// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hamiltonian Monte Carlo for explicit log-density targets with an exact gradient.
//!
//! The target supplies its log density and gradient at a point; the sampler runs a fixed
//! number of leapfrog steps per transition with a jittered step, adapts the step by Nesterov
//! dual averaging (Hoffman & Gelman 2014, alg. 5) during warmup and keeps the averaged step
//! for every retained draw. The kinetic energy uses the identity metric, so the caller passes
//! a target already scaled to roughly unit width (for example whitened at a posterior mode).
//! A trajectory whose energy error is not finite or exceeds [`DIVERGENCE`] is rejected.
//!
//! [`sample_nuts_dispersed_parallel`] and [`nuts_step`] replace the fixed trajectory with the
//! No-U-Turn sampler (Hoffman & Gelman 2014) with multinomial sampling of the trajectory
//! (Betancourt 2017): the trajectory doubles until it turns back on itself, so it crosses a
//! long curved ridge that a fixed number of leapfrog steps traverses only slowly.
//!
//! Draws use the layout of [`crate::metropolis::MetropolisDraws`], and every chain's streams
//! are keyed by its index, so the output is bit-identical for every thread count.
//! The sampler does not certify mixing: the caller gates the draws on R-hat and ESS.

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_kernels::standard_normal;

use crate::metropolis::{MetropolisDraws, MetropolisError};

/// Energy error beyond which a trajectory counts as divergent and is rejected.
pub const DIVERGENCE: f64 = 1000.0;

/// Chains, schedule and leapfrog settings of [`sample_gradient_hmc_dispersed_parallel`].
#[derive(Clone, Copy, Debug)]
pub struct GradientHmcOptions {
    /// Independent semantic chain streams (2 to 16).
    pub chains: usize,
    /// Discarded iterations per chain, during which the step is adapted.
    pub warmup: usize,
    /// Retained draws per chain.
    pub draws: usize,
    /// Leapfrog steps per transition.
    pub leapfrog_steps: u32,
    /// Step the adaptation starts from.
    pub initial_step: f64,
    /// Acceptance probability the dual averaging targets.
    pub target_accept: f64,
}

impl Default for GradientHmcOptions {
    fn default() -> Self {
        Self {
            chains: 4,
            warmup: 1000,
            draws: 1000,
            leapfrog_steps: 16,
            initial_step: 0.2,
            target_accept: 0.8,
        }
    }
}

/// Log density and gradient of the target at one point; `None` outside its support.
pub type GradientTarget<'a> = dyn Fn(&[f64]) -> Option<(f64, Vec<f64>)> + Sync + 'a;

/// Nesterov dual averaging of the leapfrog step during warmup.
#[derive(Clone, Copy, Debug)]
pub struct StepAdaptation {
    mu: f64,
    h_bar: f64,
    log_step_bar: f64,
    iteration: f64,
    target_accept: f64,
    /// Step the next warmup transition uses.
    pub step: f64,
}

impl StepAdaptation {
    /// Adaptation from `initial_step` toward `target_accept`.
    #[must_use]
    pub fn new(initial_step: f64, target_accept: f64) -> Self {
        Self {
            mu: (10.0 * initial_step).ln(),
            h_bar: 0.0,
            log_step_bar: initial_step.ln(),
            iteration: 0.0,
            target_accept,
            step: initial_step,
        }
    }

    /// Fold one warmup transition's acceptance probability into the step.
    pub fn update(&mut self, accept_prob: f64) {
        const GAMMA: f64 = 0.05;
        const T0: f64 = 10.0;
        const KAPPA: f64 = 0.75;
        self.iteration += 1.0;
        let m = self.iteration;
        let eta = 1.0 / (m + T0);
        self.h_bar = (1.0 - eta) * self.h_bar + eta * (self.target_accept - accept_prob);
        let log_step = (self.mu - m.sqrt() / GAMMA * self.h_bar).clamp(-20.0, 5.0);
        let weight = m.powf(-KAPPA);
        self.log_step_bar = weight * log_step + (1.0 - weight) * self.log_step_bar;
        self.step = log_step.exp();
    }

    /// The averaged step every retained transition uses.
    #[must_use]
    pub fn finalize(&self) -> f64 {
        self.log_step_bar.exp()
    }
}

/// One transition's outcome.
#[derive(Clone, Debug)]
pub struct Transition {
    /// Whether the proposal was accepted.
    pub accepted: bool,
    /// Metropolis acceptance probability of the proposal (0 for a divergent trajectory).
    pub accept_prob: f64,
}

/// One HMC transition from `position` (log density `logp`, gradient `gradient`), updating all
/// three in place when the proposal is accepted. The executed step is `step` jittered by a
/// uniform factor in `[0.8, 1.2]`.
pub fn transition(
    position: &mut Vec<f64>,
    logp: &mut f64,
    gradient: &mut Vec<f64>,
    step: f64,
    leapfrog_steps: u32,
    target: &GradientTarget<'_>,
    rng: &mut CausalRng,
) -> Result<Transition, MetropolisError> {
    let dim = position.len();
    let step = step * (0.8 + 0.4 * rng.next_f64());
    let momentum: Vec<f64> = (0..dim).map(|_| standard_normal(rng)).collect();
    let kinetic = 0.5 * momentum.iter().map(|p| p * p).sum::<f64>();
    let mut q = position.clone();
    let mut p = momentum;
    let mut grad = gradient.clone();
    let mut proposal_logp = *logp;
    for (pi, gi) in p.iter_mut().zip(&grad) {
        *pi += 0.5 * step * gi;
    }
    let mut diverged = false;
    for leap in 0..leapfrog_steps {
        for (qi, pi) in q.iter_mut().zip(&p) {
            *qi += step * pi;
        }
        let Some((value, next)) = target(&q) else {
            diverged = true;
            break;
        };
        if value.is_nan() || value == f64::INFINITY || next.len() != dim {
            return Err(MetropolisError::Invalid("nonfinite target"));
        }
        if !value.is_finite() || next.iter().any(|g| !g.is_finite()) {
            diverged = true;
            break;
        }
        proposal_logp = value;
        grad = next;
        let factor = if leap + 1 == leapfrog_steps { 0.5 } else { 1.0 };
        for (pi, gi) in p.iter_mut().zip(&grad) {
            *pi += factor * step * gi;
        }
    }
    let energy_error = if diverged {
        f64::INFINITY
    } else {
        let proposal_kinetic = 0.5 * p.iter().map(|v| v * v).sum::<f64>();
        (-*logp + kinetic) - (-proposal_logp + proposal_kinetic)
    };
    if !energy_error.is_finite() || -energy_error > DIVERGENCE {
        return Ok(Transition { accepted: false, accept_prob: 0.0 });
    }
    let accept_prob = energy_error.exp().min(1.0);
    let accepted = rng.next_f64() < accept_prob;
    if accepted {
        *position = q;
        *logp = proposal_logp;
        *gradient = grad;
    }
    Ok(Transition { accepted, accept_prob })
}

/// Sample a target with HMC, each chain starting at `center` plus an independent Gaussian
/// offset of standard deviation `spread` (a start outside the support is redrawn at most 64
/// times, then the target is refused). Chains run under `ctx.parallelism`; the lowest-index
/// failure is reported.
pub fn sample_gradient_hmc_dispersed_parallel(
    center: &[f64],
    spread: f64,
    options: GradientHmcOptions,
    ctx: &ExecutionContext,
    target: &GradientTarget<'_>,
) -> Result<MetropolisDraws, MetropolisError> {
    let p = center.len();
    if p == 0
        || p > 1024
        || options.chains < 2
        || options.chains > 16
        || options.draws < 8
        || options.draws > 100_000
        || options.warmup > 100_000
        || options.leapfrog_steps == 0
        || options.leapfrog_steps > 1024
        || !(options.initial_step.is_finite() && options.initial_step > 0.0)
        || !(options.target_accept > 0.0 && options.target_accept < 1.0)
        || !(spread.is_finite() && spread > 0.0)
        || center.iter().any(|v| !v.is_finite())
    {
        return Err(MetropolisError::Invalid("shape or limits"));
    }
    let bytes = p
        .checked_mul(options.chains)
        .and_then(|x| x.checked_mul(options.draws))
        .and_then(|x| x.checked_add(8 * p))
        .and_then(|x| x.checked_mul(8))
        .and_then(|x| u64::try_from(x).ok())
        .ok_or(MetropolisError::Invalid("size overflow"))?;
    let _lease = ctx
        .acquire_admission(0, bytes)
        .map_err(|error| MetropolisError::Resource(error.to_string()))?;
    let chains = ctx.map_indexed(options.chains, |chain, _inner| {
        run_chain(chain, center, spread, options, ctx, target)
    })?;
    let mut samples = Vec::with_capacity(p * options.chains * options.draws);
    let mut accepted = Vec::with_capacity(options.chains);
    for (chain_samples, count) in chains {
        samples.extend_from_slice(&chain_samples);
        accepted.push(count);
    }
    Ok(MetropolisDraws { samples, accepted })
}

fn run_chain(
    chain: usize,
    center: &[f64],
    spread: f64,
    options: GradientHmcOptions,
    ctx: &ExecutionContext,
    target: &GradientTarget<'_>,
) -> Result<(Vec<f64>, usize), MetropolisError> {
    let p = center.len();
    let mut rng =
        ctx.rng.stream(antecedent_core::derive_stream(0x4847_4d43_4348_4e53, chain as u64));
    let mut start_rng =
        ctx.rng.stream(antecedent_core::derive_stream(0x4847_4d43_5354_5254, chain as u64));
    let mut start = None;
    for _ in 0..64 {
        let q: Vec<f64> =
            center.iter().map(|x| x + spread * standard_normal(&mut start_rng)).collect();
        if let Some((value, gradient)) = target(&q)
            && value.is_finite()
            && gradient.len() == p
            && gradient.iter().all(|g| g.is_finite())
        {
            start = Some((q, value, gradient));
            break;
        }
    }
    let (mut q, mut logp, mut gradient) =
        start.ok_or(MetropolisError::Invalid("dispersed start density"))?;
    let mut adaptation = StepAdaptation::new(options.initial_step, options.target_accept);
    let mut step = options.initial_step;
    let mut samples = Vec::with_capacity(p * options.draws);
    let mut acceptance = 0_usize;
    for iteration in 0..options.warmup + options.draws {
        if ctx.cancellation.is_cancelled() {
            return Err(MetropolisError::Cancelled);
        }
        if iteration == options.warmup && options.warmup > 0 {
            step = adaptation.finalize();
        } else if iteration < options.warmup {
            step = adaptation.step;
        }
        let outcome = transition(
            &mut q,
            &mut logp,
            &mut gradient,
            step,
            options.leapfrog_steps,
            target,
            &mut rng,
        )?;
        if iteration < options.warmup {
            adaptation.update(outcome.accept_prob);
        } else {
            acceptance += usize::from(outcome.accepted);
            samples.extend_from_slice(&q);
        }
    }
    Ok((samples, acceptance))
}

/// Chains, schedule and trajectory limit of [`sample_nuts_dispersed_parallel`].
#[derive(Clone, Copy, Debug)]
pub struct NutsOptions {
    /// Independent semantic chain streams (2 to 16).
    pub chains: usize,
    /// Discarded iterations per chain, during which the step is adapted.
    pub warmup: usize,
    /// Retained draws per chain.
    pub draws: usize,
    /// Most trajectory doublings per transition (1 to 12): at most `2^max_depth - 1` leapfrog
    /// steps.
    pub max_depth: u32,
    /// Step the adaptation starts from.
    pub initial_step: f64,
    /// Mean acceptance statistic the dual averaging targets.
    pub target_accept: f64,
}

/// One end of a No-U-Turn trajectory: position, momentum, log density and its gradient.
#[derive(Clone)]
struct Phase {
    q: Vec<f64>,
    p: Vec<f64>,
    logp: f64,
    gradient: Vec<f64>,
}

/// A subtree: its two ends (`inner` adjacent to the tree it extends, `outer` its far end),
/// its multinomial proposal, the log of its total weight and its summed momentum.
struct Subtree {
    inner: Phase,
    outer: Phase,
    proposal: Phase,
    log_weight: f64,
    momentum: Vec<f64>,
    valid: bool,
    accept_sum: f64,
    steps: usize,
}

fn log_add(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a < 0.0 {
        return b;
    }
    if b.is_infinite() && b < 0.0 {
        return a;
    }
    let high = a.max(b);
    high + ((a - high).exp() + (b - high).exp()).ln()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Whether a trajectory with summed momentum `rho` and end momenta `start`, `end` has turned
/// back on itself (the identity-metric No-U-Turn criterion).
fn turned(rho: &[f64], start: &[f64], end: &[f64]) -> bool {
    dot(rho, start) <= 0.0 || dot(rho, end) <= 0.0
}

/// One leapfrog step of `step` (signed) from `from`, or `None` when the target leaves its
/// support or returns a non-finite gradient.
fn leapfrog(
    from: &Phase,
    step: f64,
    target: &GradientTarget<'_>,
) -> Result<Option<Phase>, MetropolisError> {
    let mut p: Vec<f64> =
        from.p.iter().zip(&from.gradient).map(|(p, g)| p + 0.5 * step * g).collect();
    let q: Vec<f64> = from.q.iter().zip(&p).map(|(q, p)| q + step * p).collect();
    let Some((logp, gradient)) = target(&q) else {
        return Ok(None);
    };
    if logp.is_nan() || logp == f64::INFINITY || gradient.len() != q.len() {
        return Err(MetropolisError::Invalid("nonfinite target"));
    }
    if !logp.is_finite() || gradient.iter().any(|g| !g.is_finite()) {
        return Ok(None);
    }
    for (p, g) in p.iter_mut().zip(&gradient) {
        *p += 0.5 * step * g;
    }
    Ok(Some(Phase { q, p, logp, gradient }))
}

fn hamiltonian(phase: &Phase) -> f64 {
    -phase.logp + 0.5 * dot(&phase.p, &phase.p)
}

/// Build a subtree of `2^depth` leapfrog steps from `from` in the direction of `step`'s sign,
/// with multinomial proposals relative to the initial energy `h0`. An invalid subtree (a
/// divergence or an inner U-turn) stops the trajectory.
fn build(
    from: &Phase,
    step: f64,
    depth: u32,
    h0: f64,
    target: &GradientTarget<'_>,
    rng: &mut CausalRng,
) -> Result<Subtree, MetropolisError> {
    if depth == 0 {
        let Some(next) = leapfrog(from, step, target)? else {
            return Ok(Subtree {
                inner: from.clone(),
                outer: from.clone(),
                proposal: from.clone(),
                log_weight: f64::NEG_INFINITY,
                momentum: from.p.clone(),
                valid: false,
                accept_sum: 0.0,
                steps: 1,
            });
        };
        let energy = hamiltonian(&next);
        let error = energy - h0;
        if !error.is_finite() || error > DIVERGENCE {
            return Ok(Subtree {
                inner: next.clone(),
                outer: next.clone(),
                proposal: next.clone(),
                log_weight: f64::NEG_INFINITY,
                momentum: next.p.clone(),
                valid: false,
                accept_sum: 0.0,
                steps: 1,
            });
        }
        return Ok(Subtree {
            momentum: next.p.clone(),
            inner: next.clone(),
            outer: next.clone(),
            proposal: next,
            log_weight: -error,
            valid: true,
            accept_sum: (-error).exp().min(1.0),
            steps: 1,
        });
    }
    let first = build(from, step, depth - 1, h0, target, rng)?;
    if !first.valid {
        return Ok(first);
    }
    let second = build(&first.outer, step, depth - 1, h0, target, rng)?;
    let accept_sum = first.accept_sum + second.accept_sum;
    let steps = first.steps + second.steps;
    if !second.valid {
        return Ok(Subtree { valid: false, accept_sum, steps, ..second });
    }
    let log_weight = log_add(first.log_weight, second.log_weight);
    let proposal = if rng.next_f64().ln() < second.log_weight - log_weight {
        second.proposal
    } else {
        first.proposal
    };
    let momentum: Vec<f64> =
        first.momentum.iter().zip(&second.momentum).map(|(a, b)| a + b).collect();
    let valid = !turned(&momentum, &first.inner.p, &second.outer.p);
    Ok(Subtree {
        inner: first.inner,
        outer: second.outer,
        proposal,
        log_weight,
        momentum,
        valid,
        accept_sum,
        steps,
    })
}

/// One No-U-Turn transition (multinomial sampling, Betancourt 2017) from `current`, returning
/// the next state and the mean acceptance statistic of the trajectory.
fn nuts_transition(
    current: &Phase,
    step: f64,
    max_depth: u32,
    target: &GradientTarget<'_>,
    rng: &mut CausalRng,
) -> Result<(Phase, f64), MetropolisError> {
    let start = Phase {
        p: (0..current.q.len()).map(|_| standard_normal(rng)).collect(),
        ..current.clone()
    };
    let h0 = hamiltonian(&start);
    let (mut backward, mut forward) = (start.clone(), start.clone());
    let mut sample = start.clone();
    let mut log_weight = 0.0_f64;
    let mut momentum = start.p.clone();
    let (mut accept_sum, mut steps) = (0.0, 0_usize);
    for depth in 0..max_depth {
        let forwards = rng.next_f64() < 0.5;
        let edge = if forwards { &forward } else { &backward };
        let subtree = build(edge, if forwards { step } else { -step }, depth, h0, target, rng)?;
        accept_sum += subtree.accept_sum;
        steps += subtree.steps;
        if !subtree.valid {
            break;
        }
        if rng.next_f64().ln() < subtree.log_weight - log_weight {
            sample = subtree.proposal.clone();
        }
        log_weight = log_add(log_weight, subtree.log_weight);
        for (m, s) in momentum.iter_mut().zip(&subtree.momentum) {
            *m += s;
        }
        if forwards {
            forward = subtree.outer;
        } else {
            backward = subtree.outer;
        }
        if turned(&momentum, &backward.p, &forward.p) {
            break;
        }
    }
    let accept = if steps == 0 { 0.0 } else { accept_sum / steps as f64 };
    Ok((Phase { p: Vec::new(), ..sample }, accept))
}

/// One No-U-Turn transition from `position` (log density `logp`, gradient `gradient`),
/// updating all three in place, with the step jittered by a uniform factor in `[0.8, 1.2]`
/// and at most `max_depth` trajectory doublings. `accept_prob` is the trajectory's mean
/// acceptance statistic, the quantity the step adaptation folds in.
///
/// # Errors
/// A non-finite target.
pub fn nuts_step(
    position: &mut Vec<f64>,
    logp: &mut f64,
    gradient: &mut Vec<f64>,
    step: f64,
    max_depth: u32,
    target: &GradientTarget<'_>,
    rng: &mut CausalRng,
) -> Result<Transition, MetropolisError> {
    let jittered = step * (0.8 + 0.4 * rng.next_f64());
    let current =
        Phase { q: position.clone(), p: Vec::new(), logp: *logp, gradient: gradient.clone() };
    let (next, accept_prob) = nuts_transition(&current, jittered, max_depth, target, rng)?;
    let accepted = next.q != *position;
    *position = next.q;
    *logp = next.logp;
    *gradient = next.gradient;
    Ok(Transition { accepted, accept_prob })
}

/// Sample a target with the No-U-Turn sampler under the identity metric, each chain starting
/// at `center` plus an independent Gaussian offset of standard deviation `spread`, the step
/// adapted by dual averaging toward `target_accept` during warmup and fixed afterwards.
/// Chains run under `ctx.parallelism` on streams keyed by their index, so the draws do not
/// depend on the thread count. The sampler does not certify mixing: the caller gates the
/// draws on R-hat and ESS.
///
/// # Errors
/// Invalid shapes or limits, a start outside the support after 64 redraws, a non-finite
/// target or cancellation.
pub fn sample_nuts_dispersed_parallel(
    center: &[f64],
    spread: f64,
    options: NutsOptions,
    ctx: &ExecutionContext,
    target: &GradientTarget<'_>,
) -> Result<MetropolisDraws, MetropolisError> {
    let p = center.len();
    if p == 0
        || p > 1024
        || options.chains < 2
        || options.chains > 16
        || options.draws < 8
        || options.draws > 100_000
        || options.warmup > 100_000
        || options.max_depth == 0
        || options.max_depth > 12
        || !(options.initial_step.is_finite() && options.initial_step > 0.0)
        || !(options.target_accept > 0.0 && options.target_accept < 1.0)
        || !(spread.is_finite() && spread > 0.0)
        || center.iter().any(|v| !v.is_finite())
    {
        return Err(MetropolisError::Invalid("shape or limits"));
    }
    let bytes = p
        .checked_mul(options.chains)
        .and_then(|x| x.checked_mul(options.draws))
        .and_then(|x| x.checked_add(8 * p))
        .and_then(|x| x.checked_mul(8))
        .and_then(|x| u64::try_from(x).ok())
        .ok_or(MetropolisError::Invalid("size overflow"))?;
    let _lease = ctx
        .acquire_admission(0, bytes)
        .map_err(|error| MetropolisError::Resource(error.to_string()))?;
    let chains = ctx.map_indexed(options.chains, |chain, _inner| {
        run_nuts_chain(chain, center, spread, options, ctx, target)
    })?;
    let mut samples = Vec::with_capacity(p * options.chains * options.draws);
    let mut accepted = Vec::with_capacity(options.chains);
    for (chain_samples, count) in chains {
        samples.extend_from_slice(&chain_samples);
        accepted.push(count);
    }
    Ok(MetropolisDraws { samples, accepted })
}

fn run_nuts_chain(
    chain: usize,
    center: &[f64],
    spread: f64,
    options: NutsOptions,
    ctx: &ExecutionContext,
    target: &GradientTarget<'_>,
) -> Result<(Vec<f64>, usize), MetropolisError> {
    let p = center.len();
    let mut rng =
        ctx.rng.stream(antecedent_core::derive_stream(0x4e55_5453_4348_4e53, chain as u64));
    let mut start_rng =
        ctx.rng.stream(antecedent_core::derive_stream(0x4e55_5453_5354_5254, chain as u64));
    let mut start = None;
    for _ in 0..64 {
        let q: Vec<f64> =
            center.iter().map(|x| x + spread * standard_normal(&mut start_rng)).collect();
        if let Some((logp, gradient)) = target(&q)
            && logp.is_finite()
            && gradient.len() == p
            && gradient.iter().all(|g| g.is_finite())
        {
            start = Some(Phase { q, p: Vec::new(), logp, gradient });
            break;
        }
    }
    let Phase { q: mut position, mut logp, mut gradient, .. } =
        start.ok_or(MetropolisError::Invalid("dispersed start density"))?;
    let mut adaptation = StepAdaptation::new(options.initial_step, options.target_accept);
    let mut step = options.initial_step;
    let mut samples = Vec::with_capacity(p * options.draws);
    let mut moved = 0_usize;
    for iteration in 0..options.warmup + options.draws {
        if ctx.cancellation.is_cancelled() {
            return Err(MetropolisError::Cancelled);
        }
        if iteration == options.warmup && options.warmup > 0 {
            step = adaptation.finalize();
        } else if iteration < options.warmup {
            step = adaptation.step;
        }
        let outcome = nuts_step(
            &mut position,
            &mut logp,
            &mut gradient,
            step,
            options.max_depth,
            target,
            &mut rng,
        )?;
        if iteration < options.warmup {
            adaptation.update(outcome.accept_prob);
        } else {
            moved += usize::from(outcome.accepted);
            samples.extend_from_slice(&position);
        }
    }
    Ok((samples, moved))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A strongly correlated Gaussian is sampled with the right mean and variance.
    #[test]
    fn gradient_hmc_recovers_a_correlated_gaussian() {
        let ctx = ExecutionContext::for_tests(7);
        // Precision of a bivariate Gaussian with unit variances and correlation 0.9.
        let rho = 0.9_f64;
        let det = 1.0 - rho * rho;
        let target = |x: &[f64]| {
            let (a, b) = (x[0] - 1.0, x[1] + 2.0);
            let quad = (a * a - 2.0 * rho * a * b + b * b) / det;
            let gradient = vec![-(a - rho * b) / det, -(b - rho * a) / det];
            Some((-0.5 * quad, gradient))
        };
        let options =
            GradientHmcOptions { warmup: 500, draws: 2000, ..GradientHmcOptions::default() };
        let draws =
            sample_gradient_hmc_dispersed_parallel(&[0.0, 0.0], 1.0, options, &ctx, &target)
                .unwrap();
        let n = draws.samples.len() / 2;
        let mean0 = draws.samples.chunks_exact(2).map(|x| x[0]).sum::<f64>() / n as f64;
        let mean1 = draws.samples.chunks_exact(2).map(|x| x[1]).sum::<f64>() / n as f64;
        let var0 =
            draws.samples.chunks_exact(2).map(|x| (x[0] - mean0).powi(2)).sum::<f64>() / n as f64;
        assert!((mean0 - 1.0).abs() < 0.1, "mean {mean0}");
        assert!((mean1 + 2.0).abs() < 0.1, "mean {mean1}");
        assert!((var0 - 1.0).abs() < 0.15, "variance {var0}");
        assert!(draws.accepted.iter().all(|&a| a > 1000));
    }

    /// The draws do not depend on the thread count.
    #[test]
    fn gradient_hmc_draws_do_not_depend_on_threads() {
        let target = |x: &[f64]| Some((-0.5 * x[0] * x[0], vec![-x[0]]));
        let options = GradientHmcOptions { warmup: 50, draws: 50, ..GradientHmcOptions::default() };
        let run = |threads: u32| {
            let mut ctx = ExecutionContext::for_tests(3).with_admission(
                antecedent_core::execution::AdmissionPool::fixed(threads, u64::MAX),
            );
            ctx.parallelism = antecedent_core::Parallelism::bounded(
                antecedent_core::NonZeroThreadCount::new(threads).unwrap(),
            );
            sample_gradient_hmc_dispersed_parallel(&[0.0], 1.0, options, &ctx, &target).unwrap()
        };
        assert_eq!(run(1).samples, run(4).samples);
    }

    /// The No-U-Turn sampler recovers a strongly correlated Gaussian and a curved banana's
    /// first coordinate, and its draws do not depend on the thread count.
    #[test]
    fn nuts_recovers_a_correlated_gaussian() {
        let ctx = ExecutionContext::for_tests(11);
        let rho = 0.95_f64;
        let det = 1.0 - rho * rho;
        let target = |x: &[f64]| {
            let (a, b) = (x[0] - 1.0, x[1] + 2.0);
            let quad = (a * a - 2.0 * rho * a * b + b * b) / det;
            Some((-0.5 * quad, vec![-(a - rho * b) / det, -(b - rho * a) / det]))
        };
        let options = NutsOptions {
            chains: 4,
            warmup: 500,
            draws: 1000,
            max_depth: 8,
            initial_step: 0.3,
            target_accept: 0.8,
        };
        let draws =
            sample_nuts_dispersed_parallel(&[0.0, 0.0], 1.0, options, &ctx, &target).unwrap();
        let n = draws.samples.len() / 2;
        let mean0 = draws.samples.chunks_exact(2).map(|x| x[0]).sum::<f64>() / n as f64;
        let mean1 = draws.samples.chunks_exact(2).map(|x| x[1]).sum::<f64>() / n as f64;
        let var1 =
            draws.samples.chunks_exact(2).map(|x| (x[1] - mean1).powi(2)).sum::<f64>() / n as f64;
        assert!((mean0 - 1.0).abs() < 0.1, "mean {mean0}");
        assert!((mean1 + 2.0).abs() < 0.1, "mean {mean1}");
        assert!((var1 - 1.0).abs() < 0.15, "variance {var1}");
        let single = {
            let mut ctx = ExecutionContext::for_tests(11)
                .with_admission(antecedent_core::execution::AdmissionPool::fixed(1, u64::MAX));
            ctx.parallelism = antecedent_core::Parallelism::bounded(
                antecedent_core::NonZeroThreadCount::new(1).unwrap(),
            );
            sample_nuts_dispersed_parallel(&[0.0, 0.0], 1.0, options, &ctx, &target).unwrap()
        };
        assert_eq!(single.samples, draws.samples);
    }
}
