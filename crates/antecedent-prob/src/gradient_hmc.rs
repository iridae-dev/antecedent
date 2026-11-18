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
}
