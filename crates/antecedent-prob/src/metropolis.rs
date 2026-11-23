// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bounded symmetric random-walk Metropolis for explicit log-density targets.
//! Proposals outside target support return negative infinity; NaNs are errors.

use antecedent_core::ExecutionContext;
use antecedent_kernels::{cholesky_spd_into, standard_normal, update_scatter};
use thiserror::Error;

use crate::hmc::metric_windows;

/// Stream tag of each chain's proposal and acceptance draws.
const PROPOSAL_STREAM: u64 = 0x4359_434c_4943_4d48;
/// Stream tag of each chain's dispersed start.
const START_STREAM: u64 = 0x4359_434c_5354_5254;

/// Fixed proposal and retention policy (no post-warmup adaptation).
#[derive(Clone, Copy, Debug)]
pub struct MetropolisOptions {
    /// Independent semantic chain streams.
    pub chains: usize,
    /// Discarded iterations per chain.
    pub warmup: usize,
    /// Retained draws per chain.
    pub draws: usize,
    /// Isotropic Gaussian proposal standard deviation. [`sample_metropolis_tuned`] starts
    /// from it and tunes the step and proposal shape during warmup only.
    pub proposal_scale: f64,
}

/// Retained samples and actual acceptance counts.
#[derive(Clone, Debug)]
pub struct MetropolisDraws {
    /// Layout `(chain * draws + draw) * parameters + parameter`.
    pub samples: Vec<f64>,
    /// Accepted retained proposals per chain.
    pub accepted: Vec<usize>,
}

/// Explicit target/resource failure.
#[derive(Clone, Debug, Error)]
pub enum MetropolisError {
    /// Sampling was cancelled before a complete result was available.
    #[error("Metropolis sampling cancelled")]
    Cancelled,
    /// Bad shape, limits, policy or density.
    #[error("invalid Metropolis target/policy: {0}")]
    Invalid(&'static str),
    /// Cancellation or memory admission.
    #[error("Metropolis resource refusal: {0}")]
    Resource(String),
}

impl From<antecedent_core::execution::MapExecutionError<MetropolisError>> for MetropolisError {
    fn from(error: antecedent_core::execution::MapExecutionError<MetropolisError>) -> Self {
        match error {
            antecedent_core::execution::MapExecutionError::Work(error) => error,
            antecedent_core::execution::MapExecutionError::Admission(
                antecedent_core::execution::AdmissionError::Cancelled,
            ) => Self::Cancelled,
            antecedent_core::execution::MapExecutionError::Admission(error) => {
                Self::Resource(error.to_string())
            }
        }
    }
}

/// Sample an explicit normalized-or-unnormalized density. A failed convergence
/// diagnostic remains a failed diagnostic; this routine does not certify mixing.
pub fn sample_metropolis<F>(
    initial: &[f64],
    options: MetropolisOptions,
    ctx: &ExecutionContext,
    log_density: F,
) -> Result<MetropolisDraws, MetropolisError>
where
    F: Fn(&[f64]) -> f64,
{
    sample_from_starts(initial, 0.0, options, ctx, log_density)
}

/// Like [`sample_metropolis`], but each chain starts at `center` plus an independent
/// Gaussian offset of standard deviation `spread`, so the between-chain diagnostics can see
/// a chain stuck away from the bulk. A start with a non-finite density is redrawn (at most
/// 64 times), then the target is refused. Chains run one after another on the caller; use
/// [`sample_metropolis_dispersed_parallel`] for a `Sync` density.
pub fn sample_metropolis_dispersed<F>(
    center: &[f64],
    spread: f64,
    options: MetropolisOptions,
    ctx: &ExecutionContext,
    log_density: F,
) -> Result<MetropolisDraws, MetropolisError>
where
    F: Fn(&[f64]) -> f64,
{
    if !spread.is_finite() || spread <= 0.0 {
        return Err(MetropolisError::Invalid("start spread"));
    }
    sample_from_starts(center, spread, options, ctx, log_density)
}

/// [`sample_metropolis_dispersed`] with the chains run under `ctx.parallelism`. Each chain's
/// streams are keyed by its index and results return in chain order (the lowest-index failure
/// is reported), so the draws are bit-identical for every thread count.
pub fn sample_metropolis_dispersed_parallel<F>(
    center: &[f64],
    spread: f64,
    options: MetropolisOptions,
    ctx: &ExecutionContext,
    log_density: F,
) -> Result<MetropolisDraws, MetropolisError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    if !spread.is_finite() || spread <= 0.0 {
        return Err(MetropolisError::Invalid("start spread"));
    }
    let bytes = validate_and_admit_bytes(center, options)?;
    let _lease = admit_draws(ctx, bytes)?;
    if !log_density(center).is_finite() {
        return Err(MetropolisError::Invalid("initial density"));
    }
    // Chains run under the context's parallelism. Each chain's streams are keyed by its
    // index, results return in chain order and the lowest-index failure is reported, so
    // the output is bit-identical for every thread count.
    let chains = ctx.map_indexed(options.chains, |chain, _inner| {
        run_chain(chain, center, spread, options, ctx, &log_density)
    })?;
    let mut samples = Vec::with_capacity(center.len() * options.chains * options.draws);
    let mut accepted = Vec::with_capacity(options.chains);
    for (chain_samples, count) in chains {
        samples.extend_from_slice(&chain_samples);
        accepted.push(count);
    }
    Ok(MetropolisDraws { samples, accepted })
}

/// Random-walk Metropolis whose proposal is tuned during warmup, with dispersed per-chain starts.
///
/// [`sample_metropolis`] walks every chain from one start with the declared isotropic step,
/// whatever the target's width. A step chosen for one posterior width mixes slowly once the
/// posterior narrows, widens or correlates its coordinates (for example as the sample size
/// grows), and the R-hat and ESS gates then refuse a well-posed target. Here
/// `options.proposal_scale` is only the starting step. Each chain starts at `center` plus an
/// independent Gaussian offset of standard deviation `spread` (redrawn, at most 64 times, while
/// the density there is not finite). Warmup follows the window schedule of the HMC metric
/// adaptation (Stan's): every warmup iteration moves the log step by a Robbins-Monro update
/// toward the dimension's optimal random-walk acceptance rate (Andrieu and Thoms 2008), and at
/// the end of each slow window the chain's own states in that window give a regularized
/// proposal covariance, after which the step restarts at `2.38 / sqrt(d)`, the optimal scale for
/// a Gaussian target whose covariance the proposal matches (Roberts, Gelman and Gilks 1997).
/// Retained draws use the last covariance and the step averaged over the last adaptation stage,
/// both frozen, so every retained chain is a time-homogeneous symmetric Metropolis chain and the
/// convergence diagnostics read it as one. A warmup too short for a slow window keeps the
/// declared isotropic step. Chains run one after another on the caller.
///
/// # Errors
/// Invalid shape, limits, start spread or density; admission refusal or cancellation.
pub fn sample_metropolis_tuned<F>(
    center: &[f64],
    spread: f64,
    options: MetropolisOptions,
    ctx: &ExecutionContext,
    log_density: F,
) -> Result<MetropolisDraws, MetropolisError>
where
    F: Fn(&[f64]) -> f64,
{
    let bytes = validate_and_admit_bytes(center, options)?;
    if !spread.is_finite() || spread <= 0.0 {
        return Err(MetropolisError::Invalid("start spread"));
    }
    let p = center.len();
    // One chain at a time holds its window scatter, covariance and factor, plus four vectors.
    let tuning = p
        .checked_mul(p)
        .and_then(|cells| cells.checked_mul(3))
        .and_then(|cells| cells.checked_add(4 * p))
        .and_then(|cells| cells.checked_mul(8))
        .and_then(|tuning| u64::try_from(tuning).ok())
        .and_then(|tuning| tuning.checked_add(bytes))
        .ok_or(MetropolisError::Invalid("size overflow"))?;
    let _lease = admit_draws(ctx, tuning)?;
    if !log_density(center).is_finite() {
        return Err(MetropolisError::Invalid("initial density"));
    }
    let windows = metric_windows(options.warmup);
    let mut samples = Vec::with_capacity(p * options.chains * options.draws);
    let mut accepted = Vec::with_capacity(options.chains);
    for chain in 0..options.chains {
        let (chain_samples, count) =
            run_tuned_chain(chain, center, spread, options, &windows, ctx, &log_density)?;
        samples.extend_from_slice(&chain_samples);
        accepted.push(count);
    }
    Ok(MetropolisDraws { samples, accepted })
}

/// Asymptotically optimal random-walk acceptance rate for a `dim`-dimensional Gaussian target,
/// 0.44 in one dimension falling to 0.234 (Gelman, Roberts and Gilks 1996).
fn optimal_acceptance(dim: usize) -> f64 {
    0.234 + 0.207 * (-(dim as f64 - 1.0) / 2.0).exp()
}

/// Warmup tuning state of one chain: its log step, its proposal shape and the slow window whose
/// states it is accumulating.
struct ProposalTuner<'w> {
    dim: usize,
    target_acceptance: f64,
    windows: &'w [(usize, usize)],
    cursor: usize,
    log_step: f64,
    /// Updates since the step last restarted, and the sum of the log steps they produced.
    since: usize,
    log_sum: f64,
    /// Mean, scatter and count of the current slow window's states.
    mean: Vec<f64>,
    scatter: Vec<f64>,
    count: u64,
    /// Lower Cholesky factor of the proposal covariance; `None` while the walk is isotropic.
    shape: Option<Vec<f64>>,
}

impl<'w> ProposalTuner<'w> {
    fn new(dim: usize, step: f64, windows: &'w [(usize, usize)]) -> Self {
        Self {
            dim,
            target_acceptance: optimal_acceptance(dim),
            windows,
            cursor: 0,
            log_step: step.ln(),
            since: 0,
            log_sum: 0.0,
            mean: Vec::new(),
            scatter: Vec::new(),
            count: 0,
            shape: None,
        }
    }

    /// Whether this warmup is long enough to tune at all.
    fn tuning(&self) -> bool {
        !self.windows.is_empty()
    }

    /// The exploratory step of the next warmup iteration.
    fn step(&self) -> f64 {
        self.log_step.exp()
    }

    /// Step of the retained draws: the mean log step since the last restart.
    fn frozen_step(&self) -> f64 {
        if self.since == 0 { self.step() } else { (self.log_sum / self.since as f64).exp() }
    }

    /// `current + step · L z`, or `current + step · z` while the walk is isotropic.
    fn propose(&self, current: &[f64], step: f64, z: &[f64], out: &mut [f64]) {
        let p = self.dim;
        match &self.shape {
            None => {
                for i in 0..p {
                    out[i] = current[i] + step * z[i];
                }
            }
            Some(factor) => {
                for i in 0..p {
                    let row = &factor[i * p..=i * p + i];
                    let offset: f64 = row.iter().zip(z).map(|(l, v)| l * v).sum();
                    out[i] = current[i] + step * offset;
                }
            }
        }
    }

    /// Record warmup iteration `t`: its Metropolis acceptance probability and the state the
    /// iteration left the chain in.
    fn observe(&mut self, t: usize, acceptance: f64, state: &[f64]) {
        self.since += 1;
        let gain = (self.since as f64).powf(-0.6);
        self.log_step =
            (self.log_step + gain * (acceptance - self.target_acceptance)).clamp(-700.0, 700.0);
        self.log_sum += self.log_step;
        let Some(&(start, end)) = self.windows.get(self.cursor) else { return };
        if (start..end).contains(&t) {
            if self.count == 0 {
                self.mean.clear();
                self.mean.extend_from_slice(state);
                self.scatter.clear();
                self.scatter.resize(self.dim * self.dim, 0.0);
            } else {
                update_scatter(&mut self.mean, &mut self.scatter, self.count, state);
            }
            self.count += 1;
        }
        if t + 1 == end {
            self.cursor += 1;
            if let Some(shape) = self.window_shape() {
                self.shape = Some(shape);
                self.log_step = (2.38 / (self.dim as f64).sqrt()).ln();
                self.since = 0;
                self.log_sum = 0.0;
            }
            self.count = 0;
        }
    }

    /// Cholesky factor of the closed window's covariance with its correlations shrunk toward
    /// zero by `5 / (n + 5)`, or `None` (keep the current shape) when the window is too short,
    /// a coordinate never moved or the matrix is not positive definite.
    fn window_shape(&self) -> Option<Vec<f64>> {
        if self.count < 3 {
            return None;
        }
        let p = self.dim;
        let n = self.count as f64;
        let shrink = n / (n + 5.0);
        let mut covariance = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                let value = self.scatter[i * p + j] / (n - 1.0);
                covariance[i * p + j] = if i == j { value } else { shrink * value };
            }
        }
        if (0..p).any(|i| {
            let variance = covariance[i * p + i];
            !variance.is_finite() || variance <= 0.0
        }) {
            return None;
        }
        let mut factor = vec![0.0; p * p];
        cholesky_spd_into(&covariance, p, &mut factor).ok()?;
        Some(factor)
    }
}

/// One tuned chain. Its start and proposal streams depend only on `chain` and the context seed.
fn run_tuned_chain<F>(
    chain: usize,
    center: &[f64],
    spread: f64,
    options: MetropolisOptions,
    windows: &[(usize, usize)],
    ctx: &ExecutionContext,
    log_density: &F,
) -> Result<(Vec<f64>, usize), MetropolisError>
where
    F: Fn(&[f64]) -> f64,
{
    let p = center.len();
    let mut samples = Vec::with_capacity(p * options.draws);
    let mut acceptance = 0_usize;
    let mut rng = ctx.rng.stream(antecedent_core::derive_stream(PROPOSAL_STREAM, chain as u64));
    let (mut current, mut logp) = chain_start(chain, center, spread, ctx, log_density)?;
    let mut tuner = ProposalTuner::new(p, options.proposal_scale, windows);
    let mut step = options.proposal_scale;
    let mut z = vec![0.0; p];
    let mut proposal = vec![0.0; p];
    for iteration in 0..options.warmup + options.draws {
        if ctx.cancellation.is_cancelled() {
            return Err(MetropolisError::Cancelled);
        }
        if iteration == options.warmup && tuner.tuning() {
            step = tuner.frozen_step();
        }
        for value in &mut z {
            *value = standard_normal(&mut rng);
        }
        tuner.propose(&current, step, &z, &mut proposal);
        let candidate_logp = log_density(&proposal);
        if candidate_logp.is_nan() || candidate_logp == f64::INFINITY {
            return Err(MetropolisError::Invalid("nonfinite target"));
        }
        let log_ratio = candidate_logp - logp;
        if rng.next_f64().ln() < log_ratio {
            current.copy_from_slice(&proposal);
            logp = candidate_logp;
            if iteration >= options.warmup {
                acceptance += 1;
            }
        }
        if iteration >= options.warmup {
            samples.extend_from_slice(&current);
        } else if tuner.tuning() {
            tuner.observe(iteration, log_ratio.min(0.0).exp(), &current);
            step = tuner.step();
        }
    }
    Ok((samples, acceptance))
}

fn admit_draws(
    ctx: &ExecutionContext,
    bytes: u64,
) -> Result<antecedent_core::execution::AdmissionLease, MetropolisError> {
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => MetropolisError::Cancelled,
        other => MetropolisError::Resource(other.to_string()),
    })
}

fn sample_from_starts<F>(
    initial: &[f64],
    spread: f64,
    options: MetropolisOptions,
    ctx: &ExecutionContext,
    log_density: F,
) -> Result<MetropolisDraws, MetropolisError>
where
    F: Fn(&[f64]) -> f64,
{
    let bytes = validate_and_admit_bytes(initial, options)?;
    let _lease = admit_draws(ctx, bytes)?;
    if !log_density(initial).is_finite() {
        return Err(MetropolisError::Invalid("initial density"));
    }
    let mut samples = Vec::with_capacity(initial.len() * options.chains * options.draws);
    let mut accepted = Vec::with_capacity(options.chains);
    for chain in 0..options.chains {
        let (chain_samples, count) = run_chain(chain, initial, spread, options, ctx, &log_density)?;
        samples.extend_from_slice(&chain_samples);
        accepted.push(count);
    }
    Ok(MetropolisDraws { samples, accepted })
}

/// Shape and limit checks; returns the bytes to admit for the retained samples.
fn validate_and_admit_bytes(
    initial: &[f64],
    options: MetropolisOptions,
) -> Result<u64, MetropolisError> {
    let p = initial.len();
    if p == 0
        || p > 1024
        || options.chains < 2
        || options.chains > 16
        || options.draws < 8
        || options.draws > 100_000
        || options.warmup > 100_000
        || !options.proposal_scale.is_finite()
        || options.proposal_scale <= 0.0
        || initial.iter().any(|v| !v.is_finite())
    {
        return Err(MetropolisError::Invalid("shape or limits"));
    }
    let count = p
        .checked_mul(options.chains)
        .and_then(|x| x.checked_mul(options.draws))
        .ok_or(MetropolisError::Invalid("size overflow"))?;
    u64::try_from(
        count
            .checked_add(4 * p)
            .and_then(|x| x.checked_mul(8))
            .ok_or(MetropolisError::Invalid("size overflow"))?,
    )
    .map_err(|_| MetropolisError::Invalid("size overflow"))
}

/// One chain: its start and proposal streams depend only on `chain` and the context seed,
/// never on scheduling, so chains may run in any order or concurrently with identical results.
fn run_chain<F>(
    chain: usize,
    initial: &[f64],
    spread: f64,
    options: MetropolisOptions,
    ctx: &ExecutionContext,
    log_density: &F,
) -> Result<(Vec<f64>, usize), MetropolisError>
where
    F: Fn(&[f64]) -> f64,
{
    let p = initial.len();
    let mut samples = Vec::with_capacity(p * options.draws);
    let mut acceptance = 0_usize;
    let mut rng = ctx.rng.stream(antecedent_core::derive_stream(PROPOSAL_STREAM, chain as u64));
    let (mut current, mut logp) = chain_start(chain, initial, spread, ctx, log_density)?;
    for iteration in 0..options.warmup + options.draws {
        if ctx.cancellation.is_cancelled() {
            return Err(MetropolisError::Cancelled);
        }
        let proposal: Vec<_> = current
            .iter()
            .map(|x| x + options.proposal_scale * standard_normal(&mut rng))
            .collect();
        let candidate_logp = log_density(&proposal);
        if candidate_logp.is_nan() || candidate_logp == f64::INFINITY {
            return Err(MetropolisError::Invalid("nonfinite target"));
        }
        if rng.next_f64().ln() < candidate_logp - logp {
            current = proposal;
            logp = candidate_logp;
            if iteration >= options.warmup {
                acceptance += 1;
            }
        }
        if iteration >= options.warmup {
            samples.extend_from_slice(&current);
        }
    }
    Ok((samples, acceptance))
}

/// A chain's start: `initial` itself, or for a positive `spread` an independent Gaussian offset
/// of that standard deviation, redrawn (at most 64 times) while the density there is not finite.
fn chain_start<F>(
    chain: usize,
    initial: &[f64],
    spread: f64,
    ctx: &ExecutionContext,
    log_density: &F,
) -> Result<(Vec<f64>, f64), MetropolisError>
where
    F: Fn(&[f64]) -> f64,
{
    if spread > 0.0 {
        let mut start_rng =
            ctx.rng.stream(antecedent_core::derive_stream(START_STREAM, chain as u64));
        for _ in 0..64 {
            let start: Vec<f64> =
                initial.iter().map(|x| x + spread * standard_normal(&mut start_rng)).collect();
            let logp = log_density(&start);
            if logp.is_finite() {
                return Ok((start, logp));
            }
        }
        Err(MetropolisError::Invalid("dispersed start density"))
    } else {
        Ok((initial.to_vec(), log_density(initial)))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn draw_admission_keeps_resource_and_cancellation_refusals() {
        let ctx = antecedent_core::ExecutionContext::for_tests(1)
            .with_admission(antecedent_core::execution::AdmissionPool::fixed(1, 0));
        assert!(matches!(super::admit_draws(&ctx, 1), Err(super::MetropolisError::Resource(_))));
        ctx.cancellation.cancel();
        assert!(matches!(super::admit_draws(&ctx, 1), Err(super::MetropolisError::Cancelled)));
    }

    use super::*;
    #[test]
    fn dispersed_starts_differ_by_chain_and_still_recover_moments() {
        let options =
            MetropolisOptions { chains: 4, warmup: 400, draws: 3000, proposal_scale: 1.0 };
        let run = || {
            sample_metropolis_dispersed(
                &[0.0],
                4.0,
                options,
                &ExecutionContext::for_tests(42),
                |x| -0.5 * x[0].powi(2),
            )
            .unwrap()
        };
        let a = run();
        assert_eq!(a.samples, run().samples);
        let mean = a.samples.iter().sum::<f64>() / a.samples.len() as f64;
        assert!(mean.abs() < 0.15);
        let firsts: Vec<f64> = (0..4).map(|c| a.samples[c * options.draws]).collect();
        assert!(firsts.windows(2).any(|w| w[0].to_bits() != w[1].to_bits()));
    }

    #[test]
    fn dispersed_chains_are_bit_identical_for_any_thread_budget() {
        use antecedent_core::{NonZeroThreadCount, Parallelism};
        let options = MetropolisOptions { chains: 4, warmup: 200, draws: 500, proposal_scale: 1.0 };
        let run = |threads: u32| {
            let mut ctx = ExecutionContext::for_tests(7).with_admission(
                antecedent_core::execution::AdmissionPool::fixed(threads, u64::MAX),
            );
            ctx.parallelism = Parallelism::bounded(NonZeroThreadCount::new(threads).unwrap());
            let seen = std::sync::Mutex::new(std::collections::HashSet::new());
            let draws =
                sample_metropolis_dispersed_parallel(&[0.0, 1.0], 3.0, options, &ctx, |x| {
                    seen.lock().unwrap().insert(std::thread::current().id());
                    -0.5 * (x[0].powi(2) + (x[1] - 1.0).powi(2))
                })
                .unwrap();
            let used = seen.lock().unwrap().len();
            (draws, used)
        };
        let (serial, serial_threads) = run(1);
        let (parallel, parallel_threads) = run(4);
        assert_eq!(serial_threads, 1);
        assert!(parallel_threads > 1, "the chains must actually overlap");
        let bits = |d: &MetropolisDraws| d.samples.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&serial), bits(&parallel));
        assert_eq!(serial.accepted, parallel.accepted);
    }

    #[test]
    fn normal_target_replays_and_recovers_moments() {
        let options =
            MetropolisOptions { chains: 4, warmup: 200, draws: 3000, proposal_scale: 1.0 };
        let run = || {
            sample_metropolis(&[0.0], options, &ExecutionContext::for_tests(42), |x| {
                -0.5 * x[0].powi(2)
            })
            .unwrap()
        };
        let a = run();
        let b = run();
        assert_eq!(a.samples, b.samples);
        let mean = a.samples.iter().sum::<f64>() / a.samples.len() as f64;
        let variance =
            a.samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / a.samples.len() as f64;
        assert!(mean.abs() < 0.1 && (variance - 1.0).abs() < 0.12);
        assert!(a.accepted.iter().all(|&v| v > 0 && v < options.draws));
    }

    fn moments(samples: &[f64], p: usize, coordinate: usize) -> (f64, f64) {
        let values: Vec<f64> = samples.chunks_exact(p).map(|draw| draw[coordinate]).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
        (mean, variance)
    }

    /// A step declared for a posterior about three times wider (the fixed-step walk's failure
    /// mode as the sample size grows) is retuned in warmup: the retained chains accept near the
    /// one-dimensional optimum, recover the moments and pass the convergence gate.
    #[test]
    fn tuned_walk_retunes_a_step_declared_for_a_wider_posterior() {
        let options =
            MetropolisOptions { chains: 4, warmup: 500, draws: 1500, proposal_scale: 0.14 };
        let target = |x: &[f64]| -0.5 * ((x[0] - 0.3) / 0.04).powi(2);
        let run = || {
            sample_metropolis_tuned(&[0.3], 0.14, options, &ExecutionContext::for_tests(11), target)
                .unwrap()
        };
        let a = run();
        assert_eq!(a.samples, run().samples);
        assert_eq!(a.accepted, run().accepted);
        for &accepted in &a.accepted {
            let rate = accepted as f64 / options.draws as f64;
            assert!((0.3..0.6).contains(&rate), "retained acceptance {rate}");
        }
        let (mean, variance) = moments(&a.samples, 1, 0);
        assert!((mean - 0.3).abs() < 0.01, "{mean}");
        assert!((variance / 0.0016 - 1.0).abs() < 0.25, "{variance}");
        let summary = crate::mcmc_summary(&a.samples, options.chains, options.draws, 1);
        assert!(summary.max_rhat < 1.02 && summary.min_bulk_ess > 400.0, "{summary:?}");
        let firsts: Vec<u64> = (0..4).map(|c| a.samples[c * options.draws].to_bits()).collect();
        assert!(firsts.windows(2).any(|w| w[0] != w[1]));
    }

    /// Two coordinates on scales a hundred times apart and correlated at 0.9: the windowed
    /// covariance shapes the proposal, so the narrow direction no longer caps the step.
    #[test]
    fn tuned_walk_learns_a_correlated_badly_scaled_shape() {
        let options =
            MetropolisOptions { chains: 4, warmup: 1000, draws: 2000, proposal_scale: 1.0 };
        let (sd, rho) = ([1.0, 0.01], 0.9);
        let target = |x: &[f64]| {
            let (u, v) = (x[0] / sd[0], x[1] / sd[1]);
            -0.5 * (u * u - 2.0 * rho * u * v + v * v) / (1.0 - rho * rho)
        };
        let draws = sample_metropolis_tuned(
            &[0.0, 0.0],
            0.01,
            options,
            &ExecutionContext::for_tests(5),
            target,
        )
        .unwrap();
        let (mean, variance) = moments(&draws.samples, 2, 0);
        assert!(mean.abs() < 0.2 && (variance - 1.0).abs() < 0.3, "{mean} {variance}");
        let (mean, variance) = moments(&draws.samples, 2, 1);
        assert!(mean.abs() < 0.002 && (variance / 1e-4 - 1.0).abs() < 0.3, "{mean} {variance}");
        for &accepted in &draws.accepted {
            let rate = accepted as f64 / options.draws as f64;
            assert!((0.2..0.55).contains(&rate), "retained acceptance {rate}");
        }
    }

    /// A warmup too short for a slow window keeps the declared isotropic step: the chains are
    /// exactly the fixed-step dispersed walk's. The start spread is required.
    #[test]
    fn short_tuned_warmup_keeps_the_declared_step() {
        assert!(metric_windows(99).is_empty());
        let options = MetropolisOptions { chains: 2, warmup: 16, draws: 16, proposal_scale: 0.5 };
        let ctx = ExecutionContext::for_tests(3);
        let draws =
            sample_metropolis_tuned(&[0.0], 0.5, options, &ctx, |x| -0.5 * x[0].powi(2)).unwrap();
        assert_eq!(draws.samples.len(), 32);
        let fixed =
            sample_metropolis_dispersed(&[0.0], 0.5, options, &ctx, |x| -0.5 * x[0].powi(2))
                .unwrap();
        assert_eq!(draws.samples, fixed.samples);
        assert_eq!(draws.accepted, fixed.accepted);
        assert!(matches!(
            sample_metropolis_tuned(&[0.0], 0.0, options, &ctx, |x| -0.5 * x[0].powi(2)),
            Err(MetropolisError::Invalid("start spread"))
        ));
    }
}
