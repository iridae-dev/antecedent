// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bounded symmetric random-walk Metropolis for explicit log-density targets.
//! Proposals outside target support return negative infinity; NaNs are errors.

use antecedent_core::ExecutionContext;
use antecedent_kernels::standard_normal;
use thiserror::Error;

/// Fixed proposal and retention policy (no post-warmup adaptation).
#[derive(Clone, Copy, Debug)]
pub struct MetropolisOptions {
    /// Independent semantic chain streams.
    pub chains: usize,
    /// Discarded iterations per chain.
    pub warmup: usize,
    /// Retained draws per chain.
    pub draws: usize,
    /// Isotropic Gaussian proposal standard deviation.
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
    /// Bad shape, limits, policy or density.
    #[error("invalid Metropolis target/policy: {0}")]
    Invalid(&'static str),
    /// Cancellation or memory admission.
    #[error("Metropolis resource refusal: {0}")]
    Resource(String),
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
    let _lease =
        ctx.acquire_admission(0, bytes).map_err(|e| MetropolisError::Resource(e.to_string()))?;
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
    let _lease =
        ctx.acquire_admission(0, bytes).map_err(|e| MetropolisError::Resource(e.to_string()))?;
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
    let mut rng =
        ctx.rng.stream(antecedent_core::derive_stream(0x4359_434c_4943_4d48, chain as u64));
    let (mut current, mut logp) = if spread > 0.0 {
        let mut start_rng =
            ctx.rng.stream(antecedent_core::derive_stream(0x4359_434c_5354_5254, chain as u64));
        let mut found = None;
        for _ in 0..64 {
            let start: Vec<f64> =
                initial.iter().map(|x| x + spread * standard_normal(&mut start_rng)).collect();
            let logp = log_density(&start);
            if logp.is_finite() {
                found = Some((start, logp));
                break;
            }
        }
        found.ok_or(MetropolisError::Invalid("dispersed start density"))?
    } else {
        (initial.to_vec(), log_density(initial))
    };
    for iteration in 0..options.warmup + options.draws {
        if ctx.cancellation.is_cancelled() {
            return Err(MetropolisError::Resource("cancelled".into()));
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

#[cfg(test)]
mod tests {
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
}
