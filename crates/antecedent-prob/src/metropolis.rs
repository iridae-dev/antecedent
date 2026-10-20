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
    let bytes = u64::try_from(
        count
            .checked_add(4 * p)
            .and_then(|x| x.checked_mul(8))
            .ok_or(MetropolisError::Invalid("size overflow"))?,
    )
    .map_err(|_| MetropolisError::Invalid("size overflow"))?;
    let _lease =
        ctx.acquire_admission(0, bytes).map_err(|e| MetropolisError::Resource(e.to_string()))?;
    let start_logp = log_density(initial);
    if !start_logp.is_finite() {
        return Err(MetropolisError::Invalid("initial density"));
    }
    let mut samples = Vec::with_capacity(count);
    let mut accepted = vec![0; options.chains];
    for (chain, acceptance) in accepted.iter_mut().enumerate() {
        let mut rng =
            ctx.rng.stream(antecedent_core::derive_stream(0x4359_434c_4943_4d48, chain as u64));
        let mut current = initial.to_vec();
        let mut logp = start_logp;
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
                    *acceptance += 1;
                }
            }
            if iteration >= options.warmup {
                samples.extend_from_slice(&current);
            }
        }
    }
    Ok(MetropolisDraws { samples, accepted })
}

#[cfg(test)]
mod tests {
    use super::*;
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
