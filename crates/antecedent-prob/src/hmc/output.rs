//! Posterior packing and publication diagnostics for HMC chains.

use std::sync::Arc;

use antecedent_core::diagnostic::{RouteMass, RouteNativeDiagnostic};

use crate::backend::BayesFitResult;
use crate::diagnostics::{HessianFactorization, InferenceDiagnostics};
use crate::error::ProbError;
use crate::mcmc_stats::{all_chains_moved, mcmc_summary};
use crate::posterior::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};

use super::{HmcOptions, MAX_STEP, MIN_STEP, TransitionStats};

/// Shape of the pooled chain samples: retained draws per chain, state width and the
/// coefficient / σ² split.
#[derive(Clone, Copy)]
pub(super) struct ChainLayout {
    pub(super) n_keep: usize,
    pub(super) state_dim: usize,
    pub(super) ncols: usize,
    pub(super) include_sigma2: bool,
    pub(super) total_transitions: usize,
}

pub(super) fn pack_and_gate_hmc(
    chain_samples: &[f64],
    hmc: HmcOptions,
    layout: ChainLayout,
    stats: TransitionStats,
    step_range: (f64, f64),
) -> Result<BayesFitResult, ProbError> {
    let ChainLayout { n_keep, state_dim, ncols, include_sigma2, total_transitions } = layout;
    let total_draws = n_keep.saturating_mul(hmc.n_chains);
    let n_out = if include_sigma2 { ncols + 1 } else { ncols };
    let mut values = vec![0.0; total_draws * n_out];

    for chain in 0..hmc.n_chains {
        for d in 0..n_keep {
            let src = (chain * n_keep + d) * state_dim;
            let dest_draw = chain * n_keep + d;
            for q in 0..ncols {
                values[q * total_draws + dest_draw] = chain_samples[src + q];
            }
            if include_sigma2 {
                let lambda = chain_samples[src + ncols];
                values[ncols * total_draws + dest_draw] = lambda.exp();
            }
        }
    }

    // Point estimate: posterior mean of the retained draws. The best visited
    // state is a noisy mode estimate (typical-set draws sit ~d/2 nats below the
    // mode) and can come from warmup, so it is not reported as a mode.
    let map: Vec<f64> = (0..ncols)
        .map(|q| {
            values[q * total_draws..(q + 1) * total_draws].iter().sum::<f64>() / total_draws as f64
        })
        .collect();

    // Diagnostics on unconstrained state (includes λ, not skewed σ²).
    let summary = mcmc_summary(chain_samples, hmc.n_chains, n_keep, state_dim);
    let (ess_bulk, ess_tail, rhat_max) =
        (summary.min_bulk_ess, summary.min_tail_ess, summary.max_rhat);
    let moved = all_chains_moved(chain_samples, hmc.n_chains, n_keep, state_dim);
    let mean_accept = stats.mean_accept_prob();
    let max_dh = stats.max_abs_delta_h;

    let mut diagnostics = InferenceDiagnostics {
        converged: false,
        iterations: u32::try_from(total_transitions).unwrap_or(u32::MAX),
        grad_inf_norm: 0.0,
        hessian_condition: f64::NAN,
        factorization: HessianFactorization::Mcmc,
        separation_warning: false,
        notes: vec![Arc::from(format!(
            "hmc chains={} warmup={} L={} step_size=[{:.3e}, {:.3e}]",
            hmc.n_chains, hmc.n_warmup, hmc.leapfrog_steps, step_range.0, step_range.1
        ))],
        backend_id: Arc::from("hmc"),
        n_chains: Some(u32::try_from(hmc.n_chains).unwrap_or(u32::MAX)),
        n_warmup: Some(u32::try_from(hmc.n_warmup).unwrap_or(u32::MAX)),
        ess_bulk_min: Some(ess_bulk),
        ess_tail_min: Some(ess_tail),
        rhat_max: Some(rhat_max),
        n_divergences: None,
        mean_accept_prob: Some(mean_accept),
        n_warmup_divergences: Some(stats.n_warmup_divergences),
        n_postwarmup_divergences: None,
        max_abs_delta_h: Some(max_dh),
        all_chains_moved: Some(moved),
    };
    diagnostics.record_postwarmup_divergences(stats.n_postwarmup_divergences);
    diagnostics.converged = diagnostics.mcmc_publication_ok();

    if !diagnostics.allows_posterior() {
        // The step is bounded; an adapted step pinned at a bound means the posterior scale is
        // outside what the (diagonal-metric) sampler can traverse.
        let at_bound =
            step_range.0 <= MIN_STEP * (1.0 + 1e-9) || step_range.1 >= MAX_STEP * (1.0 - 1e-9);
        let scale_hint = if at_bound {
            format!(
                "; adapted step size reached its bound ({:.3e}..{:.3e}), so the posterior \
                 scale is outside the range this sampler can traverse even with its diagonal \
                 metric: check the design for near-collinear or wildly scaled columns",
                step_range.0, step_range.1
            )
        } else {
            String::new()
        };
        let summary = format!(
            "HMC posterior refused: rhat={:?} ess_bulk={:?} ess_tail={:?} \
             postwarmup_div={:?} moved={:?} accept={:?}{scale_hint}",
            diagnostics.rhat_max,
            diagnostics.ess_bulk_min,
            diagnostics.ess_tail_min,
            diagnostics.n_postwarmup_divergences,
            diagnostics.all_chains_moved,
            diagnostics.mean_accept_prob
        );
        return Err(diagnostics.publication_error(summary.clone()).unwrap_or(ProbError::Route {
            summary,
            diagnostics: vec![
                RouteNativeDiagnostic::computational_failure(RouteMass::unreported()),
            ],
        }));
    }

    let schema = if include_sigma2 {
        let mut q: Vec<_> = (0..ncols)
            .map(|i| PosteriorQuantityKind::Coefficient { index: i, name: None })
            .collect();
        q.push(PosteriorQuantityKind::ResidualVariance);
        PosteriorSchema { quantities: Arc::from(q) }
    } else {
        PosteriorSchema::coefficients(ncols)
    };

    let draws = PosteriorDraws::from_column_major(schema, total_draws, values)?;
    Ok(BayesFitResult { draws, map, diagnostics, cov: None })
}
