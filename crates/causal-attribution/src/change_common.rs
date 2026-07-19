//! Shared change-attribution allocation / measure helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AllocationMethod, ComponentId, ExecutionContext, VariableId};
use causal_stats::gaussian_kl;

use crate::error::AttributionError;
use crate::result::ChangeAttributionResult;
use crate::shapley::{CoalitionPayoff, ShapleyEstimate, estimate_shapley, sequential_allocate};

/// How to summarize the target marginal difference.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum DifferenceMeasure {
    /// `E[Y_comparison-like] − E[Y_baseline-like]`.
    MeanDiff,
    /// Variance difference.
    VarianceDiff,
    /// Gaussian KL `KL(N(μ_S, σ_S²) ‖ N(μ₀, σ₀²))` of the hybrid outcome law vs the
    /// all-baseline coalition (pinned baseline's default target functional).
    GaussianKl,
}

/// Shared sampling / measure knobs for change attribution.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChangeOptions {
    pub measure: DifferenceMeasure,
    pub n_samples: usize,
    pub seed: u64,
}

impl ChangeOptions {
    #[must_use]
    pub const fn new(measure: DifferenceMeasure, n_samples: usize, seed: u64) -> Self {
        Self { measure, n_samples, seed }
    }

    #[must_use]
    pub const fn default_mean() -> Self {
        Self::new(DifferenceMeasure::MeanDiff, 2_000, 0)
    }
}

/// Total change from empty vs full coalition values under `measure`.
#[must_use]
pub(crate) fn total_change(measure: DifferenceMeasure, v0: f64, v_full: f64) -> f64 {
    match measure {
        DifferenceMeasure::GaussianKl => v_full,
        _ => v_full - v0,
    }
}

/// Evaluate a difference measure given hybrid `(μ, var)` and optional baseline law.
pub(crate) fn measure_value(
    measure: DifferenceMeasure,
    mask: u64,
    mu: f64,
    var: f64,
    baseline_law: Option<(f64, f64)>,
) -> Result<f64, AttributionError> {
    match measure {
        DifferenceMeasure::MeanDiff => Ok(mu),
        DifferenceMeasure::VarianceDiff => Ok(var),
        DifferenceMeasure::GaussianKl => {
            if mask == 0 {
                Ok(0.0)
            } else {
                let (mu0, var0) = baseline_law.ok_or_else(|| {
                    AttributionError::unsupported(
                        "Gaussian KL payoff missing cached baseline law",
                    )
                })?;
                Ok(gaussian_kl(mu, var, mu0, var0)?)
            }
        }
    }
}

/// Run Shapley / sequential allocation and pack a [`ChangeAttributionResult`].
pub(crate) fn run_change_allocation<P: CoalitionPayoff>(
    outcome: VariableId,
    players: &[ComponentId],
    allocation: &AllocationMethod,
    payoff: &mut P,
    total_change: f64,
    unidentified: Arc<[ComponentId]>,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    let estimate = match allocation {
        AllocationMethod::Shapley { approximation } => {
            estimate_shapley(players, approximation, payoff, ctx)?
        }
        AllocationMethod::Sequential { order } => {
            let index_of = |c: ComponentId| players.iter().position(|&p| p == c);
            sequential_allocate(order, &index_of, payoff, ctx)?
        }
        AllocationMethod::PathBased => {
            return Err(AttributionError::unsupported(
                "PathBased allocation is handled by path_decompose, not change attribution",
            ));
        }
        _ => {
            return Err(AttributionError::unsupported("unsupported AllocationMethod"));
        }
    };
    Ok(pack_change_result(outcome, total_change, estimate, unidentified))
}

fn pack_change_result(
    outcome: VariableId,
    total_change: f64,
    estimate: ShapleyEstimate,
    unidentified: Arc<[ComponentId]>,
) -> ChangeAttributionResult {
    let mc_stderr = estimate.monte_carlo_stderr;
    let component_mc = estimate.component_mc_stderr.clone().map(Arc::from);
    let interactions = Arc::from(estimate.interactions.clone());
    let cache_stats = estimate.cache_stats.clone();
    let budget = estimate.budget.clone();
    let contributions = Arc::from(estimate.into_contributions());
    ChangeAttributionResult {
        outcome,
        total_change,
        contributions,
        interactions,
        path_breakdown: Arc::from([]),
        unidentified,
        graph_sensitivity: None,
        budget,
        monte_carlo_stderr: mc_stderr,
        component_mc_stderr: component_mc,
        cache_stats,
    }
}
