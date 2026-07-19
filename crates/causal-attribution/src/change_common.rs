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

/// Run Shapley / sequential / path-based allocation and pack a [`ChangeAttributionResult`].
pub(crate) fn run_change_allocation<P: CoalitionPayoff>(
    outcome: VariableId,
    players: &[ComponentId],
    allocation: &AllocationMethod,
    payoff: &mut P,
    total_change: f64,
    unidentified: Arc<[ComponentId]>,
    ctx: &ExecutionContext,
    model_for_paths: Option<&causal_model::CompiledCausalModel>,
) -> Result<ChangeAttributionResult, AttributionError> {
    match allocation {
        AllocationMethod::Shapley { approximation } => {
            let estimate = estimate_shapley(players, approximation, payoff, ctx)?;
            Ok(pack_change_result(outcome, total_change, estimate, unidentified, Arc::from([])))
        }
        AllocationMethod::Sequential { order } => {
            let index_of = |c: ComponentId| players.iter().position(|&p| p == c);
            let estimate = sequential_allocate(order, &index_of, payoff, ctx)?;
            Ok(pack_change_result(outcome, total_change, estimate, unidentified, Arc::from([])))
        }
        AllocationMethod::PathBased => {
            let model = model_for_paths.ok_or_else(|| {
                AttributionError::unsupported(
                    "PathBased allocation requires a compiled model with linear edges",
                )
            })?;
            path_based_change_allocation(
                model,
                outcome,
                players,
                payoff,
                total_change,
                unidentified,
                ctx,
            )
        }
        _ => Err(AttributionError::unsupported("unsupported AllocationMethod")),
    }
}

fn path_based_change_allocation<P: CoalitionPayoff>(
    model: &causal_model::CompiledCausalModel,
    outcome: VariableId,
    players: &[ComponentId],
    payoff: &mut P,
    total_change: f64,
    unidentified: Arc<[ComponentId]>,
    _ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    use crate::path::path_decompose;
    use crate::result::{ComponentContribution, PathContribution};

    let outcome_dense = model
        .dense_of(outcome)
        .ok_or_else(|| AttributionError::missing_var("outcome", outcome))?;
    let full = if players.is_empty() { 0 } else { (1u64 << players.len()) - 1 };
    let v_full = payoff.value(full)?;
    let mut path_breakdown: Vec<PathContribution> = Vec::new();
    let mut contributions: Vec<ComponentContribution> = Vec::new();
    let mut evaluations = 0u64;

    for (i, &comp) in players.iter().enumerate() {
        let without = full & !(1u64 << i);
        let v_wo = payoff.value(without)?;
        let marginal = v_full - v_wo;
        evaluations += 2;

        let src = comp.variable();
        let src_dense = model
            .dense_of(src)
            .ok_or_else(|| AttributionError::missing_var("source", src))?;

        // Path shares via linear β-products; fall back to single direct share.
        let path_result = path_decompose(model, &[src], outcome, 64, 16, _ctx);
        let mut player_paths: Vec<PathContribution> = Vec::new();
        match path_result {
            Ok(res) if !res.path_breakdown.is_empty() => {
                let weight_sum: f64 = res
                    .path_breakdown
                    .iter()
                    .map(|p| p.contribution.abs())
                    .sum::<f64>()
                    .max(1e-12);
                for p in res.path_breakdown.iter() {
                    let sign = if p.contribution >= 0.0 { 1.0 } else { -1.0 };
                    let share = marginal * (p.contribution.abs() / weight_sum) * sign;
                    player_paths.push(PathContribution {
                        path: Arc::clone(&p.path),
                        contribution: share,
                    });
                }
            }
            _ => {
                // Nonlinear or no path: attribute marginal to the player→outcome hop.
                let path = if src_dense == outcome_dense {
                    Arc::from([src])
                } else {
                    Arc::from([src, outcome])
                };
                player_paths.push(PathContribution { path, contribution: marginal });
            }
        }
        let contrib_sum: f64 = player_paths.iter().map(|p| p.contribution).sum();
        contributions.push(ComponentContribution {
            component: comp,
            contribution: contrib_sum,
            stderr: None,
            ci_low: None,
            ci_high: None,
        });
        path_breakdown.extend(player_paths);
    }

    let _ = total_change;
    Ok(ChangeAttributionResult {
        outcome,
        total_change: contributions.iter().map(|c| c.contribution).sum(),
        contributions: Arc::from(contributions),
        interactions: Arc::from([]),
        path_breakdown: Arc::from(path_breakdown),
        unidentified,
        graph_sensitivity: None,
        budget: crate::result::ComputeBudget {
            evaluations,
            samples: 0,
            exact_coalitions: 0,
        },
        monte_carlo_stderr: None,
        component_mc_stderr: None,
        cache_stats: crate::result::CacheStats::default(),
    })
}

fn pack_change_result(
    outcome: VariableId,
    total_change: f64,
    estimate: ShapleyEstimate,
    unidentified: Arc<[ComponentId]>,
    path_breakdown: Arc<[crate::result::PathContribution]>,
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
        path_breakdown,
        unidentified,
        graph_sensitivity: None,
        budget,
        monte_carlo_stderr: mc_stderr,
        component_mc_stderr: component_mc,
        cache_stats,
    }
}
