//! Shared change-attribution allocation / measure helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AllocationMethod, ComponentId, ExecutionContext, VariableId};
use antecedent_stats::gaussian_kl;

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
    /// all-baseline coalition (the default target functional in Budhathoki, Janzing,
    /// Bloebaum & Ng 2021).
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
                    AttributionError::unsupported("Gaussian KL payoff missing cached baseline law")
                })?;
                Ok(gaussian_kl(mu, var, mu0, var0)?)
            }
        }
    }
}

/// Shared [`CoalitionPayoff::value`] caching shape for hybrid-outcome-law payoffs.
///
/// Both `MechanismSwapPayoff` ([`crate::distribution_change`]) and
/// `StructureSwapPayoff` ([`crate::structure_change`]) memoize the mask-0
/// all-baseline outcome law once (only needed for [`DifferenceMeasure::GaussianKl`])
/// and then evaluate the coalition's difference measure at `mask`; they differ only
/// in how they compute the outcome law itself ([`Self::law_at`]).
pub(crate) trait CachedOutcomeLawPayoff {
    /// The configured difference measure.
    fn measure(&self) -> DifferenceMeasure;
    /// The cached all-baseline `(μ₀, σ₀²)`, once computed.
    fn baseline_law(&self) -> Option<(f64, f64)>;
    /// Store the computed all-baseline law.
    fn set_baseline_law(&mut self, law: (f64, f64));
    /// Compute `(μ, σ²)` of the hybrid outcome law at `mask`.
    fn law_at(&mut self, mask: u64) -> Result<(f64, f64), AttributionError>;

    /// Shared `CoalitionPayoff::value` body.
    fn cached_payoff_value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        if matches!(self.measure(), DifferenceMeasure::GaussianKl) && self.baseline_law().is_none()
        {
            let (mu0, var0) = self.law_at(0)?;
            self.set_baseline_law((mu0, var0));
            if mask == 0 {
                return Ok(0.0);
            }
        }
        let (mu, var) = self.law_at(mask)?;
        measure_value(self.measure(), mask, mu, var, self.baseline_law())
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
    model_for_paths: Option<&antecedent_model::CompiledCausalModel>,
) -> Result<ChangeAttributionResult, AttributionError> {
    match allocation {
        AllocationMethod::Shapley { approximation } => {
            let estimate = estimate_shapley(players, approximation, payoff, ctx)?;
            Ok(pack_change_result(outcome, total_change, estimate, unidentified, Arc::from([])))
        }
        AllocationMethod::Sequential { order } => {
            if order.len() != players.len() {
                return Err(AttributionError::invalid_input(
                    "sequential allocation order must contain every player exactly once",
                ));
            }
            let mut seen = vec![false; players.len()];
            for &component in order.iter() {
                let index = players
                    .iter()
                    .position(|&player| player == component)
                    .ok_or(AttributionError::UnknownPlayer)?;
                if seen[index] {
                    return Err(AttributionError::invalid_input(
                        "sequential allocation order contains duplicate components",
                    ));
                }
                seen[index] = true;
            }
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
    model: &antecedent_model::CompiledCausalModel,
    outcome: VariableId,
    players: &[ComponentId],
    payoff: &mut P,
    total_change: f64,
    unidentified: Arc<[ComponentId]>,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    use crate::path::path_decompose;
    use crate::result::ComponentContribution;

    // PathBased is a linear-Gaussian path decomposition, not Shapley. Per-player
    // path products need not partition v(N)−v(∅) when ancestries nest, so shares
    // are renormalized to the true total (efficiency). Truncation and nonlinear
    // mechanisms are refused by `path_decompose`.
    let _ = payoff;
    let mut path_breakdown = Vec::new();
    let mut contributions: Vec<ComponentContribution> = Vec::new();
    for &comp in players {
        let res = path_decompose(model, &[comp.variable()], outcome, 64, 16, ctx)?;
        let player_sum: f64 = res.path_breakdown.iter().map(|p| p.contribution).sum();
        contributions.push(ComponentContribution {
            component: comp,
            contribution: player_sum,
            stderr: None,
            ci_low: None,
            ci_high: None,
        });
        path_breakdown.extend(res.path_breakdown.iter().cloned());
    }
    let raw: f64 = contributions.iter().map(|c| c.contribution).sum();
    if raw.abs() > 1e-15 && total_change.is_finite() {
        let scale = total_change / raw;
        for c in &mut contributions {
            c.contribution *= scale;
        }
        for p in &mut path_breakdown {
            p.contribution *= scale;
        }
    }
    let n_paths = path_breakdown.len();
    Ok(ChangeAttributionResult {
        outcome,
        total_change,
        contributions: Arc::from(contributions),
        interactions: Arc::from([]),
        path_breakdown: Arc::from(path_breakdown),
        unidentified,
        graph_sensitivity: None,
        budget: crate::result::ComputeBudget {
            evaluations: u64::try_from(n_paths).unwrap_or(u64::MAX),
            samples: 0,
            exact_coalitions: 0,
        },
        monte_carlo_stderr: None,
        component_mc_stderr: None,
        cache_stats: crate::result::CacheStats::default(),
    })
}

pub(crate) fn pack_change_result(
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

#[cfg(test)]
mod tests {
    use super::*;

    struct AdditivePayoff;

    impl CoalitionPayoff for AdditivePayoff {
        fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
            Ok(f64::from(mask.count_ones()))
        }
    }

    #[test]
    fn sequential_order_must_cover_all_players() {
        let players = [ComponentId::from_raw(1), ComponentId::from_raw(2)];
        let allocation =
            AllocationMethod::Sequential { order: Arc::from([ComponentId::from_raw(1)]) };
        let err = run_change_allocation(
            VariableId::from_raw(9),
            &players,
            &allocation,
            &mut AdditivePayoff,
            2.0,
            Arc::from([]),
            &ExecutionContext::for_tests(1),
            None,
        )
        .unwrap_err();
        assert!(matches!(err, AttributionError::InvalidInput { .. }));
    }

    #[test]
    fn sequential_order_rejects_unknown_player() {
        let players = [ComponentId::from_raw(1), ComponentId::from_raw(2)];
        let allocation = AllocationMethod::Sequential {
            order: Arc::from([ComponentId::from_raw(1), ComponentId::from_raw(3)]),
        };
        let err = run_change_allocation(
            VariableId::from_raw(9),
            &players,
            &allocation,
            &mut AdditivePayoff,
            2.0,
            Arc::from([]),
            &ExecutionContext::for_tests(1),
            None,
        )
        .unwrap_err();
        assert_eq!(err, AttributionError::UnknownPlayer);
    }
}
