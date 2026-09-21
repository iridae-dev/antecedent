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

    // PathBased is the O(n) analogue of Shapley: instead of averaging each
    // player's marginal contribution over every coalition (O(2^n) payoff
    // evaluations), it uses only the two single-player coalitions v(∅) and
    // v({i}) — the same baseline/comparison mechanism-swap `payoff` that
    // `AllocationMethod::Shapley` uses on this call site. `v({i}) − v(∅)` is the
    // effect of swapping *only* player i's mechanism from baseline to
    // comparison; a player whose fitted mechanism is unchanged between the two
    // populations therefore scores (near) exactly zero, regardless of how
    // strong its paths to the outcome are — the defect this replaces.
    //
    // This is exact whenever mechanism shifts act additively on the payoff
    // (e.g. `MeanDiff` on a linear-Gaussian model with no player sharing a
    // descendant with another player); with interacting shifts the marginals
    // need not already sum to `total_change`, so — exactly as the previous
    // implementation did — they are rescaled to the measured total
    // (efficiency), which corrects *scale* only: a truly unchanged player's
    // zero marginal survives rescaling untouched as long as some other player
    // actually moved.
    //
    // `path_decompose`'s static path-coefficient products are used only to
    // split a player's already-correct total across its individual directed
    // paths to the outcome for reporting; they never determine how much of the
    // total change that player receives.
    let v0 = payoff.value(0)?;
    let mut raw_players = Vec::with_capacity(players.len());
    for i in 0..players.len() {
        let bit = 1u64 << i;
        let v_i = payoff.value(bit)?;
        raw_players.push(v_i - v0);
    }

    let mut player_paths = Vec::with_capacity(players.len());
    for &comp in players {
        let res = path_decompose(model, &[comp.variable()], outcome, 64, 16, ctx)?;
        player_paths.push(res.path_breakdown.to_vec());
    }

    let raw: f64 = raw_players.iter().sum();
    let scale = path_efficiency_scale(raw, total_change)?;

    let mut contributions: Vec<ComponentContribution> = Vec::with_capacity(players.len());
    let mut path_breakdown = Vec::new();
    let mut n_evaluations = players.len() as u64 + 1;
    for (idx, &comp) in players.iter().enumerate() {
        let mut contribution = raw_players[idx];
        if let Some(scale) = scale {
            contribution *= scale;
        }
        contributions.push(ComponentContribution {
            component: comp,
            contribution,
            stderr: None,
            ci_low: None,
            ci_high: None,
        });
        // Apportion this player's mechanism-shift contribution across its
        // paths using static path-coefficient shares — a reporting split
        // only: it redistributes `contribution` among the player's paths and
        // does not change the player's total or the grand total.
        let paths = &player_paths[idx];
        let path_raw_total: f64 = paths.iter().map(|p| p.contribution).sum();
        for p in paths {
            n_evaluations += 1;
            let share = if path_raw_total.abs() > 1e-15 {
                p.contribution / path_raw_total
            } else if paths.is_empty() {
                0.0
            } else {
                1.0 / paths.len() as f64
            };
            path_breakdown.push(crate::result::PathContribution {
                path: Arc::clone(&p.path),
                contribution: contribution * share,
            });
        }
    }
    Ok(ChangeAttributionResult {
        outcome,
        total_change,
        contributions: Arc::from(contributions),
        interactions: Arc::from([]),
        path_breakdown: Arc::from(path_breakdown),
        unidentified,
        graph_sensitivity: None,
        budget: crate::result::ComputeBudget {
            evaluations: n_evaluations,
            samples: 0,
            exact_coalitions: 0,
        },
        monte_carlo_stderr: None,
        component_mc_stderr: None,
        cache_stats: crate::result::CacheStats::default(),
    })
}

/// Scale signed path products to the measured total without hiding an undefined allocation.
fn path_efficiency_scale(raw: f64, total_change: f64) -> Result<Option<f64>, AttributionError> {
    const ZERO_TOL: f64 = 1e-15;
    if !raw.is_finite() || !total_change.is_finite() {
        return Err(AttributionError::invalid_input(
            "PathBased allocation requires finite path shares and total change",
        ));
    }
    if raw.abs() <= ZERO_TOL {
        if total_change.abs() <= ZERO_TOL {
            return Ok(None);
        }
        return Err(AttributionError::unsupported(
            "PathBased signed path shares cancel to zero and cannot allocate a nonzero total change",
        ));
    }
    Ok(Some(total_change / raw))
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

    #[test]
    fn path_efficiency_refuses_zero_share_for_nonzero_change() {
        let err = path_efficiency_scale(0.0, 2.0).unwrap_err();
        assert!(matches!(err, AttributionError::Unsupported { .. }));
    }

    #[test]
    fn path_efficiency_allows_zero_share_for_zero_change() {
        assert_eq!(path_efficiency_scale(0.0, 0.0).unwrap(), None);
        assert_eq!(path_efficiency_scale(2.0, 6.0).unwrap(), Some(3.0));
    }
}
