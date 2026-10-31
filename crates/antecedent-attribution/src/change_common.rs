//! Shared change-attribution allocation / measure helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{AllocationMethod, ComponentId, ExecutionContext, StreamDomain, VariableId};
use antecedent_graph::DenseNodeId;
use antecedent_model::{CompiledCausalModel, MechanismWorkspace, sample_observational_into};
use antecedent_stats::{gaussian_kl, mean_var};

use crate::coalition::Coalition;
use crate::error::AttributionError;
use crate::result::ChangeAttributionResult;
use crate::shapley::{CoalitionPayoff, ShapleyEstimate, estimate_shapley, sequential_allocate};

/// How to summarize the target marginal difference.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum DifferenceMeasure {
    /// `E[Y_comparison-like] − E[Y_baseline-like]`.
    MeanDiff,
    /// Difference in population variances, including zero or arbitrarily small variances.
    VarianceDiff,
    /// Gaussian KL `KL(N(μ_S, σ_S²) ‖ N(μ₀, σ₀²))` of the hybrid outcome law vs the
    /// all-baseline coalition (the default target functional in Budhathoki, Janzing,
    /// Bloebaum & Ng 2021). Both Gaussian variances must be finite and positive;
    /// no variance regularization is applied.
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

/// Validate a per-coalition Monte Carlo sample count before any sampling work.
pub(crate) fn validate_change_sample_count(
    measure: DifferenceMeasure,
    n_samples: usize,
) -> Result<(), AttributionError> {
    let minimum = match measure {
        DifferenceMeasure::MeanDiff => 1,
        DifferenceMeasure::VarianceDiff | DifferenceMeasure::GaussianKl => 2,
    };
    if n_samples < minimum {
        return Err(AttributionError::invalid_input(
            "change attribution requires at least one draw for a mean and two draws for variance or Gaussian KL",
        ));
    }
    Ok(())
}

/// Stream domain of the distribution-change coalition payoff.
pub(crate) const DISTRIBUTION_STREAM: u64 = 0xDC01;
/// Stream domain of the structure-change coalition payoff.
pub(crate) const STRUCTURE_STREAM: u64 = 0x5C01;
/// Stream domain of the per-unit Shapley permutation seeds.
pub(crate) const UNIT_STREAM: u64 = 0x0117;

/// Stream domain of the Shapley permutation sampler.
pub(crate) const SHAPLEY_STREAM: u64 = 0x5A91;
/// Stream domain of the robust (fit-uncertainty) linear and mechanism payoffs.
pub(crate) const ROBUST_STREAM: u64 = 0x80B7;
/// Stream domain of the kernel two-sample permutation null.
pub(crate) const KERNEL_STREAM: u64 = 0x4E12;

/// Stream tag for `(domain, seed)`.
///
/// A `SplitMix64` finalizer over the seed shifted by the domain, so streams of different
/// domains cannot alias through overlapping `domain ^ seed` / `domain + seed` ranges (a
/// bare XOR maps `(DC, s ^ k)` onto `(SC, s)` for the constant `k = DC ^ SC`). For a fixed
/// domain the map is a bijection of the seed, so distinct seeds keep distinct streams.
#[must_use]
pub(crate) fn stream_tag(domain: u64, seed: u64) -> u64 {
    antecedent_core::mix64(seed.wrapping_add(domain.wrapping_mul(antecedent_core::GOLDEN_GAMMA)))
}

/// The context's attribution stream for `(domain, seed)`.
///
/// Every attribution Monte Carlo arm draws through this one function, so the result
/// depends on the context's master seed and `domain`s cannot alias through a shared
/// additive offset.
#[must_use]
pub(crate) fn attribution_stream(
    ctx: &ExecutionContext,
    domain: u64,
    seed: u64,
) -> antecedent_core::CausalRng {
    ctx.rng.stream_for(StreamDomain::Attribution, stream_tag(domain, seed))
}

/// Mean and population variance of the outcome under `model`, from `n_samples` ancestral
/// draws on the `(domain, seed)` attribution stream.
///
/// Every coalition of a change attribution uses the same stream, so coalition values share
/// their exogenous noise (common random numbers) and differences between coalitions are
/// not swamped by sampling noise. `buf` is reused across calls.
pub(crate) fn sample_outcome_law(
    model: &CompiledCausalModel,
    outcome: DenseNodeId,
    n_samples: usize,
    stream: u64,
    ctx: &ExecutionContext,
    ws: &mut MechanismWorkspace,
    buf: &mut Vec<f64>,
) -> Result<(f64, f64), AttributionError> {
    let mut rng = ctx.rng.stream_for(StreamDomain::Attribution, stream);
    if n_samples == 0 {
        return Err(AttributionError::invalid_input(
            "change attribution requires at least one draw",
        ));
    }
    let n_rows = n_samples;
    let need = n_rows.saturating_mul(model.n_nodes());
    if buf.len() < need {
        buf.resize(need, 0.0);
    }
    sample_observational_into(model, n_rows, &mut rng, ws, &mut buf[..need], ctx)?;
    let start = outcome.as_usize() * n_rows;
    let (mu, var) = mean_var(&buf[start..start + n_rows]);
    Ok((mu, var))
}

/// The outcome-summary change between two observed samples under `measure`
/// (`comparison` relative to `baseline`), on the same scale as the coalition payoff:
/// difference of means, difference of (population) variances, or
/// `KL(N(μ_c, σ_c²) ‖ N(μ_b, σ_b²))` for positive variances.
pub(crate) fn observed_change(
    measure: DifferenceMeasure,
    baseline: &[f64],
    comparison: &[f64],
) -> Result<f64, AttributionError> {
    if baseline.is_empty() || comparison.is_empty() {
        return Err(AttributionError::invalid_input("observed change needs non-empty populations"));
    }
    let (mu_b, var_b) = mean_var(baseline);
    let (mu_c, var_c) = mean_var(comparison);
    match measure {
        DifferenceMeasure::MeanDiff => Ok(mu_c - mu_b),
        DifferenceMeasure::VarianceDiff => Ok(var_c - var_b),
        DifferenceMeasure::GaussianKl => Ok(gaussian_kl(mu_c, var_c, mu_b, var_b)?),
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
    coalition: &Coalition,
    mu: f64,
    var: f64,
    baseline_law: Option<(f64, f64)>,
) -> Result<f64, AttributionError> {
    match measure {
        DifferenceMeasure::MeanDiff => Ok(mu),
        DifferenceMeasure::VarianceDiff => Ok(var),
        DifferenceMeasure::GaussianKl => {
            if coalition.is_empty() {
                gaussian_kl(mu, var, mu, var)?;
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

/// Outcome laws `(mean, variance)` already computed by a hybrid-outcome-law payoff, by coalition.
///
/// The law of a coalition is a pure function of it (every sampling stream restarts at the
/// same key on each call, which is what makes the coalition values common random numbers), so
/// the Monte Carlo allocations, which revisit the same few coalitions thousands of times,
/// can reuse it without changing a bit of any value. The memo is bounded and holds only
/// successful evaluations; once it is full a new coalition is computed and not kept.
#[derive(Clone, Debug, Default)]
pub(crate) struct LawMemo {
    laws: HashMap<Coalition, (f64, f64)>,
}

impl LawMemo {
    /// Most coalitions kept: 4096 entries of a coalition key and two floats.
    const MAX_ENTRIES: usize = 4096;

    fn get(&self, coalition: &Coalition) -> Option<(f64, f64)> {
        self.laws.get(coalition).copied()
    }

    fn insert(&mut self, coalition: &Coalition, law: (f64, f64)) {
        if self.laws.len() < Self::MAX_ENTRIES {
            self.laws.insert(coalition.clone(), law);
        }
    }
}

/// Shared [`CoalitionPayoff::value`] caching shape for hybrid-outcome-law payoffs.
///
/// Both `MechanismSwapPayoff` ([`crate::distribution_change`]) and
/// `StructureSwapPayoff` ([`crate::structure_change`]) memoize the mask-0
/// all-baseline outcome law once (only needed for [`DifferenceMeasure::GaussianKl`])
/// and then evaluate the coalition's difference measure; they differ only
/// in how they compute the outcome law itself ([`Self::law_at`]).
pub(crate) trait CachedOutcomeLawPayoff {
    /// The payoff's memo of outcome laws already computed.
    fn law_memo(&mut self) -> &mut LawMemo;
    /// The configured difference measure.
    fn measure(&self) -> DifferenceMeasure;
    /// The cached all-baseline `(μ₀, σ₀²)`, once computed.
    fn baseline_law(&self) -> Option<(f64, f64)>;
    /// Store the computed all-baseline law.
    fn set_baseline_law(&mut self, law: (f64, f64));
    /// Compute `(μ, σ²)` of the hybrid outcome law of `coalition`.
    fn law_at(&mut self, coalition: &Coalition) -> Result<(f64, f64), AttributionError>;

    /// Shared `CoalitionPayoff::value` body.
    fn cached_payoff_value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
        if matches!(self.measure(), DifferenceMeasure::GaussianKl) && self.baseline_law().is_none()
        {
            let (mu0, var0) = self.law_at(&Coalition::empty())?;
            gaussian_kl(mu0, var0, mu0, var0)?;
            self.set_baseline_law((mu0, var0));
            if coalition.is_empty() {
                return Ok(0.0);
            }
        }
        let (mu, var) = match self.law_memo().get(coalition) {
            Some(law) => law,
            None => {
                let law = self.law_at(coalition)?;
                self.law_memo().insert(coalition, law);
                law
            }
        };
        measure_value(self.measure(), coalition, mu, var, self.baseline_law())
    }
}

/// What a change allocation divides, among whom, and by which method.
#[derive(Clone, Copy)]
pub(crate) struct ChangeAllocation<'a> {
    pub outcome: VariableId,
    pub players: &'a [ComponentId],
    pub allocation: &'a AllocationMethod,
    /// Model whose graph orders path-based allocation.
    pub model_for_paths: Option<&'a antecedent_model::CompiledCausalModel>,
}

/// Run Shapley / sequential / path-based allocation and pack a [`ChangeAttributionResult`].
pub(crate) fn run_change_allocation<P: CoalitionPayoff>(
    request: &ChangeAllocation<'_>,
    payoff: &mut P,
    total_change: f64,
    unidentified: Arc<[ComponentId]>,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    let ChangeAllocation { outcome, players, allocation, model_for_paths } = *request;
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
    let v0 = payoff.value(&Coalition::empty())?;
    let mut raw_players = Vec::with_capacity(players.len());
    for i in 0..players.len() {
        let v_i = payoff.value(&Coalition::empty().with(i))?;
        raw_players.push(v_i - v0);
    }

    // One coefficient map for every player. Rebuilding it per player repeated the
    // same linear read; the path products below are unchanged.
    if ctx.cancellation.is_cancelled() {
        return Err(AttributionError::Cancelled);
    }
    let strengths = crate::path::edge_strength_map(model)?;
    let mut player_paths = Vec::with_capacity(players.len());
    for &comp in players {
        let res =
            crate::path::path_shares(model, &[comp.variable()], outcome, 64, 16, ctx, &strengths)?;
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
        observed_change: None,
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
        fit_uncertainty: None,
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
        observed_change: None,
        contributions,
        interactions,
        path_breakdown,
        unidentified,
        graph_sensitivity: None,
        budget,
        monte_carlo_stderr: mc_stderr,
        component_mc_stderr: component_mc,
        cache_stats,
        fit_uncertainty: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AdditivePayoff;

    impl CoalitionPayoff for AdditivePayoff {
        fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
            Ok(coalition.len() as f64)
        }
    }

    #[test]
    fn sequential_order_must_cover_all_players() {
        let players = [ComponentId::from_raw(1), ComponentId::from_raw(2)];
        let allocation =
            AllocationMethod::Sequential { order: Arc::from([ComponentId::from_raw(1)]) };
        let request = ChangeAllocation {
            outcome: VariableId::from_raw(9),
            players: &players,
            allocation: &allocation,
            model_for_paths: None,
        };
        let err = run_change_allocation(
            &request,
            &mut AdditivePayoff,
            2.0,
            Arc::from([]),
            &ExecutionContext::for_tests(1),
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
        let request = ChangeAllocation {
            outcome: VariableId::from_raw(9),
            players: &players,
            allocation: &allocation,
            model_for_paths: None,
        };
        let err = run_change_allocation(
            &request,
            &mut AdditivePayoff,
            2.0,
            Arc::from([]),
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert_eq!(err, AttributionError::UnknownPlayer);
    }

    #[test]
    fn stream_tags_do_not_alias_across_domains_or_seeds() {
        // A bare XOR of the domain constants maps (DISTRIBUTION, s ^ k) onto (STRUCTURE, s)
        // for k = DC01 ^ 5C01.
        let k = DISTRIBUTION_STREAM ^ STRUCTURE_STREAM;
        for s in [0u64, 1, 5, 0x8000, u64::MAX] {
            assert_ne!(
                stream_tag(DISTRIBUTION_STREAM, s ^ k),
                stream_tag(STRUCTURE_STREAM, s),
                "seed {s:#x}"
            );
            assert_ne!(
                stream_tag(DISTRIBUTION_STREAM, s.wrapping_add(0x8000)),
                stream_tag(STRUCTURE_STREAM, s),
                "seed {s:#x}"
            );
        }
        // Distinct seeds within one domain stay distinct (bijection), including neighbours.
        let tags: std::collections::HashSet<u64> =
            (0..10_000u64).map(|s| stream_tag(UNIT_STREAM, s)).collect();
        assert_eq!(tags.len(), 10_000);
    }

    #[test]
    fn observed_change_matches_closed_forms() {
        let base = [1.0, 2.0, 3.0]; // mean 2, population variance 2/3
        let cmp = [2.0, 4.0, 6.0]; // mean 4, population variance 8/3
        assert!(
            (observed_change(DifferenceMeasure::MeanDiff, &base, &cmp).unwrap() - 2.0).abs()
                < 1e-12
        );
        assert!(
            (observed_change(DifferenceMeasure::VarianceDiff, &base, &cmp).unwrap() - 2.0).abs()
                < 1e-12
        );
        // KL(N(4, 8/3) ‖ N(2, 2/3)) = ½ (σc²/σb² − 1 + ln(σb²/σc²) + (μc − μb)²/σb²)
        //                          = ½ (4 − 1 + ln(1/4) + 4/(2/3)) = ½ (3 − ln 4 + 6).
        let expected = 0.5 * (3.0 - 4.0_f64.ln() + 6.0);
        let got = observed_change(DifferenceMeasure::GaussianKl, &base, &cmp).unwrap();
        assert!((got - expected).abs() < 1e-12, "got={got} expected={expected}");
        assert!(observed_change(DifferenceMeasure::MeanDiff, &[], &cmp).is_err());
    }

    #[test]
    fn sample_count_respects_measure() {
        for measure in [
            DifferenceMeasure::MeanDiff,
            DifferenceMeasure::VarianceDiff,
            DifferenceMeasure::GaussianKl,
        ] {
            assert!(matches!(
                validate_change_sample_count(measure, 0),
                Err(AttributionError::InvalidInput { .. })
            ));
            assert!(validate_change_sample_count(measure, 2).is_ok());
        }
        assert!(validate_change_sample_count(DifferenceMeasure::MeanDiff, 1).is_ok());
        assert!(validate_change_sample_count(DifferenceMeasure::VarianceDiff, 1).is_err());
        assert!(validate_change_sample_count(DifferenceMeasure::GaussianKl, 1).is_err());
    }

    #[test]
    fn tiny_and_degenerate_variances_are_not_regularized() {
        let empty = Coalition::empty();
        let full = Coalition::from_bits(1);
        let tiny = 1e-14;
        let larger = 4e-14;
        let v0 = measure_value(DifferenceMeasure::VarianceDiff, &empty, 0.0, tiny, None).unwrap();
        let v1 = measure_value(DifferenceMeasure::VarianceDiff, &full, 0.0, larger, None).unwrap();
        assert!((v1 - v0 - 3e-14).abs() < 1e-28);
        assert_eq!(
            measure_value(DifferenceMeasure::VarianceDiff, &empty, 0.0, 0.0, None).unwrap(),
            0.0
        );
        let kl =
            measure_value(DifferenceMeasure::GaussianKl, &full, 0.0, larger, Some((0.0, tiny)))
                .unwrap();
        let expected = 0.5 * (3.0 - 4.0_f64.ln());
        assert!((kl - expected).abs() < 1e-12);
        assert!(matches!(
            measure_value(DifferenceMeasure::GaussianKl, &empty, 0.0, 0.0, None),
            Err(AttributionError::Stats(_))
        ));
        assert!(matches!(
            measure_value(DifferenceMeasure::GaussianKl, &full, 1.0, 0.0, Some((0.0, tiny))),
            Err(AttributionError::Stats(_))
        ));
        assert!(matches!(
            observed_change(DifferenceMeasure::GaussianKl, &[0.0, 0.0], &[1.0, 1.0]),
            Err(AttributionError::Stats(_))
        ));
        assert_eq!(observed_change(DifferenceMeasure::VarianceDiff, &[1.0], &[2.0]).unwrap(), 0.0);
    }

    #[test]
    fn sampled_tiny_variance_remains_visible() {
        use antecedent_graph::Dag;
        use antecedent_model::{CompiledMechanismStore, MechanismSlot};
        let model = CompiledCausalModel::compile(Dag::with_variables(1)).unwrap();
        let make = |sigma| {
            model.clone().with_mechanisms(CompiledMechanismStore {
                slots: Arc::from([MechanismSlot::LinearGaussian {
                    intercept: 0.0,
                    coeffs: Arc::from([]),
                    sigma,
                }]),
            })
        };
        let ctx = ExecutionContext::for_tests(7);
        let mut ws = MechanismWorkspace::default();
        let mut buf = Vec::new();
        let sample = |model: &CompiledCausalModel,
                      ws: &mut MechanismWorkspace,
                      buf: &mut Vec<f64>| {
            sample_outcome_law(model, DenseNodeId::from_raw(0), 128, 17, &ctx, ws, buf).unwrap().1
        };
        let a = sample(&make(1e-7), &mut ws, &mut buf);
        let b = sample(&make(2e-7), &mut ws, &mut buf);
        assert!(0.0 < a && a < 1e-12);
        assert!((b / a - 4.0).abs() < 1e-10);
        assert!(b > a);
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
