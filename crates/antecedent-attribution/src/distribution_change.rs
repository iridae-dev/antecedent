//! Distribution-change attribution (Budhathoki, Janzing, Bloebaum & Ng 2021).
//!
//! Fits mechanisms on baseline and comparison populations, then attributes the
//! change in the outcome marginal to mechanism replacements via Shapley values
//! (Budhathoki et al. 2021).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AllocationMethod, AttributionComponents, ChangeAttributionQuery, ComponentId, ExecutionContext,
    ShapleyConfig, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{BitSet, DenseNodeId, GraphWorkspace};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismSlot,
    MechanismWorkspace, SelectionPolicy, sample_observational_into,
};
use antecedent_stats::mean_var;

use crate::change_common::{ChangeOptions, run_change_allocation, total_change};
use crate::coalition::full_coalition_mask;
use crate::error::AttributionError;
use crate::prep::{require_mechanism_or_joint, resolve_change_populations, resolve_outcome_dense};
use crate::result::ChangeAttributionResult;
use crate::shapley::CoalitionPayoff;

pub use crate::change_common::DifferenceMeasure;

/// Options for distribution-change attribution.
#[derive(Clone, Debug)]
pub struct DistributionChangeOptions {
    /// Difference measure on the outcome samples.
    pub measure: DifferenceMeasure,
    /// Samples drawn per coalition evaluation.
    pub n_samples: usize,
    /// RNG seed for sampling.
    pub seed: u64,
}

impl Default for DistributionChangeOptions {
    fn default() -> Self {
        let o = ChangeOptions::default_mean();
        Self { measure: o.measure, n_samples: o.n_samples, seed: o.seed }
    }
}

/// Attribute distributional change between baseline and comparison populations.
///
/// `graph_model` supplies structure; mechanisms are fit separately on each
/// population subset. Only mechanism components that are ancestors of the
/// outcome (inclusive) participate, unless `query.components` restricts further.
///
/// # Errors
///
/// Query validation, fit/sample failures, or Shapley size limits.
pub fn distribution_change(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &DistributionChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    query.validate()?;
    require_mechanism_or_joint(query.components)?;
    if matches!(query.components, AttributionComponents::All) {
        return Err(AttributionError::unsupported(
            "AttributionComponents::All requires dual graphs; use ChangeAttribution::run_structure \
             for Structure, or InputsAndMechanisms for joint input+mechanism change",
        ));
    }

    let (baseline_data, comparison_data) = resolve_change_populations(data, query)?;

    let (baseline_mechs, _) = MechanismRegistry::standard().assign_and_fit(
        graph_model,
        &baseline_data,
        SelectionPolicy::BestScore,
    )?;
    let (comparison_mechs, _) = MechanismRegistry::standard().assign_and_fit(
        graph_model,
        &comparison_data,
        SelectionPolicy::BestScore,
    )?;

    let outcome_dense = resolve_outcome_dense(graph_model, query.outcome)?;

    let (players, player_kinds) =
        joint_players(graph_model, outcome_dense, query.max_components, query.components)?;
    if players.is_empty() {
        return Err(AttributionError::invalid_input("no components to attribute"));
    }
    crate::shapley::check_coalition_sample_budget(
        players.len(),
        &query.allocation,
        options.n_samples,
    )?;

    // Player → dense-node mapping hoisted out of the per-coalition path (was an
    // O(n_nodes) `dense_of` scan per player per coalition).
    let player_dense: Vec<Option<DenseNodeId>> =
        players.iter().map(|c| graph_model.dense_of(c.variable())).collect();
    // Persistent slot scratch: starts at the all-baseline (mask 0) store and is
    // patched incrementally between masks.
    let slot_scratch: Vec<MechanismSlot> = baseline_mechs.slots.to_vec();

    let mut payoff = MechanismSwapPayoff {
        template: graph_model.clone(),
        baseline: baseline_mechs,
        comparison: comparison_mechs,
        player_kinds,
        player_dense,
        slot_scratch,
        scratch_mask: 0,
        outcome: outcome_dense,
        measure: options.measure,
        n_samples: options.n_samples,
        seed: options.seed,
        ctx,
        ws: MechanismWorkspace::default(),
        values_buf: Vec::new(),
        baseline_law: None,
    };

    let v0 = payoff.value(0)?;
    let full_mask = full_coalition_mask(players.len())?;
    let v_full = payoff.value(full_mask)?;
    let total = total_change(options.measure, v0, v_full);

    run_change_allocation(
        query.outcome,
        &players,
        &query.allocation,
        &mut payoff,
        total,
        Arc::from([]),
        ctx,
        Some(graph_model),
    )
}

/// Convenience: Shapley Monte Carlo distribution-change with defaults.
///
/// # Errors
///
/// See [`distribution_change`].
pub fn distribution_change_shapley(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    outcome: VariableId,
    baseline: antecedent_core::PopulationSelector,
    comparison: antecedent_core::PopulationSelector,
    shapley: ShapleyConfig,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    let query = ChangeAttributionQuery::new(outcome, baseline, comparison)
        .with_components(AttributionComponents::Mechanisms)
        .with_allocation(AllocationMethod::Shapley { approximation: shapley });
    distribution_change(graph_model, data, &query, &DistributionChangeOptions::default(), ctx)
}

pub(crate) fn mechanism_players(
    model: &CompiledCausalModel,
    outcome: DenseNodeId,
    max_components: usize,
) -> Result<Vec<ComponentId>, AttributionError> {
    let (players, _) =
        joint_players(model, outcome, max_components, AttributionComponents::Mechanisms)?;
    Ok(players)
}

/// Why a node is a Shapley player in joint change attribution.
///
/// This records provenance, not behavior: on the [`distribution_change`] path every player
/// is realized the same way — a coalition bit swaps that node's fitted mechanism. For a root
/// the fitted mechanism *is* its marginal, so a mechanism swap already expresses an input
/// change; for a non-root, swapping the conditional is the only intervention that keeps the
/// causal factorization intact.
///
/// [`Input`](Self::Input) is currently unreachable here:
/// [`require_mechanism_or_joint`] rejects [`AttributionComponents::Inputs`] before
/// [`joint_players`] runs (that component set routes to `unit_change` instead), and
/// [`AttributionComponents::All`] is rejected too. Only `Mechanism` and `Both` occur.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PlayerKind {
    /// Node has a fitted mechanism and is an ancestor of the outcome.
    Mechanism,
    /// Outcome parent reached without a mechanism player. Unreachable on this path.
    Input,
    /// Outcome parent that is also a mechanism player.
    Both,
}

pub(crate) fn joint_players(
    model: &CompiledCausalModel,
    outcome: DenseNodeId,
    max_components: usize,
    components: AttributionComponents,
) -> Result<(Vec<ComponentId>, Vec<PlayerKind>), AttributionError> {
    let mut ws = GraphWorkspace::default();
    let mut anc = BitSet::default();
    model.graph.ancestors_of(&[outcome], &mut anc, &mut ws);

    let mut players = Vec::new();
    let mut kinds = Vec::new();

    if matches!(
        components,
        AttributionComponents::Mechanisms
            | AttributionComponents::InputsAndMechanisms
            | AttributionComponents::All
    ) {
        for gather in model.parent_gathers.iter() {
            let node = gather.child;
            if !anc.contains(node) {
                continue;
            }
            let var = model.output_layout.variables[node.as_usize()];
            players.push(ComponentId::from_variable(var));
            kinds.push(PlayerKind::Mechanism);
        }
    }

    if matches!(
        components,
        AttributionComponents::Inputs
            | AttributionComponents::InputsAndMechanisms
            | AttributionComponents::All
    ) {
        if let Some(gather) = model.gather_for(outcome) {
            for &p in gather.parents.iter() {
                let var = model.output_layout.variables[p.as_usize()];
                let comp = ComponentId::from_variable(var);
                if let Some(idx) = players.iter().position(|&c| c == comp) {
                    kinds[idx] = PlayerKind::Both;
                } else {
                    players.push(comp);
                    kinds.push(PlayerKind::Input);
                }
            }
        }
    }

    if players.len() > max_components {
        return Err(AttributionError::SizeLimit {
            kind: "components",
            requested: players.len(),
            max: max_components,
        });
    }
    Ok((players, kinds))
}

struct MechanismSwapPayoff<'a> {
    template: CompiledCausalModel,
    baseline: CompiledMechanismStore,
    comparison: CompiledMechanismStore,
    player_kinds: Vec<PlayerKind>,
    /// Dense node per player, hoisted at construction.
    player_dense: Vec<Option<DenseNodeId>>,
    /// Persistent hybrid-slot scratch reflecting `scratch_mask`: baseline slots
    /// everywhere except comparison slots for set mechanism-player bits.
    slot_scratch: Vec<MechanismSlot>,
    /// Mask currently applied to `slot_scratch`.
    scratch_mask: u64,
    outcome: DenseNodeId,
    measure: DifferenceMeasure,
    n_samples: usize,
    seed: u64,
    ctx: &'a ExecutionContext,
    ws: MechanismWorkspace,
    /// Reused ancestral-sample buffer (`n_samples × n_nodes`).
    values_buf: Vec<f64>,
    /// Cached `(μ₀, σ₀²)` of the all-baseline outcome law for KL payoffs.
    baseline_law: Option<(f64, f64)>,
}

impl crate::change_common::CachedOutcomeLawPayoff for MechanismSwapPayoff<'_> {
    fn measure(&self) -> DifferenceMeasure {
        self.measure
    }

    fn baseline_law(&self) -> Option<(f64, f64)> {
        self.baseline_law
    }

    fn set_baseline_law(&mut self, law: (f64, f64)) {
        self.baseline_law = Some(law);
    }

    fn law_at(&mut self, mask: u64) -> Result<(f64, f64), AttributionError> {
        self.sample_outcome_law(mask)
    }
}

impl CoalitionPayoff for MechanismSwapPayoff<'_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        crate::change_common::CachedOutcomeLawPayoff::cached_payoff_value(self, mask)
    }
}

impl MechanismSwapPayoff<'_> {
    /// Outcome law under the hybrid model selected by `mask`.
    ///
    /// A coalition bit means "use this player's comparison-fitted mechanism"; everything
    /// else stays at baseline. That mechanism swap is the *only* lever, and deliberately so.
    ///
    /// This previously also hard-set every `Input`/`Both` player to its column mean via
    /// `Intervention::set`, which `sample_with_overlay` realizes as `out.fill(v)` before
    /// mechanism sampling, then `continue`s. Two consequences, both wrong: the swapped
    /// mechanism for a `Both` player was never read (dead code), and the player's whole
    /// distribution collapsed to a point mass, so a regime difference that preserved the
    /// mean — a variance shift, a shape change — produced identical coalition values and was
    /// attributed exactly zero.
    fn sample_outcome_law(&mut self, mask: u64) -> Result<(f64, f64), AttributionError> {
        // Patch the persistent slot scratch incrementally: only bits that changed
        // since the previous mask are touched (restore to baseline on clear, swap
        // in the comparison slot on set). `Input`-kind players never swap slots,
        // exactly as before; the resulting store is value-identical to rebuilding
        // the full hybrid from baseline for every coalition.
        let diff = mask ^ self.scratch_mask;
        for (i, dense) in self.player_dense.iter().enumerate() {
            if diff & (1u64 << i) == 0 || matches!(self.player_kinds[i], PlayerKind::Input) {
                continue;
            }
            let Some(d) = dense else { continue };
            let idx = d.as_usize();
            let src = if mask & (1u64 << i) != 0 { &self.comparison } else { &self.baseline };
            self.slot_scratch[idx] = src.slots[idx].clone();
        }
        self.scratch_mask = mask;
        let store = CompiledMechanismStore { slots: self.slot_scratch.iter().cloned().collect() };

        let model = self.template.clone().with_mechanisms(store);
        let mut rng = self.ctx.rng.stream(0xDC01_u64.wrapping_add(self.seed));
        let n_rows = self.n_samples.max(1);
        let n_nodes = model.n_nodes();
        let need = n_rows.saturating_mul(n_nodes);
        if self.values_buf.len() < need {
            self.values_buf.resize(need, 0.0);
        }
        sample_observational_into(
            &model,
            n_rows,
            &mut rng,
            &mut self.ws,
            &mut self.values_buf[..need],
            self.ctx,
        )?;
        let start = self.outcome.as_usize() * n_rows;
        let col = &self.values_buf[start..start + n_rows];
        let (mu, var) = mean_var(col);
        Ok((mu, var.max(1e-12)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change_common::measure_value;
    use antecedent_core::{
        AllocationMethod, AttributionComponents, CachePolicy, CausalSchemaBuilder, MeasurementSpec,
        PopulationSelector, RoleHint, ShapleyConfig, SmallRoleSet, ToleranceClass, ValueType,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
    use antecedent_graph::{Dag, DenseNodeId};
    use antecedent_model::{MechanismRegistry, SelectionPolicy};
    use serde::Deserialize;

    fn two_period_chain() -> (CompiledCausalModel, TabularData) {
        // X → Y; baseline Y = X; comparison Y = X + 5 (mechanism change on Y only).
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i % 40) as f64 * 0.1;
            xv.push(x);
            if i < 40 {
                yv.push(1.0 + 2.0 * x);
            } else {
                yv.push(6.0 + 2.0 * x); // +5 intercept shift
            }
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        (model, data)
    }

    #[test]
    fn attributes_mechanism_shift_to_y() {
        #[derive(Deserialize)]
        struct Fixture {
            cases: Vec<Case>,
            comparison: Comparison,
        }
        #[derive(Deserialize)]
        struct Case {
            id: String,
            total_change: f64,
        }
        #[derive(Deserialize)]
        struct Comparison {
            sampled_absolute_tolerance: f64,
        }
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../conformance/attribution/distribution_change_grid/expected.json"
        ))
        .unwrap();
        let expected =
            fixture.cases.iter().find(|case| case.id == "y_intercept_plus_five").unwrap();
        let (model, data) = two_period_chain();
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cache_policy = CachePolicy::enabled(Some(1_000_000));
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 400,
            seed: 3,
        };
        let result = distribution_change(&model, &data, &query, &opts, &ctx).unwrap();
        assert!(
            (result.total_change - expected.total_change).abs()
                <= fixture.comparison.sampled_absolute_tolerance,
            "total={} expected={}",
            result.total_change,
            expected.total_change
        );
        let y_contrib = result
            .contributions
            .iter()
            .find(|c| c.component.variable() == VariableId::from_raw(1))
            .expect("y component");
        let x_contrib = result
            .contributions
            .iter()
            .find(|c| c.component.variable() == VariableId::from_raw(0))
            .map_or(0.0, |c| c.contribution);
        assert!(
            y_contrib.contribution.abs() > x_contrib.abs(),
            "y={} x={} all={:?}",
            y_contrib.contribution,
            x_contrib,
            result.contributions
        );
        // Exact Shapley efficiency: Σφ = total_change (payoff uses CRN across coalitions).
        let phi_sum: f64 = result.contributions.iter().map(|c| c.contribution).sum();
        assert!(
            (phi_sum - result.total_change).abs() < 1e-6
                || ToleranceClass::MonteCarlo.close(phi_sum, result.total_change),
            "efficiency: Σφ={phi_sum} total={}",
            result.total_change
        );
    }

    #[test]
    fn exact_shapley_efficiency_sum_phi_equals_total_change() {
        let (model, data) = two_period_chain();
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cache_policy = CachePolicy::enabled(Some(1_000_000));
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 800,
            seed: 11,
        };
        let result = distribution_change(&model, &data, &query, &opts, &ctx).unwrap();
        let phi_sum: f64 = result.contributions.iter().map(|c| c.contribution).sum();
        assert!(
            (phi_sum - result.total_change).abs() < 1e-6
                || ToleranceClass::MonteCarlo.close(phi_sum, result.total_change),
            "Σφ={phi_sum} total_change={}",
            result.total_change
        );
        assert!(result.total_change.is_finite() && result.total_change.abs() > 1.0);
    }

    /// `DifferenceMeasure::GaussianKl` end to end: exact-Shapley efficiency
    /// (`Σφ == v(N) − v(∅) == total_change`) holds for the KL payoff exactly as it
    /// does for `MeanDiff` — this is an algebraic identity of the coalition-cached
    /// Shapley telescoping sum, independent of the (nonlinear) payoff shape.
    #[test]
    fn gaussian_kl_efficiency_holds_end_to_end() {
        let (model, data) = two_period_chain();
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cache_policy = CachePolicy::enabled(Some(1_000_000));
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::GaussianKl,
            n_samples: 800,
            seed: 13,
        };
        let result = distribution_change(&model, &data, &query, &opts, &ctx).unwrap();
        let phi_sum: f64 = result.contributions.iter().map(|c| c.contribution).sum();
        assert!(
            (phi_sum - result.total_change).abs() < 1e-6,
            "Σφ={phi_sum} total_change={}",
            result.total_change
        );
        // Gaussian KL >= 0 always; the +5 intercept shift on Y must register as a
        // genuine, nonzero divergence for this test to be meaningful.
        assert!(
            result.total_change.is_finite() && result.total_change > 0.0,
            "total_change={}",
            result.total_change
        );
    }

    /// `DifferenceMeasure::GaussianKl` payoff value, pinned against a hand-computed
    /// closed-form Gaussian KL (not just checked for internal self-consistency).
    /// `measure_value` is exactly the computation `MechanismSwapPayoff::value`
    /// delegates to for the `GaussianKl` branch, so this pins the actual payoff
    /// arithmetic that was previously untested.
    #[test]
    fn measure_value_gaussian_kl_matches_closed_form() {
        // KL(N(2,3) ‖ N(0,1)) = 0.5 * (ln(1/3) + (3 + (2-0)^2)/1 - 1)
        let expected = 0.5_f64 * ((1.0_f64 / 3.0).ln() + 7.0 - 1.0);
        let got =
            measure_value(DifferenceMeasure::GaussianKl, 1, 2.0, 3.0, Some((0.0, 1.0))).unwrap();
        assert!((got - expected).abs() < 1e-12, "got={got} expected={expected}");

        // The all-baseline coalition (mask == 0) is defined as exactly zero
        // divergence, regardless of the sampled (mu, var) passed in — matching
        // `v(∅) == 0` used by the efficiency identity above.
        let empty =
            measure_value(DifferenceMeasure::GaussianKl, 0, 2.0, 3.0, Some((0.0, 1.0))).unwrap();
        assert!(empty.abs() < f64::EPSILON, "expected exact 0.0, got {empty}");

        // Missing cached baseline law with a non-empty mask is a hard error, not a
        // silent 0.0 — the payoff must have cached v(∅) first.
        assert!(measure_value(DifferenceMeasure::GaussianKl, 1, 2.0, 3.0, None).is_err());
    }

    #[test]
    fn inputs_and_mechanisms_runs() {
        let (model, data) = two_period_chain();
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_components(AttributionComponents::InputsAndMechanisms)
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let ctx = ExecutionContext::for_tests(1);
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 200,
            seed: 5,
        };
        let result = distribution_change(&model, &data, &query, &opts, &ctx).unwrap();
        assert!(result.total_change.is_finite());
        assert!(!result.contributions.is_empty());
    }

    #[test]
    fn path_based_allocation_fills_breakdown() {
        let (model, data) = two_period_chain();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&model, &data, SelectionPolicy::BestScore)
            .unwrap();
        let model = model.with_mechanisms(store);
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_allocation(AllocationMethod::PathBased);
        let ctx = ExecutionContext::for_tests(1);
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 200,
            seed: 7,
        };
        let result = distribution_change(&model, &data, &query, &opts, &ctx).unwrap();
        assert!(!result.path_breakdown.is_empty(), "path_breakdown should be populated");
        assert!(result.total_change.is_finite());
    }
}
