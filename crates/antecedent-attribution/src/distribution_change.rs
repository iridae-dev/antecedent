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
    MechanismWorkspace, SelectionPolicy,
};

use crate::change_common::{
    ChangeOptions, DISTRIBUTION_STREAM, run_change_allocation, sample_outcome_law, stream_tag,
    total_change,
};
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
    validate_distribution_change_query(query)?;
    let (baseline_data, comparison_data) = resolve_change_populations(data, query)?;
    distribution_change_on_populations(
        graph_model,
        &baseline_data,
        &comparison_data,
        query,
        options,
        ctx,
    )
}

fn validate_distribution_change_query(
    query: &ChangeAttributionQuery,
) -> Result<(), AttributionError> {
    query.validate()?;
    require_mechanism_or_joint(query.components)?;
    if matches!(query.components, AttributionComponents::All) {
        return Err(AttributionError::unsupported(
            "AttributionComponents::All requires dual graphs; use ChangeAttribution::run_structure \
             for Structure, or InputsAndMechanisms for joint input+mechanism change",
        ));
    }
    Ok(())
}

/// [`distribution_change`] plus the uncertainty from fitting the mechanisms on finite
/// populations: `replicates` row bootstraps of the two populations (each refits every mechanism
/// and recomputes the attribution with the same Shapley sampling seed), summarised as
/// percentile intervals at `level` in [`ChangeAttributionResult::fit_uncertainty`].
///
/// The point attribution is the ordinary one on the original populations. Replicates whose
/// refit fails are counted; when any fails the intervals are withheld rather than computed
/// from the survivors.
///
/// # Errors
///
/// As [`distribution_change`]; fewer than 2 replicates or a level outside `(0, 1)`;
/// [`AttributionError::Cancelled`] when the context is cancelled.
pub fn distribution_change_with_fit_uncertainty(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &DistributionChangeOptions,
    replicates: u32,
    level: f64,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    use antecedent_core::StreamDomain;
    use antecedent_data::TableView;

    use crate::population::subset_table;
    use crate::result::FitUncertainty;

    if replicates < 2 || !(level > 0.0 && level < 1.0) {
        return Err(AttributionError::invalid_input(
            "fit-uncertainty bootstrap needs >= 2 replicates and a level in (0, 1)",
        ));
    }
    validate_distribution_change_query(query)?;
    let (baseline_data, comparison_data) = resolve_change_populations(data, query)?;
    let mut point = distribution_change_on_populations(
        graph_model,
        &baseline_data,
        &comparison_data,
        query,
        options,
        ctx,
    )?;
    let n_components = point.contributions.len();

    let resample = |source: &TabularData, rng: &mut antecedent_core::CausalRng| {
        let n = source.row_count();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "next_f64() is in [0, 1), so the scaled draw is a non-negative index below n"
        )]
        let rows: Vec<usize> =
            (0..n).map(|_| ((rng.next_f64() * n as f64) as usize).min(n - 1)).collect();
        subset_table(source, &rows)
    };
    let outcomes = ctx.map_indexed(replicates as usize, |b, worker| {
        if worker.cancellation.is_cancelled() {
            return Err(AttributionError::Cancelled);
        }
        let mut rng = worker.rng.stream_for(StreamDomain::Attribution, 0xB007_5712 ^ b as u64);
        let refit = resample(&baseline_data, &mut rng).and_then(|base| {
            let cmp = resample(&comparison_data, &mut rng)?;
            distribution_change_on_populations(graph_model, &base, &cmp, query, options, worker)
        });
        match refit {
            Ok(result) if result.contributions.len() == n_components => Ok(Some((
                result.total_change,
                result.contributions.iter().map(|c| c.contribution).collect::<Vec<_>>(),
            ))),
            Err(AttributionError::Cancelled) => Err(AttributionError::Cancelled),
            _ => Ok(None),
        }
    })?;
    let failures = outcomes.iter().filter(|o| o.is_none()).count();
    let (lower, upper, total_interval) = if failures == 0 {
        let tail = 0.5 * (1.0 - level);
        let interval = |mut values: Vec<f64>| {
            values.sort_by(f64::total_cmp);
            (
                antecedent_kernels::quantile_type7_sorted(&values, tail),
                antecedent_kernels::quantile_type7_sorted(&values, 1.0 - tail),
            )
        };
        let draws: Vec<&(f64, Vec<f64>)> = outcomes.iter().flatten().collect();
        let per_component: Vec<(f64, f64)> =
            (0..n_components).map(|j| interval(draws.iter().map(|d| d.1[j]).collect())).collect();
        (
            Some(per_component.iter().map(|p| p.0).collect::<Arc<[f64]>>()),
            Some(per_component.iter().map(|p| p.1).collect::<Arc<[f64]>>()),
            Some(interval(draws.iter().map(|d| d.0).collect())),
        )
    } else {
        (None, None, None)
    };
    point.fit_uncertainty = Some(FitUncertainty {
        replicates,
        failures: u32::try_from(failures).unwrap_or(u32::MAX),
        level,
        lower,
        upper,
        total_interval,
    });
    Ok(point)
}

fn distribution_change_on_populations(
    graph_model: &CompiledCausalModel,
    baseline_data: &TabularData,
    comparison_data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &DistributionChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    let (baseline_mechs, _) = MechanismRegistry::standard().assign_and_fit(
        graph_model,
        baseline_data,
        SelectionPolicy::BestScore,
    )?;
    let (comparison_mechs, _) = MechanismRegistry::standard().assign_and_fit(
        graph_model,
        comparison_data,
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
    // The hybrid model is built once at the all-baseline (mask 0) store and its slot array
    // is patched in place between masks; nothing is cloned per coalition.
    let hybrid = graph_model
        .clone()
        .with_mechanisms(CompiledMechanismStore { slots: Arc::clone(&baseline_mechs.slots) });

    let mut payoff = MechanismSwapPayoff {
        hybrid,
        baseline: baseline_mechs,
        comparison: comparison_mechs,
        player_kinds,
        player_dense,
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
    /// Hybrid model whose slots reflect `scratch_mask`: baseline slots everywhere except
    /// comparison slots for set mechanism-player bits.
    hybrid: CompiledCausalModel,
    baseline: CompiledMechanismStore,
    comparison: CompiledMechanismStore,
    player_kinds: Vec<PlayerKind>,
    /// Dense node per player, hoisted at construction.
    player_dense: Vec<Option<DenseNodeId>>,
    /// Mask currently applied to `hybrid`.
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
        // Patch the hybrid's slot array incrementally: only bits that changed since the
        // previous mask are touched (restore to baseline on clear, swap in the comparison
        // slot on set). `Input`-kind players never swap slots; the resulting store is
        // value-identical to rebuilding the full hybrid from baseline for every coalition.
        let diff = mask ^ self.scratch_mask;
        if diff != 0 {
            let slots = unique_slots(&mut self.hybrid.mechanisms.slots);
            for (i, dense) in self.player_dense.iter().enumerate() {
                if diff & (1u64 << i) == 0 || matches!(self.player_kinds[i], PlayerKind::Input) {
                    continue;
                }
                let Some(d) = dense else { continue };
                let idx = d.as_usize();
                let src = if mask & (1u64 << i) != 0 { &self.comparison } else { &self.baseline };
                slots[idx] = src.slots[idx].clone();
            }
            self.scratch_mask = mask;
        }
        sample_outcome_law(
            &self.hybrid,
            self.outcome,
            self.n_samples,
            stream_tag(DISTRIBUTION_STREAM, self.seed),
            self.ctx,
            &mut self.ws,
            &mut self.values_buf,
        )
    }
}

/// Mutable view of a slot array this payoff owns exclusively (the hybrid model is never
/// shared); falls back to a private copy if it ever is.
fn unique_slots(slots: &mut Arc<[MechanismSlot]>) -> &mut [MechanismSlot] {
    if Arc::get_mut(slots).is_none() {
        *slots = Arc::from(slots.to_vec());
    }
    Arc::get_mut(slots).expect("slot array is uniquely owned")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change_common::measure_value;
    use antecedent_core::{
        AllocationMethod, AttributionComponents, CachePolicy, CausalSchemaBuilder, MeasurementSpec,
        PopulationSelector, RoleHint, ShapleyConfig, SmallRoleSet, ValueType,
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

    /// [`two_period_chain`] with Y perturbed by a small deterministic per-row residual.
    ///
    /// `two_period_chain`'s `Y = a + 2X` is exactly noiseless, so every row bootstrap of it
    /// refits the identical OLS line regardless of which rows are repeated: a fit-uncertainty
    /// interval over such resamples is a point, not because refitting doesn't work, but because
    /// there is no sampling variation in a deterministic fixture to reveal. Adding a residual
    /// gives different bootstrap draws different multisets of residuals and hence different
    /// fitted intercepts.
    fn two_period_chain_with_noise() -> (CompiledCausalModel, TabularData) {
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
            let base = if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x };
            // Deterministic pseudo-noise (no RNG dependency): a few incommensurate frequencies
            // summed so no small subset of rows shares a residual. Real enough that a row
            // bootstrap's different multiset of residuals refits a genuinely different line.
            let t = i as f64;
            let noise =
                0.15 * (t * 0.913_1).sin() + 0.1 * (t * 2.071_3).sin() + 0.05 * (t * 5.311_7).sin();
            yv.push(base + noise);
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

    /// The row bootstrap reports estimation uncertainty the permutation-sampling standard error
    /// cannot: exact Shapley has none of the latter (`stderr` is `None`), yet refitting on
    /// resampled 40-row populations moves the contributions.
    #[allow(clippy::float_cmp)] // exact constants: the values compared are representable results, not measurements
    #[test]
    fn fit_uncertainty_is_a_row_bootstrap_of_the_refit_attribution() {
        let (model, data) = two_period_chain_with_noise();
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 400,
            seed: 3,
        };
        let serial = ExecutionContext::for_tests(1);
        let plain = distribution_change(&model, &data, &query, &opts, &serial).unwrap();
        assert!(plain.fit_uncertainty.is_none());
        assert!(plain.contributions.iter().all(|c| c.stderr.is_none()));

        let with_fit = distribution_change_with_fit_uncertainty(
            &model, &data, &query, &opts, 40, 0.9, &serial,
        )
        .unwrap();
        // The point attribution is the ordinary one.
        assert_eq!(with_fit.total_change, plain.total_change);
        assert_eq!(with_fit.contributions.len(), plain.contributions.len());
        let fit = with_fit.fit_uncertainty.as_ref().unwrap();
        assert_eq!((fit.replicates, fit.failures), (40, 0));
        assert!((fit.level - 0.9).abs() < 1e-15);
        let (lower, upper) = (fit.lower.as_ref().unwrap(), fit.upper.as_ref().unwrap());
        assert_eq!((lower.len(), upper.len()), (with_fit.contributions.len(), lower.len()));
        for j in 0..lower.len() {
            assert!(lower[j] <= upper[j]);
        }
        // Refitting on resamples moves the y contribution: a genuine interval, not a point.
        let y_idx = with_fit
            .contributions
            .iter()
            .position(|c| c.component.variable() == VariableId::from_raw(1))
            .unwrap();
        assert!(upper[y_idx] - lower[y_idx] > 1e-6);
        let (t_lo, t_hi) = fit.total_interval.unwrap();
        assert!(t_lo <= with_fit.total_change && with_fit.total_change <= t_hi);

        // Independent of the thread budget.
        let threaded = ExecutionContext::production(1, 4);
        let again = distribution_change_with_fit_uncertainty(
            &model, &data, &query, &opts, 40, 0.9, &threaded,
        )
        .unwrap();
        assert_eq!(again.fit_uncertainty, with_fit.fit_uncertainty);

        for (replicates, level) in [(1, 0.9), (10, 0.0), (10, 1.0)] {
            assert!(
                distribution_change_with_fit_uncertainty(
                    &model, &data, &query, &opts, replicates, level, &serial
                )
                .is_err()
            );
        }
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
        // Exact Shapley efficiency is an algebraic identity of the cached telescoping sum
        // (every coalition value is deterministic under CRN), so no Monte Carlo slack.
        let phi_sum: f64 = result.contributions.iter().map(|c| c.contribution).sum();
        assert!(
            (phi_sum - result.total_change).abs() < 1e-9,
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
            (phi_sum - result.total_change).abs() < 1e-9,
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
        // Truth by construction: Y's intercept moves by exactly +5 while X's law is the same
        // 40 values in both periods, so the whole +5 is Y's mechanism and X contributes 0.
        assert!((result.total_change - 5.0).abs() < 1e-3, "total={}", result.total_change);
        let phi = |raw: u32| {
            result
                .contributions
                .iter()
                .find(|c| c.component.variable() == VariableId::from_raw(raw))
                .map_or(0.0, |c| c.contribution)
        };
        assert!((phi(1) - 5.0).abs() < 1e-3, "y={}", phi(1));
        assert!(phi(0).abs() < 1e-3, "x={}", phi(0));
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
        // Truth by construction: the +5 intercept shift on Y is the whole change.
        assert!((result.total_change - 5.0).abs() < 1e-3, "total={}", result.total_change);
        // The breakdown only apportions each player's share across its paths: it must sum
        // to the players' total.
        let by_path: f64 = result.path_breakdown.iter().map(|p| p.contribution).sum();
        assert!((by_path - result.contribution_sum()).abs() < 1e-9, "paths={by_path}");
    }

    /// Adversarial fixture: X's law is identical between populations (same 40
    /// values repeated); only Y's intercept moves by +5. The true change
    /// decomposition is X = 0, Y = `total_change` — X's mechanism never moved, so
    /// swapping it between baseline and comparison cannot move the outcome law.
    /// `PathBased` must not attribute a share of the change to X merely because
    /// X→Y has a nonzero path coefficient in the (pooled) model used for
    /// structure.
    #[test]
    fn path_based_attributes_only_the_mechanism_that_changed() {
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
            n_samples: 400,
            seed: 7,
        };
        let result = distribution_change(&model, &data, &query, &opts, &ctx).unwrap();
        let x_contrib = result
            .contributions
            .iter()
            .find(|c| c.component.variable() == VariableId::from_raw(0))
            .map_or(0.0, |c| c.contribution);
        let y_contrib = result
            .contributions
            .iter()
            .find(|c| c.component.variable() == VariableId::from_raw(1))
            .expect("y component")
            .contribution;
        assert!(
            x_contrib.abs() < 0.25,
            "X's mechanism did not change; expected ~0, got x={x_contrib} y={y_contrib} \
             total={}",
            result.total_change
        );
        assert!(
            (y_contrib - result.total_change).abs() < 0.25,
            "all of the change is Y's; expected y≈total, got x={x_contrib} y={y_contrib} \
             total={}",
            result.total_change
        );
        // Efficiency: shares still sum exactly to the measured total change.
        let sum: f64 = result.contributions.iter().map(|c| c.contribution).sum();
        assert!(
            (sum - result.total_change).abs() < 1e-6,
            "sum={sum} total={}",
            result.total_change
        );
    }
}
