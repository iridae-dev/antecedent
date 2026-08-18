//! Structure-change attribution via dual-graph parent-set Shapley.
//!
//! Given baseline and comparison DAGs over the same variables, attributes the
//! change in the outcome marginal to nodes whose parent sets differ. Hybrid
//! graphs swap comparison parent sets for coalition members; mechanisms are
//! re-fit under each hybrid (population-owned data for the parent-set owner).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{
    AllocationMethod, AttributionComponents, ChangeAttributionQuery, ComponentId, ExecutionContext,
    ShapleyConfig, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{BitSet, Dag, DenseNodeId, GraphWorkspace};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismSlot,
    MechanismWorkspace, SelectionPolicy, sample_observational, sample_observational_into,
};
use antecedent_stats::mean_var;

use crate::change_common::{ChangeOptions, run_change_allocation, total_change};
use crate::coalition::full_coalition_mask;
use crate::distribution_change::DifferenceMeasure;
use crate::error::AttributionError;
use crate::prep::{
    require_structure_components, resolve_change_populations, resolve_outcome_dense,
};
use crate::result::ChangeAttributionResult;
use crate::shapley::CoalitionPayoff;

/// Options for structure-change attribution.
#[derive(Clone, Debug)]
pub struct StructureChangeOptions {
    /// Difference measure on the outcome samples.
    pub measure: DifferenceMeasure,
    /// Samples drawn per coalition evaluation.
    pub n_samples: usize,
    /// RNG seed for sampling.
    pub seed: u64,
}

impl Default for StructureChangeOptions {
    fn default() -> Self {
        let o = ChangeOptions::default_mean();
        Self { measure: o.measure, n_samples: o.n_samples, seed: o.seed }
    }
}

/// Attribute outcome-marginal change to differing parent sets between two graphs.
///
/// `baseline_model` and `comparison_model` must share the same variable layout.
/// Only nodes whose parent sets differ and that are ancestors of the outcome
/// (in either graph) are Shapley players; other structural diffs are reported
/// in [`ChangeAttributionResult::unidentified`].
///
/// # Errors
///
/// Layout mismatch, empty players, cyclic hybrids, fit/sample failures, or
/// Shapley size limits.
pub fn structure_change(
    baseline_model: &CompiledCausalModel,
    comparison_model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &StructureChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    query.validate()?;
    require_structure_components(
        query.components,
        "structure_change requires AttributionComponents::Structure",
    )?;
    validate_shared_layout(baseline_model, comparison_model)?;

    let (baseline_data, comparison_data) = resolve_change_populations(data, query)?;
    let outcome_dense = resolve_outcome_dense(baseline_model, query.outcome)?;
    // Outcome must resolve in both layouts (already same VariableIds).
    let _ = resolve_outcome_dense(comparison_model, query.outcome)?;

    let (players, unidentified) =
        structure_players(baseline_model, comparison_model, outcome_dense, query.max_components)?;
    if players.is_empty() {
        return Err(AttributionError::invalid_input(
            "no structure components to attribute (parent sets agree on outcome ancestors)",
        ));
    }
    crate::shapley::check_coalition_sample_budget(
        players.len(),
        &query.allocation,
        options.n_samples,
    )?;

    // Player → dense-node mapping hoisted out of the per-coalition path (this is
    // invariant across coalitions; it was previously recomputed per mask).
    let variables = &baseline_model.output_layout.variables;
    let player_nodes: Vec<DenseNodeId> = players
        .iter()
        .map(|c| {
            let idx = variables.iter().position(|v| *v == c.variable()).expect("player in layout");
            DenseNodeId::from_raw(idx as u32)
        })
        .collect();

    let mut payoff = StructureSwapPayoff {
        baseline_graph: Arc::clone(&baseline_model.graph),
        comparison_graph: Arc::clone(&comparison_model.graph),
        baseline_data,
        comparison_data,
        players: players.clone(),
        player_nodes,
        outcome: outcome_dense,
        measure: options.measure,
        n_samples: options.n_samples,
        seed: options.seed,
        ctx,
        ws: MechanismWorkspace::default(),
        values_buf: Vec::new(),
        baseline_law: None,
        fits: None,
        compiled_cache: HashMap::new(),
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
        Arc::from(unidentified),
        ctx,
        Some(baseline_model),
    )
}

/// Convenience: Shapley Monte Carlo structure-change with defaults.
///
/// # Errors
///
/// See [`structure_change`].
pub fn structure_change_shapley(
    baseline_model: &CompiledCausalModel,
    comparison_model: &CompiledCausalModel,
    data: &TabularData,
    outcome: VariableId,
    baseline: antecedent_core::PopulationSelector,
    comparison: antecedent_core::PopulationSelector,
    shapley: ShapleyConfig,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    let query = ChangeAttributionQuery::new(outcome, baseline, comparison)
        .with_components(AttributionComponents::Structure)
        .with_allocation(AllocationMethod::Shapley { approximation: shapley });
    structure_change(
        baseline_model,
        comparison_model,
        data,
        &query,
        &StructureChangeOptions::default(),
        ctx,
    )
}

fn validate_shared_layout(
    baseline: &CompiledCausalModel,
    comparison: &CompiledCausalModel,
) -> Result<(), AttributionError> {
    if baseline.n_nodes() != comparison.n_nodes() {
        return Err(AttributionError::invalid_input(
            "baseline and comparison models must have the same node count",
        ));
    }
    if baseline.output_layout.variables.as_ref() != comparison.output_layout.variables.as_ref() {
        return Err(AttributionError::invalid_input(
            "baseline and comparison models must share the same VariableId layout",
        ));
    }
    Ok(())
}

fn sorted_parents(graph: &Dag, node: DenseNodeId) -> Vec<DenseNodeId> {
    let mut p = graph.parents(node).to_vec();
    p.sort_by_key(|id| id.raw());
    p
}

fn parent_sets_differ(baseline: &Dag, comparison: &Dag, node: DenseNodeId) -> bool {
    sorted_parents(baseline, node) != sorted_parents(comparison, node)
}

/// Players = differing parent sets among outcome ancestors (either graph).
/// Non-ancestor structural diffs → `unidentified`.
fn structure_players(
    baseline: &CompiledCausalModel,
    comparison: &CompiledCausalModel,
    outcome: DenseNodeId,
    max_components: usize,
) -> Result<(Vec<ComponentId>, Vec<ComponentId>), AttributionError> {
    let mut ws = GraphWorkspace::default();
    let mut anc_base = BitSet::default();
    let mut anc_cmp = BitSet::default();
    baseline.graph.ancestors_of(&[outcome], &mut anc_base, &mut ws);
    comparison.graph.ancestors_of(&[outcome], &mut anc_cmp, &mut ws);

    let n = baseline.n_nodes();
    let mut players = Vec::new();
    let mut unidentified = Vec::new();
    for i in 0..n {
        let node = DenseNodeId::from_raw(i as u32);
        if !parent_sets_differ(&baseline.graph, &comparison.graph, node) {
            continue;
        }
        let var = baseline.output_layout.variables[i];
        let comp = ComponentId::from_variable(var);
        let relevant = anc_base.contains(node) || anc_cmp.contains(node);
        if relevant {
            players.push(comp);
        } else {
            unidentified.push(comp);
        }
    }
    if players.len() > max_components {
        return Err(AttributionError::SizeLimit {
            kind: "components",
            requested: players.len(),
            max: max_components,
        });
    }
    Ok((players, unidentified))
}

/// Build a hybrid DAG: comparison parents for set player bits, else baseline.
pub(crate) fn hybrid_structure_dag(
    baseline: &Dag,
    comparison: &Dag,
    player_nodes: &[DenseNodeId],
    mask: u64,
) -> Result<Dag, AttributionError> {
    let n = baseline.node_count();
    if comparison.node_count() != n {
        return Err(AttributionError::invalid_input(
            "baseline and comparison graphs must have the same node count",
        ));
    }
    let n_u32 = u32::try_from(n).map_err(|_| AttributionError::invalid_input("too many nodes"))?;
    let mut g = Dag::with_variables(n_u32);
    for i in 0..n {
        let child = DenseNodeId::from_raw(i as u32);
        let use_comparison = player_nodes
            .iter()
            .enumerate()
            .any(|(pi, &p)| p == child && (mask & (1u64 << pi)) != 0);
        let parents =
            if use_comparison { comparison.parents(child) } else { baseline.parents(child) };
        for &p in parents {
            g.insert_directed(p, child).map_err(|e| match e {
                antecedent_graph::GraphError::Cycle { .. } => AttributionError::unsupported(
                    "hybrid structure for this coalition is cyclic; structure contribution undefined",
                ),
                other => AttributionError::Graph(other),
            })?;
        }
    }
    Ok(g)
}

/// Mechanism stores backing every coalition's hybrid model.
///
/// Only *player* nodes' parent sets vary across coalition masks; every other node
/// keeps its baseline parent set in every hybrid. A node's mechanism fit depends
/// only on `(node, parent set, population)` — `MechanismRegistry::assign_and_fit`
/// fits each node independently from its own column and its parents' columns via
/// deterministic solvers (no RNG anywhere in the fit paths, and the compiled model
/// is consulted only for the shared variable layout, which is identical across
/// hybrids). So there are exactly two distinct fits per node, not two full-model
/// refits per coalition:
///
/// * `baseline`: fit on the baseline population under the all-baseline hybrid
///   (mask 0) — supplies every non-player node and every unset player bit.
/// * `comparison`: fit on the comparison population under the full-coalition
///   hybrid — supplies each set player bit (its comparison parent set).
///
/// Hybrid parent *order* is also invariant: `hybrid_structure_dag` inserts each
/// node's parents by iterating the source graph's parent slice, so the gather-plan
/// parent order (and hence coefficient alignment) for a node is identical in every
/// hybrid that gives it the same parent-set source. Composing per-mask stores from
/// these two fits is therefore numerically identical to the previous per-coalition
/// double refit (pinned by `memoized_fits_match_per_coalition_refit`).
struct StructureFits {
    baseline: CompiledMechanismStore,
    comparison: CompiledMechanismStore,
}

struct StructureSwapPayoff<'a> {
    baseline_graph: Arc<Dag>,
    comparison_graph: Arc<Dag>,
    baseline_data: TabularData,
    comparison_data: TabularData,
    players: Vec<ComponentId>,
    /// Dense node per player (aligned with `players`), hoisted at construction.
    player_nodes: Vec<DenseNodeId>,
    outcome: DenseNodeId,
    measure: DifferenceMeasure,
    n_samples: usize,
    seed: u64,
    ctx: &'a ExecutionContext,
    ws: MechanismWorkspace,
    /// Reused ancestral-sample buffer (`n_samples × n_nodes`).
    values_buf: Vec<f64>,
    baseline_law: Option<(f64, f64)>,
    /// Memoized mechanism fits (two `assign_and_fit` calls total; see [`StructureFits`]).
    fits: Option<StructureFits>,
    /// Memoized compiled hybrid graphs keyed by coalition mask.
    compiled_cache: HashMap<u64, CompiledCausalModel>,
}

impl crate::change_common::CachedOutcomeLawPayoff for StructureSwapPayoff<'_> {
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

impl CoalitionPayoff for StructureSwapPayoff<'_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        crate::change_common::CachedOutcomeLawPayoff::cached_payoff_value(self, mask)
    }
}

impl StructureSwapPayoff<'_> {
    /// Compile (or fetch the memoized) hybrid graph for `mask`.
    ///
    /// Cyclic hybrids are not cached: the error is re-raised on every request,
    /// exactly as the previous per-coalition compile did.
    fn compiled_for(&mut self, mask: u64) -> Result<CompiledCausalModel, AttributionError> {
        if let Some(compiled) = self.compiled_cache.get(&mask) {
            return Ok(compiled.clone());
        }
        let hybrid = hybrid_structure_dag(
            &self.baseline_graph,
            &self.comparison_graph,
            &self.player_nodes,
            mask,
        )?;
        let compiled = CompiledCausalModel::compile(hybrid)?;
        self.compiled_cache.insert(mask, compiled.clone());
        Ok(compiled)
    }

    /// Fit the two memoized stores on first use (see [`StructureFits`]).
    ///
    /// This compiles the full-coalition hybrid up front, so a cyclic full
    /// coalition now errors on the first payoff evaluation instead of at
    /// `v_full`; both evaluations happen back-to-back inside `structure_change`
    /// before any allocation, so the public result is the same error. Likewise a
    /// fit failure on the comparison population surfaces under the full-coalition
    /// hybrid structure rather than per mask — previously each mask also fit the
    /// comparison population (mostly discarding the result), so failures could
    /// surface under structures whose fits were never used.
    fn ensure_fits(&mut self) -> Result<(), AttributionError> {
        if self.fits.is_some() {
            return Ok(());
        }
        let full_mask = full_coalition_mask(self.players.len())?;
        let compiled_baseline = self.compiled_for(0)?;
        let compiled_full = self.compiled_for(full_mask)?;
        let (baseline, _) = MechanismRegistry::standard().assign_and_fit(
            &compiled_baseline,
            &self.baseline_data,
            SelectionPolicy::BestScore,
        )?;
        let (comparison, _) = MechanismRegistry::standard().assign_and_fit(
            &compiled_full,
            &self.comparison_data,
            SelectionPolicy::BestScore,
        )?;
        self.fits = Some(StructureFits { baseline, comparison });
        Ok(())
    }

    fn sample_outcome_law(&mut self, mask: u64) -> Result<(f64, f64), AttributionError> {
        self.ensure_fits()?;
        let compiled = self.compiled_for(mask)?;
        let fits = self.fits.as_ref().expect("fits ensured above");

        // Compose the hybrid store from the two memoized fits: comparison-fitted
        // slots for set player bits, baseline-fitted slots everywhere else. This
        // matches the old `hybrid_mechanisms(base_store, cmp_store, ..)` output
        // exactly (see the `StructureFits` docs for why the per-node fits agree).
        let n = compiled.n_nodes();
        let mut slots: Vec<MechanismSlot> = Vec::with_capacity(n);
        for i in 0..n {
            let node = DenseNodeId::from_raw(i as u32);
            let use_comparison = self
                .player_nodes
                .iter()
                .enumerate()
                .any(|(pi, &p)| p == node && (mask & (1u64 << pi)) != 0);
            let store = if use_comparison { &fits.comparison } else { &fits.baseline };
            slots.push(store.slots[i].clone());
        }
        let model = compiled.with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) });

        let mut rng = self.ctx.rng.stream(0x5C01_u64.wrapping_add(self.seed));
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
    use antecedent_core::{
        CachePolicy, CausalSchemaBuilder, MeasurementSpec, PopulationSelector, RoleHint,
        SmallRoleSet, ToleranceClass, ValueType,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
    use antecedent_graph::DenseNodeId;
    use serde::Deserialize;

    /// Baseline X→Y vs comparison Z→Y; Y intercept/slope differ across periods.
    fn parent_swap_fixture() -> (CompiledCausalModel, CompiledCausalModel, TabularData) {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        for (name, role) in
            [("x", RoleHint::Context), ("z", RoleHint::Context), ("y", RoleHint::OutcomeCandidate)]
        {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut xv = Vec::with_capacity(n);
        let mut zv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i % 40) as f64 * 0.1;
            let z = ((i + 7) % 40) as f64 * 0.1;
            xv.push(x);
            zv.push(z);
            if i < 40 {
                yv.push(1.0 + 2.0 * x);
            } else {
                yv.push(8.0 + 3.0 * z);
            }
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(zv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());

        let mut g0 = Dag::with_variables(3);
        g0.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let mut g1 = Dag::with_variables(3);
        g1.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let baseline = CompiledCausalModel::compile(g0).unwrap();
        let comparison = CompiledCausalModel::compile(g1).unwrap();
        (baseline, comparison, data)
    }

    #[test]
    fn attributes_parent_set_change_to_y() {
        #[derive(Deserialize)]
        struct Fixture {
            cases: Vec<Case>,
            comparison: Comparison,
        }
        #[derive(Deserialize)]
        struct Case {
            id: String,
            #[serde(default)]
            changed_players: Vec<String>,
            #[serde(default)]
            unidentified: Vec<String>,
        }
        #[derive(Deserialize)]
        struct Comparison {
            absolute_tolerance: f64,
        }
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../conformance/attribution/structure_change_grid/expected.json"
        ))
        .unwrap();
        let expected = fixture.cases.iter().find(|case| case.id == "single_parent_swap").unwrap();
        let (baseline, comparison, data) = parent_swap_fixture();
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(2),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_components(AttributionComponents::Structure)
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cache_policy = CachePolicy::enabled(Some(1_000_000));
        let opts = StructureChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 600,
            seed: 5,
        };
        let result = structure_change(&baseline, &comparison, &data, &query, &opts, &ctx).unwrap();
        assert!(result.total_change.abs() > 2.0, "total={}", result.total_change);
        let y = result
            .contributions
            .iter()
            .find(|c| c.component.variable() == VariableId::from_raw(2))
            .expect("y player");
        assert_eq!(result.contributions.len(), expected.changed_players.len());
        assert!(
            (y.contribution - result.total_change).abs() < fixture.comparison.absolute_tolerance
                || ToleranceClass::MonteCarlo.close(y.contribution, result.total_change),
            "y={} total={}",
            y.contribution,
            result.total_change
        );
        assert_eq!(result.unidentified.len(), expected.unidentified.len());
    }

    /// Pins the memoization refactor: for every coalition mask, the payoff's
    /// memoized two-fit composition must reproduce, bit for bit, the previous
    /// implementation's per-coalition double refit (hybrid compile, baseline fit,
    /// comparison fit, slot mix, CRN sampling). No fit-call counter is reachable
    /// from this crate (`assign_and_fit` keeps no call statistics), so numeric
    /// equality against an inline reimplementation of the old path is the pinned
    /// evidence instead.
    #[test]
    fn memoized_fits_match_per_coalition_refit() {
        // Two players: baseline x→m, x→y vs comparison z→m, m→y.
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        for (name, role) in [
            ("x", RoleHint::Context),
            ("z", RoleHint::Context),
            ("m", RoleHint::Context),
            ("y", RoleHint::OutcomeCandidate),
        ] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut xv = Vec::with_capacity(n);
        let mut zv = Vec::with_capacity(n);
        let mut mv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i % 40) as f64 * 0.1;
            let z = ((i + 7) % 40) as f64 * 0.1;
            let m = if i < 40 { 0.5 + 1.5 * x } else { 2.0 + 0.8 * z };
            let y = if i < 40 { 1.0 + 2.0 * x } else { 3.0 + 1.2 * m };
            xv.push(x);
            zv.push(z);
            mv.push(m);
            yv.push(y);
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![xv, zv, mv, yv]
            .into_iter()
            .enumerate()
            .map(|(vi, vals)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(vi as u32),
                        Arc::from(vals),
                        validity.clone(),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());

        let x = DenseNodeId::from_raw(0);
        let z = DenseNodeId::from_raw(1);
        let m = DenseNodeId::from_raw(2);
        let y = DenseNodeId::from_raw(3);
        let mut g0 = Dag::with_variables(4);
        g0.insert_directed(x, m).unwrap();
        g0.insert_directed(x, y).unwrap();
        let mut g1 = Dag::with_variables(4);
        g1.insert_directed(z, m).unwrap();
        g1.insert_directed(m, y).unwrap();
        let baseline = CompiledCausalModel::compile(g0).unwrap();
        let comparison = CompiledCausalModel::compile(g1).unwrap();

        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(3),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_components(AttributionComponents::Structure);
        let ctx = ExecutionContext::for_tests(1);
        let (baseline_data, comparison_data) =
            crate::prep::resolve_change_populations(&data, &query).unwrap();

        let players = vec![
            ComponentId::from_variable(VariableId::from_raw(2)),
            ComponentId::from_variable(VariableId::from_raw(3)),
        ];
        let player_nodes = vec![m, y];
        let seed = 9u64;
        let n_samples = 300usize;

        let mut payoff = StructureSwapPayoff {
            baseline_graph: Arc::clone(&baseline.graph),
            comparison_graph: Arc::clone(&comparison.graph),
            baseline_data: baseline_data.clone(),
            comparison_data: comparison_data.clone(),
            players,
            player_nodes: player_nodes.clone(),
            outcome: y,
            measure: DifferenceMeasure::MeanDiff,
            n_samples,
            seed,
            ctx: &ctx,
            ws: MechanismWorkspace::default(),
            values_buf: Vec::new(),
            baseline_law: None,
            fits: None,
            compiled_cache: HashMap::new(),
        };

        for mask in 0..4u64 {
            // Reference: the pre-memoization per-coalition path, inlined.
            let hybrid =
                hybrid_structure_dag(&baseline.graph, &comparison.graph, &player_nodes, mask)
                    .unwrap();
            let compiled = CompiledCausalModel::compile(hybrid).unwrap();
            let (base_store, _) = MechanismRegistry::standard()
                .assign_and_fit(&compiled, &baseline_data, SelectionPolicy::BestScore)
                .unwrap();
            let (cmp_store, _) = MechanismRegistry::standard()
                .assign_and_fit(&compiled, &comparison_data, SelectionPolicy::BestScore)
                .unwrap();
            let mut slots = base_store.slots.to_vec();
            for (pi, &p) in player_nodes.iter().enumerate() {
                if mask & (1u64 << pi) != 0 {
                    slots[p.as_usize()] = cmp_store.slots[p.as_usize()].clone();
                }
            }
            let model =
                compiled.with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) });
            let mut rng = ctx.rng.stream(0x5C01_u64.wrapping_add(seed));
            let mut ws = MechanismWorkspace::default();
            let batch = sample_observational(&model, n_samples, &mut rng, &mut ws, &ctx).unwrap();
            let (mu_ref, var_ref) = mean_var(batch.column(y.as_usize()).unwrap());
            let var_ref = var_ref.max(1e-12);

            let (mu, var) = payoff.sample_outcome_law(mask).unwrap();
            assert!(
                mu.to_bits() == mu_ref.to_bits() && var.to_bits() == var_ref.to_bits(),
                "mask={mask}: memoized ({mu}, {var}) != refit ({mu_ref}, {var_ref})"
            );
        }
    }

    #[test]
    fn rejects_mechanism_components() {
        let (baseline, comparison, data) = parent_swap_fixture();
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(2),
            PopulationSelector::TimeRange { start: 0, end: 40 },
            PopulationSelector::TimeRange { start: 40, end: 80 },
        )
        .with_components(AttributionComponents::Mechanisms);
        let ctx = ExecutionContext::for_tests(1);
        let err = structure_change(
            &baseline,
            &comparison,
            &data,
            &query,
            &StructureChangeOptions::default(),
            &ctx,
        )
        .unwrap_err();
        assert!(matches!(err, AttributionError::Unsupported { .. }));
    }

    #[test]
    fn non_ancestor_structural_diff_is_unidentified() {
        // Baseline: X→Y, W→V. Comparison: Z→Y, W→V removed (V root).
        // Outcome Y: only Y differs among ancestors; V is unidentified.
        let n = 60usize;
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "z", "y", "w", "v"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let cols: Vec<_> = (0..5)
            .map(|vi| {
                let vals: Vec<f64> = (0..n).map(|i| (i + vi) as f64 * 0.05).collect();
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(vi as u32),
                        Arc::from(vals),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        // Overwrite y with structure-sensitive law.
        let mut yv = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i % 30) as f64 * 0.1;
            let z = ((i + 3) % 30) as f64 * 0.1;
            yv.push(if i < 30 { 1.0 + 2.0 * x } else { 7.0 + 2.5 * z });
        }
        let mut cols = cols;
        cols[2] = OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(2),
                Arc::from(yv),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        );
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());

        let x = DenseNodeId::from_raw(0);
        let z = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let w = DenseNodeId::from_raw(3);
        let v = DenseNodeId::from_raw(4);

        let mut g0 = Dag::with_variables(5);
        g0.insert_directed(x, y).unwrap();
        g0.insert_directed(w, v).unwrap();
        let mut g1 = Dag::with_variables(5);
        g1.insert_directed(z, y).unwrap();
        // V is root in comparison (no W→V).

        let baseline = CompiledCausalModel::compile(g0).unwrap();
        let comparison = CompiledCausalModel::compile(g1).unwrap();
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(2),
            PopulationSelector::TimeRange { start: 0, end: 30 },
            PopulationSelector::TimeRange { start: 30, end: 60 },
        )
        .with_components(AttributionComponents::Structure)
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cache_policy = CachePolicy::enabled(Some(1_000_000));
        let opts = StructureChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: 400,
            seed: 1,
        };
        let result = structure_change(&baseline, &comparison, &data, &query, &opts, &ctx).unwrap();
        assert!(
            result.unidentified.iter().any(|c| c.variable() == VariableId::from_raw(4)),
            "v should be unidentified: {:?}",
            result.unidentified
        );
        assert!(
            result.contributions.iter().any(|c| c.component.variable() == VariableId::from_raw(2)),
            "y should be a player"
        );
    }
}
