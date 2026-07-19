//! Structure-change attribution via dual-graph parent-set Shapley.
//!
//! Given baseline and comparison DAGs over the same variables, attributes the
//! change in the outcome marginal to nodes whose parent sets differ. Hybrid
//! graphs swap comparison parent sets for coalition members; mechanisms are
//! re-fit under each hybrid (population-owned data for the parent-set owner).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    AllocationMethod, AttributionComponents, ChangeAttributionQuery, ComponentId, ExecutionContext,
    ShapleyConfig, VariableId,
};
use causal_data::TabularData;
use causal_graph::{BitSet, Dag, DenseNodeId, GraphWorkspace};
use causal_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismWorkspace,
    SelectionPolicy, sample_observational,
};
use causal_stats::mean_var;

use crate::change_common::{measure_value, run_change_allocation, total_change, ChangeOptions};
use crate::distribution_change::{hybrid_mechanisms, DifferenceMeasure};
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

    let (players, unidentified) = structure_players(
        baseline_model,
        comparison_model,
        outcome_dense,
        query.max_components,
    )?;
    if players.is_empty() {
        return Err(AttributionError::invalid_input(
            "no structure components to attribute (parent sets agree on outcome ancestors)",
        ));
    }

    let mut payoff = StructureSwapPayoff {
        baseline_graph: Arc::clone(&baseline_model.graph),
        comparison_graph: Arc::clone(&comparison_model.graph),
        variables: Arc::clone(&baseline_model.output_layout.variables),
        baseline_data,
        comparison_data,
        players: players.clone(),
        outcome: outcome_dense,
        measure: options.measure,
        n_samples: options.n_samples,
        seed: options.seed,
        ctx,
        ws: MechanismWorkspace::default(),
        baseline_law: None,
    };

    let v0 = payoff.value(0)?;
    let full_mask = (1u64 << players.len()) - 1;
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
    baseline: causal_core::PopulationSelector,
    comparison: causal_core::PopulationSelector,
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
        let parents = if use_comparison {
            comparison.parents(child)
        } else {
            baseline.parents(child)
        };
        for &p in parents {
            g.insert_directed(p, child).map_err(|e| match e {
                causal_graph::GraphError::Cycle { .. } => AttributionError::unsupported(
                    "hybrid structure for this coalition is cyclic; structure contribution undefined",
                ),
                other => AttributionError::Graph(other),
            })?;
        }
    }
    Ok(g)
}

struct StructureSwapPayoff<'a> {
    baseline_graph: Arc<Dag>,
    comparison_graph: Arc<Dag>,
    variables: Arc<[VariableId]>,
    baseline_data: TabularData,
    comparison_data: TabularData,
    players: Vec<ComponentId>,
    outcome: DenseNodeId,
    measure: DifferenceMeasure,
    n_samples: usize,
    seed: u64,
    ctx: &'a ExecutionContext,
    ws: MechanismWorkspace,
    baseline_law: Option<(f64, f64)>,
}

impl CoalitionPayoff for StructureSwapPayoff<'_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        if matches!(self.measure, DifferenceMeasure::GaussianKl) && self.baseline_law.is_none() {
            let (mu0, var0) = self.sample_outcome_law(0)?;
            self.baseline_law = Some((mu0, var0));
            if mask == 0 {
                return Ok(0.0);
            }
        }
        let (mu, var) = self.sample_outcome_law(mask)?;
        measure_value(self.measure, mask, mu, var, self.baseline_law)
    }
}

impl StructureSwapPayoff<'_> {
    fn sample_outcome_law(&mut self, mask: u64) -> Result<(f64, f64), AttributionError> {
        let player_nodes: Vec<DenseNodeId> = self
            .players
            .iter()
            .map(|c| {
                let idx = self
                    .variables
                    .iter()
                    .position(|v| *v == c.variable())
                    .expect("player in layout");
                DenseNodeId::from_raw(idx as u32)
            })
            .collect();

        let hybrid = hybrid_structure_dag(
            &self.baseline_graph,
            &self.comparison_graph,
            &player_nodes,
            mask,
        )?;
        let compiled = CompiledCausalModel::compile(hybrid)?;

        let (base_store, _) = MechanismRegistry::standard().assign_and_fit(
            &compiled,
            &self.baseline_data,
            SelectionPolicy::BestScore,
        )?;
        let (cmp_store, _) = MechanismRegistry::standard().assign_and_fit(
            &compiled,
            &self.comparison_data,
            SelectionPolicy::BestScore,
        )?;

        let kinds = vec![crate::distribution_change::PlayerKind::Mechanism; self.players.len()];
        let mixed: CompiledMechanismStore = hybrid_mechanisms(
            &base_store,
            &cmp_store,
            &compiled,
            &self.players,
            &kinds,
            mask,
        );
        let model = compiled.with_mechanisms(mixed);

        let mut rng = self.ctx.rng.stream(0x5C01_u64.wrapping_add(self.seed));
        let batch = sample_observational(
            &model,
            self.n_samples.max(1),
            &mut rng,
            &mut self.ws,
            self.ctx,
        )?;
        let col = batch.column(self.outcome.as_usize())?;
        let (mu, var) = mean_var(col);
        Ok((mu, var.max(1e-12)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        CachePolicy, CausalSchemaBuilder, MeasurementSpec, PopulationSelector, RoleHint,
        SmallRoleSet, ToleranceClass, ValueType,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::DenseNodeId;

    /// Baseline X→Y vs comparison Z→Y; Y intercept/slope differ across periods.
    fn parent_swap_fixture() -> (CompiledCausalModel, CompiledCausalModel, TabularData) {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        for (name, role) in [
            ("x", RoleHint::Context),
            ("z", RoleHint::Context),
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
        let result =
            structure_change(&baseline, &comparison, &data, &query, &opts, &ctx).unwrap();
        assert!(
            result.total_change.abs() > 2.0,
            "total={}",
            result.total_change
        );
        let y = result
            .contributions
            .iter()
            .find(|c| c.component.variable() == VariableId::from_raw(2))
            .expect("y player");
        assert_eq!(result.contributions.len(), 1);
        assert!(
            (y.contribution - result.total_change).abs() < 1e-6
                || ToleranceClass::MonteCarlo.close(y.contribution, result.total_change),
            "y={} total={}",
            y.contribution,
            result.total_change
        );
        assert!(result.unidentified.is_empty());
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
        let result =
            structure_change(&baseline, &comparison, &data, &query, &opts, &ctx).unwrap();
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
