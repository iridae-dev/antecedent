//! Distribution-change attribution (DoWhy-GCM parity; DESIGN.md §17.2).
//!
//! Fits mechanisms on baseline and comparison populations, then attributes the
//! change in the outcome marginal to mechanism replacements via Shapley values
//! (Budhathoki et al. 2021).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{
    AllocationMethod, AttributionComponents, ChangeAttributionQuery, ComponentId, ExecutionContext,
    ShapleyConfig, VariableId,
};
use causal_data::TabularData;
use causal_graph::{BitSet, DenseNodeId, GraphWorkspace};
use causal_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismSlot,
    MechanismWorkspace, SelectionPolicy, sample_observational,
};

use crate::error::AttributionError;
use crate::population::{resolve_rows, subset_table};
use crate::result::ChangeAttributionResult;
use crate::shapley::{CoalitionPayoff, estimate_shapley, sequential_allocate};

/// How to summarize the target marginal difference.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum DifferenceMeasure {
    /// `E[Y_comparison-like] − E[Y_baseline-like]`.
    MeanDiff,
    /// Variance difference.
    VarianceDiff,
}

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
        Self { measure: DifferenceMeasure::MeanDiff, n_samples: 2_000, seed: 0 }
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
    match query.components {
        AttributionComponents::Mechanisms
        | AttributionComponents::InputsAndMechanisms
        | AttributionComponents::All => {}
        AttributionComponents::Inputs | AttributionComponents::Structure => {
            return Err(AttributionError::Message(
                "distribution_change requires Mechanisms (or All / InputsAndMechanisms)".into(),
            ));
        }
        _ => {
            return Err(AttributionError::Message(
                "unsupported AttributionComponents for distribution_change".into(),
            ));
        }
    }

    let baseline_rows = resolve_rows(data, &query.baseline)?;
    let comparison_rows = resolve_rows(data, &query.comparison)?;
    if baseline_rows.is_empty() || comparison_rows.is_empty() {
        return Err(AttributionError::Message(
            "baseline and comparison populations must be non-empty".into(),
        ));
    }
    let baseline_data = subset_table(data, &baseline_rows)?;
    let comparison_data = subset_table(data, &comparison_rows)?;

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

    let outcome_dense = graph_model.dense_of(query.outcome).ok_or_else(|| {
        AttributionError::Message(format!("outcome {} not in model", query.outcome))
    })?;

    let players = mechanism_players(graph_model, outcome_dense, query.max_components)?;
    if players.is_empty() {
        return Err(AttributionError::Message("no mechanism components to attribute".into()));
    }

    let mut payoff = MechanismSwapPayoff {
        template: graph_model.clone(),
        baseline: baseline_mechs,
        comparison: comparison_mechs,
        players: players.clone(),
        outcome: outcome_dense,
        measure: options.measure,
        n_samples: options.n_samples,
        seed: options.seed,
        ws: MechanismWorkspace::default(),
    };

    // Total change: full comparison mechanisms vs full baseline.
    let v0 = payoff.value(0)?;
    let full_mask = (1u64 << players.len()) - 1;
    let v_full = payoff.value(full_mask)?;
    let total_change = v_full - v0;

    let estimate = match &query.allocation {
        AllocationMethod::Shapley { approximation } => {
            estimate_shapley(&players, approximation, &mut payoff, ctx)?
        }
        AllocationMethod::Sequential { order } => {
            let index_of = |c: ComponentId| players.iter().position(|&p| p == c);
            sequential_allocate(order, &index_of, &mut payoff, ctx)?
        }
        AllocationMethod::PathBased => {
            return Err(AttributionError::Message(
                "PathBased allocation is handled by path_decompose, not distribution_change".into(),
            ));
        }
        _ => {
            return Err(AttributionError::Message("unsupported AllocationMethod".into()));
        }
    };

    let mc_stderr = estimate.monte_carlo_stderr;
    let component_mc = estimate.component_mc_stderr.clone().map(Arc::from);
    let interactions = Arc::from(estimate.interactions.clone());
    let cache_stats = estimate.cache_stats.clone();
    let budget = estimate.budget.clone();
    let contributions = Arc::from(estimate.into_contributions());

    Ok(ChangeAttributionResult {
        outcome: query.outcome,
        total_change,
        contributions,
        interactions,
        path_breakdown: Arc::from([]),
        unidentified: Arc::from([]),
        graph_sensitivity: None,
        budget,
        monte_carlo_stderr: mc_stderr,
        component_mc_stderr: component_mc,
        cache_stats,
    })
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
    baseline: causal_core::PopulationSelector,
    comparison: causal_core::PopulationSelector,
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
    let mut ws = GraphWorkspace::default();
    let mut anc = BitSet::default();
    model.graph.ancestors_of(&[outcome], &mut anc, &mut ws);
    let mut players = Vec::new();
    for gather in model.parent_gathers.iter() {
        let node = gather.child;
        if !anc.contains(node) {
            continue;
        }
        let var = model.output_layout.variables[node.as_usize()];
        players.push(ComponentId::from_variable(var));
    }
    // Stable topo order already from parent_gathers / node_order.
    if players.len() > max_components {
        return Err(AttributionError::SizeLimit {
            kind: "components",
            requested: players.len(),
            max: max_components,
        });
    }
    Ok(players)
}

/// Build a mechanism store that uses comparison slots for bits set in `mask`.
pub(crate) fn hybrid_mechanisms(
    baseline: &CompiledMechanismStore,
    comparison: &CompiledMechanismStore,
    model: &CompiledCausalModel,
    players: &[ComponentId],
    mask: u64,
) -> CompiledMechanismStore {
    let n = model.n_nodes();
    let mut slots: Vec<MechanismSlot> = (0..n).map(|i| baseline.slots[i].clone()).collect();
    for (i, comp) in players.iter().enumerate() {
        if mask & (1u64 << i) == 0 {
            continue;
        }
        if let Some(dense) = model.dense_of(comp.variable()) {
            slots[dense.as_usize()] = comparison.slots[dense.as_usize()].clone();
        }
    }
    CompiledMechanismStore { slots: Arc::from(slots) }
}

struct MechanismSwapPayoff {
    template: CompiledCausalModel,
    baseline: CompiledMechanismStore,
    comparison: CompiledMechanismStore,
    players: Vec<ComponentId>,
    outcome: DenseNodeId,
    measure: DifferenceMeasure,
    n_samples: usize,
    seed: u64,
    ws: MechanismWorkspace,
}

impl CoalitionPayoff for MechanismSwapPayoff {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        let store = hybrid_mechanisms(
            &self.baseline,
            &self.comparison,
            &self.template,
            &self.players,
            mask,
        );
        // Clone template shape with hybrid mechanisms (CompiledCausalModel is Clone).
        let model = self.template.clone().with_mechanisms(store);
        let mut rng = causal_core::CausalRng::from_seed(self.seed.wrapping_add(mask));
        let batch = sample_observational(
            &model,
            self.n_samples.max(1),
            &mut rng,
            &mut self.ws,
            &ExecutionContext::for_tests(self.seed),
        )?;
        let col = batch.column(self.outcome.as_usize())?;
        Ok(match self.measure {
            DifferenceMeasure::MeanDiff => col.iter().sum::<f64>() / col.len().max(1) as f64,
            DifferenceMeasure::VarianceDiff => {
                let n = col.len().max(1) as f64;
                let mean = col.iter().sum::<f64>() / n;
                col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        CachePolicy, CausalSchemaBuilder, MeasurementSpec, PopulationSelector, RoleHint,
        SmallRoleSet, ValueType,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::{Dag, DenseNodeId};

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
        assert!(result.total_change > 3.0, "total={}", result.total_change);
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
    }
}
