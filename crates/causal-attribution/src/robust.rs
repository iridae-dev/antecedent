//! Robust distribution-change attribution (`pinned baseline` `distribution_change_robust`).
//!
//! Uses fitted mechanism hybrids (same topology as [`distribution_change`]) with a
//! structural-mean payoff. Linear-family mechanisms use a fast OLS path; nonlinear
//! slots evaluate at zero noise.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{ChangeAttributionQuery, ComponentId, ExecutionContext, VariableId};
use causal_data::{TableView, TabularData};
use causal_graph::DenseNodeId;
use causal_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismSlot,
    MechanismWorkspace, ParentBatch, SelectionPolicy, evaluate_column,
};
use causal_stats::{FaerBackend, LeastSquaresWorkspace};

use crate::distribution_change::{PlayerKind, hybrid_mechanisms, mechanism_players};
use crate::error::AttributionError;
use crate::prep::{
    require_mechanism_components, require_shapley_config, resolve_change_populations,
    resolve_outcome_dense,
};
use crate::result::ChangeAttributionResult;
use crate::shapley::{CoalitionPayoff, estimate_shapley};

/// Options for the robust estimator.
#[derive(Clone, Debug)]
pub struct RobustChangeOptions {
    /// Cap on units used for regression / evaluation.
    pub max_rows: usize,
}

impl Default for RobustChangeOptions {
    fn default() -> Self {
        Self { max_rows: 10_000 }
    }
}

/// Robust attribution via mechanism hybrids + Shapley.
///
/// # Errors
///
/// Fit / size / Shapley failures.
pub fn distribution_change_robust(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &RobustChangeOptions,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    query.validate()?;
    require_mechanism_components(
        query.components,
        "distribution_change_robust requires AttributionComponents::Mechanisms",
    )?;

    let (baseline_data, comparison_data) = resolve_change_populations(data, query)?;

    if baseline_data.row_count() > options.max_rows
        || comparison_data.row_count() > options.max_rows
    {
        return Err(AttributionError::SizeLimit {
            kind: "units",
            requested: baseline_data.row_count().max(comparison_data.row_count()),
            max: options.max_rows,
        });
    }

    let outcome_dense = resolve_outcome_dense(graph_model, query.outcome)?;
    let players = mechanism_players(graph_model, outcome_dense, query.max_components)?;
    let kinds = vec![PlayerKind::Mechanism; players.len()];

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

    let all_linear = players.iter().all(|c| {
        model_slot_is_linear(&baseline_mechs, graph_model, c.variable())
            && model_slot_is_linear(&comparison_mechs, graph_model, c.variable())
    });

    let approximation = require_shapley_config(
        &query.allocation,
        "distribution_change_robust currently supports Shapley allocation",
    )?;

    let (v0, v_full, estimate) = if all_linear {
        let mut payoff = RobustLinearPayoff {
            model: graph_model,
            baseline: &baseline_data,
            comparison: &comparison_data,
            players: players.clone(),
            outcome: query.outcome,
            fitted: Vec::new(),
        };
        payoff.fit()?;
        let v0 = payoff.value(0)?;
        let full = (1u64 << players.len()) - 1;
        let v_full = payoff.value(full)?;
        let estimate = estimate_shapley(&players, approximation, &mut payoff, ctx)?;
        (v0, v_full, estimate)
    } else {
        let mut payoff = RobustMechanismPayoff {
            template: graph_model.clone(),
            baseline: baseline_mechs,
            comparison: comparison_mechs,
            baseline_data: &baseline_data,
            players: players.clone(),
            kinds,
            outcome: outcome_dense,
            ws: MechanismWorkspace::default(),
        };
        let v0 = payoff.value(0)?;
        let full = (1u64 << players.len()) - 1;
        let v_full = payoff.value(full)?;
        let estimate = estimate_shapley(&players, approximation, &mut payoff, ctx)?;
        (v0, v_full, estimate)
    };
    let total_change = v_full - v0;
    let mc_stderr = estimate.monte_carlo_stderr;
    let component_mc = estimate.component_mc_stderr.clone().map(Arc::from);
    let cache_stats = estimate.cache_stats.clone();
    let budget = estimate.budget.clone();
    let contributions = Arc::from(estimate.into_contributions());

    Ok(ChangeAttributionResult {
        outcome: query.outcome,
        total_change,
        contributions,
        interactions: Arc::from([]),
        path_breakdown: Arc::from([]),
        unidentified: Arc::from([]),
        graph_sensitivity: None,
        budget,
        monte_carlo_stderr: mc_stderr,
        component_mc_stderr: component_mc,
        cache_stats,
    })
}

fn model_slot_is_linear(
    store: &CompiledMechanismStore,
    model: &CompiledCausalModel,
    var: VariableId,
) -> bool {
    let Some(d) = model.dense_of(var) else {
        return false;
    };
    matches!(
        &store.slots[d.as_usize()],
        MechanismSlot::LinearGaussian { .. }
            | MechanismSlot::HierarchicalLinear { .. }
            | MechanismSlot::Bvar { .. }
            | MechanismSlot::Constant { .. }
    )
}

struct NodeRegression {
    baseline_beta: Vec<f64>,
    comparison_beta: Vec<f64>,
    parents: Vec<VariableId>,
}

struct RobustLinearPayoff<'a> {
    model: &'a CompiledCausalModel,
    baseline: &'a TabularData,
    comparison: &'a TabularData,
    players: Vec<ComponentId>,
    outcome: VariableId,
    fitted: Vec<NodeRegression>,
}

impl RobustLinearPayoff<'_> {
    fn fit(&mut self) -> Result<(), AttributionError> {
        let backend = FaerBackend;
        let mut ws = LeastSquaresWorkspace::default();
        self.fitted.clear();
        for &comp in &self.players {
            let dense = self
                .model
                .dense_of(comp.variable())
                .ok_or_else(|| AttributionError::missing_var("component", comp.variable()))?;
            let gather = self
                .model
                .gather_for(dense)
                .ok_or(AttributionError::MissingArtifact("missing gather"))?;
            let parents: Vec<VariableId> = gather
                .parents
                .iter()
                .map(|&p| self.model.output_layout.variables[p.as_usize()])
                .collect();
            let baseline_beta =
                fit_linear(self.baseline, comp.variable(), &parents, backend, &mut ws)?;
            let comparison_beta =
                fit_linear(self.comparison, comp.variable(), &parents, backend, &mut ws)?;
            self.fitted.push(NodeRegression { baseline_beta, comparison_beta, parents });
        }
        Ok(())
    }
}

impl CoalitionPayoff for RobustLinearPayoff<'_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        let n = self.baseline.row_count();
        let mut pred_out = vec![0.0; n];
        let mut outcome_seen = false;
        let mut node_pred: Vec<Vec<f64>> = Vec::with_capacity(self.players.len());
        for (i, &comp) in self.players.iter().enumerate() {
            let fit = &self.fitted[i];
            let beta =
                if mask & (1u64 << i) != 0 { &fit.comparison_beta } else { &fit.baseline_beta };
            let mut col = vec![0.0; n];
            for r in 0..n {
                let mut yhat = beta[0];
                for (pi, &p) in fit.parents.iter().enumerate() {
                    let x = if let Some(pj) = self.players.iter().position(|c| c.variable() == p) {
                        node_pred[pj][r]
                    } else {
                        self.baseline.float64_values(p)?[r]
                    };
                    yhat += beta.get(pi + 1).copied().unwrap_or(0.0) * x;
                }
                col[r] = yhat;
            }
            if comp.variable() == self.outcome {
                pred_out.clone_from(&col);
                outcome_seen = true;
            }
            node_pred.push(col);
        }
        if !outcome_seen {
            return Err(AttributionError::unsupported(
                "robust payoff: outcome is not among Shapley players",
            ));
        }
        Ok(pred_out.iter().sum::<f64>() / n.max(1) as f64)
    }
}

/// Nonlinear robust payoff: hybrid mechanisms, evaluate at ε=0 on baseline parents.
struct RobustMechanismPayoff<'a> {
    template: CompiledCausalModel,
    baseline: CompiledMechanismStore,
    comparison: CompiledMechanismStore,
    baseline_data: &'a TabularData,
    players: Vec<ComponentId>,
    kinds: Vec<PlayerKind>,
    outcome: DenseNodeId,
    ws: MechanismWorkspace,
}

impl CoalitionPayoff for RobustMechanismPayoff<'_> {
    fn value(&mut self, mask: u64) -> Result<f64, AttributionError> {
        let store = hybrid_mechanisms(
            &self.baseline,
            &self.comparison,
            &self.template,
            &self.players,
            &self.kinds,
            mask,
        );
        let model = self.template.clone().with_mechanisms(store);
        let n = self.baseline_data.row_count();
        let n_nodes = model.n_nodes();
        let mut values = vec![0.0; n * n_nodes];
        for (i, &var) in model.output_layout.variables.iter().enumerate() {
            let col = self.baseline_data.float64_values(var)?;
            values[i * n..(i + 1) * n].copy_from_slice(&col[..n]);
        }
        // Topo re-evaluate each node at zero noise under hybrid mechanisms.
        for &node in model.node_order.iter() {
            let gather = model
                .gather_for(node)
                .ok_or(AttributionError::MissingArtifact("missing gather"))?;
            self.ws.prepare(n, gather.n_parents().max(1));
            gather.gather(&values, n, &mut self.ws.parents);
            let parent_owned = self.ws.parents[..gather.n_parents().saturating_mul(n)].to_vec();
            let parents =
                ParentBatch { n_rows: n, n_parents: gather.n_parents(), values: &parent_owned };
            let zeros = vec![0.0; n];
            let out = &mut values[node.as_usize() * n..(node.as_usize() + 1) * n];
            evaluate_column(model.mechanisms.get(node), parents, &zeros, out, &mut self.ws)?;
        }
        let col = &values[self.outcome.as_usize() * n..(self.outcome.as_usize() + 1) * n];
        Ok(col.iter().sum::<f64>() / n.max(1) as f64)
    }
}

fn fit_linear(
    data: &TabularData,
    y_id: VariableId,
    parents: &[VariableId],
    backend: FaerBackend,
    ws: &mut LeastSquaresWorkspace,
) -> Result<Vec<f64>, AttributionError> {
    use causal_stats::DenseLinearAlgebra;
    let n = data.row_count();
    let p = parents.len() + 1;
    let y = data.float64_values(y_id)?;
    let mut x = vec![0.0; n * p];
    for r in 0..n {
        x[r] = 1.0;
    }
    for (pi, &pid) in parents.iter().enumerate() {
        let col = data.float64_values(pid)?;
        for r in 0..n {
            x[(pi + 1) * n + r] = col[r];
        }
    }
    let fit = backend.least_squares(&x, n, p, &y, ws)?;
    Ok(fit.coefficients.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        AllocationMethod, CausalSchemaBuilder, MeasurementSpec, PopulationSelector, RoleHint,
        ShapleyConfig, SmallRoleSet, ValueType,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::{Dag, DenseNodeId};

    #[test]
    fn robust_linear_still_runs() {
        let n = 60usize;
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
            let x = (i % 30) as f64 * 0.1;
            xv.push(x);
            yv.push(if i < 30 { x } else { x + 2.0 });
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
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&model, &data, SelectionPolicy::BestScore)
            .unwrap();
        let model = model.with_mechanisms(store);
        let query = ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::Rows(Arc::from((0..30).collect::<Vec<_>>())),
            PopulationSelector::Rows(Arc::from((30..60).collect::<Vec<_>>())),
        )
        .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
        let ctx = ExecutionContext::for_tests(1);
        let result = distribution_change_robust(
            &model,
            &data,
            &query,
            &RobustChangeOptions::default(),
            &ctx,
        )
        .unwrap();
        assert!(result.total_change.abs() > 0.5);
    }
}
