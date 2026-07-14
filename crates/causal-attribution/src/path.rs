//! Path-specific contribution decomposition (DESIGN.md §17.2 `PathBased`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{ComponentId, ExecutionContext, VariableId};
use causal_graph::DenseNodeId;
use causal_model::CompiledCausalModel;

use crate::error::AttributionError;
use crate::result::{
    CacheStats, ChangeAttributionResult, ComponentContribution, ComputeBudget, PathContribution,
};

/// Decompose outcome influence into directed-path shares using arrow strengths
/// along each path (product of |β| on linear edges, DP-aggregated).
///
/// # Errors
///
/// Missing nodes or path enumeration limits.
pub fn path_decompose(
    model: &CompiledCausalModel,
    sources: &[VariableId],
    outcome: VariableId,
    max_paths: usize,
    max_len: usize,
    _ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    let outcome_dense = model
        .dense_of(outcome)
        .ok_or_else(|| AttributionError::Message(format!("outcome {outcome} missing")))?;
    let strengths = edge_strength_map(model);

    let mut path_breakdown = Vec::new();
    let mut component_scores: Vec<(ComponentId, f64)> = Vec::new();
    let mut evaluations = 0u64;

    for &src in sources {
        let src_dense = model
            .dense_of(src)
            .ok_or_else(|| AttributionError::Message(format!("source {src} missing")))?;
        let paths = model.graph.directed_paths(src_dense, outcome_dense, max_paths, max_len)?;
        for path in paths {
            evaluations += 1;
            let mut share = 1.0;
            let mut vars = Vec::with_capacity(path.len());
            for w in path.windows(2) {
                let key = (w[0], w[1]);
                share *= strengths.get(&key).copied().unwrap_or(0.0);
            }
            for &n in &path {
                vars.push(model.output_layout.variables[n.as_usize()]);
            }
            // Attribute path share to the source node component.
            let src_comp = ComponentId::from_variable(src);
            if let Some(e) = component_scores.iter_mut().find(|(c, _)| *c == src_comp) {
                e.1 += share;
            } else {
                component_scores.push((src_comp, share));
            }
            path_breakdown.push(PathContribution { path: Arc::from(vars), contribution: share });
        }
    }

    let total: f64 = component_scores.iter().map(|(_, s)| *s).sum();
    let contributions: Vec<ComponentContribution> = component_scores
        .into_iter()
        .map(|(component, contribution)| ComponentContribution {
            component,
            contribution,
            stderr: None,
            ci_low: None,
            ci_high: None,
        })
        .collect();

    Ok(ChangeAttributionResult {
        outcome,
        total_change: total,
        contributions: Arc::from(contributions),
        interactions: Arc::from([]),
        path_breakdown: Arc::from(path_breakdown),
        unidentified: Arc::from([]),
        graph_sensitivity: None,
        budget: ComputeBudget { evaluations, samples: 0, exact_coalitions: 0 },
        monte_carlo_stderr: None,
        component_mc_stderr: None,
        cache_stats: CacheStats::default(),
    })
}

fn edge_strength_map(
    model: &CompiledCausalModel,
) -> std::collections::HashMap<(DenseNodeId, DenseNodeId), f64> {
    let mut m = std::collections::HashMap::new();
    for gather in model.parent_gathers.iter() {
        let child = gather.child;
        match model.mechanisms.get(child) {
            causal_model::MechanismSlot::LinearGaussian { coeffs, .. } => {
                for (i, &p) in gather.parents.iter().enumerate() {
                    m.insert((p, child), coeffs.get(i).copied().unwrap_or(0.0).abs());
                }
            }
            _ => {
                for &p in gather.parents.iter() {
                    m.insert((p, child), 0.0);
                }
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
    use causal_graph::{Dag, DenseNodeId};
    use causal_model::{MechanismRegistry, SelectionPolicy};

    #[test]
    fn path_share_on_chain() {
        let n = 30usize;
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "m", "y"] {
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
        let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let mv: Vec<f64> = xv.iter().map(|x| 2.0 * x).collect();
        let yv: Vec<f64> = mv.iter().map(|m| 3.0 * m).collect();
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(mv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
            .unwrap();
        let model = compiled.with_mechanisms(store);
        let result = path_decompose(
            &model,
            &[VariableId::from_raw(0)],
            VariableId::from_raw(2),
            10,
            8,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert!(!result.path_breakdown.is_empty());
        assert!(result.total_change > 0.0);
    }
}
