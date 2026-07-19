//! Path-specific contribution decomposition.
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
/// along each path (product of signed β on linear-Gaussian edges).
///
/// Signed products preserve cancelling paths; `total_change` is the sum of signed
/// path shares (not absolute shares). Non-linear mechanisms are refused.
///
/// # Errors
///
/// Missing nodes, path enumeration limits, or non-linear mechanisms on a path.
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
        .ok_or_else(|| AttributionError::missing_var("outcome", outcome))?;
    let strengths = edge_strength_map(model)?;

    if sources.len() > 1 {
        for (i, &src_i) in sources.iter().enumerate() {
            let src_i_dense = model
                .dense_of(src_i)
                .ok_or_else(|| AttributionError::missing_var("source", src_i))?;
            for &src_j in sources.iter().skip(i + 1) {
                let src_j_dense = model
                    .dense_of(src_j)
                    .ok_or_else(|| AttributionError::missing_var("source", src_j))?;
                if model.graph.reaches(src_i_dense, src_j_dense)
                    || model.graph.reaches(src_j_dense, src_i_dense)
                {
                    return Err(AttributionError::unsupported(
                        "path_decompose with multiple sources requires disjoint ancestry; \
                         one source is a directed ancestor of another",
                    ));
                }
            }
        }
    }

    let mut path_breakdown = Vec::new();
    let mut component_scores: Vec<(ComponentId, f64)> = Vec::new();
    let mut evaluations = 0u64;

    for &src in sources {
        let src_dense = model
            .dense_of(src)
            .ok_or_else(|| AttributionError::missing_var("source", src))?;
        let paths = model.graph.directed_paths(src_dense, outcome_dense, max_paths, max_len)?;
        for path in paths {
            evaluations += 1;
            let mut share = 1.0;
            let mut vars = Vec::with_capacity(path.len());
            for w in path.windows(2) {
                let key = (w[0], w[1]);
                let Some(&beta) = strengths.get(&key) else {
                    return Err(AttributionError::MissingEdgeCoefficient);
                };
                share *= beta;
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
) -> Result<std::collections::HashMap<(DenseNodeId, DenseNodeId), f64>, AttributionError> {
    let mut m = std::collections::HashMap::new();
    for gather in model.parent_gathers.iter() {
        let child = gather.child;
        match model.mechanisms.get(child) {
            causal_model::MechanismSlot::LinearGaussian { coeffs, .. } => {
                if coeffs.len() < gather.parents.len() {
                    return Err(AttributionError::MechanismCoeffMismatch);
                }
                for (i, &p) in gather.parents.iter().enumerate() {
                    m.insert((p, child), coeffs[i]);
                }
            }
            other if !gather.parents.is_empty() => {
                let _ = other;
                return Err(AttributionError::NonLinearGaussianMechanism);
            }
            _ => {}
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
    use causal_graph::{Dag, DenseNodeId};
    use causal_model::{
        CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismSlot,
        SelectionPolicy,
    };

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

    /// Pinned linear SEM X→M→Y with β=2,3: path product = 6 and Σ path shares = total_change.
    #[test]
    fn path_product_pins_and_sums_to_total_change() {
        let mut g = Dag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let store = CompiledMechanismStore {
            slots: Arc::from([
                MechanismSlot::Constant { value: 0.0 },
                MechanismSlot::LinearGaussian {
                    intercept: 0.0,
                    coeffs: Arc::from([2.0]),
                    sigma: 0.1,
                },
                MechanismSlot::LinearGaussian {
                    intercept: 0.0,
                    coeffs: Arc::from([3.0]),
                    sigma: 0.1,
                },
            ]),
        };
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
        assert_eq!(result.path_breakdown.len(), 1);
        assert!(
            (result.path_breakdown[0].contribution - 6.0).abs() < 1e-12,
            "path product={}",
            result.path_breakdown[0].contribution
        );
        let path_sum: f64 = result.path_breakdown.iter().map(|p| p.contribution).sum();
        assert!(
            (path_sum - result.total_change).abs() < 1e-12,
            "sum(path)={path_sum} total={}",
            result.total_change
        );
        assert!((result.total_change - 6.0).abs() < 1e-12);
    }

    fn chain_model() -> CompiledCausalModel {
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
        compiled.with_mechanisms(store)
    }

    #[test]
    fn nested_multi_source_errors() {
        let model = chain_model();
        let err = path_decompose(
            &model,
            &[VariableId::from_raw(0), VariableId::from_raw(1)],
            VariableId::from_raw(2),
            10,
            8,
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert!(matches!(err, AttributionError::Unsupported { .. }));
    }

    #[test]
    fn disjoint_multi_source_succeeds() {
        let n = 30usize;
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "z", "y"] {
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
        let zv: Vec<f64> = (0..n).map(|i| (n - i) as f64 * 0.05).collect();
        let yv: Vec<f64> = xv
            .iter()
            .zip(zv.iter())
            .enumerate()
            .map(|(i, (x, z))| 2.0 * x + 3.0 * z + 0.3 * (i as f64 * 0.13).sin())
            .collect();
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
        let _data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let store = CompiledMechanismStore {
            slots: Arc::from([
                MechanismSlot::Constant { value: 0.0 },
                MechanismSlot::Constant { value: 0.0 },
                MechanismSlot::LinearGaussian {
                    intercept: 0.0,
                    coeffs: Arc::from([2.0, 3.0]),
                    sigma: 0.1,
                },
            ]),
        };
        let model = compiled.with_mechanisms(store);
        let result = path_decompose(
            &model,
            &[VariableId::from_raw(0), VariableId::from_raw(1)],
            VariableId::from_raw(2),
            10,
            8,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        assert_eq!(result.contributions.len(), 2);
        assert!(result.total_change > 0.0);
    }
}
