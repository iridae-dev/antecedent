//! Basic anomaly attribution and arrow strength (DESIGN.md §17 slice).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{AnomalyAttributionQuery, VariableId};
use causal_data::{TableView, TabularData};
use causal_model::{
    CompiledCausalModel, MechanismWorkspace, ParentBatch, infer_noise_column, log_prob_column,
};

use crate::error::AttributionError;

/// Per-unit anomaly score for a target variable.
#[derive(Clone, Debug)]
pub struct AnomalyScores {
    /// Target variable.
    pub target: VariableId,
    /// Row indices scored.
    pub rows: Arc<[usize]>,
    /// Anomaly scores (−log p under the fitted mechanism; higher = more anomalous).
    pub scores: Arc<[f64]>,
    /// Parent contribution magnitudes |noise| attributed via residual (same length).
    pub residual_abs: Arc<[f64]>,
}

/// Score anomalies for query targets under a fitted model.
///
/// # Errors
///
/// Size limit or data/model failures.
pub fn score_anomalies(
    model: &CompiledCausalModel,
    data: &TabularData,
    query: &AnomalyAttributionQuery,
) -> Result<Vec<AnomalyScores>, AttributionError> {
    query.validate()?;
    let n = data.row_count();
    let rows: Vec<usize> = match &query.unit_rows {
        Some(r) => r.to_vec(),
        None => (0..n).collect(),
    };
    if rows.len() > query.max_units {
        return Err(AttributionError::SizeLimit {
            kind: "units",
            requested: rows.len(),
            max: query.max_units,
        });
    }
    let mut out = Vec::with_capacity(query.targets.len());
    for &target in query.targets.iter() {
        let dense = model
            .dense_of(target)
            .ok_or_else(|| AttributionError::Message(format!("target {target} not in model")))?;
        let gather = model
            .gather_for(dense)
            .ok_or_else(|| AttributionError::Message("missing gather".into()))?;
        let y_all =
            data.float64_values(target).map_err(|e| AttributionError::Message(e.to_string()))?;
        let mut parent_mat = vec![0.0; n * gather.n_parents().max(1)];
        for (pi, &p) in gather.parents.iter().enumerate() {
            let pv = model.output_layout.variables[p.as_usize()];
            let col =
                data.float64_values(pv).map_err(|e| AttributionError::Message(e.to_string()))?;
            parent_mat[pi * n..(pi + 1) * n].copy_from_slice(&col[..n]);
        }
        let parents = ParentBatch {
            n_rows: n,
            n_parents: gather.n_parents(),
            values: &parent_mat[..gather.n_parents().saturating_mul(n)],
        };
        let mut lp = vec![0.0; n];
        log_prob_column(model.mechanisms.get(dense), &y_all, parents, &mut lp)?;
        let mut noise = vec![0.0; n];
        let parents2 = ParentBatch {
            n_rows: n,
            n_parents: gather.n_parents(),
            values: &parent_mat[..gather.n_parents().saturating_mul(n)],
        };
        let _ = infer_noise_column(model.mechanisms.get(dense), &y_all, parents2, &mut noise);

        let mut scores = Vec::with_capacity(rows.len());
        let mut resid = Vec::with_capacity(rows.len());
        for &r in &rows {
            scores.push(-lp[r]);
            resid.push(noise[r].abs());
        }
        out.push(AnomalyScores {
            target,
            rows: Arc::from(rows.clone()),
            scores: Arc::from(scores),
            residual_abs: Arc::from(resid),
        });
    }
    Ok(out)
}

/// Direct arrow strength: |β| for linear Gaussian edge parent→child, else 0.
#[derive(Clone, Debug)]
pub struct ArrowStrength {
    /// Parent variable.
    pub parent: VariableId,
    /// Child variable.
    pub child: VariableId,
    /// Strength.
    pub strength: f64,
}

/// Compute arrow strengths for all edges in the compiled model.
///
/// # Errors
///
/// Model issues.
pub fn arrow_strengths(
    model: &CompiledCausalModel,
) -> Result<Vec<ArrowStrength>, AttributionError> {
    let mut out = Vec::new();
    for gather in model.parent_gathers.iter() {
        let child_var = model.output_layout.variables[gather.child.as_usize()];
        match model.mechanisms.get(gather.child) {
            causal_model::MechanismSlot::LinearGaussian { coeffs, .. } => {
                for (i, &p) in gather.parents.iter().enumerate() {
                    let parent = model.output_layout.variables[p.as_usize()];
                    let s = coeffs.get(i).copied().unwrap_or(0.0).abs();
                    out.push(ArrowStrength { parent, child: child_var, strength: s });
                }
            }
            _ => {
                for &p in gather.parents.iter() {
                    let parent = model.output_layout.variables[p.as_usize()];
                    out.push(ArrowStrength { parent, child: child_var, strength: 0.0 });
                }
            }
        }
    }
    Ok(out)
}

/// Intrinsic causal influence of parent on child via do-contrast on a unit
/// (difference in child prediction when parent is set to observed ± delta).
///
/// Hard size limit on units.
///
/// # Errors
///
/// Size / model failures.
pub fn intrinsic_influence(
    model: &CompiledCausalModel,
    data: &TabularData,
    parent: VariableId,
    child: VariableId,
    delta: f64,
    max_units: usize,
) -> Result<f64, AttributionError> {
    use causal_core::{ExecutionContext, Intervention, Value};
    use causal_model::sample_interventional;

    let n = data.row_count().min(max_units);
    if data.row_count() > max_units {
        return Err(AttributionError::SizeLimit {
            kind: "units",
            requested: data.row_count(),
            max: max_units,
        });
    }
    let mut rng = causal_core::CausalRng::from_seed(0);
    let mut ws = MechanismWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let child_dense =
        model.dense_of(child).ok_or_else(|| AttributionError::Message("child missing".into()))?;
    let pcol = data.float64_values(parent).map_err(|e| AttributionError::Message(e.to_string()))?;
    let pmean = pcol.iter().sum::<f64>() / pcol.len().max(1) as f64;
    let hi = sample_interventional(
        model,
        &[Intervention::set(parent, Value::f64(pmean + 0.5 * delta))],
        n.max(1),
        &mut rng,
        &mut ws,
        &ctx,
    )?;
    let lo = sample_interventional(
        model,
        &[Intervention::set(parent, Value::f64(pmean - 0.5 * delta))],
        n.max(1),
        &mut rng,
        &mut ws,
        &ctx,
    )?;
    let hi_m = hi.column(child_dense.as_usize())?.iter().sum::<f64>() / n.max(1) as f64;
    let lo_m = lo.column(child_dense.as_usize())?.iter().sum::<f64>() / n.max(1) as f64;
    Ok((hi_m - lo_m).abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage};
    use causal_graph::{Dag, DenseNodeId};
    use causal_model::{MechanismRegistry, SelectionPolicy};

    #[test]
    fn anomaly_and_arrow_strength() {
        let n = 30usize;
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
        let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let mut yv: Vec<f64> = xv.iter().map(|x| 1.0 + 2.0 * x).collect();
        yv[n - 1] = 100.0; // anomaly
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
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let (store, _) = MechanismRegistry::standard()
            .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
            .unwrap();
        let model = compiled.with_mechanisms(store);
        let q = AnomalyAttributionQuery::new([VariableId::from_raw(1)], 100);
        let scores = score_anomalies(&model, &data, &q).unwrap();
        assert!(scores[0].scores[n - 1] > scores[0].scores[0]);
        let arrows = arrow_strengths(&model).unwrap();
        assert!(!arrows.is_empty());
        assert!(arrows.iter().any(|a| a.strength > 0.5), "arrows={arrows:?}");
    }
}
