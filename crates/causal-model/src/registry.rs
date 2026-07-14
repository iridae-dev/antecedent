//! Mechanism registry and auto-assignment (DESIGN.md §15.3).
//!
//! Assignment returns candidates and scores; there is no silent default family.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use std::sync::Arc;

use causal_core::VariableId;
use causal_data::{TableView, TabularData};
use causal_graph::DenseNodeId;
use causal_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::compile::{CompiledCausalModel, CompiledMechanismStore, MechanismSlot, ParentGatherPlan};
use crate::error::ModelError;

/// Candidate mechanism family known to the registry.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MechanismFamily {
    /// Linear Gaussian additive noise (invertible).
    LinearGaussian,
    /// Constant (root or intercept-only).
    Constant,
    /// Discrete categorical (unconditional root or parent-conditional softmax).
    Discrete,
}

impl MechanismFamily {
    /// Registry id string.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::LinearGaussian => "linear_gaussian",
            Self::Constant => "constant",
            Self::Discrete => "discrete",
        }
    }
}

/// Scored candidate for one node.
#[derive(Clone, Debug)]
pub struct MechanismCandidate {
    /// Family.
    pub family: MechanismFamily,
    /// Validation score (higher is better; e.g. negative MSE or log-lik).
    pub score: f64,
    /// Estimated fit cost (relative).
    pub fit_cost: f64,
    /// Estimated evaluation cost (relative).
    pub eval_cost: f64,
}

/// Result of auto-assignment for one node.
#[derive(Clone, Debug)]
pub struct MechanismAssignment {
    /// Dense node.
    pub node: DenseNodeId,
    /// Variable.
    pub variable: VariableId,
    /// All scored candidates (sorted descending by score).
    pub candidates: Arc<[MechanismCandidate]>,
    /// Selected family (must be chosen explicitly from candidates).
    pub selected: MechanismFamily,
    /// Fitted slot.
    pub fitted: MechanismSlot,
}

/// Registry of mechanism families.
#[derive(Clone, Debug)]
pub struct MechanismRegistry {
    /// Families considered for continuous nodes.
    pub continuous: Arc<[MechanismFamily]>,
    /// Families considered for discrete / low-cardinality nodes.
    pub discrete: Arc<[MechanismFamily]>,
}

impl Default for MechanismRegistry {
    fn default() -> Self {
        Self::standard()
    }
}

impl MechanismRegistry {
    /// Standard Phase 7 registry.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            continuous: Arc::from(vec![MechanismFamily::LinearGaussian, MechanismFamily::Constant]),
            discrete: Arc::from(vec![MechanismFamily::Discrete, MechanismFamily::Constant]),
        }
    }

    /// Assign and fit all nodes. Requires an explicit selection policy.
    ///
    /// # Errors
    ///
    /// Data / fit failures, or empty candidate sets.
    pub fn assign_and_fit(
        &self,
        model: &CompiledCausalModel,
        data: &TabularData,
        policy: SelectionPolicy,
    ) -> Result<(CompiledMechanismStore, Vec<MechanismAssignment>), ModelError> {
        let n = model.n_nodes();
        let nrows = data.row_count();
        if nrows == 0 {
            return Err(ModelError::Shape { message: "empty data for mechanism fit".into() });
        }
        let mut slots = vec![MechanismSlot::Vacant; n];
        let mut assignments = Vec::with_capacity(n);
        let backend = FaerBackend;
        let mut ls_ws = LeastSquaresWorkspace::default();

        for gather in model.parent_gathers.iter() {
            let node = gather.child;
            let var = model.output_layout.variables[node.as_usize()];
            let y = data.float64_values(var).map_err(|e| ModelError::Data(e.to_string()))?;
            let is_discrete = is_low_cardinality(&y, 8);
            let families: &[MechanismFamily] =
                if is_discrete { &self.discrete } else { &self.continuous };

            let mut candidates = Vec::new();
            for &family in families {
                match score_family(family, gather, model, data, &y, &backend, &mut ls_ws) {
                    Ok(c) => candidates.push(c),
                    Err(_) => continue,
                }
            }
            if candidates.is_empty() {
                return Err(ModelError::Unsupported {
                    message: format!("no mechanism candidates for variable {var}"),
                });
            }
            candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            let selected = policy.select(&candidates).ok_or_else(|| ModelError::Unsupported {
                message: "selection policy produced no family".into(),
            })?;
            let fitted = fit_family(selected, gather, model, data, &y, &backend, &mut ls_ws)?;
            slots[node.as_usize()] = fitted.clone();
            assignments.push(MechanismAssignment {
                node,
                variable: var,
                candidates: Arc::from(candidates),
                selected,
                fitted,
            });
        }

        Ok((CompiledMechanismStore { slots: Arc::from(slots) }, assignments))
    }
}

/// How to pick among scored candidates (no silent fallback).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SelectionPolicy {
    /// Highest validation score.
    BestScore,
    /// Require the named family to appear; error if missing.
    RequireFamily(MechanismFamily),
}

impl SelectionPolicy {
    /// Select a family.
    #[must_use]
    pub fn select(self, candidates: &[MechanismCandidate]) -> Option<MechanismFamily> {
        match self {
            Self::BestScore => candidates.first().map(|c| c.family),
            Self::RequireFamily(fam) => candidates.iter().find(|c| c.family == fam).map(|c| c.family),
        }
    }
}

fn is_low_cardinality(y: &[f64], max_levels: usize) -> bool {
    let mut vals: Vec<i64> = y
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v * 1e6).round() as i64)
        .collect();
    vals.sort_unstable();
    vals.dedup();
    !vals.is_empty() && vals.len() <= max_levels
}

fn score_family(
    family: MechanismFamily,
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: &FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismCandidate, ModelError> {
    let fitted = fit_family(family, gather, model, data, y, backend, ls_ws)?;
    let score = match &fitted {
        MechanismSlot::LinearGaussian { intercept, coeffs, sigma } => {
            let mse = residual_mse(gather, model, data, y, *intercept, coeffs)?;
            -mse - sigma.ln().abs() * 0.01
        }
        MechanismSlot::Constant { value } => {
            let mse = y.iter().map(|yi| (yi - value).powi(2)).sum::<f64>() / y.len().max(1) as f64;
            -mse
        }
        MechanismSlot::Discrete { probs, .. } => {
            let ent: f64 = probs.iter().map(|p| if *p > 0.0 { -p * p.ln() } else { 0.0 }).sum();
            -ent
        }
        _ => f64::NEG_INFINITY,
    };
    Ok(MechanismCandidate {
        family,
        score,
        fit_cost: 1.0 + gather.n_parents() as f64,
        eval_cost: 1.0 + gather.n_parents() as f64,
    })
}

fn fit_family(
    family: MechanismFamily,
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    backend: &FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    match family {
        MechanismFamily::Constant => {
            let mean = y.iter().sum::<f64>() / n.max(1) as f64;
            Ok(MechanismSlot::Constant { value: mean })
        }
        MechanismFamily::Discrete => {
            let mut pairs: Vec<(i64, f64, usize)> = Vec::new();
            for &yi in y {
                if !yi.is_finite() {
                    continue;
                }
                let key = (yi * 1e6).round() as i64;
                if let Some(e) = pairs.iter_mut().find(|(k, _, _)| *k == key) {
                    e.2 += 1;
                } else {
                    pairs.push((key, yi, 1));
                }
            }
            if pairs.is_empty() {
                return Err(ModelError::Shape { message: "no finite values for discrete fit".into() });
            }
            let total = pairs.iter().map(|(_, _, c)| *c).sum::<usize>() as f64;
            let support: Vec<f64> = pairs.iter().map(|(_, v, _)| *v).collect();
            let probs: Vec<f64> = pairs.iter().map(|(_, _, c)| *c as f64 / total).collect();
            let k = support.len();
            let p = gather.n_parents();
            if p == 0 {
                return Ok(MechanismSlot::Discrete {
                    support: Arc::from(support),
                    probs: Arc::from(probs),
                    logit_coeffs: None,
                });
            }
            // Parent-conditional: one-vs-rest least squares on category indicators → softmax logits.
            let ncols = 1 + p;
            let mut x = vec![0.0; n * ncols];
            for r in 0..n {
                x[r] = 1.0;
            }
            for (pi, &parent) in gather.parents.iter().enumerate() {
                let var = model.output_layout.variables[parent.as_usize()];
                let col =
                    data.float64_values(var).map_err(|e| ModelError::Data(e.to_string()))?;
                let base = (1 + pi) * n;
                x[base..base + n].copy_from_slice(&col[..n]);
            }
            let mut logit_coeffs = vec![0.0; k * ncols];
            for (cat, &sv) in support.iter().enumerate() {
                let indicators: Vec<f64> = y
                    .iter()
                    .map(|&yi| if (yi - sv).abs() < 1e-12 { 1.0 } else { 0.0 })
                    .collect();
                let fit = backend
                    .least_squares(&x, n, ncols, &indicators, ls_ws)
                    .map_err(|e| ModelError::Stats(e.to_string()))?;
                let base = cat * ncols;
                logit_coeffs[base..base + ncols].copy_from_slice(&fit.coefficients[..ncols]);
            }
            Ok(MechanismSlot::Discrete {
                support: Arc::from(support),
                probs: Arc::from(probs),
                logit_coeffs: Some(Arc::from(logit_coeffs)),
            })
        }
        MechanismFamily::LinearGaussian => {
            let p = gather.n_parents();
            let ncols = 1 + p;
            let mut x = vec![0.0; n * ncols];
            for r in 0..n {
                x[r] = 1.0;
            }
            for (pi, &parent) in gather.parents.iter().enumerate() {
                let var = model.output_layout.variables[parent.as_usize()];
                let col =
                    data.float64_values(var).map_err(|e| ModelError::Data(e.to_string()))?;
                let base = (1 + pi) * n;
                x[base..base + n].copy_from_slice(&col[..n]);
            }
            let fit = backend
                .least_squares(&x, n, ncols, y, ls_ws)
                .map_err(|e| ModelError::Stats(e.to_string()))?;
            let intercept = fit.coefficients[0];
            let coeffs: Arc<[f64]> = Arc::from(fit.coefficients[1..].to_vec());
            let sigma = (fit.rss / (n.saturating_sub(ncols)).max(1) as f64).sqrt().max(1e-8);
            Ok(MechanismSlot::LinearGaussian { intercept, coeffs, sigma })
        }
    }
}

fn residual_mse(
    gather: &ParentGatherPlan,
    model: &CompiledCausalModel,
    data: &TabularData,
    y: &[f64],
    intercept: f64,
    coeffs: &[f64],
) -> Result<f64, ModelError> {
    let n = y.len();
    let mut sse = 0.0;
    let mut parent_cols: Vec<Vec<f64>> = Vec::with_capacity(gather.n_parents());
    for &parent in gather.parents.iter() {
        let var = model.output_layout.variables[parent.as_usize()];
        parent_cols.push(data.float64_values(var).map_err(|e| ModelError::Data(e.to_string()))?);
    }
    for r in 0..n {
        let mut pred = intercept;
        for (p, col) in parent_cols.iter().enumerate() {
            pred += coeffs[p] * col[r];
        }
        let e = y[r] - pred;
        sse += e * e;
    }
    Ok(sse / n.max(1) as f64)
}

/// Collection of fitted models weighted by graph posterior mass.
#[derive(Clone, Debug)]
pub struct ModelCollection {
    /// Per-graph compiled models.
    pub models: Arc<[CompiledCausalModel]>,
    /// Graph keys aligned with `models`.
    pub graph_keys: Arc<[u64]>,
    /// Normalized weights (sum to 1 over identified graphs).
    pub weights: Arc<[f64]>,
}

impl ModelCollection {
    /// Build from parallel arrays.
    ///
    /// # Errors
    ///
    /// Length mismatch or non-positive weight sum.
    pub fn new(
        models: impl Into<Arc<[CompiledCausalModel]>>,
        graph_keys: impl Into<Arc<[u64]>>,
        weights: impl Into<Arc<[f64]>>,
    ) -> Result<Self, ModelError> {
        let models = models.into();
        let graph_keys = graph_keys.into();
        let weights = weights.into();
        if models.len() != graph_keys.len() || models.len() != weights.len() {
            return Err(ModelError::Shape {
                message: "ModelCollection length mismatch".into(),
            });
        }
        let sum: f64 = weights.iter().sum();
        if !(sum > 0.0) {
            return Err(ModelError::Shape { message: "ModelCollection weights non-positive".into() });
        }
        let weights: Arc<[f64]> = Arc::from(weights.iter().map(|w| w / sum).collect::<Vec<_>>());
        Ok(Self { models, graph_keys, weights })
    }

    /// Number of graphs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Empty check.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
    use causal_graph::{Dag, DenseNodeId};

    fn toy_data() -> (TabularData, Dag) {
        let n = 40usize;
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
        let mut xv = vec![0.0; n];
        let mut yv = vec![0.0; n];
        for i in 0..n {
            xv[i] = i as f64 * 0.1;
            yv[i] = 1.0 + 2.0 * xv[i];
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
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        (TabularData::new(storage), g)
    }

    #[test]
    fn auto_assign_linear_chain() {
        let (data, g) = toy_data();
        let compiled = CompiledCausalModel::compile(g).unwrap();
        let reg = MechanismRegistry::standard();
        let (store, assigns) =
            reg.assign_and_fit(&compiled, &data, SelectionPolicy::BestScore).unwrap();
        assert_eq!(assigns.len(), 2);
        assert!(matches!(store.get(DenseNodeId::from_raw(1)), MechanismSlot::LinearGaussian { .. }));
    }
}
