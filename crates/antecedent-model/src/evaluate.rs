//! Model evaluation and falsification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::DenseNodeId;
use antecedent_stats::ci::{CiWorkspace, PartialCorrelation, SignificanceMethod};

use crate::batch::{MechanismWorkspace, ParentBatch};
use crate::compile::{CompiledCausalModel, MechanismSlot};
use crate::error::ModelError;
use crate::mechanism::{infer_noise_column, log_prob_column};

/// Model falsification / evaluation report.
#[derive(Clone, Debug)]
pub struct ModelEvaluationReport {
    /// In-sample mean log-likelihood (higher better). No holdout split is performed.
    pub in_sample_loglik: f64,
    /// Mean absolute residual for invertible nodes.
    pub mean_abs_residual: f64,
    /// Residual independence p-values vs non-parent covariates (empty if none).
    pub residual_independence_p: Arc<[f64]>,
    /// Local Markov check p-values (node ⊥ non-descendants | parents).
    pub local_markov_p: Arc<[f64]>,
    /// Permutation baseline mean log-lik under shuffled outcomes.
    pub permutation_loglik: f64,
    /// Whether the model is considered falsified under alpha.
    pub falsified: bool,
    /// Alpha used for independence tests.
    pub alpha: f64,
    /// Notes.
    pub notes: Vec<Arc<str>>,
}

/// Evaluate a fitted model on data.
#[derive(Clone, Debug)]
pub struct ModelEvaluator {
    /// Significance level for CI tests.
    pub alpha: f64,
    /// Permutation replicates for baseline.
    pub n_permutations: usize,
    /// RNG seed for permutations.
    pub seed: u64,
}

impl Default for ModelEvaluator {
    fn default() -> Self {
        Self { alpha: 0.05, n_permutations: 20, seed: 0 }
    }
}

impl ModelEvaluator {
    /// Run evaluation / falsification suite.
    ///
    /// # Errors
    ///
    /// Data / mechanism failures.
    pub fn evaluate(
        &self,
        model: &CompiledCausalModel,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<ModelEvaluationReport, ModelError> {
        let n = data.row_count();
        if n == 0 {
            return Err(ModelError::Shape { message: "empty data for evaluation".into() });
        }
        let mut notes = Vec::new();
        let in_sample_loglik = mean_loglik(model, data)?;
        let (mean_abs_residual, residuals_by_node) = residual_summary(model, data)?;
        let residual_independence_p =
            residual_independence_tests(model, data, &residuals_by_node, self.alpha, ctx)?;
        let local_markov_p = local_markov_tests(model, data, self.alpha, ctx)?;
        // Prefer the caller's execution seed when the evaluator still has the default seed.
        let perm_seed = if self.seed == 0 { ctx.rng.master_seed() } else { self.seed };
        let permutation_loglik = permutation_baseline(model, data, self.n_permutations, perm_seed)?;

        let mut falsified = false;
        for &p in &residual_independence_p {
            if p < self.alpha {
                falsified = true;
                notes.push(Arc::from("residual independence rejected at alpha"));
                break;
            }
        }
        for &p in &local_markov_p {
            if p < self.alpha {
                falsified = true;
                notes.push(Arc::from("local Markov condition rejected at alpha"));
                break;
            }
        }
        if in_sample_loglik + 1.0 < permutation_loglik {
            // Model worse than noise baseline by a wide margin.
            notes.push(Arc::from("in-sample loglik near or below permutation baseline"));
        }

        Ok(ModelEvaluationReport {
            in_sample_loglik,
            mean_abs_residual,
            residual_independence_p: Arc::from(residual_independence_p),
            local_markov_p: Arc::from(local_markov_p),
            permutation_loglik,
            falsified,
            alpha: self.alpha,
            notes,
        })
    }
}

fn mean_loglik(model: &CompiledCausalModel, data: &TabularData) -> Result<f64, ModelError> {
    let n = data.row_count();
    let mut total = 0.0;
    let mut count = 0usize;
    for gather in model.parent_gathers.iter() {
        let node = gather.child;
        let var = model.output_layout.variables[node.as_usize()];
        let y = data.float64_values(var).map_err(ModelError::from)?;
        let mut parent_mat = vec![0.0; n * gather.n_parents().max(1)];
        for (pi, &p) in gather.parents.iter().enumerate() {
            let pv = model.output_layout.variables[p.as_usize()];
            let col = data.float64_values(pv).map_err(ModelError::from)?;
            parent_mat[pi * n..(pi + 1) * n].copy_from_slice(&col[..n]);
        }
        let parents = ParentBatch {
            n_rows: n,
            n_parents: gather.n_parents(),
            values: &parent_mat[..gather.n_parents().saturating_mul(n)],
        };
        let mut lp = vec![0.0; n];
        log_prob_column(model.mechanisms.get(node), &y, parents, &mut lp)?;
        for v in lp {
            if v.is_finite() {
                total += v;
                count += 1;
            }
        }
    }
    Ok(total / count.max(1) as f64)
}

type ResidualByNode = Vec<Option<Vec<f64>>>;

fn residual_summary(
    model: &CompiledCausalModel,
    data: &TabularData,
) -> Result<(f64, ResidualByNode), ModelError> {
    let n = data.row_count();
    let mut residuals_by_node = vec![None; model.n_nodes()];
    let mut abs_sum = 0.0;
    let mut abs_count = 0usize;
    let mut ws = MechanismWorkspace::default();
    for gather in model.parent_gathers.iter() {
        let node = gather.child;
        let slot = model.mechanisms.get(node);
        if !matches!(
            slot,
            MechanismSlot::LinearGaussian { .. }
                | MechanismSlot::HierarchicalLinear { .. }
                | MechanismSlot::Bvar { .. }
        ) {
            continue;
        }
        let var = model.output_layout.variables[node.as_usize()];
        let y = data.float64_values(var).map_err(ModelError::from)?;
        ws.prepare(n, gather.n_parents().max(1));
        let mut parent_mat = vec![0.0; n * gather.n_parents().max(1)];
        for (pi, &p) in gather.parents.iter().enumerate() {
            let pv = model.output_layout.variables[p.as_usize()];
            let col = data.float64_values(pv).map_err(ModelError::from)?;
            parent_mat[pi * n..(pi + 1) * n].copy_from_slice(&col[..n]);
        }
        let parents = ParentBatch {
            n_rows: n,
            n_parents: gather.n_parents(),
            values: &parent_mat[..gather.n_parents().saturating_mul(n)],
        };
        let mut noise = vec![0.0; n];
        infer_noise_column(slot, &y, parents, &mut noise)?;
        for &e in &noise {
            abs_sum += e.abs();
            abs_count += 1;
        }
        residuals_by_node[node.as_usize()] = Some(noise);
    }
    Ok((abs_sum / abs_count.max(1) as f64, residuals_by_node))
}

fn residual_independence_tests(
    model: &CompiledCausalModel,
    data: &TabularData,
    residuals: &[Option<Vec<f64>>],
    _alpha: f64,
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, ModelError> {
    let mut ps = Vec::new();
    let test = PartialCorrelation::new();
    let mut ws = CiWorkspace::default();
    let children = child_adjacency(model);
    for (node_i, resid_opt) in residuals.iter().enumerate() {
        let Some(resid) = resid_opt else { continue };
        let gather = model.gather_for(DenseNodeId::from_raw(node_i as u32)).unwrap();
        let parent_set: std::collections::HashSet<usize> =
            gather.parents.iter().map(|p| p.as_usize()).collect();
        let descendants = descendants_of(&children, node_i);
        for other in 0..model.n_nodes() {
            // ANM residuals are independent of non-descendants (parents already skipped).
            // Dependence on descendants is expected and must not falsify a correct model.
            if other == node_i || parent_set.contains(&other) || descendants.contains(&other) {
                continue;
            }
            let ovar = model.output_layout.variables[other];
            let x = data.float64_values(ovar).map_err(ModelError::from)?;
            let cols: [&[f64]; 2] = [resid.as_slice(), x.as_slice()];
            let res = test
                .test_one(&cols, &[], SignificanceMethod::Analytic, &mut ws, ctx)
                .map_err(ModelError::from)?;
            ps.push(res.p_value);
        }
    }
    Ok(ps)
}

fn child_adjacency(model: &CompiledCausalModel) -> Vec<Vec<usize>> {
    let mut children = vec![Vec::new(); model.n_nodes()];
    for gather in model.parent_gathers.iter() {
        let child = gather.child.as_usize();
        for &p in gather.parents.iter() {
            children[p.as_usize()].push(child);
        }
    }
    children
}

fn descendants_of(children: &[Vec<usize>], node: usize) -> std::collections::HashSet<usize> {
    let mut out = std::collections::HashSet::new();
    let mut stack = children.get(node).cloned().unwrap_or_default();
    while let Some(v) = stack.pop() {
        if out.insert(v) {
            stack.extend(children.get(v).into_iter().flatten().copied());
        }
    }
    out
}

fn local_markov_tests(
    model: &CompiledCausalModel,
    data: &TabularData,
    _alpha: f64,
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, ModelError> {
    let test = PartialCorrelation::new();
    let mut ws = CiWorkspace::default();
    let mut ps = Vec::new();
    for gather in model.parent_gathers.iter() {
        let node = gather.child;
        let var = model.output_layout.variables[node.as_usize()];
        let y = data.float64_values(var).map_err(ModelError::from)?;
        let parent_ids: Vec<usize> = gather.parents.iter().map(|p| p.as_usize()).collect();
        let node_pos = model.node_order.iter().position(|d| *d == node).unwrap_or(0);
        for (oi, _) in model.node_order.iter().enumerate() {
            if oi >= node_pos || parent_ids.contains(&oi) {
                continue;
            }
            let ovar = model.output_layout.variables[oi];
            let x = data.float64_values(ovar).map_err(ModelError::from)?;
            let mut cols: Vec<&[f64]> = vec![y.as_slice(), x.as_slice()];
            let mut cond_storage: Vec<Vec<f64>> = Vec::new();
            for &p in &parent_ids {
                let pv = model.output_layout.variables[p];
                cond_storage.push(data.float64_values(pv).map_err(ModelError::from)?);
            }
            for c in &cond_storage {
                cols.push(c.as_slice());
            }
            let z: Vec<usize> = (2..cols.len()).collect();
            let res = test
                .test_one(&cols, &z, SignificanceMethod::Analytic, &mut ws, ctx)
                .map_err(ModelError::from)?;
            ps.push(res.p_value);
        }
        let _ = var;
    }
    Ok(ps)
}

fn permutation_baseline(
    model: &CompiledCausalModel,
    data: &TabularData,
    n_perm: usize,
    seed: u64,
) -> Result<f64, ModelError> {
    if n_perm == 0 {
        return Ok(f64::NEG_INFINITY);
    }
    let mut rng = CausalRng::from_seed(seed);
    // Permute a leaf outcome column and recompute mean loglik under original mechanisms
    // as a crude noise baseline (same X, shuffled Y for last node).
    let last = *model
        .node_order
        .last()
        .ok_or_else(|| ModelError::Shape { message: "empty model".into() })?;
    let var = model.output_layout.variables[last.as_usize()];
    let mut y = data.float64_values(var).map_err(ModelError::from)?;
    let mut acc = 0.0;
    for _ in 0..n_perm {
        // Fisher–Yates
        for i in (1..y.len()).rev() {
            let j = (rng.next_f64() * (i as f64 + 1.0)) as usize;
            y.swap(i, j.min(i));
        }
        // Score only the last node under shuffled y.
        let gather = model.gather_for(last).unwrap();
        let n = y.len();
        let mut parent_mat = vec![0.0; n * gather.n_parents().max(1)];
        for (pi, &p) in gather.parents.iter().enumerate() {
            let pv = model.output_layout.variables[p.as_usize()];
            let col = data.float64_values(pv).map_err(ModelError::from)?;
            parent_mat[pi * n..(pi + 1) * n].copy_from_slice(&col[..n]);
        }
        let parents = ParentBatch {
            n_rows: n,
            n_parents: gather.n_parents(),
            values: &parent_mat[..gather.n_parents().saturating_mul(n)],
        };
        let mut lp = vec![0.0; n];
        log_prob_column(model.mechanisms.get(last), &y, parents, &mut lp)?;
        acc += lp.iter().filter(|v| v.is_finite()).sum::<f64>() / n.max(1) as f64;
    }
    Ok(acc / n_perm as f64)
}

/// Mechanism predictive check: compare observed mean to predictive mean under sampling.
#[derive(Clone, Debug)]
pub struct MechanismPredictiveCheck {
    /// Sims.
    pub n_sims: usize,
    /// Seed.
    pub seed: u64,
}

impl Default for MechanismPredictiveCheck {
    fn default() -> Self {
        Self { n_sims: 50, seed: 1 }
    }
}

impl MechanismPredictiveCheck {
    /// Check one variable's mean.
    ///
    /// # Errors
    ///
    /// Sampling failures.
    pub fn check_mean(
        &self,
        model: &CompiledCausalModel,
        data: &TabularData,
        var: VariableId,
    ) -> Result<(f64, f64, f64), ModelError> {
        use crate::sample::sample_observational;
        use antecedent_core::ExecutionContext;

        let observed = data.float64_values(var).map_err(ModelError::from)?;
        let obs_mean = observed.iter().sum::<f64>() / observed.len().max(1) as f64;
        let dense = model
            .dense_of(var)
            .ok_or_else(|| ModelError::Shape { message: "variable not in model".into() })?;
        let mut rng = CausalRng::from_seed(self.seed);
        let mut ws = MechanismWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let mut means = Vec::with_capacity(self.n_sims);
        for _ in 0..self.n_sims {
            let batch = sample_observational(model, observed.len(), &mut rng, &mut ws, &ctx)?;
            let col = batch.column(dense.as_usize())?;
            means.push(col.iter().sum::<f64>() / col.len().max(1) as f64);
        }
        let pred_mean = means.iter().sum::<f64>() / means.len().max(1) as f64;
        let below = means.iter().filter(|&&m| m <= obs_mean).count() as f64;
        let p = (2.0
            * (below / means.len().max(1) as f64).min(1.0 - below / means.len().max(1) as f64))
        .min(1.0);
        Ok((obs_mean, pred_mean, p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{MechanismRegistry, SelectionPolicy};
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
    use antecedent_graph::Dag;

    #[test]
    fn evaluation_runs_on_linear_scm() {
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
        let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let yv: Vec<f64> = xv.iter().map(|x| 1.0 + 2.0 * x).collect();
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
        let rep = ModelEvaluator::default()
            .evaluate(&model, &data, &ExecutionContext::for_tests(1))
            .unwrap();
        assert!(rep.in_sample_loglik.is_finite());
        assert!(rep.mean_abs_residual < 1e-6, "resid={}", rep.mean_abs_residual);
    }
}
