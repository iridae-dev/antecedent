//! Observational and interventional batch sampling.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use causal_core::{CausalRng, ExecutionContext, Intervention, MechanismOverride, StochasticPolicy};
use causal_kernels::standard_normal;

use crate::batch::{MechanismWorkspace, NoiseBatchMut, ParentBatch, ValueBatch, ValueBatchMut};
use crate::compile::{CompiledCausalModel, MechanismSlot};
use crate::error::ModelError;
use crate::mechanism::{evaluate_column, sample_column, sample_noise_column};
use crate::overlay::{InterventionOverlay, ModelView};

/// Sample `n_rows` observational draws from a fitted model.
///
/// # Errors
///
/// Unfitted mechanisms or shape errors.
pub fn sample_observational(
    model: &CompiledCausalModel,
    n_rows: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
    _ctx: &ExecutionContext,
) -> Result<ValueBatch, ModelError> {
    let view = ModelView::observational(model);
    sample_with_overlay(&view, n_rows, rng, ws)
}

/// Sample under interventions (compiled to an overlay; model is not cloned).
///
/// # Errors
///
/// Overlay / mechanism failures.
pub fn sample_interventional(
    model: &CompiledCausalModel,
    interventions: &[Intervention],
    n_rows: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
    _ctx: &ExecutionContext,
) -> Result<ValueBatch, ModelError> {
    let overlay = InterventionOverlay::from_interventions(model, interventions)?;
    let view = ModelView::with_overlay(model, overlay);
    sample_with_overlay(&view, n_rows, rng, ws)
}

/// Core ancestral sampler with overlay.
///
/// # Errors
///
/// Mechanism failures.
pub fn sample_with_overlay(
    view: &ModelView<'_>,
    n_rows: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
) -> Result<ValueBatch, ModelError> {
    if n_rows == 0 {
        return Err(ModelError::Shape { message: "n_rows must be > 0".into() });
    }
    let model = view.model;
    let n_nodes = model.n_nodes();
    let mut values_buf = vec![0.0; n_rows * n_nodes];
    let mut values = ValueBatchMut::new(n_rows, n_nodes, &mut values_buf)?;
    let overlay = view.overlay.as_ref();

    for gather in model.parent_gathers.iter() {
        let node = gather.child;
        let idx = node.as_usize();
        ws.prepare(n_rows, gather.n_parents().max(1));
        gather.gather(values.values, n_rows, &mut ws.parents);
        let parents = ParentBatch {
            n_rows,
            n_parents: gather.n_parents(),
            values: &ws.parents[..gather.n_parents().saturating_mul(n_rows)],
        };
        // Copy parents to owned so we can mutably write the child column.
        let parent_owned = parents.values.to_vec();
        let parents = ParentBatch { n_rows, n_parents: gather.n_parents(), values: &parent_owned };

        let out = values.column_mut(idx)?;

        if let Some(v) = overlay.hard_set[idx] {
            out.fill(v);
            continue;
        }
        if let Some(policy) = &overlay.stochastic[idx] {
            sample_stochastic(policy, n_rows, rng, out)?;
            apply_shift(out, overlay.shifts[idx]);
            continue;
        }
        if let Some(soft) = &overlay.soft[idx] {
            let slot = soft_to_slot(soft, gather.n_parents())?;
            sample_column(&slot, parents, rng, out, ws)?;
            apply_shift(out, overlay.shifts[idx]);
            continue;
        }

        let slot = model.mechanisms.get(node);
        sample_column(slot, parents, rng, out, ws)?;
        apply_shift(out, overlay.shifts[idx]);
    }

    Ok(values.into_batch())
}

/// Sample under interventions conditioned on observed node values.
///
/// Strategy:
/// 1. **Rejection sampling** when conditions match within `1e-9` (exact / discrete).
/// 2. **Likelihood-weighting SIR** when rejection under-accepts: propose from `do(·)`,
///    weight by `∏_c p(condition_c | parents_c)` via [`log_prob_column`], resample.
///
/// Conditioning nodes must not be hard-intervened.
///
/// # Errors
///
/// Empty condition, intervened condition nodes, density failures, or empty weights.
pub fn sample_conditional_interventional(
    model: &CompiledCausalModel,
    interventions: &[Intervention],
    condition_nodes: &[causal_graph::DenseNodeId],
    condition_values: &[f64],
    n_rows: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<ValueBatch, ModelError> {
    if condition_nodes.is_empty() || condition_values.len() != condition_nodes.len() {
        return Err(ModelError::Shape {
            message: "conditional interventional sampling needs matching condition_nodes/values"
                .into(),
        });
    }
    if n_rows == 0 {
        return Err(ModelError::Shape { message: "n_rows must be > 0".into() });
    }
    let overlay = InterventionOverlay::from_interventions(model, interventions)?;
    for &node in condition_nodes {
        let idx = node.as_usize();
        if idx >= model.n_nodes() {
            return Err(ModelError::Shape { message: "condition node out of range".into() });
        }
        if overlay.hard_set[idx].is_some() {
            return Err(ModelError::Unsupported {
                message: "cannot condition on a hard-intervened node".into(),
            });
        }
    }

    let n_nodes = model.n_nodes();
    let mut accepted = vec![0.0; n_rows * n_nodes];
    let mut got = 0usize;
    let max_attempts = n_rows.saturating_mul(100).max(100);
    for _ in 0..max_attempts {
        if got >= n_rows {
            break;
        }
        let batch = sample_interventional(model, interventions, 1, rng, ws, ctx)?;
        let mut ok = true;
        for (i, &node) in condition_nodes.iter().enumerate() {
            let v = batch.column(node.as_usize())?[0];
            if (v - condition_values[i]).abs() > 1e-9 {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        for node in 0..n_nodes {
            accepted[node * n_rows + got] = batch.column(node)?[0];
        }
        got += 1;
    }
    if got >= n_rows {
        let _ = ctx;
        return Ok(ValueBatch { n_rows, n_nodes, values: accepted.into() });
    }

    // Likelihood-weighting / SIR for continuous conditions.
    sample_conditional_interventional_lw(
        model,
        interventions,
        condition_nodes,
        condition_values,
        n_rows,
        rng,
        ws,
        ctx,
    )
}

fn sample_conditional_interventional_lw(
    model: &CompiledCausalModel,
    interventions: &[Intervention],
    condition_nodes: &[causal_graph::DenseNodeId],
    condition_values: &[f64],
    n_rows: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<ValueBatch, ModelError> {
    use crate::mechanism::log_prob_column;

    let n_nodes = model.n_nodes();
    let n_particles = n_rows.saturating_mul(20).max(64);
    let proposal = sample_interventional(model, interventions, n_particles, rng, ws, ctx)?;
    let mut log_w = vec![0.0; n_particles];
    let mut lp_buf = vec![0.0; n_particles];

    for (ci, &node) in condition_nodes.iter().enumerate() {
        let gather = model.gather_for(node).ok_or_else(|| ModelError::Unsupported {
            message: format!("missing gather for condition node {node:?}"),
        })?;
        ws.prepare(n_particles, gather.n_parents().max(1));
        gather.gather(&proposal.values, n_particles, &mut ws.parents);
        let parent_owned = ws.parents[..gather.n_parents().saturating_mul(n_particles)].to_vec();
        let parents = ParentBatch {
            n_rows: n_particles,
            n_parents: gather.n_parents(),
            values: &parent_owned,
        };
        // Score the *conditioned* value under each particle's parents.
        let conditioned = vec![condition_values[ci]; n_particles];
        log_prob_column(model.mechanisms.get(node), &conditioned, parents, &mut lp_buf)?;
        for p in 0..n_particles {
            if !lp_buf[p].is_finite() {
                return Err(ModelError::Unsupported {
                    message: format!(
                        "conditional do: mechanism for node {node:?} cannot provide a finite density \
                         for likelihood weighting"
                    ),
                });
            }
            log_w[p] += lp_buf[p];
        }
    }

    let max_lw = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max_lw.is_finite() {
        return Err(ModelError::Unsupported {
            message: "conditional do: all likelihood weights are non-finite".into(),
        });
    }
    let mut weights = vec![0.0; n_particles];
    let mut sum_w = 0.0;
    for p in 0..n_particles {
        let w = (log_w[p] - max_lw).exp();
        weights[p] = w;
        sum_w += w;
    }
    if sum_w <= 0.0 {
        return Err(ModelError::Unsupported {
            message: "conditional do: likelihood weights sum to zero".into(),
        });
    }
    for w in &mut weights {
        *w /= sum_w;
    }

    // Systematic resampling.
    let mut accepted = vec![0.0; n_rows * n_nodes];
    let u0 = rng.next_f64() / n_rows as f64;
    let mut cdf = 0.0;
    let mut idx = 0usize;
    for i in 0..n_rows {
        let target = u0 + i as f64 / n_rows as f64;
        while idx + 1 < n_particles && cdf + weights[idx] < target {
            cdf += weights[idx];
            idx += 1;
        }
        for node in 0..n_nodes {
            accepted[node * n_rows + i] = proposal.column(node)?[idx];
            // Overwrite conditioned nodes with exact condition values.
        }
        for (ci, &node) in condition_nodes.iter().enumerate() {
            accepted[node.as_usize() * n_rows + i] = condition_values[ci];
        }
    }
    let _ = ctx;
    Ok(ValueBatch { n_rows, n_nodes, values: accepted.into() })
}

/// Posterior-predictive interventional sampling: for each coefficient draw block,
/// refresh `LinearGaussian` slots then sample. `draw_updater` mutates slots in place.
///
/// # Errors
///
/// Updater / sample failures.
pub fn sample_posterior_predictive<F>(
    model: &mut CompiledCausalModel,
    interventions: &[Intervention],
    n_rows_per_draw: usize,
    n_draws: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
    mut draw_updater: F,
    ctx: &ExecutionContext,
) -> Result<ValueBatch, ModelError>
where
    F: FnMut(usize, &mut CompiledCausalModel) -> Result<(), ModelError>,
{
    let n_nodes = model.n_nodes();
    let total_rows = n_rows_per_draw.saturating_mul(n_draws);
    let mut all = vec![0.0; total_rows * n_nodes];
    for d in 0..n_draws {
        draw_updater(d, model)?;
        let batch = sample_interventional(model, interventions, n_rows_per_draw, rng, ws, ctx)?;
        for node in 0..n_nodes {
            let src = batch.column(node)?;
            let dest_row0 = d * n_rows_per_draw;
            let dest = node * total_rows + dest_row0;
            all[dest..dest + n_rows_per_draw].copy_from_slice(src);
        }
    }
    Ok(ValueBatch { n_rows: total_rows, n_nodes, values: all.into() })
}

fn apply_shift(out: &mut [f64], shift: f64) {
    if shift != 0.0 {
        for v in out.iter_mut() {
            *v += shift;
        }
    }
}

/// Convert a soft [`MechanismOverride`] into a concrete mechanism slot.
///
/// `additive_shift` is rejected here — [`InterventionOverlay::from_interventions`] maps it
/// onto overlay shifts so sampling paths share noise semantics.
///
/// # Errors
///
/// Unknown family or shape mismatches.
pub fn soft_to_slot(
    soft: &MechanismOverride,
    n_parents: usize,
) -> Result<MechanismSlot, ModelError> {
    match soft.family_id.as_ref() {
        "constant" => {
            let v = soft.parameters.first().copied().unwrap_or(0.0);
            Ok(MechanismSlot::Constant { value: v })
        }
        "additive_shift" => Err(ModelError::Unsupported {
            message: "additive_shift soft overrides must be applied as Intervention::Shift / overlay shifts"
                .into(),
        }),
        "linear_gaussian" => {
            if soft.parameters.len() < 2 + n_parents {
                return Err(ModelError::Shape {
                    message: "linear_gaussian override needs intercept, coeffs..., sigma".into(),
                });
            }
            let intercept = soft.parameters[0];
            let coeffs = std::sync::Arc::from(soft.parameters[1..=n_parents].to_vec());
            let sigma = soft.parameters[1 + n_parents].max(1e-12);
            Ok(MechanismSlot::LinearGaussian { intercept, coeffs, sigma })
        }
        "hierarchical_linear" => {
            if soft.parameters.len() < 3 + n_parents {
                return Err(ModelError::Shape {
                    message: "hierarchical_linear override needs intercept, coeffs..., sigma, shrinkage"
                        .into(),
                });
            }
            let intercept = soft.parameters[0];
            let coeffs = std::sync::Arc::from(soft.parameters[1..=n_parents].to_vec());
            let sigma = soft.parameters[1 + n_parents].max(1e-12);
            let shrinkage = soft.parameters[2 + n_parents].max(0.0);
            Ok(MechanismSlot::HierarchicalLinear { intercept, coeffs, sigma, shrinkage })
        }
        "bvar" => {
            if soft.parameters.len() < 2 + n_parents {
                return Err(ModelError::Shape {
                    message: "bvar override needs intercept, coeffs..., sigma".into(),
                });
            }
            let intercept = soft.parameters[0];
            let coeffs = std::sync::Arc::from(soft.parameters[1..=n_parents].to_vec());
            let sigma = soft.parameters[1 + n_parents].max(1e-12);
            Ok(MechanismSlot::Bvar { intercept, coeffs, sigma })
        }
        "discrete" => soft_discrete_slot(soft, n_parents),
        "lgssm" => {
            if soft.parameters.len() < 4 {
                return Err(ModelError::Shape {
                    message: "lgssm override needs a, process_std, obs_std, initial_mean".into(),
                });
            }
            Ok(MechanismSlot::LinearGaussianStateSpace {
                a: soft.parameters[0],
                process_std: soft.parameters[1].max(1e-12),
                obs_std: soft.parameters[2].max(1e-12),
                initial_mean: soft.parameters[3],
            })
        }
        "gaussian_process" => soft_gp_slot(soft, n_parents),
        other => Err(ModelError::Unsupported {
            message: format!("unknown soft override family {other}"),
        }),
    }
}

fn soft_discrete_slot(
    soft: &MechanismOverride,
    n_parents: usize,
) -> Result<MechanismSlot, ModelError> {
    if soft.parameters.is_empty() {
        return Err(ModelError::Shape {
            message: "discrete override needs k, support..., probs/logits...".into(),
        });
    }
    let k = soft.parameters[0] as usize;
    if k == 0 {
        return Err(ModelError::Shape { message: "discrete override k must be > 0".into() });
    }
    if soft.parameters.len() < 1 + k {
        return Err(ModelError::Shape { message: "discrete override truncated support".into() });
    }
    let support: std::sync::Arc<[f64]> = std::sync::Arc::from(soft.parameters[1..=k].to_vec());
    let rest = &soft.parameters[1 + k..];
    if rest.len() == k {
        Ok(MechanismSlot::Discrete {
            support,
            probs: std::sync::Arc::from(rest.to_vec()),
            logit_coeffs: None,
        })
    } else if rest.len() == k * (1 + n_parents) {
        Ok(MechanismSlot::Discrete {
            support,
            probs: std::sync::Arc::from(vec![1.0 / k as f64; k]),
            logit_coeffs: Some(std::sync::Arc::from(rest.to_vec())),
        })
    } else {
        Err(ModelError::Shape {
            message: format!(
                "discrete override expects {k} probs or {} logits after support, got {}",
                k * (1 + n_parents),
                rest.len()
            ),
        })
    }
}

fn soft_gp_slot(soft: &MechanismOverride, n_parents: usize) -> Result<MechanismSlot, ModelError> {
    if soft.parameters.len() < 5 {
        return Err(ModelError::Shape {
            message: "gaussian_process override truncated header".into(),
        });
    }
    let length_scale = soft.parameters[0].max(1e-12);
    let variance = soft.parameters[1].max(0.0);
    let noise_std = soft.parameters[2].max(1e-12);
    let n_train = soft.parameters[3] as usize;
    let n_par = soft.parameters[4] as usize;
    if n_par != n_parents {
        return Err(ModelError::Shape {
            message: format!("gaussian_process override n_parents {n_par} != gather {n_parents}"),
        });
    }
    let need = 5 + n_train * n_par + n_train;
    if soft.parameters.len() < need {
        return Err(ModelError::Shape {
            message: format!(
                "gaussian_process override needs {need} params, got {}",
                soft.parameters.len()
            ),
        });
    }
    let x_train = std::sync::Arc::from(soft.parameters[5..5 + n_train * n_par].to_vec());
    let alpha = std::sync::Arc::from(
        soft.parameters[5 + n_train * n_par..5 + n_train * n_par + n_train].to_vec(),
    );
    Ok(MechanismSlot::GaussianProcess {
        length_scale,
        variance,
        noise_std,
        x_train,
        n_train,
        n_parents: n_par,
        alpha,
    })
}

/// Draw values from a stochastic intervention policy into `out`.
///
/// # Errors
///
/// Unsupported policy variants.
pub fn sample_stochastic(
    policy: &StochasticPolicy,
    n_rows: usize,
    rng: &mut CausalRng,
    out: &mut [f64],
) -> Result<(), ModelError> {
    match policy {
        StochasticPolicy::Bernoulli { p } => {
            for i in 0..n_rows {
                out[i] = if rng.next_f64() < *p { 1.0 } else { 0.0 };
            }
            Ok(())
        }
        StochasticPolicy::Gaussian { mean, variance } => {
            let s = variance.sqrt();
            for i in 0..n_rows {
                out[i] = mean + s * standard_normal(rng);
            }
            Ok(())
        }
        StochasticPolicy::Categorical { probs } => {
            let sum: f64 = probs.iter().sum::<f64>().max(f64::EPSILON);
            for i in 0..n_rows {
                let u = rng.next_f64() * sum;
                let mut acc = 0.0;
                let mut chosen = (probs.len() - 1) as f64;
                for (k, &p) in probs.iter().enumerate() {
                    acc += p;
                    if u <= acc {
                        chosen = k as f64;
                        break;
                    }
                }
                out[i] = chosen;
            }
            Ok(())
        }
        _ => Err(ModelError::Unsupported { message: "unknown stochastic policy".into() }),
    }
}

/// Structural path: sample noise then evaluate with overlays applied post-hoc for hard sets.
///
/// # Errors
///
/// Mechanism failures.
pub fn sample_structural_with_overlay(
    view: &ModelView<'_>,
    n_rows: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
) -> Result<(ValueBatch, Vec<f64>), ModelError> {
    let model = view.model;
    let n_nodes = model.n_nodes();
    let mut noise_buf = vec![0.0; n_rows * n_nodes];
    {
        let mut noise = NoiseBatchMut::new(n_rows, n_nodes, &mut noise_buf)?;
        for gather in model.parent_gathers.iter() {
            let idx = gather.child.as_usize();
            let col = noise.column_mut(idx)?;
            if view.overlay.hard_set[idx].is_some() || view.overlay.stochastic[idx].is_some() {
                col.fill(0.0);
            } else {
                sample_noise_column(model.mechanisms.get(gather.child), n_rows, rng, col)?;
            }
        }
    }
    let mut values_buf = vec![0.0; n_rows * n_nodes];
    let mut values = ValueBatchMut::new(n_rows, n_nodes, &mut values_buf)?;
    let overlay = view.overlay.as_ref();
    for gather in model.parent_gathers.iter() {
        let node = gather.child;
        let idx = node.as_usize();
        ws.prepare(n_rows, gather.n_parents().max(1));
        gather.gather(values.values, n_rows, &mut ws.parents);
        let parent_owned = ws.parents[..gather.n_parents().saturating_mul(n_rows)].to_vec();
        let parents = ParentBatch { n_rows, n_parents: gather.n_parents(), values: &parent_owned };
        let out = values.column_mut(idx)?;
        if let Some(v) = overlay.hard_set[idx] {
            out.fill(v);
            continue;
        }
        if let Some(policy) = &overlay.stochastic[idx] {
            sample_stochastic(policy, n_rows, rng, out)?;
            apply_shift(out, overlay.shifts[idx]);
            continue;
        }
        let noise_col = &noise_buf[idx * n_rows..(idx + 1) * n_rows];
        let slot = if let Some(soft) = &overlay.soft[idx] {
            soft_to_slot(soft, gather.n_parents())?
        } else {
            model.mechanisms.get(node).clone()
        };
        evaluate_column(&slot, parents, noise_col, out, ws)?;
        apply_shift(out, overlay.shifts[idx]);
    }
    Ok((values.into_batch(), noise_buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{MechanismRegistry, SelectionPolicy};
    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, Intervention, MeasurementSpec, RoleHint,
        SmallRoleSet, Value, ValueType, VariableId,
    };
    use causal_data::column::{Float64Column, ValidityBitmap};
    use causal_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
    use causal_graph::{Dag, DenseNodeId};
    use std::sync::Arc;

    fn fitted_chain() -> CompiledCausalModel {
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
        compiled.with_mechanisms(store)
    }

    #[test]
    fn hard_intervention_fixes_column() {
        let model = fitted_chain();
        let mut rng = CausalRng::from_seed(1);
        let mut ws = MechanismWorkspace::default();
        let t = VariableId::from_raw(0);
        let batch = sample_interventional(
            &model,
            &[Intervention::set(t, Value::f64(3.0))],
            20,
            &mut rng,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let col = batch.column(0).unwrap();
        assert!(col.iter().all(|&v| (v - 3.0).abs() < 1e-12));
    }
}
