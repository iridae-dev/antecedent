//! Observational and interventional batch sampling (DESIGN.md §15.4).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
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
        let parents = ParentBatch {
            n_rows,
            n_parents: gather.n_parents(),
            values: &parent_owned,
        };

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

/// Posterior-predictive interventional sampling: for each coefficient draw block,
/// refresh LinearGaussian slots then sample. `draw_updater` mutates slots in place.
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

fn soft_to_slot(soft: &MechanismOverride, n_parents: usize) -> Result<MechanismSlot, ModelError> {
    match soft.family_id.as_ref() {
        "constant" => {
            let v = soft.parameters.first().copied().unwrap_or(0.0);
            Ok(MechanismSlot::Constant { value: v })
        }
        "additive_shift" => {
            // Represented as constant 0 evaluated then shifted by overlay — here use
            // intercept-only linear with zero noise for structural base.
            Ok(MechanismSlot::LinearGaussian {
                intercept: soft.parameters.first().copied().unwrap_or(0.0),
                coeffs: std::sync::Arc::from(vec![0.0; n_parents]),
                sigma: 1e-12,
            })
        }
        "linear_gaussian" => {
            if soft.parameters.len() < 2 + n_parents {
                return Err(ModelError::Shape {
                    message: "linear_gaussian override needs intercept, coeffs..., sigma".into(),
                });
            }
            let intercept = soft.parameters[0];
            let coeffs = std::sync::Arc::from(soft.parameters[1..1 + n_parents].to_vec());
            let sigma = soft.parameters[1 + n_parents].max(1e-12);
            Ok(MechanismSlot::LinearGaussian { intercept, coeffs, sigma })
        }
        other => Err(ModelError::Unsupported {
            message: format!("unknown soft override family {other}"),
        }),
    }
}

fn sample_stochastic(
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
        let parents = ParentBatch {
            n_rows,
            n_parents: gather.n_parents(),
            values: &parent_owned,
        };
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
        let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
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
