//! Observational and interventional batch sampling.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::{
    CausalRng, ExecutionContext, Intervention, MechanismOverride, StochasticPolicy,
};
use antecedent_kernels::standard_normal;

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

/// Sample observational draws into a caller-owned column-major buffer.
///
/// `values` must be at least `n_rows * n_nodes`. The buffer is overwritten,
/// not grown; coalition loops can reuse one allocation across masks.
///
/// # Errors
///
/// Unfitted mechanisms, `n_rows == 0`, or a short buffer.
pub fn sample_observational_into(
    model: &CompiledCausalModel,
    n_rows: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
    values: &mut [f64],
    _ctx: &ExecutionContext,
) -> Result<(), ModelError> {
    let view = ModelView::observational(model);
    sample_with_overlay_into(&view, n_rows, rng, ws, values)
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
/// Mechanism failures, or an overlay with a node both hard-set and shifted
/// (see [`InterventionOverlay::validate`]).
pub fn sample_with_overlay(
    view: &ModelView<'_>,
    n_rows: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
) -> Result<ValueBatch, ModelError> {
    let n_nodes = view.model.n_nodes();
    let mut values_buf = vec![0.0; n_rows.saturating_mul(n_nodes)];
    sample_with_overlay_into(view, n_rows, rng, ws, &mut values_buf)?;
    Ok(ValueBatch { n_rows, n_nodes, values: std::sync::Arc::from(values_buf) })
}

/// Ancestral sample into a caller-owned column-major buffer.
///
/// # Errors
///
/// Shape, overlay, or mechanism failures.
pub fn sample_with_overlay_into(
    view: &ModelView<'_>,
    n_rows: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
    values: &mut [f64],
) -> Result<(), ModelError> {
    if n_rows == 0 {
        return Err(ModelError::Shape { message: "n_rows must be > 0".into() });
    }
    // Overlays reaching here need not have come from `from_interventions` — this is a
    // public entry point taking a caller-built `ModelView`, so the invariant the hard-set
    // branch below relies on is re-established rather than assumed.
    view.overlay.validate()?;
    let model = view.model;
    let n_nodes = model.n_nodes();
    let mut values = ValueBatchMut::new(n_rows, n_nodes, values)?;
    let overlay = view.overlay.as_ref();

    // Gather target hoisted out of the node loop (grow-only) so the parent
    // batch borrows a buffer disjoint from `ws` — `sample_column` needs
    // `&mut ws` while parents stay alive, which previously forced a fresh
    // `to_vec` per node.
    let mut parent_buf: Vec<f64> = Vec::new();
    for gather in model.parent_gathers.iter() {
        let node = gather.child;
        let idx = node.as_usize();
        let need = gather.n_parents().max(1).saturating_mul(n_rows);
        if parent_buf.len() < need {
            parent_buf.resize(need, 0.0);
        }
        gather.gather(values.values, n_rows, &mut parent_buf);
        let parents = ParentBatch {
            n_rows,
            n_parents: gather.n_parents(),
            values: &parent_buf[..gather.n_parents().saturating_mul(n_rows)],
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
            let existing = model.mechanisms.get(node);
            refuse_cross_family_soft(existing, soft)?;
            let slot = soft_to_slot(soft, gather.n_parents())?;
            sample_column(&slot, parents, rng, out, ws)?;
            apply_shift(out, overlay.shifts[idx]);
            continue;
        }

        let slot = model.mechanisms.get(node);
        sample_column(slot, parents, rng, out, ws)?;
        apply_shift(out, overlay.shifts[idx]);
    }

    Ok(())
}

/// Soft overrides must share noise semantics with the fitted mechanism. Reusing a
/// Discrete Uniform(0,1) residual as an additive Gaussian U (or the reverse) is not
/// a well-defined counterfactual.
///
/// # Errors
///
/// [`ModelError::Unsupported`] when the fitted slot and override disagree on noise kind.
pub fn refuse_cross_family_soft(
    existing: &MechanismSlot,
    soft: &MechanismOverride,
) -> Result<(), ModelError> {
    let have = noise_kind_slot(existing);
    let want = noise_kind_override(soft.family_id.as_ref());
    if have == "any" || have == want {
        return Ok(());
    }
    Err(ModelError::Unsupported {
        message: format!(
            "cross-family soft override refused: fitted `{have}` vs override `{}` (`{want}`)",
            soft.family_id
        ),
    })
}

fn noise_kind_slot(slot: &MechanismSlot) -> &'static str {
    match slot {
        MechanismSlot::Vacant | MechanismSlot::Pending { .. } | MechanismSlot::Dynamic { .. } => {
            "any"
        }
        MechanismSlot::LinearGaussian { .. }
        | MechanismSlot::HierarchicalLinear { .. }
        | MechanismSlot::LinearBasis { .. }
        | MechanismSlot::Bvar { .. } => "additive_gaussian",
        MechanismSlot::Discrete { .. } | MechanismSlot::DiscreteBasis { .. } => "discrete",
        MechanismSlot::Constant { .. } => "constant",
        MechanismSlot::LinearGaussianStateSpace { .. }
        | MechanismSlot::ConditionalLinearGaussianStateSpace { .. } => "lgssm",
        MechanismSlot::GaussianProcess { .. } => "gaussian_process",
    }
}

fn noise_kind_override(family_id: &str) -> &'static str {
    match family_id {
        "linear_gaussian" | "hierarchical_linear" | "bvar" | "additive_shift" => {
            "additive_gaussian"
        }
        "discrete" => "discrete",
        "constant" => "constant",
        "lgssm" | "conditional_lgssm" => "lgssm",
        "gaussian_process" => "gaussian_process",
        _ => "unknown",
    }
}

/// Sample under interventions conditioned on observed node values.
///
/// Strategy:
/// 1. **Rejection sampling** when conditions match within `1e-9` (exact / discrete).
/// 2. **Forward likelihood-weighting SIR** when rejection under-accepts: walk the mutilated
///    graph in topological order, clamp each conditioned node to its evidence value while
///    accumulating `∏_c p(condition_c | parents_c)`, sample every other node (including
///    descendants of evidence) from its mechanism given already-clamped parents, then
///    resample. A propose-then-clamp path would leave descendants drawn under the
///    unconditioned proposal parents — those draws are not conditional.
///
/// Conditioning nodes must not be hard-intervened.
///
/// # Errors
///
/// Empty condition, intervened condition nodes, density failures, or empty weights.
#[allow(clippy::too_many_lines)]
pub fn sample_conditional_interventional(
    model: &CompiledCausalModel,
    interventions: &[Intervention],
    condition_nodes: &[antecedent_graph::DenseNodeId],
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
        // A soft/stochastic override on a condition node would make the importance weight
        // below (`sample_conditional_interventional_lw`) inconsistent: the proposal draws
        // for that node come from `overlay.soft`/`overlay.stochastic`, but the weight is
        // computed against the model's *original* mechanism
        // (`log_prob_column(model.mechanisms.get(node), ...)`), which never consults the
        // overlay. That mismatch would silently bias the conditional estimate instead of
        // erroring, so reject it here — matching the hard-set posture above — rather than
        // letting it through.
        if overlay.soft[idx].is_some() || overlay.stochastic[idx].is_some() {
            return Err(ModelError::Unsupported {
                message: "cannot condition on a soft- or stochastic-intervened node".into(),
            });
        }
    }

    let n_nodes = model.n_nodes();
    // Exact-match rejection only makes sense for evidence with positive probability, i.e.
    // conditioning nodes whose mechanism is discrete (or a point mass). For a continuous
    // condition the acceptance probability is zero, so drawing candidates would burn the
    // whole attempt budget for nothing: go straight to likelihood weighting.
    let discrete_evidence = condition_nodes.iter().all(|&node| {
        matches!(
            model.mechanisms.get(node),
            MechanismSlot::Discrete { .. }
                | MechanismSlot::DiscreteBasis { .. }
                | MechanismSlot::Constant { .. }
        )
    });
    if discrete_evidence {
        let mut accepted = vec![0.0; n_rows * n_nodes];
        let mut got = 0usize;
        let max_attempts = n_rows.saturating_mul(100).max(100);
        // Overlay built once; candidates are drawn in batches (one workspace preparation
        // and one allocation per batch, not per candidate row).
        let overlay = InterventionOverlay::from_interventions(model, interventions)?;
        let view = ModelView::with_overlay(model, overlay);
        let batch_rows = n_rows.clamp(256, 65_536);
        let mut drawn = 0usize;
        while got < n_rows && drawn < max_attempts {
            let rows = batch_rows.min(max_attempts - drawn);
            let batch = sample_with_overlay(&view, rows, rng, ws)?;
            drawn += rows;
            let cond_cols: Vec<&[f64]> = condition_nodes
                .iter()
                .map(|node| batch.column(node.as_usize()))
                .collect::<Result<_, _>>()?;
            for r in 0..rows {
                if got >= n_rows {
                    break;
                }
                let matches_evidence = cond_cols
                    .iter()
                    .zip(condition_values)
                    .all(|(col, &target)| (col[r] - target).abs() <= 1e-9);
                if !matches_evidence {
                    continue;
                }
                for node in 0..n_nodes {
                    accepted[node * n_rows + got] = batch.column(node)?[r];
                }
                got += 1;
            }
        }
        if got >= n_rows {
            let _ = ctx;
            return Ok(ValueBatch { n_rows, n_nodes, values: accepted.into() });
        }
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

#[allow(clippy::too_many_lines)] // one linear derivation; splitting it would scatter the argument
fn sample_conditional_interventional_lw(
    model: &CompiledCausalModel,
    interventions: &[Intervention],
    condition_nodes: &[antecedent_graph::DenseNodeId],
    condition_values: &[f64],
    n_rows: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<ValueBatch, ModelError> {
    use crate::mechanism::log_prob_column;

    let n_nodes = model.n_nodes();
    let n_particles = n_rows.saturating_mul(20).max(64);
    let overlay = InterventionOverlay::from_interventions(model, interventions)?;
    overlay.validate()?;

    let mut is_condition = vec![false; n_nodes];
    let mut condition_at = vec![0.0; n_nodes];
    for (ci, &node) in condition_nodes.iter().enumerate() {
        let idx = node.as_usize();
        is_condition[idx] = true;
        condition_at[idx] = condition_values[ci];
    }

    // Forward likelihood weighting: clamp evidence in topo order and sample every
    // other node (incl. descendants of evidence) from mechanisms given those clamps.
    // Propose-from-do then overwrite evidence leaves descendants drawn under the
    // unconditioned proposal parents — not a conditional draw.
    let mut particle_buf = vec![0.0; n_particles.saturating_mul(n_nodes)];
    let mut log_w = vec![0.0; n_particles];
    let mut lp_buf = vec![0.0; n_particles];
    let mut parent_buf: Vec<f64> = Vec::new();

    {
        let mut values = ValueBatchMut::new(n_particles, n_nodes, &mut particle_buf)?;
        for gather in model.parent_gathers.iter() {
            let node = gather.child;
            let idx = node.as_usize();
            let need = gather.n_parents().max(1).saturating_mul(n_particles);
            if parent_buf.len() < need {
                parent_buf.resize(need, 0.0);
            }
            gather.gather(values.values, n_particles, &mut parent_buf);
            let parents = ParentBatch {
                n_rows: n_particles,
                n_parents: gather.n_parents(),
                values: &parent_buf[..gather.n_parents().saturating_mul(n_particles)],
            };

            if is_condition[idx] {
                let c = condition_at[idx];
                values.column_mut(idx)?.fill(c);
                let conditioned = vec![c; n_particles];
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
                continue;
            }

            let out = values.column_mut(idx)?;
            if let Some(v) = overlay.hard_set[idx] {
                out.fill(v);
                continue;
            }
            if let Some(policy) = &overlay.stochastic[idx] {
                sample_stochastic(policy, n_particles, rng, out)?;
                apply_shift(out, overlay.shifts[idx]);
                continue;
            }
            if let Some(soft) = &overlay.soft[idx] {
                let existing = model.mechanisms.get(node);
                refuse_cross_family_soft(existing, soft)?;
                let slot = soft_to_slot(soft, gather.n_parents())?;
                sample_column(&slot, parents, rng, out, ws)?;
                apply_shift(out, overlay.shifts[idx]);
                continue;
            }

            let slot = model.mechanisms.get(node);
            sample_column(slot, parents, rng, out, ws)?;
            apply_shift(out, overlay.shifts[idx]);
        }
    }

    let weights = normalized_weights(&log_w)?;

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
            accepted[node * n_rows + i] = particle_buf[node * n_particles + idx];
        }
        // Evidence nodes stay at the conditioned values (already clamped above).
        for (ci, &node) in condition_nodes.iter().enumerate() {
            accepted[node.as_usize() * n_rows + i] = condition_values[ci];
        }
    }
    let _ = ctx;
    Ok(ValueBatch { n_rows, n_nodes, values: accepted.into() })
}

/// Self-normalised importance weights from log-weights, refusing an all-non-finite or
/// zero-mass weight vector (no particle can then represent the conditional law).
fn normalized_weights(log_w: &[f64]) -> Result<Vec<f64>, ModelError> {
    let n_particles = log_w.len();
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
    Ok(weights)
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
    let family = soft.family_id.as_ref();
    if let Some(bad) = soft.parameters.iter().position(|p| !p.is_finite()) {
        return Err(ModelError::Numerical {
            message: format!(
                "soft override `{family}` parameter {bad} is not finite ({})",
                soft.parameters[bad]
            ),
        });
    }
    match family {
        "constant" => {
            match &soft.parameters[..] {
                [v] => Ok(MechanismSlot::Constant { value: *v }),
                _ => Err(ModelError::Shape {
                    message: "constant override needs exactly one value".into(),
                }),
            }
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
            let sigma = positive(soft.parameters[1 + n_parents], "linear_gaussian sigma")?;
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
            let sigma = positive(soft.parameters[1 + n_parents], "hierarchical_linear sigma")?;
            let shrinkage = soft.parameters[2 + n_parents];
            if shrinkage < 0.0 {
                return Err(ModelError::Numerical {
                    message: format!("hierarchical_linear shrinkage must be >= 0, got {shrinkage}"),
                });
            }
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
            let sigma = positive(soft.parameters[1 + n_parents], "bvar sigma")?;
            Ok(MechanismSlot::Bvar { intercept, coeffs, sigma })
        }
        "discrete" => soft_discrete_slot(soft, n_parents),
        "conditional_lgssm" => {
            if soft.parameters.len() != 5 + n_parents {
                return Err(ModelError::Shape { message: "conditional_lgssm override needs intercept, coeffs..., a, process_std, obs_std, initial_mean".into() });
            }
            let k = 1 + n_parents;
            Ok(MechanismSlot::ConditionalLinearGaussianStateSpace {
                intercept: soft.parameters[0],
                coeffs: std::sync::Arc::from(soft.parameters[1..k].to_vec()),
                a: soft.parameters[k],
                process_std: positive(soft.parameters[k + 1], "conditional_lgssm process_std")?,
                obs_std: positive(soft.parameters[k + 2], "conditional_lgssm obs_std")?,
                initial_mean: soft.parameters[k + 3],
            })
        }
        "lgssm" => {
            if soft.parameters.len() < 4 {
                return Err(ModelError::Shape {
                    message: "lgssm override needs a, process_std, obs_std, initial_mean".into(),
                });
            }
            Ok(MechanismSlot::LinearGaussianStateSpace {
                a: soft.parameters[0],
                process_std: positive(soft.parameters[1], "lgssm process_std")?,
                obs_std: positive(soft.parameters[2], "lgssm obs_std")?,
                initial_mean: soft.parameters[3],
            })
        }
        "gaussian_process" => soft_gp_slot(soft, n_parents),
        other => Err(ModelError::Unsupported {
            message: format!("unknown soft override family {other}"),
        }),
    }
}

/// A strictly positive scale parameter. A non-positive one is refused, not clamped to a
/// tiny value: `sigma = -1` silently becoming a near-deterministic mechanism reports a
/// counterfactual the caller never asked for.
fn positive(value: f64, what: &str) -> Result<f64, ModelError> {
    if value > 0.0 {
        Ok(value)
    } else {
        Err(ModelError::Numerical {
            message: format!("soft override {what} must be finite and > 0, got {value}"),
        })
    }
}

/// A non-negative integer-valued parameter (a count or a size).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the guard admits only non-negative integral values up to u32::MAX, which fit usize on every supported target"
)]
fn count_param(value: f64, what: &str) -> Result<usize, ModelError> {
    if value >= 0.0 && value.fract() == 0.0 && value <= f64::from(u32::MAX) {
        Ok(value as usize)
    } else {
        Err(ModelError::Shape {
            message: format!("soft override {what} must be a non-negative integer, got {value}"),
        })
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
    let k = count_param(soft.parameters[0], "discrete k")?;
    if k == 0 {
        return Err(ModelError::Shape { message: "discrete override k must be > 0".into() });
    }
    if soft.parameters.len() < 1 + k {
        return Err(ModelError::Shape { message: "discrete override truncated support".into() });
    }
    let support: std::sync::Arc<[f64]> = std::sync::Arc::from(soft.parameters[1..=k].to_vec());
    let rest = &soft.parameters[1 + k..];
    if rest.len() == k {
        if rest.iter().any(|p| *p < 0.0) || rest.iter().sum::<f64>() <= 0.0 {
            return Err(ModelError::Numerical {
                message: "discrete override probabilities must be non-negative with a positive sum"
                    .into(),
            });
        }
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
    let length_scale = positive(soft.parameters[0], "gaussian_process length_scale")?;
    let variance = soft.parameters[1];
    if variance < 0.0 {
        return Err(ModelError::Numerical {
            message: format!("gaussian_process variance must be >= 0, got {variance}"),
        });
    }
    let noise_std = positive(soft.parameters[2], "gaussian_process noise_std")?;
    let n_train = count_param(soft.parameters[3], "gaussian_process n_train")?;
    let n_par = count_param(soft.parameters[4], "gaussian_process n_parents")?;
    if n_par != n_parents {
        return Err(ModelError::Shape {
            message: format!("gaussian_process override n_parents {n_par} != gather {n_parents}"),
        });
    }
    // Sizes come from user parameters: checked arithmetic, never a wrapped index.
    let x_len = n_train.checked_mul(n_par);
    let need = x_len.and_then(|x| x.checked_add(n_train)).and_then(|s| s.checked_add(5));
    let (Some(x_len), Some(need)) = (x_len, need) else {
        return Err(ModelError::Shape {
            message: "gaussian_process override size overflows".into(),
        });
    };
    if soft.parameters.len() < need {
        return Err(ModelError::Shape {
            message: format!(
                "gaussian_process override needs {need} params, got {}",
                soft.parameters.len()
            ),
        });
    }
    let x_train = std::sync::Arc::from(soft.parameters[5..5 + x_len].to_vec());
    let alpha = std::sync::Arc::from(soft.parameters[5 + x_len..need].to_vec());
    Ok(MechanismSlot::GaussianProcess {
        length_scale,
        variance,
        noise_std,
        // The packed layout carries no prior mean: an override is a zero-mean GP.
        mean: 0.0,
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
/// Mechanism failures, or an overlay with a node both hard-set and shifted
/// (see [`InterventionOverlay::validate`]).
pub fn sample_structural_with_overlay(
    view: &ModelView<'_>,
    n_rows: usize,
    rng: &mut CausalRng,
    ws: &mut MechanismWorkspace,
) -> Result<(ValueBatch, Vec<f64>), ModelError> {
    view.overlay.validate()?;
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
    let mut parent_buf: Vec<f64> = Vec::new();
    for gather in model.parent_gathers.iter() {
        let node = gather.child;
        let idx = node.as_usize();
        let need = gather.n_parents().max(1).saturating_mul(n_rows);
        if parent_buf.len() < need {
            parent_buf.resize(need, 0.0);
        }
        gather.gather(values.values, n_rows, &mut parent_buf);
        let parents = ParentBatch {
            n_rows,
            n_parents: gather.n_parents(),
            values: &parent_buf[..gather.n_parents().saturating_mul(n_rows)],
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
            let existing = model.mechanisms.get(node);
            refuse_cross_family_soft(existing, soft)?;
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
    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, Intervention, MeasurementSpec, RoleHint,
        SmallRoleSet, Value, ValueType, VariableId,
    };
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
    use antecedent_graph::{Dag, DenseNodeId};
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

    /// Conditioning on a node that is *also* named with a `Stochastic` (or `Soft`) override
    /// must be refused, matching the existing hard-set posture.
    ///
    /// Without this check, `sample_conditional_interventional_lw` would draw the condition
    /// node's proposal values from the overlay's stochastic policy, but weight them under
    /// `model.mechanisms.get(node)` -- the model's *original*, un-overridden mechanism. That
    /// mismatch between what generated the draws and what scores them is an internally
    /// inconsistent importance weight, silently biasing the conditional estimate instead of
    /// erroring.
    #[test]
    fn conditioning_on_stochastic_intervened_node_is_refused() {
        let model = fitted_chain();
        let mut rng = CausalRng::from_seed(1);
        let mut ws = MechanismWorkspace::default();
        let y = VariableId::from_raw(1);
        let y_node = DenseNodeId::from_raw(1);
        let err = sample_conditional_interventional(
            &model,
            &[Intervention::Stochastic {
                variable: y,
                policy: StochasticPolicy::Gaussian { mean: 0.0, variance: 1.0 },
            }],
            &[y_node],
            &[0.5],
            10,
            &mut rng,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .unwrap_err();
        assert!(matches!(err, ModelError::Unsupported { .. }), "expected Unsupported, got {err:?}");
    }

    /// Condition on the middle of X→Y→Z and check the *unconditioned* child Z.
    /// Propose-then-clamp overwrites Y but leaves Z drawn under proposal parents;
    /// forward LW must redraw Z given Y fixed at the evidence, so the sample mean
    /// of Z sits on the linear-Gaussian conditional mean, not on E[Z | do(X)].
    #[test]
    fn conditional_do_lw_redraws_descendant_under_clamped_evidence() {
        let model = fitted_three_chain();
        let z_slot = model.mechanisms.get(DenseNodeId::from_raw(2));
        let MechanismSlot::LinearGaussian { intercept, coeffs, sigma } = z_slot else {
            panic!("expected LinearGaussian Z mechanism, got {z_slot:?}");
        };
        assert_eq!(coeffs.len(), 1, "Z should have a single parent Y");
        let y_cond = 0.0_f64;
        // Under do(X=1) the chain mean for Y is near 2; conditioning far from that
        // separates E[Z | Y=y_cond] from the propose-then-clamp mean E[Z | do(X)].
        let conditional_mean = intercept + coeffs[0] * y_cond;
        let interventional_mean = {
            let y_slot = model.mechanisms.get(DenseNodeId::from_raw(1));
            let MechanismSlot::LinearGaussian { intercept: y_int, coeffs: y_coeffs, .. } = y_slot
            else {
                panic!("expected LinearGaussian Y mechanism, got {y_slot:?}");
            };
            let e_y_do = y_int + y_coeffs[0] * 1.0;
            intercept + coeffs[0] * e_y_do
        };
        assert!(
            (conditional_mean - interventional_mean).abs() > 0.5,
            "test needs separated targets: cond={conditional_mean} do={interventional_mean}"
        );

        let mut rng = CausalRng::from_seed(11);
        let mut ws = MechanismWorkspace::default();
        let x = VariableId::from_raw(0);
        let y_node = DenseNodeId::from_raw(1);
        let n_rows = 2_048usize;
        let batch = sample_conditional_interventional(
            &model,
            &[Intervention::set(x, Value::f64(1.0))],
            &[y_node],
            &[y_cond],
            n_rows,
            &mut rng,
            &mut ws,
            &ExecutionContext::for_tests(1),
        )
        .expect("conditional do should succeed");
        let y = batch.column(1).unwrap();
        assert!(y.iter().all(|&v| (v - y_cond).abs() < 1e-12), "evidence column Y must be clamped");
        let z = batch.column(2).unwrap();
        let z_mean = z.iter().sum::<f64>() / n_rows as f64;
        // Monte Carlo SE ≈ sigma / sqrt(n); allow a few SEs plus fitting slack.
        let tol = 4.0 * sigma / (n_rows as f64).sqrt() + 0.05;
        assert!(
            (z_mean - conditional_mean).abs() < tol,
            "Z mean {z_mean} should sit on E[Z|Y={y_cond}]={conditional_mean} (tol {tol}); \
             propose-then-clamp would land near E[Z|do(X)]={interventional_mean}"
        );
        assert!(
            (z_mean - interventional_mean).abs() > (z_mean - conditional_mean).abs() + 0.25,
            "Z mean {z_mean} is closer to the do-proposal mean {interventional_mean} than to \
             the clamped conditional mean {conditional_mean}"
        );
    }

    fn fitted_three_chain() -> CompiledCausalModel {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in
            [("x", RoleHint::Context), ("y", RoleHint::Context), ("z", RoleHint::OutcomeCandidate)]
        {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        // Small noise so LinearGaussian fits with usable residual density for LW.
        let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let yv: Vec<f64> = xv
            .iter()
            .enumerate()
            .map(|(i, x)| 0.5 + 1.5 * x + 0.15 * ((i % 7) as f64 - 3.0) / 3.0)
            .collect();
        let zv: Vec<f64> = yv
            .iter()
            .enumerate()
            .map(|(i, y)| -0.25 + 0.8 * y + 0.12 * ((i % 5) as f64 - 2.0) / 2.0)
            .collect();
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(zv), validity).unwrap(),
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
    fn observational_into_matches_allocating_sample() {
        let model = fitted_chain();
        let ctx = ExecutionContext::for_tests(1);
        let n_rows = 16usize;
        let n_nodes = model.n_nodes();
        let mut rng_a = CausalRng::from_seed(7);
        let mut rng_b = CausalRng::from_seed(7);
        let mut ws_a = MechanismWorkspace::default();
        let mut ws_b = MechanismWorkspace::default();
        let batch = sample_observational(&model, n_rows, &mut rng_a, &mut ws_a, &ctx).unwrap();
        let mut buf = vec![0.0; n_rows * n_nodes];
        sample_observational_into(&model, n_rows, &mut rng_b, &mut ws_b, &mut buf, &ctx).unwrap();
        assert_eq!(&*batch.values, buf.as_slice());
        sample_observational_into(&model, n_rows, &mut rng_b, &mut ws_b, &mut buf, &ctx).unwrap();
        assert_eq!(buf.len(), n_rows * n_nodes);
    }
}
