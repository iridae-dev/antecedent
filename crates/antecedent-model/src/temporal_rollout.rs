//! Temporal rollout of a compiled model under a scheduled intervention sequence.
//!
//! The model is one time slice. A rollout evaluates it once per step, in the order the
//! step's overlay fixes, and carries the realised values of earlier steps as a history
//! window so that a value program may read a variable `k` steps back
//! ([`ValueExpr::Lag`](antecedent_core::ValueExpr::Lag)). Each step runs the same
//! per-node precedence ([`evaluate_node`]) and the same program executor
//! ([`apply_node_program`](crate::apply_node_program)) as static sampling and
//! counterfactual world evaluation; the only temporal part is how the value batch is laid
//! out: the current step's columns first, then one block of columns per step back
//! ([`history_slot`](crate::program::history_slot)), refilled from the realised steps
//! before every step.
//!
//! Noise comes from the caller's stream ([`TemporalNoise::Draw`]) or from retained
//! per-step columns ([`TemporalNoise::Retained`], abduced noise of a counterfactual), so
//! the same policy gives the same values on shared noise on every route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, InterventionSequence};

use crate::batch::{MechanismWorkspace, ParentBatch, ValueBatch, ValueBatchMut};
use crate::compile::CompiledCausalModel;
use crate::error::ModelError;
use crate::overlay::InterventionOverlay;
use crate::program::{NodeProgram, refusal_history_not_yet_observed};
use crate::sample::{NodeEvaluation, NodeNoise, evaluate_node};

/// What a rollout runs: the schedule, its length, the population and the history it keeps.
#[derive(Clone, Copy, Debug)]
pub struct TemporalPlan<'a> {
    /// Interventions with their temporal policies; step `t` applies the ones active at `t`.
    pub sequence: &'a InterventionSequence,
    /// Number of steps to roll out (times `0..n_steps`).
    pub n_steps: usize,
    /// Rows (units) evolved in parallel.
    pub n_rows: usize,
    /// Steps of history a value program may read; zero allows none.
    pub window: u32,
    /// Realised steps before time zero, most recent first (`initial_history[0]` is step
    /// `-1`). A program reading deeper than the steps that exist is refused.
    pub initial_history: &'a [ValueBatch],
}

/// Where the rollout's exogenous noise comes from.
pub enum TemporalNoise<'a> {
    /// Draw inline from one sequential stream.
    Draw(&'a mut CausalRng),
    /// Replay retained noise: `steps[t][node * n_rows + row]`, one entry per step.
    /// Stochastic policies, which have no retained noise, draw from `policy_rng`.
    Retained {
        /// Retained noise per step.
        steps: &'a [Arc<[f64]>],
        /// Stream driving a stochastic-policy node.
        policy_rng: &'a mut CausalRng,
    },
}

/// Realised values of every step.
#[derive(Clone, Debug)]
pub struct TemporalPath {
    /// Rows (units).
    pub n_rows: usize,
    /// Model nodes.
    pub n_nodes: usize,
    /// One batch per step, in time order.
    pub steps: Vec<ValueBatch>,
}

impl TemporalPath {
    /// Column of `node` at `step`.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn column(&self, step: usize, node: usize) -> Result<&[f64], ModelError> {
        self.steps
            .get(step)
            .ok_or_else(|| ModelError::Shape { message: "temporal step out of range".into() })?
            .column(node)
    }
}

/// Roll a model out over `plan.n_steps` steps under `plan.sequence`.
///
/// # Errors
///
/// Shape errors (a history batch or retained noise that does not match the model), an
/// invalid schedule or program, a program reading deeper than the history that exists at a
/// step (`policy_history_unavailable`), cancellation, or a mechanism or program failure.
pub fn rollout_temporal(
    model: &CompiledCausalModel,
    plan: &TemporalPlan<'_>,
    mut noise: TemporalNoise<'_>,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<TemporalPath, ModelError> {
    let n_nodes = model.n_nodes();
    let n_rows = plan.n_rows;
    let shape_error = |message: &str| ModelError::Shape { message: message.into() };
    if n_rows == 0 || plan.n_steps == 0 {
        return Err(shape_error("a temporal rollout needs at least one row and one step"));
    }
    let block = n_rows
        .checked_mul(n_nodes)
        .ok_or_else(|| shape_error("temporal rollout size overflows"))?;
    let width = n_nodes
        .checked_mul(
            (plan.window as usize)
                .checked_add(1)
                .ok_or_else(|| shape_error("temporal window overflows"))?,
        )
        .ok_or_else(|| shape_error("temporal rollout width overflows"))?;
    let cells =
        n_rows.checked_mul(width).ok_or_else(|| shape_error("temporal rollout size overflows"))?;
    if block.checked_mul(plan.n_steps).is_none() {
        return Err(shape_error("temporal rollout size overflows"));
    }
    if plan
        .initial_history
        .iter()
        .any(|b| b.n_rows != n_rows || b.n_nodes != n_nodes || b.values.len() != block)
    {
        return Err(shape_error("an initial history batch does not match the model and rows"));
    }
    if let TemporalNoise::Retained { steps, .. } = &noise {
        if steps.len() != plan.n_steps || steps.iter().any(|s| s.len() != block) {
            return Err(shape_error("retained noise does not match the steps, rows and model"));
        }
    }

    if ctx.cancellation.is_cancelled() {
        return Err(ModelError::Cancelled);
    }
    // The lease covers the history buffer, retained path, gather and workspace
    // scratch, and output/Arc headers through this call. Returned path lifetime
    // remains governed by the caller's retention policy.
    let path_cells = block
        .checked_mul(plan.n_steps)
        .ok_or_else(|| shape_error("temporal path size overflows"))?;
    let scratch_cells = block
        .checked_mul(2)
        .and_then(|v| v.checked_add(n_rows))
        .ok_or_else(|| shape_error("temporal scratch size overflows"))?;
    let payload = cells
        .checked_add(path_cells)
        .and_then(|v| v.checked_add(scratch_cells))
        .and_then(|v| v.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(|| shape_error("temporal rollout byte size overflows"))?;
    let headers = plan
        .n_steps
        .checked_mul(std::mem::size_of::<ValueBatch>() + 2 * std::mem::size_of::<usize>())
        .ok_or_else(|| shape_error("temporal rollout header size overflows"))?;
    let bytes = payload
        .checked_add(headers)
        .and_then(|v| v.checked_add(4096))
        .and_then(|v| u64::try_from(v).ok())
        .ok_or_else(|| shape_error("temporal rollout byte size overflows"))?;
    let _lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => ModelError::Cancelled,
        _ => ModelError::ResourceLimit {
            message: "temporal rollout buffers",
            required_bytes: bytes,
            limit_bytes: ctx.admission().available().bytes(),
        },
    })?;

    let mut path: Vec<ValueBatch> = Vec::with_capacity(plan.n_steps);
    let mut buf = vec![0.0; cells];
    let mut parent_buf: Vec<f64> = Vec::new();
    for t in 0..plan.n_steps {
        if ctx.cancellation.is_cancelled() {
            return Err(ModelError::Cancelled);
        }
        let time = i32::try_from(t).map_err(|_| shape_error("temporal step exceeds i32"))?;
        let overlay = InterventionOverlay::from_sequence_with_history(
            model,
            plan.sequence,
            time,
            plan.window,
        )?;
        let available = plan
            .initial_history
            .len()
            .checked_add(t)
            .ok_or_else(|| shape_error("temporal history length overflows"))?;
        let needed = overlay.programs.iter().flatten().map(NodeProgram::max_lag).max();
        if let Some(needed) = needed.filter(|needed| *needed as usize > available) {
            return Err(refusal_history_not_yet_observed(format!(
                "a policy active at step {t} reads {needed} steps back but only {available} \
                 earlier steps exist"
            ))
            .into_error());
        }
        buf.fill(0.0);
        for lag in 1..=available.min(plan.window as usize) {
            let source: &[f64] = if t >= lag {
                &path[t - lag].values
            } else {
                &plan.initial_history[lag - t - 1].values
            };
            let start = block * lag;
            buf[start..start + block].copy_from_slice(source);
        }
        {
            let mut values = ValueBatchMut::new(n_rows, width, &mut buf)?;
            let eval = NodeEvaluation { model, overlay: &overlay, ctx: Some(ctx) };
            for gather in overlay.gathers(model) {
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
                let node_noise = match &mut noise {
                    TemporalNoise::Draw(rng) => NodeNoise::Draw(rng),
                    TemporalNoise::Retained { steps, policy_rng } => {
                        let idx = gather.child.as_usize();
                        NodeNoise::Column {
                            noise: &steps[t][idx * n_rows..(idx + 1) * n_rows],
                            policy_rng,
                        }
                    }
                };
                evaluate_node(eval, gather, parents, node_noise, &mut values, ws)?;
            }
        }
        path.push(ValueBatch { n_rows, n_nodes, values: Arc::from(&buf[..block]) });
    }
    Ok(TemporalPath { n_rows, n_nodes, steps: path })
}
