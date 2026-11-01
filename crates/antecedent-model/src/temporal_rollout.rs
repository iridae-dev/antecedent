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

use antecedent_core::execution::{AdmissionError, AdmissionLease};
use antecedent_core::{CausalRng, ExecutionContext, Intervention, InterventionSequence};

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
    /// The admission the path's own bytes hold on the pool it was rolled out under, until the
    /// last clone of the path is dropped.
    _retention: Option<Arc<AdmissionLease>>,
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

/// Worst-case heap of one step's overlay for `sequence`: the per-node vectors, the evaluation
/// order, and one compiled program (a bounded expression, its ops, reads and callbacks) per
/// program-bearing intervention, at most one per node. The window adds no allocation: it only
/// bounds which lags a program may read. `None` on overflow.
fn overlay_bytes(model: &CompiledCausalModel, sequence: &InterventionSequence) -> Option<usize> {
    use std::mem::size_of;
    /// Bytes charged to one compiled program: [`MAX_EXPR_NODES`] expression nodes held three
    /// ways (expression, ops, callbacks) at a conservative `128` bytes each.
    const PROGRAM_BYTES: usize = antecedent_core::value_expr::MAX_EXPR_NODES * 128;
    let per_node = size_of::<Option<f64>>()
        + size_of::<f64>()
        + size_of::<Option<antecedent_core::StochasticPolicy>>()
        + size_of::<Option<antecedent_core::MechanismOverride>>()
        + size_of::<bool>()
        + size_of::<Option<NodeProgram>>();
    let mut programs = 0_usize;
    // A nested sequence may not itself nest, so two levels cover every intervention.
    for step in sequence.steps.iter() {
        match &step.intervention {
            Intervention::Transform { .. } | Intervention::Assign { .. } => programs += 1,
            Intervention::Sequence(inner) => {
                programs += inner
                    .steps
                    .iter()
                    .filter(|s| {
                        matches!(
                            s.intervention,
                            Intervention::Transform { .. } | Intervention::Assign { .. }
                        )
                    })
                    .count();
            }
            _ => {}
        }
    }
    let n_nodes = model.n_nodes();
    model
        .parent_gathers
        .len()
        .checked_mul(size_of::<u32>())?
        .checked_add(n_nodes.checked_mul(per_node)?)?
        .checked_add(programs.min(n_nodes).checked_mul(PROGRAM_BYTES)?)
}

/// Roll a model out over `plan.n_steps` steps under `plan.sequence`.
///
/// # Errors
///
/// Shape errors (a history batch or retained noise that does not match the model), an
/// invalid schedule or program, a program reading deeper than the history that exists at a
/// step (`policy_history_unavailable`), cancellation, a pool that cannot admit the path and
/// its scratch ([`ModelError::ResourceLimit`]), or a mechanism or program failure.
///
/// The returned path keeps its own bytes admitted on `ctx`'s pool until it is dropped.
pub fn rollout_temporal(
    model: &CompiledCausalModel,
    plan: &TemporalPlan<'_>,
    noise: TemporalNoise<'_>,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<TemporalPath, ModelError> {
    rollout_temporal_with_history_availability(model, plan, None, noise, ws, ctx)
}

/// Roll out with per-history-step, per-node availability. `None` means all
/// supplied history cells are available; otherwise entries are most-recent
/// first, with exactly `initial_history.len() * model.n_nodes()` booleans.
/// An unavailable cell may be stored as NaN but is rejected before any active
/// program can read it; history produced by the rollout itself is available.
///
/// # Errors
/// A malformed availability mask or any failure of [`rollout_temporal`],
/// including an active policy that reads an unavailable history cell.
#[doc(hidden)]
#[allow(
    clippy::too_many_lines,
    reason = "the scientific calculation and its checks form one bounded procedure"
)]
pub fn rollout_temporal_with_history_availability(
    model: &CompiledCausalModel,
    plan: &TemporalPlan<'_>,
    history_available: Option<&[bool]>,
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
    if history_available
        .is_some_and(|mask| plan.initial_history.len().checked_mul(n_nodes) != Some(mask.len()))
    {
        return Err(shape_error("initial history availability does not match its batches"));
    }
    if let TemporalNoise::Retained { steps, .. } = &noise {
        if steps.len() != plan.n_steps || steps.iter().any(|s| s.len() != block) {
            return Err(shape_error("retained noise does not match the steps, rows and model"));
        }
    }

    if ctx.cancellation.is_cancelled() {
        return Err(ModelError::Cancelled);
    }
    // Two leases. The path's own bytes (its columns and step headers) are the result: that
    // lease moves into the returned `TemporalPath` and is held until the path is dropped. The
    // history buffer, gather and workspace scratch and one step's overlay are transient and
    // held through this call.
    let path_cells = block
        .checked_mul(plan.n_steps)
        .ok_or_else(|| shape_error("temporal path size overflows"))?;
    let scratch_cells = block
        .checked_mul(2)
        .and_then(|v| v.checked_add(n_rows))
        .ok_or_else(|| shape_error("temporal scratch size overflows"))?;
    let headers = plan
        .n_steps
        .checked_mul(std::mem::size_of::<ValueBatch>() + 2 * std::mem::size_of::<usize>())
        .ok_or_else(|| shape_error("temporal rollout header size overflows"))?;
    let retained_bytes = path_cells
        .checked_mul(std::mem::size_of::<f64>())
        .and_then(|v| v.checked_add(headers))
        .and_then(|v| u64::try_from(v).ok())
        .ok_or_else(|| shape_error("temporal rollout byte size overflows"))?;
    let scratch_bytes = cells
        .checked_add(scratch_cells)
        .and_then(|v| v.checked_mul(std::mem::size_of::<f64>()))
        .and_then(|v| v.checked_add(overlay_bytes(model, plan.sequence)?))
        .and_then(|v| v.checked_add(4096))
        .and_then(|v| u64::try_from(v).ok())
        .ok_or_else(|| shape_error("temporal rollout byte size overflows"))?;
    let admit = |bytes: u64| {
        ctx.acquire_admission(0, bytes).map_err(|error| match error {
            AdmissionError::Cancelled => ModelError::Cancelled,
            _ => ModelError::ResourceLimit {
                message: "temporal rollout buffers",
                required_bytes: bytes,
                limit_bytes: ctx.admission().available().bytes(),
            },
        })
    };
    let retention = Arc::new(admit(retained_bytes)?);
    let _scratch = admit(scratch_bytes)?;

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
        if let Some(mask) = history_available {
            for program in overlay.programs.iter().flatten() {
                for &slot in program.lag_slots() {
                    let lag = slot / n_nodes;
                    let node = slot % n_nodes;
                    if lag > t {
                        let previous = lag - t - 1;
                        if !mask[previous * n_nodes + node] {
                            return Err(refusal_history_not_yet_observed(format!(
                                "a policy active at step {t} reads unavailable history for node {node} at lag {lag}"
                            )).into_error());
                        }
                    }
                }
            }
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
    Ok(TemporalPath { n_rows, n_nodes, steps: path, _retention: Some(retention) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{CompiledMechanismStore, MechanismSlot};
    use antecedent_core::execution::AdmissionPool;
    use antecedent_graph::{Dag, DenseNodeId};

    fn model() -> CompiledCausalModel {
        let mut graph = Dag::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        CompiledCausalModel::compile(graph).unwrap().with_mechanisms(CompiledMechanismStore {
            slots: Arc::from([
                MechanismSlot::LinearGaussian { intercept: 1.0, coeffs: Arc::from([]), sigma: 1.0 },
                MechanismSlot::LinearGaussian {
                    intercept: 0.0,
                    coeffs: Arc::from([0.5]),
                    sigma: 1.0,
                },
            ]),
        })
    }

    fn plan(sequence: &InterventionSequence, n_steps: usize, window: u32) -> TemporalPlan<'_> {
        TemporalPlan { sequence, n_steps, n_rows: 64, window, initial_history: &[] }
    }

    #[test]
    fn the_returned_path_keeps_its_bytes_admitted_until_it_is_dropped() {
        let model = model();
        let sequence = InterventionSequence::new(vec![]);
        let pool = AdmissionPool::fixed(1, 10_000_000);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let mut ws = MechanismWorkspace::default();
        let mut rng = CausalRng::from_seed(2);
        let path = rollout_temporal(
            &model,
            &plan(&sequence, 8, 0),
            TemporalNoise::Draw(&mut rng),
            &mut ws,
            &ctx,
        )
        .unwrap();
        // Scratch is released on return; the path's columns stay charged.
        let held = 10_000_000 - pool.available().bytes();
        let columns = (path.n_rows * path.n_nodes * path.steps.len() * 8) as u64;
        assert!(held >= columns && held < 2 * columns + 4096, "held {held}, columns {columns}");
        let clone = path.clone();
        drop(path);
        assert_eq!(10_000_000 - pool.available().bytes(), held);
        drop(clone);
        assert_eq!(pool.available().bytes(), 10_000_000);
    }

    #[test]
    fn a_pool_short_of_the_declared_window_or_path_is_refused_and_a_cancelled_context_stops() {
        let model = model();
        let sequence = InterventionSequence::new(vec![]);
        let limited = ExecutionContext::for_tests(1).with_memory_limit(Some(50_000));
        let mut ws = MechanismWorkspace::default();
        let mut rng = CausalRng::from_seed(2);
        let mut run = |plan: &TemporalPlan<'_>, ctx: &ExecutionContext| {
            rollout_temporal(&model, plan, TemporalNoise::Draw(&mut rng), &mut ws, ctx)
        };
        assert!(run(&plan(&sequence, 2, 0), &limited).is_ok());
        // The declared window alone overflows the pool.
        assert!(matches!(
            run(&plan(&sequence, 2, 1_000_000), &limited),
            Err(ModelError::ResourceLimit { .. })
        ));
        // So does a path that would be retained.
        assert!(matches!(
            run(&plan(&sequence, 100_000, 0), &limited),
            Err(ModelError::ResourceLimit { .. })
        ));
        // A refusal leaves nothing charged.
        assert_eq!(limited.admission().available().bytes(), 50_000);
        let cancelled = ExecutionContext::for_tests(1);
        cancelled.cancellation.cancel();
        assert!(matches!(run(&plan(&sequence, 2, 0), &cancelled), Err(ModelError::Cancelled)));
    }
}
