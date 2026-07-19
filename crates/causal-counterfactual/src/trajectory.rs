//! Counterfactual trajectories with shared-noise batching (DESIGN.md §16.1).
//!
//! Layout is flat columnar — never `Vec<Vec<Vec<_>>>`.
//! `values[time * n_worlds * n_units + world * n_units + unit]`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use std::sync::Arc;

use causal_core::{ExecutionContext, Intervention, VariableId};
use causal_graph::DenseNodeId;
use causal_model::MechanismWorkspace;

use crate::engine::{CounterfactualEngine, CounterfactualWorld, ExogenousPosterior};
use crate::error::CounterfactualError;

/// One trajectory arm: an ordered schedule of intervention sets (time axis).
#[derive(Clone, Debug)]
pub struct TrajectoryArm {
    /// Interventions applied at each time index (length = horizon).
    pub schedule: Arc<[Arc<[Intervention]>]>,
}

/// Request for shared-noise counterfactual trajectories.
#[derive(Clone, Debug)]
pub struct CounterfactualTrajectoryRequest {
    /// Trajectory arms (worlds).
    pub arms: Arc<[TrajectoryArm]>,
    /// Outcome variable.
    pub outcome: VariableId,
}

/// Columnar trajectory outcomes + streaming summaries.
#[derive(Clone, Debug)]
pub struct TrajectoryResult {
    /// Flat values: `time * n_worlds * n_units + world * n_units + unit`.
    pub values: Arc<[f64]>,
    /// Horizon (time steps).
    pub horizon: usize,
    /// Number of arms / worlds.
    pub n_worlds: usize,
    /// Units.
    pub n_units: usize,
    /// Per-(time, world) means: `mean[time * n_worlds + world]`.
    pub mean: Arc<[f64]>,
    /// Per-(time, world) standard deviations (same layout).
    pub sd: Arc<[f64]>,
    /// Outcome dense id.
    pub outcome: DenseNodeId,
}

impl TrajectoryResult {
    /// Mean at `(time, world)`.
    #[must_use]
    pub fn mean_at(&self, time: usize, world: usize) -> f64 {
        self.mean[time * self.n_worlds + world]
    }
}

/// Evaluate trajectories: abduct once (caller supplies `exo`), shared noise across arms/times.
///
/// Each arm's `schedule` length must equal `horizon`; empty schedules are refused.
///
/// # Errors
///
/// Shape / predict failures.
pub fn evaluate_trajectories(
    engine: &CounterfactualEngine,
    exo: &ExogenousPosterior,
    request: &CounterfactualTrajectoryRequest,
    ws: &mut MechanismWorkspace,
    ctx: &ExecutionContext,
) -> Result<TrajectoryResult, CounterfactualError> {
    if request.arms.is_empty() {
        return Err(CounterfactualError::model_msg("no trajectory arms"));
    }
    let horizon = request.arms[0].schedule.len();
    if horizon == 0 {
        return Err(CounterfactualError::model_msg("empty trajectory schedule"));
    }
    for arm in request.arms.iter() {
        if arm.schedule.len() != horizon {
            return Err(CounterfactualError::model_msg(
                "trajectory arms must share a common horizon",
            ));
        }
    }
    let n_worlds = request.arms.len();
    let n_units = exo.n_units;
    let outcome = engine.model.dense_of(request.outcome).ok_or_else(|| {
        CounterfactualError::model_msg(format!("unknown outcome {}", request.outcome))
    })?;

    let mut values = vec![0.0; horizon * n_worlds * n_units];
    let mut mean = vec![0.0; horizon * n_worlds];
    let mut sd = vec![0.0; horizon * n_worlds];

    for t in 0..horizon {
        let mut worlds = Vec::with_capacity(n_worlds);
        for arm in request.arms.iter() {
            worlds.push(CounterfactualWorld {
                unit_rows: None,
                interventions: Arc::clone(&arm.schedule[t]),
            });
        }
        let res = engine.predict(exo, &worlds, &[request.outcome], true, ws, ctx)?;
        for w in 0..n_worlds {
            let col = res.outcome_column(w, outcome)?;
            let dest = t * n_worlds * n_units + w * n_units;
            values[dest..dest + n_units].copy_from_slice(col);
            let m = res.streaming_outcome_mean(w, outcome);
            mean[t * n_worlds + w] = m;
            let mut var = 0.0;
            for &v in col {
                let d = v - m;
                var += d * d;
            }
            sd[t * n_worlds + w] = (var / n_units.max(1) as f64).sqrt();
        }
    }

    Ok(TrajectoryResult {
        values: Arc::from(values),
        horizon,
        n_worlds,
        n_units,
        mean: Arc::from(mean),
        sd: Arc::from(sd),
        outcome,
    })
}
