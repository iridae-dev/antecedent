//! Per-step static counterfactuals with shared-noise batching.
//!
//! The model is a *static* SCM: it has no lagged variables and no state carried
//! between rows of time. A "trajectory" here is therefore a sequence of
//! independent static counterfactuals — time index `t` evaluates the arm's
//! `schedule[t]` interventions on the same abduced noise — and nothing computed at
//! step `t` feeds step `t + 1`. An intervention scheduled at `t = 0` has no effect
//! at `t = 1` unless the schedule repeats it. Dynamics (lagged effects, carry-over)
//! need a temporal model, which this evaluator does not build.
//!
//! Layout is flat columnar — never `Vec<Vec<Vec<_>>>`.
//! `values[time * n_worlds * n_units + world * n_units + unit]`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, Intervention, VariableId};
use antecedent_graph::DenseNodeId;
use antecedent_model::MechanismWorkspace;

use crate::engine::{CounterfactualEngine, CounterfactualWorld, ExogenousPosterior};
use crate::error::CounterfactualError;

/// One trajectory arm: an ordered schedule of intervention sets (time axis).
///
/// Each entry is evaluated as its own static counterfactual; see the module docs.
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
    /// Per-(time, world) means over the units with a finite outcome:
    /// `mean[time * n_worlds + world]`.
    pub mean: Arc<[f64]>,
    /// Per-(time, world) population standard deviations (divide by the finite-unit
    /// count) over the same finite units as `mean` (same layout).
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

/// Mean and population SD over the finite entries of `col`; both `NaN` when none are
/// finite. One rule for both, so a non-finite unit cannot leave a finite mean beside a
/// `NaN` spread.
fn finite_mean_sd(col: &[f64]) -> (f64, f64) {
    let finite: Vec<f64> = col.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let var = finite.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Evaluate per-step static counterfactuals: abduct once (caller supplies `exo`), shared
/// noise across arms/times. No state is carried from one time index to the next.
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
        // Outcomes-only retention: each step reads a single outcome column, so
        // retaining all nodes multiplied peak memory by n_nodes per timestep.
        let res =
            engine.predict_retaining_outcomes(exo, &worlds, &[request.outcome], true, ws, ctx)?;
        for w in 0..n_worlds {
            let col = res.outcome_column(w, outcome)?;
            let dest = t * n_worlds * n_units + w * n_units;
            values[dest..dest + n_units].copy_from_slice(col);
            let (m, s) = finite_mean_sd(col);
            mean[t * n_worlds + w] = m;
            sd[t * n_worlds + w] = s;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean and SD share one population: `[1, 2, NaN, 3]` has finite mean 2 and
    /// population SD `sqrt(2/3)`. The SD used to become `NaN` while the mean skipped
    /// the `NaN` unit.
    #[test]
    fn mean_and_sd_use_the_same_finite_units() {
        let (mean, sd) = finite_mean_sd(&[1.0, 2.0, f64::NAN, 3.0]);
        assert!((mean - 2.0).abs() < 1e-15);
        assert!((sd - (2.0_f64 / 3.0).sqrt()).abs() < 1e-15);
        let (m, s) = finite_mean_sd(&[f64::NAN, f64::INFINITY]);
        assert!(m.is_nan() && s.is_nan());
    }
}
