//! Linear and partial-linear confounding sensitivity analysis.
//!
//! Both refuters here simulate a confounder `U` with a configurable *partial R²* on treatment
//! and outcome, growing it across a grid until the effect is "explained away" (sign flip or
//! collapse to ~0), and report the smallest tested partial R² at which that happens — the
//! robustness value. A larger robustness value means a stronger unmeasured confounder is
//! needed to overturn the conclusion.
//!
//! [`LinearSensitivity`] draws `U` from a standard normal. [`PartialLinearSensitivity`] uses
//! the same procedure with a bounded, non-Gaussian confounder shape as a cheap stand-in for
//! nonparametric misspecification sensitivity — it is *not* a Reisz-representer or
//! debiased-machine-learning sensitivity analysis. Full Reisz-representer diagnostics
//! (DESIGN.md §18.2) are deferred; both structs here only cover the linear and
//! partial-linear/nonparametric-noise variants called for in Phase 4.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::float_cmp
)]

use std::sync::Arc;

use causal_core::ExecutionContext;
use causal_data::TableView;
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{
    RefutationProblem, RefutationReport, fill_gaussian, fit_once, float64_full, sample_sd,
    with_replaced_float,
};
use crate::error::ValidationError;

/// Default partial-R² grid, ascending.
fn default_grid() -> Vec<f64> {
    vec![0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
}

fn run_grid(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
    estimator: &LinearAdjustmentAte,
    grid: &[f64],
    noise_stream: u64,
    nonparametric: bool,
) -> Result<(f64, f64, bool), ValidationError> {
    let n = problem.data.row_count();
    let t0 = float64_full(problem.data, problem.treatment())?;
    let y0 = float64_full(problem.data, problem.outcome())?;
    let sd_t = sample_sd(&t0).max(1e-12);
    let sd_y = sample_sd(&y0).max(1e-12);
    let mut u = vec![0.0; n];
    if nonparametric {
        fill_bounded(&mut u, ctx, noise_stream);
    } else {
        fill_gaussian(&mut u, ctx, noise_stream);
    }

    let mut sorted_grid = grid.to_vec();
    sorted_grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let original_sign = problem.original.ate.signum();
    let mut last_ate = problem.original.ate;
    for &r in &sorted_grid {
        let r = r.clamp(0.0, 0.999);
        let scale = (r / (1.0 - r)).sqrt();
        let t: Vec<f64> =
            t0.iter().zip(&u).map(|(&t, &u)| t + scale * sd_t * u).collect();
        let y: Vec<f64> =
            y0.iter().zip(&u).map(|(&y, &u)| y + scale * sd_y * u).collect();
        let data = with_replaced_float(problem.data, problem.treatment(), Arc::from(t))?;
        let data = with_replaced_float(&data, problem.outcome(), Arc::from(y))?;
        let est = fit_once(estimator, &data, problem.estimand, problem.query, workspace, ctx)?;
        last_ate = est.ate;
        let explained_away = est.ate.abs() < 1e-9 || est.ate.signum() != original_sign;
        if explained_away {
            return Ok((r, last_ate, true));
        }
    }
    let robustness_value = sorted_grid.last().copied().unwrap_or(1.0);
    Ok((robustness_value, last_ate, false))
}

fn fill_bounded(out: &mut [f64], ctx: &ExecutionContext, stream_id: u64) {
    let mut rng = ctx.rng.stream(stream_id);
    for slot in out.iter_mut() {
        *slot = rng.next_f64().mul_add(2.0, -1.0);
    }
}

/// Linear confounding sensitivity: simulated Gaussian confounder with configurable partial R².
#[derive(Clone, Debug)]
pub struct LinearSensitivity {
    /// Ascending grid of partial-R² values to test (shared for treatment and outcome).
    pub partial_r2_grid: Vec<f64>,
    /// Pass if the robustness value exceeds this threshold (harder to explain away).
    pub pass_threshold: f64,
    /// Estimator used for refits (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for LinearSensitivity {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearSensitivity {
    /// Defaults: grid `[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]`, pass threshold 0.1.
    #[must_use]
    pub fn new() -> Self {
        let mut estimator = LinearAdjustmentAte::new();
        estimator.bootstrap_replicates = 0;
        Self { partial_r2_grid: default_grid(), pass_threshold: 0.1, estimator }
    }

    /// Run the linear sensitivity refuter.
    ///
    /// # Errors
    ///
    /// Data or estimation failures, or an empty `partial_r2_grid`.
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        if self.partial_r2_grid.is_empty() {
            return Err(ValidationError::NotApplicable {
                message: "linear sensitivity requires a non-empty partial_r2_grid",
            });
        }
        let (robustness_value, refuted_ate, _explained_away) = run_grid(
            problem,
            workspace,
            ctx,
            &self.estimator,
            &self.partial_r2_grid,
            0xA7E0_000A_u64,
            false,
        )?;
        let passed = robustness_value >= self.pass_threshold;
        Ok(RefutationReport {
            refuter: Arc::from("sensitivity.linear"),
            original_ate: problem.original.ate,
            refuted_ate,
            comparison: robustness_value,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "effect explained away at partial R²={robustness_value}, below threshold {}",
                    self.pass_threshold
                )))
            },
            replicates: self.partial_r2_grid.len() as u32,
        })
    }
}

/// Partial-linear / nonparametric-noise variant of [`LinearSensitivity`].
///
/// Uses the identical grid search but a bounded uniform confounder shape rather than Gaussian,
/// as a minimal stand-in for nonparametric misspecification. See module docs for scope limits.
#[derive(Clone, Debug)]
pub struct PartialLinearSensitivity {
    /// Ascending grid of partial-R² values to test (shared for treatment and outcome).
    pub partial_r2_grid: Vec<f64>,
    /// Pass if the robustness value exceeds this threshold (harder to explain away).
    pub pass_threshold: f64,
    /// Estimator used for refits (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for PartialLinearSensitivity {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialLinearSensitivity {
    /// Defaults: grid `[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]`, pass threshold 0.1.
    #[must_use]
    pub fn new() -> Self {
        let mut estimator = LinearAdjustmentAte::new();
        estimator.bootstrap_replicates = 0;
        Self { partial_r2_grid: default_grid(), pass_threshold: 0.1, estimator }
    }

    /// Run the partial-linear sensitivity refuter.
    ///
    /// # Errors
    ///
    /// Data or estimation failures, or an empty `partial_r2_grid`.
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        if self.partial_r2_grid.is_empty() {
            return Err(ValidationError::NotApplicable {
                message: "partial-linear sensitivity requires a non-empty partial_r2_grid",
            });
        }
        let (robustness_value, refuted_ate, _explained_away) = run_grid(
            problem,
            workspace,
            ctx,
            &self.estimator,
            &self.partial_r2_grid,
            0xA7E0_000B_u64,
            true,
        )?;
        let passed = robustness_value >= self.pass_threshold;
        Ok(RefutationReport {
            refuter: Arc::from("sensitivity.partial_linear"),
            original_ate: problem.original.ate,
            refuted_ate,
            comparison: robustness_value,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "effect explained away at partial R²={robustness_value}, below threshold {}",
                    self.pass_threshold
                )))
            },
            replicates: self.partial_r2_grid.len() as u32,
        })
    }
}
