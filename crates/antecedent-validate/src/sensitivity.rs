//! Linear, partial-linear, and nonparametric confounding sensitivity analysis.
//!
//! [`LinearSensitivity`] and [`PartialLinearSensitivity`] simulate a confounder `U` with a
//! configurable *partial R²* on treatment and outcome under a linear (Gaussian) or
//! partial-linear (bounded) shape. [`NonparametricSensitivity`] first residualizes treatment
//! and outcome on adjustment covariates with Nadaraya–Watson (Nadaraya 1964; Watson 1964) kernel
//! regression, then runs the
//! same partial-R² grid on the residualized series — a production nonparametric path distinct
//! from the partial-linear shape stand-in.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::float_cmp
)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TableView;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::common::{
    RefutationProblem, RefutationReport, complete_case_rows, fill_gaussian, fit_once, float64_full,
    linear_estimator_no_bootstrap, refit_effect, sample_sd, with_replaced_float,
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
    let mut ids = vec![problem.treatment(), problem.outcome()];
    if problem.temporal.is_none() {
        ids.extend_from_slice(&problem.estimand.adjustment_set);
    }
    let (mask, _valid) = complete_case_rows(problem.data, &ids)?;
    // The grid is a *partial* R² — the share of variance in `T` (and `Y`) left unexplained by
    // the adjustment set `Z` that the simulated confounder accounts for. Injecting
    // `scale · SD(T) · u` calibrates against the *marginal* variance instead, so whenever `Z`
    // has real explanatory power the realized partial R² far exceeds the nominal grid value
    // (with R²(T,Z) = 0.8, a nominal 0.2 lands at 0.556) and the reported robustness is
    // misstated. Scale by the residual SD so `scale = √(r/(1−r))` targets the partial R² the
    // docs and the Cinelli–Hazlett convention promise. `NonparametricSensitivity` already
    // residualizes; this brings the linear paths in line.
    let (sd_t, sd_y) =
        residual_sd_pair_on_adjustment(problem, problem.treatment(), problem.outcome(), &mask)?;
    let sd_t = sd_t.max(1e-12);
    let sd_y = sd_y.max(1e-12);
    let mut u = vec![0.0; n];
    if nonparametric {
        fill_bounded(&mut u, ctx, noise_stream);
    } else {
        fill_gaussian(&mut u, ctx, noise_stream);
    }

    let mut sorted_grid = grid.to_vec();
    sorted_grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let original_sign = problem.original.ate.signum();
    // Worst-case orientation: load the confounder on Y against the observed effect so the
    // induced omitted-variable bias works to explain the effect away; a same-sign loading
    // could never flip a positive estimate and would spuriously kill a negative one.
    let dir = if problem.original.ate >= 0.0 { -1.0 } else { 1.0 };
    let mut last_ate = problem.original.ate;
    for &r in &sorted_grid {
        let r = r.clamp(0.0, 0.999);
        let scale = (r / (1.0 - r)).sqrt();
        let t: Vec<f64> = t0.iter().zip(&u).map(|(&t, &u)| t + scale * sd_t * u).collect();
        let y: Vec<f64> = y0.iter().zip(&u).map(|(&y, &u)| y + dir * scale * sd_y * u).collect();
        let data = with_replaced_float(problem.data, problem.treatment(), Arc::from(t))?;
        let data = with_replaced_float(&data, problem.outcome(), Arc::from(y))?;
        let est = if problem.temporal.is_some() {
            refit_effect(problem, &data, problem.estimand, &[], estimator, workspace, ctx)?
        } else {
            fit_once(estimator, &data, problem.estimand, problem.query, workspace, ctx)?
        };
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
    // Uniform on [-√3, √3): unit variance, so the partial-R² grid calibration derived for
    // a standardized confounder holds for the bounded shape too.
    let mut rng = ctx.rng.stream(stream_id);
    let sqrt3 = 3.0_f64.sqrt();
    for slot in out.iter_mut() {
        *slot = rng.next_f64().mul_add(2.0, -1.0) * sqrt3;
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
        Self {
            partial_r2_grid: default_grid(),
            pass_threshold: 0.1,
            estimator: linear_estimator_no_bootstrap(),
        }
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
            0xA7E0_000A_0000_u64,
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

/// Partial-linear sensitivity: same grid as [`LinearSensitivity`] with a bounded uniform
/// confounder shape (partial-linear misspecification), not a nonparametric residualization path.
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
        Self {
            partial_r2_grid: default_grid(),
            pass_threshold: 0.1,
            estimator: linear_estimator_no_bootstrap(),
        }
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
            0xA7E0_000B_0000_u64,
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

/// Nadaraya–Watson leave-one-out prediction of two targets sharing one
/// covariate matrix (`n × dim`, row-major).
///
/// The kernel weight depends only on the covariates, so predicting `y1` and
/// `y2` in one pass halves the O(n²·dim) distance/`exp` work versus two calls;
/// per-target accumulation order is unchanged, so each output matches the
/// single-target form bit for bit.
fn nw_loo_predict_pair(
    y1: &[f64],
    y2: &[f64],
    cov_rowmajor: &[f64],
    dim: usize,
    bandwidth: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = y1.len();
    let h2 = (bandwidth.max(1e-6)).powi(2);
    let mut out1 = vec![0.0; n];
    let mut out2 = vec![0.0; n];
    for i in 0..n {
        let xi = &cov_rowmajor[i * dim..(i + 1) * dim];
        let mut num1 = 0.0;
        let mut num2 = 0.0;
        let mut den = 0.0;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = &cov_rowmajor[j * dim..(j + 1) * dim];
            let mut d2 = 0.0;
            for d in 0..dim {
                let t = xi[d] - xj[d];
                d2 += t * t;
            }
            let w = (-0.5 * d2 / h2).exp();
            num1 += w * y1[j];
            num2 += w * y2[j];
            den += w;
        }
        out1[i] = if den > 1e-15 { num1 / den } else { y1[i] };
        out2[i] = if den > 1e-15 { num2 / den } else { y2[i] };
    }
    (out1, out2)
}

/// SD of each target after linearly regressing it on the adjustment set, i.e.
/// `(SD(first | Z), SD(second | Z))`.
///
/// Falls back to the marginal SD when there is nothing to adjust for (empty `Z`, or a
/// degenerate design the backend refuses) — with no covariates the partial and marginal
/// quantities coincide, so that fallback is exact rather than approximate.
///
/// Both targets regress on the identical `n × (|Z|+1)` design, so build it (and the
/// least-squares workspace) once and solve twice — one `least_squares` call per RHS keeps
/// each target's numeric path, and therefore its result, bit-identical to the historical
/// one-design-per-target form.
pub(crate) fn residual_sd_pair_on_adjustment(
    problem: &RefutationProblem<'_>,
    first: VariableId,
    second: VariableId,
    mask: &[bool],
) -> Result<(f64, f64), ValidationError> {
    let z_ids = problem.estimand.adjustment_set.to_vec();
    let ya = problem.data.float64_masked(first, mask).map_err(ValidationError::from)?;
    let yb = problem.data.float64_masked(second, mask).map_err(ValidationError::from)?;
    // Both fallback conditions depend on the mask and `Z` only, never on the target, so the
    // shared checks reproduce the per-target checks exactly (`ya.len() == yb.len()` by mask).
    if z_ids.is_empty() || ya.len() < z_ids.len() + 2 {
        return Ok((sample_sd(&ya), sample_sd(&yb)));
    }
    let n = ya.len();
    let Some(design) = adjustment_design(problem, mask, n, &z_ids)? else {
        return Ok((sample_sd(&ya), sample_sd(&yb)));
    };
    let mut ws = LeastSquaresWorkspace::default();
    let ncols = z_ids.len() + 1;
    let sd_a = residual_sd_given_design(&design, n, ncols, &ya, &mut ws);
    let sd_b = residual_sd_given_design(&design, n, ncols, &yb, &mut ws);
    Ok((sd_a, sd_b))
}

/// Column-major `[intercept | Z…]` design over the masked rows, or `None` when a covariate
/// extraction disagrees with `n` (callers fall back to the marginal SD, as before).
fn adjustment_design(
    problem: &RefutationProblem<'_>,
    mask: &[bool],
    n: usize,
    z_ids: &[VariableId],
) -> Result<Option<Vec<f64>>, ValidationError> {
    let ncols = z_ids.len() + 1;
    let mut design = Vec::with_capacity(n * ncols);
    design.extend(std::iter::repeat_n(1.0, n));
    for &z in z_ids {
        let col = problem.data.float64_masked(z, mask).map_err(ValidationError::from)?;
        if col.len() != n {
            return Ok(None);
        }
        design.extend_from_slice(&col);
    }
    Ok(Some(design))
}

/// Residual SD of `y` on a prebuilt design; falls back to the marginal SD on solver refusal,
/// non-finite coefficients, or a non-finite residual SD (matching the historical guards).
fn residual_sd_given_design(
    design: &[f64],
    n: usize,
    ncols: usize,
    y: &[f64],
    ws: &mut LeastSquaresWorkspace,
) -> f64 {
    let Ok(fit) = FaerBackend.least_squares(design, n, ncols, y, ws) else {
        return sample_sd(y);
    };
    if fit.coefficients.iter().any(|c| !c.is_finite()) {
        return sample_sd(y);
    }
    let residuals: Vec<f64> = (0..n)
        .map(|r| {
            let mut pred = fit.coefficients[0];
            for c in 1..ncols {
                pred += fit.coefficients[c] * design[c * n + r];
            }
            y[r] - pred
        })
        .collect();
    let sd = sample_sd(&residuals);
    if sd.is_finite() { sd } else { sample_sd(y) }
}

/// Row-major covariate matrix over the complete-case rows of `mask` (adjustment ∪ {T, Y};
/// the caller computes that mask once and shares it with the residualization step).
fn covariate_matrix(
    problem: &RefutationProblem<'_>,
    mask: &[bool],
) -> Result<(Vec<f64>, usize, usize), ValidationError> {
    let ids = problem.estimand.adjustment_set.to_vec();
    let n = mask.iter().filter(|&&k| k).count();
    if ids.is_empty() {
        return Ok((vec![1.0; n], n, 1));
    }
    let dim = ids.len();
    let mut cov = vec![0.0; n * dim];
    for (c, &z) in ids.iter().enumerate() {
        let col = problem.data.float64_masked(z, mask).map_err(ValidationError::from)?;
        for (r, &v) in col.iter().enumerate() {
            cov[r * dim + c] = v;
        }
    }
    Ok((cov, n, dim))
}

fn silverman_bandwidth(cov_rowmajor: &[f64], n: usize, dim: usize) -> f64 {
    if n == 0 || dim == 0 {
        return 1.0;
    }
    let mut sum_sd = 0.0;
    for d in 0..dim {
        let mut vals = Vec::with_capacity(n);
        for r in 0..n {
            vals.push(cov_rowmajor[r * dim + d]);
        }
        sum_sd += sample_sd(&vals);
    }
    let mean_sd = (sum_sd / dim as f64).max(1e-6);
    mean_sd * (n as f64).powf(-1.0 / (dim as f64 + 4.0))
}

/// Nonparametric sensitivity: kernel-residualize T and Y on Z, then partial-R² grid on residuals.
#[derive(Clone, Debug)]
pub struct NonparametricSensitivity {
    /// Ascending grid of partial-R² values to test on residualized series.
    pub partial_r2_grid: Vec<f64>,
    /// Pass if the robustness value exceeds this threshold.
    pub pass_threshold: f64,
    /// Optional bandwidth override; `None` uses Silverman's (1986) rule of thumb.
    pub bandwidth: Option<f64>,
}

impl Default for NonparametricSensitivity {
    fn default() -> Self {
        Self::new()
    }
}

impl NonparametricSensitivity {
    /// Defaults: same partial-R² grid as linear sensitivity, pass threshold 0.1.
    #[must_use]
    pub fn new() -> Self {
        Self { partial_r2_grid: default_grid(), pass_threshold: 0.1, bandwidth: None }
    }

    /// Run nonparametric sensitivity.
    ///
    /// # Errors
    ///
    /// Data failures or empty `partial_r2_grid`.
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
        _workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        if self.partial_r2_grid.is_empty() {
            return Err(ValidationError::NotApplicable {
                message: "nonparametric sensitivity requires a non-empty partial_r2_grid",
            });
        }
        // One complete-case mask (adjustment ∪ {T, Y}) shared by the covariate matrix and
        // the residualization pulls below — the two call sites used the identical id list.
        let mut ids = problem.estimand.adjustment_set.to_vec();
        ids.push(problem.treatment());
        ids.push(problem.outcome());
        let mask = problem.data.complete_case_mask(&ids).map_err(ValidationError::from)?;
        let (cov, n, dim) = covariate_matrix(problem, &mask)?;
        let t = problem
            .data
            .float64_masked(problem.treatment(), &mask)
            .map_err(ValidationError::from)?;
        let y =
            problem.data.float64_masked(problem.outcome(), &mask).map_err(ValidationError::from)?;
        if t.len() != n || y.len() != n {
            return Err(ValidationError::data_msg("nonparametric sensitivity row mismatch"));
        }
        let h = self.bandwidth.unwrap_or_else(|| silverman_bandwidth(&cov, n, dim));
        let (t_hat, y_hat) = nw_loo_predict_pair(&t, &y, &cov, dim, h);
        let t_res: Vec<f64> = t.iter().zip(&t_hat).map(|(&a, &b)| a - b).collect();
        let y_res: Vec<f64> = y.iter().zip(&y_hat).map(|(&a, &b)| a - b).collect();

        let residual_ate = residual_ols_ate(&t_res, &y_res);
        let sd_t = sample_sd(&t_res).max(1e-12);
        let sd_y = sample_sd(&y_res).max(1e-12);
        let mut u = vec![0.0; n];
        fill_gaussian(&mut u, ctx, 0xA7E0_000C_0000_u64);

        let mut sorted_grid = self.partial_r2_grid.clone();
        sorted_grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let original_sign = residual_ate.signum();
        // Worst-case orientation, as in `run_grid`: load U on Y against the observed sign.
        let dir = if residual_ate >= 0.0 { -1.0 } else { 1.0 };
        let mut last_ate = residual_ate;
        let mut robustness_value = sorted_grid.last().copied().unwrap_or(1.0);
        for &r in &sorted_grid {
            let r = r.clamp(0.0, 0.999);
            let scale = (r / (1.0 - r)).sqrt();
            let t_pert: Vec<f64> =
                t_res.iter().zip(&u).map(|(&tv, &uu)| tv + scale * sd_t * uu).collect();
            let y_pert: Vec<f64> =
                y_res.iter().zip(&u).map(|(&yv, &uu)| yv + dir * scale * sd_y * uu).collect();
            last_ate = residual_ols_ate(&t_pert, &y_pert);
            if last_ate.abs() < 1e-9 || last_ate.signum() != original_sign {
                robustness_value = r;
                break;
            }
        }
        let passed = robustness_value >= self.pass_threshold;
        Ok(RefutationReport {
            refuter: Arc::from("sensitivity.nonparametric"),
            original_ate: problem.original.ate,
            refuted_ate: last_ate,
            comparison: robustness_value,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "nonparametric residual effect explained away at partial R²={robustness_value}, \
                     below threshold {}",
                    self.pass_threshold
                )))
            },
            replicates: self.partial_r2_grid.len() as u32,
        })
    }
}

fn residual_ols_ate(t: &[f64], y: &[f64]) -> f64 {
    let n = t.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mean_t = t.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (&ti, &yi) in t.iter().zip(y) {
        let dt = ti - mean_t;
        num += dt * (yi - mean_y);
        den += dt * dt;
    }
    if den < 1e-15 { 0.0 } else { num / den }
}
