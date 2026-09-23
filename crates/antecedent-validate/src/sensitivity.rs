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
//! The reported `robustness_value` is the *simulated tipping partial R²*: the smallest grid
//! value at which adding `U` (added to the observed treatment and outcome, then refitting)
//! flips or annihilates the estimate. It is **not** the closed-form Cinelli–Hazlett (2020)
//! robustness value: the confounder here perturbs the observed `T` and `Y`, which also
//! attenuates the slope like measurement error, and the outcome-side R² is relative to
//! `Var(Y | Z)` rather than `Var(Y | T, Z)`. The simulated confounder is drawn uncorrelated
//! with the design (`orthogonalize_confounder`), so on an OLS design the tipping point is
//! exactly `r* = |ρ| / (1 + |ρ|)` (`ρ` the partial correlation of `T` and `Y` given `Z`), for
//! example `ρ = 0.3 → 0.231` against Cinelli–Hazlett's `0.269`; it is the more conservative
//! of the two.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::float_cmp,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TableView;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte, LinearFitKind};
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, chol_solve, cholesky_spd, form_xtx,
    form_xty,
};

use crate::common::{
    RefutationProblem, RefutationReport, complete_case_rows, fill_gaussian, fit_once, float64_full,
    linear_estimator_no_bootstrap, refit_effect, sample_sd, with_replaced_float,
};
use crate::error::ValidationError;

/// Sensitivity perturbations act on the identified lag-aligned regression rows.
/// This changes no graph certificate and makes no iid sampling-uncertainty claim.
fn with_temporal_diagnostic_rows<R>(
    problem: &RefutationProblem<'_>,
    run: impl FnOnce(&RefutationProblem<'_>) -> Result<R, ValidationError>,
) -> Result<R, ValidationError> {
    use antecedent_core::{
        CausalSchemaBuilder, Intervention, MeasurementSpec, RoleHint, SmallRoleSet, Value,
        ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    let prep = crate::common::temporal_diagnostic_design(problem)?;
    let n = prep.design.nrows;
    let mut values = vec![prep.treatment.to_vec(), prep.design.outcome.to_vec()];
    for i in 0..prep.adjustment_set.len() {
        let start = (i + 2) * n;
        values.push(prep.design.matrix[start..start + n].to_vec());
    }
    let mut schema = CausalSchemaBuilder::new();
    let mut columns = Vec::new();
    let mut ids = Vec::new();
    for (i, values) in values.into_iter().enumerate() {
        let id = VariableId::from_raw(
            u32::try_from(i)
                .map_err(|_| ValidationError::data_msg("diagnostic design exceeds u32 columns"))?,
        );
        schema
            .add_variable(
                format!("aligned_{i}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .map_err(|e| ValidationError::data_msg(e.to_string()))?;
        columns.push(OwnedColumn::Float64(Float64Column::new(
            id,
            Arc::from(values),
            ValidityBitmap::all_valid(n),
        )?));
        ids.push(id);
    }
    let schema = schema.build().map_err(|e| ValidationError::data_msg(e.to_string()))?;
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None)?);
    let mut estimand = problem.estimand.clone();
    estimand.method = Arc::from("backdoor.adjustment");
    estimand.adjustment_set = Arc::from(&ids[2..]);
    let query = antecedent_core::AverageEffectQuery::new(
        ids[0],
        ids[1],
        Arc::from([]),
        Intervention::set(ids[0], Value::Float64(prep.control)),
        Intervention::set(ids[0], Value::Float64(prep.active)),
        prep.target_population,
    );
    let aligned = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        problem.original,
        Some("linear.adjustment.ate"),
        None,
    );
    run(&aligned)
}

/// Default partial-R² grid, ascending.
fn default_grid() -> Vec<f64> {
    vec![0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
}

/// Grid tipping partial R²: the smallest probed partial R² that explains the effect away
/// (an upper bound on the true tipping point, which lies in `(previous grid value, this]`).
/// If no grid point tips the estimate, the tipping value exceeds the whole grid — report
/// `+∞` rather than the last grid point (which would look like a tipping strength that was
/// never observed).
fn grid_robustness_value(explained_away_at: Option<f64>) -> f64 {
    explained_away_at.unwrap_or(f64::INFINITY)
}

/// Pass only when the robustness value is *strictly* above the caller's bar.
/// Equality means a confounder at the threshold already kills the effect.
fn robustness_passes(robustness_value: f64, pass_threshold: f64) -> bool {
    robustness_value > pass_threshold
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
    let setup = GridSetup::new(problem, ctx, grid, noise_stream, nonparametric)?;
    if gram_applicable(problem, estimator) {
        if let Some(result) = try_run_grid_gram(problem, estimator, &setup)? {
            return Ok(result);
        }
    }
    run_grid_data_pass(problem, workspace, ctx, estimator, &setup)
}

/// Static OLS, no nested bootstrap: the grid is one Gram of `[1, T, Z, u]` plus
/// a `p×p` Cholesky per grid point. Temporal / ridge / lasso / Huber keep the
/// per-point data pass.
fn gram_applicable(problem: &RefutationProblem<'_>, estimator: &LinearAdjustmentAte) -> bool {
    problem.temporal.is_none()
        && estimator.bootstrap_replicates == 0
        && matches!(estimator.fit_kind, LinearFitKind::Ols)
}

struct GridSetup {
    t0: Vec<f64>,
    y0: Vec<f64>,
    u: Vec<f64>,
    sd_t: f64,
    sd_y: f64,
    dir: f64,
    sorted_grid: Vec<f64>,
    original_sign: f64,
    original_ate: f64,
}

impl GridSetup {
    fn new(
        problem: &RefutationProblem<'_>,
        ctx: &ExecutionContext,
        grid: &[f64],
        noise_stream: u64,
        nonparametric: bool,
    ) -> Result<Self, ValidationError> {
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
        // (with R²(T,Z) = 0.8, a nominal 0.2 lands at 0.556) and the reported tipping value is
        // misstated. Scale by the residual SD so `scale = √(r/(1−r))` targets the partial R² the
        // docs promise. `NonparametricSensitivity` already residualizes; this brings the linear
        // paths in line.
        let (sd_t, sd_y) =
            residual_sd_pair_on_adjustment(problem, problem.treatment(), problem.outcome(), &mask)?;
        // A zero (or undefined) residual SD leaves no variation for a confounder to explain.
        if !(sd_t.is_finite() && sd_t > 0.0 && sd_y.is_finite() && sd_y > 0.0) {
            return Err(ValidationError::NotApplicable {
                message: "sensitivity requires positive, finite residual variation in treatment \
                          and outcome given the adjustment set",
            });
        }
        let mut u = vec![0.0; n];
        if nonparametric {
            fill_bounded(&mut u, ctx, noise_stream);
        } else {
            fill_gaussian(&mut u, ctx, noise_stream);
        }
        // Remove the O(1/√n) sampling correlation of the drawn `U` with the design, so the
        // realized partial R² is the nominal grid value and the verdict carries no simulation
        // noise (see [`orthogonalize_confounder`]).
        let mut columns = vec![
            problem.data.float64_masked(problem.treatment(), &mask)?,
            problem.data.float64_masked(problem.outcome(), &mask)?,
        ];
        for &z in problem.estimand.adjustment_set.iter() {
            columns.push(problem.data.float64_masked(z, &mask)?);
        }
        orthogonalize_confounder(&mut u, &mask, &columns);

        let mut sorted_grid = grid.to_vec();
        sorted_grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let original_sign = problem.original.ate.signum();
        // Worst-case orientation: load the confounder on Y against the observed effect so the
        // induced omitted-variable bias works to explain the effect away; a same-sign loading
        // could never flip a positive estimate and would spuriously kill a negative one.
        let dir = if problem.original.ate >= 0.0 { -1.0 } else { 1.0 };
        Ok(Self {
            t0,
            y0,
            u,
            sd_t,
            sd_y,
            dir,
            sorted_grid,
            original_sign,
            original_ate: problem.original.ate,
        })
    }
}

/// Make the simulated confounder exactly uncorrelated with the design it perturbs.
///
/// A raw standard-normal draw has sample correlation `O(1/√n)` with `T`, `Y` and `Z`, so the
/// realized partial R² differs from the nominal grid value and the tipping point wanders with
/// the seed. Replace `u` on the complete-case rows (`mask`) by its residual from the least-squares
/// regression on `[1, columns…]`, rescaled to unit sample SD, and zero elsewhere. Then
/// `T' = T + a·u` and `Y' = Y + b·u` give `Cov(T'|Z, Y'|Z) = Cov(T|Z, Y|Z) + a·b·SD(u)²` and
/// `Var(T'|Z) = Var(T|Z) + a²·SD(u)²` exactly, so the OLS slope on the perturbed data is
/// `(S_ty + a·b) / (S_tt + a²)` in closed form, independent of the draw.
///
/// Left unchanged when the design is rank deficient or too small for the residual to exist (the
/// raw draw is then a valid, only slightly noisier, confounder).
fn orthogonalize_confounder(u: &mut [f64], mask: &[bool], columns: &[Vec<f64>]) {
    let rows: Vec<usize> = mask.iter().enumerate().filter_map(|(i, &k)| k.then_some(i)).collect();
    let m = rows.len();
    let ncols = columns.len() + 1;
    if m < ncols + 2 || columns.iter().any(|c| c.len() != m) || rows.iter().any(|&i| i >= u.len()) {
        return;
    }
    let mut design = vec![1.0; m];
    for column in columns {
        design.extend_from_slice(column);
    }
    let target: Vec<f64> = rows.iter().map(|&i| u[i]).collect();
    let Ok(fit) = FaerBackend.least_squares(
        &design,
        m,
        ncols,
        &target,
        &mut LeastSquaresWorkspace::default(),
    ) else {
        return;
    };
    if fit.rank != ncols || fit.residuals.len() != m {
        return;
    }
    let sd = sample_sd(&fit.residuals);
    if !(sd.is_finite() && sd > 0.0) || fit.residuals.iter().any(|r| !r.is_finite()) {
        return;
    }
    u.fill(0.0);
    for (&row, &r) in rows.iter().zip(&fit.residuals) {
        u[row] = r / sd;
    }
}

fn run_grid_data_pass(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
    estimator: &LinearAdjustmentAte,
    setup: &GridSetup,
) -> Result<(f64, f64, bool), ValidationError> {
    let mut last_ate = setup.original_ate;
    for &r in &setup.sorted_grid {
        let r = r.clamp(0.0, 0.999);
        last_ate = data_pass_ate(problem, workspace, ctx, estimator, setup, r)?;
        #[allow(
            clippy::float_cmp,
            reason = "signum returns exactly +/-1 (or NaN), so comparing two signs for equality is exact"
        )]
        let explained_away = last_ate.abs() < 1e-9 || last_ate.signum() != setup.original_sign;
        if explained_away {
            return Ok((grid_robustness_value(Some(r)), last_ate, true));
        }
    }
    Ok((grid_robustness_value(None), last_ate, false))
}

fn data_pass_ate(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
    estimator: &LinearAdjustmentAte,
    setup: &GridSetup,
    r: f64,
) -> Result<f64, ValidationError> {
    let r = r.clamp(0.0, 0.999);
    let scale = (r / (1.0 - r)).sqrt();
    let t: Vec<f64> =
        setup.t0.iter().zip(&setup.u).map(|(&t, &u)| t + scale * setup.sd_t * u).collect();
    let y: Vec<f64> = setup
        .y0
        .iter()
        .zip(&setup.u)
        .map(|(&y, &u)| y + setup.dir * scale * setup.sd_y * u)
        .collect();
    let data = with_replaced_float(problem.data, problem.treatment(), Arc::from(t))?;
    let data = with_replaced_float(&data, problem.outcome(), Arc::from(y))?;
    let est = if problem.temporal.is_some() {
        refit_effect(problem, &data, problem.estimand, &[], estimator, workspace, ctx)?
    } else {
        fit_once(estimator, &data, problem.estimand, problem.query, workspace, ctx)?
    };
    Ok(est.ate)
}

/// One Gram of `W = [1, T, Z, u]` over the same complete-case rows `prepare` uses.
/// Returns `None` when Cholesky refuses (caller falls back to the data pass).
fn try_run_grid_gram(
    problem: &RefutationProblem<'_>,
    estimator: &LinearAdjustmentAte,
    setup: &GridSetup,
) -> Result<Option<(f64, f64, bool)>, ValidationError> {
    let Some(gram) = SensitivityGram::compile(problem, estimator, &setup.u)? else {
        return Ok(None);
    };
    let mut last_ate = setup.original_ate;
    for &r in &setup.sorted_grid {
        let r = r.clamp(0.0, 0.999);
        let scale = (r / (1.0 - r)).sqrt();
        let a = scale * setup.sd_t;
        let b = setup.dir * scale * setup.sd_y;
        let Some(ate) = gram.ate_at(a, b) else {
            return Ok(None);
        };
        last_ate = ate;
        #[allow(
            clippy::float_cmp,
            reason = "signum returns exactly +/-1 (or NaN), so comparing two signs for equality is exact"
        )]
        let explained_away = last_ate.abs() < 1e-9 || last_ate.signum() != setup.original_sign;
        if explained_away {
            return Ok(Some((grid_robustness_value(Some(r)), last_ate, true)));
        }
    }
    Ok(Some((grid_robustness_value(None), last_ate, false)))
}

/// Sufficient statistics `WᵀW` / `WᵀY` for `W = [X | u]`, `X = [1, T, Z]`.
struct SensitivityGram {
    g: Vec<f64>,
    gy: Vec<f64>,
    p: usize,
    treatment_delta: f64,
}

impl SensitivityGram {
    fn compile(
        problem: &RefutationProblem<'_>,
        estimator: &LinearAdjustmentAte,
        u: &[f64],
    ) -> Result<Option<Self>, ValidationError> {
        if !problem.query.effect_modifiers.is_empty() {
            // Fast path is β_T · Δ on [1, T, Z]; the licensed conditional scalar is
            // (β_T + β_{T×W} Ē[W]) · Δ. Fall back to the data pass, which refits the
            // interaction model.
            return Ok(None);
        }
        // Perturbations preserve validity, so compile the original complete cases.
        let prep = estimator
            .prepare(problem.data, problem.estimand, problem.query)
            .map_err(ValidationError::from)?;
        let n = prep.design.nrows;
        let p = prep.design.ncols;
        if n == 0 || p < 2 || prep.design.row_selection.len() != n {
            return Ok(None);
        }
        let q = p + 1;
        let mut w = vec![0.0; n * q];
        w[..n * p].copy_from_slice(prep.design.matrix.as_ref());
        for (i, &row) in prep.design.row_selection.iter().enumerate() {
            if row >= u.len() {
                return Err(ValidationError::data_msg(
                    "sensitivity Gram row_selection exceeds confounder length",
                ));
            }
            w[n * p + i] = u[row];
        }
        let mut g = vec![0.0; q * q];
        form_xtx(&w, n, q, &mut g);
        let mut gy = vec![0.0; q];
        form_xty(&w, n, q, prep.design.outcome.as_ref(), &mut gy);
        Ok(Some(Self { g, gy, p, treatment_delta: prep.treatment_delta }))
    }

    fn ate_at(&self, a: f64, b: f64) -> Option<f64> {
        let p = self.p;
        let mut xtx = vec![0.0; p * p];
        let mut xty = vec![0.0; p];
        assemble_perturbed_normal_eq(&self.g, &self.gy, p, a, b, &mut xtx, &mut xty);
        let chol = cholesky_spd(&xtx, p)?;
        let beta = chol_solve(&chol, p, &xty)?;
        // Linear main-effects ATE is β_T · Δ for ATE/ATT/ATC/predicate (gcomp equals this).
        Some(beta[1] * self.treatment_delta)
    }
}

/// Assemble `X'ᵀX'` / `X'ᵀY'` for `X' = [1, T + a u, Z]` and `Y' = Y + b u`
/// from the Gram of `W = [1, T, Z, u]`.
fn assemble_perturbed_normal_eq(
    g: &[f64],
    gy: &[f64],
    p: usize,
    a: f64,
    b: f64,
    xtx: &mut [f64],
    xty: &mut [f64],
) {
    let q = p + 1;
    let u_idx = p;
    debug_assert!(g.len() >= q * q);
    debug_assert!(gy.len() >= q);
    debug_assert!(xtx.len() >= p * p);
    debug_assert!(xty.len() >= p);
    for i in 0..p {
        for j in 0..p {
            let mut v = g[i * q + j];
            if j == 1 {
                v += a * g[i * q + u_idx];
            }
            if i == 1 {
                v += a * g[j * q + u_idx];
            }
            if i == 1 && j == 1 {
                v += a * a * g[u_idx * q + u_idx];
            }
            xtx[i * p + j] = v;
        }
        let mut rhs = gy[i] + b * g[i * q + u_idx];
        if i == 1 {
            rhs += a * gy[u_idx] + a * b * g[u_idx * q + u_idx];
        }
        xty[i] = rhs;
    }
}

/// Per-grid-point ATEs along the current data-pass path (differential tests).
#[cfg(test)]
pub(crate) fn grid_ates_data_pass(
    problem: &RefutationProblem<'_>,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
    estimator: &LinearAdjustmentAte,
    grid: &[f64],
    noise_stream: u64,
    nonparametric: bool,
) -> Result<Vec<f64>, ValidationError> {
    let setup = GridSetup::new(problem, ctx, grid, noise_stream, nonparametric)?;
    let mut ates = Vec::with_capacity(setup.sorted_grid.len());
    for &r in &setup.sorted_grid {
        ates.push(data_pass_ate(problem, workspace, ctx, estimator, &setup, r)?);
    }
    Ok(ates)
}

/// Per-grid-point ATEs along the Gram path (`None` if Cholesky refuses).
#[cfg(test)]
pub(crate) fn grid_ates_gram(
    problem: &RefutationProblem<'_>,
    estimator: &LinearAdjustmentAte,
    ctx: &ExecutionContext,
    grid: &[f64],
    noise_stream: u64,
    nonparametric: bool,
) -> Result<Option<Vec<f64>>, ValidationError> {
    let setup = GridSetup::new(problem, ctx, grid, noise_stream, nonparametric)?;
    let Some(gram) = SensitivityGram::compile(problem, estimator, &setup.u)? else {
        return Ok(None);
    };
    let mut ates = Vec::with_capacity(setup.sorted_grid.len());
    for &r in &setup.sorted_grid {
        let r = r.clamp(0.0, 0.999);
        let scale = (r / (1.0 - r)).sqrt();
        let a = scale * setup.sd_t;
        let b = setup.dir * scale * setup.sd_y;
        let Some(ate) = gram.ate_at(a, b) else {
            return Ok(None);
        };
        ates.push(ate);
    }
    Ok(Some(ates))
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
    /// Pass if the robustness value *strictly exceeds* this threshold (harder to explain away).
    /// Equality fails: a confounder at the bar already kills the effect.
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
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the refutation reports its grid size as u32, and a partial-R-squared grid is far below 2^32 points"
    )]
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        if problem.temporal.is_some() {
            return with_temporal_diagnostic_rows(problem, |aligned| {
                self.refute(aligned, workspace, ctx)
            });
        }
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
        let passed = robustness_passes(robustness_value, self.pass_threshold);
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
                    "effect explained away at partial R²={robustness_value}, not strictly above threshold {}",
                    self.pass_threshold
                )))
            },
            replicates: self.partial_r2_grid.len() as u32,
        })
    }
}

/// Partial-linear sensitivity: same grid as [`LinearSensitivity`] with a bounded uniform
/// confounder shape (partial-linear misspecification), not a nonparametric residualization path.
///
/// The confounder is orthogonalized to the design (see the module docs), and least squares sees
/// only second moments, so for the OLS refit this coincides with [`LinearSensitivity`]; the
/// bounded shape matters for the non-OLS refits (ridge, lasso, Huber) and the temporal path.
#[derive(Clone, Debug)]
pub struct PartialLinearSensitivity {
    /// Ascending grid of partial-R² values to test (shared for treatment and outcome).
    pub partial_r2_grid: Vec<f64>,
    /// Pass if the robustness value *strictly exceeds* this threshold (harder to explain away).
    /// Equality fails: a confounder at the bar already kills the effect.
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
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the refutation reports its grid size as u32, and a partial-R-squared grid is far below 2^32 points"
    )]
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        if problem.temporal.is_some() {
            return with_temporal_diagnostic_rows(problem, |aligned| {
                self.refute(aligned, workspace, ctx)
            });
        }
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
        let passed = robustness_passes(robustness_value, self.pass_threshold);
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
                    "effect explained away at partial R²={robustness_value}, not strictly above threshold {}",
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
///
/// Rows are independent, so the outer loop runs on `ctx`'s pool (results come back in row order,
/// each row summed in `j` order, so the output does not depend on the thread count). Distances are
/// squared Euclidean, which is exact for covariates already on a common scale (the caller
/// standardizes); the weight shift `exp(−½ (d² − min d²) / h²)` keeps the nearest observations
/// at weight 1 even for a tiny bandwidth.
fn nw_loo_predict_pair(
    y1: &[f64],
    y2: &[f64],
    cov_rowmajor: &[f64],
    dim: usize,
    bandwidth: f64,
    ctx: &ExecutionContext,
) -> Result<(Vec<f64>, Vec<f64>), ValidationError> {
    let n = y1.len();
    let h2 = bandwidth * bandwidth;
    let rows = ctx.map_indexed(n, |i, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(ValidationError::Cancelled);
        }
        let xi = &cov_rowmajor[i * dim..(i + 1) * dim];
        let mut squared = vec![f64::INFINITY; n];
        let mut nearest = f64::INFINITY;
        for (j, slot) in squared.iter_mut().enumerate() {
            if i == j {
                continue;
            }
            let xj = &cov_rowmajor[j * dim..(j + 1) * dim];
            let mut acc = 0.0;
            for d in 0..dim {
                let diff = xi[d] - xj[d];
                acc += diff * diff;
            }
            *slot = acc;
            nearest = nearest.min(acc);
        }
        let mut num1 = 0.0;
        let mut num2 = 0.0;
        let mut den = 0.0;
        for (j, &d2) in squared.iter().enumerate() {
            if i == j {
                continue;
            }
            // The nearest observations always have weight 1, even for a tiny bandwidth.
            #[allow(
                clippy::float_cmp,
                reason = "nearest is the minimum of these very squared distances, so the nearest observations compare exactly equal"
            )]
            let w = if d2 == nearest { 1.0 } else { (-0.5 * (d2 - nearest) / h2).exp() };
            num1 += w * y1[j];
            num2 += w * y2[j];
            den += w;
        }
        Ok((num1 / den, num2 / den))
    })?;
    Ok(rows.into_iter().unzip())
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

/// `(SD(T | Z), SD(Y | Z))` on the lag-aligned rows of a temporal problem, where `Z` is the
/// unfolded adjustment set of the identified design (the dense node ids are not schema
/// columns, so the aligned design carries them). Falls back to the marginal SDs with no
/// adjustment covariates or too few rows, as [`residual_sd_pair_on_adjustment`] does.
pub(crate) fn temporal_residual_sd_pair(
    problem: &RefutationProblem<'_>,
) -> Result<(f64, f64), ValidationError> {
    let prep = crate::common::temporal_diagnostic_design(problem)?;
    let n = prep.design.nrows;
    let k = prep.adjustment_set.len();
    let treatment: &[f64] = &prep.treatment;
    let outcome: &[f64] = &prep.design.outcome;
    if k == 0 || n < k + 2 {
        return Ok((sample_sd(treatment), sample_sd(outcome)));
    }
    let mut design = vec![1.0; n];
    for i in 0..k {
        design.extend_from_slice(&prep.design.matrix[(i + 2) * n..(i + 3) * n]);
    }
    let mut ws = LeastSquaresWorkspace::default();
    Ok((
        residual_sd_given_design(&design, n, k + 1, treatment, &mut ws),
        residual_sd_given_design(&design, n, k + 1, outcome, &mut ws),
    ))
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
/// the caller computes that mask once and shares it with the residualization step), each
/// column divided by its sample SD (a constant column is left as is) so one isotropic
/// bandwidth means the same number of standard deviations in every direction.
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
    standardize_columns(&mut cov, n, dim);
    Ok((cov, n, dim))
}

/// Divide each column of a row-major `n × dim` matrix by its sample SD (a constant or
/// non-finite-SD column is left unscaled).
fn standardize_columns(cov: &mut [f64], n: usize, dim: usize) {
    for d in 0..dim {
        let column: Vec<f64> = (0..n).map(|r| cov[r * dim + d]).collect();
        let sd = sample_sd(&column);
        if sd.is_finite() && sd > 0.0 {
            for r in 0..n {
                cov[r * dim + d] /= sd;
            }
        }
    }
}

/// Silverman's (1986) multivariate normal-reference bandwidth for covariates standardized to unit
/// SD: `h = (4 / (d + 2))^(1/(d+4)) · n^(−1/(d+4))`.
fn silverman_bandwidth(n: usize, dim: usize) -> f64 {
    if n == 0 || dim == 0 {
        return 1.0;
    }
    let d = dim as f64;
    (4.0 / (d + 2.0)).powf(1.0 / (d + 4.0)) * (n as f64).powf(-1.0 / (d + 4.0))
}

/// Largest complete-case sample the O(n²·d) leave-one-out kernel smoother accepts.
const MAX_NONPARAMETRIC_ROWS: usize = 20_000;

/// Nonparametric sensitivity: kernel-residualize T and Y on Z, then partial-R² grid on residuals.
#[derive(Clone, Debug)]
pub struct NonparametricSensitivity {
    /// Ascending grid of partial-R² values to test on residualized series.
    pub partial_r2_grid: Vec<f64>,
    /// Pass if the robustness value *strictly exceeds* this threshold.
    /// Equality fails: a confounder at the bar already kills the residual effect.
    pub pass_threshold: f64,
    /// Optional bandwidth override, in standard deviations of each (standardized) covariate;
    /// `None` uses Silverman's (1986) multivariate normal-reference rule.
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
    #[allow(clippy::only_used_in_recursion)]
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the refutation reports its grid size as u32, and a partial-R-squared grid is far below 2^32 points"
    )]
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        if problem.temporal.is_some() {
            return with_temporal_diagnostic_rows(problem, |aligned| {
                self.refute(aligned, workspace, ctx)
            });
        }
        if self.partial_r2_grid.is_empty() {
            return Err(ValidationError::NotApplicable {
                message: "nonparametric sensitivity requires a non-empty partial_r2_grid",
            });
        }
        let (t_res, y_res, n) = kernel_residuals(problem, self.bandwidth, ctx)?;
        let (_, _, treatment_delta) = antecedent_estimate::prepare::treatment_contrast(
            &problem.query.active,
            &problem.query.control,
        )?;
        let residual_ate = residual_ols_ate(&t_res, &y_res);
        // The check perturbs the *residual* slope, which need not share the published
        // estimate's sign; a tipping value for a different effect says nothing about it.
        // (`residual_ate` is per unit of treatment; the published estimate is per contrast.)
        let residual_effect = residual_ate * treatment_delta;
        if !residual_effect.is_finite()
            || (residual_effect != 0.0
                && problem.original.ate != 0.0
                && residual_effect.signum() != problem.original.ate.signum())
        {
            return Err(ValidationError::NotApplicable {
                message: "the kernel-residualized effect is undefined or has the opposite sign \
                          of the published estimate, so its tipping value does not describe it",
            });
        }
        let sd_t = sample_sd(&t_res);
        let sd_y = sample_sd(&y_res);
        if !(sd_t.is_finite() && sd_t > 0.0 && sd_y.is_finite() && sd_y > 0.0) {
            return Err(ValidationError::NotApplicable {
                message: "nonparametric sensitivity requires positive residual variation in \
                          treatment and outcome after kernel residualization",
            });
        }
        let mut u = vec![0.0; n];
        fill_gaussian(&mut u, ctx, 0xA7E0_000C_0000_u64);
        orthogonalize_confounder(&mut u, &vec![true; n], &[t_res.clone(), y_res.clone()]);

        let mut sorted_grid = self.partial_r2_grid.clone();
        sorted_grid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let original_sign = residual_ate.signum();
        // Worst-case orientation, as in `run_grid`: load U on Y against the observed sign.
        let dir = if residual_ate >= 0.0 { -1.0 } else { 1.0 };
        let mut last_ate = residual_ate;
        let mut explained_away_at = None;
        for &r in &sorted_grid {
            let r = r.clamp(0.0, 0.999);
            let scale = (r / (1.0 - r)).sqrt();
            let t_pert: Vec<f64> =
                t_res.iter().zip(&u).map(|(&tv, &uu)| tv + scale * sd_t * uu).collect();
            let y_pert: Vec<f64> =
                y_res.iter().zip(&u).map(|(&yv, &uu)| yv + dir * scale * sd_y * uu).collect();
            last_ate = residual_ols_ate(&t_pert, &y_pert);
            #[allow(
                clippy::float_cmp,
                reason = "signum returns exactly +/-1 (or NaN), so comparing two signs for equality is exact"
            )]
            let sign_flipped = last_ate.signum() != original_sign;
            if last_ate.abs() < 1e-9 || sign_flipped {
                explained_away_at = Some(r);
                break;
            }
        }
        let robustness_value = grid_robustness_value(explained_away_at);
        let passed = robustness_passes(robustness_value, self.pass_threshold);
        Ok(RefutationReport {
            refuter: Arc::from("sensitivity.nonparametric"),
            original_ate: problem.original.ate,
            refuted_ate: last_ate * treatment_delta,
            comparison: robustness_value,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "nonparametric residual effect explained away at partial R²={robustness_value}, \
                     not strictly above threshold {}",
                    self.pass_threshold
                )))
            },
            replicates: self.partial_r2_grid.len() as u32,
        })
    }
}

/// Leave-one-out Nadaraya–Watson residuals of treatment and outcome on the adjustment
/// covariates, over the complete cases, with the complete-case row count.
fn kernel_residuals(
    problem: &RefutationProblem<'_>,
    bandwidth: Option<f64>,
    ctx: &ExecutionContext,
) -> Result<(Vec<f64>, Vec<f64>, usize), ValidationError> {
    // One complete-case mask (adjustment ∪ {T, Y}) shared by the covariate matrix and
    // the residualization pulls below — the two call sites used the identical id list.
    let mut ids = problem.estimand.adjustment_set.to_vec();
    ids.push(problem.treatment());
    ids.push(problem.outcome());
    let mask = problem.data.complete_case_mask(&ids).map_err(ValidationError::from)?;
    let (cov, n, dim) = covariate_matrix(problem, &mask)?;
    let t =
        problem.data.float64_masked(problem.treatment(), &mask).map_err(ValidationError::from)?;
    let y = problem.data.float64_masked(problem.outcome(), &mask).map_err(ValidationError::from)?;
    if t.len() != n || y.len() != n {
        return Err(ValidationError::data_msg("nonparametric sensitivity row mismatch"));
    }
    if n < 2 {
        return Err(ValidationError::data_msg(
            "nonparametric sensitivity requires at least 2 complete cases",
        ));
    }
    if n > MAX_NONPARAMETRIC_ROWS {
        return Err(ValidationError::NotApplicable {
            message: "nonparametric sensitivity is quadratic in the sample size and is \
                      limited to 20000 complete-case rows",
        });
    }
    let h = bandwidth.unwrap_or_else(|| silverman_bandwidth(n, dim));
    if !h.is_finite() || h <= 0.0 {
        return Err(ValidationError::data_msg(
            "nonparametric sensitivity bandwidth must be finite and positive",
        ));
    }
    let (t_hat, y_hat) = nw_loo_predict_pair(&t, &y, &cov, dim, h, ctx)?;
    let t_res: Vec<f64> = t.iter().zip(&t_hat).map(|(&a, &b)| a - b).collect();
    let y_res: Vec<f64> = y.iter().zip(&y_hat).map(|(&a, &b)| a - b).collect();
    Ok((t_res, y_res, n))
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

#[cfg(test)]
mod gram_algebra {
    use super::assemble_perturbed_normal_eq;
    use antecedent_stats::{form_xtx, form_xty};

    #[test]
    fn assemble_matches_explicit_perturbed_design() {
        // n=4, p=3: X = [1, T, Z], plus u.
        let n = 4usize;
        let p = 3usize;
        let t = [0.0, 1.0, 0.0, 1.0];
        let z = [0.2, 0.4, 0.6, 0.8];
        let y = [1.0, 3.0, 2.0, 4.0];
        let u = [0.5, -0.5, 1.0, -1.0];
        let a = 0.3;
        let b = -0.7;
        let q = p + 1;
        let mut w = vec![0.0; n * q];
        for r in 0..n {
            w[r] = 1.0;
            w[n + r] = t[r];
            w[2 * n + r] = z[r];
            w[3 * n + r] = u[r];
        }
        let mut g = vec![0.0; q * q];
        form_xtx(&w, n, q, &mut g);
        let mut gy = vec![0.0; q];
        form_xty(&w, n, q, &y, &mut gy);
        let mut xtx = vec![0.0; p * p];
        let mut xty = vec![0.0; p];
        assemble_perturbed_normal_eq(&g, &gy, p, a, b, &mut xtx, &mut xty);

        let mut xp = vec![0.0; n * p];
        let mut yp = vec![0.0; n];
        for r in 0..n {
            xp[r] = 1.0;
            xp[n + r] = t[r] + a * u[r];
            xp[2 * n + r] = z[r];
            yp[r] = y[r] + b * u[r];
        }
        let mut xtx_ref = vec![0.0; p * p];
        form_xtx(&xp, n, p, &mut xtx_ref);
        let mut xty_ref = vec![0.0; p];
        form_xty(&xp, n, p, &yp, &mut xty_ref);
        for i in 0..p * p {
            assert!((xtx[i] - xtx_ref[i]).abs() < 1e-12, "xtx[{i}]");
        }
        for i in 0..p {
            assert!((xty[i] - xty_ref[i]).abs() < 1e-12, "xty[{i}]");
        }
    }
}

#[cfg(test)]
mod kernel_regressions {
    use antecedent_core::ExecutionContext;

    use super::{
        MAX_NONPARAMETRIC_ROWS, nw_loo_predict_pair, orthogonalize_confounder, silverman_bandwidth,
        standardize_columns,
    };
    use crate::common::sample_sd;

    fn predict(
        y1: &[f64],
        y2: &[f64],
        x: &[f64],
        dim: usize,
        h: f64,
        ctx: &ExecutionContext,
    ) -> (Vec<f64>, Vec<f64>) {
        nw_loo_predict_pair(y1, y2, x, dim, h, ctx).unwrap()
    }

    #[test]
    fn leave_one_out_does_not_leak_target_when_weights_underflow() {
        let ctx = ExecutionContext::for_tests(1);
        let (first, second) =
            predict(&[100.0, 2.0, 4.0], &[50.0, 10.0, 20.0], &[0.0, 1.0, 3.0], 1, 1e-6, &ctx);
        assert_eq!(first, vec![2.0, 100.0, 2.0]);
        assert_eq!(second, vec![10.0, 50.0, 10.0]);
    }

    #[test]
    fn leave_one_out_preserves_nearest_neighbor_ties() {
        let ctx = ExecutionContext::for_tests(1);
        let (first, _) = predict(&[2.0, 100.0, 4.0], &[0.0; 3], &[-1.0, 0.0, 1.0], 1, 1e-6, &ctx);
        assert!((first[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn normalized_kernel_matches_direct_gaussian_weights() {
        let ctx = ExecutionContext::for_tests(1);
        let y = [1.0, 4.0, 9.0];
        let x = [0.0, 1.0, 2.0];
        let (prediction, _) = predict(&y, &y, &x, 1, 1.0, &ctx);
        let w1 = (-0.5_f64).exp();
        let w2 = (-2.0_f64).exp();
        assert!((prediction[0] - (w1 * 4.0 + w2 * 9.0) / (w1 + w2)).abs() < 1e-12);
        assert!((prediction[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn parallel_and_serial_smoothers_agree_bitwise() {
        // Row-independent work returned in row order: the thread count cannot change a value.
        let n = 40;
        let x: Vec<f64> = (0..n).flat_map(|i| [f64::from(i) * 0.37, f64::from(i % 7)]).collect();
        let y1: Vec<f64> = (0..n).map(|i| f64::from(i * i % 11)).collect();
        let y2: Vec<f64> = (0..n).map(|i| f64::from(i) - 3.0).collect();
        let serial = predict(&y1, &y2, &x, 2, 0.8, &ExecutionContext::for_tests(1));
        let pooled = predict(&y1, &y2, &x, 2, 0.8, &ExecutionContext::production(1, 4));
        assert_eq!(serial, pooled);
    }

    #[test]
    fn squared_distance_kernel_matches_the_gaussian_weights_in_two_dimensions() {
        // Three points in the plane, h = 2: row 0 sees (3, 4) at distance 5 and (0, 1) at
        // distance 1; weights w = exp(-d² / (2h²)) = exp(-25/8) and exp(-1/8).
        let ctx = ExecutionContext::for_tests(1);
        let x = [0.0, 0.0, 3.0, 4.0, 0.0, 1.0];
        let y = [0.0, 10.0, 20.0];
        let (prediction, _) = predict(&y, &y, &x, 2, 2.0, &ctx);
        let (wa, wb) = ((-25.0_f64 / 8.0).exp(), (-1.0_f64 / 8.0).exp());
        assert!((prediction[0] - (wa * 10.0 + wb * 20.0) / (wa + wb)).abs() < 1e-12);
    }

    #[test]
    fn standardization_gives_an_age_like_and_a_binary_covariate_the_same_scale() {
        // Column 0 has SD ~ 15 (age-like), column 1 is 0/1, column 2 is constant. After
        // standardization the first two have unit SD, so one bandwidth treats them alike; the
        // constant column is left alone rather than divided by zero.
        let n = 200_usize;
        let dim = 3;
        let mut cov = Vec::new();
        for i in 0..n {
            cov.push(20.0 + 60.0 * i as f64 / n as f64);
            cov.push((i % 2) as f64);
            cov.push(4.0);
        }
        standardize_columns(&mut cov, n, dim);
        for d in 0..2 {
            let column: Vec<f64> = (0..n).map(|r| cov[r * dim + d]).collect();
            assert!((sample_sd(&column) - 1.0).abs() < 1e-12, "column {d}");
        }
        assert!((0..n).all(|r| cov[r * dim + 2] == 4.0));
    }

    #[test]
    fn silverman_rule_matches_the_multivariate_normal_reference_formula() {
        // d = 1: (4/3)^(1/5) n^(-1/5); d = 2: 1^(1/6) n^(-1/6).
        let n = 1000_usize;
        assert!(
            (silverman_bandwidth(n, 1) - (4.0_f64 / 3.0).powf(0.2) * 1000.0_f64.powf(-0.2)).abs()
                < 1e-12
        );
        assert!((silverman_bandwidth(n, 2) - 1000.0_f64.powf(-1.0 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn the_row_cap_bounds_the_quadratic_kernel() {
        assert_eq!(MAX_NONPARAMETRIC_ROWS, 20_000);
    }

    #[test]
    fn orthogonalized_confounder_is_uncorrelated_with_the_design_and_unit_variance() {
        // Design columns t, y over 6 complete rows of 8 (rows 2 and 5 masked out).
        let mask = [true, true, false, true, true, false, true, true];
        let t = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let y = vec![0.3, 1.9, 0.2, 2.4, 1.6, 0.7];
        let mut u = vec![0.5, -1.2, 9.0, 0.7, 1.1, -9.0, -0.4, 0.9];
        orthogonalize_confounder(&mut u, &mask, &[t.clone(), y.clone()]);
        let rows: Vec<usize> = (0..8).filter(|&i| mask[i]).collect();
        let um: Vec<f64> = rows.iter().map(|&i| u[i]).collect();
        assert_eq!(u[2], 0.0);
        assert_eq!(u[5], 0.0);
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        assert!(um.iter().sum::<f64>().abs() < 1e-12, "not centred");
        assert!(dot(&um, &t).abs() < 1e-12, "correlated with t");
        assert!(dot(&um, &y).abs() < 1e-12, "correlated with y");
        assert!((sample_sd(&um) - 1.0).abs() < 1e-12, "not unit SD");
    }
}

#[cfg(test)]
mod robustness_value {
    use std::sync::Arc;

    use antecedent_core::{
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
    use antecedent_expr::ExprId;
    use antecedent_identify::IdentifiedEstimand;

    use super::{
        LinearSensitivity, NonparametricSensitivity, PartialLinearSensitivity,
        grid_robustness_value, robustness_passes,
    };
    use crate::common::RefutationProblem;

    fn strong_effect_toy() -> (TabularData, IdentifiedEstimand) {
        let n = 300usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
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
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + 0.5 * z[i]).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    fn problem_with_ate(
        data: &TabularData,
        estimand: &IdentifiedEstimand,
    ) -> (
        EstimationWorkspace,
        ExecutionContext,
        AverageEffectQuery,
        antecedent_estimate::EffectEstimate,
    ) {
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let prep = est.prepare(data, estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(11);
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        (ws, ctx, query, original)
    }

    #[test]
    fn never_explained_away_is_infinite_not_last_grid_point() {
        assert!(grid_robustness_value(None).is_infinite());
        assert_eq!(grid_robustness_value(Some(0.2)), 0.2);
        assert!(!robustness_passes(0.1, 0.1));
        assert!(robustness_passes(0.2, 0.1));
        assert!(robustness_passes(f64::INFINITY, 0.5));
    }

    #[test]
    fn linear_sensitivity_equality_at_threshold_fails() {
        let (data, estimand) = strong_effect_toy();
        let (mut ws, ctx, query, original) = problem_with_ate(&data, &estimand);
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let grid = vec![0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99];
        let probe = LinearSensitivity {
            partial_r2_grid: grid.clone(),
            pass_threshold: 0.0,
            estimator: LinearAdjustmentAte {
                bootstrap_replicates: 0,
                ..LinearAdjustmentAte::new()
            },
        };
        let tipped = probe.refute(&problem, &mut ws, &ctx).unwrap();
        assert!(
            tipped.comparison.is_finite() && tipped.comparison > 0.0,
            "expected a finite tipping partial R², got {}",
            tipped.comparison
        );
        let rv = tipped.comparison;

        let at_bar = LinearSensitivity {
            partial_r2_grid: grid,
            pass_threshold: rv,
            estimator: probe.estimator.clone(),
        };
        let equal = at_bar.refute(&problem, &mut ws, &ctx).unwrap();
        assert_eq!(equal.comparison, rv);
        assert!(!equal.passed, "RV == threshold must fail (effect already gone at the bar)");
    }

    #[test]
    fn linear_sensitivity_explained_away_past_threshold_passes() {
        let (data, estimand) = strong_effect_toy();
        let (mut ws, ctx, query, original) = problem_with_ate(&data, &estimand);
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let grid = vec![0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99];
        let probe = LinearSensitivity {
            partial_r2_grid: grid.clone(),
            pass_threshold: 0.0,
            estimator: LinearAdjustmentAte {
                bootstrap_replicates: 0,
                ..LinearAdjustmentAte::new()
            },
        };
        let tipped = probe.refute(&problem, &mut ws, &ctx).unwrap();
        let rv = tipped.comparison;
        assert!(rv.is_finite() && rv > 0.0, "comparison={rv}");

        let idx = grid.iter().position(|&r| r == rv).expect("RV must be a grid point");
        assert!(idx > 0, "need a grid point before the tipping RV to set a lower bar");
        let threshold = grid[idx - 1];
        let below = LinearSensitivity {
            partial_r2_grid: grid,
            pass_threshold: threshold,
            estimator: probe.estimator.clone(),
        };
        let past = below.refute(&problem, &mut ws, &ctx).unwrap();
        assert_eq!(past.comparison, rv);
        assert!(past.passed, "RV {rv} > threshold {threshold} must pass");
    }

    #[test]
    fn partial_linear_equality_at_threshold_fails_past_passes() {
        let (data, estimand) = strong_effect_toy();
        let (mut ws, ctx, query, original) = problem_with_ate(&data, &estimand);
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let grid = vec![0.01, 0.05, 0.1, 0.2, 0.5];
        let estimator =
            LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let tip = PartialLinearSensitivity {
            partial_r2_grid: grid.clone(),
            pass_threshold: 0.0,
            estimator: estimator.clone(),
        }
        .refute(&problem, &mut ws, &ctx)
        .unwrap();
        let rv = tip.comparison;
        assert!(rv.is_finite() && rv > 0.0, "comparison={rv}");

        let equal = PartialLinearSensitivity {
            partial_r2_grid: grid.clone(),
            pass_threshold: rv,
            estimator: estimator.clone(),
        }
        .refute(&problem, &mut ws, &ctx)
        .unwrap();
        assert!(!equal.passed);

        let past =
            PartialLinearSensitivity { partial_r2_grid: grid, pass_threshold: rv * 0.5, estimator }
                .refute(&problem, &mut ws, &ctx)
                .unwrap();
        assert!(past.passed);
    }

    #[test]
    fn nonparametric_equality_at_threshold_fails_past_passes() {
        let (data, estimand) = strong_effect_toy();
        let (mut ws, ctx, query, original) = problem_with_ate(&data, &estimand);
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let grid = vec![0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 0.99];
        let tip = NonparametricSensitivity {
            partial_r2_grid: grid.clone(),
            pass_threshold: 0.0,
            bandwidth: Some(0.5),
        }
        .refute(&problem, &mut ws, &ctx)
        .unwrap();
        let rv = tip.comparison;
        assert!(rv.is_finite() && rv > 0.0, "comparison={rv}");

        let equal = NonparametricSensitivity {
            partial_r2_grid: grid.clone(),
            pass_threshold: rv,
            bandwidth: Some(0.5),
        }
        .refute(&problem, &mut ws, &ctx)
        .unwrap();
        assert!(!equal.passed);

        let past = NonparametricSensitivity {
            partial_r2_grid: grid,
            pass_threshold: rv * 0.5,
            bandwidth: Some(0.5),
        }
        .refute(&problem, &mut ws, &ctx)
        .unwrap();
        assert!(past.passed);
    }

    #[test]
    fn never_explained_away_on_tiny_grid_reports_infinity() {
        let (data, estimand) = strong_effect_toy();
        let (mut ws, ctx, query, original) = problem_with_ate(&data, &estimand);
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let refuter = LinearSensitivity {
            partial_r2_grid: vec![1e-6],
            pass_threshold: 0.1,
            estimator: LinearAdjustmentAte {
                bootstrap_replicates: 0,
                ..LinearAdjustmentAte::new()
            },
        };
        let report = refuter.refute(&problem, &mut ws, &ctx).unwrap();
        assert!(
            report.comparison.is_infinite(),
            "surviving the grid must report +∞, got {}",
            report.comparison
        );
        assert!(report.passed);
        assert!(report.failure_condition.is_none());
    }
}

#[cfg(test)]
mod tipping_closed_form {
    use super::{LinearSensitivity, NonparametricSensitivity, PartialLinearSensitivity};
    use crate::common::RefutationProblem;
    use crate::test_support::{ate_query, backdoor, linear_original, tabular};

    /// Residual of `v` on `[1, z]` (simple regression), computed independently of the crate.
    fn residual_on(v: &[f64], z: &[f64]) -> Vec<f64> {
        let n = v.len() as f64;
        let (mv, mz) = (v.iter().sum::<f64>() / n, z.iter().sum::<f64>() / n);
        let slope = v.iter().zip(z).map(|(a, b)| (a - mv) * (b - mz)).sum::<f64>()
            / z.iter().map(|b| (b - mz) * (b - mz)).sum::<f64>();
        v.iter().zip(z).map(|(a, b)| a - mv - slope * (b - mz)).collect()
    }

    /// `(t, y, z)` with a treatment that is partly explained by `z` and an outcome with
    /// deterministic wiggle, so the partial correlation given `z` is neither 0 nor 1.
    fn design() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = 400_usize;
        let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let w = if (i / 2) % 2 == 0 { 1.0 } else { -1.0 };
                0.6 * t[i] + 0.7 * z[i] + 0.5 * w + 0.3 * (i % 3) as f64
            })
            .collect();
        (t, y, z)
    }

    /// Analytic tipping partial R²: `|ρ| / (1 + |ρ|)`, `ρ` the partial correlation of `t` and
    /// `y` given `z`.
    fn analytic_tipping(t: &[f64], y: &[f64], z: &[f64]) -> f64 {
        let (rt, ry) = (residual_on(t, z), residual_on(y, z));
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let rho = (dot(&rt, &ry) / (dot(&rt, &rt) * dot(&ry, &ry)).sqrt()).abs();
        rho / (1.0 + rho)
    }

    fn assert_brackets(reported: f64, analytic: f64, step: f64) {
        // The report is the first grid value at or beyond the true tipping point.
        assert!(
            reported >= analytic - 1e-9 && reported - step < analytic + 1e-9,
            "reported {reported} must be the first {step}-grid value >= analytic {analytic}"
        );
    }

    #[test]
    fn linear_and_partial_linear_tipping_match_the_partial_correlation_closed_form() {
        let (t, y, z) = design();
        let analytic = analytic_tipping(&t, &y, &z);
        let data = tabular(&[t, y, z]);
        let estimand = backdoor(1);
        let query = ate_query();
        let (original, mut ws, ctx) = linear_original(&data, &estimand, &query);
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let grid: Vec<f64> = (1..=99).map(|k| f64::from(k) / 100.0).collect();
        let linear = LinearSensitivity {
            partial_r2_grid: grid.clone(),
            pass_threshold: 0.0,
            ..LinearSensitivity::new()
        }
        .refute(&problem, &mut ws, &ctx)
        .unwrap();
        assert_brackets(linear.comparison, analytic, 0.01);
        let partial = PartialLinearSensitivity {
            partial_r2_grid: grid,
            pass_threshold: 0.0,
            ..PartialLinearSensitivity::new()
        }
        .refute(&problem, &mut ws, &ctx)
        .unwrap();
        // Same design, same confounder scale: the OLS tipping point does not depend on the draw.
        assert_eq!(partial.comparison, linear.comparison);
    }

    #[test]
    fn tipping_point_does_not_depend_on_the_confounder_seed() {
        let (t, y, z) = design();
        let data = tabular(&[t, y, z]);
        let estimand = backdoor(1);
        let query = ate_query();
        let (original, mut ws, _) = linear_original(&data, &estimand, &query);
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let grid: Vec<f64> = (1..=99).map(|k| f64::from(k) / 100.0).collect();
        let refuter = LinearSensitivity {
            partial_r2_grid: grid,
            pass_threshold: 0.0,
            ..LinearSensitivity::new()
        };
        let values: Vec<f64> = [1_u64, 2, 3, 99]
            .iter()
            .map(|&seed| {
                let ctx = antecedent_core::ExecutionContext::for_tests(seed);
                refuter.refute(&problem, &mut ws, &ctx).unwrap().comparison
            })
            .collect();
        assert!(values.iter().all(|&v| v == values[0]), "tipping value varies by seed: {values:?}");
    }

    #[test]
    fn nonparametric_sensitivity_refuses_an_opposite_sign_residual_effect() {
        // Published estimate positive; residual slope of y on t (after smoothing z) negative.
        let n = 200_usize;
        let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| -1.0 * t[i] + 0.2 * z[i]).collect();
        let data = tabular(&[t, y, z]);
        let estimand = backdoor(1);
        let query = ate_query();
        let (mut original, mut ws, ctx) = linear_original(&data, &estimand, &query);
        original.ate = 1.0;
        let problem = RefutationProblem::new(&data, &estimand, &query, &original, None, None);
        let err = NonparametricSensitivity::new().refute(&problem, &mut ws, &ctx).unwrap_err();
        assert!(matches!(err, crate::ValidationError::NotApplicable { .. }), "{err:?}");
    }
}
