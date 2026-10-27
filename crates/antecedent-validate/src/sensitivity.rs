//! Linear, partial-linear, and nonparametric confounding sensitivity analysis.
//!
//! [`LinearSensitivity`] and [`PartialLinearSensitivity`] simulate a confounder `U` with a
//! configurable *partial R²* on treatment and outcome under a linear (Gaussian) or
//! partial-linear (bounded) shape. [`NonparametricSensitivity`] first residualizes treatment
//! and outcome on adjustment covariates with Nadaraya–Watson (Nadaraya 1964; Watson 1964) kernel
//! regression, then runs the
//! same partial-R² grid on the residualized series — a production nonparametric path distinct
//! from the partial-linear shape stand-in. That smoother is quadratic in the sample size, so
//! above [`MAX_NONPARAMETRIC_ROWS`] complete cases [`NonparametricSensitivity`] runs
//! [`NeighborhoodSensitivity`] (the Gaussian kernel truncated to a declared number of nearest
//! neighbours, found with an exact k-d tree) and says so: the report's refuter is then
//! `sensitivity.nonparametric_neighborhood`, with that construction's own confounder stream,
//! never `sensitivity.nonparametric`.
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

/// Pass only when a probed strength at or above the caller's bar left the effect standing.
///
/// The true tipping point lies above the largest probed grid value that did *not* explain
/// the effect away (every grid value below `robustness_value` — the first one that did).
/// The verdict is sound only when that cleared value reaches the bar: comparing
/// `robustness_value` itself with the bar would pass a threshold that falls strictly
/// between two grid points although a confounder just above the lower point may already
/// kill the effect, and would pass an untipped grid whose largest point is below the bar.
/// When the bar is a grid point this equals `robustness_value > pass_threshold`, so
/// equality at the bar fails.
pub(crate) fn robustness_passes(
    probed: impl IntoIterator<Item = f64>,
    robustness_value: f64,
    pass_threshold: f64,
) -> bool {
    probed
        .into_iter()
        .filter(|&g| g < robustness_value)
        .fold(None, |cleared: Option<f64>, g| Some(cleared.map_or(g, |c| c.max(g))))
        .is_some_and(|cleared| cleared >= pass_threshold)
}

/// Failure text for a grid verdict that did not pass, naming why: the effect tipped at or
/// below the bar, it tipped above an off-grid bar the grid could not certify, or it never
/// tipped but the grid stopped short of the bar.
pub(crate) fn grid_failure_text(what: &str, axis: &str, value: f64, threshold: f64) -> String {
    if value.is_infinite() {
        format!(
            "{what} never explained away on the probed grid, but its largest {axis} is below \
             threshold {threshold}"
        )
    } else if value > threshold {
        format!(
            "{what} explained away at {axis}={value}; the largest probed {axis} that left it \
             standing is below threshold {threshold}"
        )
    } else {
        format!("{what} explained away at {axis}={value}, not strictly above threshold {threshold}")
    }
}

/// Grid verdict as a refutation report.
///
/// `informative` is always true. `failure` is rendered only when `passed` is false, and its
/// text is stored unchanged. `comparison` is the robustness value the grid just computed.
pub(crate) fn grid_report(
    refuter: &str,
    original_ate: f64,
    refuted_ate: f64,
    comparison: f64,
    passed: bool,
    failure: impl FnOnce() -> String,
    replicates: u32,
) -> RefutationReport {
    let failure_condition = if passed { None } else { Some(Arc::from(failure())) };
    RefutationReport::builder(refuter)
        .original_ate(original_ate)
        .refuted_ate(refuted_ate)
        .comparison(comparison)
        .informative(true)
        .passed(passed)
        .failure_condition(failure_condition)
        .replicates(replicates)
        .build()
}

/// Grid values as the sensitivity loops probe them (`r` clamped to `[0, 0.999]`).
fn probed_partial_r2(grid: &[f64]) -> impl Iterator<Item = f64> + '_ {
    grid.iter().map(|r| r.clamp(0.0, 0.999))
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
        && estimator.inference.bootstrap_replicates == 0
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
        let ids = problem.estimation_ids(problem.temporal.is_none());
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
        sorted_grid.sort_by(f64::total_cmp);

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
    /// Pass only if a probed partial R² at or above this threshold left the effect standing
    /// (for a threshold on the grid: the robustness value *strictly exceeds* it). Equality
    /// fails: a confounder at the bar already kills the effect.
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
            crate::streams::LINEAR_SENSITIVITY,
            false,
        )?;
        let passed = robustness_passes(
            probed_partial_r2(&self.partial_r2_grid),
            robustness_value,
            self.pass_threshold,
        );
        Ok(grid_report(
            "sensitivity.linear",
            problem.original.ate,
            refuted_ate,
            robustness_value,
            passed,
            || grid_failure_text("effect", "partial R²", robustness_value, self.pass_threshold),
            self.partial_r2_grid.len() as u32,
        ))
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
    /// Pass only if a probed partial R² at or above this threshold left the effect standing
    /// (for a threshold on the grid: the robustness value *strictly exceeds* it). Equality
    /// fails: a confounder at the bar already kills the effect.
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
            crate::streams::PARTIAL_LINEAR_SENSITIVITY,
            true,
        )?;
        let passed = robustness_passes(
            probed_partial_r2(&self.partial_r2_grid),
            robustness_value,
            self.pass_threshold,
        );
        Ok(grid_report(
            "sensitivity.partial_linear",
            problem.original.ate,
            refuted_ate,
            robustness_value,
            passed,
            || grid_failure_text("effect", "partial R²", robustness_value, self.pass_threshold),
            self.partial_r2_grid.len() as u32,
        ))
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
/// standardizes). Each row stores those squared distances once and reuses the same `f64` values
/// for the minimum and the weights; the weight shift `exp(−½ (d² − min d²) / h²)` keeps the
/// nearest observations at weight 1 even for a tiny bandwidth.
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
        // Slot `i` is unused; every other slot is the squared distance to that row.
        let mut dist2 = vec![0.0; n];
        let mut nearest = f64::INFINITY;
        for (j, slot) in dist2.iter_mut().enumerate() {
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
        for (j, &d2) in dist2.iter().enumerate() {
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

/// Cells of identical covariate rows: row indices grouped by cell (`rows`), with
/// `start[c]..start[c + 1]` the slice of cell `c`.
struct CovariateCells {
    rows: Vec<usize>,
    start: Vec<usize>,
}

impl CovariateCells {
    /// Group the rows of row-major `cov` by exact (total-order) equality, rows ascending
    /// within a cell and cells in lexicographic order of their covariates.
    fn new(cov: &[f64], n: usize, dim: usize) -> Self {
        let row = |i: usize| &cov[i * dim..(i + 1) * dim];
        let compare = |a: usize, b: usize| {
            row(a)
                .iter()
                .zip(row(b))
                .map(|(x, y)| x.total_cmp(y))
                .find(|order| order.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        let mut rows: Vec<usize> = (0..n).collect();
        rows.sort_by(|&a, &b| compare(a, b).then(a.cmp(&b)));
        let mut start: Vec<usize> = Vec::new();
        for pos in 0..n {
            if pos == 0 || compare(rows[pos - 1], rows[pos]).is_ne() {
                start.push(pos);
            }
        }
        start.push(n);
        Self { rows, start }
    }

    fn len(&self) -> usize {
        self.start.len() - 1
    }

    fn rows_of(&self, cell: usize) -> &[usize] {
        &self.rows[self.start[cell]..self.start[cell + 1]]
    }
}

/// Cells whose neighbour lists one worker handles per task.
const NEIGHBORHOOD_CELL_CHUNK: usize = 256;

/// Neighbourhood-truncated leave-one-out Nadaraya–Watson prediction of two targets sharing one
/// covariate matrix (`n × dim`, row-major), the scalable counterpart of [`nw_loo_predict_pair`].
///
/// The weight of a neighbour is `exp(−½ (d² − d₀²) / h²)` with `d₀` the distance to the nearest
/// contributing point, as in the exact smoother; only the `neighbors` nearest other covariate
/// points enter the sums (rows with identical covariates are one point carrying their count and
/// sums, and a row's own point contributes its other rows). Every row is independent given the
/// shared neighbour search, so the output does not depend on the thread count.
fn neighborhood_loo_predict_pair(
    y1: &[f64],
    y2: &[f64],
    cov_rowmajor: &[f64],
    dim: usize,
    bandwidth: f64,
    neighbors: usize,
    ctx: &ExecutionContext,
) -> Result<(Vec<f64>, Vec<f64>), ValidationError> {
    let n = y1.len();
    let word = core::mem::size_of::<f64>();
    let required = n
        .checked_mul(dim.saturating_add(14))
        .and_then(|words| words.checked_mul(word))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .unwrap_or(u64::MAX);
    if let Some(limit_bytes) = ctx.memory_limit_bytes() {
        if required > limit_bytes {
            return Err(ValidationError::Stats(antecedent_stats::StatsError::ResourceLimit {
                message: "neighborhood sensitivity smoother",
                required_bytes: required,
                limit_bytes,
            }));
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(ValidationError::Cancelled);
    }
    let cells = CovariateCells::new(cov_rowmajor, n, dim);
    let n_cells = cells.len();
    let mut cell_features = Vec::with_capacity(n_cells * dim);
    let mut count = Vec::with_capacity(n_cells);
    let mut sum1 = Vec::with_capacity(n_cells);
    let mut sum2 = Vec::with_capacity(n_cells);
    for c in 0..n_cells {
        let rows = cells.rows_of(c);
        cell_features.extend_from_slice(&cov_rowmajor[rows[0] * dim..(rows[0] + 1) * dim]);
        count.push(rows.len() as f64);
        sum1.push(rows.iter().map(|&i| y1[i]).sum::<f64>());
        sum2.push(rows.iter().map(|&i| y2[i]).sum::<f64>());
    }
    let cell_ids: Vec<usize> = (0..n_cells).collect();
    let index = antecedent_stats::MatchingIndex::exact_within(
        &cell_features,
        dim,
        &cell_ids,
        antecedent_stats::MatchingDistance::Euclidean,
        ctx,
    )?;
    // One more than `neighbors`: a cell's own point is among its nearest and may contribute
    // nothing (a single row), which must not cost a neighbour slot.
    let k = neighbors.saturating_add(1);
    let h2 = bandwidth * bandwidth;
    let chunks = n_cells.div_ceil(NEIGHBORHOOD_CELL_CHUNK);
    let parts = ctx.map_indexed(chunks, |chunk, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(ValidationError::Cancelled);
        }
        let first = chunk * NEIGHBORHOOD_CELL_CHUNK;
        let last = (first + NEIGHBORHOOD_CELL_CHUNK).min(n_cells);
        let mut out: Vec<(usize, f64, f64)> = Vec::new();
        let mut found: Vec<(usize, f64)> = Vec::new();
        for c in first..last {
            index.nearest_k(&cell_features[c * dim..(c + 1) * dim], k, &mut found)?;
            if !found.iter().any(|&(u, _)| u == c) {
                found.push((c, 0.0));
            }
            for &i in cells.rows_of(c) {
                // This row leaves its own cell's sums.
                let others = |u: usize| count[u] - f64::from(u8::from(u == c));
                let nearest = found
                    .iter()
                    .filter(|&&(u, _)| others(u) > 0.0)
                    .map(|&(_, d)| d * d)
                    .fold(f64::INFINITY, f64::min);
                let (mut num1, mut num2, mut den) = (0.0, 0.0, 0.0);
                for &(u, d) in &found {
                    let m = others(u);
                    if m <= 0.0 {
                        continue;
                    }
                    let d2 = d * d;
                    let w = if d2.total_cmp(&nearest).is_eq() {
                        1.0
                    } else {
                        (-0.5 * (d2 - nearest) / h2).exp()
                    };
                    let own = u == c;
                    num1 += w * (sum1[u] - if own { y1[i] } else { 0.0 });
                    num2 += w * (sum2[u] - if own { y2[i] } else { 0.0 });
                    den += w * m;
                }
                if den.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                    return Err(ValidationError::NotApplicable {
                        message: "neighborhood sensitivity found no other row to smooth over",
                    });
                }
                out.push((i, num1 / den, num2 / den));
            }
        }
        Ok(out)
    })?;
    let mut hat1 = vec![0.0; n];
    let mut hat2 = vec![0.0; n];
    for (i, a, b) in parts.into_iter().flatten() {
        hat1[i] = a;
        hat2[i] = b;
    }
    Ok((hat1, hat2))
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
    let mut pred = vec![0.0; n];
    crate::common::column_major_dot(design, n, &fit.coefficients, &mut pred);
    let residuals: Vec<f64> = y.iter().zip(&pred).map(|(y, pred)| y - pred).collect();
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

/// Largest complete-case sample the O(n²·d) leave-one-out kernel smoother runs on; above it
/// [`NonparametricSensitivity`] routes to the neighbourhood construction.
const MAX_NONPARAMETRIC_ROWS: usize = 20_000;

/// Nonparametric sensitivity: kernel-residualize T and Y on Z, then partial-R² grid on residuals.
#[derive(Clone, Debug)]
pub struct NonparametricSensitivity {
    /// Ascending grid of partial-R² values to test on residualized series.
    pub partial_r2_grid: Vec<f64>,
    /// Pass only if a probed partial R² at or above this threshold left the effect standing
    /// (for a threshold on the grid: the robustness value *strictly exceeds* it). Equality
    /// fails: a confounder at the bar already kills the effect.
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
    #[allow(
        clippy::only_used_in_recursion,
        reason = "part of the shared refuter signature; the temporal branch re-enters with the aligned problem"
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
        let (residuals, routed) = kernel_residuals(
            problem,
            self.bandwidth,
            ctx,
            ResidualSmoother::ExactUpTo { neighbors_beyond: DEFAULT_NEIGHBORHOOD },
        )?;
        let (refuter, stream) = if routed {
            ("sensitivity.nonparametric_neighborhood", crate::streams::NEIGHBORHOOD_SENSITIVITY)
        } else {
            ("sensitivity.nonparametric", crate::streams::NONPARAMETRIC_SENSITIVITY)
        };
        residual_grid_report(
            problem,
            ctx,
            residuals,
            &ResidualGrid {
                partial_r2_grid: &self.partial_r2_grid,
                pass_threshold: self.pass_threshold,
                refuter,
                stream,
            },
        )
    }
}

/// Grid, threshold, report id and confounder stream of one residual-grid sensitivity run.
struct ResidualGrid<'a> {
    partial_r2_grid: &'a [f64],
    pass_threshold: f64,
    refuter: &'static str,
    stream: u64,
}

/// The partial-R² grid on the residualized treatment and outcome (`n` complete cases): the
/// part of the nonparametric refuters that does not depend on the smoother.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the refutation reports its grid size as u32, and a partial-R-squared grid is far below 2^32 points"
)]
fn residual_grid_report(
    problem: &RefutationProblem<'_>,
    ctx: &ExecutionContext,
    residuals: (Vec<f64>, Vec<f64>, usize),
    spec: &ResidualGrid<'_>,
) -> Result<RefutationReport, ValidationError> {
    let (t_res, y_res, n) = residuals;
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
    fill_gaussian(&mut u, ctx, spec.stream);
    orthogonalize_confounder(&mut u, &vec![true; n], &[t_res.clone(), y_res.clone()]);

    let mut sorted_grid = spec.partial_r2_grid.to_vec();
    sorted_grid.sort_by(f64::total_cmp);
    let original_sign = residual_ate.signum();
    // Worst-case orientation, as in `run_grid`: load U on Y against the observed sign.
    let dir = if residual_ate >= 0.0 { -1.0 } else { 1.0 };
    let moments = PerturbMoments::new(&t_res, &y_res, &u);
    let mut last_ate = residual_ate;
    let mut explained_away_at = None;
    for &r in &sorted_grid {
        let r = r.clamp(0.0, 0.999);
        let scale = (r / (1.0 - r)).sqrt();
        // `t' = t + a u`, `y' = y + b u`. The OLS slope is a function of the
        // five raw second moments, so the grid does not rebuild the series.
        last_ate = moments.slope(scale * sd_t, dir * scale * sd_y);
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
    let passed = robustness_passes(
        probed_partial_r2(spec.partial_r2_grid),
        robustness_value,
        spec.pass_threshold,
    );
    Ok(grid_report(
        spec.refuter,
        problem.original.ate,
        last_ate * treatment_delta,
        robustness_value,
        passed,
        || {
            grid_failure_text(
                "nonparametric residual effect",
                "partial R²",
                robustness_value,
                spec.pass_threshold,
            )
        },
        spec.partial_r2_grid.len() as u32,
    ))
}

/// Nearest neighbours the [`NeighborhoodSensitivity`] kernel sums run over by default.
const DEFAULT_NEIGHBORHOOD: usize = 256;

/// Largest `rows x neighbors` kernel-sum work [`NeighborhoodSensitivity`] accepts. The
/// neighbour count has no separate bound: it is the same knob as the exact smoother's reach,
/// and a request whose product exceeds this budget is better run with fewer neighbours.
const MAX_NEIGHBORHOOD_WORK: u64 = 4_000_000_000;

/// Nonparametric sensitivity for samples beyond [`MAX_NONPARAMETRIC_ROWS`]: kernel-residualize
/// T and Y on Z with a nearest-neighbour-truncated Gaussian kernel, then the same partial-R²
/// grid on the residuals as [`NonparametricSensitivity`].
///
/// [`NonparametricSensitivity`] sums the Gaussian kernel over every other row, which is
/// `O(n² · d)`. This is a separate construction with its own identity
/// (`sensitivity.nonparametric_neighborhood`, its own confounder stream): the leave-one-out
/// Nadaraya–Watson prediction of row `i` sums the same product-Gaussian weights
/// `exp(-‖z_i − z_j‖² / 2h²)` (covariates standardized to unit SD, Silverman's bandwidth or the
/// caller's) over only the [`Self::neighbors`] nearest *other* covariate points, found with an
/// exact k-d tree search. Rows with identical covariates form one point carrying their count and
/// sums, so discrete adjustment sets stay exact and a tie group never competes for neighbour
/// slots by row order; the row's own point contributes its other rows. Time is
/// `O(n · (log n + neighbors))` for low-dimensional covariates and memory `O(n · d)`.
///
/// It equals the exact smoother whenever the neighbours cover every row (`neighbors >= n - 1`),
/// and approximates it when the kernel's mass lies within the neighbours. When `h` is wide
/// enough that distant rows still carry weight, the truncation is a bias the exact smoother does
/// not have, in the direction of a locally weighted neighbour average; a larger `neighbors`
/// shrinks it.
#[derive(Clone, Debug)]
pub struct NeighborhoodSensitivity {
    /// Ascending grid of partial-R² values to test on residualized series.
    pub partial_r2_grid: Vec<f64>,
    /// Pass only if a probed partial R² at or above this threshold left the effect standing
    /// (for a threshold on the grid: the robustness value *strictly exceeds* it). Equality
    /// fails: a confounder at the bar already kills the effect.
    pub pass_threshold: f64,
    /// Optional bandwidth override, in standard deviations of each (standardized) covariate;
    /// `None` uses Silverman's (1986) multivariate normal-reference rule.
    pub bandwidth: Option<f64>,
    /// Nearest other covariate points each kernel sum runs over (1 to 4096).
    pub neighbors: usize,
}

impl Default for NeighborhoodSensitivity {
    fn default() -> Self {
        Self::new()
    }
}

impl NeighborhoodSensitivity {
    /// Defaults: same partial-R² grid as linear sensitivity, pass threshold 0.1, 256 neighbours.
    #[must_use]
    pub fn new() -> Self {
        Self {
            partial_r2_grid: default_grid(),
            pass_threshold: 0.1,
            bandwidth: None,
            neighbors: DEFAULT_NEIGHBORHOOD,
        }
    }

    /// Run neighbourhood nonparametric sensitivity.
    ///
    /// # Errors
    ///
    /// Data failures, an empty `partial_r2_grid`, zero `neighbors`, cancellation,
    /// or a footprint above the context's hard memory limit.
    #[allow(
        clippy::only_used_in_recursion,
        reason = "part of the shared refuter signature; the temporal branch re-enters with the aligned problem"
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
                message: "neighborhood sensitivity requires a non-empty partial_r2_grid",
            });
        }
        if self.neighbors == 0 {
            return Err(ValidationError::NotApplicable {
                message: "neighborhood sensitivity requires at least one neighbor",
            });
        }
        let (residuals, _) = kernel_residuals(
            problem,
            self.bandwidth,
            ctx,
            ResidualSmoother::Neighborhood { neighbors: self.neighbors },
        )?;
        residual_grid_report(
            problem,
            ctx,
            residuals,
            &ResidualGrid {
                partial_r2_grid: &self.partial_r2_grid,
                pass_threshold: self.pass_threshold,
                refuter: "sensitivity.nonparametric_neighborhood",
                stream: crate::streams::NEIGHBORHOOD_SENSITIVITY,
            },
        )
    }
}

/// Which leave-one-out smoother residualizes treatment and outcome.
#[derive(Clone, Copy)]
enum ResidualSmoother {
    /// Every other row enters the kernel sums (quadratic in the sample size) up to
    /// [`MAX_NONPARAMETRIC_ROWS`] complete cases; a larger sample runs the neighbourhood
    /// smoother over `neighbors_beyond` neighbours instead.
    ExactUpTo { neighbors_beyond: usize },
    /// Only the `neighbors` nearest other covariate rows enter them.
    Neighborhood { neighbors: usize },
}

/// Leave-one-out Nadaraya–Watson residuals of treatment and outcome on the adjustment
/// covariates, over the complete cases, with the complete-case row count and whether the
/// neighbourhood smoother ran in place of the exact one.
#[allow(
    clippy::type_complexity,
    reason = "residuals, row count and the route flag are returned together"
)]
fn kernel_residuals(
    problem: &RefutationProblem<'_>,
    bandwidth: Option<f64>,
    ctx: &ExecutionContext,
    smoother: ResidualSmoother,
) -> Result<((Vec<f64>, Vec<f64>, usize), bool), ValidationError> {
    // Estimation complete cases (adjustment ∪ {T, Y} ∪ modifiers), shared by the
    // covariate matrix and the residualization pulls.
    let ids = problem.estimation_ids(true);
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
    let routed =
        matches!(smoother, ResidualSmoother::ExactUpTo { .. }) && n > MAX_NONPARAMETRIC_ROWS;
    let smoother = match smoother {
        ResidualSmoother::ExactUpTo { neighbors_beyond } if routed => {
            ResidualSmoother::Neighborhood { neighbors: neighbors_beyond }
        }
        other => other,
    };
    // Covariates are standardized to unit SD, so Silverman's (1986) multivariate
    // normal-reference factor is the bandwidth. With no covariates every kernel distance
    // is zero and the width is immaterial.
    let h = bandwidth.unwrap_or_else(|| {
        antecedent_stats::silverman_normal_reference_factor(n, dim).unwrap_or(1.0)
    });
    if !h.is_finite() || h <= 0.0 {
        return Err(ValidationError::data_msg(
            "nonparametric sensitivity bandwidth must be finite and positive",
        ));
    }
    let (t_hat, y_hat) = match smoother {
        ResidualSmoother::ExactUpTo { .. } => nw_loo_predict_pair(&t, &y, &cov, dim, h, ctx)?,
        ResidualSmoother::Neighborhood { neighbors } => {
            // Neighbours beyond the other rows add nothing, so the work is bounded by them.
            if (n as u64).saturating_mul(neighbors.min(n) as u64) > MAX_NEIGHBORHOOD_WORK {
                return Err(ValidationError::NotApplicable {
                    message: "neighborhood sensitivity work (rows x neighbors) exceeds its \
                              budget; use fewer neighbors",
                });
            }
            neighborhood_loo_predict_pair(&t, &y, &cov, dim, h, neighbors, ctx)?
        }
    };
    let t_res: Vec<f64> = t.iter().zip(&t_hat).map(|(&a, &b)| a - b).collect();
    let y_res: Vec<f64> = y.iter().zip(&y_hat).map(|(&a, &b)| a - b).collect();
    Ok(((t_res, y_res, n), routed))
}

/// Raw moments of `(t, y, u)` so `slope(a, b)` is the OLS slope of
/// `y + b u` on `t + a u` without another pass over the sample.
struct PerturbMoments {
    n: f64,
    sum_t: f64,
    sum_y: f64,
    sum_u: f64,
    tt: f64,
    ty: f64,
    tu: f64,
    yu: f64,
    uu: f64,
}

impl PerturbMoments {
    fn new(t: &[f64], y: &[f64], u: &[f64]) -> Self {
        let mut moments = Self {
            n: t.len() as f64,
            sum_t: 0.0,
            sum_y: 0.0,
            sum_u: 0.0,
            tt: 0.0,
            ty: 0.0,
            tu: 0.0,
            yu: 0.0,
            uu: 0.0,
        };
        for i in 0..t.len() {
            let (ti, yi, ui) = (t[i], y[i], u[i]);
            moments.sum_t += ti;
            moments.sum_y += yi;
            moments.sum_u += ui;
            moments.tt += ti * ti;
            moments.ty += ti * yi;
            moments.tu += ti * ui;
            moments.yu += yi * ui;
            moments.uu += ui * ui;
        }
        moments
    }

    fn slope(&self, a: f64, b: f64) -> f64 {
        if self.n < 2.0 {
            return f64::NAN;
        }
        let sum_t = self.sum_t + a * self.sum_u;
        let sum_y = self.sum_y + b * self.sum_u;
        let mean_t = sum_t / self.n;
        let mean_y = sum_y / self.n;
        let den = self.tt + 2.0 * a * self.tu + a * a * self.uu - self.n * mean_t * mean_t;
        if den < 1e-15 {
            0.0
        } else {
            let num =
                self.ty + b * self.tu + a * self.yu + a * b * self.uu - self.n * mean_t * mean_y;
            num / den
        }
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

#[cfg(test)]
mod gram_algebra {
    use super::assemble_perturbed_normal_eq;
    use antecedent_stats::{form_xtx, form_xty};

    #[test]
    fn perturb_moments_match_the_residual_ols_slope() {
        let t = [0.0, 1.0, 0.2, 0.8, 0.4];
        let y = [1.0, 3.0, 1.4, 2.2, 1.1];
        let u = [0.3, -0.2, 0.5, -0.4, 0.1];
        let moments = super::PerturbMoments::new(&t, &y, &u);
        for (a, b) in [(0.0, 0.0), (0.4, -0.2), (1.5, 0.7)] {
            let tp: Vec<f64> = t.iter().zip(&u).map(|(t, u)| t + a * u).collect();
            let yp: Vec<f64> = y.iter().zip(&u).map(|(y, u)| y + b * u).collect();
            let closed = moments.slope(a, b);
            let direct = super::residual_ols_ate(&tp, &yp);
            assert!((closed - direct).abs() < 1e-9, "a={a} b={b} closed={closed} direct={direct}");
        }
    }

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
        MAX_NONPARAMETRIC_ROWS, neighborhood_loo_predict_pair, nw_loo_predict_pair,
        orthogonalize_confounder, standardize_columns,
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
    fn the_row_cap_bounds_the_quadratic_kernel() {
        assert_eq!(MAX_NONPARAMETRIC_ROWS, 20_000);
    }

    /// Covariates on a coarse lattice (many exact duplicates and a discrete column) with two
    /// targets, from a seeded stream.
    fn lattice_problem(n: usize, dim: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x5E75);
        let mut x = Vec::with_capacity(n * dim);
        for _ in 0..n {
            for d in 0..dim {
                x.push(if d == 0 {
                    (rng.next_u64() % 4) as f64
                } else {
                    (rng.next_u64() % 7) as f64 * 0.5 + rng.next_f64() * 0.01
                });
            }
        }
        let y1: Vec<f64> = (0..n).map(|_| rng.next_f64() * 10.0 - 5.0).collect();
        let y2: Vec<f64> = (0..n).map(|_| rng.next_f64() * 4.0).collect();
        (x, y1, y2)
    }

    #[test]
    fn neighborhood_smoother_equals_the_exact_smoother_when_neighbors_cover_every_row() {
        // Duplicated covariate rows (cells of several rows), a single-row cell, two
        // dimensions, and a bandwidth narrow enough that distant rows underflow.
        let ctx = ExecutionContext::for_tests(3);
        for (dim, h) in [(1usize, 0.4), (2, 0.9), (2, 0.05)] {
            let n = 240;
            let (mut x, y1, y2) = lattice_problem(n, dim, 11 + dim as u64);
            x[0..dim].fill(1000.0);
            let exact = nw_loo_predict_pair(&y1, &y2, &x, dim, h, &ctx).unwrap();
            let truncated = neighborhood_loo_predict_pair(&y1, &y2, &x, dim, h, n, &ctx).unwrap();
            for (a, b) in exact.0.iter().zip(&truncated.0).chain(exact.1.iter().zip(&truncated.1)) {
                assert!((a - b).abs() < 1e-9 * (1.0 + a.abs()), "dim {dim} h {h}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn neighborhood_smoother_is_identical_at_any_thread_count() {
        let (x, y1, y2) = lattice_problem(900, 2, 5);
        let serial = neighborhood_loo_predict_pair(
            &y1,
            &y2,
            &x,
            2,
            0.7,
            20,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let pooled = neighborhood_loo_predict_pair(
            &y1,
            &y2,
            &x,
            2,
            0.7,
            20,
            &ExecutionContext::production(1, 4),
        )
        .unwrap();
        assert_eq!(serial, pooled);
    }

    #[test]
    fn neighborhood_smoother_leaves_the_target_out_and_keeps_ties() {
        let ctx = ExecutionContext::for_tests(1);
        // The same fixtures the exact smoother pins: an underflowing bandwidth must neither
        // leak the target nor lose nearest-neighbour ties.
        let (first, second) = neighborhood_loo_predict_pair(
            &[100.0, 2.0, 4.0],
            &[50.0, 10.0, 20.0],
            &[0.0, 1.0, 3.0],
            1,
            1e-6,
            2,
            &ctx,
        )
        .unwrap();
        assert_eq!(first, vec![2.0, 100.0, 2.0]);
        assert_eq!(second, vec![10.0, 50.0, 10.0]);
        let (tied, _) = neighborhood_loo_predict_pair(
            &[2.0, 100.0, 4.0],
            &[0.0; 3],
            &[-1.0, 0.0, 1.0],
            1,
            1e-6,
            2,
            &ctx,
        )
        .unwrap();
        assert!((tied[1] - 3.0).abs() < 1e-12);
        // Two rows sharing one covariate value: each is predicted by the other.
        let (pair, _) =
            neighborhood_loo_predict_pair(&[1.0, 5.0], &[0.0; 2], &[0.3, 0.3], 1, 1.0, 1, &ctx)
                .unwrap();
        assert_eq!(pair, vec![5.0, 1.0]);
    }

    #[test]
    fn neighborhood_smoother_tracks_the_regression_function_beyond_the_exact_cap() {
        // y = sin(2x) + noise on 30000 rows: more than the exact smoother accepts.
        let n = 30_000usize;
        assert!(n > MAX_NONPARAMETRIC_ROWS);
        let ctx = ExecutionContext::production(2, 4);
        let mut rng = ctx.rng.stream(0x7FA1);
        let mut gaussian = || antecedent_kernels::standard_normal(&mut rng);
        let x: Vec<f64> = (0..n).map(|_| gaussian()).collect();
        let truth: Vec<f64> = x.iter().map(|v| (2.0 * v).sin()).collect();
        let y: Vec<f64> = truth.iter().map(|m| m + 0.5 * gaussian()).collect();
        let h = antecedent_stats::silverman_normal_reference_factor(n, 1).unwrap();
        let (fit, _) = neighborhood_loo_predict_pair(&y, &y, &x, 1, h, 64, &ctx).unwrap();
        let rmse =
            (fit.iter().zip(&truth).map(|(f, t)| (f - t).powi(2)).sum::<f64>() / n as f64).sqrt();
        // Noise 0.5 averaged over ~64 neighbours leaves about 0.06; the bias is smaller.
        assert!(rmse < 0.12, "rmse against the true regression function: {rmse}");
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
        let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
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
        let grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5];
        assert!(!robustness_passes(grid, 0.1, 0.1));
        assert!(robustness_passes(grid, 0.2, 0.1));
        assert!(robustness_passes(grid, f64::INFINITY, 0.5));
    }

    #[test]
    fn off_grid_threshold_is_judged_by_the_last_cleared_grid_point() {
        // Tipping lies in (0.05, 0.1]; a bar of 0.07 is not shown to be cleared.
        assert!(!robustness_passes([0.05, 0.1], 0.1, 0.07));
        assert!(robustness_passes([0.05, 0.1, 0.2], 0.2, 0.07));
        // An untipped grid only certifies strengths up to its largest point.
        assert!(!robustness_passes([0.01, 0.02], f64::INFINITY, 0.1));
        assert!(robustness_passes([0.01, 0.02, 0.1], f64::INFINITY, 0.1));
        // The first grid point already tips: nothing is cleared.
        assert!(!robustness_passes([0.05, 0.1], 0.05, 0.0));
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
            estimator: LinearAdjustmentAte::new().with_bootstrap_replicates(0),
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
            estimator: LinearAdjustmentAte::new().with_bootstrap_replicates(0),
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
        let estimator = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
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

        // The bar must be a cleared grid point: the one just below the tipping value.
        let idx = grid.iter().position(|&r| r == rv).expect("RV must be a grid point");
        assert!(idx > 0, "need a grid point before the tipping RV to set a lower bar");
        let threshold = grid[idx - 1];
        let past = PartialLinearSensitivity {
            partial_r2_grid: grid,
            pass_threshold: threshold,
            estimator,
        }
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

        // The bar must be a cleared grid point: the one just below the tipping value.
        let idx = grid.iter().position(|&r| r == rv).expect("RV must be a grid point");
        assert!(idx > 0, "need a grid point before the tipping RV to set a lower bar");
        let past = NonparametricSensitivity {
            partial_r2_grid: grid.clone(),
            pass_threshold: grid[idx - 1],
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
            estimator: LinearAdjustmentAte::new().with_bootstrap_replicates(0),
        };
        let report = refuter.refute(&problem, &mut ws, &ctx).unwrap();
        assert!(
            report.comparison.is_infinite(),
            "surviving the grid must report +∞, got {}",
            report.comparison
        );
        // Surviving a grid that stops at 1e-6 certifies nothing about a 0.1 bar.
        assert!(!report.passed);
        assert!(
            report.failure_condition.as_deref().is_some_and(|f| f.contains("never explained away")),
            "{:?}",
            report.failure_condition
        );
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

/// The nonparametric sensitivity constructions at the sizes where each applies: the exact
/// smoother is unchanged below its row cap, and the neighbourhood construction beyond it
/// reproduces the exact result on a checkable subsample and brackets the tipping value the
/// known data-generating process implies.
#[cfg(test)]
mod scalable_cases {
    use antecedent_core::ExecutionContext;

    use super::{
        MAX_NONPARAMETRIC_ROWS, NeighborhoodSensitivity, NonparametricSensitivity, PerturbMoments,
        covariate_matrix, default_grid, fill_gaussian, grid_robustness_value, nw_loo_predict_pair,
        orthogonalize_confounder, residual_ols_ate,
    };
    use crate::common::{RefutationProblem, RefutationReport, sample_sd};
    use crate::test_support::{ate_query, backdoor, linear_original, tabular};

    /// Columns `[t, y, z1, z2]` and the true residuals of `t` and `y` given `(z1, z2)`.
    ///
    /// `p(z) = logistic(1.2 sin(1.5 z1) + 0.6 z2)`, `t ~ Bernoulli(p)` and
    /// `y = 0.5 t + 1.5 tanh(1.5 z1) - 0.6 z2 + 1.4 e`, so `E[t | z] = p(z)` and the true
    /// residuals are `t - p(z)` and `0.5 (t - p(z)) + 1.4 e`.
    struct Testbed {
        columns: Vec<Vec<f64>>,
        true_t_residual: Vec<f64>,
        true_y_residual: Vec<f64>,
    }

    fn testbed(n: usize, seed: u64) -> Testbed {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x30_000);
        let mut gaussian = || antecedent_kernels::standard_normal(&mut rng);
        let z1: Vec<f64> = (0..n).map(|_| gaussian()).collect();
        let z2: Vec<f64> = (0..n).map(|_| gaussian()).collect();
        let noise: Vec<f64> = (0..n).map(|_| gaussian()).collect();
        let p: Vec<f64> = (0..n)
            .map(|i| 1.0 / (1.0 + (-(1.2 * (1.5 * z1[i]).sin() + 0.6 * z2[i])).exp()))
            .collect();
        let mut uniform = ExecutionContext::for_tests(seed ^ 0x5EED).rng.stream(0x30_001);
        let t: Vec<f64> =
            p.iter().map(|&p| if uniform.next_f64() < p { 1.0 } else { 0.0 }).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 0.5 * t[i] + 1.5 * (1.5 * z1[i]).tanh() - 0.6 * z2[i] + 1.4 * noise[i])
            .collect();
        let true_t_residual: Vec<f64> = (0..n).map(|i| t[i] - p[i]).collect();
        let true_y_residual: Vec<f64> =
            (0..n).map(|i| 0.5 * true_t_residual[i] + 1.4 * noise[i]).collect();
        Testbed { columns: vec![t, y, z1, z2], true_t_residual, true_y_residual }
    }

    impl Testbed {
        fn first_rows(&self, rows: usize) -> Vec<Vec<f64>> {
            self.columns.iter().map(|c| c[..rows].to_vec()).collect()
        }

        /// Simulated tipping partial R² of the true residuals, `|rho| / (1 + |rho|)` with `rho`
        /// their correlation (module docs of `sensitivity`).
        fn true_tipping(&self) -> f64 {
            let m = self.true_t_residual.len() as f64;
            let (mt, my) = (
                self.true_t_residual.iter().sum::<f64>() / m,
                self.true_y_residual.iter().sum::<f64>() / m,
            );
            let (mut stt, mut syy, mut sty) = (0.0, 0.0, 0.0);
            for (a, b) in self.true_t_residual.iter().zip(&self.true_y_residual) {
                stt += (a - mt) * (a - mt);
                syy += (b - my) * (b - my);
                sty += (a - mt) * (b - my);
            }
            let rho = (sty / (stt * syy).sqrt()).abs();
            rho / (1.0 + rho)
        }
    }

    /// Run `refute` for the table built from `columns` (adjusting for the two covariates).
    fn report_on(
        columns: &[Vec<f64>],
        refute: impl FnOnce(
            &RefutationProblem<'_>,
            &mut antecedent_estimate::EstimationWorkspace,
        ) -> Result<RefutationReport, crate::ValidationError>,
    ) -> Result<RefutationReport, crate::ValidationError> {
        let data = tabular(columns);
        let estimand = backdoor(2);
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
        refute(&problem, &mut ws)
    }

    /// The pre-change exact refuter, verbatim: leave-one-out Nadaraya-Watson residuals from
    /// [`nw_loo_predict_pair`], the orthogonalised confounder and the partial-R² walk. Returns
    /// `(refuted effect, tipping value)`.
    fn frozen_exact_reference(
        problem: &RefutationProblem<'_>,
        ctx: &ExecutionContext,
    ) -> (f64, f64) {
        let ids = problem.estimation_ids(true);
        let mask = problem.data.complete_case_mask(&ids).unwrap();
        let (cov, n, dim) = covariate_matrix(problem, &mask).unwrap();
        let t = problem.data.float64_masked(problem.treatment(), &mask).unwrap();
        let y = problem.data.float64_masked(problem.outcome(), &mask).unwrap();
        let h = antecedent_stats::silverman_normal_reference_factor(n, dim).unwrap();
        let (t_hat, y_hat) = nw_loo_predict_pair(&t, &y, &cov, dim, h, ctx).unwrap();
        let t_res: Vec<f64> = t.iter().zip(&t_hat).map(|(&a, &b)| a - b).collect();
        let y_res: Vec<f64> = y.iter().zip(&y_hat).map(|(&a, &b)| a - b).collect();
        let residual_ate = residual_ols_ate(&t_res, &y_res);
        let (sd_t, sd_y) = (sample_sd(&t_res), sample_sd(&y_res));
        let mut u = vec![0.0; n];
        fill_gaussian(&mut u, ctx, crate::streams::NONPARAMETRIC_SENSITIVITY);
        orthogonalize_confounder(&mut u, &vec![true; n], &[t_res.clone(), y_res.clone()]);
        let mut grid = default_grid();
        grid.sort_by(f64::total_cmp);
        let original_sign = residual_ate.signum();
        let dir = if residual_ate >= 0.0 { -1.0 } else { 1.0 };
        let moments = PerturbMoments::new(&t_res, &y_res, &u);
        let mut last_ate = residual_ate;
        let mut explained_away_at = None;
        for &r in &grid {
            let scale = (r / (1.0 - r)).sqrt();
            last_ate = moments.slope(scale * sd_t, dir * scale * sd_y);
            if last_ate.abs() < 1e-9 || last_ate.signum() != original_sign {
                explained_away_at = Some(r);
                break;
            }
        }
        (last_ate, grid_robustness_value(explained_away_at))
    }

    /// The exact refuter at 10000 rows (below its cap) returns, bit for bit, what the
    /// pre-change algorithm computes on the same table.
    #[test]
    fn exact_sensitivity_retained_at_small_sizes() {
        let n = 10_000usize;
        assert!(n <= MAX_NONPARAMETRIC_ROWS);
        let columns = testbed(n, 4_539_636).columns;
        let ctx = ExecutionContext::production(71, 4);
        let report = report_on(&columns, |problem, ws| {
            NonparametricSensitivity::new().refute(problem, ws, &ctx)
        })
        .unwrap();
        assert_eq!(report.refuter.as_ref(), "sensitivity.nonparametric");
        let data = tabular(&columns);
        let estimand = backdoor(2);
        let query = ate_query();
        let (original, _, _) = linear_original(&data, &estimand, &query);
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let (frozen_effect, frozen_tipping) = frozen_exact_reference(&problem, &ctx);
        assert_eq!(report.comparison.to_bits(), frozen_tipping.to_bits());
        assert_eq!(report.refuted_ate.to_bits(), frozen_effect.to_bits());
        assert_eq!(report.replicates as usize, default_grid().len());
        // The exact path is not routed through the neighbourhood construction.
        let neighborhood = report_on(&columns, |problem, ws| {
            NeighborhoodSensitivity::new().refute(problem, ws, &ctx)
        })
        .unwrap();
        assert_eq!(neighborhood.refuter.as_ref(), "sensitivity.nonparametric_neighborhood");
    }

    /// Beyond the exact cap: the exact refuter routes to the neighbourhood construction and
    /// reports its identity; that construction (a) equals the exact refuter on a checkable subsample when its
    /// neighbourhood covers every row, and (b) reports a tipping value whose grid interval
    /// contains the tipping value implied by the known residuals.
    fn assert_scalable_sensitivity(n: usize, seed: u64) {
        assert!(n > MAX_NONPARAMETRIC_ROWS);
        let bed = testbed(n, seed);
        let ctx = ExecutionContext::production(72, 4);

        // The exact refuter does not refuse: it routes to the neighbourhood construction with its
        // default neighbours, and the report says so by its identity.
        let routed = report_on(&bed.columns, |problem, ws| {
            NonparametricSensitivity::new().refute(problem, ws, &ctx)
        })
        .unwrap();
        assert_eq!(routed.refuter.as_ref(), "sensitivity.nonparametric_neighborhood");

        let report = report_on(&bed.columns, |problem, ws| {
            NeighborhoodSensitivity::new().refute(problem, ws, &ctx)
        })
        .unwrap();
        assert_eq!(routed.comparison.to_bits(), report.comparison.to_bits());
        assert_eq!(routed.refuted_ate.to_bits(), report.refuted_ate.to_bits());
        assert_eq!(report.refuter.as_ref(), "sensitivity.nonparametric_neighborhood");
        let tipping = bed.true_tipping();
        let grid = default_grid();
        assert!(grid.iter().all(|g| (g - tipping).abs() > 0.01), "tipping {tipping} on the grid");
        let upper = grid.iter().copied().find(|&g| g > tipping).unwrap_or(f64::INFINITY);
        let lower = grid.iter().copied().filter(|&g| g < tipping).fold(0.0, f64::max);
        assert!(
            (report.comparison - upper).abs() < 1e-12,
            "reported tipping {} vs the grid value {upper} above the true tipping {tipping}",
            report.comparison
        );
        assert!(lower < tipping && tipping < report.comparison, "{lower} < {tipping}");

        // A subsample small enough for the exact smoother, with a neighbourhood that covers it.
        let rows = 4_000usize;
        let subsample = bed.first_rows(rows);
        let exact = report_on(&subsample, |problem, ws| {
            NonparametricSensitivity::new().refute(problem, ws, &ctx)
        })
        .unwrap();
        let covering = report_on(&subsample, |problem, ws| {
            NeighborhoodSensitivity { neighbors: rows, ..NeighborhoodSensitivity::new() }
                .refute(problem, ws, &ctx)
        })
        .unwrap();
        assert_eq!(covering.comparison.to_bits(), exact.comparison.to_bits());
        assert_eq!(covering.passed, exact.passed);
        assert_eq!(covering.replicates, exact.replicates);
        assert!(
            (covering.refuted_ate - exact.refuted_ate).abs()
                < 1e-8 * (1.0 + exact.refuted_ate.abs()),
            "{} vs {}",
            covering.refuted_ate,
            exact.refuted_ate
        );
    }

    #[test]
    fn nonparametric_sensitivity_beyond_20k_50000_rows() {
        assert_scalable_sensitivity(50_000, 4_965_697);
    }

    #[test]
    #[ignore = "acceptance workload: 100k rows; run with --ignored in a release build"]
    fn nonparametric_sensitivity_beyond_20k_100000_rows() {
        assert_scalable_sensitivity(100_000, 3_409_095);
    }

    /// The neighbourhood construction is a separate identity with its own provenance record, and
    /// over 2000 independent 50000-row samples its tipping interval (the grid values around the
    /// reported tipping value) contains the tipping value of the known residuals at least 93% of
    /// the time, is reported for at least 90% of samples, and is at most 1.5 times the
    /// treatment effect (0.5) wide at the median.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn scalable_sensitivity_replacement_validated_separately() {
        let record = |id: &str| {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../provenance")
                .join(format!("{id}.toml"));
            std::fs::read_to_string(path).unwrap_or_default()
        };
        assert!(
            record("validate.nonparametric_sensitivity_neighborhood")
                .contains("feature_id = \"validate.nonparametric_sensitivity_neighborhood\"")
        );
        assert!(
            record("validate.confounding_sensitivity")
                .contains("feature_id = \"validate.confounding_sensitivity\"")
        );

        let (replicates, n, effect) = (2_000u64, 50_000usize, 0.5);
        let grid = default_grid();
        let ctx = ExecutionContext::production(73, 4);
        let (mut reported, mut covered) = (0u32, 0u32);
        let mut widths: Vec<f64> = Vec::new();
        for r in 0..replicates {
            let bed = testbed(n, 5_211_204 + r);
            let Ok(report) = report_on(&bed.columns, |problem, ws| {
                NeighborhoodSensitivity::new().refute(problem, ws, &ctx)
            }) else {
                continue;
            };
            assert_eq!(report.refuter.as_ref(), "sensitivity.nonparametric_neighborhood");
            let tipping = bed.true_tipping();
            let lower = grid.iter().copied().filter(|&g| g < report.comparison).fold(0.0, f64::max);
            reported += 1;
            covered += u32::from(lower < tipping && tipping <= report.comparison);
            widths.push(report.comparison - lower);
        }
        let reporting_rate = f64::from(reported) / replicates as f64;
        assert!(reporting_rate >= 0.9, "reporting rate {reporting_rate}");
        let coverage = f64::from(covered) / f64::from(reported);
        assert!(coverage >= 0.93, "coverage {coverage} ({covered}/{reported}) below 0.93");
        widths.sort_by(f64::total_cmp);
        let median_width = widths[widths.len() / 2];
        assert!(median_width <= 1.5 * effect, "median interval width {median_width}");
        ::std::eprintln!(
            "calibration-assertion-receipt {{\"profile\":\"neighborhood_tipping_bracket.v1\",\"rows\":{n},\"attempted\":{replicates},\"reported\":{reported},\"covered\":{covered},\"conditional_coverage\":{coverage:.17},\"median_width\":{median_width:.17},\"effect\":{effect:.17}}}"
        );
    }
}
