//! Instrumental-variable estimators: Wald ratio and two-stage least squares.
//!
//! Both estimators require an `"iv"` estimand with a non-empty
//! [`IdentifiedEstimand::instruments`] slice (see `antecedent_identify::iv`). Positivity is not
//! meaningful for IV — it is not a propensity-based method — so
//! [`OverlapPolicy::ExplicitOverride`] is the only supported policy, matching
//! [`crate::adjustment::LinearAdjustmentAte`].
//!
//! [`WaldIv`] implements the simple ratio-of-differences estimator for a single binary
//! instrument. [`TwoStageLeastSquares`] handles one or more instruments (continuous or binary)
//! and optional exogenous covariates via `antecedent_stats::fit_2sls`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::manual_memcpy,
    clippy::needless_range_loop,
    clippy::similar_names
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{FaerBackend, FirstStageDiagnostics, LeastSquaresWorkspace, fit_2sls};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::se::{AnalyticSeKind, residual_sandwich_coef_se};
use crate::util::{BootstrapSeResult, bootstrap_se, stats_err};

/// Prepared IV problem: column-major instrument and exogenous-covariate designs, shared by
/// [`WaldIv`] and [`TwoStageLeastSquares`].
#[derive(Clone, Debug)]
pub struct PreparedIvProblem {
    /// Column-major `[1 | Z…]` instrument design.
    pub instruments_matrix: Arc<[f64]>,
    /// Instrument design column count (`1 + instruments.len()`).
    pub z_ncols: usize,
    /// Column-major `[1 | X…]` exogenous (non-instrumented) design.
    pub exogenous_matrix: Arc<[f64]>,
    /// Exogenous design column count (`1 + adjustment_set.len()`).
    pub x_ncols: usize,
    /// Complete-case row count.
    pub nrows: usize,
    /// Endogenous treatment, length `nrows`.
    pub treatment: Arc<[f64]>,
    /// Outcome, length `nrows`.
    pub outcome: Arc<[f64]>,
    /// Estimand method tag (always `"iv"`).
    pub method: Arc<str>,
    /// Instrument variables.
    pub instruments: Arc<[VariableId]>,
    /// Optional exogenous covariates (empty unless the estimand carries an adjustment set).
    pub adjustment_set: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Active − control treatment contrast used for the ATE scaling.
    pub treatment_delta: f64,
}

fn prepare_iv_problem(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    overlap: OverlapPolicy,
) -> Result<PreparedIvProblem, EstimationError> {
    crate::util::require_explicit_override(
        overlap,
        "IV estimators require ExplicitOverride overlap policy (not propensity-based)",
    )?;
    if estimand.method_kind().ok() != Some(antecedent_expr::EstimandMethod::Iv) {
        return Err(EstimationError::IncompatibleEstimand {
            message: "IV estimators expect an \"iv\" estimand",
        });
    }
    if estimand.instruments.is_empty() {
        return Err(EstimationError::IncompatibleEstimand {
            message: "IV estimators require a non-empty instrument set",
        });
    }
    query.validate()?;
    if !query.effect_modifiers.is_empty() {
        return Err(EstimationError::unsupported("IV estimators do not support effect modifiers"));
    }
    if query.target_population != TargetPopulation::AllObserved {
        return Err(EstimationError::unsupported(
            "IV estimators only support TargetPopulation::AllObserved",
        ));
    }
    let treatment = query.treatment;
    let outcome = query.outcome;
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    let treatment_delta = active - control;
    if treatment_delta == 0.0 {
        return Err(EstimationError::unsupported(
            "active and control treatment levels must differ",
        ));
    }

    let mut ids =
        Vec::with_capacity(2 + estimand.instruments.len() + estimand.adjustment_set.len());
    ids.push(treatment);
    ids.push(outcome);
    ids.extend_from_slice(&estimand.instruments);
    ids.extend_from_slice(&estimand.adjustment_set);
    let row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
    let t = data.float64_masked(treatment, &row_mask).map_err(EstimationError::from)?;
    let y = data.float64_masked(outcome, &row_mask).map_err(EstimationError::from)?;
    let nrows = t.len();

    let z_ncols = 1 + estimand.instruments.len();
    let mut instruments_matrix = vec![0.0; nrows * z_ncols];
    for r in 0..nrows {
        instruments_matrix[r] = 1.0;
    }
    for (i, &z_id) in estimand.instruments.iter().enumerate() {
        let col = data.float64_masked(z_id, &row_mask).map_err(EstimationError::from)?;
        let base = (1 + i) * nrows;
        for r in 0..nrows {
            instruments_matrix[base + r] = col[r];
        }
    }

    let x_ncols = 1 + estimand.adjustment_set.len();
    let mut exogenous_matrix = vec![0.0; nrows * x_ncols];
    for r in 0..nrows {
        exogenous_matrix[r] = 1.0;
    }
    for (i, &x_id) in estimand.adjustment_set.iter().enumerate() {
        let col = data.float64_masked(x_id, &row_mask).map_err(EstimationError::from)?;
        let base = (1 + i) * nrows;
        for r in 0..nrows {
            exogenous_matrix[base + r] = col[r];
        }
    }

    Ok(PreparedIvProblem {
        instruments_matrix: Arc::from(instruments_matrix),
        z_ncols,
        exogenous_matrix: Arc::from(exogenous_matrix),
        x_ncols,
        nrows,
        treatment: Arc::from(t),
        outcome: Arc::from(y),
        method: Arc::clone(&estimand.method),
        instruments: Arc::clone(&estimand.instruments),
        adjustment_set: Arc::clone(&estimand.adjustment_set),
        overlap,
        treatment_delta,
    })
}

// ---------------------------------------------------------------------------------------------
// Wald ratio estimator (single binary instrument)
// ---------------------------------------------------------------------------------------------

/// Wald (ratio-of-differences) IV estimator for a single binary instrument:
///
/// `ATE = (E[Y|Z=1] − E[Y|Z=0]) / (E[T|Z=1] − E[T|Z=0])`
///
/// Use [`TwoStageLeastSquares`] for continuous or multiple instruments.
#[derive(Clone, Debug)]
pub struct WaldIv {
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be [`OverlapPolicy::ExplicitOverride`]).
    pub overlap: OverlapPolicy,
    /// Analytic SE kind (default: delta-method on Y arms).
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids for [`AnalyticSeKind::Cluster`] (length = prepared `nrows`).
    pub cluster_ids: Option<Vec<u32>>,
    /// Multiway cluster ids.
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Optional panel time labels for panel HAC.
    pub panel_times: Option<Vec<i64>>,
}

impl Default for WaldIv {
    fn default() -> Self {
        Self::new()
    }
}

impl WaldIv {
    /// Default: 200 bootstrap replicates, explicit overlap override.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            multiway_ids: None,
            panel_times: None,
        }
    }

    /// Set the number of bootstrap replicates used for the bootstrap standard error.
    ///
    /// Defaults to 200. Set to `0` to skip bootstrapping and report only the analytic SE.
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// Set the overlap policy. Must remain [`OverlapPolicy::ExplicitOverride`] — IV is not a
    /// propensity-based method.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the analytic SE kind (default: delta-method on the Y arms).
    #[must_use]
    pub const fn with_se_kind(mut self, se_kind: AnalyticSeKind) -> Self {
        self.se_kind = se_kind;
        self
    }

    /// Set cluster ids for [`AnalyticSeKind::Cluster`] (length = prepared `nrows`).
    #[must_use]
    pub fn with_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.cluster_ids = Some(cluster_ids);
        self
    }

    /// Set multiway cluster ids (one `Vec<u32>` per clustering dimension).
    #[must_use]
    pub fn with_multiway_ids(mut self, multiway_ids: Vec<Vec<u32>>) -> Self {
        self.multiway_ids = Some(multiway_ids);
        self
    }

    /// Set panel time labels for panel HAC.
    #[must_use]
    pub fn with_panel_times(mut self, panel_times: Vec<i64>) -> Self {
        self.panel_times = Some(panel_times);
        self
    }

    /// Prepare the instrument/outcome/treatment design.
    ///
    /// # Errors
    ///
    /// Incompatible estimand (method or variable roles do not match an IV
    /// problem) or unsupported query options (e.g. effect modifiers) during
    /// problem preparation.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedIvProblem, EstimationError> {
        prepare_iv_problem(data, estimand, query, self.overlap)
    }

    /// Compute the Wald ratio ATE, with optional bootstrap.
    ///
    /// # Errors
    ///
    /// More than one instrument, a non-binary instrument, or a degenerate (zero) first stage.
    pub fn fit(
        &self,
        problem: &PreparedIvProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if problem.instruments.len() != 1 {
            return Err(EstimationError::unsupported(
                "WaldIv requires exactly one instrument; use TwoStageLeastSquares for multiple instruments",
            ));
        }
        let n = problem.nrows;
        let z: Vec<f64> = (0..n).map(|r| problem.instruments_matrix[n + r]).collect();
        if !z.iter().all(|&v| v == 0.0 || v == 1.0) {
            return Err(EstimationError::unsupported(
                "WaldIv requires a binary (0/1) instrument; use TwoStageLeastSquares for continuous instruments",
            ));
        }

        let wald = wald_ratio(&z, &problem.treatment, &problem.outcome)?;
        let ate = wald.ratio * problem.treatment_delta;
        let first_stage_diagnostics = wald_first_stage_diagnostics(&z, &problem.treatment);
        let psi = wald_influence_scores(&z, &problem.treatment, &problem.outcome, wald.ratio)?;
        let se_unit = crate::se::influence_se_kind(
            self.se_kind,
            &psi,
            problem.nrows,
            self.cluster_ids.as_deref(),
            self.multiway_ids.as_deref(),
            self.panel_times.as_deref(),
            None,
        )?;
        let se_analytic = se_if_strong_instrument(
            first_stage_diagnostics.as_ref(),
            se_unit * problem.treatment_delta.abs(),
        );

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, &z, ctx)?)
        };

        Ok(EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_first_stage_diagnostics(first_stage_diagnostics)
            .with_bootstrap(boot))
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedIvProblem,
        z: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let n = problem.nrows;
        let mut z_boot = vec![0.0; n];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        bootstrap_se(self.bootstrap_replicates, ctx, 0x5A1D_u64, n, |idx| {
            for (r, &src) in idx.iter().enumerate() {
                z_boot[r] = z[src];
                t_boot[r] = problem.treatment[src];
                y_boot[r] = problem.outcome[src];
            }
            match wald_ratio(&z_boot, &t_boot, &y_boot) {
                Ok(w) => Ok(Some(w.ratio * problem.treatment_delta)),
                Err(_) => Ok(None),
            }
        })
    }
}

struct WaldResult {
    ratio: f64,
}

/// Ratio-of-differences point estimate.
fn wald_ratio(z: &[f64], t: &[f64], y: &[f64]) -> Result<WaldResult, EstimationError> {
    let (mut sy1, mut sy0, mut st1, mut st0) = (0.0, 0.0, 0.0, 0.0);
    let (mut n1, mut n0) = (0usize, 0usize);
    for i in 0..z.len() {
        if z[i] > 0.5 {
            sy1 += y[i];
            st1 += t[i];
            n1 += 1;
        } else {
            sy0 += y[i];
            st0 += t[i];
            n0 += 1;
        }
    }
    if n1 == 0 || n0 == 0 {
        return Err(EstimationError::data_msg(
            "Wald IV requires both instrument arms (Z=0 and Z=1) to be present",
        ));
    }
    let n1f = n1 as f64;
    let n0f = n0 as f64;
    let mean_y1 = sy1 / n1f;
    let mean_y0 = sy0 / n0f;
    let mean_t1 = st1 / n1f;
    let mean_t0 = st0 / n0f;
    let denom = mean_t1 - mean_t0;
    if denom.abs() < 1e-10 {
        return Err(EstimationError::stats_msg(
            "degenerate first stage: instrument is uncorrelated with treatment",
        ));
    }
    let ratio = (mean_y1 - mean_y0) / denom;
    Ok(WaldResult { ratio })
}

/// Staiger–Stock / Stock–Yogo rule-of-thumb: analytic Wald / homoskedastic 2SLS
/// intervals are not published below this first-stage F.
const WEAK_IV_F_THRESHOLD: f64 = 10.0;

/// Analytic SE is NaN when the first stage is weaker than [`WEAK_IV_F_THRESHOLD`].
fn se_if_strong_instrument(diagnostics: Option<&FirstStageDiagnostics>, se: f64) -> f64 {
    match diagnostics {
        Some(d) if d.f_statistic.is_finite() && d.f_statistic < WEAK_IV_F_THRESHOLD => f64::NAN,
        _ => se,
    }
}

/// Weak-instrument diagnostic for the binary single-instrument Wald design.
///
/// Equivalent to an OLS F-test of `T ~ 1 + Z` for the null that the instrument's
/// coefficient is zero — i.e. the squared two-sample t-statistic comparing `T`'s `Z=1`
/// and `Z=0` arms with pooled variance, with `df1 = 1` and `df2 = n1 + n0 - 2`. Returns
/// `None` when either arm has fewer than 2 observations (pooled variance undefined),
/// mirroring [`fit_2sls`]'s [`FirstStageDiagnostics`] shape so both IV estimators expose
/// the same diagnostic type. [`WaldIv::fit`] still returns a point estimate, but
/// `se_analytic` is NaN when `f_statistic < 10`.
fn wald_first_stage_diagnostics(z: &[f64], t: &[f64]) -> Option<FirstStageDiagnostics> {
    let (mut n1, mut n0) = (0usize, 0usize);
    let (mut st1, mut st0) = (0.0, 0.0);
    for i in 0..z.len() {
        if z[i] > 0.5 {
            st1 += t[i];
            n1 += 1;
        } else {
            st0 += t[i];
            n0 += 1;
        }
    }
    if n1 < 2 || n0 < 2 {
        return None;
    }
    let n1f = n1 as f64;
    let n0f = n0 as f64;
    let mean_t1 = st1 / n1f;
    let mean_t0 = st0 / n0f;
    let (mut ss1, mut ss0) = (0.0, 0.0);
    for i in 0..z.len() {
        if z[i] > 0.5 {
            let d = t[i] - mean_t1;
            ss1 += d * d;
        } else {
            let d = t[i] - mean_t0;
            ss0 += d * d;
        }
    }
    let df2 = n1 + n0 - 2;
    let sse = ss1 + ss0;
    let ssr = (n1f * n0f / (n1f + n0f)) * (mean_t1 - mean_t0).powi(2);
    let sst = ssr + sse;
    let f_statistic = if sse > 0.0 { ssr / (sse / df2 as f64) } else { f64::INFINITY };
    let partial_r2 = if sst > 0.0 { ssr / sst } else { 0.0 };
    Some(FirstStageDiagnostics { f_statistic, df1: 1, df2, partial_r2 })
}

/// Influence-function SE for the Wald ratio `(ȳ₁−ȳ₀)/(t̄₁−t̄₀)`.
///
/// Per-row score for the ratio uses the IF of a ratio of mean contrasts. With optional
/// clustering, scores are fed to [`cluster_influence_se`].
fn wald_influence_scores(
    z: &[f64],
    t: &[f64],
    y: &[f64],
    ratio: f64,
) -> Result<Vec<f64>, EstimationError> {
    let n = z.len();
    let (mut n1, mut n0) = (0.0, 0.0);
    let (mut sy1, mut sy0, mut st1, mut st0) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        if z[i] > 0.5 {
            n1 += 1.0;
            sy1 += y[i];
            st1 += t[i];
        } else {
            n0 += 1.0;
            sy0 += y[i];
            st0 += t[i];
        }
    }
    if n1 < 1.0 || n0 < 1.0 {
        return Err(EstimationError::data_msg("Wald IV requires both instrument arms"));
    }
    let mean_y1 = sy1 / n1;
    let mean_y0 = sy0 / n0;
    let mean_t1 = st1 / n1;
    let mean_t0 = st0 / n0;
    let dt = mean_t1 - mean_t0;
    if dt.abs() < 1e-10 {
        return Err(EstimationError::stats_msg("degenerate first stage"));
    }
    let mut psi = vec![0.0; n];
    for i in 0..n {
        let (psi_dy, psi_dt) = if z[i] > 0.5 {
            ((y[i] - mean_y1) * (n as f64 / n1), (t[i] - mean_t1) * (n as f64 / n1))
        } else {
            (-(y[i] - mean_y0) * (n as f64 / n0), -(t[i] - mean_t0) * (n as f64 / n0))
        };
        psi[i] = (psi_dy - ratio * psi_dt) / dt;
    }
    Ok(psi)
}

// ---------------------------------------------------------------------------------------------
// Two-stage least squares
// ---------------------------------------------------------------------------------------------

/// Estimation workspace for [`TwoStageLeastSquares`] (reusable across bootstrap replicates).
#[derive(Clone, Debug, Default)]
pub struct TwoStageLeastSquaresWorkspace {
    /// Least-squares scratch reused by both the first- and second-stage fits.
    pub ols: LeastSquaresWorkspace,
}

/// Two-stage least squares IV estimator.
///
/// Stage 1 regresses the endogenous treatment on the FULL instrument set
/// `[instruments… | 1 | adjustment_set…]` (included exogenous regressors instrument
/// themselves); stage 2 regresses the outcome on `[fitted_T | 1 | adjustment_set…]`.
/// Supports one or more instruments (continuous or binary) and optional exogenous
/// covariates.
#[derive(Clone, Debug)]
pub struct TwoStageLeastSquares {
    /// Dense linear-algebra backend used by both least-squares stages.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be [`OverlapPolicy::ExplicitOverride`]).
    pub overlap: OverlapPolicy,
    /// Analytic SE kind.
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids for cluster / panel SE.
    pub cluster_ids: Option<Vec<u32>>,
    /// Optional multiway cluster ids for [`AnalyticSeKind::Multiway`].
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Optional panel time labels for panel HAC.
    pub panel_times: Option<Vec<i64>>,
}

impl Default for TwoStageLeastSquares {
    fn default() -> Self {
        Self::new()
    }
}

impl TwoStageLeastSquares {
    /// Default: 200 bootstrap replicates, explicit overlap override.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            multiway_ids: None,
            panel_times: None,
        }
    }

    /// Set the dense linear-algebra backend used by both least-squares stages.
    #[must_use]
    pub const fn with_backend(mut self, backend: FaerBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Set the number of bootstrap replicates used for the bootstrap standard error.
    ///
    /// Defaults to 200. Set to `0` to skip bootstrapping and report only the analytic SE.
    #[must_use]
    pub const fn with_bootstrap_replicates(mut self, replicates: u32) -> Self {
        self.bootstrap_replicates = replicates;
        self
    }

    /// Set the overlap policy. Must remain [`OverlapPolicy::ExplicitOverride`] — IV is not a
    /// propensity-based method.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the analytic SE kind (default [`AnalyticSeKind::Homoskedastic`]).
    #[must_use]
    pub const fn with_se_kind(mut self, se_kind: AnalyticSeKind) -> Self {
        self.se_kind = se_kind;
        self
    }

    /// Set cluster ids for cluster / panel SE.
    #[must_use]
    pub fn with_cluster_ids(mut self, cluster_ids: Vec<u32>) -> Self {
        self.cluster_ids = Some(cluster_ids);
        self
    }

    /// Set multiway cluster ids (one `Vec<u32>` per clustering dimension) for
    /// [`AnalyticSeKind::Multiway`].
    #[must_use]
    pub fn with_multiway_ids(mut self, multiway_ids: Vec<Vec<u32>>) -> Self {
        self.multiway_ids = Some(multiway_ids);
        self
    }

    /// Set panel time labels for panel HAC.
    #[must_use]
    pub fn with_panel_times(mut self, panel_times: Vec<i64>) -> Self {
        self.panel_times = Some(panel_times);
        self
    }

    /// Prepare the instrument/exogenous/outcome/treatment design.
    ///
    /// # Errors
    ///
    /// Incompatible estimand (method or variable roles do not match an IV
    /// problem) or unsupported query options (e.g. effect modifiers) during
    /// problem preparation.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedIvProblem, EstimationError> {
        prepare_iv_problem(data, estimand, query, self.overlap)
    }

    /// Fit 2SLS and compute the ATE, with optional bootstrap.
    ///
    /// # Errors
    ///
    /// Backend/rank failure in either stage.
    pub fn fit(
        &self,
        problem: &PreparedIvProblem,
        workspace: &mut TwoStageLeastSquaresWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        // The exogenous block carries the intercept, so pass the excluded instruments
        // without their leading intercept column (fit_2sls appends the exogenous block
        // to form the full first-stage instrument set).
        let fit = fit_2sls(
            &problem.instruments_matrix[problem.nrows..],
            problem.nrows,
            problem.z_ncols - 1,
            &problem.treatment,
            &problem.exogenous_matrix,
            problem.x_ncols,
            &problem.outcome,
            &self.backend,
            &mut workspace.ols,
        )
        .map_err(stats_err)?;
        let coef = fit.second_stage.coefficients[0];
        let ate = coef * problem.treatment_delta;
        let ncols = 1 + problem.x_ncols;
        let mut xhat = vec![0.0; problem.nrows * ncols];
        xhat[..problem.nrows].copy_from_slice(&fit.fitted_endogenous);
        xhat[problem.nrows..problem.nrows * ncols]
            .copy_from_slice(&problem.exogenous_matrix[..problem.nrows * problem.x_ncols]);
        let se_coef = if let Some(se) = residual_sandwich_coef_se(
            self.se_kind,
            &xhat,
            problem.nrows,
            ncols,
            &fit.structural_residuals,
            0,
            self.cluster_ids.as_deref(),
            self.multiway_ids.as_deref(),
            self.panel_times.as_deref(),
        )? {
            se
        } else {
            analytic_se_2sls(
                &fit.fitted_endogenous,
                &problem.exogenous_matrix,
                problem.nrows,
                problem.x_ncols,
                fit.structural_rss,
            )
        };
        let se_analytic = se_if_strong_instrument(
            Some(&fit.first_stage_diagnostics),
            se_coef * problem.treatment_delta.abs(),
        );

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, workspace, ctx)?)
        };

        Ok(EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_first_stage_diagnostics(Some(fit.first_stage_diagnostics))
            .with_bootstrap(boot))
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedIvProblem,
        workspace: &mut TwoStageLeastSquaresWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let n = problem.nrows;
        let zc = problem.z_ncols;
        let xc = problem.x_ncols;
        let mut z_boot = vec![0.0; n * zc];
        let mut x_boot = vec![0.0; n * xc];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        bootstrap_se(self.bootstrap_replicates, ctx, 0x25D5_u64, n, |idx| {
            crate::util::gather_bootstrap_vector(&mut t_boot, &problem.treatment, idx);
            crate::util::gather_bootstrap_vector(&mut y_boot, &problem.outcome, idx);
            crate::util::gather_bootstrap_design(
                &mut z_boot,
                &problem.instruments_matrix,
                n,
                zc,
                idx,
            );
            crate::util::gather_bootstrap_design(
                &mut x_boot,
                &problem.exogenous_matrix,
                n,
                xc,
                idx,
            );
            match fit_2sls(
                &z_boot[n..],
                n,
                zc - 1,
                &t_boot,
                &x_boot,
                xc,
                &y_boot,
                &self.backend,
                &mut workspace.ols,
            ) {
                Ok(fit) => Ok(Some(fit.second_stage.coefficients[0] * problem.treatment_delta)),
                Err(_) => Ok(None),
            }
        })
    }
}

/// Analytic SE for the treatment coefficient (column 0) of the 2SLS second stage:
/// `sqrt(σ̂² · [(X̂'X̂)⁻¹]₀₀)` with `X̂ = [fitted_T | exogenous]` and
/// `σ̂² = ‖y − Tβ̂ − Xγ̂‖² / (n − k)` from the STRUCTURAL residuals (actual `T`, not
/// fitted). It assumes homoskedasticity; the bootstrap SE remains the robust choice.
fn analytic_se_2sls(
    fitted_endogenous: &[f64],
    exogenous_colmajor: &[f64],
    nrows: usize,
    x_ncols: usize,
    structural_rss: f64,
) -> f64 {
    let ncols = 1 + x_ncols;
    let mut x2 = vec![0.0; nrows * ncols];
    x2[..nrows].copy_from_slice(fitted_endogenous);
    x2[nrows..nrows * ncols].copy_from_slice(&exogenous_colmajor[..nrows * x_ncols]);
    let Some(inv) = crate::util::xtx_inverse(&x2, nrows, ncols) else {
        return f64::NAN;
    };
    let sigma2 = structural_rss / (nrows as f64 - ncols as f64).max(1.0);
    (sigma2 * inv[0].max(0.0)).sqrt()
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::ExprId;
    use antecedent_expr::IdentifiedEstimand;

    use super::*;
    use crate::overlap::OverlapPolicy;
    use antecedent_kernels::standard_normal;

    /// `Z → T → Y` with `U` confounding `T-Y`: `T = Z + U + noise`, `Y = 2T + U + noise`.
    /// `Z` is a continuous instrument uncorrelated with `U`. True structural effect = 2.0.
    fn continuous_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1E70_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = (i as f64) / (n as f64) - 0.5;
            let u = standard_normal(&mut rng);
            let ti = zi + u + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
            z[i] = zi;
            t[i] = ti;
            y[i] = yi;
        }
        (build_iv_data(n, t, y, z), instrumental_estimand())
    }

    /// `Z ∈ {0,1} → T → Y` with `U` confounding `T-Y`. True structural effect = 2.0.
    fn binary_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1E71_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = (i % 2) as f64;
            let u = standard_normal(&mut rng);
            let ti = 0.5 * zi + u + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
            z[i] = zi;
            t[i] = ti;
            y[i] = yi;
        }
        (build_iv_data(n, t, y, z), instrumental_estimand())
    }

    /// Same DGP as [`binary_iv_scm`] but with the instrument's effect on `T` shrunk to
    /// `0.01` (vs `0.5`) — a deliberately weak first stage for the F-statistic test.
    fn weak_binary_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1E73_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = (i % 2) as f64;
            let u = standard_normal(&mut rng);
            let ti = 0.01 * zi + u + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
            z[i] = zi;
            t[i] = ti;
            y[i] = yi;
        }
        (build_iv_data(n, t, y, z), instrumental_estimand())
    }

    /// Same DGP as [`continuous_iv_scm`] but with the instrument's effect on `T` shrunk to
    /// `0.01` (vs `1.0`) — a deliberately weak first stage for the F-statistic test.
    fn weak_continuous_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1E74_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = (i as f64) / (n as f64) - 0.5;
            let u = standard_normal(&mut rng);
            let ti = 0.01 * zi + u + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
            z[i] = zi;
            t[i] = ti;
            y[i] = yi;
        }
        (build_iv_data(n, t, y, z), instrumental_estimand())
    }

    fn instrumental_estimand() -> IdentifiedEstimand {
        IdentifiedEstimand::instrumental(
            "iv",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        )
    }

    fn build_iv_data(n: usize, t: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> TabularData {
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
        TabularData::new(storage)
    }

    fn query() -> AverageEffectQuery {
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0)
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(21)
    }

    #[test]
    fn two_sls_recovers_effect_two() {
        let (data, estimand) = continuous_iv_scm(2000, 1);
        let est = TwoStageLeastSquares { bootstrap_replicates: 30, ..TwoStageLeastSquares::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let mut ws = TwoStageLeastSquaresWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
    }

    #[test]
    fn two_sls_analytic_se_tracks_bootstrap() {
        // Strong instrument: the structural-residual analytic SE should sit near the
        // bootstrap SE (the naive fitted-T RSS variant is systematically larger when
        // beta != 0 because it treats T-hat prediction error as regression noise).
        let (data, estimand) = continuous_iv_scm(2000, 7);
        let est = TwoStageLeastSquares { bootstrap_replicates: 60, ..TwoStageLeastSquares::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let mut ws = TwoStageLeastSquaresWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let se_boot = effect.se_bootstrap.unwrap();
        assert!(effect.se_analytic.is_finite() && effect.se_analytic > 0.0);
        let ratio = effect.se_analytic / se_boot;
        assert!(
            (0.4..=2.5).contains(&ratio),
            "analytic={} bootstrap={se_boot}",
            effect.se_analytic
        );
    }

    #[test]
    fn two_sls_first_stage_diagnostics_strong_vs_weak() {
        let est = TwoStageLeastSquares::new();

        let (strong_data, strong_estimand) = continuous_iv_scm(2000, 1);
        let strong_prep = est.prepare(&strong_data, &strong_estimand, &query()).unwrap();
        let mut strong_ws = TwoStageLeastSquaresWorkspace::default();
        let strong_effect =
            est.fit(&strong_prep, &mut strong_ws, &ctx(), AssumptionSet::new()).unwrap();
        let strong_diag =
            strong_effect.first_stage_diagnostics.expect("2SLS always reports diagnostics");
        assert_eq!(strong_diag.df1, 1);
        assert!(
            strong_diag.f_statistic > 50.0,
            "expected a strong instrument to clear F=50, got {}",
            strong_diag.f_statistic
        );

        let (weak_data, weak_estimand) = weak_continuous_iv_scm(2000, 1);
        let weak_prep = est.prepare(&weak_data, &weak_estimand, &query()).unwrap();
        let mut weak_ws = TwoStageLeastSquaresWorkspace::default();
        let weak_effect = est.fit(&weak_prep, &mut weak_ws, &ctx(), AssumptionSet::new()).unwrap();
        let weak_diag =
            weak_effect.first_stage_diagnostics.expect("2SLS always reports diagnostics");
        // F < 10 is the Staiger-Stock rule-of-thumb weak-instrument threshold, which is
        // what this diagnostic exists to let a caller check.
        assert!(
            weak_diag.f_statistic < 10.0,
            "expected a weak instrument to stay under the F=10 rule of thumb, got {}",
            weak_diag.f_statistic
        );
        assert!(
            weak_effect.se_analytic.is_nan(),
            "weak first stage must not publish a Wald/2SLS analytic SE"
        );
        assert!(strong_effect.se_analytic.is_finite() && strong_effect.se_analytic > 0.0);
        // The separation is the real claim: a weak first stage must sit an order of
        // magnitude below a strong one.
        assert!(
            strong_diag.f_statistic > 10.0 * weak_diag.f_statistic,
            "expected strong F ({}) to dwarf weak F ({})",
            strong_diag.f_statistic,
            weak_diag.f_statistic
        );
    }

    #[test]
    fn wald_iv_first_stage_diagnostics_strong_vs_weak() {
        let est = WaldIv { bootstrap_replicates: 0, ..WaldIv::new() };

        let (strong_data, strong_estimand) = binary_iv_scm(4000, 5);
        let strong_prep = est.prepare(&strong_data, &strong_estimand, &query()).unwrap();
        let strong_effect = est.fit(&strong_prep, &ctx(), AssumptionSet::new()).unwrap();
        let strong_diag =
            strong_effect.first_stage_diagnostics.expect("WaldIv always reports diagnostics");
        assert_eq!(strong_diag.df1, 1);
        assert!(
            strong_diag.f_statistic > 50.0,
            "expected a strong instrument to clear F=50, got {}",
            strong_diag.f_statistic
        );

        let (weak_data, weak_estimand) = weak_binary_iv_scm(200, 5);
        let weak_prep = est.prepare(&weak_data, &weak_estimand, &query()).unwrap();
        let weak_effect = est.fit(&weak_prep, &ctx(), AssumptionSet::new()).unwrap();
        let weak_diag =
            weak_effect.first_stage_diagnostics.expect("WaldIv always reports diagnostics");
        // F < 10 is the Staiger-Stock rule-of-thumb weak-instrument threshold, which is
        // what this diagnostic exists to let a caller check.
        assert!(
            weak_diag.f_statistic < 10.0,
            "expected a weak instrument to stay under the F=10 rule of thumb, got {}",
            weak_diag.f_statistic
        );
        assert!(
            weak_effect.se_analytic.is_nan(),
            "weak first stage must not publish a Wald/2SLS analytic SE"
        );
        assert!(strong_effect.se_analytic.is_finite() && strong_effect.se_analytic > 0.0);
        // The separation is the real claim: a weak first stage must sit an order of
        // magnitude below a strong one.
        assert!(
            strong_diag.f_statistic > 10.0 * weak_diag.f_statistic,
            "expected strong F ({}) to dwarf weak F ({})",
            strong_diag.f_statistic,
            weak_diag.f_statistic
        );
    }

    #[test]
    fn two_sls_rejects_explicit_override_violation() {
        let (data, estimand) = continuous_iv_scm(100, 2);
        let est = TwoStageLeastSquares {
            overlap: OverlapPolicy::require_diagnostics(),
            ..TwoStageLeastSquares::new()
        };
        let err = est.prepare(&data, &estimand, &query()).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn two_sls_rejects_non_iv_estimand() {
        let (data, mut estimand) = continuous_iv_scm(100, 3);
        estimand.method = Arc::from("backdoor.adjustment");
        let est = TwoStageLeastSquares::new();
        let err = est.prepare(&data, &estimand, &query()).unwrap_err();
        assert!(matches!(err, EstimationError::IncompatibleEstimand { .. }));
    }

    #[test]
    fn two_sls_rejects_empty_instruments() {
        let (data, mut estimand) = continuous_iv_scm(100, 4);
        estimand.instruments = Arc::from([]);
        let est = TwoStageLeastSquares::new();
        let err = est.prepare(&data, &estimand, &query()).unwrap_err();
        assert!(matches!(err, EstimationError::IncompatibleEstimand { .. }));
    }

    #[test]
    fn wald_iv_recovers_effect_two() {
        let (data, estimand) = binary_iv_scm(4000, 5);
        let est = WaldIv { bootstrap_replicates: 30, ..WaldIv::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let effect = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.6, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
    }

    #[test]
    fn wald_iv_rejects_continuous_instrument() {
        let (data, estimand) = continuous_iv_scm(200, 6);
        let est = WaldIv::new();
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let err = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap_err();
        assert!(matches!(err, EstimationError::Unsupported { .. }));
    }
}
