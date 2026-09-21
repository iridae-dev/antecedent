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
use antecedent_stats::{
    FaerBackend, FirstStageDiagnostics, LeastSquaresWorkspace, anderson_rubin_confidence_set,
    fit_2sls,
};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::se::AnalyticSeKind;
use crate::util::stats_err;

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
        return Err(EstimationError::refused(
            antecedent_core::reason_code!("population_not_estimable"),
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

    /// Compute the Wald ratio ATE.
    ///
    /// Licensed uncertainty is the homoskedastic Anderson–Rubin set attached to
    /// [`EffectEstimate::first_stage_diagnostics`]. Wald analytic and bootstrap
    /// standard errors are never published.
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
        let _ = ctx;
        if problem.instruments.len() != 1 {
            return Err(EstimationError::unsupported(
                "WaldIv requires exactly one instrument; use TwoStageLeastSquares for multiple instruments",
            ));
        }
        if !problem.adjustment_set.is_empty() {
            return Err(EstimationError::unsupported(
                "WaldIv does not support adjustment covariates; use TwoStageLeastSquares",
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
        let mut first_stage_diagnostics = wald_first_stage_diagnostics(&z, &problem.treatment);
        attach_anderson_rubin(
            &mut first_stage_diagnostics,
            self.se_kind,
            &problem.outcome,
            &problem.treatment,
            &z,
            1,
            &problem.exogenous_matrix,
            problem.x_ncols,
            problem.treatment_delta,
        )?;
        // Licensed IV uncertainty is the Anderson–Rubin set on first-stage
        // diagnostics — never a Wald SE (pretest or otherwise) or bootstrap SE.
        let se_analytic = f64::NAN;

        Ok(EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_n_obs(u64::try_from(n).unwrap_or(u64::MAX))
            .with_first_stage_diagnostics(first_stage_diagnostics)
            .with_se_kind(self.se_kind))
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

/// Nominal level for the licensed Anderson–Rubin IV confidence set (matches the
/// Wald interval level previously claimed by `se_analytic`).
const AR_LEVEL: f64 = 0.95;

/// Attach a homoskedastic Anderson–Rubin set for the ATE-scale structural effect.
///
/// Non-homoskedastic `se_kind` values withhold AR with an explicit reason: a robust
/// AR score test is not derived here, and the F≥10 Wald SE is never used as a fallback.
fn attach_anderson_rubin(
    diagnostics: &mut Option<FirstStageDiagnostics>,
    se_kind: AnalyticSeKind,
    y: &[f64],
    t: &[f64],
    instruments_colmajor: &[f64],
    z_ncols: usize,
    exogenous_colmajor: &[f64],
    x_ncols: usize,
    treatment_delta: f64,
) -> Result<(), EstimationError> {
    let Some(diag) = diagnostics.as_mut() else {
        return Ok(());
    };
    if !matches!(se_kind, AnalyticSeKind::Homoskedastic) {
        diag.anderson_rubin = None;
        diag.uncertainty_withheld = Some("anderson_rubin_requires_homoskedastic");
        return Ok(());
    }
    let mut ws = LeastSquaresWorkspace::default();
    let (ar, reason) = anderson_rubin_confidence_set(
        y,
        t,
        instruments_colmajor,
        y.len(),
        z_ncols,
        exogenous_colmajor,
        x_ncols,
        AR_LEVEL,
        &FaerBackend,
        &mut ws,
    )
    .map_err(stats_err)?;
    // Scale structural-β endpoints to the published ATE contrast.
    diag.anderson_rubin = ar.map(|(lo, hi, level)| {
        let (mut lo_a, mut hi_a) = (lo * treatment_delta, hi * treatment_delta);
        if treatment_delta < 0.0 {
            std::mem::swap(&mut lo_a, &mut hi_a);
        }
        (lo_a, hi_a, level)
    });
    diag.uncertainty_withheld = reason;
    Ok(())
}

/// Weak-instrument diagnostic for the binary single-instrument Wald design.
///
/// Equivalent to an OLS F-test of `T ~ 1 + Z` for the null that the instrument's
/// coefficient is zero — i.e. the squared two-sample t-statistic comparing `T`'s `Z=1`
/// and `Z=0` arms with pooled variance, with `df1 = 1` and `df2 = n1 + n0 - 2`. Returns
/// `None` when either arm has fewer than 2 observations (pooled variance undefined),
/// mirroring [`fit_2sls`]'s [`FirstStageDiagnostics`] shape so both IV estimators expose
/// the same diagnostic type.
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
    Some(FirstStageDiagnostics {
        f_statistic,
        df1: 1,
        df2,
        partial_r2,
        anderson_rubin: None,
        uncertainty_withheld: None,
    })
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

    /// Fit 2SLS and compute the ATE.
    ///
    /// Licensed uncertainty is the homoskedastic Anderson–Rubin set attached to
    /// [`EffectEstimate::first_stage_diagnostics`]. Wald analytic and bootstrap
    /// standard errors are never published.
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
        let _ = ctx;
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
        let mut diagnostics = Some(fit.first_stage_diagnostics);
        attach_anderson_rubin(
            &mut diagnostics,
            self.se_kind,
            &problem.outcome,
            &problem.treatment,
            &problem.instruments_matrix[problem.nrows..],
            problem.z_ncols - 1,
            &problem.exogenous_matrix,
            problem.x_ncols,
            problem.treatment_delta,
        )?;
        let se_analytic = f64::NAN;

        Ok(EffectEstimate::new(ate, se_analytic, assumptions, problem.overlap)
            .with_n_obs(u64::try_from(problem.nrows).unwrap_or(u64::MAX))
            .with_first_stage_diagnostics(diagnostics)
            .with_se_kind(self.se_kind))
    }
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
        assert!(effect.se_bootstrap.is_none(), "IV must not publish bootstrap SE");
        assert!(!effect.se_analytic.is_finite(), "IV must not publish Wald SE");
        let diag = effect.first_stage_diagnostics.as_ref().unwrap();
        let (lo, hi, level) = diag.anderson_rubin.expect("homoskedastic AR set");
        assert_eq!(level, 0.95);
        assert!(lo <= 2.0 && 2.0 <= hi, "AR [{lo}, {hi}]");
    }

    #[test]
    fn two_sls_withholds_wald_and_bootstrap_se() {
        let (data, estimand) = continuous_iv_scm(2000, 7);
        let est = TwoStageLeastSquares { bootstrap_replicates: 60, ..TwoStageLeastSquares::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let mut ws = TwoStageLeastSquaresWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!(effect.se_bootstrap.is_none());
        assert!(!effect.se_analytic.is_finite());
        let diag = effect.first_stage_diagnostics.as_ref().unwrap();
        assert!(diag.anderson_rubin.is_some());
        assert!(diag.uncertainty_withheld.is_none());
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
        assert!(
            weak_diag.f_statistic < 10.0,
            "expected a weak instrument to stay under the F=10 rule of thumb, got {}",
            weak_diag.f_statistic
        );
        assert!(!weak_effect.se_analytic.is_finite());
        assert!(weak_effect.se_bootstrap.is_none());
        assert!(!strong_effect.se_analytic.is_finite());
        match &weak_diag.anderson_rubin {
            Some((lo, hi, level)) => {
                assert_eq!(*level, 0.95);
                // Finite AR sets under weak IV may miss on a single draw; unbounded
                // sets must contain the truth on the infinite side.
                if !lo.is_finite() || !hi.is_finite() {
                    let covers = (!lo.is_finite() || 2.0 >= *lo) && (!hi.is_finite() || 2.0 <= *hi);
                    assert!(covers, "unbounded AR [{lo}, {hi}] must contain 2");
                }
            }
            None => {
                assert_eq!(
                    weak_diag.uncertainty_withheld,
                    Some("anderson_rubin_set_is_union"),
                    "honest withhold when AR set is a union"
                );
            }
        }
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
        assert!(
            weak_diag.f_statistic < 10.0,
            "expected a weak instrument to stay under the F=10 rule of thumb, got {}",
            weak_diag.f_statistic
        );
        assert!(!weak_effect.se_analytic.is_finite());
        assert!(weak_effect.se_bootstrap.is_none());
        assert!(!strong_effect.se_analytic.is_finite());
        match &weak_diag.anderson_rubin {
            Some((lo, hi, level)) => {
                assert_eq!(*level, 0.95);
                if !lo.is_finite() || !hi.is_finite() {
                    let covers = (!lo.is_finite() || 2.0 >= *lo) && (!hi.is_finite() || 2.0 <= *hi);
                    assert!(covers, "unbounded AR [{lo}, {hi}] must contain 2");
                }
            }
            None => {
                assert_eq!(weak_diag.uncertainty_withheld, Some("anderson_rubin_set_is_union"));
            }
        }
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
        assert!(effect.se_bootstrap.is_none());
        assert!(!effect.se_analytic.is_finite());
        let (lo, hi, _) =
            effect.first_stage_diagnostics.as_ref().unwrap().anderson_rubin.expect("AR set");
        assert!(lo <= 2.0 && 2.0 <= hi);
    }

    #[test]
    fn wald_iv_hc1_withholds_anderson_rubin() {
        let (data, estimand) = binary_iv_scm(800, 5);
        let est =
            WaldIv { bootstrap_replicates: 20, se_kind: AnalyticSeKind::Hc1, ..WaldIv::new() };
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let effect = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap();
        assert!(!effect.se_analytic.is_finite());
        assert!(effect.se_bootstrap.is_none());
        let diag = effect.first_stage_diagnostics.as_ref().unwrap();
        assert!(diag.anderson_rubin.is_none());
        assert_eq!(diag.uncertainty_withheld, Some("anderson_rubin_requires_homoskedastic"));
    }

    #[test]
    fn anderson_rubin_coverage_on_weakish_dgp() {
        // First stage ~0.45·Z at n=300 leaves a substantial F<10 share; AR coverage
        // of the acceptance set (published interval, unbounded ray, or withheld union
        // that still contains the truth) must sit near 0.95.
        let n = 300usize;
        let n_sim = 400u32;
        let mut covered = 0u32;
        let mut weak_f = 0u32;
        let mut scored = 0u32;
        let est = WaldIv { bootstrap_replicates: 0, ..WaldIv::new() };
        for s in 0..n_sim {
            let (data, estimand) = moderate_binary_iv_scm(n, 90_000 + u64::from(s));
            let prep = est.prepare(&data, &estimand, &query()).unwrap();
            let effect = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap();
            assert!(!effect.se_analytic.is_finite());
            assert!(effect.se_bootstrap.is_none());
            let diag = effect.first_stage_diagnostics.as_ref().unwrap();
            if diag.f_statistic.is_finite() && diag.f_statistic < 10.0 {
                weak_f += 1;
            }
            scored += 1;
            let z: Vec<f64> =
                (0..prep.nrows).map(|r| prep.instruments_matrix[prep.nrows + r]).collect();
            let mut ws = LeastSquaresWorkspace::default();
            let ar_true = antecedent_stats::anderson_rubin_statistic(
                &prep.outcome,
                &prep.treatment,
                &z,
                prep.nrows,
                1,
                &prep.exogenous_matrix,
                prep.x_ncols,
                2.0,
                &FaerBackend,
                &mut ws,
            )
            .unwrap();
            let df = prep.nrows.saturating_sub(1 + prep.x_ncols);
            let crit = antecedent_stats::anderson_rubin_kf_critical(0.95, 1, df);
            let accepts_truth = ar_true.is_finite() && ar_true <= crit;
            let covers = match diag.anderson_rubin {
                Some((lo, hi, level)) => {
                    assert_eq!(level, 0.95);
                    let left_ok = !lo.is_finite() || 2.0 >= lo;
                    let right_ok = !hi.is_finite() || 2.0 <= hi;
                    left_ok && right_ok
                }
                None => {
                    // Union of disjoint rays is not published as one interval, but the
                    // AR non-rejection set still covers when the test accepts at truth.
                    diag.uncertainty_withheld == Some("anderson_rubin_set_is_union")
                        && accepts_truth
                }
            };
            if covers {
                covered += 1;
            }
        }
        let rate = f64::from(covered) / f64::from(scored);
        let weak_share = f64::from(weak_f) / f64::from(scored);
        assert!(weak_share > 0.05, "DGP must produce a non-trivial F<10 share, got {weak_share}");
        assert!(
            (0.85..=0.99).contains(&rate),
            "AR coverage {rate} outside wide band around 0.95 (covered={covered}/{scored}, weak_share={weak_share})"
        );
    }

    /// Binary IV with a moderate first stage (`0.45·Z`) so many draws have F < 10.
    fn moderate_binary_iv_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1E75_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = (i % 2) as f64;
            let u = standard_normal(&mut rng);
            let ti = 0.45 * zi + u + 0.1 * standard_normal(&mut rng);
            let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
            z[i] = zi;
            t[i] = ti;
            y[i] = yi;
        }
        (build_iv_data(n, t, y, z), instrumental_estimand())
    }

    #[test]
    fn wald_iv_rejects_continuous_instrument() {
        let (data, estimand) = continuous_iv_scm(200, 6);
        let est = WaldIv::new();
        let prep = est.prepare(&data, &estimand, &query()).unwrap();
        let err = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap_err();
        assert!(matches!(err, EstimationError::Unsupported { .. }));
    }

    #[test]
    fn wald_iv_rejects_nonempty_adjustment_set_while_two_sls_recovers() {
        // T = Z + U, Y = 2T + 3U; Z binary. Unconditional Wald is 3; 2SLS is 2.
        let z = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let u = [0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0];
        let t: Vec<f64> = z.iter().zip(u).map(|(zi, ui)| zi + ui).collect();
        let y: Vec<f64> = t.iter().zip(u).map(|(ti, ui)| 2.0 * ti + 3.0 * ui).collect();
        let mut b = CausalSchemaBuilder::new();
        for (name, hint) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("z", RoleHint::Context),
            ("u", RoleHint::Context),
        ] {
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
        let cols = [t, y, z.to_vec(), u.to_vec()]
            .into_iter()
            .enumerate()
            .map(|(i, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(u32::try_from(i).unwrap()),
                        Arc::from(values),
                        ValidityBitmap::all_valid(8),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let data = TabularData::new(
            OwnedColumnarStorage::try_new(b.build().unwrap(), cols, None, None).unwrap(),
        );
        let mut estimand = instrumental_estimand();
        estimand.adjustment_set = Arc::from([VariableId::from_raw(3)]);
        let q = query();
        let wald = WaldIv { bootstrap_replicates: 8, ..WaldIv::new() };
        let wald_prep = wald.prepare(&data, &estimand, &q).unwrap();
        let err = wald.fit(&wald_prep, &ctx(), AssumptionSet::new()).unwrap_err();
        assert!(matches!(err, EstimationError::Unsupported { .. }), "{err:?}");

        let tsls = TwoStageLeastSquares::new();
        let tsls_prep = tsls.prepare(&data, &estimand, &q).unwrap();
        let mut ws = TwoStageLeastSquaresWorkspace::default();
        let effect = tsls.fit(&tsls_prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-12, "ate={}", effect.ate);
    }
}
