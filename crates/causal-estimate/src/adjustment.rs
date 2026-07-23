//! Linear adjustment ATE estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names
)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, Intervention, TargetPopulation, VariableId,
};
use causal_data::TabularData;
use causal_expr::{EstimandMethod, IdentifiedEstimand};
use causal_stats::{
    CompiledDesign, DenseLinearAlgebra, FaerBackend, LassoOptions, LeastSquaresWorkspace,
    MEstimateOptions, fit_huber_m, fit_lasso, fit_ridge, form_xtx, invert_square,
};

use crate::error::EstimationError;
use crate::overlap::{OverlapPolicy, OverlapReport};
use crate::prepare::{require_method, treatment_contrast, validate_ate_query_with_targets};
use crate::se::{AnalyticSeKind, residual_sandwich_coef_se};

/// Prepared estimation problem (compiled design retained).
#[derive(Clone, Debug)]
pub struct PreparedEstimationProblem {
    /// Compiled design.
    pub design: CompiledDesign,
    /// Estimand method tag.
    pub method: Arc<str>,
    /// Adjustment set.
    pub adjustment_set: Arc<[VariableId]>,
    /// Overlap policy applied.
    pub overlap: OverlapPolicy,
    /// Active − control treatment contrast used for the ATE scaling.
    pub treatment_delta: f64,
    /// Target population (ATT/ATC use g-computation over the arm’s covariate law).
    pub target_population: TargetPopulation,
    /// Complete-case treatment values (aligned with design rows).
    pub treatment: Arc<[f64]>,
    /// Active treatment level.
    pub active: f64,
    /// Control treatment level.
    pub control: f64,
}

/// Estimation workspace (reusable across bootstrap replicates).
#[derive(Clone, Debug, Default)]
pub struct EstimationWorkspace {
    /// OLS scratch.
    pub ols: LeastSquaresWorkspace,
}

/// Point estimate with uncertainty.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EffectEstimate {
    /// ATE point estimate `β_T * (active − control)`.
    pub ate: f64,
    /// Analytic IID standard error (homoskedastic).
    pub se_analytic: f64,
    /// Bootstrap standard error (if requested and enough survivors).
    pub se_bootstrap: Option<f64>,
    /// Successful bootstrap replicates when bootstrap was requested.
    pub bootstrap_replicates_ok: Option<u32>,
    /// Soft-failed bootstrap replicates when bootstrap was requested.
    pub bootstrap_replicates_failed: Option<u32>,
    /// Bootstrap loop observed cooperative cancellation (partial replicates).
    pub bootstrap_cancelled: bool,
    /// Adaptive bootstrap early-stop (SE relative change).
    pub bootstrap_early_stopped: bool,
    /// Assumptions carried from identification.
    pub assumptions: AssumptionSet,
    /// Overlap policy recorded on the artifact.
    pub overlap: OverlapPolicy,
    /// Propensity overlap diagnostics when computed.
    pub overlap_report: Option<OverlapReport>,
    /// Estimated retained-memory cost of fitted scratch (bytes), when known.
    pub retained_memory_bytes: Option<u64>,
}

impl EffectEstimate {
    /// Construct a point estimate without bootstrap accounting.
    #[must_use]
    pub fn new(
        ate: f64,
        se_analytic: f64,
        assumptions: AssumptionSet,
        overlap: OverlapPolicy,
    ) -> Self {
        Self {
            ate,
            se_analytic,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            bootstrap_cancelled: false,
            bootstrap_early_stopped: false,
            assumptions,
            overlap,
            overlap_report: None,
            retained_memory_bytes: None,
        }
    }

    /// Full constructor (required outside this crate because the type is `#[non_exhaustive]`).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        ate: f64,
        se_analytic: f64,
        se_bootstrap: Option<f64>,
        bootstrap_replicates_ok: Option<u32>,
        bootstrap_replicates_failed: Option<u32>,
        bootstrap_cancelled: bool,
        bootstrap_early_stopped: bool,
        assumptions: AssumptionSet,
        overlap: OverlapPolicy,
        overlap_report: Option<OverlapReport>,
        retained_memory_bytes: Option<u64>,
    ) -> Self {
        Self {
            ate,
            se_analytic,
            se_bootstrap,
            bootstrap_replicates_ok,
            bootstrap_replicates_failed,
            bootstrap_cancelled,
            bootstrap_early_stopped,
            assumptions,
            overlap,
            overlap_report,
            retained_memory_bytes,
        }
    }

    /// Attach bootstrap SE accounting (or clear when bootstrap was skipped).
    #[must_use]
    pub fn with_bootstrap(mut self, boot: Option<crate::util::BootstrapSeResult>) -> Self {
        match boot {
            None => {
                self.se_bootstrap = None;
                self.bootstrap_replicates_ok = None;
                self.bootstrap_replicates_failed = None;
                self.bootstrap_cancelled = false;
                self.bootstrap_early_stopped = false;
            }
            Some(b) => {
                self.se_bootstrap = b.se;
                self.bootstrap_replicates_ok = Some(b.replicates_ok);
                self.bootstrap_replicates_failed = Some(b.replicates_failed);
                self.bootstrap_cancelled = b.cancelled;
                self.bootstrap_early_stopped = b.early_stopped;
            }
        }
        self
    }
}

/// Linear fit family for [`LinearAdjustmentAte`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LinearFitKind {
    /// Ordinary least squares.
    Ols,
    /// Ridge with penalty `lambda` (intercept unpenalized when constant).
    Ridge {
        /// Ridge penalty λ.
        lambda: f64,
    },
    /// Lasso with penalty `lambda`.
    ///
    /// Analytic SE is permanently omitted: classical / active-set sandwich SEs are
    /// invalid after selection, and debiased Lasso changes the point estimator.
    /// Use bootstrap (`bootstrap_replicates > 0`); `se_analytic` is NaN.
    Lasso {
        /// Lasso penalty λ.
        lambda: f64,
    },
    /// Huber M-estimation with tuning constant `c`.
    Huber {
        /// Huber tuning constant (default 1.345).
        c: f64,
    },
}

impl Default for LinearFitKind {
    fn default() -> Self {
        Self::Ols
    }
}

/// Linear adjustment estimator for backdoor ATE.
#[derive(Clone, Debug)]
pub struct LinearAdjustmentAte {
    /// Backend.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy (must be explicit in).
    pub overlap: OverlapPolicy,
    /// Analytic SE estimator (default homoskedastic).
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids (length = prepared `nrows`) for cluster / panel SE.
    pub cluster_ids: Option<Vec<u32>>,
    /// Optional multiway cluster ids for [`AnalyticSeKind::Multiway`].
    pub multiway_ids: Option<Vec<Vec<u32>>>,
    /// Linear fit family (default OLS).
    pub fit_kind: LinearFitKind,
}

impl Default for LinearAdjustmentAte {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearAdjustmentAte {
    /// Default: 200 bootstrap replicates, explicit overlap override, OLS.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: OverlapPolicy::ExplicitOverride,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            multiway_ids: None,
            fit_kind: LinearFitKind::Ols,
        }
    }

    /// Prepare design from tabular data, identified estimand, and query levels.
    ///
    /// # Errors
    ///
    /// Missing columns, unsupported query options, type errors, or overlap policy not set.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        crate::util::require_explicit_override(
            self.overlap,
            "LinearAdjustmentAte requires ExplicitOverride overlap policy",
        )?;
        require_method(
            estimand,
            &[EstimandMethod::BackdoorAdjustment],
            "LinearAdjustmentAte expects backdoor.adjustment",
        )?;
        validate_ate_query_with_targets(query)?;
        let treatment = query.treatment;
        let outcome = query.outcome;
        let (active, control, treatment_delta) = treatment_contrast(&query.active, &query.control)?;

        let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
        ids.push(treatment);
        ids.push(outcome);
        ids.extend_from_slice(&estimand.adjustment_set);
        let row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
        let t = data.float64_masked(treatment, &row_mask).map_err(EstimationError::from)?;
        let y = data.float64_masked(outcome, &row_mask).map_err(EstimationError::from)?;
        let mut covs: Vec<(VariableId, Vec<f64>)> = Vec::new();
        for &z in estimand.adjustment_set.iter() {
            covs.push((z, data.float64_masked(z, &row_mask).map_err(EstimationError::from)?));
        }
        let cov_refs: Vec<(VariableId, &[f64])> =
            covs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let selected_rows: Vec<usize> =
            row_mask.iter().enumerate().filter_map(|(i, keep)| keep.then_some(i)).collect();
        let design = CompiledDesign::linear_adjustment(&t, &cov_refs, &y, &selected_rows)
            .map_err(EstimationError::from)?;
        Ok(PreparedEstimationProblem {
            design,
            method: Arc::clone(&estimand.method),
            adjustment_set: Arc::clone(&estimand.adjustment_set),
            overlap: self.overlap,
            treatment_delta,
            target_population: query.target_population.clone(),
            treatment: Arc::from(t),
            active,
            control,
        })
    }

    /// Fit ATE with optional IID bootstrap.
    ///
    /// # Errors
    ///
    /// Fit / SE failure.
    pub fn fit(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let point = self.fit_point(problem, workspace, assumptions)?;
        self.attach_bootstrap(problem, workspace, ctx, point)
    }

    /// Point estimate + analytic SE only (no bootstrap).
    ///
    /// # Errors
    ///
    /// Fit / SE failure.
    pub fn fit_point(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let (coefficients, residuals, rss, analytic_se_ok) =
            self.fit_coefficients(problem, workspace)?;
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
        let ate = gcomp_or_coef_ate(problem, &coefficients, t_col)?;
        let n = problem.design.nrows as f64;
        let p = problem.design.ncols as f64;
        let se_coef = if !analytic_se_ok {
            f64::NAN
        } else if let Some(se) = residual_sandwich_coef_se(
            self.se_kind,
            &problem.design.matrix,
            problem.design.nrows,
            problem.design.ncols,
            &residuals,
            t_col,
            self.cluster_ids.as_deref(),
            self.multiway_ids.as_deref(),
        )? {
            se
        } else {
            let sigma2 = rss / (n - p).max(1.0);
            analytic_se_treatment(
                &problem.design.matrix,
                problem.design.nrows,
                problem.design.ncols,
                t_col,
                sigma2,
            )
        };
        let se_analytic = se_coef * problem.treatment_delta.abs();

        Ok(EffectEstimate {
            ate,
            se_analytic,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            bootstrap_cancelled: false,
            bootstrap_early_stopped: false,
            assumptions,
            overlap: problem.overlap,
            overlap_report: None,
            retained_memory_bytes: None,
        })
    }

    /// Attach bootstrap SE onto a point estimate (progressive uncertainty stage).
    ///
    /// # Errors
    ///
    /// Bootstrap failure.
    pub fn attach_bootstrap(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        point: EffectEstimate,
    ) -> Result<EffectEstimate, EstimationError> {
        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            let t_col = problem
                .design
                .treatment_column()
                .ok_or_else(|| EstimationError::stats_msg("missing treatment column"))?;
            Some(self.bootstrap_se(problem, workspace, ctx, t_col)?)
        };
        Ok(point.with_bootstrap(boot))
    }

    fn fit_coefficients(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
    ) -> Result<(Vec<f64>, Vec<f64>, f64, bool), EstimationError> {
        let x = &problem.design.matrix;
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let y = &problem.design.outcome;
        match self.fit_kind {
            LinearFitKind::Ols => {
                let fit = problem
                    .design
                    .fit_ols(&self.backend, &mut workspace.ols)
                    .map_err(EstimationError::from)?;
                Ok((fit.coefficients, fit.residuals, fit.rss, true))
            }
            LinearFitKind::Ridge { lambda } => {
                let fit = fit_ridge(x, n, p, y, lambda, &self.backend, &mut workspace.ols)
                    .map_err(EstimationError::from)?;
                Ok((fit.coefficients, fit.residuals, fit.rss, true))
            }
            LinearFitKind::Lasso { lambda } => {
                let fit = fit_lasso(x, n, p, y, lambda, &LassoOptions::default())
                    .map_err(EstimationError::from)?;
                let mut residuals = vec![0.0; n];
                let mut rss = 0.0;
                for r in 0..n {
                    let mut pred = 0.0;
                    for c in 0..p {
                        pred += x[c * n + r] * fit.coefficients[c];
                    }
                    let e = y[r] - pred;
                    residuals[r] = e;
                    rss += e * e;
                }
                // Permanent policy: no analytic SE for Lasso (bootstrap only).
                Ok((fit.coefficients, residuals, rss, false))
            }
            LinearFitKind::Huber { c } => {
                let opts = MEstimateOptions { c, ..MEstimateOptions::default() };
                let fit = fit_huber_m(x, n, p, y, &opts, &self.backend, &mut workspace.ols)
                    .map_err(EstimationError::from)?;
                let mut residuals = vec![0.0; n];
                let mut rss = 0.0;
                for r in 0..n {
                    let mut pred = 0.0;
                    for c in 0..p {
                        pred += x[c * n + r] * fit.coefficients[c];
                    }
                    let e = y[r] - pred;
                    residuals[r] = e;
                    rss += e * e;
                }
                Ok((fit.coefficients, residuals, rss, true))
            }
        }
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        t_col: usize,
    ) -> Result<crate::util::BootstrapSeResult, EstimationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let mut x_boot = vec![0.0; n * p];
        let mut y_boot = vec![0.0; n];
        crate::util::bootstrap_se(self.bootstrap_replicates, ctx, 0xA7E_u64, n, |idx| {
            for (r, &src) in idx.iter().enumerate() {
                y_boot[r] = problem.design.outcome[src];
                for c in 0..p {
                    x_boot[c * n + r] = problem.design.matrix[c * n + src];
                }
            }
            let coefs = match self.fit_kind {
                LinearFitKind::Ols => {
                    match self.backend.least_squares(&x_boot, n, p, &y_boot, &mut workspace.ols) {
                        Ok(fit) => fit.coefficients,
                        Err(_) => return Ok(None),
                    }
                }
                LinearFitKind::Ridge { lambda } => {
                    match fit_ridge(
                        &x_boot,
                        n,
                        p,
                        &y_boot,
                        lambda,
                        &self.backend,
                        &mut workspace.ols,
                    ) {
                        Ok(fit) => fit.coefficients,
                        Err(_) => return Ok(None),
                    }
                }
                LinearFitKind::Lasso { lambda } => {
                    match fit_lasso(&x_boot, n, p, &y_boot, lambda, &LassoOptions::default()) {
                        Ok(fit) => fit.coefficients,
                        Err(_) => return Ok(None),
                    }
                }
                LinearFitKind::Huber { c } => {
                    let opts = MEstimateOptions { c, ..MEstimateOptions::default() };
                    match fit_huber_m(
                        &x_boot,
                        n,
                        p,
                        &y_boot,
                        &opts,
                        &self.backend,
                        &mut workspace.ols,
                    ) {
                        Ok(fit) => fit.coefficients,
                        Err(_) => return Ok(None),
                    }
                }
            };
            Ok(Some(gcomp_or_coef_ate(problem, &coefs, t_col)?))
        })
    }
}

/// G-computation of μ(active,Z)−μ(control,Z) averaged under the target arm’s covariate law.
///
/// Under a linear main-effects model this equals `β_T · Δ` for every target, including ATT/ATC.
fn gcomp_or_coef_ate(
    problem: &PreparedEstimationProblem,
    coefficients: &[f64],
    t_col: usize,
) -> Result<f64, EstimationError> {
    match problem.target_population {
        TargetPopulation::AllObserved | TargetPopulation::Predicate(_) => {
            Ok(coefficients[t_col] * problem.treatment_delta)
        }
        TargetPopulation::Treated | TargetPopulation::Untreated => {
            let n = problem.design.nrows;
            let ncols = problem.design.ncols;
            let want_treated = matches!(problem.target_population, TargetPopulation::Treated);
            let mut sum = 0.0;
            let mut count = 0usize;
            for r in 0..n {
                let treated = problem.treatment[r] > 0.5;
                if treated != want_treated {
                    continue;
                }
                let mut pred_a = 0.0;
                let mut pred_c = 0.0;
                for c in 0..ncols {
                    let x = if c == t_col {
                        (problem.active, problem.control)
                    } else {
                        let v = problem.design.matrix[c * n + r];
                        (v, v)
                    };
                    pred_a += coefficients[c] * x.0;
                    pred_c += coefficients[c] * x.1;
                }
                sum += pred_a - pred_c;
                count += 1;
            }
            if count == 0 {
                return Err(EstimationError::data_msg(
                    "target population left no rows for g-computation",
                ));
            }
            Ok(sum / count as f64)
        }
        _ => Err(EstimationError::TargetPopulation),
    }
}

pub(crate) fn intervention_f64(intervention: &Intervention) -> Result<f64, EstimationError> {
    match intervention {
        Intervention::Set { value, .. } => value.as_f64().ok_or_else(|| {
            EstimationError::unsupported(" linear adjustment requires numeric treatment levels")
        }),
        _ => Err(EstimationError::unsupported(" linear adjustment requires Set interventions")),
    }
}

fn analytic_se_treatment(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    sigma2: f64,
) -> f64 {
    let mut xtx = vec![0.0; ncols * ncols];
    form_xtx(x_colmajor, nrows, ncols, &mut xtx);
    let Some(inv) = invert_square(&xtx, ncols) else {
        return f64::NAN;
    };
    (sigma2 * inv[t_col * ncols + t_col].max(0.0)).sqrt()
}

impl crate::estimator::Estimator<TabularData> for LinearAdjustmentAte {
    type Fit = EffectEstimate;

    fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        _ctx: &ExecutionContext,
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        Self::prepare(self, data, estimand, query)
    }

    fn fit(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<Self::Fit, EstimationError> {
        Self::fit(self, problem, workspace, ctx, AssumptionSet::new())
    }
}

impl crate::estimator::TabularAteEstimator for LinearAdjustmentAte {}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, TargetPopulation, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_expr::ExprId;
    use causal_expr::IdentifiedEstimand;

    use super::*;

    fn toy() -> (TabularData, IdentifiedEstimand) {
        let n = 100usize;
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
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + z[i]).collect();
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

    #[test]
    fn overlap_report_from_propensities() {
        let ps = [0.1, 0.5, 0.9];
        let ws = [10.0, 2.0, 1.111];
        let report = OverlapReport::from_propensities(
            &ps,
            Some(&ws),
            OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: Some(0.05) },
        );
        assert!((report.propensity_min - 0.1).abs() < 1e-12);
        assert!((report.propensity_max - 0.9).abs() < 1e-12);
        assert_eq!(report.extreme_weight_count, 0); // none strictly > 10
        assert_eq!(report.clip, Some(0.05));
        assert!((report.excluded_fraction - 0.0).abs() < 1e-12);
        assert!((report.target_population_support - 1.0).abs() < 1e-12);
        assert_eq!(report.excluded_regions.len(), 2);
        assert!((report.excluded_regions[0].high - 0.05).abs() < 1e-12);
        let sens = report.clip_sensitivity.as_ref().expect("clip sensitivity");
        assert!(sens.thresholds.len() >= 2);
        assert_eq!(sens.ess.len(), sens.thresholds.len());
    }

    #[test]
    fn rejects_require_diagnostics_on_linear_path() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte {
            overlap: OverlapPolicy::require_diagnostics(),
            ..LinearAdjustmentAte::new()
        };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn recovers_ate_two() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 50, ..LinearAdjustmentAte::new() };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-8);
        assert!(effect.se_bootstrap.is_some());
    }

    #[test]
    fn scales_ate_by_level_delta() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            2.0,
        );
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        // β_T ≈ 2, delta = 2 → ATE ≈ 4
        assert!((effect.ate - 4.0).abs() < 1e-8);
    }

    #[test]
    fn recovers_att_via_gcomp() {
        let (data, estimand) = toy();
        let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let effect = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-8, "att={}", effect.ate);
    }

    #[test]
    fn hc_sandwich_kinds_yield_finite_se() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        for kind in [
            AnalyticSeKind::Hc0,
            AnalyticSeKind::Hc2,
            AnalyticSeKind::Hc3,
            AnalyticSeKind::NeweyWest { lag: 2 },
        ] {
            let est = LinearAdjustmentAte {
                bootstrap_replicates: 0,
                se_kind: kind,
                ..LinearAdjustmentAte::new()
            };
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            let mut ws = EstimationWorkspace::default();
            let effect = est
                .fit(&prep, &mut ws, &ExecutionContext::for_tests(1), AssumptionSet::new())
                .unwrap();
            assert!(effect.se_analytic.is_finite() && effect.se_analytic > 0.0, "{kind:?}");
        }
    }

    #[test]
    fn ridge_lasso_huber_fit_kinds_recover_ate() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        for kind in [
            LinearFitKind::Ridge { lambda: 1e-3 },
            LinearFitKind::Lasso { lambda: 1e-4 },
            LinearFitKind::Huber { c: 1.345 },
        ] {
            let est = LinearAdjustmentAte {
                bootstrap_replicates: 0,
                fit_kind: kind,
                ..LinearAdjustmentAte::new()
            };
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            let mut ws = EstimationWorkspace::default();
            let effect = est
                .fit(&prep, &mut ws, &ExecutionContext::for_tests(2), AssumptionSet::new())
                .unwrap();
            assert!(effect.ate.is_finite(), "{kind:?}");
            assert!((effect.ate - 2.0).abs() < 0.05, "ate={} kind={kind:?}", effect.ate);
        }
    }

    #[test]
    fn lasso_analytic_se_nan_bootstrap_finite() {
        let (data, estimand) = toy();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let no_boot = LinearAdjustmentAte {
            bootstrap_replicates: 0,
            fit_kind: LinearFitKind::Lasso { lambda: 1e-4 },
            ..LinearAdjustmentAte::new()
        };
        let prep = no_boot.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let effect = no_boot
            .fit(&prep, &mut ws, &ExecutionContext::for_tests(3), AssumptionSet::new())
            .unwrap();
        assert!(effect.se_analytic.is_nan(), "lasso se_analytic={}", effect.se_analytic);
        assert!(effect.se_bootstrap.is_none());

        let with_boot = LinearAdjustmentAte {
            bootstrap_replicates: 40,
            fit_kind: LinearFitKind::Lasso { lambda: 1e-4 },
            ..LinearAdjustmentAte::new()
        };
        let effect = with_boot
            .fit(&prep, &mut ws, &ExecutionContext::for_tests(3), AssumptionSet::new())
            .unwrap();
        assert!(effect.se_analytic.is_nan());
        let boot = effect.se_bootstrap.expect("bootstrap SE");
        assert!(boot.is_finite() && boot >= 0.0, "se_bootstrap={boot}");
    }
}
