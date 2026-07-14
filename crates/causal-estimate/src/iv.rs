//! Instrumental-variable estimators: Wald ratio and two-stage least squares .
//!
//! Both estimators require an `"iv"` estimand with a non-empty
//! [`IdentifiedEstimand::instruments`] slice (see `causal_identify::iv`). Positivity is not
//! meaningful for IV — it is not a propensity-based method — so
//! [`OverlapPolicy::ExplicitOverride`] is the only supported policy, matching
//! [`crate::adjustment::LinearAdjustmentAte`].
//!
//! [`WaldIv`] implements the simple ratio-of-differences estimator for a single binary
//! instrument. [`TwoStageLeastSquares`] handles one or more instruments (continuous or binary)
//! and optional exogenous covariates via `causal_stats::fit_2sls`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::manual_memcpy
)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation, VariableId,
};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_stats::{FaerBackend, LeastSquaresWorkspace, fit_2sls, form_xtx, invert_square};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::util::{sample_std, stats_err};

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
    if estimand.method_kind().ok() != Some(causal_expr::EstimandMethod::Iv) {
        return Err(EstimationError::IncompatibleEstimand {
            message: "IV estimators expect an \"iv\" estimand",
        });
    }
    if estimand.instruments.is_empty() {
        return Err(EstimationError::IncompatibleEstimand {
            message: "IV estimators require a non-empty instrument set",
        });
    }
    query.validate().map_err(|e| EstimationError::UnsupportedQuery(e.to_string()))?;
    if !query.effect_modifiers.is_empty() {
        return Err(EstimationError::UnsupportedQuery(
            "IV estimators do not support effect modifiers".into(),
        ));
    }
    if query.target_population != TargetPopulation::AllObserved {
        return Err(EstimationError::UnsupportedQuery(
            "IV estimators only support TargetPopulation::AllObserved".into(),
        ));
    }
    let treatment = query.treatment;
    let outcome = query.outcome;
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    let treatment_delta = active - control;
    if treatment_delta == 0.0 {
        return Err(EstimationError::UnsupportedQuery(
            "active and control treatment levels must differ".into(),
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
        Self { bootstrap_replicates: 200, overlap: OverlapPolicy::ExplicitOverride }
    }

    /// Prepare the instrument/outcome/treatment design.
    ///
    /// # Errors
    ///
    /// See [`prepare_iv_problem`].
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
            return Err(EstimationError::UnsupportedQuery(
                "WaldIv requires exactly one instrument; use TwoStageLeastSquares for multiple instruments".into(),
            ));
        }
        let n = problem.nrows;
        let z: Vec<f64> = (0..n).map(|r| problem.instruments_matrix[n + r]).collect();
        if !z.iter().all(|&v| v == 0.0 || v == 1.0) {
            return Err(EstimationError::UnsupportedQuery(
                "WaldIv requires a binary (0/1) instrument; use TwoStageLeastSquares for continuous instruments".into(),
            ));
        }

        let wald = wald_ratio(&z, &problem.treatment, &problem.outcome)?;
        let ate = wald.ratio * problem.treatment_delta;
        let se_analytic = wald.se * problem.treatment_delta.abs();

        let se_bootstrap = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, &z, ctx))
        };

        Ok(EffectEstimate {
            ate,
            se_analytic,
            se_bootstrap,
            assumptions,
            overlap: problem.overlap,
            overlap_report: None,
            retained_memory_bytes: None,
        })
    }

    fn bootstrap_se(&self, problem: &PreparedIvProblem, z: &[f64], ctx: &ExecutionContext) -> f64 {
        let mut rng = ctx.rng.stream(0x5A1D_u64);
        let n = problem.nrows;
        let mut z_boot = vec![0.0; n];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        let mut estimates = Vec::with_capacity(self.bootstrap_replicates as usize);
        for _ in 0..self.bootstrap_replicates {
            for r in 0..n {
                let idx = (rng.next_u64() as usize) % n;
                z_boot[r] = z[idx];
                t_boot[r] = problem.treatment[idx];
                y_boot[r] = problem.outcome[idx];
            }
            if let Ok(w) = wald_ratio(&z_boot, &t_boot, &y_boot) {
                estimates.push(w.ratio * problem.treatment_delta);
            }
        }
        if estimates.len() < 2 {
            return f64::NAN;
        }
        sample_std(&estimates)
    }
}

struct WaldResult {
    ratio: f64,
    se: f64,
}

/// Ratio-of-differences point estimate with a delta-method SE that ignores sampling
/// uncertainty in the first-stage denominator (a common simplification; the bootstrap SE
/// above captures the full picture by resampling the whole ratio).
fn wald_ratio(z: &[f64], t: &[f64], y: &[f64]) -> Result<WaldResult, EstimationError> {
    let (mut sy1, mut sy0, mut sy1sq, mut sy0sq, mut st1, mut st0) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut n1, mut n0) = (0usize, 0usize);
    for i in 0..z.len() {
        if z[i] > 0.5 {
            sy1 += y[i];
            sy1sq += y[i] * y[i];
            st1 += t[i];
            n1 += 1;
        } else {
            sy0 += y[i];
            sy0sq += y[i] * y[i];
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
    let var_y1 = (sy1sq / n1f - mean_y1 * mean_y1).max(0.0);
    let var_y0 = (sy0sq / n0f - mean_y0 * mean_y0).max(0.0);
    let se = (var_y1 / n1f + var_y0 / n0f).sqrt() / denom.abs();
    Ok(WaldResult { ratio, se })
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
        }
    }

    /// Prepare the instrument/exogenous/outcome/treatment design.
    ///
    /// # Errors
    ///
    /// See [`prepare_iv_problem`].
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
        let se_coef = analytic_se_2sls(
            &fit.fitted_endogenous,
            &problem.exogenous_matrix,
            problem.nrows,
            problem.x_ncols,
            fit.structural_rss,
        );
        let se_analytic = se_coef * problem.treatment_delta.abs();

        let se_bootstrap = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, workspace, ctx))
        };

        Ok(EffectEstimate {
            ate,
            se_analytic,
            se_bootstrap,
            assumptions,
            overlap: problem.overlap,
            overlap_report: None,
            retained_memory_bytes: None,
        })
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedIvProblem,
        workspace: &mut TwoStageLeastSquaresWorkspace,
        ctx: &ExecutionContext,
    ) -> f64 {
        let mut rng = ctx.rng.stream(0x25D5_u64);
        let n = problem.nrows;
        let zc = problem.z_ncols;
        let xc = problem.x_ncols;
        let mut z_boot = vec![0.0; n * zc];
        let mut x_boot = vec![0.0; n * xc];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        let mut estimates = Vec::with_capacity(self.bootstrap_replicates as usize);
        for _ in 0..self.bootstrap_replicates {
            for r in 0..n {
                let idx = (rng.next_u64() as usize) % n;
                t_boot[r] = problem.treatment[idx];
                y_boot[r] = problem.outcome[idx];
                for c in 0..zc {
                    z_boot[c * n + r] = problem.instruments_matrix[c * n + idx];
                }
                for c in 0..xc {
                    x_boot[c * n + r] = problem.exogenous_matrix[c * n + idx];
                }
            }
            let Ok(fit) = fit_2sls(
                &z_boot[n..],
                n,
                zc - 1,
                &t_boot,
                &x_boot,
                xc,
                &y_boot,
                &self.backend,
                &mut workspace.ols,
            ) else {
                continue;
            };
            estimates.push(fit.second_stage.coefficients[0] * problem.treatment_delta);
        }
        if estimates.len() < 2 {
            return f64::NAN;
        }
        sample_std(&estimates)
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
    let mut xtx = vec![0.0; ncols * ncols];
    form_xtx(&x2, nrows, ncols, &mut xtx);
    let Some(inv) = invert_square(&xtx, ncols) else {
        return f64::NAN;
    };
    let sigma2 = structural_rss / (nrows as f64 - ncols as f64).max(1.0);
    (sigma2 * inv[0].max(0.0)).sqrt()
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        AverageEffectQuery, CausalRng, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
        RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_expr::ExprId;
    use causal_expr::IdentifiedEstimand;

    use super::*;
    use crate::overlap::OverlapPolicy;

    fn standard_normal(rng: &mut CausalRng) -> f64 {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

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
        assert!(matches!(err, EstimationError::UnsupportedQuery(_)));
    }
}
