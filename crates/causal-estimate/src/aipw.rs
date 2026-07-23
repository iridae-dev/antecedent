//! Augmented inverse-probability weighting (AIPW / doubly robust) ATE estimator .
//!
//! Combines an outcome regression (μ0(Z), μ1(Z), fit separately per treatment arm) with
//! inverse-probability weighting of the residuals, so the estimator is consistent if *either*
//! the propensity model or the outcome model is correctly specified:
//!
//! ```text
//! ψ_i = (μ1(Z_i) − μ0(Z_i)) + T_i/e_i · (Y_i − μ1(Z_i)) − (1−T_i)/(1−e_i) · (Y_i − μ0(Z_i))
//! ATE = mean(ψ)
//! ```
//!
//! When the overlap policy sets a trim threshold, units whose raw propensity falls outside
//! `[trim, 1 − trim]` are excluded from the outcome-model fits and the ψ average (the estimand
//! becomes the common-support population, matching the overlap report).
//!
//! Positivity is mandatory — [`OverlapPolicy::ExplicitOverride`] is refused, matching the other
//! propensity-based estimators in [`crate::propensity`].
//!
//! Bootstrap standard errors **refit both the propensity model and the two outcome models on
//! every resample**, propagating first-stage estimation uncertainty rather than reusing the
//! point-estimate nuisance fits (see [`crate::propensity`] module docs for the same rationale).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::similar_names,
    clippy::needless_range_loop
)]

use causal_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, PopulationRegistry, TargetPopulation,
};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_stats::{
    DenseLinearAlgebra, FaerBackend, GlmOptions, LeastSquaresWorkspace, PropensityWorkspace,
    fit_propensity,
};

use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::overlap::{OverlapPolicy, OverlapReport};
use crate::propensity::{
    PreparedPropensityProblem, PropensityModel, clamp_scores, clip_of, default_propensity_overlap,
    gather, prepare_propensity_problem_with_registry, split_by_treatment, trim_of,
    trim_retained_rows,
};
use crate::se::AnalyticSeKind;
use crate::util::{BootstrapSeResult, bootstrap_se, stats_err};

/// Reusable scratch for AIPW point-estimate and bootstrap fits.
///
/// Outcome regressions are refit per treatment arm on every call; the design/outcome buffers
/// below are reused (grow-only) across bootstrap replicates to avoid per-replicate heap churn.
#[derive(Clone, Debug, Default)]
pub struct AipwWorkspace {
    /// Logistic IRLS scratch reused across propensity refits.
    pub propensity: PropensityWorkspace,
    /// OLS scratch reused across both arms' outcome-model refits.
    pub outcome: LeastSquaresWorkspace,
    treated_design: Vec<f64>,
    treated_outcome: Vec<f64>,
    control_design: Vec<f64>,
    control_outcome: Vec<f64>,
    mu0: Vec<f64>,
    mu1: Vec<f64>,
    psi: Vec<f64>,
}

/// Doubly robust (AIPW) ATE estimator.
///
// supports [`TargetPopulation::AllObserved`] only; ATT/ATC are rejected with a clear
/// [`EstimationError::Unsupported`]. Positivity is mandatory: [`OverlapPolicy::ExplicitOverride`]
/// is refused.
#[derive(Clone, Debug)]
pub struct AipwAte {
    /// Dense linear-algebra backend used for the propensity IRLS fit and outcome OLS fits.
    pub backend: FaerBackend,
    /// Bootstrap replicates (0 = skip bootstrap).
    pub bootstrap_replicates: u32,
    /// Overlap policy; must be [`OverlapPolicy::RequireDiagnostics`].
    pub overlap: OverlapPolicy,
    /// GLM fitting options for the propensity model.
    pub glm_options: GlmOptions,
    /// Analytic SE kind (IID influence / HC1 scale / cluster).
    pub se_kind: AnalyticSeKind,
    /// Optional cluster ids for [`AnalyticSeKind::Cluster`] (aligned to prepared rows).
    pub cluster_ids: Option<Vec<u32>>,
    /// Optional bindings for named predicates / custom target distributions.
    pub population_registry: Option<PopulationRegistry>,
    /// Multiway cluster ids (one `Vec<u32>` per clustering dimension).
    pub multiway_ids: Option<Vec<Vec<u32>>>,
}

impl Default for AipwAte {
    fn default() -> Self {
        Self::new()
    }
}

impl AipwAte {
    /// Defaults: 200 bootstrap replicates, clip = 0.01, no trim.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: FaerBackend,
            bootstrap_replicates: 200,
            overlap: default_propensity_overlap(),
            glm_options: GlmOptions::default(),
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            population_registry: None,
            multiway_ids: None,
        }
    }

    /// Prepare the covariate design from tabular data, identified estimand, and query.
    ///
    /// Accepts `backdoor.adjustment` / `backdoor.efficient` estimands.
    ///
    /// # Errors
    ///
    /// Overlap policy is `ExplicitOverride`, incompatible estimand, unsupported query, or
    /// missing/invalid data columns.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        prepare_propensity_problem_with_registry(
            data,
            estimand,
            query,
            self.overlap,
            self.population_registry.as_ref(),
        )
    }

    /// Fit propensity + outcome nuisance models and compute the AIPW effect, with optional
    /// bootstrap.
    ///
    /// # Errors
    ///
    /// Target population other than `AllObserved`, empty treated/control arm, or GLM/OLS
    /// backend failure.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if !matches!(
            problem.target_population,
            TargetPopulation::AllObserved
                | TargetPopulation::Treated
                | TargetPopulation::Untreated
                | TargetPopulation::Predicate(_)
        ) {
            return Err(EstimationError::unsupported(
                "AIPW supports AllObserved, Treated, Untreated, or Predicate target populations",
            ));
        }

        let model = PropensityModel::fit(
            problem,
            &self.backend,
            &mut workspace.propensity,
            &self.glm_options,
        )?;
        // Trim on RAW scores (mirrors PropensityWeighting): units outside the common-support
        // band are excluded from the outcome-model fits and the ψ average — exactly the units
        // whose T/e and (1−T)/(1−e) terms explode. The estimand becomes the common-support
        // population, matching what the overlap report below claims.
        let retained = trim_retained_rows(&model.fit.scores, trim_of(problem.overlap))?;
        let ncols = problem.design_ncols;
        let (design_used, t_used, y_used, e_used) = match &retained {
            Some(idx) => {
                let mut design = Vec::new();
                select_rows_colmajor(
                    &problem.design_matrix,
                    problem.nrows,
                    ncols,
                    idx,
                    &mut design,
                );
                (
                    design,
                    gather(&problem.treatment, idx),
                    gather(&problem.outcome, idx),
                    gather(&model.clipped_scores, idx),
                )
            }
            None => (
                problem.design_matrix.to_vec(),
                problem.treatment.to_vec(),
                problem.outcome.to_vec(),
                model.clipped_scores.clone(),
            ),
        };
        let nrows = t_used.len();
        let (beta0, beta1) = fit_outcome_models(
            &design_used,
            nrows,
            ncols,
            &t_used,
            &y_used,
            self.backend,
            workspace,
        )?;
        predict_colmajor(&design_used, nrows, ncols, &beta0, &mut workspace.mu0);
        predict_colmajor(&design_used, nrows, ncols, &beta1, &mut workspace.mu1);
        aipw_psi(
            &t_used,
            &y_used,
            &e_used,
            &workspace.mu0,
            &workspace.mu1,
            &problem.target_population,
            &mut workspace.psi,
        )?;
        let n = workspace.psi.len() as f64;
        let ate = workspace.psi.iter().sum::<f64>() / n;
        let se_analytic = crate::se::influence_se_kind(
            self.se_kind,
            &workspace.psi,
            problem.nrows,
            self.cluster_ids.as_deref(),
            self.multiway_ids.as_deref(),
            retained.as_deref(),
        )?;

        let boot = if self.bootstrap_replicates == 0 {
            None
        } else {
            Some(self.bootstrap_se(problem, workspace, ctx)?)
        };

        let overlap_report =
            Some(OverlapReport::from_propensities(&model.fit.scores, None, problem.overlap));

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
            overlap_report,
            retained_memory_bytes: None,
        }
        .with_bootstrap(boot))
    }

    fn bootstrap_se(
        &self,
        problem: &PreparedPropensityProblem,
        workspace: &mut AipwWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<BootstrapSeResult, EstimationError> {
        let clip = clip_of(problem.overlap);
        let trim = trim_of(problem.overlap);
        let n = problem.nrows;
        let ncols = problem.design_ncols;
        let mut x_boot = vec![0.0; n * ncols];
        let mut t_boot = vec![0.0; n];
        let mut y_boot = vec![0.0; n];
        bootstrap_se(self.bootstrap_replicates, ctx, 0xA1D0_u64, n, |idx| {
            for (r, &src) in idx.iter().enumerate() {
                t_boot[r] = problem.treatment[src];
                y_boot[r] = problem.outcome[src];
                for c in 0..ncols {
                    x_boot[c * n + r] = problem.design_matrix[c * n + src];
                }
            }
            let Ok(fit) = fit_propensity(
                &x_boot,
                n,
                ncols,
                &t_boot,
                &self.backend,
                &mut workspace.propensity,
                &self.glm_options,
            ) else {
                return Ok(None);
            };
            let raw = fit.scores;
            let mut e = raw.clone();
            if let Some(c) = clip {
                clamp_scores(&mut e, c);
            }
            let Ok(retained) = trim_retained_rows(&raw, trim) else {
                return Ok(None);
            };
            let (design_used, t_used, y_used, e_used) = match &retained {
                Some(rows) => {
                    let mut design = Vec::new();
                    select_rows_colmajor(&x_boot, n, ncols, rows, &mut design);
                    (design, gather(&t_boot, rows), gather(&y_boot, rows), gather(&e, rows))
                }
                None => (x_boot.clone(), t_boot.clone(), y_boot.clone(), e),
            };
            let nrows = t_used.len();
            let Ok((beta0, beta1)) = fit_outcome_models(
                &design_used,
                nrows,
                ncols,
                &t_used,
                &y_used,
                self.backend,
                workspace,
            ) else {
                return Ok(None);
            };
            predict_colmajor(&design_used, nrows, ncols, &beta0, &mut workspace.mu0);
            predict_colmajor(&design_used, nrows, ncols, &beta1, &mut workspace.mu1);
            if aipw_psi(
                &t_used,
                &y_used,
                &e_used,
                &workspace.mu0,
                &workspace.mu1,
                &problem.target_population,
                &mut workspace.psi,
            )
            .is_err()
            {
                return Ok(None);
            }
            let m = workspace.psi.len() as f64;
            Ok(Some(workspace.psi.iter().sum::<f64>() / m))
        })
    }
}

/// Extract rows `idx` from a column-major `nrows × ncols` matrix into a fresh column-major
/// `idx.len() × ncols` buffer.
fn select_rows_colmajor(
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    idx: &[usize],
    out: &mut Vec<f64>,
) {
    let m = idx.len();
    out.clear();
    out.resize(m * ncols, 0.0);
    for c in 0..ncols {
        let src_base = c * nrows;
        let dst_base = c * m;
        for (r, &i) in idx.iter().enumerate() {
            out[dst_base + r] = matrix[src_base + i];
        }
    }
}

fn select_values(values: &[f64], idx: &[usize], out: &mut Vec<f64>) {
    out.clear();
    out.extend(idx.iter().map(|&i| values[i]));
}

/// Fit separate OLS outcome models on the control (`T=0`) and treated (`T=1`) arms of
/// `design_matrix` (column-major `[1 | Z…]`), returning `(beta0, beta1)`.
///
/// # Errors
///
/// Empty treated/control arm, or an OLS backend failure (e.g. rank deficiency within an arm).
fn fit_outcome_models(
    design_matrix: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    outcome: &[f64],
    backend: FaerBackend,
    workspace: &mut AipwWorkspace,
) -> Result<(Vec<f64>, Vec<f64>), EstimationError> {
    let (treated_idx, control_idx) = split_by_treatment(treatment);
    if treated_idx.is_empty() || control_idx.is_empty() {
        return Err(EstimationError::data_msg(
            "AIPW outcome regression requires both treated and control rows",
        ));
    }

    select_rows_colmajor(design_matrix, nrows, ncols, &control_idx, &mut workspace.control_design);
    select_values(outcome, &control_idx, &mut workspace.control_outcome);
    let fit0 = backend
        .least_squares(
            &workspace.control_design,
            control_idx.len(),
            ncols,
            &workspace.control_outcome,
            &mut workspace.outcome,
        )
        .map_err(stats_err)?;
    let beta0 = fit0.coefficients;

    select_rows_colmajor(design_matrix, nrows, ncols, &treated_idx, &mut workspace.treated_design);
    select_values(outcome, &treated_idx, &mut workspace.treated_outcome);
    let fit1 = backend
        .least_squares(
            &workspace.treated_design,
            treated_idx.len(),
            ncols,
            &workspace.treated_outcome,
            &mut workspace.outcome,
        )
        .map_err(stats_err)?;
    let beta1 = fit1.coefficients;

    Ok((beta0, beta1))
}

/// Predict `design · coef` for every row of a column-major `nrows × ncols` design.
fn predict_colmajor(
    design_matrix: &[f64],
    nrows: usize,
    ncols: usize,
    coef: &[f64],
    out: &mut Vec<f64>,
) {
    out.clear();
    out.resize(nrows, 0.0);
    for (r, pred) in out.iter_mut().enumerate() {
        let mut s = 0.0;
        for c in 0..ncols {
            s += design_matrix[c * nrows + r] * coef[c];
        }
        *pred = s;
    }
}

/// Compute AIPW per-unit IF values for ATE / ATT / ATC.
fn aipw_psi(
    treatment: &[f64],
    outcome: &[f64],
    propensity: &[f64],
    mu0: &[f64],
    mu1: &[f64],
    target: &TargetPopulation,
    out: &mut Vec<f64>,
) -> Result<(), EstimationError> {
    out.clear();
    out.reserve(treatment.len());
    match target {
        TargetPopulation::AllObserved | TargetPopulation::Predicate(_) => {
            for (((&t, &y), &e), (&m0, &m1)) in
                treatment.iter().zip(outcome).zip(propensity).zip(mu0.iter().zip(mu1))
            {
                let augmented = (m1 - m0) + (t / e) * (y - m1) - ((1.0 - t) / (1.0 - e)) * (y - m0);
                out.push(augmented);
            }
        }
        TargetPopulation::Treated => {
            let pi = treatment.iter().filter(|&&t| t > 0.5).count() as f64 / treatment.len() as f64;
            if pi <= 0.0 {
                return Err(EstimationError::data_msg("ATT requires treated units"));
            }
            for (((&t, &y), &e), (&m0, &m1)) in
                treatment.iter().zip(outcome).zip(propensity).zip(mu0.iter().zip(mu1))
            {
                let aug = (t / pi) * (m1 - m0) + (t / pi) * (y - m1) / e
                    - ((1.0 - t) / pi) * (e / (1.0 - e)) * (y - m0);
                out.push(aug);
            }
        }
        TargetPopulation::Untreated => {
            let pi0 =
                treatment.iter().filter(|&&t| t <= 0.5).count() as f64 / treatment.len() as f64;
            if pi0 <= 0.0 {
                return Err(EstimationError::data_msg("ATC requires control units"));
            }
            for (((&t, &y), &e), (&m0, &m1)) in
                treatment.iter().zip(outcome).zip(propensity).zip(mu0.iter().zip(mu1))
            {
                let aug = ((1.0 - t) / pi0) * (m1 - m0) + (t / pi0) * ((1.0 - e) / e) * (y - m1)
                    - ((1.0 - t) / pi0) * (y - m0) / (1.0 - e);
                out.push(aug);
            }
        }
        _ => {
            return Err(EstimationError::unsupported(
                "AIPW unsupported target population for IF construction",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::many_single_char_names, clippy::float_cmp)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet,
        TargetPopulation, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_expr::ExprId;
    use causal_expr::IdentifiedEstimand;

    use super::*;
    use crate::overlap::OverlapPolicy;
    use causal_kernels::standard_normal;

    /// Confounded SCM: `Z ~ N(0,1)`, `T ~ Bernoulli(logit(-0.5 + Z))`, `Y = 2T + Z + noise`.
    /// True ATE = 2. Matches the propensity-estimator test fixture (`crate::propensity`).
    fn confounded_scm(n: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
        let (t, y, z) = confounded_columns(n, seed);
        build_dataset(t, y, z)
    }

    fn confounded_columns(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x1234_u64);

        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let logit = -0.5 + zi;
            let p = 1.0 / (1.0 + (-logit).exp());
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            let noise = standard_normal(&mut rng) * 0.5;
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + noise;
        }
        (t, y, z)
    }

    fn build_dataset(t: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> (TabularData, IdentifiedEstimand) {
        let n = t.len();
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
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        (TabularData::new(storage), estimand)
    }

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(7)
    }

    #[test]
    fn aipw_recovers_ate_two() {
        let (data, estimand) = confounded_scm(800, 1);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte { bootstrap_replicates: 30, ..AipwAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = AipwWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
        assert!(effect.se_bootstrap.is_some());
        assert!(effect.overlap_report.is_some());
    }

    #[test]
    fn aipw_rejects_explicit_override() {
        let (data, estimand) = confounded_scm(200, 2);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte { overlap: OverlapPolicy::ExplicitOverride, ..AipwAte::new() };
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(matches!(err, EstimationError::Overlap { .. }));
    }

    #[test]
    fn aipw_recovers_att_two() {
        let (data, estimand) = confounded_scm(1_200, 3);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_target_population(TargetPopulation::Treated);
        let est = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = AipwWorkspace::default();
        let fit = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((fit.ate - 2.0).abs() < 0.35, "att={}", fit.ate);
    }

    #[test]
    fn aipw_hc1_multiway_newey_west_finite_se() {
        let (data, estimand) = confounded_scm(400, 8);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let n = 400;
        let dim_a: Vec<u32> = (0..n).map(|i| u32::try_from(i % 20).unwrap_or(0)).collect();
        let dim_b: Vec<u32> = (0..n).map(|i| u32::try_from(i % 15).unwrap_or(0)).collect();
        for kind in
            [AnalyticSeKind::Hc1, AnalyticSeKind::Multiway, AnalyticSeKind::NeweyWest { lag: 2 }]
        {
            let est = AipwAte {
                bootstrap_replicates: 0,
                se_kind: kind,
                multiway_ids: Some(vec![dim_a.clone(), dim_b.clone()]),
                ..AipwAte::new()
            };
            let prep = est.prepare(&data, &estimand, &query).unwrap();
            let mut ws = AipwWorkspace::default();
            let fit = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
            assert!(fit.se_analytic.is_finite() && fit.se_analytic > 0.0, "kind={kind:?}");
        }
    }

    #[test]
    fn aipw_trim_excludes_extreme_propensity_unit() {
        // One treated outlier with z = -8 (raw propensity ≈ 2e-4) and y = 1000: its clipped
        // T/e term is ~100 · (1000 − μ1), which wrecks the untrimmed ψ average. Trimming on
        // the raw score must exclude it.
        let (mut t, mut y, mut z) = confounded_columns(800, 5);
        t.push(1.0);
        y.push(1000.0);
        z.push(-8.0);
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));

        let untrimmed = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let trimmed = AipwAte {
            overlap: OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.02) },
            ..untrimmed.clone()
        };
        let mut ws = AipwWorkspace::default();
        let prep = untrimmed.prepare(&data, &estimand, &query).unwrap();
        let raw = untrimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        let prep = trimmed.prepare(&data, &estimand, &query).unwrap();
        let clean = trimmed.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();

        assert!((raw.ate - 2.0).abs() > 1.0, "outlier should distort untrimmed ate={}", raw.ate);
        assert!((clean.ate - 2.0).abs() < 0.35, "trimmed ate={}", clean.ate);
        let report = clean.overlap_report.as_ref().unwrap();
        assert!(report.excluded_fraction > 0.0, "trim must report exclusions");
    }

    #[test]
    fn aipw_works_with_efficient_backdoor_estimand() {
        let (data, mut estimand) = confounded_scm(800, 4);
        estimand.method = Arc::from("backdoor.efficient");
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = AipwWorkspace::default();
        let effect = est.fit(&prep, &mut ws, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
    }
}
