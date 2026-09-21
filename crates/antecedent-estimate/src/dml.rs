//! Partially linear DML and cross-fitted AIPW via `antecedent-learn`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_pass_by_value, clippy::needless_range_loop, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_learn::{LearnerSpec, LogisticSpec, PredictionTask, RidgeSpec, diagnose};

use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::learn_nuisance::{
    aipw_scores, cached_aipw_nuisances, clip_propensity, cross_fit_nuisance, iid_se,
};
use crate::overlap::{IpwTarget, OverlapPolicy, OverlapReport};
use crate::prepare::{require_adjustment_shaped, validate_ate_query_with_targets};
use crate::propensity::{
    PreparedPropensityProblem, clip_of, default_propensity_overlap,
    prepare_propensity_problem_with_registry, trim_of, trim_retained_rows,
};
use crate::se::AnalyticSeKind;

/// Orthogonal score used by [`DmlAte`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DmlScore {
    /// Binary-treatment AIPW from OOF `μ₀`, `μ₁`, `ê`.
    Aipw,
    /// Robinson partially linear residual-on-residual score.
    PartiallyLinear,
}

impl DmlScore {
    /// Parse `aipw` / `partially_linear`.
    ///
    /// # Errors
    ///
    /// Unknown score name.
    pub fn parse(key: &str) -> Result<Self, EstimationError> {
        match key {
            "aipw" => Ok(Self::Aipw),
            "partially_linear" => Ok(Self::PartiallyLinear),
            _ => Err(EstimationError::data_msg(format!("unknown DML score {key:?}"))),
        }
    }
}

/// Cross-fitted DML / AIPW average-treatment-effect estimator.
#[derive(Clone, Debug, PartialEq)]
pub struct DmlAte {
    /// Cross-fit folds (seeded, arm-stratified plan over the distinct units).
    pub folds: usize,
    /// Outcome nuisance spec.
    pub outcome: LearnerSpec,
    /// Treatment nuisance spec.
    pub treatment: LearnerSpec,
    /// Orthogonal score.
    pub score: DmlScore,
    /// Overlap / positivity policy.
    pub overlap: OverlapPolicy,
}

impl Default for DmlAte {
    fn default() -> Self {
        Self::new()
    }
}

impl DmlAte {
    /// Ridge outcome, logistic treatment, AIPW score, five folds.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            folds: 5,
            outcome: LearnerSpec::Ridge(RidgeSpec { lambda: 1.0 }),
            treatment: LearnerSpec::Logistic(LogisticSpec { ridge_lambda: 0.0 }),
            score: DmlScore::Aipw,
            overlap: default_propensity_overlap(),
        }
    }

    /// Adapt one shared spec to each nuisance task at configuration time.
    #[must_use]
    pub const fn with_learner(mut self, spec: LearnerSpec) -> Self {
        self.outcome = spec.for_task(PredictionTask::Regression);
        self.treatment = spec.for_task(PredictionTask::BinaryProbability);
        self
    }

    /// Outcome nuisance.
    #[must_use]
    pub const fn with_outcome(mut self, spec: LearnerSpec) -> Self {
        self.outcome = spec;
        self
    }

    /// Treatment nuisance.
    #[must_use]
    pub const fn with_treatment(mut self, spec: LearnerSpec) -> Self {
        self.treatment = spec;
        self
    }

    /// Orthogonal score.
    #[must_use]
    pub const fn with_score(mut self, score: DmlScore) -> Self {
        self.score = score;
        self
    }

    /// Fold count.
    #[must_use]
    pub const fn with_folds(mut self, folds: usize) -> Self {
        self.folds = folds;
        self
    }

    /// Overlap policy.
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Prepare `[1|Z]` design, binary treatment, and outcome.
    ///
    /// # Errors
    ///
    /// Non-adjustment estimand, unsupported query, non-binary treatment,
    /// or explicit overlap override.
    pub fn prepare(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<PreparedPropensityProblem, EstimationError> {
        require_adjustment_shaped(estimand, "DML requires an adjustment-shaped estimand")?;
        validate_ate_query_with_targets(query)?;
        prepare_propensity_problem_with_registry(data, estimand, query, self.overlap, None)
    }

    /// Fit OOF nuisances and the orthogonal score.
    ///
    /// # Errors
    ///
    /// Empty folds, learner failure, or a target other than `AllObserved`.
    pub fn fit(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        if !matches!(problem.target_population, TargetPopulation::AllObserved) {
            return Err(EstimationError::TargetPopulation);
        }
        if self.folds < 2 {
            return Err(EstimationError::data_msg("DML requires at least two folds"));
        }
        if matches!(self.score, DmlScore::Aipw)
            && problem.treatment.iter().any(|&ti| ti.abs() > 1e-12 && (ti - 1.0).abs() > 1e-12)
        {
            return Err(EstimationError::data_msg(
                "DML AIPW score requires a binary treatment coded 0/1",
            ));
        }
        match self.score {
            DmlScore::Aipw => self.fit_aipw(problem, ctx, assumptions),
            DmlScore::PartiallyLinear => self.fit_plr(problem, ctx, assumptions),
        }
    }

    fn fit_aipw(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let nuisance =
            cached_aipw_nuisances(problem, self.outcome, self.treatment, self.folds, ctx)?;
        let (mu0, mu1, raw_e, treat) = nuisance.as_ref();
        let mut ehat = raw_e.clone();
        clip_propensity(&mut ehat, clip_of(problem.overlap));
        let phi =
            aipw_scores(problem.treatment.as_ref(), problem.outcome.as_ref(), &ehat, mu0, mu1);
        let yhat: Vec<f64> = problem
            .treatment
            .iter()
            .zip(mu0)
            .zip(mu1)
            .map(|((&ti, &m0), &m1)| if ti > 0.5 { m1 } else { m0 })
            .collect();
        let outcome_diag = diagnose(PredictionTask::Regression, problem.outcome.as_ref(), &yhat);
        let mut effect = finish_dml(
            &phi,
            problem,
            assumptions,
            treat.validation.logloss,
            outcome_diag.r2,
            raw_e.clone(),
        )?;
        effect.crossfit_folds = Some(self.folds);
        effect.crossfit_seed = Some(ctx.rng.master_seed());
        effect.learner_provenance = treat.model_provenance.clone();
        Ok(effect)
    }

    fn fit_plr(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        mut assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let m = cross_fit_nuisance(
            self.outcome,
            PredictionTask::Regression,
            problem.design_matrix.as_ref(),
            problem.nrows,
            problem.design_ncols,
            problem.outcome.as_ref(),
            self.folds,
            ctx,
        )?;
        let e = cross_fit_nuisance(
            self.treatment,
            PredictionTask::BinaryProbability,
            problem.design_matrix.as_ref(),
            problem.nrows,
            problem.design_ncols,
            problem.treatment.as_ref(),
            self.folds,
            ctx,
        )?;
        let n = problem.nrows;
        let retained = trim_retained_rows(&e.predictions, trim_of(problem.overlap))?;
        let mut keep = vec![true; n];
        if let Some(rows) = &retained {
            keep.fill(false);
            for &i in rows {
                keep[i] = true;
            }
        }
        let mut num = 0.0;
        let mut den = 0.0;
        let mut residual_t = vec![0.0; n];
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            let dt = problem.treatment[i] - e.predictions[i];
            let dy = problem.outcome[i] - m.predictions[i];
            residual_t[i] = dt;
            num += dt * dy;
            den += dt * dt;
        }
        if !num.is_finite() || !den.is_finite() || den <= 0.0 {
            return Err(EstimationError::stats_msg("DML PLR denominator is zero"));
        }
        let theta = num / den;
        let mut psi: Vec<f64> = (0..n)
            .map(|i| {
                let dt = residual_t[i];
                let dy = problem.outcome[i] - m.predictions[i];
                dt * dy - theta * dt * dt
            })
            .collect();
        let nf = n as f64;
        // Divide the orthogonal score by its empirical Jacobian.
        for value in &mut psi {
            *value /= den / nf;
        }
        let se = iid_se(&psi);
        let report = OverlapReport::from_propensities(
            &e.predictions,
            None,
            problem.overlap,
            Some(problem.treatment.as_ref()),
            Some(IpwTarget::Ate),
            None,
        );
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::ParametricRestriction(
                antecedent_core::ParametricAssumption {
                    id: Arc::from("dml.partially_linear.constant_effect"),
                    description: Arc::from("The Robinson slope equals the ATE under a constant conditional treatment effect; with heterogeneity it generally targets a propensity-variance-weighted effect instead. IID inference also requires consistent nuisance fits with a sufficiently small product of errors."),
                },
            ),
            source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("dml") },
            scope: antecedent_core::AssumptionScope::Estimation,
            status: antecedent_core::AssumptionStatus::Declared,
        });
        let mut effect = EffectEstimate::new(theta, se, assumptions, problem.overlap)
            .with_overlap_report(Some(report))
            .with_n_obs(u64::try_from(keep.iter().filter(|&&v| v).count()).unwrap_or(u64::MAX))
            .with_se_kind(AnalyticSeKind::Homoskedastic)
            .with_influence(Some(Arc::from(psi)));
        effect.crossfit_folds = Some(self.folds);
        effect.crossfit_seed = Some(ctx.rng.master_seed());
        effect.outcome_oof_r2 = m.validation.r2;
        effect.treatment_oof_logloss = e.validation.logloss;
        effect.learner_provenance = m.model_provenance;
        effect.learner_provenance.extend(e.model_provenance);
        Ok(effect)
    }
}

pub(crate) fn finish_dml(
    phi: &[f64],
    problem: &PreparedPropensityProblem,
    assumptions: AssumptionSet,
    treat_logloss: Option<f64>,
    outcome_r2: Option<f64>,
    raw_e: Vec<f64>,
) -> Result<EffectEstimate, EstimationError> {
    let n = phi.len();
    if n < 2 || raw_e.len() != n || phi.iter().any(|p| !p.is_finite()) {
        return Err(EstimationError::data_msg("DML requires at least two finite, aligned scores"));
    }
    let retained = trim_retained_rows(&raw_e, trim_of(problem.overlap))?;
    let rows: Vec<usize> = retained.unwrap_or_else(|| (0..n).collect());
    let kept = rows.len();
    if kept < 2 {
        return Err(EstimationError::data_msg("DML inference requires two retained rows"));
    }
    let ate = rows.iter().map(|&i| phi[i]).sum::<f64>() / kept as f64;
    // Embed the retained-population ratio influence in the original row universe.
    let mut influence = vec![0.0; n];
    for &i in &rows {
        influence[i] = (phi[i] - ate) * n as f64 / kept as f64;
    }
    let se = iid_se(&influence);
    let report = OverlapReport::from_propensities(
        &raw_e,
        None,
        problem.overlap,
        Some(problem.treatment.as_ref()),
        Some(IpwTarget::Ate),
        None,
    );
    let mut effect = EffectEstimate::new(ate, se, assumptions, problem.overlap)
        .with_overlap_report(Some(report))
        .with_n_obs(u64::try_from(kept).unwrap_or(u64::MAX))
        .with_se_kind(AnalyticSeKind::Homoskedastic)
        .with_influence(Some(Arc::from(influence)));
    effect.outcome_oof_r2 = outcome_r2;
    effect.treatment_oof_logloss = treat_logloss;
    Ok(effect)
}

#[cfg(test)]
mod tests {
    use antecedent_core::StreamDomain;

    use std::sync::Arc;

    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::{ExprId, IdentifiedEstimand};
    use antecedent_kernels::standard_normal;
    use antecedent_learn::{LinearSpec, LogisticSpec};

    use super::*;
    use crate::aipw::{AipwAte, AipwWorkspace};

    fn confounded_columns(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng =
            ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Estimate, 0x1234_u64);
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let zi = standard_normal(&mut rng);
            let logit = -0.5 + zi;
            let p = 1.0 / (1.0 + (-logit).exp());
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + standard_normal(&mut rng) * 0.5;
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
    fn ridge_logistic_recovers_linear_ate() {
        let (t, y, z) = confounded_columns(800, 1);
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = DmlAte::new();
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let effect = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.3, "ate={}", effect.ate);
        assert!(effect.overlap_report.is_some());
    }

    #[test]
    fn aipw_matches_licensed_glm_ols_on_fixture() {
        let (t, y, z) = confounded_columns(240, 4);
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let dml = DmlAte::new()
            .with_outcome(LearnerSpec::Linear(LinearSpec::default()))
            .with_treatment(LearnerSpec::Logistic(LogisticSpec::default()))
            .with_folds(5);
        let prep = dml.prepare(&data, &estimand, &query).unwrap();
        let dml_fit = dml.fit(&prep, &ctx(), AssumptionSet::new()).unwrap();
        let aipw = AipwAte { bootstrap_replicates: 0, ..AipwAte::new() };
        let aipw_prep = aipw.prepare(&data, &estimand, &query).unwrap();
        let licensed = aipw
            .fit(&aipw_prep, &mut AipwWorkspace::default(), &ctx(), AssumptionSet::new())
            .unwrap();
        assert!(
            (dml_fit.ate - licensed.ate).abs() < 0.05,
            "dml={} licensed={}",
            dml_fit.ate,
            licensed.ate
        );
    }

    #[test]
    fn aipw_refuses_non_binary_treatment() {
        let n = 40usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = DmlAte::new();
        let err = est.prepare(&data, &estimand, &query).unwrap_err();
        assert!(err.to_string().contains("binary"));
    }

    #[test]
    fn plr_recovers_linear_ate() {
        let (t, y, z) = confounded_columns(800, 8);
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = DmlAte::new().with_score(DmlScore::PartiallyLinear);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let effect = est.fit(&prep, &ctx(), AssumptionSet::new()).unwrap();
        assert!((effect.ate - 2.0).abs() < 0.35, "ate={}", effect.ate);
    }
    #[test]
    fn cached_oof_reuses_predictions_and_invalidates_changed_input() {
        let (t, y, z) = confounded_columns(120, 12);
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let estimator = DmlAte::new();
        let mut problem = estimator.prepare(&data, &estimand, &query).unwrap();
        let ctx = ExecutionContext::for_tests(44);
        let first = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap();
        assert!(first.outcome_oof_r2.is_some());
        assert!(first.treatment_oof_logloss.is_some());
        assert_eq!(first.crossfit_folds, Some(5));
        assert_eq!(first.crossfit_seed, Some(44));
        let again = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap();
        assert_eq!(first.ate.to_bits(), again.ate.to_bits());
        // Concurrent callers must initialize one shared bundle, not four fits.
        problem.learner_cache.lock().unwrap().clear();
        let bundles = std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        cached_aipw_nuisances(
                            &problem,
                            estimator.outcome,
                            estimator.treatment,
                            5,
                            &ctx,
                        )
                        .unwrap()
                    })
                })
                .collect();
            jobs.into_iter().map(|job| job.join().unwrap()).collect::<Vec<_>>()
        });
        assert!(bundles.iter().all(|bundle| Arc::ptr_eq(bundle, &bundles[0])));
        let mut reassigned = problem.clone();
        let assignments: Arc<[u32]> =
            (0..problem.nrows).map(|i| ((i / 2) % 5) as u32).collect::<Vec<_>>().into();
        reassigned.fold_assignment = Some(assignments.clone());
        let new_folds =
            cached_aipw_nuisances(&reassigned, estimator.outcome, estimator.treatment, 5, &ctx)
                .unwrap();
        assert!(!Arc::ptr_eq(&new_folds, &bundles[0]));
        assert_eq!(
            new_folds.3.fold_assignment,
            assignments.iter().map(|f| *f as u16).collect::<Vec<_>>()
        );
        let cancelled = ExecutionContext::for_tests(44);
        cancelled.cancellation.cancel();
        assert!(
            cached_aipw_nuisances(&problem, estimator.outcome, estimator.treatment, 5, &cancelled)
                .is_err()
        );
        assert!(
            cached_aipw_nuisances(&problem, estimator.outcome, estimator.treatment, 121, &ctx)
                .is_err()
        );
        assert!(
            cached_aipw_nuisances(&problem, estimator.outcome, estimator.treatment, 5, &ctx)
                .is_ok()
        );
        let mut outcomes = problem.outcome.to_vec();
        for (y, t) in outcomes.iter_mut().zip(problem.treatment.iter()) {
            *y += t * 2.0;
        }
        problem.outcome = outcomes.into();
        let changed = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap();
        assert!((changed.ate - first.ate - 2.0).abs() < 0.1);
    }

    #[test]
    fn trimming_changes_the_score_target_and_keeps_row_alignment() {
        let (t, y, z) = confounded_columns(4, 12);
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let mut prep = DmlAte::new().prepare(&data, &estimand, &query).unwrap();
        prep.overlap = OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.1) };
        let effect = finish_dml(
            &[100.0, 1.0, 3.0, 200.0],
            &prep,
            AssumptionSet::new(),
            None,
            None,
            vec![0.01, 0.3, 0.7, 0.99],
        )
        .unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-12);
        assert_eq!(effect.n_obs, Some(2));
        assert_eq!(effect.influence.as_deref().unwrap(), &[0.0, -2.0, 2.0, 0.0]);
    }

    #[test]
    fn learner_ate_known_truth_coverage() {
        // Independent SCM draws; population ATE=2, Gaussian outcome noise,
        // logistic confounding. Fixed seeds make this measurement reproducible.
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/learner_ate/fixture.json"
        ))
        .unwrap();
        let reps = fixture["replicates"].as_u64().unwrap();
        let n = usize::try_from(fixture["n"].as_u64().unwrap()).unwrap();
        let truth = fixture["ate"].as_f64().unwrap();
        let seed_start = fixture["seed_start"].as_u64().unwrap();
        for method in 0..4 {
            let mut covered = 0;
            let mut bias = 0.0;
            for seed in 0..reps {
                let (t, y, z) = confounded_columns(n, seed_start + seed);
                let (data, estimand) = build_dataset(t, y, z);
                let query = AverageEffectQuery::binary_ate(
                    VariableId::from_raw(0),
                    VariableId::from_raw(1),
                );
                let dml = DmlAte::new();
                let prep = dml.prepare(&data, &estimand, &query).unwrap();
                let effect = match method {
                    0 => dml.fit(&prep, &ctx(), AssumptionSet::new()),
                    1 => dml.with_score(DmlScore::PartiallyLinear).fit(
                        &prep,
                        &ctx(),
                        AssumptionSet::new(),
                    ),
                    2 => crate::DrLearner::new().fit(&prep, &ctx(), AssumptionSet::new()),
                    _ => crate::CausalForest::new().with_n_trees(20).fit(
                        &prep,
                        &ctx(),
                        AssumptionSet::new(),
                    ),
                }
                .unwrap();
                assert!(effect.se_analytic.is_finite() && effect.se_analytic > 0.0);
                let influence = effect.influence.as_ref().unwrap();
                assert!(influence.iter().sum::<f64>().abs() < 1e-7);
                assert!((iid_se(influence) - effect.se_analytic).abs() < 1e-10);
                covered += usize::from(
                    (effect.ate - truth).abs() <= 1.959_963_984_540_054 * effect.se_analytic,
                );
                bias += effect.ate - truth;
            }
            let coverage = covered as f64 / reps as f64;
            let bias = bias / reps as f64;
            eprintln!(
                "learner_coverage method={method} n={n} replicates={reps} coverage={coverage} bias={bias}"
            );
            assert!(
                (fixture["coverage_acceptance"][0].as_f64().unwrap()
                    ..=fixture["coverage_acceptance"][1].as_f64().unwrap())
                    .contains(&coverage),
                "method={method}, coverage={coverage}"
            );
            assert!(
                bias.abs() < fixture["absolute_bias_limit"].as_f64().unwrap(),
                "method={method}, bias={bias}"
            );
        }
    }
}
