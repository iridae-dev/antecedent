//! Partially linear DML and cross-fitted AIPW via `antecedent-learn`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, TargetPopulation};
use antecedent_data::{TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_learn::{LearnerSpec, LogisticSpec, PredictionTask, RidgeSpec, diagnose};

use crate::adjustment::EffectEstimate;
use crate::cluster_dml_aipw::{ClusterDml, DeclaredUnits};
use crate::error::EstimationError;
use crate::learn_nuisance::{
    AipwPredictions, NuisanceDesign, aipw_scores, cached_aipw_nuisances, cross_fit_nuisance,
    guarded_propensity, iid_se,
};
use crate::overlap::{IpwTarget, OverlapPolicy, OverlapReport};
use crate::prepare::{require_adjustment_shaped, validate_ate_query_with_targets};
use crate::propensity::{
    OtherLevels, PreparedPropensityProblem, clip_of, default_propensity_overlap,
    prepare_propensity_problem_with_levels, trim_of, trim_retained_rows,
};
use crate::sampling_weights::SamplingWeights;
use crate::scores::{ScoreColumn, ScoreTable};
use crate::se::AnalyticSeKind;

/// Provenance tag frozen on a [`DmlAte`] score table (followed by the two learner identities).
pub(crate) const DML_CROSSFIT_PROVENANCE: &str = "dml.crossfit.v1";

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
    /// Policy for treatment rows at neither declared level. The default refuses; excluding
    /// them restricts the target population to the rows at the two declared levels.
    pub other_levels: OtherLevels,
    /// Declared independence unit of a whole-cluster cross-fit (AIPW score only): the
    /// cross-fit plan owns whole units and the standard error is the cluster sandwich of the
    /// cross-fitted scores.
    pub units: DeclaredUnits,
    /// Sampling weights of every row of the source table (see [`crate::sampling_weights`]).
    /// Both nuisance learners are fit under them (the learner must carry prior weights), the
    /// score is averaged under them, and the standard error is that of the weighted score.
    /// A weighted fit keeps no row-identified score table, so it is not retargetable.
    pub sampling_weights: Option<SamplingWeights>,
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
            other_levels: OtherLevels::Refuse,
            units: DeclaredUnits {
                spec: None,
                first: None,
                first_source: None,
                second: None,
                second_source: None,
            },
            sampling_weights: None,
        }
    }

    /// Declare sampling weights for every row of the source table.
    #[must_use]
    pub fn with_sampling_weights(mut self, weights: SamplingWeights) -> Self {
        self.sampling_weights = Some(weights);
        self
    }

    /// Declare whole-cluster cross-fitting over the unit labels set with
    /// [`Self::with_cluster_ids`] or [`Self::with_source_cluster_ids`].
    #[must_use]
    pub fn with_cluster_dml(mut self, spec: ClusterDml) -> Self {
        self.units.spec = Some(spec);
        self
    }

    /// Unit label of every analysed row.
    #[must_use]
    pub fn with_cluster_ids(mut self, labels: Vec<u32>) -> Self {
        self.units.first = Some(labels);
        self
    }

    /// Unit label of every row of the source table (length must equal its row count).
    #[must_use]
    pub fn with_source_cluster_ids(mut self, labels: Vec<u32>) -> Self {
        self.units.first_source = Some(labels);
        self
    }

    /// Second-endpoint label of every analysed row of a dyadic unit.
    #[must_use]
    pub fn with_second_cluster_ids(mut self, labels: Vec<u32>) -> Self {
        self.units.second = Some(labels);
        self
    }

    /// Second-endpoint label of every source row of a dyadic unit.
    #[must_use]
    pub fn with_source_second_cluster_ids(mut self, labels: Vec<u32>) -> Self {
        self.units.second_source = Some(labels);
        self
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

    /// Declare what to do with rows whose treatment is at neither declared level (see
    /// [`OtherLevels`]). The default refuses.
    #[must_use]
    pub const fn with_other_levels(mut self, other_levels: OtherLevels) -> Self {
        self.other_levels = other_levels;
        self
    }

    /// Prepare `[1|Z]` design, recoded 0/1 arm indicator, and outcome.
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
        self.units.check_source_rows(data.row_count())?;
        let mut problem = prepare_propensity_problem_with_levels(
            data,
            estimand,
            query,
            self.overlap,
            None,
            self.other_levels,
            self.sampling_weights.as_ref(),
        )?;
        if self.units.spec.is_some() && !matches!(self.score, DmlScore::Aipw) {
            return Err(EstimationError::unsupported(
                "a cluster-DML declaration needs the AIPW score, whose per-arm scores the \
                 cluster sandwich reads",
            ));
        }
        self.units.declare(
            &mut problem,
            query.outcome_functional == antecedent_core::OutcomeFunctional::Mean,
            self.overlap,
        )?;
        Ok(problem)
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
        if matches!(self.score, DmlScore::Aipw) {
            require_binary_treatment(problem)?;
        }
        let estimate = match self.score {
            DmlScore::Aipw => self.fit_aipw(problem, ctx, assumptions),
            DmlScore::PartiallyLinear => self.fit_plr(problem, ctx, assumptions),
        }?;
        let estimate = if problem.sampling_weights.is_some() && self.units.spec.is_some() {
            let nuisance =
                cached_aipw_nuisances(problem, self.outcome, self.treatment, self.folds, ctx)?;
            let fold_ids: Vec<u32> =
                nuisance.3.fold_assignment.iter().map(|&f| u32::from(f)).collect();
            self.units.publish_weighted(
                estimate,
                problem,
                &fold_ids,
                u32::try_from(self.folds).unwrap_or(u32::MAX),
            )?
        } else {
            self.units.publish(estimate, problem)?
        };
        Ok(estimate.with_treatment_levels(problem.treatment_levels))
    }

    /// Form a population ATE interval under independent homoskedastic Gaussian
    /// arm-linear outcomes, including training-fold fit uncertainty and a declared
    /// finite-ridge mean correction. The default robust fit remains unchanged.
    ///
    /// # Errors
    /// Nonlinear outcome learners, dependence, unsupported target, failed fit, or invalid level.
    pub fn population_linear_interval(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
        level: f64,
    ) -> Result<crate::population_aipw::PopulationAipwInterval, EstimationError> {
        if self.units.spec.is_some() || !matches!(self.score, DmlScore::Aipw) {
            return Err(EstimationError::unsupported("population linear DML requires IID AIPW"));
        }
        let ridge = match self.outcome {
            LearnerSpec::Linear(_) => 0.0,
            LearnerSpec::Ridge(spec) => spec.lambda,
            _ => {
                return Err(EstimationError::unsupported(
                    "population linear interval requires a linear outcome learner",
                ));
            }
        };
        let estimate = self.fit(problem, ctx, assumptions)?;
        let nuisance =
            cached_aipw_nuisances(problem, self.outcome, self.treatment, self.folds, ctx)?;
        let propensity =
            guarded_propensity(&nuisance.2, crate::propensity::clip_of(problem.overlap))?;
        let folds: Vec<u32> = nuisance.3.fold_assignment.iter().map(|&f| u32::from(f)).collect();
        crate::population_aipw::interval(problem, estimate, &propensity, &folds, ridge, level)
    }

    /// Row-identified cross-fitted AIPW score table from this estimator's own out-of-fold
    /// nuisances (the same cached bundle [`Self::fit`] uses), or `None` when the fit has no
    /// such table: the partially linear score has no per-arm scores, and a trimmed fit is
    /// scored on a different population.
    ///
    /// The mean of `φ¹ - φ⁰` equals the fitted ATE, so a retarget under uniform weights
    /// returns the route's own point. The table's covariance is the plug-in iid score
    /// covariance: it is meaningful only under the cross-fit assumptions in
    /// `docs/guides/dml-scores.md` (iid rows, positivity, consistent nuisances with a small
    /// product of errors), never from a CATE prediction alone.
    ///
    /// # Errors
    ///
    /// A non-binary treatment, a cancelled or failed nuisance fit, or non-finite scores.
    pub fn score_table(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
    ) -> Result<Option<ScoreTable>, EstimationError> {
        if !matches!(self.score, DmlScore::Aipw) || !scores_retargetable(problem) {
            return Ok(None);
        }
        if self.folds < 2 {
            return Err(EstimationError::data_msg("DML requires at least two folds"));
        }
        require_binary_treatment(problem)?;
        let nuisance =
            cached_aipw_nuisances(problem, self.outcome, self.treatment, self.folds, ctx)?;
        aipw_score_table(problem, &nuisance, self.folds, [self.outcome, self.treatment], "dml")
            .map(Some)
    }

    fn fit_aipw(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let nuisance =
            cached_aipw_nuisances(problem, self.outcome, self.treatment, self.folds, ctx)?;
        let marginal = aipw_marginal_from_nuisance(problem, &nuisance, assumptions)?;
        attach_crossfit_aipw_table(
            marginal.effect,
            problem,
            &nuisance,
            CrossfitAipwTableContext {
                folds: self.folds,
                learners: [self.outcome, self.treatment],
                route: "dml",
                seed: ctx.rng.master_seed(),
                clipped: &marginal.ehat,
            },
        )
    }

    fn fit_plr(
        &self,
        problem: &PreparedPropensityProblem,
        ctx: &ExecutionContext,
        mut assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        let row_weights = problem.sampling_weights.as_ref().map(|s| &*s.weights);
        let m = cross_fit_nuisance(
            self.outcome,
            PredictionTask::Regression,
            NuisanceDesign::of(problem),
            problem.outcome.as_ref(),
            row_weights,
            self.folds,
            ctx,
        )?;
        let e = cross_fit_nuisance(
            self.treatment,
            PredictionTask::BinaryProbability,
            NuisanceDesign::of(problem),
            problem.treatment.as_ref(),
            row_weights,
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
            let weight = row_weights.map_or(1.0, |w| w[i]);
            residual_t[i] = dt;
            num += weight * dt * dy;
            den += weight * dt * dt;
        }
        if !num.is_finite() || !den.is_finite() || den <= 0.0 {
            return Err(EstimationError::stats_msg("DML PLR denominator is zero"));
        }
        let theta = num / den;
        let mut psi: Vec<f64> = (0..n)
            .map(|i| {
                let dt = residual_t[i];
                let dy = problem.outcome[i] - m.predictions[i];
                let weight = row_weights.map_or(1.0, |w| w[i]);
                weight * (dt * dy - theta * dt * dt)
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
            .with_sampling_weights(problem.sampling_weights.as_ref().map(|s| s.record))
            .with_influence(Some(Arc::from(psi)));
        effect.crossfit.folds = Some(self.folds);
        effect.crossfit.seed = Some(ctx.rng.master_seed());
        effect.crossfit.outcome_oof_r2 = m.validation.r2;
        effect.crossfit.treatment_oof_logloss = e.validation.logloss;
        effect.crossfit.learner_provenance = m.model_provenance;
        effect.crossfit.learner_provenance.extend(e.model_provenance);
        Ok(effect)
    }
}

/// Whether an AIPW-mean fit is the mean of an untrimmed per-row score table.
///
/// Trimming redefines the estimand to the retained rows, so a trimmed fit keeps no table (a
/// retarget then refuses with `score_table_unavailable` rather than reweighting scores of
/// another construction).
pub(crate) fn scores_retargetable(problem: &PreparedPropensityProblem) -> bool {
    // A weighted fit's scores describe the weighted population; a retarget under another
    // weight vector would silently drop the sampling weights, so no table is kept.
    matches!(problem.target_population, TargetPopulation::AllObserved)
        && trim_of(problem.overlap).is_none()
        && problem.sampling_weights.is_none()
}

/// Per-arm scores `μ_a + 1{T=a}(Y - μ_a)/e_a` from one cross-fitted nuisance bundle. The same
/// formulas as the licensed AIPW table (`build_binary_scores`), so `φ¹ - φ⁰` is the AIPW
/// score of the DML, `DR-Learner` and `CausalForest` marginal ATE row by row. `learners` is
/// `[outcome, treatment]`; `route` names the estimator in the provenance tag.
pub(crate) fn aipw_score_table(
    problem: &PreparedPropensityProblem,
    nuisance: &AipwPredictions,
    folds: usize,
    learners: [LearnerSpec; 2],
    route: &str,
) -> Result<ScoreTable, EstimationError> {
    aipw_score_table_clipped(problem, nuisance, folds, learners, route, None)
}

fn aipw_score_table_clipped(
    problem: &PreparedPropensityProblem,
    nuisance: &AipwPredictions,
    folds: usize,
    learners: [LearnerSpec; 2],
    route: &str,
    clipped: Option<&[f64]>,
) -> Result<ScoreTable, EstimationError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::ScoreConstruction,
        || {
            let (mu0, mu1, raw_e, treat) = nuisance;
            let n = problem.nrows;
            let clip = clip_of(problem.overlap);
            // A fit that already clipped `ê` passes that vector. Guarding again would
            // copy the raw scores and apply the same clip; the table formulas are unchanged.
            let guarded;
            let ehat_rows: &[f64] = if let Some(rows) = clipped {
                rows
            } else {
                guarded = guarded_propensity(raw_e, clip)?;
                guarded.as_slice()
            };
            let mut scores = vec![0.0; 2 * n];
            let mut propensities = vec![0.0; 2 * n];
            for i in 0..n {
                let (t, y, e) = (problem.treatment[i], problem.outcome[i], ehat_rows[i]);
                scores[i] = mu0[i] + ((1.0 - t) / (1.0 - e)) * (y - mu0[i]);
                scores[n + i] = mu1[i] + (t / e) * (y - mu1[i]);
                propensities[i] = 1.0 - raw_e[i];
                propensities[n + i] = raw_e[i];
            }
            if scores.iter().any(|v| !v.is_finite()) {
                return Err(EstimationError::data_msg("AIPW score table requires finite scores"));
            }
            AIPW_SCORE_BUILDS.with(|count| count.set(count.get().saturating_add(1)));
            Ok(ScoreTable {
                observed_arm: problem.treatment.iter().map(|t| u32::from(*t > 0.5)).collect(),
                propensities: propensities.into(),
                observed_outcome: Arc::clone(&problem.outcome),
                n_rows: n,
                row_index: Arc::clone(&problem.row_index),
                fold_ids: treat.fold_assignment.iter().map(|&f| u32::from(f)).collect(),
                n_folds: u32::try_from(folds).unwrap_or(u32::MAX),
                scores: scores.into(),
                columns: Arc::from([
                    ScoreColumn { arm: 0, threshold: None },
                    ScoreColumn { arm: 1, threshold: None },
                ]),
                adjustment_set: Arc::clone(&problem.adjustment_set),
                nuisance_provenance: Arc::from(format!(
                    "{DML_CROSSFIT_PROVENANCE};route={route};outcome={};treatment={}{}",
                    learners[0].identity(),
                    learners[1].identity(),
                    // Whole-unit folds change the dependence the scores support; the table
                    // names them, as the AIPW table does.
                    match (problem.fold_assignment.as_deref(), problem.fold_units.as_deref()) {
                        (None, Some(units)) => crate::cluster_dml_aipw::provenance_suffix(
                            units,
                            problem.fold_unit_kind
                        ),
                        _ => String::new(),
                    }
                )),
                propensity_clip: clip,
                treatment: problem.treatment_id,
                intervened: Arc::from([]),
            })
        },
    )
}

/// AIPW scores and the marginal-mean effect of one cross-fitted nuisance bundle.
///
/// DML and the DR-Learner both identify their reported ATE as the mean of these scores.
/// The DR-Learner then regresses the same `phi` in its final stage; nothing here fits that
/// stage or writes the score table.
pub(crate) struct AipwMarginal {
    /// Orthogonal scores `φ = μ¹ − μ⁰ + T(Y − μ¹)/ê − (1 − T)(Y − μ⁰)/(1 − ê)`.
    pub phi: Vec<f64>,
    /// Propensity scores after the overlap clip. The score table reuses this vector.
    pub ehat: Vec<f64>,
    /// Marginal mean of `phi` on the retained rows.
    pub effect: EffectEstimate,
}

pub(crate) fn aipw_marginal_from_nuisance(
    problem: &PreparedPropensityProblem,
    nuisance: &AipwPredictions,
    assumptions: AssumptionSet,
) -> Result<AipwMarginal, EstimationError> {
    let (mu0, mu1, raw_e, treat) = nuisance;
    let ehat = guarded_propensity(raw_e, clip_of(problem.overlap))?;
    let phi = aipw_scores(problem.treatment.as_ref(), problem.outcome.as_ref(), &ehat, mu0, mu1);
    let yhat: Vec<f64> = problem
        .treatment
        .iter()
        .zip(mu0)
        .zip(mu1)
        .map(|((&ti, &m0), &m1)| if ti > 0.5 { m1 } else { m0 })
        .collect();
    let outcome_diag = diagnose(PredictionTask::Regression, problem.outcome.as_ref(), &yhat);
    let effect =
        finish_dml(&phi, problem, assumptions, treat.validation.logloss, outcome_diag.r2, raw_e)?;
    Ok(AipwMarginal { phi, ehat, effect })
}

/// Attach the untrimmed AIPW score table and the cross-fit record of `nuisance`.
///
/// A trimmed problem keeps no table. The treatment-nuisance provenance is copied as-is;
/// a caller that fits a further stage appends that stage after this returns.
#[derive(Clone, Copy)]
pub(crate) struct CrossfitAipwTableContext<'a> {
    pub(crate) folds: usize,
    pub(crate) learners: [LearnerSpec; 2],
    pub(crate) route: &'a str,
    pub(crate) seed: u64,
    pub(crate) clipped: &'a [f64],
}

pub(crate) fn attach_crossfit_aipw_table(
    mut effect: EffectEstimate,
    problem: &PreparedPropensityProblem,
    nuisance: &AipwPredictions,
    table_context: CrossfitAipwTableContext<'_>,
) -> Result<EffectEstimate, EstimationError> {
    let CrossfitAipwTableContext { folds, learners, route, seed, clipped } = table_context;
    if scores_retargetable(problem) {
        effect = effect.with_score_table(Some(aipw_score_table_clipped(
            problem,
            nuisance,
            folds,
            learners,
            route,
            Some(clipped),
        )?));
    }
    let (_, _, _, treat) = nuisance;
    effect.crossfit.folds = Some(folds);
    effect.crossfit.seed = Some(seed);
    effect.crossfit.learner_provenance.clone_from(&treat.model_provenance);
    Ok(effect)
}

fn require_binary_treatment(problem: &PreparedPropensityProblem) -> Result<(), EstimationError> {
    if problem.treatment.iter().any(|&ti| ti.abs() > 1e-12 && (ti - 1.0).abs() > 1e-12) {
        return Err(EstimationError::data_msg(
            "DML AIPW score requires the prepared treatment recoded to a 0/1 arm indicator",
        ));
    }
    Ok(())
}

pub(crate) fn finish_dml(
    phi: &[f64],
    problem: &PreparedPropensityProblem,
    assumptions: AssumptionSet,
    treat_logloss: Option<f64>,
    outcome_r2: Option<f64>,
    raw_e: &[f64],
) -> Result<EffectEstimate, EstimationError> {
    let n = phi.len();
    if n < 2 || raw_e.len() != n || phi.iter().any(|p| !p.is_finite()) {
        return Err(EstimationError::data_msg("DML requires at least two finite, aligned scores"));
    }
    let retained = trim_retained_rows(raw_e, trim_of(problem.overlap))?;
    let rows: Vec<usize> = retained.unwrap_or_else(|| (0..n).collect());
    let kept = rows.len();
    if kept < 2 {
        return Err(EstimationError::data_msg("DML inference requires two retained rows"));
    }
    // Under sampling weights the estimate is the weighted mean of the scores and each row's
    // influence carries its weight; with none, every weight is one.
    let row_weights = problem.sampling_weights.as_ref().map(|s| &*s.weights);
    let weight_of = |i: usize| row_weights.map_or(1.0, |w| w[i]);
    let mass: f64 = rows.iter().map(|&i| weight_of(i)).sum();
    let ate = rows.iter().map(|&i| weight_of(i) * phi[i]).sum::<f64>() / mass;
    // Embed the retained-population ratio influence in the original row universe.
    let mut influence = vec![0.0; n];
    for &i in &rows {
        influence[i] = weight_of(i) * (phi[i] - ate) * n as f64 / mass;
    }
    let se = iid_se(&influence);
    let report = OverlapReport::from_propensities(
        raw_e,
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
        .with_sampling_weights(problem.sampling_weights.as_ref().map(|s| s.record))
        .with_influence(Some(Arc::from(influence)));
    effect.crossfit.outcome_oof_r2 = outcome_r2;
    effect.crossfit.treatment_oof_logloss = treat_logloss;
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
            let p = antecedent_kernels::sigmoid(logit);
            let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
            z[i] = zi;
            t[i] = ti;
            y[i] = 2.0 * ti + zi + standard_normal(&mut rng) * 0.5;
        }
        (t, y, z)
    }

    fn build_dataset(t: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> (TabularData, IdentifiedEstimand) {
        let n = t.len();
        build_dataset_with_outcome_validity(t, y, z, ValidityBitmap::all_valid(n))
    }

    fn build_dataset_with_outcome_validity(
        t: Vec<f64>,
        y: Vec<f64>,
        z: Vec<f64>,
        y_validity: ValidityBitmap,
    ) -> (TabularData, IdentifiedEstimand) {
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
                Float64Column::new(VariableId::from_raw(1), Arc::from(y), y_validity).unwrap(),
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
        assert!(effect.diagnostics.overlap_report.is_some());
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
        assert!(
            err.to_string().starts_with(
                "reason=treatment_level_undeclared: treatment_levels.undeclared_level:"
            ),
            "{err}"
        );
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
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the value is (i / 2) % 5, at most 4; fold ids are at most 4"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "fixture, two fits and the comparing assertions of one scenario"
    )]
    fn cached_oof_reuses_predictions_and_invalidates_changed_input() {
        let (t, y, z) = confounded_columns(120, 12);
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let estimator = DmlAte::new();
        let mut problem = estimator.prepare(&data, &estimand, &query).unwrap();
        let ctx = ExecutionContext::for_tests(44);
        let first = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap();
        assert!(first.crossfit.outcome_oof_r2.is_some());
        assert!(first.crossfit.treatment_oof_logloss.is_some());
        assert_eq!(first.crossfit.folds, Some(5));
        assert_eq!(first.crossfit.seed, Some(44));
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
        // A caller can alter fold metadata after physical preparation sharing. These
        // clones deliberately share learner_cache, so its own bundle key must separate
        // every fold input, not rely on the preparation-cache equality guard.
        for coordinate in 0..3 {
            let mut changed = problem.clone();
            match coordinate {
                0 => changed.fold_seed = 91,
                1 => {
                    changed.fold_units =
                        Some((0..problem.nrows).map(|i| i as u32).collect::<Vec<_>>().into())
                }
                _ => changed.fold_unit_kind = crate::cluster_dml_aipw::IndependenceUnit::Dyad,
            }
            let (bundle, fits) = antecedent_learn::fit_counts::observe_resolved_fits(|| {
                cached_aipw_nuisances(&changed, estimator.outcome, estimator.treatment, 5, &ctx)
                    .unwrap()
            });
            assert_eq!(
                fits, 15,
                "changed fold metadata rebuilds all three nuisances in five folds"
            );
            assert!(!Arc::ptr_eq(&bundle, &bundles[0]));
            let (again, fits) = antecedent_learn::fit_counts::observe_resolved_fits(|| {
                cached_aipw_nuisances(&changed, estimator.outcome, estimator.treatment, 5, &ctx)
                    .unwrap()
            });
            assert_eq!(fits, 0);
            assert!(Arc::ptr_eq(&bundle, &again));
        }
        // Hold every entry as concurrent callers would during initialization. An
        // additional key must fit correctly without expanding retained capacity.
        let active_entries = problem.learner_cache.lock().unwrap().clone();
        let busy = Arc::clone(&active_entries[0]);
        *problem.learner_cache.lock().unwrap() = vec![Arc::clone(&busy); 8];
        let mut uncached = problem.clone();
        uncached.fold_seed = 999;
        for _ in 0..2 {
            let (_, fits) = antecedent_learn::fit_counts::observe_resolved_fits(|| {
                cached_aipw_nuisances(&uncached, estimator.outcome, estimator.treatment, 5, &ctx)
                    .unwrap()
            });
            assert_eq!(fits, 15, "full in-flight cache computes extra keys without retaining them");
            assert_eq!(problem.learner_cache.lock().unwrap().len(), 8);
        }
        problem.learner_cache.lock().unwrap().clear();
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
            &[0.01, 0.3, 0.7, 0.99],
        )
        .unwrap();
        assert!((effect.ate - 2.0).abs() < 1e-12);
        assert_eq!(effect.provenance.n_obs, Some(2));
        assert_eq!(effect.scores.influence.as_deref().unwrap(), &[0.0, -2.0, 2.0, 0.0]);
    }

    fn prepared(n: usize, seed: u64, est: &DmlAte) -> PreparedPropensityProblem {
        let (t, y, z) = confounded_columns(n, seed);
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        est.prepare(&data, &estimand, &query).unwrap()
    }

    /// Fit a whole-cluster DML over 300 rows whose outcome is missing on some rows: unit labels
    /// declared for every source row keep their rows through the complete-case mask.
    fn cluster_dml_fit(units: impl Fn(&[u32], &[u32]) -> DmlAte) -> EffectEstimate {
        let n = 300usize;
        let (t, mut y, z) = confounded_columns(n, 33);
        let missing: Vec<usize> = (0..n).filter(|i| i % 11 == 3).collect();
        for &row in &missing {
            y[row] = 0.0;
        }
        let source: Vec<u32> = (0..n).map(|i| u32::try_from(i * 7 % 20).unwrap()).collect();
        let kept: Vec<u32> = source
            .iter()
            .enumerate()
            .filter(|(row, _)| !missing.contains(row))
            .map(|(_, &label)| label)
            .collect();
        let mut bits = vec![0xFFu8; n.div_ceil(8)];
        for &row in &missing {
            bits[row / 8] &= !(1u8 << (row % 8));
        }
        let validity = ValidityBitmap::from_bytes(bits, n).unwrap();
        let (data, estimand) = build_dataset_with_outcome_validity(t, y, z, validity);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let est = units(&source, &kept);
        let problem = est.prepare(&data, &estimand, &query).unwrap();
        assert_eq!(problem.nrows, n - missing.len());
        est.fit(&problem, &ctx(), AssumptionSet::new()).unwrap()
    }

    #[test]
    fn source_unit_labels_follow_their_rows_through_the_complete_case_mask() {
        let spec = ClusterDml::new(10).unwrap();
        let by_source = cluster_dml_fit(|source, _| {
            DmlAte::new().with_cluster_dml(spec).with_source_cluster_ids(source.to_vec())
        });
        let by_aligned = cluster_dml_fit(|_, kept| {
            DmlAte::new().with_cluster_dml(spec).with_cluster_ids(kept.to_vec())
        });
        assert!(by_source.se_analytic().is_finite() && by_source.se_analytic() > 0.0);
        assert_eq!(by_source.ate.to_bits(), by_aligned.ate.to_bits());
        assert_eq!(by_source.se_analytic().to_bits(), by_aligned.se_analytic().to_bits());
        let table = by_source.scores.table.as_ref().expect("the cluster fit keeps its scores");
        assert!(table.nuisance_provenance.contains("cluster_dml=unit=cluster"));
        // Labels cut to the first rows instead of following their rows are another design.
        let misaligned = cluster_dml_fit(|_, kept| {
            let shifted: Vec<u32> = kept.iter().rev().copied().collect();
            DmlAte::new().with_cluster_dml(spec).with_cluster_ids(shifted)
        });
        assert_ne!(by_source.se_analytic().to_bits(), misaligned.se_analytic().to_bits());
    }

    #[test]
    fn a_cluster_dml_declaration_needs_the_aipw_score_and_exclusive_label_kinds() {
        let spec = ClusterDml::new(10).unwrap();
        let (t, y, z) = confounded_columns(100, 5);
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let plr = DmlAte::new()
            .with_score(DmlScore::PartiallyLinear)
            .with_cluster_dml(spec)
            .with_source_cluster_ids(vec![0; 100]);
        assert!(plr.prepare(&data, &estimand, &query).unwrap_err().to_string().contains("AIPW"));
        let both = DmlAte::new()
            .with_cluster_dml(spec)
            .with_source_cluster_ids(vec![0; 100])
            .with_cluster_ids(vec![0; 100]);
        assert!(both.prepare(&data, &estimand, &query).is_err());
        let short = DmlAte::new().with_cluster_dml(spec).with_source_cluster_ids(vec![0; 99]);
        assert!(short.prepare(&data, &estimand, &query).is_err());
    }

    /// Plug-in iid contrast variance of the table's arm columns, written from raw sums:
    /// `sum (psi - mean)^2 / (n (n - 1))` with `psi = phi1 - phi0`.
    fn oracle_se(table: &ScoreTable) -> (f64, f64) {
        let n = table.n_rows;
        let psi: Vec<f64> = (0..n).map(|i| table.scores[n + i] - table.scores[i]).collect();
        let mean = psi.iter().sum::<f64>() / n as f64;
        let ss: f64 = psi.iter().map(|v| (v - mean).powi(2)).sum();
        (mean, (ss / (n as f64 * (n as f64 - 1.0))).sqrt())
    }

    #[test]
    fn the_dml_score_table_averages_to_the_fitted_ate_with_the_iid_score_covariance() {
        let est = DmlAte::new();
        let problem = prepared(300, 21, &est);
        let ctx = ctx();
        let fit = est.fit(&problem, &ctx, AssumptionSet::new()).unwrap();
        let table = fit.scores.table.as_ref().expect("an untrimmed AIPW fit keeps its scores");
        assert_eq!(table.n_rows, problem.nrows);
        assert_eq!(table.row_index, problem.row_index);
        assert_eq!(table.adjustment_set, problem.adjustment_set);
        assert_eq!(table.treatment, problem.treatment_id);
        assert_eq!(table.columns.len(), 2);
        assert!(table.nuisance_provenance.starts_with("dml.crossfit.v1;route=dml;"));
        assert!(!crate::provenance_withholds_interval(&table.nuisance_provenance));
        // Score-mean identity and the iid covariance, both from raw sums in this test.
        let (mean, se) = oracle_se(table);
        assert!((fit.ate - mean).abs() < 1e-10, "{} vs {mean}", fit.ate);
        assert!((fit.se_analytic() - se).abs() < 1e-10, "{} vs {se}", fit.se_analytic());
        let summary = table.summarize(None).unwrap();
        let contrast = table.linear_contrast(&summary, &[-1.0, 1.0]).unwrap();
        assert!((contrast.value - fit.ate).abs() < 1e-10);
        assert!((contrast.se - fit.se_analytic()).abs() < 1e-10);
        // Fold ids are the nuisance fold plan; the propensity columns are complementary.
        assert_eq!(table.n_folds, 5);
        assert!(table.fold_ids.iter().all(|&f| f < 5));
        let n = table.n_rows;
        for i in 0..n {
            assert!((table.propensities[i] + table.propensities[n + i] - 1.0).abs() < 1e-12);
        }
        // Seeded replay: the same context returns the same table to the bit, and the public
        // table builder reproduces the fit's own table.
        let again = est.score_table(&problem, &ctx).unwrap().unwrap();
        assert_eq!(&again, table);
        let replay = est.fit(&problem, &ExecutionContext::for_tests(7), AssumptionSet::new());
        assert_eq!(replay.unwrap().scores.table.as_ref(), Some(table));
    }

    #[test]
    fn the_dml_score_table_round_trips_its_wire_form() {
        let est = DmlAte::new();
        let problem = prepared(120, 9, &est);
        let table = est.score_table(&problem, &ctx()).unwrap().expect("an untrimmed AIPW table");
        let wire = table.to_wire();
        assert_eq!(ScoreTable::from_wire(wire).unwrap(), table);
    }

    #[test]
    fn the_dml_score_table_replays_under_its_seed_and_follows_it() {
        let est = DmlAte::new();
        let problem = prepared(150, 11, &est);
        let first = est.score_table(&problem, &ExecutionContext::for_tests(3)).unwrap().unwrap();
        let again = est.score_table(&problem, &ExecutionContext::for_tests(3)).unwrap().unwrap();
        assert_eq!(first, again);
        let fresh = prepared(150, 11, &est);
        let rebuilt = est.score_table(&fresh, &ExecutionContext::for_tests(3)).unwrap().unwrap();
        assert_eq!(first, rebuilt);
        // Another seed deals another fold plan: the table (its fold ids and scores) changes.
        let other = est.score_table(&fresh, &ExecutionContext::for_tests(4)).unwrap().unwrap();
        assert_ne!(first.fold_ids, other.fold_ids);
    }

    /// Hand calculation on a fixed bundle: constant nuisances `mu0 = 1`, `mu1 = 3`, `e = 0.5`
    /// give arm scores `1 + 2 (1 - T)(Y - 1)` and `3 + 2 T (Y - 3)`, written here from the
    /// problem's own `T` and `Y`, and the table's contrast covariance is their iid variance.
    #[test]
    fn the_score_table_formulas_match_a_hand_calculation_on_a_fixed_bundle() {
        let est = DmlAte::new();
        let problem = prepared(6, 3, &est);
        let n = problem.nrows;
        let bundle: AipwPredictions = (
            vec![1.0; n],
            vec![3.0; n],
            vec![0.5; n],
            antecedent_learn::CrossFittedPrediction {
                predictions: vec![0.5; n],
                fold_assignment: (0..n).map(|i| u16::try_from(i % 2).unwrap()).collect(),
                model_provenance: Vec::new(),
                validation: antecedent_learn::NuisanceDiagnostics::default(),
            },
        );
        let table =
            aipw_score_table(&problem, &bundle, 2, [est.outcome, est.treatment], "dml").unwrap();
        let mut psi = Vec::new();
        for i in 0..n {
            let (t, y) = (problem.treatment[i], problem.outcome[i]);
            let arm0 = 1.0 + 2.0 * (1.0 - t) * (y - 1.0);
            let arm1 = 3.0 + 2.0 * t * (y - 3.0);
            assert!((table.scores[i] - arm0).abs() < 1e-12);
            assert!((table.scores[n + i] - arm1).abs() < 1e-12);
            psi.push(arm1 - arm0);
        }
        let mean = psi.iter().sum::<f64>() / n as f64;
        let var = psi.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        let summary = table.summarize(None).unwrap();
        let contrast = table.linear_contrast(&summary, &[-1.0, 1.0]).unwrap();
        assert!((contrast.value - mean).abs() < 1e-12);
        assert!((contrast.se - (var / n as f64).sqrt()).abs() < 1e-12);
        let expected_folds: Vec<u32> = (0..n).map(|i| u32::try_from(i % 2).unwrap()).collect();
        assert_eq!(table.fold_ids.to_vec(), expected_folds);
    }

    #[test]
    fn fits_without_a_per_arm_score_keep_no_table() {
        // Partially linear: the final stage is a Robinson slope, not an AIPW arm score.
        let plr = DmlAte::new().with_score(DmlScore::PartiallyLinear);
        let problem = prepared(200, 5, &plr);
        assert!(plr.fit(&problem, &ctx(), AssumptionSet::new()).unwrap().scores.table.is_none());
        assert!(plr.score_table(&problem, &ctx()).unwrap().is_none());
        // Trimming redefines the population: the fitted point is a trimmed mean, so no table.
        let mut trimmed = DmlAte::new();
        trimmed.overlap = OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: Some(0.1) };
        let problem = prepared(200, 5, &trimmed);
        assert!(
            trimmed.fit(&problem, &ctx(), AssumptionSet::new()).unwrap().scores.table.is_none()
        );
        assert!(trimmed.score_table(&problem, &ctx()).unwrap().is_none());
    }

    #[test]
    fn dr_learner_and_causal_forest_tables_are_the_marginal_aipw_scores() {
        let dr = crate::DrLearner::new();
        let (t, y, z) = confounded_columns(300, 31);
        let (data, estimand) = build_dataset(t, y, z);
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let prep = dr.prepare(&data, &estimand, &query).unwrap();
        let fit = dr.fit(&prep, &ctx(), AssumptionSet::new()).unwrap();
        let table = fit.scores.table.as_ref().expect("DR-Learner keeps the marginal scores");
        assert!(table.nuisance_provenance.contains("route=dr_learner"));
        let (mean, _) = oracle_se(table);
        assert!((fit.ate - mean).abs() < 1e-10);
        assert_eq!(dr.score_table(&prep, &ctx()).unwrap().as_ref(), Some(table));
        // The forest's marginal ATE is the DML AIPW score, so its table is DML's.
        let forest = crate::CausalForest::new().with_n_trees(20);
        let forest_prep = forest.prepare(&data, &estimand, &query).unwrap();
        let forest_fit = forest.fit(&forest_prep, &ctx(), AssumptionSet::new()).unwrap();
        let forest_table = forest_fit.scores.table.as_ref().expect("the forest keeps DML scores");
        let (forest_mean, _) = oracle_se(forest_table);
        assert!((forest_fit.ate - forest_mean).abs() < 1e-10);
        assert_eq!(forest.score_table(&forest_prep, &ctx()).unwrap().as_ref(), Some(forest_table));
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
                assert!(effect.se_analytic().is_finite() && effect.se_analytic() > 0.0);
                let influence = effect.scores.influence.as_ref().unwrap();
                assert!(influence.iter().sum::<f64>().abs() < 1e-7);
                assert!((iid_se(influence) - effect.se_analytic()).abs() < 1e-10);
                covered += usize::from(
                    (effect.ate - truth).abs()
                        <= antecedent_stats::NORMAL_Q975 * effect.se_analytic(),
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

thread_local! { static AIPW_SCORE_BUILDS:std::cell::Cell<u64>=const {std::cell::Cell::new(0)}; }
/// Observe actual successful AIPW score table constructions, including nested scopes.
pub fn count_aipw_score_builds<R>(work: impl FnOnce() -> R) -> (R, u64) {
    let before = AIPW_SCORE_BUILDS.with(std::cell::Cell::get);
    let result = work();
    (result, AIPW_SCORE_BUILDS.with(std::cell::Cell::get).saturating_sub(before))
}
