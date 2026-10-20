//! Retained sequential randomized regime value over subject histories.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalExprArena, CausalQuery, DerivationMeta, DerivationTrace, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, DomainRef, EffectEstimate, EstimatorId, ExecutionContext,
    ExprNode, GraphClass, IdentificationPerformanceRecord, IdentificationResult,
    IdentifiedEstimand, IdentifiedExecuteFinish, IdentifierId, InferenceMode, OutcomeExprId,
    OverlapPolicy, PhysicalExecutionPlan, RandomizedTrialOperation, RandomizedTrialQuery,
    RefuteSuite, Strategy, Study, StudyResult, TableView, TabularData,
    finish_identified_execute_with_context,
};
use antecedent_core::LongitudinalRegimeMethod;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
};
use antecedent_estimate::longitudinal_regime::{
    RegimeValueSummary, evaluate_g_formula_value, evaluate_regime_value,
    evaluate_sequential_dr_value_with_subject_scores, g_formula_fixed_q_pointwise_interval_95,
    sequential_dr_pointwise_interval_95,
};
use antecedent_estimate::marginal_structural_model::{MsmIntervals95, MsmSummary, fit_binary_msm};

pub(crate) type CheckedLongitudinalRegimeOperation = RandomizedTrialOperation<
    antecedent_core::LongitudinalRegimeQuery,
    Option<super::BayesianConfig>,
>;

mod bayesian_regime;

impl RandomizedTrialQuery for antecedent_core::LongitudinalRegimeQuery {
    const OPERATION: &'static str = "CheckedLongitudinalRegimeOperation";
    const REFRESH_MISMATCH: &'static str =
        "longitudinal refresh requires prepared schema and subject row order";

    fn describe(&self, rows: usize, debug: &mut std::fmt::DebugStruct<'_, '_>) {
        debug.field("subjects", &rows);
        debug.field("periods", &self.periods);
    }
}

impl CheckedLongitudinalRegimeOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::LongitudinalRegime(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "longitudinal regime operation requires LongitudinalRegimeQuery".into(),
            });
        };
        query.validate()?;
        if query.subject_ids.len() != data.row_count()
            || !query.probabilities_known_by_design
            || study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "retained longitudinal regime value requires known sequential randomization, one outcome row per subject, graphless trial, and point-only frequentist execution or the Bayesian sequential regime posterior",
            });
        }
        let bayesian = match &study.inference {
            InferenceMode::Frequentist => None,
            InferenceMode::Bayesian(config) => {
                bayesian_regime::admit_bayesian_regime(query, config)?;
                Some(config.clone())
            }
        };
        data.schema().get(query.outcome).map_err(CausalError::rejected)?;
        let estimator_id = if bayesian.is_some() {
            EstimatorId::BayesianSequentialRegimeValue
        } else {
            match query.method {
                LongitudinalRegimeMethod::Ipw => EstimatorId::LongitudinalIpwRegime,
                LongitudinalRegimeMethod::GFormula => EstimatorId::LongitudinalGFormulaRegime,
                LongitudinalRegimeMethod::SequentialDoublyRobust => {
                    EstimatorId::LongitudinalSequentialDrRegime
                }
                LongitudinalRegimeMethod::MarginalStructuralModel => {
                    EstimatorId::LongitudinalMarginalStructuralModel
                }
            }
        };
        if physical.logical.query != study.query
            || !physical.binding().names(IdentifierId::RandomizedDesign, estimator_id)
        {
            return Err(CausalError::Compile {
                message: "longitudinal regime query differs from compiled plan".into(),
            });
        }
        let (identification, estimand) = longitudinal_identification(query);
        Ok(Self::freeze(study, query, data, physical, identification, estimand, bayesian))
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.begin_execute(data, ctx)?;
        let outcome = match data.column(self.query.outcome).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(col) if col.validity.is_all_valid() => {
                col.values.as_slice()
            }
            _ => {
                return Err(CausalError::Unsupported {
                    message: "longitudinal endpoint outcome must be complete continuous data",
                });
            }
        };
        if let Some(config) = self.extra.as_ref() {
            return self.execute_bayesian(data, outcome, config, ctx);
        }
        let estimation = self.estimate_regime_value(outcome)?;
        let intervals = self.regime_intervals(&estimation);
        let (diagnostic, description) = published_diagnostic(&estimation, &intervals);
        let graphless_licensed = self.graphless_licensed(&estimation.summary, &intervals);
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(
                &self.physical,
                self.identification.clone(),
                self.estimand.clone(),
                EffectEstimate::new(
                    0.0,
                    f64::NAN,
                    self.identification.required_assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                ),
                Strategy::new(IdentifierId::RandomizedDesign, estimation.estimator_id),
                self.query.outcome,
                self.query.outcome,
            )
            .identify_cached(true)
            .extra_diagnostics(vec![Diagnostic::new(
                diagnostic,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                description,
            )]),
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.treatment = None;
        if graphless_licensed {
            result.support_status = Some(crate::support::CellStatus::Licensed);
        }
        result.payload = crate::result::ResultPayload::LongitudinalRegime(self.regime_payload(
            &estimation,
            &intervals,
            graphless_licensed,
        ));
        Ok(result)
    }

    /// Smallest realized action probability and censoring survival over every decision of every subject.
    fn probability_floors(&self) -> (f64, f64) {
        let mut min_action: f64 = 1.0;
        let mut min_censor: f64 = 1.0;
        for j in 0..self.query.treatment_history.len() {
            let p = self.query.treatment_probabilities[j];
            min_action = min_action.min(if self.query.treatment_history[j] { p } else { 1.0 - p });
            min_censor = min_censor.min(self.query.censoring_probabilities[j]);
        }
        (min_action, min_censor)
    }

    /// Backward-recursive augmented value with its per-subject scores, from the validated Q predictions and history.
    fn sequential_dr_value(
        &self,
        outcome: &[f64],
    ) -> Result<(RegimeValueSummary, Vec<f64>), &'static str> {
        let (Some(observation_history), Some(q_predictions)) =
            (self.query.observation_history.as_deref(), self.query.q_predictions.as_deref())
        else {
            return Err(
                "sequential doubly robust regime value requires observation history and Q predictions",
            );
        };
        evaluate_sequential_dr_value_with_subject_scores(
            outcome,
            &self.subject_histories(),
            &antecedent_estimate::longitudinal_regime::SequentialDrInputs {
                actions: &self.query.regime_actions,
                observation_history,
                q_predictions,
            },
        )
    }

    /// The declared subject histories every estimator of this operation reads.
    fn subject_histories(&self) -> antecedent_estimate::longitudinal_regime::SubjectHistories<'_> {
        antecedent_estimate::longitudinal_regime::SubjectHistories {
            treatment: &self.query.treatment_history,
            treatment_probability: &self.query.treatment_probabilities,
            outcome_observed: &self.query.outcome_observed,
            censoring_probability: &self.query.censoring_probabilities,
            periods: self.query.periods,
            minimum_probability: self.query.minimum_probability,
        }
    }

    /// Evaluates the method's point value and keeps the MSM fit or DR subject scores the interval phase needs.
    fn estimate_regime_value(&self, outcome: &[f64]) -> Result<RegimeEstimation, CausalError> {
        let mut msm: Option<MsmSummary> = None;
        let mut dr_scores: Option<Vec<f64>> = None;
        // `LongitudinalRegimeQuery::validate`, run in the prepare pipeline,
        // refuses any query whose method-conditional fields (g-formula
        // `period_outcome_predictions`, sequential-DR `q_predictions` and
        // `observation_history`, MSM `stabilizing_numerator_probabilities`) are
        // not `Some` and aligned; an absent field is still a typed refusal here.
        let (summary, estimator_id, method, diagnostic, description) = match self.query.method {
            LongitudinalRegimeMethod::Ipw => (
                evaluate_regime_value(
                    outcome,
                    &self.subject_histories(),
                    &self.query.regime_actions,
                ),
                EstimatorId::LongitudinalIpwRegime,
                "ipw",
                "estimate.longitudinal.ipw_regime.point_only",
                "sequential inverse-probability regime value under caller-declared known randomization and censoring probabilities; no interval or calibration claim",
            ),
            LongitudinalRegimeMethod::GFormula => {
                let predictions = self.query.period_outcome_predictions.as_deref().ok_or(
                    CausalError::Unsupported {
                        message: "g-formula regime value requires period outcome predictions",
                    },
                )?;
                (
                    evaluate_g_formula_value(
                        predictions,
                        &self.query.regime_actions,
                        &self.query.treatment_probabilities,
                        &self.query.censoring_probabilities,
                        self.rows,
                        self.query.periods,
                        self.query.minimum_probability,
                    ),
                    EstimatorId::LongitudinalGFormulaRegime,
                    "g_formula",
                    "estimate.longitudinal.g_formula_regime.point_only",
                    "sequential plug-in g-formula value from caller-supplied conditional period rewards; fold exclusion is declared, not independently verified; no interval or calibration claim",
                )
            }
            LongitudinalRegimeMethod::SequentialDoublyRobust => {
                let result = self.sequential_dr_value(outcome).map(|(summary, scores)| {
                    dr_scores = Some(scores);
                    summary
                });
                (
                    result,
                    EstimatorId::LongitudinalSequentialDrRegime,
                    "sequential_dr",
                    "estimate.longitudinal.sequential_dr_regime.point_only",
                    "backward-recursive augmented regime value from caller-declared subject-excluded Q predictions and known sequential probabilities; point-only below calibrated support",
                )
            }
            LongitudinalRegimeMethod::MarginalStructuralModel => {
                let numerators = self.query.stabilizing_numerator_probabilities.as_deref().ok_or(
                    CausalError::Unsupported {
                        message: "marginal structural model requires stabilizing numerators",
                    },
                )?;
                let fit = fit_binary_msm(outcome, &self.subject_histories(), numerators)
                    .map_err(|message| CausalError::Unsupported { message })?;
                let (min_action, min_censor) = self.probability_floors();
                let summary = RegimeValueSummary {
                    value: fit.intercept,
                    effective_sample_size: fit.effective_sample_size,
                    matched_observed_fraction: fit.observed_subjects as f64 / self.rows as f64,
                    maximum_weight: fit.maximum_weight,
                    minimum_action_probability: min_action,
                    minimum_censoring_probability: min_censor,
                    score_standard_error: None,
                };
                msm = Some(fit);
                (
                    Ok(summary),
                    EstimatorId::LongitudinalMarginalStructuralModel,
                    "marginal_structural_model",
                    "estimate.longitudinal.marginal_structural_model.pointwise_cr1",
                    "additive binary marginal structural mean with stabilized sequential IPTW and subject-clustered pointwise CR1 standard errors; no interval or calibration claim",
                )
            }
        };
        let summary = summary.map_err(|message| CausalError::Unsupported { message })?;
        Ok(RegimeEstimation {
            summary,
            estimator_id,
            method,
            diagnostic,
            description,
            msm,
            dr_scores,
        })
    }

    /// Publishes whichever pointwise 95% intervals the method's support floors allow.
    fn regime_intervals(&self, estimation: &RegimeEstimation) -> RegimeIntervals {
        let summary = &estimation.summary;
        let g_formula = if self.query.method == LongitudinalRegimeMethod::GFormula
            && self.query.known_fixed_outcome_predictions
            && self.query.probabilities_known_by_design
        {
            self.query.period_outcome_predictions.as_deref().and_then(|predictions| {
                g_formula_fixed_q_pointwise_interval_95(
                    summary,
                    predictions,
                    self.rows,
                    self.query.periods,
                )
            })
        } else {
            None
        };
        let ipw = if self.query.method == LongitudinalRegimeMethod::Ipw {
            antecedent_estimate::longitudinal_regime::ipw_pointwise_interval_95(summary, self.rows)
        } else {
            None
        };
        let msm = estimation.msm.as_ref().and_then(|fit| {
            antecedent_estimate::marginal_structural_model::pointwise_intervals_95(fit, self.rows)
        });
        let dr = estimation.dr_scores.as_ref().and_then(|scores| {
            let matched = (0..self.rows)
                .filter(|&i| {
                    self.query.outcome_observed[i]
                        && (0..self.query.periods).all(|t| {
                            let j = i * self.query.periods + t;
                            self.query.treatment_history[j] == self.query.regime_actions[j]
                        })
                })
                .count();
            sequential_dr_pointwise_interval_95(
                summary,
                scores,
                self.query.periods,
                self.query.outcome_observed.iter().filter(|&&observed| observed).count(),
                matched,
            )
        });
        RegimeIntervals { g_formula, ipw, msm, dr }
    }

    /// Names why the method published no interval, when it did not.
    fn interval_reason(&self, intervals: &RegimeIntervals) -> Option<&'static str> {
        match self.query.method {
            LongitudinalRegimeMethod::Ipw if intervals.ipw.is_none() => {
                Some("insufficient_independent_subject_support_or_degenerate_score")
            }
            LongitudinalRegimeMethod::GFormula if !self.query.known_fixed_outcome_predictions => {
                Some("prediction_model_uncertainty_not_accounted")
            }
            LongitudinalRegimeMethod::GFormula if !self.query.probabilities_known_by_design => {
                Some("known_sequential_randomization_required_for_fixed_q_interval")
            }
            LongitudinalRegimeMethod::GFormula if intervals.g_formula.is_none() => {
                Some("insufficient_two_period_subject_support_or_degenerate_fixed_q_score")
            }
            LongitudinalRegimeMethod::SequentialDoublyRobust if intervals.dr.is_none() => Some(
                "insufficient_calibrated_horizon_or_trajectory_support_for_sequential_dr_interval",
            ),
            LongitudinalRegimeMethod::MarginalStructuralModel if intervals.msm.is_none() => {
                Some("insufficient_independent_subject_support_for_msm_intervals")
            }
            _ => None,
        }
    }

    /// Decides the graphless license from the published interval plus the exact retained matrix row.
    fn graphless_licensed(
        &self,
        summary: &RegimeValueSummary,
        intervals: &RegimeIntervals,
    ) -> bool {
        // The sequential-DR interval only publishes after its narrowed,
        // calibrated subject support holds, so the graphless license is a
        // function of that published interval plus the exact two- or
        // three-period row. Observed-outcome, matching-history, and censoring
        // floors are enforced upstream by the interval and stated in the row's
        // limitations; classify_graphless checks the coarse published-interval
        // support so the license cannot exist without a matching matrix row.
        let graphless_dr_licensed = self.query.method
            == LongitudinalRegimeMethod::SequentialDoublyRobust
            && intervals.dr.is_some()
            && matches!(self.query.periods, 2 | 3)
            && crate::support::license_if_graphless(
                crate::support::GraphlessSupportKey {
                    family: "longitudinal_regime",
                    design: if self.query.periods == 2 {
                        "known_sequential_randomized_two_period"
                    } else {
                        "known_sequential_randomized_three_period"
                    },
                    method: "subject_excluded_q_sequential_dr_scores",
                    inference_claim: "conditional_q_pointwise_95_normal_interval",
                },
                crate::support::GraphlessAssignmentSupport {
                    assignment_unit: "unit",
                    rows: self.rows,
                    interval_95_published: true,
                    reported_intervals: 1,
                    min_probability: summary.minimum_action_probability,
                    disjoint_nuisance_training: self.query.excluded_fold_predictions
                        && self.query.prediction_fold_ids.as_deref()
                            == Some(self.query.fold_ids.as_ref()),
                    ..Default::default()
                },
            )
            .is_some();
        // The two-period fixed-known-Q g-formula interval only publishes after
        // its own subject, positivity, and censoring floors hold, so its
        // graphless license is a function of that published interval plus the
        // exact two-period fixed-Q row. The fixed-known-Q attestation and the
        // finer floors are enforced upstream by the interval and stated in the
        // row's limitations; classify_graphless checks the coarse row support.
        let graphless_g_formula_licensed = self.query.method == LongitudinalRegimeMethod::GFormula
            && intervals.g_formula.is_some()
            && self.query.periods == 2
            && crate::support::license_if_graphless(
                crate::support::GraphlessSupportKey {
                    family: "longitudinal_regime",
                    design: "known_sequential_randomized_two_period",
                    method: "fixed_known_q_g_formula_scores",
                    inference_claim: "conditional_q_pointwise_95_normal_interval",
                },
                crate::support::GraphlessAssignmentSupport {
                    assignment_unit: "unit",
                    rows: self.rows,
                    interval_95_published: true,
                    reported_intervals: 1,
                    min_probability: summary.minimum_action_probability,
                    ..Default::default()
                },
            )
            .is_some();
        // The additive MSM publishes separate pointwise subject-clustered CR1
        // intervals for the intercept and every period coefficient only after
        // its subject, observed-outcome, and effective-weight floors hold, so
        // its license is a function of those published intervals plus the exact
        // additive-MSM row. The observed-outcome floor and positive coefficient
        // SEs are enforced upstream by the intervals and stated in the row.
        let graphless_msm_licensed = self.query.method
            == LongitudinalRegimeMethod::MarginalStructuralModel
            && intervals.msm.is_some()
            && crate::support::license_if_graphless(
                crate::support::GraphlessSupportKey {
                    family: "longitudinal_regime",
                    design: "known_sequential_randomized_additive_msm",
                    method: "stabilized_ipw_cr1_scores",
                    inference_claim: "intercept_and_period_effects_pointwise_95_normal_intervals",
                },
                crate::support::GraphlessAssignmentSupport {
                    assignment_unit: "unit",
                    rows: self.rows,
                    interval_95_published: true,
                    reported_intervals: self.query.periods + 1,
                    all_reported_intervals: true,
                    min_probability: summary.minimum_action_probability,
                    min_effective_sample_size: summary.effective_sample_size,
                    ..Default::default()
                },
            )
            .is_some();
        graphless_dr_licensed || graphless_g_formula_licensed || graphless_msm_licensed
    }

    /// Assembles the retained regime estimate from the estimation and interval phases.
    fn regime_payload(
        &self,
        estimation: &RegimeEstimation,
        intervals: &RegimeIntervals,
        graphless_licensed: bool,
    ) -> crate::LongitudinalRegimeEstimate {
        let interval_reason = self.interval_reason(intervals);
        crate::LongitudinalRegimeEstimate {
            method: Arc::from(estimation.method),
            graphless_support_status: graphless_licensed
                .then_some(crate::support::CellStatus::Licensed),
            rule_id: self.query.rule_id.clone(),
            rule_version: self.query.rule_version.clone(),
            rule_provenance: self.query.rule_provenance.clone(),
            value: estimation.summary.value,
            value_standard_error: estimation
                .msm
                .as_ref()
                .map(|fit| fit.intercept_standard_error)
                .or(estimation.summary.score_standard_error)
                .or_else(|| intervals.dr.map(|(se, _)| se))
                .or_else(|| intervals.g_formula.map(|(se, _)| se)),
            value_interval_95: intervals
                .ipw
                .or_else(|| intervals.msm.as_ref().map(|intervals| intervals.intercept))
                .or_else(|| intervals.dr.map(|(_, bounds)| bounds))
                .or_else(|| intervals.g_formula.map(|(_, bounds)| bounds)),
            period_intervals_95: intervals
                .msm
                .as_ref()
                .map(|intervals| Arc::from(intervals.period_effects.as_slice())),
            interval_reason: interval_reason.map(Arc::from),
            effective_sample_size: estimation.summary.effective_sample_size,
            matched_observed_fraction: estimation.summary.matched_observed_fraction,
            maximum_weight: estimation.summary.maximum_weight,
            minimum_action_probability: estimation.summary.minimum_action_probability,
            minimum_censoring_probability: estimation.summary.minimum_censoring_probability,
            uncertainty: Arc::from(if intervals.msm.is_some() {
                "pointwise_subject_clustered_cr1_95"
            } else if estimation.msm.is_some() {
                "pointwise_subject_clustered_cr1_no_interval"
            } else if intervals.dr.is_some() {
                "pointwise_subject_score_conditional_excluded_fold_q_95"
            } else if intervals.g_formula.is_some() {
                "pointwise_subject_score_conditional_fixed_known_q_95"
            } else if intervals.ipw.is_some() {
                "pointwise_subject_score_95"
            } else {
                "point_only_no_interval"
            }),
            probability_ownership: Arc::from("known_sequential_randomization"),
            period_effects: estimation
                .msm
                .as_ref()
                .map(|fit| Arc::from(fit.period_effects.as_slice())),
            standard_errors: estimation
                .msm
                .as_ref()
                .map(|fit| Arc::from(fit.standard_errors.as_slice())),
            stabilizing_numerator_probabilities: estimation
                .msm
                .as_ref()
                .and_then(|_| self.query.stabilizing_numerator_probabilities.clone()),
            observed_subjects: estimation.msm.as_ref().map(|fit| fit.observed_subjects),
        }
    }
}

impl Study {
    pub(super) fn execute_longitudinal_regime(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedLongitudinalRegimeOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

/// Point value of the regime plus the MSM fit or DR subject scores later phases consume.
struct RegimeEstimation {
    summary: RegimeValueSummary,
    estimator_id: EstimatorId,
    method: &'static str,
    diagnostic: &'static str,
    description: &'static str,
    msm: Option<MsmSummary>,
    dr_scores: Option<Vec<f64>>,
}

/// Pointwise 95% intervals each method may publish for the regime value.
struct RegimeIntervals {
    g_formula: Option<(f64, [f64; 2])>,
    ipw: Option<[f64; 2]>,
    msm: Option<MsmIntervals95>,
    dr: Option<(f64, [f64; 2])>,
}

/// Publishes the diagnostic code and description matching the interval the method actually reported.
fn published_diagnostic(
    estimation: &RegimeEstimation,
    intervals: &RegimeIntervals,
) -> (&'static str, &'static str) {
    if intervals.ipw.is_some() {
        (
            "estimate.longitudinal.ipw_regime.pointwise_95",
            "sequential randomized regime value with independent-subject Horvitz--Thompson score SE and pointwise 95% interval; requires known treatment/censoring probabilities and calibrated subject support",
        )
    } else if intervals.msm.is_some() {
        (
            "estimate.longitudinal.marginal_structural_model.pointwise_95",
            "additive binary MSM intercept and period effects with independent-subject CR1 pointwise 95% intervals; requires known treatment/censoring probabilities and calibrated weighted support",
        )
    } else if intervals.g_formula.is_some() {
        (
            "estimate.longitudinal.g_formula_regime.pointwise_95_conditional_fixed_q",
            "two-period g-formula value with independent-subject pointwise 95% interval conditional on the caller-declared fixed known Q law; no fitted-Q uncertainty is included",
        )
    } else if intervals.dr.is_some() {
        (
            "estimate.longitudinal.sequential_dr_regime.pointwise_95_conditional_q",
            "sequential DR value with pointwise 95% independent-subject score interval conditional on caller-declared subject-excluded Q fitting and known randomization; Q training is not verified",
        )
    } else {
        (estimation.diagnostic, estimation.description)
    }
}

/// Records one user-declared custom identification assumption.
fn declare_custom(assumptions: &mut antecedent_core::AssumptionSet, id: &str, description: &str) {
    assumptions.push(AssumptionRecord {
        assumption: Assumption::Custom { id: Arc::from(id), description: Arc::from(description) },
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Identification,
        status: AssumptionStatus::Declared,
    });
}

/// Declared sequential-randomization assumptions, plus the nuisance contract of the selected method.
fn longitudinal_assumptions(
    query: &antecedent_core::LongitudinalRegimeQuery,
) -> antecedent_core::AssumptionSet {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    for assumption in [Assumption::Consistency, Assumption::NoInterference, Assumption::Positivity]
    {
        assumptions.push(AssumptionRecord {
            assumption,
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    for (id, description) in [
        (
            "known_sequential_randomization",
            "each treatment decision is randomized with the supplied conditional probability given predecision history",
        ),
        (
            "sequential_censoring_exchangeability",
            "remaining uncensored at each decision is conditionally independent of regime outcomes given the supplied history",
        ),
        (
            "subject_history_ownership",
            "one outcome row and one fold own every decision in each subject history",
        ),
    ] {
        declare_custom(&mut assumptions, id, description);
    }
    if query.method == LongitudinalRegimeMethod::GFormula {
        if query.known_fixed_outcome_predictions {
            declare_custom(
                &mut assumptions,
                "fixed_known_outcome_predictions",
                "caller declares the conditional period reward law fixed and known before this study, with no fitted or selected Q uncertainty",
            );
        }
        declare_custom(
            &mut assumptions,
            "conditional_period_reward_validity",
            "supplied period reward predictions equal conditional means under the prescribed regime given each predecision history",
        );
        if query.excluded_fold_predictions {
            declare_custom(
                &mut assumptions,
                "subject_excluded_fold_predictions",
                "caller declares that every outcome prediction excludes the subject's entire history fold",
            );
        }
    }
    if query.method == LongitudinalRegimeMethod::SequentialDoublyRobust {
        for (id, description) in [
            (
                "sequential_q_validity",
                "supplied Q is the conditional mean of the next recursive pseudo-outcome under the prescribed regime",
            ),
            (
                "subject_excluded_fold_predictions",
                "caller declares every Q prediction excludes the subject's entire history fold",
            ),
            (
                "monotone_observation_history",
                "dropout is monotone and terminal observation agrees with the final history period",
            ),
        ] {
            declare_custom(&mut assumptions, id, description);
        }
    }
    if query.method == LongitudinalRegimeMethod::MarginalStructuralModel {
        for (id, description) in [
            (
                "additive_marginal_structural_mean",
                "the terminal marginal mean is additive in binary period treatments without interactions",
            ),
            (
                "stabilized_sequential_weights",
                "supplied treatment and censoring probabilities and prespecified numerator probabilities identify a stabilized IPTW model",
            ),
            (
                "pointwise_subject_clustered_cr1",
                "the reported CR1 sandwich treats each whole subject history as one independent cluster and does not establish interval coverage",
            ),
        ] {
            declare_custom(&mut assumptions, id, description);
        }
    }
    assumptions
}

fn longitudinal_identification(
    query: &antecedent_core::LongitudinalRegimeQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let assumptions = longitudinal_assumptions(query);
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set([query.outcome]);
    let empty = arena.empty_var_set();
    let intervention = arena.intern_intervention_set([]);
    let distribution =
        arena.intern_distribution(outcomes, empty, intervention, DomainRef::Observational);
    let functional = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(query.outcome),
        distribution,
    });
    let rule = match query.method {
        LongitudinalRegimeMethod::Ipw => "longitudinal.sequential_randomization",
        LongitudinalRegimeMethod::GFormula => "longitudinal.g_formula",
        LongitudinalRegimeMethod::SequentialDoublyRobust => "longitudinal.sequential_dr",
        LongitudinalRegimeMethod::MarginalStructuralModel => {
            "longitudinal.marginal_structural_model"
        }
    };
    arena.set_derivation(functional, DerivationMeta::rule(rule, Some(Arc::from("known sequential assignment, observation, and valid supplied conditional rewards identify the prescribed regime mean"))));
    let estimand = IdentifiedEstimand::new(
        rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut derivation = DerivationTrace::default();
    derivation.push(rule, "prescribed regime value identified by known sequential treatment and censoring probabilities and the method's declared nuisance contract");
    let identification = IdentificationResult::identified(
        CausalQuery::LongitudinalRegime(query.clone()),
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}
