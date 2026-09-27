//! Retained sequential randomized regime value over subject histories.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::LongitudinalRegimeMethod;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
};
use antecedent_estimate::longitudinal_regime::{evaluate_g_formula_value, evaluate_regime_value, evaluate_sequential_dr_value_with_subject_scores, g_formula_fixed_q_pointwise_interval_95, sequential_dr_pointwise_interval_95, RegimeValueSummary};
use antecedent_estimate::marginal_structural_model::{fit_binary_msm, MsmSummary};

#[derive(Clone)]
pub(crate) struct CheckedLongitudinalRegimeOperation {
    query: antecedent_core::LongitudinalRegimeQuery,
    schema: antecedent_core::CausalSchema,
    rows: usize,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedLongitudinalRegimeOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedLongitudinalRegimeOperation")
            .field("subjects", &self.rows)
            .field("periods", &self.query.periods)
            .finish_non_exhaustive()
    }
}

impl CheckedLongitudinalRegimeOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::LongitudinalRegime(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "longitudinal regime operation requires LongitudinalRegimeQuery".into(),
            });
        };
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        if query.subject_ids.len() != data.row_count()
            || !query.probabilities_known_by_design
            || study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || !matches!(study.inference, InferenceMode::Frequentist)
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.tiered.is_some()
        {
            return Err(CausalError::Unsupported {
                message: "retained longitudinal regime value requires known sequential randomization, one outcome row per subject, graphless trial, and point-only frequentist execution",
            });
        }
        data.schema()
            .get(query.outcome)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let estimator_id = match query.method {
            LongitudinalRegimeMethod::Ipw => EstimatorId::LongitudinalIpwRegime,
            LongitudinalRegimeMethod::GFormula => EstimatorId::LongitudinalGFormulaRegime,
            LongitudinalRegimeMethod::SequentialDoublyRobust => EstimatorId::LongitudinalSequentialDrRegime,
            LongitudinalRegimeMethod::MarginalStructuralModel => EstimatorId::LongitudinalMarginalStructuralModel,
        };
        if physical.logical.query != study.query
            || physical.logical.record.identifier.as_deref()
                != Some(IdentifierId::RandomizedDesign.as_str())
            || physical.logical.record.estimator.as_deref() != Some(estimator_id.as_str())
        {
            return Err(CausalError::Compile {
                message: "longitudinal regime query differs from compiled plan".into(),
            });
        }
        let (identification, estimand) = longitudinal_identification(query);
        Ok(Self {
            query: query.clone(),
            schema: data.schema().clone(),
            rows: data.row_count(),
            identification,
            estimand,
            physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if data.schema() != &self.schema || data.row_count() != self.rows {
            return Err(CausalError::Unsupported {
                message: "longitudinal refresh requires prepared schema and subject row order",
            });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
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
        let mut msm: Option<MsmSummary> = None;
        let mut dr_scores: Option<Vec<f64>> = None;
        let (summary, estimator_id, method, diagnostic, description) = match self.query.method {
            LongitudinalRegimeMethod::Ipw => (
                evaluate_regime_value(outcome, &self.query.treatment_history, &self.query.regime_actions,
                    &self.query.treatment_probabilities, &self.query.outcome_observed,
                    &self.query.censoring_probabilities, self.query.periods, self.query.minimum_probability),
                EstimatorId::LongitudinalIpwRegime, "ipw", "estimate.longitudinal.ipw_regime.point_only",
                "sequential inverse-probability regime value under caller-declared known randomization and censoring probabilities; no interval or calibration claim",
            ),
            LongitudinalRegimeMethod::GFormula => (
                evaluate_g_formula_value(self.query.period_outcome_predictions.as_deref().expect("validated g-formula predictions"),
                    &self.query.regime_actions, &self.query.treatment_probabilities,
                    &self.query.censoring_probabilities, self.rows, self.query.periods,
                    self.query.minimum_probability),
                EstimatorId::LongitudinalGFormulaRegime, "g_formula", "estimate.longitudinal.g_formula_regime.point_only",
                "sequential plug-in g-formula value from caller-supplied conditional period rewards; fold exclusion is declared, not independently verified; no interval or calibration claim",
            ),
            LongitudinalRegimeMethod::SequentialDoublyRobust => {
                let result = evaluate_sequential_dr_value_with_subject_scores(outcome, &self.query.outcome_observed,
                    self.query.observation_history.as_deref().expect("validated observation history"),
                    &self.query.treatment_history, &self.query.regime_actions,
                    self.query.q_predictions.as_deref().expect("validated Q predictions"),
                    &self.query.treatment_probabilities, &self.query.censoring_probabilities,
                    self.query.periods, self.query.minimum_probability)
                    .map(|(summary, scores)| { dr_scores = Some(scores); summary });
                (result, EstimatorId::LongitudinalSequentialDrRegime, "sequential_dr",
                    "estimate.longitudinal.sequential_dr_regime.point_only",
                    "backward-recursive augmented regime value from caller-declared subject-excluded Q predictions and known sequential probabilities; point-only below calibrated support")
            },
            LongitudinalRegimeMethod::MarginalStructuralModel => {
                let fit = fit_binary_msm(outcome, &self.query.treatment_history,
                    &self.query.treatment_probabilities,
                    self.query.stabilizing_numerator_probabilities.as_deref().expect("validated MSM numerators"),
                    &self.query.outcome_observed, &self.query.censoring_probabilities,
                    self.query.periods, self.query.minimum_probability)
                    .map_err(|message| CausalError::Unsupported { message })?;
                let mut min_action: f64 = 1.0;
                let mut min_censor: f64 = 1.0;
                for j in 0..self.query.treatment_history.len() {
                    let p = self.query.treatment_probabilities[j];
                    min_action = min_action.min(if self.query.treatment_history[j] { p } else { 1.0 - p });
                    min_censor = min_censor.min(self.query.censoring_probabilities[j]);
                }
                let summary = RegimeValueSummary {
                    value: fit.intercept, effective_sample_size: fit.effective_sample_size,
                    matched_observed_fraction: fit.observed_subjects as f64 / self.rows as f64,
                    maximum_weight: fit.maximum_weight,
                    minimum_action_probability: min_action,
                    minimum_censoring_probability: min_censor,
                    score_standard_error: None,
                };
                msm = Some(fit);
                (Ok(summary), EstimatorId::LongitudinalMarginalStructuralModel,
                    "marginal_structural_model", "estimate.longitudinal.marginal_structural_model.pointwise_cr1",
                    "additive binary marginal structural mean with stabilized sequential IPTW and subject-clustered pointwise CR1 standard errors; no interval or calibration claim")
            },
        };
        let summary = summary.map_err(|message| CausalError::Unsupported { message })?;
        let g_formula_interval = if self.query.method == LongitudinalRegimeMethod::GFormula
            && self.query.known_fixed_outcome_predictions
            && self.query.probabilities_known_by_design {
            g_formula_fixed_q_pointwise_interval_95(&summary,
                self.query.period_outcome_predictions.as_deref().expect("validated g-formula predictions"),
                self.rows, self.query.periods)
        } else { None };
        let ipw_interval = if self.query.method == LongitudinalRegimeMethod::Ipw {
            antecedent_estimate::longitudinal_regime::ipw_pointwise_interval_95(&summary, self.rows)
        } else { None };
        let msm_intervals = msm.as_ref().and_then(|fit|
            antecedent_estimate::marginal_structural_model::pointwise_intervals_95(fit, self.rows));
        let dr_interval = dr_scores.as_ref().and_then(|scores| {
            let matched = (0..self.rows).filter(|&i| {
                self.query.outcome_observed[i] && (0..self.query.periods).all(|t| {
                    let j = i * self.query.periods + t;
                    self.query.treatment_history[j] == self.query.regime_actions[j]
                })
            }).count();
            sequential_dr_pointwise_interval_95(&summary, scores, self.query.periods,
                self.query.outcome_observed.iter().filter(|&&observed| observed).count(), matched)
        });
        let interval_reason = match self.query.method {
            LongitudinalRegimeMethod::Ipw if ipw_interval.is_none() => Some("insufficient_independent_subject_support_or_degenerate_score"),
            LongitudinalRegimeMethod::GFormula if !self.query.known_fixed_outcome_predictions => Some("prediction_model_uncertainty_not_accounted"),
            LongitudinalRegimeMethod::GFormula if !self.query.probabilities_known_by_design => Some("known_sequential_randomization_required_for_fixed_q_interval"),
            LongitudinalRegimeMethod::GFormula if g_formula_interval.is_none() => Some("insufficient_two_period_subject_support_or_degenerate_fixed_q_score"),
            LongitudinalRegimeMethod::SequentialDoublyRobust if dr_interval.is_none() => Some("insufficient_two_period_subject_or_trajectory_support_for_sequential_dr_interval"),
            LongitudinalRegimeMethod::MarginalStructuralModel if msm_intervals.is_none() => Some("insufficient_independent_subject_support_for_msm_intervals"),
            _ => None,
        };
        let (diagnostic, description) = if ipw_interval.is_some() {
            ("estimate.longitudinal.ipw_regime.pointwise_95",
             "sequential randomized regime value with independent-subject Horvitz--Thompson score SE and pointwise 95% interval; requires known treatment/censoring probabilities and calibrated subject support")
        } else if msm_intervals.is_some() {
            ("estimate.longitudinal.marginal_structural_model.pointwise_95",
             "additive binary MSM intercept and period effects with independent-subject CR1 pointwise 95% intervals; requires known treatment/censoring probabilities and calibrated weighted support")
        } else if g_formula_interval.is_some() {
            ("estimate.longitudinal.g_formula_regime.pointwise_95_conditional_fixed_q",
             "two-period g-formula value with independent-subject pointwise 95% interval conditional on the caller-declared fixed known Q law; no fitted-Q uncertainty is included")
        } else if dr_interval.is_some() {
            ("estimate.longitudinal.sequential_dr_regime.pointwise_95_conditional_q",
             "sequential DR value with pointwise 95% independent-subject score interval conditional on caller-declared subject-excluded Q fitting and known randomization; Q training is not verified")
        } else { (diagnostic, description) };
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish {
                physical: &self.physical,
                identification: self.identification.clone(),
                estimand: self.estimand.clone(),
                estimate: EffectEstimate::new(
                    0.0,
                    f64::NAN,
                    self.identification.required_assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                ),
                identifier_id: IdentifierId::RandomizedDesign,
                estimator_id,
                treatment: self.query.outcome,
                outcome: self.query.outcome,
                identify_cached: true,
                extra_diagnostics: vec![Diagnostic::new(
                    diagnostic,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    description,
                )],
                refutations: Vec::new(),
                distribution: None,
                mediation: None,
                wall_time_ns: 0,
                bootstrap_replicates_ok: None,
                cancelled: false,
                early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.treatment = None;
        result.longitudinal_regime = Some(crate::LongitudinalRegimeEstimate {
            method: Arc::from(method),
            rule_id: self.query.rule_id.clone(),
            rule_version: self.query.rule_version.clone(),
            rule_provenance: self.query.rule_provenance.clone(),
            value: summary.value,
            value_standard_error: msm.as_ref().map(|fit| fit.intercept_standard_error)
                .or(summary.score_standard_error).or_else(|| dr_interval.map(|(se, _)| se))
                .or_else(|| g_formula_interval.map(|(se, _)| se)),
            value_interval_95: ipw_interval.or_else(|| msm_intervals.as_ref().map(|intervals| intervals.intercept))
                .or_else(|| dr_interval.map(|(_, bounds)| bounds))
                .or_else(|| g_formula_interval.map(|(_, bounds)| bounds)),
            period_intervals_95: msm_intervals.as_ref().map(|intervals| Arc::from(intervals.period_effects.as_slice())),
            interval_reason: interval_reason.map(Arc::from),
            effective_sample_size: summary.effective_sample_size,
            matched_observed_fraction: summary.matched_observed_fraction,
            maximum_weight: summary.maximum_weight,
            minimum_action_probability: summary.minimum_action_probability,
            minimum_censoring_probability: summary.minimum_censoring_probability,
            uncertainty: Arc::from(if msm_intervals.is_some() { "pointwise_subject_clustered_cr1_95" }
                else if msm.is_some() { "pointwise_subject_clustered_cr1_no_interval" }
                else if dr_interval.is_some() { "pointwise_subject_score_conditional_excluded_fold_q_95" }
                else if g_formula_interval.is_some() { "pointwise_subject_score_conditional_fixed_known_q_95" }
                else if ipw_interval.is_some() { "pointwise_subject_score_95" } else { "point_only_no_interval" }),
            probability_ownership: Arc::from("known_sequential_randomization"),
            period_effects: msm.as_ref().map(|fit| Arc::from(fit.period_effects.as_slice())),
            standard_errors: msm.as_ref().map(|fit| Arc::from(fit.standard_errors.as_slice())),
            stabilizing_numerator_probabilities: msm.as_ref().map(|_| self.query.stabilizing_numerator_probabilities.as_ref().expect("validated MSM numerator").clone()),
            observed_subjects: msm.as_ref().map(|fit| fit.observed_subjects),
        });
        Ok(result)
    }
}

impl Study {
    pub(super) fn execute_longitudinal_regime(
        &self,
        data: &TabularData,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedLongitudinalRegimeOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

fn longitudinal_identification(
    query: &antecedent_core::LongitudinalRegimeQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
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
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from(id),
                description: Arc::from(description),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
    }
    if query.method == LongitudinalRegimeMethod::GFormula {
        if query.known_fixed_outcome_predictions {
            assumptions.push(AssumptionRecord {
                assumption: Assumption::Custom {
                    id: Arc::from("fixed_known_outcome_predictions"),
                    description: Arc::from("caller declares the conditional period reward law fixed and known before this study, with no fitted or selected Q uncertainty"),
                },
                source: AssumptionSource::UserDeclared,
                scope: AssumptionScope::Identification,
                status: AssumptionStatus::Declared,
            });
        }
        assumptions.push(AssumptionRecord {
            assumption: Assumption::Custom {
                id: Arc::from("conditional_period_reward_validity"),
                description: Arc::from("supplied period reward predictions equal conditional means under the prescribed regime given each predecision history"),
            },
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Identification,
            status: AssumptionStatus::Declared,
        });
        if query.excluded_fold_predictions {
            assumptions.push(AssumptionRecord {
                assumption: Assumption::Custom {
                    id: Arc::from("subject_excluded_fold_predictions"),
                    description: Arc::from("caller declares that every outcome prediction excludes the subject's entire history fold"),
                },
                source: AssumptionSource::UserDeclared,
                scope: AssumptionScope::Identification,
                status: AssumptionStatus::Declared,
            });
        }
    }
    if query.method == LongitudinalRegimeMethod::SequentialDoublyRobust {
        for (id, description) in [
            ("sequential_q_validity", "supplied Q is the conditional mean of the next recursive pseudo-outcome under the prescribed regime"),
            ("subject_excluded_fold_predictions", "caller declares every Q prediction excludes the subject's entire history fold"),
            ("monotone_observation_history", "dropout is monotone and terminal observation agrees with the final history period"),
        ] {
            assumptions.push(AssumptionRecord { assumption: Assumption::Custom { id: Arc::from(id), description: Arc::from(description) },
                source: AssumptionSource::UserDeclared, scope: AssumptionScope::Identification, status: AssumptionStatus::Declared });
        }
    }
    if query.method == LongitudinalRegimeMethod::MarginalStructuralModel {
        for (id, description) in [
            ("additive_marginal_structural_mean", "the terminal marginal mean is additive in binary period treatments without interactions"),
            ("stabilized_sequential_weights", "supplied treatment and censoring probabilities and prespecified numerator probabilities identify a stabilized IPTW model"),
            ("pointwise_subject_clustered_cr1", "the reported CR1 sandwich treats each whole subject history as one independent cluster and does not establish interval coverage"),
        ] {
            assumptions.push(AssumptionRecord { assumption: Assumption::Custom { id: Arc::from(id), description: Arc::from(description) },
                source: AssumptionSource::UserDeclared, scope: AssumptionScope::Identification, status: AssumptionStatus::Declared });
        }
    }
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
        LongitudinalRegimeMethod::MarginalStructuralModel => "longitudinal.marginal_structural_model",
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
