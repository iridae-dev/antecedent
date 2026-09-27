//! Retained sequential randomized regime value over subject histories.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_core::LongitudinalRegimeMethod;
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
};
use antecedent_estimate::longitudinal_regime::{evaluate_g_formula_value, evaluate_regime_value};

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
        };
        let summary = summary.map_err(|message| CausalError::Unsupported { message })?;
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
            value: summary.value,
            effective_sample_size: summary.effective_sample_size,
            matched_observed_fraction: summary.matched_observed_fraction,
            maximum_weight: summary.maximum_weight,
            minimum_action_probability: summary.minimum_action_probability,
            minimum_censoring_probability: summary.minimum_censoring_probability,
            uncertainty: Arc::from("point_only_no_interval"),
            probability_ownership: Arc::from("known_sequential_randomization"),
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
