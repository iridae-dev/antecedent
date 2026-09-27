//! Retained graphless synthetic-control execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use antecedent_estimate::synthetic_control::fit_synthetic_control;
use antecedent_estimate::synthetic_control::fit_augmented_synthetic_control;
use antecedent_estimate::synthetic_control::exact_synthetic_unit_randomization_test;
use antecedent_estimate::synthetic_control::fit_synthetic_did;
use antecedent_core::SyntheticPanelMethod;

#[derive(Clone)]
pub(crate) struct CheckedSyntheticControlOperation {
    query: antecedent_core::SyntheticControlQuery,
    schema: antecedent_core::CausalSchema,
    rows: usize,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    physical: PhysicalExecutionPlan,
    result_context: IdentifiedResultContext,
}

impl std::fmt::Debug for CheckedSyntheticControlOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckedSyntheticControlOperation")
            .field("query", &self.query).field("rows", &self.rows).finish_non_exhaustive()
    }
}

impl CheckedSyntheticControlOperation {
    pub(crate) fn checked(
        study: &Study, data: &TabularData, physical: &PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        let CausalQuery::SyntheticControl(query) = &study.query else {
            return Err(CausalError::Compile { message: "synthetic-control operation requires SyntheticControlQuery".into() });
        };
        query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || !matches!(study.inference, InferenceMode::Frequentist)
            || study.refute != RefuteSuite::None
            || study.bootstrap_replicates != 0
            || !study.custom_validators.is_empty()
            || query.units.len() != data.row_count()
        {
            return Err(CausalError::Unsupported {
                message: "synthetic control requires a graphless balanced panel, one treated unit, and point-only frequentist execution",
            });
        }
        data.schema().get(query.outcome)
            .map_err(|error| CausalError::Compile { message: error.to_string() })?;
        let (identification, estimand) = synthetic_control_identification(query);
        Ok(Self {
            query: query.clone(), schema: data.schema().clone(), rows: data.row_count(),
            identification, estimand, physical: physical.clone(),
            result_context: IdentifiedResultContext::from_study(study),
        })
    }

    pub(crate) fn execute(&self, data: &TabularData, ctx: &ExecutionContext) -> Result<StudyResult, CausalError> {
        if data.schema() != &self.schema || data.row_count() != self.rows {
            return Err(CausalError::Unsupported { message: "synthetic-control refresh requires the prepared schema and row order" });
        }
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::super::stage::STAGE_ESTIMATE_POINT });
        }
        let y = match data.column(self.query.outcome).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(column) => column.values.as_slice(),
            _ => return Err(CausalError::Unsupported { message: "synthetic-control outcome must be continuous" }),
        };
        let units: Vec<String> = self.query.units.iter().map(ToString::to_string).collect();
        if self.query.method == SyntheticPanelMethod::DifferenceInDifferences {
            return self.execute_did(data, ctx, y, &units);
        }
        let augmentation = self.query.augmentation_ridge.map(|ridge|
            fit_augmented_synthetic_control(y, &units, &self.query.periods, &self.query.treated_unit,
                self.query.intervention_period, ridge)).transpose()
            .map_err(|message| CausalError::Compile { message })?;
        let fit = if let Some(augmented) = &augmentation { augmented.control.clone() } else {
            fit_synthetic_control(y, &units, &self.query.periods, &self.query.treated_unit,
                self.query.intervention_period).map_err(|message| CausalError::Compile { message })?
        };
        let effect = augmentation.as_ref().map_or(fit.effect, |augmented| augmented.effect);
        let randomization = if self.query.uniform_unit_randomization {
            Some(exact_synthetic_unit_randomization_test(y, &units, &self.query.periods,
                &self.query.treated_unit, self.query.intervention_period)
                .map_err(|message| CausalError::Compile { message })?)
        } else { None };
        let squared_mass: f64 = fit.donor_weights.iter().map(|(_, weight)| weight * weight).sum();
        let effective_donors = if squared_mass > 0.0 { 1.0 / squared_mass } else { 0.0 };
        let estimate = EffectEstimate::new(
            effect, f64::NAN, self.identification.required_assumptions.clone(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        );
        let mut result = finish_identified_execute_with_context(
            &self.result_context, Some(data), IdentifiedExecuteFinish {
                physical: &self.physical,
                identification: self.identification.clone(), estimand: self.estimand.clone(),
                estimate, identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: EstimatorId::RandomizedHt,
                treatment: self.query.outcome, outcome: self.query.outcome,
                identify_cached: false,
                extra_diagnostics: vec![Diagnostic::new(
                    "estimate.quasi.synthetic_control.support",
                    DiagnosticKind::Scientific, DiagnosticSeverity::Info,
                    format!("{} donors; {} pre-periods; {} post-periods; pre-fit RMSE {}; effective donors {}; unadjusted leave-one-donor-out placebo rank {} is descriptive and uncalibrated; donor ridge correction {:?}; exact unit-randomization p-value {:?} applies only under declared uniform one-unit assignment and the sharp null; no interval",
                        fit.donor_weights.len(), fit.n_pre_periods, fit.n_post_periods,
                        fit.pre_treatment_rmse, effective_donors, fit.placebo_rank,
                        augmentation.as_ref().map(|fit| fit.outcome_model_correction),
                        randomization.as_ref().map(|test| test.p_value)),
                )],
                refutations: Vec::new(), distribution: None, mediation: None,
                wall_time_ns: 0, bootstrap_replicates_ok: None,
                cancelled: false, early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.synthetic_control = Some(crate::SyntheticControlEstimate {
            effect,
            pre_treatment_rmse: fit.pre_treatment_rmse,
            donor_weights: fit.donor_weights.into_iter()
                .map(|(unit, weight)| (Arc::<str>::from(unit), weight)).collect(),
            placebo_effects: fit.placebo_effects.into(),
            placebo_rank: fit.placebo_rank,
            effective_donors,
            n_pre_periods: fit.n_pre_periods,
            n_post_periods: fit.n_post_periods,
            uncertainty: Arc::from(if augmentation.is_some() {
                "point_only_augmented_no_interval"
            } else if randomization.is_some() {
                "point_only_with_exact_unit_randomization_p_value_no_interval"
            } else { "point_only_with_unlicensed_placebo_rank" }),
            randomization_p_value: randomization.as_ref().map(|test| test.p_value),
            randomization_statistics: randomization.map_or_else(
                || Arc::from([]),
                |test| test.statistics.into_iter().map(|(unit, statistic)|
                    (Arc::<str>::from(unit), statistic)).collect::<Vec<_>>().into(),
            ),
            unadjusted_effect: augmentation.as_ref().map(|_| fit.effect),
            outcome_model_correction: augmentation.as_ref().map(|fit| fit.outcome_model_correction),
            augmentation_ridge: augmentation.as_ref().map(|fit| fit.ridge_penalty),
        });
        result.treatment = None;
        Ok(result)
    }

    fn execute_did(
        &self, data: &TabularData, _ctx: &ExecutionContext, y: &[f64], units: &[String],
    ) -> Result<StudyResult, CausalError> {
        let fit = fit_synthetic_did(y, units, &self.query.periods, &self.query.treated_unit,
            self.query.intervention_period).map_err(|message| CausalError::Compile { message })?;
        let estimate = EffectEstimate::new(
            fit.effect, f64::NAN, self.identification.required_assumptions.clone(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        );
        let mut result = finish_identified_execute_with_context(
            &self.result_context, Some(data), IdentifiedExecuteFinish {
                physical: &self.physical,
                identification: self.identification.clone(), estimand: self.estimand.clone(),
                estimate, identifier_id: IdentifierId::RandomizedDesign,
                estimator_id: EstimatorId::RandomizedHt,
                treatment: self.query.outcome, outcome: self.query.outcome,
                identify_cached: false,
                extra_diagnostics: vec![Diagnostic::new(
                    "estimate.quasi.synthetic_did.support",
                    DiagnosticKind::Scientific, DiagnosticSeverity::Info,
                    format!("{} donors; {} pre-periods; {} post-periods; pre-fit RMSE {}; unit and time simplex weights; no calibrated interval",
                        fit.n_donors, fit.n_pre_periods, fit.n_post_periods, fit.pre_treatment_rmse),
                )],
                refutations: Vec::new(), distribution: None, mediation: None,
                wall_time_ns: 0, bootstrap_replicates_ok: None,
                cancelled: false, early_stopped: false,
                extras: IdentifiedExecuteExtras::default(),
            },
        );
        result.synthetic_did = Some(crate::SyntheticDidEstimate {
            effect: fit.effect,
            pre_treatment_rmse: fit.pre_treatment_rmse,
            donor_weights: fit.donor_weights.into_iter().map(|(unit, weight)| (Arc::<str>::from(unit), weight)).collect(),
            time_weights: fit.time_weights.into(),
            n_donors: fit.n_donors,
            n_pre_periods: fit.n_pre_periods,
            n_post_periods: fit.n_post_periods,
            uncertainty: Arc::from("point_only_no_interval"),
        });
        result.treatment = None;
        Ok(result)
    }
}

impl Study {
    pub(super) fn execute_synthetic_control(
        &self, data: &TabularData, physical: &PhysicalExecutionPlan, ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedSyntheticControlOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

pub(crate) fn synthetic_control_identification(
    query: &antecedent_core::SyntheticControlQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    for (id, description) in [
        ("no_anticipation", "intervention does not affect pre-treatment outcomes"),
        ("stable_treatment_after_intervention", "the selected unit remains treated after intervention"),
        ("no_interference_between_units", "treatment does not change donor outcomes"),
        ("no_concurrent_treated_unit_specific_shock", "no other treated-unit-specific shock starts at intervention"),
        ("balanced_panel", "all donor and treated units are observed at the same periods"),
    ] {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom { id: Arc::from(id), description: Arc::from(description) },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    if query.uniform_unit_randomization {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: Arc::from("uniform_single_treated_unit_assignment"),
                description: Arc::from("exactly one treated unit was selected uniformly before outcomes were observed"),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    if query.augmentation_ridge.is_some() {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: Arc::from("donor_ridge_outcome_model_transports"),
                description: Arc::from("the donor-trained pre-trajectory outcome model predicts the treated unit's untreated post-period mean"),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    let (method_assumption, method_description, rule, derivation_description) = match query.method {
        SyntheticPanelMethod::Control if query.augmentation_ridge.is_some() => (
            "donor_outcome_model_correction_is_valid",
            "a donor-trained ridge outcome model corrects the remaining convex-donor imbalance",
            "synthetic_control.augmented_donor_ridge",
            "the treated post-period mean minus the convex donor mean is corrected by the donor-trained ridge prediction difference under the declared transport assumption",
        ),
        SyntheticPanelMethod::Control => (
            "convex_donor_combination_is_a_valid_counterfactual",
            "a convex combination of donor outcomes represents the treated unit without intervention",
            "synthetic_control.convex_donor_pre_fit",
            "the treated unit's post-intervention mean gap from its pre-fitted convex donor counterfactual is identified under the declared donor validity assumptions",
        ),
        SyntheticPanelMethod::DifferenceInDifferences => (
            "convex_unit_and_time_weights_represent_untreated_counterfactual_trends",
            "convex donor and pre-period weights represent the untreated trend counterfactual",
            "synthetic_did.convex_unit_time_weights",
            "the weighted post-versus-pre treated change minus the donor change is identified under the declared trend validity assumptions",
        ),
    };
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::Custom {
            id: Arc::from(method_assumption), description: Arc::from(method_description),
        },
        source: antecedent_core::AssumptionSource::UserDeclared,
        scope: antecedent_core::AssumptionScope::Identification,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let mut arena = CausalExprArena::new();
    let outcomes = arena.intern_var_set([query.outcome]);
    let empty = arena.empty_var_set();
    let intervention = arena.intern_intervention_set([]);
    let distribution = arena.intern_distribution(
        outcomes, empty, intervention, antecedent_expr::DomainRef::Observational,
    );
    let functional = arena.intern(antecedent_expr::ExprNode::Expectation {
        function: antecedent_expr::OutcomeExprId::identity(query.outcome), distribution,
    });
    arena.set_derivation(functional, antecedent_expr::DerivationMeta::rule(
        rule, Some(Arc::from("one treated unit and a balanced donor pool bind the observed panel")),
    ));
    let estimand = IdentifiedEstimand::new(rule, Arc::from([]), Arc::from([]), Arc::from([]), functional, None);
    let mut derivation = DerivationTrace::default();
    derivation.push(rule, derivation_description);
    let identification = IdentificationResult::identified(
        CausalQuery::SyntheticControl(query.clone()), vec![estimand.clone()], arena,
        derivation, assumptions, IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}
