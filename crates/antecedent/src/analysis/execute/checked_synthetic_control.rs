//! Retained graphless synthetic-control execution.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalExprArena, CausalQuery, DerivationTrace, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext, GraphClass,
    IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
    IdentifiedExecuteFinish, IdentifierId, InferenceMode, PhysicalExecutionPlan,
    RandomizedTrialOperation, RandomizedTrialQuery, RefuteSuite, Strategy, Study, StudyResult,
    TableView, TabularData, finish_identified_execute_with_context,
};
use antecedent_core::SyntheticPanelMethod;
use antecedent_estimate::synthetic_control::exact_augmented_synthetic_unit_randomization_test;
use antecedent_estimate::synthetic_control::exact_synthetic_did_unit_randomization_test;
use antecedent_estimate::synthetic_control::exact_synthetic_unit_randomization_test;
use antecedent_estimate::synthetic_control::fit_augmented_synthetic_control;
use antecedent_estimate::synthetic_control::fit_synthetic_control;
use antecedent_estimate::synthetic_control::fit_synthetic_did;
use antecedent_estimate::synthetic_control::{
    AugmentedSyntheticControlFit, SyntheticConstantEffectMethod, SyntheticControlFit,
    SyntheticUnitRandomizationTest, exact_synthetic_constant_effect_test,
};

mod bayesian;
mod rollout;

use super::BayesianConfig;
pub(super) use bayesian::bayesian_synthetic_panel_plan_ids;

/// The extra setting is the study's Bayesian configuration: with one, the retained operation
/// reports a posterior instead of the frequentist point result.
pub(crate) type CheckedSyntheticControlOperation =
    RandomizedTrialOperation<antecedent_core::SyntheticControlQuery, Option<BayesianConfig>>;

impl RandomizedTrialQuery for antecedent_core::SyntheticControlQuery {
    const OPERATION: &'static str = "CheckedSyntheticControlOperation";
    const REFRESH_MISMATCH: &'static str =
        "synthetic-control refresh requires the prepared schema and row order";
}

/// Plan id, identifier and estimator a compiled synthetic panel query records: the rollout
/// estimators of a query with intervals, otherwise those of its panel method.
pub(super) fn synthetic_query_plan_ids(
    query: &antecedent_core::SyntheticControlQuery,
) -> (&'static str, &'static str, &'static str) {
    use antecedent_core::SyntheticInterval;
    match (query.rollout.as_ref().map(|rollout| rollout.interval), query.method) {
        (Some(SyntheticInterval::Conformal), _) => (
            "quasi.synthetic_control",
            "quasi.convex_donor_counterfactual",
            "quasi.synthetic_control_conformal",
        ),
        (Some(SyntheticInterval::PlaceboVariance), SyntheticPanelMethod::Control) => (
            "quasi.synthetic_control",
            "quasi.convex_donor_counterfactual",
            "quasi.multi_unit_synthetic_control_placebo",
        ),
        (
            Some(SyntheticInterval::PlaceboVariance),
            SyntheticPanelMethod::DifferenceInDifferences,
        ) => (
            "quasi.synthetic_did",
            "quasi.convex_unit_time_trends",
            "quasi.multi_unit_synthetic_did_placebo",
        ),
        (None, method) => synthetic_panel_plan_ids(method),
    }
}

/// Plan id, identifier and estimator a compiled synthetic panel plan records.
pub(super) fn synthetic_panel_plan_ids(
    method: antecedent_core::SyntheticPanelMethod,
) -> (&'static str, &'static str, &'static str) {
    match method {
        antecedent_core::SyntheticPanelMethod::Control => (
            "quasi.synthetic_control",
            "quasi.convex_donor_counterfactual",
            "quasi.synthetic_control_simplex",
        ),
        antecedent_core::SyntheticPanelMethod::DifferenceInDifferences => {
            ("quasi.synthetic_did", "quasi.convex_unit_time_trends", "quasi.synthetic_did_simplex")
        }
    }
}

impl CheckedSyntheticControlOperation {
    pub(crate) fn checked(
        study: &Study,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
    ) -> Result<Self, CausalError> {
        let CausalQuery::SyntheticControl(query) = &study.query else {
            return Err(CausalError::Compile {
                message: "synthetic-control operation requires SyntheticControlQuery".into(),
            });
        };
        query.validate()?;
        if study.graph.class() != GraphClass::RandomizedTrial
            || study.structure_source != crate::support::StructureSource::RandomizedTrial
            || study.settings.refute != RefuteSuite::None
            || study.settings.bootstrap_replicates != 0
            || !study.declared.custom_validators.is_empty()
            || study.graph_posterior.is_some()
            || study.declared.tiered.is_some()
            || query.units.len() != data.row_count()
        {
            return Err(CausalError::Unsupported {
                message: "synthetic control requires a graphless balanced panel, one treated unit, and point-only frequentist execution or the Bayesian posterior",
            });
        }
        let bayesian = match &study.settings.inference {
            InferenceMode::Frequentist => None,
            InferenceMode::Bayesian(config) => {
                super::bayesian_quasi_common::require_built_in_model(
                    config,
                    "the Bayesian synthetic control",
                )?;
                bayesian::refuse_unread_options(query)?;
                Some(config.clone())
            }
        };
        rollout::refuse_sampling_design(query, study.declared.sampling_design)?;
        data.schema().get(query.outcome).map_err(CausalError::rejected)?;
        let (plan_id, identifier, estimator) = if bayesian.is_some() {
            bayesian_synthetic_panel_plan_ids()
        } else {
            synthetic_query_plan_ids(query)
        };
        if physical.logical.query != study.query
            || physical.logical.record.plan_id.as_ref() != plan_id
            || !physical.binding().names_wire(identifier, estimator)
        {
            return Err(CausalError::Compile {
                message: "synthetic panel design differs from its compiled plan".into(),
            });
        }
        let (identification, estimand) =
            study.design_identification_or(physical, || synthetic_control_identification(query));
        Ok(Self::freeze(study, query, data, physical, identification, estimand, bayesian))
    }

    pub(crate) fn execute(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        self.begin_execute(data, ctx)?;
        let y = match data.column(self.query.outcome).map_err(CausalError::from)? {
            antecedent_data::ColumnView::Float64(column) => column.values.as_slice(),
            _ => {
                return Err(CausalError::Unsupported {
                    message: "synthetic-control outcome must be continuous",
                });
            }
        };
        let units: Vec<String> = self.query.units.iter().map(ToString::to_string).collect();
        if self.extra.is_some() && self.query.rollout.is_some() {
            return self.execute_bayesian_rollout(data, y, ctx);
        }
        if self.extra.is_some() {
            return self.execute_bayesian(data, y, &units, ctx);
        }
        if self.query.rollout.is_some() {
            return self.execute_rollout(data, y, ctx);
        }
        if self.query.method == SyntheticPanelMethod::DifferenceInDifferences {
            return self.execute_did(data, ctx, y, &units);
        }
        let augmentation = self
            .query
            .augmentation_ridge
            .map(|ridge| {
                fit_augmented_synthetic_control(
                    y,
                    &units,
                    &self.query.periods,
                    &self.query.treated_unit,
                    self.query.intervention_period,
                    ridge,
                )
            })
            .transpose()
            .map_err(|message| CausalError::Compile { message })?;
        let fit = if let Some(augmented) = &augmentation {
            augmented.control.clone()
        } else {
            fit_synthetic_control(
                y,
                &units,
                &self.query.periods,
                &self.query.treated_unit,
                self.query.intervention_period,
            )
            .map_err(|message| CausalError::Compile { message })?
        };
        let effect = augmentation.as_ref().map_or(fit.effect, |augmented| augmented.effect);
        let randomization = self.control_randomization_test(y, &units)?;
        let squared_mass: f64 = fit.donor_weights.iter().map(|(_, weight)| weight * weight).sum();
        let effective_donors = if squared_mass > 0.0 { 1.0 / squared_mass } else { 0.0 };
        let estimate = EffectEstimate::new(
            effect,
            f64::NAN,
            self.identification.required_assumptions.clone(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        );
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(&self.physical, self.identification.clone(), self.estimand.clone(), estimate, Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::RandomizedHt), self.query.outcome, self.query.outcome).extra_diagnostics(vec![Diagnostic::new(
                    "estimate.quasi.synthetic_control.support",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!(
                        "{} donors; {} pre-periods; {} post-periods; pre-fit RMSE {}; effective donors {}; unadjusted leave-one-donor-out placebo rank {} is descriptive and uncalibrated; donor ridge correction {:?}; exact unit-randomization p-value {:?} tests the prespecified constant additive sharp null effect {} under declared uniform one-unit assignment; no interval",
                        fit.donor_weights.len(),
                        fit.n_pre_periods,
                        fit.n_post_periods,
                        fit.pre_treatment_rmse,
                        effective_donors,
                        fit.placebo_rank,
                        augmentation.as_ref().map(|fit| fit.outcome_model_correction),
                        randomization.as_ref().map(|test| test.p_value),
                        self.query.sharp_null_effect.unwrap_or(0.0)
                    ),
                )]),
        );
        result.payload = crate::result::ResultPayload::SyntheticControl(self.control_payload(
            fit,
            augmentation.as_ref(),
            randomization,
            effect,
            effective_donors,
        ));
        result.treatment = None;
        Ok(result)
    }

    /// Exact unit-randomization test of the declared sharp null, when uniform assignment was declared.
    fn control_randomization_test(
        &self,
        y: &[f64],
        units: &[String],
    ) -> Result<Option<SyntheticUnitRandomizationTest>, CausalError> {
        Ok(if self.query.uniform_unit_randomization {
            Some(
                if let Some(null_effect) = self.query.sharp_null_effect {
                    exact_synthetic_constant_effect_test(
                        y,
                        units,
                        &self.query.periods,
                        &self.query.treated_unit,
                        self.query.intervention_period,
                        null_effect,
                        self.query.augmentation_ridge.map_or(
                            SyntheticConstantEffectMethod::Control,
                            |ridge_penalty| SyntheticConstantEffectMethod::AugmentedControl {
                                ridge_penalty,
                            },
                        ),
                    )
                } else if let Some(ridge) = self.query.augmentation_ridge {
                    exact_augmented_synthetic_unit_randomization_test(
                        y,
                        units,
                        &self.query.periods,
                        &self.query.treated_unit,
                        self.query.intervention_period,
                        ridge,
                    )
                } else {
                    exact_synthetic_unit_randomization_test(
                        y,
                        units,
                        &self.query.periods,
                        &self.query.treated_unit,
                        self.query.intervention_period,
                    )
                }
                .map_err(|message| CausalError::Compile { message })?,
            )
        } else {
            None
        })
    }

    /// Assembles the retained synthetic-control estimate from the fit, augmentation, and randomization test.
    fn control_payload(
        &self,
        fit: SyntheticControlFit,
        augmentation: Option<&AugmentedSyntheticControlFit>,
        randomization: Option<SyntheticUnitRandomizationTest>,
        effect: f64,
        effective_donors: f64,
    ) -> crate::SyntheticControlEstimate {
        crate::SyntheticControlEstimate {
            effect,
            pre_treatment_rmse: fit.pre_treatment_rmse,
            donor_weights: fit
                .donor_weights
                .into_iter()
                .map(|(unit, weight)| (Arc::<str>::from(unit), weight))
                .collect(),
            placebo_effects: fit.placebo_effects.into(),
            placebo_rank: fit.placebo_rank,
            effective_donors,
            n_pre_periods: fit.n_pre_periods,
            n_post_periods: fit.n_post_periods,
            uncertainty: Arc::from(if augmentation.is_some() {
                if randomization.is_some() {
                    "point_only_augmented_with_exact_unit_randomization_p_value_no_interval"
                } else {
                    "point_only_augmented_no_interval"
                }
            } else if randomization.is_some() {
                "point_only_with_exact_unit_randomization_p_value_no_interval"
            } else {
                "point_only_with_unlicensed_placebo_rank"
            }),
            randomization_p_value: randomization.as_ref().map(|test| test.p_value),
            randomization_null_effect: randomization
                .as_ref()
                .map(|_| self.query.sharp_null_effect.unwrap_or(0.0)),
            randomization_statistics: randomization.map_or_else(
                || Arc::from([]),
                |test| {
                    test.statistics
                        .into_iter()
                        .map(|(unit, statistic)| (Arc::<str>::from(unit), statistic))
                        .collect::<Vec<_>>()
                        .into()
                },
            ),
            unadjusted_effect: augmentation.map(|_| fit.effect),
            outcome_model_correction: augmentation.map(|fit| fit.outcome_model_correction),
            augmentation_ridge: augmentation.map(|fit| fit.ridge_penalty),
        }
    }

    fn execute_did(
        &self,
        data: &TabularData,
        _ctx: &ExecutionContext,
        y: &[f64],
        units: &[String],
    ) -> Result<StudyResult, CausalError> {
        let fit = fit_synthetic_did(
            y,
            units,
            &self.query.periods,
            &self.query.treated_unit,
            self.query.intervention_period,
        )
        .map_err(|message| CausalError::Compile { message })?;
        let randomization = if self.query.uniform_unit_randomization {
            Some(
                if let Some(null_effect) = self.query.sharp_null_effect {
                    exact_synthetic_constant_effect_test(
                        y,
                        units,
                        &self.query.periods,
                        &self.query.treated_unit,
                        self.query.intervention_period,
                        null_effect,
                        SyntheticConstantEffectMethod::DifferenceInDifferences,
                    )
                } else {
                    exact_synthetic_did_unit_randomization_test(
                        y,
                        units,
                        &self.query.periods,
                        &self.query.treated_unit,
                        self.query.intervention_period,
                    )
                }
                .map_err(|message| CausalError::Compile { message })?,
            )
        } else {
            None
        };
        let estimate = EffectEstimate::new(
            fit.effect,
            f64::NAN,
            self.identification.required_assumptions.clone(),
            antecedent_estimate::OverlapPolicy::ExplicitOverride,
        );
        let mut result = finish_identified_execute_with_context(
            &self.result_context,
            Some(data),
            IdentifiedExecuteFinish::new(&self.physical, self.identification.clone(), self.estimand.clone(), estimate, Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::RandomizedHt), self.query.outcome, self.query.outcome).extra_diagnostics(vec![Diagnostic::new(
                    "estimate.quasi.synthetic_did.support",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    format!(
                        "{} donors; {} pre-periods; {} post-periods; pre-fit RMSE {}; unit and time simplex weights; exact unit-randomization p-value {:?} tests the prespecified constant additive sharp null effect {} under declared uniform one-unit assignment; no interval",
                        fit.n_donors,
                        fit.n_pre_periods,
                        fit.n_post_periods,
                        fit.pre_treatment_rmse,
                        randomization.as_ref().map(|test| test.p_value),
                        self.query.sharp_null_effect.unwrap_or(0.0)
                    ),
                )]),
        );
        result.payload = crate::result::ResultPayload::SyntheticDid(crate::SyntheticDidEstimate {
            effect: fit.effect,
            pre_treatment_rmse: fit.pre_treatment_rmse,
            donor_weights: fit
                .donor_weights
                .into_iter()
                .map(|(unit, weight)| (Arc::<str>::from(unit), weight))
                .collect(),
            time_weights: fit.time_weights.into(),
            n_donors: fit.n_donors,
            n_pre_periods: fit.n_pre_periods,
            n_post_periods: fit.n_post_periods,
            uncertainty: Arc::from(if randomization.is_some() {
                "point_only_with_exact_unit_randomization_p_value_no_interval"
            } else {
                "point_only_no_interval"
            }),
            randomization_p_value: randomization.as_ref().map(|test| test.p_value),
            randomization_null_effect: randomization
                .as_ref()
                .map(|_| self.query.sharp_null_effect.unwrap_or(0.0)),
            randomization_statistics: randomization.map_or_else(
                || Arc::from([]),
                |test| {
                    test.statistics
                        .into_iter()
                        .map(|(unit, statistic)| (Arc::<str>::from(unit), statistic))
                        .collect::<Vec<_>>()
                        .into()
                },
            ),
        });
        result.treatment = None;
        Ok(result)
    }
}

impl Study {
    pub(super) fn execute_synthetic_control(
        &self,
        data: &TabularData,
        physical: &std::sync::Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        CheckedSyntheticControlOperation::checked(self, data, physical)?.execute(data, ctx)
    }
}

/// Declared panel-design assumptions, before the method-specific counterfactual assumption is appended.
fn declared_panel_assumptions(
    query: &antecedent_core::SyntheticControlQuery,
) -> antecedent_core::AssumptionSet {
    let mut assumptions = antecedent_core::AssumptionSet::default();
    for (id, description) in [
        ("no_anticipation", "intervention does not affect pre-treatment outcomes"),
        (
            "stable_treatment_after_intervention",
            "the selected unit remains treated after intervention",
        ),
        ("no_interference_between_units", "treatment does not change donor outcomes"),
        (
            "no_concurrent_treated_unit_specific_shock",
            "no other treated-unit-specific shock starts at intervention",
        ),
        ("balanced_panel", "all donor and treated units are observed at the same periods"),
    ] {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: Arc::from(id),
                description: Arc::from(description),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    if query.uniform_unit_randomization {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: Arc::from("uniform_single_treated_unit_assignment"),
                description: Arc::from(
                    "exactly one treated unit was selected uniformly before outcomes were observed",
                ),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    if query.sharp_null_effect.is_some() {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: Arc::from("constant_additive_post_treatment_effect_sharp_null"),
                description: Arc::from("the prespecified additive effect applies to every possible selected unit and every post-intervention period"),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
    if let Some(rollout) = &query.rollout {
        push_rollout_assumptions(&mut assumptions, rollout);
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
    assumptions
}

/// Assumptions a synthetic panel with intervals adds for its adoption design, target
/// population and interval construction.
fn push_rollout_assumptions(
    assumptions: &mut antecedent_core::AssumptionSet,
    rollout: &antecedent_core::SyntheticRollout,
) {
    let declared: Vec<(&str, &str)> = match rollout.interval {
        antecedent_core::SyntheticInterval::Conformal => vec![
            (
                "post_period_residuals_exchangeable_with_pre_period_residuals",
                "under the hypothesized effect, the treated unit's synthetic-control residuals are exchangeable over time (exact) or weakly dependent and stationary (approximate)",
            ),
            (
                "target_population_treated_unit_post_periods",
                "the target is the treated unit's effect at each post period, and their average under a constant effect",
            ),
        ],
        antecedent_core::SyntheticInterval::PlaceboVariance => vec![
            (
                "staggered_adoption_with_never_treated_donors",
                "each adopting unit stays treated from its adoption period on, and the never-treated units are unaffected donors",
            ),
            (
                "treated_untreated_errors_exchangeable_with_donors",
                "the adopting units' untreated errors are exchangeable with the never-treated units', so placebo reassignments estimate the sampling variance",
            ),
            (
                "target_population_adopting_units_post_periods",
                "the target is the post-period-weighted average effect over the adopting units' post-adoption periods",
            ),
        ],
    };
    for (id, description) in declared {
        assumptions.push(antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::Custom {
                id: Arc::from(id),
                description: Arc::from(description),
            },
            source: antecedent_core::AssumptionSource::UserDeclared,
            scope: antecedent_core::AssumptionScope::Identification,
            status: antecedent_core::AssumptionStatus::Declared,
        });
    }
}

pub(crate) fn synthetic_control_identification(
    query: &antecedent_core::SyntheticControlQuery,
) -> (IdentificationResult, IdentifiedEstimand) {
    let mut assumptions = declared_panel_assumptions(query);
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
            id: Arc::from(method_assumption),
            description: Arc::from(method_description),
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
        outcomes,
        empty,
        intervention,
        antecedent_expr::DomainRef::Observational,
    );
    let functional = arena.intern(antecedent_expr::ExprNode::Expectation {
        function: antecedent_expr::OutcomeExprId::identity(query.outcome),
        distribution,
    });
    arena.set_derivation(
        functional,
        antecedent_expr::DerivationMeta::rule(
            rule,
            Some(Arc::from("one treated unit and a balanced donor pool bind the observed panel")),
        ),
    );
    let estimand = IdentifiedEstimand::new(
        rule,
        Arc::from([]),
        Arc::from([]),
        Arc::from([]),
        functional,
        None,
    );
    let mut derivation = DerivationTrace::default();
    derivation.push(rule, derivation_description);
    let identification = IdentificationResult::identified(
        CausalQuery::SyntheticControl(query.clone()),
        vec![estimand.clone()],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord::default(),
    );
    (identification, estimand)
}
