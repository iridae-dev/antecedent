//! Encoding of typed queries into their wire forms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Encode any [`CausalQuery`].
///
/// # Errors
///
/// Unsupported nested fields.
pub fn causal_query_to_wire(q: &CausalQuery) -> Result<CausalQueryWire, IoError> {
    causal_query_to_wire_with_registry(q, None)
}

/// Encode any [`CausalQuery`], resolving registry-backed populations.
///
/// # Errors
///
/// Unsupported nested fields or missing registry entries.
pub fn causal_query_to_wire_with_registry(
    q: &CausalQuery,
    registry: Option<&PopulationRegistry>,
) -> Result<CausalQueryWire, IoError> {
    Ok(match q {
        CausalQuery::AverageEffect(q) => average_effect_to_wire(q, registry)?,
        CausalQuery::TemporalEffect(q) => temporal_effect_to_wire(q, registry)?,
        CausalQuery::Counterfactual(q) => counterfactual_to_wire(q)?,
        CausalQuery::NestedCounterfactual(q) => CausalQueryWire::NestedCounterfactual {
            treatment: q.treatment.raw(),
            mediator: q.mediator.raw(),
            outcome: q.outcome.raw(),
            control_bits: q.control_value().to_bits(),
            active_bits: q.active_value().to_bits(),
        },
        CausalQuery::AnomalyAttribution(q) => anomaly_attribution_to_wire(q)?,
        CausalQuery::ChangeAttribution(q) => change_attribution_to_wire(q)?,
        CausalQuery::MechanismChange(q) => mechanism_change_to_wire(q)?,
        CausalQuery::UnitChange(q) => unit_change_to_wire(q)?,
        CausalQuery::Mediation(q) => mediation_to_wire(q, registry)?,
        CausalQuery::ConditionalEffect(q) => CausalQueryWire::ConditionalEffect {
            inner: Box::new(causal_query_to_wire_with_registry(
                &CausalQuery::AverageEffect(q.inner.clone()),
                registry,
            )?),
        },
        CausalQuery::Distribution(q) => {
            CausalQueryWire::Distribution(interventional_distribution_to_wire(q)?)
        }
        CausalQuery::PathSpecific(q) => CausalQueryWire::PathSpecific(path_specific_to_wire(q)?),
        CausalQuery::Response(q) => CausalQueryWire::Response(response_query_to_wire(q)?),
        CausalQuery::Transport(q) => CausalQueryWire::Transport(transport_query_to_wire(q)?),
        CausalQuery::Interference(q) => {
            CausalQueryWire::Interference(interference_query_to_wire(q)?)
        }
        CausalQuery::RandomizedEffect(q) => randomized_effect_to_wire(q),
        CausalQuery::PolicyValue(q) => policy_value_to_wire(q),
        CausalQuery::ContinuousDoseResponse(q) => continuous_dose_response_to_wire(q),
        CausalQuery::PanelDid(q) => panel_did_to_wire(q),
        CausalQuery::SyntheticControl(q) => synthetic_control_to_wire(q),
        CausalQuery::LocalPolynomialRatio(q) => local_polynomial_ratio_to_wire(q),
        CausalQuery::Survival(q) => survival_to_wire(q),
        CausalQuery::LongitudinalRegime(q) => longitudinal_regime_to_wire(q),
        CausalQuery::SequentialContrast(q) => sequential_contrast_to_wire(q),
        CausalQuery::HistoryMatch(q) => history_match_to_wire(q),
        CausalQuery::Discover(q) => CausalQueryWire::Discover(
            crate::structure_query_wire::DiscoverQueryWire::from_domain(q),
        ),
        CausalQuery::Attribute(q) => CausalQueryWire::Attribute(
            crate::structure_query_wire::AttributeQueryWire::from_domain(q)?,
        ),
        CausalQuery::Simulate(q) => CausalQueryWire::Simulate(
            crate::structure_query_wire::SimulateQueryWire::from_domain(q)?,
        ),
        CausalQuery::TemporalDesign(q) => CausalQueryWire::TemporalDesign(
            crate::temporal_design_query_wire::TemporalDesignQueryWire::from_domain(q)?,
        ),
        CausalQuery::IvBounds(q) => CausalQueryWire::IvBounds(IvBoundsQueryWire {
            outcome: q.outcome.raw(),
            treatment: q.treatment.raw(),
            instrument: q.instrument.raw(),
            weight: q.weight.map(antecedent_core::VariableId::raw),
        }),
        CausalQuery::Association(q) => CausalQueryWire::Association(association_query_to_wire(q)),
        _ => return Err(IoError::Convert("unsupported CausalQuery variant".into())),
    })
}

/// Encode the [`CausalQuery::AverageEffect`] payload.
fn average_effect_to_wire(
    q: &antecedent_core::AverageEffectQuery,
    registry: Option<&PopulationRegistry>,
) -> Result<CausalQueryWire, IoError> {
    Ok(CausalQueryWire::AverageEffect {
        treatment: q.treatment.raw(),
        outcome: q.outcome.raw(),
        effect_modifiers: vars_to_raw(&q.effect_modifiers),
        control: InterventionWire::from_domain(&q.control)?,
        active: InterventionWire::from_domain(&q.active)?,
        target_population: TargetPopulationWire::from_domain_with_registry(
            &q.target_population,
            registry,
        )?,
        outcome_functional: OutcomeFunctionalWire::from_domain(&q.outcome_functional),
    })
}

/// Encode the [`CausalQuery::TemporalEffect`] payload.
fn temporal_effect_to_wire(
    q: &antecedent_core::TemporalEffectQuery,
    registry: Option<&PopulationRegistry>,
) -> Result<CausalQueryWire, IoError> {
    Ok(CausalQueryWire::TemporalEffect {
        treatment: q.treatment.raw(),
        outcome: q.outcome.raw(),
        policy: TemporalPolicyWire::from_domain(&q.policy)?,
        control: InterventionWire::from_domain(&q.control)?,
        active: InterventionWire::from_domain(&q.active)?,
        horizon_steps: q.horizon_steps,
        max_history_lag: q.max_history_lag,
        target_population: TargetPopulationWire::from_domain_with_registry(
            &q.target_population,
            registry,
        )?,
        effect_modifiers: q.effect_modifiers.iter().map(|id| id.raw()).collect(),
    })
}

/// Encode the [`CausalQuery::Counterfactual`] payload.
fn counterfactual_to_wire(
    q: &antecedent_core::CounterfactualQuery,
) -> Result<CausalQueryWire, IoError> {
    Ok(CausalQueryWire::Counterfactual {
        outcomes: vars_to_raw(&q.outcomes),
        interventions: q
            .interventions
            .iter()
            .map(InterventionWire::from_domain)
            .collect::<Result<Vec<_>, _>>()?,
        control: Some(InterventionWire::from_domain(&q.control)?),
        joint_control: q
            .joint_control
            .iter()
            .map(InterventionWire::from_domain)
            .collect::<Result<Vec<_>, _>>()?,
        allow_nested: q.allow_nested,
    })
}

/// Encode the [`CausalQuery::AnomalyAttribution`] payload.
fn anomaly_attribution_to_wire(
    q: &antecedent_core::AnomalyAttributionQuery,
) -> Result<CausalQueryWire, IoError> {
    Ok(CausalQueryWire::AnomalyAttribution {
        targets: vars_to_raw(&q.targets),
        unit_rows: q
            .unit_rows
            .as_ref()
            .map(|rows| {
                rows.iter()
                    .map(|&r| u64::try_from(r).map_err(|_| IoError::TooLarge))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?,
        max_units: crate::error::fit_u64(q.max_units)?,
        reference: Some(match q.reference {
            AnomalyReference::Empirical => None,
            AnomalyReference::Fixed { center, scale } => Some((center.to_f64(), scale.to_f64())),
        }),
    })
}

/// Encode the [`CausalQuery::ChangeAttribution`] payload.
fn change_attribution_to_wire(
    q: &antecedent_core::ChangeAttributionQuery,
) -> Result<CausalQueryWire, IoError> {
    Ok(CausalQueryWire::ChangeAttribution {
        outcome: q.outcome.raw(),
        baseline: population_to_wire(&q.baseline)?,
        comparison: population_to_wire(&q.comparison)?,
        components: q.components.try_into()?,
        allocation: allocation_to_wire(&q.allocation)?,
        max_components: crate::error::fit_u64(q.max_components)?,
    })
}

/// Encode the [`CausalQuery::MechanismChange`] payload.
fn mechanism_change_to_wire(
    q: &antecedent_core::MechanismChangeQuery,
) -> Result<CausalQueryWire, IoError> {
    Ok(CausalQueryWire::MechanismChange {
        targets: vars_to_raw(&q.targets),
        baseline: population_to_wire(&q.baseline)?,
        comparison: population_to_wire(&q.comparison)?,
        significance_level_bits: q.significance_level.to_f64().to_bits(),
        max_targets: crate::error::fit_u64(q.max_targets)?,
        change_prior_bits: q.change_prior.map(|p| p.to_f64().to_bits()),
    })
}

/// Encode the [`CausalQuery::UnitChange`] payload.
fn unit_change_to_wire(q: &antecedent_core::UnitChangeQuery) -> Result<CausalQueryWire, IoError> {
    Ok(CausalQueryWire::UnitChange {
        outcome: q.outcome.raw(),
        unit_rows: q
            .unit_rows
            .as_ref()
            .map(|rows| {
                rows.iter()
                    .map(|&r| u64::try_from(r).map_err(|_| IoError::TooLarge))
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?,
        components: q.components.try_into()?,
        allocation: allocation_to_wire(&q.allocation)?,
        max_units: crate::error::fit_u64(q.max_units)?,
    })
}

/// Encode the [`CausalQuery::Mediation`] payload.
fn mediation_to_wire(
    q: &antecedent_core::MediationQuery,
    registry: Option<&PopulationRegistry>,
) -> Result<CausalQueryWire, IoError> {
    Ok(CausalQueryWire::Mediation {
        treatment: q.treatment.raw(),
        outcome: q.outcome.raw(),
        mediators: vars_to_raw(&q.mediators),
        contrast: q.contrast.into(),
        control: InterventionWire::from_domain(&q.control)?,
        active: InterventionWire::from_domain(&q.active)?,
        target_population: TargetPopulationWire::from_domain_with_registry(
            &q.target_population,
            registry,
        )?,
        horizons: q.horizons.iter().copied().collect(),
    })
}

/// Encode the [`CausalQuery::RandomizedEffect`] payload.
fn randomized_effect_to_wire(q: &antecedent_core::RandomizedEffectQuery) -> CausalQueryWire {
    CausalQueryWire::RandomizedEffect(RandomizedEffectQueryWire {
        estimand: match q.estimand {
            antecedent_core::RandomizedEstimand::IntentionToTreat => RandomizedEstimandWire::Itt,
            antecedent_core::RandomizedEstimand::ComplierAverageCausalEffect => {
                RandomizedEstimandWire::CaceLate
            }
            antecedent_core::RandomizedEstimand::TreatmentOnTreated => {
                RandomizedEstimandWire::TreatmentOnTreated
            }
        },
        design: match &q.design {
            RandomizationDesign::Bernoulli => RandomizationDesignWire::Bernoulli,
            RandomizationDesign::Complete { treated_units } => {
                RandomizationDesignWire::Complete { treated_units: *treated_units }
            }
            RandomizationDesign::Cluster { treated_clusters } => {
                RandomizationDesignWire::Cluster { treated_clusters: *treated_clusters }
            }
            RandomizationDesign::Stratified { .. } => RandomizationDesignWire::Stratified,
            RandomizationDesign::Factorial2x2 { .. } => RandomizationDesignWire::Factorial2x2,
            RandomizationDesign::MultiArm { .. } => RandomizationDesignWire::MultiArm,
            RandomizationDesign::Switchback { .. } => RandomizationDesignWire::Switchback,
        },
        outcome: q.outcome.raw(),
        realized_assignment: q.realized_assignment.to_vec(),
        assignment_probabilities: q.assignment_probabilities.to_vec(),
        assignment_units: q.assignment_units.iter().map(ToString::to_string).collect(),
        outcome_units: q.outcome_units.iter().map(ToString::to_string).collect(),
        treatment_arms: (q.treatment_arms.0.to_string(), q.treatment_arms.1.to_string()),
        blocks: match &q.design {
            RandomizationDesign::Stratified { blocks, .. } => {
                blocks.iter().map(ToString::to_string).collect()
            }
            _ => Vec::new(),
        },
        treated_per_row: match &q.design {
            RandomizationDesign::Stratified { treated_per_row, .. } => treated_per_row.to_vec(),
            _ => Vec::new(),
        },
        fixed_cuped: q.fixed_cuped.map(|(id, coefficient)| (id.raw(), coefficient)),
        ancova_covariates: q.ancova_covariates.iter().map(|id| id.raw()).collect(),
        received_treatment: q.received_treatment.as_ref().map(|receipt| receipt.to_vec()),
        exact_randomization_test: q.exact_randomization_test,
        second_factor_assignment: match &q.design {
            RandomizationDesign::Factorial2x2 { second_factor_assignment, .. } => {
                second_factor_assignment.to_vec()
            }
            _ => Vec::new(),
        },
        factorial_cell_counts: match &q.design {
            RandomizationDesign::Factorial2x2 { cell_counts, .. } => Some(*cell_counts),
            _ => None,
        },
        second_factor_arms: match &q.design {
            RandomizationDesign::Factorial2x2 { second_factor_arms, .. } => {
                Some((second_factor_arms.0.to_string(), second_factor_arms.1.to_string()))
            }
            _ => None,
        },
        multi_arm_labels: match &q.design {
            RandomizationDesign::MultiArm { arms, .. } => {
                arms.iter().map(ToString::to_string).collect()
            }
            _ => Vec::new(),
        },
        multi_arm_assignment: match &q.design {
            RandomizationDesign::MultiArm { assignment, .. } => assignment.to_vec(),
            _ => Vec::new(),
        },
        multi_arm_probabilities: match &q.design {
            RandomizationDesign::MultiArm { probabilities, .. } => probabilities.to_vec(),
            _ => Vec::new(),
        },
        periods: match &q.design {
            RandomizationDesign::Switchback { periods } => {
                periods.iter().map(ToString::to_string).collect()
            }
            _ => Vec::new(),
        },
    })
}

/// Encode the [`CausalQuery::PolicyValue`] payload.
fn policy_value_to_wire(q: &antecedent_core::PolicyValueQuery) -> CausalQueryWire {
    CausalQueryWire::PolicyValue(PolicyValueQueryWire {
        outcome: q.outcome.raw(),
        assignment: q.assignment.to_vec(),
        propensity: q.propensity.to_vec(),
        actions: q.actions.to_vec(),
        reference: q.reference.to_vec(),
        mu0: q.mu0.to_vec(),
        mu1: q.mu1.to_vec(),
        costs: q.costs.to_vec(),
        reference_costs: q.reference_costs.to_vec(),
        evaluation_subject_ids: q.evaluation_subject_ids.iter().map(ToString::to_string).collect(),
        disjoint_training_subjects: q.disjoint_training_subjects,
        crossfit_fold_ownership_valid: q.crossfit_fold_ownership_valid,
        global_constraints_present: q.global_constraints_present,
        multi_action: q.multi_action.as_ref().map(Into::into),
        uplift_bins: q.uplift_bins.to_vec(),
        uplift_bin_count: q.uplift_bin_count,
        uplift_training_subject_ids: q
            .uplift_training_subject_ids
            .iter()
            .map(ToString::to_string)
            .collect(),
        regret: q.regret.as_ref().map(|regret| FixedCandidateRegretWire {
            candidates: regret.candidates.iter().map(|candidate| candidate.to_vec()).collect(),
            selected_index: regret.selected_index,
            training_subject_ids: regret
                .training_subject_ids
                .iter()
                .map(ToString::to_string)
                .collect(),
        }),
    })
}

/// Encode the [`CausalQuery::ContinuousDoseResponse`] payload.
fn continuous_dose_response_to_wire(
    q: &antecedent_core::ContinuousDoseResponseQuery,
) -> CausalQueryWire {
    CausalQueryWire::ContinuousDoseResponse(ContinuousDoseResponseQueryWire {
        outcome: q.outcome.raw(),
        dose: q.dose.raw(),
        dose_density: q.dose_density.raw(),
        baseline_groups: q.baseline_groups.iter().map(ToString::to_string).collect(),
        target_doses: q.target_doses.to_vec(),
        bandwidth: q.bandwidth,
        min_local_support: q.min_local_support,
        density_provenance: q.density_provenance.to_string(),
        fixed_policy: q.fixed_policy.as_ref().map(|rule| {
            (
                rule.policy_doses.iter().map(|(group, dose)| (group.to_string(), *dose)).collect(),
                rule.reference_doses
                    .iter()
                    .map(|(group, dose)| (group.to_string(), *dose))
                    .collect(),
            )
        }),
    })
}

/// Encode the [`CausalQuery::PanelDid`] payload.
fn panel_did_to_wire(q: &antecedent_core::PanelDidQuery) -> CausalQueryWire {
    CausalQueryWire::PanelDid(PanelDidQueryWire {
        outcome: q.outcome.raw(),
        augmented: q.augmented.as_ref().map(|n| {
            (
                n.outcome_pre.raw(),
                n.propensity.raw(),
                n.untreated_change_prediction.raw(),
                n.predictions_cross_fitted,
            )
        }),
        repeated_cross_section: q.design
            == antecedent_core::DidSamplingDesign::RepeatedCrossSection,
        staggered_target: q.target,
        staggered_event_study: q.design == antecedent_core::DidSamplingDesign::StaggeredEventStudy,
        periods: q.periods.to_vec(),
        cohorts: q.cohorts.to_vec(),
        treated: q.treated.to_vec(),
        post: q.post.to_vec(),
        subjects: q.subjects.iter().map(ToString::to_string).collect(),
        clusters: q.clusters.iter().map(ToString::to_string).collect(),
        rollout: q
            .robust
            .as_ref()
            .map(|robust| crate::rollout_query_wire::rollout_to_wire(q.design, robust)),
        resampling: q.resampling.as_ref().map(crate::rollout_query_wire::resampling_to_wire),
    })
}

/// Encode the [`CausalQuery::SyntheticControl`] payload.
fn synthetic_control_to_wire(q: &antecedent_core::SyntheticControlQuery) -> CausalQueryWire {
    CausalQueryWire::SyntheticControl(SyntheticControlQueryWire {
        outcome: q.outcome.raw(),
        units: q.units.iter().map(ToString::to_string).collect(),
        periods: q.periods.to_vec(),
        treated_unit: q.treated_unit.to_string(),
        intervention_period: q.intervention_period,
        difference_in_differences: q.method
            == antecedent_core::SyntheticPanelMethod::DifferenceInDifferences,
        uniform_unit_randomization: q.uniform_unit_randomization,
        sharp_null_effect: q.sharp_null_effect,
        augmentation_ridge: q.augmentation_ridge,
        rollout: q.rollout.as_ref().map(crate::rollout_query_wire::synthetic_rollout_to_wire),
        observed_window_support: q.observed_window_support,
    })
}

/// Encode the [`CausalQuery::LocalPolynomialRatio`] payload.
fn local_polynomial_ratio_to_wire(
    q: &antecedent_core::LocalPolynomialRatioQuery,
) -> CausalQueryWire {
    CausalQueryWire::LocalPolynomialRatio(LocalPolynomialRatioQueryWire {
        outcome: q.outcome.raw(),
        treatment: q.treatment.raw(),
        running: q.running.raw(),
        cutoff: q.cutoff,
        bandwidth: q.bandwidth,
        kink: q.kink,
        options: q.options.as_ref().map(crate::rollout_query_wire::local_ratio_options_to_wire),
    })
}

/// Encode the [`CausalQuery::Survival`] payload.
fn survival_to_wire(q: &antecedent_core::SurvivalQuery) -> CausalQueryWire {
    CausalQueryWire::Survival(SurvivalQueryWire {
        duration: q.duration.raw(),
        event: q.event.raw(),
        treatment: q.treatment.raw(),
        tau: q.tau,
        delayed_entry: q.delayed_entry.map(antecedent_core::VariableId::raw),
        target_cause: match q.functional {
            antecedent_core::SurvivalFunctional::SurvivalAndRmst => None,
            antecedent_core::SurvivalFunctional::CumulativeIncidence { target_cause } => {
                Some(target_cause)
            }
        },
        independent_observation: matches!(
            &q.observation_assumption,
            antecedent_core::ObservationAssumption::IndependentGiven(_)
        ),
        independent_given: match &q.observation_assumption {
            antecedent_core::ObservationAssumption::IndependentGiven(vars) => {
                vars.iter().map(|id| id.raw()).collect()
            }
            _ => Vec::new(),
        },
        censoring_times: q
            .known_censoring
            .as_ref()
            .map_or_else(Vec::new, |known| known.times.to_vec()),
        censoring_columns: q
            .known_censoring
            .as_ref()
            .map_or_else(Vec::new, |known| known.columns.iter().map(|id| id.raw()).collect()),
        censoring_probability_floor: q
            .known_censoring
            .as_ref()
            .map(|known| known.minimum_probability),
    })
}

/// Encode the [`CausalQuery::LongitudinalRegime`] payload.
fn longitudinal_regime_to_wire(q: &antecedent_core::LongitudinalRegimeQuery) -> CausalQueryWire {
    CausalQueryWire::LongitudinalRegime(LongitudinalRegimeQueryWire {
        outcome: q.outcome.raw(),
        method: match q.method {
            antecedent_core::LongitudinalRegimeMethod::Ipw => "ipw",
            antecedent_core::LongitudinalRegimeMethod::GFormula => "g_formula",
            antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust => "sequential_dr",
            antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel => {
                "marginal_structural_model"
            }
        }
        .into(),
        period_outcome_predictions: q
            .period_outcome_predictions
            .as_ref()
            .map_or_else(Vec::new, |q| q.to_vec()),
        known_fixed_outcome_predictions: q.known_fixed_outcome_predictions,
        stabilizing_numerator_probabilities: q
            .stabilizing_numerator_probabilities
            .as_ref()
            .map_or_else(Vec::new, |p| p.to_vec()),
        q_predictions: q.q_predictions.as_ref().map_or_else(Vec::new, |q| q.to_vec()),
        observation_history: q.observation_history.as_ref().map_or_else(Vec::new, |o| o.to_vec()),
        prediction_fold_ids: q.prediction_fold_ids.as_ref().map_or_else(Vec::new, |f| f.to_vec()),
        periods: q.periods,
        treatment_history: q.treatment_history.to_vec(),
        regime_actions: q.regime_actions.to_vec(),
        treatment_probabilities: q.treatment_probabilities.to_vec(),
        censoring_probabilities: q.censoring_probabilities.to_vec(),
        outcome_observed: q.outcome_observed.to_vec(),
        subject_ids: q.subject_ids.iter().map(ToString::to_string).collect(),
        fold_ids: q.fold_ids.to_vec(),
        excluded_fold_predictions: q.excluded_fold_predictions,
        probabilities_known_by_design: q.probabilities_known_by_design,
        minimum_probability: q.minimum_probability,
        rule_id: q.rule_id.as_ref().map(ToString::to_string),
        rule_version: q.rule_version.as_ref().map(ToString::to_string),
        rule_provenance: q.rule_provenance.as_ref().map(ToString::to_string),
    })
}

/// Encode the [`CausalQuery::SequentialContrast`] payload.
fn sequential_contrast_to_wire(q: &antecedent_core::SequentialContrastQuery) -> CausalQueryWire {
    CausalQueryWire::SequentialContrast(SequentialContrastQueryWire {
        outcome: q.outcome.raw(),
        first_treatment: q.first_treatment.raw(),
        second_treatment: q.second_treatment.raw(),
        baseline: q.baseline.iter().map(|id| id.raw()).collect(),
        intermediate: q.intermediate.iter().map(|id| id.raw()).collect(),
        active: q.active,
        control: q.control,
        fold_of_row: q.fold_of_row.to_vec(),
        fold_count: q.fold_count,
        min_probability: q.min_probability,
    })
}

/// Encode the [`CausalQuery::HistoryMatch`] payload.
fn history_match_to_wire(q: &antecedent_core::HistoryMatchQuery) -> CausalQueryWire {
    CausalQueryWire::HistoryMatch(HistoryMatchQueryWire {
        outcome: q.outcome.raw(),
        treated: q.treated.raw(),
        history: q.history.iter().map(|id| id.raw()).collect(),
        period: q.period.map(antecedent_core::VariableId::raw),
        weight: q.weight.map(antecedent_core::VariableId::raw),
        caliper: q.caliper,
    })
}

/// Encode a noncausal association query.
#[must_use]
pub fn association_query_to_wire(q: &antecedent_core::AssociationQuery) -> AssociationQueryWire {
    use antecedent_core::{AssociationFunctional, AssociationOutcome};
    AssociationQueryWire {
        outcome: q.outcome.raw(),
        predictors: q.predictors.iter().map(|id| id.raw()).collect(),
        conditioning: q.conditioning.iter().map(|id| id.raw()).collect(),
        functional: match q.functional {
            AssociationFunctional::Auto => AssociationFunctionalWire::Auto,
            AssociationFunctional::Conditional => AssociationFunctionalWire::Conditional,
            AssociationFunctional::Coefficients => AssociationFunctionalWire::Coefficients,
            AssociationFunctional::Measure(measure) => {
                AssociationFunctionalWire::Measure { measure: measure.as_str().to_owned() }
            }
            AssociationFunctional::Predictive { folds } => {
                AssociationFunctionalWire::Predictive { folds }
            }
        },
        outcome_scale: (q.outcome_scale != AssociationOutcome::Schema)
            .then(|| q.outcome_scale.as_str().to_owned()),
        weight: q.weight.map(antecedent_core::VariableId::raw),
        refused_after: q.refused_after.as_deref().map(str::to_owned),
    }
}

/// Encode a structural transport query.
///
/// # Errors
///
/// Unsupported response-query fields.
pub fn transport_query_to_wire(q: &TransportQuery) -> Result<TransportQueryWire, IoError> {
    Ok(TransportQueryWire {
        catalog: q
            .catalog
            .as_ref()
            .map(crate::transport_catalog_wire::EvidenceCatalogWire::from_catalog),
        response: response_query_to_wire(&q.response)?,
        source_population: q.source_population.to_string(),
        target_population: q.target_population.to_string(),
        source_experiments: vars_to_raw(&q.source_experiments),
    })
}

/// Encode a randomized interference query.
///
/// # Errors
///
/// Counts that do not fit the portable wire representation.
pub fn interference_query_to_wire(q: &InterferenceQuery) -> Result<InterferenceQueryWire, IoError> {
    let assignment = match &q.assignment {
        AssignmentDesign::Bernoulli { probabilities } => {
            AssignmentDesignWire::Bernoulli { probabilities: probabilities.to_vec() }
        }
        AssignmentDesign::CompleteRandomization { treated } => {
            AssignmentDesignWire::CompleteRandomization {
                treated: u64::try_from(*treated).map_err(|_| IoError::TooLarge)?,
            }
        }
        AssignmentDesign::ClusterRandomization { clusters, treated_clusters } => {
            AssignmentDesignWire::ClusterRandomization {
                clusters: clusters.to_vec(),
                treated_clusters: u64::try_from(*treated_clusters)
                    .map_err(|_| IoError::TooLarge)?,
            }
        }
        AssignmentDesign::TwoStageSaturation {
            clusters,
            low_probability,
            high_probability,
            high_clusters,
            realized_saturation,
        } => AssignmentDesignWire::TwoStageSaturation {
            clusters: clusters.to_vec(),
            low_probability: *low_probability,
            high_probability: *high_probability,
            high_clusters: u64::try_from(*high_clusters).map_err(|_| IoError::TooLarge)?,
            realized_saturation: realized_saturation.to_vec(),
        },
        AssignmentDesign::ObservedExposure {
            clusters,
            propensity_from,
            propensity_to,
            provenance,
            assume_network_exchangeability,
        } => AssignmentDesignWire::ObservedExposure {
            clusters: clusters.to_vec(),
            propensity_from: propensity_from.to_vec(),
            propensity_to: propensity_to.to_vec(),
            provenance: match provenance {
                antecedent_core::ExposurePropensityProvenance::Known => "known",
                antecedent_core::ExposurePropensityProvenance::ExternallyEstimated => {
                    "externally_estimated"
                }
            }
            .into(),
            assume_network_exchangeability: *assume_network_exchangeability,
        },
    };
    let exposure = match &q.exposure {
        ExposureMapping::OwnTreatment => ExposureMappingWire::OwnTreatment,
        ExposureMapping::NeighborCount => ExposureMappingWire::NeighborCount,
        ExposureMapping::NeighborFraction => ExposureMappingWire::NeighborFraction,
        ExposureMapping::WeightedNeighborExposure => ExposureMappingWire::WeightedNeighborExposure,
        ExposureMapping::Custom(id) => ExposureMappingWire::Custom(id.to_string()),
    };
    let InterferenceFunctional::ExposureContrast { outcome, from, to } = &q.functional;
    Ok(InterferenceQueryWire {
        assignment,
        exposure,
        functional: InterferenceFunctionalWire::ExposureContrast {
            outcome: outcome.raw(),
            from: ExposureLevelWire { own: from.own, neighbors: from.neighbors },
            to: ExposureLevelWire { own: to.own, neighbors: to.neighbors },
        },
        probability_draws: q.probability_draws,
    })
}

/// Encode an interventional distribution query.
///
/// # Errors
///
/// Unsupported target population.
pub fn interventional_distribution_to_wire(
    q: &InterventionalDistributionQuery,
) -> Result<InterventionalDistributionQueryWire, IoError> {
    Ok(InterventionalDistributionQueryWire {
        outcomes: vars_to_raw(&q.outcomes),
        interventions: q
            .interventions
            .iter()
            .map(InterventionWire::from_domain)
            .collect::<Result<Vec<_>, _>>()?,
        conditioning: vars_to_raw(&q.conditioning),
        target_population: TargetPopulationWire::from_domain(&q.target_population)?,
    })
}

/// Encode a path-specific effect query.
///
/// # Errors
///
/// Unsupported target population.
pub fn path_specific_to_wire(
    q: &PathSpecificEffectQuery,
) -> Result<PathSpecificEffectQueryWire, IoError> {
    Ok(PathSpecificEffectQueryWire {
        treatment: q.treatment.raw(),
        outcome: q.outcome.raw(),
        path_nodes: vars_to_raw(&q.path_nodes),
        control: InterventionWire::from_domain(&q.control)?,
        active: InterventionWire::from_domain(&q.active)?,
        target_population: TargetPopulationWire::from_domain(&q.target_population)?,
        max_paths: crate::error::fit_u64(q.max_paths)?,
        max_len: crate::error::fit_u64(q.max_len)?,
    })
}
