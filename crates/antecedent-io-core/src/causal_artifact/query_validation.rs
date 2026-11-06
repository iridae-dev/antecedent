//! Query-coordinate validation for causal artifact replay.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{CausalQueryWire, IoError, causal_query_from_wire};

pub(super) fn validate_id(raw: u32, variable_count: usize) -> Result<(), IoError> {
    if usize::try_from(raw).map_or(true, |id| id >= variable_count) {
        return Err(IoError::Convert(format!(
            "variable id {raw} is outside causal payload header bounds (len={variable_count})"
        )));
    }
    Ok(())
}

pub(super) fn validate_ids(
    ids: impl IntoIterator<Item = u32>,
    variable_count: usize,
) -> Result<(), IoError> {
    for id in ids {
        validate_id(id, variable_count)?;
    }
    Ok(())
}

fn validate_intervention_ids(
    intervention: &crate::InterventionWire,
    variable_count: usize,
) -> Result<(), IoError> {
    match intervention {
        crate::InterventionWire::Set { variable, .. }
        | crate::InterventionWire::Shift { variable, .. }
        | crate::InterventionWire::Stochastic { variable, .. }
        | crate::InterventionWire::Soft { variable, .. } => validate_id(*variable, variable_count),
        crate::InterventionWire::Transform { variable, reads, .. }
        | crate::InterventionWire::Assign { variable, reads, .. } => {
            validate_id(*variable, variable_count)?;
            validate_ids(
                reads.current.iter().copied().chain(reads.history.iter().map(|(v, _)| *v)),
                variable_count,
            )
        }
        crate::InterventionWire::Sequence { steps } => {
            for step in steps {
                validate_intervention_ids(&step.intervention, variable_count)?;
            }
            Ok(())
        }
    }
}

pub(super) fn validate_response_functional_ids(
    functional: &crate::ResponseFunctionalWire,
    variable_count: usize,
) -> Result<(), IoError> {
    use crate::ResponseFunctionalWire;
    match functional {
        ResponseFunctionalWire::MeanCurve { outcome, treatment } => {
            validate_ids([*outcome, treatment.variable], variable_count)
        }
        ResponseFunctionalWire::AverageDerivative { outcome, treatment, .. }
        | ResponseFunctionalWire::PointDerivative { outcome, treatment, .. } => {
            validate_ids([*outcome, *treatment], variable_count)
        }
        ResponseFunctionalWire::DirectionalDerivative { outcomes, treatments, .. }
        | ResponseFunctionalWire::Jacobian { outcomes, treatments, .. } => {
            validate_ids(outcomes.iter().chain(treatments).copied(), variable_count)
        }
        ResponseFunctionalWire::InterventionResponse { outcome, interventions } => {
            validate_id(*outcome, variable_count)?;
            for intervention in interventions {
                validate_intervention_ids(intervention, variable_count)?;
            }
            Ok(())
        }
    }
}

fn validate_response_query_ids(
    query: &crate::ResponseQueryWire,
    variable_count: usize,
) -> Result<(), IoError> {
    use crate::ObservationSpecWire;

    validate_response_functional_ids(&query.functional, variable_count)?;
    match &query.observation {
        ObservationSpecWire::Complete => {}
        ObservationSpecWire::RightCensored { latent, observed, censoring, event }
        | ObservationSpecWire::LeftCensored { latent, observed, censoring, event } => {
            validate_ids([*latent, *observed, *censoring, *event], variable_count)?;
        }
        ObservationSpecWire::IntervalCensored { latent, lower, upper } => {
            validate_ids([*latent, *lower, *upper], variable_count)?;
        }
        ObservationSpecWire::Truncated { latent, observed, lower, upper } => {
            validate_ids([*latent, *observed], variable_count)?;
            validate_ids(lower.iter().chain(upper).copied(), variable_count)?;
        }
        ObservationSpecWire::Selected { latent, observed, indicator } => {
            validate_ids([*latent, *observed, *indicator], variable_count)?;
        }
    }
    for assumption in &query.observation_assumptions {
        match assumption {
            crate::ObservationAssumptionWire::IndependentGiven(ids)
            | crate::ObservationAssumptionWire::OutcomeIndependentGiven(ids) => {
                validate_ids(ids.iter().copied(), variable_count)?;
            }
            crate::ObservationAssumptionWire::Structural(_) => {}
        }
    }
    Ok(())
}

// allow(too_many_lines): exhaustive per-variant id validation; one match arm per query kind
#[allow(
    clippy::too_many_lines,
    reason = "one arm per query kind, each checking that kind variable ids"
)]
pub(crate) fn validate_query_ids(
    query: &CausalQueryWire,
    variable_count: usize,
) -> Result<(), IoError> {
    use CausalQueryWire as Q;
    match query {
        Q::AverageEffect { treatment, outcome, effect_modifiers, control, active, .. } => {
            validate_ids(
                [*treatment, *outcome].into_iter().chain(effect_modifiers.iter().copied()),
                variable_count,
            )?;
            validate_intervention_ids(control, variable_count)?;
            validate_intervention_ids(active, variable_count)
        }
        Q::TemporalEffect { treatment, outcome, control, active, .. } => {
            validate_ids([*treatment, *outcome], variable_count)?;
            validate_intervention_ids(control, variable_count)?;
            validate_intervention_ids(active, variable_count)
        }
        Q::Counterfactual { outcomes, interventions, control, .. } => {
            validate_ids(outcomes.iter().copied(), variable_count)?;
            for intervention in interventions {
                validate_intervention_ids(intervention, variable_count)?;
            }
            if let Some(control) = control {
                validate_intervention_ids(control, variable_count)?;
            }
            Ok(())
        }
        Q::NestedCounterfactual { treatment, mediator, outcome, .. } => {
            validate_ids([*treatment, *mediator, *outcome], variable_count)
        }
        Q::AnomalyAttribution { targets, .. } | Q::MechanismChange { targets, .. } => {
            validate_ids(targets.iter().copied(), variable_count)
        }
        Q::ChangeAttribution { outcome, .. } | Q::UnitChange { outcome, .. } => {
            validate_id(*outcome, variable_count)
        }
        Q::Discover(wire) => {
            validate_ids(wire.variable_ids(), variable_count)?;
            wire.to_domain().map(|_| ())
        }
        Q::Attribute(wire) => {
            validate_ids(wire.variable_ids(), variable_count)?;
            if let Some(embedded) = wire.embedded() {
                validate_query_ids(embedded, variable_count)?;
            }
            wire.to_domain().map(|_| ())
        }
        Q::Simulate(wire) => {
            validate_ids(wire.variable_ids(), variable_count)?;
            for intervention in wire.interventions() {
                validate_intervention_ids(intervention, variable_count)?;
            }
            wire.to_domain().map(|_| ())
        }
        Q::TemporalDesign(wire) => {
            validate_ids(wire.variable_ids(), variable_count)?;
            wire.to_domain().map(|_| ())
        }
        Q::Mediation { treatment, outcome, mediators, control, active, .. } => {
            validate_ids(
                [*treatment, *outcome].into_iter().chain(mediators.iter().copied()),
                variable_count,
            )?;
            validate_intervention_ids(control, variable_count)?;
            validate_intervention_ids(active, variable_count)
        }
        Q::ConditionalEffect { inner } => validate_query_ids(inner, variable_count),
        Q::Distribution(wire) => {
            validate_ids(wire.outcomes.iter().chain(&wire.conditioning).copied(), variable_count)?;
            for intervention in &wire.interventions {
                validate_intervention_ids(intervention, variable_count)?;
            }
            Ok(())
        }
        Q::PathSpecific(wire) => {
            validate_ids(
                [wire.treatment, wire.outcome].into_iter().chain(wire.path_nodes.iter().copied()),
                variable_count,
            )?;
            validate_intervention_ids(&wire.control, variable_count)?;
            validate_intervention_ids(&wire.active, variable_count)
        }
        Q::Response(wire) => validate_response_query_ids(wire, variable_count),
        Q::Transport(wire) => {
            validate_response_query_ids(&wire.response, variable_count)?;
            validate_ids(wire.source_experiments.iter().copied(), variable_count)?;
            if let Some(catalog) = &wire.catalog {
                for environment in &catalog.environments {
                    validate_ids(
                        environment
                            .variables
                            .iter()
                            .map(|v| v.0)
                            .chain(environment.selection_targets.iter().copied()),
                        variable_count,
                    )?;
                }
                for regime in &catalog.regimes {
                    validate_ids(
                        regime
                            .interventions
                            .iter()
                            .chain(&regime.measured)
                            .copied()
                            .chain(regime.intervention_values.iter().map(|v| v.0))
                            .chain(regime.separate_marginals.iter().flatten().copied()),
                        variable_count,
                    )?;
                }
            }
            Ok(())
        }
        Q::Interference(wire) => {
            let crate::InterferenceFunctionalWire::ExposureContrast { outcome, .. } =
                &wire.functional;
            validate_id(*outcome, variable_count)
        }
        Q::RandomizedEffect(wire) => {
            if !matches!(wire.design, crate::RandomizationDesignWire::MultiArm)
                && (!wire.multi_arm_labels.is_empty()
                    || !wire.multi_arm_assignment.is_empty()
                    || !wire.multi_arm_probabilities.is_empty())
            {
                return Err(IoError::Convert(
                    "multi-arm metadata requires a multi-arm design".into(),
                ));
            }
            if !matches!(wire.design, crate::RandomizationDesignWire::Factorial2x2)
                && (!wire.second_factor_assignment.is_empty()
                    || wire.factorial_cell_counts.is_some()
                    || wire.second_factor_arms.is_some())
            {
                return Err(IoError::Convert(
                    "factorial metadata requires a factorial design".into(),
                ));
            }
            validate_id(wire.outcome, variable_count)?;
            if let Some((covariate, _)) = wire.fixed_cuped {
                validate_id(covariate, variable_count)?;
            }
            for &covariate in &wire.ancova_covariates {
                validate_id(covariate, variable_count)?;
            }
            let design = match &wire.design {
                crate::RandomizationDesignWire::Bernoulli => {
                    antecedent_core::RandomizationDesign::Bernoulli
                }
                crate::RandomizationDesignWire::Switchback => {
                    antecedent_core::RandomizationDesign::Switchback {
                        periods: wire
                            .periods
                            .iter()
                            .map(|period| std::sync::Arc::<str>::from(period.as_str()))
                            .collect::<Vec<_>>()
                            .into(),
                    }
                }
                crate::RandomizationDesignWire::Complete { treated_units } => {
                    antecedent_core::RandomizationDesign::Complete { treated_units: *treated_units }
                }
                crate::RandomizationDesignWire::Cluster { treated_clusters } => {
                    antecedent_core::RandomizationDesign::Cluster {
                        treated_clusters: *treated_clusters,
                    }
                }
                crate::RandomizationDesignWire::Stratified => {
                    antecedent_core::RandomizationDesign::Stratified {
                        blocks: wire
                            .blocks
                            .iter()
                            .map(|x| std::sync::Arc::<str>::from(x.as_str()))
                            .collect::<Vec<_>>()
                            .into(),
                        treated_per_row: wire.treated_per_row.clone().into(),
                    }
                }
                crate::RandomizationDesignWire::Factorial2x2 => {
                    let arms = wire.second_factor_arms.as_ref().ok_or_else(|| {
                        IoError::Convert("second-factor labels are missing".into())
                    })?;
                    antecedent_core::RandomizationDesign::Factorial2x2 {
                        second_factor_assignment: wire.second_factor_assignment.clone().into(),
                        cell_counts: wire.factorial_cell_counts.ok_or_else(|| {
                            IoError::Convert("factorial cell counts are missing".into())
                        })?,
                        second_factor_arms: (
                            std::sync::Arc::<str>::from(arms.0.as_str()),
                            std::sync::Arc::<str>::from(arms.1.as_str()),
                        ),
                    }
                }
                crate::RandomizationDesignWire::MultiArm => {
                    antecedent_core::RandomizationDesign::MultiArm {
                        assignment: wire.multi_arm_assignment.clone().into(),
                        probabilities: wire.multi_arm_probabilities.clone().into(),
                        arms: wire
                            .multi_arm_labels
                            .iter()
                            .map(|arm| std::sync::Arc::<str>::from(arm.as_str()))
                            .collect::<Vec<_>>()
                            .into(),
                    }
                }
            };
            let mut query = antecedent_core::RandomizedEffectQuery::with_design(
                design,
                antecedent_core::VariableId::from_raw(wire.outcome),
                wire.realized_assignment.clone(),
                wire.assignment_probabilities.clone(),
                wire.assignment_units
                    .iter()
                    .map(|x| std::sync::Arc::<str>::from(x.as_str()))
                    .collect::<Vec<_>>(),
                wire.outcome_units
                    .iter()
                    .map(|x| std::sync::Arc::<str>::from(x.as_str()))
                    .collect::<Vec<_>>(),
                (
                    std::sync::Arc::<str>::from(wire.treatment_arms.0.as_str()),
                    std::sync::Arc::<str>::from(wire.treatment_arms.1.as_str()),
                ),
            );
            if let Some((covariate, coefficient)) = wire.fixed_cuped {
                query = query.with_fixed_cuped(
                    antecedent_core::VariableId::from_raw(covariate),
                    coefficient,
                );
            }
            if !wire.ancova_covariates.is_empty() {
                query = query.with_ancova(
                    wire.ancova_covariates
                        .iter()
                        .copied()
                        .map(antecedent_core::VariableId::from_raw)
                        .collect::<Vec<_>>(),
                );
            }
            if let Some(receipt) = &wire.received_treatment {
                query = if wire.estimand == crate::RandomizedEstimandWire::TreatmentOnTreated {
                    query.with_treatment_on_treated(receipt.clone())
                } else {
                    query.with_received_treatment(receipt.clone())
                };
            }
            if wire.exact_randomization_test {
                query = query.with_exact_randomization_test();
            }
            if (wire.estimand != crate::RandomizedEstimandWire::Itt)
                != wire.received_treatment.is_some()
            {
                return Err(IoError::Convert(
                    "receipt-adjusted estimand and treatment receipt must agree".into(),
                ));
            }
            query.validate().map_err(|e| IoError::Convert(e.to_string()))
        }
        Q::PolicyValue(wire) => {
            validate_id(wire.outcome, variable_count)?;
            antecedent_core::PolicyValueQuery {
                outcome: antecedent_core::VariableId::from_raw(wire.outcome),
                assignment: wire.assignment.clone().into(),
                propensity: wire.propensity.clone().into(),
                actions: wire.actions.clone().into(),
                reference: wire.reference.clone().into(),
                mu0: wire.mu0.clone().into(),
                mu1: wire.mu1.clone().into(),
                costs: wire.costs.clone().into(),
                reference_costs: wire.reference_costs.clone().into(),
                evaluation_subject_ids: wire
                    .evaluation_subject_ids
                    .iter()
                    .map(|x| std::sync::Arc::<str>::from(x.as_str()))
                    .collect::<Vec<_>>()
                    .into(),
                disjoint_training_subjects: wire.disjoint_training_subjects,
                crossfit_fold_ownership_valid: wire.crossfit_fold_ownership_valid,
                global_constraints_present: wire.global_constraints_present,
                multi_action: wire.multi_action.as_ref().map(Into::into),
                uplift_bins: wire.uplift_bins.clone().into(),
                uplift_bin_count: wire.uplift_bin_count,
                uplift_training_subject_ids: wire
                    .uplift_training_subject_ids
                    .iter()
                    .map(|x| std::sync::Arc::<str>::from(x.as_str()))
                    .collect::<Vec<_>>()
                    .into(),
                regret: wire.regret.as_ref().map(|regret| {
                    antecedent_core::FixedCandidateRegretInputs {
                        candidates: regret
                            .candidates
                            .iter()
                            .map(|actions| std::sync::Arc::<[bool]>::from(actions.clone()))
                            .collect::<Vec<_>>()
                            .into(),
                        selected_index: regret.selected_index,
                        training_subject_ids: regret
                            .training_subject_ids
                            .iter()
                            .map(|id| std::sync::Arc::<str>::from(id.as_str()))
                            .collect::<Vec<_>>()
                            .into(),
                    }
                }),
            }
            .validate()
            .map_err(|e| IoError::Convert(e.to_string()))
        }
        Q::ContinuousDoseResponse(wire) => {
            for id in [wire.outcome, wire.dose, wire.dose_density] {
                validate_id(id, variable_count)?;
            }
            antecedent_core::ContinuousDoseResponseQuery {
                outcome: antecedent_core::VariableId::from_raw(wire.outcome),
                dose: antecedent_core::VariableId::from_raw(wire.dose),
                dose_density: antecedent_core::VariableId::from_raw(wire.dose_density),
                baseline_groups: wire
                    .baseline_groups
                    .iter()
                    .map(|value| std::sync::Arc::<str>::from(value.as_str()))
                    .collect::<Vec<_>>()
                    .into(),
                target_doses: wire.target_doses.clone().into(),
                bandwidth: wire.bandwidth,
                min_local_support: wire.min_local_support,
                density_provenance: std::sync::Arc::from(wire.density_provenance.as_str()),
                fixed_policy: wire.fixed_policy.as_ref().map(|(policy, reference)| {
                    antecedent_core::FixedGroupDosePolicy {
                        policy_doses: policy
                            .iter()
                            .map(|(group, dose)| (std::sync::Arc::from(group.as_str()), *dose))
                            .collect::<Vec<_>>()
                            .into(),
                        reference_doses: reference
                            .iter()
                            .map(|(group, dose)| (std::sync::Arc::from(group.as_str()), *dose))
                            .collect::<Vec<_>>()
                            .into(),
                    }
                }),
            }
            .validate()
            .map_err(|error| IoError::Convert(error.to_string()))
        }
        Q::PanelDid(wire) => {
            validate_id(wire.outcome, variable_count)?;
            if let Some((pre, propensity, prediction, _)) = wire.augmented {
                validate_ids([pre, propensity, prediction], variable_count)?;
            }
            if let Some(rollout) = &wire.rollout {
                for covariate in &rollout.covariates {
                    validate_id(*covariate, variable_count)?;
                }
            }
            let query = causal_query_from_wire(&Q::PanelDid(wire.clone()))?;
            query.validate().map_err(|e| IoError::Convert(e.to_string()))
        }
        Q::SyntheticControl(wire) => {
            validate_ids([wire.outcome], variable_count)?;
            let query = antecedent_core::SyntheticControlQuery::new(
                antecedent_core::VariableId::from_raw(wire.outcome),
                wire.units
                    .iter()
                    .map(|unit| std::sync::Arc::<str>::from(unit.as_str()))
                    .collect::<Vec<_>>(),
                wire.periods.clone(),
                std::sync::Arc::<str>::from(wire.treated_unit.as_str()),
                wire.intervention_period,
            );
            let query = if wire.difference_in_differences {
                query.difference_in_differences()
            } else {
                query
            };
            let query = if wire.uniform_unit_randomization {
                query.with_uniform_unit_randomization()
            } else {
                query
            };
            let query = if let Some(effect) = wire.sharp_null_effect {
                query.with_sharp_null_effect(effect)
            } else {
                query
            };
            let mut query = if let Some(ridge) = wire.augmentation_ridge {
                query.with_augmentation(ridge)
            } else {
                query
            };
            query.rollout = wire
                .rollout
                .as_ref()
                .map(crate::rollout_query_wire::synthetic_rollout_from_wire)
                .transpose()?;
            query.observed_window_support = wire.observed_window_support;
            query.validate().map_err(|error| IoError::Convert(error.to_string()))
        }
        Q::LocalPolynomialRatio(wire) => {
            validate_ids([wire.outcome, wire.treatment, wire.running], variable_count)?;
            if let Some(options) = &wire.options {
                for covariate in &options.balance_covariates {
                    validate_id(*covariate, variable_count)?;
                }
            }
            let query = causal_query_from_wire(&Q::LocalPolynomialRatio(wire.clone()))?;
            query.validate().map_err(|error| IoError::Convert(error.to_string()))
        }
        Q::Survival(wire) => {
            if wire.censoring_probability_floor.is_none()
                && (!wire.censoring_times.is_empty() || !wire.censoring_columns.is_empty())
            {
                return Err(IoError::Convert(
                    "censoring grid requires its positivity floor".into(),
                ));
            }
            validate_id(wire.duration, variable_count)?;
            validate_id(wire.event, variable_count)?;
            validate_id(wire.treatment, variable_count)?;
            for id in wire.independent_given.iter().chain(wire.censoring_columns.iter()) {
                validate_id(*id, variable_count)?;
            }
            if let Some(entry) = wire.delayed_entry {
                validate_id(entry, variable_count)?;
            }
            if !wire.independent_observation {
                return Err(IoError::Convert(
                    "survival requires marginal independent observation declaration".into(),
                ));
            }
            antecedent_core::SurvivalQuery {
                duration: antecedent_core::VariableId::from_raw(wire.duration),
                event: antecedent_core::VariableId::from_raw(wire.event),
                treatment: antecedent_core::VariableId::from_raw(wire.treatment),
                tau: wire.tau,
                delayed_entry: wire.delayed_entry.map(antecedent_core::VariableId::from_raw),
                known_censoring: wire.censoring_probability_floor.map(|minimum_probability| {
                    antecedent_core::KnownCensoringSurvival {
                        times: std::sync::Arc::from(wire.censoring_times.clone()),
                        columns: std::sync::Arc::from(
                            wire.censoring_columns
                                .iter()
                                .copied()
                                .map(antecedent_core::VariableId::from_raw)
                                .collect::<Vec<_>>(),
                        ),
                        minimum_probability,
                    }
                }),
                observation_assumption: antecedent_core::ObservationAssumption::IndependentGiven(
                    std::sync::Arc::from(
                        wire.independent_given
                            .iter()
                            .copied()
                            .map(antecedent_core::VariableId::from_raw)
                            .collect::<Vec<_>>(),
                    ),
                ),
                functional: wire.target_cause.map_or(
                    antecedent_core::SurvivalFunctional::SurvivalAndRmst,
                    |target_cause| antecedent_core::SurvivalFunctional::CumulativeIncidence {
                        target_cause,
                    },
                ),
            }
            .validate()
            .map_err(|e| IoError::Convert(e.to_string()))
        }
        Q::LongitudinalRegime(wire) => {
            validate_id(wire.outcome, variable_count)?;
            let method = match wire.method.as_str() {
                "ipw" => antecedent_core::LongitudinalRegimeMethod::Ipw,
                "g_formula" => antecedent_core::LongitudinalRegimeMethod::GFormula,
                "sequential_dr" => {
                    antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust
                }
                "marginal_structural_model" => {
                    antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel
                }
                _ => return Err(IoError::Convert("unknown longitudinal regime method".into())),
            };
            antecedent_core::LongitudinalRegimeQuery {
                outcome: antecedent_core::VariableId::from_raw(wire.outcome),
                method,
                period_outcome_predictions: if wire.period_outcome_predictions.is_empty() {
                    None
                } else {
                    Some(wire.period_outcome_predictions.clone().into())
                },
                known_fixed_outcome_predictions: wire.known_fixed_outcome_predictions,
                stabilizing_numerator_probabilities: if wire
                    .stabilizing_numerator_probabilities
                    .is_empty()
                {
                    None
                } else {
                    Some(wire.stabilizing_numerator_probabilities.clone().into())
                },
                q_predictions: if wire.q_predictions.is_empty() {
                    None
                } else {
                    Some(wire.q_predictions.clone().into())
                },
                observation_history: if wire.observation_history.is_empty() {
                    None
                } else {
                    Some(wire.observation_history.clone().into())
                },
                prediction_fold_ids: if wire.prediction_fold_ids.is_empty() {
                    None
                } else {
                    Some(wire.prediction_fold_ids.clone().into())
                },
                periods: wire.periods,
                treatment_history: wire.treatment_history.clone().into(),
                regime_actions: wire.regime_actions.clone().into(),
                treatment_probabilities: wire.treatment_probabilities.clone().into(),
                censoring_probabilities: wire.censoring_probabilities.clone().into(),
                outcome_observed: wire.outcome_observed.clone().into(),
                subject_ids: wire
                    .subject_ids
                    .iter()
                    .map(|s| std::sync::Arc::<str>::from(s.as_str()))
                    .collect::<Vec<_>>()
                    .into(),
                fold_ids: wire.fold_ids.clone().into(),
                excluded_fold_predictions: wire.excluded_fold_predictions,
                probabilities_known_by_design: wire.probabilities_known_by_design,
                minimum_probability: wire.minimum_probability,
                rule_id: wire.rule_id.as_deref().map(std::sync::Arc::<str>::from),
                rule_version: wire.rule_version.as_deref().map(std::sync::Arc::<str>::from),
                rule_provenance: wire.rule_provenance.as_deref().map(std::sync::Arc::<str>::from),
            }
            .validate()
            .map_err(|e| IoError::Convert(e.to_string()))
        }
        Q::HistoryMatch(wire) => {
            validate_ids(
                [wire.outcome, wire.treated]
                    .into_iter()
                    .chain(wire.history.iter().copied())
                    .chain(wire.period)
                    .chain(wire.weight),
                variable_count,
            )?;
            antecedent_core::HistoryMatchQuery {
                outcome: antecedent_core::VariableId::from_raw(wire.outcome),
                treated: antecedent_core::VariableId::from_raw(wire.treated),
                history: wire
                    .history
                    .iter()
                    .map(|id| antecedent_core::VariableId::from_raw(*id))
                    .collect(),
                period: wire.period.map(antecedent_core::VariableId::from_raw),
                weight: wire.weight.map(antecedent_core::VariableId::from_raw),
                caliper: wire.caliper,
            }
            .validate()
            .map_err(|e| IoError::Convert(e.to_string()))
        }
        Q::IvBounds(wire) => {
            validate_ids(
                [wire.outcome, wire.treatment, wire.instrument].into_iter().chain(wire.weight),
                variable_count,
            )?;
            antecedent_core::IvBoundsQuery {
                outcome: antecedent_core::VariableId::from_raw(wire.outcome),
                treatment: antecedent_core::VariableId::from_raw(wire.treatment),
                instrument: antecedent_core::VariableId::from_raw(wire.instrument),
                weight: wire.weight.map(antecedent_core::VariableId::from_raw),
            }
            .validate()
            .map_err(|e| IoError::Convert(e.to_string()))
        }
        Q::Association(wire) => {
            validate_ids(
                std::iter::once(wire.outcome)
                    .chain(wire.predictors.iter().copied())
                    .chain(wire.conditioning.iter().copied())
                    .chain(wire.weight),
                variable_count,
            )?;
            crate::association_query_from_wire(wire).map(|_| ())
        }
        Q::SequentialContrast(wire) => {
            validate_ids(
                [wire.outcome, wire.first_treatment, wire.second_treatment]
                    .into_iter()
                    .chain(wire.baseline.iter().copied())
                    .chain(wire.intermediate.iter().copied()),
                variable_count,
            )?;
            antecedent_core::SequentialContrastQuery {
                outcome: antecedent_core::VariableId::from_raw(wire.outcome),
                first_treatment: antecedent_core::VariableId::from_raw(wire.first_treatment),
                second_treatment: antecedent_core::VariableId::from_raw(wire.second_treatment),
                baseline: wire
                    .baseline
                    .iter()
                    .map(|id| antecedent_core::VariableId::from_raw(*id))
                    .collect(),
                intermediate: wire
                    .intermediate
                    .iter()
                    .map(|id| antecedent_core::VariableId::from_raw(*id))
                    .collect(),
                active: wire.active,
                control: wire.control,
                fold_of_row: wire.fold_of_row.clone().into(),
                fold_count: wire.fold_count,
                min_probability: wire.min_probability,
            }
            .validate()
            .map_err(|e| IoError::Convert(e.to_string()))
        }
    }
}
