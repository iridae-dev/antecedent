//! Decoding of wire forms into typed queries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Decode [`CausalQueryWire`].
///
/// # Errors
///
/// Unknown tags, size overflows, or an invalid decoded query.
pub fn causal_query_from_wire(w: &CausalQueryWire) -> Result<CausalQuery, IoError> {
    let query = match w {
        CausalQueryWire::AverageEffect { .. }
        | CausalQueryWire::TemporalEffect { .. }
        | CausalQueryWire::Mediation { .. } => effect_query_from_wire(w)?,
        CausalQueryWire::Counterfactual { .. }
        | CausalQueryWire::NestedCounterfactual { .. }
        | CausalQueryWire::ConditionalEffect { .. } => counterfactual_query_from_wire(w)?,
        CausalQueryWire::AnomalyAttribution { .. }
        | CausalQueryWire::ChangeAttribution { .. }
        | CausalQueryWire::MechanismChange { .. }
        | CausalQueryWire::UnitChange { .. } => attribution_query_from_wire(w)?,
        CausalQueryWire::Distribution(w) => {
            CausalQuery::Distribution(interventional_distribution_from_wire(w)?)
        }
        CausalQueryWire::PathSpecific(w) => CausalQuery::PathSpecific(path_specific_from_wire(w)?),
        CausalQueryWire::Response(w) => CausalQuery::Response(response_query_from_wire(w)?),
        CausalQueryWire::Transport(w) => CausalQuery::Transport(transport_query_from_wire(w)?),
        CausalQueryWire::Interference(w) => {
            CausalQuery::Interference(interference_query_from_wire(w)?)
        }
        CausalQueryWire::RandomizedEffect(w) => randomized_effect_from_wire(w)?,
        CausalQueryWire::PolicyValue(w) => policy_value_from_wire(w)?,
        CausalQueryWire::ContinuousDoseResponse(w) => continuous_dose_response_from_wire(w)?,
        CausalQueryWire::PanelDid(w) if w.rollout.is_some() => panel_did_rollout_from_wire(w)?,
        CausalQueryWire::PanelDid(w) => panel_did_from_wire(w)?,
        CausalQueryWire::SyntheticControl(w) => synthetic_control_from_wire(w)?,
        CausalQueryWire::LocalPolynomialRatio(w) => local_polynomial_ratio_from_wire(w)?,
        CausalQueryWire::Survival(w) => survival_from_wire(w)?,
        CausalQueryWire::LongitudinalRegime(w) => longitudinal_regime_from_wire(w)?,
        CausalQueryWire::SequentialContrast(w) => sequential_contrast_from_wire(w)?,
        CausalQueryWire::HistoryMatch(w) => history_match_from_wire(w)?,
        CausalQueryWire::Discover(w) => CausalQuery::Discover(w.to_domain()?),
        CausalQueryWire::Attribute(w) => CausalQuery::Attribute(w.to_domain()?),
        CausalQueryWire::Simulate(w) => CausalQuery::Simulate(w.to_domain()?),
        CausalQueryWire::TemporalDesign(w) => CausalQuery::TemporalDesign(w.to_domain()?),
        CausalQueryWire::IvBounds(w) => iv_bounds_from_wire(w)?,
        CausalQueryWire::Association(w) => {
            CausalQuery::Association(association_query_from_wire(w)?)
        }
    };
    query.validate().map_err(|error| IoError::Convert(error.to_string()))?;
    Ok(query)
}

/// Decode the randomization design of a [`RandomizedEffectQueryWire`], refusing metadata of
/// another design.
fn randomization_design_from_wire(
    w: &RandomizedEffectQueryWire,
) -> Result<RandomizationDesign, IoError> {
    if !matches!(w.design, RandomizationDesignWire::MultiArm)
        && (!w.multi_arm_labels.is_empty()
            || !w.multi_arm_assignment.is_empty()
            || !w.multi_arm_probabilities.is_empty())
    {
        return Err(IoError::Convert("multi-arm metadata requires a multi-arm design".into()));
    }
    if !matches!(w.design, RandomizationDesignWire::Factorial2x2)
        && (!w.second_factor_assignment.is_empty()
            || w.factorial_cell_counts.is_some()
            || w.second_factor_arms.is_some())
    {
        return Err(IoError::Convert("factorial metadata requires a factorial design".into()));
    }
    Ok(match &w.design {
        RandomizationDesignWire::Bernoulli => RandomizationDesign::Bernoulli,
        RandomizationDesignWire::Complete { treated_units } => {
            RandomizationDesign::Complete { treated_units: *treated_units }
        }
        RandomizationDesignWire::Cluster { treated_clusters } => {
            RandomizationDesign::Cluster { treated_clusters: *treated_clusters }
        }
        RandomizationDesignWire::Stratified => RandomizationDesign::Stratified {
            blocks: w
                .blocks
                .iter()
                .map(|x| Arc::<str>::from(x.as_str()))
                .collect::<Vec<_>>()
                .into(),
            treated_per_row: w.treated_per_row.clone().into(),
        },
        RandomizationDesignWire::Factorial2x2 => RandomizationDesign::Factorial2x2 {
            second_factor_assignment: w.second_factor_assignment.clone().into(),
            cell_counts: w
                .factorial_cell_counts
                .ok_or_else(|| IoError::Convert("factorial cell counts are missing".into()))?,
            second_factor_arms: {
                let arms = w
                    .second_factor_arms
                    .as_ref()
                    .ok_or_else(|| IoError::Convert("second-factor labels are missing".into()))?;
                (Arc::<str>::from(arms.0.as_str()), Arc::<str>::from(arms.1.as_str()))
            },
        },
        RandomizationDesignWire::MultiArm => RandomizationDesign::MultiArm {
            assignment: w.multi_arm_assignment.clone().into(),
            probabilities: w.multi_arm_probabilities.clone().into(),
            arms: w
                .multi_arm_labels
                .iter()
                .map(|arm| Arc::<str>::from(arm.as_str()))
                .collect::<Vec<_>>()
                .into(),
        },
        RandomizationDesignWire::Switchback => RandomizationDesign::Switchback {
            periods: w
                .periods
                .iter()
                .map(|period| Arc::<str>::from(period.as_str()))
                .collect::<Vec<_>>()
                .into(),
        },
    })
}

/// Decode the [`CausalQuery::RandomizedEffect`] payload.
fn randomized_effect_from_wire(w: &RandomizedEffectQueryWire) -> Result<CausalQuery, IoError> {
    let design = randomization_design_from_wire(w)?;
    let mut query = RandomizedEffectQuery::with_design(
        design,
        VariableId::from_raw(w.outcome),
        Arc::<[bool]>::from(w.realized_assignment.clone()),
        Arc::<[f64]>::from(w.assignment_probabilities.clone()),
        w.assignment_units.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
        w.outcome_units.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
        (
            Arc::<str>::from(w.treatment_arms.0.as_str()),
            Arc::<str>::from(w.treatment_arms.1.as_str()),
        ),
    );
    if let Some((id, coefficient)) = w.fixed_cuped {
        query = query.with_fixed_cuped(VariableId::from_raw(id), coefficient);
    }
    if !w.ancova_covariates.is_empty() {
        query = query.with_ancova(
            w.ancova_covariates.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
        );
    }
    if let Some(receipt) = &w.received_treatment {
        query = if w.estimand == RandomizedEstimandWire::TreatmentOnTreated {
            query.with_treatment_on_treated(receipt.clone())
        } else {
            query.with_received_treatment(receipt.clone())
        };
    }
    if w.exact_randomization_test {
        query = query.with_exact_randomization_test();
    }
    if (w.estimand != RandomizedEstimandWire::Itt) != w.received_treatment.is_some() {
        return Err(IoError::Convert(
            "receipt-adjusted estimand and treatment receipt must agree".into(),
        ));
    }
    query.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(CausalQuery::RandomizedEffect(query))
}

/// Decode the [`CausalQuery::PolicyValue`] payload.
fn policy_value_from_wire(w: &PolicyValueQueryWire) -> Result<CausalQuery, IoError> {
    let q = PolicyValueQuery {
        outcome: VariableId::from_raw(w.outcome),
        assignment: w.assignment.clone().into(),
        propensity: w.propensity.clone().into(),
        actions: w.actions.clone().into(),
        reference: w.reference.clone().into(),
        mu0: w.mu0.clone().into(),
        mu1: w.mu1.clone().into(),
        costs: w.costs.clone().into(),
        reference_costs: w.reference_costs.clone().into(),
        evaluation_subject_ids: w
            .evaluation_subject_ids
            .iter()
            .map(|x| Arc::<str>::from(x.as_str()))
            .collect::<Vec<_>>()
            .into(),
        disjoint_training_subjects: w.disjoint_training_subjects,
        crossfit_fold_ownership_valid: w.crossfit_fold_ownership_valid,
        global_constraints_present: w.global_constraints_present,
        multi_action: w.multi_action.as_ref().map(Into::into),
        uplift_bins: w.uplift_bins.clone().into(),
        uplift_bin_count: w.uplift_bin_count,
        uplift_training_subject_ids: w
            .uplift_training_subject_ids
            .iter()
            .map(|x| Arc::<str>::from(x.as_str()))
            .collect::<Vec<_>>()
            .into(),
        regret: w.regret.as_ref().map(|regret| antecedent_core::FixedCandidateRegretInputs {
            candidates: regret
                .candidates
                .iter()
                .map(|actions| Arc::<[bool]>::from(actions.clone()))
                .collect::<Vec<_>>()
                .into(),
            selected_index: regret.selected_index,
            training_subject_ids: regret
                .training_subject_ids
                .iter()
                .map(|id| Arc::<str>::from(id.as_str()))
                .collect::<Vec<_>>()
                .into(),
        }),
    };
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(CausalQuery::PolicyValue(q))
}

/// Decode the [`CausalQuery::ContinuousDoseResponse`] payload.
fn continuous_dose_response_from_wire(
    w: &ContinuousDoseResponseQueryWire,
) -> Result<CausalQuery, IoError> {
    let q = antecedent_core::ContinuousDoseResponseQuery {
        outcome: VariableId::from_raw(w.outcome),
        dose: VariableId::from_raw(w.dose),
        dose_density: VariableId::from_raw(w.dose_density),
        baseline_groups: w
            .baseline_groups
            .iter()
            .map(|value| Arc::<str>::from(value.as_str()))
            .collect::<Vec<_>>()
            .into(),
        target_doses: w.target_doses.clone().into(),
        bandwidth: w.bandwidth,
        min_local_support: w.min_local_support,
        density_provenance: Arc::from(w.density_provenance.as_str()),
        fixed_policy: w.fixed_policy.as_ref().map(|(policy, reference)| {
            antecedent_core::FixedGroupDosePolicy {
                policy_doses: policy
                    .iter()
                    .map(|(group, dose)| (Arc::from(group.as_str()), *dose))
                    .collect::<Vec<_>>()
                    .into(),
                reference_doses: reference
                    .iter()
                    .map(|(group, dose)| (Arc::from(group.as_str()), *dose))
                    .collect::<Vec<_>>()
                    .into(),
            }
        }),
    };
    q.validate().map_err(|error| IoError::Convert(error.to_string()))?;
    Ok(CausalQuery::ContinuousDoseResponse(q))
}

/// Decode the [`CausalQuery::PanelDid`] payload that carries a rollout.
fn panel_did_rollout_from_wire(w: &PanelDidQueryWire) -> Result<CausalQuery, IoError> {
    let rollout = w.rollout.as_ref().ok_or_else(|| IoError::Convert("rollout".into()))?;
    if w.augmented.is_some()
        || w.staggered_event_study
        || w.staggered_target.is_some()
        || w.repeated_cross_section
    {
        return Err(IoError::Convert(
            "a heterogeneity-robust rollout cannot also select another sampling design".into(),
        ));
    }
    let (design, robust) = crate::rollout_query_wire::rollout_from_wire(rollout)?;
    let subjects = w.subjects.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>();
    let clusters = w.clusters.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>();
    let mut q = if design == antecedent_core::DidSamplingDesign::CovariateAdjustedTwoPeriod {
        PanelDidQuery::covariate_adjusted_two_period(
            VariableId::from_raw(w.outcome),
            w.treated.clone(),
            w.post.clone(),
            subjects,
            clusters,
            Arc::clone(&robust.covariates),
            robust.nuisance,
        )
    } else {
        PanelDidQuery::heterogeneity_robust(
            VariableId::from_raw(w.outcome),
            subjects,
            clusters,
            w.periods.clone(),
            w.cohorts.clone(),
            robust.clone(),
        )
    };
    q.robust = Some(robust);
    q.resampling =
        w.resampling.as_ref().map(crate::rollout_query_wire::resampling_from_wire).transpose()?;
    if q.treated.as_ref() != w.treated.as_slice() || q.post.as_ref() != w.post.as_slice() {
        return Err(IoError::Convert(
            "rollout treatment and post indicators disagree with cohort and period metadata".into(),
        ));
    }
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(CausalQuery::PanelDid(q))
}

/// Decode the [`CausalQuery::PanelDid`] payload.
fn panel_did_from_wire(w: &PanelDidQueryWire) -> Result<CausalQuery, IoError> {
    let q = if let Some((pre, propensity, prediction, cross_fitted)) = w.augmented {
        if w.staggered_event_study
            || w.staggered_target.is_some()
            || w.repeated_cross_section
            || !w.periods.is_empty()
            || !w.cohorts.is_empty()
        {
            return Err(IoError::Convert(
                "augmented panel DiD cannot also select another sampling design".into(),
            ));
        }
        PanelDidQuery::augmented_panel(
            VariableId::from_raw(w.outcome),
            VariableId::from_raw(pre),
            VariableId::from_raw(propensity),
            VariableId::from_raw(prediction),
            w.treated.clone(),
            w.subjects.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
            w.clusters.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
            cross_fitted,
        )
    } else if w.staggered_event_study {
        if w.repeated_cross_section || w.staggered_target.is_some() {
            return Err(IoError::Convert(
                "event study cannot also select a group-time target or repeated cross section"
                    .into(),
            ));
        }
        PanelDidQuery::staggered_event_study(
            VariableId::from_raw(w.outcome),
            w.subjects.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
            w.clusters.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
            w.periods.clone(),
            w.cohorts.clone(),
        )
    } else if let Some((cohort, period)) = w.staggered_target {
        if w.repeated_cross_section {
            return Err(IoError::Convert(
                "staggered DiD cannot also be a repeated cross section".into(),
            ));
        }
        PanelDidQuery::staggered_group_time(
            VariableId::from_raw(w.outcome),
            w.subjects.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
            w.clusters.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
            w.periods.clone(),
            w.cohorts.clone(),
            cohort,
            period,
        )
    } else {
        let constructor = if w.repeated_cross_section {
            PanelDidQuery::repeated_cross_section
        } else {
            PanelDidQuery::new
        };
        constructor(
            VariableId::from_raw(w.outcome),
            w.treated.clone(),
            w.post.clone(),
            w.subjects.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
            w.clusters.iter().map(|x| Arc::<str>::from(x.as_str())).collect::<Vec<_>>(),
        )
    };
    if (w.staggered_target.is_some() || w.staggered_event_study || w.augmented.is_some())
        && (q.treated.as_ref() != w.treated.as_slice() || q.post.as_ref() != w.post.as_slice())
    {
        return Err(IoError::Convert(
            "staggered DiD treatment and post indicators disagree with cohort and period metadata"
                .into(),
        ));
    }
    let mut q = q;
    q.resampling =
        w.resampling.as_ref().map(crate::rollout_query_wire::resampling_from_wire).transpose()?;
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(CausalQuery::PanelDid(q))
}

/// Decode the [`CausalQuery::SyntheticControl`] payload.
fn synthetic_control_from_wire(w: &SyntheticControlQueryWire) -> Result<CausalQuery, IoError> {
    let q = SyntheticControlQuery::new(
        VariableId::from_raw(w.outcome),
        w.units.iter().map(|unit| Arc::<str>::from(unit.as_str())).collect::<Vec<_>>(),
        w.periods.clone(),
        Arc::<str>::from(w.treated_unit.as_str()),
        w.intervention_period,
    );
    let q = if w.difference_in_differences { q.difference_in_differences() } else { q };
    let q = if w.uniform_unit_randomization { q.with_uniform_unit_randomization() } else { q };
    let q =
        if let Some(effect) = w.sharp_null_effect { q.with_sharp_null_effect(effect) } else { q };
    let mut q = if let Some(ridge) = w.augmentation_ridge { q.with_augmentation(ridge) } else { q };
    q.rollout = w
        .rollout
        .as_ref()
        .map(crate::rollout_query_wire::synthetic_rollout_from_wire)
        .transpose()?;
    q.observed_window_support = w.observed_window_support;
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(CausalQuery::SyntheticControl(q))
}

/// Decode the [`CausalQuery::LocalPolynomialRatio`] payload.
fn local_polynomial_ratio_from_wire(
    w: &LocalPolynomialRatioQueryWire,
) -> Result<CausalQuery, IoError> {
    let q = LocalPolynomialRatioQuery {
        outcome: VariableId::from_raw(w.outcome),
        treatment: VariableId::from_raw(w.treatment),
        running: VariableId::from_raw(w.running),
        cutoff: w.cutoff,
        bandwidth: w.bandwidth,
        kink: w.kink,
        options: w
            .options
            .as_ref()
            .map(crate::rollout_query_wire::local_ratio_options_from_wire)
            .transpose()?,
    };
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(CausalQuery::LocalPolynomialRatio(q))
}

/// Decode the [`CausalQuery::Survival`] payload.
fn survival_from_wire(w: &SurvivalQueryWire) -> Result<CausalQuery, IoError> {
    if w.censoring_probability_floor.is_none()
        && (!w.censoring_times.is_empty() || !w.censoring_columns.is_empty())
    {
        return Err(IoError::Convert("censoring grid requires its positivity floor".into()));
    }
    let q = antecedent_core::SurvivalQuery {
        duration: VariableId::from_raw(w.duration),
        event: VariableId::from_raw(w.event),
        treatment: VariableId::from_raw(w.treatment),
        tau: w.tau,
        delayed_entry: w.delayed_entry.map(VariableId::from_raw),
        known_censoring: w.censoring_probability_floor.map(|minimum_probability| {
            antecedent_core::KnownCensoringSurvival {
                times: Arc::from(w.censoring_times.clone()),
                columns: Arc::from(
                    w.censoring_columns
                        .iter()
                        .copied()
                        .map(VariableId::from_raw)
                        .collect::<Vec<_>>(),
                ),
                minimum_probability,
            }
        }),
        observation_assumption: if w.independent_observation {
            antecedent_core::ObservationAssumption::IndependentGiven(Arc::from(
                w.independent_given.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>(),
            ))
        } else {
            return Err(IoError::Convert(
                "survival requires marginal independent observation declaration".into(),
            ));
        },
        functional: w.target_cause.map_or(
            antecedent_core::SurvivalFunctional::SurvivalAndRmst,
            |target_cause| antecedent_core::SurvivalFunctional::CumulativeIncidence {
                target_cause,
            },
        ),
    };
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(CausalQuery::Survival(q))
}

/// Decode the [`CausalQuery::LongitudinalRegime`] payload.
fn longitudinal_regime_from_wire(w: &LongitudinalRegimeQueryWire) -> Result<CausalQuery, IoError> {
    let method = match w.method.as_str() {
        "ipw" => antecedent_core::LongitudinalRegimeMethod::Ipw,
        "g_formula" => antecedent_core::LongitudinalRegimeMethod::GFormula,
        "sequential_dr" => antecedent_core::LongitudinalRegimeMethod::SequentialDoublyRobust,
        "marginal_structural_model" => {
            antecedent_core::LongitudinalRegimeMethod::MarginalStructuralModel
        }
        _ => return Err(IoError::Convert("unknown longitudinal regime method".into())),
    };
    let q = antecedent_core::LongitudinalRegimeQuery {
        outcome: VariableId::from_raw(w.outcome),
        periods: w.periods,
        method,
        period_outcome_predictions: if w.period_outcome_predictions.is_empty() {
            None
        } else {
            Some(w.period_outcome_predictions.clone().into())
        },
        known_fixed_outcome_predictions: w.known_fixed_outcome_predictions,
        stabilizing_numerator_probabilities: if w.stabilizing_numerator_probabilities.is_empty() {
            None
        } else {
            Some(w.stabilizing_numerator_probabilities.clone().into())
        },
        q_predictions: if w.q_predictions.is_empty() {
            None
        } else {
            Some(w.q_predictions.clone().into())
        },
        observation_history: if w.observation_history.is_empty() {
            None
        } else {
            Some(w.observation_history.clone().into())
        },
        prediction_fold_ids: if w.prediction_fold_ids.is_empty() {
            None
        } else {
            Some(w.prediction_fold_ids.clone().into())
        },
        treatment_history: w.treatment_history.clone().into(),
        regime_actions: w.regime_actions.clone().into(),
        treatment_probabilities: w.treatment_probabilities.clone().into(),
        censoring_probabilities: w.censoring_probabilities.clone().into(),
        outcome_observed: w.outcome_observed.clone().into(),
        subject_ids: w
            .subject_ids
            .iter()
            .map(|s| Arc::<str>::from(s.as_str()))
            .collect::<Vec<_>>()
            .into(),
        fold_ids: w.fold_ids.clone().into(),
        excluded_fold_predictions: w.excluded_fold_predictions,
        probabilities_known_by_design: w.probabilities_known_by_design,
        minimum_probability: w.minimum_probability,
        rule_id: w.rule_id.as_deref().map(Arc::<str>::from),
        rule_version: w.rule_version.as_deref().map(Arc::<str>::from),
        rule_provenance: w.rule_provenance.as_deref().map(Arc::<str>::from),
    };
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(CausalQuery::LongitudinalRegime(q))
}

/// Decode the [`CausalQuery::SequentialContrast`] payload.
fn sequential_contrast_from_wire(w: &SequentialContrastQueryWire) -> Result<CausalQuery, IoError> {
    let q = antecedent_core::SequentialContrastQuery {
        outcome: VariableId::from_raw(w.outcome),
        first_treatment: VariableId::from_raw(w.first_treatment),
        second_treatment: VariableId::from_raw(w.second_treatment),
        baseline: w.baseline.iter().map(|id| VariableId::from_raw(*id)).collect(),
        intermediate: w.intermediate.iter().map(|id| VariableId::from_raw(*id)).collect(),
        active: w.active,
        control: w.control,
        fold_of_row: w.fold_of_row.clone().into(),
        fold_count: w.fold_count,
        min_probability: w.min_probability,
    };
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(CausalQuery::SequentialContrast(q))
}

/// Decode the [`CausalQuery::HistoryMatch`] payload.
fn history_match_from_wire(w: &HistoryMatchQueryWire) -> Result<CausalQuery, IoError> {
    let q = antecedent_core::HistoryMatchQuery {
        outcome: VariableId::from_raw(w.outcome),
        treated: VariableId::from_raw(w.treated),
        history: w.history.iter().map(|id| VariableId::from_raw(*id)).collect(),
        period: w.period.map(VariableId::from_raw),
        weight: w.weight.map(VariableId::from_raw),
        caliper: w.caliper,
    };
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(CausalQuery::HistoryMatch(q))
}

/// Decode the [`CausalQuery::IvBounds`] payload.
fn iv_bounds_from_wire(w: &IvBoundsQueryWire) -> Result<CausalQuery, IoError> {
    let q = antecedent_core::IvBoundsQuery {
        outcome: VariableId::from_raw(w.outcome),
        treatment: VariableId::from_raw(w.treatment),
        instrument: VariableId::from_raw(w.instrument),
        weight: w.weight.map(VariableId::from_raw),
    };
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(CausalQuery::IvBounds(q))
}

/// Decode the effect-family variants (average, temporal and mediation effects) of a
/// [`CausalQueryWire`].
fn effect_query_from_wire(w: &CausalQueryWire) -> Result<CausalQuery, IoError> {
    Ok(match w {
        CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            effect_modifiers,
            control,
            active,
            target_population,
            outcome_functional,
        } => CausalQuery::AverageEffect(
            AverageEffectQuery::new(
                VariableId::from_raw(*treatment),
                VariableId::from_raw(*outcome),
                vars_from_raw(effect_modifiers),
                control.to_domain(),
                active.to_domain(),
                target_population.to_domain()?,
            )
            .with_outcome_functional(outcome_functional.to_domain()),
        ),
        CausalQueryWire::TemporalEffect {
            treatment,
            outcome,
            policy,
            control,
            active,
            horizon_steps,
            max_history_lag,
            target_population,
            effect_modifiers,
        } => CausalQuery::TemporalEffect(TemporalEffectQuery {
            treatment: VariableId::from_raw(*treatment),
            outcome: VariableId::from_raw(*outcome),
            policy: policy.to_domain(),
            control: control.to_domain(),
            active: active.to_domain(),
            horizon_steps: *horizon_steps,
            max_history_lag: *max_history_lag,
            target_population: target_population.to_domain()?,
            effect_modifiers: effect_modifiers.iter().copied().map(VariableId::from_raw).collect(),
        }),
        CausalQueryWire::Mediation {
            treatment,
            outcome,
            mediators,
            contrast,
            control,
            active,
            target_population,
            horizons,
        } => {
            let query = MediationQuery {
                treatment: VariableId::from_raw(*treatment),
                outcome: VariableId::from_raw(*outcome),
                mediators: vars_from_raw(mediators),
                contrast: (*contrast).into(),
                control: control.to_domain(),
                active: active.to_domain(),
                target_population: target_population.to_domain()?,
                horizons: Arc::from(horizons.as_slice()),
            };
            query.validate().map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::Mediation(query)
        }
        _ => return Err(IoError::Convert("wire query is not of the effect family".into())),
    })
}

/// Decode the counterfactual-family variants (counterfactual, nested and conditional effects) of a
/// [`CausalQueryWire`].
fn counterfactual_query_from_wire(w: &CausalQueryWire) -> Result<CausalQuery, IoError> {
    Ok(match w {
        CausalQueryWire::Counterfactual {
            outcomes,
            interventions,
            control,
            joint_control,
            allow_nested,
        } => {
            let interventions: Arc<[Intervention]> =
                interventions.iter().map(InterventionWire::to_domain).collect::<Vec<_>>().into();
            let control = match control {
                Some(c) => c.to_domain(),
                None => CounterfactualQuery::default_control(&interventions),
            };
            CausalQuery::Counterfactual(CounterfactualQuery {
                outcomes: vars_from_raw(outcomes),
                interventions,
                control,
                joint_control: joint_control.iter().map(InterventionWire::to_domain).collect(),
                allow_nested: *allow_nested,
            })
        }
        CausalQueryWire::NestedCounterfactual {
            treatment,
            mediator,
            outcome,
            control_bits,
            active_bits,
        } => {
            let query = antecedent_core::NestedCounterfactualQuery::with_levels(
                antecedent_core::VariableId::from_raw(*treatment),
                antecedent_core::VariableId::from_raw(*mediator),
                antecedent_core::VariableId::from_raw(*outcome),
                f64::from_bits(*control_bits),
                f64::from_bits(*active_bits),
            )
            .map_err(|e| IoError::Convert(e.to_string()))?;
            CausalQuery::NestedCounterfactual(query)
        }
        CausalQueryWire::ConditionalEffect { inner } => {
            let CausalQuery::AverageEffect(inner_q) = causal_query_from_wire(inner)? else {
                return Err(IoError::Convert(
                    "ConditionalEffect.inner must be AverageEffect".into(),
                ));
            };
            CausalQuery::ConditionalEffect(ConditionalEffectQuery { inner: inner_q })
        }
        _ => return Err(IoError::Convert("wire query is not of the counterfactual family".into())),
    })
}

/// Decode the attribution-family variants (anomaly, change, mechanism change and unit change) of a
/// [`CausalQueryWire`].
fn attribution_query_from_wire(w: &CausalQueryWire) -> Result<CausalQuery, IoError> {
    Ok(match w {
        CausalQueryWire::AnomalyAttribution { targets, unit_rows, max_units, reference } => {
            CausalQuery::AnomalyAttribution(AnomalyAttributionQuery {
                targets: vars_from_raw(targets),
                unit_rows: unit_rows
                    .as_ref()
                    .map(|rows| {
                        rows.iter()
                            .map(|&r| usize::try_from(r).map_err(|_| IoError::TooLarge))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?
                    .map(Arc::from),
                max_units: usize::try_from(*max_units).map_err(|_| IoError::TooLarge)?,
                reference: match reference.flatten() {
                    None => AnomalyReference::Empirical,
                    Some((center, scale)) => AnomalyReference::fixed(center, scale),
                },
            })
        }
        CausalQueryWire::ChangeAttribution {
            outcome,
            baseline,
            comparison,
            components,
            allocation,
            max_components,
        } => CausalQuery::ChangeAttribution(ChangeAttributionQuery {
            outcome: VariableId::from_raw(*outcome),
            baseline: population_from_wire(baseline)?,
            comparison: population_from_wire(comparison)?,
            components: (*components).into(),
            allocation: allocation_from_wire(allocation)?,
            max_components: usize::try_from(*max_components).map_err(|_| IoError::TooLarge)?,
        }),
        CausalQueryWire::MechanismChange {
            targets,
            baseline,
            comparison,
            significance_level_bits,
            max_targets,
            change_prior_bits,
        } => CausalQuery::MechanismChange(MechanismChangeQuery {
            targets: vars_from_raw(targets),
            baseline: population_from_wire(baseline)?,
            comparison: population_from_wire(comparison)?,
            significance_level: OrderedFloatBits::from_f64(f64::from_bits(
                *significance_level_bits,
            )),
            max_targets: usize::try_from(*max_targets).map_err(|_| IoError::TooLarge)?,
            change_prior: change_prior_bits
                .map(|bits| OrderedFloatBits::from_f64(f64::from_bits(bits))),
        }),
        CausalQueryWire::UnitChange { outcome, unit_rows, components, allocation, max_units } => {
            CausalQuery::UnitChange(UnitChangeQuery {
                outcome: VariableId::from_raw(*outcome),
                unit_rows: unit_rows
                    .as_ref()
                    .map(|rows| {
                        rows.iter()
                            .map(|&r| usize::try_from(r).map_err(|_| IoError::TooLarge))
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?
                    .map(Arc::from),
                components: (*components).into(),
                allocation: allocation_from_wire(allocation)?,
                max_units: usize::try_from(*max_units).map_err(|_| IoError::TooLarge)?,
            })
        }
        _ => return Err(IoError::Convert("wire query is not of the attribution family".into())),
    })
}

/// Decode and validate a noncausal association query.
///
/// # Errors
///
/// An unknown measure or outcome scale, or a declaration the query refuses.
pub fn association_query_from_wire(
    w: &AssociationQueryWire,
) -> Result<antecedent_core::AssociationQuery, IoError> {
    use antecedent_core::{
        AssociationFunctional, AssociationMeasure, AssociationOutcome, AssociationQuery,
    };
    let functional = match &w.functional {
        AssociationFunctionalWire::Auto => AssociationFunctional::Auto,
        AssociationFunctionalWire::Conditional => AssociationFunctional::Conditional,
        AssociationFunctionalWire::Coefficients => AssociationFunctional::Coefficients,
        AssociationFunctionalWire::Measure { measure } => {
            AssociationFunctional::Measure(AssociationMeasure::from_name(measure).ok_or_else(
                || IoError::Convert(format!("unknown association measure `{measure}`")),
            )?)
        }
        AssociationFunctionalWire::Predictive { folds } => {
            AssociationFunctional::Predictive { folds: *folds }
        }
    };
    let outcome_scale = match w.outcome_scale.as_deref() {
        None => AssociationOutcome::Schema,
        Some(name) => AssociationOutcome::from_name(name).ok_or_else(|| {
            IoError::Convert(format!("unknown association outcome scale `{name}`"))
        })?,
    };
    let mut q = AssociationQuery::new(
        VariableId::from_raw(w.outcome),
        w.predictors.iter().map(|id| VariableId::from_raw(*id)).collect::<Vec<_>>(),
    )
    .with_conditioning(
        w.conditioning.iter().map(|id| VariableId::from_raw(*id)).collect::<Vec<_>>(),
    )
    .with_functional(functional)
    .with_outcome_scale(outcome_scale);
    if let Some(weight) = w.weight {
        q = q.with_weight(VariableId::from_raw(weight));
    }
    if let Some(reason) = &w.refused_after {
        q = q.following_refusal(reason.as_str());
    }
    q.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(q)
}

/// Decode and validate a structural transport query.
///
/// # Errors
///
/// Invalid response or transport semantics.
pub fn transport_query_from_wire(w: &TransportQueryWire) -> Result<TransportQuery, IoError> {
    let mut query = TransportQuery::new(
        response_query_from_wire(&w.response)?,
        w.source_population.as_str(),
        w.target_population.as_str(),
        vars_from_raw(&w.source_experiments),
    );
    if let Some(catalog) = &w.catalog {
        query = query
            .with_catalog(catalog.to_catalog()?)
            .map_err(|e| IoError::Convert(e.to_string()))?;
    }
    query.validate().map_err(|error| IoError::Convert(error.to_string()))?;
    Ok(query)
}

/// Decode and validate a randomized interference query.
///
/// # Errors
///
/// Counts that do not fit `usize` or invalid query semantics.
pub fn interference_query_from_wire(
    w: &InterferenceQueryWire,
) -> Result<InterferenceQuery, IoError> {
    let assignment = match &w.assignment {
        AssignmentDesignWire::Bernoulli { probabilities } => {
            AssignmentDesign::Bernoulli { probabilities: probabilities.clone().into() }
        }
        AssignmentDesignWire::CompleteRandomization { treated } => {
            AssignmentDesign::CompleteRandomization {
                treated: usize::try_from(*treated).map_err(|_| IoError::TooLarge)?,
            }
        }
        AssignmentDesignWire::ClusterRandomization { clusters, treated_clusters } => {
            AssignmentDesign::ClusterRandomization {
                clusters: clusters.clone().into(),
                treated_clusters: usize::try_from(*treated_clusters)
                    .map_err(|_| IoError::TooLarge)?,
            }
        }
        AssignmentDesignWire::TwoStageSaturation {
            clusters,
            low_probability,
            high_probability,
            high_clusters,
            realized_saturation,
        } => AssignmentDesign::TwoStageSaturation {
            clusters: clusters.clone().into(),
            low_probability: *low_probability,
            high_probability: *high_probability,
            high_clusters: usize::try_from(*high_clusters).map_err(|_| IoError::TooLarge)?,
            realized_saturation: realized_saturation.clone().into(),
        },
        AssignmentDesignWire::ObservedExposure {
            clusters,
            propensity_from,
            propensity_to,
            provenance,
            assume_network_exchangeability,
        } => AssignmentDesign::ObservedExposure {
            clusters: clusters.clone().into(),
            propensity_from: propensity_from.clone().into(),
            propensity_to: propensity_to.clone().into(),
            provenance: match provenance.as_str() {
                "known" => antecedent_core::ExposurePropensityProvenance::Known,
                "externally_estimated" => {
                    antecedent_core::ExposurePropensityProvenance::ExternallyEstimated
                }
                _ => {
                    return Err(IoError::Convert(
                        "unknown observational propensity provenance".into(),
                    ));
                }
            },
            assume_network_exchangeability: *assume_network_exchangeability,
        },
    };
    let exposure = match &w.exposure {
        ExposureMappingWire::OwnTreatment => ExposureMapping::OwnTreatment,
        ExposureMappingWire::NeighborCount => ExposureMapping::NeighborCount,
        ExposureMappingWire::NeighborFraction => ExposureMapping::NeighborFraction,
        ExposureMappingWire::WeightedNeighborExposure => ExposureMapping::WeightedNeighborExposure,
        ExposureMappingWire::Custom(id) => ExposureMapping::Custom(Arc::from(id.as_str())),
    };
    let InterferenceFunctionalWire::ExposureContrast { outcome, from, to } = &w.functional;
    let query = InterferenceQuery {
        assignment,
        exposure,
        functional: InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(*outcome),
            from: ExposureLevel { own: from.own, neighbors: from.neighbors },
            to: ExposureLevel { own: to.own, neighbors: to.neighbors },
        },
        probability_draws: w.probability_draws,
    };
    query.validate().map_err(|error| IoError::Convert(error.to_string()))?;
    Ok(query)
}

/// Decode an interventional distribution query.
///
/// # Errors
///
/// Row indices that do not fit `usize` or an invalid distribution query.
pub fn interventional_distribution_from_wire(
    w: &InterventionalDistributionQueryWire,
) -> Result<InterventionalDistributionQuery, IoError> {
    let query = InterventionalDistributionQuery {
        outcomes: vars_from_raw(&w.outcomes),
        interventions: w
            .interventions
            .iter()
            .map(InterventionWire::to_domain)
            .collect::<Vec<_>>()
            .into(),
        conditioning: vars_from_raw(&w.conditioning),
        target_population: w.target_population.to_domain()?,
    };
    query.validate().map_err(|error| IoError::Convert(error.to_string()))?;
    Ok(query)
}

/// Decode a path-specific effect query.
///
/// # Errors
///
/// Limits that do not fit `usize` or an invalid path-specific query.
pub fn path_specific_from_wire(
    w: &PathSpecificEffectQueryWire,
) -> Result<PathSpecificEffectQuery, IoError> {
    let query = PathSpecificEffectQuery {
        treatment: VariableId::from_raw(w.treatment),
        outcome: VariableId::from_raw(w.outcome),
        path_nodes: vars_from_raw(&w.path_nodes),
        control: w.control.to_domain(),
        active: w.active.to_domain(),
        target_population: w.target_population.to_domain()?,
        max_paths: usize::try_from(w.max_paths)
            .map_err(|_| IoError::Convert("max_paths does not fit usize".into()))?,
        max_len: usize::try_from(w.max_len)
            .map_err(|_| IoError::Convert("max_len does not fit usize".into()))?,
    };
    query.validate().map_err(|error| IoError::Convert(error.to_string()))?;
    Ok(query)
}
