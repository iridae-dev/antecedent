//! Field checks for a decoded analysis-result body.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::graphless::{
    graphless_status_ok, validate_continuous_dose_graphless, validate_did_graphless,
    validate_history_match_graphless, validate_longitudinal_graphless, validate_policy_graphless,
    validate_randomized_graphless, validate_ratio_graphless,
    validate_sequential_contrast_graphless, validate_survival_graphless,
};
use super::{
    AnalysisResultWire, StructuralResponseMixtureWire, TemporalMediationGridWire,
    TemporalMediationUncertaintyWire, UnitEffectsWire, identified_set_interval_from_wire,
};
use crate::{CausalQueryWire, IdentificationResultWire, IoError};

// allow(too_many_lines): single dispatch over every result-variant validator; splitting hides the total contract
#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
pub(super) fn validate_result(
    result: &AnalysisResultWire,
    variable_names: &[String],
    allow_legacy_graphless_missing: bool,
) -> Result<(), IoError> {
    use crate::causal_artifact::{
        validate_query_ids, validate_response_result, validate_variable_names,
    };
    if let Some(model) = &result.fitted_effect {
        // Unsupported optional prediction codecs do not erase the scientific claim.
        // The sealed body still binds their bytes; prediction loading rejects them.
        if model.version == 1 && model.predictor.version == 1 {
            model.validate().map_err(|e| IoError::Convert(e.to_string()))?;
        }
        if model.features.iter().any(|v| *v as usize >= variable_names.len()) {
            return Err(IoError::Convert("fitted effect names an unknown feature".into()));
        }
    }
    validate_variable_names(variable_names)?;
    validate_query_ids(&result.query, variable_names.len())?;
    if let Some(inference) = &result.interference_inference {
        let crate::CausalQueryWire::Interference(query) = &result.query else {
            return Err(IoError::Convert(
                "interference inference is attached to a different query".into(),
            ));
        };
        let expected = match &query.assignment {
            crate::AssignmentDesignWire::ClusterRandomization { .. } => "cluster_total_neyman_welch",
            crate::AssignmentDesignWire::TwoStageSaturation { .. } => "saturation_cluster_neyman_welch",
            crate::AssignmentDesignWire::ObservedExposure { .. } => "observational_known_exposure_cluster_t",
            _ => return Err(IoError::Convert("independent-cluster inference requires a cluster or supplied-exposure interference design".into())),
        };
        let minimum_first_stage_clusters = if matches!(
            &query.assignment,
            crate::AssignmentDesignWire::TwoStageSaturation { .. }
        ) {
            24
        } else {
            8
        };
        if inference.method != expected
            || inference.from_exposed_units == 0
            || inference.to_exposed_units == 0
            || inference.from_exposed_clusters == 0
            || inference.to_exposed_clusters == 0
            || (inference.interval.is_some() == inference.interval_unavailable_reason.is_some())
        {
            return Err(IoError::Convert(
                "interference inference method, support, or interval status is inconsistent".into(),
            ));
        }
        if let Some(interval) = &inference.interval {
            if let crate::AssignmentDesignWire::ObservedExposure { provenance, .. } =
                &query.assignment
            {
                if provenance != "known"
                    || inference.from_exposed_clusters < 8
                    || inference.to_exposed_clusters < 8
                    || interval.degrees_of_freedom < 29.0
                {
                    return Err(IoError::Convert("observational exposure interval requires known probabilities and independent-cluster support".into()));
                }
            }
            if !interval.lower.is_finite()
                || !interval.upper.is_finite()
                || interval.lower >= interval.upper
                || !interval.standard_error.is_finite()
                || interval.standard_error <= 0.0
                || !interval.degrees_of_freedom.is_finite()
                || interval.degrees_of_freedom <= 0.0
                || interval
                    .first_stage_arm_clusters
                    .iter()
                    .any(|&count| count < minimum_first_stage_clusters)
                || inference.from_exposed_clusters < 8
                || inference.to_exposed_clusters < 8
                || result
                    .estimate
                    .is_none_or(|effect| effect < interval.lower || effect > interval.upper)
            {
                return Err(IoError::Convert(
                    "interference pointwise interval violates its support or scalar claim".into(),
                ));
            }
        }
        // A published, support-passing interval on one of the three off-axis
        // cluster designs is exactly the licensed condition; every invalid or
        // unsupported interval was already refused above, so recomputing the
        // license from the published interval catches a forged status.
        let graphless_licensed = inference.interval.is_some();
        if !graphless_status_ok(
            inference.graphless_support_status.as_deref(),
            graphless_licensed,
            allow_legacy_graphless_missing,
        ) {
            return Err(IoError::Convert(
                "interference graphless license does not match the published interval support"
                    .into(),
            ));
        }
    }
    if matches!(&result.query, crate::CausalQueryWire::RandomizedEffect(query)
        if matches!(query.design, crate::RandomizationDesignWire::Switchback))
        && result.randomized_effect.is_none()
        && result.posterior_artifact.is_none()
    {
        return Err(IoError::Convert(
            "switchback artifact is missing its randomized result section".into(),
        ));
    }
    validate_randomized_graphless(result, allow_legacy_graphless_missing)?;
    // A Bayesian design result carries its posterior in place of the design-specific frequentist
    // section; exactly one of the two must be present.
    let design_posterior = result.posterior_artifact.is_some();
    if matches!(result.query, crate::CausalQueryWire::PanelDid(_))
        && result.panel_did.is_some() == design_posterior
    {
        return Err(IoError::Convert(
            "panel DiD artifact is missing its design-specific result section".into(),
        ));
    }
    if let crate::CausalQueryWire::SyntheticControl(query) = &result.query {
        if design_posterior {
            if result.synthetic_did.is_some() || result.synthetic_control.is_some() {
                return Err(IoError::Convert(
                    "a synthetic-control posterior replaces the frequentist weight sections".into(),
                ));
            }
        } else if query.difference_in_differences {
            if result.synthetic_did.is_none() || result.synthetic_control.is_some() {
                return Err(IoError::Convert(
                    "synthetic DiD artifact requires its distinct weight result section".into(),
                ));
            }
        } else if result.synthetic_control.is_none() || result.synthetic_did.is_some() {
            return Err(IoError::Convert(
                "synthetic-control artifact requires its donor-support result section".into(),
            ));
        }
    }
    if matches!(result.query, crate::CausalQueryWire::LocalPolynomialRatio(_))
        && result.local_polynomial_ratio.is_some() == design_posterior
    {
        return Err(IoError::Convert(
            "local ratio artifact is missing its design-specific result section".into(),
        ));
    }
    validate_ratio_graphless(result)?;
    if let Some(row) = &result.posterior_row {
        // A posterior row replaces the frequentist section of exactly the queries that have a
        // Bayesian counterpart on the same coordinate, and rides with its posterior artifact.
        let attached = match &result.query {
            crate::CausalQueryWire::Survival(_) => result.survival.is_none(),
            crate::CausalQueryWire::ContinuousDoseResponse(query) => {
                query.fixed_policy.is_some() && result.continuous_dose_response.is_none()
            }
            // The randomized quantile posterior rides beside the ordinary scalar posterior of
            // the quantile treatment effect.
            crate::CausalQueryWire::AverageEffect { .. } => {
                row.estimator == "bayesian.randomized_quantile"
            }
            _ => false,
        };
        if !attached || result.posterior_artifact.is_none() {
            return Err(IoError::Convert(
                "a posterior row is attached to a query without its Bayesian counterpart, beside a frequentist section or without its posterior artifact".into(),
            ));
        }
        row.validate()?;
    }
    if matches!(result.query, crate::CausalQueryWire::Survival(_))
        && result.survival.is_none()
        && result.posterior_row.is_none()
    {
        return Err(IoError::Convert(
            "survival artifact is missing its curve result section".into(),
        ));
    }
    if matches!(result.query, crate::CausalQueryWire::LongitudinalRegime(_))
        && result.longitudinal_regime.is_none()
    {
        return Err(IoError::Convert(
            "longitudinal regime artifact is missing its value section".into(),
        ));
    }
    validate_longitudinal_graphless(result, allow_legacy_graphless_missing)?;
    validate_sequential_contrast_graphless(result)?;
    validate_history_match_graphless(result)?;
    crate::causal_query_from_wire(&result.query)?;
    let identification_count = match &result.identification_variables {
        Some(variables) => {
            validate_temporal_namespace(variables, variable_names.len())?;
            variables.len()
        }
        None => variable_names.len(),
    };
    validate_identification_namespace(&result.identification, identification_count)?;
    let mut horizons = std::collections::BTreeSet::new();
    for horizon in &result.temporal_identification {
        if horizon.horizon == 0 || !horizons.insert(horizon.horizon) {
            return Err(IoError::Convert(
                "temporal identification horizons must be positive and unique".into(),
            ));
        }
        validate_temporal_namespace(&horizon.variables, variable_names.len())?;
        validate_identification_namespace(&horizon.identification, horizon.variables.len())?;
        crate::identification_from_wire(&horizon.identification)?;
    }
    crate::identification_from_wire(&result.identification)?;
    validate_interventional_distribution(result)?;
    if result.identification_variables.is_none() && result.identification.query != result.query {
        return Err(IoError::Convert(
            "identification.query does not match the enclosing query".into(),
        ));
    }
    if let crate::CausalQueryWire::TemporalEffect { horizon_steps, .. } = &result.query {
        if !result.temporal_identification.is_empty()
            && !result.temporal_identification.iter().any(|item| item.horizon == *horizon_steps)
        {
            return Err(IoError::Convert(
                "temporal certificate horizon does not match the enclosing query".into(),
            ));
        }
    }
    if result.estimate.is_some_and(|estimate| !estimate.is_finite()) {
        return Err(IoError::Convert(
            "analysis scalar estimate must be finite when present".into(),
        ));
    }
    let dose_query = matches!(result.query, crate::CausalQueryWire::ContinuousDoseResponse(_));
    let dose_section = result.continuous_dose_response.is_some();
    // A posterior row also rides on survival and quantile results, so only the dose query
    // is required to carry a dose section or a row, and only it may carry the dose section.
    if (dose_query && !dose_section && result.posterior_row.is_none())
        || (!dose_query && dose_section)
    {
        return Err(IoError::Convert(
            "continuous-dose query and result section must appear together".into(),
        ));
    }
    validate_continuous_dose_graphless(result, allow_legacy_graphless_missing)?;
    validate_policy_graphless(result)?;
    validate_did_graphless(result)?;
    if let Some(fit) = &result.synthetic_control {
        let crate::CausalQueryWire::SyntheticControl(query) = &result.query else {
            return Err(IoError::Convert(
                "synthetic-control result is attached to a different query".into(),
            ));
        };
        if query.difference_in_differences {
            return Err(IoError::Convert(
                "synthetic-control result is attached to a synthetic DiD query".into(),
            ));
        }
        let donors: std::collections::BTreeSet<&str> = query
            .units
            .iter()
            .map(String::as_str)
            .filter(|unit| *unit != query.treated_unit.as_str())
            .collect();
        let pre: std::collections::BTreeSet<i64> = query
            .periods
            .iter()
            .copied()
            .filter(|period| *period < query.intervention_period)
            .collect();
        let post: std::collections::BTreeSet<i64> = query
            .periods
            .iter()
            .copied()
            .filter(|period| *period >= query.intervention_period)
            .collect();
        let weight_names: Vec<&str> =
            fit.donor_weights.iter().map(|(unit, _)| unit.as_str()).collect();
        let squared_mass: f64 = fit.donor_weights.iter().map(|(_, weight)| weight * weight).sum();
        #[allow(
            clippy::float_cmp,
            reason = "randomization null effect must equal the frozen sharp-null query value exactly"
        )]
        if result.estimate != Some(fit.effect)
            || result.standard_error.is_some()
            || result.interval_lower.is_some()
            || result.interval_upper.is_some()
            || !fit.effect.is_finite()
            || !fit.pre_treatment_rmse.is_finite()
            || fit.pre_treatment_rmse < 0.0
            || !fit.placebo_rank.is_finite()
            || !(0.0..=1.0).contains(&fit.placebo_rank)
            || !fit.effective_donors.is_finite()
            || fit.effective_donors < 1.0
            || fit.donor_weights.len() != donors.len()
            || donors.len() < 3
            || weight_names != donors.iter().copied().collect::<Vec<_>>()
            || fit.donor_weights.iter().any(|(_, weight)| !weight.is_finite() || *weight < 0.0)
            || (fit.donor_weights.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() > 1e-8
            || (fit.effective_donors - 1.0 / squared_mass).abs() > 1e-8
            || fit.placebo_effects.len() != donors.len()
            || fit.placebo_effects.iter().any(|effect| !effect.is_finite())
            || fit.n_pre_periods != pre.len()
            || fit.n_post_periods != post.len()
            || (!query.uniform_unit_randomization
                && ((query.augmentation_ridge.is_none()
                    && fit.uncertainty != "point_only_with_unlicensed_placebo_rank")
                    || fit.randomization_p_value.is_some()
                    || fit.randomization_null_effect.is_some()
                    || !fit.randomization_statistics.is_empty()))
            || (query.augmentation_ridge.is_none()
                && (fit.unadjusted_effect.is_some()
                    || fit.outcome_model_correction.is_some()
                    || fit.augmentation_ridge.is_some()))
            || (query.augmentation_ridge.is_some()
                && ((!query.uniform_unit_randomization
                    && fit.uncertainty != "point_only_augmented_no_interval")
                    || fit.augmentation_ridge != query.augmentation_ridge
                    || !fit.unadjusted_effect.is_some_and(f64::is_finite)
                    || !fit.outcome_model_correction.is_some_and(f64::is_finite)
                    || (fit.unadjusted_effect.unwrap_or(f64::NAN)
                        - fit.outcome_model_correction.unwrap_or(f64::NAN)
                        - fit.effect)
                        .abs()
                        > 1e-8))
            || (query.uniform_unit_randomization
                && (fit.randomization_null_effect.is_some_and(|effect| {
                    !effect.is_finite() || effect != query.sharp_null_effect.unwrap_or(0.0)
                }) || (query.sharp_null_effect.is_some()
                    && fit.randomization_null_effect.is_none())
                    || !query.sharp_null_effect.unwrap_or(0.0).is_finite()
                    || (query.augmentation_ridge.is_none()
                        && fit.uncertainty
                            != "point_only_with_exact_unit_randomization_p_value_no_interval")
                    || (query.augmentation_ridge.is_some()
                        && fit.uncertainty
                            != "point_only_augmented_with_exact_unit_randomization_p_value_no_interval")
                    || fit.randomization_statistics.len() != donors.len() + 1
                    || fit.randomization_statistics.len() > 32
                    || fit
                        .randomization_statistics
                        .iter()
                        .map(|(unit, _)| unit.as_str())
                        .collect::<Vec<_>>()
                        != query
                            .units
                            .iter()
                            .map(String::as_str)
                            .collect::<std::collections::BTreeSet<_>>()
                            .into_iter()
                            .collect::<Vec<_>>()
                    || fit
                        .randomization_statistics
                        .iter()
                        .any(|(_, statistic)| !statistic.is_finite() || *statistic < 0.0)
                    || fit.randomization_p_value
                        != fit
                            .randomization_statistics
                            .iter()
                            .find(|(unit, _)| unit == &query.treated_unit)
                            .map(|(_, observed)| {
                                fit.randomization_statistics
                                    .iter()
                                    .filter(|(_, statistic)| statistic >= observed)
                                    .count() as f64
                                    / fit.randomization_statistics.len() as f64
                            })))
        {
            return Err(IoError::Convert(
                "invalid synthetic-control payload or fabricated interval".into(),
            ));
        }
    }
    if let Some(fit) = &result.synthetic_did {
        let crate::CausalQueryWire::SyntheticControl(query) = &result.query else {
            return Err(IoError::Convert(
                "synthetic DiD result is attached to a different query".into(),
            ));
        };
        let donors: std::collections::BTreeSet<&str> = query
            .units
            .iter()
            .map(String::as_str)
            .filter(|unit| *unit != query.treated_unit.as_str())
            .collect();
        let pre: std::collections::BTreeSet<i64> = query
            .periods
            .iter()
            .copied()
            .filter(|period| *period < query.intervention_period)
            .collect();
        let post: std::collections::BTreeSet<i64> = query
            .periods
            .iter()
            .copied()
            .filter(|period| *period >= query.intervention_period)
            .collect();
        #[allow(
            clippy::float_cmp,
            reason = "randomization null effect must equal the frozen sharp-null query value exactly"
        )]
        if !query.difference_in_differences
            || result.estimate != Some(fit.effect)
            || result.standard_error.is_some()
            || result.interval_lower.is_some()
            || result.interval_upper.is_some()
            || !fit.effect.is_finite()
            || !fit.pre_treatment_rmse.is_finite()
            || fit.pre_treatment_rmse < 0.0
            || donors.len() < 2
            || pre.len() < 2
            || post.is_empty()
            || fit.n_donors != donors.len()
            || fit.n_pre_periods != pre.len()
            || fit.n_post_periods != post.len()
            || fit.donor_weights.iter().map(|(unit, _)| unit.as_str()).collect::<Vec<_>>()
                != donors.iter().copied().collect::<Vec<_>>()
            || fit.donor_weights.iter().any(|(_, weight)| !weight.is_finite() || *weight < 0.0)
            || (fit.donor_weights.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() > 1e-8
            || fit.time_weights.iter().map(|(period, _)| *period).collect::<Vec<_>>()
                != pre.iter().copied().collect::<Vec<_>>()
            || fit.time_weights.iter().any(|(_, weight)| !weight.is_finite() || *weight < 0.0)
            || (fit.time_weights.iter().map(|(_, weight)| weight).sum::<f64>() - 1.0).abs() > 1e-8
            || (!query.uniform_unit_randomization
                && (fit.uncertainty != "point_only_no_interval"
                    || fit.randomization_p_value.is_some()
                    || fit.randomization_null_effect.is_some()
                    || !fit.randomization_statistics.is_empty()))
            || (query.uniform_unit_randomization
                && (fit.randomization_null_effect.is_some_and(|effect| {
                    !effect.is_finite() || effect != query.sharp_null_effect.unwrap_or(0.0)
                }) || (query.sharp_null_effect.is_some()
                    && fit.randomization_null_effect.is_none())
                    || !query.sharp_null_effect.unwrap_or(0.0).is_finite()
                    || fit.uncertainty
                        != "point_only_with_exact_unit_randomization_p_value_no_interval"
                    || fit.randomization_statistics.len() != donors.len() + 1
                    || fit.randomization_statistics.len() > 32
                    || fit
                        .randomization_statistics
                        .iter()
                        .map(|(unit, _)| unit.as_str())
                        .collect::<Vec<_>>()
                        != query
                            .units
                            .iter()
                            .map(String::as_str)
                            .collect::<std::collections::BTreeSet<_>>()
                            .into_iter()
                            .collect::<Vec<_>>()
                    || fit
                        .randomization_statistics
                        .iter()
                        .any(|(_, statistic)| !statistic.is_finite() || *statistic < 0.0)
                    || fit.randomization_p_value
                        != fit
                            .randomization_statistics
                            .iter()
                            .find(|(unit, _)| unit == &query.treated_unit)
                            .map(|(_, observed)| {
                                fit.randomization_statistics
                                    .iter()
                                    .filter(|(_, statistic)| statistic >= observed)
                                    .count() as f64
                                    / fit.randomization_statistics.len() as f64
                            })))
        {
            return Err(IoError::Convert(
                "invalid synthetic DiD weights, support, or fabricated interval".into(),
            ));
        }
    }
    validate_survival_graphless(result)?;
    if result.estimate.is_some() && !licenses_scalar_estimate(&result.identification.status) {
        return Err(IoError::Convert(
            "identification status does not license a scalar estimate".into(),
        ));
    }
    if result.standard_error.is_some_and(|se| !se.is_finite() || se < 0.0) {
        return Err(IoError::Convert(
            "analysis standard error must be finite and nonnegative".into(),
        ));
    }
    if result.standard_error.is_some() && result.estimate.is_none() {
        return Err(IoError::Convert(
            "analysis standard error requires the estimate it describes".into(),
        ));
    }
    if let Some(response) = &result.response {
        validate_response_result(response, variable_names.len())?;
    }
    if let Some(posterior) = &result.posterior_artifact {
        crate::decode_causal_posterior_bytes(posterior)?;
    }
    if let Some(grid) = &result.mediation_grid {
        validate_mediation_grid(grid, variable_names.len())?;
    }
    if let Some(unit_effects) = &result.unit_effects {
        validate_unit_effects(unit_effects)?;
    }
    validate_cate_and_learner_metrics(result)?;
    if let Some(structural) = &result.structural_response {
        validate_structural_response(structural, variable_names.len())?;
    }
    Ok(())
}

pub(crate) fn validate_interventional_distribution(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let query = match &result.query {
        CausalQueryWire::Distribution(query) => query,
        _ if result.interventional_distribution.is_some() => {
            return Err(IoError::Convert(
                "interventional distribution atoms require a distribution query".into(),
            ));
        }
        _ => return Ok(()),
    };
    let Some(distribution) = &result.interventional_distribution else {
        // Older artifacts remain readable. The independent contract consumer
        // reports the missing executable result payload as a dependency.
        return Ok(());
    };
    if distribution.atoms.is_empty() {
        return Err(IoError::Convert("interventional distribution has no atoms".into()));
    }
    let expected_outcomes = &query.outcomes;
    let expected_conditioning = &query.conditioning;
    let mut groups: Vec<(Vec<(u32, crate::ValueWire)>, f64)> = Vec::new();
    let mut numeric_mean = 0.0;
    for atom in &distribution.atoms {
        if !atom.probability.is_finite()
            || atom.probability < 0.0
            || atom.outcomes.iter().map(|(id, _)| *id).collect::<Vec<_>>() != *expected_outcomes
            || atom.conditioning.iter().map(|(id, _)| *id).collect::<Vec<_>>()
                != *expected_conditioning
        {
            return Err(IoError::Convert(
                "interventional distribution atom does not match its query or has invalid mass"
                    .into(),
            ));
        }
        if atom
            .outcomes
            .iter()
            .chain(atom.conditioning.iter())
            .any(|(_, value)| matches!(value, crate::ValueWire::Float64(v) if !v.is_finite()))
        {
            return Err(IoError::Convert(
                "interventional distribution atom contains a non-finite value".into(),
            ));
        }
        if expected_outcomes.len() == 1 {
            if let Some(value) = atom.outcomes[0].1.to_value().as_f64() {
                numeric_mean += value * atom.probability;
            } else {
                numeric_mean = f64::NAN;
            }
        } else {
            numeric_mean = f64::NAN;
        }
        if let Some((_, mass)) =
            groups.iter_mut().find(|(conditioning, _)| *conditioning == atom.conditioning)
        {
            *mass += atom.probability;
        } else {
            groups.push((atom.conditioning.clone(), atom.probability));
        }
        if distribution
            .atoms
            .iter()
            .filter(|other| {
                other.conditioning == atom.conditioning && other.outcomes == atom.outcomes
            })
            .count()
            != 1
        {
            return Err(IoError::Convert(
                "interventional distribution contains duplicate atoms".into(),
            ));
        }
    }
    if groups.iter().any(|(_, mass)| (mass - 1.0).abs() > 1e-9) {
        return Err(IoError::Convert(
            "interventional distribution atom masses must sum to one per conditioning assignment"
                .into(),
        ));
    }
    // The estimator publishes a scalar mean only for one numeric outcome at one
    // conditioning assignment; any other reported scalar has no atom meaning.
    if let Some(estimate) = result.estimate {
        let defines_mean =
            groups.len() == 1 && expected_outcomes.len() == 1 && numeric_mean.is_finite();
        if !defines_mean {
            return Err(IoError::Convert(
                "interventional distribution atoms define no mean for the reported estimate".into(),
            ));
        }
        if (estimate - numeric_mean).abs() > 1e-9 * (1.0 + numeric_mean.abs()) {
            return Err(IoError::Convert(
                "interventional distribution atoms disagree with the reported mean".into(),
            ));
        }
    }
    Ok(())
}

fn validate_structural_response(
    structural: &StructuralResponseMixtureWire,
    n_variables: usize,
) -> Result<(), IoError> {
    if let Some(interval) = &structural.identified_set_interval {
        identified_set_interval_from_wire(interval)?;
    }
    // Each mass is a fraction in [0, 1] and together they sum to one, so an
    // over-unit identified mass cannot be balanced by a negative one.
    crate::contract_section::validate_mixture_masses_within(
        [
            structural.identified_mass,
            structural.unidentified_mass,
            structural.unevaluable_mass,
            structural.subsampled_out_mass,
        ],
        1e-9,
    )
    .map_err(|e| IoError::Convert(format!("structural {e}")))?;
    if !structural.atoms.is_empty() {
        let weight: f64 = structural.atoms.iter().map(|atom| atom.weight).sum();
        if !weight.is_finite() || (weight - 1.0).abs() > 1e-9 {
            return Err(IoError::Convert("inconsistent atom totals".into()));
        }
    }
    for atom in &structural.atoms {
        if !atom.weight.is_finite() || atom.weight < 0.0 {
            return Err(IoError::Convert(
                "structural atom weight must be finite and nonnegative".into(),
            ));
        }
        if let Some(value) = &atom.value {
            crate::response_value_from_wire(value)?;
        }
        if let Some(response) = &atom.response {
            crate::causal_artifact::validate_response_result(response, n_variables)?;
        }
        if let Some(posterior) = &atom.posterior_artifact {
            crate::decode_causal_posterior_bytes(posterior)?;
        }
    }
    Ok(())
}

/// A point scalar is licensed only when identification does not refuse the estimand.
/// `not_identified` never licenses one; partial / graph-dependent mixtures may still
/// publish a conditional or retained identified-mass summary.
fn licenses_scalar_estimate(status: &str) -> bool {
    status != "not_identified"
}

fn validate_mediation_grid(
    grid: &TemporalMediationGridWire,
    base_variable_count: usize,
) -> Result<(), IoError> {
    if grid.slices.is_empty() {
        return Err(IoError::Convert("mediation grid must contain at least one slice".into()));
    }
    let mut horizons = std::collections::BTreeSet::new();
    for slice in &grid.slices {
        if slice.horizon == 0 || !horizons.insert(slice.horizon) {
            return Err(IoError::Convert(
                "mediation grid horizons must be positive and unique".into(),
            ));
        }
        if slice.method.trim().is_empty() {
            return Err(IoError::Convert("mediation grid method must be non-blank".into()));
        }
        for node in &slice.adjustment {
            if node.variable as usize >= base_variable_count {
                return Err(IoError::Convert(
                    "mediation grid adjustment names an unknown base variable".into(),
                ));
            }
        }
        let optionals = [slice.total, slice.direct, slice.mediated];
        if !slice.effect.is_finite() || optionals.iter().any(|v| v.is_some_and(|x| !x.is_finite()))
        {
            return Err(IoError::Convert(
                "mediation grid effects must be finite when present".into(),
            ));
        }
        if let Some([lower, upper]) = slice.identified_set {
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(IoError::Convert(
                    "mediation grid identified set must be finite and ordered".into(),
                ));
            }
        }
        validate_mediation_uncertainty(&slice.uncertainty)?;
    }
    Ok(())
}

fn validate_mediation_uncertainty(
    uncertainty: &TemporalMediationUncertaintyWire,
) -> Result<(), IoError> {
    match uncertainty {
        TemporalMediationUncertaintyWire::FrequentistPointwise { standard_error } => {
            if standard_error.is_some_and(|se| !se.is_finite() || se < 0.0) {
                return Err(IoError::Convert(
                    "mediation grid standard error must be finite and nonnegative".into(),
                ));
            }
        }
        TemporalMediationUncertaintyWire::BayesianPointwise {
            requested,
            total,
            direct,
            mediated,
            n_draws,
            backend,
        } => {
            for summary in [requested, total, direct, mediated] {
                if ![summary.mean, summary.standard_deviation, summary.q025, summary.q975]
                    .iter()
                    .all(|v| v.is_finite())
                    || summary.standard_deviation < 0.0
                    || summary.q025 > summary.q975
                {
                    return Err(IoError::Convert(
                        "mediation grid posterior summary must be finite and ordered".into(),
                    ));
                }
            }
            if *n_draws == 0 || backend.trim().is_empty() {
                return Err(IoError::Convert(
                    "mediation grid Bayesian uncertainty needs draws and a backend".into(),
                ));
            }
        }
        TemporalMediationUncertaintyWire::Unavailable => {}
    }
    Ok(())
}

fn validate_unit_effects(unit_effects: &UnitEffectsWire) -> Result<(), IoError> {
    if unit_effects.effects.is_empty()
        || unit_effects.effects.iter().any(|effect| !effect.is_finite())
    {
        return Err(IoError::Convert("unit effects must be a non-empty finite vector".into()));
    }
    let n = unit_effects.effects.len();
    if let Some(intervals) = &unit_effects.intervals {
        if intervals.lower.len() != n
            || intervals.upper.len() != n
            || intervals.lower.iter().any(|v| !v.is_finite())
            || intervals.upper.iter().any(|v| !v.is_finite())
            || intervals.lower.iter().zip(&intervals.upper).any(|(lo, hi)| lo > hi)
            || !(0.0..=1.0).contains(&intervals.level)
            || !intervals.level.is_finite()
            || intervals.method.trim().is_empty()
        {
            return Err(IoError::Convert(
                "unit effect intervals must match effects length with finite ordered bounds, \
                 level in [0, 1], and a non-blank method"
                    .into(),
            ));
        }
    }
    if unit_effects.extrapolative.as_ref().is_some_and(|flags| flags.len() != n) {
        return Err(IoError::Convert(
            "unit effect extrapolative flags must match effects length".into(),
        ));
    }
    Ok(())
}

fn validate_cate_and_learner_metrics(result: &AnalysisResultWire) -> Result<(), IoError> {
    if result.cate.is_none() && (result.cate_se.is_some() || result.cate_leaf_dispersion.is_some())
    {
        return Err(IoError::Convert(
            "cate standard errors and leaf dispersion require cate point predictions".into(),
        ));
    }
    if let (Some(cate), Some(dispersion)) = (&result.cate, &result.cate_leaf_dispersion) {
        if dispersion.len() != cate.len() || dispersion.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(IoError::Convert(
                "cate leaf dispersion must match cate length and be finite nonnegative".into(),
            ));
        }
    }
    match (&result.cate, &result.cate_se) {
        (None, _) => {}
        (Some(cate), cate_se) => {
            if cate.is_empty() || cate.iter().any(|v| !v.is_finite()) {
                return Err(IoError::Convert(
                    "cate predictions must be a non-empty finite vector".into(),
                ));
            }
            if let Some(se) = cate_se {
                if se.len() != cate.len() || se.iter().any(|v| !v.is_finite() || *v < 0.0) {
                    return Err(IoError::Convert(
                        "cate standard errors must match cate length and be finite nonnegative"
                            .into(),
                    ));
                }
            }
        }
    }
    if result.outcome_oof_r2.is_some_and(|v| !v.is_finite()) {
        return Err(IoError::Convert("outcome_oof_r2 must be finite when present".into()));
    }
    if result.treatment_oof_logloss.is_some_and(|v| !v.is_finite() || v < 0.0) {
        return Err(IoError::Convert(
            "treatment_oof_logloss must be finite and nonnegative".into(),
        ));
    }
    if result.crossfit_folds.is_some_and(|folds| folds < 2) {
        return Err(IoError::Convert("crossfit_folds must be at least two when present".into()));
    }
    if result.learner_provenance.iter().any(|(spec, implementation, version)| {
        spec.trim().is_empty() || implementation.trim().is_empty() || version.trim().is_empty()
    }) {
        return Err(IoError::Convert("learner provenance entries must be non-blank".into()));
    }
    Ok(())
}

fn validate_identification_namespace(
    identification: &IdentificationResultWire,
    count: usize,
) -> Result<(), IoError> {
    crate::causal_artifact::validate_query_ids(&identification.query, count)?;
    let ids = identification
        .arena
        .var_sets
        .iter()
        .flatten()
        .copied()
        .chain(identification.arena.interventions.iter().flatten().map(|value| value.variable))
        .chain(
            identification
                .estimands
                .iter()
                .flat_map(|estimand| {
                    estimand
                        .adjustment_set
                        .iter()
                        .chain(&estimand.instruments)
                        .chain(&estimand.mediators)
                })
                .copied(),
        )
        .chain(
            identification
                .estimands
                .iter()
                .filter_map(|estimand| estimand.rd_design.as_ref())
                .map(|design| design.running_variable),
        )
        .chain(identification.hedge.iter().flat_map(crate::HedgeCertificateWire::variables));
    if ids.into_iter().any(|id| id as usize >= count) {
        return Err(IoError::Convert(
            "identification variable is outside its declared namespace".into(),
        ));
    }
    Ok(())
}

fn validate_temporal_namespace(
    variables: &[crate::HorizonAdjustmentNodeWire],
    base_count: usize,
) -> Result<(), IoError> {
    let mut keys = std::collections::BTreeSet::new();
    if variables.is_empty() {
        return Err(IoError::Convert("empty temporal variable namespace".into()));
    }
    for node in variables {
        if node.variable as usize >= base_count || !keys.insert((node.variable, node.offset)) {
            return Err(IoError::Convert(
                "temporal namespace contains an unknown base variable or duplicate node".into(),
            ));
        }
    }
    Ok(())
}
