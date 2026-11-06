//! Graphless artifact checks for this family of results.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AnalysisResultWire, GraphlessAssignmentSupport, GraphlessSupportKey, INTERVAL_BOUND_RELATIVE,
    IoError, REPLAY_ABSOLUTE, REPLAY_ABSOLUTE_LOOSE, REPLAY_PAIR, graphless_admits,
    graphless_status_ok, interval_bound_replay,
};

// allow(too_many_lines): one family's full artifact validation is a single contract read top to bottom.
#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
pub(in crate::analysis_result_artifact) fn validate_longitudinal_graphless(
    result: &AnalysisResultWire,
    allow_legacy_graphless_missing: bool,
) -> Result<(), IoError> {
    let Some(regime) = &result.longitudinal_regime else {
        return Ok(());
    };
    let crate::CausalQueryWire::LongitudinalRegime(query) = &result.query else {
        return Err(IoError::Convert(
            "longitudinal regime result is attached to a different query".into(),
        ));
    };
    let eligible_ipw = query.method == "ipw"
        && query.probabilities_known_by_design
        && query.subject_ids.len() >= 500
        && regime.matched_observed_fraction * query.subject_ids.len() as f64 >= 50.0
        && regime.effective_sample_size >= 50.0
        && regime.value_standard_error.is_some_and(|se| se.is_finite() && se > 0.0);
    let eligible_msm = query.method == "marginal_structural_model"
        && query.probabilities_known_by_design
        && query.subject_ids.len() >= 300
        && regime.observed_subjects >= 200
        && regime.effective_sample_size >= 150.0
        && regime.value_standard_error.is_some_and(|se| se.is_finite() && se > 0.0)
        && regime.standard_errors.len() == query.periods
        && regime.standard_errors.iter().all(|se| se.is_finite() && *se > 0.0);
    let n = query.subject_ids.len();
    let dr_cells = n.checked_mul(query.periods);
    let dr_matching_subjects = if matches!(query.periods, 2 | 3)
        && dr_cells.is_some_and(|cells| {
            query.treatment_history.len() == cells && query.regime_actions.len() == cells
        })
        && query.outcome_observed.len() == n
    {
        (0..n)
            .filter(|&i| {
                query.outcome_observed[i]
                    && (0..query.periods).all(|t| {
                        let j = query.periods * i + t;
                        query.treatment_history[j] == query.regime_actions[j]
                    })
            })
            .count()
    } else {
        0
    };
    let dr_min_action = query
        .regime_actions
        .iter()
        .zip(&query.treatment_probabilities)
        .map(|(action, p)| if *action { *p } else { 1.0 - p })
        .fold(1.0_f64, f64::min);
    let dr_min_censor = query.censoring_probabilities.iter().copied().fold(1.0_f64, f64::min);
    let eligible_dr = query.method == "sequential_dr"
        && query.probabilities_known_by_design
        && query.excluded_fold_predictions
        && query.prediction_fold_ids == query.fold_ids
        && (query.periods == 2
            && n >= 300
            && query.outcome_observed.iter().filter(|&&observed| observed).count() >= 200
            && dr_matching_subjects >= 50
            && dr_min_action >= 0.4
            && dr_min_censor >= 0.85
            || query.periods == 3
                && n >= 800
                && query.outcome_observed.iter().filter(|&&observed| observed).count() >= 600
                && dr_matching_subjects >= 60
                && dr_min_action >= 0.5
                && dr_min_censor >= 0.9)
        && dr_cells.is_some_and(|cells| query.q_predictions.len() == cells)
        && regime.value_standard_error.is_some_and(|se| se.is_finite() && se > 0.0);
    let g_formula_expected = if query.method == "g_formula"
        && query.known_fixed_outcome_predictions
        && query.probabilities_known_by_design
    {
        antecedent_stats::longitudinal_regime::evaluate_g_formula_value(
            &query.period_outcome_predictions,
            &query.regime_actions,
            &query.treatment_probabilities,
            &query.censoring_probabilities,
            n,
            query.periods,
            query.minimum_probability,
        )
        .ok()
        .and_then(|summary| {
            antecedent_stats::longitudinal_regime::g_formula_fixed_q_pointwise_interval_95(
                &summary,
                &query.period_outcome_predictions,
                n,
                query.periods,
            )
            .map(|interval| (summary.value, interval))
        })
    } else {
        None
    };
    let expected_reason = match query.method.as_str() {
        "ipw" if regime.value_interval_95.is_none() => {
            Some("insufficient_independent_subject_support_or_degenerate_score")
        }
        "g_formula" if !query.known_fixed_outcome_predictions => {
            Some("prediction_model_uncertainty_not_accounted")
        }
        "g_formula" if !query.probabilities_known_by_design => {
            Some("known_sequential_randomization_required_for_fixed_q_interval")
        }
        "g_formula" if g_formula_expected.is_none() => {
            Some("insufficient_two_period_subject_support_or_degenerate_fixed_q_score")
        }
        "sequential_dr" if regime.value_interval_95.is_none() => {
            Some("insufficient_calibrated_horizon_or_trajectory_support_for_sequential_dr_interval")
        }
        "marginal_structural_model" if regime.value_interval_95.is_none() => {
            Some("insufficient_independent_subject_support_for_msm_intervals")
        }
        _ => None,
    };
    let expected_uncertainty = if query.method == "marginal_structural_model" {
        if regime.value_interval_95.is_some() {
            "pointwise_subject_clustered_cr1_95"
        } else {
            "pointwise_subject_clustered_cr1_no_interval"
        }
    } else if query.method == "sequential_dr" && regime.value_interval_95.is_some() {
        "pointwise_subject_score_conditional_excluded_fold_q_95"
    } else if query.method == "g_formula" && regime.value_interval_95.is_some() {
        "pointwise_subject_score_conditional_fixed_known_q_95"
    } else if regime.value_interval_95.is_some() {
        "pointwise_subject_score_95"
    } else {
        "point_only_no_interval"
    };
    // The Bayesian sequential regime result carries its posterior and reports no interval.
    let bayesian_regime = result.posterior_artifact.is_some() && query.method == "sequential_dr";
    let expected_reason = if bayesian_regime {
        Some("bayesian_posterior_reports_no_interval")
    } else {
        expected_reason
    };
    let expected_uncertainty = if bayesian_regime {
        "bayesian_bootstrap_posterior_known_sequential_randomization_no_interval"
    } else {
        expected_uncertainty
    };
    #[allow(
        clippy::float_cmp,
        reason = "wire interval bounds must equal the recomputed values exactly; drift is a real mismatch"
    )]
    let interval_valid = match regime.value_interval_95 {
        Some(bounds) => {
            let se = regime.value_standard_error.unwrap_or(f64::NAN);
            let span = antecedent_stats::normal_ppf(0.975) * se;
            let tolerance = interval_bound_replay(INTERVAL_BOUND_RELATIVE, regime.value, span);
            (eligible_ipw || eligible_msm || eligible_dr || g_formula_expected.is_some())
                && g_formula_expected.is_none_or(|(value, (expected_se, expected_bounds))| {
                    REPLAY_ABSOLUTE.close(regime.value, value)
                        && REPLAY_ABSOLUTE.close(se, expected_se)
                        && bounds == expected_bounds
                })
                && bounds[0].is_finite()
                && bounds[1].is_finite()
                && tolerance.close(bounds[0], regime.value - span)
                && tolerance.close(bounds[1], regime.value + span)
        }
        None => true,
    };
    let period_intervals_valid = if regime.period_intervals_95.is_empty() {
        regime.value_interval_95.is_none() || query.method != "marginal_structural_model"
    } else {
        eligible_msm
            && regime.value_interval_95.is_some()
            && regime.period_intervals_95.len() == query.periods
            && regime
                .period_intervals_95
                .iter()
                .zip(&regime.period_effects)
                .zip(&regime.standard_errors)
                .all(|((bounds, center), se)| {
                    let span = antecedent_stats::normal_ppf(0.975) * se;
                    let tolerance = interval_bound_replay(INTERVAL_BOUND_RELATIVE, *center, span);
                    bounds[0].is_finite()
                        && bounds[1].is_finite()
                        && tolerance.close(bounds[0], center - span)
                        && tolerance.close(bounds[1], center + span)
                })
    };
    // Support evidence is read from the same wire fields the executor used to
    // populate `GraphlessAssignmentSupport`, so every row threshold applies.
    let regime_support = GraphlessAssignmentSupport {
        assignment_unit: "unit",
        rows: n,
        interval_95_published: regime.value_interval_95.is_some(),
        reported_intervals: 1,
        min_probability: regime.minimum_action_probability,
        disjoint_nuisance_training: query.excluded_fold_predictions
            && query.prediction_fold_ids == query.fold_ids,
        ..GraphlessAssignmentSupport::default()
    };
    let has_support =
        |design: &str, method: &str, claim: &str, support: GraphlessAssignmentSupport| {
            graphless_admits(
                GraphlessSupportKey {
                    family: "longitudinal_regime",
                    design,
                    method,
                    inference_claim: claim,
                },
                support,
            )
        };
    let graphless_dr_supported = query.method == "sequential_dr"
        && matches!(query.periods, 2 | 3)
        && eligible_dr
        && regime.value_interval_95.is_some()
        && has_support(
            if query.periods == 2 {
                "known_sequential_randomized_two_period"
            } else {
                "known_sequential_randomized_three_period"
            },
            "subject_excluded_q_sequential_dr_scores",
            "conditional_q_pointwise_95_normal_interval",
            regime_support,
        );
    // Two-period fixed-known-Q g-formula: io recomputes the calibrated
    // interval in `g_formula_expected`, so the support requires that exact
    // interval to be published against the two-period fixed-Q row.
    let graphless_g_formula_supported = query.method == "g_formula"
        && query.periods == 2
        && g_formula_expected.is_some()
        && regime.value_interval_95.is_some()
        && has_support(
            "known_sequential_randomized_two_period",
            "fixed_known_q_g_formula_scores",
            "conditional_q_pointwise_95_normal_interval",
            regime_support,
        );
    // Additive MSM: io establishes support through `eligible_msm` and the
    // published intercept and per-period coefficient intervals, matched to
    // the additive-MSM row for the intercept-and-period-effects claim.
    let graphless_msm_supported = query.method == "marginal_structural_model"
        && eligible_msm
        && regime.value_interval_95.is_some()
        && regime.period_intervals_95.len() == query.periods
        && has_support(
            "known_sequential_randomized_additive_msm",
            "stabilized_ipw_cr1_scores",
            "intercept_and_period_effects_pointwise_95_normal_intervals",
            GraphlessAssignmentSupport {
                reported_intervals: regime.period_intervals_95.len() + 1,
                all_reported_intervals: regime.period_intervals_95.len() == query.periods,
                min_effective_sample_size: regime.effective_sample_size,
                ..regime_support
            },
        );
    let graphless_supported =
        graphless_dr_supported || graphless_g_formula_supported || graphless_msm_supported;
    if !graphless_status_ok(
        regime.graphless_support_status.as_deref(),
        graphless_supported,
        allow_legacy_graphless_missing,
    ) {
        return Err(IoError::Convert(
            "longitudinal regime graphless support does not match evidenced support".into(),
        ));
    }
    if regime.method != query.method
        || regime.rule_id != query.rule_id
        || regime.rule_version != query.rule_version
        || regime.rule_provenance != query.rule_provenance
        || result.estimate.is_some()
        || result.standard_error.is_some()
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
        || !regime.value.is_finite()
        || regime.value_standard_error.is_some_and(|se| {
            !se.is_finite()
                || se < 0.0
                || !(matches!(query.method.as_str(), "ipw" | "marginal_structural_model")
                    || query.method == "g_formula" && g_formula_expected.is_some()
                    || query.method == "sequential_dr" && regime.value_interval_95.is_some())
        })
        || !interval_valid
        || !period_intervals_valid
        || query.method == "sequential_dr" && regime.interval_reason.as_deref() != expected_reason
        || regime.interval_reason.as_deref().is_some_and(|reason| Some(reason) != expected_reason)
        || regime.value_interval_95.is_some() && regime.interval_reason.is_some()
        || !regime.effective_sample_size.is_finite()
        || regime.effective_sample_size <= 0.0
        || regime.effective_sample_size > query.subject_ids.len() as f64
        || !regime.matched_observed_fraction.is_finite()
        || !(0.0 < regime.matched_observed_fraction && regime.matched_observed_fraction <= 1.0)
        || !regime.maximum_weight.is_finite()
        || regime.maximum_weight <= 0.0
        || !regime.minimum_action_probability.is_finite()
        || regime.minimum_action_probability < query.minimum_probability
        || !regime.minimum_censoring_probability.is_finite()
        || regime.minimum_censoring_probability < query.minimum_probability
        || regime.uncertainty != expected_uncertainty
        || regime.probability_ownership
            != if query.probabilities_known_by_design {
                "known_sequential_randomization"
            } else {
                "caller_declared_subject_excluded_fold_predictions"
            }
    {
        return Err(IoError::Convert(
            "invalid longitudinal regime payload or fabricated interval".into(),
        ));
    }
    if query.method == "marginal_structural_model" {
        if regime.period_effects.len() != query.periods
            || regime.standard_errors.len() != query.periods
            || regime.stabilizing_numerator_probabilities
                != query.stabilizing_numerator_probabilities
            || regime.observed_subjects <= query.periods + 1
            || regime.observed_subjects > query.subject_ids.len()
            || regime.period_effects.iter().any(|v| !v.is_finite())
            || regime.standard_errors.iter().any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(IoError::Convert(
                "invalid longitudinal MSM coefficients or uncertainty".into(),
            ));
        }
    } else if !regime.period_effects.is_empty()
        || !regime.standard_errors.is_empty()
        || !regime.stabilizing_numerator_probabilities.is_empty()
        || regime.observed_subjects != 0
    {
        return Err(IoError::Convert("MSM-only result fields attached to a regime value".into()));
    }
    Ok(())
}

// A flat sequence of independent field checks; splitting would scatter the contract.
#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
pub(in crate::analysis_result_artifact) fn validate_continuous_dose_graphless(
    result: &AnalysisResultWire,
    allow_legacy_graphless_missing: bool,
) -> Result<(), IoError> {
    let Some(fit) = &result.continuous_dose_response else {
        return Ok(());
    };
    let crate::CausalQueryWire::ContinuousDoseResponse(query) = &result.query else {
        return Err(IoError::Convert(
            "continuous-dose result is attached to a different query".into(),
        ));
    };
    let groups: std::collections::BTreeSet<&str> =
        query.baseline_groups.iter().map(String::as_str).collect();
    let policy_requested = query.fixed_policy.is_some();
    let policy_interval =
        fit.fixed_policy.as_ref().is_some_and(|value| value.incremental_interval_95.is_some());
    let expected_uncertainty = if policy_interval {
        "fixed_group_kernel_smoothed_paired_pointwise_95_normal_intervals"
    } else if policy_requested {
        "fixed_group_kernel_smoothed_paired_variance_no_interval"
    } else {
        "point_only_no_interval"
    };
    #[allow(
        clippy::float_cmp,
        reason = "wire fields must equal the frozen query bytes exactly; any drift is a real mismatch"
    )]
    if result.estimate.is_some()
        || result.standard_error.is_some()
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
        || fit.bandwidth != query.bandwidth
        || fit.density_provenance != query.density_provenance
        || fit.uncertainty != expected_uncertainty
        || fit.fixed_policy.is_some() != policy_requested
        || fit.points.len() != groups.len() * query.target_doses.len()
        || (!policy_requested && fit.graphless_support_status.is_some())
    {
        return Err(IoError::Convert(
            "continuous-dose result must match its query and uncertainty".into(),
        ));
    }
    for (index, point) in fit.points.iter().enumerate() {
        let group = groups.iter().nth(index / query.target_doses.len()).copied().unwrap_or("");
        let target = query.target_doses[index % query.target_doses.len()];
        #[allow(
            clippy::float_cmp,
            reason = "target dose must equal the frozen query grid value exactly"
        )]
        if point.baseline_group != group
            || point.target_dose != target
            || point.local_rows < query.min_local_support
            || ![
                point.response,
                point.effective_sample_size,
                point.minimum_dose_density,
                point.maximum_normalized_weight,
                point.local_outcome_sd,
            ]
            .iter()
            .all(|value| value.is_finite())
            || point.effective_sample_size <= 0.0
            || point.minimum_dose_density <= 0.0
            || !(0.0..=1.0).contains(&point.maximum_normalized_weight)
            || point.local_outcome_sd < 0.0
        {
            return Err(IoError::Convert(
                "continuous-dose point or local support is invalid".into(),
            ));
        }
    }
    validate_fixed_dose_policy(
        fit,
        query,
        &groups,
        policy_interval,
        allow_legacy_graphless_missing,
    )?;
    Ok(())
}

/// Verify the optional fixed-dose policy, paired variances and support together.
fn validate_fixed_dose_policy(
    fit: &crate::analysis_result_artifact::ContinuousDoseResponseWire,
    query: &crate::query_wire::ContinuousDoseResponseQueryWire,
    groups: &std::collections::BTreeSet<&str>,
    policy_interval: bool,
    allow_legacy_graphless_missing: bool,
) -> Result<(), IoError> {
    if let Some(value) = &fit.fixed_policy {
        let Some((policy_doses, reference_doses)) = &query.fixed_policy else {
            return Err(IoError::Convert("continuous-dose policy maps are missing".into()));
        };
        let mut expected_policy = policy_doses.clone();
        let mut expected_reference = reference_doses.clone();
        expected_policy.sort_by(|a, b| a.0.cmp(&b.0));
        expected_reference.sort_by(|a, b| a.0.cmp(&b.0));
        let minimum_group_rows = groups
            .iter()
            .map(|group| {
                query.baseline_groups.iter().filter(|observed| observed.as_str() == *group).count()
            })
            .min()
            .unwrap_or(0);
        let intervals =
            [value.policy_interval_95, value.reference_interval_95, value.incremental_interval_95];
        let variances =
            [value.policy_variance, value.reference_variance, value.incremental_variance];
        let centers = [value.policy_value, value.reference_value, value.incremental_value];
        // Publication is judged separately by `policy_interval != supported`
        // below, so the evidence here is the support itself: three intervals
        // are claimed and verified against the recomputed variances.
        let supported = variances.iter().all(|variance| *variance > 0.0)
            && graphless_admits(
                GraphlessSupportKey {
                    family: "continuous_dose_policy",
                    design: "fixed_group_kernel",
                    method: "inverse_density_kernel_paired_scores",
                    inference_claim: "policy_reference_incremental_pointwise_95_normal_intervals",
                },
                GraphlessAssignmentSupport {
                    assignment_unit: "unit",
                    interval_95_published: true,
                    rows: query.baseline_groups.len(),
                    min_group_rows: minimum_group_rows,
                    min_local_rows: value.minimum_local_rows,
                    min_effective_sample_size: value.minimum_effective_sample_size,
                    max_normalized_weight: value.maximum_normalized_weight,
                    min_dose_density: value.minimum_dose_density,
                    known_density: query.density_provenance == "known",
                    reported_intervals: 3,
                    all_reported_intervals: true,
                    ..GraphlessAssignmentSupport::default()
                },
            );
        if value.policy_doses != expected_policy
            || value.reference_doses != expected_reference
            || !centers.iter().chain(&variances).all(|number| number.is_finite())
            || !REPLAY_ABSOLUTE_LOOSE
                .close(value.policy_value - value.reference_value, value.incremental_value)
            || variances.iter().any(|variance| *variance < 0.0)
            || value.minimum_local_rows < query.min_local_support
            || !value.minimum_effective_sample_size.is_finite()
            || value.minimum_effective_sample_size <= 0.0
            || !value.maximum_normalized_weight.is_finite()
            || !(0.0..=1.0).contains(&value.maximum_normalized_weight)
            || !value.minimum_dose_density.is_finite()
            || value.minimum_dose_density <= 0.0
            || policy_interval != supported
            || !graphless_status_ok(
                fit.graphless_support_status.as_deref(),
                supported,
                allow_legacy_graphless_missing,
            )
            || intervals.iter().any(|interval| interval.is_some() != policy_interval)
            || intervals.iter().zip(centers).zip(variances).any(|((interval, center), variance)| {
                interval.is_some_and(|[lower, upper]| {
                    let radius = antecedent_stats::NORMAL_Q975 * variance.sqrt();
                    !lower.is_finite()
                        || !upper.is_finite()
                        || !REPLAY_ABSOLUTE_LOOSE.close(lower, center - radius)
                        || !REPLAY_ABSOLUTE_LOOSE.close(upper, center + radius)
                })
            })
        {
            return Err(IoError::Convert(
                "continuous-dose policy value or paired interval is invalid".into(),
            ));
        }
    }
    Ok(())
}

/// Check the selected finite policy class and replay its regret bounds.
fn validate_policy_regret(
    policy: &crate::PolicyValueWire,
    query: &crate::query_wire::PolicyValueQueryWire,
) -> Result<(), IoError> {
    if let Some(regret) = &policy.regret {
        let Some(design) = &query.regret else {
            return Err(IoError::Convert(
                "finite-class regret requires its bound candidate query".into(),
            ));
        };
        let k = design.candidates.len();
        let n = query.assignment.len();
        if n < 400
            || !(2..=16).contains(&k)
            || design.selected_index >= k
            || regret.selected_index != design.selected_index
            || regret.candidate_values.len() != k
            || regret.contrast_standard_errors.len() != k
            || query.global_constraints_present
            || query.multi_action.is_some()
            || !query.mu0.is_empty()
            || !query.uplift_bins.is_empty()
            || query.propensity.iter().any(|p| !(0.2..=0.8).contains(p))
            || design.training_subject_ids.is_empty()
            || design
                .training_subject_ids
                .iter()
                .any(|id| query.evaluation_subject_ids.contains(id))
            || design.candidates.iter().any(|candidate| {
                candidate.len() != n
                    || query.assignment.iter().zip(candidate).filter(|(a, b)| a == b).count() < 50
            })
            || design.candidates[design.selected_index] != query.actions
        {
            return Err(IoError::Convert(
                "finite-class regret support or ownership is invalid".into(),
            ));
        }
        let selected = regret.candidate_values[regret.selected_index];
        let critical = antecedent_stats::normal_ppf(1.0 - 0.05 / (2.0 * (k - 1) as f64));
        let mut lower: f64 = 0.0;
        let mut upper: f64 = 0.0;
        let mut point: f64 = 0.0;
        for (j, (&value, &se)) in
            regret.candidate_values.iter().zip(&regret.contrast_standard_errors).enumerate()
        {
            if !value.is_finite()
                || !se.is_finite()
                || se < 0.0
                || (j == regret.selected_index && se != 0.0)
            {
                return Err(IoError::Convert(
                    "finite-class regret candidate score is invalid".into(),
                ));
            }
            if j == regret.selected_index {
                continue;
            }
            let difference = value - selected;
            let span = critical * se;
            point = point.max(difference);
            lower = lower.max(difference - span);
            upper = upper.max(difference + span);
        }
        let close = |a: f64, b: f64| REPLAY_PAIR.close(a, b);
        if !close(selected, policy.policy_value)
            || !close(point, regret.regret)
            || !close(lower, regret.interval_95[0])
            || !close(upper, regret.interval_95[1])
        {
            return Err(IoError::Convert(
                "finite-class regret interval does not match paired candidate contrasts".into(),
            ));
        }
    } else if query.regret.is_some() {
        return Err(IoError::Convert("finite-class regret query requires a regret result".into()));
    }
    Ok(())
}

/// Check uplift-bin membership, support and replayed interval bounds.
fn validate_policy_uplift_bins(
    policy: &crate::PolicyValueWire,
    query: &crate::query_wire::PolicyValueQueryWire,
    z: f64,
) -> Result<(), IoError> {
    if policy.uplift_bins.len() != query.uplift_bin_count
        || (!query.uplift_bins.is_empty() && query.uplift_bins.len() != query.assignment.len())
        || policy.uplift_bins.iter().enumerate().any(|(rank, bin)| {
            let rows = query
                .uplift_bins
                .iter()
                .enumerate()
                .filter(|&(_, &group)| group == rank)
                .collect::<Vec<_>>();
            let treated = rows.iter().filter(|&&(i, _)| query.assignment[i]).count();
            let controls = rows.len() - treated;
            let interval_valid = bin.interval_95.is_none_or(|bounds| {
                let span = z * bin.standard_error;
                let tolerance = interval_bound_replay(INTERVAL_BOUND_RELATIVE, bin.effect, span);
                rows.len() >= 300
                    && treated >= 50
                    && controls >= 50
                    && !query.uplift_training_subject_ids.is_empty()
                    && query
                        .uplift_training_subject_ids
                        .iter()
                        .all(|id| !query.evaluation_subject_ids.contains(id))
                    && bin.standard_error > 0.0
                    && bounds.iter().all(|value| value.is_finite())
                    && tolerance.close(bounds[0], bin.effect - span)
                    && tolerance.close(bounds[1], bin.effect + span)
            });
            bin.rank != rank
                || bin.evaluation_rows != rows.len()
                || !bin.effect.is_finite()
                || !bin.standard_error.is_finite()
                || bin.standard_error < 0.0
                || !interval_valid
        })
    {
        return Err(IoError::Convert("invalid policy uplift-bin support or interval".into()));
    }
    Ok(())
}

/// Check multi-action group membership, support and replayed interval bounds.
fn validate_policy_multi_action(
    policy: &crate::PolicyValueWire,
    query: &crate::query_wire::PolicyValueQueryWire,
    z: f64,
) -> Result<(), IoError> {
    if let Some(multi) = &query.multi_action {
        let k = multi.action_labels.len();
        if k < 2 {
            return Err(IoError::Convert(
                "multi-action CATE requires a control and an action".into(),
            ));
        }
        let mut groups = multi.cate_groups.clone();
        groups.sort_unstable();
        groups.dedup();
        if policy.multi_action_cate.len() != groups.len() * k.saturating_sub(1)
            || policy.multi_action_cate.iter().enumerate().any(|(index, point)| {
                let group = &groups[index / (k - 1)];
                let action = index % (k - 1) + 1;
                let rows = multi
                    .cate_groups
                    .iter()
                    .enumerate()
                    .filter(|(_, label)| *label == group)
                    .map(|(row, _)| row)
                    .collect::<Vec<_>>();
                let observed_action =
                    rows.iter().filter(|&&row| multi.assignment[row] == action).count();
                let observed_control =
                    rows.iter().filter(|&&row| multi.assignment[row] == 0).count();
                let strong_overlap = rows.iter().all(|&row| {
                    multi.propensities[row * k] >= 0.2
                        && multi.propensities[row * k + action] >= 0.2
                });
                let interval_valid = point.interval_95.is_none_or(|bounds| {
                    let span = z * point.standard_error;
                    let tolerance =
                        interval_bound_replay(INTERVAL_BOUND_RELATIVE, point.effect, span);
                    rows.len() >= 300
                        && observed_action >= 50
                        && observed_control >= 50
                        && strong_overlap
                        && point.standard_error > 0.0
                        && bounds.iter().all(|value| value.is_finite())
                        && tolerance.close(bounds[0], point.effect - span)
                        && tolerance.close(bounds[1], point.effect + span)
                });
                point.group != group.as_str()
                    || point.action != multi.action_labels[action].as_str()
                    || point.evaluation_rows != rows.len()
                    || point.observed_action_rows != observed_action
                    || point.observed_control_rows != observed_control
                    || !point.effect.is_finite()
                    || !point.standard_error.is_finite()
                    || point.standard_error < 0.0
                    || !interval_valid
            })
        {
            return Err(IoError::Convert("invalid multi-action CATE support or interval".into()));
        }
    } else if !policy.multi_action_cate.is_empty() {
        return Err(IoError::Convert("multi-action CATE requires a multi-action query".into()));
    }
    Ok(())
}

// allow(too_many_lines): one family's full artifact validation is a single contract read top to bottom.
#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
pub(in crate::analysis_result_artifact) fn validate_policy_graphless(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let Some(policy) = &result.policy_value else {
        return Ok(());
    };
    let expected_uncertainty = match &result.query {
        crate::CausalQueryWire::PolicyValue(_) if result.posterior_artifact.is_some() => {
            "bayesian_bootstrap_posterior_known_randomization_no_interval"
        }
        crate::CausalQueryWire::PolicyValue(query) if query.multi_action.is_some() => {
            "multi_action_ipw_row_score_standard_error_independent_subjects"
        }
        crate::CausalQueryWire::PolicyValue(query)
            if query.mu0.is_empty() && query.mu1.is_empty() =>
        {
            "ipw_row_score_standard_error_independent_subjects"
        }
        crate::CausalQueryWire::PolicyValue(_) => "row_score_standard_error_independent_subjects",
        _ => "invalid_policy_value_query",
    };
    let expected_interval = match &result.query {
        crate::CausalQueryWire::PolicyValue(query) => {
            let (n, policy_matches, reference_matches) = if let Some(multi) = &query.multi_action {
                (
                    multi.assignment.len(),
                    multi.assignment.iter().zip(&multi.actions).filter(|(a, b)| a == b).count(),
                    multi.assignment.iter().zip(&multi.reference).filter(|(a, b)| a == b).count(),
                )
            } else {
                (
                    query.assignment.len(),
                    query.assignment.iter().zip(&query.actions).filter(|(a, b)| a == b).count(),
                    query.assignment.iter().zip(&query.reference).filter(|(a, b)| a == b).count(),
                )
            };
            let multi_constraints_couple = query.multi_action.as_ref().is_some_and(|multi| {
                let n = multi.assignment.len();
                [
                    (&multi.capacities, &multi.costs, multi.budget),
                    (&multi.reference_capacities, &multi.reference_costs, multi.reference_budget),
                ]
                .into_iter()
                .any(|(capacities, costs, budget)| {
                    capacities.iter().any(|&limit| limit < n)
                        || budget.is_some_and(|limit| {
                            let maximum_cost = costs.iter().copied().fold(0.0_f64, f64::max);
                            limit + 1e-12 < n as f64 * maximum_cost
                        })
                })
            });
            n >= (if query.multi_action.is_some() || !query.mu0.is_empty() { 300 } else { 120 })
                && policy_matches >= 10
                && reference_matches >= 10
                && !query.global_constraints_present
                && !multi_constraints_couple
                && (query.multi_action.is_some()
                    || query.mu0.is_empty()
                    || query.disjoint_training_subjects
                    || query.crossfit_fold_ownership_valid)
                && policy.policy_standard_error > 0.0
                && policy.incremental_standard_error > 0.0
        }
        _ => false,
    };
    let z = antecedent_stats::normal_ppf(0.975);
    let bounds_match = |bounds: Option<[f64; 2]>, center: f64, se: f64| {
        let Some(bounds) = bounds else {
            return true;
        };
        let span = z * se;
        let tolerance = interval_bound_replay(INTERVAL_BOUND_RELATIVE, center, span);
        expected_interval
            && tolerance.close(bounds[0], center - span)
            && tolerance.close(bounds[1], center + span)
    };
    if result.estimate.is_some()
        || ![
            policy.policy_value,
            policy.reference_value,
            policy.incremental_value,
            policy.relative_value_gap,
            policy.treatment_rate,
            policy.total_cost,
            policy.policy_standard_error,
            policy.reference_standard_error,
            policy.incremental_standard_error,
            policy.propensity_min,
            policy.propensity_max,
        ]
        .iter()
        .all(|value| value.is_finite())
        || !(0.0..=1.0).contains(&policy.treatment_rate)
        || policy.total_cost < 0.0
        || policy.policy_standard_error < 0.0
        || policy.reference_standard_error < 0.0
        || policy.incremental_standard_error < 0.0
        || policy.policy_interval_95.is_some() != policy.incremental_interval_95.is_some()
        || policy
            .policy_interval_95
            .iter()
            .chain(policy.incremental_interval_95.iter())
            .any(|bounds| !bounds.iter().all(|value| value.is_finite()) || bounds[0] > bounds[1])
        || !bounds_match(
            policy.policy_interval_95,
            policy.policy_value,
            policy.policy_standard_error,
        )
        || !bounds_match(
            policy.incremental_interval_95,
            policy.incremental_value,
            policy.incremental_standard_error,
        )
        || !(0.0..=1.0).contains(&policy.propensity_min)
        || !(0.0..=1.0).contains(&policy.propensity_max)
        || policy.propensity_min > policy.propensity_max
        || policy.prediction_ownership.is_empty()
        || policy.uncertainty != expected_uncertainty
    {
        return Err(IoError::Convert(
            "invalid policy-value payload or fabricated scalar effect".into(),
        ));
    }
    let crate::CausalQueryWire::PolicyValue(query) = &result.query else {
        return Err(IoError::Convert("policy uplift bins require a policy-value query".into()));
    };
    validate_policy_regret(policy, query)?;
    validate_policy_uplift_bins(policy, query, z)?;
    validate_policy_multi_action(policy, query, z)?;
    let (design, method, claim) = antecedent_core::policy_graphless_coordinate(
        query.multi_action.is_some(),
        query.mu0.is_empty(),
        !policy.uplift_bins.is_empty(),
        !policy.multi_action_cate.is_empty(),
        query.regret.is_some(),
        !query.mu0.is_empty()
            && query.crossfit_fold_ownership_valid
            && !query.disjoint_training_subjects,
    );
    let (
        treated,
        control,
        min_action_rows,
        min_probability,
        policy_matches,
        reference_matches,
        uncoupled,
    ) = if let Some(multi) = &query.multi_action {
        let n = multi.assignment.len();
        let k = multi.action_labels.len();
        let treated = multi.assignment.iter().filter(|&&action| action != 0).count();
        let policy_matches =
            multi.assignment.iter().zip(&multi.actions).filter(|(a, b)| a == b).count();
        let reference_matches =
            multi.assignment.iter().zip(&multi.reference).filter(|(a, b)| a == b).count();
        let coupled = [
            (&multi.capacities, &multi.costs, multi.budget),
            (&multi.reference_capacities, &multi.reference_costs, multi.reference_budget),
        ]
        .into_iter()
        .any(|(capacities, costs, budget)| {
            capacities.iter().any(|&limit| limit < n)
                || budget.is_some_and(|limit| {
                    limit + 1e-12 < n as f64 * costs.iter().copied().fold(0.0_f64, f64::max)
                })
        });
        (
            treated,
            n - treated,
            (0..k)
                .map(|action| multi.assignment.iter().filter(|&&a| a == action).count())
                .min()
                .unwrap_or(0),
            multi.propensities.iter().copied().fold(f64::INFINITY, f64::min),
            policy_matches,
            reference_matches,
            !query.global_constraints_present && !coupled,
        )
    } else {
        let n = query.assignment.len();
        let treated = query.assignment.iter().filter(|&&assigned| assigned).count();
        (
            treated,
            n - treated,
            treated.min(n - treated),
            query.propensity.iter().copied().map(|p| p.min(1.0 - p)).fold(f64::INFINITY, f64::min),
            query.assignment.iter().zip(&query.actions).filter(|(a, b)| a == b).count(),
            query.assignment.iter().zip(&query.reference).filter(|(a, b)| a == b).count(),
            !query.global_constraints_present,
        )
    };
    let min_bin_rows = policy.uplift_bins.iter().map(|bin| bin.evaluation_rows).min().unwrap_or(0);
    let min_bin_arm_rows = if policy.uplift_bins.is_empty() {
        0
    } else {
        (0..policy.uplift_bins.len())
            .flat_map(|bin| {
                let treated = query
                    .uplift_bins
                    .iter()
                    .enumerate()
                    .filter(|&(row, &rank)| rank == bin && query.assignment[row])
                    .count();
                [treated, policy.uplift_bins[bin].evaluation_rows - treated]
            })
            .min()
            .unwrap_or(0)
    };
    let scalar_intervals = usize::from(policy.policy_interval_95.is_some())
        + usize::from(policy.incremental_interval_95.is_some());
    let reported_intervals = scalar_intervals
        + policy.uplift_bins.iter().filter(|bin| bin.interval_95.is_some()).count()
        + policy.multi_action_cate.iter().filter(|point| point.interval_95.is_some()).count();
    let all_reported_intervals = scalar_intervals == 2
        && policy.uplift_bins.iter().all(|bin| bin.interval_95.is_some())
        && policy.multi_action_cate.iter().all(|point| point.interval_95.is_some());
    let min_group_rows =
        policy.multi_action_cate.iter().map(|point| point.evaluation_rows).min().unwrap_or(0);
    let min_group_arm_rows = policy
        .multi_action_cate
        .iter()
        .flat_map(|point| [point.observed_action_rows, point.observed_control_rows])
        .min()
        .unwrap_or(0);
    let rank_ownership = !query.uplift_training_subject_ids.is_empty()
        && query
            .uplift_training_subject_ids
            .iter()
            .all(|id| !query.evaluation_subject_ids.contains(id));
    let supported = graphless_admits(
        GraphlessSupportKey { family: "policy_value", design, method, inference_claim: claim },
        GraphlessAssignmentSupport {
            assignment_unit: "unit",
            treated,
            control,
            interval_95_published: scalar_intervals == 2,
            rows: query.evaluation_subject_ids.len(),
            min_action_rows,
            min_probability,
            reported_intervals,
            all_reported_intervals,
            policy_matches,
            reference_matches,
            min_bin_rows,
            min_bin_arm_rows,
            min_group_rows,
            min_group_arm_rows,
            uncoupled_constraints: uncoupled,
            disjoint_nuisance_training: query.disjoint_training_subjects,
            rank_ownership,
            ..GraphlessAssignmentSupport::default()
        },
    );
    if !graphless_status_ok(policy.graphless_support_status.as_deref(), supported, true) {
        return Err(IoError::Convert(
            "policy graphless support status does not match exact design and interval evidence"
                .into(),
        ));
    }
    Ok(())
}
