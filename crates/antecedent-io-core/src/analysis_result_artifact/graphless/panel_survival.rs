//! Graphless artifact checks for this family of results.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AnalysisResultWire, GraphlessAssignmentSupport, GraphlessSupportKey, INTERVAL_BOUND_RELATIVE,
    INTERVAL_BOUND_RELATIVE_LOOSE, IoError, REPLAY_ABSOLUTE, ReplayNonFinite, ReplayScale,
    ReplayTolerance, SAMPLING_WEIGHTED_DID_UNCERTAINTY, graphless_admits, graphless_status_ok,
    interval_bound_replay,
};

/// Uncertainty label of the augmented panel's unsupported whole-cluster resample interval.
const AUGMENTED_RESAMPLE_UNCERTAINTY: &str =
    "whole_cluster_resample_supplied_nuisance_held_fixed_calibration_not_measured";

// allow(too_many_lines): one family's full artifact validation is a single contract read top to bottom.
#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
pub(in crate::analysis_result_artifact) fn validate_did_graphless(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let Some(did) = &result.panel_did else {
        return Ok(());
    };
    let crate::CausalQueryWire::PanelDid(query) = &result.query else {
        return Err(IoError::Convert(
            "panel DiD result section is attached to a different query".into(),
        ));
    };
    let mut subjects = std::collections::BTreeMap::<&str, (bool, &str)>::new();
    let mut group_clusters: [std::collections::BTreeSet<&str>; 2] = Default::default();
    let mut cell_clusters: [[std::collections::BTreeSet<&str>; 2]; 2] = Default::default();
    let mut duplicate_subject = false;
    for i in 0..query.treated.len() {
        let subject = query.subjects[i].as_str();
        let cluster = query.clusters[i].as_str();
        match subjects.insert(subject, (query.treated[i], cluster)) {
            Some((group, old_cluster)) if group != query.treated[i] || old_cluster != cluster => {
                return Err(IoError::Convert(
                    "panel DiD query changes treatment or cluster within a subject".into(),
                ));
            }
            Some(_) => duplicate_subject = true,
            None => {}
        }
        group_clusters[usize::from(query.treated[i])].insert(cluster);
        cell_clusters[usize::from(query.treated[i])][usize::from(query.post[i])].insert(cluster);
    }
    let representative = did.event_time_effects.iter().find(|effect| effect.2 >= 0);
    #[allow(
        clippy::float_cmp,
        reason = "representative effect/SE must equal the sealed event-time rows exactly"
    )]
    let event_study_valid = if query.staggered_event_study {
        let mut unit_metadata = std::collections::BTreeMap::<&str, (i64, &str)>::new();
        let mut observed_periods = std::collections::BTreeSet::new();
        for i in 0..query.subjects.len() {
            let subject = query.subjects[i].as_str();
            let metadata = (query.cohorts[i], query.clusters[i].as_str());
            if unit_metadata.insert(subject, metadata).is_some_and(|old| old != metadata) {
                return Err(IoError::Convert(
                    "event study changes cohort or cluster within subject".into(),
                ));
            }
            observed_periods.insert(query.periods[i]);
        }
        let adoption_cohorts: std::collections::BTreeSet<_> = unit_metadata
            .values()
            .map(|(cohort, _)| *cohort)
            .filter(|cohort| *cohort > 0)
            .collect();
        let expected_keys: std::collections::BTreeSet<_> = adoption_cohorts
            .iter()
            .flat_map(|cohort| {
                observed_periods
                    .iter()
                    .filter(move |period| **period != *cohort - 1)
                    .map(move |period| (*cohort, *period))
            })
            .collect();
        let actual_keys: std::collections::BTreeSet<_> =
            did.event_time_effects.iter().map(|effect| (effect.0, effect.1)).collect();
        let control_count = unit_metadata.values().filter(|(cohort, _)| *cohort == 0).count();
        !did.event_time_effects.is_empty()
            && actual_keys == expected_keys
            && actual_keys.len() == did.event_time_effects.len()
            && representative.is_some()
            && did.event_time_effects.iter().all(|effect| {
                let (cohort, period, event_time, point, treated, controls, se, clusters) = *effect;
                let expected_treated = unit_metadata.values().filter(|(g, _)| *g == cohort).count();
                let expected_clusters = unit_metadata
                    .values()
                    .filter(|(g, _)| *g == cohort || *g == 0)
                    .map(|(_, cluster)| *cluster)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len();
                cohort > 0
                    && period > 0
                    && event_time == period - cohort
                    && event_time != -1
                    && point.is_finite()
                    && se.is_finite()
                    && se >= 0.0
                    && treated == expected_treated
                    && controls == control_count
                    && clusters == expected_clusters
                    && treated >= 2
                    && controls >= 2
                    && clusters >= 4
            })
            && representative.is_some_and(|effect| {
                did.effect == effect.3 && did.standard_error == Some(effect.6)
            })
    } else {
        did.event_time_effects.is_empty()
    };
    let event_intervals_valid = if query.staggered_event_study {
        if did.event_time_intervals_95.is_empty() {
            did.interval_95.is_none()
                && did.uncertainty == "cluster_robust_standard_error_no_interval"
        } else if did.event_time_intervals_95.len() != did.event_time_effects.len() {
            false
        } else {
            let mut clusters_by_cohort =
                std::collections::BTreeMap::<i64, std::collections::BTreeSet<&str>>::new();
            for (cohort, cluster) in query.cohorts.iter().zip(&query.clusters) {
                clusters_by_cohort.entry(*cohort).or_default().insert(cluster);
            }
            let controls = clusters_by_cohort.get(&0).map_or(0, std::collections::BTreeSet::len);
            let valid = did.event_time_effects.iter().zip(&did.event_time_intervals_95).all(
                |(effect, interval)| {
                    let treated = clusters_by_cohort
                        .get(&effect.0)
                        .map_or(0, std::collections::BTreeSet::len);
                    let supported = effect.2 >= 0
                        && treated >= 24
                        && controls >= 24
                        && effect.7 >= 48
                        && effect.6.is_finite()
                        && effect.6 > 0.0;
                    match interval {
                        Some(bounds) if supported => {
                            let span = antecedent_stats::normal_ppf(0.975) * effect.6;
                            let tolerance =
                                interval_bound_replay(INTERVAL_BOUND_RELATIVE, effect.3, span);
                            bounds[0].is_finite()
                                && bounds[1].is_finite()
                                && tolerance.close(bounds[0], effect.3 - span)
                                && tolerance.close(bounds[1], effect.3 + span)
                        }
                        None => !supported,
                        _ => false,
                    }
                },
            );
            let representative_interval = did
                .event_time_effects
                .iter()
                .position(|effect| effect.2 >= 0)
                .and_then(|index| did.event_time_intervals_95[index]);
            let expected_uncertainty = if did.event_time_intervals_95.iter().any(Option::is_some) {
                "event_time_pointwise_normal_intervals_independent_clusters"
            } else {
                "cluster_robust_standard_error_no_interval"
            };
            valid
                && did.interval_95 == representative_interval
                && did.uncertainty == expected_uncertainty
        }
    } else {
        did.event_time_intervals_95.is_empty()
    };
    let (treated_subjects, comparison_subjects) = if query.staggered_event_study {
        representative.map_or((0, 0), |effect| (effect.4, effect.5))
    } else if let Some((target, _)) = query.staggered_target {
        let mut cohort_by_subject = std::collections::BTreeMap::new();
        for (subject, cohort) in query.subjects.iter().zip(&query.cohorts) {
            if cohort_by_subject.insert(subject.as_str(), *cohort).is_some_and(|old| old != *cohort)
            {
                return Err(IoError::Convert(
                    "staggered DiD query changes adoption cohort within subject".into(),
                ));
            }
        }
        (
            cohort_by_subject.values().filter(|cohort| **cohort == target).count(),
            cohort_by_subject.values().filter(|cohort| **cohort == 0).count(),
        )
    } else {
        let treated_subjects = subjects.values().filter(|(treated, _)| *treated).count();
        (treated_subjects, subjects.len() - treated_subjects)
    };
    let clusters = if query.staggered_event_study {
        representative.map_or(0, |effect| effect.7)
    } else if let Some((target, _)) = query.staggered_target {
        query
            .clusters
            .iter()
            .zip(&query.cohorts)
            .filter(|(_, cohort)| **cohort == 0 || **cohort == target)
            .map(|(cluster, _)| cluster)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    } else {
        query.clusters.iter().collect::<std::collections::BTreeSet<_>>().len()
    };
    let augmented_valid = match (&query.augmented, &did.augmented) {
        (Some((pre, propensity, prediction, declared)), Some((p_min, p_max, ess, recorded))) => {
            let variables = [query.outcome, *pre, *propensity, *prediction];
            variables.iter().collect::<std::collections::BTreeSet<_>>().len() == 4
                && !duplicate_subject
                && query.post.iter().all(|post| *post)
                && query.periods.is_empty()
                && query.cohorts.is_empty()
                && query.staggered_target.is_none()
                && !query.staggered_event_study
                && !query.repeated_cross_section
                && *declared == *recorded
                && p_min.is_finite()
                && p_max.is_finite()
                && *p_min > 0.0
                && *p_min <= *p_max
                && *p_max < 1.0
                && ess.is_finite()
                && *ess > 0.0
                && *ess <= comparison_subjects as f64 + 1e-9
                && ((did.standard_error.is_none()
                    && did.interval_95.is_none()
                    && did.uncertainty == "point_only_no_standard_error")
                    || (query.resampling.is_some()
                        && did.uncertainty == AUGMENTED_RESAMPLE_UNCERTAINTY
                        && did.interval_95.is_some()
                        && did.standard_error.is_some_and(|se| se.is_finite() && se > 0.0)))
        }
        (None, None) => true,
        _ => false,
    };
    let interval_supported = query.augmented.is_none()
        && query.staggered_target.is_none()
        && !query.staggered_event_study
        && if query.repeated_cross_section {
            !duplicate_subject && cell_clusters.iter().flatten().all(|members| members.len() >= 30)
        } else {
            group_clusters.iter().all(|members| members.len() >= 30)
        };
    let interval_valid = if query.staggered_event_study {
        let representative_scalar_valid =
            match (did.interval_95, did.standard_error, result.standard_error) {
                (Some(_), Some(section_se), Some(reported_se)) => {
                    section_se.is_finite()
                        && section_se > 0.0
                        && ReplayTolerance::new(
                            INTERVAL_BOUND_RELATIVE,
                            ReplayScale::OnePlusReplayed,
                            ReplayNonFinite::Refuse,
                        )
                        .close(reported_se, section_se)
                }
                (None, _, None) => true,
                _ => false,
            };
        representative_scalar_valid && event_intervals_valid
    } else {
        match (did.interval_95, did.standard_error) {
            (Some(bounds), Some(se)) if interval_supported && se.is_finite() && se > 0.0 => {
                let radius = antecedent_stats::NORMAL_Q975 * se;
                let tolerance =
                    interval_bound_replay(INTERVAL_BOUND_RELATIVE_LOOSE, did.effect, radius);
                bounds.iter().all(|value| value.is_finite())
                    && tolerance.close(bounds[0], did.effect - radius)
                    && tolerance.close(bounds[1], did.effect + radius)
                    && result.standard_error.is_some_and(|reported| tolerance.close(reported, se))
                    && did.uncertainty == "cluster_robust_normal_interval_independent_clusters"
            }
            (Some(bounds), Some(se))
                if query.augmented.is_some()
                    && did.uncertainty == AUGMENTED_RESAMPLE_UNCERTAINTY
                    && se.is_finite()
                    && se > 0.0 =>
            {
                // The unsupported whole-cluster resample interval: the normal interval of its
                // interquartile bootstrap standard error, never published as the scalar's.
                let radius = antecedent_stats::NORMAL_Q975 * se;
                let tolerance =
                    interval_bound_replay(INTERVAL_BOUND_RELATIVE_LOOSE, did.effect, radius);
                tolerance.close(bounds[0], did.effect - radius)
                    && tolerance.close(bounds[1], did.effect + radius)
                    && result.standard_error.is_none()
                    && did.graphless_support_status.is_none()
            }
            (None, _) => {
                result.standard_error.is_none()
                    && (query.augmented.is_some()
                        || did.uncertainty == "cluster_robust_standard_error_no_interval"
                        || (did.uncertainty == SAMPLING_WEIGHTED_DID_UNCERTAINTY
                            && !query.repeated_cross_section
                            && !query.staggered_event_study))
            }
            _ => false,
        }
    };
    let did_design =
        if query.repeated_cross_section { "repeated_cross_section_2x2" } else { "panel_2x2" };
    let did_method = if query.repeated_cross_section {
        "four_cell_cluster_scores_cr1"
    } else {
        "cluster_change_scores_cr1"
    };
    let min_cell_clusters =
        cell_clusters.iter().flatten().map(std::collections::BTreeSet::len).min().unwrap_or(0);
    let did_supported = if query.staggered_event_study {
        // Post-adoption event-time intervals: support the representative
        // group-time ATT interval when its adoption cohort and the
        // never-treated controls each clear the independent-cluster gate.
        let mut clusters_by_cohort =
            std::collections::BTreeMap::<i64, std::collections::BTreeSet<&str>>::new();
        for (cohort, cluster) in query.cohorts.iter().zip(&query.clusters) {
            clusters_by_cohort.entry(*cohort).or_default().insert(cluster);
        }
        let controls = clusters_by_cohort.get(&0).map_or(0, std::collections::BTreeSet::len);
        let representative_treated = representative.map_or(0, |effect| {
            clusters_by_cohort.get(&effect.0).map_or(0, std::collections::BTreeSet::len)
        });
        let published_intervals =
            did.event_time_intervals_95.iter().filter(|interval| interval.is_some()).count();
        interval_valid
            && event_intervals_valid
            && did.interval_95.is_some()
            && graphless_admits(
                GraphlessSupportKey {
                    family: "difference_in_differences",
                    design: "staggered_event_study",
                    method: "never_treated_event_study_cluster_cr1",
                    inference_claim: "post_adoption_event_time_pointwise_95_normal_intervals",
                },
                GraphlessAssignmentSupport {
                    assignment_unit: "cluster",
                    treated: representative_treated,
                    control: controls,
                    interval_95_published: true,
                    reported_intervals: published_intervals,
                    ..GraphlessAssignmentSupport::default()
                },
            )
    } else {
        query.augmented.is_none()
            && query.staggered_target.is_none()
            && !query.staggered_event_study
            && interval_valid
            && did.interval_95.is_some()
            && !(query.repeated_cross_section && duplicate_subject)
            && graphless_admits(
                GraphlessSupportKey {
                    family: "difference_in_differences",
                    design: did_design,
                    method: did_method,
                    inference_claim: "pointwise_95_normal_interval",
                },
                GraphlessAssignmentSupport {
                    assignment_unit: "cluster",
                    treated: group_clusters[0].len(),
                    control: group_clusters[1].len(),
                    interval_95_published: true,
                    blocks: if query.repeated_cross_section { 4 } else { 0 },
                    min_block_arm: if query.repeated_cross_section { min_cell_clusters } else { 0 },
                    reported_intervals: 1,
                    ..GraphlessAssignmentSupport::default()
                },
            )
    };
    if result.estimate != Some(did.effect)
        || !interval_valid
        || !did.effect.is_finite()
        || did.standard_error.is_some_and(|se| !se.is_finite() || se < 0.0)
        || (query.augmented.is_none() && did.standard_error.is_none())
        || did.treated_subjects != treated_subjects
        || did.comparison_subjects != comparison_subjects
        || did.clusters != clusters
        || !event_study_valid
        || !event_intervals_valid
        || !augmented_valid
        || !graphless_status_ok(did.graphless_support_status.as_deref(), did_supported, true)
        || (query.repeated_cross_section
            && (duplicate_subject
                || cell_clusters.iter().flatten().any(|members| members.len() < 2)))
        || (!query.repeated_cross_section
            && query.staggered_target.is_none()
            && !query.staggered_event_study
            && query.augmented.is_none()
            && group_clusters.iter().any(|members| members.len() < 2))
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
    {
        return Err(IoError::Convert("invalid panel DiD payload or fabricated interval".into()));
    }
    Ok(())
}

// allow(too_many_lines): one family's full artifact validation is a single contract read top to bottom.
#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
pub(in crate::analysis_result_artifact) fn validate_survival_graphless(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let Some(curve) = &result.survival else {
        return Ok(());
    };
    let crate::CausalQueryWire::Survival(query) = &result.query else {
        return Err(IoError::Convert(
            "survival result section is attached to a different query".into(),
        ));
    };
    let point_only = curve.uncertainty == "point_only_no_interval"
        && curve.rmst_difference_interval.is_none()
        && curve.difference_at_tau_interval.is_none()
        && curve.bootstrap_replicates_requested.is_none()
        && curve.bootstrap_replicates_ok.is_none();
    let pointwise_bootstrap = curve.uncertainty
        == "subject_stratified_percentile_bootstrap_pointwise_95"
        && curve.bootstrap_replicates_requested.is_some_and(|n| (199..=100_000).contains(&n))
        && curve.bootstrap_replicates_ok.is_some_and(|ok| {
            let requested = curve.bootstrap_replicates_requested.unwrap_or(0);
            ok >= 199 && ok >= requested.saturating_sub(requested / 10) && ok <= requested
        })
        && curve.difference_at_tau_interval.is_some_and(|limits| {
            limits[0].is_finite()
                && limits[1].is_finite()
                && -1.0 <= limits[0]
                && limits[0] <= limits[1]
                && limits[1] <= 1.0
        })
        && if query.target_cause.is_some() {
            curve.rmst_difference_interval.is_none()
        } else {
            curve.rmst_difference_interval.is_some_and(|limits| {
                limits[0].is_finite()
                    && limits[1].is_finite()
                    && -curve.tau <= limits[0]
                    && limits[0] <= limits[1]
                    && limits[1] <= curve.tau
            })
        };
    let band_valid = match (&curve.difference_band, &curve.band_unavailable_reason) {
        (Some(band), None) => {
            let requested = curve.bootstrap_replicates_requested.unwrap_or(0);
            query.delayed_entry.is_none()
                && query.censoring_columns.is_empty()
                && pointwise_bootstrap
                && requested >= 399
                && band.replicates_ok >= 399
                && band.replicates_ok >= requested.saturating_sub(requested / 10)
                && band.replicates_ok <= requested
                && band.times == curve.times
                && band.difference.len() == curve.times.len()
                && band.lower.len() == curve.times.len()
                && band.upper.len() == curve.times.len()
                && band.difference.iter().zip(&curve.treated).zip(&curve.control).all(
                    |((&difference, &treated), &control)| {
                        REPLAY_ABSOLUTE.close(difference, treated - control)
                    },
                )
                && band.lower.iter().zip(&band.difference).zip(&band.upper).all(
                    |((&lower, &difference), &upper)| {
                        lower.is_finite()
                            && difference.is_finite()
                            && upper.is_finite()
                            && -1.0 <= lower
                            && lower <= difference
                            && difference <= upper
                            && upper <= 1.0
                    },
                )
        }
        (None, Some(reason)) => pointwise_bootstrap && !reason.trim().is_empty(),
        (None, None) => true,
        (Some(_), Some(_)) => false,
    };
    // The paired RMST + horizon survival scalar route is fixed by the
    // query's entry and censoring facets. Delayed entry, fixed-known-G IPCW,
    // and the combined route each have a distinct (design, method) supported
    // row; cumulative incidence has no RMST companion and is never supported.
    let scalar_design = if query.delayed_entry.is_some() {
        "delayed_entry_two_arm_individual_randomized"
    } else {
        "two_arm_individual_randomized"
    };
    let scalar_method = if query.censoring_columns.is_empty() {
        "arm_stratified_subject_bootstrap_product_limit"
    } else {
        "arm_stratified_subject_bootstrap_ipcw_product_limit"
    };
    let scalar_supported = query.target_cause.is_none()
        && pointwise_bootstrap
        && curve.bootstrap_replicates_ok.is_some_and(|ok| ok >= 299)
        && curve.assignment_counts.is_some_and(|[control, treated]| {
            graphless_admits(
                GraphlessSupportKey {
                    family: "survival",
                    design: scalar_design,
                    method: scalar_method,
                    inference_claim: "rmst_and_horizon_survival_pointwise_95_percentile_intervals",
                },
                GraphlessAssignmentSupport {
                    assignment_unit: "unit",
                    treated,
                    control,
                    interval_95_published: true,
                    rows: control.saturating_add(treated),
                    reported_intervals: 2,
                    all_reported_intervals: true,
                    known_density: !query.censoring_columns.is_empty(),
                    ..GraphlessAssignmentSupport::default()
                },
            )
        });
    // The simultaneous band is its own supported claim, carried on the band's
    // own `graphless_support_status`. It publishes only for the unweighted,
    // no-entry route, and covers the survival or cumulative-incidence grid.
    let band_supported = curve.difference_band.as_ref().is_some_and(|band| {
        band_valid
            && band.replicates_ok >= 399
            && curve.assignment_counts.is_some_and(|[control, treated]| {
                let claim = if query.target_cause.is_some() {
                    "simultaneous_95_cumulative_incidence_difference_band"
                } else {
                    "simultaneous_95_survival_difference_band"
                };
                graphless_admits(
                    GraphlessSupportKey {
                        family: "survival",
                        design: "two_arm_individual_randomized",
                        method: "arm_stratified_subject_bootstrap_supremum_band",
                        inference_claim: claim,
                    },
                    GraphlessAssignmentSupport {
                        assignment_unit: "unit",
                        treated,
                        control,
                        interval_95_published: true,
                        rows: control.saturating_add(treated),
                        reported_intervals: 1,
                        all_reported_intervals: true,
                        ..GraphlessAssignmentSupport::default()
                    },
                )
            })
    });
    #[allow(
        clippy::float_cmp,
        reason = "survival curve endpoints and tau must equal the frozen query/boundary values exactly"
    )]
    if result.estimate.is_some()
        || result.standard_error.is_some()
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
        || curve.censoring_survival_provenance.as_deref()
            != (!query.censoring_columns.is_empty())
                .then_some("caller_supplied_fixed_not_fitted_or_verified")
        || !(point_only || pointwise_bootstrap)
        || !graphless_status_ok(curve.graphless_support_status.as_deref(), scalar_supported, false)
        || !graphless_status_ok(
            curve
                .difference_band
                .as_ref()
                .and_then(|band| band.graphless_support_status.as_deref()),
            band_supported,
            false,
        )
        || (curve.assignment_counts.is_some_and(|counts| counts.contains(&0)))
        || !band_valid
        || curve.tau != query.tau
        || curve.target_cause != query.target_cause
        || curve.times.len() < 2
        || curve.control.len() != curve.times.len()
        || curve.treated.len() != curve.times.len()
        || curve.times[0] != 0.0
        || curve.times.last() != Some(&curve.tau)
        || curve.times.windows(2).any(|w| !w[0].is_finite() || w[0] >= w[1])
        || !curve.tau.is_finite()
        || curve
            .control
            .iter()
            .chain(curve.treated.iter())
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || curve.minimum_event_risk_set == Some(0)
        || (query.target_cause.is_some()
            && (curve.rmst_control.is_some()
                || curve.rmst_treated.is_some()
                || curve.control[0] != 0.0
                || curve.treated[0] != 0.0))
        || (query.target_cause.is_none()
            && (curve.rmst_control.is_none()
                || curve.rmst_treated.is_none()
                || curve.control[0] != 1.0
                || curve.treated[0] != 1.0))
        || [curve.control.as_slice(), curve.treated.as_slice()].into_iter().any(|arm| {
            arm.windows(2)
                .any(|w| if query.target_cause.is_some() { w[0] > w[1] } else { w[0] < w[1] })
        })
        || curve.rmst_control.is_some_and(|v| !v.is_finite() || !(0.0..=curve.tau).contains(&v))
        || curve.rmst_treated.is_some_and(|v| !v.is_finite() || !(0.0..=curve.tau).contains(&v))
    {
        return Err(IoError::Convert("invalid survival payload or fabricated uncertainty".into()));
    }
    Ok(())
}
