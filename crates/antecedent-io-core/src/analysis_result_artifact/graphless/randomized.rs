//! Graphless artifact checks for this family of results.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AnalysisResultWire, GraphlessAssignmentSupport, GraphlessSupportKey, INTERVAL_BOUND_RELATIVE,
    IoError, MIN_EXPECTED_ARM_UNITS, MIN_SMALL_ARM_ROWS, MIN_SMALL_ARM_UNITS, REPLAY_ABSOLUTE,
    REPLAY_ABSOLUTE_LOOSE, REPLAY_ABSOLUTE_TIGHT, combination_count, graphless_admits,
    graphless_status_ok, interval_bound_replay, late_ratio_matches, sampling_weighted_uncertainty,
};

/// Recompute the design-specific conditions for publishing an interval.
fn randomized_interval_supported(
    query: &crate::query_wire::RandomizedEffectQueryWire,
    control: usize,
    treated: usize,
    expected_arm_units: f64,
    minimum_probability: f64,
) -> bool {
    match &query.design {
        crate::RandomizationDesignWire::Bernoulli => {
            matches!(
                query.estimand,
                crate::RandomizedEstimandWire::Itt
                    | crate::RandomizedEstimandWire::CaceLate
                    | crate::RandomizedEstimandWire::TreatmentOnTreated
            ) && query.ancova_covariates.len() <= 2
                && if query.estimand == crate::RandomizedEstimandWire::Itt {
                    expected_arm_units + 1e-9 >= MIN_EXPECTED_ARM_UNITS
                } else {
                    query.realized_assignment.len() >= 400
                }
                && control >= 30
                && treated >= 30
                && query
                    .assignment_probabilities
                    .iter()
                    .all(|p| *p + 1e-12 >= 0.2 && *p <= 0.8 + 1e-12)
                && (query.ancova_covariates.is_empty()
                    || query.assignment_probabilities.iter().all(|p| {
                        REPLAY_ABSOLUTE_TIGHT.close(*p, query.assignment_probabilities[0])
                    }))
        }
        crate::RandomizationDesignWire::Complete { .. }
        | crate::RandomizationDesignWire::Cluster { .. } => control >= 30 && treated >= 30,
        crate::RandomizationDesignWire::Stratified => {
            let mut blocks = std::collections::BTreeMap::<&str, (usize, usize)>::new();
            for (block, assigned) in query.blocks.iter().zip(&query.realized_assignment) {
                let counts = blocks.entry(block).or_default();
                if *assigned {
                    counts.1 += 1;
                } else {
                    counts.0 += 1;
                }
            }
            blocks.len() >= 4
                && control >= 60
                && treated >= 60
                && blocks.values().all(|(c, t)| *c >= 15 && *t >= 15)
        }
        crate::RandomizationDesignWire::Factorial2x2 => query
            .factorial_cell_counts
            .is_some_and(|counts| counts.iter().all(|count| *count >= 30)),
        crate::RandomizationDesignWire::MultiArm => {
            query.multi_arm_assignment.len() >= 400
                && query
                    .multi_arm_probabilities
                    .iter()
                    .all(|row| row.iter().all(|p| *p + 1e-12 >= 0.2))
                && (0..query.multi_arm_labels.len()).all(|arm| {
                    query.multi_arm_assignment.iter().filter(|&&assigned| assigned == arm).count()
                        >= 30
                })
        }
        crate::RandomizationDesignWire::Switchback => {
            let mut sizes = std::collections::BTreeMap::<&str, usize>::new();
            for sequence in &query.assignment_units {
                *sizes.entry(sequence.as_str()).or_default() += 1;
            }
            sizes.len() >= 30
                && control >= 30
                && treated >= 30
                && sizes
                    .values()
                    .next()
                    .is_some_and(|first| *first > 0 && sizes.values().all(|size| size == first))
                && minimum_probability + 1e-12 >= 0.2
        }
    }
}

// allow(too_many_lines): one family's full artifact validation is a single contract read top to bottom.
#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
pub(in crate::analysis_result_artifact) fn validate_randomized_graphless(
    result: &AnalysisResultWire,
    allow_legacy_graphless_missing: bool,
) -> Result<(), IoError> {
    let Some(randomized) = &result.randomized_effect else {
        return Ok(());
    };
    let crate::CausalQueryWire::RandomizedEffect(query) = &result.query else {
        return Err(IoError::Convert(
            "randomized result section is attached to a different query".into(),
        ));
    };
    let (design, point_uncertainty, interval_uncertainty) = match &query.design {
        crate::RandomizationDesignWire::Bernoulli
            if query.estimand == crate::RandomizedEstimandWire::CaceLate =>
        {
            (
                "bernoulli",
                "bernoulli_wald_cace_influence_variance_no_interval",
                Some("bernoulli_wald_cace_influence_normal_interval"),
            )
        }
        crate::RandomizationDesignWire::Bernoulli
            if query.estimand == crate::RandomizedEstimandWire::TreatmentOnTreated =>
        {
            (
                "bernoulli",
                "bernoulli_one_sided_tot_influence_variance_no_interval",
                Some("bernoulli_one_sided_tot_influence_normal_interval"),
            )
        }
        crate::RandomizationDesignWire::Bernoulli if query.fixed_cuped.is_some() => (
            "bernoulli",
            "bernoulli_fixed_cuped_ht_conservative_variance_no_interval",
            Some("bernoulli_fixed_cuped_ht_score_normal_interval"),
        ),
        crate::RandomizationDesignWire::Bernoulli if !query.ancova_covariates.is_empty() => (
            "bernoulli",
            "bernoulli_ancova_hc0_variance_no_interval",
            Some("bernoulli_ancova_hc0_normal_interval"),
        ),
        crate::RandomizationDesignWire::Bernoulli => (
            "bernoulli",
            "bernoulli_ht_conservative_variance_no_interval",
            Some("bernoulli_ht_score_normal_interval"),
        ),
        crate::RandomizationDesignWire::Complete { .. } => (
            "complete",
            "complete_neyman_variance_upper_bound_no_interval",
            Some("complete_neyman_normal_interval"),
        ),
        crate::RandomizationDesignWire::Cluster { .. } if query.fixed_cuped.is_some() => (
            "cluster",
            "cluster_fixed_cuped_neyman_variance_upper_bound_no_interval",
            Some("cluster_fixed_cuped_neyman_normal_interval"),
        ),
        crate::RandomizationDesignWire::Cluster { .. } => (
            "cluster",
            "cluster_neyman_variance_upper_bound_no_interval",
            Some("cluster_neyman_normal_interval"),
        ),
        crate::RandomizationDesignWire::Stratified => (
            "stratified",
            "stratified_neyman_variance_upper_bound_no_interval",
            Some("stratified_neyman_normal_interval"),
        ),
        crate::RandomizationDesignWire::Factorial2x2 => (
            "factorial_2x2",
            "factorial_cell_neyman_variance_upper_bound_no_interval",
            Some("factorial_cell_neyman_pointwise_normal_intervals"),
        ),
        crate::RandomizationDesignWire::Switchback => (
            "switchback",
            "switchback_independent_sequence_sandwich_variance_no_interval",
            Some("switchback_independent_sequence_student_interval"),
        ),
        crate::RandomizationDesignWire::MultiArm => (
            "multi_arm",
            "multi_arm_covariance_free_variance_bound_no_interval",
            Some("multi_arm_ht_score_pointwise_normal_intervals"),
        ),
    };
    let (control, treated) = if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
        (
            query.multi_arm_assignment.iter().filter(|&&arm| arm == 0).count(),
            query.multi_arm_assignment.iter().filter(|&&arm| arm == 1).count(),
        )
    } else if matches!(query.design, crate::RandomizationDesignWire::Cluster { .. }) {
        let mut clusters = std::collections::BTreeMap::new();
        for (unit, assignment) in query.assignment_units.iter().zip(&query.realized_assignment) {
            clusters.insert(unit, *assignment);
        }
        (
            clusters.values().filter(|assigned| !**assigned).count(),
            clusters.values().filter(|assigned| **assigned).count(),
        )
    } else {
        (
            query.realized_assignment.iter().filter(|assigned| !**assigned).count(),
            query.realized_assignment.iter().filter(|assigned| **assigned).count(),
        )
    };
    let minimum_probability = if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
        query
            .multi_arm_probabilities
            .iter()
            .flat_map(|row| row.iter().copied())
            .fold(f64::INFINITY, f64::min)
    } else {
        query
            .assignment_probabilities
            .iter()
            .copied()
            .map(|p| {
                if matches!(query.design, crate::RandomizationDesignWire::Switchback) {
                    p.min(1.0 - p)
                } else {
                    p
                }
            })
            .fold(f64::INFINITY, f64::min)
    };
    // Expected rows of the thinner arm of an independent Bernoulli design: the summed known
    // probabilities, which the two-arm score, ANCOVA and fixed-CUPED intervals are supported by.
    let expected_arm_units = if matches!(query.design, crate::RandomizationDesignWire::Bernoulli) {
        let expected_treated: f64 = query.assignment_probabilities.iter().sum();
        expected_treated.min(query.assignment_probabilities.len() as f64 - expected_treated)
    } else {
        0.0
    };
    let expected_allocations = if query.exact_randomization_test {
        let n = crate::error::fit_u64(query.realized_assignment.len())?;
        let k = crate::error::fit_u64(treated)?;
        combination_count(n, k)
    } else {
        None
    };
    let interval_supported = randomized_interval_supported(
        query,
        control,
        treated,
        expected_arm_units,
        minimum_probability,
    );
    let has_interval = randomized.interval_95.is_some();
    // The small-arm Student interval of an ANCOVA or a complete randomization is a construction
    // of its own: it stands where the normal interval is unsupported and no support row names it.
    let small_arm_uncertainty = match &query.design {
        crate::RandomizationDesignWire::Bernoulli
            if query.estimand == crate::RandomizedEstimandWire::Itt
                && query.fixed_cuped.is_none()
                && !query.ancova_covariates.is_empty() =>
        {
            Some("bernoulli_ancova_hc2_bell_mccaffrey_student_interval")
        }
        crate::RandomizationDesignWire::Complete { .. } => {
            Some("complete_difference_in_means_hc2_bell_mccaffrey_student_interval")
        }
        _ => None,
    };
    let small_arm_interval =
        has_interval && small_arm_uncertainty == Some(randomized.uncertainty.as_str());
    let normal_interval = has_interval && !small_arm_interval;
    let graphless_method = match query.design {
        crate::RandomizationDesignWire::Bernoulli
            if query.estimand == crate::RandomizedEstimandWire::TreatmentOnTreated =>
        {
            Some((
                "complier_effect",
                "bernoulli_one_sided",
                "wald_ratio_influence",
                "unit",
                "pointwise_95_normal_interval",
            ))
        }
        crate::RandomizationDesignWire::Bernoulli
            if query.estimand == crate::RandomizedEstimandWire::CaceLate =>
        {
            Some((
                "complier_effect",
                "bernoulli",
                "wald_ratio_influence",
                "unit",
                "pointwise_95_normal_interval",
            ))
        }
        crate::RandomizationDesignWire::Bernoulli if query.fixed_cuped.is_some() => Some((
            "randomized_effect",
            "bernoulli",
            "fixed_cuped_ht_score",
            "unit",
            "pointwise_95_normal_interval",
        )),
        crate::RandomizationDesignWire::Bernoulli if !query.ancova_covariates.is_empty() => Some((
            "randomized_effect",
            "bernoulli",
            "ancova_hc0",
            "unit",
            "pointwise_95_normal_interval",
        )),
        crate::RandomizationDesignWire::Bernoulli => Some((
            "randomized_effect",
            "bernoulli",
            "independent_action_ht_score",
            "unit",
            "pointwise_95_normal_interval",
        )),
        crate::RandomizationDesignWire::Complete { .. } => Some((
            "randomized_effect",
            "complete",
            "neyman_difference_in_means",
            "unit",
            "pointwise_95_normal_interval",
        )),
        crate::RandomizationDesignWire::Cluster { .. } if query.fixed_cuped.is_some() => Some((
            "randomized_effect",
            "cluster",
            "fixed_cuped_cluster_totals",
            "cluster",
            "pointwise_95_normal_interval",
        )),
        crate::RandomizationDesignWire::Cluster { .. } => Some((
            "randomized_effect",
            "cluster",
            "neyman_unit_weighted_cluster_totals",
            "cluster",
            "pointwise_95_normal_interval",
        )),
        crate::RandomizationDesignWire::Stratified => Some((
            "randomized_effect",
            "stratified",
            "blocked_neyman_difference_in_means",
            "unit",
            "pointwise_95_normal_interval",
        )),
        crate::RandomizationDesignWire::Factorial2x2 => Some((
            "randomized_effect",
            "factorial_2x2",
            "fixed_cell_neyman_contrasts",
            "unit",
            "three_pointwise_95_normal_intervals",
        )),
        crate::RandomizationDesignWire::MultiArm => Some((
            "randomized_effect",
            "multi_arm",
            "independent_action_ht_scores",
            "unit",
            "all_action_pointwise_95_normal_intervals",
        )),
        crate::RandomizationDesignWire::Switchback => Some((
            "randomized_effect",
            "switchback",
            "independent_sequence_ht_score",
            "sequence",
            "pointwise_95_student_interval",
        )),
    };
    let mut block_counts = std::collections::BTreeMap::<&str, (usize, usize)>::new();
    let support_blocks = if matches!(query.design, crate::RandomizationDesignWire::Switchback) {
        &query.assignment_units
    } else {
        &query.blocks
    };
    for (block, assigned) in support_blocks.iter().zip(&query.realized_assignment) {
        let counts = block_counts.entry(block.as_str()).or_default();
        if *assigned {
            counts.1 += 1;
        } else {
            counts.0 += 1;
        }
    }
    let min_block_arm = block_counts.values().flat_map(|(c, t)| [*c, *t]).min().unwrap_or(0);
    let balanced_sequences = matches!(query.design, crate::RandomizationDesignWire::Switchback)
        && block_counts.values().next().is_some_and(|first| {
            let size = first.0 + first.1;
            size > 0 && block_counts.values().all(|counts| counts.0 + counts.1 == size)
        });
    let min_factorial_cell =
        query.factorial_cell_counts.map_or(0, |counts| *counts.iter().min().unwrap_or(&0));
    let min_action_rows = if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
        (0..query.multi_arm_labels.len())
            .map(|arm| {
                query.multi_arm_assignment.iter().filter(|&&assigned| assigned == arm).count()
            })
            .min()
            .unwrap_or(0)
    } else {
        control.min(treated)
    };
    let min_probability = if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
        query
            .multi_arm_probabilities
            .iter()
            .flat_map(|row| row.iter().copied())
            .fold(f64::INFINITY, f64::min)
    } else {
        query
            .assignment_probabilities
            .iter()
            .copied()
            .map(|p| p.min(1.0 - p))
            .fold(f64::INFINITY, f64::min)
    };
    let reported_intervals = if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
        randomized.multi_arm_intervals_95.iter().filter(|interval| interval.is_some()).count()
    } else {
        usize::from(normal_interval)
            + usize::from(randomized.second_factor_interval_95.is_some())
            + usize::from(randomized.factorial_interaction_interval_95.is_some())
    };
    let all_reported_intervals = match query.design {
        crate::RandomizationDesignWire::Factorial2x2 => reported_intervals == 3,
        crate::RandomizationDesignWire::MultiArm => {
            randomized.multi_arm_intervals_95.len() == query.multi_arm_labels.len()
                && reported_intervals + 1 == query.multi_arm_labels.len()
        }
        _ => normal_interval,
    };
    let graphless_supported =
        graphless_method.is_some_and(|(family, design_key, method, assignment_unit, claim)| {
            graphless_admits(
                GraphlessSupportKey { family, design: design_key, method, inference_claim: claim },
                GraphlessAssignmentSupport {
                    assignment_unit,
                    treated,
                    control,
                    interval_95_published: normal_interval,
                    rows: query.realized_assignment.len(),
                    blocks: block_counts.len(),
                    min_block_arm,
                    min_factorial_cell,
                    min_action_rows,
                    min_probability,
                    min_expected_arm_units: expected_arm_units,
                    reported_intervals,
                    covariates: query.ancova_covariates.len(),
                    all_reported_intervals,
                    balanced_sequences,
                    ..GraphlessAssignmentSupport::default()
                },
            )
        });
    let expected_uncertainty = if small_arm_interval {
        small_arm_uncertainty.unwrap_or(point_uncertainty)
    } else if has_interval {
        interval_uncertainty.unwrap_or(point_uncertainty)
    } else {
        point_uncertainty
    };
    // Old point-only artifacts retain their original label and numeric variance on
    // decode only. The old quantity is not reinterpreted as the corrected bound;
    // current exports must carry the new conservative-estimator label.
    let legacy_bernoulli_point = allow_legacy_graphless_missing
        && !has_interval
        && expected_uncertainty == "bernoulli_ht_conservative_variance_no_interval"
        && randomized.uncertainty == "bernoulli_ht_design_variance_no_interval";
    let z95 = if matches!(query.design, crate::RandomizationDesignWire::Switchback) {
        antecedent_stats::student_t_ppf(0.975, (block_counts.len() - 1) as f64)
    } else {
        antecedent_stats::NORMAL_Q975
    };
    let primary_interval_valid = match (randomized.standard_error, randomized.interval_95) {
        (None, None) => result.standard_error.is_none(),
        (Some(se), Some([lower, upper])) => {
            // The Student critical value depends on degrees of freedom the artifact does not
            // carry, so the small-arm interval is checked for the shape it must have: centred on
            // the effect with a critical value between the normal's and a single degree of
            // freedom's.
            let interval_matches = if small_arm_interval {
                let critical = (upper - lower) / (2.0 * se);
                !interval_supported
                    && query.realized_assignment.len() >= MIN_SMALL_ARM_ROWS
                    && control >= MIN_SMALL_ARM_UNITS
                    && treated >= MIN_SMALL_ARM_UNITS
                    && REPLAY_ABSOLUTE_LOOSE.close((lower + upper) / 2.0, randomized.effect)
                    && critical >= z95 - 1e-8
                    && critical <= antecedent_stats::student_t_ppf(0.975, 1.0) + 1e-8
            } else {
                interval_supported
                    && interval_uncertainty.is_some()
                    && REPLAY_ABSOLUTE_LOOSE.close(lower, randomized.effect - z95 * se)
                    && REPLAY_ABSOLUTE_LOOSE.close(upper, randomized.effect + z95 * se)
            };
            interval_matches
                && se.is_finite()
                && se > 0.0
                && result.standard_error.is_some_and(|top| REPLAY_ABSOLUTE.close(top, se))
                && lower.is_finite()
                && upper.is_finite()
                && (matches!(
                    query.design,
                    crate::RandomizationDesignWire::Bernoulli
                        | crate::RandomizationDesignWire::MultiArm
                ) || REPLAY_ABSOLUTE_LOOSE.close(se * se, randomized.variance))
        }
        _ => false,
    };
    let extra_intervals_valid =
        if matches!(query.design, crate::RandomizationDesignWire::Factorial2x2) {
            match (
                randomized.second_factor_effect,
                randomized.second_factor_variance,
                randomized.second_factor_interval_95,
                randomized.factorial_interaction,
                randomized.factorial_interaction_variance,
                randomized.factorial_interaction_interval_95,
            ) {
                (
                    Some(second),
                    Some(second_var),
                    Some([sl, su]),
                    Some(interaction),
                    Some(interaction_var),
                    Some([il, iu]),
                ) if has_interval => {
                    REPLAY_ABSOLUTE_LOOSE.close(sl, second - z95 * second_var.sqrt())
                        && REPLAY_ABSOLUTE_LOOSE.close(su, second + z95 * second_var.sqrt())
                        && REPLAY_ABSOLUTE_LOOSE
                            .close(il, interaction - z95 * interaction_var.sqrt())
                        && REPLAY_ABSOLUTE_LOOSE
                            .close(iu, interaction + z95 * interaction_var.sqrt())
                }
                (_, _, None, _, _, None) if !has_interval => true,
                _ => false,
            }
        } else {
            randomized.second_factor_interval_95.is_none()
                && randomized.factorial_interaction_interval_95.is_none()
        };
    let multi_arm_intervals_valid =
        if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
            if has_interval {
                randomized.multi_arm_intervals_95.len() == query.multi_arm_labels.len()
                    && randomized.multi_arm_values.len() == query.multi_arm_labels.len()
                    && randomized.multi_arm_intervals_95.len() >= 2
                    && randomized.multi_arm_intervals_95[0].is_none()
                    && randomized.multi_arm_intervals_95.iter().enumerate().skip(1).all(
                        |(arm, interval)| {
                            interval.is_some_and(|[lower, upper]| {
                                let contrast = randomized.multi_arm_values[arm].1
                                    - randomized.multi_arm_values[0].1;
                                lower.is_finite()
                                    && upper.is_finite()
                                    && lower < upper
                                    && REPLAY_ABSOLUTE_LOOSE.close((lower + upper) / 2.0, contrast)
                            })
                        },
                    )
                    && randomized.multi_arm_intervals_95[1] == randomized.interval_95
            } else {
                randomized.multi_arm_intervals_95.is_empty()
                    || randomized.multi_arm_intervals_95.iter().all(Option::is_none)
            }
        } else {
            randomized.multi_arm_intervals_95.is_empty()
        };
    if result.estimate != Some(randomized.effect)
        || !primary_interval_valid
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
        || !randomized.effect.is_finite()
        || !randomized.variance.is_finite()
        || randomized.variance < 0.0
        || randomized.assignment_design != design
        || !graphless_status_ok(
            randomized.graphless_support_status.as_deref(),
            graphless_supported,
            allow_legacy_graphless_missing,
        )
        || (randomized.uncertainty != expected_uncertainty
            && !legacy_bernoulli_point
            && !sampling_weighted_uncertainty(&randomized.uncertainty, &query.design, has_interval))
        || !extra_intervals_valid
        || !multi_arm_intervals_valid
        || randomized.estimand
            != if query.estimand == crate::RandomizedEstimandWire::CaceLate {
                "cace_late"
            } else if query.estimand == crate::RandomizedEstimandWire::TreatmentOnTreated {
                "treatment_on_treated"
            } else if matches!(query.design, crate::RandomizationDesignWire::Factorial2x2) {
                "factorial_primary_main_effect"
            } else if matches!(query.design, crate::RandomizationDesignWire::MultiArm) {
                "multi_arm_itt"
            } else {
                "itt"
            }
        || randomized.received_treatment != query.received_treatment
        || randomized.randomization_allocations != expected_allocations
        || (matches!(query.design, crate::RandomizationDesignWire::Factorial2x2)
            != (randomized.second_factor_effect.is_some()
                && randomized.factorial_interaction.is_some()
                && randomized.second_factor_variance.is_some()
                && randomized.factorial_interaction_variance.is_some()))
        || randomized.second_factor_effect.is_some_and(|value| !value.is_finite())
        || randomized.factorial_interaction.is_some_and(|value| !value.is_finite())
        || randomized.second_factor_variance.is_some_and(|value| {
            !value.is_finite()
                || value < 0.0
                || !REPLAY_ABSOLUTE_TIGHT.close(value, randomized.variance)
        })
        || randomized.factorial_interaction_variance.is_some_and(|value| {
            !value.is_finite()
                || value < 0.0
                || !REPLAY_ABSOLUTE_TIGHT.close(value, 4.0 * randomized.variance)
        })
        || (matches!(query.design, crate::RandomizationDesignWire::MultiArm)
            == randomized.multi_arm_values.is_empty())
        || (matches!(query.design, crate::RandomizationDesignWire::MultiArm)
            && (randomized.multi_arm_values.len() != query.multi_arm_labels.len()
                || randomized.multi_arm_values.iter().enumerate().any(
                    |(i, (label, value, variance, support))| {
                        label != &query.multi_arm_labels[i]
                            || !value.is_finite()
                            || !variance.is_finite()
                            || *variance < 0.0
                            || *support
                                != query
                                    .multi_arm_assignment
                                    .iter()
                                    .filter(|&&arm| arm == i)
                                    .count()
                    },
                )
                || randomized.multi_arm_values.get(1).is_none_or(|(_, value, variance, _)| {
                    !REPLAY_ABSOLUTE
                        .close(randomized.effect, value - randomized.multi_arm_values[0].1)
                        || !REPLAY_ABSOLUTE.close(
                            randomized.variance,
                            2.0 * (variance + randomized.multi_arm_values[0].2),
                        )
                })))
        || (query.exact_randomization_test
            && randomized.randomization_p_value.is_none_or(|p| {
                let Some(allocations) = expected_allocations.filter(|count| *count > 0) else {
                    return true;
                };
                !p.is_finite() || p < 1.0 / allocations as f64 || p > 1.0
            }))
        || (!query.exact_randomization_test && randomized.randomization_p_value.is_some())
        || (query.estimand != crate::RandomizedEstimandWire::Itt
            && !late_ratio_matches(
                randomized.effect,
                randomized.intention_to_treat_effect,
                randomized.first_stage_effect,
            ))
        || (query.estimand == crate::RandomizedEstimandWire::Itt
            && (randomized.intention_to_treat_effect.is_some()
                || randomized.first_stage_effect.is_some()))
        || randomized.assignment_units != query.assignment_units
        || randomized.outcome_units != query.outcome_units
        || randomized.blocks != query.blocks
        || randomized.periods != query.periods
        || randomized.treatment_arms != query.treatment_arms
        || randomized.control_units != control
        || randomized.treatment_units != treated
        || !REPLAY_ABSOLUTE_TIGHT
            .close(randomized.minimum_assignment_probability, minimum_probability)
    {
        return Err(IoError::Convert("invalid randomized payload or fabricated interval".into()));
    }
    Ok(())
}

pub(in crate::analysis_result_artifact) fn validate_ratio_graphless(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let Some(fit) = &result.local_polynomial_ratio else {
        return Ok(());
    };
    let crate::CausalQueryWire::LocalPolynomialRatio(query) = &result.query else {
        return Err(IoError::Convert("local ratio result requires matching query".into()));
    };
    let legacy_point_only =
        fit.uncertainty == "rbc_point_with_unvalidated_hc0_standard_error_no_interval";
    let interval_available = !legacy_point_only && fit.standard_error > 0.0;
    let z = antecedent_stats::normal_ppf(0.975);
    let lower = fit.effect - z * fit.standard_error;
    let upper = fit.effect + z * fit.standard_error;
    let tolerance =
        interval_bound_replay(INTERVAL_BOUND_RELATIVE, fit.effect, z * fit.standard_error);
    let bounds_match = |bounds: Option<f64>, expected: f64| {
        if interval_available {
            bounds.is_some_and(|value| value.is_finite() && tolerance.close(value, expected))
        } else {
            bounds.is_none()
        }
    };
    let ratio_design = if fit.kink { "regression_kink" } else { "fuzzy_jump" };
    let ratio_method = if fit.kink {
        "local_quartic_slope_rbc_hc0_delta"
    } else {
        "local_quadratic_cubic_rbc_hc0_delta"
    };
    let ratio_supported = graphless_admits(
        GraphlessSupportKey {
            family: "local_polynomial_ratio",
            design: ratio_design,
            method: ratio_method,
            inference_claim: "pointwise_95_normal_interval",
        },
        GraphlessAssignmentSupport {
            assignment_unit: "unit",
            treated: fit.n_right,
            control: fit.n_left,
            interval_95_published: interval_available,
            reported_intervals: 1,
            ..GraphlessAssignmentSupport::default()
        },
    );
    #[allow(
        clippy::float_cmp,
        reason = "wire fields must equal the frozen query bytes exactly; any drift is a real mismatch"
    )]
    if fit.cutoff != query.cutoff
        || fit.bandwidth != query.bandwidth
        || fit.kink != query.kink
        || result.estimate != Some(fit.effect)
        || (if interval_available {
            result.standard_error != Some(fit.standard_error)
        } else {
            result.standard_error.is_some()
        })
        || !bounds_match(fit.ci_lower, lower)
        || !bounds_match(fit.ci_upper, upper)
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
        || !fit.effect.is_finite()
        || !fit.reduced_form.is_finite()
        || !fit.first_stage.is_finite()
        || fit.first_stage.abs() < 1e-12
        || !fit.standard_error.is_finite()
        || fit.standard_error < 0.0
        || !fit.reduced_form_standard_error.is_finite()
        || fit.reduced_form_standard_error < 0.0
        || !fit.first_stage_standard_error.is_finite()
        || fit.first_stage_standard_error < 0.0
        || fit.n_left < 3
        || fit.n_right < 3
        || (!legacy_point_only && fit.uncertainty != "rbc_hc0_delta_normal_fixed_bandwidth")
        || !graphless_status_ok(fit.graphless_support_status.as_deref(), ratio_supported, true)
    {
        return Err(IoError::Convert("invalid local ratio support or fabricated interval".into()));
    }
    Ok(())
}
