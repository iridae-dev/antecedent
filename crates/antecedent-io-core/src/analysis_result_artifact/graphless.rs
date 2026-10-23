//! Graphless-support checks for an analysis-result body.
//!
//! Each family's stored status is compared with that family's own eligibility
//! recompute. The recompute stays in this file so a forged status cannot be
//! laundered through another classifier.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::AnalysisResultWire;
use crate::IoError;
use crate::graphless_support::{GraphlessAssignmentSupport, GraphlessSupportKey, graphless_admits};

/// Expected rows in each arm of an independent Bernoulli intention-to-treat interval (the support
/// row's `min_expected_arm_units`, restated for this self-contained recompute).
const MIN_EXPECTED_ARM_UNITS: f64 = 40.0;
/// Fewest observed rows in each arm of a small-arm Student interval.
const MIN_SMALL_ARM_UNITS: usize = 2;
/// Fewest rows in all of a trial that publishes a small-arm Student interval.
const MIN_SMALL_ARM_ROWS: usize = 100;
/// Most ANCOVA covariates of a small-arm Student interval.
const MAX_SMALL_ARM_COVARIATES: usize = 2;

/// Whether a stored graphless support status matches the independently
/// recomputed license for a row. This is only the status comparison shared
/// across the families; each family's eligibility recompute (`*_licensed`) is
/// deliberately kept self-contained as a forgery check across the trust
/// boundary and is never routed through the execute/contract classifier.
/// `allow_legacy_missing` permits a `None` status to stand in for an
/// unlicensed row on artifacts written before the field existed.
pub(in crate::analysis_result_artifact) fn graphless_status_ok(
    actual: Option<&str>,
    licensed: bool,
    allow_legacy_missing: bool,
) -> bool {
    actual == licensed.then_some("licensed") || (allow_legacy_missing && actual.is_none())
}

/// `C(n, k)` with the same multiply-then-divide recurrence the artifact
/// validator used inline. Overflow, or `k > n`, refuses instead of panicking
/// or wrapping.
fn combination_count(n: u64, k: u64) -> Option<u64> {
    let mut count = 1u64;
    for i in 0..k {
        let term = n.checked_sub(i)?;
        count = count.checked_mul(term)?.checked_div(i.checked_add(1)?)?;
    }
    Some(count)
}

/// Whether a non-ITT randomized effect equals its intention-to-treat over the
/// first stage. Missing or non-finite components do not match.
fn late_ratio_matches(effect: f64, intention: Option<f64>, first_stage: Option<f64>) -> bool {
    match (intention, first_stage) {
        (Some(intention), Some(first_stage))
            if intention.is_finite() && first_stage.is_finite() && first_stage > f64::EPSILON =>
        {
            (effect - intention / first_stage).abs() <= 1e-10
        }
        _ => false,
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
            "bernoulli_ht_design_variance_no_interval",
            Some("bernoulli_ht_score_normal_interval"),
        ),
        crate::RandomizationDesignWire::Complete { .. } => (
            "complete",
            "complete_neyman_variance_upper_bound_no_interval",
            Some("complete_neyman_normal_interval"),
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
    let interval_supported = match &query.design {
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
                    || query
                        .assignment_probabilities
                        .iter()
                        .all(|p| (*p - query.assignment_probabilities[0]).abs() <= 1e-12))
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
    };
    let has_interval = randomized.interval_95.is_some();
    // The small-arm Student interval of an ANCOVA or a complete randomization is a construction
    // of its own: it stands where the normal interval is unsupported and no licence row names it.
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
    let graphless_licensed =
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
                    && query.ancova_covariates.len() <= MAX_SMALL_ARM_COVARIATES
                    && ((lower + upper) / 2.0 - randomized.effect).abs() <= 1e-8
                    && critical >= z95 - 1e-8
                    && critical <= antecedent_stats::student_t_ppf(0.975, 1.0) + 1e-8
            } else {
                interval_supported
                    && interval_uncertainty.is_some()
                    && (lower - (randomized.effect - z95 * se)).abs() <= 1e-8
                    && (upper - (randomized.effect + z95 * se)).abs() <= 1e-8
            };
            interval_matches
                && se.is_finite()
                && se > 0.0
                && result.standard_error.is_some_and(|top| (top - se).abs() <= 1e-10)
                && lower.is_finite()
                && upper.is_finite()
                && (matches!(
                    query.design,
                    crate::RandomizationDesignWire::Bernoulli
                        | crate::RandomizationDesignWire::MultiArm
                ) || (se * se - randomized.variance).abs() <= 1e-8)
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
                    (sl - (second - z95 * second_var.sqrt())).abs() <= 1e-8
                        && (su - (second + z95 * second_var.sqrt())).abs() <= 1e-8
                        && (il - (interaction - z95 * interaction_var.sqrt())).abs() <= 1e-8
                        && (iu - (interaction + z95 * interaction_var.sqrt())).abs() <= 1e-8
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
                                    && ((lower + upper) / 2.0 - contrast).abs() <= 1e-8
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
            graphless_licensed,
            allow_legacy_graphless_missing,
        )
        || (randomized.uncertainty != expected_uncertainty
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
            !value.is_finite() || value < 0.0 || (value - randomized.variance).abs() > 1e-12
        })
        || randomized.factorial_interaction_variance.is_some_and(|value| {
            !value.is_finite() || value < 0.0 || (value - 4.0 * randomized.variance).abs() > 1e-12
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
                    (randomized.effect - (value - randomized.multi_arm_values[0].1)).abs() > 1e-10
                        || (randomized.variance
                            - 2.0 * (variance + randomized.multi_arm_values[0].2))
                            .abs()
                            > 1e-10
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
        || (randomized.minimum_assignment_probability - minimum_probability).abs() > 1e-12
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
    let tolerance = 1e-10 * (1.0 + fit.effect.abs() + (z * fit.standard_error).abs());
    let bounds_match = |bounds: Option<f64>, expected: f64| {
        if interval_available {
            bounds.is_some_and(|value| value.is_finite() && (value - expected).abs() <= tolerance)
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
    let ratio_licensed = graphless_admits(
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
        || !graphless_status_ok(fit.graphless_support_status.as_deref(), ratio_licensed, true)
    {
        return Err(IoError::Convert("invalid local ratio support or fabricated interval".into()));
    }
    Ok(())
}

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
            let tolerance = 1e-10 * (1.0 + regime.value.abs() + span.abs());
            (eligible_ipw || eligible_msm || eligible_dr || g_formula_expected.is_some())
                && g_formula_expected.is_none_or(|(value, (expected_se, expected_bounds))| {
                    (regime.value - value).abs() <= 1e-10
                        && (se - expected_se).abs() <= 1e-10
                        && bounds == expected_bounds
                })
                && bounds[0].is_finite()
                && bounds[1].is_finite()
                && (bounds[0] - (regime.value - span)).abs() <= tolerance
                && (bounds[1] - (regime.value + span)).abs() <= tolerance
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
                    let tolerance = 1e-10 * (1.0 + center.abs() + span.abs());
                    bounds[0].is_finite()
                        && bounds[1].is_finite()
                        && (bounds[0] - (center - span)).abs() <= tolerance
                        && (bounds[1] - (center + span)).abs() <= tolerance
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
    let has_license =
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
    let graphless_dr_licensed = query.method == "sequential_dr"
        && matches!(query.periods, 2 | 3)
        && eligible_dr
        && regime.value_interval_95.is_some()
        && has_license(
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
    // interval in `g_formula_expected`, so the license requires that exact
    // interval to be published against the two-period fixed-Q row.
    let graphless_g_formula_licensed = query.method == "g_formula"
        && query.periods == 2
        && g_formula_expected.is_some()
        && regime.value_interval_95.is_some()
        && has_license(
            "known_sequential_randomized_two_period",
            "fixed_known_q_g_formula_scores",
            "conditional_q_pointwise_95_normal_interval",
            regime_support,
        );
    // Additive MSM: io establishes support through `eligible_msm` and the
    // published intercept and per-period coefficient intervals, matched to
    // the additive-MSM row for the intercept-and-period-effects claim.
    let graphless_msm_licensed = query.method == "marginal_structural_model"
        && eligible_msm
        && regime.value_interval_95.is_some()
        && regime.period_intervals_95.len() == query.periods
        && has_license(
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
    let graphless_licensed =
        graphless_dr_licensed || graphless_g_formula_licensed || graphless_msm_licensed;
    if !graphless_status_ok(
        regime.graphless_support_status.as_deref(),
        graphless_licensed,
        allow_legacy_graphless_missing,
    ) {
        return Err(IoError::Convert(
            "longitudinal regime graphless license does not match evidenced support".into(),
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
            || (value.policy_value - value.reference_value - value.incremental_value).abs() > 1e-8
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
                        || (lower - (center - radius)).abs() > 1e-8
                        || (upper - (center + radius)).abs() > 1e-8
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
        let tolerance = 1e-10 * (1.0 + center.abs() + span.abs());
        expected_interval
            && (bounds[0] - (center - span)).abs() <= tolerance
            && (bounds[1] - (center + span)).abs() <= tolerance
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
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * (1.0 + a.abs() + b.abs());
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
                let tolerance = 1e-10 * (1.0 + bin.effect.abs() + span.abs());
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
                    && (bounds[0] - (bin.effect - span)).abs() <= tolerance
                    && (bounds[1] - (bin.effect + span)).abs() <= tolerance
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
                    let tolerance = 1e-10 * (1.0 + point.effect.abs() + span.abs());
                    rows.len() >= 300
                        && observed_action >= 50
                        && observed_control >= 50
                        && strong_overlap
                        && point.standard_error > 0.0
                        && bounds.iter().all(|value| value.is_finite())
                        && (bounds[0] - (point.effect - span)).abs() <= tolerance
                        && (bounds[1] - (point.effect + span)).abs() <= tolerance
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
    let licensed = graphless_admits(
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
    if !graphless_status_ok(policy.graphless_support_status.as_deref(), licensed, true) {
        return Err(IoError::Convert(
            "policy graphless support status does not match exact design and interval evidence"
                .into(),
        ));
    }
    Ok(())
}

/// Uncertainty label of the augmented panel's unlicensed whole-cluster resample interval.
const AUGMENTED_RESAMPLE_UNCERTAINTY: &str =
    "whole_cluster_resample_supplied_nuisance_held_fixed_calibration_unmeasured";

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
                            let tolerance = 1e-10 * (1.0 + effect.3.abs() + span.abs());
                            bounds[0].is_finite()
                                && bounds[1].is_finite()
                                && (bounds[0] - (effect.3 - span)).abs() <= tolerance
                                && (bounds[1] - (effect.3 + span)).abs() <= tolerance
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
                        && (reported_se - section_se).abs() <= 1e-10 * (1.0 + section_se.abs())
                }
                (None, _, None) => true,
                _ => false,
            };
        representative_scalar_valid && event_intervals_valid
    } else {
        match (did.interval_95, did.standard_error) {
            (Some(bounds), Some(se)) if interval_supported && se.is_finite() && se > 0.0 => {
                let radius = antecedent_stats::NORMAL_Q975 * se;
                let tolerance = 1e-8 * (1.0 + did.effect.abs() + radius.abs());
                bounds.iter().all(|value| value.is_finite())
                    && (bounds[0] - (did.effect - radius)).abs() <= tolerance
                    && (bounds[1] - (did.effect + radius)).abs() <= tolerance
                    && result
                        .standard_error
                        .is_some_and(|reported| (reported - se).abs() <= tolerance)
                    && did.uncertainty == "cluster_robust_normal_interval_independent_clusters"
            }
            (Some(bounds), Some(se))
                if query.augmented.is_some()
                    && did.uncertainty == AUGMENTED_RESAMPLE_UNCERTAINTY
                    && se.is_finite()
                    && se > 0.0 =>
            {
                // The unlicensed whole-cluster resample interval: the normal interval of its
                // interquartile bootstrap standard error, never published as the scalar's.
                let radius = antecedent_stats::NORMAL_Q975 * se;
                let tolerance = 1e-8 * (1.0 + did.effect.abs() + radius.abs());
                (bounds[0] - (did.effect - radius)).abs() <= tolerance
                    && (bounds[1] - (did.effect + radius)).abs() <= tolerance
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
    let did_licensed = if query.staggered_event_study {
        // Post-adoption event-time intervals: license the representative
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
        || !graphless_status_ok(did.graphless_support_status.as_deref(), did_licensed, true)
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
                        (difference - (treated - control)).abs() <= 1e-10
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
    // and the combined route each have a distinct (design, method) licensed
    // row; cumulative incidence has no RMST companion and is never licensed.
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
    let scalar_licensed = query.target_cause.is_none()
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
    // The simultaneous band is its own licensed claim, carried on the band's
    // own `graphless_support_status`. It publishes only for the unweighted,
    // no-entry route, and covers the survival or cumulative-incidence grid.
    let band_licensed = curve.difference_band.as_ref().is_some_and(|band| {
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
        || !graphless_status_ok(curve.graphless_support_status.as_deref(), scalar_licensed, false)
        || !graphless_status_ok(
            curve
                .difference_band
                .as_ref()
                .and_then(|band| band.graphless_support_status.as_deref()),
            band_licensed,
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

/// Uncertainty label of a sampling-weighted two-arm intention-to-treat fit: the linearized
/// variance of the weighted Hajek arm means, with no interval.
pub(crate) const SAMPLING_WEIGHTED_RANDOMIZED_UNCERTAINTY: &str =
    "sampling_weighted_hajek_linearized_variance_no_interval";

/// Uncertainty label of a sampling-weighted balanced two-period panel `DiD`: the cluster score
/// sandwich over weighted scores, with no interval.
pub(crate) const SAMPLING_WEIGHTED_DID_UNCERTAINTY: &str =
    "sampling_weighted_cluster_robust_standard_error_no_interval";

/// Whether `label` is the sampling-weighted uncertainty, which only the unadjusted Bernoulli
/// and complete designs carry and which publishes no interval.
fn sampling_weighted_uncertainty(
    label: &str,
    design: &crate::RandomizationDesignWire,
    has_interval: bool,
) -> bool {
    label == SAMPLING_WEIGHTED_RANDOMIZED_UNCERTAINTY
        && !has_interval
        && matches!(
            design,
            crate::RandomizationDesignWire::Bernoulli
                | crate::RandomizationDesignWire::Complete { .. }
        )
}

/// Uncertainty label of the normal interval of a sequential contrast.
const SEQUENTIAL_CONTRAST_INTERVAL: &str = "normal_interval_calibration_unmeasured";
/// Uncertainty label of the modular posterior of a sequential contrast.
const SEQUENTIAL_CONTRAST_POSTERIOR: &str = "modular_bayesian_bootstrap_posterior_no_interval";

/// Recompute an observational sequential contrast from its retained scores.
///
/// The result is unlicensed: there is no support coordinate to compare, so the check is that it
/// claims none and that every reported summary is the summary of the retained scores.
#[allow(
    clippy::float_cmp,
    reason = "the declared floor is carried through the artifact unchanged, so equality is the check"
)]
pub(in crate::analysis_result_artifact) fn validate_sequential_contrast_graphless(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let query = match &result.query {
        crate::CausalQueryWire::SequentialContrast(query) => Some(query),
        _ => None,
    };
    let fail = |message: &str| Err(IoError::Convert(message.into()));
    let (Some(query), Some(fit)) = (query, &result.sequential_contrast) else {
        return if query.is_some() || result.sequential_contrast.is_some() {
            fail("a sequential contrast section needs its query, and its query its section")
        } else {
            Ok(())
        };
    };
    let rows = fit.contrast_scores.len();
    let n = rows as f64;
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let near = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + b.abs());
    let scores_ok = rows >= 2
        && fit.active_scores.len() == rows
        && fit.control_scores.len() == rows
        && fit.evaluated_rows == rows as u64
        && rows == query.fold_of_row.len()
        && fit
            .active_scores
            .iter()
            .chain(&fit.control_scores)
            .chain(&fit.contrast_scores)
            .all(|value| value.is_finite())
        && fit
            .active_scores
            .iter()
            .zip(&fit.control_scores)
            .zip(&fit.contrast_scores)
            .all(|((a, c), d)| near(*d, a - c));
    if !scores_ok
        || fit.active_regime != query.active
        || fit.control_regime != query.control
        || fit.min_probability != query.min_probability
        || !near(fit.contrast, mean(&fit.contrast_scores))
        || !near(fit.active_mean, mean(&fit.active_scores))
        || !near(fit.control_mean, mean(&fit.control_scores))
        || !fit.minimum_action_probability.is_finite()
        || fit.minimum_action_probability < fit.min_probability
        || fit.licensed
        || fit.calibration != "unmeasured"
        || result.estimate.is_some()
        || result.standard_error.is_some()
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
    {
        return fail("invalid sequential contrast payload or fabricated summary");
    }
    if result.posterior_artifact.is_some() {
        if fit.uncertainty != SEQUENTIAL_CONTRAST_POSTERIOR
            || fit.standard_error.is_some()
            || fit.interval.is_some()
        {
            return fail("a sequential contrast posterior reports no frequentist interval");
        }
        return Ok(());
    }
    let (Some(se), Some(bounds)) = (fit.standard_error, fit.interval) else {
        return fail("a sequential contrast interval needs its standard error and bounds");
    };
    if fit.uncertainty != SEQUENTIAL_CONTRAST_INTERVAL
        || !(fit.coverage > 0.0 && fit.coverage < 1.0)
    {
        return fail("invalid sequential contrast interval label or coverage");
    }
    let centre = mean(&fit.contrast_scores);
    let squares: f64 = fit.contrast_scores.iter().map(|s| (s - centre) * (s - centre)).sum();
    let expected_se = (squares / ((n - 1.0) * n)).sqrt();
    let span = antecedent_stats::normal_ppf(0.5 + fit.coverage / 2.0) * expected_se;
    let tolerance = 1e-9 * (1.0 + centre.abs() + span.abs());
    if expected_se.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
        || !near(se, expected_se)
        || (bounds[0] - (centre - span)).abs() > tolerance
        || (bounds[1] - (centre + span)).abs() > tolerance
    {
        return fail("sequential contrast interval is not the normal interval of its scores");
    }
    Ok(())
}

/// Uncertainty label of the history-matched contrast.
const HISTORY_MATCH_UNCERTAINTY: &str =
    "abadie_imbens_standard_error_calibration_unmeasured_no_interval";

/// Recompute a within-period history-matched contrast from its retained pairs.
///
/// The result is unlicensed and reports no interval, so the check is that it claims neither and
/// that the contrast, the weight total and the Abadie-Imbens standard error are those of the
/// retained pairs.
pub(in crate::analysis_result_artifact) fn validate_history_match_graphless(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let is_query = matches!(result.query, crate::CausalQueryWire::HistoryMatch(_));
    let Some(fit) = &result.history_match else {
        return if is_query {
            Err(IoError::Convert("history match artifact is missing its result section".into()))
        } else {
            Ok(())
        };
    };
    let fail = |message: &str| Err(IoError::Convert(message.into()));
    if !is_query {
        return fail("a history match section needs its query");
    }
    let near = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + b.abs());
    let pairs = fit.matched_queries.len();
    let total: f64 = fit.pair_weights.iter().sum();
    let weighted: f64 = fit.pair_weights.iter().zip(&fit.differences).map(|(w, d)| w * d).sum();
    let structure_ok = pairs > 0
        && fit.matched_donors.len() == pairs
        && fit.differences.len() == pairs
        && fit.pair_weights.len() == pairs
        && fit.matched_queries.windows(2).all(|w| w.first() < w.get(1))
        && fit.pair_weights.iter().chain(&fit.differences).all(|v| v.is_finite())
        && total > 0.0
        && near(fit.weight_total, total)
        && near(fit.contrast, weighted / total);
    if !structure_ok
        || fit.licensed
        || fit.calibration != "unmeasured"
        || fit.uncertainty != HISTORY_MATCH_UNCERTAINTY
        || result.estimate.is_some()
        || result.standard_error.is_some()
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
    {
        return fail("invalid history match payload or fabricated summary");
    }
    // Abadie-Imbens variance of a weighted mean of one-match differences, with replacement.
    let expected = if pairs < 2 || fit.pair_weights.iter().any(|w| *w <= 0.0) {
        None
    } else {
        let shares: Vec<f64> = fit.pair_weights.iter().map(|w| w / total).collect();
        let concentration: f64 = shares.iter().map(|s| s * s).sum();
        let denominator = 1.0 - concentration;
        let spread: f64 = shares
            .iter()
            .zip(&fit.differences)
            .map(|(share, d)| share * (d - fit.contrast).powi(2))
            .sum();
        let mut donated: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
        for (share, donor) in shares.iter().zip(&fit.matched_donors) {
            *donated.entry(*donor).or_insert(0.0) += share;
        }
        let reuse: f64 = donated.values().map(|s| s * s).sum();
        let variance = 0.5 * spread / denominator * (concentration + reuse);
        (denominator > 0.0 && variance.is_finite() && variance >= 0.0).then_some(variance.sqrt())
    };
    let consistent = match (fit.standard_error, expected) {
        (None, None) => true,
        (Some(reported), Some(recomputed)) => near(reported, recomputed),
        _ => false,
    };
    if consistent {
        Ok(())
    } else {
        fail("history match standard error is not the Abadie-Imbens form of its pairs")
    }
}
