//! Graphless artifact check of the independent-factor 2×2 reading of a four-cell multi-arm
//! randomized design.
//!
//! The section is recomputed from the stored query and the stored cell means and cell variances:
//! the three contrasts are fixed combinations of the cell means, every bound is the
//! Young-inequality sum of the cell variances, the intervals are published exactly when every
//! cell holds the minimum rows and the bounds are positive, and the stored support status is
//! the support row's own decision on the recomputed assignment support.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AnalysisResultWire, GraphlessAssignmentSupport, IoError, REPLAY_ABSOLUTE,
    REPLAY_ABSOLUTE_LOOSE, REPLAY_ABSOLUTE_TIGHT, graphless_admits, graphless_status_ok,
};
use crate::analysis_result_artifact::RandomizedEffectWire;
use crate::graphless_support::{
    FACTORIAL_BERNOULLI_INTERVAL_UNCERTAINTY, FACTORIAL_BERNOULLI_KEY,
    FACTORIAL_BERNOULLI_MIN_CELL_ROWS, FACTORIAL_BERNOULLI_POINT_UNCERTAINTY,
};

/// Estimand label of the primary (factor-A) contrast.
const ESTIMAND: &str = "factorial_primary_main_effect";

/// Factor-A, factor-B and interaction weights over the cells `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`.
const WEIGHTS: [[f64; 4]; 3] =
    [[-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, -0.5, 0.5], [1.0, -1.0, -1.0, 1.0]];

/// Validate a randomized section that reads a four-cell multi-arm design as independent-factor
/// 2×2 contrasts.
#[allow(
    clippy::too_many_lines,
    reason = "one section's full recompute read top to bottom; the checks are independent"
)]
pub(in crate::analysis_result_artifact) fn validate_factorial_bernoulli(
    result: &AnalysisResultWire,
    randomized: &RandomizedEffectWire,
    query: &crate::query_wire::RandomizedEffectQueryWire,
    allow_legacy_graphless_missing: bool,
) -> Result<(), IoError> {
    let invalid = || {
        Err(IoError::Convert(
            "invalid independent-factor 2x2 payload or fabricated interval".into(),
        ))
    };
    if !matches!(query.design, crate::RandomizationDesignWire::MultiArm)
        || query.multi_arm_labels.len() != 4
        || query.multi_arm_assignment.len() != query.multi_arm_probabilities.len()
        || query.multi_arm_assignment.iter().any(|cell| *cell >= 4)
        || randomized.multi_arm_values.len() != 4
    {
        return invalid();
    }
    let mut cell_rows = [0_usize; 4];
    for cell in &query.multi_arm_assignment {
        cell_rows[*cell] += 1;
    }
    let mut means = [0.0; 4];
    let mut variances = [0.0; 4];
    for (cell, (label, mean, variance, rows)) in randomized.multi_arm_values.iter().enumerate() {
        if label != &query.multi_arm_labels[cell]
            || !mean.is_finite()
            || !variance.is_finite()
            || *variance < 0.0
            || *rows != cell_rows[cell]
        {
            return invalid();
        }
        means[cell] = *mean;
        variances[cell] = *variance;
    }
    let contrast = |weights: [f64; 4]| {
        means.iter().zip(weights).map(|(mean, weight)| mean * weight).sum::<f64>()
    };
    let bound = |weights: [f64; 4]| {
        4.0 * variances
            .iter()
            .zip(weights)
            .map(|(variance, weight)| variance * weight.powi(2))
            .sum::<f64>()
    };
    let (Some(second), Some(interaction), Some(second_bound), Some(interaction_bound)) = (
        randomized.second_factor_effect,
        randomized.factorial_interaction,
        randomized.second_factor_variance,
        randomized.factorial_interaction_variance,
    ) else {
        return invalid();
    };
    let effects = [randomized.effect, second, interaction];
    let bounds = [randomized.variance, second_bound, interaction_bound];
    let recomputed = (0..3).all(|k| {
        effects[k].is_finite()
            && bounds[k].is_finite()
            && bounds[k] >= 0.0
            && REPLAY_ABSOLUTE.close(effects[k], contrast(WEIGHTS[k]))
            && REPLAY_ABSOLUTE.close(bounds[k], bound(WEIGHTS[k]))
    }) && REPLAY_ABSOLUTE_TIGHT.close(second_bound, randomized.variance)
        && REPLAY_ABSOLUTE_TIGHT.close(interaction_bound, 4.0 * randomized.variance);
    let publish = cell_rows.iter().all(|rows| *rows >= FACTORIAL_BERNOULLI_MIN_CELL_ROWS)
        && bounds.iter().all(|value| *value > 0.0);
    let intervals = [
        randomized.interval_95,
        randomized.second_factor_interval_95,
        randomized.factorial_interaction_interval_95,
    ];
    let z = antecedent_stats::NORMAL_Q975;
    let intervals_valid = if publish {
        (0..3).all(|k| {
            intervals[k].is_some_and(|[lower, upper]| {
                let span = z * bounds[k].sqrt();
                REPLAY_ABSOLUTE_LOOSE.close(lower, effects[k] - span)
                    && REPLAY_ABSOLUTE_LOOSE.close(upper, effects[k] + span)
            })
        }) && randomized
            .standard_error
            .is_some_and(|se| REPLAY_ABSOLUTE.close(se, randomized.variance.sqrt()))
            && result
                .standard_error
                .zip(randomized.standard_error)
                .is_some_and(|(top, se)| REPLAY_ABSOLUTE.close(top, se))
            && randomized.uncertainty == FACTORIAL_BERNOULLI_INTERVAL_UNCERTAINTY
    } else {
        intervals.iter().all(Option::is_none)
            && randomized.standard_error.is_none()
            && result.standard_error.is_none()
            && randomized.uncertainty == FACTORIAL_BERNOULLI_POINT_UNCERTAINTY
    };
    let minimum_probability = query
        .multi_arm_probabilities
        .iter()
        .flat_map(|row| row.iter().copied())
        .fold(f64::INFINITY, f64::min);
    let control = cell_rows[0] + cell_rows[1];
    let treated = cell_rows[2] + cell_rows[3];
    let fewest_cell_rows = cell_rows.iter().copied().min().unwrap_or(0);
    let supported = graphless_admits(
        FACTORIAL_BERNOULLI_KEY,
        GraphlessAssignmentSupport {
            assignment_unit: "unit",
            treated,
            control,
            interval_95_published: publish,
            rows: query.multi_arm_assignment.len(),
            min_factorial_cell: fewest_cell_rows,
            min_action_rows: fewest_cell_rows,
            min_probability: minimum_probability,
            reported_intervals: if publish { 3 } else { 0 },
            all_reported_intervals: publish,
            ..GraphlessAssignmentSupport::default()
        },
    );
    if result.estimate != Some(randomized.effect)
        || !recomputed
        || !intervals_valid
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
        || randomized.estimand != ESTIMAND
        || query.estimand != crate::RandomizedEstimandWire::Itt
        || query.exact_randomization_test
        || randomized.intention_to_treat_effect.is_some()
        || randomized.first_stage_effect.is_some()
        || randomized.received_treatment.is_some()
        || randomized.randomization_p_value.is_some()
        || randomized.randomization_allocations.is_some()
        || !randomized.multi_arm_intervals_95.is_empty()
        || randomized.assignment_units != query.assignment_units
        || randomized.outcome_units != query.outcome_units
        || randomized.blocks != query.blocks
        || randomized.periods != query.periods
        || randomized.treatment_arms != query.treatment_arms
        || randomized.control_units != control
        || randomized.treatment_units != treated
        || !REPLAY_ABSOLUTE_TIGHT
            .close(randomized.minimum_assignment_probability, minimum_probability)
        || !graphless_status_ok(
            randomized.graphless_support_status.as_deref(),
            supported,
            allow_legacy_graphless_missing,
        )
    {
        return invalid();
    }
    Ok(())
}
