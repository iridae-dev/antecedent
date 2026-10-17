//! Interval constructions of adaptive procedures whose coverage is measured, and
//! the calibration keys they are measured under.
//!
//! The calibration harness and the public measured route build their intervals
//! with the functions in this module and their keys with [`calibration_basis`],
//! so a coverage record describes exactly the construction a route reports. A
//! construction here carries no authority of its own: a reported interval is
//! licensed only when its key resolves to a currently attesting record.
//!
//! Two keys must never share a record set: a design measured at a fixed true
//! value (which can under-cover where a stopping rule selects on the data) is
//! keyed apart from the prior-predictive design that licenses a credible
//! interval, and a final analysis without stopping is keyed apart from the
//! designs that stop.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::CalibrationBasis;

/// Query axis of every adaptive-experiment record.
pub const QUERY: &str = "AdaptiveExperimentContrast";
/// Estimator of the Beta-Bernoulli posterior under Thompson assignment.
pub const THOMPSON_ESTIMATOR: &str = "adaptive.thompson_beta_bernoulli.posterior";
/// Estimator of the adaptively weighted epsilon-greedy contrast.
pub const WEIGHTED_ESTIMATOR: &str = "adaptive.epsilon_greedy.weighted_aipw";
/// Estimator of the time-uniform confidence sequence under epsilon-greedy.
pub const SEQUENCE_ESTIMATOR: &str = "adaptive.epsilon_greedy.confidence_sequence";
/// Estimator of the participant-level predictable plug-in confidence sequences under Thompson
/// allocation with expected-loss stopping.
pub const PARTICIPANT_SEQUENCE_ESTIMATOR: &str =
    "adaptive.thompson_beta_bernoulli.participant_plug_in_confidence_sequence";
/// Estimator of the repeated confidence interval under fixed randomization.
pub const REPEATED_INTERVAL_ESTIMATOR: &str = "adaptive.group_sequential.repeated_interval";

/// Success probability of the second arm: the functional the prior-predictive
/// records license.
pub const THOMPSON_FUNCTIONAL: &str = "arm_success_probability:treatment";
/// The same functional measured at fixed true values. It is a different key so
/// that a named boundary at equal arms never governs the licensed interval.
pub const THOMPSON_FIXED_TRUTH_FUNCTIONAL: &str = "arm_success_probability:treatment@fixed_truth";
/// Contrast of the second arm against the first.
pub const CONTRAST_FUNCTIONAL: &str = "arm_contrast:treatment_minus_control";
/// Mean of the second arm.
pub const ARM_MEAN_FUNCTIONAL: &str = "arm_mean:treatment";
/// Every arm's mean at once, read from the participants' success proportions.
pub const PARTICIPANT_ARMS_FUNCTIONAL: &str = "arm_mean:every_arm_simultaneous";
/// The data-selected best treatment's difference from the control.
pub const PARTICIPANT_SELECTED_CONTRAST_FUNCTIONAL: &str =
    "arm_contrast:selected_treatment_minus_control";

/// Assignment adapts and a posterior rule may stop the trial.
pub const STOPPED_DEPENDENCE: &str = "adaptive_assignment_stopped_by_posterior_rule";
/// Assignment adapts, repeated measures per subject, one final analysis.
pub const FINAL_DEPENDENCE: &str = "adaptive_assignment_final_analysis_repeated_measures";
/// Assignment adapts and the interval holds at every look.
pub const TIME_UNIFORM_DEPENDENCE: &str = "adaptive_assignment_time_uniform_all_looks";
/// Fixed randomization, group-sequential stopping at an efficacy boundary.
pub const GROUP_SEQUENTIAL_DEPENDENCE: &str = "fixed_randomization_group_sequential_boundary";

/// Posterior thresholds a Thompson stopping rule was measured at.
pub const THOMPSON_STOP_THRESHOLDS: [f64; 3] = [0.90, 0.95, 0.99];
/// Probability floor the weighted estimate was measured with.
pub const WEIGHTED_PROBABILITY_FLOOR: f64 = 0.01;
/// Exploration mass (epsilon) the weighted estimate was measured at; every arm
/// then has an assignment probability of at least half of it.
pub const MEASURED_EXPLORATION: f64 = 0.3;
/// Nominal level of every measured construction.
pub const MEASURED_LEVEL: f64 = 0.95;
/// Two-sided miscoverage of the confidence sequences.
pub const SEQUENCE_ALPHA: f64 = 0.05;
/// Sample size the confidence-sequence boundary was tuned for.
pub const SEQUENCE_TUNING_SAMPLES: f64 = 100.0;

/// The calibration key of one measured adaptive interval.
///
/// `rows` is the planned horizon of the procedure, not the realized count at a
/// data-dependent stopping look.
#[must_use]
pub fn calibration_basis(
    estimator: &str,
    interval_method: &str,
    inference: &str,
    functional: &str,
    dependence: &str,
    rows: u64,
) -> CalibrationBasis {
    let a = |text: &str| Arc::<str>::from(text);
    CalibrationBasis::new(
        [
            a(QUERY),
            a("NotApplicable"),
            a("fixed"),
            a("adaptive_trial_history"),
            a(inference),
            a(estimator),
            a(interval_method),
            a(""),
            a(dependence),
            a(if inference == "Bayesian" { "beta_bernoulli.uniform_prior" } else { "" }),
            a(functional),
        ],
        MEASURED_LEVEL,
        a("point"),
        rows,
        None,
        None,
        0.0,
    )
}

/// The calibration key of the participant-level procedure's confidence sequences: the
/// predictable plug-in empirical Bernstein sequences of the arms, read at every look of a
/// Thompson-allocated procedure that an expected-loss rule stops. `functional` is
/// [`PARTICIPANT_ARMS_FUNCTIONAL`] or [`PARTICIPANT_SELECTED_CONTRAST_FUNCTIONAL`] and
/// `planned_participants` the participants of the plan, not the realized count at the stop.
#[must_use]
pub fn participant_sequence_calibration_basis(
    functional: &str,
    planned_participants: u64,
) -> CalibrationBasis {
    calibration_basis(
        PARTICIPANT_SEQUENCE_ESTIMATOR,
        "confidence_sequence",
        "Frequentist",
        functional,
        STOPPED_DEPENDENCE,
        planned_participants,
    )
}

/// Equal-tailed Beta credible interval by bisection on the incomplete beta
/// function, for a posterior `Beta(alpha, beta)`.
#[must_use]
#[allow(clippy::nonminimal_bool, reason = "a NaN level or width must fail the open-interval test")]
pub fn beta_credible_interval(alpha: f64, beta: f64, level: f64) -> Option<(f64, f64)> {
    if !(level > 0.0 && level < 1.0) || !(alpha > 0.0 && beta > 0.0) {
        return None;
    }
    let quantile = |p: f64| -> Option<f64> {
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let cdf =
                antecedent_stats::adaptive_inference::regularized_incomplete_beta(mid, alpha, beta)
                    .ok()?;
            if cdf < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(0.5 * (lo + hi))
    };
    Some((quantile((1.0 - level) / 2.0)?, quantile((1.0 + level) / 2.0)?))
}

/// Normal-approximation interval around an adaptively weighted estimate.
#[must_use]
#[allow(clippy::nonminimal_bool, reason = "a NaN level or width must fail the open-interval test")]
pub fn weighted_wald_interval(estimate: f64, variance: f64, level: f64) -> Option<(f64, f64)> {
    if !(level > 0.0 && level < 1.0)
        || !estimate.is_finite()
        || !variance.is_finite()
        || variance < 0.0
    {
        return None;
    }
    let z = antecedent_stats::special::normal_ppf(0.5 + level / 2.0);
    let half = z * variance.sqrt();
    Some((estimate - half, estimate + half))
}

/// Welch difference of two arms: `(mean_t - mean_c, standard error)`.
#[must_use]
pub fn welch_difference(control: &[f64], treatment: &[f64]) -> Option<(f64, f64)> {
    let moments = |x: &[f64]| -> Option<(f64, f64)> {
        if x.len() < 2 || x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
        Some((mean, var / n))
    };
    let (mc, vc) = moments(control)?;
    let (mt, vt) = moments(treatment)?;
    let se = (vc + vt).sqrt();
    (se.is_finite() && se > 0.0).then_some((mt - mc, se))
}

/// Repeated confidence interval at one look: the estimate plus and minus the
/// look's two-sided boundary times its standard error. The intervals of every
/// look hold simultaneously with the rule's total level (Jennison and
/// Turnbull), so the one read at a data-dependent stopping look is covered.
#[must_use]
#[allow(clippy::nonminimal_bool, reason = "a NaN level or width must fail the open-interval test")]
pub fn repeated_confidence_interval(
    estimate: f64,
    standard_error: f64,
    critical: f64,
) -> Option<(f64, f64)> {
    if !estimate.is_finite()
        || !(standard_error.is_finite() && standard_error > 0.0)
        || !(critical.is_finite() && critical > 0.0)
    {
        return None;
    }
    let half = critical * standard_error;
    Some((estimate - half, estimate + half))
}
