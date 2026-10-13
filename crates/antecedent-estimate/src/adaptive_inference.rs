//! Estimators for data collected by an adaptive experiment, one per
//! inferential paradigm.
//!
//! Each entry point takes the retained history as plain columns (the arms
//! assigned, the policy's realized assignment probabilities, the observed
//! outcomes) that the caller has already checked against the experiment's
//! analysis contract, and returns a labelled result:
//!
//! * [`bayesian_posterior`]: the conjugate Beta-Bernoulli posterior of each
//!   arm with the posterior probability that each is best and the posterior
//!   expected loss of choosing it. Valid under any assignment rule that
//!   depends on the observed history only (the likelihood principle); the
//!   caller states that premise by routing here.
//! * [`expected_loss_stopping`]: a stop-or-continue decision from posterior
//!   expected loss. The decision is a function of the history; its error
//!   rates are not claimed.
//! * [`adaptively_weighted_estimate`]: augmented inverse-probability scores
//!   with a predictable outcome model and predictable weights (Hadad,
//!   Hirshberg, Zhan, Wager and Athey 2021), with the martingale plug-in
//!   variance reported as uncalibrated.
//! * [`anytime_valid_arm_sequences`]: time-uniform confidence sequences for
//!   bounded arm means (Robbins 1970; Howard, Ramdas, `McAuliffe` and Sekhon
//!   2021), reported as an uncalibrated sequence.
//! * [`group_sequential_boundaries`] and [`group_sequential_decision`]:
//!   Lan-`DeMets` alpha-spending boundaries for pre-specified looks with
//!   independent increments, and the first look at which the statistic
//!   crosses.
//!
//! No result carries an interval or a coverage statement as a claim. Every
//! guarantee slot is a [`WithheldGuarantee`] that refuses with
//! `cell_not_licensed` until whole-procedure coverage of the exact procedure
//! (assignment, outcome structure, monitoring and stopping together) has been
//! measured. The numerics are in [`antecedent_stats::adaptive_inference`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::adaptive_inference as numerics;
use antecedent_stats::adaptive_inference::AdaptiveInferenceError as NumericError;

pub use antecedent_stats::adaptive_inference::{
    ArmSequence, Boundaries, Estimand, OutcomeModelPrior, Sidedness, SpendingFunction, WeightScheme,
};

/// A coded refusal from an adaptive-inference estimator.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdaptiveInferenceError {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced detail.
    pub detail: &'static str,
    /// What was refused and what would license it.
    pub message: String,
}

impl AdaptiveInferenceError {
    fn new(code: &'static str, detail: &'static str, message: impl Into<String>) -> Self {
        Self { code, detail, message: message.into() }
    }
}

impl std::fmt::Display for AdaptiveInferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}: {}: {}",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.message
        )
    }
}

impl std::error::Error for AdaptiveInferenceError {}

fn numeric_error(error: &NumericError) -> AdaptiveInferenceError {
    match error {
        NumericError::Positivity { unit, arm, probability, floor } => AdaptiveInferenceError::new(
            antecedent_core::reason_code!("adaptive_assumption_violated"),
            "adaptive_inference.positivity_floor",
            format!(
                "arm {arm} had assignment probability {probability} below the declared floor {floor} at unit {unit}: its inverse-probability score is not supported; declare a clipping policy as its own route or analyse with a route that does not weight by assignment probability"
            ),
        ),
        NumericError::Cancelled => cancelled(),
        NumericError::Numerical(what) => AdaptiveInferenceError::new(
            antecedent_core::reason_code!("invalid_argument"),
            "adaptive_inference.numerical_failure",
            format!("{what}; no value is reported"),
        ),
        other => AdaptiveInferenceError::new(
            antecedent_core::reason_code!("invalid_argument"),
            "adaptive_inference.input_invalid",
            other.to_string(),
        ),
    }
}

fn cancelled() -> AdaptiveInferenceError {
    AdaptiveInferenceError::new(
        antecedent_core::reason_code!("cancelled_no_claim"),
        "adaptive_inference.cancelled",
        "cancelled before a result was formed; this is not a verdict on the data",
    )
}

fn check_cancelled(ctx: &ExecutionContext) -> Result<(), AdaptiveInferenceError> {
    if ctx.cancellation.is_cancelled() { Err(cancelled()) } else { Ok(()) }
}

/// A guarantee route that is closed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WithheldGuarantee {
    /// Always `cell_not_licensed`.
    pub code: &'static str,
    /// Namespaced detail naming the closed claim.
    pub detail: &'static str,
    /// Calibration reason: `estimator_grid_not_measured`.
    pub reason: &'static str,
}

impl WithheldGuarantee {
    fn closed(detail: &'static str) -> Self {
        Self {
            code: antecedent_core::reason_code!("cell_not_licensed"),
            detail,
            reason: antecedent_core::reason_code!("estimator_grid_not_measured"),
        }
    }

    /// The refusal a request for the guarantee receives.
    #[must_use]
    pub fn refusal(&self, what: &str) -> AdaptiveInferenceError {
        AdaptiveInferenceError::new(
            self.code,
            self.detail,
            format!(
                "{what} is not licensed: whole-procedure coverage of this assignment, outcome structure, monitoring and stopping rule has not been measured; the point values and uncalibrated diagnostics are the result"
            ),
        )
    }
}

// -- Bayesian posterior ----------------------------------------------------------------------

/// Observed successes and failures of one arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArmCounts {
    /// Observed outcomes equal to one.
    pub successes: u64,
    /// Observed outcomes equal to zero.
    pub failures: u64,
}

/// The posterior of one arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosteriorArm {
    /// First Beta shape.
    pub alpha: f64,
    /// Second Beta shape.
    pub beta: f64,
    /// Posterior mean.
    pub mean: f64,
    /// Posterior variance.
    pub variance: f64,
    /// Posterior probability that the arm has the largest success probability.
    pub probability_best: f64,
    /// Posterior expected loss of choosing the arm.
    pub expected_loss: f64,
}

/// Beta-Bernoulli posteriors of every arm.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianPosterior {
    /// Prior shapes used for every arm.
    pub prior: (f64, f64),
    /// One posterior per arm, in arm order.
    pub arms: Vec<PosteriorArm>,
    /// The premise under which the posterior is the same under adaptive
    /// assignment.
    pub premise: &'static str,
    /// The closed credible-interval and decision-reliability claim.
    pub guarantee: WithheldGuarantee,
}

/// Premise of the posterior's validity under adaptive assignment.
pub const POSTERIOR_PREMISE: &str = "assignment depends only on the observed history and on randomization independent of the unobserved outcomes and parameters, outcome availability does not depend on the outcome, and the stopping rule depends on the observed history only";

/// Conjugate Beta-Bernoulli posteriors from observed counts.
///
/// # Errors
/// A prior shape below one (the quadrature relies on a bounded density), a
/// count total above the bound, fewer than two arms, or cancellation.
pub fn bayesian_posterior(
    prior_alpha: f64,
    prior_beta: f64,
    counts: &[ArmCounts],
    ctx: &ExecutionContext,
) -> Result<BayesianPosterior, AdaptiveInferenceError> {
    check_cancelled(ctx)?;
    let cancel = || ctx.cancellation.is_cancelled();
    let posteriors = counts
        .iter()
        .map(|c| {
            numerics::BetaPosterior::from_counts(prior_alpha, prior_beta, c.successes, c.failures)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| numeric_error(&e))?;
    let best =
        numerics::best_arm_probabilities(&posteriors, &cancel).map_err(|e| numeric_error(&e))?;
    let loss = numerics::expected_losses(&posteriors, &cancel).map_err(|e| numeric_error(&e))?;
    let arms = posteriors
        .iter()
        .zip(best.iter().zip(&loss))
        .map(|(p, (&probability_best, &expected_loss))| PosteriorArm {
            alpha: p.alpha(),
            beta: p.beta(),
            mean: p.mean(),
            variance: p.variance(),
            probability_best,
            expected_loss,
        })
        .collect();
    Ok(BayesianPosterior {
        prior: (prior_alpha, prior_beta),
        arms,
        premise: POSTERIOR_PREMISE,
        guarantee: WithheldGuarantee::closed("adaptive_inference.posterior_interval_closed"),
    })
}

/// A stopping rule evaluated on a posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum PosteriorRule {
    /// Stop when the best arm's posterior probability of being best exceeds
    /// the threshold.
    ProbabilityBest {
        /// Threshold in `(0.5, 1)`.
        threshold: f64,
    },
    /// Stop when the best arm's posterior expected loss is below the
    /// threshold.
    ExpectedLoss {
        /// Threshold, in units of the success probability.
        loss_threshold: f64,
    },
}

/// The decision a posterior stopping rule reaches at one look.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StoppingDecision {
    /// The arm with the largest posterior probability of being best.
    pub best_arm: usize,
    /// Its posterior probability of being best.
    pub probability_best: f64,
    /// Its posterior expected loss.
    pub expected_loss: f64,
    /// Whether the rule says stop.
    pub stop: bool,
    /// The closed claim about the rule's error rates.
    pub guarantee: WithheldGuarantee,
}

/// Evaluate a posterior stopping rule.
///
/// # Errors
/// A threshold outside its range or an empty posterior.
pub fn expected_loss_stopping(
    posterior: &BayesianPosterior,
    rule: PosteriorRule,
) -> Result<StoppingDecision, AdaptiveInferenceError> {
    let valid = match rule {
        PosteriorRule::ProbabilityBest { threshold } => threshold > 0.5 && threshold < 1.0,
        PosteriorRule::ExpectedLoss { loss_threshold } => {
            loss_threshold.is_finite() && loss_threshold > 0.0
        }
    };
    if !valid || posterior.arms.is_empty() {
        return Err(AdaptiveInferenceError::new(
            antecedent_core::reason_code!("invalid_argument"),
            "adaptive_inference.input_invalid",
            "a posterior rule needs a probability threshold in (0.5, 1) or a positive loss threshold",
        ));
    }
    let (best_arm, best) = posterior
        .arms
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.probability_best.total_cmp(&b.probability_best))
        .map(|(index, arm)| (index, *arm))
        .expect("a nonempty posterior has a best arm");
    let stop = match rule {
        PosteriorRule::ProbabilityBest { threshold } => best.probability_best > threshold,
        PosteriorRule::ExpectedLoss { loss_threshold } => best.expected_loss < loss_threshold,
    };
    Ok(StoppingDecision {
        best_arm,
        probability_best: best.probability_best,
        expected_loss: best.expected_loss,
        stop,
        guarantee: WithheldGuarantee::closed("adaptive_inference.stopping_reliability_closed"),
    })
}

// -- adaptively weighted scores ------------------------------------------------------------------

/// The retained history of one analysis, as columns.
#[derive(Clone, Copy, Debug)]
pub struct WeightedHistory<'a> {
    /// Number of arms.
    pub arms: usize,
    /// Assigned arm per unit, in enrollment order.
    pub assigned: &'a [usize],
    /// Row-major `units x arms` assignment probabilities.
    pub probabilities: &'a [f64],
    /// One outcome per unit (a subject mean for repeated measures).
    pub outcomes: &'a [f64],
    /// Optional per-unit weight multiplier (measurements per subject).
    pub multiplier: Option<&'a [f64]>,
}

/// Options of the adaptively weighted estimator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedOptions {
    /// How scores are weighted.
    pub scheme: WeightScheme,
    /// Prior of the predictable outcome model.
    pub prior: OutcomeModelPrior,
    /// Smallest assignment probability an estimand arm may have had.
    pub probability_floor: f64,
}

/// An adaptively weighted estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedResult {
    /// The quantity estimated.
    pub estimand: Estimand,
    /// The weighting scheme.
    pub scheme: WeightScheme,
    /// The weighted point estimate.
    pub estimate: f64,
    /// Plug-in martingale variance, uncalibrated.
    pub uncalibrated_variance: f64,
    /// Square root of the variance, uncalibrated.
    pub uncalibrated_standard_error: f64,
    /// Units that contributed.
    pub units: usize,
    /// Effective number of units under the weights.
    pub effective_units: f64,
    /// Largest weight as a share of the total.
    pub max_weight_share: f64,
    /// The closed interval claim.
    pub guarantee: WithheldGuarantee,
}

/// Adaptively weighted augmented inverse-probability estimate of an arm mean
/// or a contrast.
///
/// # Errors
/// A malformed history, an estimand arm below the probability floor
/// (`adaptive_assumption_violated`), or cancellation.
pub fn adaptively_weighted_estimate(
    history: &WeightedHistory<'_>,
    estimand: Estimand,
    options: &WeightedOptions,
    ctx: &ExecutionContext,
) -> Result<WeightedResult, AdaptiveInferenceError> {
    check_cancelled(ctx)?;
    let cancel = || ctx.cancellation.is_cancelled();
    let scores = numerics::adaptive_scores(
        history.arms,
        history.assigned,
        history.probabilities,
        history.outcomes,
        options.prior,
        &cancel,
    )
    .map_err(|e| numeric_error(&e))?;
    let fit = numerics::weighted_estimate(
        history.arms,
        history.probabilities,
        &scores,
        estimand,
        options.scheme,
        history.multiplier,
        options.probability_floor,
    )
    .map_err(|e| numeric_error(&e))?;
    Ok(WeightedResult {
        estimand,
        scheme: options.scheme,
        estimate: fit.estimate,
        uncalibrated_variance: fit.variance,
        uncalibrated_standard_error: fit.variance.sqrt(),
        units: fit.units,
        effective_units: fit.effective_units,
        max_weight_share: fit.max_weight_share,
        guarantee: WithheldGuarantee::closed("adaptive_inference.interval_closed"),
    })
}

// -- confidence sequences --------------------------------------------------------------------

/// Confidence sequences of the arms of a bounded outcome, and optionally the
/// union-bound bounds of a contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceResult {
    /// One sequence per arm, in arm order.
    pub arms: Vec<ArmSequence>,
    /// Bounds of `plus - minus` when a contrast was requested.
    pub contrast: Option<(f64, f64)>,
    /// Two-sided miscoverage each arm's sequence was constructed at.
    pub arm_alpha: f64,
    /// Sample size the boundary is tuned for.
    pub tuning_samples: f64,
    /// Always `uncalibrated_confidence_sequence`.
    pub label: &'static str,
    /// The closed whole-procedure guarantee.
    pub guarantee: WithheldGuarantee,
}

/// Time-uniform confidence sequences for `[0, 1]`-bounded arm means.
///
/// `values_by_arm[a]` are the observed outcomes of arm `a` in the order the
/// arm was pulled. With a contrast, each arm is constructed at `alpha / 2` so
/// the contrast bounds hold jointly by a union bound.
///
/// # Errors
/// An outcome outside `[0, 1]` (`adaptive_inference.outcome_invalid`), a
/// contrast arm out of range, a bad level, or cancellation.
pub fn anytime_valid_arm_sequences(
    values_by_arm: &[Vec<f64>],
    contrast: Option<(usize, usize)>,
    alpha: f64,
    tuning_samples: f64,
    ctx: &ExecutionContext,
) -> Result<SequenceResult, AdaptiveInferenceError> {
    check_cancelled(ctx)?;
    if values_by_arm.iter().flatten().any(|x| !(x.is_finite() && (0.0..=1.0).contains(x))) {
        return Err(AdaptiveInferenceError::new(
            antecedent_core::reason_code!("invalid_argument"),
            "adaptive_inference.outcome_invalid",
            "the confidence sequence is defined for outcomes bounded in [0, 1]; rescale a known-bounded outcome or use a route for unbounded outcomes",
        ));
    }
    if contrast.is_some_and(|(plus, minus)| {
        plus == minus || plus >= values_by_arm.len() || minus >= values_by_arm.len()
    }) {
        return Err(AdaptiveInferenceError::new(
            antecedent_core::reason_code!("invalid_argument"),
            "adaptive_inference.input_invalid",
            "a contrast needs two different declared arms",
        ));
    }
    let arm_alpha = if contrast.is_some() { alpha / 2.0 } else { alpha };
    let cancel = || ctx.cancellation.is_cancelled();
    let arms = values_by_arm
        .iter()
        .map(|values| {
            numerics::bounded_mean_confidence_sequence(values, arm_alpha, tuning_samples, &cancel)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| numeric_error(&e))?;
    let contrast =
        contrast.map(|(plus, minus)| numerics::contrast_bounds(&arms[plus], &arms[minus]));
    Ok(SequenceResult {
        arms,
        contrast,
        arm_alpha,
        tuning_samples,
        label: "uncalibrated_confidence_sequence",
        guarantee: WithheldGuarantee::closed("adaptive_inference.interval_closed"),
    })
}

// -- group-sequential monitoring --------------------------------------------------------------

/// A pre-specified group-sequential rule.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupSequentialPlan {
    /// Information fraction at each look, strictly increasing in `(0, 1]`.
    pub fractions: Vec<f64>,
    /// Total alpha the rule spends.
    pub alpha: f64,
    /// Spending function.
    pub spending: SpendingFunction,
    /// Which crossings stop the trial.
    pub sidedness: Sidedness,
}

/// Efficacy boundaries of a plan.
///
/// # Errors
/// An invalid plan, a boundary outside the search bracket, or cancellation.
pub fn group_sequential_boundaries(
    plan: &GroupSequentialPlan,
    ctx: &ExecutionContext,
) -> Result<Boundaries, AdaptiveInferenceError> {
    check_cancelled(ctx)?;
    let cancel = || ctx.cancellation.is_cancelled();
    numerics::boundaries(&plan.fractions, plan.alpha, plan.spending, plan.sidedness, &cancel)
        .map_err(|e| numeric_error(&e))
}

/// The outcome of applying a group-sequential rule to the observed looks.
#[derive(Clone, Debug, PartialEq)]
pub struct GroupSequentialResult {
    /// Critical value at each planned look.
    pub critical: Vec<f64>,
    /// Alpha spent at each planned look.
    pub spend_increments: Vec<f64>,
    /// Observed standardized statistic at each performed look.
    pub statistics: Vec<f64>,
    /// First look at which the statistic crossed, if any.
    pub stopped_at: Option<usize>,
    /// The closed claim about bias-adjusted estimates and intervals after
    /// stopping.
    pub guarantee: WithheldGuarantee,
}

/// Apply a rule to the standardized statistics of the performed looks.
///
/// The plan must list at least as many looks as were performed; the
/// boundaries of the performed looks are those the rule spends at the
/// realized information fractions.
///
/// # Errors
/// More statistics than planned looks, a non-finite statistic, an invalid
/// plan, or cancellation.
pub fn group_sequential_decision(
    plan: &GroupSequentialPlan,
    statistics: &[f64],
    ctx: &ExecutionContext,
) -> Result<GroupSequentialResult, AdaptiveInferenceError> {
    if statistics.is_empty()
        || statistics.len() > plan.fractions.len()
        || statistics.iter().any(|z| !z.is_finite())
    {
        return Err(AdaptiveInferenceError::new(
            antecedent_core::reason_code!("invalid_argument"),
            "adaptive_inference.input_invalid",
            "one finite standardized statistic per performed look, at most the planned looks",
        ));
    }
    let performed = GroupSequentialPlan {
        fractions: plan.fractions[..statistics.len()].to_vec(),
        ..plan.clone()
    };
    let boundaries = group_sequential_boundaries(&performed, ctx)?;
    let stopped_at = numerics::first_crossing(statistics, &boundaries.critical, plan.sidedness);
    Ok(GroupSequentialResult {
        critical: boundaries.critical,
        spend_increments: boundaries.spend_increments,
        statistics: statistics.to_vec(),
        stopped_at,
        guarantee: WithheldGuarantee::closed("adaptive_inference.stagewise_adjustment_closed"),
    })
}

/// The Welch standardized difference of two arms' outcomes,
/// `(mean_t - mean_c) / sqrt(s_t^2 / n_t + s_c^2 / n_c)`.
///
/// # Errors
/// Fewer than two outcomes in an arm, a non-finite outcome, or a zero
/// variance in both arms.
pub fn standardized_difference(
    control: &[f64],
    treatment: &[f64],
) -> Result<f64, AdaptiveInferenceError> {
    let invalid = |message: &'static str| {
        AdaptiveInferenceError::new(
            antecedent_core::reason_code!("invalid_argument"),
            "adaptive_inference.input_invalid",
            message,
        )
    };
    if control.len() < 2 || treatment.len() < 2 {
        return Err(invalid("each arm needs at least two observed outcomes at a look"));
    }
    if control.iter().chain(treatment).any(|y| !y.is_finite()) {
        return Err(invalid("an outcome is not finite"));
    }
    let moments = |values: &[f64]| {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (n, mean, variance)
    };
    let (nc, mc, vc) = moments(control);
    let (nt, mt, vt) = moments(treatment);
    let standard_error = (vc / nc + vt / nt).sqrt();
    if standard_error.is_nan() || standard_error <= 0.0 {
        return Err(invalid("both arms have zero sample variance at this look"));
    }
    Ok((mt - mc) / standard_error)
}

// -- calibration-internal candidate intervals ---------------------------------------------------

/// A nominal Wald interval around an adaptively weighted estimate, for the
/// calibration harness only. Compiled only with `calibration-internal`.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
#[must_use]
pub fn weighted_wald_interval_internal(result: &WeightedResult, level: f64) -> Option<(f64, f64)> {
    let level_in_range = level > 0.0 && level < 1.0;
    if !level_in_range || !result.uncalibrated_variance.is_finite() {
        return None;
    }
    let z = antecedent_stats::special::normal_ppf(0.5 + level / 2.0);
    let half = z * result.uncalibrated_standard_error;
    Some((result.estimate - half, result.estimate + half))
}

/// Equal-tailed Beta credible interval of one posterior arm by bisection on
/// the incomplete beta function, for the calibration harness only.
/// Compiled only with `calibration-internal`.
#[cfg(feature = "calibration-internal")]
#[doc(hidden)]
#[must_use]
pub fn beta_credible_interval_internal(arm: &PosteriorArm, level: f64) -> Option<(f64, f64)> {
    if !(level > 0.0 && level < 1.0) {
        return None;
    }
    let quantile = |p: f64| -> Option<f64> {
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let cdf = numerics::regularized_incomplete_beta(mid, arm.alpha, arm.beta).ok()?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welch_statistic_matches_the_hand_calculation() {
        // control (1, 2, 3): mean 2, var 1; treatment (2, 4, 6): mean 4, var 4.
        // se = sqrt(1/3 + 4/3) = sqrt(5/3); z = 2 / sqrt(5/3).
        let z = standardized_difference(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap();
        assert!((z - 2.0 / (5.0_f64 / 3.0).sqrt()).abs() < 1e-14);
    }
}
