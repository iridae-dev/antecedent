//! Posterior of a fixed policy's value under a known randomization.
//!
//! Static and sequential policy values share one construction. The frequentist rows
//! ([`crate::policy_value`] and [`antecedent_stats::longitudinal_regime`]) average a per-subject
//! augmented score. Here the same score is the Bayesian-bootstrap functional of
//! [`crate::score_posterior`]: a Dirichlet-process prior of vanishing concentration on the
//! subject law, whose posterior of the mean score is the posterior of the policy value.
//!
//! **Why the score mean is the policy value.** The assignment probabilities are declared by the
//! design. For any frozen outcome prediction `μ` the augmented score has mean equal to the value
//! of the policy, so the posterior needs no outcome model to be correct and no nuisance
//! uncertainty is dropped: the only unknown is the subject law. The prior does not impose the
//! randomization; the declared probabilities do, and a design whose probabilities are not the
//! true ones moves the target, not just the interval.
//!
//! **Sequential regimes.** The score is the backward sequential doubly robust score. The
//! sequential outcome predictions must be genuinely subject-excluded (held-out) as in the
//! frequentist row; a subject-excluded prediction is a function of other subjects, so the
//! subjects' scores are conditionally independent only approximately.
//!
//! Observational sequential data, where the probabilities are fitted, is a different object
//! ([`crate::sequential_aipw_inference`]).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::longitudinal_regime::{
    RegimeValueSummary, SequentialDrInputs, SubjectHistories,
    evaluate_sequential_dr_value_with_subject_scores,
};

use crate::EstimationError;
use crate::policy_value::{BinaryPolicyDesign, binary_policy_row_scores};
use crate::score_posterior::{
    ScorePosterior, ScorePosteriorOptions, bayesian_bootstrap_score_posteriors,
};

/// Joint posterior of a binary policy, its reference and their paired difference.
///
/// The three posteriors share every draw's weights, so `incremental.draws[k]` is
/// `policy.draws[k] - reference.draws[k]`.
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryPolicyValuePosterior {
    /// Net value of the policy.
    pub policy: ScorePosterior,
    /// Net value of the reference.
    pub reference: ScorePosterior,
    /// Policy minus reference.
    pub incremental: ScorePosterior,
}

/// Posterior of a fixed binary policy value from known randomized assignment.
///
/// `nuisance` is the frozen held-out `(mu0, mu1)` outcome predictions, or `None` for the
/// inverse-probability score.
///
/// # Errors
/// The design, outcome or prediction refusals of [`binary_policy_row_scores`], and the option
/// refusals of [`bayesian_bootstrap_score_posteriors`].
pub fn bayesian_binary_policy_value(
    outcome: &[f64],
    design: &BinaryPolicyDesign<'_>,
    nuisance: Option<(&[f64], &[f64])>,
    options: ScorePosteriorOptions,
    ctx: &ExecutionContext,
) -> Result<BinaryPolicyValuePosterior, EstimationError> {
    let rows =
        binary_policy_row_scores(outcome, design, nuisance).map_err(EstimationError::data_msg)?;
    let mut posteriors = bayesian_bootstrap_score_posteriors(
        &[&rows.policy, &rows.reference, &rows.incremental],
        options,
        ctx,
    )?
    .into_iter();
    match (posteriors.next(), posteriors.next(), posteriors.next()) {
        (Some(policy), Some(reference), Some(incremental)) => {
            Ok(BinaryPolicyValuePosterior { policy, reference, incremental })
        }
        _ => Err(EstimationError::stats_msg("policy posterior lost a score column")),
    }
}

/// Posterior of a prespecified sequential regime value from known sequential randomization.
#[derive(Clone, Debug, PartialEq)]
pub struct SequentialRegimeValuePosterior {
    /// Frequentist point summary of the same scores.
    pub summary: RegimeValueSummary,
    /// Posterior of the regime value.
    pub value: ScorePosterior,
}

/// Posterior of a regime value from the sequential doubly robust subject scores.
///
/// # Errors
/// The array, positivity and observation refusals of
/// [`evaluate_sequential_dr_value_with_subject_scores`], and the option refusals of
/// [`bayesian_bootstrap_score_posteriors`].
pub fn bayesian_sequential_regime_value(
    outcomes: &[f64],
    histories: &SubjectHistories<'_>,
    sequential: &SequentialDrInputs<'_>,
    options: ScorePosteriorOptions,
    ctx: &ExecutionContext,
) -> Result<SequentialRegimeValuePosterior, EstimationError> {
    let (summary, scores) = evaluate_sequential_dr_value_with_subject_scores(
        outcomes, histories, sequential,
    )
    .map_err(|error| match error.route() {
        Some(record) => EstimationError::Prob(antecedent_prob::ProbError::Route {
            summary: error.message().to_owned(),
            diagnostics: vec![record],
        }),
        None => EstimationError::data_msg(error.message()),
    })?;
    let value =
        bayesian_bootstrap_score_posteriors(&[&scores], options, ctx)?
            .into_iter()
            .next()
            .ok_or_else(|| EstimationError::stats_msg("regime posterior lost its score column"))?;
    Ok(SequentialRegimeValuePosterior { summary, value })
}

#[cfg(test)]
mod tests {
    use super::*;

    const DRAWS: usize = 8_000;

    fn options() -> ScorePosteriorOptions {
        ScorePosteriorOptions { draws: DRAWS, coverage: 0.95 }
    }

    /// Dirichlet(1, ..., 1) posterior standard deviation of a weighted mean of `scores`.
    fn dirichlet_sd(scores: &[f64]) -> f64 {
        let n = crate::util::count(scores.len());
        let centre = scores.iter().sum::<f64>() / n;
        let spread = scores.iter().map(|s| s * s).sum::<f64>() / n - centre * centre;
        (spread / (n + 1.0)).sqrt()
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!((actual - expected).abs() <= tolerance, "actual {actual} expected {expected}");
    }

    struct Fixture {
        outcome: [f64; 4],
        assignment: [bool; 4],
        actions: [bool; 4],
        reference: [bool; 4],
        propensity: [f64; 1],
        costs: [f64; 1],
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                outcome: [3.0, 5.0, 2.0, 1.0],
                assignment: [true, true, false, false],
                actions: [true; 4],
                reference: [false; 4],
                propensity: [0.5],
                costs: [0.0],
            }
        }

        fn design(&self) -> BinaryPolicyDesign<'_> {
            BinaryPolicyDesign {
                assignment: &self.assignment,
                actions: &self.actions,
                propensity: &self.propensity,
                reference: &self.reference,
                costs: &self.costs,
                reference_costs: &self.costs,
            }
        }
    }

    #[test]
    fn inverse_probability_posterior_matches_the_hand_scores() {
        // Treat-all vs treat-none at p = 1/2: policy scores 1{A=1} Y / p = [6, 10, 0, 0],
        // reference scores 1{A=0} Y / (1 - p) = [0, 0, 4, 2].
        let fixture = Fixture::new();
        let ctx = ExecutionContext::for_tests(21);
        let result = bayesian_binary_policy_value(
            &fixture.outcome,
            &fixture.design(),
            None,
            options(),
            &ctx,
        )
        .expect("posterior");
        close(result.policy.score_mean, 4.0, 1.0e-12);
        close(result.reference.score_mean, 1.5, 1.0e-12);
        close(result.incremental.score_mean, 2.5, 1.0e-12);
        let tolerance = 0.06 * dirichlet_sd(&[6.0, 10.0, 0.0, 0.0]);
        close(result.policy.standard_deviation, dirichlet_sd(&[6.0, 10.0, 0.0, 0.0]), tolerance);
        close(
            result.incremental.standard_deviation,
            dirichlet_sd(&[6.0, 10.0, -4.0, -2.0]),
            0.06 * dirichlet_sd(&[6.0, 10.0, -4.0, -2.0]),
        );
        for ((policy, reference), incremental) in
            result.policy.draws.iter().zip(&result.reference.draws).zip(&result.incremental.draws)
        {
            close(*incremental, policy - reference, 1.0e-9);
        }
    }

    #[test]
    fn augmented_posterior_uses_the_frozen_outcome_predictions() {
        // mu1 = 2, mu0 = 1: policy scores [4, 8, 2, 2], reference scores [1, 1, 3, 1].
        let fixture = Fixture::new();
        let mu0 = [1.0; 4];
        let mu1 = [2.0; 4];
        let ctx = ExecutionContext::for_tests(22);
        let result = bayesian_binary_policy_value(
            &fixture.outcome,
            &fixture.design(),
            Some((&mu0, &mu1)),
            options(),
            &ctx,
        )
        .expect("posterior");
        close(result.policy.score_mean, 4.0, 1.0e-12);
        close(result.reference.score_mean, 1.5, 1.0e-12);
        close(result.incremental.score_mean, 2.5, 1.0e-12);
        let expected = dirichlet_sd(&[4.0, 8.0, 2.0, 2.0]);
        close(result.policy.standard_deviation, expected, 0.06 * expected);
        // The outcome model removed variance relative to the inverse-probability scores.
        assert!(result.policy.standard_deviation < dirichlet_sd(&[6.0, 10.0, 0.0, 0.0]));
    }

    #[test]
    fn a_degenerate_propensity_is_refused() {
        let mut fixture = Fixture::new();
        fixture.propensity = [1.0];
        let ctx = ExecutionContext::for_tests(23);
        let result = bayesian_binary_policy_value(
            &fixture.outcome,
            &fixture.design(),
            None,
            options(),
            &ctx,
        );
        assert!(result.is_err());
    }

    #[test]
    fn sequential_regime_posterior_matches_the_hand_scores() {
        // Always-treat regime over two periods with p = 1/2 and no censoring. Hand backward
        // recursion next = q + (next - q) / p on a matched period, next = q otherwise:
        // subject 1 (1, 1), Q (2, 3), Y 5 -> 12; subject 2 (1, 0), Q (2, 3) -> 4;
        // subject 3 (0, 1), Q (1, 2), Y 3 -> 1; subject 4 (0, 0), Q (1, 2) -> 1.
        let treatment = [true, true, true, false, false, true, false, false];
        let probability = [0.5; 8];
        let censoring = [1.0; 8];
        let observed = [true; 4];
        let actions = [true; 8];
        let q = [2.0, 3.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0];
        let outcomes = [5.0, 4.0, 3.0, 0.0];
        let histories = SubjectHistories {
            treatment: &treatment,
            treatment_probability: &probability,
            outcome_observed: &observed,
            censoring_probability: &censoring,
            periods: 2,
            minimum_probability: 0.1,
        };
        let sequential = SequentialDrInputs {
            actions: &actions,
            observation_history: &[true; 8],
            q_predictions: &q,
        };
        let ctx = ExecutionContext::for_tests(24);
        let result =
            bayesian_sequential_regime_value(&outcomes, &histories, &sequential, options(), &ctx)
                .expect("posterior");
        close(result.summary.value, 4.5, 1.0e-12);
        close(result.value.score_mean, 4.5, 1.0e-12);
        let expected = dirichlet_sd(&[12.0, 4.0, 1.0, 1.0]);
        close(result.value.standard_deviation, expected, 0.06 * expected);
        assert!(result.value.interval.0 < 4.5 && 4.5 < result.value.interval.1);
    }

    #[test]
    fn a_positivity_violation_is_refused_for_the_sequential_posterior() {
        let treatment = [true, true];
        let probability = [0.01; 2];
        let censoring = [1.0; 2];
        let observed = [true];
        let actions = [true; 2];
        let q = [1.0; 2];
        let histories = SubjectHistories {
            treatment: &treatment,
            treatment_probability: &probability,
            outcome_observed: &observed,
            censoring_probability: &censoring,
            periods: 2,
            minimum_probability: 0.1,
        };
        let sequential = SequentialDrInputs {
            actions: &actions,
            observation_history: &[true; 2],
            q_predictions: &q,
        };
        let ctx = ExecutionContext::for_tests(25);
        let error =
            bayesian_sequential_regime_value(&[1.0], &histories, &sequential, options(), &ctx)
                .expect_err("positivity is refused");
        let EstimationError::Prob(antecedent_prob::ProbError::Route { diagnostics, .. }) = error
        else {
            panic!("positivity is a route record, not another error: {error}");
        };
        assert!(
            diagnostics
                .iter()
                .any(|record| { record.failed() && record.route_name() == "positivity" })
        );
        assert!(diagnostics.iter().all(|record| record.route_name() != "censoring"));
        assert!(diagnostics.iter().all(|record| !record.is_unsupported_inference()));
    }

    #[test]
    fn a_censoring_violation_is_not_relabeled_as_positivity() {
        let treatment = [true, true];
        let probability = [0.5; 2];
        let censoring = [0.01; 2];
        let observed = [true];
        let actions = [true; 2];
        let q = [1.0; 2];
        let histories = SubjectHistories {
            treatment: &treatment,
            treatment_probability: &probability,
            outcome_observed: &observed,
            censoring_probability: &censoring,
            periods: 2,
            minimum_probability: 0.1,
        };
        let sequential = SequentialDrInputs {
            actions: &actions,
            observation_history: &[true; 2],
            q_predictions: &q,
        };
        let ctx = ExecutionContext::for_tests(26);
        let error =
            bayesian_sequential_regime_value(&[1.0], &histories, &sequential, options(), &ctx)
                .expect_err("censoring is refused");
        let EstimationError::Prob(antecedent_prob::ProbError::Route { diagnostics, .. }) = error
        else {
            panic!("censoring is a route record, not another error: {error}");
        };
        assert!(
            diagnostics
                .iter()
                .any(|record| { record.failed() && record.route_name() == "censoring" })
        );
        assert!(diagnostics.iter().all(|record| record.route_name() != "positivity"));
        assert!(diagnostics.iter().all(|record| !record.is_model_restriction()));
        assert!(diagnostics.iter().all(|record| !record.is_unsupported_inference()));
    }
}
