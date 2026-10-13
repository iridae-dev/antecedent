//! Estimators for adaptively collected data: values against independent
//! oracles, typed refusals, closed guarantee routes and cancellation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::adaptive_inference::{
    ArmCounts, Estimand, GroupSequentialPlan, OutcomeModelPrior, PosteriorRule, Sidedness,
    SpendingFunction, WeightScheme, WeightedHistory, WeightedOptions, adaptively_weighted_estimate,
    anytime_valid_arm_sequences, bayesian_posterior, expected_loss_stopping,
    group_sequential_boundaries, group_sequential_decision, standardized_difference,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected}, got {actual} (tolerance {tolerance})"
    );
}

#[test]
fn the_posterior_is_conjugate_and_carries_its_premise_and_a_closed_guarantee() {
    let counts = [ArmCounts { successes: 7, failures: 3 }, ArmCounts { successes: 2, failures: 8 }];
    let posterior = bayesian_posterior(1.0, 1.0, &counts, &ctx()).unwrap();
    close(posterior.arms[0].alpha, 8.0, 0.0, "alpha");
    close(posterior.arms[0].beta, 4.0, 0.0, "beta");
    close(posterior.arms[0].mean, 8.0 / 12.0, 1e-15, "mean");
    close(posterior.arms[1].alpha, 3.0, 0.0, "alpha of arm 1");
    close(posterior.arms[1].beta, 9.0, 0.0, "beta of arm 1");
    // Beta(8,4) against Beta(3,9): scipy quad.
    close(posterior.arms[0].probability_best, 0.985_013_438_573_810_3, 1e-9, "probability best");
    close(
        posterior.arms[0].probability_best + posterior.arms[1].probability_best,
        1.0,
        1e-9,
        "normalization",
    );
    assert!(posterior.premise.contains("observed history"));
    assert_eq!(posterior.guarantee.code, "cell_not_licensed");
    assert_eq!(posterior.guarantee.detail, "adaptive_inference.posterior_interval_closed");
    let refusal = posterior.guarantee.refusal("a credible interval");
    assert_eq!(refusal.code, "cell_not_licensed");
    assert!(refusal.message.contains("whole-procedure coverage"));
}

#[test]
fn a_posterior_rule_decides_from_the_history_and_claims_no_error_rate() {
    let counts =
        [ArmCounts { successes: 40, failures: 10 }, ArmCounts { successes: 25, failures: 25 }];
    let posterior = bayesian_posterior(1.0, 1.0, &counts, &ctx()).unwrap();
    let stop =
        expected_loss_stopping(&posterior, PosteriorRule::ProbabilityBest { threshold: 0.95 })
            .unwrap();
    assert_eq!(stop.best_arm, 0);
    assert!(stop.stop);
    let keep_going =
        expected_loss_stopping(&posterior, PosteriorRule::ProbabilityBest { threshold: 0.9999999 })
            .unwrap();
    assert!(!keep_going.stop);
    let loss =
        expected_loss_stopping(&posterior, PosteriorRule::ExpectedLoss { loss_threshold: 1e-6 })
            .unwrap();
    assert!(!loss.stop, "the expected loss of the best arm is not below one in a million");
    assert_eq!(stop.guarantee.detail, "adaptive_inference.stopping_reliability_closed");
    assert_eq!(stop.guarantee.code, "cell_not_licensed");
    assert!(
        expected_loss_stopping(&posterior, PosteriorRule::ProbabilityBest { threshold: 0.4 })
            .is_err()
    );
}

#[test]
fn weighted_estimates_match_the_hand_calculation_and_never_clip() {
    let assigned = [0_usize, 1, 1, 0];
    let probabilities = [0.5, 0.5, 0.5, 0.5, 0.3, 0.7, 0.2, 0.8];
    let outcomes = [1.0, 0.0, 1.0, 0.0];
    let history = WeightedHistory {
        arms: 2,
        assigned: &assigned,
        probabilities: &probabilities,
        outcomes: &outcomes,
        multiplier: None,
    };
    let options = WeightedOptions {
        scheme: WeightScheme::VarianceStabilizing,
        prior: OutcomeModelPrior { mean: 0.5, strength: 1.0 },
        probability_floor: 0.05,
    };
    let fit = adaptively_weighted_estimate(
        &history,
        Estimand::Contrast { plus: 1, minus: 0 },
        &options,
        &ctx(),
    )
    .unwrap();
    // Hand calculation with exact fractions and the weights
    // (1/e_a + 1/e_b)^(-1/2): 0.28890584227724664 with variance 0.7743485775674503.
    close(fit.estimate, 0.288_905_842_277_246_64, 1e-12, "estimate");
    close(fit.uncalibrated_variance, 0.774_348_577_567_450_3, 1e-9, "variance");
    close(fit.uncalibrated_standard_error, fit.uncalibrated_variance.sqrt(), 1e-15, "se");
    assert_eq!(fit.units, 4);
    assert_eq!(fit.guarantee.code, "cell_not_licensed");
    // A floor above the smallest probability of an estimand arm is a typed
    // refusal naming the unit and arm, never a clip.
    let strict = WeightedOptions { probability_floor: 0.25, ..options };
    let refusal =
        adaptively_weighted_estimate(&history, Estimand::ArmMean(0), &strict, &ctx()).unwrap_err();
    assert_eq!(refusal.code, "adaptive_assumption_violated");
    assert_eq!(refusal.detail, "adaptive_inference.positivity_floor");
    assert!(refusal.message.contains("unit 3"));
    // Estimating only the arm that was never starved is fine.
    assert!(adaptively_weighted_estimate(&history, Estimand::ArmMean(1), &strict, &ctx()).is_ok());
}

#[test]
fn a_subject_size_multiplier_reweights_the_scores() {
    let assigned = [0_usize, 1, 0, 1];
    let probabilities = [0.5; 8];
    let outcomes = [0.0, 1.0, 0.0, 1.0];
    let multiplier = [1.0, 3.0, 1.0, 1.0];
    let base = WeightedHistory {
        arms: 2,
        assigned: &assigned,
        probabilities: &probabilities,
        outcomes: &outcomes,
        multiplier: None,
    };
    let weighted = WeightedHistory { multiplier: Some(&multiplier), ..base };
    let options = WeightedOptions {
        scheme: WeightScheme::Uniform,
        prior: OutcomeModelPrior { mean: 0.5, strength: 1.0 },
        probability_floor: 0.1,
    };
    let estimand = Estimand::ArmMean(1);
    let plain = adaptively_weighted_estimate(&base, estimand, &options, &ctx()).unwrap();
    let sized = adaptively_weighted_estimate(&weighted, estimand, &options, &ctx()).unwrap();
    assert!((plain.estimate - sized.estimate).abs() > 1e-6);
    assert!(sized.effective_units < plain.effective_units);
    let bad = [1.0, 0.0, 1.0, 1.0];
    let refused = adaptively_weighted_estimate(
        &WeightedHistory { multiplier: Some(&bad), ..base },
        estimand,
        &options,
        &ctx(),
    );
    assert_eq!(refused.unwrap_err().detail, "adaptive_inference.input_invalid");
}

#[test]
fn confidence_sequences_match_the_closed_form_and_union_bound_a_contrast() {
    let mut arm_a = vec![1.0; 150];
    arm_a.extend(vec![0.0; 50]);
    let single = anytime_valid_arm_sequences(&[arm_a.clone()], None, 0.05, 100.0, &ctx()).unwrap();
    // Running intersection computed independently from the closed-form radius.
    close(single.arms[0].lower, 0.861_478_488_644_240_8, 1e-12, "lower");
    close(single.arms[0].upper, 0.865_299_150_331_128_5, 1e-12, "upper");
    close(single.arms[0].mean, 0.75, 1e-15, "mean");
    assert_eq!(single.label, "uncalibrated_confidence_sequence");
    assert_eq!(single.guarantee.code, "cell_not_licensed");
    let arm_b: Vec<f64> = (0..100).map(|i| f64::from(i % 2)).collect();
    let contrast =
        anytime_valid_arm_sequences(&[arm_a, arm_b], Some((0, 1)), 0.05, 100.0, &ctx()).unwrap();
    close(contrast.arm_alpha, 0.025, 0.0, "each arm at alpha / 2");
    let (lower, upper) = contrast.contrast.unwrap();
    close(lower, 0.150_921_659_878_689_55, 1e-12, "contrast lower");
    close(upper, 0.576_952_590_434_661_8, 1e-12, "contrast upper");
    // Outcomes outside [0, 1] are refused, not rescaled.
    let refusal =
        anytime_valid_arm_sequences(&[vec![0.5, 1.5]], None, 0.05, 10.0, &ctx()).unwrap_err();
    assert_eq!(refusal.detail, "adaptive_inference.outcome_invalid");
}

fn plan() -> GroupSequentialPlan {
    GroupSequentialPlan {
        fractions: vec![1.0 / 3.0, 2.0 / 3.0, 1.0],
        alpha: 0.025,
        spending: SpendingFunction::OBrienFlemingType,
        sidedness: Sidedness::OneSidedUpper,
    }
}

#[test]
fn group_sequential_boundaries_and_decisions_follow_the_published_values() {
    let boundaries = group_sequential_boundaries(&plan(), &ctx()).unwrap();
    for (got, want) in boundaries.critical.iter().zip([3.7103, 2.5114, 1.9930]) {
        close(*got, want, 2e-4, "published boundary");
    }
    // Crossing at the second look, with only two looks performed.
    let decision = group_sequential_decision(&plan(), &[1.0, 2.6], &ctx()).unwrap();
    assert_eq!(decision.stopped_at, Some(1));
    assert_eq!(decision.critical.len(), 2);
    // The boundary of a look does not depend on looks that were never made.
    close(decision.critical[1], boundaries.critical[1], 1e-9, "prefix boundary");
    // 2.0 exceeds the final 1.993 but not the 2.5114 of the second look.
    let none = group_sequential_decision(&plan(), &[1.0, 2.0, 1.9], &ctx()).unwrap();
    assert_eq!(none.stopped_at, None);
    let last = group_sequential_decision(&plan(), &[1.0, 2.0, 2.0], &ctx()).unwrap();
    assert_eq!(last.stopped_at, Some(2));
    assert_eq!(last.guarantee.detail, "adaptive_inference.stagewise_adjustment_closed");
    assert_eq!(last.guarantee.code, "cell_not_licensed");
    // More looks than planned, and a non-finite statistic.
    assert!(group_sequential_decision(&plan(), &[0.0; 4], &ctx()).is_err());
    assert!(group_sequential_decision(&plan(), &[f64::NAN], &ctx()).is_err());
}

#[test]
fn the_standardized_difference_matches_a_hand_calculation() {
    let z = standardized_difference(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).unwrap();
    close(z, 2.0 / (5.0_f64 / 3.0).sqrt(), 1e-14, "welch");
    assert!(standardized_difference(&[1.0], &[1.0, 2.0]).is_err());
    assert!(standardized_difference(&[1.0, 1.0], &[2.0, 2.0]).is_err());
}

#[test]
fn cancellation_is_a_refusal_not_a_verdict() {
    let ctx = ExecutionContext::for_tests(5);
    ctx.cancellation.cancel();
    let counts = [ArmCounts { successes: 1, failures: 1 }; 2];
    let refusal = bayesian_posterior(1.0, 1.0, &counts, &ctx).unwrap_err();
    assert_eq!(refusal.code, "cancelled_no_claim");
    assert_eq!(refusal.detail, "adaptive_inference.cancelled");
    let refusal = group_sequential_boundaries(&plan(), &ctx).unwrap_err();
    assert_eq!(refusal.code, "cancelled_no_claim");
}

#[cfg(feature = "calibration-internal")]
#[test]
fn the_calibration_internal_intervals_are_candidates_not_claims() {
    use antecedent_estimate::adaptive_inference::{
        beta_credible_interval_internal, weighted_wald_interval_internal,
    };
    let counts = [ArmCounts { successes: 30, failures: 10 }; 2];
    let posterior = bayesian_posterior(1.0, 1.0, &counts, &ctx()).unwrap();
    let (lower, upper) = beta_credible_interval_internal(&posterior.arms[0], 0.9).unwrap();
    assert!(lower < posterior.arms[0].mean && posterior.arms[0].mean < upper);
    // Equal-tailed 90% interval of Beta(31, 11) has mass 0.9 between the ends.
    let mass = antecedent_stats::adaptive_inference::regularized_incomplete_beta(upper, 31.0, 11.0)
        .unwrap()
        - antecedent_stats::adaptive_inference::regularized_incomplete_beta(lower, 31.0, 11.0)
            .unwrap();
    close(mass, 0.9, 1e-9, "interval mass");
    assert!(beta_credible_interval_internal(&posterior.arms[0], 1.5).is_none());
    // The published guarantee stays closed.
    assert_eq!(posterior.guarantee.code, "cell_not_licensed");
    let _ = weighted_wald_interval_internal;
}
