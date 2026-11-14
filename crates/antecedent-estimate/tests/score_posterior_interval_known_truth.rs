//! The equal-tailed posterior intervals of the sequential regime value and of the
//! inverse-probability policy value against the exact Dirichlet-weighted score law.
//!
//! Both posteriors are the Bayesian bootstrap of a mean of per-subject scores. For distinct
//! scores `s_1..s_n` and Dirichlet(1, ..., 1) weights the mean has the closed-form law
//! `P(S <= x) = 1 - sum_{i: s_i > x} (s_i - x)^(n-1) / prod_{j != i} (s_i - s_j)`, evaluated
//! here from the definition. The scores are worked by hand in the comments: the backward
//! recursion of the regime value and the inverse-probability scores of the policy, neither
//! taken from the product. Each test requires the draws to follow that law on a grid and the
//! published 95% interval to sit at its 2.5% and 97.5% points.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss, reason = "draw counts are small integers")]

use antecedent_core::ExecutionContext;
use antecedent_estimate::policy_value::BinaryPolicyDesign;
use antecedent_estimate::policy_value_posterior::{
    bayesian_binary_policy_value, bayesian_sequential_regime_value,
};
use antecedent_estimate::score_posterior::{ScorePosterior, ScorePosteriorOptions};
use antecedent_stats::longitudinal_regime::{SequentialDrInputs, SubjectHistories};

const DRAWS: usize = 200_000;

/// Exact distribution function of the Dirichlet(1, ..., 1)-weighted mean of distinct scores.
fn exact_cdf(scores: &[f64], x: f64) -> f64 {
    let power = i32::try_from(scores.len() - 1).unwrap();
    let above: f64 = scores
        .iter()
        .enumerate()
        .filter(|(_, score)| **score > x)
        .map(|(i, score)| {
            let spread: f64 = scores
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, other)| score - other)
                .product();
            (score - x).powi(power) / spread
        })
        .sum();
    1.0 - above
}

fn empirical(draws: &[f64], x: f64) -> f64 {
    draws.iter().filter(|draw| **draw <= x).count() as f64 / draws.len() as f64
}

/// The posterior's draws follow the exact law of `scores` and its interval is that law's
/// central 95% interval.
fn assert_exact_law(name: &str, posterior: &ScorePosterior, scores: &[f64]) {
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    assert!((posterior.score_mean - mean).abs() < 1e-12, "{name}: {}", posterior.score_mean);
    let low = scores.iter().copied().fold(f64::INFINITY, f64::min);
    let high = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for step in 1..20 {
        let x = low + (high - low) * f64::from(step) / 20.0;
        let exact = exact_cdf(scores, x);
        let got = empirical(&posterior.draws, x);
        let se = (exact * (1.0 - exact) / DRAWS as f64).sqrt();
        assert!((got - exact).abs() < 5.0 * se + 1e-9, "{name} cdf at {x}: {got} vs {exact}");
    }
    let (lower, upper) = posterior.interval;
    assert!((exact_cdf(scores, lower) - 0.025).abs() < 0.0025, "{name} lower {lower}");
    assert!((exact_cdf(scores, upper) - 0.975).abs() < 0.0025, "{name} upper {upper}");
}

#[test]
fn the_exact_law_is_uniform_for_two_scores_and_triangular_for_three() {
    // Two scores: the mean is uniform between them. Three scores 0, 1, 2: the density is the
    // triangle on [0, 2], so the distribution function at 1/2 is 1/8 and at 1 is 1/2.
    assert!((exact_cdf(&[1.0, 3.0], 1.5) - 0.25).abs() < 1e-15);
    assert!((exact_cdf(&[0.0, 1.0, 2.0], 0.5) - 0.125).abs() < 1e-15);
    assert!((exact_cdf(&[0.0, 1.0, 2.0], 1.0) - 0.5).abs() < 1e-15);
}

#[test]
fn the_sequential_regime_interval_is_the_exact_central_interval_of_the_hand_scores() {
    // Always-treat regime over two periods with p = 1/2 and no censoring. Backward from the
    // outcome: on a period whose action matches the regime, next = q + (next - q) / p;
    // otherwise next = q.
    //   subject 1, actions (1, 1), Q (2, 3), Y 5:   3 + (5 - 3) 2 = 7,    2 + (7 - 2) 2 = 12
    //   subject 2, actions (1, 0), Q (2, 3):        3,                    2 + (3 - 2) 2 = 4
    //   subject 3, actions (0, 1), Q (1, 2), Y 3:   2 + (3 - 2) 2 = 4,    1
    //   subject 4, actions (0, 0), Q (0.5, 2):      2,                    0.5
    //   subject 5, actions (1, 1), Q (1, 2.5), Y 4: 2.5 + (4 - 2.5) 2 = 5.5, 1 + (5.5 - 1) 2 = 10
    let scores = [12.0, 4.0, 1.0, 0.5, 10.0];
    let treatment = [true, true, true, false, false, true, false, false, true, true];
    let q = [2.0, 3.0, 2.0, 3.0, 1.0, 2.0, 0.5, 2.0, 1.0, 2.5];
    let outcomes = [5.0, 4.0, 3.0, 0.0, 4.0];
    let histories = SubjectHistories {
        treatment: &treatment,
        treatment_probability: &[0.5; 10],
        outcome_observed: &[true; 5],
        censoring_probability: &[1.0; 10],
        periods: 2,
        minimum_probability: 0.1,
    };
    let sequential = SequentialDrInputs {
        actions: &[true; 10],
        observation_history: &[true; 10],
        q_predictions: &q,
    };
    let fit = bayesian_sequential_regime_value(
        &outcomes,
        &histories,
        &sequential,
        ScorePosteriorOptions { draws: DRAWS, coverage: 0.95 },
        &ExecutionContext::for_tests(37),
    )
    .unwrap();
    assert!((fit.summary.value - 5.5).abs() < 1e-12, "{}", fit.summary.value);
    assert_exact_law("regime value", &fit.value, &scores);
}

#[test]
fn the_inverse_probability_policy_interval_is_the_exact_central_interval_of_the_hand_scores() {
    // Treat-all against treat-none at p = 1/2 with outcomes 3, 5, 2, 1 and the first two rows
    // treated: policy scores 1{A = 1} Y / p = 6, 10, 0, 0, reference scores
    // 1{A = 0} Y / (1 - p) = 0, 0, 4, 2, and incremental scores 6, 10, -4, -2.
    let design = BinaryPolicyDesign {
        assignment: &[true, true, false, false],
        actions: &[true; 4],
        propensity: &[0.5; 4],
        reference: &[false; 4],
        costs: &[0.0; 4],
        reference_costs: &[0.0; 4],
    };
    let fit = bayesian_binary_policy_value(
        &[3.0, 5.0, 2.0, 1.0],
        &design,
        None,
        ScorePosteriorOptions { draws: DRAWS, coverage: 0.95 },
        &ExecutionContext::for_tests(38),
    )
    .unwrap();
    assert_exact_law("incremental value", &fit.incremental, &[6.0, 10.0, -4.0, -2.0]);
}
