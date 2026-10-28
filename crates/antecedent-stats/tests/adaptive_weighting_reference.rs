//! Adaptively weighted augmented inverse-probability estimation against 50-digit evaluation of its
//! definition (`oracle_data/adaptive_weighting_reference.py`), and the martingale property that
//! justifies it by exact enumeration of every path of a small adaptive trial.
//!
//! The reference covers two to five arms, prior strengths from 1e-9 to 1e9, outcomes from 1e-6 to
//! 1e3, a prior mean of 1e6, assignment probabilities as small as 1/1024, arm-mean and contrast
//! estimands, both weight schemes, with and without unit multipliers. The enumeration runs a
//! Thompson-like policy over all 2^10 assignment-and-outcome paths of five units and shows that
//! the weighted score sum has expectation zero at the true arm means, for the uniform and the
//! variance-stabilizing weights, which is what makes the plug-in variance a martingale variance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::adaptive_inference::{
    Estimand, OutcomeModelPrior, WeightScheme, adaptive_scores, weighted_estimate,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/adaptive_weighting_reference.json"))
        .expect("frozen reference")
}

fn number(value: &Value) -> f64 {
    match value {
        Value::String(text) => text.parse().expect("number"),
        other => other.as_f64().expect("number"),
    }
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(number).collect()
}

fn never() -> bool {
    false
}

fn close(got: f64, want: f64, relative: f64, what: &str) {
    assert!((got - want).abs() <= 1e-300 + relative * want.abs(), "{what}: {got} vs {want}");
}

#[test]
fn scores_and_weighted_estimates_match_the_reference() {
    let reference = reference();
    let mut compared = 0;
    for case in reference["cases"].as_array().expect("cases") {
        let arms = case["arms"].as_u64().expect("arms") as usize;
        let assigned: Vec<usize> = case["assigned"]
            .as_array()
            .expect("assigned")
            .iter()
            .map(|a| a.as_u64().expect("arm") as usize)
            .collect();
        let probabilities: Vec<f64> = case["probabilities"]
            .as_array()
            .expect("probabilities")
            .iter()
            .flat_map(|row| numbers(row))
            .collect();
        let outcomes = numbers(&case["outcomes"]);
        let prior = OutcomeModelPrior {
            mean: number(&case["prior_mean"]),
            strength: number(&case["strength"]),
        };
        let scores = adaptive_scores(arms, &assigned, &probabilities, &outcomes, prior, &never)
            .expect("scores");
        let want_scores = numbers(&case["scores"]);
        // The scale of a score is its outcome scale plus the prior mean; rounding is relative to it.
        let scale = want_scores.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
        for (g, w) in scores.iter().zip(&want_scores) {
            assert!((g - w).abs() <= 1e-12 * scale + 1e-12 * w.abs(), "score {g} vs {w} in {case}");
        }
        for estimate in case["estimates"].as_array().expect("estimates") {
            let estimand = match estimate["estimand"]["contrast"].as_array() {
                Some(pair) => Estimand::Contrast {
                    plus: pair[0].as_u64().expect("plus") as usize,
                    minus: pair[1].as_u64().expect("minus") as usize,
                },
                None => {
                    Estimand::ArmMean(estimate["estimand"]["arm"].as_u64().expect("arm") as usize)
                }
            };
            let scheme = if estimate["scheme"] == "uniform" {
                WeightScheme::Uniform
            } else {
                WeightScheme::VarianceStabilizing
            };
            let multiplier =
                estimate["multiplier"].as_array().map(|m| m.iter().map(number).collect::<Vec<_>>());
            let floor = number(&estimate["floor"]) * 0.5;
            let got = weighted_estimate(
                arms,
                &probabilities,
                &scores,
                estimand,
                scheme,
                multiplier.as_deref(),
                floor,
            )
            .expect("estimate");
            let what = format!("{estimate}");
            let est = number(&estimate["estimate"]);
            assert!(
                (got.estimate - est).abs() <= 1e-11 * scale.max(est.abs()),
                "estimate {what}: {} vs {est}",
                got.estimate
            );
            let variance = number(&estimate["variance"]);
            assert!(
                (got.variance - variance).abs()
                    <= 1e-10 * variance.max(1e-6 * scale * scale * 1e-12),
                "variance {what}: {} vs {variance}",
                got.variance
            );
            close(
                got.weight_sum,
                number(&estimate["weight_sum"]),
                1e-12,
                &format!("weight sum {what}"),
            );
            close(
                got.effective_units,
                number(&estimate["effective_units"]),
                1e-11,
                &format!("effective units {what}"),
            );
            close(
                got.max_weight_share,
                number(&estimate["max_weight_share"]),
                1e-12,
                &format!("share {what}"),
            );
            compared += 1;
        }
    }
    assert!(compared >= 96, "{compared}");
}

/// Probability that the next unit goes to arm 1 after `n0`, `n1` earlier assignments and `s0`,
/// `s1` earlier successes: a Thompson-like posterior-mean rule clipped to [0.1, 0.9].
fn policy(n: [u32; 2], s: [u32; 2]) -> f64 {
    let m0 = (f64::from(s[0]) + 1.0) / (f64::from(n[0]) + 2.0);
    let m1 = (f64::from(s[1]) + 1.0) / (f64::from(n[1]) + 2.0);
    (m1 / (m0 + m1)).clamp(0.1, 0.9)
}

/// Expected `sum_i h_i (Gamma_i(arm) - mu_arm)` over all paths of five units, using the library's
/// own scores and weights (the weight sum times the estimate minus the truth).
fn expected_weighted_sum(arm: usize, scheme: WeightScheme, mu: [f64; 2], contrast: bool) -> f64 {
    let units = 5;
    let mut expectation = 0.0;
    // Path code: per unit, 2 bits: assigned arm, outcome.
    for code in 0..(1_u32 << (2 * units)) {
        let mut n = [0_u32; 2];
        let mut s = [0_u32; 2];
        let mut probability = 1.0;
        let mut probabilities = Vec::new();
        let mut assigned = Vec::new();
        let mut outcomes = Vec::new();
        for unit in 0..units {
            let a = ((code >> (2 * unit)) & 1) as usize;
            let y = ((code >> (2 * unit + 1)) & 1) as usize;
            let e1 = policy(n, s);
            probabilities.extend([1.0 - e1, e1]);
            probability *= if a == 1 { e1 } else { 1.0 - e1 };
            probability *= if y == 1 { mu[a] } else { 1.0 - mu[a] };
            assigned.push(a);
            outcomes.push(y as f64);
            n[a] += 1;
            s[a] += y as u32;
        }
        let scores = adaptive_scores(
            2,
            &assigned,
            &probabilities,
            &outcomes,
            OutcomeModelPrior { mean: 0.5, strength: 2.0 },
            &never,
        )
        .expect("scores");
        let (estimand, truth) = if contrast {
            (Estimand::Contrast { plus: 1, minus: 0 }, mu[1] - mu[0])
        } else {
            (Estimand::ArmMean(arm), mu[arm])
        };
        let fit = weighted_estimate(2, &probabilities, &scores, estimand, scheme, None, 0.05)
            .expect("estimate");
        expectation += probability * fit.weight_sum * (fit.estimate - truth);
    }
    expectation
}

#[test]
fn the_weighted_score_sum_is_a_martingale_under_an_adaptive_policy() {
    for mu in [[0.3, 0.6], [0.05, 0.95], [0.5, 0.5]] {
        for scheme in [WeightScheme::Uniform, WeightScheme::VarianceStabilizing] {
            for (arm, contrast) in [(0, false), (1, false), (0, true)] {
                let value = expected_weighted_sum(arm, scheme, mu, contrast);
                assert!(
                    value.abs() <= 1e-13,
                    "{mu:?} {scheme:?} arm {arm} contrast {contrast}: {value}"
                );
            }
        }
    }
}
