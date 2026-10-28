//! The Beta-Bernoulli posterior functionals of the adaptive-inference module against 15-digit
//! tanh-sinh quadrature and exact rational sums (`oracle_data/beta_bernoulli_reference.py`).
//!
//! The probability that each arm is best, the probability that one arm exceeds another by a
//! margin, the posterior expected loss and the regularized incomplete beta function, for flat to
//! concentrated posteriors (shapes up to 4000), skewed shapes (1, 40) and non-integer shapes; and,
//! for integer shapes, the two-arm superiority as an exact rational sum from the binomial tail.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::adaptive_inference::{
    BetaPosterior, best_arm_probabilities, expected_losses, probability_exceeds,
    probability_superior, regularized_incomplete_beta,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/beta_bernoulli_reference.json"))
        .expect("frozen reference")
}

fn number(value: &Value) -> f64 {
    match value {
        Value::String(text) => text.parse().expect("number"),
        other => other.as_f64().expect("number"),
    }
}

fn never() -> bool {
    false
}

fn arms(value: &Value) -> Vec<BetaPosterior> {
    value
        .as_array()
        .expect("shapes")
        .iter()
        .map(|s| BetaPosterior::new(number(&s[0]), number(&s[1])).expect("posterior"))
        .collect()
}

#[test]
fn best_arm_probabilities_and_losses_match_the_reference() {
    let reference = reference();
    for case in reference["best"].as_array().expect("best") {
        let got = best_arm_probabilities(&arms(&case["shapes"]), &never).expect("probabilities");
        for (g, w) in got.iter().zip(case["probabilities"].as_array().expect("want")) {
            assert!((g - number(w)).abs() <= 1e-10, "{case}: {g} vs {w}");
        }
        assert!((got.iter().sum::<f64>() - 1.0).abs() <= 1e-10, "{case}");
    }
    for case in reference["losses"].as_array().expect("losses") {
        let got = expected_losses(&arms(&case["shapes"]), &never).expect("losses");
        for (g, w) in got.iter().zip(case["losses"].as_array().expect("want")) {
            assert!((g - number(w)).abs() <= 1e-10, "{case}: {g} vs {w}");
        }
    }
}

#[test]
fn margin_exceedance_matches_the_reference() {
    let reference = reference();
    for case in reference["exceeds"].as_array().expect("exceeds") {
        let a = BetaPosterior::new(number(&case["a"][0]), number(&case["a"][1])).expect("a");
        let b = BetaPosterior::new(number(&case["b"][0]), number(&case["b"][1])).expect("b");
        let margin = number(&case["margin"]);
        let got = probability_exceeds(a, b, margin, &never).expect("exceedance");
        let want = number(&case["p"]);
        assert!((got - want).abs() <= 1e-10, "{case}: {got} vs {want}");
        if margin == 0.0 {
            let superior = probability_superior(a, b, &never).expect("superiority");
            assert!((superior - got).abs() <= 1e-12, "{case}");
        }
    }
}

#[test]
fn the_incomplete_beta_function_matches_the_reference() {
    let reference = reference();
    for case in reference["incomplete_beta"].as_array().expect("incomplete beta") {
        let got =
            regularized_incomplete_beta(number(&case["x"]), number(&case["a"]), number(&case["b"]))
                .expect("incomplete beta");
        let want = number(&case["value"]);
        assert!((got - want).abs() <= 1e-10 + 1e-10 * want, "{case}: {got} vs {want}");
    }
}

#[test]
fn integer_shape_superiority_equals_the_exact_rational_sum() {
    let reference = reference();
    for case in reference["exact_superior"].as_array().expect("exact") {
        let shapes = arms(&case["shapes"]);
        let got = probability_superior(shapes[0], shapes[1], &never).expect("superiority");
        let want = number(&case["float"]);
        assert!((got - want).abs() <= 1e-10, "{case}: {got} vs {want}");
    }
}
