//! The Dirichlet categorical-choice posterior functionals against 20-digit mpmath references
//! (`oracle_data/dirichlet_choice_reference.py`): the probability that each option has the largest
//! share, the posterior expected loss, and the exact ordered-sequence log marginal likelihood, for
//! flat to concentrated posteriors, near-flat and non-integer concentrations, and up to
//! eight options.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::adaptive_inference::DirichletPosterior;
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/dirichlet_choice_reference.json"))
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

#[test]
fn best_option_probabilities_match_the_reference() {
    for case in reference()["best"].as_array().expect("best") {
        let posterior = DirichletPosterior::new(numbers(&case["alphas"])).expect("posterior");
        let got = posterior.best_option_probabilities(&never).expect("probabilities");
        for (g, w) in got.iter().zip(numbers(&case["probabilities"])) {
            assert!((g - w).abs() <= 1e-9, "{case}: {g} vs {w}");
        }
        assert!((got.iter().sum::<f64>() - 1.0).abs() <= 1e-9, "{case}");
    }
}

#[test]
fn expected_losses_match_the_reference() {
    for case in reference()["losses"].as_array().expect("losses") {
        let posterior = DirichletPosterior::new(numbers(&case["alphas"])).expect("posterior");
        let got = posterior.expected_losses(&never).expect("losses");
        for (g, w) in got.iter().zip(numbers(&case["losses"])) {
            assert!((g - w).abs() <= 1e-9, "{case}: {g} vs {w}");
        }
    }
}

#[test]
fn log_marginal_likelihood_matches_the_reference() {
    for case in reference()["log_ml"].as_array().expect("log_ml") {
        let got = DirichletPosterior::log_marginal_likelihood(
            &numbers(&case["alphas"]),
            &numbers(&case["counts"]),
        )
        .expect("marginal likelihood");
        let want = number(&case["value"]);
        assert!((got - want).abs() <= 1e-10 * want.abs().max(1.0), "{case}: {got} vs {want}");
    }
}
