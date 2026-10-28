//! The evidence-requirement solves and the finite Bayes posterior against exact-rational and
//! 60-digit evaluation of their definitions (`oracle_data/evidence_requirement_reference.py`).
//!
//! Required Bayes factors across priors from 1e-12 to 1 - 2^-30 and levels from 1e-9 to 1 - 2^-20;
//! the smallest number of identical binary-signal observations (exact rational powers, signals that
//! settle the hypothesis in one reading, signals that cannot move the probability, a budget that
//! is exhausted and a Bayes factor of 1 + 2^-19 needing 575991 readings); the Beta-Bernoulli
//! count (an exact rational ceiling); finite Bayes posteriors with counts to 6e8; and the conjugate
//! update at shapes from 1e-3 to 1e300 and counts to 2^52.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_model::observation_query::{
    BetaBernoulli, Direction, FiniteBayesModel, ObservationQueryError, UnreachableBound,
    observations_required_binary_signal, observations_required_conjugate, required_bayes_factor,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/evidence_requirement_reference.json"))
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

fn direction(value: &Value) -> Direction {
    if value == "raise" { Direction::Raise } else { Direction::Lower }
}

fn close(got: f64, want: f64, relative: f64, absolute: f64, what: &str) {
    assert!((got - want).abs() <= absolute + relative * want.abs(), "{what}: {got} vs {want}");
}

#[test]
fn required_bayes_factors_are_the_exact_odds_ratio() {
    let reference = reference();
    for case in reference["bayes_factor"].as_array().expect("cases") {
        let got = required_bayes_factor(
            number(&case["prior"]),
            number(&case["level"]),
            direction(&case["direction"]),
        )
        .expect("factor");
        assert_eq!(
            got.already_satisfied,
            case["already_satisfied"].as_bool().expect("flag"),
            "{case}"
        );
        close(got.factor, number(&case["factor"]), 1e-12, 0.0, &format!("{case}"));
    }
}

#[test]
fn binary_signal_counts_match_exact_arithmetic() {
    let reference = reference();
    for case in reference["signal"].as_array().expect("cases") {
        let result = observations_required_binary_signal(
            number(&case["prior"]),
            number(&case["level"]),
            direction(&case["direction"]),
            number(&case["rate_h"]),
            number(&case["rate_c"]),
            case["max"].as_u64().expect("max"),
        );
        let what = format!("{case}");
        match case["status"].as_str().expect("status") {
            "satisfied" => {
                let got = result.expect("satisfied");
                assert!(got.already_satisfied && got.observations == 0, "{what}");
            }
            "ok" => {
                let got = result.unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(got.observations, case["n"].as_u64().expect("n"), "{what}");
                close(got.value_at_requirement, number(&case["value"]), 1e-9, 1e-12, &what);
                if let Some(bf) = case["bf"].as_str() {
                    close(
                        got.bayes_factor_per_observation.expect("factor"),
                        bf.parse().expect("bf"),
                        1e-12,
                        0.0,
                        &what,
                    );
                }
            }
            "no_move" => assert!(
                matches!(
                    result,
                    Err(ObservationQueryError::LevelUnreachable {
                        bound: UnreachableBound::SignalDoesNotMove,
                        ..
                    })
                ),
                "{what}: {result:?}"
            ),
            _ => assert!(
                matches!(
                    result,
                    Err(ObservationQueryError::LevelUnreachable {
                        bound: UnreachableBound::ObservationBudget { .. },
                        ..
                    })
                ),
                "{what}: {result:?}"
            ),
        }
    }
}

#[test]
fn conjugate_counts_are_the_exact_rational_ceiling() {
    let reference = reference();
    for case in reference["conjugate"].as_array().expect("cases") {
        let prior = BetaBernoulli::new(number(&case["a"]), number(&case["b"])).expect("prior");
        let got = observations_required_conjugate(
            &prior,
            number(&case["level"]),
            direction(&case["direction"]),
            1 << 24,
        )
        .unwrap_or_else(|e| panic!("{case}: {e}"));
        let what = format!("{case}");
        if case["status"] == "satisfied" {
            assert!(got.already_satisfied && got.observations == 0, "{what}");
            close(got.value_at_requirement, number(&case["mean"]), 1e-14, 0.0, &what);
        } else {
            assert_eq!(got.observations, case["n"].as_u64().expect("n"), "{what}");
            close(got.value_at_requirement, number(&case["value"]), 1e-12, 0.0, &what);
        }
    }
}

#[test]
fn finite_bayes_posteriors_match_the_reference_at_large_counts() {
    let reference = reference();
    for case in reference["bayes_models"].as_array().expect("models") {
        let prior = numbers(&case["prior"]);
        let likelihood = numbers(&case["likelihood"]);
        let levels = case["levels"].as_u64().expect("levels") as usize;
        let model = FiniteBayesModel::new(
            (0..prior.len()).map(|h| format!("h{h}")).collect(),
            prior,
            (0..levels).map(|o| format!("o{o}")).collect(),
            likelihood,
        )
        .expect("model");
        let counts: Vec<u64> = case["counts"]
            .as_array()
            .expect("counts")
            .iter()
            .map(|c| c.as_u64().expect("count"))
            .collect();
        let got = model.posterior_from_counts(&counts).expect("posterior");
        let what = format!("{case}");
        for ((_, p), w) in got.hypotheses.iter().zip(numbers(&case["posterior"])) {
            close(*p, w, 1e-9, 1e-12, &format!("posterior {what}"));
        }
        let evidence = number(&case["log_evidence"]);
        close(got.log_evidence, evidence, 1e-11, 1e-11, &format!("log evidence {what}"));
        for ((_, p), w) in got.predictive.iter().zip(numbers(&case["predictive"])) {
            close(*p, w, 1e-9, 1e-12, &format!("predictive {what}"));
        }
        // The stream form, in any order, is the same posterior as the tally.
        if counts.iter().sum::<u64>() <= 200 {
            let names: Vec<String> = counts
                .iter()
                .enumerate()
                .flat_map(|(o, c)| vec![format!("o{o}"); *c as usize])
                .collect();
            let mut stream: Vec<&str> = names.iter().map(String::as_str).collect();
            stream.reverse();
            let again = model.posterior(&stream).expect("stream posterior");
            for ((_, a), (_, b)) in again.hypotheses.iter().zip(&got.hypotheses) {
                close(*a, *b, 1e-12, 1e-14, "order does not matter");
            }
        }
    }
}

#[test]
fn beta_updates_are_exact() {
    let reference = reference();
    for case in reference["beta"].as_array().expect("cases") {
        let prior = BetaBernoulli::new(number(&case["a"]), number(&case["b"])).expect("prior");
        close(prior.mean(), number(&case["mean_prior"]), 1e-14, 0.0, &format!("{case}"));
        let updated = prior
            .update(case["s"].as_u64().expect("s"), case["f"].as_u64().expect("f"))
            .expect("update");
        close(updated.mean(), number(&case["mean_post"]), 1e-14, 0.0, &format!("{case}"));
        close(updated.alpha(), number(&case["alpha"]), 1e-15, 0.0, &format!("{case}"));
        close(updated.beta(), number(&case["beta"]), 1e-15, 0.0, &format!("{case}"));
    }
}
