//! The decision report of a posterior stopping rule against exact rational path enumeration
//! (`oracle_data/procedure_decision_reference.py`): the probability of a rule stop at each look and
//! of none, the expected enrolled units, the probability of deciding each arm, the wrong-decision
//! probability (equal arms are never wrong) and the expected regret, for posterior-expected-loss
//! and futility rules under fixed and Thompson assignment, with and without outcome delay.
//!
//! The posterior probabilities and expected losses that the rules threshold are checked in
//! `crates/antecedent-estimate/tests/adaptive_decision_stopping_posterior_reference.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support/adaptive_procedure_reference.rs"]
mod support;

use antecedent_stats::adaptive_inference::decision_report;
use serde_json::Value;
use support::{never, spec};

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/procedure_decision_reference.json"))
        .expect("frozen reference")
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12, "{what}: {got} vs {want}");
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

#[test]
fn decision_reports_match_the_exact_enumeration() {
    let reference = reference();
    let cases = reference["decisions"].as_array().expect("cases");
    assert!(cases.len() >= 6);
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let spec = spec(&case["declaration"]);
        let truth = support::fracs(&case["truth"]);
        let report = decision_report(&spec, &truth, &never).expect("a report");
        for (look, (&got, want)) in
            report.stop_probabilities.iter().zip(numbers(&case["stop_probabilities"])).enumerate()
        {
            close(got, want, &format!("{name}: stop at look {look}"));
        }
        close(report.horizon_probability, case["horizon_probability"].as_f64().unwrap(), name);
        close(report.expected_enrolled, case["expected_enrolled"].as_f64().unwrap(), name);
        for (arm, (&got, want)) in report
            .decision_probabilities
            .iter()
            .zip(numbers(&case["decision_probabilities"]))
            .enumerate()
        {
            close(got, want, &format!("{name}: decide arm {arm}"));
        }
        close(report.undecided_probability, case["undecided_probability"].as_f64().unwrap(), name);
        close(
            report.wrong_decision_probability,
            case["wrong_decision_probability"].as_f64().unwrap(),
            name,
        );
        close(report.expected_regret, case["expected_regret"].as_f64().unwrap(), name);
    }
}
