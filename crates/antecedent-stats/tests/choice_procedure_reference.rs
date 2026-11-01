//! The exact law of sequential categorical-choice procedures against an independent reference
//! (`oracle_data/choice_procedure_reference.py`): whole-batch closed-form probabilities in exact
//! rational arithmetic and mpmath posterior quantities, against the product's unit-by-unit
//! dynamic program.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::adaptive_inference::{
    ChoiceKernel, ChoiceProcedureSpec, ChoiceStoppingRule, enumerate_choice_law,
};
use serde_json::Value;

fn number(value: &Value) -> f64 {
    value.as_str().expect("text").parse().expect("number")
}

fn never() -> bool {
    false
}

fn spec(name: &str) -> (ChoiceProcedureSpec, ChoiceKernel) {
    let rule = |p: f64, loss: f64, min: u32| ChoiceStoppingRule {
        min_probability_best: Some(p),
        max_expected_loss: (loss < 1.0).then_some(loss),
        min_observations: min,
    };
    match name {
        "two_options_fixed_clear" => (
            ChoiceProcedureSpec {
                prior: vec![1.0; 2],
                batches: vec![4; 4],
                rule: rule(0.85, 1.0, 4),
            },
            ChoiceKernel::Fixed { theta: vec![0.7, 0.3] },
        ),
        "two_options_fixed_tied" => (
            ChoiceProcedureSpec {
                prior: vec![1.0; 2],
                batches: vec![4; 4],
                rule: rule(0.85, 1.0, 4),
            },
            ChoiceKernel::Fixed { theta: vec![0.5, 0.5] },
        ),
        "two_options_predictive" => (
            ChoiceProcedureSpec {
                prior: vec![1.0; 2],
                batches: vec![3; 5],
                rule: rule(0.9, 1.0, 6),
            },
            ChoiceKernel::PriorPredictive { alpha: vec![1.0; 2] },
        ),
        "three_options_fixed_loss_binding" => (
            ChoiceProcedureSpec {
                prior: vec![1.0; 3],
                batches: vec![5; 3],
                rule: rule(0.8, 0.12, 5),
            },
            ChoiceKernel::Fixed { theta: vec![0.6, 0.25, 0.15] },
        ),
        "three_options_predictive_informative_prior" => (
            ChoiceProcedureSpec {
                prior: vec![2.0; 3],
                batches: vec![4; 3],
                rule: rule(0.75, 1.0, 4),
            },
            ChoiceKernel::PriorPredictive { alpha: vec![2.0; 3] },
        ),
        other => panic!("unknown case {other}"),
    }
}

#[test]
fn the_law_matches_the_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/choice_procedure_reference.json"))
            .expect("frozen reference");
    for case in reference["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let (spec, kernel) = spec(name);
        let law = enumerate_choice_law(&spec, &kernel, &never).expect("law");
        let options = spec.prior.len();
        let (selection, undecided) = law.selection_probabilities(options);
        let close = |a: f64, b: f64, what: &str| {
            assert!((a - b).abs() <= 1e-8, "{name} {what}: {a} vs {b}");
        };
        for (g, w) in selection.iter().zip(case["selection"].as_array().expect("selection")) {
            close(*g, number(w), "selection");
        }
        close(undecided, number(&case["undecided"]), "undecided");
        close(law.expected_enrolled(), number(&case["expected_enrolled"]), "expected enrolled");
        for (g, w) in
            law.rule_stop_probabilities().iter().zip(case["rule_stops"].as_array().expect("stops"))
        {
            close(*g, number(w), "rule stop");
        }
        close(law.atoms.iter().map(|a| a.probability).sum::<f64>(), 1.0, "mass");
    }
}

#[test]
fn bounds_and_domain_are_refused() {
    let base = spec("two_options_fixed_clear").0;
    let fixed = ChoiceKernel::Fixed { theta: vec![0.7, 0.3] };
    let mut five = base.clone();
    five.prior = vec![1.0; 5];
    assert!(
        enumerate_choice_law(&five, &ChoiceKernel::Fixed { theta: vec![0.2; 5] }, &never).is_err()
    );
    let mut bad_rule = base.clone();
    bad_rule.rule.min_probability_best = Some(0.4);
    assert!(enumerate_choice_law(&bad_rule, &fixed, &never).is_err());
    assert!(
        enumerate_choice_law(&base, &ChoiceKernel::Fixed { theta: vec![0.9, 0.3] }, &never)
            .is_err()
    );
    assert!(enumerate_choice_law(&base, &fixed, &|| true).is_err());
}
