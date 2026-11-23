//! Expected linear-form terms of recorded choice options against an exact-rational reference
//! (`oracle_data/choice_record_reference.py`, `fractions.Fraction`, dyadic beliefs).
//!
//! Three value models: two rates; a rate plus a kink whose location is a fixed shape parameter
//! and a parameter-free fixed part; a three-segment piecewise-linear curve plus a fixed rate.
//! Beliefs are attribute means (affine model only) or small draw tables with and without
//! weights. The reference writes each part out as a direct function of the attributes.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordSet, OptionBelief, Pooling,
    choice_features,
};
use antecedent_design::preference_model::{
    Attribute, PreferenceExpr as E, PreferenceParameter, PreferenceSet, ValueModel,
};
use serde_json::Value;

fn attributes() -> Vec<Attribute> {
    vec![
        Attribute { id: "x0".to_owned(), units: "unit_a".to_owned() },
        Attribute { id: "x1".to_owned(), units: "unit_b".to_owned() },
    ]
}

fn parameter(id: &str, lower: f64, upper: f64) -> PreferenceParameter {
    PreferenceParameter {
        id: id.to_owned(),
        units: "utility/unit".to_owned(),
        lower,
        upper,
        assumption_id: format!("declared_{id}"),
    }
}

#[allow(clippy::unnecessary_box_returns, reason = "expression variants hold boxed operands")]
fn b(e: E) -> Box<E> {
    Box::new(e)
}

fn mul(a: E, c: E) -> E {
    E::Mul(b(a), b(c))
}

fn add(a: E, c: E) -> E {
    E::Add(b(a), b(c))
}

/// `clamp(x, from, to) - from`.
fn segment(attribute: usize, from: f64, to: f64) -> E {
    E::Sub(
        b(E::Min(b(E::Max(b(E::Attribute(attribute)), b(E::Const(from)))), b(E::Const(to)))),
        b(E::Const(from)),
    )
}

fn model(name: &str) -> ValueModel {
    let (parameters, value) = match name {
        "two_rates" => (
            vec![parameter("t0", -4.0, 4.0), parameter("t1", -4.0, 4.0)],
            add(mul(E::Parameter(0), E::Attribute(0)), mul(E::Parameter(1), E::Attribute(1))),
        ),
        "rate_and_kink" => (
            vec![parameter("t0", -4.0, 4.0), parameter("t1", -4.0, 4.0), parameter("t2", 1.5, 1.5)],
            add(
                add(
                    mul(E::Parameter(0), E::Attribute(0)),
                    mul(
                        E::Parameter(1),
                        E::Max(b(E::Sub(b(E::Attribute(1)), b(E::Parameter(2)))), b(E::Const(0.0))),
                    ),
                ),
                mul(E::Const(0.5), E::Attribute(0)),
            ),
        ),
        "three_segment" => (
            vec![
                parameter("s0", 0.0, 4.0),
                parameter("s1", 0.0, 4.0),
                parameter("s2", 0.0, 4.0),
                parameter("t3", -2.0, -2.0),
            ],
            add(
                add(
                    add(
                        mul(E::Parameter(0), segment(0, 0.0, 1.0)),
                        mul(E::Parameter(1), segment(0, 1.0, 2.0)),
                    ),
                    mul(E::Parameter(2), segment(0, 2.0, 4.0)),
                ),
                mul(E::Parameter(3), E::Attribute(1)),
            ),
        ),
        other => panic!("unknown reference model {other}"),
    };
    ValueModel {
        attributes: attributes(),
        preferences: PreferenceSet { parameters, relations: Vec::new() },
        value,
        utility_units: "utility".to_owned(),
    }
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

fn belief(value: &Value) -> OptionBelief {
    if value["means"].is_array() {
        OptionBelief::Means(floats(&value["means"]))
    } else {
        OptionBelief::Draws {
            rows: floats(&value["rows"]),
            weights: if value["weights"].is_null() {
                None
            } else {
                Some(floats(&value["weights"]))
            },
        }
    }
}

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
        "{what}: got {got}, reference {want}"
    );
}

#[test]
fn option_terms_match_the_exact_rational_reference() {
    let root: Value =
        serde_json::from_str(include_str!("oracle_data/choice_record_reference.json"))
            .expect("frozen reference");
    let cases = root["cases"].as_array().expect("cases");
    assert!(cases.len() >= 9);
    for case in cases {
        let name = case["model"].as_str().expect("model name");
        let model = model(name);
        let records: Vec<ChoiceRecord> = case["records"]
            .as_array()
            .expect("records")
            .iter()
            .map(|record| {
                let id = record["id"].as_str().expect("record id").to_owned();
                let options: Vec<ChoiceOption> = record["options"]
                    .as_array()
                    .expect("options")
                    .iter()
                    .map(|o| ChoiceOption {
                        id: o["id"].as_str().expect("option id").to_owned(),
                        belief: belief(&o["belief"]),
                    })
                    .collect();
                ChoiceRecord {
                    chosen: options[0].id.clone(),
                    decider: "decider".to_owned(),
                    options,
                    belief_source: BeliefSource::RecordedForecast {
                        source_id: "forecast".to_owned(),
                    },
                    observation_rows: vec![format!("{id}#row")],
                    id,
                }
            })
            .collect();
        let set = ChoiceRecordSet {
            attributes: attributes(),
            population: "reference population".to_owned(),
            records,
            pooling: Pooling::PerDecider,
        };
        let features = choice_features(&model, &set).expect("features");
        let n = usize::try_from(case["n_parameters"].as_u64().expect("count")).expect("fits");
        assert_eq!(features.n_parameters, n, "{name}");
        assert_eq!(features.sets.len(), case["records"].as_array().expect("records").len());
        for (got, want) in features.sets.iter().zip(case["records"].as_array().expect("records")) {
            assert_eq!(got.record_id, want["id"].as_str().expect("id"));
            let fixed = floats(&want["fixed"]);
            let terms = floats(&want["terms"]);
            assert_eq!(got.fixed.len(), fixed.len());
            assert_eq!(got.terms.len(), terms.len());
            for (i, (g, w)) in got.fixed.iter().zip(&fixed).enumerate() {
                close(*g, *w, &format!("{} fixed[{i}]", got.record_id));
            }
            for (i, (g, w)) in got.terms.iter().zip(&terms).enumerate() {
                close(*g, *w, &format!("{} terms[{i}]", got.record_id));
            }
        }
    }
}
