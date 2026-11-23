//! Compatible preference sets against an exact-rational reference
//! (`oracle_data/preference_set_learning_reference.py`, `fractions.Fraction`, dyadic inputs).
//!
//! Every case has the value `x0 + sum_k t_k * x_{k+1}`. The reference enumerates the vertices
//! of the learned polytope by brute force over sets of tight constraints and reports each
//! parameter's exact range, or, for contradictory records, the smallest conflicting records and
//! the exact smallest tolerance at which every record becomes compatible.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_design::choice_record::{
    BeliefSource, ChoiceOption, ChoiceRecord, ChoiceRecordSet, OptionBelief, Pooling,
};
use antecedent_design::preference_model::{
    Attribute, LinearRelation, PreferenceExpr as E, PreferenceParameter, PreferenceSet,
    RelationOrigin, RelationSense, ValueModel,
};
use antecedent_design::preference_set_learning::{
    ConflictKind, SetLearningOptions, SetOutcome, learn_compatible_set,
};
use serde_json::Value;

fn number(value: &Value) -> f64 {
    value.as_f64().expect("number")
}

fn index(value: &Value) -> usize {
    usize::try_from(value.as_u64().expect("index")).expect("fits")
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(number).collect()
}

fn relation(position: usize, value: &Value) -> LinearRelation {
    let sense = match value["sense"].as_str().expect("sense") {
        "at_most" => RelationSense::AtMost,
        "at_least" => RelationSense::AtLeast,
        "equal" => RelationSense::Equal,
        other => panic!("unknown sense {other}"),
    };
    LinearRelation {
        id: format!("declared.{position}"),
        terms: value["terms"]
            .as_array()
            .expect("terms")
            .iter()
            .map(|t| (index(&t[0]), number(&t[1])))
            .collect(),
        sense,
        bound: number(&value["bound"]),
        origin: RelationOrigin::Declared,
    }
}

/// `x0 + sum_k t_k * x_{k+1}` with the case's ranges and relations.
fn model(case: &Value) -> ValueModel {
    let bounds = case["bounds"].as_array().expect("bounds");
    let parameters: Vec<PreferenceParameter> = bounds
        .iter()
        .enumerate()
        .map(|(k, range)| PreferenceParameter {
            id: format!("t{k}"),
            units: "utility/unit".to_owned(),
            lower: number(&range[0]),
            upper: number(&range[1]),
            assumption_id: format!("declared_t{k}"),
        })
        .collect();
    let attributes = (0..=parameters.len())
        .map(|k| Attribute { id: format!("x{k}"), units: format!("unit_{k}") })
        .collect();
    let value = (0..parameters.len()).fold(E::Attribute(0), |sum, k| {
        E::sum(sum, E::product(E::Parameter(k), E::Attribute(k + 1)))
    });
    let relations = case["relations"]
        .as_array()
        .expect("relations")
        .iter()
        .enumerate()
        .map(|(i, r)| relation(i, r))
        .collect();
    ValueModel {
        attributes,
        preferences: PreferenceSet { parameters, relations },
        value,
        utility_units: "utility".to_owned(),
    }
}

fn records(case: &Value, model: &ValueModel) -> ChoiceRecordSet {
    let records = case["records"]
        .as_array()
        .expect("records")
        .iter()
        .map(|record| {
            let id = record["id"].as_str().expect("record id").to_owned();
            ChoiceRecord {
                decider: "decider".to_owned(),
                options: record["options"]
                    .as_array()
                    .expect("options")
                    .iter()
                    .map(|o| ChoiceOption {
                        id: o["id"].as_str().expect("option id").to_owned(),
                        belief: OptionBelief::Means(floats(&o["means"])),
                    })
                    .collect(),
                chosen: record["chosen"].as_str().expect("chosen").to_owned(),
                belief_source: BeliefSource::RecordedForecast { source_id: "forecast".to_owned() },
                observation_rows: vec![format!("{id}#row")],
                id,
            }
        })
        .collect();
    ChoiceRecordSet {
        attributes: model.attributes.clone(),
        population: "reference population".to_owned(),
        records,
        pooling: Pooling::PerDecider,
    }
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-9 * (1.0 + want.abs()), "{what}: got {got}, reference {want}");
}

#[test]
fn learned_ranges_and_conflicts_match_the_exact_rational_reference() {
    let root: Value =
        serde_json::from_str(include_str!("oracle_data/preference_set_learning_reference.json"))
            .expect("frozen reference");
    let cases = root["cases"].as_array().expect("cases");
    assert!(cases.len() >= 7);
    for case in cases {
        let name = case["name"].as_str().expect("name");
        let model = model(case);
        let set = records(case, &model);
        let options =
            SetLearningOptions { tolerance: number(&case["tolerance"]), ..Default::default() };
        let result = learn_compatible_set(&model, &set, options).expect(name);
        assert!(result.pooled.is_none() && result.overlap.is_none(), "{name}");
        assert_eq!(result.per_decider.len(), 1, "{name}");
        let expected = &case["expected"];
        match &result.per_decider[0].outcome {
            SetOutcome::Learned(learned) => {
                let want = expected["learned"].as_array().unwrap_or_else(|| {
                    panic!("{name}: learned a set where the reference is contradictory")
                });
                assert_eq!(learned.ranges.len(), want.len(), "{name}");
                for (k, (got, range)) in learned.ranges.iter().zip(want).enumerate() {
                    close(got.learned.0, number(&range[0]), &format!("{name} t{k} lower"));
                    close(got.learned.1, number(&range[1]), &format!("{name} t{k} upper"));
                }
                learned.set.validate().expect("learned set validates");
            }
            SetOutcome::Contradictory(conflict) => {
                let want = &expected["contradictory"];
                assert!(want.is_object(), "{name}: contradictory where the reference learns");
                let ids: Vec<&str> = want["records"]
                    .as_array()
                    .expect("records")
                    .iter()
                    .map(|v| v.as_str().expect("id"))
                    .collect();
                assert_eq!(conflict.records, ids, "{name}");
                assert_eq!(conflict.kind, ConflictKind::Smallest, "{name}");
                close(conflict.tolerance_needed, number(&want["tolerance_needed"]), name);
            }
        }
    }
}
