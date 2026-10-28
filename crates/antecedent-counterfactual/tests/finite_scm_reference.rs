//! Cross-world queries on finite structural models against brute-force enumeration
//! (`oracle_data/finite_scm_reference.py`).
//!
//! The reference enumerates every joint exogenous state with exact fractions, abduces each unit's
//! posterior from its factual evidence and re-evaluates every world variable by variable. One
//! hundred and twenty random models (two to three worlds, shared exogenous terms, three-level
//! variables, interventions, cross-world routes, partial and empty evidence, probability, joint
//! and mean-contrast quantities) run through the joint enumeration; a chain of seventeen binary
//! exogenous terms (131072 joint states) runs through exact variable elimination.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::RequestedReading;
use antecedent_core::ExecutionContext;
use antecedent_counterfactual::selective::{
    ExogenousInference, ExogenousMethod, ExogenousVariable, FactualUnit, FiniteExecutionRequest,
    FiniteQuantity, FiniteScm, FiniteScmExecutor, FiniteVariable, FiniteWorld, NoiseCoupling,
    WorldAtom, WorldRoute,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/finite_scm_reference.json"))
        .expect("frozen reference")
}

fn index(value: &Value) -> usize {
    value.as_u64().expect("index") as usize
}

fn atom(value: &Value) -> WorldAtom {
    WorldAtom {
        world: index(&value[0]),
        variable: index(&value[1]),
        level: value[2].as_u64().expect("level") as u8,
    }
}

fn scm(model: &Value) -> FiniteScm {
    FiniteScm {
        population: "population".to_owned(),
        history: "history".to_owned(),
        exogenous: model["exogenous"]
            .as_array()
            .expect("exogenous")
            .iter()
            .enumerate()
            .map(|(i, pmf)| ExogenousVariable {
                name: format!("u{i}"),
                pmf: pmf.as_array().expect("pmf").iter().map(|p| p.as_f64().expect("p")).collect(),
            })
            .collect(),
        variables: model["variables"]
            .as_array()
            .expect("variables")
            .iter()
            .map(|v| FiniteVariable {
                name: v["name"].as_str().expect("name").to_owned(),
                levels: index(&v["levels"]),
                parents: v["parents"].as_array().expect("parents").iter().map(index).collect(),
                exogenous: index(&v["exogenous"]),
                table: v["table"]
                    .as_array()
                    .expect("table")
                    .iter()
                    .map(|l| l.as_u64().expect("level") as u8)
                    .collect(),
            })
            .collect(),
    }
}

fn request(case: &Value) -> FiniteExecutionRequest {
    let quantity = &case["quantity"];
    let quantity = match quantity["kind"].as_str().expect("kind") {
        "probability" => FiniteQuantity::Probability(atom(&quantity["atom"])),
        "joint" => FiniteQuantity::JointProbability(
            quantity["atoms"].as_array().expect("atoms").iter().map(atom).collect(),
        ),
        _ => FiniteQuantity::MeanContrast {
            plus: (index(&quantity["plus"][0]), index(&quantity["plus"][1])),
            minus: (index(&quantity["minus"][0]), index(&quantity["minus"][1])),
        },
    };
    let scm = scm(&case["model"]);
    FiniteExecutionRequest {
        evidence_history: scm.history.clone(),
        scm,
        units: case["units"]
            .as_array()
            .expect("units")
            .iter()
            .map(|u| FactualUnit {
                unit: u["unit"].as_str().expect("unit").to_owned(),
                observed: u["observed"]
                    .as_array()
                    .expect("observed")
                    .iter()
                    .map(|o| (index(&o[0]), o[1].as_u64().expect("level") as u8))
                    .collect(),
            })
            .collect(),
        worlds: case["worlds"]
            .as_array()
            .expect("worlds")
            .iter()
            .map(|w| FiniteWorld {
                name: w["name"].as_str().expect("name").to_owned(),
                interventions: w["interventions"]
                    .as_array()
                    .expect("interventions")
                    .iter()
                    .map(|i| (index(&i[0]), i[1].as_u64().expect("level") as u8))
                    .collect(),
                routes: w["routes"]
                    .as_array()
                    .expect("routes")
                    .iter()
                    .map(|r| WorldRoute {
                        parent: index(&r[0]),
                        child: index(&r[1]),
                        source: index(&r[2]),
                    })
                    .collect(),
            })
            .collect(),
        quantity,
        coupling: NoiseCoupling::SharedAbducedExogenous,
        exogenous: ExogenousInference::ExactEnumeration,
        target_population: None,
        reading: RequestedReading::Point,
    }
}

fn check(case: &Value, expected_method: ExogenousMethod, label: &str) {
    let request = request(case);
    let mut executor = FiniteScmExecutor::new();
    let result = executor
        .execute(&request, &ExecutionContext::for_tests(3))
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    assert_eq!(result.method, expected_method, "{label}");
    assert_eq!(result.label, "point_only");
    let values = case["unit_values"].as_array().expect("unit values");
    assert_eq!(result.units.len(), values.len(), "{label}");
    for (got, want) in result.units.iter().zip(values) {
        let (value, evidence) =
            (want["value"].as_f64().expect("value"), want["evidence"].as_f64().expect("evidence"));
        assert!(
            (got.value - value).abs() <= 1e-12,
            "{label}: {} value {} vs {value}",
            got.unit,
            got.value
        );
        assert!(
            (got.evidence_probability - evidence).abs() <= 1e-12 * evidence.max(1e-300),
            "{label}: {} evidence {} vs {evidence}",
            got.unit,
            got.evidence_probability
        );
    }
    let mean = case["mean"].as_f64().expect("mean");
    assert!((result.value - mean).abs() <= 1e-12, "{label}: mean {} vs {mean}", result.value);
}

#[test]
fn random_finite_models_match_brute_force_enumeration() {
    let reference = reference();
    let cases = reference["cases"].as_array().expect("cases");
    for (i, case) in cases[..cases.len() - 1].iter().enumerate() {
        check(case, ExogenousMethod::JointEnumeration, &format!("random model {i}"));
    }
}

#[test]
fn a_model_beyond_the_enumeration_bound_is_executed_exactly_by_variable_elimination() {
    let reference = reference();
    let case = reference["cases"].as_array().expect("cases").last().expect("chain");
    check(case, ExogenousMethod::VariableElimination, "chain of seventeen exogenous terms");
}
