//! The exact spread and the declared-set envelope readings of the finite cross-world executor,
//! against definition-level enumeration with exact fractions
//! (`oracle_data/finite_scm_readings_reference.py`).
//!
//! Sixty random models (shared exogenous terms, three-level variables, interventions, cross-world
//! routes, partial and empty evidence, probability, joint and mean-contrast quantities) give each
//! unit's exact posterior distribution of the quantity over the joint exogenous states and its
//! variance; a seventeen-term chain runs through variable elimination. Envelopes are the range of
//! the point value over a declared set of parameterisations of the same structure.
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
    serde_json::from_str(include_str!("oracle_data/finite_scm_readings_reference.json"))
        .expect("frozen reference")
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
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

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-12, "{what}: {a} vs {b}");
}

fn check_spread(case: &Value, expected_method: ExogenousMethod, label: &str) {
    let mut request = request(case);
    request.reading = RequestedReading::Spread;
    let mut executor = FiniteScmExecutor::new();
    let out = executor.execute(&request, &ctx()).unwrap_or_else(|e| panic!("{label}: {e:?}"));
    assert_eq!(out.method, expected_method, "{label}");
    close(out.value, case["mean"].as_f64().unwrap(), label);
    let spread = out.spread.as_ref().expect("a spread was requested");
    assert_eq!(spread.label, "exact_noise_spread");
    let expected = &case["spread"];
    close(spread.mean_variance, expected["mean_variance"].as_f64().unwrap(), label);
    let units = expected["units"].as_array().unwrap();
    assert_eq!(spread.units.len(), units.len(), "{label}");
    for (got, want) in spread.units.iter().zip(units) {
        assert_eq!(got.unit, want["unit"].as_str().unwrap());
        close(got.variance, want["variance"].as_f64().unwrap(), label);
        let dist = want["distribution"].as_array().unwrap();
        assert_eq!(got.distribution.len(), dist.len(), "{label} {}", got.unit);
        for ((v, w), d) in got.distribution.iter().zip(dist) {
            close(*v, d[0].as_f64().unwrap(), label);
            close(*w, d[1].as_f64().unwrap(), label);
        }
    }
}

/// The spread of every random model is the exact enumerated posterior distribution and variance.
#[test]
fn the_spread_reading_matches_the_exact_posterior_distribution_of_every_unit() {
    let reference = reference();
    for (i, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        check_spread(case, ExogenousMethod::JointEnumeration, &format!("case {i}"));
    }
}

/// Past the enumeration bound an indicator quantity's spread is p(1-p) of the eliminated value.
#[test]
fn the_spread_reading_past_the_enumeration_bound_is_the_indicator_variance() {
    let reference = reference();
    check_spread(&reference["chain"], ExogenousMethod::VariableElimination, "chain");
}

/// A mean contrast has no elimination second moment: refused, with no work retained.
#[test]
fn a_mean_contrast_spread_past_the_enumeration_bound_is_refused_with_its_blocker() {
    let reference = reference();
    let mut case = reference["chain"].clone();
    case["quantity"] = serde_json::json!({"kind": "contrast", "plus": [1, 16], "minus": [0, 16]});
    let mut request = request(&case);
    request.reading = RequestedReading::Spread;
    let mut executor = FiniteScmExecutor::new();
    let err = executor.execute(&request, &ctx()).expect_err("refused");
    let r = err.refusal().expect("typed");
    assert_eq!(r.detail, "cross_world_execution.spread_needs_enumeration");
    assert_eq!(executor.counters().abductions, 0);
}

/// The point reading carries no spread, and the requested readings stay distinct.
#[test]
fn the_point_reading_carries_no_spread() {
    let reference = reference();
    let case = &reference["cases"][0];
    let mut executor = FiniteScmExecutor::new();
    let out = executor.execute(&request(case), &ctx()).unwrap();
    assert!(out.spread.is_none());
    assert_eq!(out.label, "point_only");
}

/// A degenerate posterior has zero spread and a one-point distribution.
#[test]
fn a_fully_determined_unit_has_a_point_mass_distribution_and_zero_variance() {
    let reference = reference();
    let mut seen = 0;
    for case in reference["cases"].as_array().unwrap() {
        for u in case["spread"]["units"].as_array().unwrap() {
            if u["distribution"].as_array().unwrap().len() == 1 {
                assert_eq!(u["variance_exact"].as_str().unwrap(), "0/1");
                seen += 1;
            }
        }
    }
    assert!(seen > 0, "the sample contains deterministic units");
    check_spread(&reference["cases"][0], ExogenousMethod::JointEnumeration, "first");
}

#[test]
fn the_envelope_is_the_range_of_the_exact_point_value_over_the_declared_set() {
    let reference = reference();
    let mut checked = 0;
    for (i, case) in reference["cases"].as_array().unwrap().iter().enumerate() {
        let Some(env) = case.get("envelope") else { continue };
        let members = env["members"].as_array().unwrap();
        let mut request = request(case);
        request.scm = scm(&members[0]);
        request.reading = RequestedReading::Envelope;
        let alternatives: Vec<FiniteScm> = members[1..].iter().map(scm).collect();
        let mut executor = FiniteScmExecutor::new();
        let out = executor
            .execute_envelope(&request, &alternatives, &ctx())
            .unwrap_or_else(|e| panic!("case {i}: {e:?}"));
        assert_eq!(out.label, "declared_set_envelope");
        close(out.lower, env["lower"].as_f64().unwrap(), "lower");
        close(out.upper, env["upper"].as_f64().unwrap(), "upper");
        assert_eq!(out.members.len(), members.len());
        for (m, want) in out.members.iter().zip(env["member_means"].as_array().unwrap()) {
            close(m.value, want.as_f64().unwrap(), "member");
        }
        for (k, u) in out.units.iter().enumerate() {
            close(u.lower, env["unit_lower"][k].as_f64().unwrap(), "unit lower");
            close(u.upper, env["unit_upper"][k].as_f64().unwrap(), "unit upper");
        }
        // The attaining members hold the extremes (the first on ties).
        close(out.members[out.lower_member].value, out.lower, "lower member");
        close(out.members[out.upper_member].value, out.upper, "upper member");
        checked += 1;
    }
    assert!(checked >= 10, "{checked} envelopes checked");
}

#[test]
fn the_envelope_is_refused_where_the_declared_set_cannot_be_honoured() {
    let reference = reference();
    let case = reference["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c.get("envelope").is_some())
        .unwrap();
    let members = case["envelope"]["members"].as_array().unwrap();
    let mut request = request(case);
    request.scm = scm(&members[0]);
    let alternatives: Vec<FiniteScm> = members[1..].iter().map(scm).collect();
    let mut exec = FiniteScmExecutor::new();
    let code = |e: antecedent_counterfactual::selective::CrossWorldExecutionError| {
        e.refusal().expect("typed").detail.clone()
    };

    // execute() has no parameter set to range over.
    request.reading = RequestedReading::Envelope;
    assert_eq!(
        code(exec.execute(&request, &ctx()).expect_err("refused")),
        "cross_world_execution.envelope_needs_parameter_set"
    );
    // The wrong reading, an empty set and a member of other structure.
    request.reading = RequestedReading::Point;
    assert_eq!(
        code(exec.execute_envelope(&request, &alternatives, &ctx()).expect_err("refused")),
        "cross_world_execution.envelope_reading_required"
    );
    request.reading = RequestedReading::Envelope;
    assert_eq!(
        code(exec.execute_envelope(&request, &[], &ctx()).expect_err("refused")),
        "cross_world_execution.envelope_parameter_set_empty"
    );
    let mut other = alternatives[0].clone();
    other.variables.pop();
    assert_eq!(
        code(exec.execute_envelope(&request, &[other], &ctx()).expect_err("refused")),
        "cross_world_execution.envelope_member_structure_differs"
    );
    assert_eq!(exec.counters().abductions, 0, "the refusals did no work");
}

/// Interval and posterior readings stay closed by the joint-law rule.
#[test]
fn interval_and_posterior_readings_stay_closed_beside_the_spread() {
    let reference = reference();
    for reading in [RequestedReading::Interval, RequestedReading::Posterior] {
        let mut request = request(&reference["cases"][0]);
        request.reading = reading;
        let mut exec = FiniteScmExecutor::new();
        let err = exec.execute(&request, &ctx()).expect_err("closed");
        assert_eq!(err.refusal().unwrap().code, "cell_not_licensed");
    }
}
