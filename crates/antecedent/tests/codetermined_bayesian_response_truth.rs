//! The Bayesian joint `InterventionResponse` on a `CoDetermined` tier closure against the
//! closed form of an additive outcome law.
//!
//! The fixture `conformance/estimate/codetermined_joint_level` states the design as exact cell
//! counts over `(z, t1, t2)` (the treatments are confounded by `z`) and the outcome law
//! `y = 0.2 + 0.4 t1 + 0.3 t2 + 0.2 z` up to a small deterministic wiggle. The closure
//! adjusts for `{z}`; the outcome regression of the Bayesian cell is additive in the
//! treatments, which is exactly the law, so the joint level `E[y | do(t1 = 1, t2 = 1)]` is
//! `0.2 + 0.4 + 0.3 + 0.2 mean(z)` with `mean(z)` taken from the exact counts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{TieredBackground, WithinTier};

const FIXTURE: &str =
    include_str!("../../../conformance/estimate/codetermined_joint_level/expected.json");

struct Fixture {
    cells: Vec<([f64; 3], usize)>,
    level: f64,
    tolerance: f64,
}

fn fixture() -> Fixture {
    let document: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let cells = document["design"]["cells"]
        .as_array()
        .unwrap()
        .iter()
        .map(|cell| {
            let at = |key: &str| cell[key].as_f64().unwrap();
            ([at("z"), at("t1"), at("t2")], cell["count"].as_u64().unwrap() as usize)
        })
        .collect();
    Fixture {
        cells,
        level: document["level"]["value"].as_f64().unwrap(),
        tolerance: document["tolerance"]["posterior_mean"].as_f64().unwrap(),
    }
}

/// Columns `z, t1, t2, y` from the fixture's counts; the wiggle is `0.01 sin(0.31 i)`.
fn data(fixture: &Fixture) -> TabularData {
    let (mut z, mut t1, mut t2, mut y) = (vec![], vec![], vec![], vec![]);
    for (arm, count) in &fixture.cells {
        for _ in 0..*count {
            let i = z.len() as f64;
            z.push(arm[0]);
            t1.push(arm[1]);
            t2.push(arm[2]);
            y.push(0.2 + 0.4 * arm[1] + 0.3 * arm[2] + 0.2 * arm[0] + 0.01 * (0.31 * i).sin());
        }
    }
    TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t1", t1.as_slice()),
        ("t2", t2.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

fn query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(3),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(2), Value::f64(1.0)),
        ]),
    })
}

fn build(data: TabularData, inference: InferenceMode) -> Study {
    let background = TieredBackground::from_named(
        data.schema(),
        &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
        WithinTier::CoDetermined,
    )
    .unwrap();
    Study::tabular(data)
        .tiered_background(background)
        .unwrap()
        .query(CausalQuery::Response(query()))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn value(result: &antecedent::StudyResult) -> f64 {
    match &result.response().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        other => panic!("expected a point-identified joint level, got {other:?}"),
    }
}

/// `InterventionResponse` x `CoDetermined` x explicit x Bayesian x none: the posterior mean of
/// the joint level matches the closed form, and the prepared click reproduces the one-shot run
/// from the retained plan with the builder dropped.
#[test]
fn codetermined_bayesian_joint_level_matches_the_closed_form_additive_law() {
    let fixture = fixture();
    let data = data(&fixture);
    let context = ExecutionContext::for_tests(9_121);
    let inference =
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400).prior_scale(1_000.0));
    let builder = build(data.clone(), inference);
    let one_shot = builder.run(&context).unwrap();
    let prepared = builder.prepare(&context).unwrap();
    drop(builder);
    let plan = &prepared.plan().logical.record;
    assert_eq!(plan.identifier.as_deref(), Some("generalized.adjustment"));
    let click = prepared.estimate(&data, &context).unwrap();
    for result in [&one_shot, &click] {
        assert_eq!(result.support_status().unwrap().as_str(), "licensed");
        assert!(result.refutations.is_empty());
        let got = value(result);
        assert!(
            (got - fixture.level).abs() < fixture.tolerance,
            "joint level {got} vs closed form {}",
            fixture.level
        );
    }
    assert!((value(&click) - value(&one_shot)).abs() < 1e-9);
    eprintln!("estimator {:?}", plan.estimator);
}
