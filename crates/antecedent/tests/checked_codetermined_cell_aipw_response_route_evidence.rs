//! Builder-independent evidence for the `CoDetermined` joint-cell AIPW response
//! route.
//!
//! Tiers `{z} | {t1, t2} | {y}` with binary co-determined treatments; the
//! closure ADMG carries `t1 <-> t2`, the closure adjustment set is `{z}`, and
//! the outcome law is linear in the saturated cells, so the `(1, 1)` cell has
//! the closed form the pin asserts against.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{TieredBackground, WithinTier};
use antecedent_io::consume_analysis_result;

/// Columns `z, t1, t2, y` on a balanced binary design:
/// `y = shift + 0.2 + 0.4 t1 + 0.3 t2 + 0.5 t1 t2 + 0.2 z + small noise`.
fn data(shift: f64) -> TabularData {
    let n = 960usize;
    let z: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
    let t1: Vec<f64> = (0..n).map(|i| (i / 2 % 2) as f64).collect();
    let t2: Vec<f64> = (0..n).map(|i| (i / 4 % 2) as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            shift
                + 0.2
                + 0.4 * t1[i]
                + 0.3 * t2[i]
                + 0.5 * t1[i] * t2[i]
                + 0.2 * z[i]
                + 0.03 * (i as f64 * 0.31).sin()
        })
        .collect();
    TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t1", t1.as_slice()),
        ("t2", t2.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

/// `E[y | do(t1 = 1, t2 = 1)]` with `E[z] = 1/2`.
fn truth(shift: f64) -> f64 {
    shift + 0.2 + 0.4 + 0.3 + 0.5 + 0.2 * 0.5
}

fn background(data: &TabularData) -> TieredBackground {
    TieredBackground::from_named(
        data.schema(),
        &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
        WithinTier::CoDetermined,
    )
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

fn build(data: TabularData) -> Study {
    let background = background(&data);
    Study::tabular(data)
        .tiered_background(background)
        .unwrap()
        .query(CausalQuery::Response(query()))
        .estimator(EstimatorId::CellAipw)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn value(result: &antecedent::StudyResult) -> f64 {
    match &result.response.as_ref().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        other => panic!("expected a point-identified joint cell, got {other:?}"),
    }
}

/// `InterventionResponse` × `CoDetermined` × explicit × Frequentist × none:
/// the retained plan freezes the tier closure, the joint cell, the closure
/// adjustment set and the cross-fitted cell AIPW procedure; the prepared click
/// reproduces the one-shot run and the closed-form cell mean; refresh
/// re-executes the same plan on shifted data; schema-changed data is refused;
/// and the exported artifact names the missing checked intervention response
/// operation to an independent consumer.
#[test]
fn codetermined_joint_cell_aipw_response_is_sealed_and_refreshes() {
    std::thread::Builder::new()
        .name("checked-codetermined-cell-aipw-evidence".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(codetermined_joint_cell_aipw_response_body)
        .unwrap()
        .join()
        .unwrap();
}

fn codetermined_joint_cell_aipw_response_body() {
    let initial = data(0.0);
    let context = ExecutionContext::for_tests(9_119);
    let builder = build(initial.clone());
    let one_shot = builder.run(&context).unwrap();
    let mut prepared = builder.prepare(&context).unwrap();
    drop(builder);

    let plan = prepared
        .checked_cell_aipw_response_info()
        .expect("preparation retains the checked CoDetermined joint cell route");
    assert_eq!(plan.query, query());
    assert_eq!(plan.identifier, IdentifierId::GeneralizedAdjustment);
    assert_eq!(plan.estimator, EstimatorId::CellAipw);
    assert_eq!(plan.validation, RefuteSuite::None);
    assert_eq!(plan.origin, antecedent::analysis::DagResponseOrigin::Explicit);
    assert_eq!(plan.within_tier, Some(WithinTier::CoDetermined));
    assert_eq!(plan.requested_arm, 3);
    assert_eq!(plan.adjustment_set.as_ref(), &[VariableId::from_raw(0)]);

    let click = prepared.estimate(&initial, &context).unwrap();
    for result in [&one_shot, &click] {
        assert_eq!(result.support_status.unwrap().as_str(), "licensed");
        assert_eq!(result.logical_plan.identifier.as_deref(), Some("generalized.adjustment"));
        assert_eq!(result.logical_plan.estimator.as_deref(), Some("cell.aipw"));
        assert!(result.refutations.is_empty());
        let got = value(result);
        assert!((got - truth(0.0)).abs() < 0.05, "cell (1, 1) {got} vs truth {}", truth(0.0));
    }
    assert!((value(&click) - value(&one_shot)).abs() < 1e-12);
    assert!(
        one_shot
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.as_ref() == "exec.identify.cached"),
        "one-shot run must execute its retained prepared plan"
    );

    let refreshed = prepared.refresh(data(0.4), &context).unwrap();
    assert!((value(&refreshed) - truth(0.4)).abs() < 0.05, "refreshed {}", value(&refreshed));
    let retained = prepared.checked_cell_aipw_response_info().unwrap();
    assert_eq!(retained.within_tier, Some(WithinTier::CoDetermined));
    assert_eq!(retained.adjustment_set, plan.adjustment_set);

    let three_columns = TabularData::from_f64_columns([
        ("z", vec![0.0; 8].as_slice()),
        ("t1", vec![0.0; 8].as_slice()),
        ("y", vec![0.0; 8].as_slice()),
    ])
    .unwrap();
    let error = prepared.refresh(three_columns, &context).unwrap_err();
    assert!(
        error.to_string().contains("same schema"),
        "schema-changed refresh must be refused: {error}"
    );

    let artifact = prepared
        .encode_contracted_result(&refreshed, "checked-codetermined-cell-aipw", &context)
        .unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    assert!(
        consumed.acceptance.unresolved.iter().any(|reason| {
            reason.as_ref() == "dependencies.checked_intervention_response_operation"
        }),
        "independent consumption must name the checked operation: {:?}",
        consumed.acceptance.unresolved
    );
    assert!(!consumed.acceptance.accepts_as_verified_program());
}
