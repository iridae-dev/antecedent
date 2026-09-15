//! Licensed 1.9.0 attribution / transport / interference staged cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent::{CellStatus, InterferenceSpec, RefuteSuite, Study, TransportTrialSpec};
use antecedent_core::{
    AnomalyAttributionQuery, AssignmentDesign, CausalQuery, ChangeAttributionQuery,
    ContinuousDomain, ExposureLevel, ExposureMapping, GridSpec, InterferenceFunctional,
    InterferenceQuery, PopulationSelector, ResponseFunctional, ResponseQuery, TransportQuery,
    VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_graph::{Admg, Dag, DenseNodeId};

fn ctx() -> antecedent_core::ExecutionContext {
    antecedent_core::ExecutionContext::for_tests(13)
}

fn two_period_chain() -> (TabularData, Dag) {
    let n = 80usize;
    let xv: Vec<f64> = (0..n).map(|i| (i % 40) as f64 * 0.1).collect();
    let yv: Vec<f64> = (0..n)
        .map(|i| {
            let x = (i % 40) as f64 * 0.1;
            if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x }
        })
        .collect();
    let data = TabularData::from_f64_columns([("x", xv.as_slice()), ("y", yv.as_slice())]).unwrap();
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    (data, dag)
}

fn outlier_chain() -> (TabularData, Dag) {
    let n = 20usize;
    let xv: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let yv: Vec<f64> = (0..n).map(|i| if i + 1 == n { 200.0 } else { 2.0 * i as f64 }).collect();
    let data = TabularData::from_f64_columns([("x", xv.as_slice()), ("y", yv.as_slice())]).unwrap();
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    (data, dag)
}

#[test]
fn anomaly_attribution_known_truth() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_attribution/expected.json"
    ))
    .unwrap();
    let (data, dag) = outlier_chain();
    let query = AnomalyAttributionQuery::new([VariableId::from_raw(1)], 100);
    let study = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::AnomalyAttribution(query))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    assert_eq!(study.support_status(), Some(CellStatus::Licensed));
    let prepared = study.prepare(&ctx()).unwrap();
    let result = prepared.estimate(&data, &ctx()).unwrap();
    assert_eq!(result.support_status, Some(CellStatus::Licensed));
    assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
    let scores = result.anomaly.as_ref().expect("anomaly scores");
    let y = scores.iter().find(|s| s.target == VariableId::from_raw(1)).expect("y target");
    let (top_i, top) =
        y.scores.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
    assert_eq!(
        y.rows[top_i],
        usize::try_from(pin["anomaly"]["top_row"].as_u64().unwrap()).unwrap()
    );
    assert!(*top >= pin["anomaly"]["score_min"].as_f64().unwrap(), "top score {top} below pin");
}

#[test]
fn change_attribution_known_truth() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_attribution/expected.json"
    ))
    .unwrap();
    let (data, dag) = two_period_chain();
    let query = ChangeAttributionQuery::new(
        VariableId::from_raw(1),
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
    );
    let study = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::ChangeAttribution(query))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    assert_eq!(study.support_status(), Some(CellStatus::Licensed));
    let prepared = study.prepare(&ctx()).unwrap();
    let result = prepared.estimate(&data, &ctx()).unwrap();
    assert_eq!(result.support_status, Some(CellStatus::Licensed));
    assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
    let change = result.change_attribution.as_ref().expect("change attribution");
    let expected = pin["change"]["total_change"].as_f64().unwrap();
    let tol = pin["change"]["tolerance"].as_f64().unwrap();
    assert!(
        (change.total_change - expected).abs() <= tol,
        "total_change={} pin={expected}±{tol}",
        change.total_change
    );
    assert!(change.total_change >= pin["change"]["total_change_min"].as_f64().unwrap());
}

#[test]
fn transport_direct_trial_ipw_known_truth() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/staged_transport/expected.json"
    ))
    .unwrap();
    let data = TabularData::from_f64_columns([
        ("a", &[1.0, 0.0, 0.0, 0.0][..]),
        ("y", &[3.0, 1.0, 0.0, 0.0][..]),
        ("trial", &[1.0, 1.0, 0.0, 0.0][..]),
        ("s", &[0.5, 0.5, 0.5, 0.5][..]),
        ("e", &[0.5, 0.5, 0.5, 0.5][..]),
    ])
    .unwrap();
    let mut admg = Admg::with_variables(5);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    let query = TransportQuery::new(response, "trial", "target", [VariableId::from_raw(0)]);
    let study = Study::tabular(data.clone())
        .graph(admg)
        .query(CausalQuery::Transport(query))
        .selection_targets(Arc::from([]))
        .transport_trial(TransportTrialSpec {
            trial: VariableId::from_raw(2),
            selection_probability: VariableId::from_raw(3),
            treatment_probability: VariableId::from_raw(4),
        })
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    assert_eq!(study.support_status(), Some(CellStatus::Licensed));
    let prepared = study.prepare(&ctx()).unwrap();
    let result = prepared.estimate(&data, &ctx()).unwrap();
    assert_eq!(result.support_status, Some(CellStatus::Licensed));
    assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
    let transported = result.transport.as_ref().expect("transport estimate");
    let atol = pin["tolerance"].as_f64().unwrap();
    assert!((transported.ipw - pin["ipw"].as_f64().unwrap()).abs() <= atol);
}

#[test]
fn interference_bernoulli_neighbor_count_known_truth() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/randomized_interference/expected.json"
    ))
    .unwrap();
    let outcomes: Vec<f64> =
        pin["outcomes"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let assignment: Vec<bool> =
        pin["assignment"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let units = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let network = NetworkData::try_new(
        units.clone(),
        [NetworkEdge { from: 0, to: 1, weight: 1.0 }, NetworkEdge { from: 1, to: 0, weight: 1.0 }],
    )
    .unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
        ExposureMapping::NeighborCount,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 1.0 },
            to: ExposureLevel { own: 1.0, neighbors: 0.0 },
        },
    );
    let dag = Dag::with_variables(1);
    let study = Study::tabular(units.clone())
        .graph(dag)
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    assert_eq!(study.support_status(), Some(CellStatus::Licensed));
    let prepared = study.prepare(&ctx()).unwrap();
    let result = prepared.estimate(&units, &ctx()).unwrap();
    assert_eq!(result.support_status, Some(CellStatus::Licensed));
    let estimated = result.interference.as_ref().expect("interference estimate");
    let expected = &pin["expected"];
    let atol = pin["tolerance"]["atol"].as_f64().unwrap();
    assert!(
        (estimated.contrast.horvitz_thompson
            - expected["horvitz_thompson_contrast"].as_f64().unwrap())
        .abs()
            <= atol
    );
    assert!(
        (estimated.contrast.hajek - expected["hajek_contrast"].as_f64().unwrap()).abs() <= atol
    );
    assert!(
        (estimated.contrast.conservative_variance
            - expected["conservative_variance"].as_f64().unwrap())
        .abs()
            <= atol
    );
}
