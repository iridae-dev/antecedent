//! Licensed attribution / transport / interference staged cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CellStatus, InterferenceSpec, RefuteSuite, Study, TransportTrialSpec};
use antecedent_core::{
    AnomalyAttributionQuery, AssignmentDesign, CausalQuery, ChangeAttributionQuery,
    ContinuousDomain, ExposureLevel, ExposureMapping, GridSpec, IdentificationStatus,
    InterferenceFunctional, InterferenceQuery, PopulationSelector, ResponseFunctional,
    ResponseQuery, TransportQuery, VariableId,
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
    assert_eq!(result.identification.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(result.estimand.method.as_ref(), "transport.sid.direct");
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
    assert_eq!(result.identification.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(result.estimand.method.as_ref(), "interference.design");
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

/// Prepare an interference study on two units with the conformance network.
fn two_unit_interference(y: &[f64]) -> (TabularData, antecedent::PreparedStudy) {
    let units = TabularData::from_f64_columns([("y", y)]).unwrap();
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
    let study = Study::tabular(units.clone())
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from([true, false]) })
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    (units, study.prepare(&ctx()).unwrap())
}

/// The fixed network and realized assignment freeze at prepare; the outcomes
/// are the data. An estimate or refresh click on new outcomes executes on
/// those outcomes, and the refreshed data snapshot names what it executed.
#[test]
fn interference_click_executes_on_the_clicked_outcomes() {
    let (_, prepared) = two_unit_interference(&[1.0, 4.0]);
    let (moved, fresh) = two_unit_interference(&[2.0, 4.0]);
    let expected = fresh.estimate_retained(&ctx()).unwrap();
    let expected = expected.interference.expect("interference estimate").contrast;
    // Unit 0 is exposed to (1, 0) and unit 1 to (0, 1), each with probability 1/4:
    // HT = 2/(2·0.25) − 4/(2·0.25) = −4.
    assert!((expected.horvitz_thompson + 4.0).abs() < 1e-12);

    let clicked = prepared.estimate(&moved, &ctx()).unwrap();
    assert_eq!(clicked.interference.expect("interference estimate").contrast, expected);

    let mut refreshed = prepared.clone();
    let result = refreshed.refresh(moved, &ctx()).unwrap();
    assert_eq!(result.interference.expect("interference estimate").contrast, expected);
    let again = refreshed.estimate_retained(&ctx()).unwrap();
    assert_eq!(again.interference.expect("interference estimate").contrast, expected);
    assert_eq!(
        refreshed.contract().unwrap().identities.data_snapshot,
        fresh.contract().unwrap().identities.data_snapshot,
        "the refreshed snapshot names the refreshed outcomes under the frozen network"
    );
}

/// The calibration match key of the prepared execution (the one a Python
/// `analyze` result carries) is the key the coverage harness binds a
/// `Study::run` replicate to, so a record measured by the harness binds to the
/// study's claims.
#[test]
fn design_cells_bind_calibration_under_the_harness_key() {
    let (units, prepared) = two_unit_interference(&[1.0, 4.0]);
    let network = NetworkData::try_new(
        units.clone(),
        [NetworkEdge { from: 0, to: 1, weight: 1.0 }, NetworkEdge { from: 1, to: 0, weight: 1.0 }],
    )
    .unwrap();
    let study = Study::tabular(units)
        .graph(Dag::with_variables(1))
        .query(prepared.query().clone())
        .interference(InterferenceSpec { network, assignment: Arc::from([true, false]) })
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let keys = |bases: Vec<antecedent_io::calibration::CalibrationBasisWire>| {
        bases.into_iter().map(|basis| basis.key).collect::<Vec<_>>()
    };
    let harness = study.run(&ctx()).unwrap();
    let harness = keys(harness.calibration_bases(&study.inspect().unwrap()).unwrap());
    let executed = prepared.estimate_retained(&ctx()).unwrap();
    let executed = keys(executed.calibration_bases(&prepared.contract().unwrap()).unwrap());
    assert_eq!(executed, harness);
    let primary = &executed[0];
    assert_eq!(
        (
            primary.query.as_str(),
            primary.graph_class.as_str(),
            primary.structure.as_str(),
            primary.modality.as_str(),
            primary.inference.as_str(),
            primary.estimator.as_str(),
            primary.interval_method.as_str(),
            primary.dependence.as_str(),
            primary.identification.as_str(),
        ),
        (
            "InterferenceQuery",
            "Dag",
            "fixed",
            "tabular",
            "Frequentist",
            "interference.ht_hajek",
            "analytic_se",
            "iid",
            "point"
        )
    );
}

/// A support refusal renders as `refused: reason=<code>: <message>`.
fn refused_with(error: &antecedent::CausalError) -> Option<String> {
    let text = error.to_string();
    let rest = text.strip_prefix("refused: ")?;
    antecedent_core::reason_code::split_prefix(rest).map(|(code, _)| code.to_string())
}

/// The licensed interference cell is `NeighborCount` under Bernoulli assignment;
/// another design on the same coordinate is refused, not executed under the
/// cell's license.
#[test]
fn interference_outside_the_licensed_construction_is_refused() {
    let units = TabularData::from_f64_columns([("y", &[1.0, 4.0, 2.0, 3.0][..])]).unwrap();
    let network = NetworkData::try_new(units.clone(), []).unwrap();
    let contrast = InterferenceFunctional::ExposureContrast {
        outcome: VariableId::from_raw(0),
        from: ExposureLevel { own: 0.0, neighbors: 0.0 },
        to: ExposureLevel { own: 1.0, neighbors: 0.0 },
    };
    for query in [
        InterferenceQuery::new(
            AssignmentDesign::CompleteRandomization { treated: 2 },
            ExposureMapping::NeighborCount,
            contrast.clone(),
        ),
        InterferenceQuery::new(
            AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
            ExposureMapping::NeighborFraction,
            contrast,
        ),
    ] {
        let error = Study::tabular(units.clone())
            .graph(Dag::with_variables(1))
            .query(CausalQuery::Interference(query))
            .interference(InterferenceSpec {
                network: network.clone(),
                assignment: Arc::from([false, true, false, true]),
            })
            .refute(RefuteSuite::None)
            .build()
            .expect_err("an unlicensed interference construction must refuse");
        assert_eq!(refused_with(&error).as_deref(), Some("construction_not_licensed"), "{error}");
    }
}

/// The licensed transport cell transports a mean `ResponseCurve`; a derivative
/// would be labelled with a binary IPW contrast it is not, so it is refused.
#[test]
fn transport_outside_the_licensed_construction_is_refused() {
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
    let response = ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: VariableId::from_raw(1),
        treatment: VariableId::from_raw(0),
        at: 0.5,
        order: 1,
        scale: antecedent_core::DerivativeScale::Identity,
    });
    let error = Study::tabular(data)
        .graph(admg)
        .query(CausalQuery::Transport(TransportQuery::new(
            response,
            "trial",
            "target",
            [VariableId::from_raw(0)],
        )))
        .selection_targets(Arc::from([]))
        .transport_trial(TransportTrialSpec {
            trial: VariableId::from_raw(2),
            selection_probability: VariableId::from_raw(3),
            treatment_probability: VariableId::from_raw(4),
        })
        .refute(RefuteSuite::None)
        .build()
        .expect_err("a transported derivative is not the licensed construction");
    assert_eq!(refused_with(&error).as_deref(), Some("construction_not_licensed"), "{error}");
}
