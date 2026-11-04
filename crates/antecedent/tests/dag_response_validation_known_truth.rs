//! Known-truth checks for `InterventionResponse x Dag x {explicit, accepted}` under cheap
//! and full validation, for both estimators the cell names.
//!
//! * `response.intervention_gcomp` is held to the closed-form `true_response` of
//!   `conformance/response/intervention_response` (`1 + 2*0.25 + 0.8*mean(z)` on its exact
//!   deterministic design).
//! * `cell.aipw` is held to the analytic joint-cell mean `E[Y | do(a = 1, d = 1)] = 2 + 1.5 +
//!   0.25 * mean(z)` of a deterministic design whose `z` has exactly mean zero in every cell
//!   (so the cell level, 3.5, is apart from the cell-versus-control contrast, 1.5).
//!
//! No production estimator is called to produce an expectation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "deterministic row indices are small and exact in f64"
)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, EstimatorId, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

const CHECKED: [RefuteSuite; 2] = [RefuteSuite::Cheap, RefuteSuite::Full];

fn node(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn scalar(result: &antecedent::StudyResult) -> f64 {
    result.response().map_or(result.estimate().as_effect().unwrap().ate, |r| match &r.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        _ => result.estimate().as_effect().unwrap().ate,
    })
}

fn with_graph(data: &TabularData, graph: &Dag, accepted: bool) -> antecedent::StudyBuilder {
    let builder = Study::tabular(data.clone());
    if accepted {
        builder.graph(AcceptedGraph::dag(graph.clone()))
    } else {
        builder.graph(graph.clone())
    }
}

/// `response.intervention_gcomp` on the fixture's exact design.
fn gcomp_matches_the_closed_form() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/intervention_response/expected.json"
    ))
    .unwrap();
    let n = usize::try_from(fixture["generation"]["n"].as_u64().unwrap()).unwrap();
    let z: Vec<f64> = (0..n).map(|i| (i as f64 / 17.0).sin()).collect();
    let treatment: Vec<f64> = (0..n).map(|i| z[i] + (i as f64 / 11.0).cos()).collect();
    let outcome: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * treatment[i] + 0.8 * z[i]).collect();
    let data = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(node(2), node(0)).unwrap();
    graph.insert_directed(node(2), node(1)).unwrap();
    graph.insert_directed(node(0), node(1)).unwrap();
    let level = fixture["contract"]["intervention"]["value"].as_f64().unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(level))]),
    });
    // The fixture's `true_response` and its stated formula agree on this exact design.
    let mean_z = z.iter().sum::<f64>() / n as f64;
    let truth = fixture["contract"]["true_response"].as_f64().unwrap();
    assert!((truth - (1.0 + 2.0 * level + 0.8 * mean_z)).abs() < 1e-12);
    let tolerance = fixture["tolerance"]["truth_absolute"].as_f64().unwrap();
    for accepted in [false, true] {
        for suite in CHECKED {
            let label = format!("accepted={accepted} {suite:?}");
            let result = with_graph(&data, &graph, accepted)
                .query(CausalQuery::Response(query.clone()))
                .estimator(EstimatorId::ResponseInterventionGcomp)
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(52))
                .unwrap();
            let value = scalar(&result);
            assert!((value - truth).abs() <= tolerance, "{label}: {value} vs {truth}");
            assert!(
                result.refutations.iter().any(|r| r.refuter.as_ref().contains("overlap")),
                "{label}: overlap must run on the intervention level"
            );
        }
    }
}

/// `cell.aipw` on a deterministic joint-intervention design: `Y = 2 + 1.5 a d + 0.25 z +/- 0.1`
/// with `z` cycling over a symmetric grid inside every `(a, d)` cell.
fn cell_aipw_matches_the_analytic_cell_mean() {
    let grid = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let (mut a, mut d, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (ai, di) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
        for row in 0..200usize {
            let zi = grid[row % grid.len()];
            let noise = if (row / grid.len()) % 2 == 0 { 0.1 } else { -0.1 };
            a.push(ai);
            d.push(di);
            z.push(zi);
            y.push(2.0 + 1.5 * ai * di + 0.25 * zi + noise);
        }
    }
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("d", d.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(3, 0), (3, 1), (3, 2), (0, 2), (1, 2)] {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
        ]),
    });
    let truth = 2.0 + 1.5 + 0.25 * 0.0;
    for accepted in [false, true] {
        for suite in CHECKED {
            let label = format!("accepted={accepted} {suite:?}");
            let result = with_graph(&data, &graph, accepted)
                .query(CausalQuery::Response(query.clone()))
                .estimator(EstimatorId::CellAipw)
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(207))
                .unwrap();
            let value = scalar(&result);
            assert!((value - truth).abs() < 0.05, "{label}: cell mean {value} vs {truth}");
            assert!(
                result.refutations.iter().any(|r| r.refuter.as_ref() == "sensitivity.evalue"),
                "{label}: the cell-versus-control contrast is E-valued"
            );
        }
    }
}

/// Both estimators the cell names, executed on explicit and accepted structure under cheap
/// and full validation, against their closed-form truths.
#[test]
fn dag_intervention_response_matches_the_closed_forms_under_cheap_and_full() {
    gcomp_matches_the_closed_form();
    cell_aipw_matches_the_analytic_cell_mean();
}
