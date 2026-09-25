//! Public lifecycle evidence for bounded DAG joint-cell AIPW response routes.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, EstimatorId, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, Intervention, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::consume_analysis_result;

fn data(shift: f64) -> TabularData {
    let n = 1200;
    let mut first = Vec::with_capacity(n);
    let mut second = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    for row in 0..n {
        let a = (row % 2) as f64;
        let b = ((row / 2) % 2) as f64;
        first.push(a);
        second.push(b);
        outcome
            .push(shift + 1.5 + 2.0 * a + 3.0 * b + 4.0 * a * b + 0.05 * (row as f64 * 0.31).sin());
    }
    TabularData::from_f64_columns([
        ("t1", first.as_slice()),
        ("t2", second.as_slice()),
        ("y", outcome.as_slice()),
    ])
    .unwrap()
}

fn graph() -> Dag {
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    dag
}

fn query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(0), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
        ]),
    })
}

fn result_value(result: &antecedent::StudyResult) -> f64 {
    match &result.response.as_ref().expect("response result").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        other => panic!("expected scalar response, got {other:?}"),
    }
}

fn builder(data: TabularData, query: ResponseQuery, suite: RefuteSuite, accepted: bool) -> Study {
    let study = Study::tabular(data);
    let study =
        if accepted { study.graph(AcceptedGraph::from(graph())) } else { study.graph(graph()) };
    study
        .query(CausalQuery::Response(query))
        .estimator(EstimatorId::CellAipw)
        .refute(suite)
        .build()
        .unwrap()
}

#[test]
fn accepted_and_explicit_dag_cell_aipw_cover_none_cheap_full_and_refresh() {
    let query = query();
    for accepted in [false, true] {
        for (suite_index, suite) in
            [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full].into_iter().enumerate()
        {
            let initial = data(0.0);
            let builder = builder(initial.clone(), query.clone(), suite, accepted);
            let seed = 820 + u64::try_from(suite_index).unwrap();
            let context = antecedent_core::ExecutionContext::for_tests(seed);
            let one_shot = builder.run(&context).unwrap();
            let mut prepared = builder.prepare(&context).unwrap();
            drop(builder);
            let plan = prepared
                .checked_cell_aipw_response_info()
                .expect("preparation retains the checked cell AIPW route");
            assert_eq!(plan.estimator, EstimatorId::CellAipw);
            assert_eq!(plan.requested_arm, 3);
            assert_eq!(
                plan.origin,
                if accepted {
                    antecedent::analysis::DagResponseOrigin::Accepted
                } else {
                    antecedent::analysis::DagResponseOrigin::Explicit
                }
            );
            let result = prepared.estimate(&initial, &context).unwrap();
            assert!((result_value(&result) - 10.5).abs() < 0.2);
            assert!((result_value(&result) - result_value(&one_shot)).abs() < 1e-12);
            assert_eq!(result.refutations.is_empty(), suite == RefuteSuite::None);

            let score_table = prepared.score_table().expect("cell scores support retargeting");
            let retargeted = prepared
                .retarget(&vec![1.0; score_table.n_rows], &[], &context)
                .expect("the checked mean response retains its score table");
            assert!((result_value(&retargeted) - result_value(&result)).abs() < 1e-12);

            let refreshed_data = data(0.4);
            let refreshed = prepared.refresh(refreshed_data.clone(), &context).unwrap();
            assert!((result_value(&refreshed) - 10.9).abs() < 0.2);
            assert!(prepared.checked_cell_aipw_response_info().is_some());
            let artifact = prepared
                .encode_contracted_result(&refreshed, "checked-cell-aipw", &context)
                .unwrap();
            let consumed = consume_analysis_result(&artifact).unwrap();
            assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.checked_intervention_response_operation"
            }));
            assert!(!consumed.acceptance.accepts_as_verified_program());
        }
    }
}

#[test]
fn checked_cell_aipw_quantile_is_sealed_and_executes() {
    let mut quantile_query = query();
    quantile_query.outcome_functional = antecedent_core::OutcomeFunctional::quantile(0.5);
    let study = builder(data(0.0), quantile_query.clone(), RefuteSuite::None, false);
    let context = antecedent_core::ExecutionContext::for_tests(822);
    let mut prepared = study.prepare(&context).unwrap();
    drop(study);
    assert!(prepared.checked_cell_aipw_response_info().is_some());
    assert!(prepared.score_table().is_some());

    let result = prepared.estimate(&data(0.0), &context).unwrap();
    let value = result_value(&result);
    assert!(value.is_finite());
    assert!((value - 10.5).abs() < 0.2, "median response {value}");

    let refreshed = prepared.refresh(data(0.4), &context).unwrap();
    assert!((result_value(&refreshed) - 10.9).abs() < 0.2);
}

#[test]
fn cell_aipw_route_refuses_nonbinary_intervention_levels() {
    let invalid = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(0), Value::f64(0.5)),
            Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
        ]),
    });
    let error = builder(data(0.0), invalid, RefuteSuite::None, true)
        .run(&antecedent_core::ExecutionContext::for_tests(821))
        .unwrap_err();
    assert!(error.to_string().contains("binary 0/1 Set levels"), "{error}");
}
