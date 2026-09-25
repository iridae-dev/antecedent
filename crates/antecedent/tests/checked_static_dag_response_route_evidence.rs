//! Checked static DAG response curve and intervention g-computation lifecycles.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, EstimatorId, IdentifierId, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, GridSpec, Intervention, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn data(shift: f64) -> TabularData {
    let treatment: Vec<_> = (0..800).map(|i| -1.0 + 2.0 * f64::from(i) / 799.0).collect();
    let outcome: Vec<_> = treatment
        .iter()
        .enumerate()
        .map(|(i, value)| shift + 0.4 + 1.7 * value + 0.04 * (i as f64 * 0.37).sin())
        .collect();
    TabularData::from_f64_columns([
        ("treatment", treatment.as_slice()),
        ("outcome", outcome.as_slice()),
    ])
    .unwrap()
}

fn graph() -> Dag {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn builder(
    data: TabularData,
    graph: Dag,
    accepted: bool,
    query: ResponseQuery,
    suite: RefuteSuite,
) -> Study {
    let route_builder = Study::tabular(data);
    let route_builder = if accepted {
        route_builder.graph(AcceptedGraph::from(graph))
    } else {
        route_builder.graph(graph)
    };
    route_builder.query(CausalQuery::Response(query)).refute(suite).build().unwrap()
}

fn values(response: &antecedent_core::CausalResponse) -> Vec<f64> {
    match &response.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => vec![*value],
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            mean.to_vec()
        }
        other => panic!("expected point-identified response, got {other:?}"),
    }
}

#[test]
fn mean_curve_explicit_and_accepted_keep_ordered_grid_through_refresh_and_export() {
    let initial = data(0.0);
    let graph = graph();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([-0.5, 0.0, 0.5])),
        ),
    });
    for accepted in [false, true] {
        let route_builder =
            builder(initial.clone(), graph.clone(), accepted, query.clone(), RefuteSuite::None);
        let one_shot =
            route_builder.run(&antecedent_core::ExecutionContext::for_tests(991)).unwrap();
        let context = antecedent_core::ExecutionContext::for_tests(991);
        let mut prepared = route_builder.prepare(&context).unwrap();
        let plan =
            prepared.checked_static_dag_response_info().expect("retained checked MeanCurve plan");
        assert_eq!(plan.query, query);
        assert_eq!(plan.identifier, IdentifierId::ResponseBackdoor);
        assert_eq!(plan.estimator, EstimatorId::ResponseKennedyDr);
        assert_eq!(plan.validation, RefuteSuite::None);
        assert_eq!(plan.grid_members.as_ref(), &[-0.5, 0.0, 0.5]);
        drop(route_builder);

        let first = prepared.estimate(&initial, &context).unwrap();
        let got = values(first.response.as_ref().unwrap());
        let one_shot_values = values(one_shot.response.as_ref().unwrap());
        for (prepared_value, one_shot_value) in got.iter().zip(one_shot_values) {
            assert!((prepared_value - one_shot_value).abs() < 1e-10);
        }
        for (&dose, &estimate) in [-0.5, 0.0, 0.5].iter().zip(&got) {
            let truth = 0.4 + 1.7 * dose;
            assert!((estimate - truth).abs() < 0.12, "dose {dose}: {estimate} != {truth}");
        }

        let refreshed_data = data(0.3);
        let refreshed = prepared.refresh(refreshed_data.clone(), &context).unwrap();
        let refreshed_values = values(refreshed.response.as_ref().unwrap());
        for (&dose, &estimate) in [-0.5, 0.0, 0.5].iter().zip(&refreshed_values) {
            let truth = 0.7 + 1.7 * dose;
            assert!((estimate - truth).abs() < 0.12, "refreshed dose {dose}: {estimate}");
        }

        let artifact = prepared
            .encode_contracted_result(&refreshed, "checked-dag-response-curve", &context)
            .unwrap();
        let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
        assert!(
            consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.checked_response_grid_operation"
            })
        );
        assert!(!consumed.acceptance.accepts_as_verified_program());
    }
}

#[test]
fn intervention_gcomp_explicit_and_accepted_cover_each_validation_suite() {
    let initial = data(0.0);
    let graph = graph();
    let intervention = Intervention::set(VariableId::from_raw(0), Value::f64(0.25));
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([intervention]),
    });
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let route_builder =
                builder(initial.clone(), graph.clone(), accepted, query.clone(), suite);
            let one_shot =
                route_builder.run(&antecedent_core::ExecutionContext::for_tests(992)).unwrap();
            let context = antecedent_core::ExecutionContext::for_tests(992);
            let mut prepared = route_builder.prepare(&context).unwrap();
            let plan = prepared
                .checked_static_dag_response_info()
                .expect("retained checked intervention g-computation plan");
            assert_eq!(plan.query, query);
            assert_eq!(plan.identifier, IdentifierId::ResponseBackdoor);
            assert_eq!(plan.estimator, EstimatorId::ResponseInterventionGcomp);
            assert_eq!(plan.validation, suite);
            assert!(plan.grid_members.is_empty());
            drop(route_builder);

            let first = prepared.estimate(&initial, &context).unwrap();
            let one_shot_value = values(one_shot.response.as_ref().unwrap())[0];
            let prepared_value = values(first.response.as_ref().unwrap())[0];
            assert!((prepared_value - one_shot_value).abs() < 1e-10, "{suite:?}");
            assert_eq!(first.refutations.is_empty(), suite == RefuteSuite::None);
            let got = prepared_value;
            let truth = 0.4 + 1.7 * 0.25;
            assert!((got - truth).abs() < 0.12, "{suite:?}: {got} != {truth}");

            let refreshed_data = data(0.3);
            let refreshed = prepared.refresh(refreshed_data, &context).unwrap();
            let refreshed_value = values(refreshed.response.as_ref().unwrap())[0];
            assert!((refreshed_value - truth - 0.3).abs() < 0.12, "{suite:?} refresh");

            let artifact = prepared
                .encode_contracted_result(&refreshed, "checked-dag-intervention", &context)
                .unwrap();
            let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
            assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.checked_intervention_response_operation"
            }));
            assert!(!consumed.acceptance.accepts_as_verified_program());
        }
    }
}

#[test]
fn static_response_curve_refuses_a_too_small_runtime_memory_budget() {
    let data = data(0.0);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([-0.5, 0.0, 0.5])),
        ),
    });
    let route_builder = builder(data.clone(), graph(), false, query, RefuteSuite::None);
    let prepared =
        route_builder.prepare(&antecedent_core::ExecutionContext::for_tests(993)).unwrap();
    let mut context = antecedent_core::ExecutionContext::for_tests(993);
    context.memory.hard_limit_bytes = Some(1);
    let error = prepared.estimate(&data, &context).unwrap_err();
    assert!(matches!(error, antecedent::CausalError::Resource { .. }), "{error:?}");
}
