//! Joint intervention certificates must cover every treatment.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, Dag, DenseNodeId, MarkedEdge, Pag};
use std::sync::Arc;

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/class_aware_envelope/joint.json"
    ))
    .unwrap()
}
fn node(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
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
fn graphs() -> (Dag, Cpdag, Pag) {
    let mut dag = Dag::with_variables(6);
    let mut cpdag = Cpdag::with_variables(6);
    let mut pag = Pag::with_variables(6);
    for edge in pin()["directed"].as_array().unwrap() {
        let a = node(u32::try_from(edge[0].as_u64().unwrap()).unwrap());
        let b = node(u32::try_from(edge[1].as_u64().unwrap()).unwrap());
        dag.insert_directed(a, b).unwrap();
        cpdag.insert_directed(a, b).unwrap();
        pag.insert_marked(MarkedEdge::directed(a, b)).unwrap();
    }
    (dag, cpdag, pag)
}
fn data() -> TabularData {
    let mut columns: Vec<Vec<f64>> = vec![vec![]; 6];
    for mask in 0..32 {
        let t = f64::from(mask & 1);
        let z = f64::from((mask >> 1) & 1);
        let w = f64::from((mask >> 2) & 1);
        let r = f64::from((mask >> 3) & 1);
        let v = f64::from((mask >> 4) & 1);
        let count = if (z - w).abs() < f64::EPSILON { 160 } else { 40 };
        for (column, value) in
            columns.iter_mut().zip([t, z, 1.0 + 2.0 * t + 3.0 * z + 4.0 * w, w, r, v])
        {
            column.extend(std::iter::repeat_n(value, count));
        }
    }
    TabularData::from_f64_columns(
        ["t", "z", "y", "w", "r", "v"].into_iter().zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap()
}

#[test]
fn joint_certificate_and_estimate_cover_both_targets() {
    let (dag, cpdag, pag) = graphs();
    for graph in [AcceptedGraph::from(dag), AcceptedGraph::from(cpdag), AcceptedGraph::from(pag)] {
        let identification = antecedent::identify(&graph, &CausalQuery::Response(query())).unwrap();
        assert!(identification.is_identified());
        assert_eq!(
            identification.estimands()[0].adjustment_set.as_ref(),
            &[VariableId::from_raw(3)]
        );
        let prepared = Study::tabular(data())
            .graph(graph)
            .query(query())
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ExecutionContext::for_tests(7))
            .unwrap();
        let result = prepared.estimate(&data(), &ExecutionContext::for_tests(7)).unwrap();
        let response = result.response.as_ref().unwrap();
        let value = match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
            other => panic!("expected a joint scalar, got {other:?}"),
        };
        assert!(
            (value - pin()["mean"].as_f64().unwrap()).abs() < pin()["tolerance"].as_f64().unwrap(),
            "got {value}"
        );
    }
}

#[test]
fn bayesian_joint_response_uses_the_same_certificate() {
    let (_, cpdag, pag) = graphs();
    for graph in [AcceptedGraph::from(cpdag), AcceptedGraph::from(pag)] {
        let result = Study::tabular(data())
            .graph(graph)
            .query(query())
            .refute(RefuteSuite::None)
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate()))
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(7))
            .unwrap();
        let response = result.response.as_ref().unwrap();
        let value = match &response.estimate {
            ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
            other => panic!("expected a joint scalar, got {other:?}"),
        };
        assert!(
            (value - pin()["mean"].as_f64().unwrap()).abs()
                < pin()["bayesian_tolerance"].as_f64().unwrap(),
            "got {value}"
        );
    }
}

#[test]
fn hidden_confounding_of_second_target_is_not_hidden_by_first_target() {
    let mut pag = Pag::with_variables(4);
    for edge in [
        MarkedEdge::directed(node(0), node(2)),
        MarkedEdge::directed(node(3), node(0)),
        MarkedEdge::bidirected(node(1), node(2)),
    ] {
        pag.insert_marked(edge).unwrap();
    }
    let identified =
        antecedent::identify(&AcceptedGraph::from(pag), &CausalQuery::Response(query())).unwrap();
    assert!(!identified.is_identified());
}

#[test]
fn incomplete_joint_envelope_preserves_distinct_adjustments_and_caps() {
    let mut graph = Cpdag::with_variables(6);
    for (a, b) in [(0, 2), (1, 2), (3, 2)] {
        graph.insert_directed(node(a), node(b)).unwrap();
    }
    graph.insert_undirected(node(1), node(3)).unwrap();
    let identifier = antecedent_identify::GeneralizedAdjustmentIdentifier::new();
    let envelope = identifier.identify_cpdag_response_envelope(&graph, &query()).unwrap();
    assert_eq!(envelope.cases.len(), 2);
    assert_eq!(envelope.status, antecedent_core::IdentificationStatus::PartiallyIdentified);
    assert!(envelope.cases.iter().any(|case| case.result.estimands[0].adjustment_set.is_empty()));
    assert!(
        envelope
            .cases
            .iter()
            .any(|case| case.result.estimands[0].adjustment_set.as_ref()
                == [VariableId::from_raw(3)])
    );
    let mut capped = identifier;
    capped.config.max_completions = 1;
    let envelope = capped.identify_cpdag_response_envelope(&graph, &query()).unwrap();
    assert_eq!(envelope.status, antecedent_core::IdentificationStatus::PartiallyIdentified);
    assert!(
        envelope
            .critical_graph_features
            .iter()
            .any(|f| f.kind.as_ref() == "completion_enumeration_capped")
    );
    let result = Study::tabular(data())
        .graph(graph)
        .query(query())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(7))
        .unwrap();
    let response = result.response.unwrap();
    let value = match response.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value))
        | ResponseIdentification::PartiallyIdentified(ResponseValue::Scalar(value)) => value,
        other => panic!("expected mixture scalar, got {other:?}"),
    };
    let expected =
        (pin()["mean"].as_f64().unwrap() + pin()["unadjusted_mean"].as_f64().unwrap()) / 2.0;
    assert!((value - expected).abs() < pin()["tolerance"].as_f64().unwrap(), "got {value}");
}

#[test]
fn bayesian_joint_stochastic_policies_integrate_the_linear_response() {
    use antecedent_core::StochasticPolicy;
    let (_, cpdag, pag) = graphs();
    for graph in [AcceptedGraph::from(cpdag), AcceptedGraph::from(pag)] {
        for policy in [
            StochasticPolicy::bernoulli(0.5),
            StochasticPolicy::gaussian(0.5, 0.25),
            StochasticPolicy::categorical([1.0, 1.0]),
        ] {
            let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: VariableId::from_raw(2),
                interventions: Arc::from([
                    Intervention::stochastic(VariableId::from_raw(0), policy),
                    Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
                ]),
            });
            let result = Study::tabular(data())
                .graph(graph.clone())
                .query(query)
                .refute(RefuteSuite::None)
                .inference(InferenceMode::Bayesian(BayesianConfig::conjugate()))
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(7))
                .unwrap();
            let response = result.response.unwrap();
            let ResponseIdentification::PointIdentified(ResponseValue::Scalar(mean)) =
                response.estimate
            else {
                panic!("scalar policy mean");
            };
            assert!((mean - 7.0).abs() < 0.05, "mean={mean}");
            assert!(
                response
                    .support
                    .warnings
                    .iter()
                    .any(|d| d.code.as_ref() == "response.stochastic_policy_support_unverified")
            );
        }
    }
}
