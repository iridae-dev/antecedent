//! Builder-independent evidence for Bayesian static DAG response routes.
//!
//! The law is linear, so the posterior mean of every intervention level and
//! curve member has a closed form the pins assert against directly.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::consume_analysis_result;

/// `y = shift + 0.4 + 1.7 t + small deterministic noise` on a uniform dose grid.
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

fn inference() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(192).prior_scale(30.0))
}

fn build(data: TabularData, accepted: bool, query: ResponseQuery) -> Study {
    let builder = Study::tabular(data);
    let builder =
        if accepted { builder.graph(AcceptedGraph::from(graph())) } else { builder.graph(graph()) };
    builder
        .query(CausalQuery::Response(query))
        .inference(inference())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn values(result: &antecedent::StudyResult) -> Vec<f64> {
    match &result.response.as_ref().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => vec![*value],
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            mean.to_vec()
        }
        other => panic!("expected a point-identified posterior response, got {other:?}"),
    }
}

fn intervention_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(0.25))]),
    })
}

fn curve_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([-0.5, 0.0, 0.5])),
        ),
    })
}

/// `InterventionResponse`/`ResponseCurve` × `Dag` × explicit/accepted ×
/// Bayesian × none: the retained plan fixes the Bayesian procedure, the
/// prepared click reproduces the one-shot posterior exactly, every level
/// matches the linear truth, refresh re-executes the same plan on shifted
/// data, schema-changed data is refused, and the exported artifact names the
/// missing checked response operation to an independent consumer.
#[test]
fn bayesian_static_dag_responses_are_sealed_for_explicit_and_accepted_graphs() {
    run_on_large_stack(bayesian_static_dag_responses_body);
}

fn bayesian_static_dag_responses_body() {
    let initial = data(0.0);
    for (name, query, doses, key) in [
        (
            "intervention",
            intervention_query(),
            vec![0.25],
            "dependencies.checked_intervention_response_operation",
        ),
        (
            "curve",
            curve_query(),
            vec![-0.5, 0.0, 0.5],
            "dependencies.checked_response_grid_operation",
        ),
    ] {
        for accepted in [false, true] {
            let label = format!("{name} accepted={accepted}");
            let context = ExecutionContext::for_tests(41_007);
            let builder = build(initial.clone(), accepted, query.clone());
            let one_shot = builder.run(&context).unwrap();
            let mut prepared = builder.prepare(&context).unwrap();
            drop(builder);

            let plan = prepared
                .checked_static_dag_response_info()
                .unwrap_or_else(|| panic!("{label}: retained checked Bayesian response plan"));
            assert_eq!(plan.query, query, "{label}");
            assert_eq!(plan.identifier, IdentifierId::ResponseBackdoor, "{label}");
            assert_eq!(plan.estimator, EstimatorId::ResponseBayesian, "{label}");
            assert_eq!(plan.validation, RefuteSuite::None, "{label}");
            assert!(plan.inference.starts_with("bayesian:"), "{label}: {}", plan.inference);
            assert_eq!(plan.grid_members.is_empty(), name == "intervention", "{label}");

            let first = prepared.estimate(&initial, &context).unwrap();
            assert_eq!(
                first.logical_plan.estimator.as_deref(),
                Some("response.bayesian"),
                "{label}"
            );
            assert!(first.refutations.is_empty(), "{label}: validation none");
            let got = values(&first);
            let one_shot_values = values(&one_shot);
            assert_eq!(got.len(), doses.len(), "{label}");
            for (prepared_value, one_shot_value) in got.iter().zip(&one_shot_values) {
                assert!(
                    (prepared_value - one_shot_value).abs() < 1e-9,
                    "{label}: prepared {prepared_value} vs one-shot {one_shot_value}"
                );
            }
            assert!(
                one_shot
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.code.as_ref() == "exec.identify.cached"),
                "{label}: one-shot run must execute its retained prepared plan"
            );
            for (&dose, &estimate) in doses.iter().zip(&got) {
                let truth = 0.4 + 1.7 * dose;
                assert!((estimate - truth).abs() < 0.12, "{label}: dose {dose}: {estimate}");
            }

            let refreshed = prepared.refresh(data(0.3), &context).unwrap();
            let refreshed_values = values(&refreshed);
            for (&dose, &estimate) in doses.iter().zip(&refreshed_values) {
                let truth = 0.7 + 1.7 * dose;
                assert!((estimate - truth).abs() < 0.12, "{label}: refreshed dose {dose}");
            }
            assert!(prepared.checked_static_dag_response_info().is_some(), "{label}");

            let treatment: Vec<f64> = (0..800).map(|i| f64::from(i) / 799.0).collect();
            let renamed = TabularData::from_f64_columns([("dose", treatment.as_slice())]).unwrap();
            let error = prepared.refresh(renamed, &context).unwrap_err();
            assert!(
                error.to_string().contains("same schema"),
                "{label}: schema-changed refresh must be refused: {error}"
            );

            let artifact = prepared
                .encode_contracted_result(&refreshed, "bayesian-dag-response", &context)
                .unwrap();
            let consumed = consume_analysis_result(&artifact).unwrap();
            assert!(
                consumed.acceptance.unresolved.iter().any(|reason| reason.as_ref() == key),
                "{label}: independent consumption must name {key}: {:?}",
                consumed.acceptance.unresolved
            );
            assert!(!consumed.acceptance.accepts_as_verified_program(), "{label}");
        }
    }
}

fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("checked-bayesian-static-dag-response-evidence".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
