//! Local linear slopes of a temporal response, one horizon, no interval.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, DerivativeScale, DerivativeWeighting, ExecutionContext, Lag, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TemporalPolicy, TemporalResponseSpec,
    VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

fn series() -> TimeSeriesData {
    let n = 80;
    let mut x = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut v = vec![0.0; n];
    for t in 0..n {
        x[t] = 1.0 + ((t as f64) * 0.37).sin();
        b[t] = if t % 2 == 0 { 0.0 } else { 1.0 };
        if t > 0 {
            y[t] = 3.0 + 2.0 * x[t - 1] + 0.5 * b[t - 1];
            v[t] = 1.0 + x[t - 1] - b[t - 1];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("b", b.as_slice()), ("y", y.as_slice()), ("v", v.as_slice())],
        1,
    )
    .unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let b1 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let v0 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(b1, y0).unwrap();
    graph.insert_directed(x1, v0).unwrap();
    graph.insert_directed(b1, v0).unwrap();
    graph
}

fn attach(functional: ResponseFunctional) -> ResponseQuery {
    ResponseQuery::new(functional).with_temporal(
        TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap(),
    )
}

fn functionals() -> Vec<(&'static str, ResponseFunctional)> {
    let point = |scale| ResponseFunctional::PointDerivative {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(0),
        at: 2.0,
        order: 1,
        scale,
    };
    vec![
        ("point", point(DerivativeScale::Identity)),
        ("elasticity", point(DerivativeScale::LogLog)),
        ("semi_treatment", point(DerivativeScale::LogTreatment)),
        ("semi_outcome", point(DerivativeScale::LogOutcome)),
        (
            "average",
            ResponseFunctional::AverageDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                weighting: DerivativeWeighting::Observed,
            },
        ),
        (
            "jacobian",
            ResponseFunctional::Jacobian {
                outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
                treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
                at: Arc::from([2.0, 1.0]),
                scale: DerivativeScale::Identity,
            },
        ),
        (
            "directional",
            ResponseFunctional::DirectionalDerivative {
                outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
                treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
                at: Arc::from([2.0, 1.0]),
                direction: Arc::from([1.0, 2.0]),
            },
        ),
    ]
}

#[test]
fn temporal_dag_derivatives_publish_the_linear_slope() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_derivative/expected.json"
    ))
    .unwrap();
    let data = series();
    let graph = graph();
    let ctx = ExecutionContext::for_tests(11);
    let tolerance = pin["tolerance"].as_f64().unwrap();
    for (key, functional) in functionals() {
        for accepted in [false, true] {
            for bayesian in [false, true] {
                let builder = Study::series(data.clone())
                    .query(CausalQuery::Response(attach(functional.clone())));
                let builder = if accepted {
                    builder.graph(AcceptedGraph::from(graph.clone()))
                } else {
                    builder.graph(graph.clone())
                };
                let builder = builder.refute(RefuteSuite::None);
                let builder = if bayesian {
                    builder.inference(InferenceMode::Bayesian(
                        BayesianConfig::conjugate().prior_scale(1_000.0).n_draws(8),
                    ))
                } else {
                    builder
                };
                let study = builder.clone().build().unwrap();
                assert_eq!(
                    study.support_status(),
                    Some(antecedent::support::CellStatus::Licensed),
                    "{key} accepted={accepted} bayesian={bayesian}"
                );
                let prepared = study.prepare(&ctx).unwrap();
                drop(builder);
                drop(study);
                let result = prepared.estimate_retained(&ctx).unwrap();
                let response = result.response.as_ref().unwrap();
                let ResponseIdentification::PointIdentified(value) = &response.estimate else {
                    panic!("{key} unidentified");
                };
                let got = match value {
                    ResponseValue::Scalar(x) => vec![*x],
                    ResponseValue::Jacobian { values, .. } => values.to_vec(),
                    ResponseValue::Vector(x) => x.to_vec(),
                    other => panic!("{key} unexpected {other:?}"),
                };
                let expected = if let Some(x) = pin[key].as_f64() {
                    vec![x]
                } else {
                    pin[key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
                };
                for (got, want) in got.iter().zip(&expected) {
                    assert!(
                        (got - want).abs() < tolerance,
                        "{key} accepted={accepted} bayesian={bayesian}: {got} != {want}"
                    );
                }
                assert!(result.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.temporal.linear_derivative"
                }));
                assert!(result.posterior.is_none());
                if let Some(binding) = &result.interval {
                    assert_eq!(binding.method, antecedent_core::IntervalMethod::None);
                }
                let plan = &result.logical_plan;
                let estimator = if bayesian {
                    "response.temporal.bayesian"
                } else {
                    "temporal.response.gcomp"
                };
                assert_eq!(plan.estimator.as_deref(), Some(estimator));
            }
        }
    }
}
