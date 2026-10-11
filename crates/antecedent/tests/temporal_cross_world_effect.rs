//! Linear unfolded temporal counterfactual, path, nested, and interventional mean.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, InterventionalDistributionQuery,
    Lag, NestedCounterfactualQuery, PathSpecificEffectQuery, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

fn series() -> TimeSeriesData {
    let n = 200;
    let mut x = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 0..n {
        x[t] = ((t as f64) * 0.37).sin();
        if t > 0 {
            let shock = if t % 2 == 0 { 0.3 } else { -0.3 };
            m[t] = 0.5 * x[t - 1] + shock;
            y[t] = 1.5 * x[t - 1] + 4.0 * m[t];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, m0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(m0, y0).unwrap();
    graph
}

fn queries() -> Vec<(&'static str, CausalQuery, f64)> {
    let x = VariableId::from_raw(0);
    let m = VariableId::from_raw(1);
    let y = VariableId::from_raw(2);
    vec![
        (
            "nested",
            CausalQuery::NestedCounterfactual(NestedCounterfactualQuery::new(x, m, y).unwrap()),
            1.5,
        ),
        (
            "path",
            CausalQuery::PathSpecific(
                PathSpecificEffectQuery::binary(x, y).with_path_nodes([m]),
            ),
            2.0,
        ),
        (
            "counterfactual",
            CausalQuery::Counterfactual(
                CounterfactualQuery::new(y, Arc::from([Intervention::set(x, Value::f64(1.0))]))
                    .with_control_level(0.0),
            ),
            3.5,
        ),
        (
            "distribution",
            CausalQuery::Distribution(InterventionalDistributionQuery::new(
                y,
                [Intervention::set(x, Value::f64(1.0))],
            )),
            3.5,
        ),
    ]
}

#[test]
fn temporal_dag_cross_world_points_match_the_linear_mechanism() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_cross_world/expected.json"
    ))
    .unwrap();
    let data = series();
    let graph = graph();
    let ctx = ExecutionContext::for_tests(11);
    let tolerance = pin["tolerance"].as_f64().unwrap();
    for (key, query, _) in queries() {
        let truth = pin[key].as_f64().unwrap();
        for accepted in [false, true] {
            for bayesian in [false, true] {
                let builder = Study::series(data.clone()).query(query.clone());
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
                    "{key}"
                );
                let prepared = study.prepare(&ctx).unwrap();
                drop(builder);
                drop(study);
                let result = prepared.estimate_retained(&ctx).unwrap();
                assert!(
                    (result.estimate.ate - truth).abs() < tolerance,
                    "{key} accepted={accepted} bayesian={bayesian} ate={}",
                    result.estimate.ate
                );
                assert!(result.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.temporal.cross_world"
                }));
                assert!(result.posterior.is_none());
                if let Some(binding) = &result.interval {
                    assert_eq!(binding.method, antecedent_core::IntervalMethod::None);
                }
                let plan = &result.logical_plan;
                assert_eq!(plan.estimator.as_deref(), Some("temporal.linear.adjustment"));
            }
        }
    }
}
