//! One-modifier temporal conditional pulse and sustained effects.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

fn series() -> TimeSeriesData {
    let n = 80;
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 0..n {
        x[t] = ((t as f64) * 0.37).sin();
        w[t] = if t % 2 == 0 { 0.0 } else { 1.0 };
        if t > 0 {
            y[t] = 1.5 * x[t - 1] + 0.2 * w[t] + 0.8 * x[t - 1] * w[t];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())],
        1,
    )
    .unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(w0, y0).unwrap();
    graph
}

fn query(sustained: bool) -> TemporalEffectQuery {
    let base = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
        .with_effect_modifiers([VariableId::from_raw(2)]);
    if sustained {
        base.with_policy(TemporalPolicy::sustained(-1, -1))
    } else {
        base.with_policy(TemporalPolicy::pulse(-1))
    }
}

#[test]
fn temporal_dag_conditional_pulse_and_sustained_publish_the_marginal_point() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_conditional/expected.json"
    ))
    .unwrap();
    let data = series();
    let graph = graph();
    let ctx = ExecutionContext::for_tests(11);
    let truth = pin["marginal_effect"].as_f64().unwrap();
    for sustained in [false, true] {
        for accepted in [false, true] {
            for bayesian in [false, true] {
                let builder = Study::series(data.clone()).query(CausalQuery::TemporalEffect(query(sustained)));
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
                    Some(antecedent::support::CellStatus::Licensed)
                );
                let prepared = study.prepare(&ctx).unwrap();
                drop(builder);
                drop(study);
                let result = prepared.estimate_retained(&ctx).unwrap();
                assert!(
                    (result.estimate.ate - truth).abs() < 0.05,
                    "sustained={sustained} accepted={accepted} bayesian={bayesian} ate={}",
                    result.estimate.ate
                );
                let plan = &result.logical_plan;
                assert!(
                    result.diagnostics.iter().any(|d| {
                        d.code.as_ref() == "estimate.temporal.conditional_marginal"
                    }),
                    "ate={} diagnostics={:?}",
                    result.estimate.ate,
                    result.diagnostics.iter().map(|d| d.code.as_ref()).collect::<Vec<_>>()
                );
                if let Some(binding) = &result.interval {
                    assert_eq!(binding.method, antecedent_core::IntervalMethod::None);
                }
                assert!(result.posterior.is_none());
                assert_eq!(plan.estimator.as_deref(), Some("temporal.linear.adjustment"));
                assert!(result.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.temporal.conditional_marginal"
                }));
            }
        }
    }
}
