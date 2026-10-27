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
                let builder = Study::series(data.clone())
                    .query(CausalQuery::TemporalEffect(query(sustained)));
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
                assert_eq!(study.support_status(), Some(antecedent::support::CellStatus::Licensed));
                let prepared = study.prepare(&ctx).unwrap();
                drop(builder);
                drop(study);
                let result = prepared.estimate_retained(&ctx).unwrap();
                assert!(
                    (result.estimate().as_effect().unwrap().ate - truth).abs() < 0.05,
                    "sustained={sustained} accepted={accepted} bayesian={bayesian} ate={}",
                    result.estimate().as_effect().unwrap().ate
                );
                let plan = &result.logical_plan;
                assert!(
                    result
                        .diagnostics
                        .iter()
                        .any(|d| { d.code.as_ref() == "estimate.temporal.conditional_marginal" }),
                    "ate={} diagnostics={:?}",
                    result.estimate().as_effect().unwrap().ate,
                    result.diagnostics.iter().map(|d| d.code.as_ref()).collect::<Vec<_>>()
                );
                // Bayesian: the tempered slope posterior's credible interval. Frequentist:
                // the shared circular-block interval, withheld when this exact-fit fixture
                // leaves the replicates without spread.
                assert_eq!(result.posterior.is_some(), bayesian);
                let method = result.interval().as_ref().map(|binding| binding.method);
                if bayesian {
                    assert_eq!(method, Some(antecedent_core::IntervalMethod::PosteriorQuantile));
                } else {
                    assert!(
                        matches!(
                            method,
                            Some(
                                antecedent_core::IntervalMethod::CircularBlockSe
                                    | antecedent_core::IntervalMethod::None
                            )
                        ),
                        "{method:?}"
                    );
                }
                assert_eq!(plan.estimator.as_deref(), Some("temporal.linear.adjustment"));
                assert!(
                    result
                        .diagnostics
                        .iter()
                        .any(|d| { d.code.as_ref() == "estimate.temporal.conditional_marginal" })
                );
            }
        }
    }
}

#[test]
#[allow(
    clippy::needless_range_loop,
    reason = "each step reads t[i - 1] and w[i] alongside the index-dependent signal term"
)]
fn multi_step_sustained_effect_uses_a_named_modifier() {
    let n = 250;
    let mut t = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        t[i] = match i % 5 {
            0 => -2.0,
            1 => -1.0,
            2 => 0.0,
            3 => 1.0,
            _ => 2.0,
        };
    }
    for i in 2..n {
        let t1 = t[i - 1];
        let t2 = t[i - 2];
        let noise = match i % 4 {
            0 => -1.5,
            1 => -0.5,
            2 => 0.5,
            _ => 1.5,
        };
        w[i] = 0.8 + 0.6 * t1 + noise;
        y[i] = t1 + 0.5 * t2 + 3.0 * t1 * w[i];
    }
    let mean_w = w[2..].iter().sum::<f64>() / (n - 2) as f64;
    let expected = 1.0 + 3.0 * mean_w + 0.5;
    let data = TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())],
        1,
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let treatment = VariableId::from_raw(0);
    let outcome = VariableId::from_raw(1);
    let t1 = ensure_lagged(&mut graph, treatment, Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, treatment, Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, outcome, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    let base = TemporalEffectQuery::sustained(treatment, outcome, 0, 1.0)
        .with_policy(TemporalPolicy::sustained(-2, -1));
    let modified = base.clone().with_effect_modifiers([VariableId::from_raw(2)]);
    let ctx = ExecutionContext::for_tests(11);
    let run = |query: TemporalEffectQuery| {
        let study = Study::series(data.clone())
            .graph(graph.clone())
            .query(CausalQuery::TemporalEffect(query))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let one_shot = study.run(&ctx).unwrap();
        let retained = study.prepare(&ctx).unwrap().estimate_retained(&ctx).unwrap();
        (one_shot.estimate().as_effect().unwrap().ate, retained.estimate().as_effect().unwrap().ate)
    };
    let (fresh, retained) = run(modified);
    let (plain_fresh, plain_retained) = run(base);
    assert!((fresh - expected).abs() < 0.05, "fresh {fresh} expected {expected}");
    assert!((retained - expected).abs() < 0.05, "retained {retained} expected {expected}");
    assert!((fresh - plain_fresh).abs() > 0.5, "fresh {fresh} plain {plain_fresh}");
    assert!((retained - plain_retained).abs() > 0.5, "retained {retained} plain {plain_retained}");
}
