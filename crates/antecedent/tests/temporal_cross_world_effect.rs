//! Linear unfolded temporal counterfactual, path, nested, and interventional mean.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention,
    InterventionalDistributionQuery, Lag, NestedCounterfactualQuery, PathSpecificEffectQuery,
    Value, VariableId,
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
            CausalQuery::PathSpecific(PathSpecificEffectQuery::binary(x, y).with_path_nodes([m])),
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
#[allow(
    clippy::too_many_lines,
    reason = "one mechanism check: the pinned points, the study, and the comparison stay together"
)]
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
                // The Bayesian unit counterfactual is the posterior of the slope times the
                // contrast; every other cell is the linear unfolded point.
                let posterior_cell = bayesian && key == "counterfactual";
                assert!(
                    (result.estimate.as_effect().unwrap().ate - truth).abs()
                        < if posterior_cell { tolerance.max(0.25) } else { tolerance },
                    "{key} accepted={accepted} bayesian={bayesian} ate={}",
                    result.estimate.as_effect().unwrap().ate
                );
                assert!(
                    result
                        .diagnostics
                        .iter()
                        .any(|d| { d.code.as_ref() == "estimate.temporal.cross_world" })
                );
                assert_eq!(result.posterior.is_some(), posterior_cell, "{key}");
                if !posterior_cell {
                    if let Some(binding) = &result.interval {
                        assert_eq!(binding.method, antecedent_core::IntervalMethod::None);
                    }
                }
                let plan = &result.logical_plan;
                assert_eq!(
                    plan.estimator.as_deref(),
                    Some(if posterior_cell {
                        "bayesian.temporal.gcomp"
                    } else {
                        "temporal.linear.adjustment"
                    })
                );
                let rule = result
                    .identification
                    .arena
                    .derivation(result.estimand.functional)
                    .map(|meta| meta.rule.to_string());
                assert!(result.estimand.adjustment_set.is_empty(), "{key}");
                match key {
                    "nested" => {
                        assert_eq!(rule.as_deref(), Some("temporal.cross_world.direct_slope"));
                        assert_eq!(result.estimand.method.as_ref(), "temporal_mediation.direct");
                        assert_eq!(result.estimand.mediators.as_ref(), &[VariableId::from_raw(1)]);
                        assert!(matches!(
                            result.identification.arena.node(result.estimand.functional),
                            antecedent_expr::ExprNode::Contrast { .. }
                        ));
                    }
                    "path" => {
                        assert_eq!(
                            rule.as_deref(),
                            Some("temporal.cross_world.mediator_edge_product")
                        );
                        assert_eq!(result.estimand.method.as_ref(), "path_specific.natural");
                        assert_eq!(result.estimand.mediators.as_ref(), &[VariableId::from_raw(1)]);
                        assert!(matches!(
                            result.identification.arena.node(result.estimand.functional),
                            antecedent_expr::ExprNode::Product(_)
                        ));
                    }
                    "counterfactual" | "distribution" => {
                        assert_eq!(rule.as_deref(), Some("backdoor.adjustment"));
                        assert_eq!(result.estimand.method.as_ref(), "temporal.backdoor.unfolded");
                        assert!(result.estimand.mediators.is_empty());
                    }
                    _ => panic!("{key}"),
                }
            }
        }
    }
}

#[test]
fn nested_and_path_stay_the_fitted_quantities_when_adjustment_is_nonempty() {
    let n = 80;
    let mut x = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for t in 1..n {
        z[t] = if t % 2 == 0 { 0.4 } else { -0.4 };
        x[t] = 0.5 * z[t] + ((t as f64) * 0.2).sin();
        let shock = if t % 3 == 0 { 0.3 } else { -0.2 };
        m[t] = 0.5 * x[t - 1] + shock;
        y[t] = 1.5 * x[t - 1] + 4.0 * m[t] + z[t - 1];
    }
    let data = TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("m", m.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap();
    let mut confounded = TemporalDag::empty();
    let z1 = ensure_lagged(&mut confounded, VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    let x1 = ensure_lagged(&mut confounded, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut confounded, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut confounded, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    confounded.insert_directed(z1, x1).unwrap();
    confounded.insert_directed(z1, y0).unwrap();
    confounded.insert_directed(x1, m0).unwrap();
    confounded.insert_directed(x1, y0).unwrap();
    confounded.insert_directed(m0, y0).unwrap();
    let ctx = ExecutionContext::for_tests(11);
    for (key, query, _) in queries() {
        let result = Study::series(data.clone())
            .graph(confounded.clone())
            .query(query)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
            .estimate_retained(&ctx)
            .unwrap();
        assert!(result.estimate.as_effect().unwrap().ate.is_finite(), "{key}");
        let rule = result
            .identification
            .arena
            .derivation(result.estimand.functional)
            .map(|meta| meta.rule.to_string());
        match key {
            "nested" => {
                assert_eq!(rule.as_deref(), Some("temporal.cross_world.direct_slope"));
                assert!(result.estimand.adjustment_set.is_empty(), "{key}");
            }
            "path" => {
                assert_eq!(rule.as_deref(), Some("temporal.cross_world.mediator_edge_product"));
                assert!(result.estimand.adjustment_set.is_empty(), "{key}");
            }
            "counterfactual" | "distribution" => {
                assert_eq!(rule.as_deref(), Some("backdoor.adjustment"));
            }
            _ => panic!("{key}"),
        }
    }
}
