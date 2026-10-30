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
use antecedent_graph::{TemporalCpdag, TemporalDag, ensure_lagged};

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
                    (result.estimate().as_effect().unwrap().ate - truth).abs()
                        < if posterior_cell { tolerance.max(0.25) } else { tolerance },
                    "{key} accepted={accepted} bayesian={bayesian} ate={}",
                    result.estimate().as_effect().unwrap().ate
                );
                assert!(
                    result
                        .diagnostics
                        .iter()
                        .any(|d| { d.code.as_ref() == "estimate.temporal.cross_world" })
                );
                // Every Bayesian cell publishes a posterior. A frequentist nested or unit
                // counterfactual resamples only when replicates are requested, so with the
                // omitted default it is point only and carries no interval binding; the
                // path-specific effect and the interventional mean publish their shared
                // circular-block interval.
                assert_eq!(result.posterior.is_some(), bayesian, "{key}");
                let method = result.interval().as_ref().map(|binding| binding.method);
                if bayesian {
                    assert_eq!(method, Some(antecedent_core::IntervalMethod::PosteriorQuantile));
                } else if matches!(key, "nested" | "counterfactual") {
                    assert_eq!(method, None, "{key}");
                } else {
                    assert_eq!(
                        method,
                        Some(antecedent_core::IntervalMethod::CircularBlockSe),
                        "{key}"
                    );
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
fn nested_and_path_adjust_for_a_declared_confounder_and_recover_the_structural_effect() {
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
    for (key, query, structural) in queries() {
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
        let ate = result.estimate().as_effect().unwrap().ate;
        assert!(ate.is_finite(), "{key}");
        if matches!(key, "nested" | "path") {
            // z[t-1] confounds x[t-1] and y[t]: the unadjusted regression gives 1.652 for the
            // nested effect. The generating SCM (y = 1.5 x[t-1] + 4 m + z[t-1], m = 0.5 x[t-1]
            // + shock) has nested direct effect 3/2 and path effect (1/2)(4) = 2; the shock is a
            // fixed pattern, so the adjusted regression is exact for y and within sampling
            // error of 1/2 for the x -> m edge.
            let tolerance = if key == "nested" { 1e-8 } else { 0.1 };
            assert!(
                (ate - structural).abs() < tolerance,
                "{key}: {ate} vs structural effect {structural}"
            );
        }
        let rule = result
            .identification
            .arena
            .derivation(result.estimand.functional)
            .map(|meta| meta.rule.to_string());
        match key {
            "nested" => {
                assert_eq!(rule.as_deref(), Some("temporal.cross_world.direct_slope"));
                assert!(!result.estimand.adjustment_set.is_empty(), "{key}");
            }
            "path" => {
                assert_eq!(rule.as_deref(), Some("temporal.cross_world.mediator_edge_product"));
                assert!(!result.estimand.adjustment_set.is_empty(), "{key}");
            }
            "counterfactual" | "distribution" => {
                assert_eq!(rule.as_deref(), Some("backdoor.adjustment"));
            }
            _ => panic!("{key}"),
        }
    }
}

#[test]
fn cross_world_artifacts_are_consumed_with_the_point_they_publish() {
    let data = series();
    let graph = graph();
    let ctx = ExecutionContext::for_tests(11);
    for (key, query, _) in queries() {
        for bayesian in [false, true] {
            let builder = Study::series(data.clone()).query(query.clone()).graph(graph.clone());
            let builder = builder.refute(RefuteSuite::None);
            let builder = if bayesian {
                builder.inference(InferenceMode::Bayesian(
                    BayesianConfig::conjugate().prior_scale(1_000.0).n_draws(8),
                ))
            } else {
                builder
            };
            let prepared = builder.build().unwrap().prepare(&ctx).unwrap();
            let result = prepared.estimate_retained(&ctx).unwrap();
            let bytes = prepared
                .encode_contracted_result(&result, "temporal-cross-world", &ctx)
                .unwrap_or_else(|error| panic!("{key} bayesian={bayesian}: {error}"));
            let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
            assert_eq!(consumed.body.estimate, Some(result.effect()), "{key} bayesian={bayesian}");
            assert!(consumed.contract.is_some(), "{key} bayesian={bayesian}");
        }
    }
}

/// `x → m1 → m2 → y` with a direct `x → y` edge: `m1 = 0.5 x_{t-1} + shock`, `m2 = 2 m1 +
/// shock`, `y = 1.5 x_{t-1} + 3 m2`, so the path through both mediators is `0.5 · 2 · 3`.
fn chain_series() -> TimeSeriesData {
    let n = 400;
    let mut x = vec![0.0; n];
    let mut m1 = vec![0.0; n];
    let mut m2 = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 0..n {
        x[t] = ((t as f64) * 0.37).sin();
        if t > 0 {
            let first_shock = if t % 2 == 0 { 0.3 } else { -0.3 };
            m1[t] = 0.5 * x[t - 1] + first_shock;
            m2[t] = 2.0 * m1[t] + 0.2 * ((t as f64) * 1.3).cos();
            y[t] = 1.5 * x[t - 1] + 3.0 * m2[t];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("m1", m1.as_slice()), ("m2", m2.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

fn chain_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m1 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let m2 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, m1).unwrap();
    graph.insert_directed(m1, m2).unwrap();
    graph.insert_directed(m2, y0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph
}

#[test]
fn a_multi_node_path_is_the_product_of_the_edge_coefficients_along_it() {
    let ids = [1, 2].map(VariableId::from_raw);
    let query = |nodes: [VariableId; 2]| {
        CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(3))
                .with_path_nodes(nodes),
        )
    };
    let ctx = ExecutionContext::for_tests(13);
    let run = |nodes: [VariableId; 2], class: bool| {
        Study::series(chain_series())
            .query(query(nodes))
            .graph(if class {
                AcceptedGraph::from(TemporalCpdag::from_temporal_dag(&chain_graph()))
            } else {
                AcceptedGraph::from(chain_graph())
            })
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .and_then(|prepared| prepared.estimate_retained(&ctx))
    };
    for class in [false, true] {
        let result = run(ids, class).unwrap();
        let ate = result.estimate().as_effect().unwrap().ate;
        assert!((ate - 3.0).abs() < 0.15, "two-node path product {ate}");
        assert_eq!(result.estimand.mediators.as_ref(), &ids);
        assert!(matches!(
            result.identification.arena.node(result.estimand.functional),
            antecedent_expr::ExprNode::Product(_)
        ));
        // Listed against the graph's causal order, the nodes are not a directed chain.
        let reversed =
            run([ids[1], ids[0]], class).expect_err("a reversed node order is not a chain");
        assert!(
            reversed.to_string().contains("path nodes must be listed in causal order"),
            "{reversed}"
        );
    }
}
