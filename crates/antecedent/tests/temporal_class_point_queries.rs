//! `TemporalCpdag` and `TemporalPag` conditional, derivative, and cross-world points.
//!
//! The class result is the completion-weighted mean of the existing temporal
//! linear producer. A single directed completion matches the `TemporalDag` pin.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyBuilder};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, DerivativeScale, DerivativeWeighting, ExecutionContext,
    Intervention, InterventionalDistributionQuery, Lag, NestedCounterfactualQuery, NodeRef,
    PathSpecificEffectQuery, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseValue, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, set_edge,
};
use antecedent_graph::{TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged};
use antecedent_prob::InferenceDiagnostics;

fn series_from(columns: &[(&'static str, Vec<f64>)]) -> TimeSeriesData {
    let refs: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (*name, values.as_slice())).collect();
    TimeSeriesData::from_f64_columns(refs, 1).unwrap()
}

fn with_witness(series_columns: Vec<(&'static str, Vec<f64>)>) -> TimeSeriesData {
    let n = series_columns[0].1.len();
    let mut columns = series_columns;
    columns.push(("r", (0..n).map(|t| ((t as f64) * 0.17).cos()).collect()));
    series_from(&columns)
}

fn conditional_columns() -> Vec<(&'static str, Vec<f64>)> {
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
    vec![("x", x), ("y", y), ("w", w)]
}

fn conditional_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(w0, y0).unwrap();
    graph
}

fn derivative_columns() -> Vec<(&'static str, Vec<f64>)> {
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
    vec![("x", x), ("b", b), ("y", y), ("v", v)]
}

fn derivative_dag() -> TemporalDag {
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

fn cross_columns() -> Vec<(&'static str, Vec<f64>)> {
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
    vec![("x", x), ("m", m), ("y", y)]
}

fn cross_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, m0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(m0, y0).unwrap();
    graph
}

fn directed_pag(dag: &TemporalDag) -> TemporalPag {
    let mut pag = TemporalPag::empty();
    let mut ids = Vec::new();
    for node in dag.nodes() {
        let NodeRef::Lagged { variable, lag } = *node else {
            panic!("temporal dag nodes are lagged");
        };
        ids.push(pag.add_lagged(variable, lag).unwrap());
    }
    for edge in dag.edges() {
        pag.insert_directed(ids[edge.a.as_usize()], ids[edge.b.as_usize()]).unwrap();
    }
    let witness = dag
        .nodes()
        .iter()
        .filter_map(|node| match node {
            NodeRef::Lagged { variable, .. } => Some(variable.raw()),
            _ => None,
        })
        .max()
        .unwrap()
        + 1;
    let r1 = pag.add_lagged(VariableId::from_raw(witness), Lag::from_raw(1)).unwrap();
    let treatment = dag
        .nodes()
        .iter()
        .position(|node| {
            matches!(
                node,
                NodeRef::Lagged { variable, lag } if variable.raw() == 0 && *lag == Lag::from_raw(1)
            )
        })
        .unwrap();
    pag.insert_directed(r1, ids[treatment]).unwrap();
    pag
}

fn bind(builder: StudyBuilder, dag: &TemporalDag, pag: bool, accepted: bool) -> StudyBuilder {
    if pag {
        let graph = directed_pag(dag);
        if accepted { builder.graph(AcceptedGraph::from(graph)) } else { builder.graph(graph) }
    } else {
        let graph = TemporalCpdag::from_temporal_dag(dag);
        if accepted { builder.graph(AcceptedGraph::from(graph)) } else { builder.graph(graph) }
    }
}

fn lag_bit(n: usize, from: usize, to: usize) -> u64 {
    1u64 << (from * n + to)
}

fn posterior_masks(
    base_n: usize,
    lag_edges: &[(usize, usize)],
    contemporaneous: &[(usize, usize)],
    pag: bool,
) -> (usize, AdjacencyMask, u64) {
    let n = if pag { base_n + 1 } else { base_n };
    let mut lag_mask = 0u64;
    for &(from, to) in lag_edges {
        lag_mask |= lag_bit(n, from, to);
    }
    let mut adjacency = AdjacencyMask::EMPTY;
    for &(from, to) in contemporaneous {
        adjacency = set_edge(adjacency, n, from, to, true);
    }
    if pag {
        adjacency = set_edge(adjacency, n, base_n, 0, true);
    }
    (n, adjacency, lag_mask)
}

fn class_posterior(n: usize, adjacency: AdjacencyMask, lag_mask: u64, pag: bool) -> GraphPosterior {
    let cell = n * n;
    GraphPosterior::new(
        n,
        vec![1.0],
        vec![adjacency],
        vec![0.0; cell],
        vec![0.0; cell],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("temporal-class-point"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(if pag { GraphPosteriorAtomKind::Pag } else { GraphPosteriorAtomKind::Cpdag })
    .with_lagged_marginals(1, vec![0.0; cell])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

fn bayes(builder: StudyBuilder, bayesian: bool) -> StudyBuilder {
    if bayesian {
        builder.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().prior_scale(1_000.0).n_draws(8),
        ))
    } else {
        builder
    }
}

fn run(builder: StudyBuilder) -> antecedent::StudyResult {
    let ctx = ExecutionContext::for_tests(11);
    let study = builder.clone().build().unwrap();
    assert_eq!(study.support_status(), Some(antecedent::support::CellStatus::Licensed));
    let prepared = study.prepare(&ctx).unwrap();
    drop(builder);
    drop(study);
    prepared.estimate_retained(&ctx).unwrap()
}

fn conditional_query(sustained: bool) -> TemporalEffectQuery {
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
fn temporal_class_conditional_pulse_and_sustained_publish_the_marginal_point() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_conditional/expected.json"
    ))
    .unwrap();
    let truth = pin["marginal_effect"].as_f64().unwrap();
    let columns = conditional_columns();
    let dag = conditional_dag();
    for sustained in [false, true] {
        for pag in [false, true] {
            for posterior in [false, true] {
                for accepted in [false, true] {
                    if posterior && accepted {
                        continue;
                    }
                    for bayesian in [false, true] {
                        let data =
                            if pag { with_witness(columns.clone()) } else { series_from(&columns) };
                        let (n, conditional_adj, conditional_lag) =
                            posterior_masks(3, &[(0, 1)], &[(2, 1)], pag);
                        let builder = if posterior {
                            Study::series(data)
                                .query(CausalQuery::TemporalEffect(conditional_query(sustained)))
                                .refute(RefuteSuite::None)
                                .graph_posterior(class_posterior(
                                    n,
                                    conditional_adj,
                                    conditional_lag,
                                    pag,
                                ))
                        } else {
                            bind(
                                Study::series(data)
                                    .query(CausalQuery::TemporalEffect(conditional_query(
                                        sustained,
                                    )))
                                    .refute(RefuteSuite::None),
                                &dag,
                                pag,
                                accepted,
                            )
                        };
                        let result = run(bayes(builder, bayesian));
                        assert!(
                            (result.estimate.as_effect().unwrap().ate - truth).abs() < 0.05,
                            "sustained={sustained} pag={pag} posterior={posterior} accepted={accepted} bayesian={bayesian} ate={}",
                            result.estimate.as_effect().unwrap().ate
                        );
                        assert!(result.diagnostics.iter().any(|d| {
                            d.code.as_ref() == "estimate.temporal.conditional_marginal"
                        }));
                        assert!(
                            result
                                .diagnostics
                                .iter()
                                .any(|d| { d.code.as_ref().contains("envelope") })
                        );
                        assert!(result.posterior.is_none());
                        if let Some(binding) = &result.interval {
                            assert_eq!(binding.method, antecedent_core::IntervalMethod::None);
                        }
                        assert_eq!(
                            result.logical_plan.estimator.as_deref(),
                            Some("temporal.linear.adjustment")
                        );
                    }
                }
            }
        }
    }
    let data = series_from(&columns);
    let refused = Study::series(data.clone())
        .query(CausalQuery::TemporalEffect(
            conditional_query(false)
                .with_effect_modifiers([VariableId::from_raw(2), VariableId::from_raw(9)]),
        ))
        .graph(TemporalCpdag::from_temporal_dag(&dag))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let err = refused
        .prepare(&ExecutionContext::for_tests(11))
        .unwrap()
        .estimate_retained(&ExecutionContext::for_tests(11));
    assert!(err.is_err(), "two modifiers are not a one-modifier conditional effect");
    let mapped = Study::series(data)
        .query(CausalQuery::TemporalEffect(conditional_query(false)))
        .graph(TemporalCpdag::from_temporal_dag(&dag))
        .refute(RefuteSuite::None)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().prior_from_artifact(vec![1u8, 2, 3], None),
        ))
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(11))
        .unwrap();
    assert!(mapped.estimate_retained(&ExecutionContext::for_tests(11)).is_err());
}

fn attach(functional: ResponseFunctional) -> ResponseQuery {
    ResponseQuery::new(functional)
        .with_temporal(TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap())
}

fn derivative_values() -> Vec<(&'static str, ResponseFunctional)> {
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
fn temporal_class_derivatives_publish_the_linear_slope() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_derivative/expected.json"
    ))
    .unwrap();
    let tolerance = pin["tolerance"].as_f64().unwrap();
    let columns = derivative_columns();
    let dag = derivative_dag();
    for (key, functional) in derivative_values() {
        for pag in [false, true] {
            for posterior in [false, true] {
                for accepted in [false, true] {
                    if posterior && accepted {
                        continue;
                    }
                    for bayesian in [false, true] {
                        let data =
                            if pag { with_witness(columns.clone()) } else { series_from(&columns) };
                        let (n, derivative_adj, derivative_lag) =
                            posterior_masks(4, &[(0, 2), (1, 2), (0, 3), (1, 3)], &[], pag);
                        let builder = if posterior {
                            Study::series(data)
                                .query(CausalQuery::Response(attach(functional.clone())))
                                .refute(RefuteSuite::None)
                                .graph_posterior(class_posterior(
                                    n,
                                    derivative_adj,
                                    derivative_lag,
                                    pag,
                                ))
                        } else {
                            bind(
                                Study::series(data)
                                    .query(CausalQuery::Response(attach(functional.clone())))
                                    .refute(RefuteSuite::None),
                                &dag,
                                pag,
                                accepted,
                            )
                        };
                        let result = run(bayes(builder, bayesian));
                        let response = result.response().unwrap();
                        let ResponseIdentification::PointIdentified(value) = &response.estimate
                        else {
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
                            pin[key]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|x| x.as_f64().unwrap())
                                .collect()
                        };
                        for (got, want) in got.iter().zip(&expected) {
                            assert!(
                                (got - want).abs() < tolerance,
                                "{key} pag={pag} posterior={posterior} accepted={accepted} bayesian={bayesian}: {got} != {want}"
                            );
                        }
                        assert!(
                            result.diagnostics.iter().any(|d| {
                                d.code.as_ref() == "estimate.temporal.linear_derivative"
                            })
                        );
                        assert!(result.posterior.is_none());
                        let estimator = if bayesian {
                            "response.temporal.bayesian"
                        } else {
                            "temporal.response.gcomp"
                        };
                        assert_eq!(result.logical_plan.estimator.as_deref(), Some(estimator));
                    }
                }
            }
        }
    }
}

fn cross_queries() -> Vec<(&'static str, CausalQuery)> {
    let x = VariableId::from_raw(0);
    let m = VariableId::from_raw(1);
    let y = VariableId::from_raw(2);
    vec![
        (
            "nested",
            CausalQuery::NestedCounterfactual(NestedCounterfactualQuery::new(x, m, y).unwrap()),
        ),
        (
            "path",
            CausalQuery::PathSpecific(PathSpecificEffectQuery::binary(x, y).with_path_nodes([m])),
        ),
        (
            "counterfactual",
            CausalQuery::Counterfactual(
                CounterfactualQuery::new(y, Arc::from([Intervention::set(x, Value::f64(1.0))]))
                    .with_control_level(0.0),
            ),
        ),
        (
            "distribution",
            CausalQuery::Distribution(InterventionalDistributionQuery::new(
                y,
                [Intervention::set(x, Value::f64(1.0))],
            )),
        ),
    ]
}

#[test]
fn temporal_class_nested_contract_encodes() {
    let dag = cross_dag();
    let (_, query) = cross_queries().into_iter().find(|(key, _)| *key == "nested").unwrap();
    let ctx = ExecutionContext::for_tests(11);
    let study = bind(
        Study::series(series_from(&cross_columns())).query(query).refute(RefuteSuite::None),
        &dag,
        false,
        false,
    )
    .build()
    .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate_retained(&ctx).unwrap();
    let encoded = prepared.encode_contracted_result(&result, "nested-class", &ctx);
    assert!(encoded.is_ok(), "{encoded:?}");
}

#[test]
fn temporal_class_cross_world_points_match_the_linear_mechanism() {
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_cross_world/expected.json"
    ))
    .unwrap();
    let tolerance = pin["tolerance"].as_f64().unwrap();
    let columns = cross_columns();
    let dag = cross_dag();
    for (key, query) in cross_queries() {
        let truth = pin[key].as_f64().unwrap();
        for pag in [false, true] {
            for posterior in [false, true] {
                for accepted in [false, true] {
                    if posterior && accepted {
                        continue;
                    }
                    for bayesian in [false, true] {
                        let data =
                            if pag { with_witness(columns.clone()) } else { series_from(&columns) };
                        let (n, cross_adj, cross_lag) =
                            posterior_masks(3, &[(0, 1), (0, 2)], &[(1, 2)], pag);
                        let builder = if posterior {
                            Study::series(data)
                                .query(query.clone())
                                .refute(RefuteSuite::None)
                                .graph_posterior(class_posterior(n, cross_adj, cross_lag, pag))
                        } else {
                            bind(
                                Study::series(data).query(query.clone()).refute(RefuteSuite::None),
                                &dag,
                                pag,
                                accepted,
                            )
                        };
                        let result = run(bayes(builder, bayesian));
                        assert!(
                            (result.estimate.as_effect().unwrap().ate - truth).abs() < tolerance,
                            "{key} pag={pag} posterior={posterior} accepted={accepted} bayesian={bayesian} ate={}",
                            result.estimate.as_effect().unwrap().ate
                        );
                        assert!(
                            result
                                .diagnostics
                                .iter()
                                .any(|d| { d.code.as_ref() == "estimate.temporal.cross_world" })
                        );
                        assert!(result.posterior.is_none());
                        assert_eq!(
                            result.logical_plan.estimator.as_deref(),
                            Some("temporal.linear.adjustment")
                        );
                    }
                }
            }
        }
    }
}
