//! `TemporalAdmg` twins of the licensed static ADMG analyses.
//!
//! Each variable appears once, at lag 0. Identification is general.id on that
//! contemporaneous projection, so the functional is the static ADMG functional.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "one loop covers every licensed coordinate")]

use std::sync::Arc;

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, EstimatorId, IdentifierId, InferenceMode,
    RefuteSuite, Study, SupportCell, TransportTrialSpec,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    InterventionalDistributionQuery, Lag, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, adjacency_mask_from_admg, set_edge,
};
use antecedent_graph::{Admg, DenseNodeId, TemporalAdmg};
use antecedent_io::consume_analysis_result;
use antecedent_prob::InferenceDiagnostics;

fn load_pin(path: &str) -> serde_json::Value {
    match path {
        "conformance/estimate/admg_frontdoor_functional" => serde_json::from_str(include_str!(
            "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
        ))
        .unwrap(),
        "conformance/estimate/admg_frontdoor_distribution" => serde_json::from_str(include_str!(
            "../../../conformance/estimate/admg_frontdoor_distribution/expected.json"
        ))
        .unwrap(),
        "conformance/estimate/staged_transport" => serde_json::from_str(include_str!(
            "../../../conformance/estimate/staged_transport/expected.json"
        ))
        .unwrap(),
        "conformance/estimate/bayesian_trial_transport" => serde_json::from_str(include_str!(
            "../../../conformance/estimate/bayesian_trial_transport/expected.json"
        ))
        .unwrap(),
        other => panic!("unloaded fixture {other}"),
    }
}

fn expand(pin: &serde_json::Value) -> (TabularData, Admg) {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|value| value.as_str().unwrap()).collect();
    let mut values = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let data = TabularData::from_f64_columns(
        columns.iter().zip(&values).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap();
    let node = |name: &str| {
        DenseNodeId::from_raw(
            u32::try_from(columns.iter().position(|column| *column == name).unwrap()).unwrap(),
        )
    };
    let mut graph = Admg::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        graph
            .insert_directed(node(edge[0].as_str().unwrap()), node(edge[1].as_str().unwrap()))
            .unwrap();
    }
    for edge in pin["graph"]["bidirected_edges"].as_array().unwrap() {
        graph
            .insert_bidirected(node(edge[0].as_str().unwrap()), node(edge[1].as_str().unwrap()))
            .unwrap();
    }
    (data, graph)
}

fn lift(graph: &Admg) -> TemporalAdmg {
    TemporalAdmg::from_static_contemporaneous(graph).unwrap()
}

fn suite_of(name: &str) -> RefuteSuite {
    match name {
        "none" => RefuteSuite::None,
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        other => panic!("unexpected suite {other}"),
    }
}

fn inference_of(bayesian: bool, draws: usize) -> InferenceMode {
    if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws))
    } else {
        InferenceMode::Frequentist
    }
}

fn posterior(graph: &Admg, unidentified: bool) -> GraphPosterior {
    let n = graph.node_count();
    let identified = adjacency_mask_from_admg(graph).unwrap();
    let mut cyclic = set_edge(0, n, 0, 1, true);
    cyclic = set_edge(cyclic, n, 1, 2, true);
    cyclic = set_edge(cyclic, n, 2, 0, true);
    GraphPosterior::new(
        n,
        vec![0.8, 0.2],
        if unidentified { vec![identified, cyclic] } else { vec![identified, identified] },
        vec![0.0; n * n],
        vec![0.0; n * n],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("temporal_admg_graph_posterior"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::TemporalAdmg)
}

fn response_value(result: &antecedent::StudyResult) -> f64 {
    match &result.response().expect("response").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => *value,
        other => panic!("expected a point-identified scalar, got {other:?}"),
    }
}

fn curve_means(result: &antecedent::StudyResult) -> Vec<f64> {
    match &result.response().expect("response").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            mean.to_vec()
        }
        other => panic!("expected a point-identified surface, got {other:?}"),
    }
}

fn frontdoor() -> (TabularData, Admg, f64) {
    let pin = load_pin("conformance/estimate/admg_frontdoor_functional");
    let (data, graph) = expand(&pin);
    let expected = pin["frequentist"]["expected_ate"].as_f64().unwrap();
    (data, graph, expected)
}

#[test]
fn temporal_admg_average_effect_explicit_and_accepted() {
    let (data, graph, expected) = frontdoor();
    let temporal = lift(&graph);
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(2), 0.0, 1.0);
    for accepted in [false, true] {
        for bayesian in [false, true] {
            for validation in ["none", "cheap", "full"] {
                let coordinate = format!(
                    "AverageEffect:TemporalAdmg:{}:{}:{validation}",
                    if accepted { "accepted" } else { "explicit" },
                    if bayesian { "Bayesian" } else { "Frequentist" }
                );
                let base = Study::tabular(data.clone())
                    .query(query.clone())
                    .identifier(IdentifierId::GeneralId)
                    .estimator(EstimatorId::FunctionalEffect)
                    .inference(inference_of(bayesian, 256))
                    .refute(suite_of(validation))
                    .bootstrap_replicates(0);
                let builder = if accepted {
                    base.graph(AcceptedGraph::from(temporal.clone()))
                } else {
                    base.graph(temporal.clone())
                };
                let ctx = ExecutionContext::for_tests(11_011);
                let study =
                    builder.clone().build().unwrap_or_else(|error| panic!("{coordinate}: {error}"));
                let mut prepared =
                    study.prepare(&ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
                drop(builder);
                drop(study);
                assert_eq!(prepared.support_status(), Some(CellStatus::Licensed), "{coordinate}");
                let (RetainedOperation::FunctionalEffect(program)
                | RetainedOperation::PathSpecificEffect(program)) = prepared.retained_operation()
                else {
                    panic!("{coordinate} omitted the functional program")
                };
                assert_eq!(program.mapping().source, program.mapping().executable);
                assert_eq!(
                    prepared.plan().logical.record.identifier.as_deref(),
                    Some("general.id")
                );
                let result = prepared
                    .estimate(&data, &ctx)
                    .unwrap_or_else(|error| panic!("{coordinate}: {error}"));
                let tolerance = if bayesian { 0.08 } else { 1e-9 };
                assert!(
                    (result.effect() - expected).abs() < tolerance,
                    "{coordinate}: {} vs {expected}",
                    result.effect()
                );
                if validation != "none" {
                    assert!(!result.refutations.is_empty(), "{coordinate} ran no refuter");
                }
                let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
                assert!((refreshed.effect() - result.effect()).abs() < 1e-9, "{coordinate}");
                let artifact =
                    prepared.encode_contracted_result(&refreshed, &coordinate, &ctx).unwrap();
                let consumed = consume_analysis_result(&artifact).unwrap();
                assert_eq!(consumed.body.estimate, Some(refreshed.effect()), "{coordinate}");
            }
        }
    }
}

#[test]
fn temporal_admg_average_effect_graph_posterior() {
    let (data, graph, expected) = frontdoor();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(2), 0.0, 1.0);
    let graphs = posterior(&graph, true);
    for bayesian in [false, true] {
        for validation in ["none", "cheap", "full"] {
            let coordinate = format!(
                "AverageEffect:TemporalAdmg:graph_posterior:{}:{validation}",
                if bayesian { "Bayesian" } else { "Frequentist" }
            );
            let builder = Study::tabular(data.clone())
                .graph_posterior(graphs.clone())
                .query(query.clone())
                .inference(inference_of(bayesian, 32))
                .refute(suite_of(validation))
                .bootstrap_replicates(0)
                .build()
                .unwrap_or_else(|error| panic!("{coordinate}: {error}"));
            let ctx = ExecutionContext::for_tests(11_012);
            let mut prepared =
                builder.prepare(&ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
            drop(builder);
            let RetainedOperation::GraphPosteriorEffect(plan) = prepared.retained_operation()
            else {
                panic!("{coordinate} omitted the posterior plan")
            };
            assert_eq!(plan.estimator.as_str(), "functional.effect");
            assert_eq!(plan.validation, suite_of(validation));
            let result = prepared
                .estimate(&data, &ctx)
                .unwrap_or_else(|error| panic!("{coordinate}: {error}"));
            let tolerance = if bayesian { 0.05 } else { 1e-9 };
            assert!(
                (result.estimate.as_effect().unwrap().ate - expected).abs() < tolerance,
                "{coordinate}: {} vs {expected}",
                result.estimate.as_effect().unwrap().ate
            );
            let mixture = result.structural_response.as_ref().expect("mixture");
            assert!((mixture.identified_mass - 0.8).abs() < 1e-12, "{coordinate}");
            assert!((mixture.unidentified_mass - 0.2).abs() < 1e-12, "{coordinate}");
            if validation == "none" {
                assert!(result.refutations.is_empty(), "{coordinate}");
            } else {
                assert!(!result.refutations.is_empty(), "{coordinate}");
            }
            let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
            assert!(
                (refreshed.estimate.as_effect().unwrap().ate
                    - result.estimate.as_effect().unwrap().ate)
                    .abs()
                    < 1e-9
            );
            let artifact =
                prepared.encode_contracted_result(&refreshed, &coordinate, &ctx).unwrap();
            let consumed = consume_analysis_result(&artifact).unwrap();
            assert!(
                consumed.acceptance.unresolved.iter().any(|reason| {
                    reason.as_ref() == "dependencies.checked_graph_posterior_effect_operation"
                }),
                "{coordinate}: {:?}",
                consumed.acceptance.unresolved
            );
        }
    }
}

fn intervention_query(level: f64) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(level))]),
    })
}

#[test]
fn temporal_admg_intervention_response_explicit_and_accepted() {
    let (data, graph, expected_ate) = frontdoor();
    let temporal = lift(&graph);
    let levels = [(0.0, 0.314), (1.0, 0.596)];
    assert!(((levels[1].1 - levels[0].1) - expected_ate).abs() < 1e-12);
    for accepted in [false, true] {
        for bayesian in [false, true] {
            for validation in ["none", "cheap", "full"] {
                for (level, truth) in levels {
                    let coordinate = format!(
                        "InterventionResponse:TemporalAdmg:{}:{}:{validation}",
                        if accepted { "accepted" } else { "explicit" },
                        if bayesian { "Bayesian" } else { "Frequentist" }
                    );
                    let query = intervention_query(level);
                    let base = Study::tabular(data.clone())
                        .query(CausalQuery::Response(query))
                        .identifier(IdentifierId::GeneralId)
                        .estimator(EstimatorId::FunctionalEffect)
                        .inference(inference_of(bayesian, 256))
                        .refute(suite_of(validation))
                        .bootstrap_replicates(0);
                    let builder = if accepted {
                        base.graph(AcceptedGraph::from(temporal.clone()))
                    } else {
                        base.graph(temporal.clone())
                    };
                    let ctx = ExecutionContext::for_tests(11_013);
                    let study =
                        builder.clone().build().unwrap_or_else(|e| panic!("{coordinate}: {e}"));
                    let mut prepared =
                        study.prepare(&ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
                    drop(builder);
                    drop(study);
                    assert_eq!(
                        prepared.support_status(),
                        Some(CellStatus::Licensed),
                        "{coordinate}"
                    );
                    let RetainedOperation::AdmgResponseCurve(members) =
                        prepared.retained_operation()
                    else {
                        panic!("{coordinate} omitted response members")
                    };
                    assert_eq!(
                        members[0].program().mapping().source,
                        members[0].program().mapping().executable
                    );
                    let result = prepared
                        .estimate(&data, &ctx)
                        .unwrap_or_else(|e| panic!("{coordinate}: {e}"));
                    let tolerance = if bayesian { 0.08 } else { 1e-9 };
                    assert!(
                        (response_value(&result) - truth).abs() < tolerance,
                        "{coordinate} level {level}: {}",
                        response_value(&result)
                    );
                    if validation != "none" {
                        assert!(
                            result
                                .refutations
                                .iter()
                                .any(|report| { report.refuter.as_ref() == "overlap.assessment" }),
                            "{coordinate}: {:?}",
                            result.refutations
                        );
                    }
                    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
                    let artifact =
                        prepared.encode_contracted_result(&refreshed, &coordinate, &ctx).unwrap();
                    let consumed = consume_analysis_result(&artifact).unwrap();
                    assert!(
                        consumed.acceptance.accepts_as_verified_program()
                            || !consumed.acceptance.unresolved.is_empty(),
                        "{coordinate}: {:?}",
                        consumed.acceptance.unresolved
                    );
                }
            }
        }
    }
}

#[test]
fn temporal_admg_intervention_response_graph_posterior() {
    let (data, graph, _) = frontdoor();
    let query = intervention_query(1.0);
    for bayesian in [false, true] {
        for validation in ["none", "cheap", "full"] {
            let coordinate = format!(
                "InterventionResponse:TemporalAdmg:graph_posterior:{}:{validation}",
                if bayesian { "Bayesian" } else { "Frequentist" }
            );
            let graphs = posterior(&graph, validation == "none");
            let builder = Study::tabular(data.clone())
                .graph_posterior(graphs)
                .query(CausalQuery::Response(query.clone()))
                .inference(inference_of(bayesian, 48))
                .refute(suite_of(validation))
                .bootstrap_replicates(0)
                .build()
                .unwrap_or_else(|e| panic!("{coordinate}: {e}"));
            let ctx = ExecutionContext::for_tests(11_014);
            let mut prepared =
                builder.prepare(&ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
            drop(builder);
            assert!(matches!(
                prepared.retained_operation(),
                RetainedOperation::AdmgGraphPosteriorResponse(..)
            ));
            let RetainedOperation::AdmgGraphPosteriorResponse(plan) = prepared.retained_operation()
            else {
                panic!("{coordinate} omitted the ADMG graph-posterior response plan")
            };
            assert_eq!(plan.validation, suite_of(validation));
            let result =
                prepared.estimate(&data, &ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
            let mixture = result.structural_response.as_ref().expect("mixture");
            let unidentified = if validation == "none" { 0.2 } else { 0.0 };
            assert!((mixture.unidentified_mass - unidentified).abs() < 1e-12, "{coordinate}");
            if validation != "none" {
                assert!(
                    (response_value(&result) - 0.596).abs() < if bayesian { 0.08 } else { 1e-9 },
                    "{coordinate}: {}",
                    response_value(&result)
                );
                assert!(!result.refutations.is_empty(), "{coordinate}");
            }
            let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
            let artifact =
                prepared.encode_contracted_result(&refreshed, &coordinate, &ctx).unwrap();
            let consumed = consume_analysis_result(&artifact).unwrap();
            assert!(consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.checked_admg_graph_posterior_response_operation"
            }));
        }
    }
}

fn curve_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(2),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
}

#[test]
fn temporal_admg_response_curve_explicit_and_accepted() {
    let (data, graph, expected_ate) = frontdoor();
    let temporal = lift(&graph);
    let query = curve_query();
    let expected = [0.314, 0.596];
    assert!(((expected[1] - expected[0]) - expected_ate).abs() < 1e-12);
    for accepted in [false, true] {
        for bayesian in [false, true] {
            let coordinate = format!(
                "ResponseCurve:TemporalAdmg:{}:{}:none",
                if accepted { "accepted" } else { "explicit" },
                if bayesian { "Bayesian" } else { "Frequentist" }
            );
            let base = Study::tabular(data.clone())
                .query(CausalQuery::Response(query.clone()))
                .identifier(IdentifierId::GeneralId)
                .estimator(EstimatorId::FunctionalEffect)
                .inference(inference_of(bayesian, 256))
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0);
            let builder = if accepted {
                base.graph(AcceptedGraph::from(temporal.clone()))
            } else {
                base.graph(temporal.clone())
            };
            let ctx = ExecutionContext::for_tests(11_015);
            let study = builder.clone().build().unwrap_or_else(|e| panic!("{coordinate}: {e}"));
            let mut prepared = study.prepare(&ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
            drop(builder);
            drop(study);
            assert_eq!(prepared.support_status(), Some(CellStatus::Licensed), "{coordinate}");
            assert!(matches!(
                prepared.retained_operation(),
                RetainedOperation::AdmgResponseCurve(..)
            ));
            assert_eq!(
                prepared.plan().logical.record.estimator.as_deref(),
                Some("functional.effect")
            );
            let result =
                prepared.estimate(&data, &ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
            let tolerance = if bayesian { 0.08 } else { 1e-9 };
            for (actual, truth) in curve_means(&result).iter().zip(expected) {
                assert!((actual - truth).abs() < tolerance, "{coordinate}: {actual} vs {truth}");
            }
            let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
            let artifact =
                prepared.encode_contracted_result(&refreshed, &coordinate, &ctx).unwrap();
            let consumed = consume_analysis_result(&artifact).unwrap();
            assert!(
                consumed.acceptance.accepts_as_verified_program()
                    || !consumed.acceptance.unresolved.is_empty()
            );
        }
    }
}

#[test]
fn temporal_admg_response_curve_graph_posterior() {
    let (data, graph, _) = frontdoor();
    let query = curve_query();
    let graphs = posterior(&graph, false);
    for bayesian in [false, true] {
        let coordinate = format!(
            "ResponseCurve:TemporalAdmg:graph_posterior:{}:none",
            if bayesian { "Bayesian" } else { "Frequentist" }
        );
        let builder = Study::tabular(data.clone())
            .graph_posterior(graphs.clone())
            .query(CausalQuery::Response(query.clone()))
            .inference(inference_of(bayesian, 128))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap_or_else(|e| panic!("{coordinate}: {e}"));
        let ctx = ExecutionContext::for_tests(11_016);
        let mut prepared = builder.prepare(&ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
        drop(builder);
        let RetainedOperation::AdmgGraphPosteriorResponse(plan) = prepared.retained_operation()
        else {
            panic!("{coordinate} omitted the ADMG graph-posterior response plan")
        };
        assert_eq!(plan.validation, RefuteSuite::None);
        let result = prepared.estimate(&data, &ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
        for (actual, truth) in curve_means(&result).iter().zip([0.314, 0.596]) {
            assert!((actual - truth).abs() < if bayesian { 0.08 } else { 1e-9 }, "{coordinate}");
        }
        let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
        let artifact = prepared.encode_contracted_result(&refreshed, &coordinate, &ctx).unwrap();
        let consumed = consume_analysis_result(&artifact).unwrap();
        assert!(consumed.acceptance.unresolved.iter().any(|reason| {
            reason.as_ref() == "dependencies.checked_admg_graph_posterior_response_operation"
        }));
    }
}

#[test]
fn temporal_admg_interventional_distribution_explicit_and_accepted() {
    let pin = load_pin("conformance/estimate/admg_frontdoor_distribution");
    let (data, graph) = expand(&pin);
    let temporal = lift(&graph);
    let truth: Vec<(f64, f64)> = pin["truth"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| (row["t"].as_f64().unwrap(), row["p_y1"].as_f64().unwrap()))
        .collect();
    for accepted in [false, true] {
        for bayesian in [false, true] {
            let coordinate = format!(
                "InterventionalDistribution:TemporalAdmg:{}:{}:none",
                if accepted { "accepted" } else { "explicit" },
                if bayesian { "Bayesian" } else { "Frequentist" }
            );
            for (level, p_truth) in &truth {
                let query = InterventionalDistributionQuery::new(
                    VariableId::from_raw(2),
                    [Intervention::set(VariableId::from_raw(0), Value::f64(*level))],
                );
                let base = Study::tabular(data.clone())
                    .query(CausalQuery::Distribution(query))
                    .identifier(IdentifierId::GeneralId)
                    .estimator(EstimatorId::FunctionalDistribution)
                    .inference(inference_of(bayesian, 256))
                    .refute(RefuteSuite::None)
                    .bootstrap_replicates(0);
                let builder = if accepted {
                    base.graph(AcceptedGraph::from(temporal.clone()))
                } else {
                    base.graph(temporal.clone())
                };
                let ctx = ExecutionContext::for_tests(11_017);
                let study = builder.clone().build().unwrap_or_else(|e| panic!("{coordinate}: {e}"));
                let prepared = study.prepare(&ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
                drop(builder);
                drop(study);
                assert_eq!(prepared.support_status(), Some(CellStatus::Licensed), "{coordinate}");
                let RetainedOperation::Distribution(program) = prepared.retained_operation() else {
                    panic!("distribution program")
                };
                assert_eq!(program.mapping().source, program.mapping().executable);
                let result =
                    prepared.estimate(&data, &ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
                let mean = result.distribution().expect("distribution").mean;
                assert!(
                    (mean - p_truth).abs() < if bayesian { 0.08 } else { 1e-9 },
                    "{coordinate} do(t={level}): {mean} vs {p_truth}"
                );
                let artifact =
                    prepared.encode_contracted_result(&result, &coordinate, &ctx).unwrap();
                let consumed = consume_analysis_result(&artifact).unwrap();
                assert!(
                    consumed.acceptance.accepts_as_verified_program()
                        || !consumed.acceptance.unresolved.is_empty(),
                    "{coordinate}: {:?}",
                    consumed.acceptance.unresolved
                );
            }
        }
    }
}

#[test]
fn temporal_admg_transport_explicit() {
    let ipw_pin = load_pin("conformance/estimate/staged_transport");
    let bayesian_pin = load_pin("conformance/estimate/bayesian_trial_transport");
    let ipw = ipw_pin["ipw"].as_f64().unwrap();
    let ipw_tolerance = ipw_pin["tolerance"].as_f64().unwrap();
    let data = TabularData::from_f64_columns([
        ("a", &[1.0, 0.0, 0.0, 0.0][..]),
        ("y", &[3.0, 1.0, 0.0, 0.0][..]),
        ("trial", &[1.0, 1.0, 0.0, 0.0][..]),
        ("s", &[0.5, 0.5, 0.5, 0.5][..]),
        ("e", &[0.5, 0.5, 0.5, 0.5][..]),
    ])
    .unwrap();
    let mut admg = Admg::with_variables(5);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let temporal = lift(&admg);
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    });
    let query = antecedent_core::TransportQuery::new(
        response,
        "trial",
        "target",
        [VariableId::from_raw(0)],
    );
    for bayesian in [false, true] {
        let coordinate = format!(
            "TransportQuery:TemporalAdmg:explicit:{}:none",
            if bayesian { "Bayesian" } else { "Frequentist" }
        );
        let study = Study::tabular(data.clone())
            .graph(temporal.clone())
            .query(CausalQuery::Transport(query.clone()))
            .selection_targets(Arc::from([]))
            .transport_trial(TransportTrialSpec {
                trial: VariableId::from_raw(2),
                selection_probability: VariableId::from_raw(3),
                treatment_probability: VariableId::from_raw(4),
            })
            .inference(if bayesian {
                InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(2_000))
            } else {
                InferenceMode::Frequentist
            })
            .refute(RefuteSuite::None);
        let builder = study.build().unwrap_or_else(|e| panic!("{coordinate}: {e}"));
        let ctx = ExecutionContext::for_tests(11_018);
        let prepared = builder.prepare(&ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
        drop(builder);
        let RetainedOperation::TransportTrial(plan) = prepared.retained_operation() else {
            panic!("transport plan")
        };
        assert_eq!(plan.identifier, IdentifierId::TransportSid);
        let result = prepared.estimate(&data, &ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
        if bayesian {
            let posterior = result.posterior.as_ref().expect("transport posterior");
            assert!(
                (posterior.summaries.mean[0] - bayesian_pin["expected_mean"].as_f64().unwrap())
                    .abs()
                    < bayesian_pin["mean_tolerance"].as_f64().unwrap(),
                "{coordinate}"
            );
        } else {
            let transported = result.transport().expect("transport");
            assert!(
                (transported.ipw - ipw).abs() <= ipw_tolerance,
                "{coordinate}: {}",
                transported.ipw
            );
        }
        let artifact = prepared.encode_contracted_result(&result, &coordinate, &ctx).unwrap();
        let consumed = consume_analysis_result(&artifact).unwrap();
        assert!(
            consumed.acceptance.unresolved.iter().any(|reason| {
                reason.as_ref() == "dependencies.checked_transport_trial_operation"
            })
        );
    }
}

#[test]
fn repeated_variable_temporal_admg_is_refused() {
    let mut graph = TemporalAdmg::empty();
    let past = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let now = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(past, now).unwrap();
    assert!(graph.project_preserving_functional().is_err());
    let data =
        TabularData::from_f64_columns([("t", &[0.0, 1.0][..]), ("y", &[0.0, 1.0][..])]).unwrap();
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    let built = Study::tabular(data).graph(graph).query(query).build();
    let Err(error) = built.and_then(|study| study.prepare(&ExecutionContext::for_tests(11_019)))
    else {
        panic!("a repeated variable must not identify as a different estimand");
    };
    let text = error.to_string();
    assert!(
        text.contains("projection") || text.contains("variable") || text.contains("Compile"),
        "{text}"
    );
}

#[test]
fn response_curve_refuters_are_not_applicable_on_temporal_admg() {
    let status = antecedent::support::classify(
        SupportCell::from_labels(
            "ResponseCurve",
            "TemporalAdmg",
            "explicit",
            "Frequentist",
            "cheap",
        )
        .expect("support axis"),
    );
    assert!(
        matches!(status, CellStatus::NotApplicable { .. }),
        "a function-valued curve has no ATE refuter: {status:?}"
    );
    let static_on_temporal_dag = antecedent::support::classify(
        SupportCell::from_labels(
            "ConditionalEffect",
            "TemporalDag",
            "explicit",
            "Frequentist",
            "none",
        )
        .expect("support axis"),
    );
    assert!(matches!(static_on_temporal_dag, CellStatus::NotApplicable { .. }));
    let distribution_cheap = antecedent::support::classify(
        SupportCell::from_labels(
            "InterventionalDistribution",
            "TemporalAdmg",
            "explicit",
            "Frequentist",
            "cheap",
        )
        .expect("support axis"),
    );
    assert!(
        matches!(distribution_cheap, CellStatus::Refused),
        "distribution cheap/full is not a TemporalAdmg license: {distribution_cheap:?}"
    );
    let transport_on_temporal_dag = antecedent::support::classify(
        SupportCell::from_labels(
            "TransportQuery",
            "TemporalDag",
            "explicit",
            "Frequentist",
            "none",
        )
        .expect("support axis"),
    );
    assert!(matches!(transport_on_temporal_dag, CellStatus::NotApplicable { .. }));
    let curve_posterior_cheap = antecedent::support::classify(
        SupportCell::from_labels(
            "ResponseCurve",
            "TemporalAdmg",
            "graph_posterior",
            "Frequentist",
            "cheap",
        )
        .expect("support axis"),
    );
    assert!(
        matches!(curve_posterior_cheap, CellStatus::Refused),
        "graph-posterior curve refuters stay closed: {curve_posterior_cheap:?}"
    );
}
