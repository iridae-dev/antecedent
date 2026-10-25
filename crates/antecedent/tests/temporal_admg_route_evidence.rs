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

/// One series of `n` consecutive times. Treatment is held in runs of 25 and the outcome noise is
/// AR(0.9), so the per-row influence of the trial-to-target contrast is strongly serially
/// dependent. The trial indicator is an independent coin per row, and the supplied selection and
/// treatment probabilities are the true 0.5.
#[allow(clippy::cast_precision_loss, reason = "a uniform draw from the top 53 bits of a word")]
fn dependent_transport_series(n: usize) -> TabularData {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut uniform = move || {
        state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    };
    let (mut a, mut y, mut trial) = (Vec::new(), Vec::new(), Vec::new());
    let (mut arm, mut noise) = (0.0_f64, 0.0_f64);
    for row in 0..n {
        if row % 25 == 0 {
            arm = if uniform() < 0.5 { 1.0 } else { 0.0 };
        }
        let normal =
            (-2.0 * uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * uniform()).cos();
        noise = 0.9 * noise + (1.0_f64 - 0.81).sqrt() * normal;
        a.push(arm);
        y.push(arm + noise);
        trial.push(if uniform() < 0.5 { 1.0 } else { 0.0 });
    }
    let half = vec![0.5; n];
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("trial", trial.as_slice()),
        ("s", half.as_slice()),
        ("e", half.as_slice()),
    ])
    .unwrap()
}

fn transport_study(
    data: &TabularData,
    graph: impl antecedent::IntoGraphInput,
    bayesian: bool,
) -> antecedent::StudyResult {
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
    Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Transport(query))
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
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(11_019))
        .unwrap()
}

/// A temporal transport reports an interval that respects the serial dependence of its rows: the
/// frequentist standard error is the Newey-West long-run variance of the per-row influence at
/// the series block length, and the Bayesian interval is the posterior of one Dirichlet weight
/// per circular block. Both are wider than the same rows read as independent, which a static
/// ADMG does. Both are unlicensed with calibration unmeasured.
#[test]
fn temporal_admg_transport_reports_a_serial_dependence_aware_interval() {
    let data = dependent_transport_series(1000);
    let mut admg = Admg::with_variables(5);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let temporal = lift(&admg);

    let series = transport_study(&data, temporal.clone(), false);
    let rows = transport_study(&data, admg.clone(), false);
    let series_effect = series.estimate.as_effect().expect("a temporal transport effect");
    let rows_effect = rows.estimate.as_effect().expect("a static transport effect");
    assert!(matches!(
        series_effect.uncertainty.analytic.kind,
        Some(antecedent_estimate::AnalyticSeKind::NeweyWest { lag: 10 })
    ));
    assert!(
        (series_effect.ate - rows_effect.ate).abs() < 1e-12,
        "the point estimate is the same contrast"
    );
    let (series_se, rows_se) =
        (series_effect.uncertainty.analytic.se, rows_effect.uncertainty.analytic.se);
    assert!(series_se.is_finite() && rows_se.is_finite() && rows_se > 0.0);
    assert!(series_se > 1.3 * rows_se, "Newey-West {series_se} vs independent rows {rows_se}");

    let series_posterior = transport_study(&data, temporal, true);
    let rows_posterior = transport_study(&data, admg, true);
    let sd = |result: &antecedent::StudyResult| {
        let posterior = result.posterior.as_ref().expect("a transport posterior");
        let column = posterior.effect_column().expect("the transported effect");
        posterior.summaries.sd[column]
    };
    assert!(
        sd(&series_posterior) > 1.3 * sd(&rows_posterior),
        "block posterior {} vs row posterior {}",
        sd(&series_posterior),
        sd(&rows_posterior)
    );
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
fn temporal_admg_temporal_contrasts_are_typed_boundaries() {
    // A pulse, sustained or temporal conditional contrast on a TemporalAdmg executes on its
    // unrolled window but is never licensed (the `window_contrast` tests run it), while the
    // contemporaneous projection is licensed. Temporal mediation stays a refusal.
    let mediation = antecedent::support::classify(
        SupportCell::from_labels(
            "TemporalMediationEffect",
            "TemporalAdmg",
            "explicit",
            "Frequentist",
            "none",
        )
        .expect("support axis"),
    );
    assert!(matches!(mediation, CellStatus::Refused), "{mediation:?}");
    for query in ["PulseEffect", "SustainedEffect", "TemporalConditionalPulse"] {
        for structure in ["explicit", "accepted"] {
            for inference in ["Frequentist", "Bayesian"] {
                let status = antecedent::support::classify(
                    SupportCell::from_labels(query, "TemporalAdmg", structure, inference, "none")
                        .expect("support axis"),
                );
                assert!(
                    matches!(status, CellStatus::Refused),
                    "{query}:{structure}:{inference} must stay a typed refusal: {status:?}"
                );
            }
        }
    }
    let projection = antecedent::support::classify(
        SupportCell::from_labels("AverageEffect", "TemporalAdmg", "explicit", "Bayesian", "none")
            .expect("support axis"),
    );
    assert!(matches!(projection, CellStatus::Licensed), "{projection:?}");
    // The same contrast on a graph with no latent confounding over time is licensed.
    let on_temporal_dag = antecedent::support::classify(
        SupportCell::from_labels("PulseEffect", "TemporalDag", "explicit", "Frequentist", "none")
            .expect("support axis"),
    );
    assert!(matches!(on_temporal_dag, CellStatus::Licensed), "{on_temporal_dag:?}");
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

/// The front-door fixture's rows in a fixed scrambled time order (a prime stride over
/// the expanded contingency table), so consecutive rows are not copies of one cell.
fn frontdoor_series() -> (TabularData, Admg, f64) {
    let pin = load_pin("conformance/estimate/admg_frontdoor_functional");
    let (_, graph) = expand(&pin);
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|value| value.as_str().unwrap()).collect();
    let mut values = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let n = values[0].len();
    // Within `n + 1` consecutive candidates one is 1 modulo `n`, hence coprime to it.
    let stride = (7919..=7919 + n).find(|stride| gcd(*stride, n) == 1).unwrap();
    let order: Vec<usize> = (0..n).map(|row| (row * stride) % n).collect();
    let shuffled: Vec<Vec<f64>> =
        values.iter().map(|column| order.iter().map(|&row| column[row]).collect()).collect();
    let data = TabularData::from_f64_columns(
        columns.iter().zip(&shuffled).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap();
    let expected = pin["frequentist"]["expected_ate"].as_f64().unwrap();
    (data, graph, expected)
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn projected_ate(
    data: &TabularData,
    temporal: &TemporalAdmg,
    (bayesian, replicates): (bool, u32),
    design: Option<antecedent_core::DeclaredSamplingDesign>,
    ctx: &ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(2), 0.0, 1.0);
    let builder = Study::tabular(data.clone())
        .query(query)
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalEffect)
        .inference(inference_of(bayesian, 400))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .graph(temporal.clone());
    let builder = match design {
        Some(design) => builder.sampling_design(design),
        None => builder,
    };
    builder.build()?.run(ctx)
}

#[test]
fn temporal_admg_projection_resamples_blocks_of_the_series() {
    let (data, graph, expected) = frontdoor_series();
    let temporal = lift(&graph);
    let ctx = ExecutionContext::for_tests(17);
    // Frequentist: the whole plug-in is refit on circular blocks of consecutive rows.
    let frequentist = projected_ate(&data, &temporal, (false, 199), None, &ctx).unwrap();
    assert!((frequentist.effect() - expected).abs() < 1e-9);
    let binding = frequentist.interval.as_ref().expect("an interval binding");
    assert_eq!(binding.method, antecedent_core::IntervalMethod::CircularBlockSe);
    assert_eq!(binding.dependence, "circular_block:row_block");
    let se = frequentist.estimate.as_effect().unwrap().se_bootstrap().expect("a block SE");
    assert!(se.is_finite() && se > 0.0 && se < 0.2, "se {se}");
    // The same replicates for any thread budget.
    let mut parallel = ExecutionContext::for_tests(17);
    parallel.parallelism =
        antecedent_core::Parallelism::bounded(antecedent_core::NonZeroThreadCount::new(4).unwrap());
    let threaded = projected_ate(&data, &temporal, (false, 199), None, &parallel).unwrap();
    assert_eq!(
        threaded.estimate.as_effect().unwrap().se_bootstrap().map(f64::to_bits),
        Some(se.to_bits())
    );
    // Bayesian: one Dirichlet weight per block of consecutive rows.
    let bayesian = projected_ate(&data, &temporal, (true, 0), None, &ctx).unwrap();
    let posterior = bayesian.posterior.as_ref().expect("a block Bayesian-bootstrap posterior");
    let column = posterior.effect_column().expect("an effect column");
    assert!((posterior.summaries.mean[column] - expected).abs() < 0.08);
    assert!(posterior.summaries.sd[column] > 0.0);
    // A declared independent-subject or clustered-panel design refuses: a row-level draw
    // is not allowed under either, and the table carries no subject or cluster labels to
    // draw whole units from. The refusal is typed and names the next attempt; without
    // replicates the frequentist point still stands.
    for design in [
        antecedent_core::DeclaredSamplingDesign::IndependentSubjectTrajectories,
        antecedent_core::DeclaredSamplingDesign::ClusteredPanel,
    ] {
        let refused = projected_ate(&data, &temporal, (false, 199), Some(design), &ctx)
            .expect_err("a row draw under a subject or cluster design is refused");
        let message = refused.to_string();
        assert!(message.contains("option_not_applicable"), "{design:?}: {message}");
        assert!(message.contains("no subject or cluster labels"), "{design:?}: {message}");
        assert!(message.contains("Next attempt: declare a dependent series"), "{message}");
        let bayesian_refused =
            projected_ate(&data, &temporal, (true, 0), Some(design), &ctx).unwrap_err();
        assert!(bayesian_refused.to_string().contains("Next attempt"), "{bayesian_refused}");
        let point = projected_ate(&data, &temporal, (false, 0), Some(design), &ctx).unwrap();
        assert!((point.effect() - expected).abs() < 1e-9);
    }
}

/// Pulse, sustained and conditional contrasts on one series under a `TemporalAdmg`: the template
/// is unrolled over the periods the policy needs, the contrast is identified on that window, and
/// the identified functional is evaluated on the lag-aligned window table. The cell executes
/// unlicensed; the expected values below are counted from the series, not from the engine.
mod window_contrast {
    use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult};
    use antecedent_core::{
        DeclaredSamplingDesign, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy,
        VariableId,
    };
    use antecedent_data::TimeSeriesData;
    use antecedent_graph::TemporalAdmg;
    use antecedent_io::consume_analysis_result;

    const DISCLOSURE: &str = "estimate.temporal_admg.window";

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    /// `x(t-1) -> y(t)` and `w(t-1) -> y(t)`; optionally `x(t-1) <-> y(t)`.
    fn template(confounded: bool) -> TemporalAdmg {
        let mut graph = TemporalAdmg::empty();
        let x1 = graph.add_lagged(v(0), Lag::from_raw(1)).unwrap();
        let w1 = graph.add_lagged(v(2), Lag::from_raw(1)).unwrap();
        let y0 = graph.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(x1, y0).unwrap();
        graph.insert_directed(w1, y0).unwrap();
        if confounded {
            graph.insert_bidirected(x1, y0).unwrap();
        }
        graph
    }

    /// A deterministic bit stream.
    fn bits(len: usize, seed: u32) -> Vec<f64> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                f64::from((state >> 16) & 1)
            })
            .collect()
    }

    /// `y(t) = x(t-1)` where `w(t-1) = 1` and `0` elsewhere, flipped at four times.
    fn series(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, TimeSeriesData) {
        let x = bits(len, 0x9E37_79B9);
        let w = bits(len, 0x85EB_CA6B);
        let y: Vec<f64> = (0..len)
            .map(|t| {
                let drive = if t > 0 && (w[t - 1] - 1.0).abs() < 1e-12 { x[t - 1] } else { 0.0 };
                if [9, 33, 71, 120].contains(&t) { 1.0 - drive } else { drive }
            })
            .collect();
        let data = TimeSeriesData::from_f64_columns(
            [("x", x.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())],
            1,
        )
        .unwrap();
        (x, y, w, data)
    }

    /// The treatment at time `t` on the outcome at time `t + 1`.
    fn next_period() -> TemporalEffectQuery {
        TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_horizon_steps(2)
    }

    fn study(
        data: &TimeSeriesData,
        graph: TemporalAdmg,
        query: TemporalEffectQuery,
        inference: InferenceMode,
        replicates: u32,
    ) -> Study {
        Study::series(data.clone())
            .graph(graph)
            .temporal_query(query)
            .inference(inference)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(replicates)
            .build()
            .unwrap()
    }

    /// The pulse contrast the window identifies: `w(t)` is a parent of `y(t+1)` with no parent
    /// of its own in the window, so the identified functional is
    /// `sum_w P(w) (E[y(t+1) | x(t)=1, w(t)=w] - E[y(t+1) | x(t)=0, w(t)=w])` with `P(w)` the
    /// law of `w(t)` over the window positions.
    fn counted_contrast(x: &[f64], y: &[f64], w: &[f64]) -> f64 {
        counted_conditional(x, y, w)
    }

    /// `E[y(t+1) | x(t) = 1] - E[y(t+1) | x(t) = 0]` over the window positions, which does not
    /// adjust for `w(t)` and so is not the identified contrast (it differs in a finite sample).
    fn naive_contrast(x: &[f64], y: &[f64]) -> f64 {
        let mean = |level: f64| {
            let rows: Vec<f64> = (1..x.len() - 1)
                .filter(|t| (x[*t] - level).abs() < 1e-12)
                .map(|t| y[t + 1])
                .collect();
            rows.iter().sum::<f64>() / rows.len() as f64
        };
        mean(1.0) - mean(0.0)
    }

    /// The same contrast within each level of the modifier read at the treated time, averaged
    /// over the modifier's law across the window positions.
    fn counted_conditional(x: &[f64], y: &[f64], w: &[f64]) -> f64 {
        let positions: Vec<usize> = (1..x.len() - 1).collect();
        let mean = |level: f64, stratum: f64| {
            let rows: Vec<f64> = positions
                .iter()
                .filter(|t| (x[**t] - level).abs() < 1e-12 && (w[**t] - stratum).abs() < 1e-12)
                .map(|t| y[*t + 1])
                .collect();
            rows.iter().sum::<f64>() / rows.len() as f64
        };
        [0.0, 1.0]
            .into_iter()
            .map(|stratum| {
                let weight = positions.iter().filter(|t| (w[**t] - stratum).abs() < 1e-12).count()
                    as f64
                    / positions.len() as f64;
                weight * (mean(1.0, stratum) - mean(0.0, stratum))
            })
            .sum()
    }

    fn discloses(result: &StudyResult) -> bool {
        result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == DISCLOSURE)
    }

    #[test]
    fn a_pulse_is_the_counted_contrast_and_stays_unlicensed() {
        let (x, y, w, data) = series(300);
        let counted = counted_contrast(&x, &y, &w);
        assert!(counted > 0.2 && counted < 0.8, "{counted}");
        let built = study(&data, template(false), next_period(), InferenceMode::Frequentist, 0);
        assert_eq!(built.support_status(), Some(CellStatus::Refused));
        let ctx = ExecutionContext::for_tests(31_001);
        let plan = built.compile(&ctx).unwrap();
        assert_eq!(plan.lowered().unwrap().route(), "TemporalEffectAdmg");
        let result = built.run(&ctx).unwrap();
        assert!((result.effect() - counted).abs() < 1e-12, "{} vs {counted}", result.effect());
        assert_eq!(result.support_status, Some(CellStatus::Refused));
        assert!(discloses(&result));
        assert_eq!(result.lowered_program_identity(), plan.lowered_program_identity());
        // The prepared handle freezes the identification and reproduces the point; the artifact
        // names the unfolded window coordinates and survives a round trip.
        let mut prepared = built.prepare(&ctx).unwrap();
        let again = prepared.estimate_series(&data, &ctx).unwrap();
        assert!((again.effect() - counted).abs() < 1e-12);
        assert!(again.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
        let artifact =
            prepared.encode_contracted_result(&again, "temporal_admg_pulse", &ctx).unwrap();
        let consumed = consume_analysis_result(&artifact).unwrap();
        assert_eq!(consumed.body.estimate, Some(again.effect()));
        assert!(matches!(
            prepared.retained_operation(),
            antecedent::analysis::RetainedOperation::TemporalAdmgEffect
        ));
        let (changed_x, changed_y, changed_w, changed) = series(320);
        let expected =
            study(&changed, template(false), next_period(), InferenceMode::Frequentist, 0)
                .run(&ctx)
                .unwrap();
        let refreshed = prepared.refresh_series(changed, &ctx).unwrap();
        assert_eq!(refreshed.effect().to_bits(), expected.effect().to_bits());
        assert!(
            (refreshed.effect() - counted_contrast(&changed_x, &changed_y, &changed_w)).abs()
                < 1e-12
        );
        assert_eq!(
            refreshed.lowered_program_identity(),
            prepared.plan().lowered_program_identity()
        );
        let artifact = prepared
            .encode_contracted_result(&refreshed, "retained-temporal-admg-refresh", &ctx)
            .unwrap();
        let consumed = consume_analysis_result(&artifact).unwrap();
        assert_eq!(consumed.body.estimate, Some(refreshed.effect()));
    }

    #[test]
    fn overlapping_windows_are_resampled_in_blocks_and_never_row_by_row() {
        let (x, y, w, data) = series(300);
        let counted = counted_contrast(&x, &y, &w);
        let ctx = ExecutionContext::for_tests(31_002);
        let result = study(&data, template(false), next_period(), InferenceMode::Frequentist, 60)
            .run(&ctx)
            .unwrap();
        assert!((result.effect() - counted).abs() < 1e-12);
        let effect = result.estimate.as_effect().unwrap();
        assert_eq!(
            effect.uncertainty.block.family,
            Some(antecedent_estimate::CircularBlockFamily::RowBlock)
        );
        let se = effect.se_bootstrap().expect("a block bootstrap standard error");
        assert!(se.is_finite() && se > 0.0, "{se}");
        // The window spans three times, so a block is at least three windows long.
        let disclosure = result
            .diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code.as_ref() == DISCLOSURE)
            .expect("the window disclosure");
        let block: usize = disclosure
            .fields
            .iter()
            .find(|(key, _)| key.as_ref() == "block_length")
            .map(|(_, value)| value.parse().unwrap())
            .expect("the block length is disclosed");
        assert!(block >= 3, "{block}");
        // A declared independent-subject or clustered design refuses a row-level draw.
        for design in [
            DeclaredSamplingDesign::IndependentSubjectTrajectories,
            DeclaredSamplingDesign::ClusteredPanel,
        ] {
            let refused = Study::series(data.clone())
                .graph(template(false))
                .temporal_query(next_period())
                .refute(RefuteSuite::None)
                .bootstrap_replicates(20)
                .sampling_design(design)
                .build()
                .unwrap()
                .run(&ctx)
                .unwrap_err()
                .to_string();
            assert!(refused.contains("Next attempt"), "{design:?}: {refused}");
        }
    }

    #[test]
    fn the_bayesian_form_shares_one_dirichlet_weight_per_block() {
        let (x, y, w, data) = series(300);
        let counted = counted_contrast(&x, &y, &w);
        let ctx = ExecutionContext::for_tests(31_003);
        let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400));
        let result = study(&data, template(false), next_period(), inference, 0).run(&ctx).unwrap();
        let posterior = result.posterior.as_ref().expect("a block Bayesian-bootstrap posterior");
        let column = posterior.effect_column().expect("an effect column");
        assert!((posterior.summaries.mean[column] - counted).abs() < 0.1);
        assert!(posterior.summaries.sd[column] > 0.0);
        assert!(discloses(&result));
        assert_eq!(result.support_status, Some(CellStatus::Refused));
    }

    #[test]
    fn a_conditional_contrast_is_idc_on_the_same_window() {
        let (x, y, w, data) = series(400);
        let counted = counted_conditional(&x, &y, &w);
        let unconditional = naive_contrast(&x, &y);
        assert!((counted - unconditional).abs() > 1e-3, "the modifier must matter");
        let query = next_period().with_effect_modifiers([v(2)]);
        let ctx = ExecutionContext::for_tests(31_004);
        let built = study(&data, template(false), query, InferenceMode::Frequentist, 0);
        assert_eq!(built.support_status(), Some(CellStatus::Refused));
        let result = built.run(&ctx).unwrap();
        assert!((result.effect() - counted).abs() < 1e-12, "{} vs {counted}", result.effect());
        assert!(discloses(&result));
        // The conditional contrast is a sealed artifact an independent consumer reads back.
        let prepared = built.prepare(&ctx).unwrap();
        let again = prepared.estimate_series(&data, &ctx).unwrap();
        assert!((again.effect() - counted).abs() < 1e-12);
        let artifact = prepared
            .encode_contracted_result(&again, "temporal_admg_conditional_pulse", &ctx)
            .unwrap();
        let consumed = consume_analysis_result(&artifact).unwrap();
        assert_eq!(consumed.body.estimate, Some(again.effect()));
    }

    #[test]
    fn a_conditional_sustained_contrast_is_idc_on_the_same_window() {
        let (x, y, w, data) = series(400);
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0)
            .with_policy(TemporalPolicy::sustained(0, 1))
            .with_horizon_steps(2)
            .with_effect_modifiers([v(2)]);
        let ctx = ExecutionContext::for_tests(31_008);
        let built = study(&data, template(false), query, InferenceMode::Frequentist, 0);
        let result = built.run(&ctx).unwrap();
        assert!(result.effect().is_finite());
        assert!((result.effect() - counted_conditional(&x, &y, &w)).abs() < 0.5);
        assert!(discloses(&result));
        let prepared = built.prepare(&ctx).unwrap();
        let again = prepared.estimate_series(&data, &ctx).unwrap();
        let artifact = prepared
            .encode_contracted_result(&again, "temporal_admg_conditional_sustained", &ctx)
            .unwrap();
        let consumed = consume_analysis_result(&artifact).unwrap();
        assert_eq!(consumed.body.estimate, Some(again.effect()));
    }

    #[test]
    fn a_sustained_contrast_sets_every_treated_period() {
        let (x, y, w, data) = series(300);
        // x(t-1) -> y(t) only: treating x at t and t+1 and reading y at t+1 equals the pulse at t,
        // and the treatment at t+1 does not reach y(t+1).
        let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0)
            .with_policy(TemporalPolicy::sustained(0, 1))
            .with_horizon_steps(2);
        let ctx = ExecutionContext::for_tests(31_005);
        let result = study(&data, template(false), query.clone(), InferenceMode::Frequentist, 0)
            .run(&ctx)
            .unwrap();
        assert!(result.effect().is_finite());
        assert!((result.effect() - counted_contrast(&x, &y, &w)).abs() < 0.5);
        assert!(discloses(&result));
        let built = study(&data, template(false), query, InferenceMode::Frequentist, 0);
        let prepared = built.prepare(&ctx).unwrap();
        let again = prepared.estimate_series(&data, &ctx).unwrap();
        let artifact =
            prepared.encode_contracted_result(&again, "temporal_admg_sustained", &ctx).unwrap();
        let consumed = consume_analysis_result(&artifact).unwrap();
        assert_eq!(consumed.body.estimate, Some(again.effect()));
    }

    #[test]
    fn a_window_that_holds_a_hedge_is_refused_with_the_variables_and_the_next_attempts() {
        let (_, _, _, data) = series(200);
        let built = study(&data, template(true), next_period(), InferenceMode::Frequentist, 0);
        let ctx = ExecutionContext::for_tests(31_006);
        for refusal in
            [built.run(&ctx).unwrap_err().to_string(), built.prepare(&ctx).unwrap_err().to_string()]
        {
            assert!(refusal.contains("effect_not_identified"), "{refusal}");
            assert!(refusal.contains("hedge") && refusal.contains('@'), "{refusal}");
            assert!(refusal.contains("next attempts"), "{refusal}");
        }
    }

    #[test]
    fn a_modifier_the_treatment_reaches_is_refused_with_its_next_attempt() {
        let (_, _, _, data) = series(200);
        let mut graph = template(false);
        // x(t) -> w(t): the modifier read at the treated time is a consequence of the treatment.
        let x0 = graph.add_lagged(v(0), Lag::CONTEMPORANEOUS).unwrap();
        let w0 = graph.add_lagged(v(2), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(x0, w0).unwrap();
        let query = next_period().with_effect_modifiers([v(2)]);
        let ctx = ExecutionContext::for_tests(31_007);
        let refusal = study(&data, graph, query, InferenceMode::Frequentist, 0)
            .run(&ctx)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("descendant") && refusal.contains("next attempt"), "{refusal}");
    }

    fn study_with_suite(
        data: &TimeSeriesData,
        query: TemporalEffectQuery,
        inference: InferenceMode,
        suite: RefuteSuite,
    ) -> Study {
        Study::series(data.clone())
            .graph(template(false))
            .temporal_query(query)
            .inference(inference)
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    }

    fn refuters(result: &StudyResult) -> Vec<&str> {
        result.refutations.iter().map(|report| report.refuter.as_ref()).collect()
    }

    #[test]
    fn cheap_and_full_validation_run_the_window_native_suite_on_both_inferences() {
        let (x, y, w, data) = series(300);
        let counted = counted_contrast(&x, &y, &w);
        for bayesian in [false, true] {
            for (suite, expected) in [
                (RefuteSuite::None, Vec::new()),
                (RefuteSuite::Cheap, vec!["window.stretch_80", "window.placebo_treatment"]),
                (
                    RefuteSuite::Full,
                    vec![
                        "window.stretch_80",
                        "window.stretch_50",
                        "window.placebo_treatment",
                        "window.split_half",
                    ],
                ),
            ] {
                let inference = if bayesian {
                    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200))
                } else {
                    InferenceMode::Frequentist
                };
                let built = study_with_suite(&data, next_period(), inference, suite);
                assert_eq!(built.support_status(), Some(CellStatus::Refused));
                let ctx = ExecutionContext::for_tests(31_010);
                let result = built.run(&ctx).unwrap();
                assert_eq!(refuters(&result), expected, "bayesian={bayesian} suite={suite:?}");
                let tolerance = if bayesian { 0.1 } else { 1e-12 };
                assert!((result.effect() - counted).abs() < tolerance, "{}", result.effect());
                assert!(discloses(&result));
                for report in &result.refutations {
                    assert!(
                        (report.original_ate - result.effect()).abs() < 1e-12
                            || report.refuter.as_ref() == "window.split_half"
                    );
                    assert!((0.0..=1.0).contains(&report.comparison), "{report:?}");
                }
                // Rotating the treatment removes every link to the outcome: the contrast of the
                // rotated series is far from the counted one and indistinguishable from zero.
                if let Some(placebo) = result
                    .refutations
                    .iter()
                    .find(|report| report.refuter.as_ref() == "window.placebo_treatment")
                {
                    assert_eq!(placebo.replicates, 20);
                    assert!(placebo.refuted_ate.abs() < 0.15, "{}", placebo.refuted_ate);
                    assert!(counted - placebo.refuted_ate.abs() > 0.3);
                    assert!(placebo.passed, "{placebo:?}");
                }
            }
        }
    }

    #[test]
    fn a_series_whose_effect_vanishes_halfway_fails_the_split_half_check() {
        let len = 400;
        let x = bits(len, 0x9E37_79B9);
        let w = bits(len, 0x85EB_CA6B);
        let y: Vec<f64> = (0..len)
            .map(|t| {
                if t > 0 && t < len / 2 && (w[t - 1] - 1.0).abs() < 1e-12 { x[t - 1] } else { 0.0 }
            })
            .collect();
        let data = TimeSeriesData::from_f64_columns(
            [("x", x.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())],
            1,
        )
        .unwrap();
        let first_half = counted_contrast(&x[..len / 2], &y[..len / 2], &w[..len / 2]);
        assert!(first_half > 0.3, "{first_half}");
        let ctx = ExecutionContext::for_tests(31_011);
        let result =
            study_with_suite(&data, next_period(), InferenceMode::Frequentist, RefuteSuite::Full)
                .run(&ctx)
                .unwrap();
        let split = result
            .refutations
            .iter()
            .find(|report| report.refuter.as_ref() == "window.split_half")
            .expect("the split-half report");
        assert!(split.informative);
        assert!(!split.passed, "{split:?}");
        assert!(split.comparison < 0.01, "{}", split.comparison);
        // The first half recovers the counted contrast of its own stretch; the second is nil.
        assert!((split.original_ate - first_half).abs() < 0.1, "{}", split.original_ate);
        assert!(split.refuted_ate.abs() < 1e-9, "{}", split.refuted_ate);
        // The pooled contrast sits between the halves, so the pooled stretch refits cannot see
        // the break: only the split-half check does.
        let pooled = result.effect();
        assert!(pooled > 0.1 && pooled < first_half, "{pooled} vs {first_half}");
    }

    #[test]
    fn a_graph_the_series_does_not_describe_and_a_history_cap_are_refused_at_compile() {
        let (_, _, _, data) = series(100);
        let ctx = ExecutionContext::for_tests(31_008);
        let capped = study(
            &data,
            template(false),
            next_period().with_max_history_lag(Some(1)),
            InferenceMode::Frequentist,
            0,
        );
        let refusal = capped.compile(&ctx).unwrap_err().to_string();
        assert!(refusal.contains("option_not_applicable"), "{refusal}");
        let mut outside = TemporalAdmg::empty();
        let a = outside.add_lagged(v(0), Lag::from_raw(1)).unwrap();
        let b = outside.add_lagged(v(7), Lag::CONTEMPORANEOUS).unwrap();
        outside.insert_directed(a, b).unwrap();
        let query = TemporalEffectQuery::pulse(v(0), v(7), 1.0).with_horizon_steps(2);
        let refusal = study(&data, outside, query, InferenceMode::Frequentist, 0)
            .compile(&ctx)
            .unwrap_err()
            .to_string();
        assert!(refusal.contains("unknown_variable"), "{refusal}");
    }
}
