//! Builder-independent evidence for Bayesian average effects on static CPDAG and PAG envelopes.
//!
//! The CPDAG fixture (`conformance/estimate/cpdag_ate_envelope`) has two completions
//! with different effects and no unidentified mass; the PAG fixture
//! (`conformance/estimate/pag_ate_envelope_identified`) has seven completions with one
//! unidentified. Both carry a seeded conjugate `bayesian.gcomp` envelope pin that the
//! sibling numeric-pin tests validate against closed forms; here the same pins must be
//! reproduced from the retained checked plan after the builder is gone.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, GraphClass, IdentifierId, InferenceMode,
    RefuteSuite, Study,
};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};
use antecedent_validate::PredictiveCheckKind;

fn cpdag_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/cpdag_ate_envelope/expected.json"
    ))
    .unwrap()
}

fn pag_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/pag_ate_envelope_identified/expected.json"
    ))
    .unwrap()
}

fn columns(pin: &serde_json::Value) -> Vec<&str> {
    pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect()
}

fn expand_contingency(pin: &serde_json::Value) -> TabularData {
    let columns = columns(pin);
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (i, name) in columns.iter().enumerate() {
            values[i].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(name, col)| (*name, col.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

/// The same rows under a different column order: every variable ID changes, so a
/// retained plan must refuse to bind it.
fn reordered(pin: &serde_json::Value) -> TabularData {
    let mut columns = columns(pin);
    columns.rotate_left(1);
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (i, name) in columns.iter().enumerate() {
            values[i].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(name, col)| (*name, col.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn index(pin: &serde_json::Value, name: &str) -> u32 {
    u32::try_from(columns(pin).iter().position(|c| *c == name).unwrap()).unwrap()
}

fn endpoint(mark: &str) -> Endpoint {
    match mark {
        "tail" => Endpoint::Tail,
        "arrow" => Endpoint::Arrow,
        "circle" => Endpoint::Circle,
        other => panic!("unknown endpoint {other}"),
    }
}

fn cpdag_from_pin(pin: &serde_json::Value) -> Cpdag {
    let mut cpdag = Cpdag::with_variables(u32::try_from(columns(pin).len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        cpdag
            .insert_directed(
                DenseNodeId::from_raw(index(pin, edge[0].as_str().unwrap())),
                DenseNodeId::from_raw(index(pin, edge[1].as_str().unwrap())),
            )
            .unwrap();
    }
    for edge in pin["graph"]["undirected_edges"].as_array().unwrap() {
        cpdag
            .insert_undirected(
                DenseNodeId::from_raw(index(pin, edge[0].as_str().unwrap())),
                DenseNodeId::from_raw(index(pin, edge[1].as_str().unwrap())),
            )
            .unwrap();
    }
    cpdag
}

fn pag_from_pin(pin: &serde_json::Value) -> Pag {
    let mut pag = Pag::with_variables(u32::try_from(columns(pin).len()).unwrap());
    for edge in pin["graph"]["marked_edges"].as_array().unwrap() {
        pag.insert_marked(MarkedEdge {
            a: DenseNodeId::from_raw(index(pin, edge[0].as_str().unwrap())),
            b: DenseNodeId::from_raw(index(pin, edge[1].as_str().unwrap())),
            at_a: endpoint(edge[2].as_str().unwrap()),
            at_b: endpoint(edge[3].as_str().unwrap()),
            middle: MiddleMark::Empty,
        })
        .unwrap();
    }
    pag
}

fn query_from_pin(pin: &serde_json::Value) -> AverageEffectQuery {
    AverageEffectQuery::with_levels(
        VariableId::from_raw(index(pin, pin["query"]["treatment"].as_str().unwrap())),
        VariableId::from_raw(index(pin, pin["query"]["outcome"].as_str().unwrap())),
        pin["query"]["control_level"].as_f64().unwrap(),
        pin["query"]["active_level"].as_f64().unwrap(),
    )
}

fn bayesian_config(pin: &serde_json::Value) -> BayesianConfig {
    let block = &pin["bayesian"];
    assert_eq!(block["backend"], "conjugate");
    assert_eq!(block["estimator"], "bayesian.gcomp");
    BayesianConfig::conjugate()
        .n_draws(usize::try_from(block["n_draws"].as_u64().unwrap()).unwrap())
        .prior_scale(block["prior_scale"].as_f64().unwrap())
}

enum ClassGraph {
    Cpdag(Cpdag),
    Pag(Pag),
}

struct Fixture {
    graph_class: GraphClass,
    graph: ClassGraph,
    pin: serde_json::Value,
    /// Seeded posterior mean of the envelope mixture; see the numeric-pin tests for
    /// the closed-form checks that license these values.
    expected_ate: f64,
}

fn fixtures() -> [Fixture; 2] {
    let cpdag = cpdag_pin();
    let pag = pag_pin();
    [
        Fixture {
            graph_class: GraphClass::Cpdag,
            graph: ClassGraph::Cpdag(cpdag_from_pin(&cpdag)),
            pin: cpdag,
            expected_ate: 0.461_605_123_227_517_9,
        },
        Fixture {
            graph_class: GraphClass::Pag,
            graph: ClassGraph::Pag(pag_from_pin(&pag)),
            pin: pag,
            expected_ate: 0.376_238_176_353_573,
        },
    ]
}

fn assert_validation(result: &antecedent::StudyResult, suite: RefuteSuite, label: &str) {
    if suite == RefuteSuite::None {
        assert!(result.refutations.is_empty(), "{label}: validation none must emit no reports");
        assert!(result.predictive_checks.is_empty(), "{label}: validation none must not run PPC");
        return;
    }
    assert!(!result.refutations.is_empty(), "{label}: {suite:?} must execute a refuter");
    assert!(
        result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Prior),
        "{label}: {suite:?} must attach prior PPC"
    );
    assert!(
        result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Posterior),
        "{label}: {suite:?} must attach posterior PPC"
    );
    let posterior = result.posterior.as_ref().expect("Bayesian envelope posterior");
    assert_eq!(
        posterior.prior_sensitivity.is_some(),
        suite == RefuteSuite::Full,
        "{label}: prior sensitivity is attached exactly under full validation"
    );
}

#[test]
fn bayesian_static_class_effect_executes_from_retained_plan_after_builder_is_dropped() {
    run_on_large_stack(|| {
        for fixture in fixtures() {
            let pin = &fixture.pin;
            let data = expand_contingency(pin);
            let query = query_from_pin(pin);
            let config = bayesian_config(pin);
            let seed = pin["bayesian"]["seed"].as_u64().unwrap();
            let ctx = ExecutionContext::for_tests(seed);
            let completion_count =
                usize::try_from(pin["identification"]["completion_count"].as_u64().unwrap())
                    .unwrap();
            let identified_mass = pin["identification"]["identified_mass"].as_f64().unwrap();
            let unidentified_mass = pin["identification"]["unidentified_mass"].as_f64().unwrap();
            for accepted in [false, true] {
                for (suite_label, suite) in [
                    ("none", RefuteSuite::None),
                    ("cheap", RefuteSuite::Cheap),
                    ("full", RefuteSuite::Full),
                ] {
                    let coordinate = format!(
                        "AverageEffect:{:?}:{}:Bayesian:{suite_label}",
                        fixture.graph_class,
                        if accepted { "accepted" } else { "explicit" }
                    );
                    let base = Study::tabular(data.clone());
                    let base = match (&fixture.graph, accepted) {
                        (ClassGraph::Cpdag(graph), true) => {
                            base.graph(AcceptedGraph::from(graph.clone()))
                        }
                        (ClassGraph::Cpdag(graph), false) => base.graph(graph.clone()),
                        (ClassGraph::Pag(graph), true) => {
                            base.graph(AcceptedGraph::from(graph.clone()))
                        }
                        (ClassGraph::Pag(graph), false) => base.graph(graph.clone()),
                    };
                    let builder = base
                        .query(query.clone())
                        .inference(InferenceMode::Bayesian(config.clone()))
                        .refute(suite)
                        .bootstrap_replicates(0)
                        .build()
                        .unwrap();
                    let one_shot = builder.clone().run(&ctx).unwrap();
                    let mut prepared = builder.prepare(&ctx).unwrap();
                    drop(builder);

                    assert!(prepared.has_checked_static_class_effect_operation(), "{coordinate}");
                    let plan = prepared
                        .checked_static_class_effect_info()
                        .expect("retained Bayesian static class effect operation");
                    assert_eq!(plan.graph_class, fixture.graph_class, "{coordinate}");
                    assert_eq!(plan.query, query, "{coordinate}");
                    assert_eq!(
                        plan.identifier,
                        IdentifierId::GeneralizedAdjustment,
                        "{coordinate}"
                    );
                    assert_eq!(plan.estimator, EstimatorId::BayesianGcomp, "{coordinate}");
                    assert_eq!(
                        plan.inference,
                        InferenceMode::Bayesian(config.clone()),
                        "{coordinate}"
                    );
                    assert_eq!(plan.validation, suite, "{coordinate}");
                    assert_eq!(plan.completion_count, completion_count, "{coordinate}");
                    assert!((plan.identified_mass - identified_mass).abs() < 1e-15, "{coordinate}");
                    assert!(
                        (plan.unresolved_mass - unidentified_mass).abs() < 1e-15,
                        "{coordinate}"
                    );
                    assert_eq!(
                        prepared.plan().logical.record.identifier.as_deref(),
                        Some("generalized.adjustment"),
                        "{coordinate}"
                    );
                    assert_eq!(
                        prepared.plan().logical.record.estimator.as_deref(),
                        Some("bayesian.gcomp"),
                        "{coordinate}"
                    );

                    let result = prepared.estimate(&data, &ctx).unwrap();
                    assert!(
                        (result.estimate.ate - fixture.expected_ate).abs() < 1e-9,
                        "{coordinate}: envelope posterior mean {} vs pin {}",
                        result.estimate.ate,
                        fixture.expected_ate
                    );
                    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.gcomp"));
                    assert!(
                        result
                            .diagnostics
                            .iter()
                            .any(|d| d.code.as_ref() == "exec.identify.cached"),
                        "{coordinate}: click must execute the retained envelope"
                    );
                    let posterior = result.posterior.as_ref().expect("Bayesian envelope posterior");
                    let unidentified_share =
                        unidentified_mass / (identified_mass + unidentified_mass);
                    assert!(
                        (posterior.unidentified_mass - unidentified_share).abs() < 1e-12,
                        "{coordinate}: posterior unidentified mass {} vs envelope share {unidentified_share}",
                        posterior.unidentified_mass
                    );
                    assert_validation(&result, suite, &coordinate);

                    // The one-shot facade executes the same retained plan.
                    assert_eq!(
                        one_shot.estimate.ate.to_bits(),
                        result.estimate.ate.to_bits(),
                        "{coordinate}: one-shot and prepared clicks must agree bitwise"
                    );
                    assert!(
                        one_shot
                            .diagnostics
                            .iter()
                            .any(|d| d.code.as_ref() == "exec.identify.cached"),
                        "{coordinate}: one-shot run must execute the retained proof"
                    );

                    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
                    assert_eq!(refreshed.estimate.ate.to_bits(), result.estimate.ate.to_bits());
                    let rebound = prepared.checked_static_class_effect_info().unwrap();
                    assert_eq!(rebound.query, plan.query);
                    assert_eq!(rebound.inference, plan.inference);
                    assert_eq!(rebound.completion_count, plan.completion_count);
                    assert!(
                        prepared.refresh(reordered(pin), &ctx).is_err(),
                        "{coordinate}: schema-changing refresh must be refused"
                    );

                    let artifact = prepared
                        .encode_contracted_result(
                            &refreshed,
                            &format!("checked-bayesian-static-class-{suite_label}"),
                            &ctx,
                        )
                        .unwrap();
                    let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                    assert!(consumed.contract.is_some());
                    assert!(
                        consumed.acceptance.unresolved.iter().any(|reason| {
                            reason.as_ref() == "dependencies.checked_static_class_effect_operation"
                        }),
                        "{coordinate}: {:?}",
                        consumed.acceptance.unresolved
                    );
                    assert!(!consumed.acceptance.accepts_as_verified_program());
                }
            }
        }
    });
}

/// Class envelopes execute through deep prepare/estimate frames in debug builds.
fn run_on_large_stack(run: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("checked-bayesian-class-evidence".into())
        .stack_size(8 * 1024 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
}
