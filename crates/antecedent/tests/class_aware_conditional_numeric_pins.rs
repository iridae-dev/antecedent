//! 1.4 numeric pins for licensed Cpdag/Pag ConditionalEffect cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ConditionalEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};
use antecedent_validate::PredictiveCheckKind;

fn ate_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/cpdag_ate_envelope/expected.json"
    ))
    .unwrap()
}

fn class_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/class_aware_envelope/expected.json"
    ))
    .unwrap()
}

fn expand_contingency(pin: &serde_json::Value) -> TabularData {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (i, name) in columns.iter().enumerate() {
            let value = cell[*name].as_f64().unwrap();
            values[i].extend(std::iter::repeat_n(value, count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(name, col)| (*name, col.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn node(columns: &[&str], name: &str) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(columns.iter().position(|c| *c == name).unwrap()).unwrap())
}

fn endpoint(mark: &str) -> Endpoint {
    match mark {
        "tail" => Endpoint::Tail,
        "arrow" => Endpoint::Arrow,
        "circle" => Endpoint::Circle,
        "conflict" => Endpoint::Conflict,
        other => panic!("unknown endpoint {other}"),
    }
}

fn cpdag_from_pin(pin: &serde_json::Value) -> Cpdag {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let spec = &pin["graph"];
    let mut cpdag = Cpdag::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in spec["directed_edges"].as_array().unwrap() {
        cpdag
            .insert_directed(
                node(&columns, edge[0].as_str().unwrap()),
                node(&columns, edge[1].as_str().unwrap()),
            )
            .unwrap();
    }
    for edge in spec["undirected_edges"].as_array().unwrap() {
        cpdag
            .insert_undirected(
                node(&columns, edge[0].as_str().unwrap()),
                node(&columns, edge[1].as_str().unwrap()),
            )
            .unwrap();
    }
    cpdag
}

fn pag_from_class_pin(pin: &serde_json::Value) -> Pag {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut pag = Pag::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["pag"]["graph"]["marked_edges"].as_array().unwrap() {
        pag.insert_marked(MarkedEdge {
            a: node(&columns, edge[0].as_str().unwrap()),
            b: node(&columns, edge[1].as_str().unwrap()),
            at_a: endpoint(edge[2].as_str().unwrap()),
            at_b: endpoint(edge[3].as_str().unwrap()),
            middle: MiddleMark::Empty,
        })
        .unwrap();
    }
    pag
}

fn conditional_query(pin: &serde_json::Value) -> ConditionalEffectQuery {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let treatment = node(&columns, pin["query"]["treatment"].as_str().unwrap());
    let outcome = node(&columns, pin["query"]["outcome"].as_str().unwrap());
    let modifier = node(&columns, pin["conditional"]["modifier"].as_str().unwrap());
    let inner = AverageEffectQuery::with_levels(
        VariableId::from_raw(treatment.raw()),
        VariableId::from_raw(outcome.raw()),
        pin["query"]["control_level"].as_f64().unwrap(),
        pin["query"]["active_level"].as_f64().unwrap(),
    )
    .with_effect_modifiers([VariableId::from_raw(modifier.raw())]);
    ConditionalEffectQuery::try_new(inner).unwrap()
}

#[derive(Default)]
struct RecordingProgress(std::sync::Mutex<Vec<String>>);

impl antecedent_core::ProgressSink for RecordingProgress {
    fn report(&self, _fraction: f64, stage: &str) {
        self.0.lock().unwrap().push(stage.to_owned());
    }
}

fn recording_ctx(seed: u64) -> (ExecutionContext, Arc<RecordingProgress>) {
    let sink = Arc::new(RecordingProgress::default());
    let mut ctx = ExecutionContext::for_tests(seed);
    ctx.progress = Some(Arc::clone(&sink) as Arc<dyn antecedent_core::ProgressSink>);
    (ctx, sink)
}

fn identify_computations(sink: &RecordingProgress) -> usize {
    sink.0.lock().unwrap().iter().filter(|stage| stage.as_str() == "identify.compute").count()
}

fn cached_count(result: &antecedent::StudyResult) -> usize {
    result.diagnostics.iter().filter(|d| d.code.as_ref() == "exec.identify.cached").count()
}

fn assert_validation_presence(
    result: &antecedent::StudyResult,
    suite: RefuteSuite,
    bayesian: bool,
) {
    match suite {
        RefuteSuite::None => {
            assert!(result.refutations.is_empty(), "validation none must emit no reports");
        }
        RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
            assert!(!result.refutations.is_empty(), "{suite:?} must execute a refuter");
            if !bayesian {
                assert!(
                    result
                        .diagnostics
                        .iter()
                        .any(|d| d.code.as_ref() == "refute.envelope.effect_mixture"),
                    "Frequentist {suite:?} must mix effect refuters across completions"
                );
            }
            if bayesian {
                assert!(
                    result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Prior),
                    "{suite:?} must attach prior PPC"
                );
            }
        }
    }
}

enum ClassGraph {
    Cpdag(Cpdag),
    Pag(Pag),
}

fn build_conditional(
    data: &TabularData,
    graph: &ClassGraph,
    accepted: bool,
    query: ConditionalEffectQuery,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    let builder = Study::tabular(data.clone());
    let builder = match (graph, accepted) {
        (ClassGraph::Cpdag(cpdag), true) => builder.graph(AcceptedGraph::from(cpdag.clone())),
        (ClassGraph::Cpdag(cpdag), false) => builder.graph(cpdag.clone()),
        (ClassGraph::Pag(pag), true) => builder.graph(AcceptedGraph::from(pag.clone())),
        (ClassGraph::Pag(pag), false) => builder.graph(pag.clone()),
    };
    builder.query(query).inference(inference).refute(suite).bootstrap_replicates(0).build().unwrap()
}

#[test]
fn class_aware_conditional_pins_z_conditional_effect() {
    let pin = ate_pin();
    let class = class_pin();
    let data = expand_contingency(&pin);
    let query = conditional_query(&pin);
    let freq = &pin["conditional"]["frequentist"];
    let bayes = &pin["conditional"]["bayesian"];
    let freq_ate = freq["expected_ate"].as_f64().unwrap();
    let freq_tol = freq["absolute_tolerance"].as_f64().unwrap();
    let bayes_ate = bayes["expected_ate"].as_f64().unwrap();
    let bayes_tol = bayes["absolute_tolerance"].as_f64().unwrap();
    let seed = bayes["seed"].as_u64().unwrap();
    let graphs = [
        ("cpdag", ClassGraph::Cpdag(cpdag_from_pin(&pin))),
        ("pag", ClassGraph::Pag(pag_from_class_pin(&class))),
    ];
    for (class_name, graph) in graphs {
        for accepted in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let freq_study = build_conditional(
                    &data,
                    &graph,
                    accepted,
                    query.clone(),
                    InferenceMode::Frequentist,
                    suite,
                );
                let (ctx, sink) = recording_ctx(seed);
                let fresh = freq_study.clone().run(&ctx).unwrap();
                assert_eq!(identify_computations(&sink), 1);
                let prepared: PreparedStudy = freq_study.prepare(&ctx).unwrap();
                let click = prepared.estimate(&data, &ctx).unwrap();
                assert_eq!(cached_count(&fresh), 0);
                assert_eq!(cached_count(&click), 1);
                assert_eq!(
                    fresh.logical_plan.estimator.as_deref(),
                    Some(freq["estimator"].as_str().unwrap())
                );
                for result in [&fresh, &click] {
                    assert_eq!(format!("{:?}", result.identification.status), "GraphDependent");
                    assert!((result.estimate.ate - freq_ate).abs() < freq_tol, "{class_name} freq");
                    assert_validation_presence(result, suite, false);
                }

                let bayes_study = build_conditional(
                    &data,
                    &graph,
                    accepted,
                    query.clone(),
                    InferenceMode::Bayesian(
                        BayesianConfig::conjugate()
                            .n_draws(usize::try_from(bayes["n_draws"].as_u64().unwrap()).unwrap())
                            .prior_scale(bayes["prior_scale"].as_f64().unwrap()),
                    ),
                    suite,
                );
                let (ctx, _) = recording_ctx(seed);
                let fresh = bayes_study.clone().run(&ctx).unwrap();
                let prepared: PreparedStudy = bayes_study.prepare(&ctx).unwrap();
                let click = prepared.estimate(&data, &ctx).unwrap();
                assert_eq!(
                    fresh.logical_plan.estimator.as_deref(),
                    Some(bayes["estimator"].as_str().unwrap())
                );
                for result in [&fresh, &click] {
                    assert!(
                        (result.estimate.ate - bayes_ate).abs() < bayes_tol,
                        "{class_name} bayes"
                    );
                    assert_validation_presence(result, suite, true);
                    assert_eq!(cached_count(&click), 1);
                }
            }
        }
    }
}
