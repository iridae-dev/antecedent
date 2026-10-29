//! Numeric pins for licensed `Cpdag`/`Pag` `ConditionalEffect` cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

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

/// The modifier-stratified contrast `sum_z P(z) [P(y=1 | t=1, z) - P(y=1 | t=0, z)]` read straight
/// off the contingency table by definition. With the modifier `z` conditioned on, every completion
/// of the equivalence class adjusts for `z`, so this is the conditional effect whatever the
/// orientation of the undirected `z - t` edge; it does not run any product code.
fn stratified_contrast_from_table(pin: &serde_json::Value) -> f64 {
    let cells = pin["contingency_table"].as_array().unwrap();
    let count = |z: f64, t: f64, y: Option<f64>| -> f64 {
        cells
            .iter()
            .filter(|c| {
                c["z"].as_f64().unwrap() == z
                    && c["t"].as_f64().unwrap() == t
                    && y.is_none_or(|y| c["y"].as_f64().unwrap() == y)
            })
            .map(|c| c["count"].as_f64().unwrap())
            .sum()
    };
    let total: f64 = cells.iter().map(|c| c["count"].as_f64().unwrap()).sum();
    [0.0, 1.0]
        .into_iter()
        .map(|z| {
            let weight = (count(z, 0.0, None) + count(z, 1.0, None)) / total;
            let p1 = count(z, 1.0, Some(1.0)) / count(z, 1.0, None);
            let p0 = count(z, 0.0, Some(1.0)) / count(z, 0.0, None);
            weight * (p1 - p0)
        })
        .sum()
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
    // The pinned posterior band is 0.01; the observed deviation is 1.5e-3 on 64 draws.
    let bayes_tol = bayes["absolute_tolerance"].as_f64().unwrap().min(5e-3);
    let seed = bayes["seed"].as_u64().unwrap();
    let table_truth = stratified_contrast_from_table(&pin);
    assert!(
        (freq_ate - table_truth).abs() < 1e-12 && (bayes_ate - table_truth).abs() < 1e-12,
        "pinned conditional effects {freq_ate} / {bayes_ate} vs table definition {table_truth}"
    );
    let graphs = [
        ("cpdag", ClassGraph::Cpdag(cpdag_from_pin(&pin))),
        ("pag", ClassGraph::Pag(pag_from_class_pin(&class))),
    ];
    for (class_name, graph) in graphs {
        if let ClassGraph::Pag(pag) = &graph {
            let env = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
                .identify_pag_envelope(pag, &query.inner)
                .unwrap();
            assert_eq!(
                env.identified_weight.0, 0.0,
                "conditioning cannot rescue MAG non-amenability"
            );
            continue;
        }
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
                    assert!(
                        (result.estimate().as_effect().unwrap().ate - freq_ate).abs() < freq_tol,
                        "{class_name} freq"
                    );
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
                        (result.estimate().as_effect().unwrap().ate - bayes_ate).abs() < bayes_tol,
                        "{class_name} bayes"
                    );
                    assert_validation_presence(result, suite, true);
                    assert_eq!(cached_count(&click), 1);
                }
            }
        }
    }
}

#[test]
fn checked_pag_conditional_certificate_retains_completion_masses() {
    let n = 80;
    let t: Vec<f64> = (0..n).map(|i| f64::from(i) / f64::from(n)).collect();
    let y: Vec<f64> = t.iter().map(|value| 1.0 + 2.0 * value).collect();
    let r: Vec<f64> = (0..n).map(|i| f64::from(i % 2)).collect();
    let s: Vec<f64> = (0..n).map(|i| f64::from(i).sin()).collect();
    let v: Vec<f64> = (0..n).map(|i| f64::from(i).cos()).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("r", r.as_slice()),
        ("s", s.as_slice()),
        ("v", v.as_slice()),
    ])
    .unwrap();
    let mut pag = Pag::with_variables(5);
    pag.insert_marked(MarkedEdge {
        a: DenseNodeId::from_raw(2),
        b: DenseNodeId::from_raw(0),
        at_a: Endpoint::Tail,
        at_b: Endpoint::Arrow,
        middle: MiddleMark::Empty,
    })
    .unwrap();
    pag.insert_marked(MarkedEdge {
        a: DenseNodeId::from_raw(0),
        b: DenseNodeId::from_raw(1),
        at_a: Endpoint::Tail,
        at_b: Endpoint::Arrow,
        middle: MiddleMark::Empty,
    })
    .unwrap();
    pag.insert_marked(MarkedEdge {
        a: DenseNodeId::from_raw(3),
        b: DenseNodeId::from_raw(4),
        at_a: Endpoint::Circle,
        at_b: Endpoint::Circle,
        middle: MiddleMark::Empty,
    })
    .unwrap();
    let inner =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0)
            .with_effect_modifiers([VariableId::from_raw(2)]);
    let query = ConditionalEffectQuery::try_new(inner).unwrap();
    let study = Study::tabular(data.clone())
        .graph(pag.clone())
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let expected = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
        .identify_pag_envelope(&pag, &conditional_query(&ate_pin()).inner)
        .unwrap();
    let ctx = ExecutionContext::for_tests(19);
    let fresh = study.clone().run(&ctx).unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let click = prepared.estimate(&data, &ctx).unwrap();
    for result in [&fresh, &click] {
        let Some(certificate) = &result.certificate else {
            panic!("checked conditional result omitted its identification certificate");
        };
        let crate_identification = &certificate.identification;
        let antecedent::Identification::Envelope { envelope, .. } = crate_identification else {
            panic!("PAG conditional result did not retain its completion envelope");
        };
        // The circle-circle pair s o-o v has three completions (s -> v, v -> s, s <-> v) and the
        // chain r -> t -> y with the modifier r is amenable in each, so all three mass units
        // are identified and none is not.
        assert_eq!(envelope.cases.len(), 3);
        assert_eq!(format!("{:?}", envelope.identified_weight), "ProbabilityMass(3.0)");
        assert_eq!(format!("{:?}", envelope.unidentified_weight), "ProbabilityMass(0.0)");
        assert_eq!(envelope.cases.len(), expected.cases.len());
        assert_eq!(envelope.identified_weight, expected.identified_weight);
        assert_eq!(envelope.unidentified_weight, expected.unidentified_weight);
    }
}
