//! 1.1 numeric pins for licensed PAG and ADMG ATE cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};
use antecedent_validate::PredictiveCheckKind;

fn pag_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/pag_ate_envelope/expected.json"
    ))
    .unwrap()
}

fn admg_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
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

fn endpoint(mark: &str) -> Endpoint {
    match mark {
        "tail" => Endpoint::Tail,
        "arrow" => Endpoint::Arrow,
        "circle" => Endpoint::Circle,
        "conflict" => Endpoint::Conflict,
        other => panic!("unknown endpoint {other}"),
    }
}

fn node(columns: &[&str], name: &str) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(columns.iter().position(|c| *c == name).unwrap()).unwrap())
}

fn pag_from_pin(pin: &serde_json::Value) -> Pag {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut pag = Pag::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["marked_edges"].as_array().unwrap() {
        let a = edge[0].as_str().unwrap();
        let b = edge[1].as_str().unwrap();
        pag.insert_marked(MarkedEdge {
            a: node(&columns, a),
            b: node(&columns, b),
            at_a: endpoint(edge[2].as_str().unwrap()),
            at_b: endpoint(edge[3].as_str().unwrap()),
            middle: MiddleMark::Empty,
        })
        .unwrap();
    }
    pag
}

fn admg_from_pin(pin: &serde_json::Value) -> Admg {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut admg = Admg::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        admg.insert_directed(
            node(&columns, edge[0].as_str().unwrap()),
            node(&columns, edge[1].as_str().unwrap()),
        )
        .unwrap();
    }
    for edge in pin["graph"]["bidirected_edges"].as_array().unwrap() {
        admg.insert_bidirected(
            node(&columns, edge[0].as_str().unwrap()),
            node(&columns, edge[1].as_str().unwrap()),
        )
        .unwrap();
    }
    admg
}

fn query_from_pin(pin: &serde_json::Value) -> AverageEffectQuery {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let treatment = node(&columns, pin["query"]["treatment"].as_str().unwrap());
    let outcome = node(&columns, pin["query"]["outcome"].as_str().unwrap());
    AverageEffectQuery::with_levels(
        VariableId::from_raw(treatment.raw()),
        VariableId::from_raw(outcome.raw()),
        pin["query"]["control_level"].as_f64().unwrap(),
        pin["query"]["active_level"].as_f64().unwrap(),
    )
}

/// Records every progress label so a test can prove identification was
/// computed exactly where the contract says (fresh run, prepare) and never on
/// a prepared estimate or refresh click.
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

fn assert_pag_envelope_diagnostic(result: &antecedent::StudyResult, pin: &serde_json::Value) {
    let identification = &pin["identification"];
    let expected = format!(
        "identified_mass={}, unidentified_mass={}, cases={}",
        identification["identified_mass"].as_f64().unwrap(),
        identification["unidentified_mass"].as_f64().unwrap(),
        identification["completion_count"].as_u64().unwrap()
    );
    assert!(
        result.diagnostics.iter().any(|d| {
            d.code.as_ref() == "identify.pag.envelope" && d.message.contains(&expected)
        }),
        "PAG envelope diagnostic must report the pinned completion masses ({expected})"
    );
}

fn assert_validation_presence(
    result: &antecedent::StudyResult,
    suite: RefuteSuite,
    bayesian: bool,
) {
    match suite {
        RefuteSuite::None => {
            assert!(result.refutations.is_empty(), "validation none must emit no reports");
            assert!(result.predictive_checks.is_empty(), "validation none must not run PPC");
        }
        RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => {
            assert!(!result.refutations.is_empty(), "{suite:?} must execute a refuter");
            if bayesian {
                assert!(
                    result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Prior),
                    "{suite:?} must attach prior PPC"
                );
                assert!(
                    result
                        .predictive_checks
                        .iter()
                        .any(|c| c.kind == PredictiveCheckKind::Posterior),
                    "{suite:?} must attach posterior PPC"
                );
            }
        }
    }
}

fn assert_prepared_reuse(
    fresh: &antecedent::StudyResult,
    click: &antecedent::StudyResult,
    refreshed: &antecedent::StudyResult,
    expected: f64,
    tolerance: f64,
) {
    assert!((click.estimate.ate - expected).abs() < tolerance);
    assert!((fresh.estimate.ate - expected).abs() < tolerance);
    assert!((refreshed.estimate.ate - expected).abs() < tolerance);
    assert!((click.estimate.ate - fresh.estimate.ate).abs() < 1e-12);
    assert!((refreshed.estimate.ate - click.estimate.ate).abs() < 1e-12);
    assert_eq!(click.support_status.unwrap().as_str(), "licensed");
    assert_eq!(cached_count(fresh), 0);
    assert_eq!(cached_count(click), 1);
    assert_eq!(cached_count(refreshed), 1, "same-schema refresh must reuse identification");
}

fn run_prepared(
    study: &Study,
    data: &TabularData,
    seed: u64,
    expected_identifier: &str,
    expected_estimator: &str,
) -> (antecedent::StudyResult, antecedent::StudyResult, antecedent::StudyResult) {
    let (ctx, sink) = recording_ctx(seed);
    let fresh = study.clone().run(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1, "a fresh run identifies exactly once");
    let mut prepared: PreparedStudy = study.prepare(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 2, "prepare identifies exactly once");
    assert_eq!(
        prepared.plan().logical.record.identifier.as_deref(),
        Some(expected_identifier),
        "prepared plan must record the pinned identifier"
    );
    assert_eq!(
        prepared.plan().logical.record.estimator.as_deref(),
        Some(expected_estimator),
        "prepared plan must record the pinned estimator"
    );
    let click = prepared.estimate(data, &ctx).unwrap();
    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    assert_eq!(
        identify_computations(&sink),
        2,
        "estimate and refresh clicks must reuse the prepared identification"
    );
    (fresh, click, refreshed)
}

#[test]
fn pag_ate_envelope_numeric_pin() {
    let pin = pag_pin();
    let data = expand_contingency(&pin);
    let pag = pag_from_pin(&pin);
    let query = query_from_pin(&pin);
    let freq = &pin["frequentist"];
    let bayes = &pin["bayesian"];
    let identifier = pin["identification"]["identifier"].as_str().unwrap();
    let freq_estimator = freq["estimator"].as_str().unwrap();
    let bayes_estimator = bayes["estimator"].as_str().unwrap();
    let bayes_seed = bayes["seed"].as_u64().unwrap();
    let freq_ate = freq["expected_ate"].as_f64().unwrap();
    let freq_tol = freq["absolute_tolerance"].as_f64().unwrap();
    let bayes_ate = bayes["expected_ate"].as_f64().unwrap();
    let bayes_tol = bayes["absolute_tolerance"].as_f64().unwrap();
    let unidentified = pin["identification"]["unidentified_mass"].as_f64().unwrap();
    assert_eq!(pin["identification"]["status"], "PartiallyIdentified");
    assert_eq!(bayes["backend"], "conjugate");

    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let mut freq_builder = Study::tabular(data.clone());
            freq_builder = if accepted {
                freq_builder.graph(AcceptedGraph::from(pag.clone()))
            } else {
                freq_builder.graph(pag.clone())
            };
            let freq_study = freq_builder
                .query(query.clone())
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            let (fresh, click, refreshed) =
                run_prepared(&freq_study, &data, bayes_seed, identifier, freq_estimator);
            assert_eq!(format!("{:?}", fresh.identification.status), "PartiallyIdentified");
            for result in [&fresh, &click, &refreshed] {
                assert_validation_presence(result, suite, false);
                assert_pag_envelope_diagnostic(result, &pin);
            }
            assert_prepared_reuse(&fresh, &click, &refreshed, freq_ate, freq_tol);

            let mut bayes_builder = Study::tabular(data.clone());
            bayes_builder = if accepted {
                bayes_builder.graph(AcceptedGraph::from(pag.clone()))
            } else {
                bayes_builder.graph(pag.clone())
            };
            let bayes_study = bayes_builder
                .query(query.clone())
                .inference(InferenceMode::Bayesian(
                    BayesianConfig::conjugate()
                        .n_draws(usize::try_from(bayes["n_draws"].as_u64().unwrap()).unwrap())
                        .prior_scale(bayes["prior_scale"].as_f64().unwrap()),
                ))
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            let (fresh, click, refreshed) =
                run_prepared(&bayes_study, &data, bayes_seed, identifier, bayes_estimator);
            assert_prepared_reuse(&fresh, &click, &refreshed, bayes_ate, bayes_tol);
            for result in [&fresh, &click, &refreshed] {
                assert_validation_presence(result, suite, true);
                assert_pag_envelope_diagnostic(result, &pin);
                let posterior = result.posterior.as_ref().expect("PAG Bayesian envelope");
                assert!((posterior.unidentified_mass - unidentified).abs() < 1e-15);
                if matches!(suite, RefuteSuite::Full) {
                    assert!(
                        posterior.prior_sensitivity.is_some(),
                        "Bayesian full validation must attach prior sensitivity"
                    );
                } else {
                    assert!(posterior.prior_sensitivity.is_none());
                }
            }
        }
    }
}

#[test]
fn admg_frontdoor_functional_effect_numeric_pin() {
    let pin = admg_pin();
    let data = expand_contingency(&pin);
    let admg = admg_from_pin(&pin);
    let query = query_from_pin(&pin);
    let identifier = pin["identification"]["identifier"].as_str().unwrap();
    let estimator = pin["frequentist"]["estimator"].as_str().unwrap();
    let expected = pin["frequentist"]["expected_ate"].as_f64().unwrap();
    let tolerance = pin["frequentist"]["absolute_tolerance"].as_f64().unwrap();
    assert_eq!(pin["identification"]["status"], "NonparametricallyIdentified");

    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let mut builder = Study::tabular(data.clone());
            builder = if accepted {
                builder.graph(AcceptedGraph::from(admg.clone()))
            } else {
                builder.graph(admg.clone())
            };
            let study =
                builder.query(query.clone()).refute(suite).bootstrap_replicates(0).build().unwrap();
            let (fresh, click, refreshed) = run_prepared(&study, &data, 1, identifier, estimator);
            assert_eq!(format!("{:?}", fresh.identification.status), "NonparametricallyIdentified");
            for result in [&fresh, &click, &refreshed] {
                assert_validation_presence(result, suite, false);
                assert!(result.posterior.is_none());
            }
            assert_prepared_reuse(&fresh, &click, &refreshed, expected, tolerance);
        }
    }
}
