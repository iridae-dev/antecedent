//! 1.9 positive multi-completion PAG evidence (R-10).
//!
//! `conformance/estimate/pag_ate_envelope_identified` is a PAG whose seven MAG
//! completions include six that identify by adjustment with visible edges out
//! of `t`, at two materially different effects (adjust `{z}`: direct effect;
//! adjust `{}`: total effect through `z -> m -> y`), plus one completion with no
//! visible edge out of `t` that stays unidentified (mass 1 of 7). The
//! Frequentist mass-weighted ATE / CATE and their joint-IF SEs are pinned
//! against the independent numpy reference committed next to the fixture;
//! the Bayesian envelope mean and SD are seeded output pins.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};
use antecedent_validate::PredictiveCheckKind;

pub fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/pag_ate_envelope_identified/expected.json"
    ))
    .unwrap()
}

fn columns(pin: &serde_json::Value) -> Vec<&str> {
    pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect()
}

pub fn expand_contingency(pin: &serde_json::Value) -> TabularData {
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

pub fn pag_from_pin(pin: &serde_json::Value) -> Pag {
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

pub fn ate_query(pin: &serde_json::Value) -> AverageEffectQuery {
    AverageEffectQuery::with_levels(
        VariableId::from_raw(index(pin, pin["query"]["treatment"].as_str().unwrap())),
        VariableId::from_raw(index(pin, pin["query"]["outcome"].as_str().unwrap())),
        pin["query"]["control_level"].as_f64().unwrap(),
        pin["query"]["active_level"].as_f64().unwrap(),
    )
}

pub fn conditional_query(pin: &serde_json::Value) -> ConditionalEffectQuery {
    let modifier = index(pin, pin["conditional"]["modifier"].as_str().unwrap());
    ConditionalEffectQuery::try_new(
        ate_query(pin).with_effect_modifiers([VariableId::from_raw(modifier)]),
    )
    .unwrap()
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

/// Fresh run, prepare and estimate click, plus a same-schema refresh except
/// under `full` (the refuter suite is the slow part in debug builds; refresh
/// reuse is already pinned at `none`/`cheap`).
fn run_prepared(
    study: &Study,
    data: &TabularData,
    seed: u64,
    suite: RefuteSuite,
) -> Vec<antecedent::StudyResult> {
    let (ctx, sink) = recording_ctx(seed);
    let fresh = study.clone().run(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1, "a fresh run identifies exactly once");
    let mut prepared: PreparedStudy = study.prepare(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 2, "prepare identifies exactly once");
    let click = prepared.estimate(data, &ctx).unwrap();
    assert_eq!(cached_count(&fresh), 0);
    assert_eq!(cached_count(&click), 1);
    assert!((click.estimate.ate - fresh.estimate.ate).abs() < 1e-12);
    let mut out = vec![fresh, click];
    if suite != RefuteSuite::Full {
        let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
        assert_eq!(cached_count(&refreshed), 1, "same-schema refresh must reuse the envelope");
        out.push(refreshed);
    }
    assert_eq!(identify_computations(&sink), 2, "clicks must reuse the prepared envelope");
    out
}

fn assert_envelope(result: &antecedent::StudyResult, pin: &serde_json::Value) {
    let id = &pin["identification"];
    let expected = format!(
        "identified_mass={}, unidentified_mass={}, cases={}",
        id["identified_mass"].as_f64().unwrap(),
        id["unidentified_mass"].as_f64().unwrap(),
        id["completion_count"].as_u64().unwrap()
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "identify.pag.envelope" && d.message.contains(&expected)),
        "PAG envelope diagnostic must report the pinned completion masses ({expected})"
    );
    assert_eq!(result.support_status.unwrap().as_str(), "licensed");
    assert_eq!(format!("{:?}", result.identification.status), id["status"].as_str().unwrap());
}

fn assert_validation(result: &antecedent::StudyResult, suite: RefuteSuite, bayesian: bool) {
    if suite == RefuteSuite::None {
        assert!(result.refutations.is_empty(), "validation none must emit no reports");
        assert!(result.predictive_checks.is_empty(), "validation none must not run PPC");
        return;
    }
    assert!(!result.refutations.is_empty(), "{suite:?} must execute a refuter");
    if bayesian {
        assert!(result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Prior));
        assert!(result.predictive_checks.iter().any(|c| c.kind == PredictiveCheckKind::Posterior));
    } else {
        assert!(
            result.diagnostics.iter().any(|d| d.code.as_ref() == "refute.envelope.effect_mixture"),
            "Frequentist {suite:?} must mix effect refuters across completions"
        );
    }
}

fn build(
    data: &TabularData,
    pag: &Pag,
    accepted: bool,
    query: CausalQuery,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    let builder = Study::tabular(data.clone());
    let builder = if accepted {
        builder.graph(AcceptedGraph::from(pag.clone()))
    } else {
        builder.graph(pag.clone())
    };
    builder.query(query).inference(inference).refute(suite).bootstrap_replicates(0).build().unwrap()
}

fn bayes(block: &serde_json::Value) -> InferenceMode {
    InferenceMode::Bayesian(
        BayesianConfig::conjugate()
            .n_draws(usize::try_from(block["n_draws"].as_u64().unwrap()).unwrap())
            .prior_scale(block["prior_scale"].as_f64().unwrap()),
    )
}

/// The envelope itself: seven completions, six identified at the pinned
/// adjustment sets (so the mixture really averages two different effects),
/// one unidentified.
#[test]
fn pag_identified_envelope_completions_match_the_fixture() {
    let pin = pin();
    let pag = pag_from_pin(&pin);
    for query in [ate_query(&pin), conditional_query(&pin).inner] {
        let env = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
            .identify_pag_envelope(&pag, &query)
            .unwrap();
        let id = &pin["identification"];
        assert_eq!(env.cases.len() as u64, id["completion_count"].as_u64().unwrap());
        assert_eq!(env.identified_weight.0, id["identified_mass"].as_f64().unwrap());
        assert_eq!(env.unidentified_weight.0, id["unidentified_mass"].as_f64().unwrap());
        let sets: Vec<Vec<String>> = env
            .cases
            .iter()
            .filter(|c| !c.result.estimands.is_empty())
            .map(|c| {
                c.result.estimands[0]
                    .adjustment_set
                    .iter()
                    .map(|v| columns(&pin)[v.raw() as usize].to_owned())
                    .collect()
            })
            .collect();
        let expected: Vec<Vec<String>> = pin["identified_completions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                c["adjustment_set"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_owned())
                    .collect()
            })
            .collect();
        assert_eq!(sets, expected, "identified completion adjustment sets");
        assert!(sets.iter().any(Vec::is_empty) && sets.iter().any(|s| !s.is_empty()));
    }
}

#[test]
fn pag_identified_envelope_average_effect_pins() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let pag = pag_from_pin(&pin);
    let query = CausalQuery::AverageEffect(ate_query(&pin));
    let freq = &pin["frequentist"];
    let bayes_block = &pin["bayesian"];
    let tol = freq["absolute_tolerance"].as_f64().unwrap();
    let seed = bayes_block["seed"].as_u64().unwrap();
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let study =
                build(&data, &pag, accepted, query.clone(), InferenceMode::Frequentist, suite);
            let results = run_prepared(&study, &data, seed, suite);
            for result in &results {
                assert_envelope(result, &pin);
                assert_validation(result, suite, false);
                assert_eq!(result.logical_plan.estimator.as_deref(), freq["estimator"].as_str());
                assert!(
                    (result.estimate.ate - freq["expected_ate"].as_f64().unwrap()).abs() < tol,
                    "PAG envelope ATE {}",
                    result.estimate.ate
                );
                assert!(
                    (result.estimate.se_analytic - freq["expected_se"].as_f64().unwrap()).abs()
                        < tol,
                    "PAG envelope joint-IF SE {} vs numpy reference",
                    result.estimate.se_analytic
                );
                assert!(
                    !result
                        .diagnostics
                        .iter()
                        .any(|d| d.code.as_ref()
                            == "estimate.envelope.se_omits_between_atom_variance")
                );
            }

            let study = build(&data, &pag, accepted, query.clone(), bayes(bayes_block), suite);
            let results = run_prepared(&study, &data, seed, suite);
            for result in &results {
                assert_envelope(result, &pin);
                assert_validation(result, suite, true);
                assert_eq!(
                    result.logical_plan.estimator.as_deref(),
                    bayes_block["estimator"].as_str()
                );
                // The posterior reports unidentified mass as a probability:
                // 1 of 7 equally weighted completions, never renormalized away.
                let posterior = result.posterior.as_ref().unwrap();
                let id = &pin["identification"];
                let unidentified = id["unidentified_mass"].as_f64().unwrap()
                    / id["completion_count"].as_f64().unwrap();
                assert!((posterior.unidentified_mass - unidentified).abs() < 1e-15);
                // Exact envelope moments (posterior summaries), not a draw average.
                let sd = result.estimate.se_analytic;
                eprintln!("bayes ate mean={:.17} sd={sd:.17}", result.estimate.ate);
                assert!(
                    (result.estimate.ate - bayes_block["expected_mean"].as_f64().unwrap()).abs()
                        < bayes_block["absolute_tolerance"].as_f64().unwrap(),
                    "Bayesian PAG envelope mean {}",
                    result.estimate.ate
                );
                assert!(
                    (sd - bayes_block["expected_sd"].as_f64().unwrap()).abs()
                        < bayes_block["absolute_tolerance"].as_f64().unwrap(),
                    "Bayesian PAG envelope SD {sd}"
                );
                if suite == RefuteSuite::Full {
                    assert!(posterior.prior_sensitivity.is_some());
                }
            }
        }
    }
}

#[test]
fn pag_identified_envelope_conditional_effect_pins() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let pag = pag_from_pin(&pin);
    let query = CausalQuery::ConditionalEffect(conditional_query(&pin));
    let freq = &pin["conditional"]["frequentist"];
    let bayes_block = &pin["conditional"]["bayesian"];
    let tol = freq["absolute_tolerance"].as_f64().unwrap();
    let seed = bayes_block["seed"].as_u64().unwrap();
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let study =
                build(&data, &pag, accepted, query.clone(), InferenceMode::Frequentist, suite);
            let results = run_prepared(&study, &data, seed, suite);
            for result in &results {
                assert_envelope(result, &pin);
                assert_validation(result, suite, false);
                assert_eq!(result.logical_plan.estimator.as_deref(), freq["estimator"].as_str());
                assert!(
                    (result.estimate.ate - freq["expected_ate"].as_f64().unwrap()).abs() < tol,
                    "PAG envelope CATE {}",
                    result.estimate.ate
                );
                assert!(
                    (result.estimate.se_analytic - freq["expected_se"].as_f64().unwrap()).abs()
                        < tol,
                    "PAG envelope CATE joint-IF SE {} vs numpy reference",
                    result.estimate.se_analytic
                );
            }

            let study = build(&data, &pag, accepted, query.clone(), bayes(bayes_block), suite);
            let results = run_prepared(&study, &data, seed, suite);
            for result in &results {
                assert_envelope(result, &pin);
                assert_validation(result, suite, true);
                assert_eq!(
                    result.logical_plan.estimator.as_deref(),
                    bayes_block["estimator"].as_str()
                );
                eprintln!(
                    "bayes cate mean={:.17} sd={:.17}",
                    result.estimate.ate, result.estimate.se_analytic
                );
                assert!(
                    (result.estimate.ate - bayes_block["expected_mean"].as_f64().unwrap()).abs()
                        < bayes_block["absolute_tolerance"].as_f64().unwrap(),
                    "Bayesian PAG CATE envelope mean {}",
                    result.estimate.ate
                );
                assert!(
                    (result.estimate.se_analytic - bayes_block["expected_sd"].as_f64().unwrap())
                        .abs()
                        < bayes_block["absolute_tolerance"].as_f64().unwrap(),
                    "Bayesian PAG CATE envelope SD {}",
                    result.estimate.se_analytic
                );
            }
        }
    }
}
