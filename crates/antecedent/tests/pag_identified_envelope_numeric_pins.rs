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

fn pin() -> serde_json::Value {
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

fn ate_query(pin: &serde_json::Value) -> AverageEffectQuery {
    AverageEffectQuery::with_levels(
        VariableId::from_raw(index(pin, pin["query"]["treatment"].as_str().unwrap())),
        VariableId::from_raw(index(pin, pin["query"]["outcome"].as_str().unwrap())),
        pin["query"]["control_level"].as_f64().unwrap(),
        pin["query"]["active_level"].as_f64().unwrap(),
    )
}

fn conditional_query(pin: &serde_json::Value) -> ConditionalEffectQuery {
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

fn response_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/class_aware_envelope/pag_identified.json"
    ))
    .unwrap()
}

/// Fresh run and prepared click of a response study; the click must reuse the
/// frozen envelope.
fn fresh_and_click(study: &Study, data: &TabularData, seed: u64) -> Vec<antecedent::StudyResult> {
    let (ctx, sink) = recording_ctx(seed);
    let fresh = study.clone().run(&ctx).unwrap();
    let click = study.prepare(&ctx).unwrap().estimate(data, &ctx).unwrap();
    assert_eq!(identify_computations(&sink), 2, "the click must reuse the prepared envelope");
    assert_eq!(cached_count(&fresh), 0);
    assert_eq!(cached_count(&click), 1);
    vec![fresh, click]
}

fn atom_scalars(result: &antecedent::StudyResult) -> Vec<Option<f64>> {
    use antecedent_core::ResponseValue;
    let mixture = result.structural_response.as_ref().expect("per-completion response atoms");
    assert_eq!(
        mixture.weight_basis,
        antecedent::result::StructuralWeightBasis::CompletionEnumeration
    );
    mixture
        .atoms
        .iter()
        .map(|atom| match &atom.value {
            Some(ResponseValue::Scalar(value)) => Some(*value),
            None => None,
            other => panic!("expected scalar completion responses, got {other:?}"),
        })
        .collect()
}

/// `InterventionResponse × Pag` for both inferences on explicit and accepted
/// structure. On the PAG ATE fixture every completion that identifies the ATE
/// by adjustment must give `do(t=1) − do(t=0)` equal to that completion's
/// numpy reference effect (`frequentist.completion_effects`), the completions
/// disagree (0.350 vs 0.422), and the published response is the completion
/// identified set: its lower/upper are the extreme completion levels, never a
/// weighted mean.
#[test]
fn pag_identified_envelope_intervention_response_matches_completion_effects() {
    use antecedent_core::{
        Intervention, ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue,
        Value,
    };
    let pin = pin();
    let rpin = response_pin();
    let spec = &rpin["intervention"];
    assert_eq!(spec["source"], "conformance/estimate/pag_ate_envelope_identified");
    let data = expand_contingency(&pin);
    let pag = pag_from_pin(&pin);
    let reference: Vec<f64> = pin["frequentist"]["completion_effects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    // Which enumerated completions identify the ATE by adjustment, in the
    // order `completion_effects` lists them.
    let cases = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
        .identify_pag_envelope(&pag, &ate_query(&pin))
        .unwrap()
        .cases;
    let adjusted: Vec<usize> =
        (0..cases.len()).filter(|&i| !cases[i].result.estimands.is_empty()).collect();
    assert_eq!(adjusted.len(), reference.len());
    let treatment = VariableId::from_raw(index(&pin, "t"));
    let outcome = VariableId::from_raw(index(&pin, "y"));
    let query = |level: f64| {
        CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome,
            interventions: Arc::from([Intervention::set(treatment, Value::f64(level))]),
        }))
    };
    let seed = spec["bayesian"]["seed"].as_u64().unwrap();
    for bayesian in [false, true] {
        let (inference, tolerance, estimator) = if bayesian {
            (
                bayes(&spec["bayesian"]),
                spec["bayesian_contrast_tolerance"].as_f64().unwrap(),
                "response.bayesian",
            )
        } else {
            (
                InferenceMode::Frequentist,
                spec["frequentist_contrast_tolerance"].as_f64().unwrap(),
                "response.intervention_gcomp",
            )
        };
        for accepted in [false, true] {
            let label = format!("bayesian={bayesian} accepted={accepted}");
            let run = |level: f64| {
                let study = build(
                    &data,
                    &pag,
                    accepted,
                    query(level),
                    inference.clone(),
                    RefuteSuite::None,
                );
                fresh_and_click(&study, &data, seed)
            };
            let high = run(1.0);
            let low = run(0.0);
            for (hi, lo) in high.iter().zip(&low) {
                for result in [hi, lo] {
                    assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{label}");
                    assert_eq!(result.logical_plan.estimator.as_deref(), Some(estimator));
                    assert_eq!(
                        format!("{:?}", result.identification.status),
                        "PartiallyIdentified"
                    );
                    assert!(
                        result
                            .diagnostics
                            .iter()
                            .any(|d| d.code.as_ref() == "identify.pag.envelope"),
                        "{label}: the PAG envelope must be disclosed"
                    );
                    assert!(result.refutations.is_empty(), "{label}: validation none");
                }
                let hi_atoms = atom_scalars(hi);
                let lo_atoms = atom_scalars(lo);
                assert_eq!(hi_atoms.len(), cases.len(), "{label}: one atom per completion");
                for (&case, &truth) in adjusted.iter().zip(&reference) {
                    let contrast = hi_atoms[case].unwrap() - lo_atoms[case].unwrap();
                    assert!(
                        (contrast - truth).abs() < tolerance,
                        "{label}: completion {case} contrast {contrast} vs reference {truth}"
                    );
                }
                let levels: Vec<f64> = hi_atoms.iter().flatten().copied().collect();
                let (min, max) = levels
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
                assert!(max - min > 0.02, "{label}: the completions must disagree");
                let ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(envelope)) =
                    &hi.response.as_ref().unwrap().estimate
                else {
                    panic!("{label}: disagreeing completions publish the identified set");
                };
                assert_eq!(envelope.lower.as_ref(), &[min], "{label}");
                assert_eq!(envelope.upper.as_ref(), &[max], "{label}");
                if bayesian {
                    assert!(hi.diagnostics.iter().any(|d| {
                        d.code.as_ref() == "estimate.envelope.response_posterior_not_mixed"
                    }));
                }
            }
        }
    }
}

/// `ResponseCurve × Pag` for both inferences on explicit and accepted
/// structure (`class_aware_envelope/pag_identified.json`, curve section): a
/// continuous linear law on a PAG whose four MAG completions all identify by
/// adjusting `{z}`. The agreeing completions publish a point-identified curve
/// whose `[0, 1]` levels match the structural sample-covariate levels and
/// whose contrast matches both the structural slope and the `AverageEffect`
/// cell on the same PAG and table.
#[allow(clippy::many_single_char_names)]
#[test]
fn pag_identified_envelope_response_curve_pins_structural_contrast() {
    use antecedent_core::{
        ContinuousDomain, GridSpec, ResponseFunctional, ResponseIdentification, ResponseQuery,
        ResponseValue,
    };
    let rpin = response_pin();
    let spec = &rpin["curve"];
    let n = usize::try_from(spec["law"]["n"].as_u64().unwrap()).unwrap();
    let wave = |i: usize, freq: f64| (i as f64 * freq).sin();
    let z: Vec<f64> = (0..n).map(|i| wave(i, 0.37)).collect();
    let r: Vec<f64> = (0..n).map(|i| wave(i, 0.53)).collect();
    let t: Vec<f64> = (0..n).map(|i| 0.5 + 0.5 * z[i] + 0.4 * r[i] + 0.5 * wave(i, 0.61)).collect();
    let noise: Vec<f64> = (0..n).map(|i| 0.02 * wave(i, 0.29)).collect();
    let slope = spec["true_contrast"].as_f64().unwrap();
    let y: Vec<f64> = (0..n).map(|i| 0.1 + slope * t[i] + 0.2 * z[i] + noise[i]).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let grid: Vec<f64> =
        spec["grid"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let truth: Vec<f64> =
        grid.iter().map(|a| 0.1 + slope * a + 0.2 * mean(&z) + mean(&noise)).collect();
    let columns: Vec<&str> =
        spec["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert_eq!(columns, ["t", "y", "z", "r"]);
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("r", r.as_slice()),
    ])
    .unwrap();
    let mut pag = Pag::with_variables(4);
    let node = |name: &str| {
        DenseNodeId::from_raw(
            u32::try_from(columns.iter().position(|c| *c == name).unwrap()).unwrap(),
        )
    };
    for edge in spec["graph"]["marked_edges"].as_array().unwrap() {
        pag.insert_marked(MarkedEdge {
            a: node(edge[0].as_str().unwrap()),
            b: node(edge[1].as_str().unwrap()),
            at_a: endpoint(edge[2].as_str().unwrap()),
            at_b: endpoint(edge[3].as_str().unwrap()),
            middle: MiddleMark::Empty,
        })
        .unwrap();
    }
    let (treatment, outcome) = (VariableId::from_raw(0), VariableId::from_raw(1));
    let curve = CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome,
        treatment: ContinuousDomain::new(treatment, GridSpec::Values(grid.clone().into())),
    }));
    let id = &spec["identification"];
    let envelope = format!(
        "identified_mass={}, unidentified_mass={}, cases={}",
        id["identified_mass"].as_f64().unwrap(),
        id["unidentified_mass"].as_f64().unwrap(),
        id["completion_count"].as_u64().unwrap()
    );
    for bayesian in [false, true] {
        let block = if bayesian { &spec["bayesian"] } else { &spec["frequentist"] };
        let inference = if bayesian { bayes(block) } else { InferenceMode::Frequentist };
        let level_tol = block["level_tolerance"].as_f64().unwrap();
        let contrast_tol = block["contrast_tolerance"].as_f64().unwrap();
        for accepted in [false, true] {
            let label = format!("bayesian={bayesian} accepted={accepted}");
            let ate = build(
                &data,
                &pag,
                accepted,
                CausalQuery::AverageEffect(AverageEffectQuery::with_levels(
                    treatment, outcome, grid[0], grid[1],
                )),
                inference.clone(),
                RefuteSuite::None,
            )
            .run(&ExecutionContext::for_tests(1))
            .unwrap()
            .estimate
            .ate;
            let study =
                build(&data, &pag, accepted, curve.clone(), inference.clone(), RefuteSuite::None);
            for result in fresh_and_click(&study, &data, 1) {
                assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{label}");
                assert_eq!(
                    result.logical_plan.estimator.as_deref(),
                    block["estimator"].as_str(),
                    "{label}"
                );
                assert!(
                    result.diagnostics.iter().any(|d| d.code.as_ref() == "identify.pag.envelope"
                        && d.message.contains(&envelope)),
                    "{label}: envelope must report {envelope}"
                );
                let atoms = &result.structural_response.as_ref().expect("completion atoms").atoms;
                assert_eq!(atoms.len() as u64, id["completion_count"].as_u64().unwrap());
                let ResponseIdentification::PointIdentified(ResponseValue::Surface {
                    mean: curve,
                    ..
                }) = &result.response.as_ref().unwrap().estimate
                else {
                    panic!("{label}: agreeing completions publish a point-identified curve");
                };
                for atom in atoms {
                    let Some(ResponseValue::Surface { mean, .. }) = &atom.value else {
                        panic!("{label}: every completion is evaluable");
                    };
                    assert_eq!(mean.as_ref(), curve.as_ref(), "{label}: completions agree");
                }
                for (value, level) in curve.iter().zip(&truth) {
                    assert!((value - level).abs() < level_tol, "{label}: level {value} vs {level}");
                }
                let contrast = curve[1] - curve[0];
                assert!(
                    (contrast - slope).abs() < contrast_tol,
                    "{label}: curve contrast {contrast} vs structural {slope}"
                );
                assert!(
                    (contrast - ate).abs() < contrast_tol,
                    "{label}: curve contrast {contrast} vs AverageEffect {ate}"
                );
                assert!(result.refutations.is_empty(), "{label}: validation none");
            }
        }
    }
}
