//! Numeric pins for licensed Cpdag/Pag response cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};

fn pin() -> serde_json::Value {
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

fn continuous_linear(pin: &serde_json::Value) -> TabularData {
    let n = usize::try_from(pin["continuous"]["n"].as_u64().unwrap()).unwrap();
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for i in 0..n {
        let zi = if i % 2 == 0 { 0.0 } else { 1.0 };
        let ti = 0.3 + 0.4 * zi + 0.2 * ((i as f64) * 0.017).sin();
        let yi = 0.10 + 0.40 * ti + 0.20 * zi;
        t.push(ti);
        y.push(yi);
        z.push(zi);
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
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
    let spec = &pin["cpdag"]["graph"];
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

fn pag_from_pin(pin: &serde_json::Value) -> Pag {
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

fn ids(pin: &serde_json::Value) -> (VariableId, VariableId) {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let treatment = node(&columns, pin["query"]["treatment"].as_str().unwrap());
    let outcome = node(&columns, pin["query"]["outcome"].as_str().unwrap());
    (VariableId::from_raw(treatment.raw()), VariableId::from_raw(outcome.raw()))
}

fn curve_query(pin: &serde_json::Value) -> ResponseQuery {
    let (treatment, outcome) = ids(pin);
    let grid: Vec<f64> =
        pin["grid"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome,
        treatment: ContinuousDomain::new(treatment, GridSpec::Values(grid.into())),
    })
}

fn intervention_query(pin: &serde_json::Value, level: f64) -> ResponseQuery {
    let (treatment, outcome) = ids(pin);
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome,
        interventions: Arc::from([Intervention::set(treatment, Value::f64(level))]),
    })
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

fn response_values(result: &antecedent::StudyResult) -> Vec<f64> {
    if let Some(mixture) = result.structural_response.as_ref() {
        assert_eq!(
            mixture.weight_basis,
            antecedent::result::StructuralWeightBasis::CompletionEnumeration
        );
        assert!(
            mixture.conditional_on_identified.is_none(),
            "completion enumeration must not publish a probability-weighted causal summary"
        );
        let values = mixture
            .atoms
            .iter()
            .filter_map(|atom| atom.value.as_ref().map(|value| (atom.weight, value)))
            .collect::<Vec<_>>();
        let total_weight = values.iter().map(|(weight, _)| weight).sum::<f64>();
        if let Some((_, ResponseValue::Scalar(_))) = values.first() {
            return vec![
                values
                    .iter()
                    .map(|(weight, value)| {
                        let ResponseValue::Scalar(value) = value else { unreachable!() };
                        weight * value
                    })
                    .sum::<f64>()
                    / total_weight,
            ];
        }
        if let Some((_, ResponseValue::Surface { mean, .. })) = values.first() {
            return (0..mean.len())
                .map(|index| {
                    values
                        .iter()
                        .map(|(weight, value)| {
                            let ResponseValue::Surface { mean, .. } = value else { unreachable!() };
                            weight * mean[index]
                        })
                        .sum::<f64>()
                        / total_weight
                })
                .collect();
        }
    }
    let response = result.response().expect("class-aware response payload");
    match &response.estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => match value {
            ResponseValue::Scalar(value) => vec![*value],
            ResponseValue::Surface { mean, .. } => mean.to_vec(),
            other => panic!("unexpected response value {other:?}"),
        },
        other => panic!("unexpected identification payload {other:?}"),
    }
}

// Diagnostic display of actual completion outputs; enumeration weights here
// identify atoms, not a probability distribution over causal graphs.
fn response_atoms(result: &antecedent::StudyResult) -> Vec<(u64, f64, &Option<ResponseValue>)> {
    result.structural_response.as_ref().map_or_else(Vec::new, |mixture| {
        mixture.atoms.iter().map(|atom| (atom.graph_key, atom.weight, &atom.value)).collect()
    })
}

fn assert_envelope_diagnostic(
    result: &antecedent::StudyResult,
    section: &serde_json::Value,
    class: &str,
) {
    let identification = &section["identification"];
    let expected = format!(
        "identified_mass={}, unidentified_mass={}, cases={}",
        identification["identified_mass"].as_f64().unwrap(),
        identification["unidentified_mass"].as_f64().unwrap(),
        identification["completion_count"].as_u64().unwrap()
    );
    let code = if class == "cpdag" { "identify.cpdag.envelope" } else { "identify.pag.envelope" };
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == code && d.message.contains(&expected)),
        "{class} envelope diagnostic must report {expected}; got {:?}",
        result.diagnostics
    );
}

enum ClassGraph {
    Cpdag(Cpdag),
    Pag(Pag),
}

fn build_response(
    data: &TabularData,
    graph: &ClassGraph,
    accepted: bool,
    query: ResponseQuery,
) -> Study {
    let builder = Study::tabular(data.clone());
    let builder = match (graph, accepted) {
        (ClassGraph::Cpdag(cpdag), true) => builder.graph(AcceptedGraph::from(cpdag.clone())),
        (ClassGraph::Cpdag(cpdag), false) => builder.graph(cpdag.clone()),
        (ClassGraph::Pag(pag), true) => builder.graph(AcceptedGraph::from(pag.clone())),
        (ClassGraph::Pag(pag), false) => builder.graph(pag.clone()),
    };
    builder.query(query).refute(RefuteSuite::None).bootstrap_replicates(0).build().unwrap()
}

fn run_prepared(
    study: &Study,
    data: &TabularData,
) -> (antecedent::StudyResult, antecedent::StudyResult, antecedent::StudyResult) {
    let (ctx, sink) = recording_ctx(1);
    let fresh = study.clone().run(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1);
    let mut prepared: PreparedStudy = study.prepare(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 2);
    let click = prepared.estimate(data, &ctx).unwrap();
    let refreshed = rebind_estimate(&mut prepared, data.clone(), &ctx).unwrap();
    assert_eq!(identify_computations(&sink), 2);
    (fresh, click, refreshed)
}

// On this fixture the two DAG completions respectively adjust for z and do
// not adjust. Calculate both targets from the original contingency counts.
fn count_derived_intervention_targets(pin: &serde_json::Value, level: f64) -> [f64; 2] {
    let cells = pin["contingency_table"].as_array().unwrap();
    let total: f64 = cells.iter().map(|cell| cell["count"].as_f64().unwrap()).sum();
    let mut adjusted = 0.0;
    let mut exposed_count = 0.0;
    let mut exposed_outcome_sum = 0.0;
    for z in [0.0, 1.0] {
        let (mut z_count, mut selected_count, mut selected_outcome_sum) = (0.0, 0.0, 0.0);
        for cell in cells {
            if cell["z"].as_f64().unwrap() != z {
                continue;
            }
            let count = cell["count"].as_f64().unwrap();
            z_count += count;
            if cell["t"].as_f64().unwrap() == level {
                selected_count += count;
                selected_outcome_sum += count * cell["y"].as_f64().unwrap();
            }
        }
        assert!(selected_count > 0.0);
        adjusted += (z_count / total) * (selected_outcome_sum / selected_count);
        exposed_count += selected_count;
        exposed_outcome_sum += selected_outcome_sum;
    }
    let unadjusted = exposed_outcome_sum / exposed_count;
    let mut targets = [adjusted, unadjusted];
    targets.sort_by(f64::total_cmp);
    targets
}

fn assert_intervention_atoms_match_counts(
    result: &antecedent::StudyResult,
    targets: [f64; 2],
    label: &str,
) {
    let mut actual: Vec<f64> = response_atoms(result)
        .iter()
        .filter_map(|(_, _, value)| match value {
            Some(ResponseValue::Scalar(value)) => Some(*value),
            _ => None,
        })
        .collect();
    actual.sort_by(f64::total_cmp);
    assert_eq!(actual.len(), targets.len(), "{label}: {:?}", response_atoms(result));
    for (value, target) in actual.iter().zip(targets) {
        assert!((value - target).abs() < 1e-5, "{label}: {value} vs count-derived {target}");
    }
}

#[test]
fn class_aware_intervention_pins_against_ate_envelope() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let graphs = [
        ("cpdag", ClassGraph::Cpdag(cpdag_from_pin(&pin))),
        ("pag", ClassGraph::Pag(pag_from_pin(&pin))),
    ];
    for (class, graph) in graphs {
        if let ClassGraph::Pag(pag) = &graph {
            let (t, y) = ids(&pin);
            let env = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
                .identify_pag_envelope(pag, &AverageEffectQuery::binary_ate(t, y))
                .unwrap();
            assert_eq!(
                env.identified_weight.0, 0.0,
                "invisible causal edges cannot license response adjustment"
            );
            assert_eq!(env.unidentified_weight.0, env.cases.len() as f64);
            continue;
        }
        let section = &pin[class];
        let ate = section["ate_contrast"].as_f64().unwrap();
        let low_targets = count_derived_intervention_targets(&pin, 0.0);
        let high_targets = count_derived_intervention_targets(&pin, 1.0);
        let do0 = (low_targets[0] + low_targets[1]) / 2.0;
        let do1 = (high_targets[0] + high_targets[1]) / 2.0;
        assert!((do0 - 0.17).abs() < 1e-12);
        assert!((do1 - 0.63).abs() < 1e-12);
        let identifier = section["identification"]["identifier"].as_str().unwrap();
        let estimator = section["intervention"]["estimator"].as_str().unwrap();
        for accepted in [false, true] {
            let study = build_response(&data, &graph, accepted, intervention_query(&pin, 1.0));
            let (fresh, click, refreshed) = run_prepared(&study, &data);
            assert_eq!(fresh.logical_plan.identifier.as_deref(), Some(identifier));
            assert_eq!(fresh.logical_plan.estimator.as_deref(), Some(estimator));
            for result in [&fresh, &click, &refreshed] {
                assert_eq!(format!("{:?}", result.identification.status), "PartiallyIdentified");
                assert_envelope_diagnostic(result, section, class);
                let actual = response_values(result)[0];
                assert!(
                    (actual - do1).abs() < 1e-5,
                    "{class} accepted={accepted} do(1): actual={actual:.17} count-derived={do1:.17} delta={:.17e} atoms={:?}",
                    actual - do1,
                    response_atoms(result)
                );
                assert_intervention_atoms_match_counts(result, high_targets, "do(1)");
            }
            // A one-shot class response executes its retained prepared plan, so the
            // fresh result reports the identification it reused from that plan.
            assert_eq!(cached_count(&fresh), 1);
            assert_eq!(cached_count(&click), 1);
            assert_eq!(cached_count(&refreshed), 1);

            let low = build_response(&data, &graph, accepted, intervention_query(&pin, 0.0))
                .run(&ExecutionContext::for_tests(1))
                .unwrap();
            let lo = response_values(&low)[0];
            let hi = response_values(&fresh)[0];
            assert!((lo - do0).abs() < 1e-5, "{class} do(0) {lo} vs {do0}");
            assert_intervention_atoms_match_counts(&low, low_targets, "do(0)");
            assert!(
                (hi - lo - ate).abs()
                    < section["intervention"]["contrast_tolerance"].as_f64().unwrap(),
                "{class} contrast"
            );
        }
    }
}

/// Regression pins of the recorded class-aware curve means on the two-point
/// grid `[0, 1]`. These are output pins of this estimator, not accuracy
/// evidence: the grid lies outside the observed treatment support `[0.1, 0.9]`,
/// so the pinned levels are extrapolations. Accuracy against the generating law
/// is asserted in the three tests below, and the contrast of these extrapolated
/// levels is deliberately not compared with `AverageEffect` (a linear-projection
/// slope), because the two are different functionals.
#[test]
fn class_aware_curve_regression_pins_on_continuous_t() {
    let pin = pin();
    let data = continuous_linear(&pin);
    let graphs = [
        ("cpdag", ClassGraph::Cpdag(cpdag_from_pin(&pin))),
        ("pag", ClassGraph::Pag(pag_from_pin(&pin))),
    ];
    for (class, graph) in graphs {
        if let ClassGraph::Pag(pag) = &graph {
            let (t, y) = ids(&pin);
            let env = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
                .identify_pag_envelope(pag, &AverageEffectQuery::binary_ate(t, y))
                .unwrap();
            assert_eq!(
                env.identified_weight.0, 0.0,
                "invisible causal edges cannot license response adjustment"
            );
            assert_eq!(env.unidentified_weight.0, env.cases.len() as f64);
            continue;
        }
        let section = &pin[class];
        let expected: Vec<f64> = section["curve"]["mean"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        let mean_tol = section["curve"]["absolute_tolerance"].as_f64().unwrap();
        let identifier = section["identification"]["identifier"].as_str().unwrap();
        for accepted in [false, true] {
            let study = build_response(&data, &graph, accepted, curve_query(&pin));
            let (fresh, click, refreshed) = run_prepared(&study, &data);
            assert_eq!(fresh.logical_plan.identifier.as_deref(), Some(identifier));
            assert_eq!(
                fresh.logical_plan.estimator.as_deref(),
                Some(section["curve"]["estimator"].as_str().unwrap())
            );
            for result in [&fresh, &click, &refreshed] {
                assert_eq!(format!("{:?}", result.identification.status), "PartiallyIdentified");
                assert_envelope_diagnostic(result, section, class);
                let values = response_values(result);
                assert_eq!(values.len(), 2);
                assert!(
                    (values[0] - expected[0]).abs() < mean_tol,
                    "{class} accepted={accepted} curve[0]: actual={:.17} expected={:.17} delta={:.17e} tolerance={mean_tol:.17e}; all_actual={values:?} all_expected={expected:?} atoms={:?}",
                    values[0],
                    expected[0],
                    values[0] - expected[0],
                    response_atoms(result)
                );
                assert!(
                    (values[1] - expected[1]).abs() < mean_tol,
                    "{class} accepted={accepted} curve[1]: actual={:.17} expected={:.17} delta={:.17e} tolerance={mean_tol:.17e}; all_actual={values:?} all_expected={expected:?} atoms={:?}",
                    values[1],
                    expected[1],
                    values[1] - expected[1],
                    response_atoms(result)
                );
                // Out-of-support grid: the result must say so rather than be read as an effect.
                assert_eq!(
                    format!("{:?}", result.response().unwrap().support.status),
                    "OutsideEmpiricalSupport",
                    "{class} grid [0, 1] lies outside the observed treatment support"
                );
            }
            // A one-shot class response executes its retained prepared plan, so the
            // fresh result reports the identification it reused from that plan.
            assert_eq!(cached_count(&fresh), 1);
            assert_eq!(cached_count(&click), 1);
            assert_eq!(cached_count(&refreshed), 1);
        }
    }
}

fn build_bayesian_response(
    data: &TabularData,
    graph: &ClassGraph,
    accepted: bool,
    query: ResponseQuery,
    pin: &serde_json::Value,
) -> Study {
    let bayes = &pin["bayesian"];
    let builder = Study::tabular(data.clone());
    let builder = match (graph, accepted) {
        (ClassGraph::Cpdag(cpdag), true) => builder.graph(AcceptedGraph::from(cpdag.clone())),
        (ClassGraph::Cpdag(cpdag), false) => builder.graph(cpdag.clone()),
        (ClassGraph::Pag(pag), true) => builder.graph(AcceptedGraph::from(pag.clone())),
        (ClassGraph::Pag(pag), false) => builder.graph(pag.clone()),
    };
    builder
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate()
                .n_draws(usize::try_from(bayes["n_draws"].as_u64().unwrap()).unwrap())
                .prior_scale(bayes["prior_scale"].as_f64().unwrap()),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

#[test]
fn class_aware_bayesian_intervention_pins_against_ate_envelope() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let bayes = &pin["bayesian"];
    let diagnostic = bayes["diagnostic"].as_str().unwrap();
    let seed = bayes["seed"].as_u64().unwrap();
    let graphs = [
        ("cpdag", ClassGraph::Cpdag(cpdag_from_pin(&pin)), "cpdag_ate_contrast"),
        ("pag", ClassGraph::Pag(pag_from_pin(&pin)), "pag_ate_contrast"),
    ];
    for (class, graph, contrast_key) in graphs {
        if let ClassGraph::Pag(pag) = &graph {
            let (t, y) = ids(&pin);
            let env = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
                .identify_pag_envelope(pag, &AverageEffectQuery::binary_ate(t, y))
                .unwrap();
            assert_eq!(env.identified_weight.0, 0.0);
            continue;
        }
        // `cpdag_ate_contrast` is the same fixture value pinned in
        // conformance/estimate/cpdag_ate_envelope/expected.json's `bayesian.expected_ate`; see
        // cpdag_ate_numeric_pins.rs::cpdag_ate_envelope_numeric_pin for why it moved to
        // 0.4616051232275179 (rank-coupled Bayesian mixture correlation shift from 6890a67, not
        // a change in the frequentist point estimate, which still reproduces the fixture
        // bit-for-bit) and why the sibling closed-form test still validates it.
        let expected = if contrast_key == "cpdag_ate_contrast" {
            0.461_605_123_227_517_9
        } else {
            bayes[contrast_key].as_f64().unwrap()
        };
        let tol = if contrast_key == "cpdag_ate_contrast" {
            1e-9
        } else {
            bayes["contrast_tolerance"].as_f64().unwrap()
        };
        for accepted in [false, true] {
            let high = build_bayesian_response(
                &data,
                &graph,
                accepted,
                intervention_query(&pin, 1.0),
                &pin,
            );
            let (ctx, sink) = recording_ctx(seed);
            let fresh = high.clone().run(&ctx).unwrap();
            assert_eq!(identify_computations(&sink), 1);
            let prepared: PreparedStudy = high.prepare(&ctx).unwrap();
            let click = prepared.estimate(&data, &ctx).unwrap();
            // A one-shot class response executes its retained prepared plan, so the
            // fresh result reports the identification it reused from that plan.
            assert_eq!(cached_count(&fresh), 1);
            assert_eq!(cached_count(&click), 1);
            assert_eq!(fresh.logical_plan.estimator.as_deref(), Some("response.bayesian"));
            for result in [&fresh, &click] {
                assert!(
                    result.diagnostics.iter().any(|d| d.code.as_ref() == diagnostic),
                    "{class} must disclose {diagnostic}"
                );
                let hi = response_values(result)[0];
                let low = build_bayesian_response(
                    &data,
                    &graph,
                    accepted,
                    intervention_query(&pin, 0.0),
                    &pin,
                )
                .run(&ExecutionContext::for_tests(seed))
                .unwrap();
                let lo = response_values(&low)[0];
                assert!(
                    (hi - lo - expected).abs() < tol,
                    "{class} Bayesian contrast {} vs {expected}",
                    hi - lo
                );
                // Independent anchor: the equal-mass mixture of the adjusted (0.4) and the
                // unadjusted (0.52) completions of the hand-derived counts is 0.46.
                assert!((hi - lo - 0.46).abs() < 5e-3, "{class} contrast {} vs 0.46", hi - lo);
            }
        }
    }
}

/// Bayesian `ResponseCurve × Cpdag` on explicit and accepted structure: the
/// same MEC envelope evaluated pointwise with `response.bayesian` on the
/// discrete table. The two-point `[0, 1]` curve contrast must equal the
/// fixture's CPDAG Bayesian ATE contrast, per-completion draws stay unmixed
/// (the disclosure diagnostic fires), and the prepared click reuses the frozen
/// envelope.
#[test]
fn class_aware_bayesian_curve_pins_against_ate_envelope() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let bayes = &pin["bayesian"];
    let diagnostic = bayes["diagnostic"].as_str().unwrap();
    let seed = bayes["seed"].as_u64().unwrap();
    // See cpdag_ate_numeric_pins.rs::cpdag_ate_envelope_numeric_pin and the sibling
    // `class_aware_bayesian_intervention_pins_against_ate_envelope` test above for why the
    // fixture's stale `cpdag_ate_contrast` (0.4615641945101853) is re-pinned here.
    let expected = 0.461_605_123_227_517_9;
    let tol = 1e-9;
    let section = &pin["cpdag"];
    let graph = ClassGraph::Cpdag(cpdag_from_pin(&pin));
    for accepted in [false, true] {
        let study = build_bayesian_response(&data, &graph, accepted, curve_query(&pin), &pin);
        let (ctx, sink) = recording_ctx(seed);
        let fresh = study.clone().run(&ctx).unwrap();
        assert_eq!(identify_computations(&sink), 1);
        let prepared: PreparedStudy = study.prepare(&ctx).unwrap();
        let click = prepared.estimate(&data, &ctx).unwrap();
        assert_eq!(identify_computations(&sink), 2, "the click must reuse the envelope");
        // A one-shot class response executes its retained prepared plan, so the
        // fresh result reports the identification it reused from that plan.
        assert_eq!(cached_count(&fresh), 1);
        assert_eq!(cached_count(&click), 1);
        assert_eq!(fresh.logical_plan.estimator.as_deref(), Some("response.bayesian"));
        for result in [&fresh, &click] {
            assert_eq!(result.support_status().unwrap().as_str(), "licensed");
            assert_eq!(format!("{:?}", result.identification.status), "PartiallyIdentified");
            assert_envelope_diagnostic(result, section, "cpdag");
            assert!(
                result.diagnostics.iter().any(|d| d.code.as_ref() == diagnostic),
                "accepted={accepted}: must disclose {diagnostic}"
            );
            let values = response_values(result);
            assert_eq!(values.len(), 2);
            assert!(
                (values[1] - values[0] - expected).abs() < tol,
                "accepted={accepted}: Bayesian curve contrast {} vs {expected}",
                values[1] - values[0]
            );
            assert!(
                (values[1] - values[0] - 0.46).abs() < 5e-3,
                "accepted={accepted}: contrast {} vs the count-derived 0.46",
                values[1] - values[0]
            );
        }
    }
}

/// Bind `data` on a new branch, materialize and estimate it, then adopt the branch as the handle.
/// A failure leaves the handle unchanged, as the immutable lifecycle does.
fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}

fn dag_from_pin() -> antecedent_graph::Dag {
    let mut dag = antecedent_graph::Dag::with_variables(3);
    // columns are [t, y, z]: z -> t, z -> y, t -> y.
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag
}

fn curve_on(pin: &serde_json::Value, grid: &[f64]) -> ResponseQuery {
    let (treatment, outcome) = ids(pin);
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome,
        treatment: ContinuousDomain::new(treatment, GridSpec::Values(grid.to_vec().into())),
    })
}

fn raw_columns(data: &TabularData) -> [Vec<f64>; 3] {
    let column = |index: usize| match &data.storage().columns()[index] {
        antecedent_data::OwnedColumn::Float64(column) => column.values.as_slice().to_vec(),
        _ => panic!("fixture columns are Float64"),
    };
    [column(0), column(1), column(2)]
}

/// Coefficient of the first regressor in `y ~ 1 + regressors` by normal equations
/// with Gauss-Jordan elimination: an independent calculation of a completion's
/// regression-adjustment effect.
fn ols_first_coefficient(y: &[f64], regressors: &[&[f64]]) -> f64 {
    let p = regressors.len() + 1;
    let column = |j: usize, i: usize| if j == 0 { 1.0 } else { regressors[j - 1][i] };
    let mut m = vec![vec![0.0; p + 1]; p];
    for i in 0..y.len() {
        for a in 0..p {
            for b in 0..p {
                m[a][b] += column(a, i) * column(b, i);
            }
            m[a][p] += column(a, i) * y[i];
        }
    }
    for pivot in 0..p {
        let best = (pivot..p).max_by(|&a, &b| m[a][pivot].abs().total_cmp(&m[b][pivot].abs()));
        m.swap(pivot, best.unwrap());
        let d = m[pivot][pivot];
        for c in 0..=p {
            m[pivot][c] /= d;
        }
        for r in 0..p {
            if r != pivot {
                let f = m[r][pivot];
                for c in 0..=p {
                    m[r][c] -= f * m[pivot][c];
                }
            }
        }
    }
    m[1][p]
}

fn surface_mean(value: &Option<ResponseValue>) -> Vec<f64> {
    match value {
        Some(ResponseValue::Surface { mean, .. }) => mean.to_vec(),
        other => panic!("expected a surface, got {other:?}"),
    }
}

/// The two completion curves of the class, ordered `[unadjusted, adjusted]` by their
/// first level (the unadjusted completion absorbs the confounding of `z`).
fn completion_curves(result: &antecedent::StudyResult) -> [Vec<f64>; 2] {
    let mixture = result.structural_response.as_ref().expect("the class publishes its completions");
    let mut curves: Vec<Vec<f64>> =
        mixture.atoms.iter().map(|atom| surface_mean(&atom.value)).collect();
    assert_eq!(curves.len(), 2);
    curves.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let adjusted = curves.pop().unwrap();
    let unadjusted = curves.pop().unwrap();
    [unadjusted, adjusted]
}

/// Closed-form `E[Y | T = a]` of the noiseless fixture: `z` equals 1 exactly when
/// `t > 0.5` (checked on the data in the test), so the unadjusted completion's
/// target is `0.10 + 0.40 a + 0.20 * 1[a > 0.5]`.
fn unadjusted_truth(a: f64) -> f64 {
    0.10 + 0.40 * a + if a > 0.5 { 0.20 } else { 0.0 }
}

/// (a) A correctly specified, unambiguous linear case recovers the generating effect.
#[test]
fn class_aware_unambiguous_linear_case_recovers_generating_effect() {
    let pin = pin();
    let data = continuous_linear(&pin);
    let (t, y) = ids(&pin);
    let ctx = ExecutionContext::for_tests(1);
    let run = |query: antecedent_core::CausalQuery| {
        Study::tabular(data.clone())
            .graph(AcceptedGraph::dag(dag_from_pin()))
            .query(query)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap()
    };
    // AverageEffect: the generating coefficient 0.4, to numerical precision.
    let ate = run(antecedent_core::CausalQuery::from(AverageEffectQuery::binary_ate(t, y)));
    assert!((ate.estimate().as_effect().unwrap().ate - 0.4).abs() < 1e-9);
    // Curve on an in-support grid: the generating levels E[Y|do(t)] = 0.1 + 0.4 t + 0.2 E[z]
    // with E[z] = 0.5. The Kennedy curve carries penalized-nuisance and local-quadratic
    // smoothing error, observed at 1.8e-3 per level here.
    let curve = run(antecedent_core::CausalQuery::from(curve_on(&pin, &[0.2, 0.8])));
    let response = curve.response().unwrap();
    assert_eq!(format!("{:?}", response.support.status), "Supported");
    assert!(matches!(response.estimate, ResponseIdentification::PointIdentified(_)));
    let levels = response_values(&curve);
    for (level, truth) in levels.iter().zip([0.28, 0.52]) {
        assert!((level - truth).abs() < 3e-3, "curve level {level} vs generating {truth}");
    }
    let slope = (levels[1] - levels[0]) / 0.6;
    assert!((slope - 0.4).abs() < 0.011, "curve secant slope {slope} vs generating 0.4");
    // Off the observed support [0.1, 0.9] the same curve is an extrapolation and says so.
    let extrapolated = run(antecedent_core::CausalQuery::from(curve_on(&pin, &[0.0, 1.0])));
    let support = &extrapolated.response().unwrap().support;
    assert_eq!(format!("{:?}", support.status), "OutsideEmpiricalSupport");
    assert_eq!(
        format!("{:?}", support.point_status),
        "Some([OutsideEmpiricalSupport, OutsideEmpiricalSupport])"
    );
}

/// (b) An ambiguous class keeps each completion's answer and their disagreement.
#[test]
fn class_aware_ambiguous_curve_keeps_each_completions_answer() {
    let pin = pin();
    let data = continuous_linear(&pin);
    let [t_col, y_col, z_col] = raw_columns(&data);
    // z is a deterministic function of t: z = 1 iff t >= 0.5.
    let t_where = |level: f64| t_col.iter().zip(&z_col).filter(move |(_, z)| **z == level);
    let max_t_z0 = t_where(0.0).map(|(t, _)| *t).fold(f64::MIN, f64::max);
    let min_t_z1 = t_where(1.0).map(|(t, _)| *t).fold(f64::MAX, f64::min);
    assert!(max_t_z0 <= 0.5 + 1e-12 && min_t_z1 >= 0.5 - 1e-12);
    let graph = ClassGraph::Cpdag(cpdag_from_pin(&pin));
    let ctx = ExecutionContext::for_tests(1);

    // Curve completions on an in-support grid.
    let grid = [0.2, 0.8];
    let curve = build_response(&data, &graph, false, curve_on(&pin, &grid)).run(&ctx).unwrap();
    assert_eq!(format!("{:?}", curve.response().unwrap().support.status), "Supported");
    let [unadjusted, adjusted] = completion_curves(&curve);
    for (index, a) in grid.iter().enumerate() {
        let generating = 0.1 + 0.4 * a + 0.2 * 0.5;
        assert!(
            (adjusted[index] - generating).abs() < 3e-3,
            "adjusted {} vs {generating}",
            adjusted[index]
        );
        assert!(
            (unadjusted[index] - unadjusted_truth(*a)).abs() < 1e-3,
            "unadjusted {} vs E[Y|T={a}] {}",
            unadjusted[index],
            unadjusted_truth(*a)
        );
        assert!(
            (adjusted[index] - unadjusted[index]).abs() > 0.09,
            "the completions disagree at {a} and the disagreement is preserved"
        );
    }
    // The published answer is the pointwise envelope of the two curves, not one curve.
    let ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(envelope)) =
        &curve.response().unwrap().estimate
    else {
        panic!("a disagreeing class publishes a partially identified envelope");
    };
    for index in 0..grid.len() {
        assert_eq!(envelope.lower[index], unadjusted[index].min(adjusted[index]));
        assert_eq!(envelope.upper[index], unadjusted[index].max(adjusted[index]));
    }

    // AverageEffect completions: adjusted 0.4, unadjusted the OLS slope of y on t.
    let (treatment, outcome) = ids(&pin);
    let ate = Study::tabular(data.clone())
        .graph(cpdag_from_pin(&pin))
        .query(AverageEffectQuery::binary_ate(treatment, outcome))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let mut atoms: Vec<f64> = ate
        .structural_response
        .as_ref()
        .unwrap()
        .atoms
        .iter()
        .map(|atom| match atom.value {
            Some(ResponseValue::Scalar(value)) => value,
            ref other => panic!("{other:?}"),
        })
        .collect();
    atoms.sort_by(f64::total_cmp);
    let slope_adjusted = ols_first_coefficient(&y_col, &[&t_col, &z_col]);
    let slope_unadjusted = ols_first_coefficient(&y_col, &[&t_col]);
    assert!((slope_adjusted - 0.4).abs() < 1e-9);
    assert!((atoms[0] - slope_adjusted).abs() < 1e-9);
    assert!((atoms[1] - slope_unadjusted).abs() < 1e-9);
    assert!(slope_unadjusted > 0.7, "the unadjusted slope absorbs z: {slope_unadjusted}");
}

/// (c) The smoothed curve on the out-of-support grid `[0, 1]`: the gap between its
/// contrast (0.499) and `AverageEffect` (0.569) is explained, and the extrapolation
/// is asserted through the support flag rather than through a contrast tolerance.
#[test]
fn class_aware_curve_gap_to_average_effect_is_explained_and_flagged_as_extrapolation() {
    let pin = pin();
    let data = continuous_linear(&pin);
    let [t_col, y_col, z_col] = raw_columns(&data);
    let graph = ClassGraph::Cpdag(cpdag_from_pin(&pin));
    let ctx = ExecutionContext::for_tests(1);
    let (treatment, outcome) = ids(&pin);

    let query = curve_on(&pin, &[0.0, 1.0]);
    let curve = build_response(&data, &graph, false, query).run(&ctx).unwrap();
    // Grid points 0 and 1 lie outside the observed treatment support [0.1, 0.9]:
    // extrapolation, reported as such per coordinate.
    let lo_t = t_col.iter().copied().fold(f64::MAX, f64::min);
    let hi_t = t_col.iter().copied().fold(f64::MIN, f64::max);
    assert!(0.0 < lo_t && hi_t < 1.0);
    let support = &curve.response().unwrap().support;
    assert_eq!(format!("{:?}", support.status), "OutsideEmpiricalSupport");
    assert_eq!(
        format!("{:?}", support.point_status),
        "Some([OutsideEmpiricalSupport, OutsideEmpiricalSupport])"
    );
    assert!(
        support.warnings.iter().any(|d| d.code.as_ref() == "response.outside_empirical_support")
    );

    let [unadjusted, adjusted] = completion_curves(&curve);
    // Adjusted completion: the line 0.2 + 0.4 t, up to smoothing error (about 1e-3).
    assert!((adjusted[0] - 0.2).abs() < 2e-3 && (adjusted[1] - 0.6).abs() < 2e-3);
    // Unadjusted completion: the boundary lines of E[Y|T=t] extended to 0 and 1
    // (0.10 + 0.40 * 0 on the z = 0 branch, 0.10 + 0.40 * 1 + 0.20 on the z = 1 branch).
    assert!((unadjusted[0] - 0.1).abs() < 1e-6 && (unadjusted[1] - 0.7).abs() < 1e-6);
    let curve_contrast = (unadjusted[1] - unadjusted[0] + adjusted[1] - adjusted[0]) / 2.0;
    let published = response_values(&curve);
    assert!((published[1] - published[0] - curve_contrast).abs() < 1e-12);

    let ate = build_ate_study(&data, &pin, treatment, outcome).run(&ctx).unwrap();
    let ate_value = ate.estimate().as_effect().unwrap().ate;
    let slope_unadjusted = ols_first_coefficient(&y_col, &[&t_col]);
    let slope_adjusted = ols_first_coefficient(&y_col, &[&t_col, &z_col]);
    assert!((ate_value - (slope_unadjusted + slope_adjusted) / 2.0).abs() < 1e-9);
    // The whole gap is accounted for completion by completion.
    let gap = ate_value - curve_contrast;
    let unadjusted_gap = (slope_unadjusted - (unadjusted[1] - unadjusted[0])) / 2.0;
    let adjusted_gap = (slope_adjusted - (adjusted[1] - adjusted[0])) / 2.0;
    assert!((gap - (unadjusted_gap + adjusted_gap)).abs() < 1e-9);
    assert!(adjusted_gap.abs() < 2e-3, "adjusted completion: smoothing only, {adjusted_gap}");
    assert!(unadjusted_gap > 0.06, "unadjusted completion carries the gap: {unadjusted_gap}");
    // That gap is a difference of functionals, not an estimation error: the unadjusted
    // target E[Y|T=t] is a line plus a step at 0.5, so its OLS slope (0.737) is a
    // linear-projection weight, while the curve contrast between points on its two
    // boundary lines is 0.6. Even inside the support the secant over [0.1, 0.9] is
    // 0.65, still below the projection slope.
    let query = curve_on(&pin, &[0.1, 0.9]);
    let inside = build_response(&data, &graph, false, query).run(&ctx).unwrap();
    let [inside_unadjusted, _] = completion_curves(&inside);
    let secant = (inside_unadjusted[1] - inside_unadjusted[0]) / 0.8;
    assert!((secant - 0.65).abs() < 1e-3, "secant {secant}");
    assert!(slope_unadjusted - secant > 0.08);
}

fn build_ate_study(
    data: &TabularData,
    pin: &serde_json::Value,
    t: VariableId,
    y: VariableId,
) -> Study {
    Study::tabular(data.clone())
        .graph(cpdag_from_pin(pin))
        .query(AverageEffectQuery::binary_ate(t, y))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}
