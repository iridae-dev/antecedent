//! 1.4 numeric pins for licensed Cpdag/Pag response cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

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
    let response = result.response.as_ref().expect("class-aware response payload");
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
    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    assert_eq!(identify_computations(&sink), 2);
    (fresh, click, refreshed)
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
        let do0 = section["intervention"]["do_0"].as_f64().unwrap();
        let do1 = section["intervention"]["do_1"].as_f64().unwrap();
        let tol = section["intervention"]["absolute_tolerance"].as_f64().unwrap();
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
                assert!((response_values(result)[0] - do1).abs() < tol, "{class} do(1)");
            }
            assert_eq!(cached_count(&fresh), 0);
            assert_eq!(cached_count(&click), 1);
            assert_eq!(cached_count(&refreshed), 1);

            let low = build_response(&data, &graph, accepted, intervention_query(&pin, 0.0))
                .run(&ExecutionContext::for_tests(1))
                .unwrap();
            let lo = response_values(&low)[0];
            let hi = response_values(&fresh)[0];
            assert!((lo - do0).abs() < tol, "{class} do(0) {lo}");
            assert!(
                (hi - lo - ate).abs()
                    < section["intervention"]["contrast_tolerance"].as_f64().unwrap(),
                "{class} contrast"
            );
        }
    }
}

#[test]
fn class_aware_curve_pins_against_ate_on_continuous_t() {
    let pin = pin();
    let data = continuous_linear(&pin);
    let (treatment, outcome) = ids(&pin);
    let ate_query = AverageEffectQuery::binary_ate(treatment, outcome);
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
        let contrast_tol = section["curve"]["contrast_tolerance"].as_f64().unwrap();
        let identifier = section["identification"]["identifier"].as_str().unwrap();
        for accepted in [false, true] {
            let ate = {
                let builder = Study::tabular(data.clone());
                let builder = match (&graph, accepted) {
                    (ClassGraph::Cpdag(cpdag), true) => {
                        builder.graph(AcceptedGraph::from(cpdag.clone()))
                    }
                    (ClassGraph::Cpdag(cpdag), false) => builder.graph(cpdag.clone()),
                    (ClassGraph::Pag(pag), true) => builder.graph(AcceptedGraph::from(pag.clone())),
                    (ClassGraph::Pag(pag), false) => builder.graph(pag.clone()),
                };
                builder
                    .query(ate_query.clone())
                    .refute(RefuteSuite::None)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(1))
                    .unwrap()
                    .estimate
                    .ate
            };
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
                    "{class} curve[0] {}",
                    values[0]
                );
                assert!(
                    (values[1] - expected[1]).abs() < mean_tol,
                    "{class} curve[1] {}",
                    values[1]
                );
                assert!(
                    (values[1] - values[0] - ate).abs() < contrast_tol,
                    "{class} curve contrast {} vs ate {ate}",
                    values[1] - values[0]
                );
            }
            assert_eq!(cached_count(&fresh), 0);
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
        let expected = bayes[contrast_key].as_f64().unwrap();
        let tol = bayes["contrast_tolerance"].as_f64().unwrap();
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
            assert_eq!(cached_count(&fresh), 0);
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
            }
        }
    }
}
