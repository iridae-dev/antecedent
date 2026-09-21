//! Numeric pins for licensed PAG and ADMG ATE cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, EstimatorId, IdentifierId, InferenceMode, PreparedStudy, RefuteSuite, Study,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    InterventionalDistributionQuery, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseValue, Value, VariableId,
};
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
fn invisible_pag_pin_is_not_identified_by_adjustment() {
    let pin = pag_pin();
    let pag = pag_from_pin(&pin);
    let query = query_from_pin(&pin);
    let envelope = antecedent_identify::GeneralizedAdjustmentIdentifier::new()
        .identify_pag_envelope(&pag, &query)
        .unwrap();
    assert_eq!(envelope.identified_weight.0, 0.0);
    assert_eq!(envelope.unidentified_weight.0, envelope.cases.len() as f64);
    assert!(envelope.cases.iter().all(|case| case.result.estimands.is_empty()));
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

fn response_mean(result: &antecedent::StudyResult) -> f64 {
    match result.response.as_ref().and_then(|r| match &r.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(v)) => Some(*v),
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            mean.first().copied()
        }
        _ => None,
    }) {
        Some(v) if v.is_finite() => v,
        _ => result.estimate.ate,
    }
}

fn surface_means(result: &antecedent::StudyResult) -> Vec<f64> {
    match result.response.as_ref().and_then(|r| match &r.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            Some(mean.to_vec())
        }
        _ => None,
    }) {
        Some(v) => v,
        None => vec![result.estimate.ate],
    }
}

/// Bidirected front-door ADMG: `InterventionResponse` / `ResponseCurve` means at
/// do(T=0) and do(T=1) match the licensed `InterventionalDistribution` / ATE
/// functional on the same graph.
#[test]
fn admg_frontdoor_response_pins_against_distribution() {
    let pin = admg_pin();
    let data = expand_contingency(&pin);
    let admg = admg_from_pin(&pin);
    let query = query_from_pin(&pin);
    let expected_ate = pin["frequentist"]["expected_ate"].as_f64().unwrap();
    let tolerance = pin["frequentist"]["absolute_tolerance"].as_f64().unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let t = query.treatment;
    let y = query.outcome;

    let ate = Study::tabular(data.clone())
        .graph(admg.clone())
        .query(query.clone())
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalEffect)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    assert!((ate.estimate.ate - expected_ate).abs() < tolerance);

    let mut dist_means = Vec::new();
    let mut ir_means = Vec::new();
    for level in [0.0, 1.0] {
        let dist_query =
            InterventionalDistributionQuery::new(y, [Intervention::set(t, Value::f64(level))]);
        let dist = Study::tabular(data.clone())
            .graph(admg.clone())
            .query(CausalQuery::Distribution(dist_query))
            .identifier(IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalDistribution)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        let dist_mean = dist.distribution.as_ref().expect("distribution").mean;
        dist_means.push(dist_mean);

        let ir = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([Intervention::set(t, Value::f64(level))]),
        });
        for accepted in [false, true] {
            for bayesian in [false, true] {
                let mut builder = Study::tabular(data.clone());
                builder = if accepted {
                    builder.graph(AcceptedGraph::from(admg.clone()))
                } else {
                    builder.graph(admg.clone())
                };
                let study = builder
                    .query(CausalQuery::Response(ir.clone()))
                    .identifier(IdentifierId::GeneralId)
                    .estimator(EstimatorId::FunctionalEffect)
                    .inference(if bayesian {
                        InferenceMode::Bayesian(antecedent::BayesianConfig::conjugate().n_draws(64))
                    } else {
                        InferenceMode::Frequentist
                    })
                    .refute(RefuteSuite::None)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap();
                let (fresh, click, _) = run_prepared(
                    &study,
                    &data,
                    1,
                    IdentifierId::GeneralId.as_str(),
                    EstimatorId::FunctionalEffect.as_str(),
                );
                for result in [&fresh, &click] {
                    assert!(
                        result
                            .diagnostics
                            .iter()
                            .any(|d| d.code.as_ref() == "identify.response.general_id"),
                        "accepted={accepted} bayesian={bayesian} level={level}"
                    );
                    let mean = response_mean(result);
                    let bound = if bayesian { 0.05 } else { 1e-9 };
                    assert!(
                        (mean - dist_mean).abs() < bound,
                        "IR mean {mean} != dist {dist_mean} at do(T={level}) accepted={accepted} bayesian={bayesian}"
                    );
                }
                if !bayesian && !accepted {
                    ir_means.push(response_mean(&fresh));
                }
            }
        }
    }
    assert!(((ir_means[1] - ir_means[0]) - expected_ate).abs() < tolerance);
    assert!(((dist_means[1] - dist_means[0]) - expected_ate).abs() < tolerance);

    let curve = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(t, GridSpec::Values(Arc::from([0.0, 1.0]))),
    });
    let curve_result = Study::tabular(data.clone())
        .graph(admg.clone())
        .query(CausalQuery::Response(curve))
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalEffect)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let means = surface_means(&curve_result);
    assert_eq!(means.len(), 2);
    assert!((means[0] - dist_means[0]).abs() < 1e-9);
    assert!((means[1] - dist_means[1]).abs() < 1e-9);
}

/// Exact 2048-row population table of the napkin SCM `W -> Z -> X -> Y`, `W <-> X` (latent
/// `U1`), `W <-> Y` (latent `U2`) with `P(U1=1) = 1/2`, `P(U2=1) = 1/4`,
/// `P(W=1|u1,u2) = (1 + u1 + u2)/4`, `P(Z=1|w) = (1 + 2w)/4`, `P(X=1|z,u1) = (1 + z + u1)/4`,
/// `P(Y=1|x,u2) = (1 + x + x·u2)/4`. Every joint mass is a multiple of `1/2048`, so the table
/// is the law itself. Returns the data (columns `w, z, x, y`) and `E[Y | do(X=x)]` for
/// `x = 0, 1`, enumerated from the mechanisms: `1/4` and `1/2 + P(U2=1)/4 = 9/16`.
#[allow(clippy::many_single_char_names, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn napkin_population() -> (TabularData, [f64; 2]) {
    let p = |one: bool, p1: f64| if one { p1 } else { 1.0 - p1 };
    let mut cols: [Vec<f64>; 4] = Default::default();
    let mut do_mean = [0.0_f64; 2];
    for bits in 0..64_u32 {
        let b = |k: u32| (bits >> k) & 1 == 1;
        let (u1, u2, w, z, x, y) = (b(0), b(1), b(2), b(3), b(4), b(5));
        let f = |v: bool| f64::from(u8::from(v));
        let latent = p(u1, 0.5) * p(u2, 0.25);
        let mass = latent
            * p(w, (1.0 + f(u1) + f(u2)) / 4.0)
            * p(z, (1.0 + 2.0 * f(w)) / 4.0)
            * p(x, (1.0 + f(z) + f(u1)) / 4.0)
            * p(y, (1.0 + f(x) + f(x) * f(u2)) / 4.0);
        let count = mass * 2048.0;
        assert!((count - count.round()).abs() < 1e-9, "mass {mass} is not a multiple of 1/2048");
        for (col, value) in cols.iter_mut().zip([w, z, x, y]) {
            col.extend(std::iter::repeat_n(f(value), count.round() as usize));
        }
    }
    // do(X = x) cuts X's mechanism: E[Y | do(x)] = sum_u2 P(u2) P(Y=1 | x, u2).
    for (x, slot) in do_mean.iter_mut().enumerate() {
        for u2 in [0.0, 1.0] {
            let pu2 = if u2 == 1.0 { 0.25 } else { 0.75 };
            *slot += pu2 * (1.0 + x as f64 + x as f64 * u2) / 4.0;
        }
    }
    assert_eq!(cols[0].len(), 2048);
    let pairs: Vec<(&str, &[f64])> =
        ["w", "z", "x", "y"].into_iter().zip(cols.iter().map(Vec::as_slice)).collect();
    (TabularData::from_f64_columns(pairs).unwrap(), do_mean)
}

fn napkin_admg() -> Admg {
    let mut admg = Admg::with_variables(4);
    for (a, b) in [(0, 1), (1, 2), (2, 3)] {
        admg.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(0, 2), (0, 3)] {
        admg.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    admg
}

fn free_variable_diagnostic(result: &antecedent::StudyResult) -> &antecedent_core::Diagnostic {
    result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "estimate.functional.free_variables_averaged")
        .expect("the result must say how the functional's free variable was resolved")
}

/// The napkin functional `sum_w P(x,y|z,w)P(w) / sum_w P(x|z,w)P(w)` keeps `z` free: it
/// equals `P(y | do(x))` at every supported `z`. An evaluator that sums over `z` returns
/// twice the effect here.
#[test]
fn napkin_effect_with_a_free_variable_is_the_enumerated_effect() {
    let (data, do_mean) = napkin_population();
    let truth = do_mean[1] - do_mean[0];
    assert!((truth - 0.3125).abs() < 1e-15);
    let x = VariableId::from_raw(2);
    let y = VariableId::from_raw(3);
    let ctx = ExecutionContext::for_tests(3);
    for bayesian in [false, true] {
        let mut builder = Study::tabular(data.clone())
            .graph(napkin_admg())
            .query(AverageEffectQuery::with_levels(x, y, 0.0, 1.0))
            .identifier(IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalEffect)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0);
        if bayesian {
            builder = builder.inference(InferenceMode::Bayesian(
                antecedent::BayesianConfig::conjugate().n_draws(256),
            ));
        }
        let result = builder.build().unwrap().run(&ctx).unwrap();
        if bayesian {
            // Posterior mean of a Bayesian bootstrap over 2048 rows.
            assert!((result.estimate.ate - truth).abs() < 0.05, "ate={}", result.estimate.ate);
        } else {
            assert!((result.estimate.ate - truth).abs() < 1e-12, "ate={}", result.estimate.ate);
        }
        let diagnostic = free_variable_diagnostic(&result);
        assert!(diagnostic.message.contains("variable 1"), "{}", diagnostic.message);
    }

    for (level, expected) in [(0.0, do_mean[0]), (1.0, do_mean[1])] {
        let query =
            InterventionalDistributionQuery::new(y, [Intervention::set(x, Value::f64(level))]);
        let result = Study::tabular(data.clone())
            .graph(napkin_admg())
            .query(CausalQuery::Distribution(query))
            .identifier(IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalDistribution)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        let distribution = result.distribution.as_ref().expect("distribution");
        assert!((distribution.mean - expected).abs() < 1e-12, "mean={}", distribution.mean);
        let total: f64 = distribution.atoms.iter().map(|a| a.probability).sum();
        assert!((total - 1.0).abs() < 1e-12, "atoms sum to {total}");
        free_variable_diagnostic(&result);

        let response = ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([Intervention::set(x, Value::f64(level))]),
        });
        let result = Study::tabular(data.clone())
            .graph(napkin_admg())
            .query(CausalQuery::Response(response))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        assert!((response_mean(&result) - expected).abs() < 1e-12, "{}", response_mean(&result));
    }
}
