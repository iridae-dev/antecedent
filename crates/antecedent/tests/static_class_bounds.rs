//! A partially identified static class (CPDAG / PAG) answer carries its
//! identified set, and class responses export and reload.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// `t`, `y`, `z`, `x`, `g` are the variable names of the laws below.
#![allow(clippy::doc_markdown, clippy::float_cmp)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, PreparedStudy, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ClaimKind, ConditionalEffectQuery, ContinuousDomain,
    ExecutionContext, GridSpec, Intervention, ResponseFunctional, ResponseQuery, ResponseValue,
    Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, Dag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};
use antecedent_io::consume_analysis_result;

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn gaussian(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        let mut sum = 0.0;
        for _ in 0..12 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            sum += (state >> 11) as f64 / (1u64 << 53) as f64;
        }
        sum - 6.0
    }
}

/// `z — t`, `z -> y`, `t -> y`: two completions, `z -> t` (adjust `{z}`) and
/// `t -> z` (adjust nothing), whose ATEs differ by the confounding path.
fn two_completion_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(3);
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_undirected(d(2), d(0)).unwrap();
    g
}

/// The completion `z -> t` of [`two_completion_cpdag`].
fn completion_z_to_t() -> Dag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(2), d(0)).unwrap();
    g
}

/// The completion `t -> z` of [`two_completion_cpdag`].
fn completion_t_to_z() -> Dag {
    let mut g = Dag::with_variables(3);
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(0), d(2)).unwrap();
    g
}

/// `z ~ N(0,1)`, `t = 0.8 z + e`, `y = t + z + e` (columns `t, y, z`).
fn cpdag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = 0.8 * z[i] + g();
        y[i] = t[i] + z[i] + g();
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

/// The licensed-compiler static fixture shape: three smooth continuous columns.
fn static_like(n: usize) -> TabularData {
    let column = |j: usize| -> Vec<f64> {
        (0..n).map(|i| ((i + 3 * j) as f64 * 0.17).sin() + 0.25 * (i % 4) as f64).collect()
    };
    let (t, y, z) = (column(0), column(1), column(2));
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn ate_query() -> CausalQuery {
    CausalQuery::AverageEffect(AverageEffectQuery::with_levels(v(0), v(1), 0.0, 1.0))
}

fn prepare(
    data: &TabularData,
    graph: impl Into<antecedent::AcceptedGraph>,
    query: CausalQuery,
    inference: InferenceMode,
    ctx: &ExecutionContext,
) -> PreparedStudy {
    Study::tabular(data.clone())
        .graph(graph.into())
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
}

fn atom_values(result: &StudyResult) -> Vec<f64> {
    result
        .structural_response
        .as_ref()
        .expect("partially identified class result must carry its structural mixture")
        .atoms
        .iter()
        .filter_map(|atom| match atom.value {
            Some(ResponseValue::Scalar(value)) => Some(value),
            _ => None,
        })
        .collect()
}

fn identified_set(result: &StudyResult) -> (f64, f64) {
    let set = result
        .structural_response
        .as_ref()
        .expect("partially identified class result must carry its structural mixture")
        .identified_set
        .as_ref()
        .expect("a partially identified class answer must carry bounds");
    assert_eq!(set.dimension, 0, "a scalar identified set is coordinate-free");
    assert!(set.grid.is_empty());
    (set.lower[0], set.upper[0])
}

fn claim_kind(prepared: &PreparedStudy, result: &StudyResult, ctx: &ExecutionContext) -> ClaimKind {
    let contract = prepared.contract().unwrap();
    result.claim(&contract, ctx).unwrap().kind
}

fn round_trip(prepared: &PreparedStudy, result: &StudyResult, ctx: &ExecutionContext, id: &str) {
    let bytes = prepared
        .encode_contracted_result(result, id, ctx)
        .unwrap_or_else(|err| panic!("{id}: export failed: {err}"));
    let consumed =
        consume_analysis_result(&bytes).unwrap_or_else(|err| panic!("{id}: consume failed: {err}"));
    assert!(
        consumed.acceptance.accepts_as_verified_program(),
        "{id}: consume did not accept a verified program"
    );
}

#[test]
fn partially_identified_cpdag_ate_carries_the_completion_identified_set() {
    let ctx = ExecutionContext::for_tests(31);
    let data = cpdag_data(600, 4_021);
    let prepared =
        prepare(&data, two_completion_cpdag(), ate_query(), InferenceMode::Frequentist, &ctx);
    let result = prepared.estimate(&data, &ctx).unwrap();

    // The two completions, estimated on their own as ordinary DAGs.
    let per_completion: Vec<f64> = [completion_z_to_t(), completion_t_to_z()]
        .into_iter()
        .map(|dag| {
            let one = prepare(&data, dag, ate_query(), InferenceMode::Frequentist, &ctx);
            one.estimate(&data, &ctx).unwrap().estimate.ate
        })
        .collect();
    let lo = per_completion.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = per_completion.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(hi - lo > 0.1, "completions must disagree clearly: {per_completion:?}");

    assert_eq!(
        result.identification.status,
        antecedent_core::IdentificationStatus::PartiallyIdentified
    );
    assert_eq!(claim_kind(&prepared, &result, &ctx), ClaimKind::Bounds);
    let (set_lo, set_hi) = identified_set(&result);
    assert!((set_lo - lo).abs() < 1e-9, "lower bound {set_lo} is not the min completion {lo}");
    assert!((set_hi - hi).abs() < 1e-9, "upper bound {set_hi} is not the max completion {hi}");

    let mut values = atom_values(&result);
    values.sort_by(f64::total_cmp);
    assert_eq!(values.len(), 2, "both completions keep their own value");
    assert!((values[0] - lo).abs() < 1e-9 && (values[1] - hi).abs() < 1e-9);

    let mixture = result.structural_response.as_ref().unwrap();
    assert_eq!(mixture.weight_basis, antecedent::StructuralWeightBasis::CompletionEnumeration);
    assert_eq!(mixture.identified_mass, 1.0);
    assert_eq!(mixture.unidentified_mass, 0.0);
    assert_eq!(mixture.unevaluable_mass, 0.0);
    assert_eq!(mixture.subsampled_out_mass, 0.0);
    assert!(
        result.estimate.ate >= set_lo && result.estimate.ate <= set_hi,
        "the reported mixture must lie inside its identified set"
    );
    round_trip(&prepared, &result, &ctx, "cpdag-ate-partial");
}

#[test]
fn point_identified_cpdag_ate_still_reports_a_point() {
    let ctx = ExecutionContext::for_tests(32);
    let data = cpdag_data(600, 4_022);
    // Fully oriented: one completion, one adjustment set, one number.
    let oriented = Cpdag::from_dag(&completion_z_to_t());
    let prepared = prepare(&data, oriented, ate_query(), InferenceMode::Frequentist, &ctx);
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(
        result.identification.status,
        antecedent_core::IdentificationStatus::NonparametricallyIdentified
    );
    assert!(
        result.structural_response.is_none(),
        "a point-identified class answer is not an identified set"
    );
    assert_eq!(claim_kind(&prepared, &result, &ctx), ClaimKind::Point);
    assert!(result.estimate.ate.is_finite());
    round_trip(&prepared, &result, &ctx, "cpdag-ate-point");
}

#[test]
fn bayesian_partially_identified_cpdag_ate_carries_the_identified_set() {
    let ctx = ExecutionContext::for_tests(33);
    let data = cpdag_data(600, 4_023);
    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64));
    let prepared = prepare(&data, two_completion_cpdag(), ate_query(), inference, &ctx);
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(claim_kind(&prepared, &result, &ctx), ClaimKind::Bounds);
    let (lo, hi) = identified_set(&result);
    assert!(hi - lo > 0.1, "Bayesian completions must disagree clearly: [{lo}, {hi}]");
    let mut values = atom_values(&result);
    values.sort_by(f64::total_cmp);
    assert_eq!(values, vec![lo, hi], "the bounds are the per-completion posterior means");
    round_trip(&prepared, &result, &ctx, "cpdag-ate-partial-bayes");
}

/// `x` (a modifier of `t`'s effect on `y`) makes this a ConditionalEffect on the
/// same two-completion class.
fn conditional_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = 0.8 * z[i] + g();
        x[i] = f64::from(u8::from(i % 2 == 0));
        y[i] = t[i] + 0.8 * t[i] * x[i] + z[i] + g();
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap()
}

fn conditional_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(4);
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(3), d(1)).unwrap();
    g.insert_undirected(d(2), d(0)).unwrap();
    g
}

#[test]
fn partially_identified_cpdag_conditional_effect_carries_the_identified_set() {
    let ctx = ExecutionContext::for_tests(34);
    let data = conditional_data(600, 4_024);
    let query = CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::with_levels(v(0), v(1), 0.0, 1.0).with_effect_modifiers([v(3)]),
        )
        .unwrap(),
    );
    let prepared = prepare(&data, conditional_cpdag(), query, InferenceMode::Frequentist, &ctx);
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(claim_kind(&prepared, &result, &ctx), ClaimKind::Bounds);
    let (lo, hi) = identified_set(&result);
    assert!(hi - lo > 0.1, "conditional completions must disagree clearly: [{lo}, {hi}]");
    let mut values = atom_values(&result);
    values.sort_by(f64::total_cmp);
    assert_eq!(values, vec![lo, hi], "the bounds are the per-completion conditional effects");
    round_trip(&prepared, &result, &ctx, "cpdag-conditional-partial");
}

/// `z -> t -> y` as a PAG: `z` points into `t` and is not adjacent to `y`, so
/// `t -> y` is visible and the intervention response is identified by
/// generalized adjustment, on the Pag response owner. Without the witness
/// (`t -> y`, `z -> y`) the edge is invisible and the response is refused.
fn response_pag() -> Pag {
    let mut g = Pag::with_variables(3);
    g.insert_directed(d(2), d(0)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g
}

/// The identified multi-completion PAG of
/// `conformance/response/class_aware_envelope/pag_identified.json`
/// (`r o-> t`, `z o-> t`, `z -> y`, `t -> y`; columns `t, y, z, r`), whose
/// completions keep a dose curve on the adjustment envelope.
fn curve_pag() -> Pag {
    let mut g = Pag::with_variables(4);
    for (a, b) in [(3u32, 0u32), (2, 0)] {
        g.insert_marked(MarkedEdge {
            a: d(a),
            b: d(b),
            at_a: Endpoint::Circle,
            at_b: Endpoint::Arrow,
            middle: MiddleMark::Empty,
        })
        .unwrap();
    }
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g
}

/// The law of that fixture: `z = sin(0.37 i)`, `r = sin(0.53 i)`,
/// `t = 0.5 + 0.5 z + 0.4 r + 0.5 sin(0.61 i)`, `y = 0.1 + 0.4 t + 0.2 z`.
fn curve_data(n: usize) -> TabularData {
    let wave = |i: usize, freq: f64| (i as f64 * freq).sin();
    let z: Vec<f64> = (0..n).map(|i| wave(i, 0.37)).collect();
    let r: Vec<f64> = (0..n).map(|i| wave(i, 0.53)).collect();
    let t: Vec<f64> = (0..n).map(|i| 0.5 + 0.5 * z[i] + 0.4 * r[i] + 0.5 * wave(i, 0.61)).collect();
    let y: Vec<f64> =
        (0..n).map(|i| 0.1 + 0.4 * t[i] + 0.2 * z[i] + 0.02 * wave(i, 0.29)).collect();
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("r", r.as_slice()),
    ])
    .unwrap()
}

fn intervention_response_query() -> CausalQuery {
    CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: v(1),
        interventions: Arc::from([Intervention::set(v(0), Value::f64(1.0))]),
    }))
}

fn mean_curve_query() -> CausalQuery {
    CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
    }))
}

#[test]
fn class_responses_export_and_reload_on_cpdag_and_pag() {
    let ctx = ExecutionContext::for_tests(35);
    // `partial` records the identification each fixture is chosen for: the
    // CPDAG cases publish a completion identified set (the payload shape that
    // must survive the round trip), the PAG cases a point.
    let cases: Vec<(&str, CausalQuery, antecedent::AcceptedGraph, TabularData, bool)> = vec![
        (
            "intervention_response-cpdag",
            intervention_response_query(),
            antecedent::AcceptedGraph::from(two_completion_cpdag()),
            static_like(48),
            true,
        ),
        (
            "intervention_response-pag",
            intervention_response_query(),
            antecedent::AcceptedGraph::from(response_pag()),
            static_like(48),
            false,
        ),
        (
            "response_curve-cpdag",
            mean_curve_query(),
            antecedent::AcceptedGraph::from(two_completion_cpdag()),
            static_like(48),
            true,
        ),
        (
            "response_curve-pag",
            mean_curve_query(),
            antecedent::AcceptedGraph::from(curve_pag()),
            curve_data(200),
            false,
        ),
    ];
    for (name, query, graph, data, partial) in cases {
        for (inference_name, inference) in [
            ("frequentist", InferenceMode::Frequentist),
            ("bayesian", InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32))),
        ] {
            let id = format!("{name}-{inference_name}");
            let prepared = prepare(&data, graph.clone(), query.clone(), inference, &ctx);
            let result = prepared
                .estimate(&data, &ctx)
                .unwrap_or_else(|err| panic!("{id}: execute failed: {err}"));
            let response = result.response.as_ref().unwrap_or_else(|| panic!("{id}: no response"));
            assert_eq!(
                response.identification_status
                    == antecedent_core::IdentificationStatus::PartiallyIdentified,
                partial,
                "{id}: fixture no longer covers the identification it was chosen for"
            );
            round_trip(&prepared, &result, &ctx, &id);
        }
    }
}
