//! Numeric pins for licensed `TemporalCpdag` / `TemporalPag` Pulse cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines)]
#![allow(
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

// The pinned temporal-PAG structure, in one owner. This file keeps its own
// series builder: it stamps an explicit schema and time index rather than
// taking `from_f64_columns`'s defaults, so it is a different construction, not
// a copy of `pinned_pag_series`.
use common::fixtures::pinned_pag as identified_pag;

use std::sync::Arc;

use antecedent::{AcceptedGraph, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec, Intervention,
    Lag, MeasurementSpec, MechanismOverride, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseValue, RoleHint, SmallRoleSet, TemporalEffectQuery, TemporalPolicy,
    TemporalResponseSpec, Value, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalCpdag, TemporalPag};
use common::lagged_ols::{Eval, lagged_ols_level};

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_class_envelope/expected.json"
    ))
    .unwrap()
}

fn series(pin: &serde_json::Value) -> TimeSeriesData {
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        z[i] = if i % 2 == 0 { 0.0 } else { 1.0 };
        t[i] = 0.3 + 0.4 * z[i] + 0.05 * ((i as f64) * 0.017).sin();
        if i > 0 {
            y[i] = 1.0 + 2.0 * t[i - 1] + 0.5 * z[i - 1];
        }
    }
    let mut builder = CausalSchemaBuilder::new();
    for name in ["t", "y", "z"] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn pulse_query(pin: &serde_json::Value) -> CausalQuery {
    let mut q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
    q.policy = TemporalPolicy::pulse(-1);
    q.horizon_steps = u32::try_from(pin["query"]["horizon_steps"].as_u64().unwrap()).unwrap();
    CausalQuery::TemporalEffect(q)
}

fn sustained_query(pin: &serde_json::Value) -> CausalQuery {
    let mut q =
        TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), 0, 1.0);
    q.policy = TemporalPolicy::sustained(-1, -1);
    q.horizon_steps = u32::try_from(pin["query"]["horizon_steps"].as_u64().unwrap()).unwrap();
    CausalQuery::TemporalEffect(q)
}

fn lagged(g: &mut TemporalCpdag, var: u32, lag: u32) -> antecedent_graph::DenseNodeId {
    g.add_lagged(VariableId::from_raw(var), Lag::from_raw(lag)).unwrap()
}

fn cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = lagged(&mut g, 0, 1);
    let y0 = lagged(&mut g, 1, 0);
    let z1 = lagged(&mut g, 2, 1);
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_undirected(z1, t1).unwrap();
    g
}

fn pag() -> TemporalPag {
    let mut g = TemporalPag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_circle_circle_with_middle(z1, t1, antecedent_graph::MiddleMark::Empty).unwrap();
    g
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

enum ClassGraph {
    Cpdag(TemporalCpdag),
    Pag(TemporalPag),
}

fn build(data: &TimeSeriesData, graph: &ClassGraph, accepted: bool, query: CausalQuery) -> Study {
    let builder = Study::series(data.clone());
    let builder = match (graph, accepted) {
        (ClassGraph::Cpdag(g), true) => builder.graph(AcceptedGraph::from(g.clone())),
        (ClassGraph::Cpdag(g), false) => builder.graph(g.clone()),
        (ClassGraph::Pag(g), true) => builder.graph(AcceptedGraph::from(g.clone())),
        (ClassGraph::Pag(g), false) => builder.graph(g.clone()),
    };
    builder.query(query).refute(RefuteSuite::None).bootstrap_replicates(0).build().unwrap()
}

#[test]
fn temporal_class_pulse_pins_and_reuses_envelope() {
    let pin = pin();
    let data = series(&pin);
    let query = pulse_query(&pin);
    let graphs = [
        ("cpdag", ClassGraph::Cpdag(cpdag()), "identify.temporal_cpdag.envelope"),
        ("pag", ClassGraph::Pag(pag()), "identify.temporal_pag.envelope"),
    ];
    for (class, graph, diag) in graphs {
        if class == "pag" {
            for accepted in [false, true] {
                let error = build(&data, &graph, accepted, query.clone())
                    .run(&ExecutionContext::for_tests(1))
                    .unwrap_err();
                assert!(error.to_string().contains("no identified mass"));
            }
            continue;
        }
        let section = &pin[class];
        let expected = section["pulse"]["ate"].as_f64().unwrap();
        let tol = section["pulse"]["absolute_tolerance"].as_f64().unwrap();
        for accepted in [false, true] {
            let study = build(&data, &graph, accepted, query.clone());
            let (ctx, sink) = recording_ctx(1);
            let fresh = study.clone().run(&ctx).unwrap();
            assert_eq!(identify_computations(&sink), 1);
            let prepared: PreparedStudy = study.prepare(&ctx).unwrap();
            assert_eq!(identify_computations(&sink), 2);
            let click = prepared.estimate_series(&data, &ctx).unwrap();
            assert_eq!(identify_computations(&sink), 2);
            assert_eq!(
                fresh.logical_plan.identifier.as_deref(),
                Some(section["identification"]["identifier"].as_str().unwrap())
            );
            assert_eq!(
                fresh.logical_plan.estimator.as_deref(),
                Some(section["pulse"]["estimator"].as_str().unwrap())
            );
            for result in [&fresh, &click] {
                assert_eq!(format!("{:?}", result.identification.status), "PartiallyIdentified");
                assert!(
                    result.diagnostics.iter().any(|d| d.code.as_ref() == diag),
                    "{class} missing {diag}: {:?}",
                    result.diagnostics
                );
                assert!(
                    (result.estimate.ate - expected).abs() < tol,
                    "{class} ate {} vs {expected}",
                    result.estimate.ate
                );
            }
            assert!(fresh.diagnostics.iter().all(|d| d.code.as_ref() != "exec.identify.cached"));
            assert!(click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
            assert!(
                fresh.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.envelope.se_omits_between_atom_variance"
                }),
                "{class} with zero replicates must disclose envelope SE limitation"
            );
            let bootstrapped = match &graph {
                ClassGraph::Cpdag(g) => Study::series(data.clone()).graph(g.clone()),
                ClassGraph::Pag(g) => Study::series(data.clone()).graph(g.clone()),
            }
            .query(query.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(16)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(1))
            .unwrap();
            let se =
                bootstrapped.estimate.se_bootstrap.unwrap_or_else(|| panic!("{class} mixture SE"));
            assert!(se.is_finite() && se > 0.0, "{class} mixture SE {se}");
            assert!(
                bootstrapped.diagnostics.iter().any(|d| {
                    d.code.as_ref() == "estimate.temporal_class.frequentist.shared_block"
                }),
                "{class} must publish shared circular-block SE"
            );
            assert!(
                bootstrapped.diagnostics.iter().all(|d| {
                    d.code.as_ref() != "estimate.envelope.se_omits_between_atom_variance"
                }),
                "{class} must not omit between-atom variance when atoms align"
            );
            assert!((bootstrapped.estimate.ate - expected).abs() < tol);
        }
        let cheap = match &graph {
            ClassGraph::Cpdag(g) => Study::series(data.clone()).graph(g.clone()),
            ClassGraph::Pag(g) => Study::series(data.clone()).graph(g.clone()),
        }
        .query(query.clone())
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
        assert!(!cheap.refutations.is_empty(), "{class} licensed cheap must execute a refuter");
        assert!(
            cheap.diagnostics.iter().any(|d| d.code.as_ref() == "refute.envelope.effect_mixture"),
            "{class} cheap must mix refuters across temporal completions"
        );
        assert!((cheap.estimate.ate - expected).abs() < tol);
    }
}

/// Deterministic, non-degenerate class-response law: `r → {z, t}`, `z → t`,
/// `{t, z}@-1 → y`, with small aperiodic wiggles so every lagged design is full rank
/// and completions that do and do not adjust for `z` disagree.
fn class_response_columns(n: usize) -> Vec<Vec<f64>> {
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut r = vec![0.0; n];
    for i in 0..n {
        let s = i as f64;
        r[i] = (s * 0.29).sin();
        z[i] = 0.4 * r[i] + (s * 0.13).cos() + 0.3 * (s * 1.7).sin();
        t[i] = 0.3 + 0.5 * r[i] + 0.2 * z[i] + 0.4 * (s * 2.3).cos();
        if i > 0 {
            y[i] = 1.0 + 2.0 * t[i - 1] + 0.6 * z[i - 1] + 0.1 * (s * 0.77).sin();
        }
    }
    vec![t, y, z, r]
}

fn class_response_series(columns: &[Vec<f64>], with_r: bool) -> TimeSeriesData {
    let names = ["t", "y", "z", "r"];
    let used = if with_r { 4 } else { 3 };
    let named: Vec<(&str, &[f64])> = (0..used).map(|i| (names[i], columns[i].as_slice())).collect();
    TimeSeriesData::from_f64_columns(named, 1).unwrap()
}

/// `z@-1 — t@-1` with both → `y@0`: one completion adjusts for `z`, the other
/// treats `z` as a mediator, so the two completions' surfaces differ.
fn two_completion_cpdag() -> TemporalCpdag {
    cpdag()
}

/// Shielded `z@-1 o-o t@-1` with `r@-1 → {z, t}@-1` and both → `y@0`: directed MAG
/// completions identify, the latent `z ↔ t` completion does not.
fn mixed_id_pag() -> TemporalPag {
    let mut g = TemporalPag::empty();
    let t1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = g.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    let r1 = g.add_lagged(VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    g.insert_directed(r1, z1).unwrap();
    g.insert_directed(r1, t1).unwrap();
    g.insert_circle_circle_with_middle(z1, t1, antecedent_graph::MiddleMark::Empty).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g
}

fn class_response_queries() -> Vec<(&'static str, CausalQuery, Vec<Eval>)> {
    let temporal = || TemporalResponseSpec::new(vec![1u32, 2], TemporalPolicy::pulse(-1), None);
    vec![
        (
            "curve",
            CausalQuery::Response(
                ResponseQuery::new(ResponseFunctional::MeanCurve {
                    outcome: VariableId::from_raw(1),
                    treatment: ContinuousDomain::new(
                        VariableId::from_raw(0),
                        GridSpec::Values(Arc::from([0.0, 1.0])),
                    ),
                })
                .with_temporal(temporal().unwrap()),
            ),
            vec![Eval::Dose(0.0), Eval::Dose(1.0)],
        ),
        (
            "set",
            CausalQuery::Response(
                ResponseQuery::new(ResponseFunctional::InterventionResponse {
                    outcome: VariableId::from_raw(1),
                    interventions: Arc::from([Intervention::set(
                        VariableId::from_raw(0),
                        Value::f64(1.0),
                    )]),
                })
                .with_temporal(temporal().unwrap()),
            ),
            vec![Eval::Dose(1.0)],
        ),
        (
            "shift",
            CausalQuery::Response(
                ResponseQuery::new(ResponseFunctional::InterventionResponse {
                    outcome: VariableId::from_raw(1),
                    interventions: Arc::from([Intervention::soft(
                        VariableId::from_raw(0),
                        MechanismOverride::additive_shift(0.5),
                    )]),
                })
                .with_temporal(temporal().unwrap()),
            ),
            vec![Eval::Shift(0.5)],
        ),
    ]
}

/// The closed-form structural truth of this law's horizon-1 response
/// (`conformance/estimate/temporal_class_response_truth`): `(value per cell,
/// tolerance)` for the completion with the given adjustment set. The z-adjusting
/// completion identifies the interventional level `1 + 2 x + 0.6 mean(z) + mean(w)`; the
/// completion without `z` identifies the association, whose closed form is the
/// omitted-variable-bias line through the sample means.
fn horizon_one_truth(kind: &str, adjustment: &[(usize, i32)]) -> (Vec<f64>, f64) {
    let truth: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_class_response_truth/expected.json"
    ))
    .unwrap();
    let key = match adjustment {
        [] => "z_as_mediator",
        [(2, -1)] => "adjusting_z",
        other => panic!("no closed-form truth for adjustment set {other:?}"),
    };
    let values = truth["horizon_1"][key][kind]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    (values, truth["tolerance"][key].as_f64().unwrap())
}

/// `ResponseCurve` / `InterventionResponse` × `TemporalCpdag` / `TemporalPag` (Frequentist):
/// every identified completion atom equals an independent lag-aligned OLS
/// g-computation under that atom's adjustment set, the adjustment sets are the
/// graph-theoretic ones, and the published identified set is exactly the pointwise
/// min/max of those per-completion surfaces. At horizon 1 each atom also equals the
/// closed-form structural value of its completion (the interventional level for the
/// z-adjusting completion, the omitted-variable-bias association for the other), which
/// does not share the estimator's lag convention.
#[test]
fn temporal_class_response_returns_completion_identified_set() {
    let columns = class_response_columns(300);
    let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
    let z_at_origin = vec![(2usize, -1i32)];
    let cases = [
        (
            "cpdag",
            ClassGraph::Cpdag(two_completion_cpdag()),
            class_response_series(&columns, false),
            vec![vec![vec![], z_at_origin.clone()], vec![vec![]]],
        ),
        (
            "pag",
            ClassGraph::Pag(mixed_id_pag()),
            class_response_series(&columns, true),
            vec![vec![z_at_origin.clone()], vec![vec![]]],
        ),
    ];
    let horizons = [1u32, 2];
    for (class, graph, data, expected_adjustments) in &cases {
        for accepted in [false, true] {
            for (kind, query, evals) in class_response_queries() {
                let label = format!("{class}/{kind}/accepted={accepted}");
                let result = build(data, graph, accepted, query)
                    .run(&ExecutionContext::for_tests(17))
                    .unwrap();
                let structural = result.structural_response.as_ref().expect("completion metadata");
                assert_eq!(
                    structural.weight_basis,
                    antecedent::result::StructuralWeightBasis::CompletionEnumeration
                );
                assert!(structural.conditional_on_identified.is_none());
                let envelope = structural.identified_set.as_ref().expect("identified set");
                assert_eq!(envelope.dimension, if kind == "curve" { 2 } else { 1 });
                assert_eq!(envelope.lower.len(), evals.len() * horizons.len());
                assert!(matches!(
                    result.response.as_ref().unwrap().estimate,
                    ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(_))
                        | ResponseIdentification::GraphDependent(_)
                ));
                for (h_index, &horizon) in horizons.iter().enumerate() {
                    let mut seen = std::collections::BTreeSet::new();
                    let mut surfaces = Vec::new();
                    for atom in structural.atoms.iter().filter(|atom| {
                        usize::try_from(atom.graph_key >> 32).unwrap() == h_index
                            && atom.value.is_some()
                    }) {
                        let response = atom.response.as_ref().expect("atom response");
                        let adjustment: Vec<(usize, i32)> =
                            response.horizon_identification.as_ref().expect("atom identification")
                                [0]
                            .adjustment
                            .iter()
                            .map(|key| (key.variable.raw() as usize, key.offset))
                            .collect();
                        let expected: Vec<f64> = evals
                            .iter()
                            .map(|&eval| {
                                lagged_ols_level(&refs, 1, 0, -1, horizon, &adjustment, eval)
                            })
                            .collect();
                        let Some(ResponseValue::Surface { mean, .. }) = atom.value.as_ref() else {
                            panic!("{label}: atom value is not a surface");
                        };
                        for (got, want) in mean.iter().zip(&expected) {
                            assert!(
                                (got - want).abs() < 1e-8,
                                "{label} h={horizon} adj={adjustment:?}: atom {got} vs OLS {want}"
                            );
                        }
                        if horizon == 1 {
                            let (structural_values, tolerance) =
                                horizon_one_truth(kind, &adjustment);
                            assert_eq!(mean.len(), structural_values.len(), "{label}");
                            for (got, truth) in mean.iter().zip(&structural_values) {
                                assert!(
                                    (got - truth).abs() < tolerance,
                                    "{label} adj={adjustment:?}: atom {got} vs closed-form {truth} (tolerance {tolerance})"
                                );
                            }
                        }
                        seen.insert(adjustment);
                        surfaces.push(expected);
                    }
                    let want: std::collections::BTreeSet<_> =
                        expected_adjustments[h_index].iter().cloned().collect();
                    assert_eq!(seen, want, "{label} h={horizon}: completion adjustment sets");
                    for (cell, _) in evals.iter().enumerate() {
                        let lo = surfaces.iter().map(|s| s[cell]).fold(f64::INFINITY, f64::min);
                        let hi = surfaces.iter().map(|s| s[cell]).fold(f64::NEG_INFINITY, f64::max);
                        let at = cell * horizons.len() + h_index;
                        assert!((envelope.lower[at] - lo).abs() < 1e-8, "{label} lower[{at}]");
                        assert!((envelope.upper[at] - hi).abs() < 1e-8, "{label} upper[{at}]");
                    }
                }
                if *class == "cpdag" && kind != "shift" {
                    // The adjust / do-not-adjust completions genuinely disagree at h=1.
                    let at = (evals.len() - 1) * horizons.len();
                    assert!(
                        envelope.upper[at] - envelope.lower[at] > 0.1,
                        "{label}: degenerate set"
                    );
                }
            }
        }
    }
}

/// The identified set must contain the structural value, not only agree with an in-test
/// OLS that shares the estimator's lag convention. `class_response_columns` with noise
/// added to the outcome equation `y_t = 1 + 2 t_{t-1} + 0.6 z_{t-1} + 0.3 e_t`: at h = 1 the
/// level of `E[y | do(t@-1 := 1)]` is `1 + 2 + 0.6 mean(z)`. The completion that adjusts for
/// `z` estimates it within sampling error, so the set's lower end sits within a few
/// standard errors of it, and a set whose completions all share a shifted alignment
/// (both ends off the same way) is caught. The `z`-as-mediator completion is biased up.
#[test]
fn temporal_class_response_identified_set_contains_the_structural_level() {
    const N: usize = 300;
    let mut columns = class_response_columns(N);
    let mut noise = common::calibration::gaussian(0x20_2609);
    for i in 1..N {
        columns[1][i] = 1.0 + 2.0 * columns[0][i - 1] + 0.6 * columns[2][i - 1] + 0.3 * noise();
    }
    let mean_z = columns[2][..N - 1].iter().sum::<f64>() / (N - 1) as f64;
    let truth = 1.0 + 2.0 * 1.0 + 0.6 * mean_z;
    let (_, query, _) =
        class_response_queries().into_iter().find(|(kind, ..)| *kind == "set").unwrap();
    let cases = [
        (
            "cpdag",
            ClassGraph::Cpdag(two_completion_cpdag()),
            class_response_series(&columns, false),
        ),
        ("pag", ClassGraph::Pag(mixed_id_pag()), class_response_series(&columns, true)),
    ];
    for (class, graph, data) in &cases {
        let result =
            build(data, graph, false, query.clone()).run(&ExecutionContext::for_tests(17)).unwrap();
        let envelope = result
            .structural_response
            .as_ref()
            .and_then(|s| s.identified_set.as_ref())
            .expect("identified set");
        // Cell 0 is horizon 1 (cells are dose-major, horizons 1 and 2 within a dose).
        let (lower, upper) = (envelope.lower[0], envelope.upper[0]);
        assert!(
            lower - 0.2 <= truth && truth <= upper + 0.2,
            "{class}: identified set [{lower:.4}, {upper:.4}] misses the structural level {truth:.4}"
        );
        assert!(
            (lower - truth).abs() < 0.2,
            "{class}: the z-adjusting completion must estimate the level: {lower:.4} vs {truth:.4}"
        );
        assert!(upper >= lower);
    }
}

#[test]
fn temporal_class_single_step_sustained_matches_pulse() {
    let pin = pin();
    let data = series(&pin);
    let graphs = [("cpdag", ClassGraph::Cpdag(cpdag())), ("pag", ClassGraph::Pag(pag()))];
    for (class, graph) in graphs {
        if class == "pag" {
            for query in [pulse_query(&pin), sustained_query(&pin)] {
                let error = build(&data, &graph, false, query)
                    .run(&ExecutionContext::for_tests(1))
                    .unwrap_err();
                assert!(error.to_string().contains("no identified mass"));
            }
            continue;
        }
        let pulse = build(&data, &graph, false, pulse_query(&pin))
            .run(&ExecutionContext::for_tests(1))
            .unwrap()
            .estimate
            .ate;
        let sustained = build(&data, &graph, false, sustained_query(&pin))
            .run(&ExecutionContext::for_tests(1))
            .unwrap()
            .estimate
            .ate;
        assert!(
            (pulse - sustained).abs() < 1e-10,
            "{class} pulse {pulse} vs sustained {sustained}"
        );
    }
}

fn identified_pag_pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_class_envelope/identified_pag.json"
    ))
    .unwrap()
}

/// Deterministic series of `identified_pag.json` (`law`), rebuilt exactly by
/// `identified_pag_reference.py`.
#[allow(clippy::many_single_char_names)]
fn identified_pag_series(pin: &serde_json::Value) -> TimeSeriesData {
    let n = usize::try_from(pin["n"].as_u64().unwrap()).unwrap();
    let names: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut cols = vec![vec![0.0; n]; names.len()];
    let [t, y, z, m, v] = [0, 1, 2, 3, 4];
    for i in 0..n {
        let x = i as f64;
        cols[z][i] = (0.37 * x).sin() + 0.5 * (1.3 * x).cos();
        cols[t][i] = 0.6 * cols[z][i] + 0.8 * (0.23 * x + 0.4).sin();
        cols[v][i] = 0.5 * cols[t][i] + (0.41 * x).cos();
        cols[m][i] = 0.7 * cols[z][i] + 0.6 * (0.29 * x + 0.2).cos();
        if i > 0 {
            cols[y][i] = 1.0 + 2.0 * cols[t][i - 1] + 1.5 * cols[m][i - 1] + 0.3 * (0.53 * x).sin();
        }
    }
    let mut builder = CausalSchemaBuilder::new();
    for name in &names {
        builder
            .add_variable(
                *name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns = cols
        .into_iter()
        .enumerate()
        .map(|(k, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(k).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

/// R-10: positive multi-completion `TemporalPag` evidence for Frequentist Pulse
/// and single-step Sustained. Six of seven stationary MAG completions identify
/// at two different lag-1 effects (adjust `z@-1`: 1.96; adjust nothing: 2.72);
/// the reported point is their equal-weight mixture, pinned against the numpy
/// reference. With replicates the mixture SE is the shared circular-block SE
/// (every completion refit on the same resample), finite and positive.
#[test]
fn temporal_pag_identified_multi_completion_pulse_and_sustained() {
    let pin = identified_pag_pin();
    let data = identified_pag_series(&pin);
    let pag = identified_pag(&pin);
    let expected = pin["pulse_ate"].as_f64().unwrap();
    let tol = pin["absolute_tolerance"].as_f64().unwrap();
    let replicates = u32::try_from(pin["bootstrap_replicates"].as_u64().unwrap()).unwrap();
    let id = &pin["identification"];
    let masses = format!(
        "identified_mass={}, unidentified_mass={}",
        id["identified_mass"].as_f64().unwrap(),
        id["unidentified_mass"].as_f64().unwrap()
    );
    let horizon = u32::try_from(pin["query"]["horizon_steps"].as_u64().unwrap()).unwrap();
    let mut pulse =
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
    pulse.policy = TemporalPolicy::pulse(-1);
    pulse.horizon_steps = horizon;
    let mut sustained =
        TemporalEffectQuery::sustained(VariableId::from_raw(0), VariableId::from_raw(1), 0, 1.0);
    sustained.policy = TemporalPolicy::sustained(-1, -1);
    sustained.horizon_steps = horizon;
    for (label, query) in [
        ("pulse", CausalQuery::TemporalEffect(pulse)),
        ("sustained", CausalQuery::TemporalEffect(sustained)),
    ] {
        for accepted in [false, true] {
            for (suite, boot) in [
                (RefuteSuite::None, 0),
                (RefuteSuite::None, replicates),
                (RefuteSuite::Cheap, replicates),
                (RefuteSuite::Full, 0),
            ] {
                let builder = Study::series(data.clone());
                let builder = if accepted {
                    builder.graph(AcceptedGraph::from(pag.clone()))
                } else {
                    builder.graph(pag.clone())
                };
                let study = builder
                    .query(query.clone())
                    .refute(suite)
                    .bootstrap_replicates(boot)
                    .build()
                    .unwrap();
                let (ctx, sink) = recording_ctx(1);
                let fresh = study.clone().run(&ctx).unwrap();
                let prepared: PreparedStudy = study.prepare(&ctx).unwrap();
                let click = prepared.estimate_series(&data, &ctx).unwrap();
                assert_eq!(identify_computations(&sink), 2, "click must reuse the envelope");
                assert!(
                    click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached")
                );
                for result in [&fresh, &click] {
                    let tag = format!("{label} accepted={accepted} {suite:?} boot={boot}");
                    assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{tag}");
                    assert_eq!(
                        format!("{:?}", result.identification.status),
                        id["status"].as_str().unwrap(),
                        "{tag}"
                    );
                    assert!(
                        result.diagnostics.iter().any(|d| {
                            d.code.as_ref() == "identify.temporal_pag.envelope"
                                && d.message.contains(&masses)
                        }),
                        "{tag}: envelope masses {masses}: {:?}",
                        result.diagnostics
                    );
                    assert!(
                        (result.estimate.ate - expected).abs() < tol,
                        "{tag}: mixture {} vs numpy reference {expected}",
                        result.estimate.ate
                    );
                    let shared_block = result.diagnostics.iter().any(|d| {
                        d.code.as_ref() == "estimate.temporal_class.frequentist.shared_block"
                    });
                    let omitted = result.diagnostics.iter().any(|d| {
                        d.code.as_ref() == "estimate.envelope.se_omits_between_atom_variance"
                    });
                    if boot > 0 {
                        let se = result.estimate.se_bootstrap.expect("shared-block mixture SE");
                        assert!(se.is_finite() && se > 0.0, "{tag}: SE {se}");
                        assert!(shared_block, "{tag}: shared circular-block diagnostic");
                        assert!(!omitted, "{tag}: finite SE must not claim omitted variance");
                    } else {
                        assert!(omitted, "{tag}: zero replicates must disclose the missing SE");
                    }
                    if suite == RefuteSuite::None {
                        assert!(result.refutations.is_empty(), "{tag}");
                    } else {
                        assert!(!result.refutations.is_empty(), "{tag}: refuters must run");
                        assert!(
                            result
                                .diagnostics
                                .iter()
                                .any(|d| { d.code.as_ref() == "refute.envelope.effect_mixture" }),
                            "{tag}: refuters mix across completions"
                        );
                    }
                }
                assert!((fresh.estimate.ate - click.estimate.ate).abs() < 1e-12);
            }
        }
    }
}

/// Every licensed Frequentist `TemporalCpdag` Pulse and single-step Sustained
/// coordinate (explicit and accepted structure × none / cheap / full) runs on the
/// pinned two-completion law and returns the pinned mixture. Cheap and full run
/// the refuters on each completion and mix them (`refute.envelope.effect_mixture`);
/// none runs no refuter.
#[test]
fn temporal_cpdag_frequentist_pulse_and_sustained_all_structures_and_suites() {
    let pin = pin();
    let data = series(&pin);
    let section = &pin["cpdag"];
    let expected = section["pulse"]["ate"].as_f64().unwrap();
    let tol = section["pulse"]["absolute_tolerance"].as_f64().unwrap();
    for (label, query) in [("pulse", pulse_query(&pin)), ("sustained", sustained_query(&pin))] {
        for accepted in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let builder = Study::series(data.clone());
                let builder = if accepted {
                    builder.graph(AcceptedGraph::from(cpdag()))
                } else {
                    builder.graph(cpdag())
                };
                let result = builder
                    .query(query.clone())
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(1))
                    .unwrap();
                let case = format!("{label} accepted={accepted} {suite:?}");
                assert_eq!(result.support_status.unwrap().as_str(), "licensed", "{case}");
                assert!(
                    (result.estimate.ate - expected).abs() < tol,
                    "{case}: ate {} vs {expected}",
                    result.estimate.ate
                );
                if suite == RefuteSuite::None {
                    assert!(result.refutations.is_empty(), "{case}: none runs no refuter");
                } else {
                    assert!(!result.refutations.is_empty(), "{case}: suite must execute");
                    assert!(
                        result
                            .diagnostics
                            .iter()
                            .any(|d| d.code.as_ref() == "refute.envelope.effect_mixture"),
                        "{case}: refuters must be mixed across completions"
                    );
                }
            }
        }
    }
}
