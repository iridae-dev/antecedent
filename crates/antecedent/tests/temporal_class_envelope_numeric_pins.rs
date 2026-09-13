//! 1.4 numeric pins for licensed `TemporalCpdag` / `TemporalPag` Pulse cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec, Lag,
    MeasurementSpec, ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue,
    RoleHint, SmallRoleSet, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalCpdag, TemporalPag};

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

#[test]
fn temporal_class_response_returns_completion_identified_set() {
    let pin = pin();
    let data = series(&pin);
    let mut directed_cpdag = TemporalCpdag::empty();
    let treatment_cpdag = lagged(&mut directed_cpdag, 0, 1);
    let outcome_cpdag = lagged(&mut directed_cpdag, 1, 0);
    let witness_cpdag = lagged(&mut directed_cpdag, 2, 1);
    directed_cpdag.insert_directed(witness_cpdag, treatment_cpdag).unwrap();
    directed_cpdag.insert_directed(treatment_cpdag, outcome_cpdag).unwrap();
    let mut directed_pag = TemporalPag::empty();
    let treatment_pag = directed_pag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let outcome_pag =
        directed_pag.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let witness_pag = directed_pag.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    directed_pag.insert_directed(witness_pag, treatment_pag).unwrap();
    directed_pag.insert_directed(treatment_pag, outcome_pag).unwrap();
    let query = CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0])),
            ),
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1u32, 2], TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    );
    for graph in [ClassGraph::Cpdag(directed_cpdag), ClassGraph::Pag(directed_pag)] {
        for accepted in [false, true] {
            let result = build(&data, &graph, accepted, query.clone())
                .run(&ExecutionContext::for_tests(17))
                .unwrap();
            let structural = result.structural_response.as_ref().expect("completion metadata");
            assert_eq!(
                structural.weight_basis,
                antecedent::result::StructuralWeightBasis::CompletionEnumeration
            );
            assert!(structural.conditional_on_identified.is_none());
            let envelope = structural.identified_set.as_ref().expect("identified set");
            assert_eq!(envelope.dimension, 2);
            assert_eq!(envelope.lower.len(), 4);
            assert_eq!(envelope.upper.len(), 4);
            assert!(matches!(
                result.response.as_ref().unwrap().estimate,
                ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(_))
                    | ResponseIdentification::GraphDependent(_)
            ));
        }
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

fn identified_pag(pin: &serde_json::Value) -> TemporalPag {
    use antecedent_graph::{Endpoint, MarkedEdge, MiddleMark};
    let names: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mark = |m: &str| match m {
        "tail" => Endpoint::Tail,
        "arrow" => Endpoint::Arrow,
        "circle" => Endpoint::Circle,
        other => panic!("unknown endpoint {other}"),
    };
    let node = |g: &mut TemporalPag, name: &str, lag: &serde_json::Value| {
        let var = u32::try_from(names.iter().position(|c| *c == name).unwrap()).unwrap();
        let lag = Lag::from_raw(u32::try_from(lag.as_u64().unwrap()).unwrap());
        g.add_lagged(VariableId::from_raw(var), lag).unwrap()
    };
    let mut g = TemporalPag::empty();
    for edge in pin["marked_edges"].as_array().unwrap() {
        let a = node(&mut g, edge[0].as_str().unwrap(), &edge[1]);
        let b = node(&mut g, edge[2].as_str().unwrap(), &edge[3]);
        g.insert_marked(MarkedEdge {
            a,
            b,
            at_a: mark(edge[4].as_str().unwrap()),
            at_b: mark(edge[5].as_str().unwrap()),
            middle: MiddleMark::Empty,
        })
        .unwrap();
    }
    g
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
