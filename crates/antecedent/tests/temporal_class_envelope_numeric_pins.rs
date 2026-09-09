//! 1.4 numeric pins for licensed `TemporalCpdag` / `TemporalPag` Pulse cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint,
    SmallRoleSet, TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
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
                "{class} must disclose envelope SE limitation"
            );
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
fn temporal_class_single_step_sustained_matches_pulse() {
    let pin = pin();
    let data = series(&pin);
    let graphs = [("cpdag", ClassGraph::Cpdag(cpdag())), ("pag", ClassGraph::Pag(pag()))];
    for (class, graph) in graphs {
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
