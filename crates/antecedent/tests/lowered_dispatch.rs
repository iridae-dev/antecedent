//! Dispatch executes only the lowered program `Study::compile` attached.
//!
//! A physical plan compiled from the logical plan alone carries no lowered program
//! and is refused before any route runs. A context already inside a lowered
//! dispatch refuses to compile, prepare or run another study. A one-shot run
//! identifies once: the execution takes the identification the lowered program
//! retained at compile instead of identifying again.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex};

use antecedent::{
    AnomalyAttributionQuery, AverageEffectQuery, CausalQuery, Dag, DenseNodeId,
    RandomizedEffectQuery, RefuteSuite, Study, TemporalDag, TemporalEffectQuery, TemporalPolicy,
};
use antecedent_core::{ExecutionContext, Lag, ProgressSink, VariableId};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::ensure_lagged;

#[path = "fixtures_common/mod.rs"]
mod common;

use common::fixtures::{confounded_scm, mixture_graph_posterior};

#[derive(Default)]
struct RecordingProgress(Mutex<Vec<String>>);

impl ProgressSink for RecordingProgress {
    fn report(&self, _fraction: f64, stage: &str) {
        self.0.lock().unwrap().push(stage.to_owned());
    }
}

fn recording_ctx(seed: u64) -> (ExecutionContext, Arc<RecordingProgress>) {
    let sink = Arc::new(RecordingProgress::default());
    let mut ctx = ExecutionContext::for_tests(seed);
    ctx.progress = Some(Arc::clone(&sink) as Arc<dyn ProgressSink>);
    (ctx, sink)
}

fn identify_computations(sink: &RecordingProgress) -> usize {
    sink.0.lock().unwrap().iter().filter(|stage| stage.as_str() == "identify.compute").count()
}

fn static_effect_study() -> Study {
    let (data, dag, query) = confounded_scm(128, 41);
    Study::tabular(data)
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// `t -> y` and a binary confounder `z`, with the posterior over the three graphs
/// `mixture_graph_posterior` weighs.
fn mixture_data() -> TabularData {
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    let mut confounder = Vec::new();
    for _ in 0..10 {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -0.2 } else { 0.2 };
                treatment.push(t);
                confounder.push(z);
                outcome.push(2.0 * t + 2.0 * z + epsilon);
            }
        }
    }
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", confounder.as_slice()),
    ])
    .unwrap()
}

fn graph_posterior_study() -> Study {
    Study::tabular(mixture_data())
        .graph_posterior(mixture_graph_posterior())
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// `y_t = 0.8 x_{t-1}` plus a small periodic term, under `x_{t-1} -> y_t`.
fn temporal_effect_study() -> Study {
    let rows = 96;
    let mut x = vec![0.0; rows];
    let mut y = vec![0.0; rows];
    for t in 0..rows {
        let clock = t as f64;
        x[t] = (0.37 * clock).sin();
        if t > 0 {
            y[t] = 0.8 * x[t - 1] + 0.1 * (1.3 * clock).cos();
        }
    }
    let series =
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap();
    let (treatment, outcome) = (VariableId::from_raw(0), VariableId::from_raw(1));
    let mut graph = TemporalDag::empty();
    let lagged = ensure_lagged(&mut graph, treatment, Lag::from_raw(1)).unwrap();
    let current = ensure_lagged(&mut graph, outcome, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(lagged, current).unwrap();
    let query = TemporalEffectQuery::pulse(treatment, outcome, 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    Study::series(series)
        .graph(graph)
        .temporal_query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// `y = 2x` with one outlier, under `x -> y`.
fn structural_study() -> Study {
    let x: Vec<f64> = (0..20).map(f64::from).collect();
    let y: Vec<f64> = (0..20).map(|i| if i == 19 { 200.0 } else { 2.0 * f64::from(i) }).collect();
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
            [VariableId::from_raw(1)],
            100,
        )))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

/// A Bernoulli-randomized two-arm intention-to-treat effect.
fn design_study() -> Study {
    let n = 200;
    let assignment: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let outcomes: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * f64::from(u8::from(assignment[i])) + 0.5 * (i as f64 * 0.3).sin())
        .collect();
    let units: Vec<Arc<str>> = (0..n).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect();
    let query = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        assignment,
        vec![0.5; n],
        units.clone(),
        units,
        ("control", "treated"),
    );
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    Study::tabular(data).query(query).build().unwrap()
}

#[test]
fn dispatch_refuses_a_plan_without_a_lowered_program() {
    let ctx = ExecutionContext::for_tests(3);
    let study = static_effect_study();
    let plan = study.compile_logical().unwrap().compile_physical(&ctx).unwrap();
    assert!(plan.lowered().is_none(), "the logical plan alone lowers no program");
    let refusal = study.execute(&plan, &ctx).unwrap_err();
    assert!(refusal.to_string().contains("route_not_lowered"), "{refusal}");
    let compiled = study.compile(&ctx).unwrap();
    let lowered = compiled.lowered().expect("compile attaches a lowered program");
    let result = study.execute(&compiled, &ctx).unwrap();
    assert_eq!(result.lowered_program_identity(), Some(lowered.identity()));
}

#[test]
fn lowered_dispatch_refuses_public_builder_reentry() {
    let ctx = ExecutionContext::for_tests(3);
    let study = static_effect_study();
    let plan = study.compile(&ctx).unwrap();
    let inside = ctx.enter_lowered_dispatch(plan.lowered().unwrap().identity());
    let compiled = study.compile(&inside).unwrap_err();
    assert!(compiled.to_string().contains("public_builder_reentry"), "{compiled}");
    assert!(study.prepare(&inside).is_err(), "prepare inside a lowered dispatch must refuse");
    assert!(study.run(&inside).is_err(), "a one-shot run inside a lowered dispatch must refuse");
    let other = graph_posterior_study();
    assert!(other.run(&inside).is_err(), "no other study runs inside a lowered dispatch");
}

#[test]
fn one_shot_run_identifies_once_per_family() {
    // The posterior atoms are identified by builders that always report
    // `identify.compute`: compile identifies them once and the run reuses them.
    let (ctx, sink) = recording_ctx(7);
    graph_posterior_study().run(&ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1, "graph posterior: one-shot run identifies once");
    // Every other family reports a computation only where it identifies outside a
    // retained product; a one-shot run never reports two.
    for (family, study) in [
        ("static effect", static_effect_study()),
        ("temporal", temporal_effect_study()),
        ("structural", structural_study()),
        ("design", design_study()),
    ] {
        let (ctx, sink) = recording_ctx(7);
        study.run(&ctx).unwrap_or_else(|error| panic!("{family}: {error}"));
        assert!(identify_computations(&sink) <= 1, "{family}: one-shot run identifies once");
    }
}

#[test]
fn compiled_execution_takes_the_identification_the_program_retained() {
    let study = graph_posterior_study();
    let (ctx, sink) = recording_ctx(11);
    let plan = study.compile(&ctx).unwrap();
    assert!(plan.lowered().unwrap().identification_retained());
    assert_eq!(identify_computations(&sink), 1, "compile identifies every atom once");
    let result = study.execute(&plan, &ctx).unwrap();
    assert_eq!(identify_computations(&sink), 1, "execution installs the retained atoms");
    assert_eq!(result.lowered_program_identity(), plan.lowered_program_identity());
    let cached = |code: &str| code == "exec.identify.cached";
    assert!(
        !result.diagnostics.iter().any(|diagnostic| cached(diagnostic.code.as_ref())),
        "a compiled plan is not a prepared handle; its published diagnostics are unchanged"
    );
}
