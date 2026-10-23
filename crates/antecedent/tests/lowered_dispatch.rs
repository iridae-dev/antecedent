//! Dispatch executes only the lowered program `Study::compile` attached.
//!
//! A physical plan compiled from the logical plan alone carries no lowered program
//! and is refused before any route runs. A context already inside a lowered
//! dispatch refuses to compile, prepare or run another study. A one-shot run
//! identifies once: compile reports `identify.compute` when its lowered program
//! retains an identification, and the execution takes that retained product
//! instead of identifying again.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex};

use antecedent::{
    AnomalyAttributionQuery, AverageEffectQuery, CausalQuery, Dag, DenseNodeId,
    RandomizedEffectQuery, RefuteSuite, Study, TemporalDag, TemporalEffectQuery, TemporalPolicy,
};
use antecedent_core::{
    ExecutionContext, Intervention, Lag, ProgressSink, ResponseFunctional, ResponseQuery, Value,
    VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Cpdag, TemporalCpdag, ensure_lagged};

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

/// The confounded fixture under the equivalence class of its DAG.
fn class_effect_study() -> Study {
    let (data, dag, query) = confounded_scm(128, 43);
    Study::tabular(data)
        .graph(Cpdag::from_dag(&dag))
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// The mean outcome of the confounded fixture under `do(t = 1)`.
fn static_response_study() -> Study {
    let (data, dag, query) = confounded_scm(128, 47);
    let response = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: query.outcome,
        interventions: [Intervention::set(query.treatment, Value::f64(1.0))].into(),
    });
    Study::tabular(data)
        .graph(dag)
        .query(CausalQuery::Response(response))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// The lagged series of `temporal_effect_study` and its temporal graph.
fn lagged_series() -> (TimeSeriesData, TemporalDag, TemporalEffectQuery) {
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
    (series, graph, query)
}

/// `y_t = 0.8 x_{t-1}` plus a small periodic term, under `x_{t-1} -> y_t`.
fn temporal_effect_study() -> Study {
    let (series, graph, query) = lagged_series();
    Study::series(series)
        .graph(graph)
        .temporal_query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// The lagged series under the temporal equivalence class of its graph.
fn temporal_class_study() -> Study {
    let (series, graph, query) = lagged_series();
    Study::series(series)
        .graph(TemporalCpdag::from_temporal_dag(&graph))
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
    // Compile identifies what preparation would and reports `identify.compute` once
    // (single-graph, class, temporal, structural and design products alike; the
    // posterior builders report their own). Every executor then takes the prepared
    // cache, the product the plan retained, or only then a fresh identification,
    // which reports too: a one-shot run that identified twice reports two.
    for (family, study) in [
        ("graph posterior", graph_posterior_study()),
        ("static effect", static_effect_study()),
        ("class effect", class_effect_study()),
        ("static response", static_response_study()),
        ("temporal", temporal_effect_study()),
        ("temporal class", temporal_class_study()),
        ("structural", structural_study()),
        ("design", design_study()),
    ] {
        let (ctx, sink) = recording_ctx(7);
        let result = study.run(&ctx).unwrap_or_else(|error| panic!("{family}: {error}"));
        assert_eq!(identify_computations(&sink), 1, "{family}: a one-shot run identifies once");
        assert!(result.lowered_program_identity().is_some(), "{family}: the run is lowered");
    }
}

#[test]
fn a_compiled_single_graph_plan_executes_its_retained_identification() {
    // An execution of a compiled plan takes the retained product: compile identifies
    // once and the execution adds no identification. A series study is not
    // a prepared handle, so its published result is the one a fresh identification produces.
    for (family, study) in [
        ("class effect", class_effect_study()),
        ("structural", structural_study()),
        ("temporal class", temporal_class_study()),
    ] {
        let (ctx, sink) = recording_ctx(13);
        let plan = study.compile(&ctx).unwrap();
        assert!(plan.lowered().unwrap().identification_retained(), "{family}: retained");
        assert_eq!(identify_computations(&sink), 1, "{family}: compile identifies once");
        let reused = study.execute(&plan, &ctx).unwrap();
        assert_eq!(identify_computations(&sink), 1, "{family}: execution reuses the product");
        // A tabular study executes a compiled plan through its prepared handle, which
        // reports the retained identification it installed; a series study does not.
        if family == "temporal class" {
            assert!(
                !reused
                    .diagnostics
                    .iter()
                    .any(|diagnostic| diagnostic.code.as_ref() == "exec.identify.cached"),
                "{family}: a compiled plan publishes the diagnostics of a fresh identification"
            );
        }
        // A plan compiled for another study keeps nothing it could reuse here.
        let (other_ctx, other_sink) = recording_ctx(13);
        let foreign = static_effect_study().compile(&other_ctx).unwrap();
        let fresh = study.execute(&foreign, &other_ctx);
        assert!(fresh.is_err(), "{family}: a foreign plan's lowered cell does not match");
        assert_eq!(identify_computations(&other_sink), 1, "{family}: only the foreign compile");
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
