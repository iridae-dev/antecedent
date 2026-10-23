//! One plan view for every route family.
//!
//! A compiled plan and a prepared handle each yield a `PlanView` whose layers
//! agree with the lowered program and whose identity is the lowered identity.
//! Building the view identifies, fits, scans, samples and leases nothing. A graph
//! posterior is viewed as one mixture over its atoms. `Study::inspect` stays the
//! non-identifying preflight beside it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "masses are compared to exact sums of the same weights")]

use std::sync::{Arc, Mutex};

use antecedent::analysis::RetainedOperation;
use antecedent::planner::plan_view::IdentificationKind;
use antecedent::planner::{PhysicalExecutionPlan, PlanStage, PlanView, RouteFamily};
use antecedent::{
    AnomalyAttributionQuery, AverageEffectQuery, CausalQuery, Dag, DenseNodeId,
    RandomizedEffectQuery, RefuteSuite, Study, TemporalDag, TemporalEffectQuery, TemporalPolicy,
};
use antecedent_contract::SlotAvailability;
use antecedent_core::execution_attempt::{PhysicalEvent, observe_execution};
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

/// Work `build` ran: instrumented operations, physical events, dense solves and
/// posterior work. Every one must be zero for a view.
fn observe_no_work<T>(label: &str, build: impl FnOnce() -> T) -> T {
    let ((attempt, solves), bayes) = antecedent_prob::fit_counts::count_bayesian_work(|| {
        antecedent_stats::fit_counts::count_least_squares_solves(|| {
            observe_execution(|| Ok::<_, std::convert::Infallible>(build()))
        })
    });
    for (operation, counts) in attempt.report.operations() {
        assert_eq!(counts.attempted, 0, "{label}: building the view ran {operation:?}");
    }
    for event in [PhysicalEvent::Scan, PhysicalEvent::ModelFit, PhysicalEvent::Copy] {
        assert_eq!(attempt.report.physical(event).unwrap_or(0), 0, "{label}: {event:?}");
    }
    assert_eq!(solves, 0, "{label}: a dense solve ran");
    assert_eq!(bayes.model_fits, 0, "{label}: a posterior fit ran");
    assert_eq!(bayes.posterior_draws, 0, "{label}: posterior draws were emitted");
    match attempt.result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

/// The view agrees with the lowered program layer by layer.
fn assert_agrees(label: &str, view: &PlanView, plan: &PhysicalExecutionPlan) {
    let lowered = plan.lowered().expect("compile attaches a lowered program");
    assert_eq!(view.identity(), lowered.identity(), "{label}: identity");
    assert_eq!(Some(view.identity()), plan.lowered_program_identity(), "{label}");
    assert_eq!(view.route().as_str(), lowered.route(), "{label}: route");
    assert_eq!(view.depth(), lowered.depth(), "{label}: depth");
    assert_eq!(view.statistical().identity(), lowered.statistical().identity(), "{label}");
    assert_eq!(view.evidence(), lowered.evidence(), "{label}: evidence program");
    assert_eq!(view.claim(), lowered.claim(), "{label}: claim demand");
    assert_eq!(view.uncertainty().reading(), lowered.claim().uncertainty(), "{label}");
    assert_eq!(view.uncertainty().claim_kind(), lowered.claim().kind(), "{label}");
    assert_eq!(view.physical().design(), lowered.physical().design(), "{label}: design");
    assert_eq!(view.physical().operators(), lowered.physical().operators(), "{label}");
    assert_eq!(view.physical().record(), Some(&plan.record), "{label}: physical record");
    assert_eq!(
        view.physical().quoted_width().is_some(),
        view.physical().not_design_based().is_none(),
        "{label}: a design is quoted or names why it is not"
    );
    assert_eq!(view.support().coordinate(), lowered.scientific().support_coordinate());
    assert_eq!(view.support().licensed(), lowered.scientific().licensed(), "{label}");
    assert_eq!(
        view.identification().kind() == IdentificationKind::Retained,
        lowered.identification_retained(),
        "{label}: identification kind"
    );
    assert_eq!(view.mixture().is_some(), view.family() == RouteFamily::GraphPosterior);
    let json = view.to_json();
    assert_eq!(json["identity"], lowered.identity().to_hex(), "{label}: JSON identity");
    assert_eq!(json["route"], lowered.route(), "{label}: JSON route");
    assert_eq!(json["family"], view.family().as_str(), "{label}: JSON family");
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

fn families() -> Vec<(&'static str, RouteFamily, Study)> {
    vec![
        ("graph posterior", RouteFamily::GraphPosterior, graph_posterior_study()),
        ("static effect", RouteFamily::StaticEffect, static_effect_study()),
        ("class effect", RouteFamily::StaticEffect, class_effect_study()),
        ("static response", RouteFamily::StaticResponse, static_response_study()),
        ("temporal", RouteFamily::Temporal, temporal_effect_study()),
        ("temporal class", RouteFamily::Temporal, temporal_class_study()),
        ("structural", RouteFamily::Structural, structural_study()),
        ("design", RouteFamily::DesignBased, design_study()),
    ]
}

#[test]
fn a_compiled_plan_views_every_layer_without_doing_work() {
    for (label, family, study) in families() {
        let (ctx, sink) = recording_ctx(5);
        let plan = study.compile(&ctx).unwrap_or_else(|error| panic!("{label}: {error}"));
        let compiled = identify_computations(&sink);
        let pool = ctx.admission().available();
        let view = observe_no_work(label, || plan.plan_view().unwrap());
        assert_eq!(identify_computations(&sink), compiled, "{label}: the view identified");
        assert_eq!(ctx.admission().available(), pool, "{label}: the view leased from the pool");
        assert_eq!(view.stage(), PlanStage::Compiled, "{label}");
        assert_eq!(view.family(), family, "{label}: family");
        assert!(view.operation().is_none(), "{label}: a compiled view names no operation");
        assert_agrees(label, &view, &plan);
    }
}

#[test]
fn study_plan_view_is_compile_and_nothing_more() {
    for (label, family, study) in families() {
        let (ctx, sink) = recording_ctx(9);
        let pool = ctx.admission().available();
        let ((view, solves), bayes) = antecedent_prob::fit_counts::count_bayesian_work(|| {
            antecedent_stats::fit_counts::count_least_squares_solves(|| {
                study.plan_view(&ctx).unwrap_or_else(|error| panic!("{label}: {error}"))
            })
        });
        assert_eq!(solves, 0, "{label}: compile for a view solved a design");
        assert_eq!(bayes.model_fits + bayes.posterior_draws, 0, "{label}: posterior work");
        assert!(identify_computations(&sink) <= 1, "{label}: compile identifies at most once");
        assert_eq!(ctx.admission().available(), pool, "{label}: the pool is left as it was");
        assert_eq!(view.family(), family, "{label}");
        assert_eq!(view.support().status(), study.inspect().unwrap().support_status, "{label}");
        let plan = study.compile(&ExecutionContext::for_tests(9)).unwrap();
        assert_eq!(view.identity(), plan.lowered_program_identity().unwrap(), "{label}");
        assert!(view.physical().kernel_policy().is_some(), "{label}: compile's kernel policy");
    }
}

#[test]
fn a_prepared_handle_views_the_installed_identification_and_the_retained_operation() {
    for (label, _, study) in families() {
        let ctx = ExecutionContext::for_tests(13);
        let Ok(prepared) = study.prepare(&ctx) else {
            // A family whose one-shot route has no prepared handle keeps its compiled view.
            continue;
        };
        let pool = ctx.admission().available();
        let view = observe_no_work(label, || prepared.plan_view().unwrap());
        assert_eq!(ctx.admission().available(), pool, "{label}: the view leased from the pool");
        assert_eq!(view.stage(), PlanStage::Prepared, "{label}");
        assert_eq!(view.operation(), Some(prepared.retained_operation().name()), "{label}");
        assert_eq!(
            Some(view.identity()),
            prepared.plan().lowered_program_identity(),
            "{label}: the prepared view reads the program the handle executes"
        );
        assert_eq!(view.support().status(), prepared.support_status(), "{label}");
        assert_agrees(label, &view, prepared.plan());
        if view.identification().installed_at_prepare() {
            assert!(!view.identification().retained().is_empty(), "{label}");
        }
        let result = prepared.estimate_retained(&ctx);
        if let Ok(result) = result {
            assert_eq!(result.lowered_program_identity(), Some(view.identity()), "{label}");
        }
    }
}

#[test]
fn the_static_effect_view_states_its_identification_design_and_uncertainty() {
    let ctx = ExecutionContext::for_tests(17);
    let study = static_effect_study();
    let prepared = study.prepare(&ctx).unwrap();
    let view = prepared.plan_view().unwrap();
    assert_eq!(view.route().as_str(), "StaticAteDag");
    let identification = view.identification();
    assert_eq!(identification.kind(), IdentificationKind::Retained);
    assert!(identification.installed_at_prepare(), "preparation installed the product");
    let retained = &identification.retained()[0];
    assert_eq!(retained.kind(), "static");
    assert!(retained.status().is_some(), "a static product states its status");
    assert!(!retained.estimands().is_empty(), "the selected estimand is shown");
    assert!(
        !retained.estimands()[0].adjustment_set().is_empty(),
        "the confounded fixture is identified by adjusting for its confounder"
    );
    assert!(!identification.assumptions().is_empty(), "the assumptions are listed");
    assert!(view.statistical().estimator().is_some(), "the estimator is named");
    assert_eq!(view.uncertainty().realization(), "frequentist");
    let json = view.to_json();
    assert_eq!(json["identification"]["retained"][0]["kind"], "static");
    assert_eq!(json["stage"], "prepared");
    if let RetainedOperation::CheckedLinear(_) = prepared.retained_operation() {
        assert_eq!(view.operation(), Some("linear_adjustment"));
        assert!(view.physical().quoted_width().is_some(), "a linear adjustment quotes its width");
    }
}

#[test]
fn a_graph_posterior_is_one_mixture_over_its_atoms() {
    let ctx = ExecutionContext::for_tests(19);
    let study = graph_posterior_study();
    let view = study.plan_view(&ctx).unwrap();
    assert_eq!(view.family(), RouteFamily::GraphPosterior);
    assert_eq!(view.route().as_str(), "GraphPosteriorAverageEffect");
    let mixture = view.mixture().expect("a graph posterior is viewed as a mixture");
    assert!(mixture.identification_retained(), "compile identified every atom once");
    assert!(!mixture.atoms().is_empty());
    let identified = mixture.identified_mass().unwrap();
    let unidentified = mixture.unidentified_mass().unwrap();
    assert!((identified + unidentified - mixture.total_weight()).abs() < 1e-9);
    for atom in mixture.atoms() {
        assert!(atom.weight() > 0.0, "every atom carries posterior weight");
        if atom.identified() == Some(true) {
            assert!(atom.estimand().is_some(), "an identified atom names its estimand");
        }
    }
    assert!(matches!(
        mixture.composition(),
        antecedent::planner::lowered::Composition::AtomMixture { .. }
    ));
    let json = view.to_json();
    assert_eq!(
        json["mixture"]["atoms"].as_array().map(Vec::len),
        Some(mixture.atoms().len()),
        "the JSON lists every atom"
    );
    let prepared = study.prepare(&ctx).unwrap();
    let prepared_view = prepared.plan_view().unwrap();
    assert_eq!(prepared_view.identity(), view.identity(), "one program before and after prepare");
    assert_eq!(
        prepared_view.mixture().map(|mixture| mixture.atoms().len()),
        Some(mixture.atoms().len()),
        "preparation installs the atoms compile retained"
    );
}

#[test]
fn inspect_stays_the_non_identifying_preflight_beside_the_view() {
    let study = static_effect_study();
    let (ctx, sink) = recording_ctx(23);
    let inspected = study.inspect().unwrap();
    assert!(matches!(inspected.reasoning.identification, SlotAvailability::Unavailable { .. }));
    assert!(matches!(inspected.reasoning.uncertainty, SlotAvailability::Unavailable { .. }));
    assert!(inspected.identities.identification_product.is_none());
    assert_eq!(identify_computations(&sink), 0, "inspect does not identify");
    let view = study.plan_view(&ctx).unwrap();
    assert_eq!(view.identification().kind(), IdentificationKind::Retained);
    assert!(view.identification().status().is_some(), "the view shows what inspect withholds");
}

#[test]
fn a_plan_without_a_lowered_program_has_no_view() {
    let ctx = ExecutionContext::for_tests(29);
    let study = static_effect_study();
    let plan = study.compile_logical().unwrap().compile_physical(&ctx).unwrap();
    let refusal = plan.plan_view().unwrap_err();
    assert!(refusal.to_string().contains("route_not_lowered"), "{refusal}");
}
