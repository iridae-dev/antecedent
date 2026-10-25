//! Plan view stages: what each inspection stage shows and the work it does.
//!
//! Every family's view lists its statistical, evidence and decision, and physical
//! operations, the physical backend chosen with the alternatives it rejected, the
//! route-memory quote by component, what is reused and the assumptions relied on.
//! The compiled, study and prepared stages each build their view without a scan, a
//! copy, a fit, a draw or a lease. A graph-posterior mixture is viewed as one
//! evidence composition over its atoms. A memory quote over the limit is refused at
//! compile, before any buffer is copied. The facts a result carries keep measured
//! counts, admission reservations and plan estimates apart.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "masses are compared to exact sums of the same weights")]

use std::sync::Arc;

use antecedent::analysis::planning::compile_planning;
use antecedent::design::{
    PlannedDesign, PlanningAssumptions, PlanningCriterion, PlanningGoal, PlanningRequest,
};
use antecedent::discovery::StaticDiscoverParams;
use antecedent::discovery_defaults::resolve_ci;
use antecedent::{AssociationQuery, DiscoverySpec};
use antecedent_discovery::BackgroundKnowledge;

use antecedent::planner::plan_view::{OperationLayer, PlanView};
use antecedent::planner::{PhysicalExecutionPlan, PlanStage, RouteFamily};
use antecedent::{
    AnomalyAttributionQuery, AverageEffectQuery, CausalQuery, Dag, DenseNodeId,
    RandomizedEffectQuery, RefuteSuite, Study, TemporalDag, TemporalEffectQuery, TemporalPolicy,
};
use antecedent_core::execution_attempt::{PhysicalEvent, observe_execution};
use antecedent_core::execution_facts::AccountingBasis;
use antecedent_core::{
    ExecutionContext, Intervention, Lag, ResponseFunctional, ResponseQuery, Value, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Cpdag, TemporalCpdag, ensure_lagged};

#[path = "fixtures_common/mod.rs"]
mod common;

use common::fixtures::{confounded_scm, mixture_graph_posterior};

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

/// Work `build` ran: every instrumented operation and physical event must be zero.
fn observe_no_work<T>(label: &str, build: impl FnOnce() -> T) -> T {
    let attempt = observe_execution(|| Ok::<_, std::convert::Infallible>(build()));
    for (operation, counts) in attempt.report.operations() {
        assert_eq!(counts.attempted, 0, "{label}: building the view ran {operation:?}");
    }
    for event in [PhysicalEvent::Scan, PhysicalEvent::ModelFit, PhysicalEvent::Copy] {
        assert_eq!(attempt.report.physical(event).unwrap_or(0), 0, "{label}: {event:?}");
    }
    assert_eq!(attempt.report.physical(PhysicalEvent::CopiedBytes).unwrap_or(0), 0, "{label}");
    match attempt.result {
        Ok(value) => value,
        Err(never) => match never {},
    }
}

/// Operations a view must list, counted from the program the view reads.
fn assert_operations_derive_from_the_program(label: &str, view: &PlanView) {
    let operations = view.operations();
    let count = |layer: OperationLayer| {
        operations.iter().filter(|operation| operation.layer() == layer).count()
    };
    let statistical = view.statistical();
    let expected_statistical = statistical.operators().len()
        + statistical.models().len()
        + usize::from(statistical.folds().is_some())
        + usize::from(statistical.resampling().is_some())
        + usize::from(statistical.posterior().is_some());
    assert_eq!(count(OperationLayer::Statistical), expected_statistical, "{label}: statistical");
    let evidence = view.evidence();
    let decision = evidence.decision();
    let expected_evidence = evidence.sources().len()
        + 1
        + decision.map_or(0, |decision| 1 + usize::from(decision.information_value().is_some()))
        + usize::from(evidence.planning().and_then(|planning| planning.simulation()).is_some())
        + evidence.chain().map_or(0, |chain| chain.plan().stages.len());
    assert_eq!(count(OperationLayer::Evidence), expected_evidence, "{label}: evidence");
    let physical = view.physical();
    let expected_physical = physical.operators().len()
        + physical.materializations().len()
        + usize::from(physical.backend().is_some());
    assert_eq!(count(OperationLayer::Physical), expected_physical, "{label}: physical");
    let json = view.to_json();
    assert_eq!(
        json["operations"].as_array().map(Vec::len),
        Some(operations.len()),
        "{label}: JSON lists every operation"
    );
}

/// Backend, memory, materialization, reuse and assumptions of a view taken from a plan.
fn assert_view_states_resources_and_reuse(label: &str, view: &PlanView) {
    let physical = view.physical();
    let backend = physical.backend().unwrap_or_else(|| panic!("{label}: no backend"));
    assert_eq!(backend.rejected().len(), 3, "{label}: three alternatives are rejected");
    assert!(
        backend.rejected().iter().all(|rejected| rejected.alternative() != backend.selected()),
        "{label}: the selected alternative is not also rejected"
    );
    let memory = physical.memory_quote().unwrap_or_else(|| panic!("{label}: no memory quote"));
    let summed: u64 = memory.components().iter().map(|(_, bytes)| *bytes).sum();
    assert_eq!(summed, memory.total(), "{label}: the components are the quote");
    assert!(memory.total() > 0, "{label}: absence is not zero bytes");
    assert!(
        view.operations().iter().any(|operation| operation.layer() == OperationLayer::Physical),
        "{label}: a plan has physical operations"
    );
    assert!(!physical.materializations().is_empty(), "{label}: materialization is shown");
    let json = view.to_json();
    assert_eq!(json["physical"]["backend"]["selected"], backend.selected().as_str(), "{label}");
    assert_eq!(json["physical"]["memory"]["total"], memory.total(), "{label}");
    assert_eq!(
        json["physical"]["memory"]["components"].as_array().map(Vec::len),
        Some(8),
        "{label}: every memory component is listed"
    );
    assert_eq!(
        json["reuse"]["identification_retained"],
        view.reuse().identification_retained(),
        "{label}"
    );
    assert_eq!(
        json["assumptions"].as_array().map(Vec::len),
        Some(view.assumptions().len()),
        "{label}"
    );
}

#[test]
fn every_family_view_lists_its_operations_backend_memory_and_reuse() {
    for (label, family, study) in families() {
        let ctx = ExecutionContext::for_tests(61);
        let view = study.plan_view(&ctx).unwrap_or_else(|error| panic!("{label}: {error}"));
        assert_eq!(view.family(), family, "{label}");
        assert_operations_derive_from_the_program(label, &view);
        assert_view_states_resources_and_reuse(label, &view);
    }
}

#[test]
fn each_stage_builds_its_view_without_scanning_copying_fitting_or_leasing() {
    for (label, _, study) in families() {
        let ctx = ExecutionContext::for_tests(67);
        let plan: PhysicalExecutionPlan =
            study.compile(&ctx).unwrap_or_else(|error| panic!("{label}: {error}"));
        let pool = ctx.admission().available();
        let compiled = observe_no_work(label, || plan.plan_view().unwrap());
        assert_eq!(compiled.stage(), PlanStage::Compiled, "{label}");
        let again = observe_no_work(label, || plan.plan_view().unwrap());
        assert_eq!(
            compiled.to_json(),
            again.to_json(),
            "{label}: the view is a function of the plan, not of the number of times it is read"
        );
        assert_eq!(ctx.admission().available(), pool, "{label}: the compiled view leased");
        assert_view_states_resources_and_reuse(label, &compiled);
        let Ok(prepared) = study.prepare(&ctx) else {
            continue;
        };
        let pool = ctx.admission().available();
        let view = observe_no_work(label, || prepared.plan_view().unwrap());
        assert_eq!(view.stage(), PlanStage::Prepared, "{label}");
        assert_eq!(ctx.admission().available(), pool, "{label}: the prepared view leased");
        assert_operations_derive_from_the_program(label, &view);
        assert_view_states_resources_and_reuse(label, &view);
        assert_eq!(
            view.reuse().retained_operation(),
            view.operation(),
            "{label}: reuse names the sealed operation"
        );
    }
}

#[test]
fn a_graph_posterior_mixture_is_one_evidence_composition_over_its_atoms() {
    let ctx = ExecutionContext::for_tests(71);
    let study = graph_posterior_study();
    let view = study.plan_view(&ctx).unwrap();
    let mixture = view.mixture().expect("a graph posterior is a mixture");
    let operations = view.operations();
    let compose: Vec<_> = operations
        .iter()
        .filter(|operation| {
            operation.layer() == OperationLayer::Evidence && operation.name() == "compose"
        })
        .collect();
    assert_eq!(compose.len(), 1, "one composition, not one plan per atom");
    assert!(compose[0].subject().starts_with("atom_mixture"), "{}", compose[0].subject());
    assert!(!mixture.atoms().is_empty());
    let summed: f64 = mixture.atoms().iter().map(|atom| atom.weight()).sum();
    assert!((summed - mixture.total_weight()).abs() < 1e-12);
    assert!(view.reuse().identification_retained(), "compile identified every atom once");
    // The mixture is viewed from retained values alone.
    let plan = study.compile(&ctx).unwrap();
    let from_plan = observe_no_work("graph posterior", || plan.plan_view().unwrap());
    assert_eq!(from_plan.identity(), view.identity());
    assert!(!from_plan.operations().is_empty());
}

// The estimator is one with no streamed reduction: a streamable estimator takes the streamed
// route under a tight limit instead of refusing (a graph posterior fits each atom's own design
// and always takes the streamable default), so the claim pinned here is the refusal of a
// resident design whose quote exceeds the limit.
#[test]
fn a_memory_quote_over_the_limit_is_refused_before_anything_is_copied() {
    let ctx = ExecutionContext::for_tests(73);
    let (data, dag, query) = confounded_scm(128, 41);
    let study = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(antecedent::EstimatorId::GlmAdjustment)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let view = study.plan_view(&ctx).unwrap();
    let quoted = view.physical().memory_quote().unwrap().total();
    assert!(quoted > 1);
    let tight = ExecutionContext::for_tests(73).with_memory_limit(Some(quoted - 1));
    let pool = tight.admission().available();
    let attempt = observe_execution(|| study.compile(&tight));
    let error = attempt.result.unwrap_err();
    assert!(error.to_string().contains(&format!("exceeds memory limit {}", quoted - 1)), "{error}");
    assert_eq!(attempt.report.physical(PhysicalEvent::Copy).unwrap_or(0), 0, "no copy");
    assert_eq!(attempt.report.physical(PhysicalEvent::CopiedBytes).unwrap_or(0), 0, "no bytes");
    assert_eq!(attempt.report.physical(PhysicalEvent::Scan).unwrap_or(0), 0, "no scan");
    assert_eq!(tight.admission().available(), pool, "a refusal charges nothing");
    // Naming the next attempt: the refusal says what to lower.
    assert!(error.to_string().contains("lower the declared copies, workers, or retained outputs"));
}

#[test]
fn a_result_keeps_measured_facts_reservations_and_estimates_apart() {
    let ctx = ExecutionContext::for_tests(79);
    let result = static_effect_study().run(&ctx).unwrap();
    let facts = result.execution_facts().expect("a run is observed");
    assert!(facts.is_native(), "the engine emitted these facts");
    let accounting = result.execution_accounting().expect("accounting follows the facts");
    assert!(accounting.get("model_fits").unwrap().value().unwrap() >= 1, "the fit was counted");
    for name in [
        "scans",
        "copies",
        "copied_bytes",
        "allocations",
        "allocated_bytes",
        "model_fits",
        "cache_hits",
        "spill_bytes",
    ] {
        assert_eq!(accounting.get(name).unwrap().basis(), AccountingBasis::Measured, "{name}");
    }
    for name in ["peak_admitted_bytes", "peak_admitted_threads", "retained_output_bytes"] {
        assert_eq!(
            accounting.get(name).unwrap().basis(),
            AccountingBasis::AdmissionReservation,
            "{name}"
        );
    }
    let planned = accounting.get("planned_peak_memory_bytes").unwrap();
    assert_eq!(planned.basis(), AccountingBasis::Estimate);
    assert_eq!(planned.value(), result.physical_plan.estimated_peak_memory_bytes);
    let measured = accounting.with_basis(AccountingBasis::Measured).count();
    let reserved = accounting.with_basis(AccountingBasis::AdmissionReservation).count();
    let estimated = accounting.with_basis(AccountingBasis::Estimate).count();
    assert_eq!(measured + reserved + estimated, accounting.quantities().len());
    assert!(measured > 0 && reserved > 0 && estimated > 0);
}

fn noise(i: usize) -> f64 {
    ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5
}

fn discovery_study() -> Study {
    let n = 240;
    let z: Vec<f64> = (0..n).map(|i| noise(i)).collect();
    let t: Vec<f64> = (0..n).map(|i| 0.8 * z[i] + 0.3 * noise(i + 1000)).collect();
    let y: Vec<f64> = (0..n).map(|i| 0.9 * t[i] + 0.5 * z[i] + 0.3 * noise(i + 2000)).collect();
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let params = StaticDiscoverParams {
        alpha: 0.05,
        max_cond_size: 2,
        fdr: None,
        ci: resolve_ci("parcorr", None).unwrap(),
        screen_pc: false,
        max_subset: None,
        knowledge: BackgroundKnowledge::default(),
    };
    Study::tabular(data)
        .discovery(
            DiscoverySpec::pc(params, "parcorr"),
            vec![VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)],
        )
        .build()
        .unwrap()
}

fn association_study() -> Study {
    let x: Vec<f64> = (0..120).map(|i| f64::from(i) / 12.0).collect();
    let y: Vec<f64> = x.iter().enumerate().map(|(i, x)| 1.0 + 2.0 * x + noise(i)).collect();
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    Study::tabular(data)
        .query(CausalQuery::Association(AssociationQuery::new(
            VariableId::from_raw(1),
            [VariableId::from_raw(0)],
        )))
        .build()
        .unwrap()
}

#[test]
fn discovery_and_association_views_state_operations_backend_memory_and_reuse() {
    let studies: [(&str, RouteFamily, Study); 2] = [
        ("discovery", RouteFamily::Discovery, discovery_study()),
        ("association", RouteFamily::Associational, association_study()),
    ];
    // A structure search recurses deeply; give it the stack its own tests use.
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || {
            for (label, family, study) in studies {
                let ctx = ExecutionContext::for_tests(83);
                let plan = study.compile(&ctx).unwrap_or_else(|error| panic!("{label}: {error}"));
                let compiled = observe_no_work(label, || plan.plan_view().unwrap());
                assert_eq!(compiled.family(), family, "{label}");
                assert_operations_derive_from_the_program(label, &compiled);
                assert_view_states_resources_and_reuse(label, &compiled);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn a_planning_view_lists_the_simulation_its_program_carries() {
    let request = PlanningRequest::new(
        PlannedDesign::Cluster { mean_cluster_size: 12.0, icc: 0.05, cluster_size_cv: 0.0 },
        PlanningAssumptions::continuous(1.0).with_effect(0.3).with_attrition(0.1),
        PlanningCriterion::Power { target: 0.8 },
        PlanningGoal::SampleSize { max_units: 10_000 },
    )
    .with_simulation(100, 3);
    let ctx = ExecutionContext::for_tests(89);
    let plan = compile_planning(&request, &ctx).unwrap();
    let view = observe_no_work("planning", || plan.plan_view());
    assert_eq!(view.family(), RouteFamily::Planning);
    assert_operations_derive_from_the_program("planning", &view);
    let operations = view.operations();
    assert!(
        operations
            .iter()
            .any(|operation| operation.name() == "simulate"
                && operation.subject() == "100 replicates"),
        "the program's simulation is an evidence operation"
    );
    assert!(view.physical().backend().is_none(), "a standalone plan has no physical record");
}

#[test]
fn posterior_storage_is_quoted_before_admission_and_survives_the_retained_workflow() {
    let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 8.0).collect();
    let y: Vec<f64> = x.iter().enumerate().map(|(i, x)| 1.0 + 2.0 * x + noise(i)).collect();
    // Unweighted Gaussian rows use closed-form conjugate marginals and retain
    // no sampled draws. Nonuniform declared weights select the actual Bayesian
    // bootstrap, whose retained output must scale with n_draws.
    let weight: Vec<f64> = (0..80).map(|i| 1.0 + f64::from(i % 3)).collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("y", y.as_slice()),
        ("weight", weight.as_slice()),
    ])
    .unwrap();
    let study = |draws| {
        Study::tabular(data.clone())
            .query(CausalQuery::Association(
                AssociationQuery::new(VariableId::from_raw(1), [VariableId::from_raw(0)])
                    .with_weight(VariableId::from_raw(2)),
            ))
            .inference(antecedent::InferenceMode::Bayesian(
                antecedent::BayesianConfig::conjugate().n_draws(draws),
            ))
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    };
    let ctx = ExecutionContext::for_tests(97);
    let small = study(32).compile(&ctx).unwrap();
    let large_study = study(512);
    let large = large_study.compile(&ctx).unwrap();
    assert_eq!(small.lowered().unwrap().statistical().posterior().unwrap().draws(), 32);
    assert_eq!(large.lowered().unwrap().statistical().posterior().unwrap().draws(), 512);
    let small_quote = small.memory_quote.unwrap();
    let quote = large.memory_quote.unwrap();
    assert!(quote.retained_scores_draws() > small_quote.retained_scores_draws());
    assert!(quote.total() > small_quote.total());
    let tight = ExecutionContext::for_tests(97).with_memory_limit(Some(quote.total() - 1));
    let available = tight.admission().available();
    let refused = observe_execution(|| large_study.compile(&tight));
    assert!(matches!(refused.result, Err(antecedent::CausalError::Resource { .. })));
    for event in [PhysicalEvent::Scan, PhysicalEvent::Copy, PhysicalEvent::ModelFit] {
        assert_eq!(refused.report.physical(event).unwrap_or(0), 0);
    }
    assert_eq!(tight.admission().available(), available, "a quote is never a reservation");

    let mut prepared = large_study.prepare(&ctx).unwrap();
    assert_eq!(prepared.plan_view().unwrap().physical().memory_quote(), Some(quote));
    let result = prepared.estimate(&data, &ctx).unwrap();
    let refreshed = prepared.refresh(data.clone(), &ctx).unwrap();
    assert_eq!(
        result.association().unwrap().estimate.terms,
        refreshed.association().unwrap().estimate.terms
    );
    let accounting = refreshed.execution_accounting().unwrap();
    let planned = accounting.get("planned_peak_memory_bytes").unwrap();
    assert_eq!(planned.basis(), AccountingBasis::Estimate);
    assert_eq!(planned.value(), Some(quote.total()));
    assert_eq!(
        accounting.get("peak_admitted_bytes").unwrap().basis(),
        AccountingBasis::AdmissionReservation
    );
    let bytes =
        prepared.encode_contracted_result(&refreshed, "statistical-memory-quote", &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    assert!(consumed.contract.is_some());
    assert!(consumed.body.association.is_some());
    assert_eq!(
        consumed.body.native_execution,
        antecedent_io::native_execution_wire::NativeExecutionWire::from_facts(
            refreshed.execution_facts().unwrap()
        )
    );
}

#[test]
fn fold_memory_tracks_the_live_window_instead_of_every_fold_and_learner() {
    use antecedent_core::execution::AdmissionPool;
    use antecedent_core::{NonZeroThreadCount, Parallelism};
    let mut ctx =
        ExecutionContext::for_tests(101).with_admission(AdmissionPool::fixed(2, u64::MAX));
    ctx.parallelism = Parallelism::bounded(NonZeroThreadCount::new(2).unwrap());
    let compile = |folds| {
        let (data, dag, query) = confounded_scm(128, 41);
        let mut estimator = antecedent_estimate::DmlAte::new();
        estimator.folds = folds;
        Study::tabular(data)
            .graph(dag)
            .query(query)
            .estimator(estimator)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .compile(&ctx)
            .unwrap()
    };
    let five = compile(5);
    let ten = compile(10);
    assert_eq!(
        five.plan_view().unwrap().physical().backend().unwrap().selected(),
        antecedent_core::physical_choice::PhysicalAlternative::MaterializedDesign,
        "numeric source columns do not grant a learner executor a direct-view backend"
    );
    let quote = five.memory_quote.unwrap();
    assert_eq!(quote.copy_count(), 2, "portable learner folds share the two-worker window");
    assert_eq!(quote.learner_fold_copies(), quote.encoded_width() * 2);
    assert_eq!(
        ten.memory_quote.unwrap(),
        quote,
        "more sequential waves do not retain more designs"
    );

    let predictive = Study::tabular(mixture_data())
        .query(CausalQuery::Association(
            AssociationQuery::new(
                VariableId::from_raw(1),
                [VariableId::from_raw(0), VariableId::from_raw(2)],
            )
            .with_functional(antecedent_core::AssociationFunctional::Predictive { folds: 5 }),
        ))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .compile(&ctx)
        .unwrap();
    assert_eq!(
        predictive.memory_quote.unwrap().copy_count(),
        1,
        "the predictive association fits and consumes one fold at a time"
    );
}
