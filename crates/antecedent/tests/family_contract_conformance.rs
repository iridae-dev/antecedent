//! Cross-family contract conformance.
//!
//! Every route family obeys the same contracts, not merely compiles. The row table
//! `parity/capabilities/family_conformance.toml` is generated from the public route
//! table and holds one row per route family; this suite runs the same six obligations
//! on a representative study of each covered family:
//!
//! - uncertainty is published through the typed slot with its declared components, or
//!   the route is refused naming its next attempt;
//! - reuse (refresh, retarget) invalidates exactly the dependency keys it should;
//! - a decision demand is satisfied or refused by carrier;
//! - the saved result is read by an independent consumer that re-derives its identity
//!   and its claim;
//! - the plan view agrees with the lowered program and with `Study::inspect`;
//! - every refusal names its next attempt.
//!
//! A route id the table does not list fails the build, as does a family without a row.
//! A violation is an integration gap: the finding names the study and the obligation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "fixture indices convert exactly and exact bit patterns are compared"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Display;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent::analysis::planning::compile_planning;
use antecedent::analysis::propagation_chain::{
    ChainCoefficientWire, ChainOpWire, ChainReading, ChainRefusal, ChainSourceWire, ChainStageWire,
    ChainUnaryWire, LawMeaning, PropagationChain, PropagationChainRecordWire, STATUS_CURRENT,
    STATUS_INVALIDATED, STATUS_REWEIGHTED, SharedFactorKind, StageKind, TableConstruction,
    compile_propagation_chain, consume_propagation_chain,
};
use antecedent::analysis::staleness::StalenessPolicy;
use antecedent::analysis::{
    DECISION_NEXT_ATTEMPTS, DecisionCarrierError, DecisionCompileRequest, DecisionDemand,
    MissingCarrier, PublishedCarriers, score_decision,
};
use antecedent::design::{
    PlannedDesign, PlanningAssumptions, PlanningCriterion, PlanningGoal, PlanningRequest,
};
use antecedent::discovery::StaticDiscoverParams;
use antecedent::discovery_defaults::resolve_ci;
use antecedent::planner::lowered::{CarrierDemand, LoweredProgram, UncertaintyReading};
use antecedent::planner::{PhysicalExecutionPlan, PlanView, RouteFamily};
use antecedent::{
    AnomalyAttributionQuery, AssociationQuery, BayesianConfig, CausalError, CausalQuery,
    CompileCause, Dag, DenseNodeId, DiscoverySpec, EstimatorId, InferenceMode, PreparedStudy,
    RandomizedEffectQuery, RefuteSuite, Study, TemporalDag, TemporalEffectQuery, TemporalPolicy,
};
use antecedent_contract::{EvidencePort, SlotAvailability, TransformIntent};
use antecedent_core::recalc::Stage;
use antecedent_core::{
    ContractIdentities, ExecutionContext, Intervention, Lag, ResponseFunctional, ResponseQuery,
    Value, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_design::{AffineUtility, DecisionProblem, DesignError, Utility};
use antecedent_discovery::BackgroundKnowledge;
use antecedent_graph::{Cpdag, TemporalCpdag, ensure_lagged};
use antecedent_io::cyclic_family_artifact::{
    CertificateDemandWire, CyclicComponentWire, CyclicFamilyAnswer, CyclicFamilyReport,
    CyclicFamilyRequest, CyclicInterventionWire, CyclicWorldOutcomeAnswer, CyclicWorldWire,
    LinearCyclicWire, consume_cyclic_family, consume_cyclic_family_contract,
};
use antecedent_io::{
    claim_kind_name_of_section, consume_analysis_result, contract_seal, result_digest,
};
use antecedent_state::{InvalidationTarget, TemporalDisposition, TemporalFactor};
use antecedent_stats::normal_ppf;

#[path = "fixtures_common/mod.rs"]
mod common;

use common::fixtures::{confounded_scm, mixture_graph_posterior};

// ---------------------------------------------------------------------------------------
// Findings

/// Violations of one representative study.
struct Findings {
    label: &'static str,
    items: Vec<String>,
}

impl Findings {
    fn new(label: &'static str) -> Self {
        Self { label, items: Vec::new() }
    }

    fn check(&mut self, ok: bool, what: impl Display) {
        if !ok {
            self.items.push(format!("{}: {what}", self.label));
        }
    }

    /// A refusal conforms only when it names what to try next.
    fn refused(&mut self, what: &str, error: &CausalError) {
        if !names_next_attempt(error) {
            self.items.push(format!(
                "{}: {what} refused without naming a next attempt: {error}",
                self.label
            ));
        }
    }
}

fn names_next_attempt(error: &CausalError) -> bool {
    error.to_string().to_lowercase().contains("next attempt") || error.remedy().is_some()
}

fn same_bits(left: f64, right: f64) -> bool {
    left.to_bits() == right.to_bits()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

/// Debug builds of the dispatcher carry frames close to the default test-thread stack.
fn on_large_stack<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new()
        .name("family-conformance".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .expect("spawn conformance thread")
        .join()
        .expect("conformance body panicked")
}

/// Run `obligation` on every representative study, and `shared` on every family that has no
/// `Study` entry, and fail with every violation found.
fn run_obligation(obligation: fn(&mut Rep) -> Vec<String>, shared: Obligation) {
    let findings = on_large_stack(move || {
        let mut found = Vec::new();
        for mut rep in representatives() {
            found.extend(obligation(&mut rep));
        }
        for other in others() {
            found.extend(other_findings(&other, shared));
        }
        found
    });
    assert!(findings.is_empty(), "integration gaps:\n{}", findings.join("\n"));
}

// ---------------------------------------------------------------------------------------
// Representative studies

type Table = fn() -> TabularData;
type Series = fn() -> TimeSeriesData;

/// The data a representative retains, with a same-schema replacement and an identical copy.
enum Retained {
    Table { same: Table, changed: Table },
    Series { same: Series, changed: Series },
}

struct Rep {
    label: &'static str,
    family: RouteFamily,
    study: Study,
    retained: Retained,
}

fn static_effect() -> Rep {
    let (data, dag, query) = confounded_scm(128, 41);
    let study = Study::tabular(data)
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    Rep {
        label: "static_effect",
        family: RouteFamily::StaticEffect,
        study,
        retained: Retained::Table {
            same: || confounded_scm(128, 41).0,
            changed: || confounded_scm(128, 42).0,
        },
    }
}

fn static_effect_doubly_robust() -> Rep {
    let (data, dag, query) = confounded_scm(96, 12);
    let study = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    Rep {
        label: "static_effect_doubly_robust",
        family: RouteFamily::StaticEffect,
        study,
        retained: Retained::Table {
            same: || confounded_scm(96, 12).0,
            changed: || confounded_scm(96, 13).0,
        },
    }
}

fn static_effect_bayesian() -> Rep {
    let (data, dag, query) = confounded_scm(128, 41);
    let study = Study::tabular(data)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    Rep {
        label: "static_effect_bayesian",
        family: RouteFamily::StaticEffect,
        study,
        retained: Retained::Table {
            same: || confounded_scm(128, 41).0,
            changed: || confounded_scm(128, 42).0,
        },
    }
}

fn static_effect_class() -> Rep {
    let (data, dag, query) = confounded_scm(128, 43);
    let study = Study::tabular(data)
        .graph(Cpdag::from_dag(&dag))
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    Rep {
        label: "static_effect_class",
        family: RouteFamily::StaticEffect,
        study,
        retained: Retained::Table {
            same: || confounded_scm(128, 43).0,
            changed: || confounded_scm(128, 44).0,
        },
    }
}

fn static_response() -> Rep {
    let (data, dag, query) = confounded_scm(128, 47);
    let response = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: query.outcome,
        interventions: [Intervention::set(query.treatment, Value::f64(1.0))].into(),
    });
    let study = Study::tabular(data)
        .graph(dag)
        .query(CausalQuery::Response(response))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    Rep {
        label: "static_response",
        family: RouteFamily::StaticResponse,
        study,
        retained: Retained::Table {
            same: || confounded_scm(128, 47).0,
            changed: || confounded_scm(128, 48).0,
        },
    }
}

fn chain_data(spike: f64) -> TabularData {
    let x: Vec<f64> = (0..20).map(f64::from).collect();
    let y: Vec<f64> = (0..20).map(|i| if i == 19 { spike } else { 2.0 * f64::from(i) }).collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn structural_attribution() -> Rep {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let study = Study::tabular(chain_data(200.0))
        .graph(graph)
        .query(CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new(
            [VariableId::from_raw(1)],
            100,
        )))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    Rep {
        label: "structural_attribution",
        family: RouteFamily::Structural,
        study,
        retained: Retained::Table { same: || chain_data(200.0), changed: || chain_data(150.0) },
    }
}

fn lagged_series(phase: f64) -> (TimeSeriesData, TemporalDag, TemporalEffectQuery) {
    let rows = 96;
    let mut x = vec![0.0; rows];
    let mut y = vec![0.0; rows];
    for t in 0..rows {
        let clock = t as f64;
        x[t] = (0.37 * clock).sin();
        if t > 0 {
            y[t] = 0.8 * x[t - 1] + 0.1 * (1.3 * clock + phase).cos();
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

fn temporal_effect() -> Rep {
    let (series, graph, query) = lagged_series(0.0);
    let study = Study::series(series)
        .graph(graph)
        .temporal_query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    Rep {
        label: "temporal_effect",
        family: RouteFamily::Temporal,
        study,
        retained: Retained::Series {
            same: || lagged_series(0.0).0,
            changed: || lagged_series(0.7).0,
        },
    }
}

fn temporal_effect_class() -> Rep {
    let (series, graph, query) = lagged_series(0.0);
    let study = Study::series(series)
        .graph(TemporalCpdag::from_temporal_dag(&graph))
        .temporal_query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    Rep {
        label: "temporal_effect_class",
        family: RouteFamily::Temporal,
        study,
        retained: Retained::Series {
            same: || lagged_series(0.0).0,
            changed: || lagged_series(0.7).0,
        },
    }
}

fn design_parts(frequency: f64) -> (TabularData, RandomizedEffectQuery) {
    let n = 200;
    let assignment: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let outcomes: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * f64::from(u8::from(assignment[i])) + 0.5 * (i as f64 * frequency).sin()
        })
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
    (data, query)
}

fn randomized_design() -> Rep {
    let (data, query) = design_parts(0.3);
    let study = Study::tabular(data).query(query).build().unwrap();
    Rep {
        label: "randomized_design",
        family: RouteFamily::DesignBased,
        study,
        retained: Retained::Table {
            same: || design_parts(0.3).0,
            changed: || design_parts(0.45).0,
        },
    }
}

fn mixture_data(spread: f64) -> TabularData {
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    let mut confounder = Vec::new();
    for _ in 0..10 {
        for (z, t, count) in [(0.0, 0.0, 6), (0.0, 1.0, 2), (1.0, 0.0, 2), (1.0, 1.0, 6)] {
            for row in 0..count {
                let epsilon = if row % 2 == 0 { -spread } else { spread };
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

fn graph_posterior_effect() -> Rep {
    let study = Study::tabular(mixture_data(0.2))
        .graph_posterior(mixture_graph_posterior())
        .query(antecedent::AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    Rep {
        label: "graph_posterior_effect",
        family: RouteFamily::GraphPosterior,
        study,
        retained: Retained::Table { same: || mixture_data(0.2), changed: || mixture_data(0.25) },
    }
}

fn noise(i: usize) -> f64 {
    ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5
}

fn line_data(offset: usize) -> TabularData {
    let x: Vec<f64> = (0..120).map(|i| (i as f64) / 12.0).collect();
    let y: Vec<f64> =
        x.iter().enumerate().map(|(i, x)| 1.0 + 2.0 * x + noise(i + offset)).collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn association_static() -> Rep {
    let query = AssociationQuery::new(VariableId::from_raw(1), [VariableId::from_raw(0)]);
    let study =
        Study::tabular(line_data(0)).query(CausalQuery::Association(query)).build().unwrap();
    Rep {
        label: "association_static",
        family: RouteFamily::Associational,
        study,
        retained: Retained::Table { same: || line_data(0), changed: || line_data(500) },
    }
}

fn association_series(offset: usize) -> TimeSeriesData {
    let n = 96;
    let (mut x, mut y) = (vec![0.0; n], vec![0.0; n]);
    for t in 1..n {
        x[t] = 0.8 * x[t - 1] + noise(t + offset);
        y[t] = 0.5 * x[t] + 0.6 * y[t - 1] + noise(t + 1000 + offset);
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1_000_000_000)
        .unwrap()
}

fn association_query() -> AssociationQuery {
    AssociationQuery::new(VariableId::from_raw(1), [VariableId::from_raw(0)])
}

fn association_temporal() -> Rep {
    let study = Study::series(association_series(0))
        .query(CausalQuery::Association(association_query()))
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    Rep {
        label: "association_temporal",
        family: RouteFamily::Associational,
        study,
        retained: Retained::Series {
            same: || association_series(0),
            changed: || association_series(500),
        },
    }
}

fn association_static_bayesian() -> Rep {
    let study = Study::tabular(line_data(0))
        .query(CausalQuery::Association(association_query()))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200)))
        .build()
        .unwrap();
    Rep {
        label: "association_static_bayesian",
        family: RouteFamily::Associational,
        study,
        retained: Retained::Table { same: || line_data(0), changed: || line_data(500) },
    }
}

fn association_temporal_bayesian() -> Rep {
    let study = Study::series(association_series(0))
        .query(CausalQuery::Association(association_query()))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200)))
        .build()
        .unwrap();
    Rep {
        label: "association_temporal_bayesian",
        family: RouteFamily::Associational,
        study,
        retained: Retained::Series {
            same: || association_series(0),
            changed: || association_series(500),
        },
    }
}

fn discovery_data(shift: usize) -> TabularData {
    let n = 240;
    let z: Vec<f64> = (0..n).map(|i| noise(i + shift)).collect();
    let t: Vec<f64> = (0..n).map(|i| 0.8 * z[i] + 0.3 * noise(i + 1000 + shift)).collect();
    let y: Vec<f64> =
        (0..n).map(|i| 0.9 * t[i] + 0.5 * z[i] + 0.3 * noise(i + 2000 + shift)).collect();
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

/// A structure search compiled through the compiler: PC over three variables.
fn discovery_search() -> Rep {
    let params = StaticDiscoverParams {
        alpha: 0.05,
        max_cond_size: 2,
        fdr: None,
        ci: resolve_ci("parcorr", None).unwrap(),
        screen_pc: false,
        max_subset: None,
        knowledge: BackgroundKnowledge::default(),
    };
    let study = Study::tabular(discovery_data(0))
        .discovery(
            DiscoverySpec::pc(params, "parcorr"),
            vec![VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)],
        )
        .build()
        .unwrap();
    Rep {
        label: "discovery_search",
        family: RouteFamily::Discovery,
        study,
        retained: Retained::Table { same: || discovery_data(0), changed: || discovery_data(500) },
    }
}

fn representatives() -> Vec<Rep> {
    vec![
        static_effect(),
        static_effect_doubly_robust(),
        static_effect_bayesian(),
        static_effect_class(),
        static_response(),
        structural_attribution(),
        temporal_effect(),
        temporal_effect_class(),
        randomized_design(),
        graph_posterior_effect(),
        discovery_search(),
        association_static(),
        association_temporal(),
        association_static_bayesian(),
        association_temporal_bayesian(),
    ]
}

fn prepare(rep: &Rep, found: &mut Findings) -> Option<PreparedStudy> {
    match rep.study.prepare(&ctx()) {
        Ok(prepared) => Some(prepared),
        Err(error) => {
            found.refused("prepare", &error);
            None
        }
    }
}

// ---------------------------------------------------------------------------------------
// Plan view

/// The view agrees with the lowered program layer by layer.
fn view_agrees(found: &mut Findings, stage: &str, view: &PlanView, plan: &PhysicalExecutionPlan) {
    let Some(lowered) = plan.lowered() else {
        found.check(false, format!("{stage}: compile attached no lowered program"));
        return;
    };
    found.check(view.identity() == lowered.identity(), format!("{stage}: view identity"));
    found.check(
        Some(view.identity()) == plan.lowered_program_identity(),
        format!("{stage}: plan identity"),
    );
    found.check(view.route().as_str() == lowered.route(), format!("{stage}: route"));
    found.check(view.depth() == lowered.depth(), format!("{stage}: depth"));
    found.check(
        view.statistical().identity() == lowered.statistical().identity(),
        format!("{stage}: statistical program"),
    );
    found.check(view.evidence() == lowered.evidence(), format!("{stage}: evidence program"));
    found.check(view.claim() == lowered.claim(), format!("{stage}: claim demand"));
    found.check(
        view.uncertainty().reading() == lowered.claim().uncertainty(),
        format!("{stage}: uncertainty reading"),
    );
    found.check(
        view.uncertainty().claim_kind() == lowered.claim().kind(),
        format!("{stage}: claim kind"),
    );
    found.check(
        view.physical().design() == lowered.physical().design(),
        format!("{stage}: physical design"),
    );
    found.check(
        view.physical().operators() == lowered.physical().operators(),
        format!("{stage}: physical operators"),
    );
    found
        .check(view.physical().record() == Some(&plan.record), format!("{stage}: physical record"));
    found.check(
        view.support().coordinate() == lowered.scientific().support_coordinate(),
        format!("{stage}: support coordinate"),
    );
    found.check(
        view.support().licensed() == lowered.scientific().licensed(),
        format!("{stage}: licensed"),
    );
    let json = view.to_json();
    found.check(json["identity"] == lowered.identity().to_hex(), format!("{stage}: JSON identity"));
    found.check(json["route"] == lowered.route(), format!("{stage}: JSON route"));
    found.check(json["family"] == view.family().as_str(), format!("{stage}: JSON family"));
}

fn plan_view_findings(rep: &mut Rep) -> Vec<String> {
    let mut found = Findings::new(rep.label);
    let context = ctx();
    let plan = match rep.study.compile(&context) {
        Ok(plan) => plan,
        Err(error) => {
            found.refused("compile", &error);
            return found.items;
        }
    };
    let view = match plan.plan_view() {
        Ok(view) => view,
        Err(error) => {
            found.check(false, format!("compiled plan has no view: {error}"));
            return found.items;
        }
    };
    found.check(
        view.family() == rep.family,
        format!("family {:?} != {:?}", view.family(), rep.family),
    );
    view_agrees(&mut found, "compiled", &view, &plan);

    // The preflight inspect states what the view states, and withholds what it cannot know.
    match rep.study.inspect() {
        Ok(inspected) => {
            found.check(
                view.support().status() == inspected.support_status,
                "inspect and the view disagree on the support status",
            );
            found.check(
                inspected.identities.program.is_none()
                    && inspected.identities.identification_product.is_none(),
                "inspect reports a program or identification product it has not compiled",
            );
            found.check(
                matches!(inspected.reasoning.uncertainty, SlotAvailability::Unavailable { .. }),
                "inspect publishes an uncertainty slot before any execution",
            );
        }
        Err(error) => found.check(false, format!("inspect failed: {error}")),
    }

    // The study-level view is the compiled view, and the prepared view reads the same program.
    match rep.study.plan_view(&context) {
        Ok(direct) => {
            found.check(direct.identity() == view.identity(), "Study::plan_view identity")
        }
        Err(error) => found.check(false, format!("Study::plan_view failed: {error}")),
    }
    if let Some(prepared) = prepare(rep, &mut found) {
        match prepared.plan_view() {
            Ok(prepared_view) => {
                found.check(
                    prepared_view.identity() == view.identity(),
                    "preparation changed the lowered program identity",
                );
                found.check(
                    prepared_view.operation() == Some(prepared.retained_operation().name()),
                    "the prepared view does not name the retained operation",
                );
                view_agrees(&mut found, "prepared", &prepared_view, prepared.plan());
            }
            Err(error) => found.check(false, format!("prepared handle has no view: {error}")),
        }
    }
    found.items
}

// ---------------------------------------------------------------------------------------
// Uncertainty

fn uncertainty_findings(rep: &mut Rep) -> Vec<String> {
    let mut found = Findings::new(rep.label);
    let context = ctx();
    let Some(prepared) = prepare(rep, &mut found) else { return found.items };
    let result = match prepared.estimate_retained(&context) {
        Ok(result) => result,
        Err(error) => {
            found.refused("estimate", &error);
            return found.items;
        }
    };
    let view = match prepared.plan_view() {
        Ok(view) => view,
        Err(error) => {
            found.check(false, format!("prepared handle has no view: {error}"));
            return found.items;
        }
    };
    let reading = view.uncertainty().reading();
    let realization = view.uncertainty().realization();
    let claim = match prepared.execution_contract(&result, &context) {
        Ok((_, claim)) => claim,
        Err(error) => {
            found.refused("claim", &error);
            return found.items;
        }
    };
    if let SlotAvailability::Available(slot) = &claim.reasoning.uncertainty {
        found.check(!slot.components.is_empty(), "the uncertainty slot declares no component");
        for component in slot.components.iter() {
            found.check(
                !component.target.is_empty(),
                format!("a {} component reports no target", component.source.as_str()),
            );
        }
        found.check(
            reading == UncertaintyReading::None || slot.components.iter().any(|c| !c.omitted),
            format!("every component is omitted though the program reads {}", reading.as_str()),
        );
    } else if let SlotAvailability::Unavailable { reason } = &claim.reasoning.uncertainty {
        found.check(
            reading == UncertaintyReading::None,
            format!("slot unavailable ({reason}) though the program reads {}", reading.as_str()),
        );
    }
    found.check(
        ["calibrated", "scope_not_assessed", "unavailable"].contains(&&*claim.calibration.status),
        format!("calibration axis `{}` is not a declared status", claim.calibration.status),
    );
    match rep.label {
        "static_effect" => {
            found.check(
                reading == UncertaintyReading::Sampling && realization == "frequentist",
                format!("frequentist effect reads {} / {realization}", reading.as_str()),
            );
            found.check(result.posterior.is_none(), "a frequentist effect carries a posterior");
        }
        "static_effect_bayesian" => {
            found.check(
                reading == UncertaintyReading::Posterior && realization == "bayesian",
                format!("Bayesian effect reads {} / {realization}", reading.as_str()),
            );
            found.check(result.posterior.is_some(), "a Bayesian effect carries no posterior");
        }
        _ => {}
    }
    found.items
}

// ---------------------------------------------------------------------------------------
// Decision demand

fn affine_problem(slope: f64) -> DecisionProblem<usize, f64> {
    let utility = AffineUtility::new(vec![0.0, -0.5], vec![0.0, slope]).unwrap();
    DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![])
}

/// A utility that declares no affine form.
struct Quadratic;

impl Utility<usize, f64> for Quadratic {
    fn evaluate_batch(
        &self,
        actions: &[usize],
        outcomes: &[f64],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        for (row, &action) in actions.iter().enumerate() {
            for (col, &state) in outcomes.iter().enumerate() {
                out[row * outcomes.len() + col] = action as f64 * state * state;
            }
        }
        Ok(())
    }
}

fn request(carrier: CarrierDemand) -> DecisionCompileRequest {
    DecisionCompileRequest::new(carrier)
        .with_contract_identity("family-conformance")
        .with_utility_unit("net_margin")
}

fn carrier_refusal(error: &CausalError) -> Option<DecisionCarrierError> {
    let CausalError::Rejected(failure) = error.peeled() else { return None };
    let CompileCause::Decision(refusal) = failure.cause() else { return None };
    Some(**refusal)
}

fn decision_findings(rep: &mut Rep) -> Vec<String> {
    let mut found = Findings::new(rep.label);
    let context = ctx();
    let realization =
        rep.study.plan_view(&context).ok().map(|view| view.uncertainty().realization());
    let plain =
        rep.study.compile(&context).ok().and_then(|plan| plan.lowered().map(|l| l.identity()));
    for carrier in
        [CarrierDemand::Mean, CarrierDemand::ParameterPosterior, CarrierDemand::OutcomeLaw]
    {
        let name = format!("{carrier:?}");
        match rep.study.compile_for_decision(&affine_problem(2.0), &request(carrier), &context) {
            Ok(plan) => {
                let lowered = plan.lowered();
                let decision = lowered.and_then(|l| l.evidence().decision());
                found.check(
                    decision.is_some_and(|d| d.carrier() == carrier),
                    format!("{name}: layer four does not carry the demanded carrier"),
                );
                found.check(
                    lowered.map(|l| l.identity()) != plain || plain.is_none(),
                    format!("{name}: a decision left the program identity unchanged"),
                );
                match plan.plan_view() {
                    Ok(view) => found.check(
                        view.uncertainty().decision_request() == carrier.evidence_request(),
                        format!("{name}: the view does not show the decision's evidence port"),
                    ),
                    Err(error) => {
                        found.check(false, format!("{name}: decision plan has no view: {error}"))
                    }
                }
                found.check(
                    !(carrier == CarrierDemand::ParameterPosterior
                        && realization == Some("frequentist")),
                    "a frequentist realization satisfied a parameter-posterior demand",
                );
            }
            Err(error) => {
                match carrier_refusal(&error) {
                    Some(refusal) => {
                        found.check(
                            refusal.demanded == carrier,
                            format!("{name}: refusal names another carrier"),
                        );
                        found.check(
                            refusal.missing_carrier() == MissingCarrier::of(carrier),
                            format!("{name}: refusal names the wrong missing carrier"),
                        );
                        found.check(
                            DECISION_NEXT_ATTEMPTS
                                .iter()
                                .any(|attempt| attempt.name() == refusal.next_attempt.name()),
                            format!("{name}: refusal names an unregistered next attempt"),
                        );
                        found.check(
                            error.to_string().contains("next attempt:"),
                            format!("{name}: refusal does not print its next attempt"),
                        );
                        if carrier == CarrierDemand::ParameterPosterior {
                            found.check(
                                realization != Some("bayesian"),
                                "a Bayesian realization was refused a parameter-posterior demand",
                            );
                            found.check(
                            refusal.next_attempt.name() == "bayesian_realization"
                                || realization == Some("envelope"),
                            format!("{name}: a sampling realization did not name the Bayesian attempt"),
                        );
                        }
                    }
                    None => found.refused(&format!("{name} decision compile"), &error),
                }
            }
        }
    }
    // A mean carries only a utility affine in the state.
    let opaque: DecisionProblem<usize, f64> =
        DecisionProblem::new(vec![0, 1], Arc::new(Quadratic), vec![]);
    match rep.study.compile_for_decision(&opaque, &request(CarrierDemand::Mean), &context) {
        Ok(_) => found.check(false, "a mean carried a utility that declares no affine form"),
        Err(error) => match carrier_refusal(&error) {
            Some(refusal) => found.check(
                refusal.next_attempt.name() == "distributional_carrier",
                "a nonlinear utility under a mean did not name the distributional carrier",
            ),
            None => found.refused("nonlinear decision compile", &error),
        },
    }
    // The executed answer scores by carrier: a mean is scored, a risk needs an outcome law.
    if let Some(prepared) = prepare(rep, &mut found) {
        if let Ok(result) = prepared.estimate_retained(&context) {
            let published = PublishedCarriers { mean: result.effect(), posterior_mean: None };
            match score_decision(
                DecisionDemand::AffineMean { slope: 2.0, intercept: -0.5 },
                published,
            ) {
                Ok(scored) => {
                    found.check(
                        scored.carrier == EvidencePort::Mean,
                        "a mean demand scored from another port",
                    );
                    found.check(
                        scored.value.is_finite() == published.mean.is_finite(),
                        "the affine score does not follow the published answer",
                    );
                }
                Err(missing) => {
                    found.check(false, format!("a mean demand was refused: {missing:?}"))
                }
            }
            found.check(
                score_decision(DecisionDemand::Risk, published) == Err(MissingCarrier::OutcomeLaw),
                "an outcome risk was scored from a mean",
            );
            found.check(
                score_decision(
                    DecisionDemand::AffinePosterior { slope: 1.0, intercept: 0.0 },
                    published,
                ) == Err(MissingCarrier::ParameterPosterior),
                "a posterior demand was scored without a posterior",
            );
        }
    }
    found.items
}

// ---------------------------------------------------------------------------------------
// Saved result and independent consumer

fn artifact_findings(rep: &mut Rep) -> Vec<String> {
    let mut found = Findings::new(rep.label);
    let context = ctx();
    let Some(prepared) = prepare(rep, &mut found) else { return found.items };
    let result = match prepared.estimate_retained(&context) {
        Ok(result) => result,
        Err(error) => {
            found.refused("estimate", &error);
            return found.items;
        }
    };
    let id = format!("family-conformance-{}", rep.label);
    let bytes = match prepared.encode_contracted_result(&result, &id, &context) {
        Ok(bytes) => bytes,
        Err(error) => {
            found.refused("save", &error);
            return found.items;
        }
    };
    match prepared.encode_contracted_result(&result, &id, &context) {
        Ok(again) => found.check(again == bytes, "saving the same result twice differs"),
        Err(error) => found.check(false, format!("second save failed: {error}")),
    }
    let consumed = match consume_analysis_result(&bytes) {
        Ok(consumed) => consumed,
        Err(error) => {
            found.check(false, format!("the consumer cannot read the saved result: {error}"));
            return found.items;
        }
    };
    // Verified, or restricted by named dependencies: never silently accepted.
    if !consumed.acceptance.accepts_as_claim() {
        found.check(
            !consumed.acceptance.unresolved.is_empty()
                && consumed.acceptance.unresolved.iter().all(|reason| !reason.is_empty()),
            format!(
                "the consumer refuses the claim without naming a dependency: {:?}",
                consumed.acceptance.restriction
            ),
        );
    }
    let Some(contract) = consumed.contract.as_ref() else {
        found.check(false, "the saved result carries no contract section");
        return found.items;
    };
    let Some(section) = contract.claim.as_ref() else {
        found.check(false, "the saved result carries no claim");
        return found.items;
    };
    // The consumer re-derives the seal, the claim kind and the body digest.
    match contract_seal(
        &contract.identities,
        &contract.reasoning,
        &contract.graph_class,
        &contract.structure_source,
        contract.identifier.as_deref(),
        contract.estimator.as_deref(),
    ) {
        Ok(seal) => {
            found.check(seal == contract.seal, "the consumer's seal differs from the saved one")
        }
        Err(error) => found.check(false, format!("seal cannot be re-derived: {error}")),
    }
    found.check(
        claim_kind_name_of_section(&consumed.body, &contract.reasoning.identification)
            == section.kind,
        "the consumer's claim kind rule differs from the saved claim kind",
    );
    found.check(result_digest(&consumed.body).is_ok(), "the saved body has no digest");
    // The producer's claim, re-derived in process, is the claim the consumer reads.
    match prepared.execution_contract(&result, &context) {
        Ok((produced, claim)) => {
            found.check(
                claim.claim_id.as_bytes() == &section.claim_id,
                "the saved claim id is not the producer's claim id",
            );
            found.check(
                claim.kind.as_str() == section.kind,
                format!("claim kind {} saved as {}", claim.kind.as_str(), section.kind),
            );
            found.check(
                claim.value.map(f64::to_bits) == section.value_bits,
                "the saved claim value differs from the producer's",
            );
            let ids = &produced.identities;
            found.check(*ids.target.as_bytes() == contract.identities.target, "target identity");
            found.check(
                *ids.identification.as_bytes() == contract.identities.identification,
                "identification identity",
            );
            found.check(
                ids.program.map(|program| *program.as_bytes()) == Some(contract.identities.program),
                "program identity",
            );
            found.check(
                *ids.data_snapshot.as_bytes() == contract.identities.data_snapshot,
                "data snapshot identity",
            );
        }
        Err(error) => found.check(false, format!("producer claim failed: {error}")),
    }
    // A flipped byte is refused or no longer accepted as the claim.
    let mut tampered = bytes;
    let middle = tampered.len() / 2;
    tampered[middle] ^= 0x40;
    found.check(
        consume_analysis_result(&tampered).map_or(true, |c| !c.acceptance.accepts_as_claim()),
        "a tampered artifact is still accepted as the claim",
    );
    found.items
}

// ---------------------------------------------------------------------------------------
// Reuse

fn identity_delta(
    before: &ContractIdentities,
    after: &ContractIdentities,
) -> BTreeSet<&'static str> {
    let mut delta = BTreeSet::new();
    if before.target != after.target {
        delta.insert("target");
    }
    if before.identification != after.identification {
        delta.insert("identification");
    }
    if before.identification_product != after.identification_product {
        delta.insert("identification_product");
    }
    if before.program != after.program {
        delta.insert("program");
    }
    if before.inference_binding != after.inference_binding {
        delta.insert("inference_binding");
    }
    if before.observation != after.observation {
        delta.insert("observation");
    }
    if before.data_snapshot != after.data_snapshot {
        delta.insert("data_snapshot");
    }
    delta
}

fn identities_of(prepared: &PreparedStudy, found: &mut Findings) -> Option<ContractIdentities> {
    match prepared.contract() {
        Ok(contract) => Some(contract.identities),
        Err(error) => {
            found.check(false, format!("contract unavailable: {error}"));
            None
        }
    }
}

/// Temporal products the log records since `from`, and whether all results were invalidated.
fn logged_since(prepared: &PreparedStudy, from: usize) -> (BTreeSet<String>, bool, bool) {
    let mut products = BTreeSet::new();
    let mut all_results = false;
    let mut other = false;
    for entry in &prepared.temporal_invalidations().entries[from..] {
        match &entry.target {
            InvalidationTarget::TemporalProduct(product) => {
                products.insert(format!("{product:?}"));
            }
            InvalidationTarget::AllResults => all_results = true,
            _ => other = true,
        }
    }
    (products, all_results, other)
}

fn refresh_findings(rep: &Rep, prepared: &mut PreparedStudy, found: &mut Findings) {
    let context = ctx();
    let Some(base) = identities_of(prepared, found) else { return };
    let base_view = prepared.plan_view().ok().map(|view| view.identity());
    let base_effect = prepared.estimate_retained(&context).ok().map(|result| result.effect());
    match &rep.retained {
        Retained::Table { same, changed } => {
            if rep.label == "static_effect" {
                staleness_findings(prepared, same(), changed(), found);
            }
            match rebind_estimate(prepared, same(), &context) {
                Ok(result) => {
                    if let Some(after) = identities_of(prepared, found) {
                        found.check(
                            identity_delta(&base, &after).is_empty(),
                            format!(
                                "refresh with the unchanged data invalidated {:?}",
                                identity_delta(&base, &after)
                            ),
                        );
                    }
                    if rep.label != "static_effect_bayesian" {
                        found.check(
                            base_effect.is_none_or(|effect| same_bits(effect, result.effect())),
                            "refresh with the unchanged data moved the estimate",
                        );
                    }
                }
                Err(error) => found.refused("refresh", &error),
            }
            match rebind_estimate(prepared, changed(), &context) {
                Ok(_) => {
                    if let Some(after) = identities_of(prepared, found) {
                        found.check(
                            identity_delta(&base, &after) == BTreeSet::from(["data_snapshot"]),
                            format!(
                                "refresh with replaced rows invalidated {:?}, not the data snapshot alone",
                                identity_delta(&base, &after)
                            ),
                        );
                    }
                    found.check(
                        prepared.plan_view().ok().map(|view| view.identity()) == base_view,
                        "refresh with replaced rows changed the lowered program",
                    );
                }
                Err(error) => found.refused("refresh", &error),
            }
        }
        Retained::Series { same, changed } => {
            let start = prepared.temporal_invalidations().entries.len();
            match prepared.refresh_series(same(), &context) {
                Ok(_) => {
                    if let Some(after) = identities_of(prepared, found) {
                        found.check(
                            identity_delta(&base, &after).is_empty(),
                            format!(
                                "refresh with the unchanged series invalidated {:?}",
                                identity_delta(&base, &after)
                            ),
                        );
                    }
                    let (products, all_results, other) = logged_since(prepared, start);
                    found.check(
                        products.is_empty() && !all_results && !other,
                        format!(
                            "refresh with the unchanged series logged invalidations {products:?}"
                        ),
                    );
                }
                Err(error) => found.refused("refresh", &error),
            }
            let start = prepared.temporal_invalidations().entries.len();
            match prepared.refresh_series(changed(), &context) {
                Ok(_) => {
                    if let Some(after) = identities_of(prepared, found) {
                        found.check(
                            identity_delta(&base, &after) == BTreeSet::from(["data_snapshot"]),
                            format!(
                                "refresh with a replaced period invalidated {:?}, not the data snapshot alone",
                                identity_delta(&base, &after)
                            ),
                        );
                    }
                    let oracle =
                        TemporalDisposition::of(&BTreeSet::from([TemporalFactor::Period]), false);
                    let expected: BTreeSet<String> =
                        oracle.invalidated.iter().map(|product| format!("{product:?}")).collect();
                    let (products, all_results, other) = logged_since(prepared, start);
                    found.check(
                        products == expected && all_results == !expected.is_empty() && !other,
                        format!(
                            "period refresh logged {products:?}, expected exactly {expected:?}"
                        ),
                    );
                }
                Err(error) => found.refused("refresh", &error),
            }
        }
    }
}

/// The staleness diagnostic names exactly the keys a refresh recomputes.
fn staleness_findings(
    prepared: &PreparedStudy,
    same: TabularData,
    changed: TabularData,
    found: &mut Findings,
) {
    let policy = StalenessPolicy { subsample: 60, permutations: 99, ..StalenessPolicy::default() };
    match prepared.staleness(&same, &policy, &ctx()) {
        Ok(report) => found.check(
            report.changed_keys.is_empty() && report.invalidated.is_empty(),
            format!("staleness of the retained data names keys {:?}", report.changed_keys),
        ),
        Err(error) => found.refused("staleness", &error),
    }
    match prepared.staleness(&changed, &policy, &ctx()) {
        Ok(report) => found.check(
            report
                .changed_keys
                .iter()
                .all(|key| matches!(key, Stage::DataSnapshot | Stage::RowDesign)),
            format!(
                "staleness of replaced rows names keys beyond the data: {:?}",
                report.changed_keys
            ),
        ),
        Err(error) => found.refused("staleness", &error),
    }
}

fn retarget_findings(prepared: &PreparedStudy, found: &mut Findings) {
    let context = ctx();
    let Some(base) = identities_of(prepared, found) else { return };
    let preview = match prepared.preview_transform(TransformIntent::Retarget) {
        Ok(preview) => preview,
        Err(error) => {
            found.check(false, format!("retarget preview failed: {error}"));
            return;
        }
    };
    let base_effect = prepared.estimate_retained(&context).ok().map(|result| result.effect());
    let base_score = prepared.score_reuse_identity().ok().flatten();
    let Some(rows) = prepared.score_table().map(|table| table.n_rows) else {
        found.check(
            preview.refused,
            "retarget previews as possible on a handle with no score table",
        );
        match prepared.retarget(&vec![1.0; 8], &[], &context) {
            Ok(_) => found.check(false, "a handle with no score table retargeted"),
            Err(error) => found.check(
                preview.refusal.as_deref() == Some(error.to_string().as_str()),
                "the retarget preview does not report the refusal the apply raises",
            ),
        }
        return;
    };
    found.check(!preview.refused, "retarget previews as refused on a handle with a score table");
    let tilted: Vec<f64> = (0..rows).map(|row| 1.0 + 0.5 * (row % 3) as f64).collect();
    let mut digests = Vec::new();
    for (name, weights) in [("uniform", vec![1.0; rows]), ("tilted", tilted)] {
        let retargeted = match prepared.retarget(&weights, &[], &context) {
            Ok(result) => result,
            Err(error) => {
                found.refused(&format!("{name} retarget"), &error);
                continue;
            }
        };
        if name == "uniform" {
            found.check(
                base_effect.is_none_or(|effect| {
                    (effect - retargeted.effect()).abs() <= 1e-8 * effect.abs().max(1.0)
                }),
                "uniform weights moved the estimate",
            );
        }
        match prepared.execution_contract(&retargeted, &context) {
            Ok((contract, _)) => {
                let delta = identity_delta(&base, &contract.identities);
                found.check(
                    !delta.contains("data_snapshot")
                        && !delta.contains("observation")
                        && !delta.contains("inference_binding"),
                    format!("{name} retarget invalidated {delta:?}"),
                );
                // Constant weights leave the population unchanged and bind no target weights;
                // only a nonconstant retarget binds them.
                found.check(
                    contract.target_weights.is_some() == (name != "uniform"),
                    format!("{name} retarget binds the wrong target weights"),
                );
                found.check(
                    contract.score_reuse == base_score,
                    format!("{name} retarget changed the score-reuse key"),
                );
                digests.push(contract.target_weights);
            }
            Err(error) => found.check(false, format!("{name} retarget has no contract: {error}")),
        }
    }
    if let [Some(uniform), Some(tilted)] = digests[..] {
        found.check(uniform != tilted, "different weights bound the same target-weights key");
    }
    if let Some(after) = identities_of(prepared, found) {
        found.check(
            identity_delta(&base, &after).is_empty(),
            "a retarget changed the handle's own identities",
        );
    }
}

fn reuse_findings(rep: &mut Rep) -> Vec<String> {
    let mut found = Findings::new(rep.label);
    let Some(mut prepared) = prepare(rep, &mut found) else { return found.items };
    retarget_findings(&prepared, &mut found);
    refresh_findings(rep, &mut prepared, &mut found);
    found.items
}

// ---------------------------------------------------------------------------------------
// Refusals

fn refusal_findings(rep: &mut Rep) -> Vec<String> {
    let mut found = Findings::new(rep.label);
    let context = ctx();
    let Some(mut prepared) = prepare(rep, &mut found) else { return found.items };
    match &rep.retained {
        Retained::Table { .. } => {
            let wrong =
                TabularData::from_f64_columns([("unrelated", &[0.0, 1.0, 2.0, 3.0][..])]).unwrap();
            match rebind_estimate(&mut prepared, wrong.clone(), &context) {
                Ok(_) => found.check(false, "refresh accepted data of another schema"),
                Err(error) => found.refused("refresh on another schema", &error),
            }
            match prepared.estimate(&wrong, &context) {
                Ok(_) => found.check(false, "estimate accepted data of another schema"),
                Err(error) => found.refused("estimate on another schema", &error),
            }
        }
        Retained::Series { .. } => {
            let wrong = TimeSeriesData::from_f64_columns(
                [("unrelated", &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0][..])],
                1,
            )
            .unwrap();
            match prepared.refresh_series(wrong.clone(), &context) {
                Ok(_) => found.check(false, "refresh accepted a series of another schema"),
                Err(error) => found.refused("refresh on another schema", &error),
            }
            match prepared.estimate_series(&wrong, &context) {
                Ok(_) => found.check(false, "estimate accepted a series of another schema"),
                Err(error) => found.refused("estimate on another schema", &error),
            }
        }
    }
    // A refused refresh leaves the handle usable.
    if let Err(error) = prepared.estimate_retained(&context) {
        found.refused("estimate after a refused refresh", &error);
    }
    // A retarget the handle cannot perform, or that is mis-shaped, names what to try.
    if prepared.score_table().is_some() {
        match prepared.retarget(&[1.0], &[], &context) {
            Ok(_) => found.check(false, "retarget accepted weights of the wrong length"),
            Err(error) => found.refused("retarget with mis-shaped weights", &error),
        }
    } else if let Err(error) = prepared.retarget(&[1.0], &[], &context) {
        found.refused("retarget without a score table", &error);
    }
    found.items
}

// ---------------------------------------------------------------------------------------
// Families with their own entry
//
// A cyclic family, an experiment plan and a propagation chain are not built from a `Study`:
// each prepares or compiles through its own public entry and answers with its own result. The
// same six obligations run through that entry.

/// The six shared obligations.
#[derive(Clone, Copy)]
enum Obligation {
    Uncertainty,
    Reuse,
    Decision,
    Artifact,
    PlanView,
    Refusal,
}

/// The public entry a family without a `Study` representative runs through.
#[derive(Clone, Copy)]
enum Entry {
    Cyclic,
    Planning,
    Chain,
}

struct Other {
    label: &'static str,
    family: RouteFamily,
    entry: Entry,
}

fn others() -> Vec<Other> {
    vec![
        Other { label: "cyclic_feedback_pair", family: RouteFamily::Cyclic, entry: Entry::Cyclic },
        Other {
            label: "planning_cluster_power",
            family: RouteFamily::Planning,
            entry: Entry::Planning,
        },
        Other {
            label: "propagation_chain_five_stage",
            family: RouteFamily::PropagationChain,
            entry: Entry::Chain,
        },
    ]
}

fn other_findings(other: &Other, obligation: Obligation) -> Vec<String> {
    let mut found = Findings::new(other.label);
    match other.entry {
        Entry::Cyclic => cyclic_findings(&mut found, obligation),
        Entry::Planning => planning_findings(&mut found, obligation),
        Entry::Chain => chain_findings(&mut found, obligation),
    }
    found.items
}

/// A refusal outside `CausalError` conforms only when its text names the next attempt.
fn names_attempt_text(found: &mut Findings, what: &str, error: &impl Display) {
    let text = error.to_string();
    found.check(
        text.to_lowercase().contains("next attempt"),
        format!("{what} refused without naming a next attempt: {text}"),
    );
}

/// The view of a program held without a physical plan record agrees with the program.
fn view_matches_program(
    found: &mut Findings,
    stage: &str,
    view: &PlanView,
    lowered: &LoweredProgram,
) {
    found.check(view.identity() == lowered.identity(), format!("{stage}: view identity"));
    found.check(view.route().as_str() == lowered.route(), format!("{stage}: route"));
    found.check(view.depth() == lowered.depth(), format!("{stage}: depth"));
    found.check(
        view.statistical().identity() == lowered.statistical().identity(),
        format!("{stage}: statistical program"),
    );
    found.check(view.evidence() == lowered.evidence(), format!("{stage}: evidence program"));
    found.check(view.claim() == lowered.claim(), format!("{stage}: claim demand"));
    found.check(
        view.uncertainty().reading() == lowered.claim().uncertainty(),
        format!("{stage}: uncertainty reading"),
    );
    found.check(
        view.uncertainty().claim_kind() == lowered.claim().kind(),
        format!("{stage}: claim kind"),
    );
    found.check(
        view.physical().design() == lowered.physical().design(),
        format!("{stage}: physical design"),
    );
    found.check(
        view.physical().operators() == lowered.physical().operators(),
        format!("{stage}: physical operators"),
    );
    found.check(
        view.support().coordinate() == lowered.scientific().support_coordinate(),
        format!("{stage}: support coordinate"),
    );
    found.check(
        view.support().licensed() == lowered.scientific().licensed(),
        format!("{stage}: licensed"),
    );
    let json = view.to_json();
    found.check(json["identity"] == lowered.identity().to_hex(), format!("{stage}: JSON identity"));
    found.check(json["route"] == lowered.route(), format!("{stage}: JSON route"));
    found.check(json["family"] == view.family().as_str(), format!("{stage}: JSON family"));
}

/// A published mean is scored by carrier: a mean is scored, a risk needs an outcome law, and a
/// posterior functional needs a parameter posterior.
fn scoring_findings(found: &mut Findings, mean: f64) {
    let published = PublishedCarriers { mean, posterior_mean: None };
    match score_decision(DecisionDemand::AffineMean { slope: 2.0, intercept: -0.5 }, published) {
        Ok(scored) => {
            found.check(
                scored.carrier == EvidencePort::Mean,
                "a mean demand scored from another port",
            );
            found.check(
                scored.value.is_finite() == mean.is_finite(),
                "the affine score does not follow the published answer",
            );
        }
        Err(missing) => found.check(false, format!("a mean demand was refused: {missing:?}")),
    }
    found.check(
        score_decision(DecisionDemand::Risk, published) == Err(MissingCarrier::OutcomeLaw),
        "an outcome risk was scored from a mean",
    );
    found.check(
        score_decision(DecisionDemand::AffinePosterior { slope: 1.0, intercept: 0.0 }, published)
            == Err(MissingCarrier::ParameterPosterior),
        "a posterior demand was scored without a posterior",
    );
}

// ---- Cyclic family

/// `A = 0.4 B + U_A`, `B = 0.5 A + U_B`: the observed world and `do(A = clamp)`.
fn feedback_pair(clamp: f64) -> CyclicFamilyRequest {
    CyclicFamilyRequest::LinearWorlds {
        variables: vec![0, 1],
        model: LinearCyclicWire {
            coefficients: vec![0.0, 0.4, 0.5, 0.0],
            intercept: vec![0.0; 2],
            noise_mean: vec![0.0; 2],
            noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        },
        component: CyclicComponentWire {
            component_id: "feedback-pair".into(),
            source_id: "specification".into(),
            certifying_evidence: vec!["independent-bench-test".into()],
        },
        worlds: vec![
            CyclicWorldWire { world: 0, interventions: vec![], initial_states: Vec::new() },
            CyclicWorldWire {
                world: 1,
                interventions: vec![CyclicInterventionWire::Set { variable: 0, value: clamp }],
                initial_states: Vec::new(),
            },
        ],
        certificate: CertificateDemandWire::Required,
    }
}

/// Equilibrium mean of one world of a cyclic report.
fn world_mean(report: &CyclicFamilyReport, world: usize) -> Option<Vec<f64>> {
    let CyclicFamilyAnswer::Worlds(worlds) = &report.answer else { return None };
    match &worlds.get(world)?.outcome {
        CyclicWorldOutcomeAnswer::GaussianLaw { mean, .. } => Some(mean.clone()),
        _ => None,
    }
}

fn cyclic_findings(found: &mut Findings, obligation: Obligation) {
    let context = ctx();
    let request = feedback_pair(0.0);
    let prepared = match Study::prepare_cyclic_family(&request, &context) {
        Ok(prepared) => prepared,
        Err(error) => {
            names_attempt_text(found, "prepare", &error);
            return;
        }
    };
    let view = prepared.plan_view();
    match obligation {
        Obligation::PlanView => {
            found.check(view.family() == RouteFamily::Cyclic, "the view names another family");
            view_matches_program(found, "prepared", &view, prepared.lowered());
            let inspected = prepared.inspect();
            found.check(
                inspected.lowered_program_identity == view.identity(),
                "inspect names another program than the view",
            );
            found.check(inspected.preparation_evaluations == 0, "preparing evaluated the model");
            found.check(
                view.support().status().is_none() && !view.support().licensed(),
                "a cyclic family claims a support-matrix cell",
            );
            match prepared.contract() {
                Ok(contract) => found.check(
                    contract.lowered_program_identity == view.identity(),
                    "the contract names another program than the view",
                ),
                Err(error) => found.check(false, format!("contract unavailable: {error}")),
            }
        }
        Obligation::Uncertainty => {
            let reading = view.uncertainty().reading();
            match prepared.estimate(&context) {
                Ok(report) => {
                    found.check(
                        report.lowered_program_identity() == Some(view.identity()),
                        "the report does not carry the program identity",
                    );
                    found.check(
                        reading == UncertaintyReading::None,
                        format!("a specified model reads {}", reading.as_str()),
                    );
                    match &report.answer {
                        CyclicFamilyAnswer::Worlds(worlds) => {
                            found.check(worlds.len() == 2, "the report omits a declared world");
                            for world in worlds {
                                found.check(
                                    world.certificate.certified,
                                    format!("world {} is not certified", world.world),
                                );
                            }
                        }
                        other => {
                            found.check(false, format!("typed worlds expected, got {other:?}"))
                        }
                    }
                }
                Err(error) => names_attempt_text(found, "estimate", &error),
            }
        }
        Obligation::Reuse => {
            let Ok(base) = prepared.contract().map(|contract| contract.identities) else {
                found.check(false, "contract unavailable");
                return;
            };
            match prepared.rebind(&request, &context) {
                Ok(again) => {
                    found.check(
                        again.contract().is_ok_and(|contract| contract.identities == base),
                        "rebinding the unchanged request moved an identity",
                    );
                    found.check(
                        again.lowered().identity() == view.identity(),
                        "rebinding the unchanged request moved the program",
                    );
                }
                Err(error) => names_attempt_text(found, "rebind", &error),
            }
            let changed = feedback_pair(2.0);
            match prepared.rebind(&changed, &context) {
                Ok(rebound) => {
                    if let Ok(contract) = rebound.contract() {
                        let delta = identity_delta(&base, &contract.identities);
                        found.check(delta.contains("target"), "a changed world kept the target");
                        found.check(
                            !delta.contains("data_snapshot")
                                && !delta.contains("observation")
                                && !delta.contains("inference_binding"),
                            format!("a changed world invalidated {delta:?}"),
                        );
                    }
                    match rebound.estimate(&context) {
                        Ok(report) => found.check(
                            world_mean(&report, 1)
                                .is_some_and(|mean| (mean[1] - 1.0).abs() <= 1e-12),
                            "the rebound world did not solve do(A = 2) to B = 1",
                        ),
                        Err(error) => names_attempt_text(found, "rebound estimate", &error),
                    }
                }
                Err(error) => names_attempt_text(found, "rebind", &error),
            }
        }
        Obligation::Decision => {
            found.check(
                prepared.lowered().evidence().decision().is_none(),
                "a family that demands no decision carries a decision carrier",
            );
            match prepared.estimate(&context) {
                Ok(report) => match world_mean(&report, 1) {
                    Some(mean) => scoring_findings(found, mean[1]),
                    None => found.check(false, "the report carries no equilibrium mean"),
                },
                Err(error) => names_attempt_text(found, "estimate", &error),
            }
        }
        Obligation::Artifact => {
            let (report, bytes) = match prepared.export(&context) {
                Ok(exported) => exported,
                Err(error) => {
                    names_attempt_text(found, "export", &error);
                    return;
                }
            };
            found.check(
                prepared.inspect().identity == report.identity,
                "the handle and its report name different requests",
            );
            // A consumer on another context verifies the contract from bytes and replays.
            let other = ExecutionContext::for_tests(1);
            match consume_cyclic_family_contract(&bytes, &report.identity, &other) {
                Ok((contract, replayed)) => {
                    found.check(
                        serde_json::to_value(&replayed).ok() == serde_json::to_value(&report).ok(),
                        "the consumer's replay differs from the producer's report",
                    );
                    found.check(
                        contract.request_identity == report.identity,
                        "the consumer's contract names another request",
                    );
                    found.check(
                        prepared
                            .contract()
                            .is_ok_and(|own| own.identities == contract.contract_identities()),
                        "the consumer's identities are not the producer's",
                    );
                    let mut tampered = bytes.clone();
                    let middle = tampered.len() / 2;
                    tampered[middle] ^= 0x40;
                    found.check(
                        consume_cyclic_family_contract(&tampered, &report.identity, &other).map_or(
                            true,
                            |(changed, again)| {
                                changed != contract
                                    || serde_json::to_value(&again).ok()
                                        != serde_json::to_value(&report).ok()
                            },
                        ),
                        "a tampered artifact is still accepted as the claim",
                    );
                }
                Err(error) => found
                    .check(false, format!("the consumer cannot read the saved family: {error}")),
            }
            found.check(
                consume_cyclic_family(&bytes, &"0".repeat(64), &other).is_err(),
                "a substituted identity was accepted",
            );
        }
        Obligation::Refusal => {
            // A family another entry owns, a substituted identity and a malformed one.
            if let Err(error) = Study::prepare_recovered_linear(&request, &context) {
                names_attempt_text(found, "recovered-linear entry on a specified family", &error);
            } else {
                found.check(false, "the recovered-linear entry accepted a specified family");
            }
            if let Err(error) = Study::prepare_experimental_recovery(&request, &context) {
                names_attempt_text(found, "recovery entry on a specified family", &error);
            } else {
                found.check(false, "the recovery entry accepted a specified family");
            }
            if let Ok((_, bytes)) = prepared.export(&context) {
                match consume_cyclic_family(&bytes, &"0".repeat(64), &context) {
                    Ok(_) => found.check(false, "a substituted identity was accepted"),
                    Err(error) => names_attempt_text(found, "consume on another identity", &error),
                }
                match consume_cyclic_family(&bytes, "short", &context) {
                    Ok(_) => found.check(false, "a malformed identity was accepted"),
                    Err(error) => {
                        names_attempt_text(found, "consume on a malformed identity", &error);
                    }
                }
            }
        }
    }
}

// ---- Experiment plan

fn cluster_plan(attrition: f64) -> PlanningRequest {
    PlanningRequest::new(
        PlannedDesign::Cluster { mean_cluster_size: 12.0, icc: 0.05, cluster_size_cv: 0.0 },
        PlanningAssumptions::continuous(1.0).with_effect(0.3).with_attrition(attrition),
        PlanningCriterion::Power { target: 0.8 },
        PlanningGoal::Evaluate { units: 40 },
    )
    .with_simulation(200, 3)
}

fn planning_findings(found: &mut Findings, obligation: Obligation) {
    let context = ctx();
    let request = cluster_plan(0.1);
    let plan = match compile_planning(&request, &context) {
        Ok(plan) => plan,
        Err(error) => {
            found.refused("planning compile", &error);
            return;
        }
    };
    let view = plan.plan_view();
    match obligation {
        Obligation::PlanView => {
            found.check(view.family() == RouteFamily::Planning, "the view names another family");
            view_matches_program(found, "compiled", &view, plan.lowered());
            found.check(view.identity() == plan.identity(), "plan identity");
            found.check(
                view.to_json()["evidence"]["planning"]["request_identity"]
                    == request.identity().to_hex(),
                "the view names another request than the one compiled",
            );
            found.check(
                view.physical().backend().is_none(),
                "a standalone plan reports a physical backend",
            );
            found.check(
                plan.simulation_quote().is_some(),
                "compile did not quote the declared simulation",
            );
        }
        Obligation::Uncertainty => match plan.execute(&context) {
            Ok(done) => {
                let reading = view.uncertainty().reading();
                let realization = view.uncertainty().realization();
                found.check(
                    done.program_identity == view.identity(),
                    "the result names another program than the view",
                );
                found.check(
                    reading == UncertaintyReading::Sampling && realization == "frequentist",
                    format!("a power plan reads {} / {realization}", reading.as_str()),
                );
                found.check(done.result.analytic.value.is_finite(), "the analytic power is unset");
                match &done.result.simulation {
                    Some(simulated) => {
                        found.check(
                            simulated.monte_carlo_se.is_finite() && simulated.monte_carlo_se > 0.0,
                            "the simulated criterion publishes no Monte Carlo error",
                        );
                        found.check(
                            (simulated.value - done.result.analytic.value).abs()
                                <= 4.0 * simulated.monte_carlo_se + 0.05,
                            "the analytic and simulated criteria disagree beyond Monte Carlo error",
                        );
                    }
                    None => {
                        found.check(false, "a declared simulation published no characteristics");
                    }
                }
            }
            Err(error) => found.refused("planning execute", &error),
        },
        Obligation::Reuse => {
            let again = ExecutionContext::for_tests(99);
            match compile_planning(&request.clone(), &again) {
                Ok(recompiled) => found.check(
                    recompiled.identity() == plan.identity(),
                    "recompiling an unchanged request moved the program identity",
                ),
                Err(error) => found.refused("recompile", &error),
            }
            let changed = PlanningRequest {
                assumptions: request.assumptions.with_attrition(0.2),
                ..request.clone()
            };
            let (Ok(first), Ok(second)) = (plan.execute(&context), plan.execute(&context)) else {
                found.check(false, "the plan did not execute");
                return;
            };
            found.check(first == second, "executing one plan twice differs");
            match compile_planning(&changed, &context) {
                Ok(other) => {
                    found.check(
                        other.identity() != plan.identity(),
                        "a changed attrition kept the program identity",
                    );
                    match other.execute(&context) {
                        Ok(done) => {
                            found.check(
                                done.result.request_identity == changed.identity()
                                    && done.result.request_identity
                                        != first.result.request_identity,
                                "the result does not name the changed request",
                            );
                            found.check(
                                done.result.analysed_individuals
                                    < first.result.analysed_individuals,
                                "more attrition did not shrink the analysed individuals",
                            );
                        }
                        Err(error) => found.refused("changed plan execute", &error),
                    }
                }
                Err(error) => found.refused("changed compile", &error),
            }
        }
        Obligation::Decision => {
            found.check(
                plan.lowered().evidence().decision().is_none(),
                "a plan that demands no decision carries a decision carrier",
            );
            match plan.execute(&context) {
                Ok(done) => scoring_findings(found, done.result.analytic.value),
                Err(error) => found.refused("planning execute", &error),
            }
        }
        Obligation::Artifact => {
            // A plan binds no data and has no byte artifact: a consumer holding the request
            // compiles it afresh, on another context, and re-derives the program and the claim.
            let Ok(done) = plan.execute(&context) else {
                found.check(false, "the plan did not execute");
                return;
            };
            let other = ExecutionContext::for_tests(99);
            match compile_planning(&request.clone(), &other) {
                Ok(fresh) => {
                    found.check(
                        fresh.identity() == done.program_identity,
                        "the consumer's program is not the executed program",
                    );
                    found.check(
                        fresh.request().identity() == done.result.request_identity,
                        "the consumer's request is not the executed request",
                    );
                    found.check(
                        fresh
                            .lowered()
                            .evidence()
                            .planning()
                            .is_some_and(|p| p.request_identity() == done.result.request_identity),
                        "layer four carries another request than the result",
                    );
                }
                Err(error) => found.refused("consumer compile", &error),
            }
            let substituted = PlanningRequest {
                criterion: PlanningCriterion::Power { target: 0.9 },
                ..request.clone()
            };
            match compile_planning(&substituted, &other) {
                Ok(fresh) => found.check(
                    fresh.identity() != done.program_identity,
                    "a substituted criterion kept the program identity",
                ),
                Err(error) => found.refused("substituted compile", &error),
            }
            found.check(!done.result.calibration.is_empty(), "the result declares no calibration");
        }
        Obligation::Refusal => {
            let assumptions = PlanningAssumptions::continuous(1.0).with_effect(0.3);
            for design in [
                PlannedDesign::Interference { exposure: None },
                PlannedDesign::OutcomeAdaptive { rule: None },
            ] {
                let unplannable = PlanningRequest::new(
                    design,
                    assumptions,
                    PlanningCriterion::Power { target: 0.8 },
                    PlanningGoal::Evaluate { units: 100 },
                );
                match compile_planning(&unplannable, &context) {
                    Ok(_) => found.check(false, "an unplannable design compiled"),
                    Err(error) => found.refused("unplannable design", &error),
                }
            }
            let tiny = ExecutionContext::for_tests(1).with_memory_limit(Some(10_000));
            match compile_planning(&request, &tiny) {
                Ok(_) => found.check(false, "a store the pool cannot hold compiled"),
                Err(error) => found.refused("simulation store", &error),
            }
        }
    }
}

// ---- Propagation chain

const CHAIN_ROWS: usize = 1597;
const CHAIN_STRIDE: usize = 987;

fn chain_grid(mean: f64, sd: f64) -> Vec<f64> {
    (0..CHAIN_ROWS).map(|i| mean + sd * normal_ppf((i as f64 + 0.5) / CHAIN_ROWS as f64)).collect()
}

fn chain_partner(mean: f64, sd: f64) -> Vec<f64> {
    let base = chain_grid(mean, sd);
    (0..CHAIN_ROWS).map(|i| base[(i * CHAIN_STRIDE) % CHAIN_ROWS]).collect()
}

/// Measurement mapping, transport, a derived net, a nonlinear utility and a preposterior update.
fn chain_draft(reading: ChainReading, meaning: LawMeaning) -> PropagationChainRecordWire {
    let theta = chain_grid(0.5, 0.6);
    PropagationChain::draft(
        TableConstruction::JointDraws,
        reading,
        "update",
        vec![
            ChainSourceWire::draws("effect", meaning, "t", &theta)
                .with_factor(SharedFactorKind::Row, "trial-rows"),
            ChainSourceWire::draws("cost", meaning, "t", &chain_partner(0.2, 0.3))
                .with_factor(SharedFactorKind::Prior, "cost-prior"),
        ],
        vec![
            ChainStageWire::new(
                "mapped",
                StageKind::MeasurementMapping,
                &["effect"],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(0.8),
                    ChainCoefficientWire::fixed(0.1),
                ),
            ),
            ChainStageWire::new(
                "transported",
                StageKind::Transport,
                &["mapped"],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(0.9),
                    ChainCoefficientWire::fixed(0.0),
                ),
            ),
            ChainStageWire::new(
                "net",
                StageKind::DerivedQuantity,
                &["transported", "cost"],
                ChainOpWire::binary("difference"),
            ),
            ChainStageWire::new(
                "utility",
                StageKind::DecisionUtility,
                &["net"],
                ChainOpWire::unary(ChainUnaryWire::exponential_utility(1.0)),
            )
            .with_decision_reference(0.0),
            ChainStageWire::new(
                "update",
                StageKind::PreposteriorUpdate,
                &["effect", "utility"],
                ChainOpWire::preposterior(1.0, 4.0, 0.0, &chain_partner(0.0, 1.0)),
            ),
        ],
    )
}

fn chain_names_attempt(found: &mut Findings, what: &str, refusal: &ChainRefusal) {
    found.check(
        !refusal.next_attempt().name().is_empty(),
        format!("{what} refused without naming a next attempt: {refusal:?}"),
    );
}

fn chain_findings(found: &mut Findings, obligation: Obligation) {
    let context = ctx();
    let plan = match compile_propagation_chain(
        chain_draft(ChainReading::Spread, LawMeaning::Posterior),
        &context,
    ) {
        Ok(plan) => plan,
        Err(refusal) => {
            chain_names_attempt(found, "compile", &refusal);
            return;
        }
    };
    let view = plan.plan_view();
    let chain = match plan.execute(&context) {
        Ok(chain) => chain,
        Err(refusal) => {
            chain_names_attempt(found, "execute", &refusal);
            return;
        }
    };
    match obligation {
        Obligation::PlanView => {
            found.check(
                view.family() == RouteFamily::PropagationChain,
                "the view names another family",
            );
            view_matches_program(found, "compiled", &view, plan.lowered());
            found.check(view.identity() == plan.identity(), "plan identity");
            found.check(
                chain.lowered_program_identity() == view.identity(),
                "the sealed chain stamps another program than the view",
            );
            let stages = view
                .operations()
                .iter()
                .filter(|operation| operation.name() == "propagate")
                .count();
            found.check(stages == 5, format!("the view lists {stages} of 5 propagated stages"));
        }
        Obligation::Uncertainty => {
            let reading = view.uncertainty().reading();
            let realization = view.uncertainty().realization();
            found.check(
                reading != UncertaintyReading::Posterior || realization == "bayesian",
                format!("a posterior chain reads {realization}"),
            );
            found.check(
                reading != UncertaintyReading::Sampling || realization == "frequentist",
                format!("a sampling chain reads {realization}"),
            );
            found.check(chain.label() == "uncalibrated_spread", "the spread carries another label");
            let update = chain.answer();
            found.check(
                update
                    .and_then(|answer| answer.value("value_of_information"))
                    .is_some_and(f64::is_finite),
                "the update publishes no finite value of information",
            );
            found.check(
                update.and_then(|answer| answer.value("probability_decision_changes")).is_some(),
                "the update publishes no probability of changing the decision",
            );
            // An interval of the composition is closed until its coverage is measured.
            match chain.interval() {
                Ok(()) => found.check(false, "an interval of a composition was published"),
                Err(refusal) => found.check(
                    refusal.next_attempt().name() == "measure_composed_coverage",
                    "the closed interval does not name the coverage measurement",
                ),
            }
            found.check(chain.posterior().is_err(), "a spread reading published a posterior");
        }
        Obligation::Reuse => {
            let dependents = plan.dependents_of("cost");
            match chain.withdraw("cost", &context) {
                Ok(withdrawn) => {
                    for result in withdrawn.results() {
                        let reached = dependents.contains(&result.stage.as_str());
                        found.check(
                            reached == (result.status == STATUS_INVALIDATED),
                            format!("withdrawing `cost` mis-marked stage `{}`", result.stage),
                        );
                        if !reached {
                            found.check(
                                Some(result) == chain.stage(&result.stage),
                                format!(
                                    "an unreached stage `{}` changed its numbers",
                                    result.stage
                                ),
                            );
                        }
                    }
                    found.check(
                        withdrawn.lowered_program_identity() != chain.lowered_program_identity(),
                        "a withdrawn source kept the program identity",
                    );
                    found.check(
                        chain.stage("net").is_some_and(|net| net.status == STATUS_CURRENT),
                        "withdrawing a source modified the chain it was called on",
                    );
                }
                Err(refusal) => chain_names_attempt(found, "withdraw", &refusal),
            }
            let tilt: Vec<f64> = chain_grid(0.5, 0.6).iter().map(|effect| 0.5 * effect).collect();
            match chain.reweight("effect", &tilt, &context) {
                Ok(reweighted) => found.check(
                    reweighted
                        .stage("mapped")
                        .is_some_and(|mapped| mapped.status == STATUS_REWEIGHTED),
                    "re-weighting a source did not re-weight its dependents",
                ),
                Err(refusal) => chain_names_attempt(found, "reweight", &refusal),
            }
        }
        Obligation::Decision => {
            found.check(
                plan.lowered().evidence().decision().is_none(),
                "a chain that demands no decision carries a decision carrier",
            );
            // A decision updated by a future study needs posterior draws: a sampling law is
            // refused naming the attempt that supplies them.
            match PropagationChain::seal(
                chain_draft(ChainReading::Spread, LawMeaning::Sampling),
                &context,
            ) {
                Ok(_) => found.check(false, "a sampling law was updated as a posterior"),
                Err(refusal) => found.check(
                    refusal.next_attempt().name() == "supply_posterior_draws",
                    "the refused update does not name the posterior draws",
                ),
            }
            match chain.answer().and_then(|answer| answer.value("value_of_information")) {
                Some(value) => scoring_findings(found, value),
                None => found.check(false, "the chain published no value to score"),
            }
        }
        Obligation::Artifact => {
            let bytes = match chain.export() {
                Ok(bytes) => bytes,
                Err(refusal) => {
                    chain_names_attempt(found, "export", &refusal);
                    return;
                }
            };
            match consume_propagation_chain(&bytes, Some(chain.identity()), &context) {
                Ok(consumed) => found.check(
                    consumed.results == chain.results(),
                    "the consumer's replay differs from the producer's stored results",
                ),
                Err(refusal) => {
                    found.check(false, format!("the consumer cannot read the chain: {refusal:?}"));
                }
            }
            found.check(
                chain.declaration_identity().is_ok_and(|identity| identity == chain.identity()),
                "the sealed identity is not the declaration identity",
            );
            let mut tampered = bytes.clone();
            let middle = tampered.len() / 2;
            tampered[middle] ^= 0x40;
            found.check(
                consume_propagation_chain(&tampered, Some(chain.identity()), &context)
                    .map_or(true, |consumed| consumed.results != chain.results()),
                "a tampered chain is still accepted",
            );
            found.check(
                consume_propagation_chain(&bytes, Some(&"0".repeat(64)), &context).is_err(),
                "a substituted identity was accepted",
            );
        }
        Obligation::Refusal => {
            match PropagationChain::seal(
                chain_draft(ChainReading::Spread, LawMeaning::Sampling),
                &context,
            ) {
                Ok(_) => found.check(false, "a sampling law was updated as a posterior"),
                Err(refusal) => chain_names_attempt(found, "a sampling-law update", &refusal),
            }
            if let Err(refusal) = chain.interval() {
                chain_names_attempt(found, "the composed interval", &refusal);
            }
            if let Err(refusal) = chain.posterior() {
                chain_names_attempt(found, "a posterior reading of a spread", &refusal);
            }
            match chain.withdraw("no-such-source", &context) {
                Ok(_) => found.check(false, "a source the chain does not have was withdrawn"),
                Err(refusal) => {
                    chain_names_attempt(found, "withdrawing an unknown source", &refusal);
                }
            }
            match chain.reweight("effect", &[1.0], &context) {
                Ok(_) => found.check(false, "re-weighting accepted one weight for every row"),
                Err(refusal) => chain_names_attempt(found, "mis-shaped log weights", &refusal),
            }
            let scoped = context.enter_lowered_dispatch(plan.identity());
            match compile_propagation_chain(
                chain_draft(ChainReading::Spread, LawMeaning::Posterior),
                &scoped,
            ) {
                Ok(_) => found.check(false, "a chain compiled inside a lowered dispatch"),
                Err(refusal) => chain_names_attempt(found, "a nested compile", &refusal),
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Row table

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn quoted(value: &str) -> Vec<String> {
    value.split('"').skip(1).step_by(2).map(str::to_owned).collect()
}

/// Route ids of the public route table, in file order.
fn manifest_route_ids() -> Vec<String> {
    let text = fs::read_to_string(workspace_root().join("parity/capabilities/route_manifest.toml"))
        .expect("route manifest");
    let mut ids = Vec::new();
    let mut in_route = false;
    for line in text.lines() {
        if line.starts_with("[[") {
            in_route = line == "[[route]]";
        } else if in_route {
            if let Some(value) = line.strip_prefix("id = ") {
                ids.push(quoted(value).remove(0));
            }
        }
    }
    ids
}

type Row = BTreeMap<String, Vec<String>>;

fn conformance_rows() -> Vec<Row> {
    let text =
        fs::read_to_string(workspace_root().join("parity/capabilities/family_conformance.toml"))
            .expect("family conformance table");
    let mut rows: Vec<Row> = Vec::new();
    for line in text.lines() {
        if line == "[[family]]" {
            rows.push(Row::new());
        } else if let (Some(row), Some((key, value))) = (rows.last_mut(), line.split_once(" = ")) {
            row.insert(key.to_owned(), quoted(value));
        }
    }
    rows
}

/// Family names declared by `RouteFamily`, read from its source so a new variant needs a row.
fn declared_families() -> BTreeSet<String> {
    let source = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/planner/plan_view.rs"),
    )
    .expect("plan view source");
    let body = source
        .split("pub enum RouteFamily {")
        .nth(1)
        .and_then(|rest| rest.split("\n}").next())
        .expect("RouteFamily declaration");
    body.lines()
        .filter_map(|line| line.strip_prefix("    "))
        .filter(|line| line.ends_with(',') && line.chars().next().is_some_and(char::is_uppercase))
        .map(|line| {
            let mut name = String::new();
            for (index, ch) in line.trim_end_matches(',').chars().enumerate() {
                if ch.is_uppercase() && index > 0 {
                    name.push('_');
                }
                name.extend(ch.to_lowercase());
            }
            name
        })
        .collect()
}

#[test]
fn every_route_in_the_route_table_has_a_conformance_row() {
    let manifest = manifest_route_ids();
    assert!(manifest.len() > 200, "the route table parsed to {} routes", manifest.len());
    let mut listed: BTreeMap<String, String> = BTreeMap::new();
    for row in conformance_rows() {
        let family = row["id"][0].clone();
        for route in &row["routes"] {
            if let Some(other) = listed.insert(route.clone(), family.clone()) {
                panic!("route `{route}` is in the rows of `{other}` and `{family}`");
            }
        }
    }
    let missing: Vec<&String> = manifest.iter().filter(|id| !listed.contains_key(*id)).collect();
    assert!(
        missing.is_empty(),
        "routes without a conformance row (run scripts/generate_family_conformance.py): {missing:?}"
    );
    let manifest: BTreeSet<&String> = manifest.iter().collect();
    let stale: Vec<&String> = listed.keys().filter(|id| !manifest.contains(id)).collect();
    assert!(stale.is_empty(), "conformance rows for routes that no longer exist: {stale:?}");
}

#[test]
fn every_route_family_has_a_row_that_is_covered_or_states_its_gap() {
    let rows = conformance_rows();
    let named: BTreeSet<String> = rows.iter().map(|row| row["id"][0].clone()).collect();
    assert_eq!(named, declared_families(), "one row per route family");
    let source = include_str!("family_contract_conformance.rs");
    let studies = on_large_stack(|| {
        let mut studies: Vec<(RouteFamily, &'static str)> =
            representatives().into_iter().map(|rep| (rep.family, rep.label)).collect();
        studies.extend(others().into_iter().map(|other| (other.family, other.label)));
        studies
    });
    for row in &rows {
        let family = &row["id"][0];
        let status = row["status"][0].as_str();
        let present: BTreeSet<&str> =
            studies.iter().filter(|(f, _)| f.as_str() == family).map(|(_, label)| *label).collect();
        match status {
            "covered" => {
                let listed: BTreeSet<&str> =
                    row["representatives"].iter().map(String::as_str).collect();
                assert!(!listed.is_empty(), "`{family}` is covered by no study");
                assert_eq!(listed, present, "`{family}` representatives differ from the suite");
                for obligation in [
                    "uncertainty_ports",
                    "reuse",
                    "decision_demand",
                    "artifact_consumer",
                    "plan_view",
                    "refusal_next_attempt",
                ] {
                    let test = &row[obligation][0];
                    assert!(
                        source.contains(&format!("fn {test}()")),
                        "`{family}` names `{test}` for {obligation}, which this suite does not define"
                    );
                }
            }
            "gap" => {
                assert!(present.is_empty(), "`{family}` declares a gap but the suite runs a study");
                for key in ["gap", "next_step"] {
                    assert!(!row[key][0].trim().is_empty(), "`{family}` gap states no {key}");
                }
            }
            other => panic!("`{family}` has status `{other}`"),
        }
    }
}

// ---------------------------------------------------------------------------------------
// Obligations

#[test]
fn uncertainty_is_published_through_typed_ports_or_refused_naming_the_next_attempt() {
    run_obligation(uncertainty_findings, Obligation::Uncertainty);
}

#[test]
fn reuse_invalidates_exactly_the_dependency_keys() {
    run_obligation(reuse_findings, Obligation::Reuse);
}

#[test]
fn a_decision_demand_is_satisfied_or_refused_by_carrier() {
    run_obligation(decision_findings, Obligation::Decision);
}

#[test]
fn a_saved_result_is_re_derived_by_an_independent_consumer() {
    run_obligation(artifact_findings, Obligation::Artifact);
}

#[test]
fn the_plan_view_agrees_with_the_lowered_program() {
    run_obligation(plan_view_findings, Obligation::PlanView);
}

#[test]
fn every_refusal_names_its_next_attempt() {
    run_obligation(refusal_findings, Obligation::Refusal);
}

/// Bind `data` on a new branch, materialize and estimate it, then adopt the branch as the handle.
/// A failure leaves the handle unchanged, as the immutable lifecycle does.
fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: antecedent_data::TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}
