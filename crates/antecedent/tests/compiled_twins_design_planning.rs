//! The compiled planning and ranking programs and the direct kernel entries they replace give
//! the same results on the kernels' own fixtures.
//!
//! Each compiled program calls the same kernel with the same inputs, so every report, plan,
//! ranking and flag is compared whole (`PartialEq` where the type has it, and the `Debug`
//! rendering, which distinguishes every `f64` bit pattern but NaN payloads). Each family also
//! reaches a hand-derived or independently computed value through the compiled route, and
//! names its route and support unit.
//!
//! These results carry no sampling interval, so there is no calibration basis to compare.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

#[path = "../../antecedent-estimate/tests/common/mod.rs"]
mod common;

use std::sync::Arc;

use antecedent::analysis::design_planning_programs::{
    ChoicePosterior, compile_choice_study_ranking, compile_design_ranking,
    compile_graph_identification, compile_identification_repair, compile_structure_identification,
    compile_study_addition_plan, compile_z_transport_evidence_plan,
    compile_z_transport_failure_snapshot,
};
use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceCatalogDelta,
    EvidenceKind, EvidenceRegime, ExecutionContext, InterventionAssignment, QueryId, RegimeBinding,
    RegimeId, RegimeKind, SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_design::choice_evsi::{
    StudyCandidate as ChoiceStudyCandidate, rank_hierarchical_studies, rank_studies,
};
use antecedent_design::repair::RepairClassification;
use antecedent_design::structure_identification::{
    GraphIdentFlag, GraphPosteriorSample, identified_under_intervention, identifies,
    structure_posterior,
};
use antecedent_design::{
    BackdoorRepairFamily, CandidateDesign, DesignCost, DesignEvaluationContext, DesignObjective,
    DesignRankConfig, DesignRanker, DurableStudyCandidate, ExpectedEvidence, ExperimentPlan,
    MeasurementPlan, RepairLimits, RepairObjective, StudyCandidate, StudyCost,
    StudyCostDeclaration, StudyKind, TransportEvidenceCandidate, UnitRules, ZTransportPlanSpec,
    plan_studies, plan_z_transport_evidence, repair, snapshot_z_transport_failure,
};
use antecedent_graph::{Admg, Dag, DenseNodeId, SelectionDiagram};
use antecedent_identify::{SidLimits, StudyPlanLimits, StudyPlanRoute, ZTransportQuery};
use antecedent_prob::WeightedGraphSamples;
use antecedent_stats::adaptive_inference::HierarchicalChoicePrior;
use common::mz_fixture::{X, Y as MZ_Y, Z1 as MZ_Z1, Z2 as MZ_Z2, evidence, graph, query, sources};
use common::z_scm::vid;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn vars(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(v).collect::<Vec<_>>().into()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

/// The compiled program names its family as its route and reads the support decision of the
/// family's unit.
macro_rules! assert_route {
    ($plan:expr, $route:literal, $unit:literal) => {{
        assert_eq!($plan.lowered().route(), $route);
        let support = $plan.plan_view().unit_support().expect("the route is bound to a unit");
        assert_eq!(support.unit(), $unit);
    }};
}

// --- identification repair (repair_facade.rs back-door fixture) ------------------------------

const T: u32 = 0;
const Y: u32 = 1;
const Z1: u32 = 2;
const Z2: u32 = 3;

fn backdoor_family() -> BackdoorRepairFamily {
    let mut graph = Dag::with_variables(7);
    for (from, to) in [(Z1, T), (Z1, Y), (Z2, T), (Z2, Y), (T, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    BackdoorRepairFamily::try_new(graph, v(T), v(Y), "pop", [v(T), v(Y)], &[]).unwrap()
}

fn observation(label: &str, measured: &[u32], units: u64) -> DurableStudyCandidate {
    DurableStudyCandidate {
        label: Arc::from(label),
        kind: StudyKind::Observation,
        population: Arc::from("pop"),
        interventions: vars(&[]),
        measured: vars(measured),
        joint_measurement: true,
        sample_size: 200,
        recruitment: Arc::from("consecutive recruitment"),
        timing: Arc::from("baseline"),
        unit_rules: UnitRules {
            unit: Arc::from("patient"),
            cluster: None,
            whole_cluster_sampling: false,
        },
        cost: StudyCostDeclaration { units, unit_label: Arc::from("USD"), sample_budget: 200 },
        feasible: true,
        feasibility_notes: Arc::from([]),
        expected_evidence: Arc::from([ExpectedEvidence {
            population: Arc::from("pop"),
            interventions: vars(&[]),
            intervention_values: Arc::from([]),
            conditioned_on: Arc::from([]),
            measured: vars(measured),
            distribution: DistributionAvailability::Joint,
        }]),
        external_provider: None,
    }
}

#[test]
fn compiled_identification_repair_equals_the_direct_search_and_needs_the_joint_study() {
    let ctx = ctx();
    let a = observation("a", &[T, Y, Z1], 5);
    let b = observation("b", &[T, Y, Z2], 5);
    let c = observation("c", &[T, Y, Z1, Z2], 20);
    let candidates = vec![a.clone(), b.clone(), c.clone()];
    let direct = repair(
        &backdoor_family(),
        &candidates,
        RepairObjective::MinimizeCost,
        RepairLimits::default(),
        &ctx,
    )
    .unwrap();
    let plan = compile_identification_repair(
        backdoor_family(),
        candidates,
        RepairObjective::MinimizeCost,
        RepairLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_route!(plan, "IdentificationRepair", "design.identification_repair");
    let compiled = plan.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    assert_eq!(format!("{compiled:?}"), format!("{direct:?}"));
    // Hand derivation: Z1 and Z2 each confound T -> Y, so the only admissible adjustment set is
    // {Z1, Z2} and only the study measuring all four jointly repairs the contract.
    let best = compiled.best().unwrap();
    assert_eq!(best.candidates, vec![c.semantic_id()]);
    assert_eq!(best.classification, RepairClassification::VerifiedSufficient);
    for alone in [&a, &b] {
        let outcome =
            compiled.outcomes.iter().find(|o| o.candidates == vec![alone.semantic_id()]).unwrap();
        assert_eq!(outcome.classification, RepairClassification::Insufficient);
    }
}

// --- study addition planning (study_planning_lifecycle.rs R-443 Figure 1(c,d)) ---------------

fn study(
    id: &str,
    population: &str,
    on: &[usize],
    levels: Option<Vec<Vec<(usize, bool)>>>,
    measured: &[usize],
    cost: u64,
) -> StudyCandidate {
    let assign = |variable: usize, level: bool| InterventionAssignment {
        variable: vid(variable),
        value: Value::Bool(level),
    };
    StudyCandidate {
        id: Arc::from(id),
        population: Arc::from(population),
        interventions: on.iter().map(|v| vid(*v)).collect(),
        levels: levels.map(|levels| {
            levels
                .into_iter()
                .map(|level| level.into_iter().map(|(v, l)| assign(v, l)).collect())
                .collect()
        }),
        measured: measured.iter().map(|v| vid(*v)).collect(),
        recruitment: Arc::from("independent recruitment, one arm per level"),
        cost: StudyCost { units: cost, sample_budget: 500 },
        requires: Arc::from([]),
        conflicts: Arc::from([]),
        feasibility_constraints: Arc::from([Arc::from("ethics approval in hand")]),
    }
}

fn mz_base() -> EvidenceCatalog {
    let (catalog, _) = evidence();
    EvidenceCatalog::try_new(
        Arc::clone(&catalog.environments),
        catalog.regimes.iter().filter(|r| r.id.raw() == 0).cloned().collect::<Vec<_>>(),
        catalog.bindings.iter().filter(|b| b.regime.raw() == 0).cloned().collect::<Vec<_>>(),
        None,
    )
    .unwrap()
}

fn mz_candidates() -> Vec<StudyCandidate> {
    vec![
        study(
            "a_do_z2",
            "a",
            &[MZ_Z2],
            Some(vec![vec![(MZ_Z2, false)], vec![(MZ_Z2, true)]]),
            &[MZ_Z1, X, MZ_Y],
            3,
        ),
        study("b_do_z1", "b", &[MZ_Z1], Some(vec![vec![(MZ_Z1, false)]]), &[X, MZ_Z2, MZ_Y], 2),
        study("b_observe", "b", &[], None, &[MZ_Z1, X, MZ_Z2, MZ_Y], 1),
    ]
}

#[test]
fn compiled_study_addition_plan_equals_the_direct_planner_and_selects_the_complementary_pair() {
    let ctx = ExecutionContext::for_tests(3);
    let route = StudyPlanRoute::Mz(query(sources()));
    let direct = plan_studies(
        &graph(),
        &route,
        &mz_base(),
        &mz_candidates(),
        StudyPlanLimits::default(),
        &ctx,
    )
    .unwrap();
    let plan = compile_study_addition_plan(
        graph(),
        route,
        mz_base(),
        mz_candidates(),
        StudyPlanLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_route!(plan, "StudyAdditionPlanning", "design.study_planning");
    let compiled = plan.execute(&ctx).unwrap();
    assert_eq!(format!("{compiled:?}"), format!("{direct:?}"));
    // R-443 Figure 1(c,d): no single declared study repairs the multi-source failure; only the
    // a do(Z2) + b do(Z1 = 0) pair does, at 3 + 2 = 5 cost units.
    let top = compiled.proposal(0).expect("the complementary pair is sufficient");
    assert_eq!(top.proposal().candidates, [Arc::from("a_do_z2"), Arc::from("b_do_z1")]);
    assert_eq!(compiled.plan().proposals[0].cost_units, 5);
    assert!(compiled.plan().minimal);
}

// --- z-transport evidence planning (z_transport_planner.rs measuring fixture) ----------------

fn z_query() -> ZTransportQuery {
    ZTransportQuery {
        outcomes: vars(&[3]),
        treatments: vars(&[2]),
        controllable: vars(&[1]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: v(1),
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

/// Z -> X -> Y with an observed confounder W -> X, W -> Y, and the source experiment on Z.
fn z_fixture() -> (SelectionDiagram, EvidenceCatalog) {
    let mut graph = Admg::with_variables(5);
    for (a, b) in [(1, 2), (2, 3), (0, 2), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let coordinates = (0..5)
        .map(|raw| VariableCoordinate {
            variable: v(raw),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect::<Vec<_>>();
    let environment = |name: &str| {
        Environment::try_new(name, coordinates.clone(), Arc::<[VariableId]>::from([])).unwrap()
    };
    let experiment = EvidenceRegime::try_new(
        RegimeId::from_raw(1),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [v(1)],
        [InterventionAssignment { variable: v(1), value: Value::Bool(false) }],
        [v(0), v(2), v(3)],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let base = EvidenceCatalog::try_new(
        [environment("source"), environment("target")],
        [experiment],
        [RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(1),
            snapshot_identity: Arc::from("source-do-z0"),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        None,
    )
    .unwrap();
    (diagram, base)
}

fn measure(base: &EvidenceCatalog, id: &str, population: &str) -> TransportEvidenceCandidate {
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(2),
        RegimeKind::Observational,
        EvidenceKind::Proposed,
        [],
        [],
        (0..5).map(v).collect::<Vec<_>>(),
        population,
        DistributionAvailability::Joint,
    )
    .unwrap();
    TransportEvidenceCandidate {
        id: Arc::from(id),
        delta: EvidenceCatalogDelta::try_new(base, [regime]).unwrap(),
        design: CandidateDesign::Measure(MeasurementPlan {
            variables: (0..5).map(v).collect::<Vec<_>>().into(),
            cost: DesignCost { amount: 2.0, sample_budget: 50 },
            tag: 1,
        }),
        recruitment_sampling: Arc::from(format!("{population} population")),
        feasibility_constraints: Arc::from([]),
        cost: DesignCost { amount: 2.0, sample_budget: 50 },
    }
}

#[test]
fn compiled_z_transport_snapshot_and_plan_equal_the_direct_planner_and_measure_the_target() {
    let ctx = ExecutionContext::for_tests(0);
    let (diagram, base) = z_fixture();
    let direct_snapshot =
        snapshot_z_transport_failure(&diagram, &z_query(), &base, SidLimits::default(), &ctx)
            .unwrap();
    let snapshot_plan = compile_z_transport_failure_snapshot(
        diagram,
        z_query(),
        base.clone(),
        SidLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_route!(snapshot_plan, "ZTransportEvidencePlanning", "design.identification_repair");
    let snapshot = snapshot_plan.execute(&ctx).unwrap();
    assert_eq!(format!("{snapshot:?}"), format!("{direct_snapshot:?}"));

    let spec = ZTransportPlanSpec {
        candidates: vec![
            measure(&base, "measure-source", "source"),
            measure(&base, "measure-target", "target"),
        ],
        max_evaluated: 2,
        identification_limits: SidLimits::default(),
    };
    let snapshot = Arc::new(snapshot);
    let direct = plan_z_transport_evidence(&snapshot, &spec, &ctx).unwrap();
    let plan = compile_z_transport_evidence_plan(Arc::clone(&snapshot), spec, &ctx).unwrap();
    assert_route!(plan, "ZTransportEvidencePlanning", "design.identification_repair");
    let compiled = plan.execute(&ctx).unwrap();
    assert_eq!(format!("{compiled:?}"), format!("{direct:?}"));
    // The recursive formula cites the target's observational joint: measuring the target
    // repairs the plan, the same measurement in the source does not.
    assert_eq!(compiled.ranked_sufficient, [Arc::<str>::from("measure-target")]);
}

// --- structure identification and design ranking ---------------------------------------------

/// Two structures over T -> Y: without confounding (identified) and with T <-> Y (not
/// identified until an experiment sets T).
fn samples() -> Vec<GraphPosteriorSample> {
    let mut plain = Admg::with_variables(2);
    plain.insert_directed(DenseNodeId::from_raw(T), DenseNodeId::from_raw(Y)).unwrap();
    let mut confounded = plain.clone();
    confounded.insert_bidirected(DenseNodeId::from_raw(T), DenseNodeId::from_raw(Y)).unwrap();
    vec![
        GraphPosteriorSample { weight: 1.0, key: 1, graph: plain },
        GraphPosteriorSample { weight: 1.0, key: 2, graph: confounded },
    ]
}

#[test]
fn compiled_structure_identification_equals_the_id_algorithm_on_each_structure() {
    let ctx = ctx();
    let samples = samples();
    let single = compile_graph_identification(samples[1].graph.clone(), v(T), v(Y), &ctx).unwrap();
    assert_route!(single, "StructureIdentification", "design.evidence_plan");
    assert_eq!(single.execute(&ctx).unwrap(), identifies(&samples[1].graph, v(T), v(Y)).unwrap());

    let plan =
        compile_structure_identification(samples.clone(), v(T), v(Y), Some(vec![v(T)]), &ctx)
            .unwrap();
    assert_route!(plan, "StructureIdentification", "design.evidence_plan");
    let compiled = plan.execute(&ctx).unwrap();
    assert_eq!(compiled.posterior, structure_posterior(&samples, v(T), v(Y)).unwrap());
    assert_eq!(
        compiled.under_intervention.as_deref(),
        Some(identified_under_intervention(&samples, v(T), v(Y), &[v(T)]).unwrap().as_slice())
    );
    // Hand derivation: T <-> Y blocks identification of P(Y | do(T)); setting T removes the
    // bidirected edge at T, so both mutilated structures identify.
    assert_eq!(
        compiled.posterior.identified.as_ref(),
        [GraphIdentFlag::Identified, GraphIdentFlag::Unidentified]
    );
    assert_eq!(
        compiled.under_intervention.as_deref(),
        Some([GraphIdentFlag::Identified, GraphIdentFlag::Identified].as_slice())
    );
}

#[test]
fn compiled_design_ranking_equals_the_direct_ranker_and_scores_the_identified_mass_gain() {
    let ctx = ExecutionContext::for_tests(3);
    let readout = compile_structure_identification(samples(), v(T), v(Y), Some(vec![v(T)]), &ctx)
        .unwrap()
        .execute(&ctx)
        .unwrap();
    let graphs: WeightedGraphSamples = readout.posterior;
    let flags = readout.under_intervention.unwrap();
    let query = QueryId::from_raw(0);
    let candidates = vec![CandidateDesign::Intervene(ExperimentPlan {
        targets: Arc::from([v(T)]),
        cost: DesignCost::zero(),
        tag: 1,
    })];
    let eval = DesignEvaluationContext::<(), ()> {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: Some(&flags),
        graph_features: None,
    };
    let ranker = DesignRanker::new().with_config(DesignRankConfig {
        min_batches: 2,
        max_batches: 4,
        batch_size: 4,
        rank_uncertainty_threshold: 1.0,
    });
    let objective = DesignObjective::IncreaseIdentificationProbability { query };
    let direct = ranker.rank(&objective, &candidates, &eval, &ctx).unwrap();
    let plan = compile_design_ranking(&ranker, objective, candidates, &eval, &ctx).unwrap();
    assert_route!(plan, "DesignRanking", "design.ranking_artifact");
    let compiled = plan.execute(&eval, &ctx).unwrap();
    assert_eq!(format!("{compiled:?}"), format!("{direct:?}"));
    // Hand derivation: half the posterior mass is identified today; the experiment on T
    // identifies both structures, so the identified mass rises from 1/2 to 1.
    assert!((compiled.ranked[0].score - 0.5).abs() < 1e-12);

    // A ranking compiled for one posterior is not executed under another.
    let other = WeightedGraphSamples::new(
        vec![1.0, 3.0],
        vec![GraphIdentFlag::Identified, GraphIdentFlag::Unidentified],
        vec![1, 2],
    )
    .unwrap();
    let changed = DesignEvaluationContext::<(), ()> { graphs: &other, ..eval };
    assert!(plan.execute(&changed, &ctx).is_err());
}

// --- choice study ranking --------------------------------------------------------------------

#[test]
fn compiled_choice_study_ranking_equals_the_pooled_ranker_and_the_binomial_evsi() {
    let ctx = ExecutionContext::for_tests(5);
    let candidates = vec![
        ChoiceStudyCandidate { label: "one".into(), observations: 1, cost: 0.0 },
        ChoiceStudyCandidate { label: "three".into(), observations: 3, cost: 0.05 },
    ];
    let direct = rank_studies(&[1.0, 1.0], &[1.0, 1.0], &candidates, 17).unwrap();
    let plan = compile_choice_study_ranking(
        ChoicePosterior::Pooled { alpha: vec![1.0, 1.0], seed: 17 },
        vec![1.0, 1.0],
        candidates,
        &ctx,
    )
    .unwrap();
    assert_route!(plan, "ChoiceStudyRanking", "design.evsi");
    let compiled = plan.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    assert_eq!(format!("{compiled:?}"), format!("{direct:?}"));
    // Hand derivation under a uniform Beta(1, 1) share with equal values: today's best value is
    // 1/2; after one choice the chosen option's posterior mean is 2/3, so the EVSI of one more
    // choice is 2/3 - 1/2 = 1/6.
    let one = compiled.iter().find(|rank| rank.candidate.observations == 1).unwrap();
    assert!((one.evsi - 1.0 / 6.0).abs() < 1e-12, "{one:?}");
}

#[test]
fn compiled_choice_study_ranking_equals_the_hierarchical_ranker_and_its_reference_grid() {
    let ctx = ExecutionContext::for_tests(9);
    let prior = HierarchicalChoicePrior {
        shapes: vec![1.0; 3],
        log_concentration_mean: 5.0_f64.ln(),
        log_concentration_sd: 1.0,
    };
    let respondents = vec![
        vec![2, 1, 0],
        vec![0, 2, 1],
        vec![1, 0, 2],
        vec![1, 1, 1],
        vec![2, 1, 0],
        vec![0, 2, 1],
        vec![1, 0, 2],
    ];
    let candidates = vec![
        ChoiceStudyCandidate { label: "one".into(), observations: 1, cost: 0.0 },
        ChoiceStudyCandidate { label: "two".into(), observations: 2, cost: 0.0 },
    ];
    let direct =
        rank_hierarchical_studies(&prior, &respondents, &[1.0, 1.0, 1.0], &candidates, &ctx)
            .unwrap();
    let plan = compile_choice_study_ranking(
        ChoicePosterior::Hierarchical { prior, respondents },
        vec![1.0, 1.0, 1.0],
        candidates,
        &ctx,
    )
    .unwrap();
    assert_route!(plan, "ChoiceStudyRanking", "design.evsi");
    let compiled = plan.execute(&ctx).unwrap();
    assert_eq!(compiled, direct);
    assert_eq!(format!("{compiled:?}"), format!("{direct:?}"));
    // The independent three-option grid the kernel's own reference test reads.
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "oracle_data/hierarchical_choice_evsi_three_reference.json"
    ))
    .unwrap();
    for row in &compiled {
        let index = row.candidate.observations as usize - 1;
        assert!((row.evsi - expected["evsi"][index].as_f64().unwrap()).abs() < 2e-5, "{row:?}");
    }
}
