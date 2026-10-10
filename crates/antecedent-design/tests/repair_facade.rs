//! F9/F13 evidence obligations and the generic identification-repair facade,
//! exercised on two distinct families: catalog-aware classical transport and
//! back-door adjustment. Expected outcomes are derived by hand from the
//! theorems, not from the code under test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::assumption::{AssumptionSource, AssumptionStatus};
use antecedent_core::{
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceCatalogDelta,
    EvidenceObligationKind, ExecutionContext, FactorNeed, InterventionAssignment, ObligationKind,
    ObligationRecord, ObligationScope, SearchLimits, SearchStop, SupportRegion, SupportReport,
    SupportStatus, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_design::obligation_adapters::{
    delta_obligations, factor_obligations, support_obligations,
};
use antecedent_design::repair::{REPAIR_MAX_CANDIDATES, RepairClassification};
use antecedent_design::{
    BackdoorRepairFamily, DurableStudyCandidate, ExpectedEvidence, RepairFamily, RepairLimits,
    RepairObjective, RepairOutcome, RepairReport, StudyCostDeclaration, StudyKind,
    TransportRepairFamily, UnitRules, repair,
};
use antecedent_graph::{Admg, Dag, DenseNodeId, SelectionDiagram};
use antecedent_identify::{ClassicalTransportQuery, SidLimits};

// Back-door graph: Z1 and Z2 each confound T -> Y; W1..W3 are isolated.
const T: u32 = 0;
const Y: u32 = 1;
const Z1: u32 = 2;
const Z2: u32 = 3;
const W1: u32 = 4;
const W2: u32 = 5;
const W3: u32 = 6;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn vars(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(v).collect::<Vec<_>>().into()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn dag7() -> Dag {
    let mut graph = Dag::with_variables(7);
    for (from, to) in [(Z1, T), (Z1, Y), (Z2, T), (Z2, Y), (T, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn backdoor_family() -> BackdoorRepairFamily {
    BackdoorRepairFamily::try_new(dag7(), v(T), v(Y), "pop", [v(T), v(Y)], &[]).unwrap()
}

fn study(
    label: &str,
    kind: StudyKind,
    population: &str,
    interventions: &[u32],
    measured: &[u32],
    joint: bool,
    units: u64,
) -> DurableStudyCandidate {
    DurableStudyCandidate {
        label: Arc::from(label),
        kind,
        population: Arc::from(population),
        interventions: vars(interventions),
        measured: vars(measured),
        joint_measurement: joint,
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
            population: Arc::from(population),
            interventions: vars(interventions),
            intervention_values: Arc::from([]),
            conditioned_on: Arc::from([]),
            measured: vars(measured),
            distribution: if joint {
                DistributionAvailability::Joint
            } else {
                DistributionAvailability::SeparateMarginals { variables: vars(measured) }
            },
        }]),
        external_provider: None,
    }
}

/// An observational joint law in the back-door contract's population.
fn observation(label: &str, measured: &[u32], units: u64) -> DurableStudyCandidate {
    study(label, StudyKind::Observation, "pop", &[], measured, true, units)
}

fn ids(candidates: &[&DurableStudyCandidate]) -> Vec<Arc<str>> {
    let mut ids: Vec<Arc<str>> = candidates.iter().map(|c| c.semantic_id()).collect();
    ids.sort();
    ids
}

fn outcome<'a>(
    report: &'a RepairReport,
    candidates: &[&DurableStudyCandidate],
) -> &'a RepairOutcome {
    let wanted = ids(candidates);
    report.outcomes.iter().find(|o| o.candidates == wanted).expect("the subset has an outcome")
}

fn run(
    family: &dyn RepairFamily,
    candidates: &[DurableStudyCandidate],
    objective: RepairObjective,
    limits: RepairLimits,
) -> RepairReport {
    repair(family, candidates, objective, limits, &ctx()).unwrap()
}

fn defaults() -> RepairLimits {
    RepairLimits::default()
}

// ------------------------------------------------------------- obligations

#[test]
fn f9_failed_backdoor_contract_exposes_a_joint_law_obligation_with_its_proof_step() {
    let family = backdoor_family();
    let obligations = family.unresolved_obligations();
    assert_eq!(obligations.len(), 1);
    let joint = &obligations[0];
    assert_eq!(joint.kind, EvidenceObligationKind::ProvideJointLaw);
    // The only admissible adjustment set is {Z1, Z2}; with T and Y that is the
    // one joint law the contract is waiting for.
    assert_eq!(joint.variables.as_ref(), [v(T), v(Y), v(Z1), v(Z2)]);
    assert!(joint.regime.joint && joint.regime.interventions.is_empty());
    assert_eq!(joint.population.as_deref(), Some("pop"));
    assert_eq!(joint.provenance.family.as_ref(), "backdoor");
    assert_eq!(joint.provenance.proof_step.as_deref(), Some("backdoor.adjustment_set:0,1,2,3"));
}

#[test]
fn f9_failed_transport_contract_exposes_the_source_factor_and_its_search_stage() {
    let (family, _) = transport_family();
    let obligations = family.unresolved_obligations();
    assert_eq!(obligations.len(), 1);
    assert_eq!(obligations[0].kind, EvidenceObligationKind::Intervene);
    assert_eq!(obligations[0].population.as_deref(), Some("source"));
    assert_eq!(obligations[0].regime.interventions.as_ref(), [v(X)]);
    assert_eq!(obligations[0].variables.as_ref(), [v(TY)]);
    assert!(
        obligations[0].provenance.proof_step.as_deref().is_some_and(|s| s.starts_with("searched:"))
    );
    assert!(obligations[0].reason.contains("solver:"));
}

#[test]
fn f9_transport_factors_support_and_planner_deltas_become_obligations() {
    // An unmet transport factor from the catalog's own unmet-dependency report.
    let catalog = transport_catalog();
    let need = FactorNeed {
        population: "source",
        variables: &[v(TY)],
        conditioned_on: &[],
        interventions: &[v(X)],
    };
    let factors =
        factor_obligations(&catalog, &[(Arc::from("source_y"), need)], "transport", "contract:x")
            .unwrap();
    assert_eq!(factors.len(), 1);
    assert_eq!(factors[0].kind, EvidenceObligationKind::Intervene);
    assert_eq!(factors[0].provenance.proof_step.as_deref(), Some("factor:source_y"));
    // A satisfied need yields nothing.
    let satisfied = transport_catalog_with_experiment();
    assert!(
        factor_obligations(&satisfied, &[(Arc::from("source_y"), need)], "transport", "c")
            .unwrap()
            .is_empty()
    );

    // Support coordinates without evidence: one establish-support obligation
    // per label class and one sample obligation for weak overlap.
    let report = SupportReport {
        status: SupportStatus::MissingEvidence,
        query_region: SupportRegion { minima: Arc::from([0.0]), maxima: Arc::from([3.0]) },
        diagnostics: Vec::new(),
        warnings: Vec::new(),
        point_status: Some(Arc::from([
            SupportStatus::MissingEvidence,
            SupportStatus::Supported,
            SupportStatus::WeakOverlap,
            SupportStatus::OutsideEmpiricalSupport,
        ])),
    };
    let support = support_obligations(&report, "target", &[v(T)], "support:report").unwrap();
    assert_eq!(support.len(), 2);
    assert_eq!(support[0].kind, EvidenceObligationKind::EstablishSupport);
    assert_eq!(
        support[0].required_slots.as_ref(),
        [Arc::<str>::from("coordinate:0"), Arc::<str>::from("coordinate:3")]
    );
    assert_eq!(support[1].kind, EvidenceObligationKind::IncreaseSample);
    assert_eq!(support[1].required_slots.as_ref(), [Arc::<str>::from("coordinate:2")]);

    // A planner delta restated as obligations keeps each regime id as proof step.
    let candidate = study("exp", StudyKind::Experiment, "source", &[X], &[TY], true, 3);
    let delta = EvidenceCatalogDelta::try_new(
        &EvidenceCatalog::empty(),
        candidate.proposed_regimes(7).unwrap(),
    )
    .unwrap();
    let from_delta = delta_obligations(&delta, "planner:test").unwrap();
    assert_eq!(from_delta.len(), 1);
    assert_eq!(from_delta[0].kind, EvidenceObligationKind::Intervene);
    assert_eq!(from_delta[0].provenance.proof_step.as_deref(), Some("regime:7"));
}

// ------------------------------------------------------------------ backdoor

#[test]
fn f13_backdoor_two_separate_studies_cannot_meet_the_joint_regime_but_one_study_can() {
    let a = observation("a", &[T, Y, Z1], 5);
    let b = observation("b", &[T, Y, Z2], 5);
    let c = observation("c", &[T, Y, Z1, Z2], 20);
    let report = run(
        &backdoor_family(),
        &[a.clone(), b.clone(), c.clone()],
        RepairObjective::MinimizeCost,
        defaults(),
    );
    // Each confounder alone leaves the other backdoor path open.
    assert_eq!(outcome(&report, &[&a]).classification, RepairClassification::Insufficient);
    assert_eq!(outcome(&report, &[&b]).classification, RepairClassification::Insufficient);
    // Z1 and Z2 measured in two separate studies are not one joint law over
    // (T, Y, Z1, Z2): the pair is insufficient although every name is present.
    let pair = outcome(&report, &[&a, &b]);
    assert_eq!(pair.classification, RepairClassification::Insufficient);
    assert!(
        pair.reasons.iter().any(|r| r.contains("no admissible set exists")),
        "{:?}",
        pair.reasons
    );
    assert!(!pair.unmet.is_empty() && pair.addressed.is_empty());
    // One study measuring all four jointly repairs the contract.
    let repaired = outcome(&report, &[&c]);
    assert_eq!(repaired.classification, RepairClassification::VerifiedSufficient);
    let derivation = repaired.derivation.as_ref().unwrap();
    assert!(derivation.verified);
    assert_eq!(derivation.checker, "backdoor.adjustment");
    assert_eq!(derivation.steps[1], "adjustment_set:[2,3]");
    assert_eq!(repaired.addressed.len(), 1);
    // Supersets of the sufficient study are skipped, not evaluated.
    assert_eq!(report.receipt.dominated_skipped, 3);
    assert_eq!(report.outcomes.len(), 4);
    assert_eq!(report.receipt.operations_consumed, 4);
    assert_eq!(report.ranked_sufficient.len(), 1);
    assert_eq!(report.best().unwrap().candidates, ids(&[&c]));
    assert!(report.status().is_none() && report.receipt.stop.is_none());
}

#[test]
fn f13_insufficient_candidate_keeps_its_failure_reasons_and_the_report_is_wrong_contract() {
    let a = observation("a", &[T, Y, Z1], 5);
    let report = run(&backdoor_family(), &[a.clone()], RepairObjective::MinimizeCost, defaults());
    let only = outcome(&report, &[&a]);
    assert_eq!(only.classification, RepairClassification::Insufficient);
    assert!(only.derivation.is_none() && !only.reasons.is_empty());
    assert!(report.ranked_sufficient.is_empty());
    assert_eq!(
        report.status(),
        Some(("transport_not_certified", "identification_repair.wrong_contract"))
    );
}

#[test]
fn f13_a_variable_name_match_without_population_or_regime_stays_insufficient() {
    // Every needed variable is measured jointly, but in another population.
    let other_population =
        study("other", StudyKind::Observation, "elsewhere", &[], &[T, Y, Z1, Z2], true, 4);
    // Right variables, but an experiment on Z1 is not an observational law.
    let experiment = study("exp", StudyKind::Experiment, "pop", &[Z1], &[T, Y, Z2], true, 4);
    let report = run(
        &backdoor_family(),
        &[other_population.clone(), experiment.clone()],
        RepairObjective::MinimizeCost,
        defaults(),
    );
    assert_eq!(
        outcome(&report, &[&other_population]).classification,
        RepairClassification::Insufficient
    );
    assert_eq!(outcome(&report, &[&experiment]).classification, RepairClassification::Insufficient);
    assert!(report.ranked_sufficient.is_empty());
}

#[test]
fn f13_invalid_candidates_are_reported_and_never_searched() {
    let c = observation("c", &[T, Y, Z1, Z2], 20);
    let mut free = c.clone();
    free.label = Arc::from("free");
    free.cost.units = 0;
    let mut infeasible = observation("infeasible", &[T, Y, Z1, Z2, W1], 9);
    infeasible.feasible = false;
    // Observing separate regimes while claiming the joint law.
    let mut liar = observation("liar", &[T, Y, Z1, Z2, W2], 9);
    liar.joint_measurement = false;
    let report = run(
        &backdoor_family(),
        &[c.clone(), free.clone(), infeasible.clone(), liar.clone(), c.clone()],
        RepairObjective::MinimizeCost,
        defaults(),
    );
    for bad in [&free, &infeasible, &liar] {
        let o = outcome(&report, &[bad]);
        assert_eq!(o.classification, RepairClassification::Invalid);
        assert!(!o.reasons.is_empty());
    }
    assert!(
        outcome(&report, &[&free]).reasons[0].contains("study_candidate.wrong_contract"),
        "{:?}",
        outcome(&report, &[&free]).reasons
    );
    // The duplicate of C is invalid; the first C is searched and repairs.
    let duplicates = report
        .outcomes
        .iter()
        .filter(|o| o.candidates == ids(&[&c]) && o.classification == RepairClassification::Invalid)
        .count();
    assert_eq!(duplicates, 1);
    assert_eq!(report.best().unwrap().candidates, ids(&[&c]));
}

#[test]
fn f13_candidate_order_does_not_change_the_report() {
    let a = observation("a", &[T, Y, Z1], 5);
    let b = observation("b", &[T, Y, Z2], 5);
    let c = observation("c", &[T, Y, Z1, Z2], 20);
    let forward = run(
        &backdoor_family(),
        &[a.clone(), b.clone(), c.clone()],
        RepairObjective::MinimizeCost,
        defaults(),
    );
    let backward = run(&backdoor_family(), &[c, b, a], RepairObjective::MinimizeCost, defaults());
    assert_eq!(forward.outcomes, backward.outcomes);
    assert_eq!(forward.ranked_sufficient, backward.ranked_sufficient);
    assert_eq!(forward.receipt.explored, backward.receipt.explored);
}

#[test]
fn f13_objective_ranks_sufficient_subsets_and_cost_units_must_agree() {
    let cheap_wide = observation("cheap-wide", &[T, Y, Z1, Z2], 20);
    let mut dear_narrow = observation("dear-narrow", &[T, Y, Z1, Z2, W1], 30);
    dear_narrow.cost.sample_budget = 10;
    let candidates = [cheap_wide.clone(), dear_narrow.clone()];
    let by_cost = run(&backdoor_family(), &candidates, RepairObjective::MinimizeCost, defaults());
    assert_eq!(by_cost.best().unwrap().candidates, ids(&[&cheap_wide]));
    let by_samples =
        run(&backdoor_family(), &candidates, RepairObjective::MinimizeSampleBudget, defaults());
    assert_eq!(by_samples.best().unwrap().candidates, ids(&[&dear_narrow]));

    let mut euros = dear_narrow;
    euros.cost.unit_label = Arc::from("EUR");
    let error = repair(
        &backdoor_family(),
        &[cheap_wide, euros],
        RepairObjective::MinimizeCost,
        defaults(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(
        (error.code, error.detail),
        ("design_cost_units_mismatch", "identification_repair.cost_units_mismatch")
    );
}

#[test]
fn f13_budget_truncated_search_keeps_explored_and_unevaluated_and_never_says_impossible() {
    let candidates = [
        observation("a", &[T, Y, Z1], 5),
        observation("b", &[T, Y, Z2], 5),
        observation("w1", &[T, Y, W1], 1),
        observation("w2", &[T, Y, W2], 1),
        observation("w3", &[T, Y, W3], 1),
    ];
    // 5 singles + 10 pairs = 15 subsets; three operations evaluate three.
    let limits = RepairLimits { search: SearchLimits { operations: 3, depth: 2 }, ..defaults() };
    let report = run(&backdoor_family(), &candidates, RepairObjective::MinimizeCost, limits);
    let stop = report.receipt.stop.as_ref().expect("the operation limit binds");
    assert_eq!(stop.stop, SearchStop::Operations);
    assert_eq!(report.receipt.operations_consumed, 3);
    assert_eq!(report.receipt.explored.len(), 3);
    assert_eq!(report.receipt.unevaluated_total, 12);
    assert_eq!(report.receipt.unevaluated.len(), 12);
    assert_eq!(stop.explored, report.receipt.explored);
    assert_eq!(stop.unevaluated, report.receipt.unevaluated);
    let count = |class| report.outcomes.iter().filter(|o| o.classification == class).count();
    assert_eq!(count(RepairClassification::Unevaluated), 12);
    assert_eq!(count(RepairClassification::Insufficient), 3);
    assert_eq!(count(RepairClassification::VerifiedSufficient), 0);
    // Exhaustion is a resource outcome, not a verdict.
    assert_eq!(report.status(), Some(("transport_budget_cancel", "identification_repair.budget")));
    // A memory cap below the live-state estimate stops before any subset.
    let memory = RepairLimits { memory_limit_bytes: 100, ..defaults() };
    let starved = run(&backdoor_family(), &candidates, RepairObjective::MinimizeCost, memory);
    assert_eq!(starved.receipt.stop.as_ref().unwrap().stop, SearchStop::Memory);
    assert_eq!(starved.receipt.unevaluated_total, 5 + 10 + 10);
    assert!(starved.outcomes.iter().all(|o| o.classification == RepairClassification::Unevaluated));
}

#[test]
fn f13_subsets_larger_than_the_declared_depth_are_not_examined_and_say_so() {
    let a = observation("a", &[T, Y, Z1], 5);
    let b = observation("b", &[T, Y, Z2], 5);
    let limits = RepairLimits { search: SearchLimits { operations: 100, depth: 1 }, ..defaults() };
    let report = run(&backdoor_family(), &[a, b], RepairObjective::MinimizeCost, limits);
    assert!(report.receipt.beyond_declared_depth);
    assert!(report.outcomes.iter().all(|o| o.candidates.len() == 1));
    assert!(report.receipt.stop.is_none());
}

#[test]
fn f13_bounds_and_cancellation_are_refusals_or_reports_not_verdicts() {
    let family = backdoor_family();
    let many: Vec<_> = (0..=REPAIR_MAX_CANDIDATES as u64)
        .map(|n| observation(&format!("n{n}"), &[T, Y, Z1], n + 1))
        .collect();
    let error =
        repair(&family, &many, RepairObjective::MinimizeCost, defaults(), &ctx()).unwrap_err();
    assert_eq!(
        (error.code, error.detail),
        ("invalid_argument", "identification_repair.bounds_exceeded")
    );
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let error = repair(
        &family,
        &[observation("a", &[T, Y, Z1], 1)],
        RepairObjective::MinimizeCost,
        defaults(),
        &cancelled,
    )
    .unwrap_err();
    assert_eq!(
        (error.code, error.detail),
        ("transport_budget_cancel", "identification_repair.budget")
    );
    assert!(error.receipt.is_some());
}

#[test]
fn f9_establish_assumption_is_never_marked_satisfied_by_a_study() {
    let record = ObligationRecord::new(
        "assume:positivity",
        ObligationScope::Program,
        AssumptionSource::UserDeclared,
        ObligationKind::CheckNotRun,
        AssumptionStatus::Declared,
        "positivity of treatment given the adjustment set",
    );
    let family =
        BackdoorRepairFamily::try_new(dag7(), v(T), v(Y), "pop", [v(T), v(Y)], &[record]).unwrap();
    assert!(
        family
            .unresolved_obligations()
            .iter()
            .any(|o| o.kind == EvidenceObligationKind::EstablishAssumption)
    );
    let c = observation("c", &[T, Y, Z1, Z2], 20);
    let report = run(&family, &[c.clone()], RepairObjective::MinimizeCost, defaults());
    let repaired = outcome(&report, &[&c]);
    // The back-door checker identifies, but the study does not establish the
    // assumption, so the subset is not verified sufficient.
    assert_eq!(repaired.classification, RepairClassification::NotCertified);
    assert!(repaired.derivation.as_ref().is_some_and(|d| d.verified));
    assert!(repaired.reasons[0].contains("evidence_obligations.wrong_contract"));
    assert!(report.ranked_sufficient.is_empty());
    assert_eq!(
        report.status(),
        Some(("transport_missing_evidence", "evidence_obligations.wrong_contract"))
    );
}

#[test]
fn f13_a_contract_that_already_identifies_is_not_a_repair_target() {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let error =
        BackdoorRepairFamily::try_new(graph, v(0), v(1), "pop", [v(0), v(1)], &[]).unwrap_err();
    assert_eq!(error.detail, "identification_repair.not_a_failure");
}

// ----------------------------------------------------------------- transport

const X: u32 = 0;
const TY: u32 = 1;

fn transport_catalog() -> EvidenceCatalog {
    let coordinate =
        |raw| VariableCoordinate { variable: v(raw), domain: VariableDomain::Binary, unit: None };
    let source = Environment::try_new("source", [coordinate(X), coordinate(TY)], [v(X)]).unwrap();
    let target = Environment::try_new("target", [coordinate(X), coordinate(TY)], []).unwrap();
    EvidenceCatalog::try_new([source, target], [], [], None).unwrap()
}

fn transport_catalog_with_experiment() -> EvidenceCatalog {
    let base = transport_catalog();
    let candidate = study("exp", StudyKind::Experiment, "source", &[X], &[TY], true, 3);
    let mut regimes = candidate.proposed_regimes(0).unwrap();
    for regime in &mut regimes {
        regime.evidence_kind = antecedent_core::EvidenceKind::Available;
        regime.label = None;
        regime.study = None;
    }
    EvidenceCatalog::try_new(
        Arc::clone(&base.environments),
        regimes,
        Arc::clone(&base.bindings),
        base.target_sampling,
    )
    .unwrap()
}

/// X -> Y with the selection mechanism on X: P(y | do(x)) transports from the
/// source experiment, and nothing is available yet.
fn transport_family() -> (TransportRepairFamily, SelectionDiagram) {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(X), DenseNodeId::from_raw(TY)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(X)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(TY)]),
        treatments: Arc::from([v(X)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let family = TransportRepairFamily::try_new(
        diagram.clone(),
        query,
        transport_catalog(),
        SidLimits::default(),
        &ctx(),
    )
    .unwrap();
    (family, diagram)
}

#[test]
fn f13_transport_family_is_repaired_only_by_the_source_experiment_that_binds_the_factor() {
    let (family, _) = transport_family();
    let experiment = study("exp", StudyKind::Experiment, "source", &[X], &[TY], true, 3);
    // Same variable, wrong regime: a source observation is not do(X).
    let observation = study("obs", StudyKind::Observation, "source", &[], &[TY], true, 1);
    // A population the catalog does not declare.
    let mars = study("mars", StudyKind::Observation, "mars", &[], &[X, TY], true, 1);
    let report = run(
        &family,
        &[experiment.clone(), observation.clone(), mars.clone()],
        RepairObjective::MinimizeCost,
        defaults(),
    );
    assert_eq!(report.family, "transport");
    let repaired = outcome(&report, &[&experiment]);
    assert_eq!(repaired.classification, RepairClassification::VerifiedSufficient);
    let derivation = repaired.derivation.as_ref().unwrap();
    assert!(derivation.verified);
    assert_eq!(derivation.checker, "classical_transport.catalog");
    assert!(derivation.steps.last().is_some_and(|s| s.starts_with("leaf_factors:")));
    assert_eq!(repaired.addressed.len(), 1);
    assert_eq!(
        outcome(&report, &[&observation]).classification,
        RepairClassification::Insufficient
    );
    assert_eq!(outcome(&report, &[&mars]).classification, RepairClassification::Invalid);
    assert_eq!(report.best().unwrap().candidates, ids(&[&experiment]));
    // The experiment plus anything else is dominated, not re-evaluated.
    assert!(report.receipt.dominated_skipped >= 2);
}

#[test]
fn f13_two_distinct_families_are_repaired_by_their_own_verified_candidates() {
    let (transport, _) = transport_family();
    let experiment = study("exp", StudyKind::Experiment, "source", &[X], &[TY], true, 3);
    let transport_report =
        run(&transport, &[experiment.clone()], RepairObjective::MinimizeCost, defaults());
    let covariates = observation("covariates", &[T, Y, Z1, Z2], 20);
    let backdoor_report =
        run(&backdoor_family(), &[covariates.clone()], RepairObjective::MinimizeCost, defaults());
    assert_eq!(transport_report.family, "transport");
    assert_eq!(backdoor_report.family, "backdoor");
    assert!(transport_report.best().is_some() && backdoor_report.best().is_some());
    // Each candidate repairs only the family whose theorem it satisfies: the
    // same variable names are not enough across contracts.
    let crossed = run(&backdoor_family(), &[experiment], RepairObjective::MinimizeCost, defaults());
    assert!(crossed.best().is_none());
    let crossed = run(&transport, &[covariates], RepairObjective::MinimizeCost, defaults());
    assert!(crossed.best().is_none());
}

#[test]
fn f13_sample_increase_has_no_transport_checker_and_is_not_certified() {
    let (family, _) = transport_family();
    let mut more_rows = study("rows", StudyKind::SampleIncrease, "source", &[X], &[TY], true, 2);
    more_rows.expected_evidence = Arc::from([]);
    let report = run(&family, &[more_rows.clone()], RepairObjective::MinimizeCost, defaults());
    assert_eq!(outcome(&report, &[&more_rows]).classification, RepairClassification::NotCertified);
    assert!(report.best().is_none());
}

// A level-valued intervention keeps its levels as hypothetical evidence only.
#[test]
fn f13_leveled_experiment_is_a_restricted_domain_and_does_not_supply_an_unrestricted_response() {
    let (family, _) = transport_family();
    let mut leveled = study("leveled", StudyKind::Experiment, "source", &[X], &[TY], true, 3);
    leveled.expected_evidence = Arc::from([ExpectedEvidence {
        population: Arc::from("source"),
        interventions: vars(&[X]),
        intervention_values: Arc::from([InterventionAssignment {
            variable: v(X),
            value: Value::f64(1.0),
        }]),
        conditioned_on: Arc::from([]),
        measured: vars(&[TY]),
        distribution: DistributionAvailability::Joint,
    }]);
    let report = run(&family, &[leveled.clone()], RepairObjective::MinimizeCost, defaults());
    // A value-restricted assignment cannot supply an unrestricted symbolic
    // response, so the transport theorem's factor stays unmet.
    assert_ne!(
        outcome(&report, &[&leveled]).classification,
        RepairClassification::VerifiedSufficient
    );
}
