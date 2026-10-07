//! F13 `repair_search_receipt_v1`: the durable identification-repair artifact
//! and its independent consumer. Expected outcomes follow the theorems (a
//! back-door adjustment set must be read from one joint law; the transport
//! source factor needs the source experiment), not the code under test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::assumption::{AssumptionSource, AssumptionStatus};
use antecedent_core::{
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceObligationKind,
    ExecutionContext, ObligationKind, ObligationRecord, ObligationScope, SearchLimits,
    VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_design::repair_artifact::DurableStudyCandidateWire;
use antecedent_design::{
    BackdoorRepairFamily, DurableStudyCandidate, ExpectedEvidence, RepairClassification,
    RepairConsumeLimits, RepairFamily, RepairFamilyRef, RepairLimits, RepairObjective,
    RepairReport, RepairReportArtifact, StudyCostDeclaration, StudyKind, TransportRepairFamily,
    UnitRules, repair,
};
use antecedent_graph::{Admg, Dag, DenseNodeId, SelectionDiagram};
use antecedent_identify::{ClassicalTransportQuery, SidLimits};

const T: u32 = 0;
const Y: u32 = 1;
const Z1: u32 = 2;
const Z2: u32 = 3;
const W1: u32 = 4;
const W2: u32 = 5;
const W3: u32 = 6;
const X: u32 = 0;
const TY: u32 = 1;

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

fn observation(label: &str, measured: &[u32], units: u64) -> DurableStudyCandidate {
    study(label, StudyKind::Observation, "pop", &[], measured, true, units)
}

fn transport_catalog() -> EvidenceCatalog {
    let coordinate =
        |raw| VariableCoordinate { variable: v(raw), domain: VariableDomain::Binary, unit: None };
    let source = Environment::try_new("source", [coordinate(X), coordinate(TY)], [v(X)]).unwrap();
    let target = Environment::try_new("target", [coordinate(X), coordinate(TY)], []).unwrap();
    EvidenceCatalog::try_new([source, target], [], [], None).unwrap()
}

fn transport_family() -> TransportRepairFamily {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(X), DenseNodeId::from_raw(TY)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(X)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(TY)]),
        treatments: Arc::from([v(X)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    TransportRepairFamily::try_new(
        diagram,
        query,
        transport_catalog(),
        SidLimits::default(),
        &ctx(),
    )
    .unwrap()
}

fn run(
    family: &dyn RepairFamily,
    candidates: &[DurableStudyCandidate],
    limits: RepairLimits,
) -> RepairReport {
    repair(family, candidates, RepairObjective::MinimizeCost, limits, &ctx()).unwrap()
}

fn backdoor_candidates() -> Vec<DurableStudyCandidate> {
    vec![
        observation("a", &[T, Y, Z1], 5),
        observation("b", &[T, Y, Z2], 5),
        observation("c", &[T, Y, Z1, Z2], 20),
    ]
}

fn backdoor_artifact() -> (RepairReportArtifact, RepairReport, Vec<DurableStudyCandidate>) {
    let family = backdoor_family();
    let candidates = backdoor_candidates();
    let report = run(&family, &candidates, RepairLimits::default());
    let artifact =
        RepairReportArtifact::build(RepairFamilyRef::Backdoor(&family), &candidates, &report)
            .unwrap();
    (artifact, report, candidates)
}

fn transport_artifact() -> (RepairReportArtifact, RepairReport) {
    let family = transport_family();
    let candidates = vec![
        study("exp", StudyKind::Experiment, "source", &[X], &[TY], true, 3),
        study("obs", StudyKind::Observation, "source", &[], &[TY], true, 1),
    ];
    let report = run(&family, &candidates, RepairLimits::default());
    let artifact =
        RepairReportArtifact::build(RepairFamilyRef::Transport(&family), &candidates, &report)
            .unwrap();
    (artifact, report)
}

fn consume(
    artifact: &RepairReportArtifact,
) -> Result<RepairReport, antecedent_design::RepairArtifactError> {
    artifact.consume(RepairConsumeLimits::default(), &ctx())
}

fn detail(artifact: &RepairReportArtifact) -> &'static str {
    consume(artifact).expect_err("the artifact is refused").detail
}

// -------------------------------------------------------------- round trips

#[test]
fn f13_backdoor_artifact_round_trips_through_the_container_and_replays_identically() {
    let (artifact, report, _) = backdoor_artifact();
    let bytes = artifact.to_bytes("repair-backdoor").unwrap();
    let decoded = RepairReportArtifact::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, artifact);
    let replayed = consume(&decoded).unwrap();
    assert_eq!(replayed.outcomes, report.outcomes);
    assert_eq!(replayed.ranked_sufficient, report.ranked_sufficient);
    assert_eq!(replayed.obligations, report.obligations);
    assert_eq!(replayed.receipt.explored, report.receipt.explored);
    assert_eq!(artifact.report.outcome, "repaired");
    assert_eq!(artifact.premises.family, "backdoor");
}

#[test]
fn f13_transport_artifact_round_trips_and_stores_the_hypothetical_delta() {
    let (artifact, report) = transport_artifact();
    let bytes = artifact.to_bytes("repair-transport").unwrap();
    let decoded = RepairReportArtifact::from_bytes(&bytes).unwrap();
    let replayed = consume(&decoded).unwrap();
    assert_eq!(replayed.outcomes, report.outcomes);
    let sufficient = artifact
        .report
        .outcomes
        .iter()
        .find(|o| o.classification == "verified_sufficient")
        .expect("the source experiment repairs the contract");
    let delta = sufficient.delta.as_ref().expect("a decided subset keeps its delta");
    assert_eq!(delta.len(), 1);
    assert_eq!(delta[0].population, "source");
    assert_eq!(delta[0].interventions, vec![X]);
    assert_eq!(delta[0].evidence_kind, "proposed");
    assert_eq!(
        sufficient.derivation.as_ref().map(|d| (d.checker.as_str(), d.verified)),
        Some(("classical_transport.catalog", true))
    );
    assert!(artifact.data.catalog.is_some() && artifact.data.base_law.is_none());
}

#[test]
fn f9_artifact_keeps_obligation_ids_kinds_and_the_source_proof_step() {
    let (artifact, report, _) = backdoor_artifact();
    let stored = artifact.obligations().unwrap();
    assert_eq!(stored, report.obligations);
    assert_eq!(stored[0].kind, EvidenceObligationKind::ProvideJointLaw);
    assert_eq!(stored[0].provenance.proof_step.as_deref(), Some("backdoor.adjustment_set:0,1,2,3"));
    let (transport, _) = transport_artifact();
    let transport_stored = transport.obligations().unwrap();
    assert_eq!(transport_stored[0].kind, EvidenceObligationKind::Intervene);
    assert!(
        transport_stored[0]
            .provenance
            .proof_step
            .as_deref()
            .is_some_and(|step| step.starts_with("searched:"))
    );
}

#[test]
fn f9_separate_studies_cannot_meet_the_joint_regime_in_the_stored_report() {
    let (artifact, _, _) = backdoor_artifact();
    let pair = artifact
        .report
        .outcomes
        .iter()
        .find(|o| o.candidates.len() == 2)
        .expect("the pair of single-confounder studies was evaluated");
    assert_eq!(pair.classification, "insufficient");
    assert!(pair.addressed.is_empty() && !pair.unmet.is_empty());
    assert!(pair.derivation.is_none());
}

#[test]
fn f13_assumption_obligations_survive_the_artifact_and_block_certification() {
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
    let candidates = vec![observation("c", &[T, Y, Z1, Z2], 20)];
    let report = run(&family, &candidates, RepairLimits::default());
    let artifact =
        RepairReportArtifact::build(RepairFamilyRef::Backdoor(&family), &candidates, &report)
            .unwrap();
    assert_eq!(artifact.premises.backdoor.as_ref().unwrap().assumptions.len(), 1);
    let replayed = consume(&artifact).unwrap();
    assert_eq!(replayed.outcomes, report.outcomes);
    assert_eq!(artifact.report.outcomes[0].classification, "not_certified");
    assert_eq!(artifact.report.outcome, "none_certified");
}

// ------------------------------------------------------------------ replay

#[test]
fn f13_replay_recomputes_the_verdict_from_the_stored_delta() {
    let (artifact, _) = transport_artifact();
    // Keep the claimed "verified_sufficient" but empty its stored delta: the
    // family's own checker, run on the stored delta, no longer identifies.
    let mut emptied = artifact.clone();
    let at = emptied
        .report
        .outcomes
        .iter()
        .position(|o| o.classification == "verified_sufficient")
        .unwrap();
    emptied.report.outcomes[at].delta = Some(Vec::new());
    let emptied = emptied.sealed().unwrap();
    assert_eq!(detail(&emptied), "repair_artifact.replay_mismatch");

    // Claim the insufficient observation was sufficient, keeping its delta.
    let mut forged = artifact.clone();
    let obs =
        forged.report.outcomes.iter().position(|o| o.classification == "insufficient").unwrap();
    forged.report.outcomes[obs].classification = "verified_sufficient".to_owned();
    let forged = forged.sealed().unwrap();
    let error = consume(&forged).unwrap_err();
    assert_eq!(error.detail, "repair_artifact.replay_mismatch");
    assert!(error.message.starts_with("classification:"), "{}", error.message);
}

#[test]
fn f13_resealed_mutation_of_candidate_obligation_classification_or_receipt_is_refused() {
    let (artifact, _, _) = backdoor_artifact();

    // A changed classification, resealed.
    let mut classification = artifact.clone();
    let at = classification
        .report
        .outcomes
        .iter()
        .position(|o| o.classification == "verified_sufficient")
        .unwrap();
    classification.report.outcomes[at].classification = "insufficient".to_owned();
    assert_eq!(detail(&classification.sealed().unwrap()), "repair_artifact.replay_mismatch");

    // A changed receipt, resealed.
    let mut receipt = artifact.clone();
    receipt.report.receipt.dominated_skipped += 1;
    let error = consume(&receipt.sealed().unwrap()).unwrap_err();
    assert_eq!(error.detail, "repair_artifact.replay_mismatch");
    assert!(error.message.starts_with("receipt:"), "{}", error.message);

    // A changed obligation (its content no longer matches its id), resealed.
    let mut obligation = artifact.clone();
    obligation.report.obligations[0].reason = "back-door adjustment needs nothing".to_owned();
    // Either the id check or the replay comparison refuses it; both are refusals.
    let refused = detail(&obligation.sealed().unwrap());
    assert!(
        refused == "repair_artifact.invalid_artifact"
            || refused == "repair_artifact.replay_mismatch",
        "{refused}"
    );

    // A changed obligation re-derived with a fresh id still differs from the
    // obligation the contract owes.
    let mut swapped = artifact.clone();
    swapped.report.obligations[0].kind = "measure".to_owned();
    assert!(consume(&swapped.sealed().unwrap()).is_err());

    // A changed candidate with an honestly recomputed semantic id: the stored
    // outcomes no longer replay from it.
    let mut candidate = artifact.clone();
    let mut changed = candidate.candidates().unwrap()[0].clone();
    changed.cost.units += 100;
    candidate.premises.candidates[0] = DurableStudyCandidateWire::from_candidate(&changed);
    assert_eq!(detail(&candidate.sealed().unwrap()), "repair_artifact.replay_mismatch");

    // Any edit left unsealed fails the digest.
    let mut unsealed = artifact;
    unsealed.report.outcome = "repaired-ish".to_owned();
    assert_eq!(detail(&unsealed), "repair_artifact.invalid_artifact");
}

#[test]
fn f13_unknown_version_and_kind_are_refused_and_stored_limits_are_bounded() {
    let (artifact, _, _) = backdoor_artifact();
    let mut future = artifact.clone();
    future.version = 2;
    let future = future.sealed().unwrap();
    assert_eq!(detail(&future), "repair_artifact.unsupported_version");
    let bytes = future.to_bytes("repair-v2").unwrap();
    assert_eq!(
        RepairReportArtifact::from_bytes(&bytes).unwrap_err().detail,
        "repair_artifact.unsupported_version"
    );

    // Stored limits above the consumer's maxima refuse before any work.
    let small = RepairConsumeLimits {
        search: SearchLimits { operations: 1, depth: 1 },
        ..RepairConsumeLimits::default()
    };
    let error = artifact.consume(small, &ctx()).unwrap_err();
    assert_eq!(error.detail, "repair_artifact.bounds_exceeded");

    // A corrupt container is refused by the bounded reader.
    let good = artifact.to_bytes("repair-ok").unwrap();
    assert!(RepairReportArtifact::from_bytes(&good[..good.len() / 2]).is_err());
}

// ------------------------------------------------------- truncated search

#[test]
fn f13_truncated_search_is_retained_as_unevaluated_never_as_impossible() {
    let family = backdoor_family();
    let candidates = vec![
        observation("a", &[T, Y, Z1], 5),
        observation("b", &[T, Y, Z2], 5),
        observation("w1", &[T, Y, W1], 1),
        observation("w2", &[T, Y, W2], 1),
        observation("w3", &[T, Y, W3], 1),
    ];
    let limits = RepairLimits {
        search: SearchLimits { operations: 3, depth: 2 },
        ..RepairLimits::default()
    };
    let report = run(&family, &candidates, limits);
    let artifact =
        RepairReportArtifact::build(RepairFamilyRef::Backdoor(&family), &candidates, &report)
            .unwrap();
    let decoded =
        RepairReportArtifact::from_bytes(&artifact.to_bytes("repair-truncated").unwrap()).unwrap();
    // 5 singles + 10 pairs; three operations decide three.
    let unevaluated =
        decoded.report.outcomes.iter().filter(|o| o.classification == "unevaluated").count();
    assert_eq!(unevaluated, 12);
    assert_eq!(decoded.report.receipt.unevaluated_total, 12);
    assert_eq!(decoded.report.receipt.unevaluated.len(), 12);
    assert_eq!(decoded.report.receipt.explored.len(), 3);
    assert!(decoded.report.receipt.stop.as_ref().is_some_and(|s| s.stop == "search.operations"));
    assert_eq!(decoded.report.outcome, "exhausted");
    assert_eq!(
        decoded.report.status.as_ref().map(|(_, d)| d.as_str()),
        Some("identification_repair.budget")
    );
    assert!(
        decoded
            .report
            .outcomes
            .iter()
            .all(|o| o.delta.is_none() == (o.classification == "unevaluated"))
    );
    let rendered = format!("{decoded:?}");
    assert!(!rendered.contains("impossible"));
    // The unevaluated subsets replay as unevaluated, not as failures.
    let replayed = consume(&decoded).unwrap();
    assert_eq!(
        replayed
            .outcomes
            .iter()
            .filter(|o| o.classification == RepairClassification::Unevaluated)
            .count(),
        12
    );
    // Reclassifying an unevaluated subset as insufficient is refused.
    let mut lied = decoded;
    let at = lied.report.outcomes.iter().position(|o| o.classification == "unevaluated").unwrap();
    lied.report.outcomes[at].classification = "insufficient".to_owned();
    assert!(consume(&lied.sealed().unwrap()).is_err());
}

#[test]
fn f13_beyond_depth_is_stored_as_not_examined() {
    let family = backdoor_family();
    let candidates = vec![observation("a", &[T, Y, Z1], 5), observation("b", &[T, Y, Z2], 5)];
    let limits = RepairLimits {
        search: SearchLimits { operations: 100, depth: 1 },
        ..RepairLimits::default()
    };
    let report = run(&family, &candidates, limits);
    let artifact =
        RepairReportArtifact::build(RepairFamilyRef::Backdoor(&family), &candidates, &report)
            .unwrap();
    assert!(artifact.report.receipt.beyond_declared_depth);
    assert!(artifact.report.outcomes.iter().all(|o| o.candidates.len() == 1));
    assert_eq!(artifact.report.outcome, "none_certified");
    assert!(consume(&artifact).is_ok());
}

#[test]
fn f13_a_report_for_another_contract_cannot_be_exported_under_this_family() {
    let (_, report, candidates) = backdoor_artifact();
    let other = transport_family();
    let error =
        RepairReportArtifact::build(RepairFamilyRef::Transport(&other), &candidates, &report)
            .unwrap_err();
    assert_eq!(error.detail, "repair_artifact.invalid_artifact");
}
