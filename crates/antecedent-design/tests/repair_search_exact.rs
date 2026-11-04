//! The minimal repair of a failed back-door contract, derived by hand, and its stored receipt.
//!
//! Nodes T=0, Y=1, Z1=2, Z2=3, W=4, D=5; edges Z1->T, Z1->Y, Z2->T, Z2->Y, T->Y, W->Z1, T->D.
//! The only minimal adjustment set is {Z1, Z2} (W is upstream of a blocked path, D is a
//! descendant of T), and it must be read from one joint law, so a subset of studies repairs the
//! contract exactly when one member jointly measures both Z1 and Z2 (with T and Y). Candidates:
//! `full` (T,Y,Z1,Z2,W,D) cost 30 budget 100; `minimal` (T,Y,Z1,Z2) cost 20 budget 200;
//! `cheap` (T,Y,Z1) cost 5; `decoy` (T,Y,W,D) cost 1. Hence the sufficient subsets are those
//! containing `full` or `minimal`; the cheapest is {minimal} (20) and, with sample budget as
//! the objective, {full} (100), although `decoy` is the cheapest single study overall.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{DistributionAvailability, ExecutionContext, VariableId};
use antecedent_design::{
    BackdoorRepairFamily, DurableStudyCandidate, ExpectedEvidence, RepairClassification,
    RepairConsumeLimits, RepairFamilyRef, RepairLimits, RepairObjective, RepairReportArtifact,
    StudyCostDeclaration, StudyKind, UnitRules, repair,
};
use antecedent_graph::{Dag, DenseNodeId};

const T: u32 = 0;
const Y: u32 = 1;
const Z1: u32 = 2;
const Z2: u32 = 3;
const W: u32 = 4;
const D: u32 = 5;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn vars(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(v).collect::<Vec<_>>().into()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn dag6() -> Dag {
    let mut graph = Dag::with_variables(6);
    for (from, to) in [(Z1, T), (Z1, Y), (Z2, T), (Z2, Y), (T, Y), (W, Z1), (T, D)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn backdoor_family() -> BackdoorRepairFamily {
    BackdoorRepairFamily::try_new(dag6(), v(T), v(Y), "pop", [v(T), v(Y)], &[]).unwrap()
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

fn candidates() -> Vec<DurableStudyCandidate> {
    let mut full = observation("full", &[T, Y, Z1, Z2, W, D], 30);
    full.cost.sample_budget = 100;
    vec![
        full,
        observation("minimal", &[T, Y, Z1, Z2], 20),
        observation("cheap", &[T, Y, Z1], 5),
        observation("decoy", &[T, Y, W, D], 1),
    ]
}

fn artifact_for(objective: RepairObjective) -> (RepairReportArtifact, Vec<DurableStudyCandidate>) {
    let family = backdoor_family();
    let cands = candidates();
    let report = repair(&family, &cands, objective, RepairLimits::default(), &ctx()).unwrap();
    let artifact =
        RepairReportArtifact::build(RepairFamilyRef::Backdoor(&family), &cands, &report).unwrap();
    (artifact, cands)
}

fn id_of(c: &DurableStudyCandidate) -> String {
    c.semantic_id().to_string()
}

#[test]
fn the_replayed_receipt_holds_exactly_the_hand_derived_sufficient_subsets_and_best() {
    for (objective, best_label) in [
        (RepairObjective::MinimizeCost, "minimal"),
        (RepairObjective::MinimizeSampleBudget, "full"),
    ] {
        let (artifact, cands) = artifact_for(objective);
        let by = |label: &str| id_of(cands.iter().find(|c| &*c.label == label).unwrap());
        let report = artifact.consume(RepairConsumeLimits::default(), &ctx()).unwrap();
        let (full, minimal) = (by("full"), by("minimal"));
        for outcome in &report.outcomes {
            let has = |id: &String| outcome.candidates.iter().any(|c| &**c == id.as_str());
            let expected = if has(&full) || has(&minimal) {
                RepairClassification::VerifiedSufficient
            } else {
                RepairClassification::Insufficient
            };
            assert_eq!(outcome.classification, expected, "{:?}", outcome.candidates);
        }
        let best = report.best().expect("a verified repair exists");
        assert_eq!(best.candidates.len(), 1);
        assert_eq!(&*best.candidates[0], by(best_label).as_str());
        let derivation = best.derivation.as_ref().unwrap();
        assert!(derivation.verified);
    }
}

#[test]
fn the_stored_costs_of_the_best_subsets_are_the_declared_sums() {
    let (artifact, _) = artifact_for(RepairObjective::MinimizeCost);
    let report = artifact.consume(RepairConsumeLimits::default(), &ctx()).unwrap();
    assert_eq!(report.best().unwrap().cost_units, 20);
    let (artifact, _) = artifact_for(RepairObjective::MinimizeSampleBudget);
    let report = artifact.consume(RepairConsumeLimits::default(), &ctx()).unwrap();
    assert_eq!(report.best().unwrap().sample_budget, 100);
}
