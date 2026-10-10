//! Surrogate experiments repair a z-transport contract only as a joint law
//! at the checked population and intervention level.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::repair::{
    DurableStudyCandidate, ExpectedEvidence, RepairClassification, RepairConsumeLimits,
    RepairFamily, RepairFamilyRef, RepairLimits, RepairObjective, StudyCostDeclaration, StudyKind,
    UnitRules, ZTransportRepairFamily, consume_repair_artifact, repair_contract,
    repair_with_artifact, unresolved_obligations,
};
use antecedent_core::{
    DistributionAvailability, Environment, EvidenceCatalog, ExecutionContext,
    InterventionAssignment, SearchLimits, SearchStop, Value, VariableCoordinate, VariableDomain,
    VariableId,
};
use antecedent_design::{
    ZTransportFailureSnapshot, ZTransportFailureSnapshotWire, snapshot_z_transport_failure,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{SidLimits, ZTransportQuery};
use std::sync::Arc;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}
fn vars(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(v).collect::<Vec<_>>().into()
}
fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(101)
}

fn snapshot() -> ZTransportFailureSnapshot {
    // Registered surrogate: W→Z→X→Y, W→Y, W↔Y, Z↔Y, Z↔X.
    // An experiment on Z breaks its confounding; a joint over W,X,Y is needed
    // to recover the X response. Separate marginals cannot supply this law.
    let mut graph = Admg::with_variables(4);
    for (a, b) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let query = ZTransportQuery {
        outcomes: vars(&[3]),
        treatments: vars(&[2]),
        controllable: vars(&[1]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: v(1),
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    // Intentionally noncanonical insertion order also exercises portable replay.
    let environments = ["target", "source"].map(|name| {
        Environment::try_new(
            name,
            (0..4)
                .map(|raw| VariableCoordinate {
                    variable: v(raw),
                    domain: VariableDomain::Binary,
                    unit: None,
                })
                .collect::<Vec<_>>(),
            [],
        )
        .unwrap()
    });
    let catalog = EvidenceCatalog::try_new(environments, [], [], None).unwrap();
    snapshot_z_transport_failure(&diagram, &query, &catalog, SidLimits::default(), &ctx()).unwrap()
}

fn experiment(label: &str, population: &str, level: bool, joint: bool) -> DurableStudyCandidate {
    DurableStudyCandidate {
        label: Arc::from(label),
        kind: StudyKind::Experiment,
        population: Arc::from(population),
        interventions: vars(&[1]),
        measured: vars(&[0, 2, 3]),
        joint_measurement: joint,
        sample_size: 200,
        recruitment: Arc::from("consecutive"),
        timing: Arc::from("baseline"),
        unit_rules: UnitRules {
            unit: Arc::from("patient"),
            cluster: None,
            whole_cluster_sampling: false,
        },
        cost: StudyCostDeclaration { units: 10, unit_label: Arc::from("USD"), sample_budget: 200 },
        feasible: true,
        feasibility_notes: Arc::from([]),
        external_provider: None,
        expected_evidence: Arc::from([ExpectedEvidence {
            population: Arc::from(population),
            interventions: vars(&[1]),
            intervention_values: Arc::from([InterventionAssignment {
                variable: v(1),
                value: Value::Bool(level),
            }]),
            conditioned_on: Arc::from([]),
            measured: vars(&[0, 2, 3]),
            distribution: if joint {
                DistributionAvailability::Joint
            } else {
                DistributionAvailability::SeparateMarginals { variables: vars(&[0, 2, 3]) }
            },
        }]),
    }
}

#[test]
fn public_obligations_drive_z_transport_repair_and_preserve_proof_leaves() {
    let family = ZTransportRepairFamily::try_new(snapshot(), SidLimits::default()).unwrap();
    let obligations = unresolved_obligations(&family);
    assert!(!obligations.is_empty());
    assert!(obligations.iter().all(|o| o.provenance.family.as_ref() == "z_transport"
        && o.provenance.proof_step.as_deref().is_some_and(|step| step.starts_with("leaf:"))));
    let correct = experiment("correct", "source", false, true);
    let wrong_population = experiment("target-experiment", "target", false, true);
    let wrong_level = experiment("wrong-level", "source", true, true);
    let separate = experiment("marginals", "source", false, false);
    let candidates = [correct.clone(), wrong_population, wrong_level, separate];
    let report = repair_contract(
        &family,
        &candidates,
        RepairObjective::MinimizeCost,
        RepairLimits {
            search: SearchLimits { operations: 100, depth: 1 },
            ..RepairLimits::default()
        },
        &ctx(),
    )
    .unwrap();
    let sufficient = report.best().unwrap();
    assert_eq!(sufficient.candidates, vec![correct.semantic_id()]);
    assert_eq!(sufficient.classification, RepairClassification::VerifiedSufficient);
    assert!(sufficient.unmet.is_empty());
    let proof = sufficient.derivation.as_ref().unwrap();
    assert_eq!(proof.checker, "z_transport.catalog");
    assert!(proof.verified);
    for obligation in obligations {
        assert!(proof.steps.contains(&obligation.provenance.proof_step.unwrap().to_string()));
    }
    assert_eq!(
        report
            .outcomes
            .iter()
            .filter(|o| o.classification == RepairClassification::VerifiedSufficient)
            .count(),
        1
    );
    for candidate in &candidates[1..] {
        let refused =
            report.outcomes.iter().find(|o| o.candidates == vec![candidate.semantic_id()]).unwrap();
        assert_eq!(refused.classification, RepairClassification::Insufficient);
        assert!(!refused.reasons.is_empty());
        assert!(refused.derivation.is_none());
    }
}

#[test]
fn portable_failure_replay_reproduces_obligations_and_repair() {
    let original = snapshot();
    let bytes = serde_json::to_vec(&original.to_wire().unwrap()).unwrap();
    let wire: ZTransportFailureSnapshotWire = serde_json::from_slice(&bytes).unwrap();
    let restored =
        ZTransportFailureSnapshot::from_wire(&wire, SidLimits::default(), &ctx()).unwrap();
    let a = ZTransportRepairFamily::try_new(original, SidLimits::default()).unwrap();
    let b = ZTransportRepairFamily::try_new(restored, SidLimits::default()).unwrap();
    assert_eq!(a.contract_id(), b.contract_id());
    assert_eq!(unresolved_obligations(&a), unresolved_obligations(&b));
    let candidates = [experiment("correct", "source", false, true)];
    let run = |family: &dyn RepairFamily| {
        repair_contract(
            family,
            &candidates,
            RepairObjective::MinimizeCost,
            RepairLimits::default(),
            &ctx(),
        )
        .unwrap()
    };
    assert_eq!(run(&a), run(&b));
    let mut edited = wire;
    edited.catalog_digest = "changed".into();
    assert!(ZTransportFailureSnapshot::from_wire(&edited, SidLimits::default(), &ctx()).is_err());
}

#[test]
fn stopped_z_transport_repair_keeps_candidates_unevaluated() {
    let family = ZTransportRepairFamily::try_new(snapshot(), SidLimits::default()).unwrap();
    let candidates =
        [experiment("first", "source", true, true), experiment("second", "source", false, true)];
    let report = repair_contract(
        &family,
        &candidates,
        RepairObjective::MinimizeCost,
        RepairLimits {
            search: SearchLimits { operations: 1, depth: 1 },
            ..RepairLimits::default()
        },
        &ctx(),
    )
    .unwrap();
    let stop = report.receipt.stop.as_ref().expect("the operation budget binds");
    assert_eq!(stop.stop, SearchStop::Operations);
    assert_eq!(report.receipt.operations_consumed, 1);
    assert_eq!(stop.explored, report.receipt.explored);
    assert_eq!(stop.unevaluated, report.receipt.unevaluated);
    assert_eq!(report.receipt.unevaluated_total, 1);
    assert_eq!(
        report
            .outcomes
            .iter()
            .filter(|o| o.classification == RepairClassification::Unevaluated)
            .count(),
        1
    );
}

#[test]
fn cancelled_z_transport_repair_never_certifies_a_candidate() {
    let family = ZTransportRepairFamily::try_new(snapshot(), SidLimits::default()).unwrap();
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    let error = repair_contract(
        &family,
        &[experiment("correct", "source", false, true)],
        RepairObjective::MinimizeCost,
        RepairLimits::default(),
        &cancelled,
    )
    .unwrap_err();
    assert_eq!(error.detail, "identification_repair.budget");
    let candidate = experiment("correct", "source", false, true);
    assert!(family.check(&[&candidate], &cancelled).is_err());
}

#[test]
fn z_transport_repair_artifact_replays_full_and_stopped_searches_and_refuses_forged_deltas() {
    let family = ZTransportRepairFamily::try_new(snapshot(), SidLimits::default()).unwrap();
    let candidates =
        [experiment("correct", "source", false, true), experiment("wrong", "source", true, true)];
    for operations in [1, 100] {
        let (report, artifact) = repair_with_artifact(
            RepairFamilyRef::ZTransport(&family),
            &candidates,
            RepairObjective::MinimizeCost,
            RepairLimits {
                search: SearchLimits { operations, depth: 1 },
                ..RepairLimits::default()
            },
            &ctx(),
        )
        .unwrap();
        let bytes = artifact.to_bytes("z-repair").unwrap();
        let (_, replay) =
            consume_repair_artifact(&bytes, RepairConsumeLimits::default(), &ctx()).unwrap();
        assert_eq!(report, replay);
        if operations == 1 {
            assert_eq!(replay.receipt.unevaluated_total, 1);
        } else {
            let mut forged = artifact;
            let sufficient = forged
                .report
                .outcomes
                .iter_mut()
                .find(|o| o.classification == "verified_sufficient")
                .unwrap();
            sufficient.delta = Some(vec![]);
            let error = forged
                .sealed()
                .unwrap()
                .consume(RepairConsumeLimits::default(), &ctx())
                .unwrap_err();
            assert_eq!(error.detail, "repair_artifact.replay_mismatch");
        }
    }
}
