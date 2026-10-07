//! F10 durable study candidates: validation, joint-law honesty, canonical
//! order-independent identity and the hypothetical regimes a study proposes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceCatalogDelta, EvidenceKind,
    EvidenceObligation, EvidenceObligationKind, EvidenceObligationSpec, InterventionAssignment,
    ObligationProvenance, ObligationRegime, ObligationScope, RegimeKind, Value, VariableId,
};
use antecedent_design::{
    CandidateDesign, DurableStudyCandidate, ExpectedEvidence, StudyCostDeclaration, StudyKind,
    UnitRules,
};

const A: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn vars(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(v).collect::<Vec<_>>().into()
}

fn level(variable: u32, value: f64) -> Arc<[InterventionAssignment]> {
    Arc::from([InterventionAssignment { variable: v(variable), value: Value::f64(value) }])
}

fn evidence(level_value: f64, measured: &[u32], joint: bool) -> ExpectedEvidence {
    ExpectedEvidence {
        population: Arc::from("T"),
        interventions: vars(&[A]),
        intervention_values: level(A, level_value),
        conditioned_on: Arc::from([]),
        measured: vars(measured),
        distribution: if joint {
            DistributionAvailability::Joint
        } else {
            DistributionAvailability::SeparateMarginals { variables: vars(measured) }
        },
    }
}

/// Candidate C of the frozen fixture: joint (Y, Z) under do(A = 1) in population
/// T with n = 100 and cost 10 USD.
fn candidate_c() -> DurableStudyCandidate {
    DurableStudyCandidate {
        label: Arc::from("trial-1"),
        kind: StudyKind::Experiment,
        population: Arc::from("T"),
        interventions: vars(&[A]),
        measured: vars(&[Y, Z]),
        joint_measurement: true,
        sample_size: 100,
        recruitment: Arc::from("randomized recruitment at three sites"),
        timing: Arc::from("12 week follow-up"),
        unit_rules: UnitRules {
            unit: Arc::from("patient"),
            cluster: Some(Arc::from("site")),
            whole_cluster_sampling: true,
        },
        cost: StudyCostDeclaration { units: 10, unit_label: Arc::from("USD"), sample_budget: 100 },
        feasible: true,
        feasibility_notes: Arc::from([Arc::from("treatment is manipulable")]),
        expected_evidence: Arc::from([evidence(1.0, &[Y, Z], true)]),
        external_provider: None,
    }
}

fn joint_obligation() -> EvidenceObligation {
    EvidenceObligation::try_new(EvidenceObligationSpec {
        kind: EvidenceObligationKind::ProvideJointLaw,
        scope: ObligationScope::Factor,
        variables: vars(&[Y, Z]),
        population: Some(Arc::from("T")),
        regime: ObligationRegime {
            interventions: vars(&[A]),
            conditioned_on: Arc::from([]),
            joint: true,
        },
        reason: Arc::from("joint (Y, Z) under do(A) is required"),
        required_slots: Arc::from([Arc::from("factor:joint_yz")]),
        min_additional_samples: None,
        provenance: ObligationProvenance {
            family: Arc::from("transport"),
            source: Arc::from("contract:test"),
            proof_step: Some(Arc::from("leaf:2")),
        },
    })
    .unwrap()
}

fn refusal(candidate: &DurableStudyCandidate) -> (&'static str, &'static str) {
    let error = candidate.validate().unwrap_err();
    (error.code, error.detail)
}

#[test]
fn f10_joint_candidate_validates_and_matches_the_joint_obligation() {
    let c = candidate_c();
    c.validate().unwrap();
    assert!(c.semantic_id().starts_with("sc1:"));
    let offers = c.offers();
    assert_eq!(offers.len(), 1);
    assert!(offers[0].joint);
    assert_eq!(offers[0].additional_samples, Some(100));
    assert!(joint_obligation().addressed_by(&offers[0]));
}

#[test]
fn f10_candidate_observing_separate_regimes_cannot_claim_the_joint_factor() {
    let wrong = ("design_signal_invalid", "study_candidate.wrong_contract");
    // The study does not observe (Y, Z) together yet claims one joint law.
    let mut separate = candidate_c();
    separate.joint_measurement = false;
    assert_eq!(refusal(&separate), wrong);
    // Honestly declaring separate marginals validates but never addresses the
    // joint obligation.
    separate.expected_evidence = Arc::from([evidence(1.0, &[Y, Z], false)]);
    separate.validate().unwrap();
    let offers = separate.offers();
    assert!(!offers[0].joint);
    assert!(!joint_obligation().addressed_by(&offers[0]));
    // Evidence over a variable the study does not measure is refused.
    let mut outside = candidate_c();
    outside.measured = vars(&[Y]);
    assert_eq!(refusal(&outside), wrong);
}

#[test]
fn f10_missing_unit_timing_or_cost_semantics_refuse() {
    let wrong = ("design_signal_invalid", "study_candidate.wrong_contract");
    let mut free = candidate_c();
    free.cost.units = 0;
    assert_eq!(refusal(&free), wrong);
    let mut unitless = candidate_c();
    unitless.cost.unit_label = Arc::from(" ");
    assert_eq!(refusal(&unitless), wrong);
    let mut no_unit = candidate_c();
    no_unit.unit_rules.unit = Arc::from("");
    assert_eq!(refusal(&no_unit), wrong);
    let mut no_timing = candidate_c();
    no_timing.timing = Arc::from("");
    assert_eq!(refusal(&no_timing), wrong);
    let mut no_recruitment = candidate_c();
    no_recruitment.recruitment = Arc::from("  ");
    assert_eq!(refusal(&no_recruitment), wrong);
    let mut empty = candidate_c();
    empty.sample_size = 0;
    assert_eq!(refusal(&empty), wrong);
    let mut silent = candidate_c();
    silent.expected_evidence = Arc::from([]);
    assert_eq!(refusal(&silent), wrong);
    let mut observing = candidate_c();
    observing.kind = StudyKind::Observation;
    assert_eq!(refusal(&observing), wrong);
}

#[test]
fn f10_reordering_preserves_semantic_identity() {
    let mut two_levels = candidate_c();
    two_levels.expected_evidence =
        Arc::from([evidence(0.0, &[Y, Z], true), evidence(1.0, &[Y, Z], true)]);
    two_levels.feasibility_notes = Arc::from([Arc::from("alpha"), Arc::from("beta")]);
    two_levels.validate().unwrap();

    let mut shuffled = two_levels.clone();
    shuffled.expected_evidence =
        Arc::from([evidence(1.0, &[Z, Y], true), evidence(0.0, &[Z, Y], true)]);
    shuffled.measured = vars(&[Z, Y]);
    shuffled.feasibility_notes = Arc::from([Arc::from("beta"), Arc::from("alpha")]);
    shuffled.validate().unwrap();
    assert_eq!(two_levels.semantic_id(), shuffled.semantic_id());

    // The human label is not part of the semantic identity; content is.
    let mut relabelled = two_levels.clone();
    relabelled.label = Arc::from("renamed");
    assert_eq!(two_levels.semantic_id(), relabelled.semantic_id());
    let mut larger = two_levels.clone();
    larger.sample_size = 101;
    assert_ne!(two_levels.semantic_id(), larger.semantic_id());
    let mut provided = two_levels.clone();
    provided.external_provider = Some(Arc::from("registry:trial-network"));
    provided.validate().unwrap();
    assert_ne!(two_levels.semantic_id(), provided.semantic_id());
    let mut dearer = two_levels.clone();
    dearer.cost.units = 11;
    assert_ne!(two_levels.semantic_id(), dearer.semantic_id());
}

#[test]
fn f10_proposed_regimes_are_proposed_and_tied_to_the_semantic_id() {
    let c = candidate_c();
    let regimes = c.proposed_regimes(5).unwrap();
    assert_eq!(regimes.len(), 1);
    assert_eq!(regimes[0].id.raw(), 5);
    assert_eq!(regimes[0].evidence_kind, EvidenceKind::Proposed);
    assert_eq!(regimes[0].kind, RegimeKind::Experimental);
    assert_eq!(regimes[0].study.as_deref(), Some(&*c.semantic_id()));
    assert_eq!(regimes[0].label.as_deref().map(|l| l.ends_with("#0")), Some(true));
    // A proposed regime is a hypothesis: the delta never makes it available.
    let delta = EvidenceCatalogDelta::try_new(&EvidenceCatalog::empty(), regimes).unwrap();
    assert_eq!(delta.proposed_regimes[0].evidence_kind, EvidenceKind::Proposed);
}

#[test]
fn f10_design_action_bridges_to_the_existing_planners() {
    let c = candidate_c();
    let CandidateDesign::Intervene(plan) = c.design_action() else {
        panic!("an experiment bridges to an intervention plan");
    };
    assert_eq!(plan.targets.as_ref(), [v(A)]);
    assert!((plan.cost.amount - 10.0).abs() < 1e-12);
    assert_eq!(plan.cost.sample_budget, 100);

    let mut increase = candidate_c();
    increase.kind = StudyKind::SampleIncrease;
    increase.expected_evidence = Arc::from([]);
    increase.validate().unwrap();
    assert!(matches!(
        increase.design_action(),
        CandidateDesign::IncreaseSamplingRate(plan) if plan.additional_samples == 100
    ));
    assert!(increase.proposed_regimes(0).unwrap().is_empty());
    // A sample increase delivers no new regime and so claims none.
    increase.expected_evidence = Arc::from([evidence(1.0, &[Y, Z], true)]);
    assert!(increase.validate().is_err());
}
