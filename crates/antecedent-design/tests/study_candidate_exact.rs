//! Which study candidates address a joint-law obligation, derived by hand.
//!
//! The obligation owed is the joint law of (Y, Z) in population T under do(A). A study
//! addresses it exactly when it delivers, in population T and with the same intervention set
//! do(A), a law that covers both Y and Z jointly. Hand table for the six candidates below:
//! joint (Y,Z) under do(A) in T: yes; joint (Y,Z,W) under do(A) in T: yes (a superset);
//! separate marginals of Y and Z: no (not one joint law); joint (Y,Z) in population S: no;
//! joint (Y,Z) with no intervention: no (wrong regime); joint (Y) only: no (Z missing).
//! The cheapest addressing candidate is therefore the 10 USD one, not the 15 USD superset.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{
    EvidenceObligation, EvidenceObligationKind, EvidenceObligationSpec, ObligationProvenance,
    ObligationRegime, ObligationScope,
};
use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceCatalogDelta, EvidenceKind,
    InterventionAssignment, RegimeKind, Value, VariableId,
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
        quantities: std::collections::BTreeMap::new(),
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

const W: u32 = 3;

fn variant(
    label: &str,
    population: &str,
    interventions: &[u32],
    measured: &[u32],
    joint: bool,
    cost: u64,
) -> DurableStudyCandidate {
    let mut c = candidate_c();
    c.label = Arc::from(label);
    c.population = Arc::from(population);
    c.interventions = vars(interventions);
    c.measured = vars(measured);
    c.cost.units = cost;
    let level_values = if interventions.is_empty() { Arc::from([]) } else { level(A, 1.0) };
    c.expected_evidence = Arc::from([ExpectedEvidence {
        population: Arc::from(population),
        interventions: vars(interventions),
        intervention_values: level_values,
        conditioned_on: Arc::from([]),
        measured: vars(measured),
        distribution: if joint {
            DistributionAvailability::Joint
        } else {
            DistributionAvailability::SeparateMarginals { variables: vars(measured) }
        },
    }]);
    c.joint_measurement = joint || measured.len() == 1;
    if interventions.is_empty() {
        c.kind = StudyKind::Observation;
    }
    c
}

#[test]
fn the_addressing_table_matches_the_hand_derivation_and_picks_the_cheapest() {
    let obligation = joint_obligation();
    let table = [
        (variant("exact", "T", &[A], &[Y, Z], true, 10), true),
        (variant("superset", "T", &[A], &[Y, Z, W], true, 15), true),
        (variant("marginals", "T", &[A], &[Y, Z], false, 5), false),
        (variant("other_population", "S", &[A], &[Y, Z], true, 3), false),
        (variant("observational", "T", &[], &[Y, Z], true, 2), false),
        (variant("y_only", "T", &[A], &[Y], true, 1), false),
    ];
    for (candidate, expected) in &table {
        candidate.validate().unwrap();
        let addressed = candidate.offers().iter().any(|offer| obligation.addressed_by(offer));
        assert_eq!(addressed, *expected, "{}", candidate.label);
    }
    let cheapest = table
        .iter()
        .filter(|(c, _)| c.offers().iter().any(|o| obligation.addressed_by(o)))
        .min_by_key(|(c, _)| c.cost.units)
        .map(|(c, _)| c.label.to_string());
    assert_eq!(cheapest.as_deref(), Some("exact"));
}

#[test]
fn identity_ignores_order_and_label_but_tracks_every_declared_semantic() {
    let base = candidate_c();
    let mut reordered = base.clone();
    reordered.label = Arc::from("renamed");
    reordered.measured = vars(&[Z, Y]);
    reordered.expected_evidence = Arc::from([evidence(1.0, &[Z, Y], true)]);
    assert_eq!(base.semantic_id(), reordered.semantic_id());
    let mut bigger = base.clone();
    bigger.sample_size = 101;
    let mut dearer = base.clone();
    dearer.cost.units = 11;
    let mut other_level = base.clone();
    other_level.expected_evidence = Arc::from([evidence(0.0, &[Y, Z], true)]);
    for changed in [&bigger, &dearer, &other_level] {
        assert_ne!(base.semantic_id(), changed.semantic_id());
    }
    // The proposed regime of the one expected law: experimental, proposed, tied to the id.
    let regimes = base.proposed_regimes(7).unwrap();
    assert_eq!(regimes.len(), 1);
    assert_eq!(regimes[0].kind, RegimeKind::Experimental);
    assert_eq!(regimes[0].evidence_kind, EvidenceKind::Proposed);
    assert_eq!(regimes[0].study.as_deref(), Some(&*base.semantic_id()));
}

#[test]
fn a_joint_claim_without_joint_measurement_is_refused() {
    let mut c = candidate_c();
    c.joint_measurement = false;
    let error = c.validate().unwrap_err();
    assert_eq!(error.detail, antecedent_core::reason_details::study_candidate::WRONG_CONTRACT);
}
