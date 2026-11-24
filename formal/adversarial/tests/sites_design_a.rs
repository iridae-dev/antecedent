//! Identity sites of `formal/identity_sites.toml` in antecedent-design (group A), checked on
//! the production encoders through the public API with the category laws of `common`.
//!
//! * `DurableStudyCandidate::semantic_id`: variable lists, feasibility notes and expected laws
//!   are sets; free text never moves content between fields (framing injective, including on
//!   text containing the v1 separators).
//! * `ExpectedEvidence::canonical`: the v1 content string, a multiset of variables per list.
//! * `EvidencePlan::identity`: an option's resolved and unresolved obligation ids are two
//!   framed lists; an unverified route's obligation ids are framed.
//! * `SignalRequest::fingerprint`: the exact-request rule (lineage order is irrelevant, a
//!   repeated lineage id is a different request).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{DistributionAvailability, VariableId};
use antecedent_design::signal::{SignalLimits, SignalRequest};
use antecedent_design::{
    DurableStudyCandidate, EvidencePlan, ExpectedEvidence, PlanCost, PlanOption, PlanOutcome,
    PlanValueBasis, RepairClassification, StudyCostDeclaration, StudyKind, UnitRules,
    UnverifiedRoute,
};
use common::{adversarial_text, config, distinct_permuted, injective, multiset_laws, set_laws};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Study candidates.
// ---------------------------------------------------------------------------

/// Free-text parts of a candidate.
#[derive(Clone, Debug)]
struct CandidateText {
    recruitment: String,
    timing: String,
    notes: Vec<String>,
    provider: Option<String>,
}

fn candidate_text() -> impl Strategy<Value = CandidateText> {
    (
        adversarial_text(),
        adversarial_text(),
        prop::collection::vec(adversarial_text(), 0..3),
        prop::option::of(adversarial_text()),
    )
        .prop_map(|(recruitment, timing, notes, provider)| CandidateText {
            recruitment,
            timing,
            notes,
            provider,
        })
}

fn evidence(measured: &[u32]) -> ExpectedEvidence {
    ExpectedEvidence {
        population: Arc::from("target"),
        interventions: Arc::from([]),
        intervention_values: Arc::from([]),
        conditioned_on: Arc::from([]),
        measured: measured.iter().map(|&raw| VariableId::from_raw(raw)).collect(),
        distribution: DistributionAvailability::Joint,
    }
}

fn candidate(text: &CandidateText, measured: &[u32], laws: &[Vec<u32>]) -> DurableStudyCandidate {
    DurableStudyCandidate {
        label: Arc::from("label"),
        kind: StudyKind::Observation,
        population: Arc::from("target"),
        interventions: Arc::from([]),
        measured: measured.iter().map(|&raw| VariableId::from_raw(raw)).collect(),
        joint_measurement: true,
        sample_size: 10,
        recruitment: Arc::from(text.recruitment.as_str()),
        timing: Arc::from(text.timing.as_str()),
        unit_rules: UnitRules {
            unit: Arc::from("patient"),
            cluster: None,
            whole_cluster_sampling: false,
        },
        cost: StudyCostDeclaration { units: 1, unit_label: Arc::from("usd"), sample_budget: 10 },
        feasible: true,
        feasibility_notes: text.notes.iter().map(|note| Arc::from(note.as_str())).collect(),
        expected_evidence: laws.iter().map(|law| evidence(law)).collect(),
        external_provider: text.provider.as_deref().map(Arc::from),
    }
}

fn plain_text() -> CandidateText {
    CandidateText {
        recruitment: "mail".into(),
        timing: "now".into(),
        notes: vec!["n".into()],
        provider: None,
    }
}

/// The candidate's text as the identity rule reads it: notes as a set.
fn text_key(text: &CandidateText) -> (String, String, Vec<String>, Option<String>) {
    let mut notes = text.notes.clone();
    notes.sort();
    notes.dedup();
    (text.recruitment.clone(), text.timing.clone(), notes, text.provider.clone())
}

// ---------------------------------------------------------------------------
// Evidence plans.
// ---------------------------------------------------------------------------

fn option(resolves: &[String], unresolved: &[String]) -> PlanOption {
    PlanOption {
        rank: 0,
        route: "backdoor",
        members: vec![Arc::from("sc1:member")],
        labels: vec![Arc::from("member")],
        gate: RepairClassification::NotCertified,
        reasons: vec![],
        derivation: None,
        resolves: resolves.iter().map(|id| Arc::from(id.as_str())).collect(),
        unresolved: unresolved.iter().map(|id| Arc::from(id.as_str())).collect(),
        assumptions_remaining: vec![],
        cost: PlanCost { units: 1, unit_label: Arc::from("usd"), sample_budget: 10 },
        value: None,
        value_note: None,
        rank_uncertain: false,
    }
}

fn plan(
    resolves: &[String],
    unresolved: &[String],
    unverified: &[(String, Vec<String>)],
) -> EvidencePlan {
    EvidencePlan {
        routes: vec![],
        unverified: unverified
            .iter()
            .map(|(route, obligations)| UnverifiedRoute {
                route: Arc::from(route.as_str()),
                obligations: obligations.iter().map(|id| Arc::from(id.as_str())).collect(),
                reason: Arc::from("no verifier"),
            })
            .collect(),
        options: vec![option(resolves, unresolved)],
        outcome: PlanOutcome::NoneCertified,
        cost_unit: Arc::from("usd"),
        basis: PlanValueBasis::None,
        cost_map: None,
        utility_unit: None,
        value_identity: None,
    }
}

type PlanParts = (Vec<String>, Vec<String>, Vec<(String, Vec<String>)>);

fn plan_parts() -> impl Strategy<Value = PlanParts> {
    let ids = || prop::collection::vec("[ab]{1,2}", 0..3);
    (ids(), ids(), prop::collection::vec(("[ab]{1,2}", ids()), 0..3))
}

// ---------------------------------------------------------------------------
// Signal requests.
// ---------------------------------------------------------------------------

fn quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn request(lineage: &[String]) -> SignalRequest {
    SignalRequest {
        candidate_id: "sc1:candidate".into(),
        prior_id: "prior".into(),
        state_quantity: quantity("state"),
        observation_quantity: quantity("observation"),
        sample_size: 10,
        rng_seed: 7,
        evidence_lineage: lineage.to_vec(),
        conditional_independence: "ci".into(),
        limits: SignalLimits::default(),
    }
}

proptest! {
    #![proptest_config(config())]

    /// `DurableStudyCandidate::semantic_id`: notes, measured variables and expected laws are
    /// sets; two candidates share the identity exactly when their text agrees (notes as a set),
    /// whatever separators the text contains.
    #[test]
    fn site_study_candidate_semantic_id(
        (notes, shuffled_notes) in distinct_permuted("[a-c]{1,3}", 0..4),
        (measured, shuffled_measured) in distinct_permuted(0u32..8, 1..5),
        left in candidate_text(),
        right in candidate_text(),
    ) {
        let with_notes = |notes: &[String]| {
            let mut text = plain_text();
            text.notes = notes.to_vec();
            Some(candidate(&text, &[0, 1], &[vec![0]]).semantic_id())
        };
        set_laws(&notes, &shuffled_notes, with_notes)?;
        set_laws(&measured, &shuffled_measured, |measured| {
            Some(candidate(&plain_text(), measured, &[vec![0]]).semantic_id())
        })?;
        let laws: Vec<Vec<u32>> = measured.iter().map(|&variable| vec![variable]).collect();
        let shuffled_laws: Vec<Vec<u32>> =
            shuffled_measured.iter().map(|&variable| vec![variable]).collect();
        set_laws(&laws, &shuffled_laws, |laws| {
            Some(candidate(&plain_text(), &[0, 1], laws).semantic_id())
        })?;
        injective(&left, &right, text_key, |text| {
            Some(candidate(text, &[0, 1], &[vec![0]]).semantic_id())
        })?;
    }

    /// `ExpectedEvidence::canonical`: the v1 content string lists each variable list sorted
    /// with repeats kept (a multiset).
    #[test]
    fn site_expected_evidence_canonical(
        (measured, shuffled) in common::permuted(0u32..6, 1..5),
    ) {
        multiset_laws(&measured, &shuffled, |measured| Some(evidence(measured).canonical()))?;
    }

    /// `EvidencePlan::identity`: an option's resolved and unresolved ids, and each unverified
    /// route's obligation ids, are framed lists; moving an id between lists, or between an
    /// obligation list and the next route, changes the identity.
    #[test]
    fn site_evidence_plan_identity(left in plan_parts(), right in plan_parts()) {
        injective(&left, &right, Clone::clone, |(resolves, unresolved, unverified)| {
            Some(plan(resolves, unresolved, unverified).identity())
        })?;
        let (resolves, unresolved, unverified) = &left;
        let current = plan(resolves, unresolved, unverified);
        prop_assert!(current.accepts_identity(&current.identity_v1()));
    }

    /// `SignalRequest::fingerprint`: the exact-request rule. Lineage order does not matter; a
    /// repeated lineage id is a different request.
    #[test]
    fn site_signal_request_fingerprint(
        (lineage, shuffled) in common::permuted("[a-c]{1,2}", 1..4),
    ) {
        multiset_laws(&lineage, &shuffled, |lineage| Some(request(lineage).fingerprint()))?;
    }
}
