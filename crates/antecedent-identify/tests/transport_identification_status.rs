//! Every transport binding reports its identification status in one vocabulary.
//!
//! The canonical spellings are `TransportOutcomeKind::IDENTIFICATION`. The legacy
//! spellings a binding still emits where they are serialized or public (a scenario's
//! `status()`, a temporal decision's `status()`, a Python stage's `outcome`) read as
//! the same kind the binding's `identification_status()` reports.

use std::sync::Arc;

use antecedent_core::{SearchReceipt, SearchStop, TransportOutcomeKind as K};
use antecedent_identify::sid::scenarios::ScenarioOutcome;
use antecedent_identify::sid::temporal_sequence::TemporalOutcome;
use antecedent_identify::sid::{
    ConditionalTransportDecision, MixedSourceDecision, MzTransportDecision,
    TwoSourceZTransportDecision,
};
use antecedent_identify::{CatalogTransportResult, ClassicalTransportResult};

fn receipt() -> SearchReceipt {
    SearchReceipt {
        stop: SearchStop::Operations,
        operations_limit: 1,
        depth_limit: 1,
        memory_limit_bytes: Some(1),
        operations_consumed: Some(1),
        depth_reached: Some(0),
        explored: vec![],
        unevaluated: vec![],
    }
}

fn notes() -> Arc<[Arc<str>]> {
    Arc::from([Arc::from("scope")])
}

/// The legacy spelling a binding emits reads as the kind it reports.
fn same(legacy: &str, status: K) {
    assert!(K::IDENTIFICATION.contains(&status), "{} is an identification status", status.as_str());
    assert_eq!(K::from_identification_status(legacy), Some(status), "{legacy}");
    assert_eq!(K::from_identification_status(status.as_str()), Some(status));
}

#[test]
fn scenario_and_temporal_status_spellings_read_as_their_canonical_kind() {
    let scenarios = [
        (ScenarioOutcome::MissingEvidence { obligations: notes() }, K::MissingEvidence),
        (ScenarioOutcome::NotCertified { obligations: notes() }, K::NotCertified),
        (ScenarioOutcome::Unevaluated { stop: SearchStop::Cancelled }, K::BudgetCancel),
    ];
    for (outcome, kind) in scenarios {
        assert_eq!(outcome.identification_status(), kind);
        same(outcome.status(), kind);
    }
    let temporal = [
        (TemporalOutcome::MissingEvidence { obligations: notes() }, K::MissingEvidence),
        (TemporalOutcome::NotCertified { obligations: notes() }, K::NotCertified),
        (TemporalOutcome::Stopped { stop: SearchStop::Depth }, K::BudgetCancel),
    ];
    for (outcome, kind) in temporal {
        assert_eq!(outcome.identification_status(), kind);
        same(outcome.status(), kind);
    }
    // The impossibility spelling these two bindings serialize.
    same("structurally_unidentified", K::ProvenNonTransportable);
}

#[test]
fn stage_outcome_spellings_read_as_their_canonical_kind() {
    // `exhausted` is the Python `outcome` of every budget-stopped stage.
    for status in [
        MzTransportDecision::Exhausted(receipt()).identification_status(),
        MixedSourceDecision::Exhausted(receipt()).identification_status(),
        ConditionalTransportDecision::Exhausted(receipt()).identification_status(),
    ] {
        assert_eq!(status, K::BudgetCancel);
        same("exhausted", status);
    }
    let named = MixedSourceDecision::NamedRoute { route: "z_transport", stages: vec![] };
    assert_eq!(named.identification_status(), K::Identified);
    same("named_route", named.identification_status());
    same("combined_identified", K::Identified);
    let declined = TwoSourceZTransportDecision::NotCertified { reason: "scope" };
    assert_eq!(declined.identification_status(), K::NotCertified);
    same("not_certified", declined.identification_status());
    assert_eq!(ClassicalTransportResult::NotCertified.identification_status(), K::NotCertified);
    let catalog = CatalogTransportResult::MissingEvidence {
        searched: Arc::from([Arc::from("classical")]),
        obligations: notes(),
    };
    assert_eq!(catalog.identification_status(), K::MissingEvidence);
    same("missing_evidence", catalog.identification_status());
}
