//! Transport provenance in the evidence ledger against hand-counted rows.
//!
//! Every expected number below is counted by hand from the declared row ranges: a native
//! cohort of rows 0..30 and a transported claim fitted on rows 100..160 hold 30 + 60 = 90
//! slots and 90 distinct observations; a transported claim fitted on rows 20..50 shares the
//! ten rows 20..30 with the native cohort. The transport identity a result reports is the
//! digest of the declared identity, so an identical declaration reproduces it and a changed
//! target population changes it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::Range;

use antecedent_contract::evidence_ledger::{
    CombinationRequest, EvidenceLedger, LedgerError, ObservationId, PoolingDependence, Revalidation,
    SourceDeclaration, SourceId, SourceKind, SynthesisOperation, TransportIdentity,
};
use antecedent_core::SemanticDigest;

fn obs(range: Range<u32>) -> Vec<ObservationId> {
    range.map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
}

fn identity(target: &str) -> TransportIdentity {
    TransportIdentity {
        source_populations: vec!["trial_a".to_owned(), "trial_b".to_owned()],
        target_population: target.to_owned(),
        selection_diagram: SemanticDigest::from_bytes([7; 32]),
        assumptions: vec!["s-admissible adjustment set".to_owned()],
    }
}

fn native(ledger: &mut EvidenceLedger, label: &str, rows: Range<u32>) -> SourceId {
    ledger
        .add_source(SourceDeclaration::new(label, SourceKind::ObservationSet).with_observations(obs(rows)))
        .expect("native source")
}

fn transported(
    ledger: &mut EvidenceLedger,
    label: &str,
    rows: Range<u32>,
    target: &str,
) -> SourceId {
    ledger
        .add_source(
            SourceDeclaration::new(label, SourceKind::ExternalClaim)
                .with_observations(obs(rows))
                .with_transport(identity(target)),
        )
        .expect("transported source")
}

fn transport_request(label: &str, transported: SourceId, native: SourceId) -> CombinationRequest {
    CombinationRequest {
        label: label.to_owned(),
        operation: SynthesisOperation::CausalTransport {
            transported: vec![transported],
            native: vec![native],
        },
        replaces: None,
    }
}

fn expect_refusal(error: &LedgerError, code: &str, detail: &str) {
    assert_eq!(error.code, code, "{error}");
    assert_eq!(error.detail, detail, "{error}");
}

/// Disjoint native and transported rows combine; the result counts 30 + 60 = 90 slots and 90
/// distinct observations with none repeated, and carries exactly one transport identity, the
/// digest of the declared identity.
#[test]
fn disjoint_transport_counts_hand_rows_and_reports_the_declared_identity() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "target_cohort", 0..30);
    let claim = transported(&mut ledger, "effect_from_trials", 100..160, "target");
    let result = ledger.combine(transport_request("transport", claim, cohort)).expect("disjoint");
    let record = ledger.result(result).expect("record");
    assert_eq!(record.mass.sources, 2);
    assert_eq!(record.mass.declared_slots, 90);
    assert_eq!(record.mass.distinct_observations, 90);
    assert_eq!(record.mass.repeated_slots, 0);
    assert_eq!(record.transport_identities, vec![identity("target").digest()]);
}

/// The reported identity is a function of the declared content: another target population is
/// another identity, and the same declaration made twice is the same identity.
#[test]
fn the_reported_identity_follows_the_declared_content() {
    assert_eq!(identity("target").digest(), identity("target").digest());
    assert_ne!(identity("target").digest(), identity("elsewhere").digest());
    let mut fewer = identity("target");
    fewer.assumptions.clear();
    assert_ne!(identity("target").digest(), fewer.digest(), "the declared assumptions are part of it");
}

/// A transported claim fitted on rows 20..50 shares rows 20..30 with a native cohort of rows
/// 0..30: the ten shared rows make it the same evidence counted twice, whatever else is declared.
#[test]
fn a_transported_claim_that_reuses_ten_native_rows_is_refused() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "target_cohort", 0..30);
    let claim = transported(&mut ledger, "claim_from_target_rows", 20..50, "target");
    let error = ledger
        .combine(transport_request("transport", claim, cohort))
        .expect_err("ten rows are shared");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.native_and_transported");

    // Moving the transported rows to 30..60 removes the overlap and the same request is accepted.
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "target_cohort", 0..30);
    let claim = transported(&mut ledger, "claim_from_other_rows", 30..60, "target");
    let result = ledger.combine(transport_request("transport", claim, cohort)).expect("disjoint");
    let record = ledger.result(result).expect("record");
    assert_eq!(record.mass.distinct_observations, 60);
    assert_eq!(record.mass.repeated_slots, 0);
}

/// A source without a transport identity cannot fill the transported slot, and a transported
/// source cannot fill the native slot.
#[test]
fn the_transported_and_native_slots_are_not_interchangeable() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "target_cohort", 0..30);
    let bare = native(&mut ledger, "bare_claim_data", 300..310);
    let claim = transported(&mut ledger, "effect_from_trials", 100..160, "target");
    let other = transported(&mut ledger, "other_claim", 400..410, "target");

    let error = ledger
        .combine(transport_request("bare", bare, cohort))
        .expect_err("no identity, not transported");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.transport_identity_missing");

    let error = ledger
        .combine(transport_request("swapped", claim, other))
        .expect_err("a transported claim is not native");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.transported_as_native");
}

fn covariance_pool(label: &str, ids: &[SourceId]) -> CombinationRequest {
    CombinationRequest {
        label: label.to_owned(),
        operation: SynthesisOperation::StatisticalPooling {
            sources: ids.to_vec(),
            dependence: PoolingDependence::DeclaredCovariance {
                covariance: SemanticDigest::from_bytes([9; 32]),
                has_cross_covariance: true,
            },
        },
        replaces: None,
    }
}

/// Two trial claims on rows 100..160 and 140..200 share rows 140..160. Pooled under a declared
/// covariance they hold 60 + 60 = 120 slots over the union 100..200 = 100 distinct rows, so 20
/// slots are repeated; the result carries the transport identity of both claims (two digests).
#[test]
fn overlapping_transported_claims_pool_with_hand_counted_union_and_two_identities() {
    let mut ledger = EvidenceLedger::new();
    let a = transported(&mut ledger, "trial_a_claim", 100..160, "target");
    let b = transported(&mut ledger, "trial_b_claim", 140..200, "target");
    let result = ledger.combine(covariance_pool("pool", &[a, b])).expect("covariance carries overlap");
    let record = ledger.result(result).expect("record");
    assert_eq!(record.mass.declared_slots, 120);
    assert_eq!(record.mass.distinct_observations, 100);
    assert_eq!(record.mass.repeated_slots, 20);
    // Both claims declare the same transport identity, and a result reports distinct identities
    // (digests), so the shared declaration is reported once, not once per claim.
    assert_eq!(record.transport_identities, vec![identity("target").digest()]);

    // A second claim that declares another target population contributes its own identity.
    let mut ledger = EvidenceLedger::new();
    let a = transported(&mut ledger, "trial_a_claim", 100..160, "target");
    let b = transported(&mut ledger, "trial_b_claim", 140..200, "elsewhere");
    let result = ledger.combine(covariance_pool("pool", &[a, b])).expect("covariance carries overlap");
    let record = ledger.result(result).expect("record");
    assert_eq!(record.transport_identities.len(), 2);
    assert!(record.transport_identities.contains(&identity("target").digest()));
    assert!(record.transport_identities.contains(&identity("elsewhere").digest()));
}

/// Citing a pooled result of overlapping rows 0..100 and 50..150 reports the pooled mass (200
/// slots, 150 distinct, 50 repeated) every time, never a multiple.
#[test]
fn citing_an_overlapping_pool_reports_its_hand_counted_union_each_time() {
    let mut ledger = EvidenceLedger::new();
    let a = native(&mut ledger, "a", 0..100);
    let b = native(&mut ledger, "b", 50..150);
    let pooled = ledger.combine(covariance_pool("pool", &[a, b])).expect("pooled");
    for label in ["first_citation", "second_citation", "third_citation"] {
        let cite = ledger
            .combine(CombinationRequest {
                label: label.to_owned(),
                operation: SynthesisOperation::EvidenceReuse { of: pooled },
                replaces: None,
            })
            .expect("reuse");
        let mass = &ledger.result(cite).expect("record").mass;
        assert_eq!(
            (mass.declared_slots, mass.distinct_observations, mass.repeated_slots),
            (200, 150, 50)
        );
    }
}

/// A repeated-use declaration over overlapping stages rows 0..30 and 20..50 counts 60 slots over
/// 50 distinct rows (10 repeated) and is flagged data_reuse.
#[test]
fn repeated_use_over_overlapping_stages_counts_the_union_once() {
    let mut ledger = EvidenceLedger::new();
    let a = native(&mut ledger, "interim", 0..30);
    let b = native(&mut ledger, "final", 20..50);
    let result = ledger
        .combine(CombinationRequest {
            label: "interim_then_final".to_owned(),
            operation: SynthesisOperation::RepeatedObservationUse {
                sources: vec![a, b],
                stage: "final_analysis".to_owned(),
            },
            replaces: None,
        })
        .expect("declared");
    let record = ledger.result(result).expect("record");
    assert_eq!(record.mass.declared_slots, 60);
    assert_eq!(record.mass.distinct_observations, 50);
    assert_eq!(record.mass.repeated_slots, 10);
    assert!(record.data_reuse);
}

/// Sources a = 0..40, b = 20..60, c = 100..140 and the pools P(a,b), Q(b,c), R(a,b,c), S(a,c).
/// Before any change R holds 120 slots over 100 distinct rows (a and b share 20). Withdrawing b
/// leaves P and Q with one member (withdrawn), R with two (recompute) and S untouched. The
/// recomputed R on {a, c} counts 80 slots over 80 distinct rows, no repeats, and replaces R.
#[test]
fn withdrawing_an_overlapping_source_reclassifies_and_the_recompute_recounts_the_union() {
    let mut ledger = EvidenceLedger::new();
    let a = native(&mut ledger, "a", 0..40);
    let b = native(&mut ledger, "b", 20..60);
    let c = native(&mut ledger, "c", 100..140);
    let p = ledger.combine(covariance_pool("P", &[a, b])).expect("P");
    let q = ledger.combine(covariance_pool("Q", &[b, c])).expect("Q");
    let r = ledger.combine(covariance_pool("R", &[a, b, c])).expect("R");
    let independent = CombinationRequest {
        label: "S".to_owned(),
        operation: SynthesisOperation::StatisticalPooling {
            sources: vec![a, c],
            dependence: PoolingDependence::Independent,
        },
        replaces: None,
    };
    let s = ledger.combine(independent).expect("S");
    let before = ledger.result(r).expect("R").mass.clone();
    assert_eq!((before.declared_slots, before.distinct_observations, before.repeated_slots), (120, 100, 20));

    let applied = ledger.withdraw_source(b, "consent withdrawn").expect("withdraw").clone();
    let class = |id| applied.effect.revalidation.iter().find(|x| x.result == id).expect("classified").class;
    assert_eq!(class(p), Revalidation::Withdrawn);
    assert_eq!(class(q), Revalidation::Withdrawn);
    assert_eq!(class(r), Revalidation::Recompute);
    assert_eq!(class(s), Revalidation::StillValid);

    let recomputed = ledger
        .combine(CombinationRequest {
            label: "R2".to_owned(),
            operation: SynthesisOperation::StatisticalPooling {
                sources: vec![a, c],
                dependence: PoolingDependence::Independent,
            },
            replaces: Some(r),
        })
        .expect("recompute on the remaining sources");
    let mass = &ledger.result(recomputed).expect("R2").mass;
    assert_eq!((mass.declared_slots, mass.distinct_observations, mass.repeated_slots), (80, 80, 0));
    assert_eq!(ledger.standing(r), Some(Revalidation::Withdrawn));
    assert_eq!(ledger.standing(recomputed), Some(Revalidation::StillValid));
}
