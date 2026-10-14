//! Identity properties of the recalculation standing inputs: field sensitivity, absence as a
//! value, domain separation between input kinds and branches, and length-delimited encoding.
//! Plain deterministic mutation tests; the workspace carries no property-test dependency.

use std::collections::BTreeSet;

use antecedent_core::recalc::{
    Branch, CalibrationInput, CouplingInput, ProviderStandingInput, RepresentationInput,
    StageIdentity,
};

fn id(tag: &str) -> StageIdentity {
    StageIdentity::of("identity_properties.fixture", &[tag.as_bytes()])
}

fn all_distinct(identities: &[StageIdentity]) -> bool {
    let set: BTreeSet<[u8; 32]> = identities.iter().map(|i| *i.as_bytes()).collect();
    set.len() == identities.len()
}

#[test]
fn calibration_variants_and_fields_are_all_distinct() {
    let a = id("record-a");
    let b = id("record-b");
    let c = id("coordinate-c");
    let d = id("coordinate-d");
    let identities = [
        CalibrationInput::Unmeasured.identity(),
        CalibrationInput::PointOnly.identity(),
        CalibrationInput::Held { record: a, coordinate: c }.identity(),
        CalibrationInput::Held { record: b, coordinate: c }.identity(),
        CalibrationInput::Held { record: a, coordinate: d }.identity(),
        CalibrationInput::Held { record: c, coordinate: a }.identity(),
        CalibrationInput::Withdrawn { record: a }.identity(),
        CalibrationInput::Withdrawn { record: b }.identity(),
    ];
    assert!(all_distinct(&identities));
    // Same input, same identity.
    assert_eq!(
        CalibrationInput::Held { record: a, coordinate: c }.identity(),
        CalibrationInput::Held { record: a, coordinate: c }.identity()
    );
    // A record that is withdrawn is not the record held, whatever coordinate it had.
    assert_ne!(
        CalibrationInput::Withdrawn { record: a }.identity(),
        CalibrationInput::Held { record: a, coordinate: StageIdentity::ABSENT }.identity()
    );
}

#[test]
fn absence_digests_differently_from_every_held_record() {
    let held = CalibrationInput::Held { record: StageIdentity::ABSENT, coordinate: StageIdentity::ABSENT };
    assert_ne!(CalibrationInput::Unmeasured.identity(), held.identity());
    assert_ne!(CalibrationInput::Unmeasured.identity(), StageIdentity::ABSENT);
    assert_ne!(CouplingInput::None.identity(), StageIdentity::ABSENT);
    assert_ne!(
        ProviderStandingInput::Unverified.identity(Branch::new(0).unwrap()),
        StageIdentity::ABSENT
    );
}

#[test]
fn representation_is_length_delimited_and_field_sensitive() {
    let base = RepresentationInput::new("linear_gaussian_scm", "3").identity();
    assert_eq!(base, RepresentationInput::new("linear_gaussian_scm", "3").identity());
    assert_ne!(base, RepresentationInput::new("linear_gaussian_scm", "4").identity());
    assert_ne!(base, RepresentationInput::new("linear_gaussian_scm_", "3").identity());
    // Moving the boundary between family and version must change the identity.
    assert_ne!(
        RepresentationInput::new("ab", "c").identity(),
        RepresentationInput::new("a", "bc").identity()
    );
    // Swapping family and version must change it.
    assert_ne!(
        RepresentationInput::new("x", "y").identity(),
        RepresentationInput::new("y", "x").identity()
    );
}

#[test]
fn coupling_variants_are_distinct_and_carry_their_source() {
    let a = id("source-a");
    let b = id("source-b");
    let identities = [
        CouplingInput::None.identity(),
        CouplingInput::DeclaredIndependent.identity(),
        CouplingInput::Unknown.identity(),
        CouplingInput::SharedExogenous { source: a }.identity(),
        CouplingInput::SharedExogenous { source: b }.identity(),
        CouplingInput::AlignedJoint { alignment: a }.identity(),
        CouplingInput::AlignedJoint { alignment: b }.identity(),
    ];
    assert!(all_distinct(&identities));
}

#[test]
fn identical_payload_in_different_input_kinds_never_collides() {
    // Domain separation: the same 32 bytes under different input kinds.
    let x = id("shared-payload");
    let identities = [
        CalibrationInput::Withdrawn { record: x }.identity(),
        CouplingInput::SharedExogenous { source: x }.identity(),
        CouplingInput::AlignedJoint { alignment: x }.identity(),
        ProviderStandingInput::Verified { level: "attested".into(), receipt: x }
            .identity(Branch::new(0).unwrap()),
    ];
    assert!(all_distinct(&identities));
}

#[test]
fn provider_standing_is_branch_level_and_receipt_sensitive() {
    let b0 = Branch::new(0).unwrap();
    let b1 = Branch::new(1).unwrap();
    let r = id("receipt-1");
    let verified = |level: &str, receipt| ProviderStandingInput::Verified {
        level: level.into(),
        receipt,
    };
    let identities = [
        ProviderStandingInput::Unverified.identity(b0),
        ProviderStandingInput::Unverified.identity(b1),
        ProviderStandingInput::Revoked.identity(b0),
        ProviderStandingInput::Revoked.identity(b1),
        verified("attested", r).identity(b0),
        verified("attested", r).identity(b1),
        verified("verified", r).identity(b0),
        verified("attested", id("receipt-2")).identity(b0),
    ];
    assert!(all_distinct(&identities));
}

#[test]
fn stage_identity_parts_are_length_prefixed_and_labelled() {
    let one = StageIdentity::of("label", &[b"ab", b"c"]);
    assert_ne!(one, StageIdentity::of("label", &[b"a", b"bc"]));
    assert_ne!(one, StageIdentity::of("label", &[b"abc"]));
    assert_ne!(one, StageIdentity::of("other", &[b"ab", b"c"]));
    assert_ne!(StageIdentity::of("label", &[]), StageIdentity::of("label", &[b""]));
    assert_eq!(one, StageIdentity::of("label", &[b"ab", b"c"]));
}
