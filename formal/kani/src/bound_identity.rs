//! `antecedent_io_core::bound_identity::bind` binds a presented identity only
//! against an expected identity the consumer retained.
//!
//! The facet digest is replaced by an arbitrary value (`facet_digest_any`), so
//! these proofs use no property of the hash: they hold for every digest function.
//! The Lean statement of the same properties is `Antecedent.Acceptance`.

use antecedent_core::SemanticDigest;
use antecedent_io_core::bound_identity::{
    BoundIdentity, BoundMatch, CyclicSolution, ExpectedIdentity, IdentitySlot, SelfDescribedDigest,
    bind,
};

fn any_digest() -> SemanticDigest {
    SemanticDigest::from_bytes(kani::any())
}

fn any_slot() -> IdentitySlot {
    let index: usize = kani::any();
    kani::assume(index < IdentitySlot::ALL.len());
    IdentitySlot::ALL[index]
}

fn any_cyclic() -> CyclicSolution {
    if kani::any() { CyclicSolution::Absent } else { CyclicSolution::Tag(any_digest()) }
}

fn any_bound() -> BoundIdentity {
    BoundIdentity {
        slot: any_slot(),
        payload: any_digest(),
        coordinates: any_digest(),
        prior: any_digest(),
        alignment: any_digest(),
        cyclic_solution: any_cyclic(),
        self_description: if kani::any() {
            Some(SelfDescribedDigest { advertised: any_digest() })
        } else {
            None
        },
    }
}

fn any_expected() -> ExpectedIdentity {
    ExpectedIdentity {
        payload: any_digest(),
        coordinates: any_digest(),
        prior: any_digest(),
        alignment: any_digest(),
        cyclic_solution: any_cyclic(),
    }
}

/// An arbitrary digest function in place of BLAKE3.
fn facet_digest_any(_bound: &BoundIdentity) -> SemanticDigest {
    any_digest()
}

fn facets_equal(bound: &BoundIdentity, expected: &ExpectedIdentity) -> bool {
    bound.payload == expected.payload
        && bound.coordinates == expected.coordinates
        && bound.prior == expected.prior
        && bound.alignment == expected.alignment
        && bound.cyclic_solution == expected.cyclic_solution
}

/// `Bound` means the slot matched, an expectation was retained, and every facet equals it.
#[kani::proof]
#[kani::stub(antecedent_io_core::bound_identity::facet_digest, facet_digest_any)]
#[kani::unwind(33)]
fn bound_only_against_the_retained_expectation() {
    let required = any_slot();
    let bound = any_bound();
    let expected = if kani::any() { Some(any_expected()) } else { None };
    if let Ok(BoundMatch::Bound { slot }) = bind(required, &bound, expected.as_ref()) {
        assert!(slot == required && bound.slot == required);
        let Some(expected) = expected else {
            panic!("bound without a retained expectation");
        };
        assert!(facets_equal(&bound, &expected));
    }
}

/// Without a retained expectation the outcome is never `Bound`.
#[kani::proof]
#[kani::stub(antecedent_io_core::bound_identity::facet_digest, facet_digest_any)]
#[kani::unwind(33)]
fn no_expectation_is_never_bound() {
    let required = any_slot();
    let bound = any_bound();
    let outcome = bind(required, &bound, None);
    assert!(!matches!(outcome, Ok(BoundMatch::Bound { .. })));
}

/// A presentation in the right slot whose facets equal the expectation, and that makes no
/// self-description, is bound.
#[kani::proof]
#[kani::stub(antecedent_io_core::bound_identity::facet_digest, facet_digest_any)]
#[kani::unwind(33)]
fn matching_presentation_is_bound() {
    let mut bound = any_bound();
    bound.self_description = None;
    let expected = ExpectedIdentity {
        payload: bound.payload,
        coordinates: bound.coordinates,
        prior: bound.prior,
        alignment: bound.alignment,
        cyclic_solution: bound.cyclic_solution,
    };
    let outcome = bind(bound.slot, &bound, Some(&expected));
    assert!(matches!(outcome, Ok(BoundMatch::Bound { slot }) if slot == bound.slot));
}

/// The claim slot `accept_claim_expecting` uses binds exactly when the replayed claim id
/// equals the claim id the consumer retained, for every pair of ids.
#[kani::proof]
#[kani::stub(antecedent_io_core::bound_identity::facet_digest, facet_digest_any)]
#[kani::unwind(33)]
fn claim_binds_only_the_retained_claim_id() {
    let replayed = any_digest();
    let retained = any_digest();
    let outcome = bind(
        IdentitySlot::Claim,
        &BoundIdentity::claim(replayed),
        Some(&ExpectedIdentity::claim(retained)),
    );
    assert_eq!(matches!(outcome, Ok(BoundMatch::Bound { .. })), replayed == retained);
}
