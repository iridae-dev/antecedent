//! `bind` binds a presented identity only against the expectation the consumer retained.
//!
//! The real BLAKE3 `facet_digest` is used. Positive cases present facets equal to the
//! expectation, with no self-description or an honest one. Negative cases mutate exactly one
//! expected facet, retain no expectation, advertise a false self-description, or present the
//! wrong slot; each must be refused with the matching typed variant or never reach `Bound`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::SemanticDigest;
use antecedent_io_core::bound_identity::{
    BoundIdentity, BoundMatch, BoundRefusal, CyclicSolution, ExpectedIdentity, IdentitySlot,
    SelfDescribedDigest, bind, facet_digest,
};
use proptest::prelude::*;

fn config() -> ProptestConfig {
    ProptestConfig { cases: 256, failure_persistence: None, ..ProptestConfig::default() }
}

fn digest() -> impl Strategy<Value = SemanticDigest> {
    any::<[u8; 32]>().prop_map(SemanticDigest::from_bytes)
}

fn slot() -> impl Strategy<Value = IdentitySlot> {
    prop::sample::select(IdentitySlot::ALL.to_vec())
}

fn cyclic() -> impl Strategy<Value = CyclicSolution> {
    prop_oneof![Just(CyclicSolution::Absent), digest().prop_map(CyclicSolution::Tag)]
}

/// A presented identity with no self-description.
fn bound() -> impl Strategy<Value = BoundIdentity> {
    (slot(), digest(), digest(), digest(), digest(), cyclic()).prop_map(
        |(slot, payload, coordinates, prior, alignment, cyclic_solution)| BoundIdentity {
            slot,
            payload,
            coordinates,
            prior,
            alignment,
            cyclic_solution,
            self_description: None,
        },
    )
}

/// The expectation a consumer would retain for exactly these facets.
fn retained(bound: &BoundIdentity) -> ExpectedIdentity {
    ExpectedIdentity {
        payload: bound.payload,
        coordinates: bound.coordinates,
        prior: bound.prior,
        alignment: bound.alignment,
        cyclic_solution: bound.cyclic_solution,
    }
}

fn flipped(digest: SemanticDigest, index: usize, mask: u8) -> SemanticDigest {
    let mut bytes = *digest.as_bytes();
    bytes[index % 32] ^= mask;
    SemanticDigest::from_bytes(bytes)
}

proptest! {
    #![proptest_config(config())]

    /// Facets equal to the retained expectation in the required slot are bound, whether the
    /// carrier makes no self-description or advertises the recomputed facet digest.
    #[test]
    fn matching_expectation_is_bound(mut presented in bound(), described in any::<bool>()) {
        let expected = retained(&presented);
        let required = presented.slot;
        if described {
            presented.self_description =
                Some(SelfDescribedDigest { advertised: facet_digest(&presented) });
        }
        prop_assert_eq!(
            bind(required, &presented, Some(&expected)),
            Ok(BoundMatch::Bound { slot: required })
        );
    }

    /// Changing exactly one retained facet is refused with that facet's typed variant.
    #[test]
    fn mismatched_expectation_is_rejected(
        presented in bound(),
        facet in 0usize..5,
        index in any::<usize>(),
        mask in 1..=255u8,
        replacement in digest(),
        described in any::<bool>(),
    ) {
        let mut presented = presented;
        if described {
            presented.self_description =
                Some(SelfDescribedDigest { advertised: facet_digest(&presented) });
        }
        let slot = presented.slot;
        let mut expected = retained(&presented);
        let refusal = match facet {
            0 => {
                expected.payload = flipped(expected.payload, index, mask);
                BoundRefusal::PayloadMismatch { slot }
            }
            1 => {
                expected.coordinates = flipped(expected.coordinates, index, mask);
                BoundRefusal::CoordinatesMismatch { slot }
            }
            2 => {
                expected.prior = flipped(expected.prior, index, mask);
                BoundRefusal::PriorMismatch { slot }
            }
            3 => {
                expected.alignment = flipped(expected.alignment, index, mask);
                BoundRefusal::AlignmentMismatch { slot }
            }
            _ => {
                expected.cyclic_solution = match expected.cyclic_solution {
                    CyclicSolution::Absent => CyclicSolution::Tag(replacement),
                    CyclicSolution::Tag(tag) if index % 2 == 0 => {
                        CyclicSolution::Tag(flipped(tag, index / 2, mask))
                    }
                    CyclicSolution::Tag(_) => CyclicSolution::Absent,
                };
                BoundRefusal::CyclicSolutionMismatch { slot }
            }
        };
        prop_assert_eq!(bind(slot, &presented, Some(&expected)), Err(refusal));
    }

    /// With no retained expectation the outcome is never `Bound`: it is `NotRequired` for a
    /// consistent presentation and a refusal otherwise.
    #[test]
    fn no_retained_expectation_is_never_bound(
        presented in bound(),
        advertised in prop::option::of(digest()),
        honest in any::<bool>(),
        wrong_slot in slot(),
    ) {
        let mut presented = presented;
        presented.self_description = if honest {
            Some(SelfDescribedDigest { advertised: facet_digest(&presented) })
        } else {
            advertised.map(|advertised| SelfDescribedDigest { advertised })
        };
        let outcome = bind(presented.slot, &presented, None);
        prop_assert!(!matches!(outcome, Ok(BoundMatch::Bound { .. })), "bound without expectation");
        if honest || presented.self_description.is_none() {
            prop_assert_eq!(outcome, Ok(BoundMatch::NotRequired { slot: presented.slot }));
        }
        let other = bind(wrong_slot, &presented, None);
        prop_assert!(!matches!(other, Ok(BoundMatch::Bound { .. })), "bound without expectation");
    }

    /// A self-description advertising any digest other than the recomputed facet digest is a
    /// false self-description, even when the retained expectation matches.
    #[test]
    fn false_self_description_is_rejected(
        presented in bound(),
        advertised in digest(),
        with_expectation in any::<bool>(),
    ) {
        let mut presented = presented;
        prop_assume!(advertised != facet_digest(&presented));
        presented.self_description = Some(SelfDescribedDigest { advertised });
        let slot = presented.slot;
        let expected = retained(&presented);
        let outcome = bind(slot, &presented, with_expectation.then_some(&expected));
        prop_assert_eq!(outcome, Err(BoundRefusal::FalseSelfDescription { slot }));
    }

    /// Presenting an identity for a slot other than the required one is a slot mismatch, even
    /// when every facet matches.
    #[test]
    fn wrong_slot_is_rejected(presented in bound(), required in slot(), described in any::<bool>()) {
        prop_assume!(required != presented.slot);
        let mut presented = presented;
        if described {
            presented.self_description =
                Some(SelfDescribedDigest { advertised: facet_digest(&presented) });
        }
        let expected = retained(&presented);
        prop_assert_eq!(
            bind(required, &presented, Some(&expected)),
            Err(BoundRefusal::SlotMismatch { presented: presented.slot, expected: required })
        );
    }
}
