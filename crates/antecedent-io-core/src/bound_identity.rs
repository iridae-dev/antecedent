//! Bound target, program, source, inference, and claim identities.
//!
//! [`bind`] compares one presented identity with the expected identity a
//! consumer retained independently. [`bind_all`] does that for each slot.
//! `None` means that consumer does not require an expected identity for the
//! slot. A self-described digest is not that retained identity.
//!
//! A presented identity mismatches when its payload, coordinates, prior,
//! alignment, or cyclic-solution tag differs from the retained expectation.
//! A self-description advertises one digest for the presented facets. The
//! consumer recomputes that digest from the facets it was handed
//! ([`facet_digest`]) and compares the two. A carrier cannot supply both sides of
//! the comparison. An advertisement that differs is a false self-description and
//! is refused. A matching one passes that check, and the retained expectation is
//! still compared.
//!
//! This module does not load an artifact. [`CyclicSolution::Tag`] is the
//! `cyclic.solution.v1` tag; comparing it does not solve a cyclic system.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::SemanticDigest;
use thiserror::Error;

/// Contract named by [`CyclicSolution::Tag`].
///
/// The tag is this contract's identity. Reading it does not solve the system.
pub const CYCLIC_SOLUTION_CONTRACT: &str = "cyclic.solution.v1";

/// Layer a consumer binds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum IdentitySlot {
    /// Causal target.
    Target,
    /// Licensed program.
    Program,
    /// Source snapshot.
    Source,
    /// Inference binding.
    Inference,
    /// Claim envelope.
    Claim,
}

impl IdentitySlot {
    /// Closed set, in the order [`bind_all`] reports.
    pub const ALL: [Self; 5] =
        [Self::Target, Self::Program, Self::Source, Self::Inference, Self::Claim];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Target => "target",
            Self::Program => "program",
            Self::Source => "source",
            Self::Inference => "inference",
            Self::Claim => "claim",
        }
    }
}

impl core::fmt::Display for IdentitySlot {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Declared cyclic-solution tag.
///
/// [`Self::Absent`] is the only absence. A zero digest is a real
/// [`Self::Tag`], not an absent one. Either tag is compared as a value.
/// Neither variant solves `cyclic.solution.v1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub enum CyclicSolution {
    /// This identity carries no cyclic-solution tag.
    Absent,
    /// Digest of the declared [`CYCLIC_SOLUTION_CONTRACT`] semantics.
    Tag(SemanticDigest),
}

/// Digest the carrier advertises for the facets of its own identity.
///
/// The consumer does not trust the advertisement. [`bind`] recomputes
/// [`facet_digest`] from the presented slot, payload, coordinates, prior,
/// alignment, and cyclic-solution tag, and compares that value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct SelfDescribedDigest {
    /// Bytes the carrier says are the digest of the presented facets.
    pub advertised: SemanticDigest,
}

/// Presented identity for one [`IdentitySlot`].
///
/// `self_description` is the carrier's own claim about two digests. It is
/// not an expected identity, and it is not retained by the consumer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct BoundIdentity {
    /// Slot the carrier says this value identifies.
    pub slot: IdentitySlot,
    /// Payload digest.
    pub payload: SemanticDigest,
    /// Coordinate digest.
    pub coordinates: SemanticDigest,
    /// Prior digest.
    pub prior: SemanticDigest,
    /// Alignment digest.
    pub alignment: SemanticDigest,
    /// Cyclic-solution tag. Absent and a tag are different identities.
    pub cyclic_solution: CyclicSolution,
    /// Self-description, when the carrier says two digests match.
    pub self_description: Option<SelfDescribedDigest>,
}

/// Expected identity retained by a consumer, independently of [`BoundIdentity`].
///
/// The consumer stores this value itself. It is not read from the presented
/// identity and not implied by [`SelfDescribedDigest`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct ExpectedIdentity {
    /// Payload digest the consumer retained.
    pub payload: SemanticDigest,
    /// Coordinate digest the consumer retained.
    pub coordinates: SemanticDigest,
    /// Prior digest the consumer retained.
    pub prior: SemanticDigest,
    /// Alignment digest the consumer retained.
    pub alignment: SemanticDigest,
    /// Cyclic-solution tag the consumer retained.
    pub cyclic_solution: CyclicSolution,
}

/// The five presented identities, one per [`IdentitySlot`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct BoundIdentities {
    /// Presented target identity. Its slot is [`IdentitySlot::Target`].
    pub target: BoundIdentity,
    /// Presented program identity. Its slot is [`IdentitySlot::Program`].
    pub program: BoundIdentity,
    /// Presented source identity. Its slot is [`IdentitySlot::Source`].
    pub source: BoundIdentity,
    /// Presented inference identity. Its slot is [`IdentitySlot::Inference`].
    pub inference: BoundIdentity,
    /// Presented claim identity. Its slot is [`IdentitySlot::Claim`].
    pub claim: BoundIdentity,
}

/// Expected identities retained where a consumer requires them.
///
/// `None` means that slot is not required. A presented self-description does
/// not fill `None`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct ExpectedIdentities {
    /// Retained target identity, when the consumer requires one.
    pub target: Option<ExpectedIdentity>,
    /// Retained program identity, when the consumer requires one.
    pub program: Option<ExpectedIdentity>,
    /// Retained source identity, when the consumer requires one.
    pub source: Option<ExpectedIdentity>,
    /// Retained inference identity, when the consumer requires one.
    pub inference: Option<ExpectedIdentity>,
    /// Retained claim identity, when the consumer requires one.
    pub claim: Option<ExpectedIdentity>,
}

/// Result of comparing one presented identity with a consumer's requirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub enum BoundMatch {
    /// Payload, coordinates, prior, alignment, and cyclic-solution tag match
    /// the retained expected identity.
    Bound {
        /// Slot that matched.
        slot: IdentitySlot,
    },
    /// The consumer does not require an expected identity for this slot.
    ///
    /// The presented self-description was consistent, and it was not used as
    /// a retained identity.
    NotRequired {
        /// Slot the consumer left unbound.
        slot: IdentitySlot,
    },
}

/// Why a presented identity was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum BoundRefusal {
    /// The presented slot is not the slot the consumer is binding.
    #[error("presented {presented} identity does not match the {expected} slot")]
    SlotMismatch {
        /// Slot written on the presented identity.
        presented: IdentitySlot,
        /// Slot the consumer is binding.
        expected: IdentitySlot,
    },
    /// Payload digest differs from the retained expected identity.
    #[error("bound {slot} payload differs from the retained expected identity")]
    PayloadMismatch {
        /// Slot being bound.
        slot: IdentitySlot,
    },
    /// Coordinate digest differs from the retained expected identity.
    #[error("bound {slot} coordinates differ from the retained expected identity")]
    CoordinatesMismatch {
        /// Slot being bound.
        slot: IdentitySlot,
    },
    /// Prior digest differs from the retained expected identity.
    #[error("bound {slot} prior differs from the retained expected identity")]
    PriorMismatch {
        /// Slot being bound.
        slot: IdentitySlot,
    },
    /// Alignment digest differs from the retained expected identity.
    #[error("bound {slot} alignment differs from the retained expected identity")]
    AlignmentMismatch {
        /// Slot being bound.
        slot: IdentitySlot,
    },
    /// Cyclic-solution tag differs from the retained expected identity.
    ///
    /// The tag was compared. No cyclic system was solved.
    #[error("bound {slot} cyclic-solution tag differs from the retained expected identity")]
    CyclicSolutionMismatch {
        /// Slot being bound.
        slot: IdentitySlot,
    },
    /// The carrier says its digest matches, and the two byte strings differ.
    #[error("bound {slot} digest describes itself as matching while the bytes differ")]
    FalseSelfDescription {
        /// Slot being bound.
        slot: IdentitySlot,
    },
}

/// Digest of the facets of `bound`, recomputed by the consumer.
///
/// It covers the slot, payload, coordinates, prior, alignment, and cyclic-solution
/// tag, each with a fixed width so no two facets can trade bytes. It does not
/// read [`BoundIdentity::self_description`].
#[must_use]
pub fn facet_digest(bound: &BoundIdentity) -> SemanticDigest {
    let mut hasher = blake3::Hasher::new_derive_key("antecedent bound identity facets");
    hasher.update(&[slot_tag(bound.slot)]);
    hasher.update(bound.payload.as_bytes());
    hasher.update(bound.coordinates.as_bytes());
    hasher.update(bound.prior.as_bytes());
    hasher.update(bound.alignment.as_bytes());
    match bound.cyclic_solution {
        CyclicSolution::Absent => {
            hasher.update(&[0]);
        }
        CyclicSolution::Tag(tag) => {
            hasher.update(&[1]);
            hasher.update(tag.as_bytes());
        }
    }
    SemanticDigest::from_bytes(*hasher.finalize().as_bytes())
}

/// Bind `bound` as `required`.
///
/// `expected` is the identity the consumer retained for this slot. `None`
/// means the consumer does not require one: a consistent presentation is
/// [`BoundMatch::NotRequired`] and is not [`BoundMatch::Bound`].
///
/// Slot agreement is checked first. A self-description whose advertised bytes
/// differ from [`facet_digest`] of the presented facets is then refused. A
/// matching advertisement is accepted and do not skip the retained comparison. The first differing
/// facet is reported, in order: payload, coordinates, prior, alignment,
/// cyclic-solution tag.
///
/// # Errors
///
/// The presented slot disagrees with `required`, the self-description is
/// false, or a required facet differs from `expected`.
pub fn bind(
    required: IdentitySlot,
    bound: &BoundIdentity,
    expected: Option<&ExpectedIdentity>,
) -> Result<BoundMatch, BoundRefusal> {
    if slot_tag(bound.slot) != slot_tag(required) {
        return Err(BoundRefusal::SlotMismatch { presented: bound.slot, expected: required });
    }
    if falsely_self_described(bound) {
        return Err(BoundRefusal::FalseSelfDescription { slot: required });
    }
    let Some(expected) = expected else {
        return Ok(BoundMatch::NotRequired { slot: required });
    };
    if !same_digest(bound.payload, expected.payload) {
        return Err(BoundRefusal::PayloadMismatch { slot: required });
    }
    if !same_digest(bound.coordinates, expected.coordinates) {
        return Err(BoundRefusal::CoordinatesMismatch { slot: required });
    }
    if !same_digest(bound.prior, expected.prior) {
        return Err(BoundRefusal::PriorMismatch { slot: required });
    }
    if !same_digest(bound.alignment, expected.alignment) {
        return Err(BoundRefusal::AlignmentMismatch { slot: required });
    }
    if !same_cyclic(bound.cyclic_solution, expected.cyclic_solution) {
        return Err(BoundRefusal::CyclicSolutionMismatch { slot: required });
    }
    Ok(BoundMatch::Bound { slot: required })
}

/// Bind every slot in [`IdentitySlot::ALL`] order.
///
/// Each field of `expected` is that slot's retained identity. The first
/// refusal in that order is returned. A `None` field is [`BoundMatch::NotRequired`]
/// for its slot when the presentation is consistent.
///
/// # Errors
///
/// Any slot's presented identity is refused by [`bind`].
pub fn bind_all(
    bound: &BoundIdentities,
    expected: &ExpectedIdentities,
) -> Result<[BoundMatch; 5], BoundRefusal> {
    let target = bind(IdentitySlot::Target, &bound.target, expected.target.as_ref())?;
    let program = bind(IdentitySlot::Program, &bound.program, expected.program.as_ref())?;
    let source = bind(IdentitySlot::Source, &bound.source, expected.source.as_ref())?;
    let inference = bind(IdentitySlot::Inference, &bound.inference, expected.inference.as_ref())?;
    let claim = bind(IdentitySlot::Claim, &bound.claim, expected.claim.as_ref())?;
    Ok([target, program, source, inference, claim])
}

const fn slot_tag(slot: IdentitySlot) -> u8 {
    match slot {
        IdentitySlot::Target => 0,
        IdentitySlot::Program => 1,
        IdentitySlot::Source => 2,
        IdentitySlot::Inference => 3,
        IdentitySlot::Claim => 4,
    }
}

/// Byte equality, so the comparison can stay a const fn on Rust 1.85.
const fn same_digest(left: SemanticDigest, right: SemanticDigest) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    let mut index = 0;
    while index < 32 {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

const fn same_cyclic(left: CyclicSolution, right: CyclicSolution) -> bool {
    match (left, right) {
        (CyclicSolution::Absent, CyclicSolution::Absent) => true,
        (CyclicSolution::Tag(left), CyclicSolution::Tag(right)) => same_digest(left, right),
        (CyclicSolution::Absent, CyclicSolution::Tag(_))
        | (CyclicSolution::Tag(_), CyclicSolution::Absent) => false,
    }
}

/// `Some` advertises the digest of the presented facets. A different advertised
/// value is a false self-description. Absent means the carrier made no claim.
fn falsely_self_described(bound: &BoundIdentity) -> bool {
    match bound.self_description {
        Some(described) => !same_digest(described.advertised, facet_digest(bound)),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::from_bytes([byte; 32])
    }

    const fn presented(slot: IdentitySlot) -> BoundIdentity {
        BoundIdentity {
            slot,
            payload: digest(1),
            coordinates: digest(2),
            prior: digest(3),
            alignment: digest(4),
            cyclic_solution: CyclicSolution::Absent,
            self_description: None,
        }
    }

    const fn retained() -> ExpectedIdentity {
        ExpectedIdentity {
            payload: digest(1),
            coordinates: digest(2),
            prior: digest(3),
            alignment: digest(4),
            cyclic_solution: CyclicSolution::Absent,
        }
    }

    fn honest(bound: &BoundIdentity) -> SelfDescribedDigest {
        SelfDescribedDigest { advertised: facet_digest(bound) }
    }

    const fn require_all(retained: ExpectedIdentity) -> ExpectedIdentities {
        ExpectedIdentities {
            target: Some(retained),
            program: Some(retained),
            source: Some(retained),
            inference: Some(retained),
            claim: Some(retained),
        }
    }

    const fn presented_all() -> BoundIdentities {
        BoundIdentities {
            target: presented(IdentitySlot::Target),
            program: presented(IdentitySlot::Program),
            source: presented(IdentitySlot::Source),
            inference: presented(IdentitySlot::Inference),
            claim: presented(IdentitySlot::Claim),
        }
    }

    #[test]
    fn matching_bytes_are_accepted() {
        let retained = retained();
        for slot in IdentitySlot::ALL {
            let mut bound = presented(slot);
            bound.self_description = Some(honest(&bound));
            assert_eq!(bind(slot, &bound, Some(&retained)), Ok(BoundMatch::Bound { slot }));
            bound.self_description = None;
            assert_eq!(bind(slot, &bound, Some(&retained)), Ok(BoundMatch::Bound { slot }));
        }
        let mut tagged = presented(IdentitySlot::Source);
        tagged.cyclic_solution = CyclicSolution::Tag(digest(5));
        tagged.self_description = Some(honest(&tagged));
        let mut expected = retained;
        expected.cyclic_solution = CyclicSolution::Tag(digest(5));
        assert_eq!(
            bind(IdentitySlot::Source, &tagged, Some(&expected)),
            Ok(BoundMatch::Bound { slot: IdentitySlot::Source }),
        );
    }

    #[test]
    fn tampered_payload_mismatches_retained_expected() {
        let retained = retained();
        for slot in IdentitySlot::ALL {
            let mut bound = presented(slot);
            bound.payload = digest(9);
            bound.self_description = Some(honest(&bound));
            assert_eq!(
                bind(slot, &bound, Some(&retained)),
                Err(BoundRefusal::PayloadMismatch { slot }),
            );
        }
    }

    #[test]
    fn tampered_coordinates_mismatch_retained_expected() {
        let retained = retained();
        for slot in IdentitySlot::ALL {
            let mut bound = presented(slot);
            bound.coordinates = digest(8);
            bound.self_description = Some(honest(&bound));
            assert_eq!(
                bind(slot, &bound, Some(&retained)),
                Err(BoundRefusal::CoordinatesMismatch { slot }),
            );
        }
    }

    #[test]
    fn tampered_prior_mismatches_retained_expected() {
        let retained = retained();
        for slot in IdentitySlot::ALL {
            let mut bound = presented(slot);
            bound.prior = digest(6);
            bound.self_description = Some(honest(&bound));
            assert_eq!(
                bind(slot, &bound, Some(&retained)),
                Err(BoundRefusal::PriorMismatch { slot }),
            );
        }
    }

    #[test]
    fn tampered_alignment_mismatches_retained_expected() {
        let retained = retained();
        for slot in IdentitySlot::ALL {
            let mut bound = presented(slot);
            bound.alignment = digest(5);
            bound.self_description = Some(honest(&bound));
            assert_eq!(
                bind(slot, &bound, Some(&retained)),
                Err(BoundRefusal::AlignmentMismatch { slot }),
            );
        }
    }

    #[test]
    fn cyclic_solution_tag_mismatch_is_not_a_solved_system() {
        assert_eq!(CYCLIC_SOLUTION_CONTRACT, "cyclic.solution.v1");
        let retained_absent = retained();
        for slot in IdentitySlot::ALL {
            let mut tagged = presented(slot);
            tagged.cyclic_solution = CyclicSolution::Tag(digest(5));
            tagged.self_description = Some(honest(&tagged));
            assert_eq!(
                bind(slot, &tagged, Some(&retained_absent)),
                Err(BoundRefusal::CyclicSolutionMismatch { slot }),
            );
            let mut other = tagged;
            other.cyclic_solution = CyclicSolution::Tag(digest(6));
            other.self_description = Some(honest(&other));
            let mut retained_tag = retained_absent;
            retained_tag.cyclic_solution = CyclicSolution::Tag(digest(5));
            assert_eq!(
                bind(slot, &other, Some(&retained_tag)),
                Err(BoundRefusal::CyclicSolutionMismatch { slot }),
            );
            assert_eq!(bind(slot, &tagged, Some(&retained_tag)), Ok(BoundMatch::Bound { slot }),);
        }
        let mut zero = presented(IdentitySlot::Target);
        zero.cyclic_solution = CyclicSolution::Tag(digest(0));
        let mut retained_zero = retained_absent;
        retained_zero.cyclic_solution = CyclicSolution::Tag(digest(0));
        assert_eq!(
            bind(IdentitySlot::Target, &zero, Some(&retained_absent)),
            Err(BoundRefusal::CyclicSolutionMismatch { slot: IdentitySlot::Target }),
        );
        assert_eq!(
            bind(IdentitySlot::Target, &zero, Some(&retained_zero)),
            Ok(BoundMatch::Bound { slot: IdentitySlot::Target }),
        );
    }

    #[test]
    fn false_self_description_is_refused() {
        let retained = retained();
        for slot in IdentitySlot::ALL {
            let mut bound = presented(slot);
            bound.self_description = Some(SelfDescribedDigest { advertised: digest(1) });
            assert_ne!(bound.self_description.unwrap().advertised, facet_digest(&bound));
            assert_eq!(bound.payload.as_bytes(), retained.payload.as_bytes());
            assert_eq!(
                bind(slot, &bound, Some(&retained)),
                Err(BoundRefusal::FalseSelfDescription { slot }),
            );
            assert_eq!(bind(slot, &bound, None), Err(BoundRefusal::FalseSelfDescription { slot }),);
        }
    }

    #[test]
    fn stale_self_description_is_recomputed_and_refused() {
        let retained = retained();
        for slot in IdentitySlot::ALL {
            // Described honestly, then a facet is changed: the carrier's earlier
            // advertisement no longer matches the recomputed digest.
            let mut bound = presented(slot);
            bound.self_description = Some(honest(&bound));
            bound.alignment = digest(7);
            assert_eq!(
                bind(slot, &bound, Some(&retained)),
                Err(BoundRefusal::FalseSelfDescription { slot }),
            );
        }
    }

    #[test]
    fn facet_digest_depends_on_every_facet_and_ignores_the_description() {
        let base = presented(IdentitySlot::Target);
        let reference = facet_digest(&base);
        let mut described = base;
        described.self_description = Some(honest(&base));
        assert_eq!(facet_digest(&described), reference);
        let variants = [
            BoundIdentity { slot: IdentitySlot::Claim, ..base },
            BoundIdentity { payload: digest(9), ..base },
            BoundIdentity { coordinates: digest(9), ..base },
            BoundIdentity { prior: digest(9), ..base },
            BoundIdentity { alignment: digest(9), ..base },
            BoundIdentity { cyclic_solution: CyclicSolution::Tag(digest(0)), ..base },
        ];
        for variant in variants {
            assert_ne!(facet_digest(&variant), reference);
        }
    }

    #[test]
    fn matching_self_description_is_not_an_independent_binding() {
        let retained = retained();
        let mut bound = presented(IdentitySlot::Claim);
        bound.payload = digest(9);
        let described = honest(&bound);
        bound.self_description = Some(described);
        assert_eq!(described.advertised, facet_digest(&bound));
        assert_eq!(
            bind(IdentitySlot::Claim, &bound, Some(&retained)),
            Err(BoundRefusal::PayloadMismatch { slot: IdentitySlot::Claim }),
        );
    }

    #[test]
    fn self_description_without_retained_expected_is_not_a_binding() {
        for slot in IdentitySlot::ALL {
            let mut bound = presented(slot);
            bound.self_description = Some(honest(&bound));
            assert_eq!(bind(slot, &bound, None), Ok(BoundMatch::NotRequired { slot }));
        }
    }

    #[test]
    fn retained_expectations_bind_each_required_slot() {
        let retained = retained();
        let bound = presented_all();
        let matches = [
            BoundMatch::Bound { slot: IdentitySlot::Target },
            BoundMatch::Bound { slot: IdentitySlot::Program },
            BoundMatch::Bound { slot: IdentitySlot::Source },
            BoundMatch::Bound { slot: IdentitySlot::Inference },
            BoundMatch::Bound { slot: IdentitySlot::Claim },
        ];
        assert_eq!(bind_all(&bound, &require_all(retained)), Ok(matches));

        let partial = ExpectedIdentities {
            target: Some(retained),
            program: None,
            source: None,
            inference: None,
            claim: Some(retained),
        };
        assert_eq!(
            bind_all(&bound, &partial),
            Ok([
                BoundMatch::Bound { slot: IdentitySlot::Target },
                BoundMatch::NotRequired { slot: IdentitySlot::Program },
                BoundMatch::NotRequired { slot: IdentitySlot::Source },
                BoundMatch::NotRequired { slot: IdentitySlot::Inference },
                BoundMatch::Bound { slot: IdentitySlot::Claim },
            ]),
        );
    }

    #[test]
    fn presented_slot_must_match_the_field() {
        let mut bound = presented_all();
        bound.program.slot = IdentitySlot::Claim;
        assert_eq!(
            bind_all(&bound, &require_all(retained())),
            Err(BoundRefusal::SlotMismatch {
                presented: IdentitySlot::Claim,
                expected: IdentitySlot::Program,
            }),
        );
        assert_eq!(
            bind(IdentitySlot::Target, &bound.program, Some(&retained())),
            Err(BoundRefusal::SlotMismatch {
                presented: IdentitySlot::Claim,
                expected: IdentitySlot::Target,
            }),
        );
    }
}
