//! The claim-id payload and its digest.
//!
//! Re-exported from [`crate::identity`]. It carries the contract claim section,
//! which the estimator-bound consumer owns, so it sits above
//! `antecedent-io-core`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::IoError;
use crate::identity::digest_wire;
use antecedent_core::{IDENTITY_FORMAT, IdentityDomain, SemanticDigest};
use serde::{Deserialize, Serialize};

/// Claim-id payload.
///
/// `seal` binds every contract identity, the four reasoning slots, and the
/// section's audit fields ([`crate::contract_seal`]); `claim` is the claim
/// section with a zeroed `claim_id`; `result` is [`crate::result_digest`] of
/// the executed result body. Nothing a verified consume reports is outside
/// these three.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClaimIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Contract seal.
    pub seal: [u8; 32],
    /// Claim fields with `claim_id` zeroed.
    pub claim: crate::ClaimSectionWire,
    /// Digest of the executed result body.
    pub result: [u8; 32],
}

impl ClaimIdentityWire {
    /// Bind a claim identity. Format is always [`IDENTITY_FORMAT`].
    #[must_use]
    pub fn new(seal: [u8; 32], claim: &crate::ClaimSectionWire, result: [u8; 32]) -> Self {
        Self {
            format: IDENTITY_FORMAT,
            seal,
            claim: crate::ClaimSectionWire { claim_id: [0; 32], ..claim.clone() },
            result,
        }
    }
}

/// Digest a claim envelope identity.
///
/// # Errors
///
/// CBOR encode failure.
pub fn claim_digest(wire: &ClaimIdentityWire) -> Result<SemanticDigest, IoError> {
    digest_wire(IdentityDomain::Claim, wire)
}
