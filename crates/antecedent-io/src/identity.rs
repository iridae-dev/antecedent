//! Canonical encoding and domain-separated digests for contract identities.
//!
//! The digest rules live in `antecedent-io-core`; the claim-id payload, which
//! embeds the contract claim section, is defined next to its consumer and
//! joined here so the public path is unchanged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io_core::identity::*;

pub use crate::claim_identity::{ClaimIdentityWire, claim_digest};
