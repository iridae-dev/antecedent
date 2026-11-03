//! The evidence digest shared by the transport point artifacts.
//!
//! The premises and data digests of these artifacts bind the graph, query, proof and the
//! catalog's and laws' snapshot identities. Neither binds a law's probabilities or axes, the
//! request, the producer's limits, the program or the reported result, so an artifact whose
//! laws and result were replaced together still carries both. The evidence digest closes that:
//! it is the canonical hash of the whole stored artifact with only its own slot emptied, so a
//! field added later is bound without anyone remembering to list it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::IdentityDomain;
use serde::Serialize;

use crate::IoError;

/// Hex evidence digest of `wire` under `tag`, with `unseal` emptying the stored evidence slot
/// (and nothing else) on a private copy.
///
/// # Errors
/// Canonical encoding failure.
pub(crate) fn evidence_digest<T: Serialize + Clone>(
    tag: &'static str,
    wire: &T,
    unseal: impl FnOnce(&mut T),
) -> Result<String, IoError> {
    let mut unsealed = wire.clone();
    unseal(&mut unsealed);
    crate::digest_hex(IdentityDomain::TransportCertificate, &(tag, unsealed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Serialize)]
    struct Wire {
        limit: u64,
        evidence_digest: String,
    }

    fn digest(wire: &Wire) -> String {
        evidence_digest("test_evidence_v1", wire, |w| w.evidence_digest = String::new()).unwrap()
    }

    #[test]
    fn the_digest_ignores_only_its_own_slot() {
        let wire = Wire { limit: 8, evidence_digest: String::new() };
        let sealed = Wire { evidence_digest: digest(&wire), ..wire.clone() };
        assert_eq!(digest(&sealed), sealed.evidence_digest);
        assert_ne!(digest(&Wire { limit: 7, ..sealed.clone() }), sealed.evidence_digest);
    }
}
