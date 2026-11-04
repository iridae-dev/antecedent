//! Digest and identity helpers for hosts: hex spellings, content digests and the contract
//! identity layers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// The seven identity layers of one compiled contract.
pub use antecedent_core::ContractIdentities;
/// A 32-byte domain-separated digest.
pub use antecedent_core::SemanticDigest;

/// Lowercase hex spelling of `bytes`, two characters per byte.
#[must_use]
pub fn to_hex(bytes: &[u8]) -> String {
    antecedent_core::identity::to_hex(bytes)
}

/// Unframed BLAKE3 digest of `bytes` as 64 lowercase hex characters, for content addressing.
#[must_use]
pub fn blake3_hex(bytes: &[u8]) -> String {
    antecedent_core::blake3_hex(bytes)
}

/// Parse 64 hexadecimal characters (either case) into a digest; `None` for any other text.
#[must_use]
pub fn parse_digest(text: &str) -> Option<SemanticDigest> {
    antecedent_core::digest_from_hex(text).map(SemanticDigest::from_bytes)
}

/// Digest of one opaque byte string under the domain `tag`.
#[must_use]
pub fn digest_bytes(tag: &str, bytes: &[u8]) -> SemanticDigest {
    SemanticDigest::from_bytes(antecedent_core::CanonicalWriter::digest_bytes(tag, bytes))
}

/// Digest of an ordered list of labels under the domain `tag`.
#[must_use]
pub fn digest_labels<S: AsRef<str>>(tag: &str, labels: &[S]) -> SemanticDigest {
    SemanticDigest::from_bytes(antecedent_core::CanonicalWriter::new(tag).texts(labels).finish())
}

/// Digest of a wire value as a portable claim envelope, over its canonical CBOR encoding.
///
/// # Errors
/// The value cannot be encoded.
pub fn digest_claim_wire<T: serde::Serialize>(
    value: &T,
) -> Result<SemanticDigest, antecedent_io_core::IoError> {
    antecedent_io_core::identity::digest_wire(antecedent_core::IdentityDomain::Claim, value)
}
