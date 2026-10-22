//! Digest-backed identities used by the evidence ledger.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use antecedent_core::{CanonicalWriter, SemanticDigest};

/// Version word written into every ledger identity. Bump only when a
/// canonical encoding in this module tree changes.
pub const LEDGER_IDENTITY_VERSION: u64 = 1;

macro_rules! ledger_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        pub struct $name(SemanticDigest);

        impl $name {
            /// Wrap a digest.
            #[must_use]
            pub const fn from_digest(digest: SemanticDigest) -> Self {
                Self(digest)
            }

            /// The underlying digest.
            #[must_use]
            pub const fn digest(&self) -> SemanticDigest {
                self.0
            }

            /// Lowercase hex spelling (64 characters).
            #[must_use]
            pub fn to_hex(&self) -> String {
                self.0.to_hex()
            }

            /// Parse the hex spelling written by [`Self::to_hex`].
            #[must_use]
            pub fn from_hex(text: &str) -> Option<Self> {
                parse_digest_hex(text).map(Self)
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0.to_hex())
            }
        }
    };
}

ledger_id! {
    /// Identity of one observation (row, unit or draw), supplied by the
    /// producer so that two evidence sources that used the same observation
    /// carry the same id and their overlap is computable.
    ObservationId
}

ledger_id! {
    /// Content identity of one evidence source: its kind, label, observation
    /// basis, lineage and transport identity.
    SourceId
}

ledger_id! {
    /// Identity of one derived result (a pooled, borrowed, transported,
    /// reused or repeatedly used claim).
    ResultId
}

ledger_id! {
    /// Identity of one applied ledger operation. Chains the previous
    /// operation, so the head identity binds the whole history.
    OperationId
}

impl ObservationId {
    /// Identity of an observation from a stable key within a named domain,
    /// for example a data snapshot digest and a row index.
    #[must_use]
    pub fn from_key(domain: &str, key: &[u8]) -> Self {
        let mut writer = CanonicalWriter::new("antecedent.evidence_ledger.observation");
        writer.word(LEDGER_IDENTITY_VERSION).text(domain).bytes(key);
        Self(SemanticDigest::from_bytes(writer.finish()))
    }
}

/// Parse 64 lowercase or uppercase hex characters into a digest.
#[must_use]
pub fn parse_digest_hex(text: &str) -> Option<SemanticDigest> {
    antecedent_core::digest_from_hex(text).map(SemanticDigest::from_bytes)
}

pub(crate) fn finish(writer: &CanonicalWriter) -> SemanticDigest {
    SemanticDigest::from_bytes(writer.finish())
}

pub(crate) fn digest_word(writer: &mut CanonicalWriter, digest: SemanticDigest) {
    writer.bytes(digest.as_bytes());
}
