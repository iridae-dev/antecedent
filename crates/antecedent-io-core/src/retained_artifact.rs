//! Admit an artifact only as the artifact its consumer retained.
//!
//! A consumer that will load an artifact later retains its identity apart from the bytes:
//! [`RetainedArtifactIdentity::of_bytes`] over the bytes the producer wrote, kept in the
//! consumer's own store, a registry, or a message from the producer. It never reads that
//! identity from the bytes it is about to admit, so an artifact cannot supply the identity it
//! is checked against.
//!
//! Every artifact loader offers an `*_expecting` form that takes a
//! [`RetainedArtifactIdentity`]. The form recomputes [`artifact_identity`] from the bytes it
//! was handed and binds it to the retained identity through
//! [`crate::bound_identity::bind`] ([`bind_retained_artifact`]) before the bytes are read.
//! Bytes that are not the retained artifact are refused with `external_binding_mismatch`; the
//! retained artifact then goes through the loader's own checks unchanged.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::SemanticDigest;

use crate::bound_identity::{
    BoundIdentity, BoundMatch, BoundRefusal, ExpectedIdentity, IdentitySlot, bind,
};
use crate::error::IoError;

/// BLAKE3 derive-key context of [`artifact_identity`].
pub const ARTIFACT_IDENTITY_DOMAIN: &str = "antecedent retained artifact bytes v1";

/// The identity of an artifact a consumer retained, apart from the bytes it later admits.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[must_use]
pub struct RetainedArtifactIdentity {
    /// The artifact digest ([`artifact_identity`]) the consumer expects.
    pub digest: SemanticDigest,
}

impl RetainedArtifactIdentity {
    /// The artifact digest the consumer retained.
    pub const fn new(digest: SemanticDigest) -> Self {
        Self { digest }
    }

    /// The identity to retain for `bytes`, computed when the producer writes them (or when the
    /// consumer first stores them), never from bytes the consumer is about to admit.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Self::new(artifact_identity(bytes))
    }

    /// Parse the 64-character hex form ([`Self::to_hex`]).
    ///
    /// # Errors
    ///
    /// `invalid_argument` unless `text` is 64 hex characters.
    pub fn from_hex(text: &str) -> Result<Self, IoError> {
        antecedent_core::digest_from_hex(text)
            .map(|bytes| Self::new(SemanticDigest::from_bytes(bytes)))
            .ok_or_else(|| IoError::Refused {
                code: antecedent_core::reason_code!("invalid_argument"),
                message: "a retained artifact identity is 64 hex characters".into(),
            })
    }

    /// Lowercase hex of the digest.
    #[must_use]
    pub fn to_hex(&self) -> String {
        self.digest.to_hex()
    }
}

/// Digest of an artifact's bytes, as the consumer recomputes it from the bytes it was handed.
///
/// BLAKE3 derive-key ([`ARTIFACT_IDENTITY_DOMAIN`]) over the bytes exactly as handed over:
/// one changed, added or removed byte is another artifact.
#[must_use]
pub fn artifact_identity(bytes: &[u8]) -> SemanticDigest {
    SemanticDigest::from_bytes(antecedent_core::derive_key_digest(
        ARTIFACT_IDENTITY_DOMAIN,
        &[bytes],
    ))
}

/// Bind a presented artifact digest to the identity the consumer retained, through
/// [`bind`].
///
/// # Errors
///
/// The typed refusal of [`bind`] when `presented` is not the retained digest.
pub fn bind_artifact_identity(
    presented: SemanticDigest,
    retained: &RetainedArtifactIdentity,
) -> Result<(), BoundRefusal> {
    let expected = ExpectedIdentity::artifact(retained.digest);
    match bind(IdentitySlot::Claim, &BoundIdentity::artifact(presented), Some(&expected))? {
        BoundMatch::Bound { .. } => Ok(()),
        // An expectation was passed, so `bind` cannot leave the slot unbound; refusing keeps
        // the only accepting outcome `Bound`.
        BoundMatch::NotRequired { slot } => Err(BoundRefusal::PayloadMismatch { slot }),
    }
}

/// Admit `bytes` only as the artifact the consumer retained.
///
/// Recomputes [`artifact_identity`] from `bytes` and binds it to `retained`
/// ([`bind_artifact_identity`]). Every `*_expecting` loader calls this before it reads the
/// bytes.
///
/// # Errors
///
/// `external_binding_mismatch` when `bytes` are not the retained artifact.
pub fn bind_retained_artifact(
    bytes: &[u8],
    retained: &RetainedArtifactIdentity,
) -> Result<(), IoError> {
    bind_retained_digest(artifact_identity(bytes), retained)
}

pub(crate) fn bind_retained_digest(
    digest: SemanticDigest,
    retained: &RetainedArtifactIdentity,
) -> Result<(), IoError> {
    bind_artifact_identity(digest, retained).map_err(|refusal| IoError::Refused {
        code: antecedent_core::reason_code!("external_binding_mismatch"),
        message: format!(
            "retained artifact identity: the bytes are not the artifact the consumer \
                 retained ({refusal})"
        ),
    })
}

/// Run `load` on `bytes` only after they bind to the retained identity
/// ([`bind_retained_artifact`]).
///
/// The one body every `*_expecting` loader shares.
///
/// # Errors
///
/// `external_binding_mismatch` (converted into the loader's error) when `bytes` are not the
/// retained artifact; otherwise whatever `load` returns.
pub fn load_expecting<T, E: From<IoError>>(
    bytes: &[u8],
    retained: &RetainedArtifactIdentity,
    load: impl FnOnce(&[u8]) -> Result<T, E>,
) -> Result<T, E> {
    bind_retained_artifact(bytes, retained).map_err(E::from)?;
    load(bytes)
}

/// Text every retained-artifact refusal carries, so a refusal converted into another error type
/// can still be recognised as the binding refusal.
pub const RETAINED_ARTIFACT_REFUSAL: &str = "retained artifact identity";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_retained_bytes_are_admitted() {
        let bytes = b"artifact".to_vec();
        let retained = RetainedArtifactIdentity::of_bytes(&bytes);
        assert_eq!(bind_retained_artifact(&bytes, &retained), Ok(()));
        assert_eq!(RetainedArtifactIdentity::from_hex(&retained.to_hex()), Ok(retained));
        let loaded: Result<usize, IoError> = load_expecting(&bytes, &retained, |b| Ok(b.len()));
        assert_eq!(loaded, Ok(8));
    }

    #[test]
    fn other_bytes_are_refused_before_they_are_read() {
        let retained = RetainedArtifactIdentity::of_bytes(b"artifact");
        for other in [&b"artifacT"[..], b"artifact ", b"artifac", b""] {
            let refusal = bind_retained_artifact(other, &retained).unwrap_err();
            assert!(matches!(
                &refusal,
                IoError::Refused { code: "external_binding_mismatch", message }
                    if message.contains(RETAINED_ARTIFACT_REFUSAL)
            ));
            let loaded: Result<(), IoError> =
                load_expecting(other, &retained, |_| panic!("read before binding"));
            assert_eq!(loaded, Err(refusal));
        }
        assert!(RetainedArtifactIdentity::from_hex("00").is_err());
    }
}
