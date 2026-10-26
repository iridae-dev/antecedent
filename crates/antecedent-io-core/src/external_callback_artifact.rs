//! Portable attested callback output; closures and provider authority are never serialized.
//!
//! New output is an `external_callback_v1` artifact in the shared sectioned
//! container: a CBOR section of execution identities and the original claim
//! artifact as an opaque section. The earlier `ANTCALLBACK1` frame stays
//! readable; nothing writes it any more.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::IoError;
use crate::external_claim_artifact::{ExternalClaimArtifact, ExternalClaimIdentity};
use serde::{Deserialize, Serialize};

/// Frame magic of the pre-container artifact. Read-only.
const LEGACY_MAGIC: &[u8] = b"ANTCALLBACK1";
const ARTIFACT_KIND: &str = crate::artifact_kinds::EXTERNAL_CALLBACK;
const META_SECTION: &str = "external_callback.meta";
const CLAIM_SECTION: &str = "external_callback.claim";
const META_VERSION: u16 = 1;
/// Maximum bytes before decoding an execution envelope.
pub const MAX_CALLBACK_ARTIFACT_BYTES: usize = 6 * 1024 * 1024;
/// An independently checked original external claim plus declared execution identities.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalCallbackArtifact {
    /// Full actual callback input identity, including raw payload and RNG.
    pub request_digest: [u8; 32],
    /// Exact descriptor, including provider version, environment and replay policy.
    pub provider_digest: [u8; 32],
    /// Declared replay policy, not a purity proof.
    pub policy: String,
    /// Original external claim's independent expected identity.
    pub identity: ExternalClaimIdentity,
    /// Original external claim artifact, carrying finite values and external trust.
    pub artifact: Vec<u8>,
}

/// The CBOR section: everything but the opaque claim bytes.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CallbackMeta {
    version: u16,
    request_digest: [u8; 32],
    provider_digest: [u8; 32],
    policy: String,
    identity: ExternalClaimIdentity,
}

impl ExternalCallbackArtifact {
    /// The container's artifact id, a function of the provider.
    #[must_use]
    pub fn artifact_id(&self) -> String {
        format!("external-callback:{}", self.identity.provider_id)
    }

    /// Whether `bytes` are a callback artifact, in the container or the original frame.
    #[must_use]
    pub fn recognizes(bytes: &[u8]) -> bool {
        bytes.starts_with(LEGACY_MAGIC)
            || crate::container_kind(bytes).is_some_and(|kind| kind == ARTIFACT_KIND)
    }

    /// Encode a validated original claim and execution envelope.
    /// # Errors
    /// Invalid original claim, non-attested trust, or excessive bytes.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        self.validate()?;
        let artifact_id = self.artifact_id();
        crate::artifact_codec::encode_sectioned_io(
            &crate::ArtifactContainer {
                kind: ARTIFACT_KIND,
                note: "external_callback",
                max_bytes: MAX_CALLBACK_ARTIFACT_BYTES,
                artifact_id: &artifact_id,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            META_SECTION,
            &CallbackMeta {
                version: META_VERSION,
                request_digest: self.request_digest,
                provider_digest: self.provider_digest,
                policy: self.policy.clone(),
                identity: self.identity.clone(),
            },
            &[crate::OutSection {
                id: CLAIM_SECTION,
                content_type: crate::CONTENT_BYTES,
                bytes: &self.artifact,
            }],
            "unsupported external callback layout",
        )
    }

    /// Decode and independently validate the existing scientific claim.
    /// This checks declarations and bytes; it confers no provider numerical verification.
    /// # Errors
    /// Bounds, format, or original external claim refusal.
    pub fn consume(bytes: &[u8], request: &[u8; 32], provider: &[u8; 32]) -> Result<Self, IoError> {
        let value = Self::inspect(bytes)?;
        if &value.request_digest != request || &value.provider_digest != provider {
            return Err(IoError::Convert("callback artifact execution identity mismatch".into()));
        }
        Ok(value)
    }

    /// Decode and validate without matching execution identities: the claim and
    /// its declared identity must agree, but nothing says which request it answers.
    /// # Errors
    /// Bounds, format, or original external claim refusal.
    pub fn inspect(bytes: &[u8]) -> Result<Self, IoError> {
        if bytes.len() > MAX_CALLBACK_ARTIFACT_BYTES {
            return Err(IoError::Convert("invalid callback artifact envelope".into()));
        }
        let value = if bytes.starts_with(LEGACY_MAGIC) {
            Self::decode_legacy(bytes)?
        } else {
            Self::decode_container(bytes)?
        };
        value.validate()?;
        Ok(value)
    }

    fn decode_container(bytes: &[u8]) -> Result<Self, IoError> {
        // The version is read from the untyped value and gated before the typed
        // shape is interpreted, so a newer body is `UnsupportedVersion`, not a
        // shape error.
        let (meta, [artifact]): (CallbackMeta, _) =
            crate::artifact_codec::decode_sectioned_io_versioned(
                ARTIFACT_KIND,
                MAX_CALLBACK_ARTIFACT_BYTES,
                META_SECTION,
                &[CLAIM_SECTION],
                "invalid callback artifact envelope",
                META_VERSION,
                bytes,
            )?;
        Ok(Self {
            request_digest: meta.request_digest,
            provider_digest: meta.provider_digest,
            policy: meta.policy,
            identity: meta.identity,
            artifact,
        })
    }

    /// The `ANTCALLBACK1` magic is the legacy frame's version: it carries no
    /// version word, its fields are closed (`deny_unknown_fields`) and its size
    /// is bounded by the envelope cap checked in [`Self::inspect`].
    fn decode_legacy(bytes: &[u8]) -> Result<Self, IoError> {
        let mut remaining = &bytes[LEGACY_MAGIC.len()..];
        let value: Self = ciborium::from_reader(&mut remaining)
            .map_err(|error| IoError::Convert(error.to_string()))?;
        if !remaining.is_empty() {
            return Err(IoError::Convert("callback artifact execution identity mismatch".into()));
        }
        Ok(value)
    }

    fn validate(&self) -> Result<(), IoError> {
        if !matches!(
            self.policy.as_str(),
            "deterministic" | "seeded" | "stateful" | "side_effecting"
        ) || self.identity.trust
            != crate::external_claim_artifact::ExternalClaimTrust::ExternallyAttested
        {
            return Err(IoError::Convert("callback output has unsupported replay or trust".into()));
        }
        ExternalClaimArtifact::from_bytes(&self.artifact, &self.identity)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrelated_bytes_are_not_recognized_or_decoded() {
        assert!(!ExternalCallbackArtifact::recognizes(b"not an artifact"));
        assert!(ExternalCallbackArtifact::recognizes(b"ANTCALLBACK1\x00"));
        assert!(ExternalCallbackArtifact::inspect(b"not an artifact").is_err());
        assert!(ExternalCallbackArtifact::inspect(b"ANTCALLBACK1\xff").is_err());
    }
}
