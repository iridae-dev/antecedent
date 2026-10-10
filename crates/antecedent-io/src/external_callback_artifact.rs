//! Portable attested callback output; closures and provider authority are never serialized.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::IoError;
use crate::external_claim_artifact::{ExternalClaimArtifact, ExternalClaimIdentity};
use serde::{Deserialize, Serialize};

const MAGIC: &[u8] = b"ANTCALLBACK1";
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
impl ExternalCallbackArtifact {
    /// Encode a validated original claim and execution envelope.
    /// # Errors
    /// Invalid original claim, non-attested trust, or excessive bytes.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        self.validate()?;
        let mut bytes = MAGIC.to_vec();
        ciborium::into_writer(self, &mut bytes)
            .map_err(|error| IoError::Convert(error.to_string()))?;
        if bytes.len() > MAX_CALLBACK_ARTIFACT_BYTES {
            return Err(IoError::Convert("callback artifact exceeds byte bound".into()));
        }
        Ok(bytes)
    }
    /// Decode and independently validate the existing scientific claim.
    /// This checks declarations and bytes; it confers no provider numerical verification.
    /// # Errors
    /// Bounds, format, or original external claim refusal.
    pub fn consume(bytes: &[u8], request: &[u8; 32], provider: &[u8; 32]) -> Result<Self, IoError> {
        if bytes.len() > MAX_CALLBACK_ARTIFACT_BYTES || !bytes.starts_with(MAGIC) {
            return Err(IoError::Convert("invalid callback artifact envelope".into()));
        }
        let mut remaining = &bytes[MAGIC.len()..];
        let value: Self = ciborium::from_reader(&mut remaining)
            .map_err(|error| IoError::Convert(error.to_string()))?;
        if !remaining.is_empty()
            || &value.request_digest != request
            || &value.provider_digest != provider
        {
            return Err(IoError::Convert("callback artifact execution identity mismatch".into()));
        }
        value.validate()?;
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
