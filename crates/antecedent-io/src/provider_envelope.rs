//! Fixed native validation and artifact envelope for external Python providers.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 12] = b"ANTEPROV2\0\0\0";
const MAX_HEADER: usize = 1 << 20;
const MAX_ARTIFACT: usize = 32 << 20;

/// Host trust assigned to a separately installed provider result.
/// Native licensing is deliberately absent from this external envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalProviderTrust {
    /// The provider or caller attests the result; the host has not verified it.
    ExternallyAttested,
    /// The host verified this exact request against independent fixtures.
    VerifiedExtension,
}

impl ExternalProviderTrust {
    fn as_str(self) -> &'static str {
        match self {
            Self::ExternallyAttested => "externally_attested",
            Self::VerifiedExtension => "verified_extension",
        }
    }
}

/// Versioned scientific and provenance claims attached to an external result.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderEnvelopeHeader {
    /// Envelope schema version, currently 1.
    pub version: u16,
    /// Explicitly registered host name for the provider.
    pub provider: String,
    /// Query family declared by the provider contract.
    pub query_family: String,
    /// Host trust boundary; native licensing cannot be encoded here.
    pub trust: ExternalProviderTrust,
    /// Digest of the provider's declarative contract.
    pub spec_digest: String,
    /// Digest of the exact request exercised by this result.
    pub request_digest: String,
    /// Host fixture evidence receipt, present only for verified results.
    pub verification_evidence_digest: Option<String>,
    /// Flattened row-major finite estimate.
    pub estimate: Vec<f64>,
    /// Estimate dimensions; an empty shape denotes a scalar.
    pub shape: Vec<usize>,
    /// Optional flattened finite uncertainty payload.
    pub uncertainty: Option<Vec<f64>>,
    /// Dimensions of the uncertainty payload, scalar or equal to `shape`.
    pub uncertainty_shape: Option<Vec<usize>>,
    /// Explicit semantics such as `point_only` or externally attested interval.
    pub uncertainty_semantics: String,
    /// Provider and caller assumptions retained in reported order.
    pub assumptions: Vec<String>,
    /// Provider reported support, separate from host trust.
    pub provider_support_status: String,
    /// Host and provider provenance. `trust_boundary` must match `trust`.
    pub provenance: BTreeMap<String, String>,
    /// BLAKE3 digest of the opaque external artifact, when present.
    pub external_artifact_digest: Option<String>,
}

/// Native envelope plus opaque provider-specific artifact bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderEnvelope {
    /// Validated versioned claims.
    pub header: ProviderEnvelopeHeader,
    /// Opaque receipt encoded by the provider's declared artifact codec.
    pub external_artifact: Option<Vec<u8>>,
}

fn digest64(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn count(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1_usize, |product, dimension| product.checked_mul(*dimension))
}

fn validate(envelope: &ProviderEnvelope) -> Result<(), String> {
    let h = &envelope.header;
    if h.version != 1 {
        return Err("unsupported provider envelope version".into());
    }
    if !nonempty(&h.provider)
        || !nonempty(&h.query_family)
        || !nonempty(&h.uncertainty_semantics)
        || !nonempty(&h.provider_support_status)
        || h.assumptions.iter().any(|assumption| !nonempty(assumption))
    {
        return Err("provider envelope requires non-empty identity and claims".into());
    }
    if !digest64(&h.spec_digest) || !digest64(&h.request_digest) {
        return Err("provider spec and request digests must be 64 hex digits".into());
    }
    if h.trust == ExternalProviderTrust::VerifiedExtension {
        if !h.verification_evidence_digest.as_deref().is_some_and(digest64)
            || h.provenance.get("verification_evidence_digest")
                != h.verification_evidence_digest.as_ref()
        {
            return Err("verified extension requires matching host evidence digest".into());
        }
    } else if h.verification_evidence_digest.is_some()
        || h.provenance.contains_key("verification_evidence_digest")
    {
        return Err("externally attested result cannot carry verification evidence".into());
    }
    if h.provenance.get("registry_name") != Some(&h.provider)
        || h.provenance.get("trust_boundary").map(String::as_str) != Some(h.trust.as_str())
        || h.provenance.iter().any(|(key, value)| !nonempty(key) || !nonempty(value))
    {
        return Err("provider envelope provenance disagrees with host identity or trust".into());
    }
    if count(&h.shape) != Some(h.estimate.len())
        || h.estimate.iter().any(|value| !value.is_finite())
    {
        return Err("provider estimate shape or values are invalid".into());
    }
    match (&h.uncertainty, &h.uncertainty_shape) {
        (None, None) => {}
        (Some(values), Some(shape))
            if h.uncertainty_semantics != "point_only"
                && (shape.is_empty() || shape == &h.shape)
                && count(shape) == Some(values.len())
                && values.iter().all(|value| value.is_finite()) => {}
        _ => return Err("provider uncertainty shape or semantics are invalid".into()),
    }
    match (&envelope.external_artifact, &h.external_artifact_digest) {
        (None, None) => {}
        (Some(artifact), Some(digest))
            if artifact.len() <= MAX_ARTIFACT
                && digest == &blake3::hash(artifact).to_hex().to_string() => {}
        _ => return Err("provider artifact digest or size is invalid".into()),
    }
    Ok(())
}

/// Validate and seal an external result through the native fixed envelope.
/// The opaque provider artifact is retained byte for byte.
pub fn seal_provider_envelope(mut envelope: ProviderEnvelope) -> Result<Vec<u8>, String> {
    envelope.header.external_artifact_digest = envelope
        .external_artifact
        .as_ref()
        .map(|artifact| blake3::hash(artifact).to_hex().to_string());
    validate(&envelope)?;
    let header = serde_json::to_vec(&envelope.header).map_err(|error| error.to_string())?;
    if header.len() > MAX_HEADER {
        return Err("provider envelope header is too large".into());
    }
    let artifact = envelope.external_artifact.as_deref().unwrap_or(&[]);
    let mut bytes = Vec::with_capacity(MAGIC.len() + 8 + header.len() + artifact.len() + 32);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&(artifact.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(artifact);
    let checksum = blake3::hash(&bytes);
    bytes.extend_from_slice(checksum.as_bytes());
    Ok(bytes)
}

/// Check the native checksum and decode an externally attested envelope.
/// A verified claim requires host-held evidence and must use
/// [`open_verified_provider_envelope`] instead.
pub fn open_provider_envelope(bytes: &[u8]) -> Result<ProviderEnvelope, String> {
    decode_provider_envelope(bytes, None)
}

/// Decode a verified result only when its spec, request, and evidence digests
/// match the host registry's independently held verification receipt.
pub fn open_verified_provider_envelope(
    bytes: &[u8],
    expected_spec_digest: &str,
    expected_request_digest: &str,
    expected_evidence_digest: &str,
) -> Result<ProviderEnvelope, String> {
    decode_provider_envelope(
        bytes,
        Some((expected_spec_digest, expected_request_digest, expected_evidence_digest)),
    )
}

fn decode_provider_envelope(
    bytes: &[u8],
    expected_verification: Option<(&str, &str, &str)>,
) -> Result<ProviderEnvelope, String> {
    let prefix = MAGIC.len() + 8;
    if bytes.len() < prefix + 32 || &bytes[..MAGIC.len()] != MAGIC {
        return Err("provider envelope magic or length is invalid".into());
    }
    let header_len = u32::from_le_bytes(
        bytes[MAGIC.len()..MAGIC.len() + 4]
            .try_into()
            .map_err(|_| "provider header length is invalid")?,
    ) as usize;
    let artifact_len = u32::from_le_bytes(
        bytes[MAGIC.len() + 4..prefix]
            .try_into()
            .map_err(|_| "provider artifact length is invalid")?,
    ) as usize;
    if header_len > MAX_HEADER
        || artifact_len > MAX_ARTIFACT
        || bytes.len() != prefix + header_len + artifact_len + 32
    {
        return Err("provider envelope section lengths are invalid".into());
    }
    let signed = &bytes[..bytes.len() - 32];
    if blake3::hash(signed).as_bytes() != &bytes[bytes.len() - 32..] {
        return Err("provider envelope checksum mismatch".into());
    }
    let header: ProviderEnvelopeHeader =
        serde_json::from_slice(&bytes[prefix..prefix + header_len])
            .map_err(|error| error.to_string())?;
    if header.external_artifact_digest.is_none() && artifact_len != 0 {
        return Err("provider envelope carries an unclaimed artifact section".into());
    }
    match (header.trust, expected_verification) {
        (ExternalProviderTrust::ExternallyAttested, None) => {}
        (ExternalProviderTrust::VerifiedExtension, Some((spec, request, evidence)))
            if header.spec_digest == spec
                && header.request_digest == request
                && header.verification_evidence_digest.as_deref() == Some(evidence) => {}
        _ => return Err("provider envelope trust requires matching host verification".into()),
    }
    let external_artifact = (header.external_artifact_digest.is_some())
        .then(|| bytes[prefix + header_len..prefix + header_len + artifact_len].to_vec());
    let envelope = ProviderEnvelope { header, external_artifact };
    validate(&envelope)?;
    Ok(envelope)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> ProviderEnvelope {
        ProviderEnvelope {
            header: ProviderEnvelopeHeader {
                version: 1,
                provider: "external-econml".into(),
                query_family: "effect".into(),
                trust: ExternalProviderTrust::ExternallyAttested,
                spec_digest: "a".repeat(64),
                request_digest: "b".repeat(64),
                verification_evidence_digest: None,
                estimate: vec![2.0],
                shape: vec![],
                uncertainty: None,
                uncertainty_shape: None,
                uncertainty_semantics: "point_only".into(),
                assumptions: vec!["backdoor adjustment".into()],
                provider_support_status: "caller_asserted".into(),
                provenance: BTreeMap::from([
                    ("registry_name".into(), "external-econml".into()),
                    ("trust_boundary".into(), "externally_attested".into()),
                    ("adapter".into(), "EconMLProviderAdapter".into()),
                ]),
                external_artifact_digest: None,
            },
            external_artifact: Some(b"original-econml-receipt".to_vec()),
        }
    }

    #[test]
    fn econml_receipt_round_trips_without_reencoding() {
        let bytes = seal_provider_envelope(fixture()).unwrap();
        let decoded = open_provider_envelope(&bytes).unwrap();
        assert_eq!(decoded.external_artifact.as_deref(), Some(&b"original-econml-receipt"[..]));
        assert_eq!(decoded.header.estimate, [2.0]);
        assert_eq!(seal_provider_envelope(decoded).unwrap(), bytes);
    }

    #[test]
    fn checksum_shape_trust_and_point_only_refusals() {
        let mut bytes = seal_provider_envelope(fixture()).unwrap();
        bytes[MAGIC.len() + 8] ^= 1;
        assert!(open_provider_envelope(&bytes).unwrap_err().contains("checksum"));
        let mut bad = fixture();
        bad.header.shape = vec![2];
        assert!(seal_provider_envelope(bad).unwrap_err().contains("shape"));
        let mut bad = fixture();
        bad.header.trust = ExternalProviderTrust::VerifiedExtension;
        assert!(seal_provider_envelope(bad).unwrap_err().contains("evidence"));
        let mut bad = fixture();
        bad.header.uncertainty = Some(vec![0.1]);
        bad.header.uncertainty_shape = Some(vec![]);
        assert!(seal_provider_envelope(bad).unwrap_err().contains("uncertainty"));
    }
}
