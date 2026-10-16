//! Fixed native validation and artifact envelope for external Python providers.
//!
//! New results are written as a `provider_result_v1` artifact in the shared
//! sectioned container: a CBOR header section and the provider's opaque
//! artifact bytes, each under the container checksum. The original
//! `ANTEPROV2` frame stays readable, so results sealed by an earlier release
//! still open; nothing writes it any more.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::convert::{from_cbor, to_cbor};

/// Frame magic of the pre-container envelope. Read-only.
const LEGACY_MAGIC: &[u8; 12] = b"ANTEPROV2\0\0\0";
const ARTIFACT_KIND: &str = crate::artifact_kinds::PROVIDER_RESULT;
const HEADER_SECTION: &str = "provider_result.header";
const ARTIFACT_SECTION: &str = "provider_result.artifact";
const MAX_HEADER: usize = 1 << 20;
const MAX_ARTIFACT: usize = 32 << 20;
/// Largest sealed result, container framing included.
const MAX_ENVELOPE_BYTES: usize = MAX_HEADER + MAX_ARTIFACT + (1 << 20);

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

impl ProviderEnvelopeHeader {
    /// The container's artifact id: a function of the registered provider, so
    /// a replayed result seals to the same bytes.
    #[must_use]
    pub fn artifact_id(&self) -> String {
        format!("provider-result:{}", self.provider)
    }
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
                && digest == &antecedent_core::blake3_hex(artifact) => {}
        _ => return Err("provider artifact digest or size is invalid".into()),
    }
    Ok(())
}

/// Validate and seal an external result through the shared container.
/// The opaque provider artifact is retained byte for byte.
pub fn seal_provider_envelope(mut envelope: ProviderEnvelope) -> Result<Vec<u8>, String> {
    envelope.header.external_artifact_digest =
        envelope.external_artifact.as_ref().map(|artifact| antecedent_core::blake3_hex(artifact));
    validate(&envelope)?;
    let header = to_cbor(&envelope.header).map_err(|error| error.to_string())?;
    if header.len() > MAX_HEADER {
        return Err("provider envelope header is too large".into());
    }
    let artifact = envelope.external_artifact.as_deref().unwrap_or(&[]);
    let artifact_id = envelope.header.artifact_id();
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "provider_result",
            max_bytes: MAX_ENVELOPE_BYTES,
            artifact_id: &artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[
            crate::OutSection {
                id: HEADER_SECTION,
                content_type: crate::CONTENT_CBOR,
                bytes: &header,
            },
            crate::OutSection {
                id: ARTIFACT_SECTION,
                content_type: crate::CONTENT_BYTES,
                bytes: artifact,
            },
        ],
    )
    .map_err(|error| error.into_io("unsupported provider result layout").to_string())
}

/// Whether `bytes` are a provider result, in the container or the original frame.
#[must_use]
pub fn is_provider_envelope(bytes: &[u8]) -> bool {
    bytes.starts_with(LEGACY_MAGIC)
        || crate::container_kind(bytes).is_some_and(|kind| kind == ARTIFACT_KIND)
}

/// Check the checksum and decode an externally attested envelope.
/// A verified claim requires host-held evidence and must use
/// [`open_verified_provider_envelope`] instead.
pub fn open_provider_envelope(bytes: &[u8]) -> Result<ProviderEnvelope, String> {
    decode_provider_envelope(bytes, Expectation::Attested)
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
        Expectation::Verified(
            expected_spec_digest,
            expected_request_digest,
            expected_evidence_digest,
        ),
    )
}

/// Decode and structurally validate an envelope without crediting its trust.
///
/// For generic readers: the header's `trust` is what the producer *declared*.
/// A `verified_extension` claim is believed only through
/// [`open_verified_provider_envelope`] against host-held evidence.
pub fn inspect_provider_envelope(bytes: &[u8]) -> Result<ProviderEnvelope, String> {
    decode_provider_envelope(bytes, Expectation::Declared)
}

#[derive(Clone, Copy)]
enum Expectation<'a> {
    /// Only an externally attested envelope opens.
    Attested,
    /// A verified envelope opens only under these spec, request and evidence digests.
    Verified(&'a str, &'a str, &'a str),
    /// Structural decoding only; trust stays a declaration.
    Declared,
}

fn decode_provider_envelope(
    bytes: &[u8],
    expectation: Expectation<'_>,
) -> Result<ProviderEnvelope, String> {
    let (header, external_artifact) = if bytes.starts_with(LEGACY_MAGIC) {
        decode_legacy_frame(bytes)?
    } else {
        decode_container_frame(bytes)?
    };
    match (header.trust, expectation) {
        (_, Expectation::Declared)
        | (ExternalProviderTrust::ExternallyAttested, Expectation::Attested) => {}
        (
            ExternalProviderTrust::VerifiedExtension,
            Expectation::Verified(spec, request, evidence),
        ) if header.spec_digest == spec
            && header.request_digest == request
            && header.verification_evidence_digest.as_deref() == Some(evidence) => {}
        _ => return Err("provider envelope trust requires matching host verification".into()),
    }
    let envelope = ProviderEnvelope { header, external_artifact };
    validate(&envelope)?;
    Ok(envelope)
}

fn decode_container_frame(
    bytes: &[u8],
) -> Result<(ProviderEnvelopeHeader, Option<Vec<u8>>), String> {
    let layout = "unsupported provider result layout";
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_ENVELOPE_BYTES,
        &[HEADER_SECTION, ARTIFACT_SECTION],
        bytes,
    )
    .map_err(|error| error.into_io(layout).to_string())?;
    let Ok([header, artifact]) = <[Vec<u8>; 2]>::try_from(parts) else {
        return Err(layout.into());
    };
    if header.len() > MAX_HEADER || artifact.len() > MAX_ARTIFACT {
        return Err("provider envelope section lengths are invalid".into());
    }
    let header: ProviderEnvelopeHeader = from_cbor(&header).map_err(|error| error.to_string())?;
    if header.external_artifact_digest.is_none() && !artifact.is_empty() {
        return Err("provider envelope carries an unclaimed artifact section".into());
    }
    let artifact = header.external_artifact_digest.is_some().then_some(artifact);
    Ok((header, artifact))
}

/// The `ANTEPROV2` frame: magic, header and artifact lengths, JSON header,
/// artifact, trailing BLAKE3 of everything before it.
fn decode_legacy_frame(bytes: &[u8]) -> Result<(ProviderEnvelopeHeader, Option<Vec<u8>>), String> {
    let prefix = LEGACY_MAGIC.len() + 8;
    if bytes.len() < prefix + 32 {
        return Err("provider envelope magic or length is invalid".into());
    }
    let header_len = u32::from_le_bytes(
        bytes[LEGACY_MAGIC.len()..LEGACY_MAGIC.len() + 4]
            .try_into()
            .map_err(|_| "provider header length is invalid")?,
    ) as usize;
    let artifact_len = u32::from_le_bytes(
        bytes[LEGACY_MAGIC.len() + 4..prefix]
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
    let artifact = header
        .external_artifact_digest
        .is_some()
        .then(|| bytes[prefix + header_len..prefix + header_len + artifact_len].to_vec());
    Ok((header, artifact))
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

    /// The pre-container frame, kept only to prove old bytes still open.
    fn legacy_frame(envelope: &ProviderEnvelope) -> Vec<u8> {
        let mut header = envelope.header.clone();
        header.external_artifact_digest =
            envelope.external_artifact.as_ref().map(|a| antecedent_core::blake3_hex(a));
        let header = serde_json::to_vec(&header).unwrap();
        let artifact = envelope.external_artifact.as_deref().unwrap_or(&[]);
        let mut bytes = LEGACY_MAGIC.to_vec();
        bytes.extend_from_slice(&u32::try_from(header.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&u32::try_from(artifact.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(artifact);
        let checksum = blake3::hash(&bytes);
        bytes.extend_from_slice(checksum.as_bytes());
        bytes
    }

    #[test]
    fn econml_receipt_round_trips_without_reencoding() {
        let bytes = seal_provider_envelope(fixture()).unwrap();
        assert!(is_provider_envelope(&bytes));
        assert!(bytes.starts_with(b"CAUSAL\0\0"), "new results use the shared container");
        let decoded = open_provider_envelope(&bytes).unwrap();
        assert_eq!(decoded.external_artifact.as_deref(), Some(&b"original-econml-receipt"[..]));
        assert_eq!(decoded.header.estimate, [2.0]);
        assert_eq!(seal_provider_envelope(decoded).unwrap(), bytes);
    }

    #[test]
    fn a_container_checksum_shape_trust_and_point_only_refusals() {
        let mut bytes = seal_provider_envelope(fixture()).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        assert!(open_provider_envelope(&bytes).is_err());
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

    #[test]
    fn an_original_frame_still_opens_and_reseals_into_the_container() {
        let original = fixture();
        let old = legacy_frame(&original);
        assert!(is_provider_envelope(&old));
        let decoded = open_provider_envelope(&old).unwrap();
        assert_eq!(decoded.external_artifact, original.external_artifact);
        assert_eq!(decoded.header.estimate, [2.0]);
        let mut tampered = old.clone();
        tampered[LEGACY_MAGIC.len() + 8] ^= 1;
        assert!(open_provider_envelope(&tampered).unwrap_err().contains("checksum"));
        let sealed = seal_provider_envelope(decoded).unwrap();
        assert_eq!(open_provider_envelope(&sealed).unwrap(), original_with_digest(original));
    }

    fn original_with_digest(mut envelope: ProviderEnvelope) -> ProviderEnvelope {
        envelope.header.external_artifact_digest =
            envelope.external_artifact.as_ref().map(|a| antecedent_core::blake3_hex(a));
        envelope
    }

    #[test]
    fn inspection_decodes_a_verified_claim_without_crediting_it() {
        let evidence = "c".repeat(64);
        let mut verified = fixture();
        verified.header.trust = ExternalProviderTrust::VerifiedExtension;
        verified.header.verification_evidence_digest = Some(evidence.clone());
        verified.header.provenance.insert("trust_boundary".into(), "verified_extension".into());
        verified.header.provenance.insert("verification_evidence_digest".into(), evidence.clone());
        let bytes = seal_provider_envelope(verified).unwrap();
        assert!(open_provider_envelope(&bytes).is_err());
        let inspected = inspect_provider_envelope(&bytes).unwrap();
        assert_eq!(inspected.header.trust, ExternalProviderTrust::VerifiedExtension);
        let opened =
            open_verified_provider_envelope(&bytes, &"a".repeat(64), &"b".repeat(64), &evidence)
                .unwrap();
        assert_eq!(opened, inspected);
    }
}
