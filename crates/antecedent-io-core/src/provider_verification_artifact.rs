//! Portable receipt for providers verified at exact requests.
//!
//! One wire for every provider that was checked against independent probes,
//! whether it supplied a typed scientific object or ran as an executable
//! Python plugin. The receipt keeps request identities, contract digests and
//! passed probes, never the contracts themselves. A consumer retains the
//! receipt digest independently and supplies it on load; a changed field,
//! including a resealed one, is refused. Holding a receipt grants no native
//! licence and no calibration.

// Refusals are the cold path of a once-per-verification call, so the large `Err`
// variant is not worth boxing.
#![allow(clippy::result_large_err)]

use antecedent_contract::{
    ExecutionContract, ExternalVerificationError, ProviderObjectIdentity,
    ProviderVerificationReceipt, VerificationProbe, VerificationSubject, VerifiedContractKind,
    VerifiedRequest, probe_from_name, probe_name, verify_provider_requests,
};
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;
use crate::error::IoError;
use crate::external_binding_wire::ExternalBindingRefusalWire;
use crate::external_claim_artifact::VerificationProbeWire;

const ARTIFACT_KIND: &str = crate::artifact_kinds::PROVIDER_VERIFICATION_RECEIPT;
const META_SECTION: &str = "provider_verification.meta";
/// Maximum artifact bytes accepted before decode allocation.
pub const MAX_PROVIDER_VERIFICATION_ARTIFACT_BYTES: usize = 4 * 1024 * 1024;
/// Maximum verified requests in one receipt.
pub const MAX_VERIFIED_REQUESTS: usize = 4_096;
/// Maximum probes recorded for one request.
pub const MAX_PROBES_PER_REQUEST: usize = 64;

/// Exact provider object and request, on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderIdentityWire {
    /// Provider service or registry name.
    pub provider_id: String,
    /// Object identity within the provider.
    pub object_id: String,
    /// Provider implementation version, or the digest of its declarative spec.
    pub version_id: String,
    /// Input data or model snapshot.
    pub snapshot_id: String,
    /// Exact request fingerprint.
    pub request_id: String,
}

impl From<&ProviderObjectIdentity> for ProviderIdentityWire {
    fn from(identity: &ProviderObjectIdentity) -> Self {
        Self {
            provider_id: identity.provider_id.clone(),
            object_id: identity.object_id.clone(),
            version_id: identity.version_id.clone(),
            snapshot_id: identity.snapshot_id.clone(),
            request_id: identity.request_id.clone(),
        }
    }
}

impl From<ProviderIdentityWire> for ProviderObjectIdentity {
    fn from(wire: ProviderIdentityWire) -> Self {
        Self {
            provider_id: wire.provider_id,
            object_id: wire.object_id,
            version_id: wire.version_id,
            snapshot_id: wire.snapshot_id,
            request_id: wire.request_id,
        }
    }
}

/// One executable-provider request to verify, with its independent probes.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionRequestWire {
    /// Exact identity of the run being verified.
    pub identity: ProviderIdentityWire,
    /// Whether repeated runs must agree exactly.
    pub deterministic: bool,
    /// Whether the run reports an uncertainty the reference constrains.
    pub reports_uncertainty: bool,
    /// BLAKE3 digest of the emitted opaque artifact, when one was emitted.
    pub artifact_blake3: Option<String>,
    /// Independently computed probes.
    pub probes: Vec<VerificationProbeWire>,
}

/// A batch of executable-provider requests verified under one named origin.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionVerificationWire {
    /// Who held the independent reference.
    pub evidence_origin: String,
    /// Requests, in order.
    pub requests: Vec<ExecutionRequestWire>,
}

/// One verified request in a stored receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerifiedRequestWire {
    /// Exact identity that was verified.
    pub identity: ProviderIdentityWire,
    /// `VerifiedContractKind` name.
    pub kind: String,
    /// Canonical digest of the contract that was checked.
    pub contract_digest: String,
    /// Probes that passed.
    pub probes: Vec<VerificationProbeWire>,
}

/// Stored receipt: requests plus the digest a consumer retains independently.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderVerificationReceiptWire {
    /// Wire format version.
    pub version: u16,
    /// Who held the independent reference, when named.
    pub evidence_origin: Option<String>,
    /// Verified requests, in order.
    pub requests: Vec<VerifiedRequestWire>,
    /// [`ProviderVerificationReceipt::digest`] of the content above.
    pub digest: String,
}

fn probe_wire(probe: &VerificationProbe) -> VerificationProbeWire {
    VerificationProbeWire {
        kind: probe_name(probe.kind).to_owned(),
        observed: probe.observed,
        expected: probe.expected,
        tolerance: probe.tolerance,
    }
}

fn convert(message: impl Into<String>) -> IoError {
    IoError::Convert(message.into())
}

fn probe_from_wire(wire: &VerificationProbeWire) -> Result<VerificationProbe, IoError> {
    Ok(VerificationProbe {
        kind: probe_from_name(&wire.kind)
            .ok_or_else(|| convert(format!("unknown verification probe `{}`", wire.kind)))?,
        observed: wire.observed,
        expected: wire.expected,
        tolerance: wire.tolerance,
    })
}

impl From<&ProviderVerificationReceipt> for ProviderVerificationReceiptWire {
    fn from(receipt: &ProviderVerificationReceipt) -> Self {
        Self {
            version: 1,
            evidence_origin: receipt.evidence_origin().map(str::to_owned),
            requests: receipt
                .requests()
                .iter()
                .map(|request| VerifiedRequestWire {
                    identity: request.identity().into(),
                    kind: request.kind().as_str().to_owned(),
                    contract_digest: request.contract_digest().to_owned(),
                    probes: request.probes().iter().map(probe_wire).collect(),
                })
                .collect(),
            digest: receipt.digest(),
        }
    }
}

impl ProviderVerificationReceiptWire {
    /// Rebuild the typed receipt, re-checking every retained field.
    ///
    /// # Errors
    /// An unsupported version, unknown kind or probe, a failed or malformed
    /// probe, or a recorded digest that is not the content's digest refuses.
    pub fn into_receipt(self) -> Result<ProviderVerificationReceipt, IoError> {
        if self.version != 1 {
            return Err(IoError::UnsupportedVersion { version: u32::from(self.version) });
        }
        if self.requests.len() > MAX_VERIFIED_REQUESTS
            || self.requests.iter().any(|r| r.probes.len() > MAX_PROBES_PER_REQUEST)
        {
            return Err(IoError::TooLarge);
        }
        let requests = self
            .requests
            .into_iter()
            .map(|request| {
                let kind = VerifiedContractKind::from_name(&request.kind)
                    .ok_or_else(|| convert(format!("unknown verified kind `{}`", request.kind)))?;
                let probes = request
                    .probes
                    .iter()
                    .map(probe_from_wire)
                    .collect::<Result<Vec<_>, IoError>>()?;
                VerifiedRequest::from_parts(
                    request.identity.into(),
                    kind,
                    request.contract_digest,
                    probes,
                )
                .map_err(|error| verification_error(&error))
            })
            .collect::<Result<Vec<_>, IoError>>()?;
        let receipt = ProviderVerificationReceipt::from_requests(self.evidence_origin, requests)
            .map_err(|error| verification_error(&error))?;
        if receipt.digest() != self.digest {
            return Err(convert("provider verification receipt digest differs from its content"));
        }
        Ok(receipt)
    }
}

fn verification_error(error: &ExternalVerificationError) -> IoError {
    let refusal = error.to_refusal();
    IoError::Refused {
        code: refusal.code,
        message: format!(
            "{}: provider verification receipt is not a passing receipt",
            refusal.detail
        ),
    }
}

/// Verify executable-provider requests against their independent probes.
///
/// The probes are recomputed here: nothing opaque crosses the boundary, and no
/// host language decides what passes.
///
/// # Errors
/// Any malformed declaration, missing or failed probe refuses with the
/// structured external-verification refusal, naming the failing request.
pub fn verify_execution_requests(
    wire: &ExecutionVerificationWire,
) -> Result<ProviderVerificationReceipt, ExternalBindingRefusalWire> {
    let malformed = |message: String| ExternalBindingRefusalWire {
        code: antecedent_core::reason_code!("invalid_argument").to_owned(),
        stage: "verify".into(),
        detail: "external_object_verification.invalid_probe".into(),
        offending: Some(message),
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    };
    if wire.requests.len() > MAX_VERIFIED_REQUESTS
        || wire.requests.iter().any(|r| r.probes.len() > MAX_PROBES_PER_REQUEST)
    {
        return Err(malformed("verification request exceeds its bounds".into()));
    }
    let contracts: Vec<ExecutionContract> = wire
        .requests
        .iter()
        .map(|request| ExecutionContract {
            identity: request.identity.clone().into(),
            deterministic: request.deterministic,
            reports_uncertainty: request.reports_uncertainty,
            artifact_blake3: request.artifact_blake3.clone(),
        })
        .collect();
    let probes = wire
        .requests
        .iter()
        .map(|request| {
            request
                .probes
                .iter()
                .map(probe_from_wire)
                .collect::<Result<Vec<_>, IoError>>()
                .map_err(|error| malformed(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let subjects: Vec<(VerificationSubject<'_>, &[VerificationProbe])> = contracts
        .iter()
        .zip(&probes)
        .map(|(contract, probes)| (VerificationSubject::Execution(contract), probes.as_slice()))
        .collect();
    verify_provider_requests(&wire.evidence_origin, &subjects)
        .map_err(|error| ExternalBindingRefusalWire::from(error.to_refusal()))
}

/// Serialize a receipt through the checksummed sectioned container.
///
/// # Errors
/// A blank artifact id or an oversized payload refuses.
pub fn receipt_to_bytes(
    receipt: &ProviderVerificationReceipt,
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    if artifact_id.trim().is_empty() {
        return Err(convert("missing artifact id"));
    }
    let meta = to_cbor(&ProviderVerificationReceiptWire::from(receipt))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "provider_verification_receipt",
            max_bytes: MAX_PROVIDER_VERIFICATION_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection { id: META_SECTION, content_type: crate::CONTENT_CBOR, bytes: &meta }],
    )
    .map_err(|error| error.into_io("unsupported provider verification layout"))
}

/// Load a receipt and check it against the digest the consumer retained.
///
/// # Errors
/// Oversized, corrupt or changed bytes, an internally inconsistent receipt, or
/// a digest other than `expected_digest` refuse before any field is exposed.
pub fn receipt_from_bytes(
    bytes: &[u8],
    expected_digest: &str,
) -> Result<ProviderVerificationReceipt, IoError> {
    if bytes.len() > MAX_PROVIDER_VERIFICATION_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_PROVIDER_VERIFICATION_ARTIFACT_BYTES,
        &[META_SECTION],
        bytes,
    )
    .map_err(|error| error.into_io("unsupported provider verification layout"))?;
    let Ok([meta]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(convert("unsupported provider verification layout"));
    };
    // The version is gated from the untyped value before the typed shape is read.
    let wire: ProviderVerificationReceiptWire =
        crate::decode_with_version(&meta, |version: u16| {
            if version == 1 {
                Ok(())
            } else {
                Err(IoError::UnsupportedVersion { version: u32::from(version) })
            }
        })?;
    if wire.digest != expected_digest {
        return Err(IoError::Refused {
            code: antecedent_core::reason_code!("external_binding_mismatch"),
            message: "provider_verification.digest_expected: receipt differs from the digest the consumer retained".into(),
        });
    }
    wire.into_receipt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(request_id: &str, observed: f64) -> ExecutionRequestWire {
        let probe = |kind: &str, observed: f64| VerificationProbeWire {
            kind: kind.into(),
            observed,
            expected: 0.0,
            tolerance: 1e-9,
        };
        ExecutionRequestWire {
            identity: ProviderIdentityWire {
                provider_id: "external".into(),
                object_id: "effect".into(),
                version_id: "a".repeat(64),
                snapshot_id: "rows".into(),
                request_id: request_id.into(),
            },
            deterministic: true,
            reports_uncertainty: false,
            artifact_blake3: Some("b".repeat(64)),
            probes: vec![
                probe("known_truth", observed),
                probe("shape", 0.0),
                probe("seeded_behavior", 0.0),
            ],
        }
    }

    fn wire(requests: Vec<ExecutionRequestWire>) -> ExecutionVerificationWire {
        ExecutionVerificationWire { evidence_origin: "fixtures/v1".into(), requests }
    }

    #[test]
    fn verified_receipt_round_trips_bytes_under_the_retained_digest() {
        let receipt =
            verify_execution_requests(&wire(vec![request("r1", 0.0), request("r2", 0.0)])).unwrap();
        let bytes = receipt_to_bytes(&receipt, "provider-verification").unwrap();
        let loaded = receipt_from_bytes(&bytes, &receipt.digest()).unwrap();
        assert_eq!(loaded, receipt);
        let refused = receipt_from_bytes(&bytes, &"0".repeat(64)).unwrap_err();
        assert!(refused.to_string().contains("retained"), "{refused}");
    }

    #[test]
    fn a_failed_request_names_its_position_and_probe() {
        let refusal =
            verify_execution_requests(&wire(vec![request("r1", 0.0), request("r2", 1.0)]))
                .unwrap_err();
        assert_eq!(refusal.detail, "external_object_verification.failed_probe");
        assert_eq!(refusal.offending.as_deref(), Some("request[1].known_truth"));
    }

    #[test]
    fn a_resealed_receipt_with_a_failed_probe_is_refused_on_load() {
        let receipt = verify_execution_requests(&wire(vec![request("r1", 0.0)])).unwrap();
        let mut forged = ProviderVerificationReceiptWire::from(&receipt);
        forged.requests[0].probes[0].observed = 5.0;
        assert!(forged.clone().into_receipt().is_err());
        let mut digest_only = ProviderVerificationReceiptWire::from(&receipt);
        digest_only.evidence_origin = Some("someone-else".into());
        assert!(digest_only.into_receipt().is_err(), "digest no longer matches the content");
    }
}
