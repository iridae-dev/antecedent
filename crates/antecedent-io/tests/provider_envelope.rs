//! Compile and exercise the standalone provider envelope before shared lib wiring.
// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../src/provider_envelope.rs"]
mod provider_envelope;

use std::collections::BTreeMap;

use provider_envelope::{ExternalProviderTrust, ProviderEnvelope, ProviderEnvelopeHeader,
    open_provider_envelope, open_verified_provider_envelope, seal_provider_envelope};

#[test]
fn verified_external_result_requires_host_receipt_and_preserves_provenance() {
    let evidence = "c".repeat(64);
    let envelope = ProviderEnvelope {
        header: ProviderEnvelopeHeader {
            version: 1,
            provider: "external-provider".into(), query_family: "effect".into(),
            trust: ExternalProviderTrust::VerifiedExtension,
            spec_digest: "a".repeat(64), request_digest: "b".repeat(64),
            verification_evidence_digest: Some(evidence.clone()),
            estimate: vec![1.0, 2.0], shape: vec![2],
            uncertainty: Some(vec![0.1, 0.2]), uncertainty_shape: Some(vec![2]),
            uncertainty_semantics: "pointwise_standard_error".into(),
            assumptions: vec!["independent assignment".into()],
            provider_support_status: "caller_asserted".into(),
            provenance: BTreeMap::from([
                ("registry_name".into(), "external-provider".into()),
                ("trust_boundary".into(), "verified_extension".into()),
                ("verification_evidence_digest".into(), evidence),
                ("entry_point".into(), "external_pkg:create_provider".into()),
            ]),
            external_artifact_digest: None,
        },
        external_artifact: Some(b"opaque-provider-artifact".to_vec()),
    };
    let bytes = seal_provider_envelope(envelope).unwrap();
    assert!(open_provider_envelope(&bytes).unwrap_err().contains("host verification"));
    let decoded = open_verified_provider_envelope(&bytes, &"a".repeat(64),
        &"b".repeat(64), &"c".repeat(64)).unwrap();
    assert_eq!(decoded.header.trust, ExternalProviderTrust::VerifiedExtension);
    assert_eq!(decoded.header.provenance["entry_point"], "external_pkg:create_provider");
    assert_eq!(decoded.external_artifact.unwrap(), b"opaque-provider-artifact");
    assert!(open_verified_provider_envelope(&bytes, &"a".repeat(64),
        &"d".repeat(64), &"c".repeat(64)).is_err());
}
