//! A newer body with a changed shape is `UnsupportedVersion`, not a shape
//! error: the version is read from the untyped value and gated before the
//! typed structure is interpreted. Composition verification also refuses any
//! required feature this reader does not define.

use std::collections::BTreeMap;

use antecedent_contract::composition_verification::{CompositionGraph, CompositionNode, NodeKind};
use antecedent_core::SemanticDigest;
use antecedent_io_core::composition_verification_artifact::{
    CompositionVerificationArtifact, CompositionVerificationArtifactError,
    MAX_COMPOSITION_VERIFICATION_ARTIFACT_BYTES,
};
use antecedent_io_core::external_callback_artifact::ExternalCallbackArtifact;
use antecedent_io_core::provider_verification_artifact::receipt_from_bytes;
use antecedent_io_core::{
    ArtifactContainer, CONTENT_BYTES, CONTENT_CBOR, IoError, OutSection, encode_container,
};
use ciborium::Value;

fn newer_body() -> Vec<u8> {
    let value = Value::Map(vec![
        (Value::Text("version".into()), Value::Integer(2.into())),
        (Value::Text("a_field_this_reader_does_not_know".into()), Value::Text("x".into())),
    ]);
    let mut bytes = Vec::new();
    ciborium::into_writer(&value, &mut bytes).expect("cbor");
    bytes
}

#[test]
fn a_newer_provider_verification_receipt_is_an_unsupported_version() {
    let meta = newer_body();
    let bytes = encode_container(
        &ArtifactContainer {
            kind: "provider_verification_receipt_v1",
            note: "provider_verification_receipt",
            max_bytes: 1024 * 1024,
            artifact_id: "receipt",
            library_version: "1.0.0",
        },
        &[OutSection {
            id: "provider_verification.meta",
            content_type: CONTENT_CBOR,
            bytes: &meta,
        }],
    )
    .expect("container");
    match receipt_from_bytes(&bytes, &"0".repeat(64)) {
        Err(IoError::UnsupportedVersion { version }) => assert_eq!(version, 2),
        other => panic!("expected an unsupported version, got {other:?}"),
    }
}

#[test]
fn a_newer_external_callback_envelope_is_an_unsupported_version() {
    let meta = newer_body();
    let bytes = encode_container(
        &ArtifactContainer {
            kind: "external_callback_v1",
            note: "external_callback",
            max_bytes: 1024 * 1024,
            artifact_id: "external-callback:lab",
            library_version: "1.0.0",
        },
        &[
            OutSection { id: "external_callback.meta", content_type: CONTENT_CBOR, bytes: &meta },
            OutSection { id: "external_callback.claim", content_type: CONTENT_BYTES, bytes: b"x" },
        ],
    )
    .expect("container");
    match ExternalCallbackArtifact::inspect(&bytes) {
        Err(IoError::UnsupportedVersion { version }) => assert_eq!(version, 2),
        other => panic!("expected an unsupported version, got {other:?}"),
    }
}

#[test]
fn a_composition_verification_artifact_refuses_a_required_feature_it_does_not_define() {
    let mut graph = CompositionGraph::new();
    graph
        .add_node(CompositionNode::new(
            "claim",
            NodeKind::Claim,
            SemanticDigest::from_bytes([6; 32]),
        ))
        .expect("node");
    let (_, mut meta) =
        CompositionVerificationArtifact::seal(None, &graph, &BTreeMap::new(), &[], None)
            .expect("seal");
    assert!(meta.required_features.is_empty());
    meta.required_features.push("future_semantics".to_owned());
    CompositionVerificationArtifact::reseal_identity(&mut meta);
    assert_eq!(
        CompositionVerificationArtifact::from_meta(&meta, None).expect_err("required feature"),
        CompositionVerificationArtifactError::UnsupportedSemantics("required feature")
    );
    // The byte cap is the bound the content limits justify, not an unbounded file.
    assert_eq!(MAX_COMPOSITION_VERIFICATION_ARTIFACT_BYTES, 32 * 1024 * 1024);
}
