//! Source-backed bundle projections resolved by actual checked execution.

use std::sync::Arc;

use antecedent_design::composition_bundle::{
    BundleConsumer, BundleStage, NodeFailure, NodeKind, NodeVerifier, UpstreamNode, VerifiedNode,
};
use antecedent_design::composition_verifiers::{
    projection_node_description, register_standard_verifiers,
};
use antecedent_design::source_projection_artifact::SourceProjectionArtifact;
use antecedent_io::error::IoError;

use super::source_evidence::SourceResolution;

struct ResolvedProjectionVerifier {
    kind: NodeKind,
    receipts: Arc<[SourceResolution]>,
}

impl NodeVerifier for ResolvedProjectionVerifier {
    fn kind(&self) -> NodeKind {
        self.kind
    }
    fn verify(
        &self,
        _: &str,
        bytes: &[u8],
        upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let artifact = SourceProjectionArtifact::inspect(bytes).map_err(|error| NodeFailure {
            stage: BundleStage::SwappedEvidence,
            reason: error.to_string(),
        })?;
        let report = artifact.report();
        if artifact.is_native() {
            let receipt = self
                .receipts
                .iter()
                .find(|receipt| receipt.source_artifact_digest() == report.source_digest)
                .ok_or_else(|| NodeFailure {
                    stage: BundleStage::CallbackUnavailable,
                    reason: "composition_bundle.callback_unavailable".into(),
                })?;
            if report.unresolved.iter().any(|dependency| {
                !receipt
                    .resolved_dependencies()
                    .iter()
                    .any(|resolved| resolved.as_ref() == dependency)
            }) {
                return Err(NodeFailure {
                    stage: BundleStage::CallbackUnavailable,
                    reason: "composition_bundle.callback_unavailable".into(),
                });
            }
        }
        let mut described = projection_node_description(self.kind, &artifact, upstream)?;
        if artifact.is_native() {
            described.facts.insert("source_resolution".into(), "actual_checked_execution".into());
            described.facts.insert("native_execution_authority_issued".into(), "false".into());
        }
        Ok(described)
    }
}

/// Original bundle consumer plus source projections resolved only by opaque actual-execution receipts.
/// A receipt supplies no live fit, provider authority or calibration license.
///
/// # Errors
/// More than 64 separately executed source receipts exceed the bounded consumer scope.
pub fn source_resolved_consumer(
    receipts: Vec<SourceResolution>,
) -> Result<BundleConsumer, IoError> {
    if receipts.len() > 64 {
        return Err(IoError::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: "source_projection.limits_exceeded".into(),
        });
    }
    let receipts: Arc<[SourceResolution]> = receipts.into();
    let consumer = [
        NodeKind::CausalContract,
        NodeKind::Attestation,
        NodeKind::QuantityCoordinates,
        NodeKind::Transformation,
    ]
    .into_iter()
    .fold(BundleConsumer::new(), |consumer, kind| {
        consumer.register(Box::new(ResolvedProjectionVerifier {
            kind,
            receipts: Arc::clone(&receipts),
        }))
    });
    Ok(register_standard_verifiers(consumer))
}
