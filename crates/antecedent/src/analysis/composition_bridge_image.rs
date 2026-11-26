//! Recompile and evaluate the image before exposing its bounds or joint law to a bundle.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::{
    ArtifactDescription, BundleStage, NodeFailure, NodeKind, NodeVerifier, UpstreamNode,
    VerifiedNode,
};
use crate::Study;
use crate::analysis::kernel_programs::bridge_images::{
    BridgeImageRequest, BridgeImageResult, StoredBridgeImage,
};
use antecedent_core::ExecutionContext;

fn failure(reason: impl ToString) -> NodeFailure {
    NodeFailure { stage: BundleStage::TamperedQuantity, reason: reason.to_string() }
}

pub(super) fn describe(
    bytes: &[u8],
    ctx: &ExecutionContext,
    upstream: &[UpstreamNode<'_>],
) -> Result<ArtifactDescription, NodeFailure> {
    let stored = StoredBridgeImage::decode(bytes).map_err(failure)?;
    let source = match &stored.request {
        BridgeImageRequest::IdentifiedSet { source, .. } => source.claim_identity.to_hex(),
        BridgeImageRequest::JointOutcomes { source, .. } => {
            let bytes = source.to_bytes("source").map_err(failure)?;
            antecedent_design::composition_verifiers::describe_artifact(
                NodeKind::Distribution,
                &bytes,
            )?
            .identity
        }
    };
    if upstream.iter().any(|node| node.identity != source.as_str()) {
        return Err(failure("an upstream node differs from the image's retained source"));
    }
    let plan = Study::compile_bridge_image(stored.request.clone(), ctx).map_err(failure)?;
    let result = plan.execute(ctx).map_err(failure)?;
    let program = plan.lowered().identity();
    stored.verify(program, &result).map_err(failure)?;
    let identity = result.identity().to_hex();
    let (mut facts, values) = match result {
        BridgeImageResult::IdentifiedSet(v) => (
            BTreeMap::from([
                (super::FACT_LAW.into(), "identified_set".into()),
                (
                    super::FACT_COORDINATES.into(),
                    antecedent_design::composition_verifiers::coordinate_id(&(&v.quantity).into()),
                ),
            ]),
            vec![("lower".into(), v.set.lower), ("upper".into(), v.set.upper)],
        ),
        BridgeImageResult::JointOutcomes(v) => {
            let bytes = v.distribution.to_bytes("result").map_err(failure)?;
            let law = antecedent_design::composition_verifiers::describe_artifact(
                NodeKind::Distribution,
                &bytes,
            )?;
            (law.facts, law.values)
        }
    };
    facts.insert("image_program".into(), program.to_hex());
    facts.insert("image_source".into(), source);
    Ok(ArtifactDescription { kind: NodeKind::ScientificBridgeImage, identity, facts, values })
}

pub(super) struct BridgeImageVerifier(pub(super) ExecutionContext);

impl NodeVerifier for BridgeImageVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::ScientificBridgeImage
    }
    fn verify(
        &self,
        _id: &str,
        bytes: &[u8],
        upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let value = describe(bytes, &self.0, upstream)?;
        Ok(VerifiedNode { identity: value.identity, facts: value.facts, values: value.values })
    }
}
