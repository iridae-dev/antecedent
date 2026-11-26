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
        BridgeImageRequest::IntervalThreshold { .. } => String::new(),
        BridgeImageRequest::TrialConfidence(_) => {
            stored.request.input_identity().map_err(failure)?.to_hex()
        }
        BridgeImageRequest::IdentifiedSet { source, .. } => source.claim_identity.to_hex(),
        BridgeImageRequest::JointOutcomes { source, .. }
        | BridgeImageRequest::JointPosterior { source, .. } => {
            let bytes = source.to_bytes("source").map_err(failure)?;
            antecedent_design::composition_verifiers::describe_artifact(
                NodeKind::Distribution,
                &bytes,
            )?
            .identity
        }
    };
    let plan = Study::compile_bridge_image(stored.request.clone(), ctx).map_err(failure)?;
    let result = plan.execute(ctx).map_err(failure)?;
    let program = plan.lowered().identity();
    stored.verify(program, &result).map_err(failure)?;
    let identity = result.identity().to_hex();
    let source = match &result {
        BridgeImageResult::IntervalThreshold(v) => {
            antecedent_core::SemanticDigest::from_bytes(v.interval_identity).to_hex()
        }
        _ => source,
    };
    if upstream.iter().any(|node| node.identity != source.as_str()) {
        return Err(failure("an upstream node differs from the image's retained source"));
    }
    let (mut facts, values) = match result {
        BridgeImageResult::IntervalThreshold(v) => {
            let status = serde_json::to_value(&v).map_err(failure)?;
            let facts = BTreeMap::from([
                ("interval_reading".into(), status["reading"].as_str().unwrap_or_default().into()),
                ("decision".into(), status["verdict"].as_str().unwrap_or_default().into()),
                ("decision_calibration".into(), v.decision_calibration().into()),
            ]);
            (
                facts,
                vec![
                    ("lower".into(), v.endpoints[0]),
                    ("upper".into(), v.endpoints[1]),
                    ("threshold".into(), v.threshold),
                ],
            )
        }
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
        BridgeImageResult::TrialConfidence(v) => {
            let facts = BTreeMap::from([
                ("interval_reading".into(), "sampling_confidence".into()),
                ("calibration_record".into(), v.calibration.record_id.unwrap_or_default()),
                ("calibration_sha".into(), v.calibration.calibration_sha.unwrap_or_default()),
            ]);
            let values = vec![
                ("estimate".into(), v.interval.point),
                ("lower".into(), v.interval.endpoints[0]),
                ("upper".into(), v.interval.endpoints[1]),
            ];
            (facts, values)
        }
        BridgeImageResult::JointPosterior(v) => {
            let bytes = v.posterior.to_bytes("posterior").map_err(failure)?;
            let law = antecedent_design::composition_verifiers::describe_artifact(
                NodeKind::Distribution,
                &bytes,
            )?;
            let mut facts = law.facts;
            facts.insert("interval_reading".into(), "posterior_credible".into());
            let mut values = law.values;
            values.extend([
                ("lower".into(), v.endpoints[0]),
                ("upper".into(), v.endpoints[1]),
                ("requested_mass".into(), v.level),
                ("attained_mass".into(), v.attained_mass),
            ]);
            (facts, values)
        }
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
