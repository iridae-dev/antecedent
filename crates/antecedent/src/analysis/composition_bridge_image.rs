//! Recompile and evaluate the image before exposing its bounds or joint law to a bundle.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::{
    ArtifactDescription, BundleStage, NodeFailure, NodeKind, NodeVerifier, UpstreamNode,
    VerifiedNode,
};
use crate::analysis::kernel_programs::bridge_images::{
    BridgeImageError, BridgeImageRequest, BridgeImageResult, ExactImageError, StoredBridgeImage,
    replay_stored_image,
};
use antecedent_core::ExecutionContext;
use antecedent_design::composition_verifiers::describe_checked_distribution;

fn failure(reason: &(impl ToString + ?Sized)) -> NodeFailure {
    NodeFailure { stage: BundleStage::TamperedQuantity, reason: reason.to_string() }
}

fn replay_failure(error: &ExactImageError) -> NodeFailure {
    let stage = if error.is_cancelled() {
        BundleStage::ExecutionCancelled
    } else if error.is_resource_refusal() {
        BundleStage::ResourceRefused
    } else {
        BundleStage::TamperedQuantity
    };
    NodeFailure { stage, reason: error.to_string() }
}

pub(super) fn describe(
    bytes: &[u8],
    ctx: &ExecutionContext,
    upstream: &[UpstreamNode<'_>],
) -> Result<ArtifactDescription, NodeFailure> {
    let (request, binding) = StoredBridgeImage::decode_with_context(bytes, ctx)
        .map_err(|error| replay_failure(&error.into()))?
        .into_parts();
    check_patient_source(&request, upstream)?;
    let source = match &request {
        BridgeImageRequest::IntervalThreshold { .. } => String::new(),
        BridgeImageRequest::TrialConfidence(_) => {
            request.input_identity().map_err(|e| replay_failure(&e.into()))?.to_hex()
        }
        BridgeImageRequest::IdentifiedSet { source, .. } => source.claim_identity.to_hex(),
        BridgeImageRequest::JointOutcomes { source, .. }
        | BridgeImageRequest::JointPosterior { source, .. } => {
            describe_checked_distribution(source)?.identity
        }
    };
    let (program, result) =
        replay_stored_image(request, &binding, ctx).map_err(|error| replay_failure(&error))?;
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
    let _result_workspace = result.admit_readout(ctx).map_err(|e| replay_failure(&e.into()))?;
    let (mut facts, values) = readout(result)?;
    if ctx.cancellation.is_cancelled() {
        return Err(replay_failure(
            &BridgeImageError::from(antecedent_io::IoError::StreamCancelled).into(),
        ));
    }
    facts.insert("image_program".into(), program.to_hex());
    facts.insert("image_source".into(), source);
    Ok(ArtifactDescription { kind: NodeKind::ScientificBridgeImage, identity, facts, values })
}

type Readout = (BTreeMap<String, String>, Vec<(String, f64)>);

fn readout(result: BridgeImageResult) -> Result<Readout, NodeFailure> {
    Ok(match result {
        BridgeImageResult::IntervalThreshold(v) => {
            let status = serde_json::to_value(&v).map_err(|error| failure(&error))?;
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
            let reading =
                if v.is_calibrated() { "sampling_confidence" } else { "sampling_nominal" };
            let mut facts = BTreeMap::from([
                ("interval_reading".into(), reading.into()),
                ("calibration_status".into(), v.calibration.status.as_str().into()),
            ]);
            if v.is_calibrated() {
                facts.insert(
                    "calibration_record".into(),
                    v.calibration.record_id.unwrap_or_default(),
                );
                facts.insert(
                    "calibration_sha".into(),
                    v.calibration.calibration_sha.unwrap_or_default(),
                );
            }
            let values = vec![
                ("estimate".into(), v.interval.point),
                ("lower".into(), v.interval.endpoints[0]),
                ("upper".into(), v.interval.endpoints[1]),
            ];
            (facts, values)
        }
        BridgeImageResult::JointPosterior(v) => {
            let law = describe_checked_distribution(&v.posterior)?;
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
            let law = describe_checked_distribution(&v.distribution)?;
            (law.facts, law.values)
        }
    })
}

fn check_patient_source(
    request: &BridgeImageRequest,
    upstream: &[UpstreamNode<'_>],
) -> Result<(), NodeFailure> {
    for patient in
        upstream.iter().filter(|node| node.kind == NodeKind::PatientSurrogateIdentification)
    {
        let BridgeImageRequest::IdentifiedSet { source, .. } = request else {
            return Err(NodeFailure {
                stage: BundleStage::UnsupportedLaw,
                reason: "a population patient-surrogate result supplies an identified set".into(),
            });
        };
        if patient.facts.get(super::FACT_LAW).map(String::as_str) != Some("identified_set") {
            return Err(NodeFailure {
                stage: BundleStage::UnsupportedLaw,
                reason: "a patient-surrogate source needs a verified population identified set"
                    .into(),
            });
        }
        let coordinate =
            antecedent_design::composition_verifiers::coordinate_id(&(&source.quantity).into());
        let bound = |name| {
            patient.values.iter().find_map(|(key, value)| (key == name).then_some(value.to_bits()))
        };
        if patient.identity != source.claim_identity.to_hex()
            || patient.facts.get(super::FACT_COORDINATES) != Some(&coordinate)
            || bound("lower") != Some(source.set.lower.to_bits())
            || bound("upper") != Some(source.set.upper.to_bits())
        {
            return Err(failure(
                "the image source coordinate or bounds differ from the verified patient-surrogate result",
            ));
        }
    }
    Ok(())
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
