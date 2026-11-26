//! The sensitivity artifact's standard bundle adapter delegates to its native replay.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::ExecutionContext;

use super::{
    ArtifactDescription, BundleStage, NodeFailure, NodeKind, NodeVerifier, UpstreamNode,
    VerifiedNode,
};
use crate::analysis::longitudinal_joint_sensitivity::{
    JointSensitivityError, LongitudinalSensitivityChoice, consume_artifact,
};
use crate::error::CausalError;

pub(super) fn describe(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<ArtifactDescription, NodeFailure> {
    let result = consume_artifact(bytes, None, ctx).map_err(|error| {
        let stage = match error {
            JointSensitivityError::Cancelled
            | JointSensitivityError::Compile(CausalError::Cancelled { .. }) => {
                BundleStage::ExecutionCancelled
            }
            JointSensitivityError::Resource(_)
            | JointSensitivityError::Compile(CausalError::Resource { .. }) => {
                BundleStage::ResourceRefused
            }
            JointSensitivityError::Identity => BundleStage::SwappedEvidence,
            _ => BundleStage::TamperedQuantity,
        };
        NodeFailure { stage, reason: error.to_string() }
    })?;
    let report = result.report();
    let calibration = result.calibration();
    let mut facts = BTreeMap::from([
        ("program_identity".into(), result.program_identity().into()),
        ("inference".into(), "whole_subject_bootstrap_supremum".into()),
        ("calibration".into(), calibration.status.as_str().into()),
        ("population".into(), result.spec().population.clone()),
        ("outcome".into(), result.spec().outcome.clone()),
        ("units".into(), result.spec().unit.clone()),
    ]);
    if let Some(record) = calibration.record_id {
        facts.insert("calibration_record".into(), record);
    }
    let mut values = vec![
        ("assumption_lower".into(), report.assumption_range[0]),
        ("assumption_upper".into(), report.assumption_range[1]),
        ("sampling_lower".into(), report.extrema_intervals[0][0]),
        ("sampling_upper".into(), report.extrema_intervals[1][1]),
        ("joint_band_radius".into(), report.sampling_radius),
    ];
    if let Some(decision) = report.decision {
        let extrema_calibration = calibration.secondary.first();
        let extrema_status =
            extrema_calibration.map_or("not_measured", |slot| slot.status.as_str());
        facts.insert("decision_extrema_calibration".into(), extrema_status.into());
        if let Some(record) = extrema_calibration.and_then(|slot| slot.record_id.as_ref()) {
            facts.insert("decision_extrema_calibration_record".into(), record.clone());
        }
        let label = |decision| match decision {
            LongitudinalSensitivityChoice::PreferActive => "prefer_active",
            LongitudinalSensitivityChoice::PreferControl => "prefer_control",
            LongitudinalSensitivityChoice::Undetermined => "undetermined",
        };
        facts.insert("assumption_choice".into(), label(decision.assumption_choice).into());
        facts.insert("sampling_choice".into(), label(decision.sampling_choice).into());
        values.push(("threshold".into(), decision.threshold));
    }
    Ok(ArtifactDescription {
        kind: NodeKind::LongitudinalJointSensitivity,
        identity: result.identity().to_hex(),
        facts,
        values,
    })
}

pub(super) struct Verifier(pub(super) ExecutionContext);

impl NodeVerifier for Verifier {
    fn kind(&self) -> NodeKind {
        NodeKind::LongitudinalJointSensitivity
    }

    fn verify(
        &self,
        _id: &str,
        bytes: &[u8],
        upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        if !upstream.is_empty() {
            return Err(NodeFailure {
                stage: BundleStage::SwappedEvidence,
                reason: "the sensitivity report already retains the exact common subject source"
                    .into(),
            });
        }
        let description = describe(bytes, &self.0)?;
        Ok(VerifiedNode {
            identity: description.identity,
            facts: description.facts,
            values: description.values,
        })
    }
}
