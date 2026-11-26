//! Three-way decision for a threshold on a bridged target quantity.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::bridge_image_artifact::{BridgeImageError, BridgeImageResult};
use crate::bridge_joint_outcomes::JointOutcomeBridgeError;
use crate::quantity_wire::ScientificQuantityWire;
use antecedent_contract::{ScientificQuantity, bridge::draws_unusable};
use antecedent_core::{CanonicalWriter, SemanticDigest};
use serde::{Deserialize, Serialize};

/// A threshold decision retains the interval's interpretation and any measurement on its source.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeIntervalDecision {
    /// Digest binding the source result, named target coordinate, and threshold.
    pub identity: [u8; 32],
    /// Interval result supplying both endpoints.
    pub interval_identity: [u8; 32],
    /// Coordinate in which the cutoff was declared.
    pub quantity: ScientificQuantityWire,
    /// Decision cutoff.
    pub threshold: f64,
    /// Strictly above, strictly below, or unable to resolve this threshold.
    pub verdict: ThresholdVerdict,
    /// Interval from which the verdict follows.
    pub endpoints: [f64; 2],
    /// Meaning of the endpoints.
    pub reading: BridgeIntervalReading,
    /// Requested coverage or posterior probability, conditional on the named reading.
    pub level: f64,
    /// Actual finite posterior mass for a posterior interval.
    pub attained_mass: Option<f64>,
    /// Coverage measurement of the whole source trial-refit interval.
    pub interval_coverage_record: Option<String>,
    /// Digest of the source interval's measurement.
    pub interval_calibration_sha: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Reported direction at the declared cutoff. Equality with a bound is unresolved.
pub enum ThresholdVerdict {
    Above,
    Below,
    Unresolved,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Interpretation of the threshold decision's input.
pub enum BridgeIntervalReading {
    PosteriorCredible,
    SamplingConfidence,
    /// Numerically verified trial bounds without current coverage evidence.
    SamplingNominal,
}

impl BridgeIntervalDecision {
    /// This terminal has no separate repeated-experiment measurement of its own choices.
    #[must_use]
    pub const fn decision_calibration(&self) -> &'static str {
        "not_measured"
    }

    pub(crate) fn evaluate(
        input: &BridgeImageResult,
        coordinate: &ScientificQuantity,
        threshold: f64,
    ) -> Result<Self, BridgeImageError> {
        let (quantity, endpoints, reading, level, attained_mass, record, sha) =
            match input {
                BridgeImageResult::JointPosterior(v) => (
                    &v.quantity,
                    v.endpoints,
                    BridgeIntervalReading::PosteriorCredible,
                    v.level,
                    Some(v.attained_mass),
                    None,
                    None,
                ),
                BridgeImageResult::TrialConfidence(v) => {
                    let (reading, record, sha) = if v.is_calibrated() {
                        (
                            BridgeIntervalReading::SamplingConfidence,
                            v.calibration.record_id.clone(),
                            v.calibration.calibration_sha.clone(),
                        )
                    } else {
                        (BridgeIntervalReading::SamplingNominal, None, None)
                    };
                    (
                        &v.interval.quantity,
                        v.interval.endpoints,
                        reading,
                        v.interval.basis.key.level,
                        None,
                        record,
                        sha,
                    )
                }
                _ => return Err(JointOutcomeBridgeError::from(draws_unusable(
                    "threshold source",
                    "supply a posterior credible interval or a measured trial confidence interval",
                ))
                .into()),
            };
        if !threshold.is_finite() || coordinate.require_same_coordinate(quantity).is_err() {
            return Err(JointOutcomeBridgeError::from(draws_unusable(
                "threshold coordinate",
                "declare a finite threshold in the interval's target coordinate and units",
            ))
            .into());
        }
        let verdict = if endpoints[0] > threshold {
            ThresholdVerdict::Above
        } else if endpoints[1] < threshold {
            ThresholdVerdict::Below
        } else {
            ThresholdVerdict::Unresolved
        };
        let mut key = CanonicalWriter::new("antecedent.bridge.interval_threshold.v1");
        key.bytes(input.identity().as_bytes())
            .text(&coordinate.canonical_identity())
            .word(threshold.to_bits());
        let id = SemanticDigest::from_bytes(key.finish());
        Ok(Self {
            identity: *id.as_bytes(),
            interval_identity: *input.identity().as_bytes(),
            quantity: quantity.into(),
            threshold,
            verdict,
            endpoints,
            reading,
            level,
            attained_mass,
            interval_coverage_record: record,
            interval_calibration_sha: sha,
        })
    }
}
