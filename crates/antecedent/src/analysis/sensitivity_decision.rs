//! F17: consume an existing assumption-sensitivity result in a durable decision.
//!
//! The facade routes three things and owns no scientific rule of its own:
//!
//! * [`PreparedZTransport::sensitivity_artifact`] runs the 2.2 joint mechanism
//!   sensitivity on a prepared z stage and adapts its result into a
//!   composition-ready [`SensitivityArtifact`] (assumption coordinate, effect surface,
//!   coordinate support, uncertainty relationship, provenance);
//! * [`decide`] evaluates a [`DecisionContract`] over the artifact's surface and
//!   classifies the decision as an invariant action, an assumption-dependent switch
//!   with its tipping coordinate, or no robust action;
//! * [`export`] and [`consume`] move the artifact through the bounded sectioned
//!   container; consumption recomputes the identity digests and the decision outcome
//!   and refuses a resealed mutation.
//!
//! An assumption range is never a probability and never a sampling interval. The
//! sampling interval, when one exists, is reported next to the decision
//! ([`SamplingReport`]); composing it with the range is refused with
//! `composition_not_licensed` unless the method is listed in
//! [`LICENSED_SAMPLING_COMPOSITIONS`] (empty today). Decisions are evaluated only at
//! the declared grid points; nothing between them is evaluated.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, ScientificQuantity};
use antecedent_io::IoError;
use antecedent_validate::JointDeviationSpec;

pub use antecedent_design::sensitivity_decision::*;
pub use antecedent_io::sensitivity_artifact::{
    MAX_SENSITIVITY_ACTIONS, MAX_SENSITIVITY_ARTIFACT_BYTES, MAX_SENSITIVITY_GRID_POINTS,
    MAX_SENSITIVITY_QUANTITIES, MAX_SENSITIVITY_RANGED_QUANTITIES, MAX_SENSITIVITY_UTILITY_DEPTH,
    SENSITIVITY_ARTIFACT_FEATURE, SENSITIVITY_ARTIFACT_VERSION, SENSITIVITY_INFERENCE_CLAIM,
};

use antecedent_design::decision_contract::DecisionContract;

use super::PreparedZTransport;

impl PreparedZTransport {
    /// Run the 2.2 joint mechanism sensitivity of the retained checked formula and
    /// adapt its assumption range into a composition-ready [`SensitivityArtifact`].
    ///
    /// With one declared factor the surface is exact on `grid_points` equally spaced
    /// contamination fractions; with two factors only the unperturbed point and the
    /// declared box are exact, so `grid_points` must be 2. The 2.2 withheld sampling
    /// status is carried unchanged: the range stays an assumption range.
    ///
    /// # Errors
    ///
    /// The reason-coded refusal of the joint sensitivity contract, or of the
    /// adaptation (a `grid_points` the result cannot fill exactly, an unsupported
    /// factor set, mixed estimands in `actions`).
    pub fn sensitivity_artifact(
        &self,
        spec: &JointDeviationSpec,
        effect: &ScientificQuantity,
        grid_points: usize,
        actions: Vec<ActionUtility>,
        causal_contract_id: &str,
        ctx: &ExecutionContext,
    ) -> Result<SensitivityArtifact, IoError> {
        let result = self.joint_mechanism_sensitivity(spec, ctx)?;
        SensitivityArtifact::from_joint_result(
            &result,
            effect,
            grid_points,
            actions,
            causal_contract_id,
        )
    }
}

/// Decide over the surface of `artifact` under `contract`.
///
/// Equivalent to [`evaluate_sensitivity_decision`]; see it for the refusals.
///
/// # Errors
///
/// A requested sampling composition that is not licensed, a contract the surface
/// cannot answer (unlike units, an unsupported criterion, an input the surface does
/// not carry, weights on a ranged surface), an unsupported grid point inside the
/// evaluated range, or a malformed range.
pub fn decide(
    contract: &DecisionContract,
    artifact: &SensitivityArtifact,
    spec: &SensitivityDecisionSpec,
) -> Result<SensitivityDecisionResult, SensitivityDecisionError> {
    evaluate_sensitivity_decision(contract, artifact, spec)
}

/// Serialize an artifact through the bounded sectioned container.
///
/// # Errors
///
/// An empty id, an encoding failure or an oversized payload.
pub fn export(artifact: &SensitivityArtifact, artifact_id: &str) -> Result<Vec<u8>, IoError> {
    artifact.to_bytes(artifact_id)
}

/// Consume an artifact by recomputation; `expected` is an identity the consumer
/// retained independently of the bytes.
///
/// # Errors
///
/// Corruption, another major version, any validation refusal, a changed identity or a
/// stored decision outcome that does not replay.
pub fn consume(
    bytes: &[u8],
    expected: Option<&SensitivityIdentity>,
) -> Result<SensitivityArtifact, IoError> {
    SensitivityArtifact::from_bytes(bytes, expected)
}
