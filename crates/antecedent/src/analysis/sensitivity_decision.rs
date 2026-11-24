//! Consume an existing assumption-sensitivity result in a durable decision.
//!
//! The facade routes three things and owns no scientific rule of its own:
//!
//! * [`PreparedZTransport::sensitivity_artifact`] runs the joint mechanism
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
//! A transported response's point value is shifted by a declared bound through
//! [`super::transport_sensitivity`], which builds the same [`SensitivityArtifact`].
//!
//! An assumption range is never a probability and never a sampling interval. The
//! sampling interval, when one exists, is reported next to the decision
//! ([`SamplingReport`]); composing it with the range is refused with
//! `composition_not_supported` unless the method is listed in
//! [`SUPPORTED_SAMPLING_COMPOSITIONS`] (empty today). Decisions are evaluated only at
//! the declared grid points; nothing between them is evaluated.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::ExecutionContext;
use antecedent_io::IoError;
use antecedent_validate::JointDeviationSpec;

pub use antecedent_design::sensitivity_decision::{
    ASSUMPTION_RANGE_STATEMENT, ActionUtility, AssumptionCoordinate, AssumptionRangeStatement,
    DecisionSwitch, IdentifiedBound, NoRobustReason, PointSupport,
    SENSITIVITY_DECISION_INTERPRETATION, SUPPORTED_SAMPLING_COMPOSITIONS, SamplingInterval,
    SamplingReport, SamplingStatus, ScenarioCoverage, SensitivityArtifact,
    SensitivityDecisionError, SensitivityDecisionResult, SensitivityDecisionSpec,
    SensitivityIdentity, SensitivityOutcome, SensitivityParts, SensitivityProvenance,
    SourceTipping, SurfaceQuantity, UncertaintyRelationship, UtilityTerm, contract_from_artifact,
};
pub use antecedent_io::sensitivity_artifact::{
    MAX_SENSITIVITY_ACTIONS, MAX_SENSITIVITY_ARTIFACT_BYTES, MAX_SENSITIVITY_GRID_POINTS,
    MAX_SENSITIVITY_QUANTITIES, MAX_SENSITIVITY_SCENARIO_EVALUATIONS,
    MAX_SENSITIVITY_SCENARIO_NODE_EVALUATIONS, MAX_SENSITIVITY_UTILITY_DEPTH,
    MAX_SENSITIVITY_UTILITY_NODES, SENSITIVITY_ARTIFACT_FEATURE, SENSITIVITY_ARTIFACT_VERSION,
    SENSITIVITY_INFERENCE_CLAIM,
};

use antecedent_design::decision_contract::DecisionContract;

use super::PreparedZTransport;

/// Evaluate `contract` over the surface of `artifact`.
///
/// Every evaluated grid point and range-vertex scenario becomes one structural atom of
/// `evaluate_structural`, so the contract's `StructuralPolicy` decides how the
/// assumption scenarios are combined (invariant best action, worst case, or Bayes over
/// declared genuine weights). The result also classifies the decision along the
/// coordinate (see `SensitivityOutcome`).
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::sensitivity_decision::evaluate_sensitivity_decision`].
///
/// # Errors
/// Refuses (a) a requested sampling composition that is not supported
/// (`composition_not_supported`, code `route_not_supported`); (b) an invalid contract, an
/// unsupported criterion, unlike units or an input the surface does not carry, weights on
/// a ranged surface, a weight count that does not match the grid, and `Bayes` without
/// genuine weights (`wrong_contract`, code `decision_contract_unsatisfied`); (c) an
/// unsupported grid point inside the evaluated range (`unsupported_coordinate`, code
/// `quantity_semantics_mismatch`); (d) a malformed or empty range (`invalid_surface`).
#[deprecated(
    since = "3.1.0",
    note = "use the compiled SensitivityDecision route, which calls this kernel function with the same inputs"
)]
pub fn evaluate_sensitivity_decision(
    contract: &antecedent_design::DecisionContract,
    artifact: &antecedent_io_core::sensitivity_artifact::SensitivityArtifact,
    spec: &antecedent_design::sensitivity_decision::SensitivityDecisionSpec,
) -> Result<
    antecedent_design::sensitivity_decision::SensitivityDecisionResult,
    antecedent_design::sensitivity_decision::SensitivityDecisionError,
> {
    antecedent_design::sensitivity_decision::evaluate_sensitivity_decision(contract, artifact, spec)
}

impl PreparedZTransport {
    /// Run the joint mechanism sensitivity of the retained checked formula and
    /// adapt its assumption range into a composition-ready [`SensitivityArtifact`].
    ///
    /// With one declared factor the surface is exact on `grid_points` equally spaced
    /// contamination fractions; with two factors only the unperturbed point and the
    /// declared box are exact, so `grid_points` must be 2. The withheld sampling
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
/// A requested sampling composition that is not supported, a contract the surface
/// cannot answer (unlike units, an unsupported criterion, an input the surface does
/// not carry, weights on a ranged surface), an unsupported grid point inside the
/// evaluated range, or a malformed range.
pub fn decide(
    contract: &DecisionContract,
    artifact: &SensitivityArtifact,
    spec: &SensitivityDecisionSpec,
) -> Result<SensitivityDecisionResult, SensitivityDecisionError> {
    antecedent_design::sensitivity_decision::evaluate_sensitivity_decision(contract, artifact, spec)
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
