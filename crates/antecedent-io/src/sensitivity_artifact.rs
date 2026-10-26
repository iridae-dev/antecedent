//! Sensitivity-surface artifact (`sensitivity_decision_v1`) and its recomputing consumer.
//!
//! The artifact lives in `antecedent-io-core`. The validator's joint mechanism
//! sensitivity types are re-exported here so a decision-level consumer can adapt a
//! result without naming the validation crate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io_core::sensitivity_artifact::{
    ASSUMPTION_RANGE_STATEMENT, ActionUtility, AssumptionCoordinate, AssumptionRangeStatement,
    DecisionSwitch, IdentifiedBound, JointFactorView, JointSensitivityView,
    LICENSED_SAMPLING_COMPOSITIONS, MAX_SENSITIVITY_ACTIONS, MAX_SENSITIVITY_ARTIFACT_BYTES,
    MAX_SENSITIVITY_GRID_POINTS, MAX_SENSITIVITY_QUANTITIES, MAX_SENSITIVITY_SCENARIO_EVALUATIONS,
    MAX_SENSITIVITY_UTILITY_DEPTH, NoRobustReason, PointInput, PointLeaders, PointSupport,
    SENSITIVITY_ARTIFACT_FEATURE, SENSITIVITY_ARTIFACT_VERSION, SENSITIVITY_CLAIM_LINK_ID,
    SENSITIVITY_INFERENCE_CLAIM, SamplingInterval, SamplingStatus, SensitivityArtifact,
    SensitivityIdentity, SensitivityMeta, SensitivityOutcome, SensitivityParts,
    SensitivityProvenance, SourceTipping, SurfaceQuantity, UncertaintyRelationship, UtilityTerm,
    classify_decision_points,
};
pub use antecedent_validate::{
    AxisTipping, JointFactor, JointFactorBound, JointMechanismSensitivityResult,
    JointSensitivityLimits, JointSensitivityRange, JointSensitivityReceipt,
    JointSensitivityUncertainty, TippingBracket, TippingStatus,
};

#[cfg(feature = "test-util")]
#[doc(hidden)]
pub use antecedent_io_core::sensitivity_artifact::{decode_parts, encode_parts};
