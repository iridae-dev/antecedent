//! Sensitivity-surface artifact (`sensitivity_decision_v1`) and its recomputing consumer.
//!
//! The artifact lives in `antecedent-io-core`. The validator's joint mechanism
//! sensitivity types are re-exported here so a decision-level consumer can adapt a
//! result without naming the validation crate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io_core::sensitivity_artifact::*;
pub use antecedent_validate::{
    AxisTipping, JointFactor, JointFactorBound, JointMechanismSensitivityResult,
    JointSensitivityLimits, JointSensitivityRange, JointSensitivityReceipt,
    JointSensitivityUncertainty, TippingBracket, TippingStatus,
};
