//! Estimators for identified causal functionals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod adjustment;
pub mod error;
pub mod temporal_adjustment;

pub use adjustment::{
    EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy,
    PreparedEstimationProblem,
};
pub use error::EstimationError;
pub use temporal_adjustment::TemporalLinearAdjustment;
