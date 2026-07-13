//! Estimators for identified causal functionals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod adjustment;
pub mod error;
pub mod propensity;
pub mod temporal_adjustment;

pub use adjustment::{
    EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy, OverlapReport,
    PreparedEstimationProblem,
};
pub use error::EstimationError;
pub use propensity::{
    DistanceMatching, PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityMatching,
    PropensityModel, PropensityStratification, PropensityWeighting, default_propensity_overlap,
};
pub use temporal_adjustment::TemporalLinearAdjustment;
