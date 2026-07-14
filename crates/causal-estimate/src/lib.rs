//! Estimators for identified causal functionals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod adjustment;
pub mod aipw;
pub mod bayesian;
pub mod error;
pub mod frontdoor;
pub mod glm_adjustment;
pub mod iv;
pub mod propensity;
pub mod rd;
pub mod temporal_adjustment;
mod util;

pub use adjustment::{
    ClipSensitivity, EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, OverlapPolicy,
    OverlapReport, PreparedEstimationProblem, PropensityInterval,
};
pub use aipw::{AipwAte, AipwWorkspace};
pub use bayesian::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianGlmMechanism,
    CausalPosterior, CompiledGCompAte, GCompAteEvaluator, PosteriorFunctionalEvaluator,
    PreparedBayesianProblem, nonidentified_with_prior,
};
pub use error::EstimationError;
pub use frontdoor::{FrontDoorTwoStage, FrontDoorWorkspace, PreparedFrontDoorProblem};
pub use glm_adjustment::{GlmAdjustmentAte, GlmAdjustmentWorkspace, PreparedGlmProblem};
pub use iv::{PreparedIvProblem, TwoStageLeastSquares, TwoStageLeastSquaresWorkspace, WaldIv};
pub use propensity::{
    DistanceMatching, PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityMatching,
    PropensityModel, PropensityStratification, PropensityWeighting, default_propensity_overlap,
};
pub use rd::{PreparedRdProblem, RdWorkspace, SharpRegressionDiscontinuity};
pub use temporal_adjustment::TemporalLinearAdjustment;
