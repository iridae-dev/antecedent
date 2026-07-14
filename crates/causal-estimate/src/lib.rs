//! Estimators for identified causal functionals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod adjustment;
pub mod aipw;
pub mod bayesian;
pub mod conditional;
pub mod envelope;
pub mod error;
pub mod estimator;
pub mod gcomp;
pub mod overlap;
pub mod prepare;
pub mod frontdoor;
pub mod glm_adjustment;
pub mod iv;
pub mod prediction;
pub mod propensity;
pub mod rd;
pub mod temporal_adjustment;
pub mod temporal_mediation;
mod util;

pub use adjustment::{EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, PreparedEstimationProblem};
pub use estimator::{Estimator, FittedEstimator, TabularAteEstimator};
pub use overlap::{ClipSensitivity, OverlapPolicy, OverlapReport, PropensityInterval};
pub use causal_expr::EstimandMethod;
pub use aipw::{AipwAte, AipwWorkspace};
pub use bayesian::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianGlmMechanism,
    CausalPosterior, CompiledGCompAte, GCompAteEvaluator, PosteriorFunctionalEvaluator,
    PreparedBayesianProblem, nonidentified_with_prior,
};
pub use envelope::{EnvelopeOptions, GraphEffectDraws, aggregate_effect_envelope};
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
pub use conditional::ConditionalLinearAdjustment;
pub use prediction::TemporalLinearPredictor;
pub use temporal_mediation::{TemporalEffectSurface, TemporalMediationEstimate, TemporalMediationEstimator};

