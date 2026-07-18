//! Estimators for identified causal functionals.
//!
//! Estimators consume an [`IdentifiedEstimand`](causal_expr::IdentifiedEstimand) —
//! they never choose confounders or assert identifiability.
//!
//! ```
//! use causal_estimate::LinearAdjustmentAte;
//!
//! let est = LinearAdjustmentAte::default();
//! let _ = est;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod adjustment;
pub mod aipw;
pub mod bayesian;
pub mod conditional;
pub mod design_compile;
pub mod envelope;
pub mod error;
pub mod estimator;
pub mod frontdoor;
pub mod gcomp;
pub mod glm_adjustment;
pub mod iv;
pub mod overlap;
pub mod prediction;
pub mod prepare;
pub mod propensity;
pub mod rd;
pub mod se;
pub mod temporal_adjustment;
pub mod temporal_mediation;
pub mod util;

pub use adjustment::{
    EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, LinearFitKind, PreparedEstimationProblem,
};
pub use se::DEFAULT_RIDGE_ON_SEPARATION;
pub use design_compile::{CovariateSpec, compile_adjustment_design};
pub use se::{AnalyticSeKind, LinearSeKind};
pub use util::BootstrapSeResult;
pub use aipw::{AipwAte, AipwWorkspace};
pub use bayesian::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianGlmMechanism,
    CausalPosterior, CompiledGCompAte, GCompAteEvaluator, PosteriorFunctionalEvaluator,
    PreparedBayesianProblem, nonidentified_with_prior,
};
pub use causal_expr::EstimandMethod;
pub use conditional::ConditionalLinearAdjustment;
pub use envelope::{EnvelopeOptions, GraphEffectDraws, aggregate_effect_envelope};
pub use error::EstimationError;
pub use estimator::{Estimator, TabularAteEstimator};
pub use frontdoor::{FrontDoorTwoStage, FrontDoorWorkspace, PreparedFrontDoorProblem};
pub use glm_adjustment::{GlmAdjustmentAte, GlmAdjustmentWorkspace, PreparedGlmProblem};
pub use iv::{PreparedIvProblem, TwoStageLeastSquares, TwoStageLeastSquaresWorkspace, WaldIv};
pub use overlap::{ClipSensitivity, OverlapPolicy, OverlapReport, PropensityInterval};
pub use prediction::TemporalLinearPredictor;
pub use propensity::{
    DistanceMatching, PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityMatching,
    PropensityModel, PropensityStratification, PropensityWeighting, default_propensity_overlap,
};
pub use rd::{PreparedRdProblem, RdWorkspace, SharpRegressionDiscontinuity};
pub use temporal_adjustment::TemporalLinearAdjustment;
pub use temporal_mediation::{
    TemporalEffectSurface, TemporalMediationEstimate, TemporalMediationEstimator,
};
