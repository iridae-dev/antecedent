//! Estimators for identified causal functionals.
//!
//! Estimators consume an [`IdentifiedEstimand`](antecedent_expr::IdentifiedEstimand) —
//! they never choose confounders or assert identifiability.
//!
//! ```
//! use antecedent_estimate::LinearAdjustmentAte;
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
pub mod bayesian_mediation;
pub mod cell_aipw;
pub mod conditional;
pub mod crossfit_aipw;
pub mod design_compile;
pub mod envelope;
pub mod error;
pub mod estimator;
pub mod frontdoor;
pub mod functional_distribution;
pub mod gcomp;
pub mod glm_adjustment;
pub mod interference;
pub mod iv;
pub mod joint_if;
pub mod observation;
pub mod overlap;
pub mod prediction;
pub mod prepare;
pub mod propensity;
pub mod quantile;
pub mod rd;
pub mod response;
pub mod retarget;
pub mod scores;
pub mod se;
pub mod temporal_adjustment;
pub mod temporal_mediation;
pub mod temporal_response;
pub mod temporal_sequential;
pub mod transport;
pub mod util;

#[cfg(test)]
mod calibration_coverage;

pub use adjustment::{
    CandidateSelectionRecord, EffectEstimate, EstimationWorkspace, LinearAdjustmentAte,
    LinearFitKind, PreparedEstimationProblem,
};
pub use aipw::{AipwAte, AipwWorkspace};
pub use antecedent_expr::EstimandMethod;
pub use antecedent_stats::FirstStageDiagnostics;
pub use bayesian::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianGlmMechanism,
    BayesianTemporalGcomp, CausalPosterior, CompiledGCompAte, GCompAteEvaluator, HydrateMapping,
    PosteriorFunctionalEvaluator, PreparedBayesianProblem, coefficient_names_from_design,
    hydrate_prior, hydrate_prior_from_posterior, hydrate_prior_from_quantity_summaries,
    nonidentified_with_prior,
};
pub use cell_aipw::{
    CellSaturatedAipw, ContinuousCellSpec, MAX_JOINT_BINARY, POINT_CDE_UNLICENSED, contrast_named,
    interaction_contrast,
};
pub use conditional::{ConditionalArmScores, ConditionalLinearAdjustment};
pub use crossfit_aipw::{
    AIPW_CROSSFIT_PROVENANCE, DEFAULT_AIPW_FOLDS, WeightedSupport, build_binary_scores,
    crossfit_binary_scores, thresholds_of, weighted_support,
};
pub use design_compile::{CovariateSpec, compile_adjustment_design};
pub use envelope::{EnvelopeOptions, GraphEffectDraws, aggregate_effect_envelope};
pub use error::EstimationError;
pub use estimator::{Estimator, TabularAteEstimator};
pub use frontdoor::{FrontDoorTwoStage, FrontDoorWorkspace, PreparedFrontDoorProblem};
pub use functional_distribution::{
    DistributionAtom, FunctionalDistribution, FunctionalDistributionWorkspace, FunctionalEffect,
    InterventionalDistributionEstimate, PreparedFunctionalDistribution, PreparedFunctionalEffect,
};
pub use glm_adjustment::{GlmAdjustmentAte, GlmAdjustmentWorkspace, PreparedGlmProblem};
pub use interference::{InterferenceEstimate, estimate_interference, own_treatment_level};
pub use iv::{PreparedIvProblem, TwoStageLeastSquares, TwoStageLeastSquaresWorkspace, WaldIv};
pub use joint_if::{
    JointCovariance, frozen_weight_mixture_scores, joint_influence_covariance, kish_n_eff,
    max_t_critical, monotone_decreasing, monotone_increasing, weighted_mean,
};
pub use observation::{
    ObservationAdjustedOutcome, ObservationEstimatorOptions, ObservationMechanismEstimator,
    SelectedOutcomeCorrection,
};
pub use overlap::{ClipSensitivity, IpwTarget, OverlapPolicy, OverlapReport, PropensityInterval};
pub use prediction::TemporalLinearPredictor;
pub use propensity::{
    CaliperScale, DistanceMatching, PreparedPropensityProblem, PropensityEstimationWorkspace,
    PropensityMatching, PropensityModel, PropensityStratification, PropensityWeighting,
    default_propensity_overlap,
};
pub use quantile::{MIN_QUANTILE_DENSITY, empirical_threshold_grid, invert_cdf_quantile};
pub use rd::{PreparedRdProblem, RdWorkspace, SharpRegressionDiscontinuity};
pub use response::{ContinuousResponseEstimator, ContinuousResponseOptions, ResponseInfluence};
pub use retarget::{
    DirectedAncestry, MIN_WEIGHTED_ARM_N_EFF, RetargetRefusal, RetargetResult, check_depends_on,
    exceedance_cdf_values, retarget, summarize_functional,
};
pub use scores::{
    LinearContrast, ScoreColumn, ScoreInference, ScoreSummary, ScoreTable, ScoreTableWire,
    inference_from_influence_columns,
};
pub use se::DEFAULT_RIDGE_ON_SEPARATION;
pub use se::{AnalyticSeKind, LinearSeKind};
pub use temporal_adjustment::TemporalLinearAdjustment;
pub use temporal_mediation::{
    TemporalEffectSurface, TemporalMediationEstimate, TemporalMediationEstimator,
};
pub use temporal_response::TemporalResponseEstimator;
pub use transport::{
    TransportEffectEstimate, TransportOverlapDiagnostic, TransportOverlapReport,
    TransportResponseGridEstimate, transport_augmented_response_grid, trial_to_target_effect,
};
pub use util::BootstrapSeResult;

mod static_mediation;
pub use static_mediation::estimate_static_mediation;
