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
pub mod identified_set;
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
pub mod serial_dependence;
pub mod temporal_adjustment;
pub mod temporal_block;
pub mod temporal_mediation;
pub mod temporal_observed_bayes;
pub mod temporal_response;
pub mod temporal_sequential;
pub mod temporal_sequential_tuples;
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
    BayesianTemporalGcomp, CausalPosterior, CompiledGCompAte, GCompAteEvaluator,
    HMC_DRAW_FLOOR_NOTE_PREFIX, HMC_MIN_DRAWS, HydrateMapping, PosteriorFunctionalEvaluator,
    PreparedBayesianProblem, coefficient_names_from_design, hmc_draw_floor_from_notes,
    hydrate_prior, hydrate_prior_from_posterior, hydrate_prior_from_quantity_summaries,
    nonidentified_with_prior, require_bayesian_n_draws,
};
pub use cell_aipw::{
    CellSaturatedAipw, ContinuousCellSpec, MAX_JOINT_BINARY, POINT_CDE_UNLICENSED,
    cell_minus_control_contrast, contrast_named, family_cell_contrast, interaction_contrast,
};
pub use conditional::{ConditionalArmScores, ConditionalLinearAdjustment};
pub use crossfit_aipw::{
    AIPW_CROSSFIT_PROVENANCE, DEFAULT_AIPW_FOLDS, WeightedSupport, build_binary_scores,
    crossfit_binary_scores, thresholds_of, weighted_support,
};
pub use design_compile::{CovariateSpec, compile_adjustment_design};
pub use envelope::{
    EnvelopeOptions, GraphEffectDraws, aggregate_effect_envelope,
    aggregate_mixture_functional_envelope, couple_mixture_functional_draws,
};
pub use error::EstimationError;
pub use estimator::{Estimator, TabularAteEstimator};
pub use frontdoor::{FrontDoorTwoStage, FrontDoorWorkspace, PreparedFrontDoorProblem};
pub use functional_distribution::{
    DistributionAtom, FunctionalDistribution, FunctionalDistributionWorkspace, FunctionalEffect,
    InterventionalDistributionEstimate, PreparedFunctionalDistribution, PreparedFunctionalEffect,
    functional_cell_unevaluable, support_from_functional_eval,
};
pub use glm_adjustment::{GlmAdjustmentAte, GlmAdjustmentWorkspace, PreparedGlmProblem};
pub use identified_set::{
    IdentifiedSetInterval, IdentifiedSetIntervalMethod, imbens_manski_critical_value,
    imbens_manski_posterior_draws, imbens_manski_shared_replicates,
};
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
pub use serial_dependence::{
    DEPENDENCE_ASSUMPTION_ID, DEPENDENCE_NOTE_PREFIX, DependenceScope, SerialDependence,
    TemperingFactor, long_run_tempering_factor, tempering_kappa_from_notes,
};
pub use temporal_adjustment::{
    TEMPORAL_COEF_LAG_MARKER, TemporalDependenceSe, TemporalLinearAdjustment,
    is_temporal_coefficient_name, temporal_coefficient_names,
};
pub use temporal_block::{
    AlignedRows, MIN_EFFECTIVE_ROWS, RowBlockBootstrap, RowBlockDraws, aligned_block_bootstrap,
    common_time_window, dependence_block_length, effective_rows, fixed_b_scale,
    politis_white_block_length, row_block_bootstrap, row_block_bootstrap_vec,
};
pub use temporal_mediation::{
    MediationPosteriorSummary, TemporalEffectSurface, TemporalMediationBlockSe,
    TemporalMediationEstimate, TemporalMediationEstimator, TemporalMediationGrid,
    TemporalMediationIdentifiedSet, TemporalMediationSlice, TemporalMediationUncertainty,
};
pub use temporal_response::{
    MaxDeviationBand, PreparedTemporalSurface, SIMULTANEOUS_BAND_CRITICAL, SIMULTANEOUS_BAND_LOWER,
    SIMULTANEOUS_BAND_MIN_REPLICATES, SIMULTANEOUS_BAND_UPPER, SIMULTANEOUS_BAND_WITHHELD,
    TemporalInterventionPlan, TemporalResponseEstimator, block_dispersion_inflation,
    circular_block_positions, clear_simultaneous_band, inflate_replicates, max_deviation_band,
    max_deviation_band_columns, plan_from_response_query, plan_temporal_intervention,
    publish_simultaneous_band, temporal_block_length,
};
pub use temporal_sequential::{
    SequentialContrastDesign, SequentialMechanismOverlay, SequentialNodeOverlay,
    estimate_sequence_mechanisms, estimate_sequence_overlays, estimate_sustained_window,
};
pub use temporal_sequential_tuples::{
    PreparedSequenceLevel, SequenceColumnReplacement, prepare_sequence_level,
};
pub use transport::{
    TransportEffectEstimate, TransportOverlapDiagnostic, TransportOverlapReport,
    TransportResponseGridEstimate, transport_augmented_response_grid, trial_to_target_effect,
};
pub use util::BootstrapSeResult;

mod static_mediation;
pub use static_mediation::{
    MediationPriorBridge, estimate_static_mediation, estimate_static_mediation_bayesian,
};
