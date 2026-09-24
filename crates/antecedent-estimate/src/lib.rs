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
pub mod ar_kernel;
pub mod bayesian;
pub mod bayesian_iv;
pub mod bayesian_mediation;
pub mod bayesian_rd;
pub mod bayesian_robust_ate;
pub mod causal_forest;
pub mod cell_aipw;
pub mod conditional;
pub mod crossfit_aipw;
pub mod design_compile;
pub mod dml;
pub mod dr;
pub mod empirical_table;
pub mod envelope;
pub mod error;
pub mod estimator;
pub mod frontdoor;
pub mod frontdoor_functional;
pub mod functional_distribution;
pub mod gcomp;
pub mod glm_adjustment;
pub mod identified_set;
pub mod interference;
pub mod iv;
pub mod joint_if;
mod learn_nuisance;
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
pub mod statistical_transport;
pub mod temporal_adjustment;
pub mod temporal_block;
pub mod temporal_mediation;
pub mod temporal_observed_bayes;
pub mod temporal_response;
pub mod temporal_response_dispersion;
pub mod temporal_sequential;
pub mod temporal_sequential_tuples;
pub mod transport;
pub mod util;

#[cfg(test)]
#[allow(clippy::doc_markdown)]
mod calibration_coverage;

pub use adjustment::{
    BlockResampling, CandidateSelectionRecord, EffectEstimate, EstimationWorkspace,
    LinearAdjustmentAte, LinearFitKind, PreparedEstimationProblem,
};
pub mod learned_trial;
pub use learned_trial::{
    TrialAipwEstimate, TrialAipwInput, TrialAipwOptions, TrialSampling, estimate_trial_aipw,
    validate_trial_aipw, validate_trial_query,
};
mod fitted_effect;
pub use fitted_effect::FittedEffect;

pub use aipw::{AipwAte, AipwWorkspace};
pub use antecedent_expr::EstimandMethod;
pub use antecedent_learn::{
    ForestSpec, GbtSpec, LearnerProvenance, LearnerSpec, LinearSpec, LogisticSpec, NeuralSpec,
    RidgeSpec,
};
pub use antecedent_stats::FirstStageDiagnostics;
pub use ar_kernel::kernel_bias_factor;
pub use bayesian::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianGlmMechanism,
    BayesianTemporalGcomp, CausalPosterior, CompiledGCompAte, GCompAteEvaluator,
    HMC_DRAW_FLOOR_NOTE_PREFIX, HMC_MIN_DRAWS, HydrateMapping, PosteriorFunctionalEvaluator,
    PreparedBayesianProblem, coefficient_names_from_design, hmc_draw_floor_from_notes,
    hydrate_prior, hydrate_prior_from_posterior, hydrate_prior_from_quantity_summaries,
    nonidentified_with_prior, require_bayesian_n_draws,
};
pub use causal_forest::CausalForest;
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
pub use dml::{DmlAte, DmlScore};
pub use dr::DrLearner;
pub use empirical_table::{
    BayesianTransportLawDraw, BayesianTransportLawProvider, EMPIRICAL_SUPPORT_BAYESIAN_BOOTSTRAP,
    EMPIRICAL_TABLE_DIRICHLET, EMPIRICAL_TABLE_PLUGIN, EmpiricalTableEstimator,
    EmpiricalTableOptions, RegimeSample, STATE_SPACE_DIRICHLET, StatisticalTransportInput,
    assemble_point_laws, assemble_statistical_laws, catalog_axes, dependence_refusal,
    draw_bayesian_transport_laws, draw_empirical_support_transport_law,
    draw_state_space_dirichlet_transport_law, fit_empirical_joint, licensed_iid_dependence,
    licensed_iid_regimes,
};
pub use envelope::{
    EnvelopeOptions, GraphEffectDraws, aggregate_effect_envelope,
    aggregate_mixture_functional_envelope, couple_mixture_functional_draws,
};
pub use error::EstimationError;
pub use estimator::{Estimator, TabularAteEstimator};
pub use frontdoor::{
    FrontDoorTwoStage, FrontDoorWorkspace, PreparedFrontDoorProblem,
    linear_path_product_restriction,
};
pub use frontdoor_functional::{FrontDoorFunctional, FrontDoorOutcomeModel};
pub use functional_distribution::{
    AtomUncertainty, DistributionAtom, FREE_VARIABLES_AVERAGED_CODE, FunctionalDistribution,
    FunctionalDistributionWorkspace, FunctionalEffect, InterventionalDistributionEstimate,
    PreparedFunctionalDistribution, PreparedFunctionalEffect, ProbabilityInterval,
    ProbabilityIntervalUnavailable, free_variables_diagnostic, functional_cell_unevaluable,
    functional_free_variables, logit_probability_interval, support_from_functional_eval,
};
pub use glm_adjustment::{GlmAdjustmentAte, GlmAdjustmentWorkspace, PreparedGlmProblem};
pub use identified_set::{
    IdentifiedSetInterval, IdentifiedSetIntervalMethod, imbens_manski_critical_value,
    imbens_manski_posterior_draws, imbens_manski_shared_replicates,
};
pub use interference::{
    BayesianInterferenceEstimate, InterferenceEstimate, estimate_interference,
    estimate_interference_bayesian, own_treatment_level,
};
pub use iv::{PreparedIvProblem, TwoStageLeastSquares, TwoStageLeastSquaresWorkspace, WaldIv};
pub use joint_if::{
    JointCovariance, frozen_weight_mixture_scores, joint_influence_covariance, kish_n_eff,
    max_t_critical, monotone_decreasing, monotone_increasing, weighted_mean,
};
pub use observation::{
    LAGGED_OUTCOME_REGRESSOR_REFUSAL, ObservationAdjustedOutcome, ObservationEstimatorOptions,
    ObservationMechanismEstimator, SelectedOutcomeCorrection, temporal_curve_outcome_regressors,
    temporal_sequence_outcome_regressors,
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
    DirectedAncestry, MIN_WEIGHTED_ARM_N_EFF, RetargetRefusal, RetargetResult, changes_target,
    check_depends_on, exceedance_cdf_values, retarget, summarize_functional,
};
pub use scores::{
    LinearContrast, ScoreColumn, ScoreInference, ScoreSummary, ScoreSupport, ScoreTable,
    ScoreTableWire, inference_from_influence_columns,
};
pub use se::DEFAULT_RIDGE_ON_SEPARATION;
pub use se::{AnalyticSeKind, LinearSeKind};
pub use serial_dependence::{
    DEPENDENCE_ASSUMPTION_ID, DEPENDENCE_NOTE_PREFIX, DependenceScope, SerialDependence,
    TemperingFactor, long_run_tempering_factor, tempering_capped_from_notes,
    tempering_inestimable_from_notes, tempering_kappa_from_notes,
};
pub use statistical_transport::{
    BayesianStatisticalTransportEstimate, PERCENTILE_BOOTSTRAP, POSTERIOR_EQUAL_TAIL,
    StatisticalTransportEstimate, TransportUncertaintyRow, evaluate_bayesian_statistical_transport,
    evaluate_statistical_transport, percentile_interval,
};
pub use temporal_adjustment::{
    TEMPORAL_COEF_LAG_MARKER, TemporalDependenceSe, TemporalLinearAdjustment,
    is_temporal_coefficient_name, temporal_coefficient_names,
};
pub use temporal_block::{
    AlignedRows, CircularBlockFamily, RowBlockDraws, aligned_block_bootstrap, block_effective_rows,
    circular_fixed_b_scale, common_time_window, dependence_block_length, effective_rows,
    fixed_b_scale, kernel_bias_scale, normal_equation_scores, normal_equation_scores_of_residuals,
    politis_white_block_length, row_block_bootstrap_vec, score_effective_rows,
    testing_block_length,
};
pub use temporal_mediation::{
    MediationPosteriorSummary, PreparedTemporalMediation, SharedMediationBlockSe,
    TemporalEffectSurface, TemporalMediationBlockSe, TemporalMediationEstimate,
    TemporalMediationEstimator, TemporalMediationGrid, TemporalMediationIdentifiedSet,
    TemporalMediationSlice, TemporalMediationUncertainty, shared_mediation_block_bootstrap,
};
pub use temporal_response::{
    MaxDeviationBand, PreparedTemporalSurface, ResponseBlockLength, SIMULTANEOUS_BAND_CRITICAL,
    SIMULTANEOUS_BAND_LOWER, SIMULTANEOUS_BAND_MIN_REPLICATES, SIMULTANEOUS_BAND_UPPER,
    SIMULTANEOUS_BAND_WITHHELD, TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC,
    TEMPORAL_RESPONSE_BAND_WITHHELD, TEMPORAL_RESPONSE_BLOCK_CAPPED,
    TEMPORAL_RESPONSE_BLOCK_LENGTH, TEMPORAL_RESPONSE_FEW_REPLICATES,
    TEMPORAL_RESPONSE_PERSISTENCE_BOUNDARY, TemporalInterventionPlan, TemporalResponseEstimator,
    block_dispersion_inflation, circular_block_positions, circular_block_positions_into,
    clear_simultaneous_band, disclose_response_block_bootstrap, inflate_replicates,
    max_deviation_band, max_deviation_band_columns, plan_from_response_query,
    plan_temporal_intervention, publish_simultaneous_band, temporal_block_length,
};
pub use temporal_response_dispersion::{
    CellDispersion, RESPONSE_SHORT_SERIES_ROWS, influence_effective_rows,
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
    TransportResponseGridEstimate, evaluate_exact_transport, prepare_exact_transport,
    transport_augmented_response_grid, trial_to_target_bayesian_bootstrap, trial_to_target_effect,
    trial_to_target_ipw_se,
};
pub use util::BootstrapSeResult;

mod static_mediation;
pub use static_mediation::{
    MediationPriorBridge, estimate_static_mediation, estimate_static_mediation_bayesian,
    linear_no_interaction_restriction,
};
