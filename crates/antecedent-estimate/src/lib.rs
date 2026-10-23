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

pub mod adaptive_inference;
pub mod adaptive_measured_interval;
pub mod adaptive_resample_waves;
pub mod adjustment;
pub mod admg_conditional_transport;
pub mod aipw;
pub mod ancova;
pub mod ar_kernel;
pub mod bayesian;
pub mod bayesian_iv;
pub mod bayesian_local_ratio;
pub mod bayesian_mediation;
pub mod bayesian_panel_did;
pub mod bayesian_rd;
pub mod bayesian_robust_ate;
pub mod bayesian_staggered_event_study;
pub mod bayesian_synthetic_control;
pub mod bootstrap_wave;
pub mod categorical_treatment;
pub mod categorical_treatment_posterior;
pub mod causal_forest;
pub mod cell_aipw;
pub mod cohort_linear_effect;
pub mod conditional;
pub mod constraint_likelihood;
pub mod continuous_dose;
pub mod continuous_dose_posterior;
pub mod count_fitted_posterior;
pub mod counterfactual_transport;
pub mod counterfactual_transport_sampled;
pub mod cpdag_scenarios;
pub mod crossfit_aipw;
pub mod crossfit_cache;
mod crossfit_plan;
pub mod cyclic_adam;
pub mod cyclic_dynamic_inference;
pub mod cyclic_dynamics;
pub mod cyclic_functional;
pub mod cyclic_gaussian_functional;
pub mod cyclic_gradient;
pub mod cyclic_likelihood;
mod cyclic_optim;
pub mod design_compile;
pub mod design_width;
pub mod dml;
pub mod dr;
pub mod effect_constancy;
pub mod effect_estimate;
pub mod empirical_table;
pub mod envelope;
pub mod error;
pub mod estimator;
pub mod experimental_fit;
pub mod factorial_bernoulli;
pub mod frontdoor;
pub mod frontdoor_functional;
pub mod functional_distribution;
pub mod gaussian_linear_posterior;
pub mod gcomp;
pub mod glm_adjustment;
pub mod history_match;
pub mod ou_mechanisms;
pub mod polynomial_scm;
pub mod population_aipw;
pub mod population_atc;
pub use antecedent_stats::identified_set;
pub mod heterogeneity_robust_did;
pub mod interference;
pub mod interference_observational;
pub mod interference_saturation;
pub mod iv;
pub mod joint_bayesian_transport;
pub mod joint_if;
pub mod joint_panel;
pub mod lagged_instrument;
pub mod latent_class_effects;
mod learn_nuisance;
pub mod local_polynomial_ratio;
pub mod local_ratio_design;
pub mod matching_posterior_boundary;
pub mod mechanism_discrepancy;
pub mod missing_policy;
pub mod mixed_source;
pub mod multi_arm;
pub mod multiple_imputation;
pub mod nested_markov_binary;
pub mod nested_markov_uncertainty;
pub mod nonlinear_mediation;
pub mod nuisance_cache;
pub mod observation;
pub mod observational_policy;
pub mod observational_survival;
pub mod overlap;
pub mod penalized_propensity_posterior;
pub mod policy_value;
pub mod policy_value_posterior;
mod posterior_lattice;
pub mod posterior_mean;
pub mod prediction;
pub mod prepare;
pub mod propensity;
pub mod quantile;
pub mod quantile_posterior;
pub mod randomized_missing;
pub mod randomized_neyman;
pub mod randomized_posterior;
pub mod randomized_scores;
pub mod rd;
pub mod recovery;
pub mod recovery_chain;
pub mod recovery_sampled;
pub mod response;
pub mod retarget;
pub mod sampling_weights;
pub mod scenario_covariance;
pub mod scenario_invariance_report;
pub mod score_posterior;
pub mod scores;
pub mod se;
pub mod sequential_aipw_inference;
pub mod sequential_crossfit;
pub mod serial_dependence;
pub mod staggered_event_study;
pub mod statistical_transport;
pub mod survival;
pub mod survival_posterior;
pub mod switchback;
pub mod switchback_posterior;
pub mod synthetic_control;
pub mod synthetic_rollout;
pub mod temporal_adjustment;
pub mod temporal_block;
pub mod temporal_curve_posterior_meaning;
pub mod temporal_dependent_interval;
pub mod temporal_design;
pub mod temporal_history_fit;
pub mod temporal_initial_state;
pub mod temporal_link;
pub mod temporal_mediation;
pub mod temporal_observed_bayes;
pub mod temporal_penalized_score;
pub mod temporal_refresh;
pub mod temporal_response;
pub mod temporal_response_dispersion;
pub mod temporal_sequential;
pub mod temporal_sequential_tuples;
pub mod temporal_slope_design;
pub mod temporal_transport;
pub mod tiled_prediction;
pub mod trajectory_survival;
pub mod transport;
pub mod transport_scenarios;
pub mod util;
pub mod vector_treatment;
pub mod white_innovation_posterior;

#[cfg(test)]
mod calibration_coverage;

pub use adjustment::{
    CheckedLinearAdjustmentAte, EstimationWorkspace, LinearAdjustmentAte, LinearFitKind,
    PreparedEstimationProblem,
};
pub use effect_estimate::{
    AnalyticSe, BlockResampling, BlockSeRecord, BootstrapAccounting, CandidateSelectionRecord,
    CrossfitRecord, EffectEstimate, EstimateDiagnostics, EstimateHeterogeneity, EstimateProvenance,
    EstimateScores, EstimateUncertainty, FamilyInference, JointRecord, PreparationSummary,
    ScenarioBand,
};
pub mod cluster_dml_aipw;
pub mod descriptive_comparison;
pub mod dose_basis_regression;
pub mod dose_grid_functional;
pub mod dose_grid_posterior;
pub mod estimator_menu;
pub mod fit_plan;
pub mod joint_cell_factorized;
pub mod learned_continuous;
pub mod learned_joint_transport;
pub mod learned_trial;
pub use antecedent_stats::longitudinal_regime;
pub mod marginal_structural_model;
pub mod matched_case_control;
pub mod measurement;
pub mod smoothed_dose;
pub mod smoothed_dose_posterior;
pub use antecedent_learn::{FittedEffect, FittedEffectError};
pub use estimator_menu::{
    EstimatorMenu, EstimatorMenuEntry, MenuContext, MenuRefusal, transport_estimator_menu,
    transport_estimator_menu_with,
};
#[cfg(feature = "calibration-internal")]
pub use learned_continuous::learned_continuous_interval_internal;
pub use learned_continuous::{
    FoldProvenance, LEARNED_CONTINUOUS_BOUNDS, LearnedContinuousEstimate, LearnedContinuousOptions,
    LearnedContinuousUncertainty, check_membership_overlap, estimate_learned_continuous,
    parse_learned_continuous_sampling, parse_learned_continuous_target,
    refuse_learned_continuous_interval, validate_learned_continuous,
};
pub use learned_trial::{
    TrialAipwEstimate, TrialAipwInput, TrialAipwOptions, TrialSampling, estimate_trial_aipw,
    learned_trial_uncertainty, validate_trial_aipw, validate_trial_query,
};
pub use smoothed_dose::{
    DoseBasis, SMOOTHED_DOSE_BOUNDS, SmoothedDoseEstimate, SmoothedDoseInput, SmoothedDoseOptions,
    SmoothedDoseUncertainty, estimate_smoothed_dose, smoothed_dose_estimator_menu,
    validate_smoothed_dose,
};
#[cfg(feature = "calibration-internal")]
pub use smoothed_dose::{SmoothedDoseIntervalRun, smoothed_dose_interval_internal};

pub use admg_conditional_transport::{
    AdmgConditionalExactPlan, prepare_exact_admg_conditional_transport,
};
pub use aipw::{
    AipwAte, AipwWorkspace, CheckedAipwLowering, CheckedAipwPreparation, CheckedAipwProcedure,
};
pub use antecedent_expr::EstimandMethod;
pub use antecedent_learn::{
    ForestSpec, GbtSpec, LearnerProvenance, LearnerSpec, LinearSpec, LogisticSpec, NeuralSpec,
    RidgeSpec,
};
pub use antecedent_stats::FirstStageDiagnostics;
pub use ar_kernel::kernel_bias_factor;
pub(crate) use bayesian::require_bayesian_n_draws;
pub use bayesian::{
    BayesianBackendKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianTemporalGcomp,
    CausalPosterior, CompiledGCompAte, DRAW_CONTROL_NOTE_PREFIX, EndpointMonteCarloError,
    GCompAteEvaluator, HydrateMapping, HydrateSource, HydrateTarget,
    LAPLACE_RANK_INTERVAL_BELOW_DRAWS, MIN_CONTROLLED_DRAWS, PosteriorFunctionalEvaluator,
    PreparedBayesianProblem, coefficient_covariance_from_draws, endpoint_mcse_from_notes,
    endpoint_monte_carlo_error, hmc_draw_floor_from_notes, hydrate_prior,
    hydrate_prior_from_coefficient_moments, nonidentified_with_prior, supported_min_draws,
};
pub use causal_forest::CausalForest;
pub use cell_aipw::{
    CellSaturatedAipw, CellScoreSpec, ContinuousCellSpec, MAX_JOINT_BINARY, POINT_CDE_UNLICENSED,
    SharedCellInputs, cell_minus_control_contrast, contrast_named, family_cell_contrast,
    interaction_contrast,
};
pub use cluster_dml_aipw::{
    ClusterDml, ClusterDmlReceipt, DEFAULT_MIN_CLUSTERS, DEFAULT_MIN_COMPONENTS_PER_FOLD,
    DeclaredUnits, IndependenceUnit, MIN_CLUSTERS_FLOOR, MIN_COMPONENTS_PER_FOLD_FLOOR,
    cluster_fold_plan, endpoint_components, flexible_learner_dependence_refusal,
};
pub use conditional::{ConditionalArmScores, ConditionalLinearAdjustment};
pub use counterfactual_transport::{
    CounterfactualTransportEvidence, CounterfactualTransportPoint, LoweredCounterfactualItem,
    evaluate_counterfactual_transport, lower_counterfactual_transport_items,
};
pub use crossfit_aipw::{
    DEFAULT_AIPW_FOLDS, WeightedSupport, build_binary_scores, crossfit_binary_scores,
};
pub use crossfit_cache::{CrossfitCacheStats, CrossfitNuisanceCache, NuisanceScopeUse};
pub use descriptive_comparison::{
    AdjustedEstimate, ArmSummary, Availability, DescriptiveComparison, MeanPair, RawContrast,
    ReportingContrast, ReportingScale, ReportingTransform, Unavailable, compare_raw_adjusted,
    raw_contrast, refuse_column_attribution, refuse_transform_interval, transform_mean_pair,
};
pub use dml::{DmlAte, DmlScore};
pub use dr::{CateProfile, DrLearner, PointwiseCateEstimate};
pub use empirical_table::{
    BayesianTransportLawProvider, EMPIRICAL_SUPPORT_BAYESIAN_BOOTSTRAP, EMPIRICAL_TABLE_PLUGIN,
    EmpiricalTableEstimator, EmpiricalTableOptions, RegimeSample, STATE_SPACE_DIRICHLET,
    StatisticalTransportInput, assemble_point_laws, catalog_axes, dependence_refusal,
    fit_empirical_joint, licensed_iid_dependence, licensed_iid_regimes,
};
pub use envelope::{
    EnvelopeOptions, GraphEffectDraws, aggregate_effect_envelope,
    aggregate_mixture_functional_envelope,
};
pub use error::{EstimationError, ExactF64, RefusalFields};
pub use estimator::Estimator;
pub use fit_plan::JointFitPlan;
pub use frontdoor::{
    CheckedFrontDoorLowering, CheckedFrontDoorPreparation, FrontDoorTwoStage, FrontDoorWorkspace,
    PreparedFrontDoorProblem, linear_path_product_restriction,
};
pub use frontdoor_functional::{FrontDoorFunctional, FrontDoorOutcomeModel};
pub use functional_distribution::{
    AtomUncertainty, DistributionAtom, FunctionalDistribution, FunctionalDistributionWorkspace,
    FunctionalEffect, InterventionalDistributionEstimate, PreparedFunctionalDistribution,
    PreparedFunctionalEffect, ProbabilityInterval, ProbabilityIntervalUnavailable,
    free_variables_diagnostic, functional_cell_unevaluable, functional_free_variables,
    logit_probability_interval, support_from_functional_eval,
};
pub use glm_adjustment::{GlmAdjustmentAte, GlmAdjustmentWorkspace, PreparedGlmProblem};
pub use identified_set::{
    IdentifiedSetInterval, IdentifiedSetIntervalMethod, imbens_manski_critical_value,
    imbens_manski_posterior_draws, imbens_manski_shared_replicates,
};
pub use interference::{
    BayesianInterferenceEstimate, ClusterTotalInterval, InterferenceEstimate,
    estimate_cluster_interference_total, estimate_cluster_interference_total_with_inference,
    estimate_interference, estimate_interference_bayesian, own_treatment_level,
};
pub use interference_observational::{
    ObservationalExposureEstimate, ObservationalExposureSpec, estimate_observational_exposure,
};
pub use interference_saturation::{
    SaturationInterferenceEstimate, estimate_saturation_interference,
};
pub use iv::{
    CheckedIvLowering, CheckedIvPreparation, CheckedIvProcedure, PreparedIvProblem,
    TwoStageLeastSquares, TwoStageLeastSquaresWorkspace, WaldIv,
};
pub use joint_cell_factorized::{
    CellEstimate, CellOrderingSpread, CellRefusal, CellStatus, FactorizedJointConfig,
    FactorizedJointFit, JointCellReport, JointContrast, MAX_FACTORIZED_COMPONENTS, MAX_ORDERINGS,
    NormalizationCheck, OrderingSensitivity, declared_joint_nuisance, fit_factorized_joint_cells,
    orderings_for, refuse_joint_inference, refuse_joint_learner_inference,
};
pub use joint_if::{
    JointCovariance, frozen_weight_mixture_scores, joint_influence_covariance, max_t_critical,
    max_t_critical_polled, max_t_critical_with_df_polled, monotone_increasing, weighted_mean,
};
pub use matched_case_control::{
    ConditionalOddsRatio, MATCHED_ESTIMAND, MATCHED_SAMPLING, MatchedSetCounts,
    conditional_odds_ratio, parse_matched_estimand, parse_matched_sampling,
    refuse_matched_interval,
};
pub use mixed_source::{evaluate_exact_mixed_source, prepare_exact_mixed_source};
pub use observation::{
    LAGGED_OUTCOME_REGRESSOR_REFUSAL, ObservationAdjustedOutcome, ObservationEstimatorOptions,
    ObservationMechanismEstimator, SelectedOutcomeCorrection, temporal_curve_outcome_regressors,
    temporal_sequence_outcome_regressors,
};
pub use overlap::{ClipSensitivity, IpwTarget, OverlapPolicy, OverlapReport, PropensityInterval};
pub use prediction::TemporalLinearPredictor;
pub use propensity::{
    BalancedWeightingInterval, CaliperScale, DEFAULT_RIDGE_GRID, DeclaredTreatmentLevels,
    DistanceMatching, FailedFit, FallbackRecord, FoldSupport, GENERALIZED_PROPENSITY_PROVENANCE,
    KNN_DISTANCE_MATCHING_CONSTRUCTION, KnnDistanceMatching, LevelMean, LevelRows, MultiLevelAipw,
    MultiLevelEstimate, MultiLevelPopulation, MultiLevelTarget, NuisanceFallback, OtherLevels,
    PROPENSITY_FIT_STAGE, PenalizedReport, PenalizedVarianceReport, PopulationLinearVariance,
    PreparedPropensityProblem, PropensityEstimationWorkspace, PropensityMatching,
    PropensityNuisance, PropensityPenalty, PropensityStratification, PropensityWeighting,
    REFIT_BOOTSTRAP_UNCERTAINTY_KIND, ReplicatePenalties, RidgeTuning, crossfit_influence_se,
    default_propensity_overlap, provenance_marks_penalized, provenance_withholds_interval,
};
pub use quantile::empirical_threshold_grid;
pub use rd::{
    CheckedRdLowering, CheckedRdPreparation, PreparedRdProblem, RdWorkspace,
    SharpRegressionDiscontinuity,
};
pub use recovery::{RecoveredLaw, evaluate_exact_recovery, evaluate_recovered_effect};
pub use recovery_chain::{ChainPatternLaw, ChainRecoveredLaw, evaluate_chain_recovery};
pub use recovery_sampled::{
    ObservationPattern, ObservationRow, SAMPLED_RECOVERY_CALIBRATION,
    SAMPLED_RECOVERY_MAX_BINARY_VARIABLES, SAMPLED_RECOVERY_MAX_REPLICATES,
    SAMPLED_RECOVERY_MAX_ROWS, SAMPLED_RECOVERY_MIN_REPLICATES, SampledEffectInterval,
    SampledObservationInput, SampledRecoveryConfig, SampledRecoveryDetail,
    SampledRecoveryDiagnostics, SampledRecoveryError, SampledRecoveryReceipt,
    SampledRecoveryResult, derive_sampled_recovery, estimate_sampled_recovery,
    refuse_component_variance_only, replay_sampled_recovery, sampled_recovery_route_frozen,
};
pub use response::{ContinuousResponseEstimator, ContinuousResponseOptions, ResponseInfluence};
pub use retarget::{
    DirectedAncestry, RetargetRefusal, RetargetResult, changes_target, check_depends_on,
    exceedance_cdf_values, retarget, summarize_functional, weighted_overlap_fields,
    weighted_support_fields,
};
pub use sampling_weights::{RetainedSamplingWeights, SamplingWeightRecord, SamplingWeights};
pub use scores::{
    LinearContrast, ScoreColumn, ScoreInference, ScoreSummary, ScoreSupport, ScoreTable,
    ScoreTableWire, inference_from_influence_columns,
};
pub use se::AnalyticSeKind;
pub use se::DEFAULT_RIDGE_ON_SEPARATION;
pub use serial_dependence::{
    DEPENDENCE_ASSUMPTION_ID, DependenceScope, SerialDependence, tempering_capped_from_notes,
    tempering_inestimable_from_notes, tempering_kappa_from_notes, window_segments,
};
pub use statistical_transport::{
    BayesianStatisticalTransportEstimate, INTERVAL_NUMERICAL_FAILURE, NominalZTransportInterval,
    PERCENTILE_BOOTSTRAP, POSTERIOR_EQUAL_TAIL, StatisticalTransportEstimate,
    TransportUncertaintyRow, Z_TRANSPORT_INTERVAL_NOT_MEASURED, bayesian_z_transport_interval,
    evaluate_bayesian_statistical_transport, evaluate_bayesian_statistical_transport_grid,
    evaluate_statistical_transport, mz_interval_withheld_reason, mz_sampling_dependence,
    nominal_z_transport_interval, percentile_interval, z_transport_bootstrap_law_draws,
};
#[cfg(feature = "calibration-internal")]
pub use statistical_transport::{
    MzRequestInterval, MzTransportIntervals, mz_transport_bootstrap_interval,
    mz_transport_bootstrap_law_draws,
};
pub use temporal_adjustment::{
    GapSelection, NestedDirectFitMoments, TemporalDependenceSe, TemporalDesignInputs,
    TemporalLinearAdjustment, TemporalSlopeSpec, is_temporal_coefficient_name,
    nested_direct_fit_moments, temporal_coefficient_names, temporal_conditional_marginal_effect,
    temporal_conditional_marginal_effect_with_gaps, temporal_linear_slopes,
};
pub use temporal_block::{
    AlignedRows, BlockBootstrap, CircularBlockFamily, GappedAlignedRows, RowBlockDraws,
    aligned_block_bootstrap, circular_fixed_b_scale, common_time_window, dependence_block_length,
    effective_rows, gapped_aligned_block_bootstrap, gapped_row_block_bootstrap, kernel_bias_scale,
    normal_equation_scores, normal_equation_scores_of_residuals, row_block_bootstrap_vec,
    score_effective_rows,
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
    TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC, TEMPORAL_RESPONSE_BAND_WITHHELD,
    TEMPORAL_RESPONSE_BLOCK_LENGTH, TEMPORAL_RESPONSE_FEW_REPLICATES, TemporalIdentification,
    TemporalInterventionPlan, TemporalResponseEstimator, block_dispersion_inflation,
    circular_block_positions_into, clear_simultaneous_band, disclose_response_block_bootstrap,
    inflate_replicates, max_deviation_band, plan_from_response_query, publish_simultaneous_band,
};
pub use temporal_response_dispersion::{CellDispersion, RESPONSE_SHORT_SERIES_ROWS};
pub use temporal_sequential::{
    SequenceTarget, SequentialContrastDesign, SequentialInference, SequentialMechanismOverlay,
    SequentialNodeOverlay, SequentialProblem, estimate_sequence_mechanisms,
    estimate_sequence_overlays, estimate_sustained_window, estimate_sustained_window_gaps_if,
    estimate_sustained_window_with_gaps,
};
pub use temporal_sequential_tuples::{
    PreparedSequenceLevel, SequenceColumnReplacement, prepare_sequence_level,
};
pub use temporal_slope_design::{
    SlopeFit, SlopePosterior, SlopePosteriorSpec, SlopeRefitScratch, TemporalSlopeDesign,
};
pub use transport::{
    ResponseGridInputs, TransportEffectEstimate, TransportOverlapDiagnostic,
    TransportOverlapReport, TransportResponseGridEstimate, evaluate_exact_mz_transport,
    evaluate_exact_transport, evaluate_exact_z_transport, prepare_exact_mz_transport,
    prepare_exact_transport, prepare_exact_z_transport, refuse_cancelled, refuse_eval,
    transport_augmented_response_grid, transport_outcome_kind, trial_aipw_row_scores,
    trial_to_target_bayesian_bootstrap, trial_to_target_effect, trial_to_target_ipw_se,
};
pub use util::{BootstrapSeResult, DrawSpec, ReplicatePolicy};

mod splitmix;
mod static_mediation;
pub use static_mediation::{
    MediationPriorBridge, StaticMediationProblem, estimate_static_mediation,
    estimate_static_mediation_bayesian, estimate_static_mediation_bayesian_outcome_prior,
    linear_no_interaction_restriction,
};
pub mod augmented_panel_did;
/// Whole-procedure resampling shared by the rollout designs.
pub mod rollout_resampling;

/// Retained adjusted-regression execution and measured model fits.
pub mod adjustment_resume;
pub mod measurement_transport;
pub mod measurement_transport_pooled;
