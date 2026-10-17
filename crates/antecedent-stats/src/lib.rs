//! Statistical algorithms and linear-algebra backends.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod adaptive_inference;
pub mod ci;
pub mod cluster;
pub mod covariance;
mod cox_ipcw;
pub mod dcor;
pub mod describe;
pub mod design;
pub mod divergence;
pub mod error;
pub mod faer_backend;
pub mod fdr;
pub mod fit_counts;
pub mod gam;
pub mod glm;
pub mod gram;
pub mod identified_set;
pub mod interference;
pub mod joint_law;
pub mod linalg;
pub mod longitudinal_regime;
pub mod m_estimate;
pub mod matching;
pub mod measurement_error;
pub mod observation;
pub use cox_ipcw::{CoxBreslowFit, CoxIpcwFit, cox_breslow, cox_ipcw};
pub mod propensity;
pub mod quantile;
pub mod regularized;
pub mod response;
pub mod special;
pub mod synthesis;
pub mod synthesis_transport;
#[cfg(test)]
mod test_util;
pub mod twosls;
mod welford;

pub use ci::{
    BayesFactorCi, BlockTest, CalibrationReport, CiBatchRequest, CiBatchResult, CiPreparationPlan,
    CiQuery, CiResult, CiWorkspace, ConditionalIndependence, ConditionalIndependenceTest,
    ConfidenceMethod, GPDC_NYSTROM_MAX_INDUCING_POINTS, GSquared, Gpdc, GpdcNystrom, KnnDependence,
    KnnDependenceWorkspace, MixedKnnDependence, MultivariatePartialCorrelation, OracleCi,
    PairwiseMultivariateCi, PartialCorrelation, PosteriorDependenceCi, PosteriorPredictiveCi,
    PreparedCiTest, RegressionCi, RobustPartialCorrelation, SignificanceMethod, SymbolicCmi,
    WeightedPartialCorrelation, calibrate_parcorr_like, ci_from_name, ensure_alpha_resolvable,
};
#[cfg(test)]
pub(crate) use cluster::panel_effective_lag;
pub use cluster::{
    MAX_CLUSTER_DIMENSIONS, bartlett_weight, cluster_meat_scalar, combine_inclusion_exclusion,
    effective_nw_lag, few_cluster_t_ratio, intern_cluster_tuples, multiway_subset_masks,
    newey_west_meat_scalar, panel_hac_meat_scalar,
};
pub use covariance::{
    Hc2Coefficient, SandwichKind, coefficient_covariance, hc2_coefficient_bell_mccaffrey,
    score_coefficient_covariance, symmetric_eigen,
};
pub use dcor::distance_correlation;
pub use describe::{iid_se, mean_and_se};
pub use design::{
    BasisKind, CompiledDesign, ContrastCodingKind, DesignColumn, DesignColumnMap, DesignColumnRole,
    RecordedContrast, RecordedSmooth, StandardizationRecord, StandardizedColumn,
    standardize_columns,
};
pub use divergence::{
    DEFAULT_MECHANISM_PERMUTATIONS, PermutationTestResult, change_point_two_sample,
    classifier_two_sample, gaussian_kl, kernel_two_sample, kernel_two_sample_with_permutations,
    likelihood_ratio_permutations, max_abs_cusum, mean_diff_two_sample, mean_var,
    residual_likelihood_ratio, sample_std,
};
pub use error::{GlmRefusalKind, StatsError};
pub use faer_backend::FaerBackend;
pub use fdr::{
    FdrAdjustment, MultipleTestingMethod, adjust_pvalues, benjamini_hochberg, benjamini_yekutieli,
    bonferroni, holm,
};
pub use gam::{
    AdditiveDesign, GamFit, GamOptions, GamWorkspace, PredictorMatrix, SmoothSpec, expand_bspline,
    fit_gam, fit_gam_weighted, fit_gam_weighted_design, predict_gam,
};
pub use glm::{
    DEFAULT_RIDGE_ON_SEPARATION, GlmDesignRef, GlmFamily, GlmFit, GlmOptions, MultinomialDesignRef,
    MultinomialFit, NbAlphaPolicy, fit_glm, fit_glm_ridge, fit_glm_ridge_weighted,
    fit_glm_weighted, fit_logistic_weighted, fit_multinomial_logit, fit_multinomial_logit_weighted,
};
pub use gram::{
    accumulate_xtx_xty_row, chol_log_det, chol_solve, cholesky_spd, column_is_constant, form_xtx,
    form_xty, invert_square,
};
pub use identified_set::{
    IdentifiedSetInterval, IdentifiedSetIntervalMethod, imbens_manski_critical_value,
    imbens_manski_posterior_draws, imbens_manski_shared_replicates,
};
pub use interference::{
    ExposureProbabilities, ExposureProbabilityMethod, RandomizationContrast, RandomizationMean,
    exposure_probabilities, exposures, randomization_contrast, randomization_mean,
};
pub use joint_law::{
    CALIBRATION_STATUS, CORRELATION_RANGE_LABEL, CorrelationRangeEnvelope,
    DRAW_EXCHANGEABLE_RANK_CONSTRUCTION, DRAW_INTERPOLATED_CONSTRUCTION, DecisionVerdict,
    DrawSpread, ExceedanceDecision, IntervalCandidate, JointLawError, LinearMoments,
    MIXTURE_EQUAL_TAIL_CONSTRUCTION, Marginal, MixtureMoments, MomentSummary,
    NORMAL_APPROXIMATION_CONSTRUCTION, PairStructure, PairedMoments, STRUCTURAL_ENVELOPE_LABEL,
    UNCALIBRATED_SPREAD_LABEL, ValueBounds, ValueFunctional, aligned_functional_draws,
    aligned_functional_mean, aligned_linear_draws, aligned_map_draws, composed_exceedance_decision,
    correlation_range_envelope, covariance_envelope, difference_cdf_envelope,
    draw_interval_candidate, draw_spread, frechet_expectation_envelope, functional_envelope,
    linear_combination_moments, linear_variance_envelope, marginal_interval_candidate,
    mixture_exceedance, mixture_marginal, mixture_moments, normal_interval_candidate,
    paired_moments, product_moment_envelope, sum_cdf_envelope,
};
pub use linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresFit, LeastSquaresWorkspace};
pub use m_estimate::{MEstimateFit, MEstimateOptions, fit_huber_m};
#[allow(deprecated, reason = "the retained released name stays reachable")]
pub use matching::EXACT_MATCHING_ROW_LIMIT;
pub use matching::{MatchingDistance, MatchingIndex, nearest_euclidean_scalar};
pub use observation::{
    GaussianObservation, KaplanMeierIpcw, ObservationProbabilityFit, censoring_tail_restriction,
    fit_observation_logistic, gaussian_observation_log_likelihood, kaplan_meier_ipcw,
    selected_outcome_pseudo_values,
};
pub use propensity::{
    PropensityFit, PropensityWorkspace, fit_propensity, fit_propensity_diagnostic,
    fit_propensity_in_place, fit_propensity_weighted, fit_propensity_weighted_in_place,
    predict_propensity,
};
pub use quantile::{
    QuantileRule, equal_tail_interval_sorted, mad_sigma, median_sorted, quantile_sorted,
};
pub use regularized::{
    LassoFit, LassoOptions, first_col_is_exact_ones, fit_lasso, fit_lasso_with_ones_column,
    fit_ridge, predict_lasso, ridge_gram_inverse,
};
pub use response::{
    GaussianMixtureDensity, LocalPolynomialBiasCorrected, LocalPolynomialInfluence,
    LocalPolynomialPoint, LocalQuadraticWorkspace, gaussian_density,
    gaussian_local_quadratic_bias_corrected, gaussian_local_quadratic_influence,
    gaussian_local_quadratic_influence_prechecked, gaussian_local_quadratic_influence_with,
    gaussian_local_quadratic_weighted, silverman_bandwidth, silverman_normal_reference_factor,
};
pub use special::{
    NORMAL_Q975, chi_square_sf, gamma_q, gauss_hermite_standard_normal, ln_gamma, normal_ppf,
    student_t_ppf, student_t_sf,
};
pub use synthesis::{
    BetweenStudyMethod, CovarianceCheck, DEPENDENCE_ENVELOPE_LABEL, DependenceEnvelope,
    HierarchicalPosterior, MeanPrior, PooledEstimate, RandomEffectsEstimate, SynthesisError,
    TauPrior, check_covariance, equicorrelated_gls, fixed_effect_gls,
    fixed_effect_inverse_variance, hierarchical_normal_borrowing, pooled_dependence_envelope,
    random_effects_dersimonian_laird, random_effects_reml,
};
pub use twosls::{
    FirstStageDiagnostics, IvBlocks, TwoSlsFit, anderson_rubin_confidence_set,
    anderson_rubin_kf_critical, anderson_rubin_statistic, f_critical, fit_2sls, fit_wls,
};
pub use welford::Welford;
