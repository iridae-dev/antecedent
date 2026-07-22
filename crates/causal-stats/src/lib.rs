//! Statistical algorithms and linear-algebra backends.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod ci;
pub mod covariance;
pub mod design;
pub mod divergence;
pub mod error;
pub mod faer_backend;
pub mod fdr;
pub mod gam;
pub mod glm;
pub mod gram;
pub mod linalg;
pub mod m_estimate;
pub mod matching;
pub mod propensity;
pub mod regularized;
pub mod special;
pub mod twosls;

pub use ci::{
    BayesFactorCi, CalibrationReport, CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery,
    CiResult, CiWorkspace, ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod,
    GSquared, Gpdc, KnnDependence, KnnDependenceWorkspace, MixedKnnDependence,
    MultivariatePartialCorrelation, OracleCi, PairwiseMultivariateCi, PartialCorrelation,
    PosteriorDependenceCi, PosteriorPredictiveCi, PreparedCiTest, RegressionCi,
    RobustPartialCorrelation, SignificanceMethod, SymbolicCmi, WeightedPartialCorrelation,
    analytic_confidence_level, analytic_parcorr_ci, calibrate_parcorr_like, ci_from_name,
    nonparametric_permutation_count, pairwise_multivariate_test,
};
pub use covariance::{SandwichKind, coefficient_covariance, score_coefficient_covariance};
pub use design::{
    BasisKind, CompiledDesign, ContrastCodingKind, DesignColumn, DesignColumnMap, DesignColumnRole,
    RecordedContrast, RecordedSmooth, StandardizationRecord, StandardizedColumn,
    standardize_columns,
};
pub use divergence::{
    change_point_known_split, change_point_scan, change_point_two_sample, classifier_two_sample,
    gaussian_kl, kernel_two_sample, max_abs_cusum, mean_diff_two_sample, mean_var,
    residual_likelihood_ratio, sample_std,
};
pub use error::StatsError;
pub use faer_backend::FaerBackend;
pub use fdr::{
    FdrAdjustment, MultipleTestingMethod, adjust_pvalues, benjamini_hochberg, benjamini_yekutieli,
    bonferroni, holm,
};
pub use gam::{
    GamFit, GamOptions, GamWorkspace, SmoothSpec, compile_additive_design, expand_bspline, fit_gam,
    fitted_from_gam, predict_gam,
};
pub use glm::{
    DEFAULT_RIDGE_ON_SEPARATION, GlmDesignRef, GlmFamily, GlmFit, GlmOptions, MultinomialDesignRef,
    MultinomialFit, NbAlphaPolicy, fit_glm, fit_multinomial_logit,
};
pub use gram::{accumulate_xtx, accumulate_xtx_xty_row, form_xtx, invert_square};
pub use linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresFit, LeastSquaresWorkspace};
pub use m_estimate::{MEstimateFit, MEstimateOptions, fit_huber_m};
pub use matching::{
    EXACT_MATCHING_ROW_LIMIT, MatchingDistance, MatchingIndex, nearest_euclidean_scalar,
};
pub use propensity::{
    PropensityFit, PropensityWorkspace, fit_propensity, fit_propensity_diagnostic,
    predict_propensity,
};
pub use regularized::{LassoFit, LassoOptions, fit_lasso, fit_ridge};
pub use special::{
    digamma, gamma_q, ln_gamma, normal_ppf, regularized_incomplete_beta, student_t_sf, trigamma,
};
pub use twosls::{TwoSlsFit, fit_2sls, fit_wls};
