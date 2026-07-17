//! Statistical algorithms and linear-algebra backends.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod ci;
pub mod design;
pub mod divergence;
pub mod error;
#[cfg(feature = "faer")]
pub mod faer_backend;
pub mod fdr;
pub mod glm;
pub mod gram;
pub mod linalg;
pub mod matching;
pub mod propensity;
pub mod special;
pub mod twosls;

pub use ci::{
    CalibrationReport, CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult,
    CiWorkspace, ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod, GSquared,
    Gpdc, KnnCmi, KnnCmiWorkspace, MixedKnnCmi, MultivariatePartialCorrelation, OracleCi,
    PartialCorrelation, PreparedCiTest, RegressionCi, RobustPartialCorrelation, SignificanceMethod,
    SymbolicCmi, WeightedPartialCorrelation, analytic_confidence_level, analytic_parcorr_ci,
    calibrate_parcorr_like, ci_from_name, nonparametric_permutation_count,
};
pub use design::{CompiledDesign, DesignColumnRole};
pub use divergence::{
    classifier_two_sample, gaussian_kl, mean_diff_two_sample, mean_var, residual_likelihood_ratio,
    sample_std,
};
pub use error::StatsError;
#[cfg(feature = "faer")]
pub use faer_backend::FaerBackend;
pub use fdr::{
    FdrAdjustment, MultipleTestingMethod, adjust_pvalues, benjamini_hochberg, benjamini_yekutieli,
    bonferroni, holm,
};
pub use glm::{
    GlmDesignRef, GlmFamily, GlmFit, GlmOptions, MultinomialDesignRef, MultinomialFit, fit_glm,
    fit_multinomial_logit,
};
pub use gram::{accumulate_xtx, accumulate_xtx_xty_row, form_xtx, invert_square};
pub use linalg::{DenseLinearAlgebra, LeastSquaresFit, LeastSquaresWorkspace};
pub use matching::{
    EXACT_MATCHING_ROW_LIMIT, MatchingDistance, MatchingIndex, nearest_euclidean_scalar,
};
pub use propensity::{
    PropensityFit, PropensityWorkspace, fit_propensity, fit_propensity_diagnostic,
    predict_propensity,
};
pub use special::{
    gamma_q, ln_gamma, normal_ppf, regularized_incomplete_beta, student_t_sf,
};
pub use twosls::{TwoSlsFit, fit_2sls, fit_wls};
