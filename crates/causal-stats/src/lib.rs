//! Statistical algorithms and linear-algebra backends.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod ci;
pub mod design;
pub mod error;
#[cfg(feature = "faer")]
pub mod faer_backend;
pub mod fdr;
pub mod glm;
pub mod gram;
pub mod linalg;
pub mod matching;
pub mod propensity;
pub mod twosls;

pub use ci::{
    CalibrationReport, CiBatchRequest, CiBatchResult, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependence, Gpdc, GSquared, KnnCmi, KnnCmiWorkspace, MixedKnnCmi,
    MultivariatePartialCorrelation, OracleCi, PartialCorrelation, RegressionCi,
    RobustPartialCorrelation, SignificanceMethod, SymbolicCmi, WeightedPartialCorrelation,
    analytic_parcorr_ci, calibrate_parcorr_like,
};
pub use design::{CompiledDesign, DesignColumnRole};
pub use error::StatsError;
#[cfg(feature = "faer")]
pub use faer_backend::FaerBackend;
pub use fdr::benjamini_hochberg;
pub use glm::{GlmDesignRef, GlmFamily, GlmFit, GlmOptions, fit_glm};
pub use gram::{form_xtx, invert_square};
pub use linalg::{DenseLinearAlgebra, LeastSquaresFit, LeastSquaresWorkspace};
pub use matching::{
    EXACT_MATCHING_ROW_LIMIT, MatchingDistance, MatchingIndex, nearest_euclidean_scalar,
};
pub use propensity::{PropensityFit, PropensityWorkspace, fit_propensity, predict_propensity};
pub use twosls::{TwoSlsFit, fit_2sls, fit_wls};
