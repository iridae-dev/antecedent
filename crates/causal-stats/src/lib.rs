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

pub use ci::{
    CiBatchRequest, CiBatchResult, CiQuery, CiResult, CiWorkspace, ConditionalIndependence,
    PartialCorrelation, SignificanceMethod,
};
pub use design::{CompiledDesign, DesignColumnRole};
pub use error::StatsError;
#[cfg(feature = "faer")]
pub use faer_backend::FaerBackend;
pub use fdr::benjamini_hochberg;
pub use glm::{GlmDesignRef, GlmFamily, GlmFit, GlmOptions, fit_glm};
pub use gram::{form_xtx, invert_square};
pub use linalg::{DenseLinearAlgebra, LeastSquaresFit, LeastSquaresWorkspace};
