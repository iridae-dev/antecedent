//! Statistical algorithms and linear-algebra backends.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod design;
pub mod error;
#[cfg(feature = "faer")]
pub mod faer_backend;
pub mod linalg;

pub use design::{CompiledDesign, DesignColumnRole};
pub use error::StatsError;
#[cfg(feature = "faer")]
pub use faer_backend::FaerBackend;
pub use linalg::{DenseLinearAlgebra, LeastSquaresFit, LeastSquaresWorkspace};
