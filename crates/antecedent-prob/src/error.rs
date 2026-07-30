//! Probability / inference errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

/// Errors from prior construction, posterior storage, or inference backends.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum ProbError {
    /// Shape / dimension mismatch.
    #[error("shape error: {message}")]
    Shape {
        /// Context.
        message: &'static str,
    },
    /// Invalid prior or configuration.
    #[error("invalid prior: {message}")]
    InvalidPrior {
        /// Context.
        message: &'static str,
    },
    /// Inference failed to converge or produce a usable approximation.
    #[error("inference error: {message}")]
    Inference {
        /// Context.
        message: &'static str,
    },
    /// Numerical failure (singular Hessian, separation, etc.).
    #[error("numerical error: {message}")]
    Numerical {
        /// Context.
        message: String,
    },
    /// Missing required diagnostics for a reported posterior.
    #[error("missing diagnostics: {message}")]
    MissingDiagnostics {
        /// Context.
        message: String,
    },
}
