//! Probability / inference errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

/// Errors from prior construction, posterior storage, or inference backends.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProbError {
    /// Shape / dimension mismatch.
    Shape {
        /// Context.
        message: &'static str,
    },
    /// Invalid prior or configuration.
    InvalidPrior {
        /// Context.
        message: &'static str,
    },
    /// Inference failed to converge or produce a usable approximation.
    Inference {
        /// Context.
        message: &'static str,
    },
    /// Numerical failure (singular Hessian, separation, etc.).
    Numerical {
        /// Context.
        message: String,
    },
    /// Missing required diagnostics for a reported posterior.
    MissingDiagnostics {
        /// Context.
        message: &'static str,
    },
}

impl fmt::Display for ProbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { message } => write!(f, "shape error: {message}"),
            Self::InvalidPrior { message } => write!(f, "invalid prior: {message}"),
            Self::Inference { message } => write!(f, "inference error: {message}"),
            Self::Numerical { message } => write!(f, "numerical error: {message}"),
            Self::MissingDiagnostics { message } => {
                write!(f, "missing diagnostics: {message}")
            }
        }
    }
}

impl std::error::Error for ProbError {}
