//! Probability / inference errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::diagnostic::RouteNativeDiagnostic;
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
    /// A typed refusal: the request is not one the library will run, as opposed
    /// to a numerical failure of a request it would run. `code` is the stable
    /// machine-readable reason code (`parity/reason_codes.toml`).
    #[error("{message}")]
    Refused {
        /// Stable reason code.
        code: &'static str,
        /// Human-readable context, led by the code.
        message: String,
    },
    /// A route check failed, or the route reported model restriction,
    /// computational failure, or unsupported inference.
    ///
    /// `diagnostics` is the record. `summary` is prose and is not parsed.
    #[error("{summary}")]
    Route {
        /// What the route measured, for a human reader.
        summary: String,
        /// Route records, including a failed check when one failed.
        diagnostics: Vec<RouteNativeDiagnostic>,
    },
}

impl ProbError {
    /// Route records on this error.
    ///
    /// Empty when this error is not a route outcome. A failed check in
    /// [`Self::Route`] is included.
    #[must_use = "a failed route check is dropped if the records are discarded"]
    pub fn route_diagnostics(&self) -> &[RouteNativeDiagnostic] {
        match self {
            Self::Route { diagnostics, .. } => diagnostics,
            _ => &[],
        }
    }
}
