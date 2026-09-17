//! Design-crate errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

/// Errors from experiment / measurement design evaluation.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DesignError {
    /// Empty candidate list.
    #[error("no candidate designs to rank")]
    EmptyCandidates,
    /// Empty graph / model posterior draws.
    #[error("empty posterior ensemble for design evaluation")]
    EmptyPosterior,
    /// Shape / length mismatch in inputs.
    #[error("design shape error: {0}")]
    Shape(String),
    /// Invalid configuration (budget, threshold, etc.).
    #[error("invalid design config: {0}")]
    Config(String),
    /// Numerical failure (singular Gram, non-finite score).
    #[error("design numerical failure: {0}")]
    Numerical(String),
    /// Probability / stats layer failure.
    #[error("probability error: {0}")]
    Prob(String),
    /// No action of a decision problem meets its chance threshold, so there is no
    /// decision to value.
    #[error("no admissible action: {0}")]
    NoAdmissibleAction(String),
    /// In-process utility / constraint callback failed.
    #[error("callback {name}: {message}")]
    Callback {
        /// Callback name (`utility` or a constraint name).
        name: String,
        /// Failure detail.
        message: String,
    },
}
