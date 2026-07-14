//! State-crate errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

/// Errors from incremental causal state.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum StateError {
    /// Unknown registry id.
    #[error("unknown state id: {0}")]
    UnknownId(String),
    /// Shape / length mismatch.
    #[error("state shape error: {0}")]
    Shape(String),
    /// Cache budget refused an insert.
    #[error("cache budget exceeded (need {need} bytes, remaining {remaining})")]
    CacheBudget {
        /// Requested bytes.
        need: u64,
        /// Remaining budget.
        remaining: u64,
    },
    /// Numerical failure in sufficient statistics.
    #[error("state numerical failure: {0}")]
    Numerical(String),
    /// Invalid event / configuration.
    #[error("invalid state event: {0}")]
    InvalidEvent(String),
}
