//! Stats-layer errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

/// Statistical / linear algebra errors.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum StatsError {
    /// Shape mismatch.
    #[error("shape error: {message}")]
    Shape {
        /// Context.
        message: &'static str,
    },
    /// Rank deficiency / singular design.
    #[error("rank deficient: rank={rank} ncols={ncols}")]
    RankDeficient {
        /// Detected rank.
        rank: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Materially non-positive variance after inclusion–exclusion (not FP noise).
    #[error("non-positive variance: {message}")]
    NonPositiveVariance {
        /// Context.
        message: &'static str,
    },
    /// Backend failure.
    #[error("backend error: {0}")]
    Backend(String),
}
