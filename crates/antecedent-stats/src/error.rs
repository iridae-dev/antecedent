//! Stats-layer errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

/// Statistical / linear algebra errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StatsError {
    /// Shape mismatch.
    Shape {
        /// Context.
        message: &'static str,
    },
    /// Rank deficiency / singular design.
    RankDeficient {
        /// Detected rank.
        rank: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Backend failure.
    Backend(String),
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { message } => write!(f, "shape error: {message}"),
            Self::RankDeficient { rank, ncols } => {
                write!(f, "rank deficient: rank={rank} ncols={ncols}")
            }
            Self::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for StatsError {}
