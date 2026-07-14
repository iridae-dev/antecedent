//! Estimation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_data::DataError;
use causal_prob::ProbError;
use causal_stats::StatsError;
use thiserror::Error;

/// Estimation failures.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum EstimationError {
    /// Data/schema issue.
    #[error(transparent)]
    Data(#[from] DataError),
    /// Stats backend.
    #[error(transparent)]
    Stats(#[from] StatsError),
    /// Probability / posterior backend.
    #[error(transparent)]
    Prob(#[from] ProbError),
    /// Missing overlap override when required.
    #[error("{message}")]
    Overlap {
        /// Message.
        message: &'static str,
    },
    /// Incompatible estimand.
    #[error("{message}")]
    IncompatibleEstimand {
        /// Message.
        message: &'static str,
    },
    /// Query options unsupported by this estimator.
    #[error("{0}")]
    UnsupportedQuery(String),
}

impl EstimationError {
    /// Ad-hoc data-layer message (maps to [`DataError::InvalidArgument`]).
    #[must_use]
    pub fn data_msg(message: impl Into<String>) -> Self {
        Self::Data(DataError::InvalidArgument { message: message.into() })
    }

    /// Ad-hoc stats-layer message (maps to [`StatsError::Backend`]).
    #[must_use]
    pub fn stats_msg(message: impl Into<String>) -> Self {
        Self::Stats(StatsError::Backend(message.into()))
    }
}
