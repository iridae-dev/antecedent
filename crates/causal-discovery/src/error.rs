//! Discovery errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_data::DataError;
use causal_graph::GraphError;
use causal_stats::StatsError;
use thiserror::Error;

/// Discovery failures.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum DiscoveryError {
    /// Data / sample preparation.
    #[error(transparent)]
    Data(#[from] DataError),
    /// Stats / CI failure.
    #[error(transparent)]
    Stats(#[from] StatsError),
    /// Graph mutation / validation failure.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Score / state cache failure.
    #[error(transparent)]
    State(#[from] causal_state::StateError),
    /// Probability / ensemble construction failure.
    #[error(transparent)]
    Prob(#[from] causal_prob::ProbError),
    /// Unsupported configuration.
    #[error("{message}")]
    Unsupported {
        /// Message.
        message: &'static str,
    },
    /// Orientation / rule-scheduling precondition or message.
    #[error("orientation: {0}")]
    Orientation(String),
    /// Resource / memory budget exceeded.
    #[error("resource: {0}")]
    Resource(String),
    /// Cooperative cancellation.
    #[error("cancelled")]
    Cancelled,
}

impl DiscoveryError {
    /// Fixed unsupported configuration.
    #[must_use]
    pub const fn unsupported(message: &'static str) -> Self {
        Self::Unsupported { message }
    }

    /// Ad-hoc data-layer message.
    #[must_use]
    pub fn data_msg(message: impl Into<String>) -> Self {
        Self::Data(DataError::InvalidArgument { message: message.into() })
    }

    /// Ad-hoc stats-layer message (CI / numerical only — not orientation).
    #[must_use]
    pub fn stats_msg(message: impl Into<String>) -> Self {
        Self::Stats(StatsError::Backend(message.into()))
    }
}
