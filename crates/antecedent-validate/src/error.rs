//! Validation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::DataError;
use antecedent_discovery::DiscoveryError;
use antecedent_estimate::EstimationError;
use antecedent_prob::ProbError;
use antecedent_stats::StatsError;
use thiserror::Error;

/// Validation / refutation failures.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ValidationError {
    /// Data transformation failed.
    #[error(transparent)]
    Data(#[from] DataError),
    /// Estimation failed inside a refuter.
    #[error(transparent)]
    Estimation(#[from] EstimationError),
    /// Stats backend failure.
    #[error(transparent)]
    Stats(#[from] StatsError),
    /// Probability / posterior backend.
    #[error(transparent)]
    Prob(#[from] ProbError),
    /// Discovery failure inside a stability check.
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    /// Refuter not applicable to the problem.
    #[error("{message}")]
    NotApplicable {
        /// Reason.
        message: &'static str,
    },
}

impl ValidationError {
    /// Ad-hoc data-layer message.
    #[must_use]
    pub fn data_msg(message: impl Into<String>) -> Self {
        Self::Data(DataError::InvalidArgument { message: message.into() })
    }

    /// Ad-hoc estimation message.
    #[must_use]
    pub fn estimation_msg(message: impl Into<String>) -> Self {
        Self::Estimation(EstimationError::stats_msg(message))
    }
}
