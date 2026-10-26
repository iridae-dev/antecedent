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
#[non_exhaustive]
pub enum ValidationError {
    /// Execution was cancelled before a complete validation report was available.
    #[error("validation cancelled")]
    Cancelled,
    /// Execution could not fund validation's required work.
    #[error("validation resource refusal: {message}")]
    Resource {
        /// The refused resource demand.
        message: String,
    },
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
    /// A refuter refused a result that is not a stats-backend failure.
    ///
    /// Display stays `backend error: …`, the text [`Self::estimation_msg`] used to
    /// produce by packing the message into [`EstimationError`].
    #[error("backend error: {message}")]
    Refused {
        /// What was refused.
        message: String,
    },
}

impl ValidationError {
    /// Ad-hoc data-layer message.
    #[must_use]
    pub fn data_msg(message: impl Into<String>) -> Self {
        Self::Data(DataError::InvalidArgument { message: message.into() })
    }

    /// A validation refusal whose text is `backend error: {message}`.
    ///
    /// This is not an [`EstimationError`]: the stats layer did not fail. Callers
    /// that matched [`Self::Estimation`] for these messages should match [`Self::Refused`].
    #[must_use]
    pub fn estimation_msg(message: impl Into<String>) -> Self {
        Self::Refused { message: message.into() }
    }
}
