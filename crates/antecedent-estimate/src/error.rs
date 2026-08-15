//! Estimation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::QueryError;
use antecedent_data::DataError;
use antecedent_prob::ProbError;
use antecedent_stats::StatsError;
use thiserror::Error;

/// Estimation failures.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
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
    /// Query validation failed (`AverageEffectQuery::validate`, …).
    #[error(transparent)]
    Query(#[from] QueryError),
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
    /// Effect modifiers not supported on this estimator path.
    #[error("effect modifiers are not supported on this estimator path")]
    EffectModifiers,
    /// Target population not supported on this estimator path.
    #[error("only TargetPopulation::AllObserved is supported on this estimator path")]
    TargetPopulation,
    /// Query options unsupported by this estimator (fixed message).
    #[error("{message}")]
    Unsupported {
        /// Explanation.
        message: &'static str,
    },
    /// The estimator refused because identification was not certified for this query.
    ///
    /// Distinct from [`Self::Data`]: the inputs are well formed, and the refusal is about
    /// what the graph licenses. Owned rather than `&'static str` because it carries the
    /// certificate's own reason and message.
    #[error("{message}")]
    NotCertified {
        /// Refusal, including the certificate's reason and message.
        message: String,
    },
}

impl EstimationError {
    /// Fixed unsupported query option.
    #[must_use]
    pub const fn unsupported(message: &'static str) -> Self {
        Self::Unsupported { message }
    }

    /// Refusal to estimate a quantity whose identification was not certified.
    #[must_use]
    pub fn not_certified(stage: &str, reason: &str, message: &str) -> Self {
        Self::NotCertified {
            message: format!(
                "{stage} refused: identification was not certified ({reason}): {message}"
            ),
        }
    }

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
