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
    /// Target population not supported on this estimator path: a refusal with
    /// the registered `population_not_estimable` reason code.
    #[error(
        "{}{}: only TargetPopulation::AllObserved is supported on this estimator path",
        antecedent_core::reason_code::PREFIX,
        antecedent_core::reason_code!("population_not_estimable")
    )]
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
    /// A banked posterior's coefficient count does not match the target design's, so
    /// its coefficients cannot be mapped one-to-one onto a prior. Callers that treat
    /// "not this mechanism's prior" as a fallback match this variant, never its text;
    /// the message still carries the registered `prior_dimension_mismatch` reason so a
    /// caller with no fallback (the prior transfer is terminal) surfaces a coded refusal.
    #[error(
        "{}prior_dimension_mismatch: posterior coefficient dimension {posterior} != expected n_coef {design}",
        antecedent_core::reason_code::PREFIX
    )]
    PriorDimensionMismatch {
        /// Coefficients in the banked posterior.
        posterior: usize,
        /// Coefficients in the target design.
        design: usize,
    },
    /// A resampled empirical law received no rows, so it has no defined table. Bootstrap
    /// loops count this as a failed replicate; anywhere else it is an ordinary error.
    #[error("{message}")]
    EmptyEmpiricalSample {
        /// Located description of the empty sample.
        message: String,
    },
    /// A refusal carrying a registered runtime reason code
    /// (`parity/reason_codes.toml`), rendered `reason=<code>: <message>` so every
    /// boundary reads the code the same way.
    #[error("{}{code}: {message}", antecedent_core::reason_code::PREFIX)]
    Refused {
        /// Registered reason code (checked with `antecedent_core::reason_code!`).
        code: &'static str,
        /// What was refused and what the caller can do instead.
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

    /// A refusal with a registered runtime reason code.
    #[must_use]
    pub fn refused(code: &'static str, message: impl Into<String>) -> Self {
        Self::Refused { code, message: message.into() }
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
