//! Learner-layer errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

use crate::learner::{LearnerCapabilities, PredictionTask};

/// Prediction / learner errors.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum LearnError {
    /// Worker or scratch admission stopped the learner.
    #[error(transparent)]
    Admission(antecedent_core::execution::AdmissionError),
    /// Shape mismatch between design, target, weights, or output.
    #[error("shape error: {message}")]
    Shape {
        /// Context.
        message: &'static str,
    },
    /// A prediction, target or weight the computation needs to be finite is not.
    #[error("non-finite value: {message}")]
    NonFinite {
        /// Context.
        message: &'static str,
    },
    /// Requested prediction task is not what this factory implements.
    #[error("task mismatch: requested {requested:?}, supported {supported:?}")]
    TaskMismatch {
        /// Task the caller asked for.
        requested: PredictionTask,
        /// Task this factory implements.
        supported: PredictionTask,
    },
    /// Public spec has no resolved provider yet.
    #[error("provider unavailable for {spec}")]
    ProviderUnavailable {
        /// Public [`crate::LearnerSpec`] name.
        spec: &'static str,
        /// Capabilities the eventual provider must satisfy.
        required: LearnerCapabilities,
    },
    /// The cross-fit stopped because its work was cancelled; no fold result is reported.
    #[error("cross-fit stopped because the work was cancelled")]
    Cancelled,
    /// A declared count or size is above the limit the learner admits. The request is
    /// refused before any allocation or fit; nothing is shortened to fit.
    #[error("resource limit: {message}")]
    ResourceLimit {
        /// Which limit the request passed.
        message: &'static str,
    },
    /// Sparse or layout combination this factory cannot consume.
    #[error("unsupported design: {message}")]
    Unsupported {
        /// Context.
        message: &'static str,
    },
    /// Statistical backend failure.
    #[error(transparent)]
    Stats(#[from] antecedent_stats::StatsError),
    /// Probability backend failure.
    #[error(transparent)]
    Probability(#[from] antecedent_prob::ProbError),
    /// Kernel view construction failure.
    #[error("view error: {0}")]
    View(String),
    /// Provider backend failure.
    #[error("backend error: {0}")]
    Backend(String),
}

impl LearnError {
    /// Whether the learner refused a request that passed one of its declared limits, so the
    /// refusal belongs to the `resource` family: the request is too large, not unanswerable.
    #[must_use]
    pub fn is_resource_limit(&self) -> bool {
        match self {
            Self::ResourceLimit { .. } => true,
            Self::Admission(error) => {
                !matches!(error, antecedent_core::execution::AdmissionError::Cancelled)
            }
            _ => false,
        }
    }
}
