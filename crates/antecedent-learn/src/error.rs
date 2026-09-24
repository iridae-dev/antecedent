//! Learner-layer errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

use crate::learner::{LearnerCapabilities, PredictionTask};

/// Prediction / learner errors.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum LearnError {
    /// Shape mismatch between design, target, weights, or output.
    #[error("shape error: {message}")]
    Shape {
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
