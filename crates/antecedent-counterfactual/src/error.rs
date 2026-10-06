//! Counterfactual evaluation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_model::ModelError;
use thiserror::Error;

/// Counterfactual errors.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum CounterfactualError {
    /// Model / shape issue.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// Missing factual values required for abduction.
    #[error("missing factual: {message}")]
    MissingFactual {
        /// Variable.
        message: String,
    },
    /// Nested interventions not allowed.
    #[error("nested counterfactuals not enabled")]
    NestedNotAllowed,
    /// Abduction is ill-posed or not exact inversion: the observed values do not
    /// determine each unit's exogenous terms (a deterministic mechanism given
    /// varying data, or a posterior or prior-drawn exogenous term). This is not a
    /// model-adequacy test: an invertible mechanism always regenerates the data it
    /// abduces from.
    #[error("abduction is not exact inversion: {message}")]
    AbductionNotExact {
        /// What failed to invert.
        message: String,
    },
    /// Cooperative cancellation was observed between evaluation steps.
    #[error("cancelled during cross-world evaluation")]
    Cancelled,
    /// Numerical failure.
    #[error("numerical error: {message}")]
    Numerical {
        /// Context.
        message: String,
    },
}

impl CounterfactualError {
    /// Ad-hoc model message.
    #[must_use]
    pub fn model_msg(message: impl Into<String>) -> Self {
        Self::Model(ModelError::Unsupported { message: message.into() })
    }
}
