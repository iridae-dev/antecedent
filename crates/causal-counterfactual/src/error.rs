//! Counterfactual evaluation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_model::ModelError;
use thiserror::Error;

/// Counterfactual errors.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
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
        Self::Model(ModelError::Unsupported {
            message: message.into(),
        })
    }
}
