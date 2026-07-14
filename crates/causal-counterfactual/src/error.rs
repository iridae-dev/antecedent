//! Counterfactual evaluation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

/// Counterfactual errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CounterfactualError {
    /// Model / shape issue.
    Model(String),
    /// Missing factual values required for abduction.
    MissingFactual {
        /// Variable.
        message: String,
    },
    /// Nested interventions not allowed.
    NestedNotAllowed,
    /// Numerical failure.
    Numerical {
        /// Context.
        message: String,
    },
}

impl fmt::Display for CounterfactualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(m) => write!(f, "counterfactual model error: {m}"),
            Self::MissingFactual { message } => write!(f, "missing factual: {message}"),
            Self::NestedNotAllowed => write!(f, "nested counterfactuals not enabled"),
            Self::Numerical { message } => write!(f, "numerical error: {message}"),
        }
    }
}

impl std::error::Error for CounterfactualError {}

impl From<causal_model::ModelError> for CounterfactualError {
    fn from(e: causal_model::ModelError) -> Self {
        Self::Model(e.to_string())
    }
}
