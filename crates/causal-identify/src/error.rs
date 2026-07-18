//! Identification errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::VariableId;
use causal_graph::GraphError;
use thiserror::Error;

/// Identification failures.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum IdentificationError {
    /// Treatment/outcome missing from graph.
    #[error("unknown variable {id}")]
    UnknownVariable {
        /// Variable.
        id: VariableId,
    },
    /// Query type not supported.
    #[error("unsupported query: {message}")]
    UnsupportedQuery {
        /// Explanation.
        message: &'static str,
    },
    /// Temporal backdoor is Pulse-only; Sustained needs g-formula / sequential ID.
    #[error(
        "temporal backdoor identification supports Pulse policies only; \
         sustained interventions require sequential (g-formula) identification"
    )]
    SustainedPolicyUnsupported,
    /// No adjustment set exists / not identified.
    #[error("not identified: {message}")]
    NotIdentified {
        /// Explanation.
        message: &'static str,
    },
    /// Result limit exceeded during enumeration.
    #[error("adjustment enumeration exceeded limit {limit}")]
    ResultLimitExceeded {
        /// Configured limit.
        limit: usize,
    },
    /// Graph error.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Index / configuration message that is not a raw [`GraphError`].
    #[error("{0}")]
    Message(String),
}

impl IdentificationError {
    /// Ad-hoc message helper.
    #[must_use]
    pub fn msg(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }

    /// Fixed unsupported query.
    #[must_use]
    pub const fn unsupported(message: &'static str) -> Self {
        Self::UnsupportedQuery { message }
    }
}
