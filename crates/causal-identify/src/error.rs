//! Identification errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

use causal_core::VariableId;

/// Identification failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IdentificationError {
    /// Treatment/outcome missing from graph.
    UnknownVariable {
        /// Variable.
        id: VariableId,
    },
    /// Query type not supported.
    UnsupportedQuery {
        /// Explanation.
        message: &'static str,
    },
    /// No adjustment set exists / not identified.
    NotIdentified {
        /// Explanation.
        message: &'static str,
    },
    /// Result limit exceeded during enumeration.
    ResultLimitExceeded {
        /// Configured limit.
        limit: usize,
    },
    /// Graph error.
    Graph(String),
}

impl fmt::Display for IdentificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVariable { id } => write!(f, "unknown variable {id}"),
            Self::UnsupportedQuery { message } => write!(f, "unsupported query: {message}"),
            Self::NotIdentified { message } => write!(f, "not identified: {message}"),
            Self::ResultLimitExceeded { limit } => {
                write!(f, "adjustment enumeration exceeded limit {limit}")
            }
            Self::Graph(msg) => write!(f, "graph error: {msg}"),
        }
    }
}

impl std::error::Error for IdentificationError {}
