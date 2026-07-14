//! Model-layer errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

/// Errors from compiling, fitting, or sampling causal models.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelError {
    /// Graph / shape inconsistency.
    Shape {
        /// Context.
        message: String,
    },
    /// Graph is cyclic or has no topological order.
    NotDag {
        /// Context.
        message: String,
    },
    /// Mechanism missing for a node.
    MissingMechanism {
        /// Dense node index.
        node: u32,
    },
    /// Unsupported intervention or mechanism family.
    Unsupported {
        /// Context.
        message: String,
    },
    /// Numerical failure.
    Numerical {
        /// Context.
        message: String,
    },
    /// Graph error passthrough.
    Graph(String),
    /// Data error passthrough.
    Data(String),
    /// Stats error passthrough.
    Stats(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape { message } => write!(f, "model shape error: {message}"),
            Self::NotDag { message } => write!(f, "not a DAG: {message}"),
            Self::MissingMechanism { node } => write!(f, "missing mechanism for node {node}"),
            Self::Unsupported { message } => write!(f, "unsupported: {message}"),
            Self::Numerical { message } => write!(f, "numerical error: {message}"),
            Self::Graph(msg) => write!(f, "graph error: {msg}"),
            Self::Data(msg) => write!(f, "data error: {msg}"),
            Self::Stats(msg) => write!(f, "stats error: {msg}"),
        }
    }
}

impl std::error::Error for ModelError {}
