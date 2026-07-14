//! Model-layer errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_data::DataError;
use causal_graph::GraphError;
use causal_stats::StatsError;
use thiserror::Error;

/// Errors from compiling, fitting, or sampling causal models.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum ModelError {
    /// Graph / shape inconsistency.
    #[error("model shape error: {message}")]
    Shape {
        /// Context.
        message: String,
    },
    /// Graph is cyclic or has no topological order.
    #[error("not a DAG: {message}")]
    NotDag {
        /// Context.
        message: String,
    },
    /// Mechanism missing for a node.
    #[error("missing mechanism for node {node}")]
    MissingMechanism {
        /// Dense node index.
        node: u32,
    },
    /// Unsupported intervention or mechanism family.
    #[error("unsupported: {message}")]
    Unsupported {
        /// Context.
        message: String,
    },
    /// Numerical failure.
    #[error("numerical error: {message}")]
    Numerical {
        /// Context.
        message: String,
    },
    /// Graph error passthrough.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Data error passthrough.
    #[error(transparent)]
    Data(#[from] DataError),
    /// Stats error passthrough.
    #[error(transparent)]
    Stats(#[from] StatsError),
}
