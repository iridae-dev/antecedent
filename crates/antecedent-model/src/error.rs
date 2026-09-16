//! Model-layer errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::DataError;
use antecedent_graph::GraphError;
use antecedent_stats::StatsError;
use thiserror::Error;

/// Errors from compiling, fitting, or sampling causal models.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
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
    /// An iterative mechanism fit ran out of iterations short of its tolerance.
    ///
    /// Kept apart from [`Self::Numerical`] because it is the one fit failure a
    /// caller can act on: the remedy is a named one (rescale the parent columns,
    /// or drop the offending mechanism family), and the facade turns this into a
    /// reason-coded refusal rather than surfacing a raw deviance.
    #[error("mechanism fit did not converge: {message}")]
    NotConverged {
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
