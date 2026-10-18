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
    /// A cooperative cancellation poll reported cancellation, so the fit stopped
    /// before producing any mechanism. Only the `*_polled` fitting entry points
    /// return it; the unpolled ones never poll.
    #[error("mechanism fit cancelled")]
    Cancelled,
    /// A planned allocation exceeds the execution context's hard memory limit.
    #[error("resource limit: {message} needs {required_bytes} bytes, limit {limit_bytes}")]
    ResourceLimit {
        /// What was being planned.
        message: &'static str,
        /// Bytes the plan needs (`u64::MAX` when the size overflows).
        required_bytes: u64,
        /// The hard limit in force.
        limit_bytes: u64,
    },
    /// A candidate set, per-node constraint or validation-split declaration is
    /// inconsistent with the model or the data (an unknown variable, a family of
    /// the wrong outcome type, an empty candidate set, clustered splits without
    /// units). Raised before any family is fit.
    #[error("invalid mechanism selection: {message}")]
    InvalidSelection {
        /// What is inconsistent.
        message: String,
    },
    /// The family a policy or a per-node constraint requires could not be scored
    /// or fit on its node; the message carries the family's own failure.
    #[error("required mechanism family unavailable: {message}")]
    RequiredFamilyUnavailable {
        /// Node, family and the failure reason.
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
