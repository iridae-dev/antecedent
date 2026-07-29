//! Graph construction and validation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Lag, VariableId};
use thiserror::Error;

/// Graph-layer errors.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum GraphError {
    /// Unknown dense node.
    #[error("unknown dense node {id}")]
    UnknownNode {
        /// Dense id.
        id: u32,
    },
    /// Unknown variable name at an API boundary.
    #[error("unknown variable name '{name}'")]
    UnknownVariableName {
        /// Requested name.
        name: String,
    },
    /// Edge would introduce a directed cycle.
    #[error("edge {from}->{to} would create a cycle")]
    Cycle {
        /// Source dense id.
        from: u32,
        /// Target dense id.
        to: u32,
    },
    /// Invalid endpoint combination for this graph class.
    #[error("invalid endpoints: {message}")]
    InvalidEndpoints {
        /// Explanation.
        message: &'static str,
    },
    /// Contemporaneous self-edge is invalid.
    #[error("contemporaneous self-edge on {variable}")]
    ContemporaneousSelfEdge {
        /// Variable.
        variable: VariableId,
    },
    /// Duplicate edge.
    #[error("duplicate edge {from}->{to}")]
    DuplicateEdge {
        /// From.
        from: u32,
        /// To.
        to: u32,
    },
    /// Lagged self-edge with lag 0.
    #[error("invalid lag {lag}")]
    InvalidLag {
        /// Lag value.
        lag: Lag,
    },
    /// Edge points from the future into the past (source lag nearer the present
    /// than target lag).
    #[error("edge {from}->{to} points from the future ({from_lag}) into the past ({to_lag})")]
    FutureToPast {
        /// Source dense id.
        from: u32,
        /// Target dense id.
        to: u32,
        /// Source lag.
        from_lag: Lag,
        /// Target lag.
        to_lag: Lag,
    },
    /// Node capacity exceeded.
    #[error("too many nodes")]
    TooManyNodes,
    /// Bounded path search hit `max_paths` or `max_len` before exploring all candidates.
    ///
    /// Returned when m-separation would otherwise conclude "separated" after an incomplete
    /// search (an unexplored active path may still exist). Finding an active path remains
    /// conclusive even under truncation.
    #[error("path search budget exhausted (max_paths={max_paths}, max_len={max_len})")]
    SearchBudgetExhausted {
        /// Path-count budget.
        max_paths: usize,
        /// Path-length budget.
        max_len: usize,
    },
}
