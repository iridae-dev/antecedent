//! Graph construction and validation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

use antecedent_core::{Lag, VariableId};

/// Graph-layer errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphError {
    /// Unknown dense node.
    UnknownNode {
        /// Dense id.
        id: u32,
    },
    /// Unknown variable name at an API boundary.
    UnknownVariableName {
        /// Requested name.
        name: String,
    },
    /// Edge would introduce a directed cycle.
    Cycle {
        /// Source dense id.
        from: u32,
        /// Target dense id.
        to: u32,
    },
    /// Invalid endpoint combination for this graph class.
    InvalidEndpoints {
        /// Explanation.
        message: &'static str,
    },
    /// Contemporaneous self-edge is invalid.
    ContemporaneousSelfEdge {
        /// Variable.
        variable: VariableId,
    },
    /// Duplicate edge.
    DuplicateEdge {
        /// From.
        from: u32,
        /// To.
        to: u32,
    },
    /// Lagged self-edge with lag 0.
    InvalidLag {
        /// Lag value.
        lag: Lag,
    },
    /// Node capacity exceeded.
    TooManyNodes,
    /// Bounded path search hit `max_paths` or `max_len` before exploring all candidates.
    ///
    /// Returned when m-separation would otherwise conclude "separated" after an incomplete
    /// search (an unexplored active path may still exist). Finding an active path remains
    /// conclusive even under truncation.
    SearchBudgetExhausted {
        /// Path-count budget.
        max_paths: usize,
        /// Path-length budget.
        max_len: usize,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNode { id } => write!(f, "unknown dense node {id}"),
            Self::UnknownVariableName { name } => write!(f, "unknown variable name '{name}'"),
            Self::Cycle { from, to } => write!(f, "edge {from}->{to} would create a cycle"),
            Self::InvalidEndpoints { message } => write!(f, "invalid endpoints: {message}"),
            Self::ContemporaneousSelfEdge { variable } => {
                write!(f, "contemporaneous self-edge on {variable}")
            }
            Self::DuplicateEdge { from, to } => write!(f, "duplicate edge {from}->{to}"),
            Self::InvalidLag { lag } => write!(f, "invalid lag {lag}"),
            Self::TooManyNodes => write!(f, "too many nodes"),
            Self::SearchBudgetExhausted { max_paths, max_len } => {
                write!(f, "path search budget exhausted (max_paths={max_paths}, max_len={max_len})")
            }
        }
    }
}

impl std::error::Error for GraphError {}
