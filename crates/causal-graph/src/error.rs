//! Graph construction and validation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

use causal_core::{Lag, VariableId};

/// Graph-layer errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphError {
    /// Unknown dense node.
    UnknownNode {
        /// Dense id.
        id: u32,
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
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownNode { id } => write!(f, "unknown dense node {id}"),
            Self::Cycle { from, to } => write!(f, "edge {from}->{to} would create a cycle"),
            Self::InvalidEndpoints { message } => write!(f, "invalid endpoints: {message}"),
            Self::ContemporaneousSelfEdge { variable } => {
                write!(f, "contemporaneous self-edge on {variable}")
            }
            Self::DuplicateEdge { from, to } => write!(f, "duplicate edge {from}->{to}"),
            Self::InvalidLag { lag } => write!(f, "invalid lag {lag}"),
            Self::TooManyNodes => write!(f, "too many nodes"),
        }
    }
}

impl std::error::Error for GraphError {}
