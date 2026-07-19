//! Stable node identity before dense graph indexing.
//!
//! Shared by sample planning (`causal-data`) and graph types (`causal-graph`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ids::{EnvironmentId, Lag, VariableId};

/// Stable node identity before dense indexing.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum NodeRef {
    /// Static graph node.
    Static(VariableId),
    /// Lagged temporal node (`variable` at `t - lag`).
    Lagged {
        /// Variable.
        variable: VariableId,
        /// Non-negative lag (`0` = contemporaneous).
        lag: Lag,
    },
    /// Context-aware node.
    Context {
        /// Variable.
        variable: VariableId,
        /// Optional environment.
        environment: Option<EnvironmentId>,
    },
}

impl NodeRef {
    /// Variable id carried by this node reference.
    #[must_use]
    pub const fn variable(self) -> VariableId {
        match self {
            Self::Static(v) | Self::Lagged { variable: v, .. } | Self::Context { variable: v, .. } => {
                v
            }
        }
    }

    /// Lag if this is a lagged node; `None` for static/context.
    #[must_use]
    pub const fn lag(self) -> Option<Lag> {
        match self {
            Self::Lagged { lag, .. } => Some(lag),
            Self::Static(_) | Self::Context { .. } => None,
        }
    }
}
