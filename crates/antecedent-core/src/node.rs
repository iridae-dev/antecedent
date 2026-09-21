//! Stable node identity before dense graph indexing.
//!
//! Shared by sample planning (`antecedent-data`) and graph types (`antecedent-graph`).
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
    /// Slot of a finite unfolding: `variable` at a signed time `offset` from the analysis
    /// origin (`Lagged { lag }` is `Unfolded { offset: -lag }` restricted to the past).
    ///
    /// Unfolded graphs label their nodes with this so that a window slot is never mistaken for
    /// a schema variable (`Static`) by code generic over `nodes()`.
    Unfolded {
        /// Variable.
        variable: VariableId,
        /// Signed offset from the analysis origin (negative = history).
        offset: i32,
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
            Self::Static(v)
            | Self::Lagged { variable: v, .. }
            | Self::Unfolded { variable: v, .. }
            | Self::Context { variable: v, .. } => v,
        }
    }

    /// Lag if this is a lagged node; `None` for static/context.
    #[must_use]
    pub const fn lag(self) -> Option<Lag> {
        match self {
            Self::Lagged { lag, .. } => Some(lag),
            Self::Static(_) | Self::Unfolded { .. } | Self::Context { .. } => None,
        }
    }

    /// Whether this is a node of a static (non-temporal) graph: a schema variable or a slot of a
    /// finite unfolding.
    #[must_use]
    pub const fn is_static_graph_node(self) -> bool {
        matches!(self, Self::Static(_) | Self::Unfolded { .. })
    }
}
