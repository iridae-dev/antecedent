//! Node references, dense ids, and edge endpoints (DESIGN.md §6).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{EnvironmentId, Lag, VariableId};

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

/// Compact dense node index used in algorithmic paths.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DenseNodeId(u32);

impl DenseNodeId {
    /// From raw index.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw index.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// As usize.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

/// Edge endpoint mark.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Endpoint {
    /// Tail (undirected / directed origin).
    Tail,
    /// Arrow head.
    Arrow,
    /// Circle (PAG; not used in Phase 0 DAG constructors).
    Circle,
}

/// Directed marked edge between dense nodes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct MarkedEdge {
    /// Endpoint A node.
    pub a: DenseNodeId,
    /// Endpoint B node.
    pub b: DenseNodeId,
    /// Mark at A.
    pub at_a: Endpoint,
    /// Mark at B.
    pub at_b: Endpoint,
}

impl MarkedEdge {
    /// Directed edge `from -> to` (tail at from, arrow at to).
    #[must_use]
    pub const fn directed(from: DenseNodeId, to: DenseNodeId) -> Self {
        Self { a: from, b: to, at_a: Endpoint::Tail, at_b: Endpoint::Arrow }
    }

    /// Whether this is a DAG-legal directed edge.
    #[must_use]
    pub const fn is_dag_directed(self) -> bool {
        matches!(
            (self.at_a, self.at_b),
            (Endpoint::Tail, Endpoint::Arrow) | (Endpoint::Arrow, Endpoint::Tail)
        )
    }

    /// Oriented parent -> child for a DAG directed edge.
    #[must_use]
    pub fn parent_child(self) -> Option<(DenseNodeId, DenseNodeId)> {
        match (self.at_a, self.at_b) {
            (Endpoint::Tail, Endpoint::Arrow) => Some((self.a, self.b)),
            (Endpoint::Arrow, Endpoint::Tail) => Some((self.b, self.a)),
            _ => None,
        }
    }
}
