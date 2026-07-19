//! Dense ids and edge endpoints.
//!
//! [`NodeRef`] lives in `causal-core` so sample planning can use it without
//! depending on this crate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use causal_core::NodeRef;

use crate::error::GraphError;

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

    /// Fallible index → dense id.
    pub fn try_from_usize(i: usize) -> Result<Self, GraphError> {
        let raw = u32::try_from(i).map_err(|_| GraphError::TooManyNodes)?;
        Ok(Self::from_raw(raw))
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
    /// Circle (PAG; not used in DAG constructors).
    Circle,
    /// Conflict (`x` in pinned baseline). Orientation rules proposed incompatible marks.
    ///
    /// CPDAG / PCMCI+ contemporaneous conflicts use Conflict–Conflict (`x-x`).
    /// PAG / LPCMCI may also use asymmetric forms (`x→`, `←x`).
    Conflict,
}

/// LPCMCI middle mark on an edge (Gerhardus & Runge 2020).
///
/// Intermediate search state; a converged PAG has only [`MiddleMark::Empty`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum MiddleMark {
    /// Unknown (`?`) — no adjacency claim yet.
    Unknown,
    /// Left (`L`) — search among parents of the later/ordered endpoint exhausted.
    Left,
    /// Right (`R`) — search among parents of the earlier/ordered endpoint exhausted.
    Right,
    /// Both (`!`) — `Left` and `Right` both hold.
    Both,
    /// Empty (`-`) — definite adjacency in the MAG / final PAG.
    #[default]
    Empty,
}

impl MiddleMark {
    /// Whether this is a definite adjacency mark (empty middle).
    #[must_use]
    pub const fn is_definite(self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Merge an existing middle mark with an update (pinned baseline `_apply_middle_mark`).
    #[must_use]
    pub const fn apply(self, update: Self) -> Self {
        use MiddleMark::{Both, Empty, Left, Right, Unknown};
        match (self, update) {
            (Empty, _) | (_, Empty) => Empty,
            (Both, _) | (_, Both) => Both,
            (Left, Right) | (Right, Left) => Both,
            (Unknown, x) => x,
            (x, Unknown) => x,
            (Left, Left) => Left,
            (Right, Right) => Right,
        }
    }
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
    /// LPCMCI middle mark (default [`MiddleMark::Empty`] outside LPCMCI search).
    pub middle: MiddleMark,
}

impl MarkedEdge {
    /// Directed edge `from -> to` (tail at from, arrow at to).
    #[must_use]
    pub const fn directed(from: DenseNodeId, to: DenseNodeId) -> Self {
        Self {
            a: from,
            b: to,
            at_a: Endpoint::Tail,
            at_b: Endpoint::Arrow,
            middle: MiddleMark::Empty,
        }
    }

    /// Undirected edge `a — b` (tail–tail). Canonicalizes so `a.raw() <= b.raw()`.
    #[must_use]
    pub fn undirected(a: DenseNodeId, b: DenseNodeId) -> Self {
        if a.raw() <= b.raw() {
            Self {
                a,
                b,
                at_a: Endpoint::Tail,
                at_b: Endpoint::Tail,
                middle: MiddleMark::Empty,
            }
        } else {
            Self {
                a: b,
                b: a,
                at_a: Endpoint::Tail,
                at_b: Endpoint::Tail,
                middle: MiddleMark::Empty,
            }
        }
    }

    /// Whether this is a DAG-legal directed edge.
    #[must_use]
    pub const fn is_dag_directed(self) -> bool {
        matches!(
            (self.at_a, self.at_b),
            (Endpoint::Tail, Endpoint::Arrow) | (Endpoint::Arrow, Endpoint::Tail)
        )
    }

    /// Whether this is an undirected CPDAG edge (tail–tail).
    #[must_use]
    pub const fn is_undirected(self) -> bool {
        matches!((self.at_a, self.at_b), (Endpoint::Tail, Endpoint::Tail))
    }

    /// Whether this is a bidirected ADMG edge (arrow–arrow).
    #[must_use]
    pub const fn is_bidirected(self) -> bool {
        matches!((self.at_a, self.at_b), (Endpoint::Arrow, Endpoint::Arrow))
    }

    /// Whether this is a conflict edge (`x-x`, both endpoints [`Endpoint::Conflict`]).
    #[must_use]
    pub const fn is_conflict(self) -> bool {
        matches!((self.at_a, self.at_b), (Endpoint::Conflict, Endpoint::Conflict))
    }

    /// Bidirected edge `a ↔ b`. Canonicalizes so `a.raw() <= b.raw()`.
    #[must_use]
    pub fn bidirected(a: DenseNodeId, b: DenseNodeId) -> Self {
        if a.raw() <= b.raw() {
            Self {
                a,
                b,
                at_a: Endpoint::Arrow,
                at_b: Endpoint::Arrow,
                middle: MiddleMark::Empty,
            }
        } else {
            Self {
                a: b,
                b: a,
                at_a: Endpoint::Arrow,
                at_b: Endpoint::Arrow,
                middle: MiddleMark::Empty,
            }
        }
    }

    /// Conflict edge `a x-x b` (pinned baseline). Canonicalizes so `a.raw() <= b.raw()`.
    #[must_use]
    pub fn conflict(a: DenseNodeId, b: DenseNodeId) -> Self {
        if a.raw() <= b.raw() {
            Self {
                a,
                b,
                at_a: Endpoint::Conflict,
                at_b: Endpoint::Conflict,
                middle: MiddleMark::Empty,
            }
        } else {
            Self {
                a: b,
                b: a,
                at_a: Endpoint::Conflict,
                at_b: Endpoint::Conflict,
                middle: MiddleMark::Empty,
            }
        }
    }

    /// Same endpoints with a different middle mark.
    #[must_use]
    pub const fn with_middle(mut self, middle: MiddleMark) -> Self {
        self.middle = middle;
        self
    }

    /// Whether marks are legal for a CPDAG (directed, undirected, or `x-x`; no Circle).
    #[must_use]
    pub const fn is_cpdag_legal(self) -> bool {
        matches!(
            (self.at_a, self.at_b),
            (Endpoint::Tail, Endpoint::Arrow)
                | (Endpoint::Arrow, Endpoint::Tail)
                | (Endpoint::Tail, Endpoint::Tail)
                | (Endpoint::Conflict, Endpoint::Conflict)
        )
    }

    /// Whether marks are legal for an ADMG (directed or bidirected; no Circle/Conflict).
    #[must_use]
    pub const fn is_admg_legal(self) -> bool {
        matches!(
            (self.at_a, self.at_b),
            (Endpoint::Tail, Endpoint::Arrow)
                | (Endpoint::Arrow, Endpoint::Tail)
                | (Endpoint::Arrow, Endpoint::Arrow)
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
