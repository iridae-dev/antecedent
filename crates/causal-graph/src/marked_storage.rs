//! Shared marked-adjacency helpers for CPDAG / PAG storage (DESIGN §6.3).
//!
//! Graph types remain distinct; only adjacency entry layout and directed
//! reachability scratch reuse are shared.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::GraphError;
use crate::types::{DenseNodeId, Endpoint, MarkedEdge, MiddleMark};
use crate::workspace::GraphWorkspace;

/// Adjacency entry: neighbor plus marks at self and at neighbor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct AdjEntry {
    pub(crate) neighbor: DenseNodeId,
    pub(crate) at_self: Endpoint,
    pub(crate) at_neighbor: Endpoint,
    pub(crate) middle: MiddleMark,
}

impl AdjEntry {
    #[inline]
    pub(crate) const fn new(
        neighbor: DenseNodeId,
        at_self: Endpoint,
        at_neighbor: Endpoint,
        middle: MiddleMark,
    ) -> Self {
        Self { neighbor, at_self, at_neighbor, middle }
    }

    #[inline]
    pub(crate) const fn is_directed_out(self) -> bool {
        matches!((self.at_self, self.at_neighbor), (Endpoint::Tail, Endpoint::Arrow))
    }
}

/// Push both halves of a marked edge into adjacency lists.
pub(crate) fn push_marked_pair(adj: &mut [Vec<AdjEntry>], edge: MarkedEdge) {
    adj[edge.a.as_usize()].push(AdjEntry::new(edge.b, edge.at_a, edge.at_b, edge.middle));
    adj[edge.b.as_usize()].push(AdjEntry::new(edge.a, edge.at_b, edge.at_a, edge.middle));
}

/// Marked edge between `a` and `b` if present.
#[must_use]
pub(crate) fn edge_between(
    adj: &[Vec<AdjEntry>],
    a: DenseNodeId,
    b: DenseNodeId,
) -> Option<MarkedEdge> {
    if a.as_usize() >= adj.len() || b.as_usize() >= adj.len() {
        return None;
    }
    for e in &adj[a.as_usize()] {
        if e.neighbor == b {
            return Some(MarkedEdge {
                a,
                b,
                at_a: e.at_self,
                at_b: e.at_neighbor,
                middle: e.middle,
            });
        }
    }
    None
}

/// Iterator over definite directed children (Tail→Arrow from `id`).
pub(crate) fn directed_children(
    adj: &[Vec<AdjEntry>],
    id: DenseNodeId,
) -> impl Iterator<Item = DenseNodeId> + '_ {
    adj.get(id.as_usize()).into_iter().flatten().filter(|e| e.is_directed_out()).map(|e| e.neighbor)
}

/// Whether `from` reaches `to` via definite directed edges, reusing `ws`.
#[must_use]
pub(crate) fn reaches_directed(
    adj: &[Vec<AdjEntry>],
    ws: &mut GraphWorkspace,
    from: DenseNodeId,
    to: DenseNodeId,
) -> bool {
    if from == to {
        return true;
    }
    if from.as_usize() >= adj.len() || to.as_usize() >= adj.len() {
        return false;
    }
    ws.prepare(adj.len());
    ws.frontier.push(from);
    ws.visited.insert(from);
    while let Some(u) = ws.frontier.pop() {
        for c in directed_children(adj, u) {
            if c == to {
                return true;
            }
            if !ws.visited.contains(c) {
                ws.visited.insert(c);
                ws.frontier.push(c);
            }
        }
    }
    false
}

/// Update endpoint marks on an existing edge (both adjacency halves); middle unchanged.
pub(crate) fn set_marks(
    adj: &mut [Vec<AdjEntry>],
    a: DenseNodeId,
    b: DenseNodeId,
    at_a: Endpoint,
    at_b: Endpoint,
) -> Result<(), GraphError> {
    let mut found = false;
    for e in &mut adj[a.as_usize()] {
        if e.neighbor == b {
            e.at_self = at_a;
            e.at_neighbor = at_b;
            found = true;
            break;
        }
    }
    if !found {
        return Err(GraphError::UnknownNode { id: a.raw() });
    }
    found = false;
    for e in &mut adj[b.as_usize()] {
        if e.neighbor == a {
            e.at_self = at_b;
            e.at_neighbor = at_a;
            found = true;
            break;
        }
    }
    if !found {
        return Err(GraphError::UnknownNode { id: b.raw() });
    }
    Ok(())
}

/// Update the middle mark on an existing edge (both adjacency halves).
pub(crate) fn set_middle(
    adj: &mut [Vec<AdjEntry>],
    a: DenseNodeId,
    b: DenseNodeId,
    middle: MiddleMark,
) -> Result<(), GraphError> {
    let mut found = false;
    for e in &mut adj[a.as_usize()] {
        if e.neighbor == b {
            e.middle = middle;
            found = true;
            break;
        }
    }
    if !found {
        return Err(GraphError::UnknownNode { id: a.raw() });
    }
    found = false;
    for e in &mut adj[b.as_usize()] {
        if e.neighbor == a {
            e.middle = middle;
            found = true;
            break;
        }
    }
    if !found {
        return Err(GraphError::UnknownNode { id: b.raw() });
    }
    Ok(())
}

/// Remove both halves of the edge between `a` and `b`.
pub(crate) fn remove_edge(adj: &mut [Vec<AdjEntry>], a: DenseNodeId, b: DenseNodeId) {
    adj[a.as_usize()].retain(|e| e.neighbor != b);
    adj[b.as_usize()].retain(|e| e.neighbor != a);
}
