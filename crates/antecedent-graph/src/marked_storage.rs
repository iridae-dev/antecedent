//! Shared marked-adjacency helpers for CPDAG / PAG storage .
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

/// Iterator over definite directed parents (Arrow→Tail into `id`).
pub(crate) fn directed_parents(
    adj: &[Vec<AdjEntry>],
    id: DenseNodeId,
) -> impl Iterator<Item = DenseNodeId> + '_ {
    adj.get(id.as_usize())
        .into_iter()
        .flatten()
        .filter(|e| matches!((e.at_self, e.at_neighbor), (Endpoint::Arrow, Endpoint::Tail)))
        .map(|e| e.neighbor)
}

/// Iterator over undirected (Tail–Tail) neighbors of `id`.
pub(crate) fn undirected_neighbors(
    adj: &[Vec<AdjEntry>],
    id: DenseNodeId,
) -> impl Iterator<Item = DenseNodeId> + '_ {
    adj.get(id.as_usize())
        .into_iter()
        .flatten()
        .filter(|e| matches!((e.at_self, e.at_neighbor), (Endpoint::Tail, Endpoint::Tail)))
        .map(|e| e.neighbor)
}

/// All marked edges (each pair once): parent-first for directed, `a.raw() <= b.raw()` for
/// undirected/conflict.
pub(crate) fn all_marked_edges(adj: &[Vec<AdjEntry>]) -> Vec<MarkedEdge> {
    let mut out = Vec::new();
    for (i, nbrs) in adj.iter().enumerate() {
        let a = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
        for e in nbrs {
            if a.raw() < e.neighbor.raw()
                || (a.raw() == e.neighbor.raw()
                    && matches!((e.at_self, e.at_neighbor), (Endpoint::Tail, Endpoint::Arrow)))
            {
                out.push(MarkedEdge {
                    a,
                    b: e.neighbor,
                    at_a: e.at_self,
                    at_b: e.at_neighbor,
                    middle: e.middle,
                });
            } else if a.raw() > e.neighbor.raw() {
                // skip reverse half
            } else if matches!((e.at_self, e.at_neighbor), (Endpoint::Arrow, Endpoint::Tail)) {
                out.push(MarkedEdge::directed(e.neighbor, a));
            }
        }
    }
    out.sort_by_key(|e| (e.a.raw(), e.b.raw(), e.at_a as u8, e.at_b as u8));
    out.dedup();
    out
}

/// Neighbors with marks at `(self, neighbor)` from `id`'s perspective (PAG marks).
pub(crate) fn neighbors(
    adj: &[Vec<AdjEntry>],
    id: DenseNodeId,
) -> impl Iterator<Item = (DenseNodeId, Endpoint, Endpoint)> + '_ {
    adj[id.as_usize()].iter().map(|e| (e.neighbor, e.at_self, e.at_neighbor))
}

/// Shared tail of `insert_marked` across CPDAG/PAG-family graphs, after each type's own
/// legality/self-loop/node-kind validation has already succeeded: duplicate check, directed-
/// cycle check, then push. Verified character-identical across `Cpdag`, `TemporalCpdag`,
/// `Pag`, and `TemporalPag`.
pub(crate) fn insert_marked_finish(
    adj: &mut [Vec<AdjEntry>],
    edge: MarkedEdge,
) -> Result<(), GraphError> {
    if edge_between(adj, edge.a, edge.b).is_some() {
        return Err(GraphError::DuplicateEdge { from: edge.a.raw(), to: edge.b.raw() });
    }
    if let Some((from, to)) = edge.parent_child() {
        let mut ws = GraphWorkspace::default();
        if reaches_directed(adj, &mut ws, to, from) {
            return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
        }
    }
    push_marked_pair(adj, edge);
    Ok(())
}

/// Shared tail of CPDAG `orient_undirected`, after node validation: look up the edge, require
/// it to be undirected, reject directed cycles, then orient.
pub(crate) fn orient_undirected_finish(
    adj: &mut [Vec<AdjEntry>],
    from: DenseNodeId,
    to: DenseNodeId,
) -> Result<(), GraphError> {
    let Some(edge) = edge_between(adj, from, to) else {
        return Err(GraphError::UnknownNode { id: from.raw() });
    };
    if !edge.is_undirected() {
        return Err(GraphError::InvalidEndpoints {
            message: "orient_undirected requires an undirected Tail–Tail edge",
        });
    }
    let mut ws = GraphWorkspace::default();
    if reaches_directed(adj, &mut ws, to, from) {
        return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
    }
    set_marks(adj, from, to, Endpoint::Tail, Endpoint::Arrow)
}

/// Shared tail of CPDAG `mark_conflict`, after node validation: require the edge to exist,
/// then pin it as `x-x`.
pub(crate) fn mark_conflict_finish(
    adj: &mut [Vec<AdjEntry>],
    a: DenseNodeId,
    b: DenseNodeId,
) -> Result<(), GraphError> {
    if edge_between(adj, a, b).is_none() {
        return Err(GraphError::UnknownNode { id: a.raw() });
    }
    set_marks(adj, a, b, Endpoint::Conflict, Endpoint::Conflict)
}

/// Shared tail of PAG-family `set_marks`, once the edge's `previous` state is known: replacing
/// marks that complete a directed edge must reject directed cycles (restoring `previous` on
/// rejection); otherwise it's a plain mark update.
pub(crate) fn set_marks_finish(
    adj: &mut [Vec<AdjEntry>],
    a: DenseNodeId,
    b: DenseNodeId,
    at_a: Endpoint,
    at_b: Endpoint,
    previous: MarkedEdge,
) -> Result<(), GraphError> {
    let edge = MarkedEdge { a, b, at_a, at_b, middle: previous.middle };
    if let Some((from, to)) = edge.parent_child() {
        remove_edge(adj, a, b);
        let mut ws = GraphWorkspace::default();
        let cycle = reaches_directed(adj, &mut ws, to, from);
        if cycle {
            push_marked_pair(adj, previous);
            return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
        }
        push_marked_pair(adj, edge);
        return Ok(());
    }
    set_marks(adj, a, b, at_a, at_b)
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
