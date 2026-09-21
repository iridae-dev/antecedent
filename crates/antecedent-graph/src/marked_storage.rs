//! Shared marked-adjacency helpers for CPDAG / PAG storage.
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
    adj.get(id.as_usize()).into_iter().flatten().map(|e| (e.neighbor, e.at_self, e.at_neighbor))
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
        if reaches_directed_scratch(adj, to, from) {
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
    if reaches_directed_scratch(adj, to, from) {
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
        let cycle = reaches_directed_scratch(adj, to, from);
        if cycle {
            push_marked_pair(adj, previous);
            return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
        }
        push_marked_pair(adj, edge);
        return Ok(());
    }
    set_marks(adj, a, b, at_a, at_b)
}

/// [`reaches_directed`] on a per-thread scratch workspace, so the inner loops of PAG / CPDAG
/// construction and orientation do not allocate a fresh workspace per inserted edge.
fn reaches_directed_scratch(adj: &[Vec<AdjEntry>], from: DenseNodeId, to: DenseNodeId) -> bool {
    thread_local! {
        static SCRATCH: std::cell::RefCell<GraphWorkspace> =
            std::cell::RefCell::new(GraphWorkspace::default());
    }
    SCRATCH.with(|cell| match cell.try_borrow_mut() {
        Ok(mut ws) => reaches_directed(adj, &mut ws, from, to),
        Err(_) => reaches_directed(adj, &mut GraphWorkspace::default(), from, to),
    })
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

/// Read-only accessors shared verbatim by [`crate::Cpdag`] and [`crate::TemporalCpdag`]:
/// both store `nodes: Vec<NodeRef>` and `adj: Vec<Vec<AdjEntry>>`, and differ only in which
/// node kinds and edge orientations their insertion methods admit. Expanded inside each
/// type's inherent `impl`; the call site imports the names used here.
macro_rules! impl_cpdag_accessors {
    () => {
        /// Node count.
        #[must_use]
        pub fn node_count(&self) -> usize {
            self.nodes.len()
        }

        /// Whether empty.
        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.nodes.is_empty()
        }

        /// Nodes in dense order.
        #[must_use]
        pub fn nodes(&self) -> &[NodeRef] {
            &self.nodes
        }

        /// Whether any edge exists between `a` and `b`.
        #[must_use]
        pub fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool {
            self.edge_between(a, b).is_some()
        }

        /// Marked edge between `a` and `b` if present.
        #[must_use]
        pub fn edge_between(&self, a: DenseNodeId, b: DenseNodeId) -> Option<MarkedEdge> {
            marked_storage::edge_between(&self.adj, a, b)
        }

        /// All marked edges (each pair once).
        #[must_use]
        pub fn edges(&self) -> Vec<MarkedEdge> {
            marked_storage::all_marked_edges(&self.adj)
        }

        /// Directed children of `id`.
        #[must_use]
        pub fn children(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
            marked_storage::directed_children(&self.adj, id).collect()
        }

        /// Directed parents of `id`.
        #[must_use]
        pub fn parents(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
            marked_storage::directed_parents(&self.adj, id).collect()
        }

        /// Undirected neighbors of `id`.
        #[must_use]
        pub fn undirected_neighbors(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
            marked_storage::undirected_neighbors(&self.adj, id).collect()
        }

        /// Borrowed directed-child iterator.
        pub fn children_iter(&self, id: DenseNodeId) -> impl Iterator<Item = DenseNodeId> + '_ {
            marked_storage::directed_children(&self.adj, id)
        }

        /// Count conflict (`x-x`) edges.
        #[must_use]
        pub fn conflict_edge_count(&self) -> usize {
            self.edges().iter().filter(|e| e.is_conflict()).count()
        }

        /// Count undirected (Tail–Tail) edges.
        #[must_use]
        pub fn undirected_edge_count(&self) -> usize {
            self.edges().iter().filter(|e| e.is_undirected()).count()
        }

        /// Count directed edges.
        #[must_use]
        pub fn directed_edge_count(&self) -> usize {
            self.edges().iter().filter(|e| e.parent_child().is_some()).count()
        }

        /// Directed reachability reusing a caller-owned workspace.
        #[must_use]
        pub fn reaches_directed_with(
            &self,
            ws: &mut GraphWorkspace,
            from: DenseNodeId,
            to: DenseNodeId,
        ) -> bool {
            marked_storage::reaches_directed(&self.adj, ws, from, to)
        }

        fn validate_node(&self, id: DenseNodeId) -> Result<(), GraphError> {
            if id.as_usize() >= self.node_count() {
                Err(GraphError::UnknownNode { id: id.raw() })
            } else {
                Ok(())
            }
        }
    };
}
pub(crate) use impl_cpdag_accessors;

#[cfg(test)]
mod tests {
    use crate::pag::Pag;

    use super::*;

    #[test]
    fn cycle_checks_stay_correct_when_the_scratch_is_reused_across_graph_sizes() {
        // Large graph first, so the per-thread scratch is sized for 300 nodes.
        let mut big = Pag::with_variables(300);
        for i in 0..299u32 {
            big.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
        }
        assert!(matches!(
            big.insert_directed(DenseNodeId::from_raw(299), DenseNodeId::from_raw(0)),
            Err(GraphError::Cycle { .. })
        ));
        // Small graph afterwards: stale visited bits from the big graph must not leak.
        let mut small = Pag::with_variables(3);
        let (a, b, c) =
            (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
        small.insert_directed(a, b).unwrap();
        small.insert_directed(a, c).unwrap();
        small.insert_directed(c, b).unwrap();
        assert!(matches!(small.insert_directed(b, a), Err(GraphError::Cycle { .. })));
        assert!(matches!(small.insert_directed(b, c), Err(GraphError::DuplicateEdge { .. })));
    }
}
