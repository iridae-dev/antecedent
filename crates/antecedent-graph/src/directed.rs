//! One directed graph shared by [`Dag`](crate::Dag), [`Admg`](crate::Admg), and
//! [`TemporalDag`](crate::TemporalDag).
//!
//! Those public types are wrappers: an ADMG adds bidirected edges, and a
//! temporal DAG adds lag rules. The node table and the directed edges live
//! here. Both are behind `Arc`, so a wrapper copy bumps refcounts until one
//! side adds a node or a directed edge. Equality compares the node table and the directed
//! edge set, never the insertion order of an adjacency row or the insertion scratch.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use crate::algo::{bfs_reaches, insert_sorted, kahn_order};
use crate::error::GraphError;
use crate::types::{DenseNodeId, MarkedEdge, NodeRef};
use crate::workspace::GraphWorkspace;

/// Node labels plus outgoing and incoming directed adjacency.
///
/// [`Dag`](crate::Dag), [`Admg`](crate::Admg), and [`TemporalDag`](crate::TemporalDag)
/// each own one of these and add only the checks that belong to that type.
#[derive(Clone)]
pub(crate) struct DirectedCore {
    nodes: Arc<Vec<NodeRef>>,
    children: Arc<Vec<Vec<DenseNodeId>>>,
    parents: Arc<Vec<Vec<DenseNodeId>>>,
    /// Every directed edge, keyed `(parent, child)`: the duplicate check is one lookup rather
    /// than a scan of an insertion-ordered row.
    edge_index: Arc<HashSet<(DenseNodeId, DenseNodeId)>>,
    insert_ws: GraphWorkspace,
}

/// The edge index is derived from the adjacency rows and its hash order is not stable between
/// two graphs with the same edges, so the rendering leaves it out: equal graphs render alike.
impl std::fmt::Debug for DirectedCore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectedCore")
            .field("nodes", &self.nodes)
            .field("children", &self.children)
            .field("parents", &self.parents)
            .field("insert_ws", &self.insert_ws)
            .finish_non_exhaustive()
    }
}

impl Default for DirectedCore {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectedCore {
    /// Empty node table and adjacency.
    pub(crate) fn new() -> Self {
        Self {
            nodes: Arc::new(Vec::new()),
            children: Arc::new(Vec::new()),
            parents: Arc::new(Vec::new()),
            edge_index: Arc::new(HashSet::new()),
            insert_ws: GraphWorkspace::default(),
        }
    }

    /// Node labels in dense order.
    pub(crate) fn nodes(&self) -> &[NodeRef] {
        &self.nodes
    }

    /// Shared node labels. A copy shares the allocation until one side adds a node.
    #[cfg(test)]
    pub(crate) fn shared_nodes(&self) -> Arc<Vec<NodeRef>> {
        Arc::clone(&self.nodes)
    }

    /// Node count.
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the node table is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Append `node` and one empty adjacency row.
    ///
    /// # Errors
    ///
    /// [`GraphError::TooManyNodes`] when the dense id would not fit in `u32`.
    pub(crate) fn push_labeled(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        let nodes = Arc::make_mut(&mut self.nodes);
        let id = u32::try_from(nodes.len()).map_err(|_| GraphError::TooManyNodes)?;
        nodes.push(node);
        self.push_node();
        Ok(DenseNodeId::from_raw(id))
    }

    /// One empty adjacency row. Called once per added node.
    pub(crate) fn push_node(&mut self) {
        Arc::make_mut(&mut self.children).push(Vec::new());
        Arc::make_mut(&mut self.parents).push(Vec::new());
    }

    /// Outgoing lists, node order.
    pub(crate) fn child_adj(&self) -> &[Vec<DenseNodeId>] {
        &self.children
    }

    /// Incoming lists, node order.
    pub(crate) fn parent_adj(&self) -> &[Vec<DenseNodeId>] {
        &self.parents
    }

    /// Children of `id`, or empty when `id` is outside the graph.
    pub(crate) fn children_of(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.children.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Parents of `id`, or empty when `id` is outside the graph.
    pub(crate) fn parents_of(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.parents.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Whether `from` reaches `to`, including the case `from == to`.
    ///
    /// An unknown node reaches only itself. [`Admg`](crate::Admg) and
    /// [`TemporalDag`](crate::TemporalDag) both use this predicate.
    pub(crate) fn reaches(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        let mut ws = GraphWorkspace::default();
        self.reaches_with(from, to, &mut ws)
    }

    /// Directed edges as marked edges, parent then child, in node order.
    pub(crate) fn marked_edges(&self) -> impl Iterator<Item = MarkedEdge> + '_ {
        self.children.iter().enumerate().flat_map(|(i, kids)| {
            let from = DenseNodeId::from_index(i);
            kids.iter().copied().map(move |to| MarkedEdge::directed(from, to))
        })
    }

    /// Directed edges as raw `(parent, child)` dense-id pairs in ascending order.
    pub(crate) fn edge_pairs_sorted(&self) -> Vec<(u32, u32)> {
        sorted_row_pairs(&self.children, |_, _| true)
    }

    /// Whether `from -> to` is already present.
    pub(crate) fn has_edge(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        self.edge_index.contains(&(from, to))
    }

    /// Reachability using a reusable workspace.
    pub(crate) fn reaches_with(
        &self,
        from: DenseNodeId,
        to: DenseNodeId,
        ws: &mut GraphWorkspace,
    ) -> bool {
        bfs_reaches(&self.children, from, to, ws)
    }

    /// Kahn order, or `None` when the directed edges contain a cycle.
    pub(crate) fn topological_order(&self) -> Option<Vec<DenseNodeId>> {
        kahn_order(&self.parents, &self.children)
    }

    /// Duplicate, then a cycle through `to` back to `from`, then the edge.
    ///
    /// The caller has already checked that both endpoints exist and that any
    /// graph-specific rule (self-loop, temporal order) accepts the edge.
    pub(crate) fn try_insert(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
    ) -> Result<(), GraphError> {
        if self.has_edge(from, to) {
            return Err(GraphError::DuplicateEdge { from: from.raw(), to: to.raw() });
        }
        let mut ws = core::mem::take(&mut self.insert_ws);
        let cycle = bfs_reaches(&self.children, to, from, &mut ws);
        self.insert_ws = ws;
        if cycle {
            return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
        }
        self.push_unchecked(from, to);
        Ok(())
    }

    /// Cancellation-aware cycle check for an admitted temporal unrolling.
    pub(crate) fn try_insert_temporal(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<(), GraphError> {
        use antecedent_core::TemporalMaterializationError;
        if ctx.cancellation.is_cancelled() {
            return Err(TemporalMaterializationError::Cancelled.into());
        }
        if self.has_edge(from, to) {
            return Err(GraphError::DuplicateEdge { from: from.raw(), to: to.raw() });
        }
        let mut ws = core::mem::take(&mut self.insert_ws);
        ws.prepare(self.children.len());
        ws.frontier.push(to);
        ws.visited.insert(to);
        let mut cycle = false;
        while let Some(node) = ws.frontier.pop() {
            if ctx.cancellation.is_cancelled() {
                self.insert_ws = ws;
                return Err(TemporalMaterializationError::Cancelled.into());
            }
            for &child in &self.children[node.as_usize()] {
                if ctx.cancellation.is_cancelled() {
                    self.insert_ws = ws;
                    return Err(TemporalMaterializationError::Cancelled.into());
                }
                if child == from {
                    cycle = true;
                    break;
                }
                if !ws.visited.contains(child) {
                    ws.visited.insert(child);
                    ws.frontier.push(child);
                }
            }
            if cycle {
                break;
            }
        }
        self.insert_ws = ws;
        if cycle {
            return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
        }
        self.push_unchecked(from, to);
        Ok(())
    }

    /// Append an edge. The caller guarantees both nodes exist, the edge is new,
    /// and the directed graph stays acyclic.
    pub(crate) fn push_unchecked(&mut self, from: DenseNodeId, to: DenseNodeId) {
        Arc::make_mut(&mut self.children)[from.as_usize()].push(to);
        Arc::make_mut(&mut self.parents)[to.as_usize()].push(from);
        Arc::make_mut(&mut self.edge_index).insert((from, to));
    }

    /// Append `from -> to` when it is not already present. Does not search for a cycle.
    pub(crate) fn push_if_absent(&mut self, from: DenseNodeId, to: DenseNodeId) -> bool {
        if self.has_edge(from, to) {
            return false;
        }
        self.push_unchecked(from, to);
        true
    }

    /// Directed edge-set equality: the same `(parent, child)` pairs, whatever order the edges
    /// were inserted in. Neither the row order nor the insertion scratch is part of the graph.
    pub(crate) fn same_edges(&self, other: &Self) -> bool {
        // The edge index holds exactly the pairs in the adjacency rows, so set equality of the
        // index is edge-set equality (and each parent row is determined by the children).
        Arc::ptr_eq(&self.edge_index, &other.edge_index) || self.edge_index == other.edge_index
    }

    /// Node-table and edge-list equality. The insertion scratch is not part of the graph.
    pub(crate) fn same_directed(&self, other: &Self) -> bool {
        self.nodes.as_slice() == other.nodes.as_slice() && self.same_edges(other)
    }

    /// Whether appending `edges` in order adds only new edges and stays acyclic.
    ///
    /// Endpoints outside the graph are rejected here so the caller can fall
    /// back to [`Self::try_insert`], which is what reports [`GraphError::UnknownNode`].
    /// A self-loop fails the acyclicity check, matching one-edge insertion on a DAG.
    pub(crate) fn accepts_new_edges(&self, edges: &[(DenseNodeId, DenseNodeId)]) -> bool {
        let n = self.children.len();
        let mut seen = std::collections::HashSet::with_capacity(edges.len());
        for &(from, to) in edges {
            if from.as_usize() >= n || to.as_usize() >= n {
                return false;
            }
            if !seen.insert((from, to)) || self.has_edge(from, to) {
                return false;
            }
        }
        if edges.is_empty() {
            return true;
        }
        let mut trial = self.clone();
        trial.push_edges(edges);
        trial.topological_order().is_some()
    }

    /// Append edges. The caller has already accepted them with [`Self::accepts_new_edges`]
    /// or an equivalent per-edge check.
    pub(crate) fn push_edges(&mut self, edges: &[(DenseNodeId, DenseNodeId)]) {
        for &(from, to) in edges {
            self.push_unchecked(from, to);
        }
    }
}

/// Sorted directed adjacency that may contain cycles.
///
/// [`DirectedCore`] appends and rejects a cycle. This store keeps each child
/// list and each parent list in ascending dense-id order and does not search
/// for a cycle. [`Self::known`] and [`crate::algo::insert_sorted`] are the one copy
/// of those operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DirectedStore {
    nodes: Vec<NodeRef>,
    children: Vec<Vec<DenseNodeId>>,
    parents: Vec<Vec<DenseNodeId>>,
}

impl DirectedStore {
    /// Empty node table and adjacency.
    pub(crate) const fn empty() -> Self {
        Self { nodes: Vec::new(), children: Vec::new(), parents: Vec::new() }
    }

    /// Node labels in dense order.
    pub(crate) fn nodes(&self) -> &[NodeRef] {
        &self.nodes
    }

    /// Node count.
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the node table is empty.
    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Outgoing lists, node order.
    pub(crate) fn child_rows(&self) -> &[Vec<DenseNodeId>] {
        &self.children
    }

    /// Incoming lists, node order.
    pub(crate) fn parent_rows(&self) -> &[Vec<DenseNodeId>] {
        &self.parents
    }

    /// Children of `id`, or empty when `id` is outside the graph.
    pub(crate) fn children_of(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.children.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Parents of `id`, or empty when `id` is outside the graph.
    pub(crate) fn parents_of(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.parents.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Whether `id` indexes a graph of `node_count` nodes.
    pub(crate) fn known(node_count: usize, id: DenseNodeId) -> Result<(), GraphError> {
        if id.as_usize() >= node_count {
            Err(GraphError::UnknownNode { id: id.raw() })
        } else {
            Ok(())
        }
    }

    /// Append `node` and one empty adjacency row.
    ///
    /// # Errors
    ///
    /// [`GraphError::TooManyNodes`] when the dense id would not fit in `u32`.
    pub(crate) fn push_labeled(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        let id = u32::try_from(self.nodes.len()).map_err(|_| GraphError::TooManyNodes)?;
        self.nodes.push(node);
        self.children.push(Vec::new());
        self.parents.push(Vec::new());
        Ok(DenseNodeId::from_raw(id))
    }

    /// Record `from -> to` in both adjacency lists.
    ///
    /// The caller has checked that both nodes exist and that the edge is not a
    /// self-loop. A duplicate is not inserted.
    ///
    /// # Errors
    ///
    /// [`GraphError::DuplicateEdge`] when the directed edge is already present.
    pub(crate) fn link(&mut self, from: DenseNodeId, to: DenseNodeId) -> Result<(), GraphError> {
        if !insert_sorted(&mut self.children[from.as_usize()], to) {
            return Err(GraphError::DuplicateEdge { from: from.raw(), to: to.raw() });
        }
        if !insert_sorted(&mut self.parents[to.as_usize()], from) {
            let kids = &mut self.children[from.as_usize()];
            let at = kids.binary_search(&to).expect("child edge was just inserted");
            kids.remove(at);
            return Err(GraphError::DuplicateEdge { from: from.raw(), to: to.raw() });
        }
        Ok(())
    }

    /// Directed edges as raw `(parent, child)` pairs in ascending order.
    pub(crate) fn edge_pairs_sorted(&self) -> Vec<(u32, u32)> {
        sorted_row_pairs(&self.children, |_, _| true)
    }
}

/// Whether two adjacency tables hold the same neighbors per node, ignoring the order inside
/// each row. Equal rows are accepted without sorting; only differing rows are compared sorted.
pub(crate) fn same_rows_as_sets(a: &[Vec<DenseNodeId>], b: &[Vec<DenseNodeId>]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            x == y
                || (x.len() == y.len() && {
                    let (mut x, mut y) = (x.clone(), y.clone());
                    x.sort_unstable();
                    y.sort_unstable();
                    x == y
                })
        })
}

/// `(row index, entry)` pairs of adjacency `rows` that `keep` accepts, ascending.
///
/// The one mechanism behind every sorted edge-pair listing: directed edges keep every pair,
/// a symmetric bidirected row keeps each unordered pair once with `to > from`.
pub(crate) fn sorted_row_pairs(
    rows: &[Vec<DenseNodeId>],
    mut keep: impl FnMut(u32, u32) -> bool,
) -> Vec<(u32, u32)> {
    let mut pairs = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        let from = u32::try_from(i).expect("node fit");
        pairs.extend(row.iter().map(|to| (from, to.raw())).filter(|&(f, t)| keep(f, t)));
    }
    pairs.sort_unstable();
    pairs
}
