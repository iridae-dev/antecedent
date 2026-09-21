//! Partial ancestral graphs (PAGs) with circle marks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use std::sync::Arc;

use antecedent_core::VariableId;

use crate::error::GraphError;
use crate::marked_storage::{self, AdjEntry};
use crate::types::{DenseNodeId, Endpoint, MarkedEdge, MiddleMark, NodeRef};
use crate::workspace::GraphWorkspace;

/// Static PAG over variables .
#[derive(Clone, Debug)]
pub struct Pag {
    nodes: Vec<NodeRef>,
    adj: Vec<Vec<AdjEntry>>,
}

impl Pag {
    /// Empty PAG.
    #[must_use]
    pub fn empty() -> Self {
        Self { nodes: Vec::new(), adj: Vec::new() }
    }

    /// One static node per variable `0..n`.
    #[must_use]
    pub fn with_variables(n: u32) -> Self {
        let mut g = Self::empty();
        for i in 0..n {
            let _ = g.add_node(NodeRef::Static(VariableId::from_raw(i)));
        }
        g
    }

    /// Schema-aligned PAG with named directed edges (`VariableId` raw == dense id).
    ///
    /// # Errors
    ///
    /// Unknown names or invalid inserts.
    pub fn from_named_edges(
        schema: &antecedent_core::CausalSchema,
        edges: &[(&str, &str)],
    ) -> Result<Self, GraphError> {
        let n = crate::named::schema_node_count(schema)?;
        let mut g = Self::with_variables(n);
        for &(from_name, to_name) in edges {
            let (from, to) = crate::named::resolve_named_edge(schema, from_name, to_name)?;
            g.insert_directed(from, to)?;
        }
        Ok(g)
    }

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

    /// Add a static node.
    ///
    /// # Errors
    ///
    /// Non-static or capacity.
    pub fn add_node(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        if !matches!(node, NodeRef::Static(_)) {
            return Err(GraphError::InvalidEndpoints { message: "Pag accepts only Static nodes" });
        }
        let id = u32::try_from(self.nodes.len()).map_err(|_| GraphError::TooManyNodes)?;
        self.nodes.push(node);
        self.adj.push(Vec::new());
        Ok(DenseNodeId::from_raw(id))
    }

    fn validate_node(&self, id: DenseNodeId) -> Result<(), GraphError> {
        if id.as_usize() >= self.node_count() {
            return Err(GraphError::UnknownNode { id: id.raw() });
        }
        Ok(())
    }

    pub(crate) fn validate_node_pub(&self, id: DenseNodeId) -> Result<(), GraphError> {
        self.validate_node(id)
    }

    /// Whether marks are legal for a PAG (any Tail/Arrow/Circle/Conflict pair on distinct nodes).
    ///
    /// Structural constraints (duplicates, directed cycles) are checked on insert.
    #[must_use]
    pub const fn is_pag_legal(edge: MarkedEdge) -> bool {
        edge.a.raw() != edge.b.raw()
    }

    /// Insert a PAG-legal marked edge.
    ///
    /// # Errors
    ///
    /// Unknown nodes, duplicates, self-loops, or directed cycles from arrowheads.
    pub fn insert_marked(&mut self, edge: MarkedEdge) -> Result<(), GraphError> {
        if !Self::is_pag_legal(edge) {
            return Err(GraphError::InvalidEndpoints { message: "Pag rejects self-loops" });
        }
        self.validate_node(edge.a)?;
        self.validate_node(edge.b)?;
        if edge.a == edge.b {
            return Err(GraphError::InvalidEndpoints { message: "Pag rejects self-loops" });
        }
        marked_storage::insert_marked_finish(&mut self.adj, edge)
    }

    /// Directed `from -> to`.
    ///
    /// # Errors
    ///
    /// See [`Self::insert_marked`].
    pub fn insert_directed(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
    ) -> Result<(), GraphError> {
        self.insert_marked(MarkedEdge::directed(from, to))
    }

    /// Circle-arrow `from o→ to`.
    ///
    /// # Errors
    ///
    /// See [`Self::insert_marked`].
    pub fn insert_circle_arrow(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
    ) -> Result<(), GraphError> {
        self.insert_marked(MarkedEdge {
            a: from,
            b: to,
            at_a: Endpoint::Circle,
            at_b: Endpoint::Arrow,
            middle: MiddleMark::Empty,
        })
    }

    /// Circle-circle `a o–o b`.
    ///
    /// # Errors
    ///
    /// See [`Self::insert_marked`].
    pub fn insert_circle_circle(
        &mut self,
        a: DenseNodeId,
        b: DenseNodeId,
    ) -> Result<(), GraphError> {
        let (a, b) = if a.raw() <= b.raw() { (a, b) } else { (b, a) };
        self.insert_marked(MarkedEdge {
            a,
            b,
            at_a: Endpoint::Circle,
            at_b: Endpoint::Circle,
            middle: MiddleMark::Empty,
        })
    }

    /// Bidirected `a ↔ b`.
    ///
    /// # Errors
    ///
    /// See [`Self::insert_marked`].
    pub fn insert_bidirected(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        self.insert_marked(MarkedEdge::bidirected(a, b))
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

    /// Neighbors with marks.
    pub fn neighbors(
        &self,
        id: DenseNodeId,
    ) -> impl Iterator<Item = (DenseNodeId, Endpoint, Endpoint)> + '_ {
        marked_storage::neighbors(&self.adj, id)
    }

    /// Set marks on an existing edge (from `a`'s perspective).
    ///
    /// # Errors
    ///
    /// Missing edge or cycle after orientation.
    pub fn set_marks(
        &mut self,
        a: DenseNodeId,
        b: DenseNodeId,
        at_a: Endpoint,
        at_b: Endpoint,
    ) -> Result<(), GraphError> {
        self.validate_node(a)?;
        self.validate_node(b)?;
        if !self.has_edge(a, b) {
            return Err(GraphError::UnknownNode { id: a.raw() });
        }
        let previous =
            marked_storage::edge_between(&self.adj, a, b).expect("edge present after has_edge");
        marked_storage::set_marks_finish(&mut self.adj, a, b, at_a, at_b, previous)
    }

    /// Mark an existing edge as an `x-x` conflict ([`Endpoint::Conflict`]–[`Endpoint::Conflict`]).
    ///
    /// # Errors
    ///
    /// Missing edge or unknown nodes.
    pub fn mark_conflict(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        self.set_marks(a, b, Endpoint::Conflict, Endpoint::Conflict)
    }

    /// Remove an edge (both adjacency halves).
    ///
    /// # Errors
    ///
    /// Unknown nodes or missing edge.
    pub fn remove_edge(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        self.validate_node(a)?;
        self.validate_node(b)?;
        if self.edge_between(a, b).is_none() {
            return Err(GraphError::UnknownNode { id: a.raw() });
        }
        marked_storage::remove_edge(&mut self.adj, a, b);
        Ok(())
    }

    /// Directed children (definite Tail→Arrow from this node).
    #[must_use]
    pub fn directed_children(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
        marked_storage::directed_children(&self.adj, id).collect()
    }

    /// Borrowed directed-child iterator (reachability hot path).
    pub fn directed_children_iter(
        &self,
        id: DenseNodeId,
    ) -> impl Iterator<Item = DenseNodeId> + '_ {
        marked_storage::directed_children(&self.adj, id)
    }

    /// Whether `from` reaches `to` via definite directed edges only.
    #[must_use]
    pub fn reaches_directed(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        let mut ws = GraphWorkspace::default();
        self.reaches_directed_with(&mut ws, from, to)
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
}

/// Path whose every non-endpoint has definite collider or non-collider status.
///
/// A non-endpoint `V` between `A` and `B` is a definite collider when both path
/// edges have an arrowhead at `V`. It is a definite non-collider when a path
/// edge has a tail at `V`, or when the marks at `V` are not both arrowheads and
/// `A`, `B` are not adjacent (Zhang 2008 states this for circle-circle; a
/// remaining circle beside an arrowhead is covered by the same argument): an
/// unshielded triple that is not marked as a collider is a collider in no
/// member of the class. A conflict mark records
/// that the orientation is unknown, so it never yields a definite status.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DefiniteStatusPath {
    /// Ordered nodes on the path.
    pub nodes: Vec<DenseNodeId>,
}

/// Bounded enumeration of definite-status paths, with a truncation flag.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DefiniteStatusPathSearch {
    /// Paths found within the budget.
    pub paths: Vec<DefiniteStatusPath>,
    /// `true` if `max_paths` / `max_len` cut the search short (result may be incomplete).
    pub truncated: bool,
}

impl Pag {
    /// Enumerate definite-status paths from `x` to `y` up to `max_paths` (bounded).
    ///
    /// # Errors
    ///
    /// Unknown nodes.
    pub fn definite_status_paths(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        max_paths: usize,
        max_len: usize,
    ) -> Result<DefiniteStatusPathSearch, GraphError> {
        self.validate_node(x)?;
        self.validate_node(y)?;
        let mut out = Vec::new();
        if max_paths == 0 || max_len == 0 {
            return Ok(DefiniteStatusPathSearch { paths: out, truncated: true });
        }
        let mut capped = false;
        let cut = self.walk_simple_paths(x, y, max_len, |path| {
            if out.len() >= max_paths {
                capped = true;
                return false;
            }
            if self.path_is_definite_status(path) {
                out.push(DefiniteStatusPath { nodes: path.to_vec() });
            }
            true
        });
        Ok(DefiniteStatusPathSearch { paths: out, truncated: capped || cut })
    }

    /// Depth-first walk over the simple paths from `x` to `y` of at most `max_len`
    /// nodes. `visit` returns `false` to stop. Returns whether `max_len` left a
    /// path unexplored.
    pub(crate) fn walk_simple_paths(
        &self,
        x: DenseNodeId,
        y: DenseNodeId,
        max_len: usize,
        mut visit: impl FnMut(&[DenseNodeId]) -> bool,
    ) -> bool {
        let mut cut = false;
        let mut stack = vec![vec![x]];
        while let Some(path) = stack.pop() {
            let last = *path.last().expect("nonempty");
            if path.len() > 1 && last == y {
                if !visit(&path) {
                    break;
                }
                continue;
            }
            let onward = self.neighbors(last).map(|(nbr, _, _)| nbr).filter(|n| !path.contains(n));
            if path.len() >= max_len {
                // Neighbors exist that we refuse to expand → incomplete.
                cut |= onward.count() > 0;
                continue;
            }
            for nbr in onward {
                let mut next = path.clone();
                next.push(nbr);
                stack.push(next);
            }
        }
        cut
    }

    /// Marks at `v` on the path edges from `pred` and to `succ`.
    fn marks_at(
        &self,
        pred: DenseNodeId,
        v: DenseNodeId,
        succ: DenseNodeId,
    ) -> Option<(Endpoint, Endpoint)> {
        let e1 = self.edge_between(pred, v)?;
        let e2 = self.edge_between(v, succ)?;
        Some((if e1.a == v { e1.at_a } else { e1.at_b }, if e2.a == v { e2.at_a } else { e2.at_b }))
    }

    pub(crate) fn path_is_definite_status(&self, path: &[DenseNodeId]) -> bool {
        path.windows(3).all(|w| {
            let Some((from_pred, from_succ)) = self.marks_at(w[0], w[1], w[2]) else {
                return false;
            };
            let definite_collider = from_pred == Endpoint::Arrow && from_succ == Endpoint::Arrow;
            // Members keep the PAG's unshielded colliders and add none, so an
            // unshielded triple without two arrowheads is a non-collider in all
            // of them even when only one mark at `V` is a circle.
            let conflict = from_pred == Endpoint::Conflict || from_succ == Endpoint::Conflict;
            let definite_noncollider = from_pred == Endpoint::Tail
                || from_succ == Endpoint::Tail
                || (!conflict && !definite_collider && !self.has_edge(w[0], w[2]));
            definite_collider || definite_noncollider
        })
    }

    /// Whether some member of the class could have `path` m-connecting given `z`.
    ///
    /// Each non-endpoint is judged on its own marks: it may be a non-collider
    /// unless both marks are arrowheads, and it may be a collider unless a mark
    /// is a tail or the triple is unshielded and not marked as a collider. A
    /// possible collider is open when it, or a node it may be an ancestor of, is
    /// in `z`. Every path that is m-connecting in a member passes this test, so
    /// the absence of such a path certifies separation in every member.
    pub(crate) fn path_possibly_active_given(
        &self,
        path: &[DenseNodeId],
        z: &[DenseNodeId],
    ) -> bool {
        let unknown = |m: Endpoint| matches!(m, Endpoint::Circle | Endpoint::Conflict);
        path.windows(3).all(|w| {
            let Some((from_pred, from_succ)) = self.marks_at(w[0], w[1], w[2]) else {
                return false;
            };
            let in_z = z.contains(&w[1]);
            let arrows = from_pred == Endpoint::Arrow && from_succ == Endpoint::Arrow;
            let conflict = from_pred == Endpoint::Conflict || from_succ == Endpoint::Conflict;
            let may_collide = (from_pred == Endpoint::Arrow || unknown(from_pred))
                && (from_succ == Endpoint::Arrow || unknown(from_succ))
                && (arrows || conflict || self.has_edge(w[0], w[2]));
            (!arrows && !in_z) || (may_collide && (in_z || self.possible_descendant_in(w[1], z)))
        })
    }

    /// True if some node of `z` is reachable from `v` along edges that no mark
    /// forbids reading as `a -> b` (no arrowhead at `a`, no tail at `b`).
    fn possible_descendant_in(&self, v: DenseNodeId, z: &[DenseNodeId]) -> bool {
        let mut seen = vec![false; self.node_count()];
        let mut stack = vec![v];
        seen[v.as_usize()] = true;
        while let Some(a) = stack.pop() {
            for (b, at_a, at_b) in self.neighbors(a) {
                if at_a == Endpoint::Arrow || at_b == Endpoint::Tail || seen[b.as_usize()] {
                    continue;
                }
                if z.contains(&b) {
                    return true;
                }
                seen[b.as_usize()] = true;
                stack.push(b);
            }
        }
        false
    }

    /// Whether a definite-status path is active given `z` (m-connecting).
    ///
    /// A collider is open if it **or any definite directed descendant** is in `z`.
    #[must_use]
    pub fn path_active_given(&self, path: &[DenseNodeId], z: &[DenseNodeId]) -> bool {
        if path.len() < 2 {
            return false;
        }
        let in_z = |n: DenseNodeId| z.iter().any(|&v| v == n);
        if in_z(path[0]) || in_z(path[path.len() - 1]) {
            return true;
        }
        for i in 1..path.len() - 1 {
            let pred = path[i - 1];
            let v = path[i];
            let succ = path[i + 1];
            let e1 = self.edge_between(pred, v).expect("path edge");
            let e2 = self.edge_between(v, succ).expect("path edge");
            let mark_from_pred = if e1.a == v { e1.at_a } else { e1.at_b };
            let mark_from_succ = if e2.a == v { e2.at_a } else { e2.at_b };
            let collider = matches!(mark_from_pred, Endpoint::Arrow)
                && matches!(mark_from_succ, Endpoint::Arrow);
            if collider {
                if !in_z(v) && !self.collider_descendant_in_z(v, z) {
                    return false;
                }
            } else if in_z(v) {
                return false;
            }
        }
        true
    }

    /// True if some node in `z` is a definite directed descendant of `v`.
    fn collider_descendant_in_z(&self, v: DenseNodeId, z: &[DenseNodeId]) -> bool {
        z.iter().any(|&d| d != v && self.reaches_directed(v, d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_circle_marks() {
        let mut g = Pag::with_variables(2);
        g.insert_circle_arrow(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert!(g.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
    }

    #[test]
    fn remove_edge_clears_both_halves() {
        let mut g = Pag::with_variables(2);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        g.insert_directed(a, b).unwrap();
        g.remove_edge(a, b).unwrap();
        assert!(!g.has_edge(a, b));
        assert!(g.remove_edge(a, b).is_err());
    }

    #[test]
    fn definite_status_chain() {
        let mut g = Pag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        let paths = g.definite_status_paths(a, c, 10, 8).unwrap();
        assert!(!paths.paths.is_empty());
        assert!(g.path_active_given(&paths.paths[0].nodes, &[]));
        assert!(!g.path_active_given(&paths.paths[0].nodes, &[b]));
    }
}

/// Review artifact for a discovered static PAG (pending circle marks).
#[derive(Clone, Debug)]
pub struct PagReview {
    /// Proposed PAG.
    pub graph: Pag,
    /// Edges that still have at least one circle endpoint `(a,b)` with `a.raw() <= b.raw()`.
    pub pending_circles: Arc<[(DenseNodeId, DenseNodeId)]>,
    /// Algorithm id.
    pub algorithm: Arc<str>,
}

impl PagReview {
    /// Build review listing all circle-bearing edges.
    #[must_use]
    pub fn from_pag(graph: Pag, algorithm: impl Into<Arc<str>>) -> Self {
        let mut pending = Vec::new();
        for i in 0..graph.node_count() {
            let a = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
            for (b, at_a, at_b) in graph.neighbors(a) {
                if b.raw() < a.raw() {
                    continue;
                }
                if matches!(at_a, Endpoint::Circle) || matches!(at_b, Endpoint::Circle) {
                    pending.push((a, b));
                }
            }
        }
        Self { graph, pending_circles: Arc::from(pending), algorithm: algorithm.into() }
    }

    /// Whether no circle marks remain.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.pending_circles.is_empty()
    }
}

#[cfg(test)]
mod review_tests {
    use super::*;

    #[test]
    fn review_lists_circle_edges() {
        let mut g = Pag::with_variables(2);
        g.insert_circle_circle(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let review = PagReview::from_pag(g, "fci");
        assert_eq!(review.pending_circles.len(), 1);
        assert!(!review.is_complete());
    }

    #[test]
    fn directed_only_is_complete() {
        let mut g = Pag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let review = PagReview::from_pag(g, "fci");
        assert!(review.is_complete());
    }
}
