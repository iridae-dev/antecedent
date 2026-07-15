//! Partial ancestral graphs (PAGs) with circle marks (DESIGN.md §6.2 / §6.5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use causal_core::VariableId;

use crate::error::GraphError;
use crate::marked_storage::{self, AdjEntry};
use crate::types::{DenseNodeId, Endpoint, MarkedEdge, NodeRef};
use crate::workspace::GraphWorkspace;

/// Static PAG over variables (DESIGN §6.2).
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

    /// Whether marks are legal for a PAG (any Tail/Arrow/Circle pair on distinct nodes).
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
        if self.has_edge(edge.a, edge.b) {
            return Err(GraphError::DuplicateEdge { from: edge.a.raw(), to: edge.b.raw() });
        }
        if let Some((from, to)) = edge.parent_child() {
            if self.reaches_directed(to, from) {
                return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
            }
        }
        marked_storage::push_marked_pair(&mut self.adj, edge);
        Ok(())
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
        self.insert_marked(MarkedEdge { a, b, at_a: Endpoint::Circle, at_b: Endpoint::Circle })
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
        self.adj[id.as_usize()].iter().map(|e| (e.neighbor, e.at_self, e.at_neighbor))
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
        let edge = MarkedEdge { a, b, at_a, at_b };
        if let Some((from, to)) = edge.parent_child() {
            let previous =
                marked_storage::edge_between(&self.adj, a, b).expect("edge present after has_edge");
            marked_storage::remove_edge(&mut self.adj, a, b);
            let cycle = self.reaches_directed(to, from);
            if cycle {
                marked_storage::push_marked_pair(&mut self.adj, previous);
                return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
            }
            marked_storage::push_marked_pair(&mut self.adj, edge);
            return Ok(());
        }
        marked_storage::set_marks(&mut self.adj, a, b, at_a, at_b)
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
        let mut truncated = false;
        let mut stack = vec![vec![x]];
        while let Some(path) = stack.pop() {
            if out.len() >= max_paths {
                truncated = true;
                break;
            }
            let last = *path.last().expect("nonempty");
            if path.len() > 1 && last == y {
                if self.path_is_definite_status(&path) {
                    out.push(DefiniteStatusPath { nodes: path });
                }
                continue;
            }
            if path.len() >= max_len {
                // Neighbors exist that we refuse to expand → incomplete.
                for (nbr, _, _) in self.neighbors(last) {
                    if path.len() >= 2 && path[path.len() - 2] == nbr {
                        continue;
                    }
                    if path.contains(&nbr) {
                        continue;
                    }
                    truncated = true;
                    break;
                }
                continue;
            }
            for (nbr, _, _) in self.neighbors(last) {
                if path.len() >= 2 && path[path.len() - 2] == nbr {
                    continue; // no immediate backtrack
                }
                if path.contains(&nbr) {
                    continue;
                }
                let mut next = path.clone();
                next.push(nbr);
                stack.push(next);
            }
        }
        Ok(DefiniteStatusPathSearch { paths: out, truncated })
    }

    fn path_is_definite_status(&self, path: &[DenseNodeId]) -> bool {
        if path.len() < 2 {
            return true;
        }
        for i in 1..path.len() - 1 {
            let pred = path[i - 1];
            let v = path[i];
            let succ = path[i + 1];
            let Some(e1) = self.edge_between(pred, v) else {
                return false;
            };
            let Some(e2) = self.edge_between(v, succ) else {
                return false;
            };
            let mark_from_pred = if e1.a == v { e1.at_a } else { e1.at_b };
            let mark_from_succ = if e2.a == v { e2.at_a } else { e2.at_b };
            let definite_collider = matches!(mark_from_pred, Endpoint::Arrow)
                && matches!(mark_from_succ, Endpoint::Arrow);
            let definite_noncollider = matches!(mark_from_pred, Endpoint::Tail)
                || matches!(mark_from_succ, Endpoint::Tail);
            if !(definite_collider || definite_noncollider) {
                return false;
            }
        }
        true
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
