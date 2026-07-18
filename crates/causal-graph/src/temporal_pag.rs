//! Temporal PAG over lagged nodes (DESIGN.md §6.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use std::sync::Arc;

use causal_core::{Lag, VariableId};

use crate::error::GraphError;
use crate::marked_storage::{self, AdjEntry};
use crate::pag::Pag;
use crate::types::{DenseNodeId, Endpoint, MarkedEdge, MiddleMark, NodeRef};
use crate::workspace::GraphWorkspace;

/// Temporal PAG: lagged nodes with ancestral-graph marks including circles.
#[derive(Clone, Debug)]
pub struct TemporalPag {
    nodes: Vec<NodeRef>,
    adj: Vec<Vec<AdjEntry>>,
}

impl TemporalPag {
    /// Empty temporal PAG.
    #[must_use]
    pub fn empty() -> Self {
        Self { nodes: Vec::new(), adj: Vec::new() }
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

    /// Add a lagged node.
    ///
    /// # Errors
    ///
    /// Non-lagged or capacity.
    pub fn add_node(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        match node {
            NodeRef::Lagged { .. } => {}
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalPag accepts only Lagged nodes",
                });
            }
        }
        let id = u32::try_from(self.nodes.len()).map_err(|_| GraphError::TooManyNodes)?;
        self.nodes.push(node);
        self.adj.push(Vec::new());
        Ok(DenseNodeId::from_raw(id))
    }

    /// Add `variable` at `lag`.
    ///
    /// # Errors
    ///
    /// Capacity.
    pub fn add_lagged(
        &mut self,
        variable: VariableId,
        lag: Lag,
    ) -> Result<DenseNodeId, GraphError> {
        self.add_node(NodeRef::Lagged { variable, lag })
    }

    fn validate_node(&self, id: DenseNodeId) -> Result<(), GraphError> {
        if id.as_usize() >= self.node_count() {
            return Err(GraphError::UnknownNode { id: id.raw() });
        }
        Ok(())
    }

    /// Insert a marked edge with temporal constraints (no future→past directed).
    ///
    /// # Errors
    ///
    /// Unknown nodes, duplicates, temporal violations, cycles.
    pub fn insert_marked(&mut self, edge: MarkedEdge) -> Result<(), GraphError> {
        self.validate_node(edge.a)?;
        self.validate_node(edge.b)?;
        if edge.a == edge.b {
            return Err(GraphError::InvalidEndpoints { message: "TemporalPag rejects self-loops" });
        }
        if let (
            NodeRef::Lagged { variable: v1, lag: l1 },
            NodeRef::Lagged { variable: v2, lag: l2 },
        ) = (self.nodes[edge.a.as_usize()], self.nodes[edge.b.as_usize()])
        {
            if v1 == v2 && l1 == l2 && l1.is_contemporaneous() {
                return Err(GraphError::ContemporaneousSelfEdge { variable: v1 });
            }
            // Directed edge must not point future → past (smaller lag = closer to present).
            if let Some((from, to)) = edge.parent_child() {
                if let (NodeRef::Lagged { lag: lf, .. }, NodeRef::Lagged { lag: lt, .. }) =
                    (self.nodes[from.as_usize()], self.nodes[to.as_usize()])
                {
                    // Lag is non-negative steps into the past; edge from larger lag to smaller
                    // lag goes past→present (allowed). from lag < to lag means future→past.
                    if lf.raw() < lt.raw() {
                        return Err(GraphError::InvalidEndpoints {
                            message: "TemporalPag rejects future-to-past directed edges",
                        });
                    }
                }
            }
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

    /// Directed insert.
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

    /// Circle-arrow insert with empty middle mark.
    ///
    /// # Errors
    ///
    /// See [`Self::insert_marked`].
    pub fn insert_circle_arrow(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
    ) -> Result<(), GraphError> {
        self.insert_circle_arrow_with_middle(from, to, MiddleMark::Empty)
    }

    /// Circle-arrow insert with an LPCMCI middle mark (typically [`MiddleMark::Left`] for lagged).
    ///
    /// # Errors
    ///
    /// See [`Self::insert_marked`].
    pub fn insert_circle_arrow_with_middle(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
        middle: MiddleMark,
    ) -> Result<(), GraphError> {
        self.insert_marked(MarkedEdge {
            a: from,
            b: to,
            at_a: Endpoint::Circle,
            at_b: Endpoint::Arrow,
            middle,
        })
    }

    /// Circle-circle contemporaneous insert with middle mark (typically [`MiddleMark::Unknown`]).
    ///
    /// # Errors
    ///
    /// See [`Self::insert_marked`].
    pub fn insert_circle_circle_with_middle(
        &mut self,
        a: DenseNodeId,
        b: DenseNodeId,
        middle: MiddleMark,
    ) -> Result<(), GraphError> {
        let (lo, hi) = if a.raw() <= b.raw() { (a, b) } else { (b, a) };
        self.insert_marked(MarkedEdge {
            a: lo,
            b: hi,
            at_a: Endpoint::Circle,
            at_b: Endpoint::Circle,
            middle,
        })
    }

    /// Whether edge exists.
    #[must_use]
    pub fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool {
        self.edge_between(a, b).is_some()
    }

    /// Edge between nodes.
    #[must_use]
    pub fn edge_between(&self, a: DenseNodeId, b: DenseNodeId) -> Option<MarkedEdge> {
        marked_storage::edge_between(&self.adj, a, b)
    }

    /// Neighbors.
    pub fn neighbors(
        &self,
        id: DenseNodeId,
    ) -> impl Iterator<Item = (DenseNodeId, Endpoint, Endpoint)> + '_ {
        self.adj[id.as_usize()].iter().map(|e| (e.neighbor, e.at_self, e.at_neighbor))
    }

    /// Set marks on existing edge.
    ///
    /// When the new marks form a definite directed edge, rejects orientations that
    /// would create a directed cycle (same check as [`crate::pag::Pag::set_marks`]).
    /// On cycle, previous marks are restored and [`GraphError::Cycle`] is returned.
    ///
    /// # Errors
    ///
    /// Missing edge or directed cycle after orientation.
    pub fn set_marks(
        &mut self,
        a: DenseNodeId,
        b: DenseNodeId,
        at_a: Endpoint,
        at_b: Endpoint,
    ) -> Result<(), GraphError> {
        self.validate_node(a)?;
        self.validate_node(b)?;
        let Some(previous) = self.edge_between(a, b) else {
            return Err(GraphError::UnknownNode { id: a.raw() });
        };
        let edge = MarkedEdge { a, b, at_a, at_b, middle: previous.middle };
        if let Some((from, to)) = edge.parent_child() {
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

    /// Mark an existing edge as a pinned baseline `x-x` conflict ([`Endpoint::Conflict`]–[`Endpoint::Conflict`]).
    ///
    /// # Errors
    ///
    /// Missing edge or unknown nodes.
    pub fn mark_conflict(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        self.set_marks(a, b, Endpoint::Conflict, Endpoint::Conflict)
    }

    /// Set / merge the LPCMCI middle mark on an existing edge.
    ///
    /// # Errors
    ///
    /// Missing edge.
    pub fn apply_middle(
        &mut self,
        a: DenseNodeId,
        b: DenseNodeId,
        update: MiddleMark,
    ) -> Result<(), GraphError> {
        self.validate_node(a)?;
        self.validate_node(b)?;
        let Some(e) = self.edge_between(a, b) else {
            return Err(GraphError::UnknownNode { id: a.raw() });
        };
        marked_storage::set_middle(&mut self.adj, a, b, e.middle.apply(update))
    }

    /// Replace the middle mark (no merge).
    ///
    /// # Errors
    ///
    /// Missing edge.
    pub fn set_middle(
        &mut self,
        a: DenseNodeId,
        b: DenseNodeId,
        middle: MiddleMark,
    ) -> Result<(), GraphError> {
        self.validate_node(a)?;
        self.validate_node(b)?;
        if self.edge_between(a, b).is_none() {
            return Err(GraphError::UnknownNode { id: a.raw() });
        }
        marked_storage::set_middle(&mut self.adj, a, b, middle)
    }

    /// Middle mark between `a` and `b`, if adjacent.
    #[must_use]
    pub fn middle_between(&self, a: DenseNodeId, b: DenseNodeId) -> Option<MiddleMark> {
        self.edge_between(a, b).map(|e| e.middle)
    }

    /// Remove an edge (both adjacency halves).
    ///
    /// # Errors
    ///
    /// Unknown nodes.
    pub fn remove_edge(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        self.validate_node(a)?;
        self.validate_node(b)?;
        if self.edge_between(a, b).is_none() {
            return Err(GraphError::UnknownNode { id: a.raw() });
        }
        marked_storage::remove_edge(&mut self.adj, a, b);
        Ok(())
    }

    /// Force all middle marks to [`MiddleMark::Empty`] (pinned baseline finalization).
    pub fn clear_middle_marks(&mut self) {
        for list in &mut self.adj {
            for e in list.iter_mut() {
                e.middle = MiddleMark::Empty;
            }
        }
    }

    /// Directed reachability.
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

    /// View as static [`Pag`] for algorithms that only need adjacency/marks.
    ///
    /// Nodes become `Static(VariableId::from_raw(dense))` — only for algorithm reuse.
    #[must_use]
    pub fn as_static_pag_for_alg(&self) -> Pag {
        let n = u32::try_from(self.node_count()).expect("node fit");
        let mut p = Pag::with_variables(n);
        for i in 0..self.node_count() {
            let a = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
            for e in &self.adj[i] {
                if e.neighbor.raw() < a.raw() {
                    continue;
                }
                let edge = MarkedEdge {
                    a,
                    b: e.neighbor,
                    at_a: e.at_self,
                    at_b: e.at_neighbor,
                    middle: e.middle,
                };
                let _ = p.insert_marked(edge);
            }
        }
        p
    }

    /// Definite-status paths (delegates via static projection of marks).
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
    ) -> Result<crate::pag::DefiniteStatusPathSearch, GraphError> {
        self.as_static_pag_for_alg().definite_status_paths(x, y, max_paths, max_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::Lag;

    #[test]
    fn rejects_future_to_past_directed() {
        let mut g = TemporalPag::empty();
        let past = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
        let present = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(0)).unwrap();
        // present -> past is future to past
        assert!(g.insert_directed(present, past).is_err());
        // past -> present ok
        g.insert_directed(past, present).unwrap();
    }

    #[test]
    fn allows_circle_arrow() {
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(0)).unwrap();
        g.insert_circle_arrow(a, b).unwrap();
        assert!(g.has_edge(a, b));
    }

    #[test]
    fn set_marks_rejects_directed_cycle_and_restores() {
        // c → b → a with a o→ c; completing a → c would cycle (c already reaches a).
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let c = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(c, b).unwrap();
        g.insert_directed(b, a).unwrap();
        g.insert_circle_arrow(a, c).unwrap();
        let err = g.set_marks(a, c, Endpoint::Tail, Endpoint::Arrow).unwrap_err();
        assert!(matches!(err, GraphError::Cycle { .. }));
        let e = g.edge_between(a, c).unwrap();
        let (at_a, at_c) = if e.a == a { (e.at_a, e.at_b) } else { (e.at_b, e.at_a) };
        assert!(matches!(at_a, Endpoint::Circle));
        assert!(matches!(at_c, Endpoint::Arrow));
    }

    #[test]
    fn middle_marks_and_remove_edge() {
        let mut g = TemporalPag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_circle_arrow_with_middle(a, b, MiddleMark::Left).unwrap();
        assert_eq!(g.middle_between(a, b), Some(MiddleMark::Left));
        g.apply_middle(a, b, MiddleMark::Right).unwrap();
        assert_eq!(g.middle_between(a, b), Some(MiddleMark::Both));
        g.clear_middle_marks();
        assert_eq!(g.middle_between(a, b), Some(MiddleMark::Empty));
        g.remove_edge(a, b).unwrap();
        assert!(!g.has_edge(a, b));
    }
}

/// Review artifact for a discovered temporal PAG (pending circle marks).
#[derive(Clone, Debug)]
pub struct TemporalPagReview {
    /// Proposed PAG.
    pub graph: TemporalPag,
    /// Edges that still have at least one circle endpoint `(a,b)` with `a.raw() <= b.raw()`.
    pub pending_circles: Arc<[(DenseNodeId, DenseNodeId)]>,
    /// Algorithm id.
    pub algorithm: Arc<str>,
}

impl TemporalPagReview {
    /// Build review listing all circle-bearing edges.
    #[must_use]
    pub fn from_pag(graph: TemporalPag, algorithm: impl Into<Arc<str>>) -> Self {
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
