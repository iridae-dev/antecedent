//! Temporal PAG over lagged nodes (DESIGN.md §6.2 / Phase 8).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use std::sync::Arc;

use causal_core::{Lag, VariableId};

use crate::error::GraphError;
use crate::pag::{DefiniteStatusPath, Pag, PagAdjEntry};
use crate::types::{DenseNodeId, Endpoint, MarkedEdge, NodeRef};
use crate::workspace::GraphWorkspace;

/// Temporal PAG: lagged nodes with ancestral-graph marks including circles.
#[derive(Clone, Debug)]
pub struct TemporalPag {
    nodes: Vec<NodeRef>,
    adj: Vec<Vec<PagAdjEntry>>,
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
            return Err(GraphError::InvalidEndpoints {
                message: "TemporalPag rejects self-loops",
            });
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
                if let (
                    NodeRef::Lagged { lag: lf, .. },
                    NodeRef::Lagged { lag: lt, .. },
                ) = (self.nodes[from.as_usize()], self.nodes[to.as_usize()])
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
            return Err(GraphError::DuplicateEdge {
                from: edge.a.raw(),
                to: edge.b.raw(),
            });
        }
        if let Some((from, to)) = edge.parent_child() {
            if self.reaches_directed(to, from) {
                return Err(GraphError::Cycle {
                    from: from.raw(),
                    to: to.raw(),
                });
            }
        }
        self.adj[edge.a.as_usize()].push(PagAdjEntry {
            neighbor: edge.b,
            at_self: edge.at_a,
            at_neighbor: edge.at_b,
        });
        self.adj[edge.b.as_usize()].push(PagAdjEntry {
            neighbor: edge.a,
            at_self: edge.at_b,
            at_neighbor: edge.at_a,
        });
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

    /// Circle-arrow insert.
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

    /// Whether edge exists.
    #[must_use]
    pub fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool {
        self.edge_between(a, b).is_some()
    }

    /// Edge between nodes.
    #[must_use]
    pub fn edge_between(&self, a: DenseNodeId, b: DenseNodeId) -> Option<MarkedEdge> {
        if a.as_usize() >= self.node_count() || b.as_usize() >= self.node_count() {
            return None;
        }
        for e in &self.adj[a.as_usize()] {
            if e.neighbor == b {
                return Some(MarkedEdge {
                    a,
                    b,
                    at_a: e.at_self,
                    at_b: e.at_neighbor,
                });
            }
        }
        None
    }

    /// Neighbors.
    #[must_use]
    pub fn neighbors(
        &self,
        id: DenseNodeId,
    ) -> impl Iterator<Item = (DenseNodeId, Endpoint, Endpoint)> + '_ {
        self.adj[id.as_usize()]
            .iter()
            .map(|e| (e.neighbor, e.at_self, e.at_neighbor))
    }

    /// Set marks on existing edge.
    ///
    /// # Errors
    ///
    /// Missing edge.
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
        for e in &mut self.adj[a.as_usize()] {
            if e.neighbor == b {
                e.at_self = at_a;
                e.at_neighbor = at_b;
            }
        }
        for e in &mut self.adj[b.as_usize()] {
            if e.neighbor == a {
                e.at_self = at_b;
                e.at_neighbor = at_a;
            }
        }
        Ok(())
    }

    fn directed_children(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
        self.adj[id.as_usize()]
            .iter()
            .filter(|e| matches!((e.at_self, e.at_neighbor), (Endpoint::Tail, Endpoint::Arrow)))
            .map(|e| e.neighbor)
            .collect()
    }

    /// Directed reachability.
    #[must_use]
    pub fn reaches_directed(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        if from == to {
            return true;
        }
        let mut ws = GraphWorkspace::default();
        ws.prepare(self.node_count());
        ws.frontier.push(from);
        ws.visited.insert(from);
        while let Some(u) = ws.frontier.pop() {
            for c in self.directed_children(u) {
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

    /// View as static [`Pag`] for algorithms that only need adjacency/marks.
    ///
    /// Nodes become `Static(VariableId::from_raw(dense))` — only for algorithm reuse.
    #[must_use]
    pub fn as_static_pag_for_alg(&self) -> Pag {
        let n = self.node_count() as u32;
        let mut p = Pag::with_variables(n);
        for i in 0..self.node_count() {
            let a = DenseNodeId::from_raw(i as u32);
            for e in &self.adj[i] {
                if e.neighbor.raw() < a.raw() {
                    continue;
                }
                let edge = MarkedEdge {
                    a,
                    b: e.neighbor,
                    at_a: e.at_self,
                    at_b: e.at_neighbor,
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
    ) -> Result<Vec<DefiniteStatusPath>, GraphError> {
        self.as_static_pag_for_alg()
            .definite_status_paths(x, y, max_paths, max_len)
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
            let a = DenseNodeId::from_raw(i as u32);
            for (b, at_a, at_b) in graph.neighbors(a) {
                if b.raw() < a.raw() {
                    continue;
                }
                if matches!(at_a, Endpoint::Circle) || matches!(at_b, Endpoint::Circle) {
                    pending.push((a, b));
                }
            }
        }
        Self {
            graph,
            pending_circles: Arc::from(pending),
            algorithm: algorithm.into(),
        }
    }

    /// Whether no circle marks remain.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.pending_circles.is_empty()
    }
}
