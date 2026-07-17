//! CPDAG and temporal CPDAG .
//!
//! Undirected contemporaneous marks use [`Endpoint::Tail`]–[`Endpoint::Tail`].
//! Orientation conflicts use [`Endpoint::Conflict`]–[`Endpoint::Conflict`] (`x-x`).
//! [`Endpoint::Circle`] is rejected (reserved for PAG/LPCMCI).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use causal_core::{Lag, TemporalNodeKey, VariableId};

use crate::error::GraphError;
use crate::marked_storage::{self, AdjEntry};
use crate::temporal::TemporalDag;
use crate::types::{DenseNodeId, Endpoint, MarkedEdge, NodeRef};
use crate::workspace::GraphWorkspace;

/// Temporal CPDAG over lagged variable nodes (DESIGN §6.2).
#[derive(Clone, Debug)]
pub struct TemporalCpdag {
    nodes: Vec<NodeRef>,
    adj: Vec<Vec<AdjEntry>>,
}

impl TemporalCpdag {
    /// Empty temporal CPDAG.
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

    /// Add a lagged or context node.
    ///
    /// Context nodes are allowed for J-PCMCI+ / multi-dataset graphs .
    /// Static nodes remain rejected (use a static CPDAG / DAG).
    ///
    /// # Errors
    ///
    /// Static refs or capacity overflow.
    pub fn add_node(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        match node {
            NodeRef::Lagged { .. } | NodeRef::Context { .. } => {}
            NodeRef::Static(_) => {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalCpdag accepts Lagged or Context nodes (not Static)",
                });
            }
        }
        let id = u32::try_from(self.nodes.len()).map_err(|_| GraphError::TooManyNodes)?;
        self.nodes.push(node);
        self.adj.push(Vec::new());
        Ok(DenseNodeId::from_raw(id))
    }

    /// Convenience: add `variable` at `lag`.
    ///
    /// # Errors
    ///
    /// Capacity overflow.
    pub fn add_lagged(
        &mut self,
        variable: VariableId,
        lag: Lag,
    ) -> Result<DenseNodeId, GraphError> {
        self.add_node(NodeRef::Lagged { variable, lag })
    }

    /// Convenience: add a context node (optional environment tag).
    ///
    /// # Errors
    ///
    /// Capacity overflow.
    pub fn add_context(
        &mut self,
        variable: VariableId,
        environment: Option<causal_core::EnvironmentId>,
    ) -> Result<DenseNodeId, GraphError> {
        self.add_node(NodeRef::Context { variable, environment })
    }

    /// Insert a CPDAG-legal marked edge (directed or undirected).
    ///
    /// # Errors
    ///
    /// Unknown nodes, duplicates, illegal marks, contemporaneous self-edges,
    /// lagged self-loops (reported as [`GraphError::Cycle`]), or directed cycles.
    pub fn insert_marked(&mut self, edge: MarkedEdge) -> Result<(), GraphError> {
        if !edge.is_cpdag_legal() {
            return Err(GraphError::InvalidEndpoints {
                message: "CPDAG accepts only Tail–Arrow, Tail–Tail, or Conflict–Conflict marks",
            });
        }
        self.validate_node(edge.a)?;
        self.validate_node(edge.b)?;
        // Match TemporalDag::insert_directed: compare NodeRefs, not dense ids,
        // so duplicate nodes at the same (variable, lag) are also self-edges.
        if let (
            NodeRef::Lagged { variable: v1, lag: l1 },
            NodeRef::Lagged { variable: v2, lag: l2 },
        ) = (self.nodes[edge.a.as_usize()], self.nodes[edge.b.as_usize()])
        {
            if v1 == v2 && l1 == l2 && l1.is_contemporaneous() {
                return Err(GraphError::ContemporaneousSelfEdge { variable: v1 });
            }
        }
        if edge.a == edge.b {
            // A lagged self-loop on a single node is a cycle once oriented,
            // matching TemporalDag which reports Cycle for dense self-loops.
            return Err(GraphError::Cycle { from: edge.a.raw(), to: edge.b.raw() });
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

    /// Insert directed edge `from -> to`.
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

    /// Insert undirected edge `a — b`.
    ///
    /// # Errors
    ///
    /// See [`Self::insert_marked`].
    pub fn insert_undirected(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        self.insert_marked(MarkedEdge::undirected(a, b))
    }

    /// Orient an existing undirected edge as `from -> to`.
    ///
    /// # Errors
    ///
    /// Missing edge, already directed, cycle, or unknown nodes.
    pub fn orient_undirected(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
    ) -> Result<(), GraphError> {
        self.validate_node(from)?;
        self.validate_node(to)?;
        let Some(edge) = self.edge_between(from, to) else {
            return Err(GraphError::UnknownNode { id: from.raw() });
        };
        if !edge.is_undirected() {
            return Err(GraphError::InvalidEndpoints {
                message: "orient_undirected requires an undirected Tail–Tail edge",
            });
        }
        if self.reaches_directed(to, from) {
            return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
        }
        self.set_marks(from, to, Endpoint::Tail, Endpoint::Arrow)?;
        Ok(())
    }

    /// Mark an existing edge as a Tigramite `x-x` conflict ([`Endpoint::Conflict`]–[`Endpoint::Conflict`]).
    ///
    /// # Errors
    ///
    /// Missing edge or unknown nodes.
    pub fn mark_conflict(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        self.validate_node(a)?;
        self.validate_node(b)?;
        if self.edge_between(a, b).is_none() {
            return Err(GraphError::UnknownNode { id: a.raw() });
        }
        self.set_marks(a, b, Endpoint::Conflict, Endpoint::Conflict)
    }

    /// Whether any edge exists between `a` and `b`.
    #[must_use]
    pub fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool {
        self.edge_between(a, b).is_some()
    }

    /// Marked edge between `a` and `b` if present (marks oriented from `a`'s perspective as `at_a`).
    #[must_use]
    pub fn edge_between(&self, a: DenseNodeId, b: DenseNodeId) -> Option<MarkedEdge> {
        marked_storage::edge_between(&self.adj, a, b)
    }

    /// All marked edges (each undirected/directed pair once, with `a.raw() <= b.raw()` for undirected
    /// and parent-first for directed).
    #[must_use]
    pub fn edges(&self) -> Vec<MarkedEdge> {
        let mut out = Vec::new();
        for (i, nbrs) in self.adj.iter().enumerate() {
            let a = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
            for e in nbrs {
                if a.raw() < e.neighbor.raw()
                    || (a.raw() == e.neighbor.raw()
                        && matches!((e.at_self, e.at_neighbor), (Endpoint::Tail, Endpoint::Arrow)))
                {
                    out.push(MarkedEdge { a, b: e.neighbor, at_a: e.at_self, at_b: e.at_neighbor, middle: e.middle });
                } else if a.raw() > e.neighbor.raw() {
                    // skip reverse half
                } else if matches!((e.at_self, e.at_neighbor), (Endpoint::Arrow, Endpoint::Tail)) {
                    // directed stored from child side only when a > neighbor — emit parent-first
                    out.push(MarkedEdge::directed(e.neighbor, a));
                }
            }
        }
        out.sort_by_key(|e| (e.a.raw(), e.b.raw(), e.at_a as u8, e.at_b as u8));
        out.dedup();
        out
    }

    /// Directed children of `id` (outgoing arrows).
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
        marked_storage::directed_children(&self.adj, id).collect()
    }

    /// Directed parents of `id` (incoming arrows).
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
        if id.as_usize() >= self.node_count() {
            return Vec::new();
        }
        self.adj[id.as_usize()]
            .iter()
            .filter(|e| matches!((e.at_self, e.at_neighbor), (Endpoint::Arrow, Endpoint::Tail)))
            .map(|e| e.neighbor)
            .collect()
    }

    /// Undirected neighbors of `id`.
    #[must_use]
    pub fn undirected_neighbors(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
        if id.as_usize() >= self.node_count() {
            return Vec::new();
        }
        self.adj[id.as_usize()]
            .iter()
            .filter(|e| matches!((e.at_self, e.at_neighbor), (Endpoint::Tail, Endpoint::Tail)))
            .map(|e| e.neighbor)
            .collect()
    }

    /// Borrowed directed-child iterator (orientation hot path).
    pub fn children_iter(&self, id: DenseNodeId) -> impl Iterator<Item = DenseNodeId> + '_ {
        marked_storage::directed_children(&self.adj, id)
    }

    /// Build from a directed [`TemporalDag`] (all edges become directed marks).
    #[must_use]
    pub fn from_temporal_dag(dag: &TemporalDag) -> Self {
        let mut g = Self::empty();
        for node in dag.nodes() {
            let _ = g.add_node(*node);
        }
        for e in dag.edges() {
            if let Some((from, to)) = e.parent_child() {
                let _ = g.insert_directed(from, to);
            }
        }
        g
    }

    /// Extract a directed [`TemporalDag`] from directed edges only (drops undirected).
    ///
    /// # Errors
    ///
    /// Propagates insert failures (should not happen for a valid CPDAG's directed subset).
    pub fn to_directed_skeleton(&self) -> Result<TemporalDag, GraphError> {
        let mut dag = TemporalDag::empty();
        for node in &self.nodes {
            dag.add_node(*node)?;
        }
        for e in self.edges() {
            if let Some((from, to)) = e.parent_child() {
                dag.insert_directed(from, to)?;
            }
        }
        Ok(dag)
    }

    /// Convert to a [`TemporalDag`] only when no undirected or conflict edges remain.
    ///
    /// # Errors
    ///
    /// [`GraphError::InvalidEndpoints`] if any Tail–Tail or Conflict–Conflict edge remains,
    /// or insert failure.
    pub fn try_into_temporal_dag(&self) -> Result<TemporalDag, GraphError> {
        for e in self.edges() {
            if e.is_undirected() {
                return Err(GraphError::InvalidEndpoints {
                    message: "cannot complete TemporalCpdag to TemporalDag while undirected edges remain",
                });
            }
            if e.is_conflict() {
                return Err(GraphError::InvalidEndpoints {
                    message: "cannot complete TemporalCpdag to TemporalDag while conflict edges remain",
                });
            }
        }
        self.to_directed_skeleton()
    }

    /// Count conflict (`x-x`) edges.
    #[must_use]
    pub fn conflict_edge_count(&self) -> usize {
        self.edges().iter().filter(|e| e.is_conflict()).count()
    }

    /// Map dense id to a serializable [`TemporalNodeKey`].
    #[must_use]
    pub fn temporal_key(&self, id: DenseNodeId) -> Option<TemporalNodeKey> {
        match self.nodes.get(id.as_usize())? {
            NodeRef::Lagged { variable, lag } => {
                let offset = -i32::try_from(lag.raw()).ok()?;
                Some(TemporalNodeKey { variable: *variable, offset })
            }
            _ => None,
        }
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

    fn reaches_directed(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        let mut ws = GraphWorkspace::default();
        marked_storage::reaches_directed(&self.adj, &mut ws, from, to)
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

    fn set_marks(
        &mut self,
        a: DenseNodeId,
        b: DenseNodeId,
        at_a: Endpoint,
        at_b: Endpoint,
    ) -> Result<(), GraphError> {
        marked_storage::set_marks(&mut self.adj, a, b, at_a, at_b)
    }

    fn validate_node(&self, id: DenseNodeId) -> Result<(), GraphError> {
        if id.as_usize() >= self.node_count() {
            Err(GraphError::UnknownNode { id: id.raw() })
        } else {
            Ok(())
        }
    }
}

/// Alias for documentation / DESIGN naming (`Cpdag` over temporal nodes).
pub type Cpdag = TemporalCpdag;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undirected_then_orient() {
        let mut g = TemporalCpdag::empty();
        let x = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_undirected(x, y).unwrap();
        assert!(g.edge_between(x, y).unwrap().is_undirected());
        g.orient_undirected(x, y).unwrap();
        let e = g.edge_between(x, y).unwrap();
        assert_eq!(e.parent_child(), Some((x, y)));
        assert_eq!(g.children(x), vec![y]);
        assert!(g.undirected_neighbors(x).is_empty());
    }

    #[test]
    fn mark_conflict_sets_x_x() {
        let mut g = TemporalCpdag::empty();
        let x = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_undirected(x, y).unwrap();
        g.mark_conflict(x, y).unwrap();
        let e = g.edge_between(x, y).unwrap();
        assert!(e.is_conflict());
        assert_eq!(g.conflict_edge_count(), 1);
        assert!(g.undirected_neighbors(x).is_empty());
        assert!(g.try_into_temporal_dag().is_err());
    }

    #[test]
    fn rejects_circle_marks() {
        let mut g = TemporalCpdag::empty();
        let x = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let y = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let edge = MarkedEdge {
            a: x,
            b: y,
            at_a: Endpoint::Circle,
            at_b: Endpoint::Arrow,
            middle: crate::types::MiddleMark::Empty,
        };
        assert!(matches!(g.insert_marked(edge), Err(GraphError::InvalidEndpoints { .. })));
    }

    #[test]
    fn from_temporal_dag_preserves_directed() {
        let mut dag = TemporalDag::empty();
        let a = dag.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let b = dag.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        dag.insert_directed(a, b).unwrap();
        let cpdag = TemporalCpdag::from_temporal_dag(&dag);
        assert_eq!(cpdag.edge_between(a, b).unwrap().parent_child(), Some((a, b)));
        let back = cpdag.to_directed_skeleton().unwrap();
        assert!(back.reaches(a, b));
    }

    #[test]
    fn rejects_contemporaneous_self_edges() {
        let mut g = TemporalCpdag::empty();
        let x = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        assert!(matches!(
            g.insert_undirected(x, x),
            Err(GraphError::ContemporaneousSelfEdge { .. })
        ));
        assert!(matches!(g.insert_directed(x, x), Err(GraphError::ContemporaneousSelfEdge { .. })));
        // Duplicate node at the same (variable, lag) is still a self-edge.
        let x2 = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        assert!(matches!(
            g.insert_directed(x, x2),
            Err(GraphError::ContemporaneousSelfEdge { .. })
        ));
        assert!(g.edges().is_empty());
    }

    #[test]
    fn rejects_lagged_self_loops() {
        let mut g = TemporalCpdag::empty();
        let x1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        assert!(matches!(g.insert_undirected(x1, x1), Err(GraphError::Cycle { .. })));
        assert!(matches!(g.insert_directed(x1, x1), Err(GraphError::Cycle { .. })));
        assert_eq!(g.undirected_edge_count(), 0);
        assert!(g.edges().is_empty());
    }

    #[test]
    fn accepts_context_nodes_without_coercing() {
        use causal_core::EnvironmentId;
        let mut g = TemporalCpdag::empty();
        let c = g.add_context(VariableId::from_raw(0), Some(EnvironmentId::from_raw(1))).unwrap();
        let y = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        match g.nodes()[c.as_usize()] {
            NodeRef::Context { variable, environment } => {
                assert_eq!(variable, VariableId::from_raw(0));
                assert_eq!(environment, Some(EnvironmentId::from_raw(1)));
            }
            _ => panic!("expected Context node"),
        }
        g.insert_directed(c, y).unwrap();
        assert_eq!(g.edge_between(c, y).unwrap().parent_child(), Some((c, y)));
        assert!(g.add_node(NodeRef::Static(VariableId::from_raw(2))).is_err());
    }

    #[test]
    fn marked_edge_undirected_canonical() {
        let a = DenseNodeId::from_raw(2);
        let b = DenseNodeId::from_raw(1);
        let e = MarkedEdge::undirected(a, b);
        assert!(e.is_undirected());
        assert!(e.is_cpdag_legal());
        assert_eq!(e.a.raw(), 1);
        assert_eq!(e.b.raw(), 2);
    }
}
