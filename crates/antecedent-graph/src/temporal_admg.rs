//! Temporal ADMG: lagged nodes, directed edges, and bidirected latent edges.
//!
//! A [`TemporalAdmg`] whose variables each appear once projects onto a static
//! [`Admg`] with the same variable ids and the same mixed-graph edges. General
//! identification of that projection is the same functional. A variable that
//! occupies more than one lag does not have that projection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;

use antecedent_core::{Lag, TemporalNodeKey, VariableId};

use crate::admg::Admg;
use crate::algo::bfs_reaches;
use crate::error::GraphError;
use crate::types::{DenseNodeId, MarkedEdge, NodeRef};
use crate::workspace::GraphWorkspace;

/// Acyclic directed mixed graph over lagged (`VariableId`, `Lag`) nodes.
#[derive(Clone, Debug)]
pub struct TemporalAdmg {
    nodes: Vec<NodeRef>,
    children: Vec<Vec<DenseNodeId>>,
    parents: Vec<Vec<DenseNodeId>>,
    bidirected: Vec<Vec<DenseNodeId>>,
    insert_ws: GraphWorkspace,
}

impl TemporalAdmg {
    /// Empty temporal ADMG.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            children: Vec::new(),
            parents: Vec::new(),
            bidirected: Vec::new(),
            insert_ws: GraphWorkspace::default(),
        }
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
    /// Non-lagged node refs or capacity overflow.
    pub fn add_node(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        match node {
            NodeRef::Lagged { .. } => {}
            _ => {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg accepts only Lagged nodes",
                });
            }
        }
        let id = u32::try_from(self.nodes.len()).map_err(|_| GraphError::TooManyNodes)?;
        self.nodes.push(node);
        self.children.push(Vec::new());
        self.parents.push(Vec::new());
        self.bidirected.push(Vec::new());
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

    /// Insert directed edge with the same temporal rules as [`crate::TemporalDag`].
    ///
    /// # Errors
    ///
    /// Unknown nodes, duplicates, cycles, contemporaneous self-edges, or edges
    /// that point from the future into the past.
    pub fn insert_directed(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
    ) -> Result<(), GraphError> {
        self.validate_node(from)?;
        self.validate_node(to)?;
        if let (
            NodeRef::Lagged { variable: v1, lag: l1 },
            NodeRef::Lagged { variable: v2, lag: l2 },
        ) = (self.nodes[from.as_usize()], self.nodes[to.as_usize()])
        {
            if v1 == v2 && l1 == l2 && l1.is_contemporaneous() {
                return Err(GraphError::ContemporaneousSelfEdge { variable: v1 });
            }
            crate::types::reject_future_to_past(&self.nodes, from, to)?;
        }
        if self.children[from.as_usize()].contains(&to) {
            return Err(GraphError::DuplicateEdge { from: from.raw(), to: to.raw() });
        }
        let mut ws = core::mem::take(&mut self.insert_ws);
        let cycle = bfs_reaches(&self.children, to, from, &mut ws);
        self.insert_ws = ws;
        if cycle {
            return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
        }
        self.children[from.as_usize()].push(to);
        self.parents[to.as_usize()].push(from);
        Ok(())
    }

    /// Insert bidirected edge `a ↔ b` (latent confounder).
    ///
    /// Bidirected edges have no time direction. Distinct lagged nodes may be
    /// joined, including across lags.
    ///
    /// # Errors
    ///
    /// Unknown nodes, self-loop, or duplicate.
    pub fn insert_bidirected(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        self.validate_node(a)?;
        self.validate_node(b)?;
        if a == b {
            return Err(GraphError::InvalidEndpoints {
                message: "TemporalAdmg rejects bidirected self-loops",
            });
        }
        if self.bidirected[a.as_usize()].contains(&b) {
            return Err(GraphError::DuplicateEdge { from: a.raw(), to: b.raw() });
        }
        self.bidirected[a.as_usize()].push(b);
        self.bidirected[b.as_usize()].push(a);
        Ok(())
    }

    /// Children.
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.children.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Parents.
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.parents.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Bidirected neighbors.
    #[must_use]
    pub fn bidirected_neighbors(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.bidirected.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Whether any bidirected edge is present.
    #[must_use]
    pub fn has_bidirected(&self) -> bool {
        self.bidirected.iter().any(|neighbors| !neighbors.is_empty())
    }

    /// Directed and bidirected edges as marked edges.
    #[must_use]
    pub fn edges(&self) -> Vec<MarkedEdge> {
        let mut edges = Vec::new();
        for (i, kids) in self.children.iter().enumerate() {
            let from = DenseNodeId::from_index(i);
            for &to in kids {
                edges.push(MarkedEdge::directed(from, to));
            }
        }
        for (i, neighbors) in self.bidirected.iter().enumerate() {
            let a = DenseNodeId::from_index(i);
            for &b in neighbors {
                if a < b {
                    edges.push(MarkedEdge::bidirected(a, b));
                }
            }
        }
        edges
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

    /// Project onto a static [`Admg`] when that projection is the same functional.
    ///
    /// Each variable may appear once. The static node keeps that variable id, and
    /// directed and bidirected edges keep their dense endpoints. General
    /// identification then addresses the same variables on the same mixed graph.
    ///
    /// # Errors
    ///
    /// A variable occupies more than one lag, or a node is not lagged. That graph
    /// does not have this projection.
    pub fn project_preserving_functional(&self) -> Result<Admg, GraphError> {
        let mut seen = HashSet::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let NodeRef::Lagged { variable, .. } = node else {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg projection requires lagged nodes",
                });
            };
            if !seen.insert(*variable) {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg projection preserves the functional only when each \
                              variable appears once",
                });
            }
        }
        let mut admg = Admg::empty();
        for node in &self.nodes {
            let NodeRef::Lagged { variable, .. } = node else { unreachable!("checked above") };
            admg.add_node(NodeRef::Static(*variable))?;
        }
        for (i, kids) in self.children.iter().enumerate() {
            let from = DenseNodeId::from_index(i);
            for &to in kids {
                admg.insert_directed(from, to)?;
            }
        }
        for (i, neighbors) in self.bidirected.iter().enumerate() {
            let a = DenseNodeId::from_index(i);
            for &b in neighbors {
                if a < b {
                    admg.insert_bidirected(a, b)?;
                }
            }
        }
        Ok(admg)
    }

    /// Lift a static [`Admg`] to contemporaneous lagged nodes.
    ///
    /// Variable ids and dense order are unchanged, so
    /// [`Self::project_preserving_functional`] returns an isomorphic graph.
    ///
    /// # Errors
    ///
    /// The static graph contains a non-static node, or edge insertion fails.
    pub fn from_static_contemporaneous(admg: &Admg) -> Result<Self, GraphError> {
        let mut graph = Self::empty();
        for node in admg.nodes() {
            let NodeRef::Static(variable) = node else {
                return Err(GraphError::InvalidEndpoints {
                    message: "TemporalAdmg lift accepts only static ADMG nodes",
                });
            };
            graph.add_lagged(*variable, Lag::CONTEMPORANEOUS)?;
        }
        for i in 0..admg.node_count() {
            let from = DenseNodeId::from_index(i);
            for &to in admg.children(from) {
                graph.insert_directed(from, to)?;
            }
            for &to in admg.bidirected_neighbors(from) {
                if from < to {
                    graph.insert_bidirected(from, to)?;
                }
            }
        }
        Ok(graph)
    }

    fn validate_node(&self, id: DenseNodeId) -> Result<(), GraphError> {
        if id.as_usize() >= self.node_count() {
            Err(GraphError::UnknownNode { id: id.raw() })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contemporaneous_lift_projects_to_the_same_mixed_graph() {
        let mut admg = Admg::with_variables(3);
        admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        admg.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let temporal = TemporalAdmg::from_static_contemporaneous(&admg).unwrap();
        let projected = temporal.project_preserving_functional().unwrap();
        assert!(projected.has_bidirected());
        assert_eq!(projected.children(DenseNodeId::from_raw(0)), &[DenseNodeId::from_raw(1)]);
        assert_eq!(
            projected.bidirected_neighbors(DenseNodeId::from_raw(0)),
            &[DenseNodeId::from_raw(2)]
        );
    }

    #[test]
    fn repeated_variable_does_not_project() {
        let mut graph = TemporalAdmg::empty();
        let past = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = graph.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(past, now).unwrap();
        assert!(graph.project_preserving_functional().is_err());
    }

    #[test]
    fn rejects_future_to_past_directed_edge() {
        let mut graph = TemporalAdmg::empty();
        let past = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        assert!(matches!(graph.insert_directed(now, past), Err(GraphError::FutureToPast { .. })));
    }
}
