//! Temporal DAG over lagged variable nodes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::TemporalNodeKey;
use causal_core::{Lag, VariableId};

use crate::error::GraphError;
use crate::types::{DenseNodeId, MarkedEdge, NodeRef};
use crate::workspace::GraphWorkspace;

/// Directed acyclic graph over lagged (`VariableId`, `Lag`) nodes.
#[derive(Clone, Debug)]
pub struct TemporalDag {
    nodes: Vec<NodeRef>,
    children: Vec<Vec<DenseNodeId>>,
    parents: Vec<Vec<DenseNodeId>>,
}

impl TemporalDag {
    /// Empty temporal DAG.
    #[must_use]
    pub fn empty() -> Self {
        Self { nodes: Vec::new(), children: Vec::new(), parents: Vec::new() }
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
                    message: "TemporalDag accepts only Lagged nodes",
                });
            }
        }
        let id = u32::try_from(self.nodes.len()).map_err(|_| GraphError::TooManyNodes)?;
        self.nodes.push(node);
        self.children.push(Vec::new());
        self.parents.push(Vec::new());
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

    /// Insert directed edge with temporal rules.
    ///
    /// Contemporaneous self-edges are rejected. A self-loop on a single dense
    /// node is always a [`GraphError::Cycle`]; lagged self-influence is modeled
    /// as an edge between two distinct nodes (e.g. `X@t-1 -> X@t`).
    ///
    /// # Errors
    ///
    /// Unknown nodes, duplicates, cycles, or contemporaneous self-edges.
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
        }
        if self.children[from.as_usize()].contains(&to) {
            return Err(GraphError::DuplicateEdge { from: from.raw(), to: to.raw() });
        }
        if self.reaches(to, from) {
            return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
        }
        self.children[from.as_usize()].push(to);
        self.parents[to.as_usize()].push(from);
        Ok(())
    }

    /// Children.
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
        &self.children[id.as_usize()]
    }

    /// Iterate directed edges as marked edges.
    pub fn edges(&self) -> impl Iterator<Item = MarkedEdge> + '_ {
        self.children.iter().enumerate().flat_map(|(i, kids)| {
            let from = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
            kids.iter().map(move |&to| MarkedEdge::directed(from, to))
        })
    }

    /// Reachability.
    #[must_use]
    pub fn reaches(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        if from == to {
            return true;
        }
        let mut ws = GraphWorkspace::default();
        ws.prepare(self.node_count());
        ws.frontier.push(from);
        ws.visited.insert(from);
        while let Some(n) = ws.frontier.pop() {
            for &c in self.children(n) {
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
    fn rejects_contemporaneous_self_edge() {
        let mut g = TemporalDag::empty();
        let n = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        assert!(matches!(g.insert_directed(n, n), Err(GraphError::ContemporaneousSelfEdge { .. })));
    }

    #[test]
    fn allows_lagged_self_edge() {
        let mut g = TemporalDag::empty();
        let past = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(past, now).unwrap();
        assert!(g.reaches(past, now));
    }
}
