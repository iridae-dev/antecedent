//! Temporal DAG over lagged variable nodes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::TemporalNodeKey;
use antecedent_core::{Lag, VariableId};

use crate::directed::DirectedCore;
use crate::error::GraphError;
use crate::types::{DenseNodeId, MarkedEdge, NodeRef};
use crate::workspace::GraphWorkspace;

/// Directed acyclic graph over lagged (`VariableId`, `Lag`) nodes.
///
/// Directed nodes and edges are the shared directed core an [`Admg`](crate::Admg)
/// also wraps. This type adds the lag rules.
#[derive(Clone, Debug)]
pub struct TemporalDag {
    directed: DirectedCore,
}

impl PartialEq for TemporalDag {
    fn eq(&self, other: &Self) -> bool {
        self.directed.same_directed(&other.directed)
    }
}

impl Eq for TemporalDag {}

impl TemporalDag {
    /// Empty temporal DAG.
    #[must_use]
    pub fn empty() -> Self {
        Self { directed: DirectedCore::new() }
    }

    /// Node count.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.directed.node_count()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.directed.is_empty()
    }

    /// Nodes in dense order.
    #[must_use]
    pub fn nodes(&self) -> &[NodeRef] {
        self.directed.nodes()
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
        self.directed.push_labeled(node)
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
    /// `from`'s lag must be greater than or equal to `to`'s lag: larger `Lag`
    /// values sit further in the past (`Lag::CONTEMPORANEOUS` is the present),
    /// so an edge is only valid running from the past (or same time) toward the
    /// present. An edge whose source is nearer the present than its target
    /// would point from the future into the past and is rejected.
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
        ) = (self.nodes()[from.as_usize()], self.nodes()[to.as_usize()])
        {
            if v1 == v2 && l1 == l2 && l1.is_contemporaneous() {
                return Err(GraphError::ContemporaneousSelfEdge { variable: v1 });
            }
            crate::types::reject_future_to_past(self.nodes(), from, to)?;
        }
        self.directed.try_insert(from, to)
    }

    /// Insert directed edges, using one acyclicity check when every edge is legal.
    ///
    /// A temporal rejection, duplicate, or cycle is inserted one edge at a time,
    /// so the prefix and the error match [`Self::insert_directed`].
    pub(crate) fn insert_directed_edges(
        &mut self,
        edges: &[(DenseNodeId, DenseNodeId)],
    ) -> Result<(), GraphError> {
        if edges.iter().all(|&(from, to)| self.temporal_precheck(from, to))
            && self.directed.accepts_new_edges(edges)
        {
            self.directed.push_edges(edges);
            return Ok(());
        }
        for &(from, to) in edges {
            self.insert_directed(from, to)?;
        }
        Ok(())
    }

    /// Whether `insert_directed` would get past the temporal rules to the cycle check.
    fn temporal_precheck(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        if self.validate_node(from).is_err() || self.validate_node(to).is_err() {
            return false;
        }
        let (NodeRef::Lagged { variable: v1, lag: l1 }, NodeRef::Lagged { variable: v2, lag: l2 }) =
            (self.nodes()[from.as_usize()], self.nodes()[to.as_usize()])
        else {
            return false;
        };
        if v1 == v2 && l1 == l2 && l1.is_contemporaneous() {
            return false;
        }
        crate::types::reject_future_to_past(self.nodes(), from, to).is_ok()
    }

    /// Children.
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
        &self.directed.child_adj()[id.as_usize()]
    }

    /// Parents.
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
        &self.directed.parent_adj()[id.as_usize()]
    }

    /// Iterate directed edges as marked edges.
    pub fn edges(&self) -> impl Iterator<Item = MarkedEdge> + '_ {
        self.directed.marked_edges()
    }

    /// Directed edges as raw `(parent, child)` dense-id pairs in ascending order.
    #[must_use]
    pub fn directed_edge_pairs_sorted(&self) -> Vec<(u32, u32)> {
        self.directed.edge_pairs_sorted()
    }

    /// Reachability.
    #[must_use]
    pub fn reaches(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        self.directed.reaches(from, to)
    }

    /// Reachability with a reusable workspace.
    pub fn reaches_with(
        &self,
        from: DenseNodeId,
        to: DenseNodeId,
        ws: &mut GraphWorkspace,
    ) -> bool {
        self.directed.reaches_with(from, to, ws)
    }

    /// Map dense id to a serializable [`TemporalNodeKey`].
    #[must_use]
    pub fn temporal_key(&self, id: DenseNodeId) -> Option<TemporalNodeKey> {
        match self.nodes().get(id.as_usize())? {
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
    fn directed_edge_pairs_are_sorted_dense_pairs() {
        let mut g = TemporalDag::empty();
        assert!(g.directed_edge_pairs_sorted().is_empty());
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
        let c = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(b, c).unwrap();
        g.insert_directed(a, c).unwrap();
        g.insert_directed(a, b).unwrap();
        assert_eq!(g.directed_edge_pairs_sorted(), vec![(0, 1), (0, 2), (1, 2)]);
    }

    #[test]
    fn allows_lagged_self_edge() {
        let mut g = TemporalDag::empty();
        let past = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(past, now).unwrap();
        assert!(g.reaches(past, now));
    }

    #[test]
    fn rejects_future_to_past_edge() {
        let mut g = TemporalDag::empty();
        let past = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        // `now` (lag 0) -> `past` (lag 1) points from the present into the past.
        assert!(matches!(g.insert_directed(now, past), Err(GraphError::FutureToPast { .. })));
    }

    #[test]
    fn batch_insert_matches_one_edge_at_a_time() {
        let mut nodes = TemporalDag::empty();
        let past = nodes.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = nodes.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let other = nodes.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        let cases = [
            vec![(past, now), (past, other)],
            vec![(now, past)],
            vec![(now, now)],
            vec![(past, now), (past, now)],
        ];
        for edges in cases {
            let mut one = nodes.clone();
            let one_err = edges.iter().try_for_each(|&(from, to)| one.insert_directed(from, to));
            let mut batch = nodes.clone();
            let batch_err = batch.insert_directed_edges(&edges);
            assert_eq!(one_err, batch_err);
            assert_eq!(one, batch);
        }
    }
}
