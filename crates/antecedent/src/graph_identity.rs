//! The single definition of "the same graph" for frozen-operation checks.
//!
//! A checked operation freezes the graph it was sealed against and is later
//! asked whether the graph in hand is that one. `PartialEq` on [`Dag`] also
//! compares adjacency-list order, which depends on edge insertion order and
//! says nothing about the causal structure, so the checks compare the node
//! list and the sorted directed edge set instead.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::{Dag, NodeRef, TemporalDag};

/// A graph that exposes its node list and sorted directed edge pairs.
pub(crate) trait NodesAndEdges {
    /// The node list in storage order.
    fn node_list(&self) -> &[NodeRef];
    /// Directed edges as ascending `(parent, child)` dense-id pairs.
    fn sorted_edges(&self) -> Vec<(u32, u32)>;
}

impl NodesAndEdges for Dag {
    fn node_list(&self) -> &[NodeRef] {
        self.nodes()
    }
    fn sorted_edges(&self) -> Vec<(u32, u32)> {
        self.directed_edge_pairs_sorted()
    }
}

impl NodesAndEdges for TemporalDag {
    fn node_list(&self) -> &[NodeRef] {
        self.nodes()
    }
    fn sorted_edges(&self) -> Vec<(u32, u32)> {
        self.directed_edge_pairs_sorted()
    }
}

fn same_structure<G: NodesAndEdges>(a: &G, b: &G) -> bool {
    a.node_list() == b.node_list() && a.sorted_edges() == b.sorted_edges()
}

/// Whether `a` and `b` have the same nodes and the same directed edge set.
pub(crate) fn same_dag_structure(a: &Dag, b: &Dag) -> bool {
    same_structure(a, b)
}

/// Whether `a` and `b` have the same nodes and the same directed edge set.
pub(crate) fn same_temporal_dag_structure(a: &TemporalDag, b: &TemporalDag) -> bool {
    same_structure(a, b)
}

/// The sorted directed edge set of a graph frozen into an operation, computed once so
/// each later check sorts only the graph in hand.
#[derive(Clone, Debug)]
pub(crate) struct FrozenEdges(Vec<(u32, u32)>);

impl FrozenEdges {
    /// Freeze the edge set of `graph`.
    pub(crate) fn of<G: NodesAndEdges>(graph: &G) -> Self {
        Self(graph.sorted_edges())
    }

    /// Whether `other` has the nodes of `frozen` (the graph these edges were taken from)
    /// and exactly the frozen edge set.
    pub(crate) fn matches<G: NodesAndEdges>(&self, frozen: &G, other: &G) -> bool {
        other.node_list() == frozen.node_list() && other.sorted_edges() == self.0
    }
}
