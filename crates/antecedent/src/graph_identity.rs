//! The single definition of "the same graph" for frozen-operation checks.
//!
//! A checked operation freezes the graph it was sealed against and is later
//! asked whether the graph in hand is that one. `PartialEq` on [`Dag`] also
//! compares adjacency-list order, which depends on edge insertion order and
//! says nothing about the causal structure, so the checks compare the node
//! list and the sorted directed edge set instead.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::{Dag, TemporalDag};

/// Whether `a` and `b` have the same nodes and the same directed edge set.
pub(crate) fn same_dag_structure(a: &Dag, b: &Dag) -> bool {
    a.nodes() == b.nodes() && a.directed_edge_pairs_sorted() == b.directed_edge_pairs_sorted()
}

/// Whether `a` and `b` have the same nodes and the same directed edge set.
pub(crate) fn same_temporal_dag_structure(a: &TemporalDag, b: &TemporalDag) -> bool {
    a.nodes() == b.nodes() && a.directed_edge_pairs_sorted() == b.directed_edge_pairs_sorted()
}
