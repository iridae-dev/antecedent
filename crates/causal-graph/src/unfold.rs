//! Lazy finite unfolding of temporal DAGs and graph-review artifacts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]

use std::sync::Arc;

use causal_core::VariableId;
use causal_data::{TemporalIndexer, TemporalNodeKey};

use crate::dag::Dag;
use crate::error::GraphError;
use crate::temporal::TemporalDag;
use crate::types::{DenseNodeId, NodeRef};

/// Result of unfolding a [`TemporalDag`] over a finite [`TemporalIndexer`] window.
#[derive(Clone, Debug)]
pub struct UnfoldedTemporalGraph {
    /// Static DAG whose dense ids match [`TemporalIndexer::dense_id`].
    pub dag: Dag,
    /// Indexer used for the unfolding.
    pub indexer: TemporalIndexer,
}

impl TemporalDag {
    /// Unfold lagged summary edges into a finite static DAG.
    ///
    /// For each directed edge `A(τ_from) → B(τ_to)` and every absolute time `t`
    /// where both endpoints lie in the indexer window, inserts
    /// `A@t+offset_from → B@t+offset_to` with `offset = -lag`.
    ///
    /// # Errors
    ///
    /// Unknown/non-lagged nodes, indexer construction issues, or cycle insertion.
    pub fn unfold(&self, indexer: TemporalIndexer) -> Result<UnfoldedTemporalGraph, GraphError> {
        let n = indexer.dense_len();
        let n_u32 = u32::try_from(n).map_err(|_| GraphError::TooManyNodes)?;
        let mut dag = Dag::with_variables(n_u32);
        // Remap: Dag::with_variables creates Static VariableId nodes 0..n-1.
        // Dense ids align with indexer dense ids by construction.

        for (from_i, _) in self.nodes().iter().enumerate() {
            let from = DenseNodeId::from_raw(u32::try_from(from_i).expect("fit"));
            let from_key = self.temporal_key(from).ok_or(GraphError::InvalidEndpoints {
                message: "unfold requires lagged nodes",
            })?;
            for &to in self.children(from) {
                let to_key = self.temporal_key(to).ok_or(GraphError::InvalidEndpoints {
                    message: "unfold requires lagged nodes",
                })?;
                insert_replicated_edges(&mut dag, &indexer, from_key, to_key)?;
            }
        }

        Ok(UnfoldedTemporalGraph { dag, indexer })
    }
}

fn insert_replicated_edges(
    dag: &mut Dag,
    indexer: &TemporalIndexer,
    from_key: TemporalNodeKey,
    to_key: TemporalNodeKey,
) -> Result<(), GraphError> {
    // Absolute offsets in the window are keyed as TemporalNodeKey.offset.
    // Replicate for every shift s such that both (offset+s) stay in window.
    let min_off = -(indexer.history() as i32);
    let max_off = (indexer.horizon() as i32) - 1;
    // We interpret summary keys as relative to a reference t; replicate by
    // adding delta to both offsets for all delta where both remain in-range.
    for delta in min_off..=max_off {
        let a = TemporalNodeKey {
            variable: from_key.variable,
            offset: from_key.offset.saturating_add(delta),
        };
        let b = TemporalNodeKey {
            variable: to_key.variable,
            offset: to_key.offset.saturating_add(delta),
        };
        if a.offset < min_off || a.offset > max_off || b.offset < min_off || b.offset > max_off {
            continue;
        }
        let from_dense = indexer.dense_id(a).map_err(|_| GraphError::InvalidEndpoints {
            message: "unfold endpoint outside window",
        })?;
        let to_dense = indexer.dense_id(b).map_err(|_| GraphError::InvalidEndpoints {
            message: "unfold endpoint outside window",
        })?;
        let from = DenseNodeId::from_raw(from_dense);
        let to = DenseNodeId::from_raw(to_dense);
        if from == to {
            continue;
        }
        match dag.insert_directed(from, to) {
            Ok(()) | Err(GraphError::DuplicateEdge { .. }) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Review-required temporal graph artifact (Phase 3 consumes; Phase 2 produces).
#[derive(Clone, Debug)]
pub struct TemporalGraphReview {
    /// Proposed discovery graph.
    pub graph: TemporalDag,
    /// Edges awaiting explicit acceptance (serializable keys).
    pub pending_edges: Arc<[(TemporalNodeKey, TemporalNodeKey)]>,
    /// Algorithm id that produced the proposal.
    pub algorithm: Arc<str>,
}

impl TemporalGraphReview {
    /// Construct a review artifact listing all current edges as pending.
    #[must_use]
    pub fn from_graph(graph: TemporalDag, algorithm: impl Into<Arc<str>>) -> Self {
        let mut pending = Vec::new();
        for (i, _) in graph.nodes().iter().enumerate() {
            let from = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
            let Some(from_key) = graph.temporal_key(from) else {
                continue;
            };
            for &to in graph.children(from) {
                if let Some(to_key) = graph.temporal_key(to) {
                    pending.push((from_key, to_key));
                }
            }
        }
        Self { graph, pending_edges: Arc::from(pending), algorithm: algorithm.into() }
    }

    /// Accept a pending edge by endpoints (no-op if absent).
    #[must_use]
    pub fn accept_edge(mut self, from: TemporalNodeKey, to: TemporalNodeKey) -> Self {
        let pending: Vec<_> =
            self.pending_edges.iter().copied().filter(|e| *e != (from, to)).collect();
        self.pending_edges = Arc::from(pending);
        self
    }

    /// Whether all pending edges have been accepted.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.pending_edges.is_empty()
    }
}

/// Helper for tests / discovery: ensure a lagged node exists.
pub fn ensure_lagged(
    graph: &mut TemporalDag,
    variable: VariableId,
    lag: causal_core::Lag,
) -> Result<DenseNodeId, GraphError> {
    for (i, n) in graph.nodes().iter().enumerate() {
        if let NodeRef::Lagged { variable: v, lag: l } = n {
            if *v == variable && *l == lag {
                return Ok(DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
            }
        }
    }
    graph.add_lagged(variable, lag)
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use causal_core::{Lag, VariableId};
    use causal_data::TemporalIndexer;

    use super::*;
    use crate::dsep::DSeparationWorkspace;

    #[test]
    fn unfold_replicates_lagged_edge() {
        let mut g = TemporalDag::empty();
        let past = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(past, now).unwrap();
        // history=1, horizon=2 → offsets -1,0,1
        let indexer = TemporalIndexer::new(2, 1, 2).unwrap();
        let unfolded = g.unfold(indexer).unwrap();
        assert_eq!(unfolded.dag.node_count(), 6); // 2 vars * 3 slices
        // At least one edge should exist
        let mut edge_count = 0usize;
        for i in 0..unfolded.dag.node_count() {
            edge_count += unfolded.dag.children(DenseNodeId::from_raw(i as u32)).len();
        }
        assert!(edge_count >= 1);
    }

    #[test]
    fn unfold_dsep_on_chain() {
        // X(1) -> Y(0) -> Z(0) contemporaneous Y->Z plus lag X->Y
        let mut g = TemporalDag::empty();
        let x1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let z0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();
        g.insert_directed(y0, z0).unwrap();
        let indexer = TemporalIndexer::new(3, 1, 1).unwrap(); // offsets -1,0
        let unfolded = g.unfold(indexer).unwrap();
        let mut ws = DSeparationWorkspace::default();
        // Pick dense ids for Y@0 and Z@0
        let y = DenseNodeId::from_raw(
            unfolded.indexer.dense_id(TemporalNodeKey { variable: VariableId::from_raw(1), offset: 0 }).unwrap(),
        );
        let z = DenseNodeId::from_raw(
            unfolded.indexer.dense_id(TemporalNodeKey { variable: VariableId::from_raw(2), offset: 0 }).unwrap(),
        );
        // Without conditioning, may be connected via edge
        let _ = unfolded.dag.is_d_separated(y, z, &[], &mut ws);
    }

    #[test]
    fn review_accept_clears_pending() {
        let mut g = TemporalDag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(a, b).unwrap();
        let review = TemporalGraphReview::from_graph(g, "pcmci");
        assert!(!review.is_complete());
        let from = TemporalNodeKey { variable: VariableId::from_raw(0), offset: -1 };
        let to = TemporalNodeKey { variable: VariableId::from_raw(1), offset: 0 };
        let review = review.accept_edge(from, to);
        assert!(review.is_complete());
    }
}
