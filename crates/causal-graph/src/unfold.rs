//! Lazy finite unfolding of temporal DAGs and graph-review artifacts.
//!
//! Stationary algorithms query edges on demand via [`LazyUnfoldedTemporalGraph`].
//! Full materialisation is available when a static [`Dag`] is required (Phase 3).
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

/// Eager materialisation of a [`TemporalDag`] over a finite indexer window.
#[derive(Clone, Debug)]
pub struct UnfoldedTemporalGraph {
    /// Static DAG whose dense ids match [`TemporalIndexer::dense_id`].
    pub dag: Dag,
    /// Indexer used for the unfolding.
    pub indexer: TemporalIndexer,
}

/// Lazy finite unfolding: template edges are replicated on query, not upfront.
#[derive(Clone, Debug)]
pub struct LazyUnfoldedTemporalGraph {
    /// Stationary / lagged summary graph.
    pub template: TemporalDag,
    /// Finite window indexer.
    pub indexer: TemporalIndexer,
}

impl TemporalDag {
    /// Lazy unfold over a finite indexer window (Phase 2 default).
    ///
    /// # Errors
    ///
    /// Unknown/non-lagged template nodes.
    pub fn unfold_lazy(
        &self,
        indexer: TemporalIndexer,
    ) -> Result<LazyUnfoldedTemporalGraph, GraphError> {
        // Validate all nodes are lagged up front.
        for (i, _) in self.nodes().iter().enumerate() {
            let id = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
            let _ = self
                .temporal_key(id)
                .ok_or(GraphError::InvalidEndpoints { message: "unfold requires lagged nodes" })?;
        }
        Ok(LazyUnfoldedTemporalGraph { template: self.clone(), indexer })
    }

    /// Eager unfold into a static [`Dag`] (materialises the full window).
    ///
    /// # Errors
    ///
    /// Unknown/non-lagged nodes, indexer issues, or cycle insertion.
    pub fn unfold(&self, indexer: TemporalIndexer) -> Result<UnfoldedTemporalGraph, GraphError> {
        self.unfold_lazy(indexer)?.materialize()
    }
}

impl LazyUnfoldedTemporalGraph {
    /// Whether a concrete directed edge exists under the template replication.
    ///
    /// # Errors
    ///
    /// Endpoints outside the indexer window.
    pub fn has_edge(&self, from: TemporalNodeKey, to: TemporalNodeKey) -> Result<bool, GraphError> {
        let _ = self.indexer.dense_id(from).map_err(|_| GraphError::InvalidEndpoints {
            message: "unfold endpoint outside window",
        })?;
        let _ = self.indexer.dense_id(to).map_err(|_| GraphError::InvalidEndpoints {
            message: "unfold endpoint outside window",
        })?;
        for (from_i, _) in self.template.nodes().iter().enumerate() {
            let from_id = DenseNodeId::from_raw(u32::try_from(from_i).expect("fit"));
            let Some(from_key) = self.template.temporal_key(from_id) else {
                continue;
            };
            for &to_id in self.template.children(from_id) {
                let Some(to_key) = self.template.temporal_key(to_id) else {
                    continue;
                };
                if edge_matches(from_key, to_key, from, to) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Materialise all in-window replicated edges into a static DAG.
    ///
    /// # Errors
    ///
    /// Indexer / cycle insertion failures.
    pub fn materialize(&self) -> Result<UnfoldedTemporalGraph, GraphError> {
        let n = self.indexer.dense_len();
        let n_u32 = u32::try_from(n).map_err(|_| GraphError::TooManyNodes)?;
        let mut dag = Dag::with_variables(n_u32);

        for (from_i, _) in self.template.nodes().iter().enumerate() {
            let from = DenseNodeId::from_raw(u32::try_from(from_i).expect("fit"));
            let from_key = self
                .template
                .temporal_key(from)
                .ok_or(GraphError::InvalidEndpoints { message: "unfold requires lagged nodes" })?;
            for &to in self.template.children(from) {
                let to_key =
                    self.template.temporal_key(to).ok_or(GraphError::InvalidEndpoints {
                        message: "unfold requires lagged nodes",
                    })?;
                insert_replicated_edges(&mut dag, &self.indexer, from_key, to_key)?;
            }
        }

        Ok(UnfoldedTemporalGraph { dag, indexer: self.indexer.clone() })
    }
}

fn edge_matches(
    template_from: TemporalNodeKey,
    template_to: TemporalNodeKey,
    concrete_from: TemporalNodeKey,
    concrete_to: TemporalNodeKey,
) -> bool {
    template_from.variable == concrete_from.variable
        && template_to.variable == concrete_to.variable
        && concrete_from.offset.wrapping_sub(template_from.offset)
            == concrete_to.offset.wrapping_sub(template_to.offset)
}

fn insert_replicated_edges(
    dag: &mut Dag,
    indexer: &TemporalIndexer,
    from_key: TemporalNodeKey,
    to_key: TemporalNodeKey,
) -> Result<(), GraphError> {
    let min_off = -(indexer.history() as i32);
    let max_off = (indexer.horizon() as i32) - 1;
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

/// Review-required temporal graph artifact (Phase 2 produces; Phase 3 consumes).
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
    fn lazy_has_edge_matches_materialize() {
        let mut g = TemporalDag::empty();
        let past = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(past, now).unwrap();
        let indexer = TemporalIndexer::new(2, 1, 2).unwrap();
        let lazy = g.unfold_lazy(indexer.clone()).unwrap();
        let from = TemporalNodeKey { variable: VariableId::from_raw(0), offset: -1 };
        let to = TemporalNodeKey { variable: VariableId::from_raw(1), offset: 0 };
        assert!(lazy.has_edge(from, to).unwrap());
        let unfolded = lazy.materialize().unwrap();
        assert_eq!(unfolded.dag.node_count(), 6);
    }

    #[test]
    fn unfold_replicates_lagged_edge() {
        let mut g = TemporalDag::empty();
        let past = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let now = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(past, now).unwrap();
        let indexer = TemporalIndexer::new(2, 1, 2).unwrap();
        let unfolded = g.unfold(indexer).unwrap();
        assert_eq!(unfolded.dag.node_count(), 6);
        let mut edge_count = 0usize;
        for i in 0..unfolded.dag.node_count() {
            edge_count += unfolded.dag.children(DenseNodeId::from_raw(i as u32)).len();
        }
        assert!(edge_count >= 1);
    }

    #[test]
    fn unfold_dsep_on_chain() {
        let mut g = TemporalDag::empty();
        let x1 = g.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        let z0 = g.add_lagged(VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();
        g.insert_directed(y0, z0).unwrap();
        let indexer = TemporalIndexer::new(3, 1, 1).unwrap();
        let unfolded = g.unfold(indexer).unwrap();
        let mut ws = DSeparationWorkspace::default();
        let y = DenseNodeId::from_raw(
            unfolded
                .indexer
                .dense_id(TemporalNodeKey { variable: VariableId::from_raw(1), offset: 0 })
                .unwrap(),
        );
        let z = DenseNodeId::from_raw(
            unfolded
                .indexer
                .dense_id(TemporalNodeKey { variable: VariableId::from_raw(2), offset: 0 })
                .unwrap(),
        );
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
