//! Graph review interaction surface for discovery outputs.
//!
//! Discovery is a separate, explicit step (see [`crate::discovery`]); the review gate
//! itself now lives in [`crate::accepted::AcceptedGraph`] construction. This module
//! keeps the interactive edit surface (`accept_edge`, `orient_edge`, `require_edge`,
//! `accept_all`) callers use to resolve a review artifact before handing it to
//! [`AcceptedGraph::accept`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use antecedent_core::{ExecutionContext, TemporalEffectQuery};
use antecedent_data::{DiscoveryEstimationSplit, TemporalNodeKey, TimeSeriesData};
use antecedent_graph::{DenseNodeId, TemporalCpdagReview, TemporalDag, TemporalGraphReview};

use crate::accepted::AcceptedGraph;
use crate::error::CausalError;
use crate::planner::{PhysicalExecutionPlan, compile_logical_temporal_effect};

/// Pending review session that must complete before estimation (DAG discovery).
#[derive(Clone, Debug)]
pub struct PendingGraphReview {
    /// Review artifact.
    pub review: TemporalGraphReview,
}

impl PendingGraphReview {
    /// Wrap a review artifact.
    #[must_use]
    pub fn new(review: TemporalGraphReview) -> Self {
        Self { review }
    }

    /// Accept one pending edge.
    #[must_use]
    pub fn accept_edge(mut self, from: TemporalNodeKey, to: TemporalNodeKey) -> Self {
        self.review = self.review.accept_edge(from, to);
        self
    }

    /// Require that an edge exists in the proposed graph and accept it.
    ///
    /// # Errors
    ///
    /// Edge not present in the proposed graph.
    pub fn require_edge(
        self,
        from: TemporalNodeKey,
        to: TemporalNodeKey,
    ) -> Result<Self, CausalError> {
        let pending = self.review.pending_edges.iter().any(|e| *e == (from, to));
        if pending {
            return Ok(self.accept_edge(from, to));
        }
        if edge_in_graph(&self.review.graph, from, to) {
            return Ok(self);
        }
        Err(CausalError::review_required_msg(format!(
            "required edge {from:?} -> {to:?} not in proposed graph"
        )))
    }

    /// Accept all remaining pending edges.
    #[must_use]
    pub fn accept_all(mut self) -> Self {
        self.review.pending_edges = std::sync::Arc::from([]);
        self
    }

    /// Complete the review into an [`AcceptedGraph`].
    ///
    /// Row-count re-checks and logical/physical compilation now happen at
    /// [`crate::analysis::StudyBuilder::build`] time — this only enforces that no
    /// pending edges remain.
    ///
    /// # Errors
    ///
    /// [`CausalError::ReviewRequired`] when pending edges remain.
    pub fn finish(self) -> Result<AcceptedGraph, CausalError> {
        AcceptedGraph::accept(self.review)
    }

    /// Borrow the reviewed temporal DAG.
    #[must_use]
    pub fn graph(&self) -> &TemporalDag {
        &self.review.graph
    }
}

/// Pending review for a PCMCI+ temporal CPDAG.
///
/// Directed edges must be accepted; undirected marks must be explicitly oriented
/// before completion to a [`TemporalDag`]. Auto-accept never drops undirected edges.
#[derive(Clone, Debug)]
pub struct PendingCpdagReview {
    /// CPDAG review artifact.
    pub review: TemporalCpdagReview,
}

impl PendingCpdagReview {
    /// Wrap a CPDAG review.
    #[must_use]
    pub fn new(review: TemporalCpdagReview) -> Self {
        Self { review }
    }

    /// Accept one pending directed edge.
    #[must_use]
    pub fn accept_edge(mut self, from: TemporalNodeKey, to: TemporalNodeKey) -> Self {
        self.review = self.review.accept_edge(from, to);
        self
    }

    /// Orient an undirected edge as `from -> to`.
    ///
    /// # Errors
    ///
    /// Missing undirected edge, cycle, or unknown nodes.
    pub fn orient_edge(
        mut self,
        from: TemporalNodeKey,
        to: TemporalNodeKey,
    ) -> Result<Self, CausalError> {
        self.review = self
            .review
            .orient_edge(from, to)
            .map_err(|e| CausalError::review_required_msg(e.to_string()))?;
        Ok(self)
    }

    /// Accept all directed pending edges.
    ///
    /// Does **not** orient or drop undirected edges — call [`Self::orient_edge`] first.
    #[must_use]
    pub fn accept_all_directed(mut self) -> Self {
        self.review.pending_edges = std::sync::Arc::from([]);
        self
    }

    /// Complete the review into an [`AcceptedGraph`] (only when no undirected marks remain).
    ///
    /// Row-count re-checks and logical/physical compilation now happen at
    /// [`crate::analysis::StudyBuilder::build`] time.
    ///
    /// # Errors
    ///
    /// [`CausalError::ReviewRequired`] when pending edges or undirected marks remain.
    pub fn finish(self) -> Result<AcceptedGraph, CausalError> {
        AcceptedGraph::accept(self.review)
    }
}

fn edge_in_graph(graph: &TemporalDag, from: TemporalNodeKey, to: TemporalNodeKey) -> bool {
    let mut from_id = None;
    let mut to_id = None;
    for i in 0..graph.nodes().len() {
        let id = DenseNodeId::from_raw(i as u32);
        if let Some(k) = graph.temporal_key(id) {
            if k == from {
                from_id = Some(id);
            }
            if k == to {
                to_id = Some(id);
            }
        }
    }
    match (from_id, to_id) {
        (Some(f), Some(t)) => graph.children(f).iter().any(|c| *c == t),
        _ => false,
    }
}

/// Compile a temporal effect with a supplied (already reviewed) graph.
///
/// # Errors
///
/// Compile failures.
pub fn compile_temporal_with_graph(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    query: &TemporalEffectQuery,
    split: Option<DiscoveryEstimationSplit>,
    ctx: &ExecutionContext,
) -> Result<PhysicalExecutionPlan, CausalError> {
    let logical = compile_logical_temporal_effect(data, graph, query, split, false)?;
    logical.compile_physical_with_graph(ctx, Some(graph.clone()))
}

#[cfg(test)]
mod tests {
    use antecedent_core::{Lag, VariableId};
    use antecedent_graph::{TemporalCpdag, TemporalDag, TemporalGraphReview, ensure_lagged};

    use super::*;

    fn tiny_review() -> TemporalGraphReview {
        let mut g = TemporalDag::empty();
        let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();
        TemporalGraphReview::from_graph(g, "pcmci")
    }

    #[test]
    fn accept_edge_completes_review() {
        let r = tiny_review();
        assert!(!r.is_complete());
        let (a, b) = r.pending_edges[0];
        let done = r.accept_edge(a, b);
        assert!(done.is_complete());
        let pending = PendingGraphReview::new(done);
        assert!(pending.review.is_complete());
        let accepted = pending.finish().unwrap();
        assert_eq!(accepted.class(), crate::accepted::GraphClass::TemporalDag);
    }

    #[test]
    fn require_missing_edge_errors() {
        let r = tiny_review();
        let pending = PendingGraphReview::new(r);
        let missing_from = TemporalNodeKey { variable: VariableId::from_raw(9), offset: 0 };
        let missing_to = TemporalNodeKey { variable: VariableId::from_raw(8), offset: 0 };
        assert!(pending.require_edge(missing_from, missing_to).is_err());
    }

    #[test]
    fn incomplete_review_refuses_finish() {
        let r = tiny_review();
        let err = PendingGraphReview::new(r).finish().unwrap_err();
        assert!(matches!(err, CausalError::ReviewRequired { .. }));
    }

    #[test]
    fn cpdag_finish_refuses_undirected() {
        let mut g = TemporalCpdag::empty();
        let a = g.add_lagged(VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
        let b = g.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_undirected(a, b).unwrap();
        let review = TemporalCpdagReview::from_cpdag(g, "pcmci_plus");
        assert!(!review.pending_undirected.is_empty());
        let pending = PendingCpdagReview::new(review).accept_all_directed();
        assert!(!pending.review.is_complete());
        let err = pending.finish().unwrap_err();
        assert!(matches!(err, CausalError::ReviewRequired { .. }));
    }
}
