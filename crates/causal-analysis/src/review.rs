//! Graph review gate for discovery outputs (DESIGN.md §21.3).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use causal_core::{ExecutionContext, TemporalEffectQuery};
use causal_data::{DiscoveryEstimationSplit, TemporalNodeKey, TimeSeriesData};
use causal_graph::{DenseNodeId, TemporalDag, TemporalGraphReview};

use crate::error::AnalysisError;
use crate::planner::{CompiledAnalysis, LogicalAnalysisPlan, compile_logical_temporal_effect};

/// Pending review session that must complete before estimation.
#[derive(Clone, Debug)]
pub struct PendingGraphReview {
    /// Review artifact.
    pub review: TemporalGraphReview,
    /// Series length (for diagnostics).
    pub series_len: usize,
    /// Temporal effect query.
    pub query: TemporalEffectQuery,
    /// Optional discovery/estimation split.
    pub split: Option<DiscoveryEstimationSplit>,
}

impl PendingGraphReview {
    /// Wrap a review artifact with query context.
    #[must_use]
    pub fn new(
        review: TemporalGraphReview,
        series_len: usize,
        query: TemporalEffectQuery,
        split: Option<DiscoveryEstimationSplit>,
    ) -> Self {
        Self { review, series_len, query, split }
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
    ) -> Result<Self, AnalysisError> {
        let pending = self.review.pending_edges.iter().any(|e| *e == (from, to));
        if pending {
            return Ok(self.accept_edge(from, to));
        }
        if edge_in_graph(&self.review.graph, from, to) {
            return Ok(self);
        }
        Err(AnalysisError::ReviewRequired {
            message: format!("required edge {from:?} -> {to:?} not in proposed graph"),
        })
    }

    /// Accept all remaining pending edges.
    #[must_use]
    pub fn accept_all(mut self) -> Self {
        self.review.pending_edges = std::sync::Arc::from([]);
        self
    }

    /// Finish review into a compiled Ready plan carrying the accepted graph.
    ///
    /// # Errors
    ///
    /// Incomplete review or compile failure.
    pub fn finish(
        self,
        data: &TimeSeriesData,
        ctx: &ExecutionContext,
    ) -> Result<CompiledAnalysis, AnalysisError> {
        let _ = self.series_len;
        if !self.review.is_complete() {
            return Err(AnalysisError::ReviewRequired {
                message: format!(
                    "{} pending edges remain; accept or require them before estimation",
                    self.review.pending_edges.len()
                ),
            });
        }
        let logical = compile_logical_temporal_effect(
            data,
            &self.review.graph,
            &self.query,
            self.split,
            false,
        )?;
        let physical = logical.compile_physical_with_graph(ctx, Some(self.review.graph.clone()))?;
        Ok(CompiledAnalysis::Ready(physical))
    }

    /// Borrow the reviewed temporal DAG.
    #[must_use]
    pub fn graph(&self) -> &TemporalDag {
        &self.review.graph
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
) -> Result<CompiledAnalysis, AnalysisError> {
    let logical = compile_logical_temporal_effect(data, graph, query, split, false)?;
    let physical = logical.compile_physical_with_graph(ctx, Some(graph.clone()))?;
    Ok(CompiledAnalysis::Ready(physical))
}

/// Wrap discovery output as review-required.
#[must_use]
pub fn compile_review_required(review: TemporalGraphReview) -> CompiledAnalysis {
    CompiledAnalysis::ReviewRequired(review)
}

/// Refuse when the logical plan still requires review.
///
/// # Errors
///
/// [`AnalysisError::ReviewRequired`] when the flag is set.
pub fn ensure_review_complete(plan: &LogicalAnalysisPlan) -> Result<(), AnalysisError> {
    if plan.record.graph_review_required {
        return Err(AnalysisError::ReviewRequired {
            message: "graph review required before estimation".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use causal_core::{Lag, TemporalEffectQuery, VariableId};
    use causal_graph::{TemporalDag, TemporalGraphReview, ensure_lagged};

    use super::*;

    fn tiny_review() -> TemporalGraphReview {
        let mut g = TemporalDag::empty();
        let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();
        TemporalGraphReview::from_graph(g, "pcmci")
    }

    #[test]
    fn incomplete_review_blocks_estimation_flag() {
        assert!(matches!(
            ensure_review_complete(&LogicalAnalysisPlan {
                record: causal_core::LogicalAnalysisPlanRecord {
                    plan_id: std::sync::Arc::from("t"),
                    data_classification: causal_core::DataClassification::Temporal,
                    discovery_algorithm: Some(std::sync::Arc::from("pcmci")),
                    graph_review_required: true,
                    identifier: None,
                    estimator: None,
                    validation_suite: None,
                    query_variables: std::sync::Arc::from([]),
                },
                query: causal_core::CausalQuery::TemporalEffect(TemporalEffectQuery::pulse(
                    VariableId::from_raw(0),
                    VariableId::from_raw(1),
                    1.0,
                )),
                split: None,
                row_count_hint: 100,
            }),
            Err(AnalysisError::ReviewRequired { .. })
        ));
    }

    #[test]
    fn accept_edge_completes_review() {
        let r = tiny_review();
        assert!(!r.is_complete());
        let (a, b) = r.pending_edges[0];
        let done = r.accept_edge(a, b);
        assert!(done.is_complete());
        let pending = PendingGraphReview::new(
            done,
            100,
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0),
            None,
        );
        assert!(pending.review.is_complete());
    }

    #[test]
    fn require_missing_edge_errors() {
        let r = tiny_review();
        let pending = PendingGraphReview::new(
            r,
            100,
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0),
            None,
        );
        let missing_from = TemporalNodeKey { variable: VariableId::from_raw(9), offset: 0 };
        let missing_to = TemporalNodeKey { variable: VariableId::from_raw(8), offset: 0 };
        assert!(pending.require_edge(missing_from, missing_to).is_err());
    }
}
