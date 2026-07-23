//! Batch multi-query: one table, N average-effect estimates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{AverageEffectQuery, ExecutionContext};
use causal_data::TabularData;
use causal_graph::Dag;

use crate::error::AnalysisError;
use crate::result::CausalAnalysisResult;

use super::builder::{CausalAnalysisBuilder, RefuteSuite};
use super::execute::CausalAnalysis;
use super::latency::LatencyMode;

/// Shared-table batch of static average-effect queries.
///
/// Binds data once; each query runs identify → project → estimate independently
/// (shared ingest, not shared physical plan — plans stay per-query).
#[derive(Clone, Debug)]
pub struct BatchAnalysis {
    data: TabularData,
    graph: Dag,
    bootstrap_replicates: u32,
    refute: RefuteSuite,
    latency_mode: Option<LatencyMode>,
    identifier: Option<String>,
    estimator: Option<String>,
}

impl BatchAnalysis {
    /// Start a batch over `data` and a static DAG.
    #[must_use]
    pub fn new(data: TabularData, graph: Dag) -> Self {
        Self {
            data,
            graph,
            bootstrap_replicates: 50,
            refute: RefuteSuite::PlaceboAndRcc,
            latency_mode: None,
            identifier: None,
            estimator: None,
        }
    }

    /// Bootstrap replicates for every query.
    #[must_use]
    pub fn bootstrap_replicates(mut self, n: u32) -> Self {
        self.bootstrap_replicates = n;
        self
    }

    /// Refute suite for every query.
    #[must_use]
    pub fn refute(mut self, suite: RefuteSuite) -> Self {
        self.refute = suite;
        self
    }

    /// Optional latency tier applied to every query.
    #[must_use]
    pub fn latency_mode(mut self, mode: LatencyMode) -> Self {
        self.latency_mode = Some(mode);
        self
    }

    /// Optional identifier id string.
    #[must_use]
    pub fn identifier(mut self, id: impl Into<String>) -> Self {
        self.identifier = Some(id.into());
        self
    }

    /// Optional estimator id string.
    #[must_use]
    pub fn estimator(mut self, id: impl Into<String>) -> Self {
        self.estimator = Some(id.into());
        self
    }

    /// Estimate each query against the shared table.
    ///
    /// # Errors
    ///
    /// Empty query list, or any per-query analysis failure.
    pub fn estimate_many(
        &self,
        queries: &[AverageEffectQuery],
        ctx: &ExecutionContext,
    ) -> Result<Vec<CausalAnalysisResult>, AnalysisError> {
        if queries.is_empty() {
            return Err(AnalysisError::Compile {
                message: "batch estimate_many requires at least one query".into(),
            });
        }
        let mut out = Vec::with_capacity(queries.len());
        for q in queries {
            let mut builder = CausalAnalysisBuilder::new()
                .data(self.data.clone())
                .graph(self.graph.clone())
                .query(q.clone())
                .refute(self.refute)
                .bootstrap_replicates(self.bootstrap_replicates);
            if let Some(mode) = self.latency_mode {
                builder = builder.latency_mode(mode);
            }
            if let Some(id) = self.identifier.as_deref() {
                builder = builder.identifier(id);
            }
            if let Some(est) = self.estimator.as_deref() {
                builder = builder.estimator(est);
            }
            let analysis: CausalAnalysis = builder.build()?;
            out.push(analysis.run(ctx)?);
        }
        Ok(out)
    }
}
