//! Batch multi-query: one table, N average-effect estimates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, ExecutionContext};
use antecedent_data::TabularData;
use antecedent_graph::Dag;

use crate::error::CausalError;
use crate::result::StudyResult;

use super::builder::RefuteSuite;
use super::execute::Study;
use super::latency::LatencyMode;
use crate::strategy_table::{EstimatorId, IdentifierId};

/// Shared-table batch of static average-effect queries.
///
/// Binds data once; each query runs identify → project → estimate independently
/// (shared ingest, not shared physical plan — plans stay per-query).
#[derive(Clone, Debug)]
pub struct BatchStudy {
    data: TabularData,
    graph: Dag,
    bootstrap_replicates: u32,
    refute: RefuteSuite,
    latency_mode: Option<LatencyMode>,
    identifier: Option<IdentifierId>,
    estimator: Option<EstimatorId>,
}

impl BatchStudy {
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

    /// Optional identification strategy applied to every query.
    ///
    /// Parse a wire name with `"backdoor.adjustment".parse::<IdentifierId>()?`.
    #[must_use]
    pub const fn identifier(mut self, id: IdentifierId) -> Self {
        self.identifier = Some(id);
        self
    }

    /// Optional estimator applied to every query.
    ///
    /// Parse a wire name with `"propensity.weighting".parse::<EstimatorId>()?`.
    #[must_use]
    pub const fn estimator(mut self, id: EstimatorId) -> Self {
        self.estimator = Some(id);
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
    ) -> Result<Vec<StudyResult>, CausalError> {
        if queries.is_empty() {
            return Err(CausalError::Compile {
                message: "batch estimate_many requires at least one query".into(),
            });
        }
        let mut out = Vec::with_capacity(queries.len());
        for q in queries {
            let mut builder = Study::tabular(self.data.clone())
                .graph(self.graph.clone())
                .query(q.clone())
                .refute(self.refute)
                .bootstrap_replicates(self.bootstrap_replicates);
            if let Some(mode) = self.latency_mode {
                builder = builder.latency_mode(mode);
            }
            if let Some(id) = self.identifier {
                builder = builder.identifier(id);
            }
            if let Some(est) = self.estimator {
                builder = builder.estimator(est);
            }
            let analysis: Study = builder.build()?;
            out.push(analysis.run(ctx)?);
        }
        Ok(out)
    }
}
