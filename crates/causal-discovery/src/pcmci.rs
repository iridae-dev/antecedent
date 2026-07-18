//! Public lagged PCMCI algorithm (DESIGN.md §13.4–13.5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::{ExecutionContext, VariableId};
use causal_data::TimeSeriesData;
use causal_graph::TemporalGraphReview;
use causal_stats::{ConditionalIndependence, FdrAdjustment};

use crate::constraints::DiscoveryConstraints;
use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::evidence::{graph_evidence_from_scored_with_sepsets, threshold_scored_links};
use crate::result::{AlgorithmRecord, DagDiscoveryResult};

/// Lagged PCMCI discovery algorithm.
#[derive(Clone, Debug)]
pub struct Pcmci {
    /// Engine.
    pub engine: PcmciEngine,
    /// Multiple-testing adjustment over the MCI family (`None` = off).
    pub fdr: Option<FdrAdjustment>,
}

impl Default for Pcmci {
    fn default() -> Self {
        Self::new()
    }
}

impl Pcmci {
    /// Default PCMCI (BH FDR on, alpha 0.05).
    #[must_use]
    pub fn new() -> Self {
        Self { engine: PcmciEngine::new(), fdr: Some(FdrAdjustment::bh()) }
    }

    /// Configure constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.engine.constraints = constraints;
        self
    }

    /// Enable / disable BH FDR (pinned baseline contemporaneous exclusion).
    #[must_use]
    pub fn with_fdr(mut self, fdr: bool) -> Self {
        self.fdr = fdr.then(FdrAdjustment::bh);
        self
    }

    /// Full FDR / FWER configuration.
    #[must_use]
    pub fn with_fdr_adjustment(mut self, fdr: Option<FdrAdjustment>) -> Self {
        self.fdr = fdr;
        self
    }

    /// Replace the CI test on the shared engine.
    #[must_use]
    pub fn with_ci(mut self, ci: Arc<dyn ConditionalIndependence + Send + Sync>) -> Self {
        self.engine = self.engine.with_ci(ci);
        self
    }

    /// Run lagged PCMCI on `variables` in `data`.
    ///
    /// MCI scores the full constrained candidate family (all allowed
    /// `(X_{t−τ}, Y_t)` pairs); PC parent sets supply conditioning only. When
    /// FDR is set, that family is adjusted, then alpha retains links.
    ///
    /// # Errors
    ///
    /// Propagates engine / data failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<DagDiscoveryResult, DiscoveryError> {
        let mut result = self.engine.run_pc_mci(data, variables, workspace, ctx)?;
        let alpha = self.engine.constraints.alpha;

        let scored = threshold_scored_links(
            result.evidence.links.iter().copied().collect(),
            self.fdr,
            alpha,
        );

        result.evidence = graph_evidence_from_scored_with_sepsets(scored, &result.sepsets)?;
        result.algorithm = AlgorithmRecord {
            id: Arc::from("pcmci"),
            config: Arc::from(format!(
                "alpha={},max_lag={},fdr={:?}",
                alpha,
                self.engine.constraints.temporal.max_lag.raw(),
                self.fdr
            )),
        };
        result.review = TemporalGraphReview::from_graph(
            result.evidence.graph.clone(),
            result.algorithm.id.clone(),
        );
        result.performance.links_retained = result.evidence.links.len() as u64;
        Ok(result)
    }
}
