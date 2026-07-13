//! Public lagged PCMCI algorithm (DESIGN.md §13.4–13.5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::{ExecutionContext, VariableId};
use causal_data::TimeSeriesData;
use causal_graph::TemporalGraphReview;
use causal_stats::benjamini_hochberg;

use crate::constraints::DiscoveryConstraints;
use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::evidence::graph_evidence_from_scored;
use crate::result::{AlgorithmRecord, DiscoveryResult, ScoredLink};

/// Lagged PCMCI discovery algorithm.
#[derive(Clone, Debug)]
pub struct Pcmci {
    /// Engine.
    pub engine: PcmciEngine,
    /// Apply Benjamini–Hochberg FDR to the full MCI family before alpha keep.
    pub fdr: bool,
}

impl Default for Pcmci {
    fn default() -> Self {
        Self::new()
    }
}

impl Pcmci {
    /// Default PCMCI (FDR on, alpha 0.05).
    #[must_use]
    pub fn new() -> Self {
        Self { engine: PcmciEngine::new(), fdr: true }
    }

    /// Configure constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.engine.constraints = constraints;
        self
    }

    /// Enable / disable FDR.
    #[must_use]
    pub fn with_fdr(mut self, fdr: bool) -> Self {
        self.fdr = fdr;
        self
    }

    /// Run lagged PCMCI on `variables` in `data`.
    ///
    /// MCI scores the full candidate family from PC parents. When `fdr` is set,
    /// Benjamini–Hochberg adjusts that family, then alpha retains links.
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
    ) -> Result<DiscoveryResult, DiscoveryError> {
        let mut result = self.engine.run_pc_mci(data, variables, workspace, ctx)?;
        let alpha = self.engine.constraints.alpha;

        let mut scored: Vec<ScoredLink> = result.evidence.links.iter().copied().collect();
        if self.fdr && !scored.is_empty() {
            let pvals: Vec<f64> = scored.iter().map(|l| l.p_value).collect();
            let adj = benjamini_hochberg(&pvals);
            for (link, &p_adj) in scored.iter_mut().zip(adj.iter()) {
                link.p_value = p_adj;
            }
        }
        scored.retain(|s| s.p_value < alpha);

        result.evidence = graph_evidence_from_scored(scored)?;
        result.algorithm = AlgorithmRecord {
            id: Arc::from("pcmci"),
            config: Arc::from(format!(
                "alpha={},max_lag={},fdr={}",
                alpha,
                self.engine.constraints.temporal.max_lag.raw(),
                self.fdr
            )),
        };
        result.review =
            TemporalGraphReview::from_graph(result.evidence.graph.clone(), result.algorithm.id.clone());
        result.performance.links_retained = result.evidence.links.len() as u64;
        Ok(result)
    }
}
