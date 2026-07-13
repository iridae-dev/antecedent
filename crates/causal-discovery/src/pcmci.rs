//! Public lagged PCMCI algorithm (DESIGN.md §13.4–13.5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::{ExecutionContext, VariableId};
use causal_data::TimeSeriesData;
use causal_stats::benjamini_hochberg;

use crate::constraints::DiscoveryConstraints;
use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::result::{AlgorithmRecord, DiscoveryResult};

/// Lagged PCMCI discovery algorithm.
#[derive(Clone, Debug)]
pub struct Pcmci {
    /// Engine.
    pub engine: PcmciEngine,
    /// Apply Benjamini–Hochberg FDR to MCI p-values before thresholding.
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
        if self.fdr && !result.evidence.links.is_empty() {
            let pvals: Vec<f64> = result.evidence.links.iter().map(|l| l.p_value).collect();
            let adj = benjamini_hochberg(&pvals);
            let alpha = self.engine.constraints.alpha;
            let mut kept = Vec::new();
            let mut graph = causal_graph::TemporalDag::empty();
            for (link, &p_adj) in result.evidence.links.iter().zip(adj.iter()) {
                if p_adj < alpha {
                    let mut scored = *link;
                    scored.p_value = p_adj;
                    let from = causal_graph::ensure_lagged(
                        &mut graph,
                        scored.link.source,
                        scored.link.source_lag,
                    )
                    .map_err(|e| DiscoveryError::Data(e.to_string()))?;
                    let to = causal_graph::ensure_lagged(
                        &mut graph,
                        scored.link.target,
                        scored.link.target_lag,
                    )
                    .map_err(|e| DiscoveryError::Data(e.to_string()))?;
                    let _ = graph.insert_directed(from, to);
                    kept.push(scored);
                }
            }
            result.evidence.links = Arc::from(kept);
            result.evidence.graph = graph;
            result.performance.links_retained = result.evidence.links.len() as u64;
        }
        result.algorithm = AlgorithmRecord {
            id: Arc::from("pcmci"),
            config: Arc::from(format!(
                "alpha={},max_lag={},fdr={}",
                self.engine.constraints.alpha,
                self.engine.constraints.temporal.max_lag.raw(),
                self.fdr
            )),
        };
        Ok(result)
    }
}
