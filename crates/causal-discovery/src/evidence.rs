//! Build temporal graph evidence from scored links.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_graph::{TemporalDag, ensure_lagged};
use causal_stats::benjamini_hochberg;

use crate::error::DiscoveryError;
use crate::result::{GraphEvidence, ScoredLink};

/// Optionally FDR-adjust then retain links with `p_value < alpha`.
#[must_use]
pub fn threshold_scored_links(mut scored: Vec<ScoredLink>, fdr: bool, alpha: f64) -> Vec<ScoredLink> {
    if fdr && !scored.is_empty() {
        let pvals: Vec<f64> = scored.iter().map(|l| l.p_value).collect();
        let adj = benjamini_hochberg(&pvals);
        for (link, &p_adj) in scored.iter_mut().zip(adj.iter()) {
            link.p_value = p_adj;
        }
    }
    scored.retain(|s| s.p_value < alpha);
    scored
}

/// Insert scored links into a fresh [`TemporalDag`] and wrap as evidence.
///
/// # Errors
///
/// Propagates lag-node registration failures.
pub fn graph_evidence_from_scored(links: Vec<ScoredLink>) -> Result<GraphEvidence, DiscoveryError> {
    let mut graph = TemporalDag::empty();
    for s in &links {
        let from = ensure_lagged(&mut graph, s.link.source, s.link.source_lag)
            .map_err(|e| DiscoveryError::Data(e.to_string()))?;
        let to = ensure_lagged(&mut graph, s.link.target, s.link.target_lag)
            .map_err(|e| DiscoveryError::Data(e.to_string()))?;
        let _ = graph.insert_directed(from, to);
    }
    Ok(GraphEvidence { graph, links: Arc::from(links) })
}
