//! Build temporal graph evidence from scored links.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_graph::{TemporalDag, ensure_lagged};

use crate::error::DiscoveryError;
use crate::result::{GraphEvidence, ScoredLink};

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
