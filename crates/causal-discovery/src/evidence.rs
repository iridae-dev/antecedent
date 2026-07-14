//! Build temporal graph evidence from scored links.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{Lag, VariableId};
use causal_graph::{DenseNodeId, TemporalCpdag, TemporalDag, ensure_lagged};
use causal_stats::benjamini_hochberg;

use crate::error::DiscoveryError;
use crate::result::{
    CpdagGraphEvidence, DagGraphEvidence, EdgeEvidence, EvidenceSource, PcSepsets, ScoredLink,
};

/// Optionally FDR-adjust then retain links whose (adjusted) p-value is below `alpha`.
///
/// Raw p-values are preserved on [`ScoredLink::p_value`]; the BH-adjusted values are
/// recorded on [`ScoredLink::adjusted_p_value`] and drive retention when `fdr` is set.
#[must_use]
pub fn threshold_scored_links(
    mut scored: Vec<ScoredLink>,
    fdr: bool,
    alpha: f64,
) -> Vec<ScoredLink> {
    if fdr && !scored.is_empty() {
        let pvals: Vec<f64> = scored.iter().map(|l| l.p_value).collect();
        let adj = benjamini_hochberg(&pvals);
        for (link, &p_adj) in scored.iter_mut().zip(adj.iter()) {
            link.adjusted_p_value = Some(p_adj);
        }
    }
    scored.retain(|s| s.adjusted_p_value.unwrap_or(s.p_value) < alpha);
    scored
}

fn edge_evidence_from_scored(links: &[ScoredLink], sepsets: &PcSepsets) -> Arc<[EdgeEvidence]> {
    links
        .iter()
        .copied()
        .map(|s| {
            let key = (s.link.source, s.link.source_lag, s.link.target, s.link.target_lag);
            let sep = sepsets
                .get(&key)
                .cloned()
                .map_or_else(|| Arc::from([]), |s| Arc::<[_]>::from(vec![s]));
            let mut ev = EdgeEvidence::from_scored(s, sep);
            ev.adjusted_p_value = s.adjusted_p_value;
            ev
        })
        .collect()
}

/// Insert scored links into a fresh [`TemporalDag`] and wrap as evidence.
///
/// # Errors
///
/// Propagates lag-node registration failures.
pub fn graph_evidence_from_scored(
    links: Vec<ScoredLink>,
) -> Result<DagGraphEvidence, DiscoveryError> {
    graph_evidence_from_scored_with_sepsets(links, &PcSepsets::default())
}

/// Insert scored links into a fresh [`TemporalDag`] with separating-set evidence.
///
/// # Errors
///
/// Propagates lag-node registration failures.
pub fn graph_evidence_from_scored_with_sepsets(
    links: Vec<ScoredLink>,
    sepsets: &PcSepsets,
) -> Result<DagGraphEvidence, DiscoveryError> {
    let mut graph = TemporalDag::empty();
    for s in &links {
        let from = ensure_lagged(&mut graph, s.link.source, s.link.source_lag)
            .map_err(|e| DiscoveryError::Data(e.to_string()))?;
        let to = ensure_lagged(&mut graph, s.link.target, s.link.target_lag)
            .map_err(|e| DiscoveryError::Data(e.to_string()))?;
        let _ = graph.insert_directed(from, to);
    }
    let edge_evidence = edge_evidence_from_scored(&links, sepsets);
    Ok(DagGraphEvidence {
        graph,
        edge_evidence,
        links: Arc::from(links),
        source: EvidenceSource::Discovery { algorithm: Arc::from("pcmci") },
    })
}

/// Wrap an oriented [`TemporalCpdag`] and scored links as CPDAG evidence.
#[must_use]
pub fn cpdag_evidence_from_oriented(
    graph: TemporalCpdag,
    links: Vec<ScoredLink>,
    sepsets: &PcSepsets,
) -> CpdagGraphEvidence {
    let edge_evidence = edge_evidence_from_scored(&links, sepsets);
    CpdagGraphEvidence {
        graph,
        edge_evidence,
        links: Arc::from(links),
        source: EvidenceSource::Discovery { algorithm: Arc::from("pcmci_plus") },
    }
}

/// Build a temporal CPDAG from scored links (lagged directed, contemporaneous undirected).
///
/// # Errors
///
/// Node / edge insertion failures.
pub fn cpdag_from_scored_links(
    links: &[ScoredLink],
    variables: &[VariableId],
    max_lag: u32,
) -> Result<TemporalCpdag, DiscoveryError> {
    let mut cpdag = TemporalCpdag::empty();
    let mut node_ids = HashMap::<(u32, u32), DenseNodeId>::new();
    for &v in variables {
        for lag in 0..=max_lag {
            let id = cpdag
                .add_lagged(v, Lag::from_raw(lag))
                .map_err(|e| DiscoveryError::Data(e.to_string()))?;
            node_ids.insert((v.raw(), lag), id);
        }
    }
    for link in links {
        let Some(&src) = node_ids.get(&(link.link.source.raw(), link.link.source_lag.raw())) else {
            continue;
        };
        let Some(&tgt) = node_ids.get(&(link.link.target.raw(), link.link.target_lag.raw())) else {
            continue;
        };
        if cpdag.has_edge(src, tgt) {
            continue;
        }
        let contemporaneous =
            link.link.source_lag.is_contemporaneous() && link.link.target_lag.is_contemporaneous();
        let insert = if contemporaneous {
            cpdag.insert_undirected(src, tgt)
        } else {
            cpdag.insert_directed(src, tgt)
        };
        insert.map_err(|e| DiscoveryError::Data(e.to_string()))?;
    }
    Ok(cpdag)
}
