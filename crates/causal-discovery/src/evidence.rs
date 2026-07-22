//! Build temporal graph evidence from scored links.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::needless_pass_by_value
)]

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{Lag, VariableId};
use causal_graph::{DenseNodeId, TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged};
use causal_stats::{FdrAdjustment, adjust_pvalues};

use crate::error::DiscoveryError;
use crate::result::{
    CpdagGraphEvidence, DagGraphEvidence, EdgeEvidence, EvidenceSource, PagGraphEvidence,
    PcSepsets, ScoredLink,
};

/// Optionally FDR-adjust then retain links whose (adjusted) p-value is `<= alpha`.
///
/// Matches pinned baseline's significance boundary (`p <= alpha` kept). Raw p-values are
/// preserved on [`ScoredLink::p_value`]; adjusted values are recorded on
/// [`ScoredLink::adjusted_p_value`] and drive retention when FDR is configured.
///
/// When `fdr.exclude_contemporaneous` is set (pinned baseline default), lag-0 links are
/// left unadjusted and thresholded on their raw p-values — only lagged tests enter
/// the correction family (`get_corrected_pvalues`).
#[must_use]
pub fn threshold_scored_links(
    mut scored: Vec<ScoredLink>,
    fdr: Option<FdrAdjustment>,
    alpha: f64,
) -> Vec<ScoredLink> {
    if let Some(cfg) = fdr {
        if !scored.is_empty() {
            let mut family_idx = Vec::new();
            let mut family_p = Vec::new();
            for (i, link) in scored.iter().enumerate() {
                let contemp = link.link.source_lag.is_contemporaneous()
                    && link.link.target_lag.is_contemporaneous();
                if cfg.exclude_contemporaneous && contemp {
                    continue;
                }
                family_idx.push(i);
                family_p.push(link.p_value);
            }
            if !family_p.is_empty() {
                let adj = adjust_pvalues(&family_p, cfg.method);
                for (slot, &i) in family_idx.iter().enumerate() {
                    scored[i].adjusted_p_value = Some(adj[slot]);
                }
            }
        }
    }
    scored.retain(|s| s.adjusted_p_value.unwrap_or(s.p_value) <= alpha);
    scored
}

/// Backward-compatible BH toggle (excludes contemporaneous links, matching pinned baseline).
#[must_use]
pub fn threshold_scored_links_bh(
    scored: Vec<ScoredLink>,
    fdr: bool,
    alpha: f64,
) -> Vec<ScoredLink> {
    threshold_scored_links(scored, fdr.then(FdrAdjustment::bh), alpha)
}

/// Keep a contemporaneous undirected adjacency only when **both** directed tests survive.
///
/// pinned baseline symmetrizes lag-0 links: X—Y at τ=0 requires both X→Y and Y→X adjacency
/// survivors (intersection). Lagged links pass through unchanged. When both directions
/// survive, the retained score uses the more conservative (max) p-value and the
/// corresponding statistic.
#[must_use]
pub fn symmetrize_contemporaneous_links(links: Vec<ScoredLink>) -> Vec<ScoredLink> {
    let mut lagged = Vec::new();
    let mut contemp: HashMap<(u32, u32), (Option<ScoredLink>, Option<ScoredLink>)> = HashMap::new();
    for s in links {
        let contemp_pair =
            s.link.source_lag.is_contemporaneous() && s.link.target_lag.is_contemporaneous();
        if !contemp_pair {
            lagged.push(s);
            continue;
        }
        let a = s.link.source.raw();
        let b = s.link.target.raw();
        let key = if a <= b { (a, b) } else { (b, a) };
        let entry = contemp.entry(key).or_insert((None, None));
        if a <= b {
            // source is lower id → "forward" slot when key ordered by ids
            if s.link.source.raw() == key.0 {
                entry.0 = Some(s);
            } else {
                entry.1 = Some(s);
            }
        } else if s.link.source.raw() == key.0 {
            entry.0 = Some(s);
        } else {
            entry.1 = Some(s);
        }
    }
    let mut out = lagged;
    for ((lo, hi), (fwd, rev)) in contemp {
        let (Some(mut a), Some(b)) = (fwd, rev) else {
            continue;
        };
        // Conservative p across both directions.
        let (p_a, p_b) =
            (a.adjusted_p_value.unwrap_or(a.p_value), b.adjusted_p_value.unwrap_or(b.p_value));
        if p_b > p_a {
            a.p_value = b.p_value;
            a.adjusted_p_value = b.adjusted_p_value;
            a.statistic = b.statistic;
        }
        // Canonicalize source = lower variable id for stable CPDAG insertion.
        if a.link.source.raw() != lo {
            a.link.source = VariableId::from_raw(lo);
            a.link.target = VariableId::from_raw(hi);
        }
        let _ = hi;
        out.push(a);
    }
    out
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
    let mut kept = Vec::with_capacity(links.len());
    for s in &links {
        let from = ensure_lagged(&mut graph, s.link.source, s.link.source_lag)
            .map_err(DiscoveryError::from)?;
        let to = ensure_lagged(&mut graph, s.link.target, s.link.target_lag)
            .map_err(DiscoveryError::from)?;
        match graph.insert_directed(from, to) {
            Ok(()) => kept.push(*s),
            Err(
                causal_graph::GraphError::Cycle { .. }
                | causal_graph::GraphError::DuplicateEdge { .. },
            ) => {
                // Keep links/graph aligned: cycle-forming or duplicate edges stay out of both.
            }
            Err(e) => return Err(DiscoveryError::from(e)),
        }
    }
    let edge_evidence = edge_evidence_from_scored(&kept, sepsets);
    Ok(DagGraphEvidence {
        graph,
        edge_evidence,
        links: Arc::from(kept),
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
            let id = cpdag.add_lagged(v, Lag::from_raw(lag)).map_err(DiscoveryError::from)?;
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
        insert.map_err(DiscoveryError::from)?;
    }
    Ok(cpdag)
}

/// Build a temporal PAG from scored links.
///
/// Contemporaneous pairs are inserted as `o–o`. Lagged links are inserted as `o→`
/// (circle at the earlier/source node, arrow at the later/target) per LPCMCI
/// initialization (Gerhardus & Runge 2020) — a tail would assert ancestorship, which
/// is not yet justified for a mere lagged dependence.
///
/// # Errors
///
/// Node / edge insertion failures.
pub fn pag_from_scored_links(
    links: &[ScoredLink],
    variables: &[VariableId],
    max_lag: u32,
) -> Result<TemporalPag, DiscoveryError> {
    let mut pag = TemporalPag::empty();
    let mut node_ids = HashMap::<(u32, u32), DenseNodeId>::new();
    for &v in variables {
        for lag in 0..=max_lag {
            let id = pag.add_lagged(v, Lag::from_raw(lag)).map_err(DiscoveryError::from)?;
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
        if pag.has_edge(src, tgt) {
            continue;
        }
        let contemporaneous =
            link.link.source_lag.is_contemporaneous() && link.link.target_lag.is_contemporaneous();
        let insert = if contemporaneous {
            // Circle-circle for uncertain contemporaneous adjacency (latent-aware).
            pag.insert_marked(causal_graph::MarkedEdge {
                a: if src.raw() <= tgt.raw() { src } else { tgt },
                b: if src.raw() <= tgt.raw() { tgt } else { src },
                at_a: causal_graph::Endpoint::Circle,
                at_b: causal_graph::Endpoint::Circle,
                middle: causal_graph::MiddleMark::Empty,
            })
        } else {
            // Lagged: o→ with arrow at the later node by time order.
            pag.insert_circle_arrow(src, tgt)
        };
        insert.map_err(DiscoveryError::from)?;
    }
    Ok(pag)
}

/// Wrap an oriented [`TemporalPag`] and scored links as PAG evidence.
#[must_use]
pub fn pag_evidence_from_oriented(
    graph: TemporalPag,
    links: Vec<ScoredLink>,
    sepsets: &PcSepsets,
) -> PagGraphEvidence {
    let edge_evidence = edge_evidence_from_scored(&links, sepsets);
    PagGraphEvidence {
        graph,
        edge_evidence,
        links: Arc::from(links),
        source: EvidenceSource::Discovery { algorithm: Arc::from("lpcmci") },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::LaggedLink;
    use causal_graph::Endpoint;

    #[test]
    fn lagged_links_initialize_as_circle_arrow() {
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let links = [ScoredLink {
            link: LaggedLink {
                source: vars[0],
                source_lag: Lag::from_raw(1),
                target: vars[1],
                target_lag: Lag::CONTEMPORANEOUS,
            },
            statistic: 0.5,
            p_value: 0.01,
            adjusted_p_value: None,
        }];
        let pag = pag_from_scored_links(&links, &vars, 1).unwrap();
        // Nodes: (v0,0), (v0,1), (v1,0), (v1,1) in nested loop order.
        let mut src_id = None;
        let mut tgt_id = None;
        for i in 0..pag.node_count() {
            let id = DenseNodeId::from_raw(i as u32);
            let node = &pag.nodes()[i];
            match node {
                causal_graph::NodeRef::Lagged { variable, lag }
                    if *variable == vars[0] && lag.raw() == 1 =>
                {
                    src_id = Some(id);
                }
                causal_graph::NodeRef::Lagged { variable, lag }
                    if *variable == vars[1] && lag.is_contemporaneous() =>
                {
                    tgt_id = Some(id);
                }
                _ => {}
            }
        }
        let src = src_id.expect("source node");
        let tgt = tgt_id.expect("target node");
        let e = pag.edge_between(src, tgt).expect("lagged edge");
        let (at_src, at_tgt) = if e.a == src { (e.at_a, e.at_b) } else { (e.at_b, e.at_a) };
        assert!(matches!(at_src, Endpoint::Circle), "circle at earlier node");
        assert!(matches!(at_tgt, Endpoint::Arrow), "arrow at later node");
    }

    #[test]
    fn symmetrize_requires_both_contemporaneous_directions() {
        let x = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let one_way = vec![ScoredLink {
            link: LaggedLink {
                source: x,
                source_lag: Lag::CONTEMPORANEOUS,
                target: y,
                target_lag: Lag::CONTEMPORANEOUS,
            },
            statistic: 0.9,
            p_value: 0.01,
            adjusted_p_value: None,
        }];
        assert!(symmetrize_contemporaneous_links(one_way).is_empty());

        let both = vec![
            ScoredLink {
                link: LaggedLink {
                    source: x,
                    source_lag: Lag::CONTEMPORANEOUS,
                    target: y,
                    target_lag: Lag::CONTEMPORANEOUS,
                },
                statistic: 0.9,
                p_value: 0.01,
                adjusted_p_value: None,
            },
            ScoredLink {
                link: LaggedLink {
                    source: y,
                    source_lag: Lag::CONTEMPORANEOUS,
                    target: x,
                    target_lag: Lag::CONTEMPORANEOUS,
                },
                statistic: 0.4,
                p_value: 0.04,
                adjusted_p_value: None,
            },
        ];
        let out = symmetrize_contemporaneous_links(both);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].p_value, 0.04); // conservative max-p
        assert_eq!(out[0].link.source, x);
        assert_eq!(out[0].link.target, y);
    }

    #[test]
    fn symmetrize_passes_lagged_unchanged() {
        let link = ScoredLink {
            link: LaggedLink {
                source: VariableId::from_raw(0),
                source_lag: Lag::from_raw(1),
                target: VariableId::from_raw(1),
                target_lag: Lag::CONTEMPORANEOUS,
            },
            statistic: 0.5,
            p_value: 0.01,
            adjusted_p_value: None,
        };
        let out = symmetrize_contemporaneous_links(vec![link]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].link.source_lag.raw(), 1);
    }

    #[test]
    fn exclude_contemporaneous_leaves_lag0_unadjusted() {
        let lagged = ScoredLink {
            link: LaggedLink {
                source: VariableId::from_raw(0),
                source_lag: Lag::from_raw(1),
                target: VariableId::from_raw(1),
                target_lag: Lag::CONTEMPORANEOUS,
            },
            statistic: 0.5,
            p_value: 0.01,
            adjusted_p_value: None,
        };
        let contemp = ScoredLink {
            link: LaggedLink {
                source: VariableId::from_raw(0),
                source_lag: Lag::CONTEMPORANEOUS,
                target: VariableId::from_raw(1),
                target_lag: Lag::CONTEMPORANEOUS,
            },
            statistic: 0.4,
            p_value: 0.02,
            adjusted_p_value: None,
        };
        let out = threshold_scored_links(
            vec![lagged, contemp],
            Some(FdrAdjustment::bh()),
            1.0, // keep all
        );
        assert_eq!(out.len(), 2);
        let lagged_out = out.iter().find(|s| s.link.source_lag.raw() == 1).unwrap();
        let contemp_out = out.iter().find(|s| s.link.source_lag.is_contemporaneous()).unwrap();
        assert!(lagged_out.adjusted_p_value.is_some());
        assert!(contemp_out.adjusted_p_value.is_none());
    }

    #[test]
    fn include_contemporaneous_adjusts_lag0() {
        let contemp = ScoredLink {
            link: LaggedLink {
                source: VariableId::from_raw(0),
                source_lag: Lag::CONTEMPORANEOUS,
                target: VariableId::from_raw(1),
                target_lag: Lag::CONTEMPORANEOUS,
            },
            statistic: 0.4,
            p_value: 0.02,
            adjusted_p_value: None,
        };
        let cfg = FdrAdjustment::bh().with_exclude_contemporaneous(false);
        let out = threshold_scored_links(vec![contemp], Some(cfg), 1.0);
        assert_eq!(out.len(), 1);
        assert!(out[0].adjusted_p_value.is_some());
    }
}
