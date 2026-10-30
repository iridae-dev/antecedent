//! The outcome of one discovery run in the neutral form the adapters retain, with the
//! identity that decides whether two runs reached the same result.
//!
//! The *result identity* covers everything the algorithm returned that is not an execution
//! detail: the structure, the per-edge evidence (statistic, p-values, interval, separating
//! set, provenance), the separating sets, the diagnostics and the per-pass records. It
//! excludes thread counts, byte counts and work counters. Two runs with equal result
//! identity are an algorithmic match; the identity says nothing about whether either result
//! is close to a true graph.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CanonicalWriter;

use crate::fci::StaticPagDiscoveryResult;
use crate::pc::StaticCpdagDiscoveryResult;
use crate::result::{CpdagDiscoveryResult, DagDiscoveryResult, DiscoveryResult};

use super::identity::{Digest, SELECTIVE_IDENTITY_VERSION};
use super::structure::DiscoveredStructure;

const RESULT_TAG: &str = "antecedent.selective_discovery.result.v1";

/// The native result of the algorithm that ran.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum NativeResult {
    /// Static PC.
    StaticCpdag(StaticCpdagDiscoveryResult),
    /// Static FCI.
    StaticPag(StaticPagDiscoveryResult),
    /// PCMCI.
    TemporalDag(DagDiscoveryResult),
    /// PCMCI+.
    TemporalCpdag(CpdagDiscoveryResult),
}

/// One run's retained outcome.
#[derive(Clone, Debug)]
pub struct DiscoveryOutcome {
    /// The discovered structure in canonical form.
    pub structure: DiscoveredStructure,
    /// What the algorithm returned.
    pub native: NativeResult,
    /// `(code, message)` of every diagnostic, in the algorithm's order.
    pub diagnostics: Vec<(String, String)>,
    /// Diagnostics recording exhausted search bounds or parent-limit removals without
    /// separating evidence; those absences do not establish a complete structure.
    pub bounded_search_receipts: Vec<(String, String)>,
    /// Whether a bound left the search incomplete or removed an unseparated adjacency.
    pub search_incomplete: bool,
    /// Whether orientation evidence conflicted (a collider or an orientation rule disagreed).
    pub orientation_conflict: bool,
    /// Identity of the structure.
    pub structure_identity: Digest,
    /// Identity of everything the algorithm returned except execution detail.
    pub result_identity: Digest,
}

fn write_link(writer: &mut CanonicalWriter, link: &crate::result::LaggedLink) {
    writer
        .word(u64::from(link.source.raw()))
        .word(u64::from(link.source_lag.raw()))
        .word(u64::from(link.target.raw()))
        .word(u64::from(link.target_lag.raw()));
}

fn write_lagged_parents(writer: &mut CanonicalWriter, parents: &[crate::result::LaggedParent]) {
    writer.word(parents.len() as u64);
    for (variable, lag) in parents {
        writer.word(u64::from(variable.raw())).word(u64::from(lag.raw()));
    }
}

fn result_identity<G, R>(result: &DiscoveryResult<G, R>, structure: Digest) -> Digest {
    let mut w = CanonicalWriter::new(RESULT_TAG);
    w.word(SELECTIVE_IDENTITY_VERSION).bytes(&structure);
    w.text(&result.algorithm.id).text(&result.algorithm.config);

    let mut edges: Vec<_> = result.evidence.edge_evidence.iter().collect();
    edges.sort_by_key(|e| e.link);
    w.word(edges.len() as u64);
    for edge in edges {
        write_link(&mut w, &edge.link);
        w.opt_real(edge.statistic).opt_real(edge.p_value).opt_real(edge.adjusted_p_value);
        match edge.interval {
            Some((lo, hi)) => w.word(1).real(lo).real(hi),
            None => w.word(0),
        };
        match &edge.separating_set {
            Some(set) => {
                w.word(1);
                write_lagged_parents(&mut w, set);
            }
            None => {
                w.word(0);
            }
        }
        let tags: Vec<&str> = edge.provenance.iter().map(|tag| &**tag).collect();
        w.texts(&tags);
    }

    let mut links: Vec<_> = result.evidence.links.iter().collect();
    links.sort_by_key(|l| l.link);
    w.word(links.len() as u64);
    for scored in links {
        write_link(&mut w, &scored.link);
        w.real(scored.statistic).real(scored.p_value).opt_real(scored.adjusted_p_value);
    }

    let mut sepsets: Vec<_> = result.sepsets.iter().collect();
    sepsets.sort_by_key(|(key, _)| **key);
    w.word(sepsets.len() as u64);
    for ((x, lx, y, ly), set) in sepsets {
        w.word(u64::from(x.raw()))
            .word(u64::from(lx.raw()))
            .word(u64::from(y.raw()))
            .word(u64::from(ly.raw()));
        write_lagged_parents(&mut w, set);
    }

    w.word(result.diagnostics.len() as u64);
    for d in &result.diagnostics {
        w.text(&d.code).text(&d.message);
    }
    w.word(result.iterations.len() as u64);
    for it in &result.iterations {
        w.text(&it.label).word(it.ci_tests);
    }
    w.finish()
}

fn diagnostics_of<G, R>(result: &DiscoveryResult<G, R>) -> Vec<(String, String)> {
    result.diagnostics.iter().map(|d| (d.code.to_string(), d.message.to_string())).collect()
}

impl DiscoveryOutcome {
    fn assemble<G, R>(
        structure: DiscoveredStructure,
        result: &DiscoveryResult<G, R>,
        native: NativeResult,
    ) -> Self {
        let diagnostics = diagnostics_of(result);
        let bounded: Vec<(String, String)> = diagnostics
            .iter()
            .filter(|(code, _)| {
                code.contains("budget") || code.ends_with(".parent_limit_removed_adjacencies")
            })
            .cloned()
            .collect();
        let conflict = diagnostics.iter().any(|(code, _)| {
            code.contains("orientation_conflict") || code == "orientation.conflicts"
        }) || structure
            .edges()
            .iter()
            .any(|e| matches!(e.unresolved_reason(), Some("orientation_conflict")));
        let structure_identity = structure.identity();
        let result_identity = result_identity(result, structure_identity);
        Self {
            structure,
            native,
            search_incomplete: !bounded.is_empty(),
            bounded_search_receipts: bounded,
            diagnostics,
            orientation_conflict: conflict,
            structure_identity,
            result_identity,
        }
    }

    /// From a static PC result.
    #[must_use]
    pub fn from_static_cpdag(result: &StaticCpdagDiscoveryResult) -> Self {
        let structure = DiscoveredStructure::from_cpdag(&result.evidence.graph);
        Self::assemble(structure, result, NativeResult::StaticCpdag(result.clone()))
    }

    /// From a static FCI result.
    #[must_use]
    pub fn from_static_pag(result: &StaticPagDiscoveryResult) -> Self {
        let structure = DiscoveredStructure::from_pag(&result.evidence.graph);
        Self::assemble(structure, result, NativeResult::StaticPag(result.clone()))
    }

    /// From a PCMCI result.
    #[must_use]
    pub fn from_temporal_dag(result: &DagDiscoveryResult) -> Self {
        let structure = DiscoveredStructure::from_temporal_dag(&result.evidence.graph);
        Self::assemble(structure, result, NativeResult::TemporalDag(result.clone()))
    }

    /// From a PCMCI+ result.
    #[must_use]
    pub fn from_temporal_cpdag(result: &CpdagDiscoveryResult) -> Self {
        let structure = DiscoveredStructure::from_temporal_cpdag(&result.evidence.graph);
        Self::assemble(structure, result, NativeResult::TemporalCpdag(result.clone()))
    }
}
