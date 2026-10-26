//! Shared post-MCI discovery pipeline helpers.
//!
//! Sepset remapping and result-finishing used by PCMCI+, LPCMCI, and J-PCMCI+.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_graph::{DenseNodeId, NodeRef};

use crate::error::DiscoveryError;
use crate::orientation::OrientationState;
use crate::result::{AlgorithmRecord, DiscoveryDiagnostic, DiscoveryPerformanceRecord, PcSepsets};

/// Resolve `v`'s dense id from a `VariableId -> array index` map (index == dense id when
/// variables were added to the graph in that same order).
///
/// # Errors
///
/// `v` not present in `var_index`.
pub(crate) fn dense_of(
    var_index: &HashMap<VariableId, usize>,
    v: VariableId,
) -> Result<DenseNodeId, DiscoveryError> {
    let idx = *var_index
        .get(&v)
        .ok_or_else(|| DiscoveryError::data_msg(format!("unknown variable {v:?}")))?;
    Ok(DenseNodeId::from_raw(crate::indexing::dense_u32(idx)))
}

/// Map lagged `(variable, lag)` pairs — and context nodes as `(variable, 0)` — to dense ids.
#[must_use]
pub(crate) fn lagged_node_index(nodes: &[NodeRef]) -> HashMap<(u32, u32), DenseNodeId> {
    let mut node_ids = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        match node {
            NodeRef::Lagged { variable, lag } => {
                node_ids.insert(
                    (variable.raw(), lag.raw()),
                    DenseNodeId::from_raw(crate::indexing::dense_u32(i)),
                );
            }
            NodeRef::Context { variable, .. } => {
                node_ids.insert(
                    (variable.raw(), 0),
                    DenseNodeId::from_raw(crate::indexing::dense_u32(i)),
                );
            }
            NodeRef::Static(_) | NodeRef::Unfolded { .. } => {}
        }
    }
    node_ids
}

/// Remap PC sepsets into an [`OrientationState`] keyed by dense node ids.
///
/// Sepset keys are directional; the orientation state is unordered. Entries are
/// inserted in sorted key order so the winning entry for a pair recorded in both
/// directions is deterministic (`HashMap` iteration order is not).
#[must_use]
pub(crate) fn orientation_state_from_sepsets<S: BuildHasher>(
    node_ids: &HashMap<(u32, u32), DenseNodeId, S>,
    sepsets: &PcSepsets,
) -> OrientationState {
    let mut state = OrientationState::default();
    let mut sepset_entries: Vec<_> = sepsets.iter().collect();
    sepset_entries
        .sort_by_key(|((s, slag, t, tlag), _)| (s.raw(), slag.raw(), t.raw(), tlag.raw()));
    for ((s, slag, t, tlag), sep) in sepset_entries {
        let Some(&sa) = node_ids.get(&(s.raw(), slag.raw())) else {
            continue;
        };
        let Some(&tb) = node_ids.get(&(t.raw(), tlag.raw())) else {
            continue;
        };
        let mapped: Vec<DenseNodeId> =
            sep.iter().filter_map(|(v, l)| node_ids.get(&(v.raw(), l.raw())).copied()).collect();
        state.set_sepset(sa, tb, Arc::from(mapped));
    }
    state
}

/// Build an [`AlgorithmRecord`] from id + config digest.
#[must_use]
pub(crate) fn algorithm_record(id: &str, config: impl Into<String>) -> AlgorithmRecord {
    AlgorithmRecord { id: Arc::from(id), config: Arc::from(config.into()) }
}

/// Append a discovery diagnostic.
pub(crate) fn push_diagnostic(
    diagnostics: &mut Vec<DiscoveryDiagnostic>,
    code: &str,
    message: impl Into<String>,
) {
    diagnostics
        .push(DiscoveryDiagnostic { code: Arc::from(code), message: Arc::from(message.into()) });
}

/// The static structure-learning algorithms whose identity is recorded in results.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum StaticAlgorithm {
    Pc,
    Fci,
    Rfci,
    Ges,
    DirectLingam,
    Notears,
}

impl StaticAlgorithm {
    /// The id the algorithm carries in algorithm records, assumptions and evidence sources.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Pc => "pc",
            Self::Fci => "fci",
            Self::Rfci => "rfci",
            Self::Ges => "ges",
            Self::DirectLingam => "direct_lingam",
            Self::Notears => "notears",
        }
    }
}

/// Multiple-testing control of the static algorithms: Benjamini–Hochberg over every tested
/// undirected edge when `enabled`, none otherwise.
pub(crate) fn static_fdr(enabled: bool) -> Option<antecedent_stats::FdrAdjustment> {
    enabled.then(static_bh_fdr)
}

/// Benjamini–Hochberg over all static edges, the default of the static algorithms.
pub(crate) fn static_bh_fdr() -> antecedent_stats::FdrAdjustment {
    antecedent_stats::FdrAdjustment::bh().with_exclude_contemporaneous(false)
}

/// A count as the `u64` of a result record.
///
/// # Errors
///
/// [`DiscoveryError::Resource`] when the count does not fit, rather than a saturated value
/// that would read as a real one.
pub(crate) fn count_u64(count: usize) -> Result<u64, DiscoveryError> {
    u64::try_from(count)
        .map_err(|_| DiscoveryError::Resource(format!("count {count} does not fit in u64")))
}

/// Copy performance counters and set retained-link count.
#[must_use]
pub(crate) fn with_links_retained(
    mut performance: DiscoveryPerformanceRecord,
    links_retained: usize,
) -> DiscoveryPerformanceRecord {
    performance.links_retained = links_retained as u64;
    performance
}
