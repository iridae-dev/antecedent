//! Shared post-MCI discovery pipeline helpers.
//!
//! Sepset remapping and result-finishing used by PCMCI+, LPCMCI, and J-PCMCI+.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::sync::Arc;

use causal_graph::{DenseNodeId, NodeRef};

use crate::orientation::OrientationState;
use crate::result::{AlgorithmRecord, DiscoveryDiagnostic, DiscoveryPerformanceRecord, PcSepsets};

/// Map lagged `(variable, lag)` pairs to dense node ids for orientation.
#[must_use]
pub fn lagged_node_index(nodes: &[NodeRef]) -> HashMap<(u32, u32), DenseNodeId> {
    let mut node_ids = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        if let NodeRef::Lagged { variable, lag } = node {
            node_ids.insert((variable.raw(), lag.raw()), DenseNodeId::from_raw(i as u32));
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
pub fn orientation_state_from_sepsets<S: BuildHasher>(
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
pub fn algorithm_record(id: &str, config: impl Into<String>) -> AlgorithmRecord {
    AlgorithmRecord { id: Arc::from(id), config: Arc::from(config.into()) }
}

/// Append a discovery diagnostic.
pub fn push_diagnostic(
    diagnostics: &mut Vec<DiscoveryDiagnostic>,
    code: &str,
    message: impl Into<String>,
) {
    diagnostics
        .push(DiscoveryDiagnostic { code: Arc::from(code), message: Arc::from(message.into()) });
}

/// Copy performance counters and set retained-link count.
#[must_use]
pub fn with_links_retained(
    mut performance: DiscoveryPerformanceRecord,
    links_retained: usize,
) -> DiscoveryPerformanceRecord {
    performance.links_retained = links_retained as u64;
    performance
}
