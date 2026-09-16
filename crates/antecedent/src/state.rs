//! incremental antecedent-state facade helpers.
//!
//! `apply` stays events/invalidation only. The facade inspects stale layers,
//! recomputes through prepared handles, and publishes only the data/program
//! version actually computed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CacheBudget, ContractIdentities, QueryId, SemanticDigest, StateVersion};

use crate::error::CausalError;

pub use antecedent_state::{
    CachedResult, CausalState, ConstraintId, DataBatchRef, DataCatalog, DataVersion,
    GraphConstraintRecord, GraphEvidenceRecord, GraphEvidenceStore, GraphScoreCacheKey,
    GraphScoreData, GraphScoreFamily, InterventionRecord, InvalidationEntry, InvalidationLog,
    InvalidationTarget, LagIndexCacheEntry, LagIndexCacheKey, LgssmParams, LinearOlsSuffStats,
    LocalScoreCache, ModelRecord, ModelStore, ParentSetOp, ParticleFilterState, QueryRecord,
    QueryStore, ResultPublication, ResultStore, RetentionPolicy, RollingMechanismDiagnostics,
    StateError, StateEvent, StreamingCovariance, SuffStatStore, evict_mechanism_diag,
    full_graph_score, insert_mechanism_diag,
};

/// Construct a fresh [`CausalState`] with the given cache budget.
#[must_use]
pub fn new_antecedent_state(budget: CacheBudget) -> CausalState {
    CausalState::new(budget)
}

/// Apply a state event without auto-rerunning analyses.
///
/// # Errors
///
/// Propagates state update failures.
pub fn apply_state_event(
    state: &mut CausalState,
    event: StateEvent,
) -> Result<StateVersion, CausalError> {
    state.apply(event).map_err(CausalError::from)
}

/// Complete lineage digest of the contract a result was computed under.
///
/// Covers every identity layer — target (including its population),
/// identification premises and products, program, inference binding,
/// observation, and data snapshot — so two estimands, two priors, or two
/// bootstrap budgets never share a lineage key.
#[must_use]
pub fn result_lineage_fingerprint(identities: &ContractIdentities) -> SemanticDigest {
    let mut bytes = Vec::new();
    for reference in identities.refs().iter() {
        bytes.extend_from_slice(reference.domain.as_str().as_bytes());
        bytes.extend_from_slice(reference.digest.as_bytes());
    }
    SemanticDigest::from_bytes(antecedent_io::payload_digest("state.result_lineage", &bytes))
}

/// Leading bytes of a lineage digest, for the store's `u64` key slot.
fn store_fingerprint(lineage: &SemanticDigest) -> u64 {
    let mut head = [0u8; 8];
    head.copy_from_slice(&lineage.as_bytes()[..8]);
    u64::from_le_bytes(head)
}

/// Publish recomputed results for `expected`, binding what was recomputed.
///
/// Inspect stale layers with [`CausalState::stale_queries`], recompute through
/// [`crate::PreparedStudy`], then call this with the identities of each
/// recomputed contract. A version that advanced during recompute, a failed or
/// cancelled recompute that never reaches this function, and a republication
/// of the data snapshot recorded before the latest data event are all refused:
/// nothing here can mark an uncomputed result fresh.
///
/// # Errors
///
/// Stale expected version, a snapshot that predates the latest data event,
/// unknown query, or cache budget refusal.
pub fn publish_recomputed_results(
    state: &mut CausalState,
    expected: StateVersion,
    updates: &[(QueryId, &ContractIdentities, u64)],
) -> Result<(), CausalError> {
    let updates: Vec<ResultPublication> = updates
        .iter()
        .map(|&(query, identities, bytes)| {
            let lineage = result_lineage_fingerprint(identities);
            ResultPublication {
                query,
                fingerprint: store_fingerprint(&lineage),
                bytes,
                lineage: Some(lineage),
                data_snapshot: Some(identities.data_snapshot),
            }
        })
        .collect();
    state.publish_results_at(expected, &updates).map_err(CausalError::from)
}
