//! incremental antecedent-state facade helpers.
//!
//! `apply` stays events/invalidation only. The facade inspects stale layers,
//! recomputes through prepared handles, and publishes only the data/program
//! version actually computed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CacheBudget, ContractIdentities, QueryId, StateVersion};

use crate::error::CausalError;

pub use antecedent_state::{
    CachedResult, CausalState, ConstraintId, DataBatchRef, DataCatalog, DataVersion,
    GraphConstraintRecord, GraphEvidenceRecord, GraphEvidenceStore, GraphScoreCacheKey,
    GraphScoreData, GraphScoreFamily, InterventionRecord, InvalidationEntry, InvalidationLog,
    InvalidationTarget, LagIndexCacheEntry, LagIndexCacheKey, LgssmParams, LinearOlsSuffStats,
    LocalScoreCache, ModelRecord, ModelStore, ParentSetOp, ParticleFilterState, QueryRecord,
    QueryStore, ResultStore, RetentionPolicy, RollingMechanismDiagnostics, StateError, StateEvent,
    StreamingCovariance, SuffStatStore, evict_mechanism_diag, full_graph_score,
    insert_mechanism_diag,
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

/// Fingerprint bound to the data snapshot and program actually computed.
#[must_use]
pub fn result_lineage_fingerprint(identities: &ContractIdentities) -> u64 {
    let snapshot = identities.data_snapshot.as_bytes();
    let program = identities.program.as_bytes();
    u64::from_le_bytes([
        snapshot[0],
        snapshot[1],
        snapshot[2],
        snapshot[3],
        program[0],
        program[1],
        program[2],
        program[3],
    ])
}

/// Publish caller-computed result fingerprints only for `expected`.
///
/// Inspect stale layers with [`CausalState::stale_queries`], recompute through
/// [`crate::PreparedStudy`], then call this. A version that advanced during
/// recompute, or a failed/cancelled recompute that never reaches this function,
/// cannot mark results fresh.
///
/// # Errors
///
/// Stale expected version, unknown query, or cache budget refusal.
pub fn publish_recomputed_results(
    state: &mut CausalState,
    expected: StateVersion,
    updates: &[(QueryId, u64, u64)],
) -> Result<(), CausalError> {
    state.refresh_results_at(expected, updates).map_err(CausalError::from)
}
