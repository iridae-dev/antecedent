//! incremental antecedent-state facade helpers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CacheBudget;

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
) -> Result<antecedent_core::StateVersion, CausalError> {
    state.apply(event).map_err(CausalError::from)
}
