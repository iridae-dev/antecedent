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
    CachedResult, CausalState, ChangedInput, ConstraintId, DataBatchRef, DataCatalog, DataVersion,
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
    let mut stream = antecedent_core::CanonicalWriter::new("antecedent.state.result_lineage.v1");
    for reference in identities.refs().iter() {
        stream.text(reference.domain.as_str()).bytes(reference.digest.as_bytes());
    }
    SemanticDigest::from_bytes(stream.finish())
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
/// nothing here can mark an uncomputed result fresh. A stage input that changed
/// since the previous publication (data, graph evidence, a constraint, an
/// assumption, a recorded intervention) must show in the identity layer that
/// reads it, so a premise event answered by an identical lineage is refused too.
/// The result's three-state class is [`CausalState::revalidation`].
///
/// A query published here is lineage-bound: the unchecked
/// `CausalState::refresh_results` path refuses it.
///
/// # Errors
///
/// Stale expected version, a snapshot or identity layer that predates a
/// changed stage input, unknown query, or cache budget refusal.
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
                identities: Some(*identities),
            }
        })
        .collect();
    state.publish_results_at(expected, &updates).map_err(CausalError::from)
}

#[cfg(test)]
mod tests {
    use antecedent_core::ContractIdentities;

    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::from_bytes([byte; 32])
    }

    #[test]
    fn the_lineage_fingerprint_is_a_pinned_golden_vector_over_every_layer() {
        let identities = ContractIdentities::new(
            digest(1),
            digest(2),
            Some(digest(3)),
            Some(digest(4)),
            digest(5),
            digest(6),
            digest(7),
        );
        assert_eq!(
            result_lineage_fingerprint(&identities).to_hex(),
            "752044c4ff36b8e2237c59563c505fafb619f4eb7b3bce86a10e9d3bc04f7dca"
        );
        let moved = ContractIdentities::new(
            digest(1),
            digest(2),
            Some(digest(3)),
            Some(digest(4)),
            digest(5),
            digest(6),
            digest(8),
        );
        assert_ne!(result_lineage_fingerprint(&moved), result_lineage_fingerprint(&identities));
    }
}
