//! [`CausalState`] apply / invalidate / recompute.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use std::sync::Arc;

use antecedent_core::recalc::{ChangeKind, Revalidation, Stage, StageIdentities, StageIdentity};
use antecedent_core::{AssumptionSet, CacheBudget, ContractIdentities, QueryId, StateVersion};

use crate::error::StateError;
use crate::event::StateEvent;
use crate::invalidation::{ChangedInput, InvalidationLog, InvalidationReason, InvalidationTarget};
use crate::refresh_contract::{RefreshChange, RefreshDisposition, RefreshRoute, route_contract};
use crate::store::{
    CachedResult, DataCatalog, DataVersion, GraphEvidenceStore, ModelStore, QueryStore,
    ResultStore, SuffStatStore,
};

/// One recomputed result a caller publishes as fresh.
#[derive(Clone, Copy, Debug)]
pub struct ResultPublication {
    /// Query whose result was recomputed.
    pub query: QueryId,
    /// Store-boundary fingerprint (truncated lineage digest).
    pub fingerprint: u64,
    /// Bytes retained for this result.
    pub bytes: u64,
    /// Complete lineage digest over every contract identity layer.
    pub lineage: Option<antecedent_core::SemanticDigest>,
    /// Data-snapshot identity the result was computed on.
    pub data_snapshot: Option<antecedent_core::SemanticDigest>,
    /// Every contract identity layer the result was computed under. With it, a
    /// republication is checked layer by layer against the stage inputs that changed since
    /// the previous publication.
    pub identities: Option<ContractIdentities>,
}

/// The contract identity layer a changed stage input must move in a real recomputation.
/// `None` for a stage no single layer reflects.
fn layer_unchanged(
    stage: Stage,
    was: &ContractIdentities,
    now: &ContractIdentities,
) -> Option<bool> {
    match stage {
        Stage::DataSnapshot => Some(was.data_snapshot == now.data_snapshot),
        Stage::Graph => Some(
            was.identification == now.identification
                && was.identification_product == now.identification_product,
        ),
        Stage::Regime => Some(was.program == now.program),
        Stage::Evidence => Some(was.observation == now.observation),
        Stage::Query => Some(was.target == now.target),
        _ => None,
    }
}

/// Incremental causal analysis state (embeddable; no service runtime).
#[derive(Clone, Debug)]
pub struct CausalState {
    /// Monotonic version.
    pub version: StateVersion,
    /// Data catalog.
    pub data_catalog: DataCatalog,
    /// Graph evidence / constraints.
    pub graph_evidence: GraphEvidenceStore,
    /// Assumptions.
    pub assumptions: AssumptionSet,
    /// Models.
    pub models: ModelStore,
    /// Queries.
    pub queries: QueryStore,
    /// Cached results.
    pub cached_results: ResultStore,
    /// Invalidation log.
    pub invalidations: InvalidationLog,
    /// Cache budget.
    pub cache_budget: CacheBudget,
    /// Sufficient statistics.
    pub suff_stats: SuffStatStore,
    /// Recorded interventions.
    pub interventions: Vec<crate::store::InterventionRecord>,
}

impl Default for CausalState {
    fn default() -> Self {
        Self::new(CacheBudget::new(64 * 1024 * 1024))
    }
}

impl CausalState {
    /// Fresh state with the given cache budget.
    #[must_use]
    pub fn new(cache_budget: CacheBudget) -> Self {
        Self {
            version: StateVersion::ZERO,
            data_catalog: DataCatalog::new(),
            graph_evidence: GraphEvidenceStore::default(),
            assumptions: AssumptionSet::new(),
            models: ModelStore::default(),
            queries: QueryStore::default(),
            cached_results: ResultStore::default(),
            invalidations: InvalidationLog::new(),
            cache_budget,
            suff_stats: SuffStatStore::default(),
            interventions: Vec::new(),
        }
    }

    /// Apply an event: bump version, update stores, record invalidations.
    /// Does **not** automatically recompute expensive analyses.
    ///
    /// # Errors
    ///
    /// Invalid event payloads.
    pub fn apply(&mut self, event: StateEvent) -> Result<StateVersion, StateError> {
        // Validate fallible events before advancing the state clock. A rejected
        // event must be observationally atomic: callers use `version` as the
        // freshness boundary for cached causal results, so consuming a version
        // on an operation that did not happen would make that evidence stamp lie.
        if let StateEvent::RemoveConstraint(id) = &event {
            if !self.graph_evidence.constraints.iter().any(|constraint| constraint.id == id.0) {
                return Err(StateError::UnknownId(format!("constraint {}", id.0)));
            }
        }
        self.version = self.version.next();
        let v = self.version;
        let input = self.changed_input(&event, v);
        match event {
            StateEvent::AppendData(batch) => {
                self.data_catalog.batches.push(batch);
                self.data_catalog.version = self.data_catalog.version.next();
                self.invalidate_data_dependents(
                    v,
                    InvalidationReason::AppendData,
                    input,
                    RefreshChange::AppendRows,
                );
            }
            StateEvent::ReplaceData(new_ver) => {
                self.data_catalog.version = new_ver;
                self.data_catalog.batches.clear();
                self.invalidate_data_dependents(
                    v,
                    InvalidationReason::ReplaceData,
                    input,
                    RefreshChange::SourceChanged,
                );
                if matches!(
                    route_contract(RefreshRoute::LinearOlsGram, RefreshChange::SourceChanged),
                    RefreshDisposition::InvalidateAndRecompute { .. }
                ) {
                    self.suff_stats.ols.clear();
                }
                if matches!(
                    route_contract(RefreshRoute::StreamingCovariance, RefreshChange::SourceChanged),
                    RefreshDisposition::InvalidateAndRecompute { .. }
                ) {
                    self.suff_stats.cov.clear();
                }
                // Drain through the budget-aware path so the bytes each slot was charged
                // are returned, not stranded until the budget is exhausted.
                let keys: Vec<_> = self.suff_stats.mechanism_diags.keys().cloned().collect();
                for key in keys {
                    crate::mechanism_diag::evict_mechanism_diag(
                        &mut self.suff_stats.mechanism_diags,
                        &key,
                        &mut self.cache_budget,
                    );
                }
                self.invalidations.push(
                    v,
                    InvalidationTarget::LagIndexes,
                    InvalidationReason::ReplaceData,
                );
                self.invalidations.push(
                    v,
                    InvalidationTarget::GraphScores,
                    InvalidationReason::ReplaceData,
                );
                self.invalidations.push(
                    v,
                    InvalidationTarget::ParticleFilters,
                    InvalidationReason::ReplaceData,
                );
                self.invalidations.push(
                    v,
                    InvalidationTarget::MechanismDiagnostics,
                    InvalidationReason::ReplaceData,
                );
            }
            StateEvent::AddGraphEvidence(rec) => {
                self.graph_evidence.evidence.push(rec);
                self.invalidate_all_results(v, InvalidationReason::AddGraphEvidence, input);
            }
            StateEvent::AddConstraint(rec) => {
                self.graph_evidence.constraints.push(rec);
                self.invalidate_all_results(v, InvalidationReason::AddConstraint, input);
            }
            StateEvent::RemoveConstraint(id) => {
                self.graph_evidence.constraints.retain(|c| c.id != id.0);
                self.invalidate_all_results(v, InvalidationReason::RemoveConstraint, input);
            }
            StateEvent::UpdateAssumption(rec) => {
                self.assumptions.push(rec);
                self.invalidate_all_results(v, InvalidationReason::UpdateAssumption, input);
            }
            StateEvent::RegisterQuery(query) => {
                let _id = self.queries.register(query);
                // Registration alone does not invalidate others.
            }
            StateEvent::RecordIntervention(rec) => {
                self.interventions.push(rec);
                self.invalidate_all_results(v, InvalidationReason::RecordIntervention, input);
            }
        }
        Ok(v)
    }

    /// The stage input an event changes, with the digest of what it changed to. The digest
    /// binds the event's content and its position in the history, so two events with equal
    /// content are still two changes. `None` for an event that changes no stage input.
    fn changed_input(&self, event: &StateEvent, at: StateVersion) -> Option<ChangedInput> {
        let at_bytes = at.raw().to_le_bytes();
        let (stage, change, identity) = match event {
            StateEvent::AppendData(batch) => (
                Stage::DataSnapshot,
                ChangeKind::Modified,
                StageIdentity::of(
                    "state.append_data.v1",
                    &[
                        &at_bytes,
                        batch.id.as_bytes(),
                        &batch.nrows.to_le_bytes(),
                        &batch.bytes.to_le_bytes(),
                    ],
                ),
            ),
            StateEvent::ReplaceData(version) => (
                Stage::DataSnapshot,
                ChangeKind::Modified,
                StageIdentity::of(
                    "state.replace_data.v1",
                    &[&at_bytes, &version.raw().to_le_bytes()],
                ),
            ),
            StateEvent::AddGraphEvidence(rec) => (
                Stage::Graph,
                ChangeKind::Added,
                StageIdentity::of(
                    "state.graph_evidence.v1",
                    &[&at_bytes, rec.id.as_bytes(), &rec.fingerprint.to_le_bytes()],
                ),
            ),
            StateEvent::AddConstraint(rec) => (
                Stage::Graph,
                ChangeKind::Added,
                StageIdentity::of(
                    "state.constraint_added.v1",
                    &[&at_bytes, rec.id.as_bytes(), &rec.fingerprint.to_le_bytes()],
                ),
            ),
            StateEvent::RemoveConstraint(id) => (
                Stage::Graph,
                ChangeKind::Removed,
                StageIdentity::of("state.constraint_removed.v1", &[&at_bytes, id.0.as_bytes()]),
            ),
            // The assumption set is a premise of identification, which the graph stage
            // feeds: an updated assumption is a modified premise.
            StateEvent::UpdateAssumption(_) => (
                Stage::Graph,
                ChangeKind::Modified,
                StageIdentity::of(
                    "state.assumption_updated.v1",
                    &[&at_bytes, &(self.assumptions.len() as u64).to_le_bytes()],
                ),
            ),
            // A recorded intervention is an observation of what was done: it moves the data
            // the result is computed on, not the declared regime of the query.
            StateEvent::RecordIntervention(rec) => (
                Stage::DataSnapshot,
                ChangeKind::Added,
                StageIdentity::of(
                    "state.intervention.v1",
                    &[&at_bytes, rec.id.as_bytes(), &rec.fingerprint.to_le_bytes()],
                ),
            ),
            StateEvent::RegisterQuery(_) => return None,
        };
        Some(ChangedInput { stage, change, identity })
    }

    /// Own-input identities of the state's declared inputs: the graph (evidence, constraints,
    /// assumptions), the data snapshot (catalog version and batches) and the recorded
    /// interventions. Two states with equal identities hold equal inputs.
    #[must_use]
    pub fn input_identities(&self) -> StageIdentities {
        fn refs(parts: &[Vec<u8>]) -> Vec<&[u8]> {
            parts.iter().map(Vec::as_slice).collect()
        }
        let mut graph: Vec<Vec<u8>> = Vec::new();
        for rec in &self.graph_evidence.evidence {
            graph.push(rec.id.as_bytes().to_vec());
            graph.push(rec.fingerprint.to_le_bytes().to_vec());
        }
        for rec in &self.graph_evidence.constraints {
            graph.push(rec.id.as_bytes().to_vec());
            graph.push(rec.fingerprint.to_le_bytes().to_vec());
        }
        graph.push((self.assumptions.len() as u64).to_le_bytes().to_vec());
        let mut data: Vec<Vec<u8>> = vec![self.data_catalog.version.raw().to_le_bytes().to_vec()];
        for batch in &self.data_catalog.batches {
            data.push(batch.id.as_bytes().to_vec());
            data.push(batch.nrows.to_le_bytes().to_vec());
            data.push(batch.bytes.to_le_bytes().to_vec());
        }
        for rec in &self.interventions {
            data.push(rec.id.as_bytes().to_vec());
            data.push(rec.fingerprint.to_le_bytes().to_vec());
        }
        StageIdentities::new()
            .with(Stage::Graph, StageIdentity::of("state.graph.v1", &refs(&graph)))
            .with(Stage::DataSnapshot, StageIdentity::of("state.data.v1", &refs(&data)))
    }

    /// The three-state class of a query's result: `still_valid` when nothing it rests on
    /// changed since it was published, `recompute` when a stage input changed (or it was never
    /// computed), `withdrawn` when a premise it rested on was removed. `None` for an
    /// unregistered query.
    #[must_use]
    pub fn revalidation(&self, query: QueryId) -> Option<Revalidation> {
        let rec = self.queries.queries.get(&query)?;
        let pending = rec
            .pending_inputs
            .iter()
            .fold(Revalidation::StillValid, |class, input| class.worst(input.revalidation()));
        Some(match rec.result_valid_at {
            None => pending.worst(Revalidation::Recompute),
            Some(since) => pending.worst(self.invalidations.revalidation_since(query, since)),
        })
    }

    /// Whether a query's cached result is stale.
    #[must_use]
    pub fn is_stale(&self, query: QueryId) -> bool {
        let Some(rec) = self.queries.queries.get(&query) else {
            return true;
        };
        match rec.result_valid_at {
            None => true,
            Some(since) => self.invalidations.query_stale_since(query, since),
        }
    }

    /// List registered queries that are currently stale.
    #[must_use]
    pub fn stale_queries(&self) -> Vec<QueryId> {
        self.queries.queries.keys().copied().filter(|q| self.is_stale(*q)).collect()
    }

    /// Caller-driven recomputation hook for results that carry no lineage: marks selected
    /// queries fresh after the caller supplies new result fingerprints. Does not run
    /// estimators itself.
    ///
    /// This is the unchecked path and is refused for any query whose last publication
    /// carried a lineage: such a query is republished only through [`Self::publish_results`]
    /// with the identities of the recomputation, so freshness is never asserted for a result
    /// that cannot show what it was recomputed from.
    ///
    /// # Errors
    ///
    /// Unknown query, a lineage-bound query, or cache budget refusal.
    pub fn refresh_results(&mut self, updates: &[(QueryId, u64, u64)]) -> Result<(), StateError> {
        let updates: Vec<ResultPublication> = updates
            .iter()
            .map(|&(query, fingerprint, bytes)| ResultPublication {
                query,
                fingerprint,
                bytes,
                lineage: None,
                data_snapshot: None,
                identities: None,
            })
            .collect();
        self.publish_results(&updates)
    }

    /// Publish recomputed results, refusing a republication that the data
    /// events say cannot be a recomputation.
    ///
    /// A result whose data snapshot equals the one published under an earlier
    /// data-catalog version was not recomputed on the current data, whatever
    /// its fingerprint says.
    ///
    /// # Errors
    ///
    /// Unknown query, a snapshot that predates the latest data event, or cache
    /// budget refusal.
    pub fn publish_results(&mut self, updates: &[ResultPublication]) -> Result<(), StateError> {
        for update in updates {
            if !self.queries.queries.contains_key(&update.query) {
                return Err(StateError::UnknownId(format!("query {}", update.query.raw())));
            }
            let Some(record) = self.queries.queries.get(&update.query) else {
                continue;
            };
            let Some(previous) = record.last_published else {
                continue;
            };
            if update.lineage.is_none() {
                return Err(StateError::InvalidEvent(format!(
                    "query {} was published with a lineage; republish it with the identities of \
                     the recomputation, not with a bare fingerprint",
                    update.query.raw()
                )));
            }
            if let Some(published) = update.data_snapshot {
                if previous.data_version != self.data_catalog.version
                    && published == previous.data_snapshot
                {
                    return Err(StateError::InvalidEvent(format!(
                        "query {} republishes the data snapshot recorded before the latest data \
                         event; recompute on the current data before publishing",
                        update.query.raw()
                    )));
                }
            }
            // Every stage input that changed since the previous publication must be
            // reflected in the identity layer that reads it: an identical lineage after a
            // premise, constraint or intervention event is not a recomputation.
            for input in &record.pending_inputs {
                let unchanged = match (&previous.identities, &update.identities) {
                    (Some(was), Some(now)) => layer_unchanged(input.stage, was, now),
                    _ => (update.lineage == Some(previous.lineage)).then_some(true),
                };
                if unchanged == Some(true) {
                    return Err(StateError::InvalidEvent(format!(
                        "query {} republishes the {} layer recorded before the {} change; \
                         recompute under the current inputs before publishing",
                        update.query.raw(),
                        input.stage,
                        input.stage,
                    )));
                }
            }
        }
        self.commit_results(updates)
    }

    fn commit_results(&mut self, updates: &[ResultPublication]) -> Result<(), StateError> {
        // Stage the whole batch first. `ResultStore::insert` is individually
        // atomic, but applying it directly to `self` would leave a prefix of the
        // batch committed when a later entry is unknown or exceeds the budget.
        // A refresh batch represents one caller-supplied recomputation boundary,
        // so publish all freshness stamps together or none of them.
        for update in updates {
            if !self.queries.queries.contains_key(&update.query) {
                return Err(StateError::UnknownId(format!("query {}", update.query.raw())));
            }
        }
        let mut staged_results = self.cached_results.clone();
        let mut staged_budget = self.cache_budget;
        for update in updates {
            staged_results.insert(
                CachedResult {
                    query: update.query,
                    fingerprint: update.fingerprint,
                    bytes: update.bytes,
                    computed_at: self.version,
                },
                &mut staged_budget,
            )?;
        }
        self.cached_results = staged_results;
        self.cache_budget = staged_budget;
        let data_version = self.data_catalog.version;
        for update in updates {
            let Some(record) = self.queries.queries.get_mut(&update.query) else {
                return Err(StateError::UnknownId(format!("query {}", update.query.raw())));
            };
            record.result_valid_at = Some(self.version);
            record.pending_inputs.clear();
            if let (Some(lineage), Some(data_snapshot)) = (update.lineage, update.data_snapshot) {
                record.last_published = Some(crate::store::PublishedLineage {
                    lineage,
                    data_snapshot,
                    data_version,
                    identities: update.identities,
                });
            }
        }
        Ok(())
    }

    /// Like [`Self::refresh_results`], but refuse if `expected` is no longer
    /// the current state version. A stale commit cannot mark results fresh.
    ///
    /// # Errors
    ///
    /// Version mismatch, unknown query, or cache budget refusal.
    pub fn refresh_results_at(
        &mut self,
        expected: StateVersion,
        updates: &[(QueryId, u64, u64)],
    ) -> Result<(), StateError> {
        self.require_version(expected)?;
        self.refresh_results(updates)
    }

    /// Like [`Self::publish_results`], refusing a stale expected version.
    ///
    /// # Errors
    ///
    /// Version mismatch, a snapshot that predates the latest data event,
    /// unknown query, or cache budget refusal.
    pub fn publish_results_at(
        &mut self,
        expected: StateVersion,
        updates: &[ResultPublication],
    ) -> Result<(), StateError> {
        self.require_version(expected)?;
        self.publish_results(updates)
    }

    fn require_version(&self, expected: StateVersion) -> Result<(), StateError> {
        if self.version != expected {
            return Err(StateError::InvalidEvent(format!(
                "stale commit: expected version {} but state is {}",
                expected.raw(),
                self.version.raw()
            )));
        }
        Ok(())
    }

    /// Current data version.
    #[must_use]
    pub fn data_version(&self) -> DataVersion {
        self.data_catalog.version
    }

    fn invalidate_data_dependents(
        &mut self,
        v: StateVersion,
        reason: InvalidationReason,
        input: Option<ChangedInput>,
        change: RefreshChange,
    ) {
        // Each kept route answers the change as its contract says: an in-place update the
        // caller applies, or a drop and recompute. The caches are cleared only when the
        // contract says so, so the table and this method cannot drift apart.
        let drops = |route| {
            matches!(
                route_contract(route, change),
                RefreshDisposition::InvalidateAndRecompute { .. }
            )
        };
        self.invalidate_all_results(v, reason, input);
        for key in self.suff_stats.ols.keys().cloned().collect::<Vec<_>>() {
            self.invalidations.push(
                v,
                InvalidationTarget::SuffStat(key),
                format!("{}:ols_may_need_append", reason.as_str()),
            );
        }
        self.invalidations.push(v, InvalidationTarget::LagIndexes, reason);
        // Lag indexes become stale on data change; drop metadata.
        if drops(RefreshRoute::LagIndexCache) {
            self.suff_stats.lag_indexes.clear();
        }
        // Graph scores are data-dependent; clear until a streaming family exists.
        if drops(RefreshRoute::GraphScoreCache) {
            self.suff_stats.graph_scores.clear();
        }
        self.invalidations.push(v, InvalidationTarget::GraphScores, reason);
        // Particle filters: drop on append (caller re-inits / steps a fresh filter).
        if drops(RefreshRoute::ParticleFilter) {
            self.suff_stats.particle_filters.clear();
        }
        self.invalidations.push(v, InvalidationTarget::ParticleFilters, reason);
        // Mechanism diagnostics stream via caller append; log stale without clearing.
        self.invalidations.push(v, InvalidationTarget::MechanismDiagnostics, reason);
    }

    fn invalidate_all_results(
        &mut self,
        v: StateVersion,
        reason: InvalidationReason,
        input: Option<ChangedInput>,
    ) {
        let ids: Vec<QueryId> = self.cached_results.results.keys().copied().collect();
        let log = |log: &mut InvalidationLog,
                   target: InvalidationTarget,
                   reason: InvalidationReason| match input {
            Some(input) => log.push_input(v, target, reason, input),
            None => log.push(v, target, reason),
        };
        for q in ids {
            self.cached_results.remove(q, &mut self.cache_budget);
            log(&mut self.invalidations, InvalidationTarget::QueryResult(q), reason);
            if let Some(rec) = self.queries.queries.get_mut(&q) {
                rec.result_valid_at = None;
            }
        }
        log(&mut self.invalidations, InvalidationTarget::AllResults, reason);
        if let Some(input) = input {
            for rec in self.queries.queries.values_mut() {
                rec.pending_inputs
                    .retain(|p| !(p.stage == input.stage && p.change == input.change));
                rec.pending_inputs.push(input);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{AverageEffectQuery, CacheBudget, CausalQuery, VariableId};

    use crate::event::StateEvent;
    use crate::store::DataBatchRef;
    use crate::suff_stats::LinearOlsSuffStats;

    /// `ReplaceData` drops every diagnostics slot, so the bytes they were charged must
    /// come back; clearing the map alone left the budget permanently smaller.
    #[test]
    fn replace_data_returns_mechanism_diagnostic_budget() {
        use crate::mechanism_diag::{RollingMechanismDiagnostics, insert_mechanism_diag};

        let mut state = CausalState::new(CacheBudget::new(1 << 20));
        for name in ["a", "b"] {
            insert_mechanism_diag(
                &mut state.suff_stats.mechanism_diags,
                Arc::from(name),
                RollingMechanismDiagnostics::new(2, 4).unwrap(),
                &mut state.cache_budget,
            )
            .unwrap();
        }
        assert!(state.cache_budget.used_bytes > 0);
        state.apply(StateEvent::ReplaceData(DataVersion::default().next())).unwrap();
        assert!(state.suff_stats.mechanism_diags.is_empty());
        assert_eq!(state.cache_budget.used_bytes, 0);
    }

    #[test]
    fn append_marks_results_stale_without_autorun() {
        let mut state = CausalState::new(CacheBudget::new(1024));
        let q = state.queries.register(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        )));
        state.refresh_results(&[(q, 42, 16)]).expect("cache");
        assert!(!state.is_stale(q));
        state
            .apply(StateEvent::AppendData(DataBatchRef {
                id: Arc::from("b1"),
                nrows: 10,
                bytes: 80,
            }))
            .expect("apply");
        assert!(state.is_stale(q));
        assert!(state.cached_results.results.is_empty());
    }

    #[test]
    fn refresh_results_at_rejects_a_stale_expected_version() {
        let mut state = CausalState::new(CacheBudget::new(1024));
        let q = state.queries.register(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        )));
        let expected = state.version;
        state
            .apply(StateEvent::AppendData(DataBatchRef {
                id: Arc::from("b_stale"),
                nrows: 4,
                bytes: 32,
            }))
            .expect("append");
        let err = state.refresh_results_at(expected, &[(q, 1, 16)]).expect_err("stale");
        assert!(matches!(err, StateError::InvalidEvent(_)));
        assert!(state.is_stale(q));
        assert!(state.cached_results.results.is_empty());
        state.refresh_results_at(state.version, &[(q, 1, 16)]).expect("current version");
        assert!(!state.is_stale(q));
    }

    #[test]
    fn cache_budget_refuses_oversized_insert() {
        let mut state = CausalState::new(CacheBudget::new(8));
        let q = state.queries.register(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        )));
        let err = state.refresh_results(&[(q, 1, 64)]).expect_err("budget");
        assert!(matches!(err, StateError::CacheBudget { .. }));
    }

    #[test]
    fn rejected_constraint_removal_does_not_advance_version() {
        use crate::store::ConstraintId;

        let mut state = CausalState::new(CacheBudget::new(1024));
        let before = state.version;
        let err = state
            .apply(StateEvent::RemoveConstraint(ConstraintId(Arc::from("missing"))))
            .expect_err("unknown constraint");
        assert!(matches!(err, StateError::UnknownId(_)));
        assert_eq!(state.version, before);
        assert!(state.invalidations.entries.is_empty());
    }

    #[test]
    fn refresh_batch_is_atomic_on_unknown_query() {
        let mut state = CausalState::new(CacheBudget::new(1024));
        let known = state.queries.register(CausalQuery::AverageEffect(
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)),
        ));
        let unknown = QueryId::from_raw(99);
        let err =
            state.refresh_results(&[(known, 1, 16), (unknown, 2, 16)]).expect_err("unknown query");
        assert!(matches!(err, StateError::UnknownId(_)));
        assert!(state.cached_results.results.is_empty());
        assert_eq!(state.cache_budget.used_bytes, 0);
        assert!(state.is_stale(known));
    }

    #[test]
    fn refresh_batch_is_atomic_on_late_budget_failure() {
        let mut state = CausalState::new(CacheBudget::new(24));
        let query = |treatment| {
            CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
                VariableId::from_raw(treatment),
                VariableId::from_raw(2),
            ))
        };
        let first = state.queries.register(query(0));
        let second = state.queries.register(query(1));
        let err = state
            .refresh_results(&[(first, 1, 16), (second, 2, 16)])
            .expect_err("aggregate budget");
        assert!(matches!(err, StateError::CacheBudget { .. }));
        assert!(state.cached_results.results.is_empty());
        assert_eq!(state.cache_budget.used_bytes, 0);
        assert!(state.is_stale(first));
        assert!(state.is_stale(second));
    }

    fn identities(identification: u8, snapshot: u8) -> ContractIdentities {
        let d = |byte: u8| antecedent_core::SemanticDigest::from_bytes([byte; 32]);
        ContractIdentities::new(d(1), d(identification), None, None, d(5), d(6), d(snapshot))
    }

    fn publication(query: QueryId, identities: &ContractIdentities) -> ResultPublication {
        ResultPublication {
            query,
            fingerprint: 1,
            bytes: 8,
            lineage: Some(identities.identification),
            data_snapshot: Some(identities.data_snapshot),
            identities: Some(*identities),
        }
    }

    fn one_query(state: &mut CausalState) -> QueryId {
        state.queries.register(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        )))
    }

    fn constraint(id: &str) -> StateEvent {
        StateEvent::AddConstraint(crate::store::GraphConstraintRecord {
            id: Arc::from(id),
            fingerprint: 7,
        })
    }

    /// A premise event leaves the data snapshot alone, so the old data-snapshot check cannot
    /// see it: the identification layer has to move for the republication to count.
    #[test]
    fn a_constraint_event_is_not_answered_by_an_identical_identification_layer() {
        let mut state = CausalState::new(CacheBudget::new(1024));
        let q = one_query(&mut state);
        let first = identities(10, 20);
        state.publish_results(&[publication(q, &first)]).expect("first publication");
        assert_eq!(state.revalidation(q), Some(Revalidation::StillValid));

        state.apply(constraint("c1")).expect("constraint");
        assert_eq!(state.revalidation(q), Some(Revalidation::Recompute));
        let err = state.publish_results(&[publication(q, &first)]).expect_err("same layer");
        assert!(matches!(&err, StateError::InvalidEvent(m) if m.contains("graph layer")));
        assert_eq!(state.revalidation(q), Some(Revalidation::Recompute));

        // the same data but a re-derived identification is a recomputation
        let rederived = identities(11, 20);
        state.publish_results(&[publication(q, &rederived)]).expect("recomputed");
        assert_eq!(state.revalidation(q), Some(Revalidation::StillValid));
        assert!(!state.is_stale(q));
    }

    /// Removing a premise withdraws what rested on it; adding one only recomputes.
    #[test]
    fn a_removed_constraint_withdraws_and_an_added_one_recomputes() {
        let mut state = CausalState::new(CacheBudget::new(1024));
        let q = one_query(&mut state);
        let first = identities(10, 20);
        state.apply(constraint("c1")).expect("constraint");
        state.publish_results(&[publication(q, &first)]).expect("publication");
        assert_eq!(state.revalidation(q), Some(Revalidation::StillValid));
        state
            .apply(StateEvent::RemoveConstraint(crate::store::ConstraintId(Arc::from("c1"))))
            .expect("remove");
        assert_eq!(state.revalidation(q), Some(Revalidation::Withdrawn));
        assert!(state.is_stale(q));
        assert_eq!(state.revalidation(QueryId::from_raw(99)), None);
    }

    /// The staleness is tied to the stage input identity: equal inputs, equal identities.
    #[test]
    fn input_identities_move_with_the_events_that_change_them() {
        let mut state = CausalState::new(CacheBudget::new(1024));
        let before = state.input_identities();
        assert_eq!(before, CausalState::new(CacheBudget::new(1024)).input_identities());
        state.apply(constraint("c1")).expect("constraint");
        let after = state.input_identities();
        assert_ne!(before.own(Stage::Graph), after.own(Stage::Graph));
        assert_eq!(before.own(Stage::DataSnapshot), after.own(Stage::DataSnapshot));
        let entry = state.invalidations.entries.last().expect("entry");
        let input = entry.input.expect("graph input");
        assert_eq!((input.stage, input.change), (Stage::Graph, ChangeKind::Added));
        assert_eq!(entry.revalidation(), Revalidation::Recompute);
    }

    /// A bare fingerprint cannot mark a lineage-bound result fresh.
    #[test]
    fn a_bare_refresh_cannot_republish_a_lineage_bound_result() {
        let mut state = CausalState::new(CacheBudget::new(1024));
        let q = one_query(&mut state);
        state.publish_results(&[publication(q, &identities(10, 20))]).expect("publication");
        state.apply(constraint("c1")).expect("constraint");
        let err = state.refresh_results(&[(q, 5, 8)]).expect_err("lineage-bound");
        assert!(matches!(err, StateError::InvalidEvent(_)));
        assert!(state.is_stale(q));
        // a query that never carried a lineage keeps the caller-driven path
        let plain = one_query(&mut state);
        state.refresh_results(&[(plain, 5, 8)]).expect("unbound query");
        assert!(!state.is_stale(plain));
    }

    #[test]
    fn incremental_ols_slot_matches_full_recompute() {
        let mut state = CausalState::new(CacheBudget::unlimited());
        let key: Arc<str> = Arc::from("ate_ols");
        state
            .suff_stats
            .ols
            .insert(Arc::clone(&key), LinearOlsSuffStats::new(2).expect("dimension"));
        let rows_a = [1.0, 0.0, 1.0, 1.0];
        let y_a = [1.0, 3.0];
        let rows_b = [1.0, 2.0, 1.0, 3.0];
        let y_b = [5.0, 7.0];
        state.suff_stats.ols.get_mut(&key).unwrap().append_batch(&rows_a, &y_a).unwrap();
        state.suff_stats.ols.get_mut(&key).unwrap().append_batch(&rows_b, &y_b).unwrap();
        let mut full = LinearOlsSuffStats::new(2).expect("dimension");
        let mut all_rows = rows_a.to_vec();
        all_rows.extend_from_slice(&rows_b);
        let mut all_y = y_a.to_vec();
        all_y.extend_from_slice(&y_b);
        full.append_batch(&all_rows, &all_y).unwrap();
        let b_inc = state.suff_stats.ols[&key].solve_beta().unwrap();
        let b_full = full.solve_beta().unwrap();
        assert!((b_inc[0] - b_full[0]).abs() < 1e-10);
        assert!((b_inc[1] - b_full[1]).abs() < 1e-10);
    }

    #[test]
    fn response_query_invalidates_on_append_without_implicit_refresh() {
        use antecedent_core::{ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery};

        let mut state = CausalState::new(CacheBudget::new(1024));
        let q = state.queries.register(CausalQuery::Response(ResponseQuery::new(
            ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Values(Arc::from(vec![0.0_f64, 1.0])),
                ),
            },
        )));
        state.refresh_results(&[(q, 7, 16)]).expect("cache");
        assert!(!state.is_stale(q));
        state
            .apply(StateEvent::AppendData(DataBatchRef {
                id: Arc::from("b_resp"),
                nrows: 4,
                bytes: 32,
            }))
            .expect("append");
        assert!(state.is_stale(q));
        assert!(state.cached_results.results.is_empty());
    }

    #[test]
    fn temporal_response_query_invalidates_on_append_without_implicit_refresh() {
        use antecedent_core::{
            ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TemporalPolicy,
            TemporalResponseSpec,
        };

        let mut state = CausalState::new(CacheBudget::new(1024));
        let temporal =
            TemporalResponseSpec::new(vec![1, 2], TemporalPolicy::pulse(-1), None).unwrap();
        let q = state.queries.register(CausalQuery::Response(
            ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Values(Arc::from(vec![0.0_f64, 1.0])),
                ),
            })
            .with_temporal(temporal),
        ));
        state.refresh_results(&[(q, 9, 32)]).expect("cache");
        assert!(!state.is_stale(q));
        state
            .apply(StateEvent::AppendData(DataBatchRef {
                id: Arc::from("b_temporal_resp"),
                nrows: 8,
                bytes: 64,
            }))
            .expect("append");
        assert!(state.is_stale(q));
        assert!(state.cached_results.results.is_empty());
    }

    #[test]
    fn data_events_follow_the_route_contract_for_the_kept_summaries() {
        use crate::suff_stats::StreamingCovariance;
        let batch = |id: &str| DataBatchRef { id: Arc::from(id), nrows: 4, bytes: 32 };
        for (event_is_append, change) in
            [(true, RefreshChange::AppendRows), (false, RefreshChange::SourceChanged)]
        {
            let mut state = CausalState::new(CacheBudget::unlimited());
            let ols: Arc<str> = Arc::from("ols");
            let cov: Arc<str> = Arc::from("cov");
            state
                .suff_stats
                .ols
                .insert(Arc::clone(&ols), LinearOlsSuffStats::new(2).expect("dimension"));
            state
                .suff_stats
                .cov
                .insert(Arc::clone(&cov), StreamingCovariance::new(2).expect("dimension"));
            let event = if event_is_append {
                StateEvent::AppendData(batch("b"))
            } else {
                StateEvent::ReplaceData(DataVersion::default().next())
            };
            state.apply(event).expect("apply");
            for (route, kept) in [
                (RefreshRoute::LinearOlsGram, state.suff_stats.ols.contains_key(&ols)),
                (RefreshRoute::StreamingCovariance, state.suff_stats.cov.contains_key(&cov)),
            ] {
                let expect_kept = route_contract(route, change) == RefreshDisposition::ExactUpdate;
                assert_eq!(kept, expect_kept, "{route:?} after {change:?}");
            }
        }
    }
}
