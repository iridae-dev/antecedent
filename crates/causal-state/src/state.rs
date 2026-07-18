//! [`CausalState`] apply / invalidate / recompute (DESIGN.md §20).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_pass_by_value)]

use std::sync::Arc;

use causal_core::{AssumptionSet, CacheBudget, QueryId, StateVersion};

use crate::error::StateError;
use crate::event::StateEvent;
use crate::invalidation::{InvalidationLog, InvalidationTarget};
use crate::store::{
    CachedResult, DataCatalog, DataVersion, GraphEvidenceStore, ModelStore, QueryStore,
    ResultStore, SuffStatStore,
};

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
        self.version = self.version.next();
        let v = self.version;
        match event {
            StateEvent::AppendData(batch) => {
                self.data_catalog.batches.push(batch);
                self.data_catalog.version = self.data_catalog.version.next();
                self.invalidate_data_dependents(v, "append_data");
            }
            StateEvent::ReplaceData(new_ver) => {
                self.data_catalog.version = new_ver;
                self.data_catalog.batches.clear();
                self.invalidate_data_dependents(v, "replace_data");
                self.suff_stats.ols.clear();
                self.suff_stats.cov.clear();
                self.suff_stats.lag_indexes.clear();
                self.suff_stats.graph_scores.clear();
                self.suff_stats.particle_filters.clear();
                self.suff_stats.mechanism_diags.clear();
                self.invalidations.push(v, InvalidationTarget::LagIndexes, "replace_data");
                self.invalidations.push(v, InvalidationTarget::GraphScores, "replace_data");
                self.invalidations.push(v, InvalidationTarget::ParticleFilters, "replace_data");
                self.invalidations.push(
                    v,
                    InvalidationTarget::MechanismDiagnostics,
                    "replace_data",
                );
            }
            StateEvent::AddGraphEvidence(rec) => {
                self.graph_evidence.evidence.push(rec);
                self.invalidate_all_results(v, "add_graph_evidence");
            }
            StateEvent::AddConstraint(rec) => {
                self.graph_evidence.constraints.push(rec);
                self.invalidate_all_results(v, "add_constraint");
            }
            StateEvent::RemoveConstraint(id) => {
                let before = self.graph_evidence.constraints.len();
                self.graph_evidence.constraints.retain(|c| c.id != id.0);
                if self.graph_evidence.constraints.len() == before {
                    return Err(StateError::UnknownId(format!("constraint {}", id.0)));
                }
                self.invalidate_all_results(v, "remove_constraint");
            }
            StateEvent::UpdateAssumption(rec) => {
                self.assumptions.push(rec);
                self.invalidate_all_results(v, "update_assumption");
            }
            StateEvent::RegisterQuery(query) => {
                let _id = self.queries.register(query);
                // Registration alone does not invalidate others.
            }
            StateEvent::RecordIntervention(rec) => {
                self.interventions.push(rec);
                self.invalidate_all_results(v, "record_intervention");
            }
        }
        Ok(v)
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

    /// Caller-driven recomputation hook: marks selected queries fresh after the
    /// caller supplies new result fingerprints. Does not run estimators itself.
    ///
    /// # Errors
    ///
    /// Unknown query or cache budget refusal.
    pub fn refresh_results(&mut self, updates: &[(QueryId, u64, u64)]) -> Result<(), StateError> {
        for &(query, fingerprint, bytes) in updates {
            if !self.queries.queries.contains_key(&query) {
                return Err(StateError::UnknownId(format!("query {}", query.raw())));
            }
            self.cached_results.insert(
                CachedResult { query, fingerprint, bytes, computed_at: self.version },
                &mut self.cache_budget,
            )?;
            if let Some(rec) = self.queries.queries.get_mut(&query) {
                rec.result_valid_at = Some(self.version);
            }
        }
        Ok(())
    }

    /// Current data version.
    #[must_use]
    pub fn data_version(&self) -> DataVersion {
        self.data_catalog.version
    }

    fn invalidate_data_dependents(&mut self, v: StateVersion, reason: &str) {
        self.invalidate_all_results(v, reason);
        for key in self.suff_stats.ols.keys().cloned().collect::<Vec<_>>() {
            self.invalidations.push(
                v,
                InvalidationTarget::SuffStat(key),
                format!("{reason}:ols_may_need_append"),
            );
        }
        self.invalidations.push(v, InvalidationTarget::LagIndexes, reason);
        // Lag indexes become stale on data change; drop metadata.
        self.suff_stats.lag_indexes.clear();
        // Graph scores are data-dependent; clear until a streaming family exists.
        self.suff_stats.graph_scores.clear();
        self.invalidations.push(v, InvalidationTarget::GraphScores, reason);
        // Particle filters: drop on append (caller re-inits / steps a fresh filter).
        self.suff_stats.particle_filters.clear();
        self.invalidations.push(v, InvalidationTarget::ParticleFilters, reason);
        // Mechanism diagnostics stream via caller append; log stale without clearing.
        self.invalidations.push(v, InvalidationTarget::MechanismDiagnostics, reason);
    }

    fn invalidate_all_results(&mut self, v: StateVersion, reason: &str) {
        let ids: Vec<QueryId> = self.cached_results.results.keys().copied().collect();
        for q in ids {
            self.cached_results.remove(q, &mut self.cache_budget);
            self.invalidations.push(v, InvalidationTarget::QueryResult(q), Arc::from(reason));
            if let Some(rec) = self.queries.queries.get_mut(&q) {
                rec.result_valid_at = None;
            }
        }
        self.invalidations.push(v, InvalidationTarget::AllResults, reason);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{AverageEffectQuery, CacheBudget, CausalQuery, VariableId};

    use crate::event::StateEvent;
    use crate::store::DataBatchRef;
    use crate::suff_stats::LinearOlsSuffStats;

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
    fn incremental_ols_slot_matches_full_recompute() {
        let mut state = CausalState::new(CacheBudget::unlimited());
        let key: Arc<str> = Arc::from("ate_ols");
        state.suff_stats.ols.insert(Arc::clone(&key), LinearOlsSuffStats::new(2));
        let rows_a = [1.0, 0.0, 1.0, 1.0];
        let y_a = [1.0, 3.0];
        let rows_b = [1.0, 2.0, 1.0, 3.0];
        let y_b = [5.0, 7.0];
        state.suff_stats.ols.get_mut(&key).unwrap().append_batch(&rows_a, &y_a).unwrap();
        state.suff_stats.ols.get_mut(&key).unwrap().append_batch(&rows_b, &y_b).unwrap();
        let mut full = LinearOlsSuffStats::new(2);
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
}
