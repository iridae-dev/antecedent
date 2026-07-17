//! Versioned stores inside [`crate::CausalState`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{CacheBudget, CausalQuery, ModelId, QueryId, StateVersion};

use crate::error::StateError;
use crate::graph_score::LocalScoreCache;
use crate::particle_filter::ParticleFilterState;
use crate::retention::RetentionPolicy;
use crate::suff_stats::{
    LagIndexCacheEntry, LagIndexCacheKey, LinearOlsSuffStats, StreamingCovariance,
};

/// Opaque data batch reference (no borrowed buffers / process handles).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct DataBatchRef {
    /// Stable batch id.
    pub id: Arc<str>,
    /// Row count in the batch.
    pub nrows: u64,
    /// Byte size estimate.
    pub bytes: u64,
}

/// Data catalog version stamp.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub struct DataVersion(u64);

impl DataVersion {
    /// Raw counter.
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Next version.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

/// Catalog of registered data batches (does not retain full history by default).
#[derive(Clone, Debug)]
pub struct DataCatalog {
    /// Current data version.
    pub version: DataVersion,
    /// Active batches (may be empty under sufficient-stat-only retention).
    pub batches: Vec<DataBatchRef>,
    /// Retention for the catalog itself.
    pub retention: RetentionPolicy,
}

impl Default for DataCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl DataCatalog {
    /// Fresh catalog that prefers sufficient statistics over raw history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            version: DataVersion(0),
            batches: Vec::new(),
            retention: RetentionPolicy::SufficientStatisticsOnly,
        }
    }
}

/// Opaque graph-evidence record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphEvidenceRecord {
    /// Evidence id.
    pub id: Arc<str>,
    /// Opaque payload fingerprint / key.
    pub fingerprint: u64,
    /// Bytes retained.
    pub bytes: u64,
}

/// Graph constraint record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphConstraintRecord {
    /// Constraint id.
    pub id: Arc<str>,
    /// Detail fingerprint.
    pub fingerprint: u64,
}

/// Constraint id for removal.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ConstraintId(pub Arc<str>);

/// Store of graph evidence and constraints.
#[derive(Clone, Debug, Default)]
pub struct GraphEvidenceStore {
    /// Evidence records.
    pub evidence: Vec<GraphEvidenceRecord>,
    /// Active constraints.
    pub constraints: Vec<GraphConstraintRecord>,
}

/// Intervention record (library does not execute external actions).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InterventionRecord {
    /// Intervention id.
    pub id: Arc<str>,
    /// Fingerprint of the intervention payload.
    pub fingerprint: u64,
}

/// Registered model artifact handle.
#[derive(Clone, Debug)]
pub struct ModelRecord {
    /// Model id.
    pub id: ModelId,
    /// Fingerprint / version of fitted artifact.
    pub fingerprint: u64,
    /// Bytes.
    pub bytes: u64,
    /// Retention.
    pub retention: RetentionPolicy,
}

/// Model store.
#[derive(Clone, Debug, Default)]
pub struct ModelStore {
    /// Models by id.
    pub models: HashMap<ModelId, ModelRecord>,
    /// Next id.
    next_id: u32,
}

impl ModelStore {
    /// Register a model; returns assigned id.
    pub fn register(&mut self, fingerprint: u64, bytes: u64) -> ModelId {
        let id = ModelId::from_raw(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.models.insert(
            id,
            ModelRecord {
                id,
                fingerprint,
                bytes,
                retention: RetentionPolicy::SufficientStatisticsOnly,
            },
        );
        id
    }
}

/// Registered query with cache freshness version.
#[derive(Clone, Debug)]
pub struct QueryRecord {
    /// Query id.
    pub id: QueryId,
    /// Query payload.
    pub query: CausalQuery,
    /// Version at which the cached result is valid (`None` = never computed).
    pub result_valid_at: Option<StateVersion>,
}

/// Query store.
#[derive(Clone, Debug, Default)]
pub struct QueryStore {
    /// Queries by id.
    pub queries: HashMap<QueryId, QueryRecord>,
    next_id: u32,
}

impl QueryStore {
    /// Register a query.
    pub fn register(&mut self, query: CausalQuery) -> QueryId {
        let id = QueryId::from_raw(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.queries.insert(id, QueryRecord { id, query, result_valid_at: None });
        id
    }
}

/// Cached analysis result (opaque bytes / fingerprint; reconstructible).
#[derive(Clone, Debug)]
pub struct CachedResult {
    /// Query id.
    pub query: QueryId,
    /// Result fingerprint.
    pub fingerprint: u64,
    /// Bytes retained.
    pub bytes: u64,
    /// State version when computed.
    pub computed_at: StateVersion,
}

/// Bounded result cache.
#[derive(Clone, Debug, Default)]
pub struct ResultStore {
    /// Results by query.
    pub results: HashMap<QueryId, CachedResult>,
}

impl ResultStore {
    /// Insert respecting cache budget; refuses when over budget (no silent semantics change).
    ///
    /// # Errors
    ///
    /// Budget exceeded.
    pub fn insert(
        &mut self,
        result: CachedResult,
        budget: &mut CacheBudget,
    ) -> Result<(), StateError> {
        let old_bytes = self.results.get(&result.query).map_or(0, |r| r.bytes);
        let net = result.bytes.saturating_sub(old_bytes);
        if !budget.can_admit(net) {
            return Err(StateError::CacheBudget { need: net, remaining: budget.remaining() });
        }
        budget.used_bytes =
            budget.used_bytes.saturating_sub(old_bytes).saturating_add(result.bytes);
        self.results.insert(result.query, result);
        Ok(())
    }

    /// Remove and free budget.
    pub fn remove(&mut self, query: QueryId, budget: &mut CacheBudget) {
        if let Some(r) = self.results.remove(&query) {
            budget.used_bytes = budget.used_bytes.saturating_sub(r.bytes);
        }
    }

    /// Clear all, freeing budget.
    pub fn clear(&mut self, budget: &mut CacheBudget) {
        for r in self.results.values() {
            budget.used_bytes = budget.used_bytes.saturating_sub(r.bytes);
        }
        self.results.clear();
    }
}

/// Named sufficient-stat slots.
#[derive(Clone, Debug, Default)]
pub struct SuffStatStore {
    /// Linear OLS slots.
    pub ols: HashMap<Arc<str>, LinearOlsSuffStats>,
    /// Streaming covariance slots.
    pub cov: HashMap<Arc<str>, StreamingCovariance>,
    /// Lag-index cache metadata.
    pub lag_indexes: HashMap<LagIndexCacheKey, LagIndexCacheEntry>,
    /// Graph local-score caches (keyed by opaque slot name).
    pub graph_scores: HashMap<Arc<str>, LocalScoreCache>,
    /// Particle-filter states (keyed by opaque slot name).
    pub particle_filters: HashMap<Arc<str>, ParticleFilterState>,
}
