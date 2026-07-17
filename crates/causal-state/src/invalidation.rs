//! Invalidation log and dependency tracking.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{ModelId, QueryId, StateVersion};

/// What became stale after an event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InvalidationTarget {
    /// Cached result for a query.
    QueryResult(QueryId),
    /// Model artifact.
    Model(ModelId),
    /// All cached results.
    AllResults,
    /// Sufficient-statistic slot (opaque key).
    SuffStat(Arc<str>),
    /// Lag-index cache entries.
    LagIndexes,
    /// Graph-score cache slots.
    GraphScores,
    /// Particle-filter state slots.
    ParticleFilters,
}

/// One invalidation record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidationEntry {
    /// State version after the causing event.
    pub at_version: StateVersion,
    /// Target that is now stale.
    pub target: InvalidationTarget,
    /// Human-readable reason.
    pub reason: Arc<str>,
}

/// Append-only invalidation log.
#[derive(Clone, Debug, Default)]
pub struct InvalidationLog {
    /// Entries in application order.
    pub entries: Vec<InvalidationEntry>,
}

impl InvalidationLog {
    /// Empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an invalidation.
    pub fn push(
        &mut self,
        at_version: StateVersion,
        target: InvalidationTarget,
        reason: impl Into<Arc<str>>,
    ) {
        self.entries.push(InvalidationEntry { at_version, target, reason: reason.into() });
    }

    /// Whether `query` has an unresolved invalidation after `since`.
    #[must_use]
    pub fn query_stale_since(&self, query: QueryId, since: StateVersion) -> bool {
        self.entries.iter().any(|e| {
            e.at_version.raw() > since.raw()
                && matches!(
                    e.target,
                    InvalidationTarget::QueryResult(q) if q == query
                )
                || (e.at_version.raw() > since.raw()
                    && matches!(e.target, InvalidationTarget::AllResults))
        })
    }
}
