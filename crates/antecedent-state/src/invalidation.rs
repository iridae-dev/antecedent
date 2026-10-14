//! Invalidation log and dependency tracking.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::recalc::{ChangeKind, Revalidation, Stage, StageIdentity};
use antecedent_core::{ModelId, QueryId, StateVersion};

/// The declared input of the recalculation stage model an event changed.
///
/// Staleness is not a bare boolean: it names the stage whose own input moved, how it moved
/// and the digest of what it moved to, so a stale result classifies with the same
/// three-state vocabulary as a stage plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChangedInput {
    /// The stage whose own input changed.
    pub stage: Stage,
    /// How it changed.
    pub change: ChangeKind,
    /// Digest of the input after the change (event content and position in the history).
    pub identity: StageIdentity,
}

impl ChangedInput {
    /// The class an input change gives a result resting on it: an input that was removed
    /// withdraws the result, any other change asks for recomputation.
    #[must_use]
    pub const fn revalidation(&self) -> Revalidation {
        self.change.revalidation()
    }
}

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
    /// Rolling mechanism-diagnostics slots.
    MechanismDiagnostics,
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
    /// The stage input the causing event changed, when the event changes one.
    pub input: Option<ChangedInput>,
}

impl InvalidationEntry {
    /// The three-state class this entry gives what it invalidates. An entry that names no
    /// changed input can only ask for recomputation.
    #[must_use]
    pub fn revalidation(&self) -> Revalidation {
        self.input.as_ref().map_or(Revalidation::Recompute, ChangedInput::revalidation)
    }
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
        self.entries.push(InvalidationEntry {
            at_version,
            target,
            reason: reason.into(),
            input: None,
        });
    }

    /// Record an invalidation caused by a change of one stage input.
    pub fn push_input(
        &mut self,
        at_version: StateVersion,
        target: InvalidationTarget,
        reason: impl Into<Arc<str>>,
        input: ChangedInput,
    ) {
        self.entries.push(InvalidationEntry {
            at_version,
            target,
            reason: reason.into(),
            input: Some(input),
        });
    }

    /// The most severe class of the entries after `since` that concern `query` (an entry for
    /// the query itself or for all results). `still_valid` when none does.
    #[must_use]
    pub fn revalidation_since(&self, query: QueryId, since: StateVersion) -> Revalidation {
        self.entries
            .iter()
            .filter(|e| e.at_version.raw() > since.raw())
            .filter(|e| match &e.target {
                InvalidationTarget::QueryResult(q) => *q == query,
                InvalidationTarget::AllResults => true,
                _ => false,
            })
            .fold(Revalidation::StillValid, |class, e| class.worst(e.revalidation()))
    }

    /// Whether `query` has an unresolved invalidation after `since`: its class is not
    /// `still_valid` (see [`Self::revalidation_since`]).
    #[must_use]
    pub fn query_stale_since(&self, query: QueryId, since: StateVersion) -> bool {
        self.revalidation_since(query, since) != Revalidation::StillValid
    }
}
