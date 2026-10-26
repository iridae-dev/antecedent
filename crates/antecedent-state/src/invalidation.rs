//! Invalidation log and dependency tracking.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
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
    /// One prepared product of a temporal window (see [`TemporalProduct`]).
    TemporalProduct(TemporalProduct),
}

/// A factor of a temporal window whose change moves the products that read it.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TemporalFactor {
    /// The observation period moved.
    Period,
    /// The observed population drifted.
    Population,
    /// The tick cadence changed.
    Cadence,
    /// The horizon changed, including an appended slice.
    Horizon,
    /// The initial-state law changed.
    InitialLaw,
    /// The lag alignment of the coordinates or the intervention history changed.
    History,
    /// The selection targets changed.
    Selection,
    /// The intervention regimes changed.
    Regime,
    /// The graph or the proof premises changed.
    Premise,
}

/// A prepared product of a temporal window, in dependency order.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TemporalProduct {
    /// Identification of the window's query, with its proof premises and initial law.
    Identification,
    /// Sampling design: period, population, cadence, horizon, selection and regimes.
    Design,
    /// Mechanism fit and initial law. Period-bound unless invariance was declared.
    Model,
    /// Observations bound to a period, population, cadence and regime.
    Evidence,
    /// The prior of the next period's Bayesian fit, built from the earlier periods'
    /// posterior. It reads exactly what the model reads: a posterior carried forward as a
    /// prior is a stationarity statement, so wherever the model does not survive the prior
    /// restarts at the originally stated one.
    PriorRefresh,
}

impl TemporalProduct {
    /// Every product, in dependency order.
    pub const ALL: [Self; 5] =
        [Self::Identification, Self::Design, Self::Model, Self::Evidence, Self::PriorRefresh];

    /// Whether this product reads `factor`. `period_invariant` is the caller's declaration that
    /// the mechanism is the same law across the periods compared; a temporal index never
    /// supplies it.
    #[must_use]
    pub const fn reads(self, factor: TemporalFactor, period_invariant: bool) -> bool {
        use TemporalFactor as F;
        match self {
            Self::Identification => matches!(
                factor,
                F::Horizon | F::InitialLaw | F::History | F::Selection | F::Regime | F::Premise
            ),
            Self::Design => matches!(
                factor,
                F::Period | F::Population | F::Cadence | F::Horizon | F::Selection | F::Regime
            ),
            Self::Model | Self::PriorRefresh => match factor {
                F::Population
                | F::Cadence
                | F::Horizon
                | F::InitialLaw
                | F::History
                | F::Regime => true,
                F::Period => !period_invariant,
                F::Selection | F::Premise => false,
            },
            Self::Evidence => {
                matches!(factor, F::Period | F::Population | F::Cadence | F::Regime)
            }
        }
    }

    /// Stable label, for example `prior_refresh`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Identification => "identification",
            Self::Design => "design",
            Self::Model => "model",
            Self::Evidence => "evidence",
            Self::PriorRefresh => "prior_refresh",
        }
    }
}

/// Which products a set of factor changes invalidates and which it leaves in place.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalDisposition {
    /// Products that read at least one changed factor.
    pub invalidated: BTreeSet<TemporalProduct>,
    /// Products that read none of them and are reused.
    pub retained: BTreeSet<TemporalProduct>,
}

impl TemporalDisposition {
    /// The exact affected set of `changes`.
    #[must_use]
    pub fn of(changes: &BTreeSet<TemporalFactor>, period_invariant: bool) -> Self {
        let mut invalidated = BTreeSet::new();
        let mut retained = BTreeSet::new();
        for product in TemporalProduct::ALL {
            if changes.iter().any(|factor| product.reads(*factor, period_invariant)) {
                invalidated.insert(product);
            } else {
                retained.insert(product);
            }
        }
        Self { invalidated, retained }
    }
}

/// The state event that made a target stale.
///
/// Logged as [`InvalidationEntry::reason`] through its stable [`Self::as_str`] label, so a
/// cause cannot drift or be misspelled between the places that record it.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum InvalidationReason {
    /// A batch of data was appended.
    AppendData,
    /// The data catalog was replaced.
    ReplaceData,
    /// Graph evidence was added.
    AddGraphEvidence,
    /// A constraint was added.
    AddConstraint,
    /// A constraint was removed.
    RemoveConstraint,
    /// An assumption was updated.
    UpdateAssumption,
    /// An intervention was recorded.
    RecordIntervention,
}

impl InvalidationReason {
    /// Stable label, for example `replace_data`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AppendData => "append_data",
            Self::ReplaceData => "replace_data",
            Self::AddGraphEvidence => "add_graph_evidence",
            Self::AddConstraint => "add_constraint",
            Self::RemoveConstraint => "remove_constraint",
            Self::UpdateAssumption => "update_assumption",
            Self::RecordIntervention => "record_intervention",
        }
    }
}

impl core::fmt::Display for InvalidationReason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<InvalidationReason> for Arc<str> {
    fn from(reason: InvalidationReason) -> Self {
        Self::from(reason.as_str())
    }
}

/// One invalidation record.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InvalidationEntry {
    /// State version after the causing event.
    pub at_version: StateVersion,
    /// Target that is now stale.
    pub target: InvalidationTarget,
    /// Human-readable reason; an [`InvalidationReason`] label for state events.
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

    /// Record a change of temporal window factors for every family at once.
    ///
    /// Every product that reads a changed factor gets one entry, and the cached results that
    /// rest on those products are recomputed (a result entry for all queries) whenever any
    /// product moved. Products that read none of the changed factors get no entry: they are
    /// reused. Returns the exact disposition.
    pub fn push_temporal_change(
        &mut self,
        at_version: StateVersion,
        changes: &BTreeSet<TemporalFactor>,
        period_invariant: bool,
        reason: &str,
    ) -> TemporalDisposition {
        let disposition = TemporalDisposition::of(changes, period_invariant);
        for product in &disposition.invalidated {
            self.push(
                at_version,
                InvalidationTarget::TemporalProduct(*product),
                format!("{reason}: {} reads a changed factor", product.label()),
            );
        }
        if !disposition.invalidated.is_empty() {
            self.push(at_version, InvalidationTarget::AllResults, reason);
        }
        disposition
    }

    /// Whether `product` may still be used after `since`: no change after it moved a factor the
    /// product reads.
    #[must_use]
    pub fn temporal_product_reusable_since(
        &self,
        product: TemporalProduct,
        since: StateVersion,
    ) -> bool {
        !self.entries.iter().any(|e| {
            e.at_version.raw() > since.raw()
                && e.target == InvalidationTarget::TemporalProduct(product)
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn set(factors: &[TemporalFactor]) -> BTreeSet<TemporalFactor> {
        factors.iter().copied().collect()
    }

    fn products(items: &[TemporalProduct]) -> BTreeSet<TemporalProduct> {
        items.iter().copied().collect()
    }

    #[test]
    fn each_factor_invalidates_exactly_the_products_that_read_it() {
        use TemporalFactor as F;
        use TemporalProduct as P;
        let cases: Vec<(F, bool, Vec<P>)> = vec![
            (F::Period, false, vec![P::Design, P::Model, P::Evidence, P::PriorRefresh]),
            (F::Period, true, vec![P::Design, P::Evidence]),
            (F::Population, true, vec![P::Design, P::Model, P::Evidence, P::PriorRefresh]),
            (F::Cadence, true, vec![P::Design, P::Model, P::Evidence, P::PriorRefresh]),
            (F::Horizon, true, vec![P::Identification, P::Design, P::Model, P::PriorRefresh]),
            (F::InitialLaw, true, vec![P::Identification, P::Model, P::PriorRefresh]),
            (F::History, true, vec![P::Identification, P::Model, P::PriorRefresh]),
            (F::Selection, true, vec![P::Identification, P::Design]),
            (F::Regime, true, P::ALL.to_vec()),
            (F::Premise, true, vec![P::Identification]),
            (F::InitialLaw, false, vec![P::Identification, P::Model, P::PriorRefresh]),
        ];
        for (factor, invariant, expected) in cases {
            let disposition = TemporalDisposition::of(&set(&[factor]), invariant);
            assert_eq!(
                disposition.invalidated,
                products(&expected),
                "{factor:?} invariant={invariant}"
            );
            let all: BTreeSet<_> = P::ALL.into_iter().collect();
            let union: BTreeSet<_> =
                disposition.invalidated.union(&disposition.retained).copied().collect();
            assert_eq!(union, all);
            assert!(disposition.invalidated.is_disjoint(&disposition.retained));
        }
        assert!(TemporalDisposition::of(&BTreeSet::new(), false).invalidated.is_empty());
    }

    #[test]
    fn the_log_records_one_entry_per_affected_product_and_reuses_the_rest() {
        let mut log = InvalidationLog::new();
        let v1 = StateVersion::from_raw(1);
        let v2 = StateVersion::from_raw(2);
        let disposition =
            log.push_temporal_change(v2, &set(&[TemporalFactor::Cadence]), true, "cadence changed");
        assert_eq!(
            disposition.invalidated,
            products(&[
                TemporalProduct::Design,
                TemporalProduct::Model,
                TemporalProduct::Evidence,
                TemporalProduct::PriorRefresh
            ])
        );
        let targets: Vec<_> = log
            .entries
            .iter()
            .filter_map(|e| match e.target {
                InvalidationTarget::TemporalProduct(p) => Some(p),
                _ => None,
            })
            .collect();
        assert_eq!(targets.len(), 4);
        assert!(!targets.contains(&TemporalProduct::Identification));
        assert!(log.temporal_product_reusable_since(TemporalProduct::Identification, v1));
        assert!(!log.temporal_product_reusable_since(TemporalProduct::Model, v1));
        assert!(log.temporal_product_reusable_since(TemporalProduct::Model, v2));
        assert!(log.query_stale_since(QueryId::from_raw(3), v1));
        // A change nothing reads leaves the log and every cached result alone.
        let before = log.entries.len();
        let none = log.push_temporal_change(
            StateVersion::from_raw(3),
            &BTreeSet::new(),
            false,
            "no change",
        );
        assert!(none.invalidated.is_empty());
        assert_eq!(log.entries.len(), before);
    }
}
