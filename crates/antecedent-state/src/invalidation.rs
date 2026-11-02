//! Invalidation log and dependency tracking.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::recalc::{ChangeKind, Revalidation, Stage, StageIdentity};
use antecedent_core::{CanonicalWriter, ModelId, QueryId, SemanticDigest, StateVersion};

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

const ENTRY_TAG: &str = "antecedent.state.invalidation.entry.v1";
const CHAIN_TAG: &str = "antecedent.state.invalidation.chain.v1";

impl InvalidationEntry {
    fn digest(&self) -> [u8; 32] {
        let mut writer = CanonicalWriter::new(ENTRY_TAG);
        writer.word(self.at_version.raw());
        match &self.target {
            InvalidationTarget::QueryResult(query) => {
                writer.word(0).word(u64::from(query.raw()));
            }
            InvalidationTarget::Model(model) => {
                writer.word(1).word(u64::from(model.raw()));
            }
            InvalidationTarget::AllResults => {
                writer.word(2);
            }
            InvalidationTarget::SuffStat(key) => {
                writer.word(3).text(key);
            }
            InvalidationTarget::LagIndexes => {
                writer.word(4);
            }
            InvalidationTarget::GraphScores => {
                writer.word(5);
            }
            InvalidationTarget::ParticleFilters => {
                writer.word(6);
            }
            InvalidationTarget::MechanismDiagnostics => {
                writer.word(7);
            }
            InvalidationTarget::TemporalProduct(product) => {
                writer.word(8).text(product.label());
            }
        }
        writer.text(&self.reason);
        match &self.input {
            None => {
                writer.word(0);
            }
            Some(input) => {
                let change = match input.change {
                    ChangeKind::Added => 0,
                    ChangeKind::Removed => 1,
                    ChangeKind::Modified => 2,
                };
                writer.word(1).text(&input.stage.label()).word(change);
                writer.bytes(input.identity.as_bytes());
            }
        }
        writer.finish()
    }

    /// The head of a digest chain after this entry: a digest of `head` and of this entry's
    /// content. Folding the entries a log pruned, oldest first, from
    /// [`InvalidationLog::PRUNED_GENESIS`] gives [`InvalidationLog::pruned_head`], so an auditor
    /// who kept those entries can check them against what the log committed to.
    #[must_use]
    pub fn chained_onto(&self, head: SemanticDigest) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(CHAIN_TAG);
        writer.bytes(head.as_bytes()).bytes(&self.digest());
        SemanticDigest::from_bytes(writer.finish())
    }
}

/// What a log committed to when it trimmed its oldest entries.
#[derive(Clone, Copy, Debug)]
struct Pruned {
    through: StateVersion,
    entries: u64,
    head: SemanticDigest,
}

/// A request for entries that a log no longer holds.
///
/// Entries of versions up to [`Self::pruned_through`] were trimmed, so a reader asking for
/// everything after an earlier version cannot be shown all of it. The log kept their count and
/// a chain head instead, so the gap is stated and checkable, never a silent omission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrunedHistory {
    /// The version after which entries were asked for.
    pub requested_after: StateVersion,
    /// Greatest version whose entries were trimmed.
    pub pruned_through: StateVersion,
    /// How many entries were trimmed in all.
    pub pruned_entries: u64,
    /// Chain head over every trimmed entry ([`InvalidationEntry::chained_onto`]).
    pub head: SemanticDigest,
}

impl core::fmt::Display for PrunedHistory {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "invalidation entries through version {} were trimmed ({} entries, chain head {}) \
             and the request asks for those after version {}: classify from a version at or \
             after {}, or recompute",
            self.pruned_through,
            self.pruned_entries,
            self.head.to_hex(),
            self.requested_after,
            self.pruned_through,
        )
    }
}

impl std::error::Error for PrunedHistory {}

/// Versions at which one result's class was raised, with dominated records dropped.
///
/// Sorted by version ascending with the class strictly descending, so the worst class among
/// the records after `since` is that of the first record after it. A record is dominated when
/// a later-or-equal one is at least as severe: every `since` that sees the first sees the second.
#[derive(Clone, Debug, Default)]
struct Staircase(Vec<(u64, Revalidation)>);

impl Staircase {
    fn record(&mut self, version: u64, class: Revalidation) {
        if self.0.iter().any(|&(v, c)| v >= version && c >= class) {
            return;
        }
        self.0.retain(|&(v, c)| !(v <= version && c <= class));
        let at = self.0.partition_point(|&(v, _)| v < version);
        self.0.insert(at, (version, class));
    }

    fn worst_after(&self, since: u64) -> Revalidation {
        let at = self.0.partition_point(|&(v, _)| v <= since);
        self.0.get(at).map_or(Revalidation::StillValid, |&(_, class)| class)
    }
}

/// What the staleness queries read, independent of how many entries the log still holds.
#[derive(Clone, Debug, Default)]
struct StalenessIndex {
    queries: BTreeMap<QueryId, Staircase>,
    all_results: Staircase,
    products: BTreeMap<TemporalProduct, u64>,
}

impl StalenessIndex {
    fn record(&mut self, entry: &InvalidationEntry) {
        let version = entry.at_version.raw();
        match &entry.target {
            InvalidationTarget::QueryResult(query) => {
                self.queries.entry(*query).or_default().record(version, entry.revalidation());
            }
            InvalidationTarget::AllResults => {
                self.all_results.record(version, entry.revalidation());
            }
            InvalidationTarget::TemporalProduct(product) => {
                let latest = self.products.entry(*product).or_insert(version);
                *latest = (*latest).max(version);
            }
            _ => {}
        }
    }
}

/// Append-only invalidation log.
///
/// Staleness queries read an index kept in step with [`Self::push`], so they cost the number
/// of distinct classes, not the number of entries. `entries` is the audit trail: append through
/// the `push*` methods only, and a state bounds it with [`Self::trim_audit`], which never
/// changes a staleness answer. Trimmed entries stay committed to by a digest chain
/// ([`Self::pruned_head`]), and a reader of the trail goes through [`Self::entries_after`],
/// which refuses a request the trimmed history cannot answer in full.
#[derive(Clone, Debug, Default)]
pub struct InvalidationLog {
    /// Entries in application order, after any [`Self::pruned_through`] version.
    pub entries: Vec<InvalidationEntry>,
    index: StalenessIndex,
    pruned: Option<Pruned>,
}

impl InvalidationLog {
    /// The chain head before any entry was trimmed.
    pub const PRUNED_GENESIS: SemanticDigest = SemanticDigest::from_bytes([0; 32]);

    /// Empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Greatest version whose entries were trimmed from [`Self::entries`], if any. Staleness
    /// queries still account for them; a reader of the trail goes through
    /// [`Self::entries_after`], which refuses to omit them silently.
    #[must_use]
    pub fn pruned_through(&self) -> Option<StateVersion> {
        self.pruned.map(|pruned| pruned.through)
    }

    /// How many entries were trimmed in all.
    #[must_use]
    pub fn pruned_entries(&self) -> u64 {
        self.pruned.map_or(0, |pruned| pruned.entries)
    }

    /// Chain head over every trimmed entry, oldest first, from [`Self::PRUNED_GENESIS`]
    /// ([`InvalidationEntry::chained_onto`]); [`Self::PRUNED_GENESIS`] when none was trimmed.
    #[must_use]
    pub fn pruned_head(&self) -> SemanticDigest {
        self.pruned.map_or(Self::PRUNED_GENESIS, |pruned| pruned.head)
    }

    /// Entries recorded after `since`, in the order they were pushed.
    ///
    /// # Errors
    ///
    /// [`PrunedHistory`] when entries of a version after `since` were trimmed, so the answer
    /// would be missing some of them. A request at or after [`Self::pruned_through`] is
    /// answered in full.
    pub fn entries_after(
        &self,
        since: StateVersion,
    ) -> Result<impl Iterator<Item = &InvalidationEntry> + '_, PrunedHistory> {
        if let Some(pruned) = self.pruned.filter(|pruned| since < pruned.through) {
            return Err(PrunedHistory {
                requested_after: since,
                pruned_through: pruned.through,
                pruned_entries: pruned.entries,
                head: pruned.head,
            });
        }
        Ok(self.entries.iter().filter(move |entry| entry.at_version > since))
    }

    /// Drop the oldest whole versions once `entries` holds more than twice `keep`, leaving
    /// about `keep`, and fold what was dropped into the chain head. A version is never split,
    /// so one with more than `keep` entries stays whole. Trimming in batches keeps the cost
    /// amortized.
    pub fn trim_audit(&mut self, keep: usize) {
        let keep = keep.max(1);
        if self.entries.len() <= keep.saturating_mul(2) {
            return;
        }
        let boundary = self.entries[self.entries.len() - keep - 1].at_version;
        let cut = self.entries.partition_point(|entry| entry.at_version <= boundary);
        let before = self.pruned;
        let head = self.entries[..cut]
            .iter()
            .fold(before.map_or(Self::PRUNED_GENESIS, |pruned| pruned.head), |head, entry| {
                entry.chained_onto(head)
            });
        self.entries.drain(..cut);
        self.entries.shrink_to_fit();
        self.pruned = Some(Pruned {
            through: before.map_or(boundary, |pruned| pruned.through.max(boundary)),
            entries: before.map_or(0, |pruned| pruned.entries).saturating_add(cut as u64),
            head,
        });
    }

    fn append(&mut self, entry: InvalidationEntry) {
        self.index.record(&entry);
        self.entries.push(entry);
    }

    /// Record an invalidation.
    pub fn push(
        &mut self,
        at_version: StateVersion,
        target: InvalidationTarget,
        reason: impl Into<Arc<str>>,
    ) {
        self.append(InvalidationEntry { at_version, target, reason: reason.into(), input: None });
    }

    /// Record an invalidation caused by a change of one stage input.
    pub fn push_input(
        &mut self,
        at_version: StateVersion,
        target: InvalidationTarget,
        reason: impl Into<Arc<str>>,
        input: ChangedInput,
    ) {
        self.append(InvalidationEntry {
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
        self.index.products.get(&product).is_none_or(|latest| *latest <= since.raw())
    }

    /// The most severe class of the entries after `since` that concern `query` (an entry for
    /// the query itself or for all results). `still_valid` when none does.
    #[must_use]
    pub fn revalidation_since(&self, query: QueryId, since: StateVersion) -> Revalidation {
        let since = since.raw();
        let own = self
            .index
            .queries
            .get(&query)
            .map_or(Revalidation::StillValid, |stairs| stairs.worst_after(since));
        own.worst(self.index.all_results.worst_after(since))
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

    fn scan(log: &InvalidationLog, query: QueryId, since: StateVersion) -> Revalidation {
        log.entries
            .iter()
            .filter(|e| e.at_version.raw() > since.raw())
            .filter(|e| match &e.target {
                InvalidationTarget::QueryResult(q) => *q == query,
                InvalidationTarget::AllResults => true,
                _ => false,
            })
            .fold(Revalidation::StillValid, |class, e| class.worst(e.revalidation()))
    }

    fn changed(change: ChangeKind) -> ChangedInput {
        ChangedInput {
            stage: Stage::Graph,
            change,
            identity: StageIdentity::of("test.invalidation.v1", &[b"x".as_slice()]),
        }
    }

    #[test]
    fn the_staleness_index_matches_a_scan_of_every_entry_and_survives_trimming() {
        let mut log = InvalidationLog::new();
        let queries = [QueryId::from_raw(0), QueryId::from_raw(1), QueryId::from_raw(2)];
        for version in 1..=300_u64 {
            let at = StateVersion::from_raw(version);
            let query = queries[(version % 3) as usize];
            match version % 5 {
                0 => log.push_input(
                    at,
                    InvalidationTarget::QueryResult(query),
                    "withdrawn",
                    changed(ChangeKind::Removed),
                ),
                1 => log.push_input(
                    at,
                    InvalidationTarget::AllResults,
                    "all",
                    changed(ChangeKind::Added),
                ),
                2 => log.push(at, InvalidationTarget::QueryResult(query), "no input"),
                _ => log.push(at, InvalidationTarget::LagIndexes, "unrelated"),
            }
        }
        let untrimmed = log.clone();
        for query in queries {
            for since in 0..=301_u64 {
                let since = StateVersion::from_raw(since);
                assert_eq!(
                    log.revalidation_since(query, since),
                    scan(&untrimmed, query, since),
                    "query {query:?} since {since:?}"
                );
            }
        }
        log.trim_audit(40);
        assert_eq!(log.entries.len(), 40, "the oldest whole versions were trimmed");
        let boundary = log.pruned_through().expect("a trim records its boundary");
        assert!(log.entries.iter().all(|entry| entry.at_version > boundary));
        for query in queries {
            for since in 0..=301_u64 {
                let since = StateVersion::from_raw(since);
                assert_eq!(
                    log.revalidation_since(query, since),
                    scan(&untrimmed, query, since),
                    "trimmed: query {query:?} since {since:?}"
                );
            }
        }
    }

    fn to_len(count: u64) -> usize {
        usize::try_from(count).expect("a test count fits in usize")
    }

    fn push_versions(log: &mut InvalidationLog, versions: std::ops::RangeInclusive<u64>) {
        for version in versions {
            log.push_input(
                StateVersion::from_raw(version),
                InvalidationTarget::AllResults,
                "all",
                changed(ChangeKind::Modified),
            );
        }
    }

    #[test]
    fn a_trimmed_log_commits_to_what_it_dropped_and_refuses_to_omit_it_silently() {
        let mut log = InvalidationLog::new();
        push_versions(&mut log, 1..=200);
        let mut mirror = log.clone();
        assert_eq!(log.pruned_head(), InvalidationLog::PRUNED_GENESIS);
        assert_eq!((log.pruned_entries(), log.pruned_through()), (0, None));
        assert_eq!(log.entries_after(StateVersion::ZERO).unwrap().count(), 200);

        log.trim_audit(50);
        let through = log.pruned_through().expect("a trim records its boundary");
        let dropped = log.pruned_entries();
        assert_eq!(to_len(dropped) + log.entries.len(), 200);
        let chain = |entries: &[InvalidationEntry]| {
            entries
                .iter()
                .fold(InvalidationLog::PRUNED_GENESIS, |head, entry| entry.chained_onto(head))
        };
        assert_eq!(log.pruned_head(), chain(&mirror.entries[..to_len(dropped)]));
        assert_ne!(log.pruned_head(), InvalidationLog::PRUNED_GENESIS);

        // The tail is answered in full; anything earlier is refused with the commitment.
        assert_eq!(log.entries_after(through).unwrap().count(), log.entries.len());
        let earlier = StateVersion::from_raw(through.raw() - 1);
        let refused = log.entries_after(earlier).err().expect("trimmed history is refused");
        assert_eq!(refused.requested_after, earlier);
        assert_eq!(refused.pruned_through, through);
        assert_eq!(refused.pruned_entries, dropped);
        assert_eq!(refused.head, log.pruned_head());
        assert!(refused.to_string().contains(&log.pruned_head().to_hex()));

        // A second trim extends the same chain.
        push_versions(&mut log, 201..=400);
        push_versions(&mut mirror, 201..=400);
        log.trim_audit(50);
        let dropped_twice = log.pruned_entries();
        assert!(dropped_twice > dropped);
        assert_eq!(log.pruned_head(), chain(&mirror.entries[..to_len(dropped_twice)]));
        assert_eq!(
            to_len(dropped_twice) + log.entries.len(),
            mirror.entries.len(),
            "every entry is either held or committed to"
        );
    }

    #[test]
    fn an_entry_digest_binds_its_version_target_reason_and_input() {
        let entry = |version, target, reason: &str, input| InvalidationEntry {
            at_version: StateVersion::from_raw(version),
            target,
            reason: reason.into(),
            input,
        };
        let base = || entry(1, InvalidationTarget::QueryResult(QueryId::from_raw(1)), "r", None);
        let genesis = InvalidationLog::PRUNED_GENESIS;
        let reference = base().chained_onto(genesis);
        assert_eq!(base().chained_onto(genesis), reference);
        let variants = [
            entry(2, InvalidationTarget::QueryResult(QueryId::from_raw(1)), "r", None),
            entry(1, InvalidationTarget::QueryResult(QueryId::from_raw(2)), "r", None),
            entry(1, InvalidationTarget::Model(ModelId::from_raw(1)), "r", None),
            entry(1, InvalidationTarget::QueryResult(QueryId::from_raw(1)), "s", None),
            entry(
                1,
                InvalidationTarget::QueryResult(QueryId::from_raw(1)),
                "r",
                Some(changed(ChangeKind::Removed)),
            ),
        ];
        for variant in variants {
            assert_ne!(variant.chained_onto(genesis), reference, "{variant:?}");
        }
        assert_ne!(
            entry(
                1,
                InvalidationTarget::QueryResult(QueryId::from_raw(1)),
                "r",
                Some(changed(ChangeKind::Added)),
            )
            .chained_onto(genesis),
            entry(
                1,
                InvalidationTarget::QueryResult(QueryId::from_raw(1)),
                "r",
                Some(changed(ChangeKind::Modified)),
            )
            .chained_onto(genesis),
        );
        // Order matters: the chain commits to the sequence, not the set.
        let (a, b) = (base(), entry(2, InvalidationTarget::AllResults, "r", None));
        assert_ne!(
            b.chained_onto(a.chained_onto(genesis)),
            a.chained_onto(b.chained_onto(genesis))
        );
    }

    #[test]
    fn an_entry_stays_within_the_size_the_audit_window_is_derived_from() {
        // `StateLimits` budgets 128 bytes per entry: the entry itself and a short reason string.
        assert!(std::mem::size_of::<InvalidationEntry>() <= 96);
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
