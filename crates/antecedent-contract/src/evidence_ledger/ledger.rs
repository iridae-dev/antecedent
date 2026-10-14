//! The runtime evidence ledger: sources, overlap, operations and results.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::{CanonicalWriter, SemanticDigest};

use crate::claim::EvidenceDependence;

use super::ids::{
    LEDGER_IDENTITY_VERSION, ObservationId, OperationId, ResultId, SourceId, digest_word, finish,
};
use super::operation::{
    AppliedOperation, ClaimClass, CombinationRequest, DetectedOverlap, LedgerOperation,
    OperationEffect, PoolingDependence, ResultRecord, ResultRevalidation, ResultStatus,
    Revalidation, SynthesisOperation,
};
use super::overlap::{
    MassAccount, Overlap, ReentryKind, RelationKind, intersection, intersection_count, is_subset,
    union_sorted,
};
use super::refusal::{LedgerError, RefusalKind};
use super::source::{
    EffectiveObservations, EvidenceSource, Lineage, ObservationBasis, SourceDeclaration,
    SourceKind, SourceStatus,
};

/// Most sources a ledger holds.
pub const MAX_SOURCES: usize = 1024;
/// Most results a ledger holds.
pub const MAX_RESULTS: usize = 4096;
/// Most operations a ledger logs.
pub const MAX_OPERATIONS: usize = 16_384;
/// Most observations one source declares.
pub const MAX_OBSERVATIONS_PER_SOURCE: usize = 1 << 20;
/// Most observation ids held across all sources.
pub const MAX_TOTAL_OBSERVATIONS: usize = 1 << 22;
/// Most sources one combination names.
pub const MAX_COMBINATION_SOURCES: usize = 64;
/// Most bytes in a label, reason or stage name.
pub const MAX_TEXT_BYTES: usize = 256;
/// Most populations or assumptions in one transport identity.
pub const MAX_TRANSPORT_ITEMS: usize = 64;

macro_rules! refuse {
    ($kind:ident, $code:literal, $detail:literal, $($message:tt)+) => {
        LedgerError {
            kind: RefusalKind::$kind,
            code: antecedent_core::reason_code!($code),
            detail: $detail,
            message: format!($($message)+),
            sources: Vec::new(),
        }
    };
}

macro_rules! double_counted {
    ($detail:literal, $($message:tt)+) => {
        refuse!(DoubleCounted, "evidence_double_counted", $detail, $($message)+)
    };
}

macro_rules! overlap_unknown {
    ($detail:literal, $($message:tt)+) => {
        refuse!(OverlapUnknown, "evidence_overlap_unknown", $detail, $($message)+)
    };
}

macro_rules! retired {
    ($detail:literal, $($message:tt)+) => {
        refuse!(SourceRetired, "evidence_source_withdrawn", $detail, $($message)+)
    };
}

macro_rules! invalid {
    ($detail:literal, $($message:tt)+) => {
        refuse!(InvalidArgument, "invalid_argument", $detail, $($message)+)
    };
}

macro_rules! not_licensed {
    ($detail:literal, $($message:tt)+) => {
        refuse!(NotLicensed, "cell_not_licensed", $detail, $($message)+)
    };
}

macro_rules! limit {
    ($detail:literal, $($message:tt)+) => {
        refuse!(LimitExceeded, "invalid_argument", $detail, $($message)+)
    };
}

/// How a retired source is lost to a dependent result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Loss {
    /// Withdrawn: gone for good.
    Permanent,
    /// Superseded, replaced or updated: a successor exists.
    Replaceable,
}

/// A source declaration that passed canonical validation.
struct CanonicalSource {
    declaration: SourceDeclaration,
    id: SourceId,
    effective: EffectiveObservations,
}

/// Retirement kind of a source-changing operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Retire {
    Update,
    Replace,
    Supersede,
    Withdraw,
}

/// A combination that passed the overlap policy.
struct CheckedCombination {
    id: ResultId,
    label: String,
    operation: SynthesisOperation,
    mass: MassAccount,
    claim_class: ClaimClass,
    observations_digest: Option<SemanticDigest>,
    transport_identities: Vec<SemanticDigest>,
    data_reuse: bool,
    dependence_declared: bool,
    replaces: Option<ResultId>,
}

enum Plan {
    Add {
        source: CanonicalSource,
        overlaps: Vec<DetectedOverlap>,
    },
    Retire {
        retire: Retire,
        target: SourceId,
        reason: Option<String>,
        replacement: Option<(CanonicalSource, Vec<DetectedOverlap>)>,
    },
    Revalidate,
    Combine(CheckedCombination),
}

/// The runtime record of which observations, experiments, priors, fitted
/// models and external claims contributed to which results, how they
/// overlap, and what changed since.
///
/// The ledger never promotes unknown overlap to independence, counts each
/// observation once per result and refuses to register a combination that
/// would count one twice.
#[derive(Clone, Debug)]
pub struct EvidenceLedger {
    sources: Vec<EvidenceSource>,
    source_index: BTreeMap<SourceId, usize>,
    results: Vec<ResultRecord>,
    result_index: BTreeMap<ResultId, usize>,
    log: Vec<AppliedOperation>,
    head: OperationId,
    total_observations: usize,
}

impl Default for EvidenceLedger {
    fn default() -> Self {
        Self::new()
    }
}

fn genesis() -> OperationId {
    let mut writer = CanonicalWriter::new("antecedent.evidence_ledger.genesis");
    writer.word(LEDGER_IDENTITY_VERSION);
    OperationId::from_digest(finish(&writer))
}

fn text_ok(text: &str) -> bool {
    !text.trim().is_empty() && text.len() <= MAX_TEXT_BYTES
}

impl EvidenceLedger {
    /// An empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            source_index: BTreeMap::new(),
            results: Vec::new(),
            result_index: BTreeMap::new(),
            log: Vec::new(),
            head: genesis(),
            total_observations: 0,
        }
    }

    // -- read access ------------------------------------------------------------------

    /// Sources in the order they were added.
    #[must_use]
    pub fn sources(&self) -> &[EvidenceSource] {
        &self.sources
    }

    /// One source.
    #[must_use]
    pub fn source(&self, id: SourceId) -> Option<&EvidenceSource> {
        self.source_index.get(&id).map(|&i| &self.sources[i])
    }

    /// Results in the order they were registered.
    #[must_use]
    pub fn results(&self) -> &[ResultRecord] {
        &self.results
    }

    /// One result.
    #[must_use]
    pub fn result(&self, id: ResultId) -> Option<&ResultRecord> {
        self.result_index.get(&id).map(|&i| &self.results[i])
    }

    /// Applied operations in order.
    #[must_use]
    pub fn operations(&self) -> &[AppliedOperation] {
        &self.log
    }

    /// Identity of the last applied operation (a hash chain over the history).
    #[must_use]
    pub const fn head(&self) -> OperationId {
        self.head
    }

    /// Digest of the whole ledger state: every source with its standing, every
    /// result with its standing and accounting, and the head.
    #[must_use]
    pub fn state_digest(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new("antecedent.evidence_ledger.state");
        writer.word(LEDGER_IDENTITY_VERSION);
        let mut sources: Vec<&EvidenceSource> = self.sources.iter().collect();
        sources.sort_by_key(|s| s.id);
        writer.word(sources.len() as u64);
        for source in sources {
            digest_word(&mut writer, source.id.digest());
            source.status.encode(&mut writer);
        }
        let mut results: Vec<&ResultRecord> = self.results.iter().collect();
        results.sort_by_key(|r| r.id);
        writer.word(results.len() as u64);
        for result in results {
            digest_word(&mut writer, result.id.digest());
            result.status.encode(&mut writer);
            writer.text(result.claim_class.as_str());
            writer.word(result.mass.sources);
            writer.word(result.mass.declared_slots);
            writer.word(result.mass.distinct_observations);
            writer.word(u64::from(result.data_reuse));
            writer.word(u64::from(result.dependence_declared));
        }
        digest_word(&mut writer, self.head.digest());
        finish(&writer)
    }

    /// Source-loss classification of one result against the current standings.
    ///
    /// This is the ledger's own observation (which sources were lost), not the
    /// class a consumer may act on: it knows nothing of representation,
    /// support, provider trust, calibration or coupling. Bind it to a claim
    /// node with `composition_verification::ledger_effects` and read the class
    /// from `revalidate`.
    #[must_use]
    pub fn standing(&self, id: ResultId) -> Option<Revalidation> {
        self.result(id).map(|r| r.status.revalidation())
    }

    // -- overlap queries ----------------------------------------------------------------

    fn require_known(&self, id: SourceId) -> Result<&EvidenceSource, LedgerError> {
        self.source(id).ok_or_else(|| {
            invalid!("evidence_ledger.unknown_source", "source {id} is not in the ledger")
                .with_sources([id])
        })
    }

    fn overlap_of(a: &EvidenceSource, b: &EvidenceSource) -> Overlap {
        match (&a.effective, &b.effective) {
            (EffectiveObservations::Known(x), EffectiveObservations::Known(y)) => {
                let shared = intersection_count(x, y);
                if shared == 0 {
                    Overlap::Independent
                } else {
                    Overlap::SharedObservations { shared }
                }
            }
            // An empty observation set overlaps nothing, known or not.
            (EffectiveObservations::Known(x), EffectiveObservations::Unknown(_))
            | (EffectiveObservations::Unknown(_), EffectiveObservations::Known(x))
                if x.is_empty() =>
            {
                Overlap::Independent
            }
            (EffectiveObservations::Unknown(reason), _)
            | (_, EffectiveObservations::Unknown(reason)) => Overlap::Unknown { reason },
        }
    }

    /// Overlap of two sources' observations.
    ///
    /// # Errors
    /// An id that is not in the ledger.
    pub fn overlap(&self, a: SourceId, b: SourceId) -> Result<Overlap, LedgerError> {
        Ok(Self::overlap_of(self.require_known(a)?, self.require_known(b)?))
    }

    /// The claim contract's coarse dependence label for two sources.
    ///
    /// # Errors
    /// An id that is not in the ledger.
    pub fn dependence(&self, a: SourceId, b: SourceId) -> Result<EvidenceDependence, LedgerError> {
        Ok(self.overlap(a, b)?.dependence())
    }

    /// Observations two sources share, sorted. Empty when independent or unknown.
    ///
    /// # Errors
    /// An id that is not in the ledger.
    pub fn shared_observations(
        &self,
        a: SourceId,
        b: SourceId,
    ) -> Result<Vec<ObservationId>, LedgerError> {
        let (sa, sb) = (self.require_known(a)?, self.require_known(b)?);
        Ok(match (sa.effective.known(), sb.effective.known()) {
            (Some(x), Some(y)) => intersection(x, y),
            _ => Vec::new(),
        })
    }

    /// The source and everything it was derived from.
    fn lineage_closure(&self, id: SourceId) -> Vec<SourceId> {
        let mut seen: BTreeSet<SourceId> = BTreeSet::new();
        let mut stack = vec![id];
        while let Some(current) = stack.pop() {
            if !seen.insert(current) {
                continue;
            }
            if let Some(source) = self.source(current) {
                stack.extend(source.declaration.lineage.parents());
            }
        }
        seen.into_iter().collect()
    }

    /// How two sources relate, in the composition vocabulary.
    ///
    /// # Errors
    /// An id that is not in the ledger.
    pub fn relation(&self, a: SourceId, b: SourceId) -> Result<RelationKind, LedgerError> {
        Ok(match self.overlap(a, b)? {
            Overlap::Independent => RelationKind::Independent,
            Overlap::Unknown { .. } => RelationKind::UnknownDependence,
            Overlap::SharedObservations { .. } => {
                let left: BTreeSet<SourceId> = self.lineage_closure(a).into_iter().collect();
                let common: Vec<&EvidenceSource> = self
                    .lineage_closure(b)
                    .into_iter()
                    .filter(|id| left.contains(id))
                    .filter_map(|id| self.source(id))
                    .collect();
                let has = |kind: SourceKind| common.iter().any(|s| s.declaration.kind == kind);
                if has(SourceKind::ObservationSet) || has(SourceKind::Experiment) {
                    RelationKind::SharedData
                } else if has(SourceKind::Prior) {
                    RelationKind::SharedPrior
                } else if has(SourceKind::FittedModel) {
                    RelationKind::SharedFittedModel
                } else {
                    RelationKind::SharedData
                }
            }
        })
    }

    fn reentry_between(&self, a: &EvidenceSource, b: &EvidenceSource) -> ReentryKind {
        let from_lineage = |descendant: &EvidenceSource, ancestor: SourceId| {
            if descendant.id == ancestor || !self.lineage_closure(descendant.id).contains(&ancestor)
            {
                return None;
            }
            Some(match descendant.declaration.lineage {
                Lineage::Duplicate { .. } => ReentryKind::Duplicate,
                Lineage::Transform { .. } => ReentryKind::Transform,
                Lineage::PriorPlusLikelihood { .. } => ReentryKind::PriorPlusLikelihood,
                Lineage::DerivedFromUsedData { .. } => ReentryKind::DerivedFromUsedData,
                Lineage::Subset { .. } | Lineage::Original => ReentryKind::SharedObservations,
            })
        };
        if let Some(kind) = from_lineage(b, a.id).or_else(|| from_lineage(a, b.id)) {
            return kind;
        }
        let kinds = (a.declaration.kind, b.declaration.kind);
        let derived = |k: SourceKind| k.is_model_or_claim();
        if derived(kinds.0) || derived(kinds.1) {
            ReentryKind::DerivedFromUsedData
        } else if (kinds.0 == SourceKind::Prior) != (kinds.1 == SourceKind::Prior) {
            ReentryKind::PriorPlusLikelihood
        } else {
            ReentryKind::SharedObservations
        }
    }

    /// Mass accounting over a set of sources: each observation counted once.
    ///
    /// # Errors
    /// An id that is not in the ledger.
    pub fn mass_account(&self, sources: &[SourceId]) -> Result<MassAccount, LedgerError> {
        let mut ids: Vec<SourceId> = sources.to_vec();
        ids.sort_unstable();
        ids.dedup();
        let mut known: Vec<&[ObservationId]> = Vec::new();
        let mut unknown = Vec::new();
        let mut slots = 0_u64;
        for id in &ids {
            let source = self.require_known(*id)?;
            match source.effective.known() {
                Some(rows) => {
                    slots += rows.len() as u64;
                    known.push(rows);
                }
                None => unknown.push(*id),
            }
        }
        let distinct = union_sorted(known).len() as u64;
        Ok(MassAccount {
            sources: ids.len() as u64,
            declared_slots: slots,
            distinct_observations: distinct,
            repeated_slots: slots - distinct,
            unknown_sources: unknown,
        })
    }

    // -- validation -------------------------------------------------------------------

    /// First non-active member of the source's lineage closure.
    fn loss_of(&self, id: SourceId) -> Option<(SourceId, Loss)> {
        let mut replaceable = None;
        for member in self.lineage_closure(id) {
            let Some(source) = self.source(member) else { continue };
            match source.status {
                SourceStatus::Active => {}
                SourceStatus::Withdrawn { .. } => return Some((member, Loss::Permanent)),
                _ => {
                    if replaceable.is_none() {
                        replaceable = Some((member, Loss::Replaceable));
                    }
                }
            }
        }
        replaceable
    }

    fn require_usable(&self, id: SourceId) -> Result<&EvidenceSource, LedgerError> {
        let source = self.require_known(id)?;
        match self.loss_of(id) {
            None => Ok(source),
            Some((member, Loss::Permanent)) => Err(retired!(
                "evidence_ledger.source_withdrawn",
                "source {member} (needed by {id}) is withdrawn; remove it or register evidence that does not rest on it"
            )
            .with_sources([member])),
            Some((member, Loss::Replaceable)) => {
                let successor = match self.source(member).map(|s| &s.status) {
                    Some(
                        SourceStatus::Superseded { by, .. }
                        | SourceStatus::Replaced { by, .. }
                        | SourceStatus::Updated { by, .. },
                    ) => format!("use its successor {by}"),
                    _ => "use its successor".to_owned(),
                };
                Err(retired!(
                    "evidence_ledger.source_superseded",
                    "source {member} (needed by {id}) was superseded, replaced or updated; {successor}"
                )
                .with_sources([member]))
            }
        }
    }

    fn canonical_source(
        &self,
        declaration: &SourceDeclaration,
    ) -> Result<CanonicalSource, LedgerError> {
        if !text_ok(&declaration.label) {
            return Err(invalid!(
                "evidence_ledger.invalid_declaration",
                "a source label must be non-empty and at most {MAX_TEXT_BYTES} bytes"
            ));
        }
        let mut declaration = declaration.clone();
        // Observation rows: canonical order, no repeated observation inside one set.
        if let ObservationBasis::Declared(rows) = &mut declaration.observations {
            if rows.len() > MAX_OBSERVATIONS_PER_SOURCE {
                return Err(limit!(
                    "evidence_ledger.limit_exceeded",
                    "a source declares at most {MAX_OBSERVATIONS_PER_SOURCE} observations"
                ));
            }
            rows.sort_unstable();
            if rows.windows(2).any(|w| w[0] == w[1]) {
                return Err(double_counted!(
                    "evidence_ledger.duplicate_row",
                    "an observation appears more than once inside one source; declare each observation once"
                ));
            }
        }
        // Lineage: parents exist, are usable and are not repeated.
        match &mut declaration.lineage {
            Lineage::DerivedFromUsedData { inputs } => {
                if inputs.is_empty() {
                    return Err(invalid!(
                        "evidence_ledger.invalid_declaration",
                        "a derived source names the sources it was built from"
                    ));
                }
                inputs.sort_unstable();
                if inputs.windows(2).any(|w| w[0] == w[1]) {
                    return Err(double_counted!(
                        "evidence_ledger.repeated_source",
                        "a source is listed more than once among the inputs"
                    ));
                }
            }
            Lineage::PriorPlusLikelihood { prior, likelihood } => {
                if prior == likelihood {
                    return Err(double_counted!(
                        "evidence_ledger.repeated_source",
                        "the prior and the likelihood are the same source"
                    ));
                }
                if self.require_known(*prior)?.declaration.kind != SourceKind::Prior {
                    return Err(invalid!(
                        "evidence_ledger.invalid_declaration",
                        "the prior of a prior-plus-likelihood source must have kind prior"
                    ));
                }
            }
            _ => {}
        }
        let parents = declaration.lineage.parents();
        for parent in &parents {
            self.require_usable(*parent)?;
        }
        // Transport provenance.
        if let Some(transport) = &mut declaration.transport {
            if declaration.kind == SourceKind::ObservationSet {
                return Err(invalid!(
                    "evidence_ledger.invalid_declaration",
                    "raw observations are not transported; transport a claim or model"
                ));
            }
            transport.source_populations.sort_unstable();
            transport.source_populations.dedup();
            transport.assumptions.sort_unstable();
            transport.assumptions.dedup();
            if transport.source_populations.is_empty()
                || transport.source_populations.len() > MAX_TRANSPORT_ITEMS
                || transport.assumptions.len() > MAX_TRANSPORT_ITEMS
                || transport.source_populations.iter().any(|p| !text_ok(p))
                || transport.assumptions.iter().any(|p| !text_ok(p))
                || !text_ok(&transport.target_population)
            {
                return Err(invalid!(
                    "evidence_ledger.invalid_declaration",
                    "a transport identity names its source populations, target population and assumptions"
                ));
            }
            // A transport that states no assumption licenses nothing, and a
            // target that is one of its own sources is not a transport at all;
            // the decision and handoff routes refuse both, so refuse here
            // rather than late and under another code.
            if transport.assumptions.is_empty() {
                return Err(invalid!(
                    "evidence_ledger.invalid_declaration",
                    "a transport identity states at least one invariance or selection assumption"
                ));
            }
            if transport
                .source_populations
                .iter()
                .any(|p| *p == transport.target_population)
            {
                return Err(invalid!(
                    "evidence_ledger.invalid_declaration",
                    "a transport target population is not one of its source populations"
                ));
            }
        }
        let effective = self.resolve_effective(&declaration)?;
        let id = declaration.identity();
        Ok(CanonicalSource { declaration, id, effective })
    }

    /// Resolve observations through lineage, refusing rows that contradict it.
    fn resolve_effective(
        &self,
        declaration: &SourceDeclaration,
    ) -> Result<EffectiveObservations, LedgerError> {
        let parent_rows = |id: SourceId| -> Option<Arc<Vec<ObservationId>>> {
            match self.source(id).map(|s| &s.effective) {
                Some(EffectiveObservations::Known(rows)) => Some(Arc::clone(rows)),
                _ => None,
            }
        };
        let mismatch = || {
            invalid!(
                "evidence_ledger.lineage_rows_mismatch",
                "the declared observations contradict the declared lineage"
            )
        };
        let declared = match &declaration.observations {
            ObservationBasis::Declared(rows) => Some(rows),
            ObservationBasis::Undeclared => None,
        };
        match &declaration.lineage {
            Lineage::Original => Ok(declared.map_or(
                EffectiveObservations::Unknown("observations were not declared"),
                |rows| EffectiveObservations::Known(Arc::new(rows.clone())),
            )),
            Lineage::Duplicate { of } | Lineage::Transform { of, .. } => {
                match (declared, parent_rows(*of)) {
                    (Some(rows), Some(parent)) => {
                        if rows.as_slice() != parent.as_slice() {
                            return Err(mismatch());
                        }
                        Ok(EffectiveObservations::Known(parent))
                    }
                    (Some(rows), None) => Ok(EffectiveObservations::Known(Arc::new(rows.clone()))),
                    (None, Some(parent)) => Ok(EffectiveObservations::Known(parent)),
                    (None, None) => Ok(EffectiveObservations::Unknown(
                        "a copy or transform of a source with unknown observations",
                    )),
                }
            }
            Lineage::Subset { of } => match (declared, parent_rows(*of)) {
                (Some(rows), Some(parent)) => {
                    if !is_subset(rows, &parent) {
                        return Err(mismatch());
                    }
                    Ok(EffectiveObservations::Known(Arc::new(rows.clone())))
                }
                (Some(rows), None) => Ok(EffectiveObservations::Known(Arc::new(rows.clone()))),
                (None, _) => {
                    Ok(EffectiveObservations::Unknown("the rows of a subset were not declared"))
                }
            },
            Lineage::PriorPlusLikelihood { .. } | Lineage::DerivedFromUsedData { .. } => {
                let parents = declaration.lineage.parents();
                let rows: Vec<Option<Arc<Vec<ObservationId>>>> =
                    parents.iter().map(|p| parent_rows(*p)).collect();
                let all_known = rows.iter().all(Option::is_some);
                let union =
                    all_known.then(|| union_sorted(rows.iter().flatten().map(|r| r.as_slice())));
                match (declared, union) {
                    (Some(declared), Some(union)) => {
                        if !is_subset(declared, &union) {
                            return Err(mismatch());
                        }
                        Ok(EffectiveObservations::Known(Arc::new(declared.clone())))
                    }
                    (Some(declared), None) => {
                        Ok(EffectiveObservations::Known(Arc::new(declared.clone())))
                    }
                    (None, Some(union)) => Ok(EffectiveObservations::Known(Arc::new(union))),
                    (None, None) => Ok(EffectiveObservations::Unknown(
                        "built from sources with unknown observations",
                    )),
                }
            }
        }
    }

    fn overlaps_with_existing(
        &self,
        source: &CanonicalSource,
        skip: Option<SourceId>,
    ) -> Vec<DetectedOverlap> {
        let Some(rows) = source.effective.known() else { return Vec::new() };
        let mut found = Vec::new();
        for existing in &self.sources {
            if Some(existing.id) == skip || existing.id == source.id {
                continue;
            }
            if let Some(other) = existing.effective.known() {
                let shared = intersection_count(rows, other);
                if shared > 0 {
                    found.push(DetectedOverlap { existing: existing.id, shared });
                }
            }
        }
        found
    }

    // -- combination policy ---------------------------------------------------------------

    fn check_listed(&self, ids: &[SourceId]) -> Result<Vec<SourceId>, LedgerError> {
        if ids.len() > MAX_COMBINATION_SOURCES {
            return Err(limit!(
                "evidence_ledger.limit_exceeded",
                "a combination names at most {MAX_COMBINATION_SOURCES} sources"
            ));
        }
        let mut sorted = ids.to_vec();
        sorted.sort_unstable();
        if let Some(window) = sorted.windows(2).find(|w| w[0] == w[1]) {
            return Err(double_counted!(
                "evidence_ledger.repeated_source",
                "source {} is listed more than once; each source counts once per result",
                window[0]
            )
            .with_sources([window[0]]));
        }
        for id in &sorted {
            self.require_usable(*id)?;
        }
        Ok(sorted)
    }

    fn double_counting(&self, a: SourceId, b: SourceId, shared: u64) -> LedgerError {
        let (sa, sb) = (&self.sources[self.source_index[&a]], &self.sources[self.source_index[&b]]);
        let reentry = self.reentry_between(sa, sb);
        let how = reentry.as_str();
        let base = format!(
            "sources {a} and {b} share {shared} observation(s) (re-entry: {how}); counting both as independent evidence would count each shared observation twice. Pool them with a declared covariance, declare an assumption range, or drop one source"
        );
        let error = match reentry {
            ReentryKind::Duplicate => {
                double_counted!("evidence_ledger.reentry_duplicate", "{base}")
            }
            ReentryKind::Transform => {
                double_counted!("evidence_ledger.reentry_transform", "{base}")
            }
            ReentryKind::PriorPlusLikelihood => {
                double_counted!("evidence_ledger.reentry_prior_plus_likelihood", "{base}")
            }
            ReentryKind::DerivedFromUsedData => {
                double_counted!("evidence_ledger.reentry_derived_from_used_data", "{base}")
            }
            ReentryKind::SharedObservations => {
                double_counted!("evidence_ledger.shared_observations", "{base}")
            }
        };
        error.with_sources([a, b])
    }

    fn unknown_overlap(a: SourceId, b: SourceId, reason: &str) -> LedgerError {
        overlap_unknown!(
            "evidence_ledger.overlap_unknown",
            "the observations behind {a} and {b} are not known ({reason}); independence is not assumed. Declare the row-level observations, supply a covariance with cross terms, or request an assumption range"
        )
        .with_sources([a, b])
    }

    /// Check that every pair in `ids` is independent.
    fn require_pairwise_independent(&self, ids: &[SourceId]) -> Result<(), LedgerError> {
        let mut unknown: Option<LedgerError> = None;
        for (i, &a) in ids.iter().enumerate() {
            for &b in &ids[i + 1..] {
                match self.overlap(a, b)? {
                    Overlap::Independent => {}
                    Overlap::SharedObservations { shared } => {
                        return Err(self.double_counting(a, b, shared));
                    }
                    Overlap::Unknown { reason } => {
                        unknown.get_or_insert_with(|| Self::unknown_overlap(a, b, reason));
                    }
                }
            }
        }
        unknown.map_or(Ok(()), Err)
    }

    fn require_disjoint_groups(
        &self,
        left: &[SourceId],
        right: &[SourceId],
        shared_detail: impl Fn(&Self, SourceId, SourceId, u64) -> LedgerError,
    ) -> Result<(), LedgerError> {
        let mut unknown: Option<LedgerError> = None;
        for &a in left {
            for &b in right {
                match self.overlap(a, b)? {
                    Overlap::Independent => {}
                    Overlap::SharedObservations { shared } => {
                        return Err(shared_detail(self, a, b, shared));
                    }
                    Overlap::Unknown { reason } => {
                        unknown.get_or_insert_with(|| Self::unknown_overlap(a, b, reason));
                    }
                }
            }
        }
        unknown.map_or(Ok(()), Err)
    }

    fn native_and_transported(a: SourceId, b: SourceId, shared: u64) -> LedgerError {
        double_counted!(
            "evidence_ledger.native_and_transported",
            "native source and transported claim ({a}, {b}) rest on {shared} common observation(s): the same source cannot count as native target data and as transported evidence"
        )
        .with_sources([a, b])
    }

    fn account(
        &self,
        ids: &[SourceId],
    ) -> Result<(MassAccount, Option<SemanticDigest>, Vec<SemanticDigest>), LedgerError> {
        let mass = self.mass_account(ids)?;
        let mut known: Vec<&[ObservationId]> = Vec::new();
        let mut transports = Vec::new();
        for id in ids {
            let source = self.require_known(*id)?;
            if let Some(rows) = source.effective.known() {
                known.push(rows);
            }
            if let Some(transport) = &source.declaration.transport {
                transports.push(transport.digest());
            }
        }
        transports.sort_unstable();
        transports.dedup();
        let digest = mass.unknown_sources.is_empty().then(|| {
            let rows = union_sorted(known);
            let mut writer = CanonicalWriter::new("antecedent.evidence_ledger.observations");
            writer.word(LEDGER_IDENTITY_VERSION);
            writer.word(rows.len() as u64);
            for row in rows {
                digest_word(&mut writer, row.digest());
            }
            finish(&writer)
        });
        Ok((mass, digest, transports))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one arm per combination operation, each a flat sequence of independent admissibility checks"
    )]
    fn check_combination(
        &self,
        request: &CombinationRequest,
    ) -> Result<CheckedCombination, LedgerError> {
        if !text_ok(&request.label) {
            return Err(invalid!(
                "evidence_ledger.invalid_declaration",
                "a result label must be non-empty and at most {MAX_TEXT_BYTES} bytes"
            ));
        }
        if self.results.len() >= MAX_RESULTS {
            return Err(limit!(
                "evidence_ledger.limit_exceeded",
                "a ledger holds at most {MAX_RESULTS} results"
            ));
        }
        if let Some(replaced) = request.replaces {
            let replaced_record = self.result(replaced).ok_or_else(|| {
                invalid!("evidence_ledger.unknown_result", "result {replaced} is not in the ledger")
            })?;
            if matches!(replaced_record.status, ResultStatus::Replaced { .. }) {
                return Err(invalid!(
                    "evidence_ledger.unknown_result",
                    "result {replaced} was already replaced by a recomputation"
                ));
            }
        }
        let mut canonical = request.operation.clone();
        let mut claim_class = ClaimClass::PointOnly;
        let mut data_reuse = false;
        let mut dependence_declared = false;
        let direct: Vec<SourceId>;
        let precomputed: Option<(MassAccount, Option<SemanticDigest>, Vec<SemanticDigest>)>;
        match &mut canonical {
            SynthesisOperation::StatisticalPooling { sources, dependence } => {
                if sources.len() < 2 {
                    return Err(invalid!(
                        "evidence_ledger.operation_arity",
                        "pooling combines at least two sources"
                    ));
                }
                let ids = self.check_listed(sources)?;
                sources.clone_from(&ids);
                // A claim that carries transport identity is never native data.
                let (transported, native): (Vec<SourceId>, Vec<SourceId>) =
                    ids.iter().copied().partition(|id| {
                        self.source(*id).is_some_and(|s| s.declaration.transport.is_some())
                    });
                self.require_disjoint_groups(&native, &transported, |_, a, b, shared| {
                    Self::native_and_transported(a, b, shared)
                })?;
                match *dependence {
                    PoolingDependence::Independent => self.require_pairwise_independent(&ids)?,
                    PoolingDependence::DeclaredCovariance { has_cross_covariance, .. } => {
                        if !has_cross_covariance {
                            self.require_pairwise_independent_for_diagonal(&ids)?;
                        }
                        dependence_declared = has_cross_covariance;
                    }
                    PoolingDependence::AssumptionRange { rho_lo, rho_hi } => {
                        let k = ids.len() as f64;
                        let lower_bound = -1.0 / (k - 1.0);
                        if !rho_lo.is_finite()
                            || !rho_hi.is_finite()
                            || rho_lo >= rho_hi
                            || rho_hi >= 1.0
                            || rho_lo <= lower_bound
                        {
                            return Err(invalid!(
                                "evidence_ledger.invalid_declaration",
                                "an assumption range needs finite bounds with lo < hi inside the positive definite interval (-1/(k-1), 1)"
                            ));
                        }
                        claim_class = ClaimClass::AssumptionRange;
                        dependence_declared = true;
                    }
                }
                direct = ids;
                precomputed = None;
            }
            SynthesisOperation::BayesianBorrowing { priors, likelihood } => {
                if likelihood.is_empty() {
                    return Err(invalid!(
                        "evidence_ledger.operation_arity",
                        "borrowing needs at least one likelihood-bearing source"
                    ));
                }
                let mut all = priors.clone();
                all.extend(likelihood.iter().copied());
                let ids = self.check_listed(&all)?;
                for prior in priors.iter() {
                    if self.require_known(*prior)?.declaration.kind != SourceKind::Prior {
                        return Err(invalid!(
                            "evidence_ledger.invalid_declaration",
                            "a borrowed prior must have kind prior"
                        )
                        .with_sources([*prior]));
                    }
                }
                priors.sort_unstable();
                likelihood.sort_unstable();
                self.require_pairwise_independent(likelihood)?;
                self.require_pairwise_independent(priors)?;
                self.require_disjoint_groups(priors, likelihood, |ledger, a, b, shared| {
                    let (sa, sb) =
                        (&ledger.sources[ledger.source_index[&a]], &ledger.sources[ledger.source_index[&b]]);
                    match ledger.reentry_between(sa, sb) {
                        ReentryKind::SharedObservations | ReentryKind::PriorPlusLikelihood => {
                            double_counted!(
                                "evidence_ledger.reentry_prior_plus_likelihood",
                                "prior {a} and likelihood {b} share {shared} observation(s): the data would inform the estimate once as prior and once as likelihood"
                            )
                            .with_sources([a, b])
                        }
                        _ => ledger.double_counting(a, b, shared),
                    }
                })?;
                direct = ids;
                precomputed = None;
            }
            SynthesisOperation::CausalTransport { transported, native } => {
                if transported.is_empty() {
                    return Err(invalid!(
                        "evidence_ledger.operation_arity",
                        "transport carries at least one transported claim"
                    ));
                }
                if let Some(id) = transported.iter().find(|id| native.contains(id)) {
                    return Err(double_counted!(
                        "evidence_ledger.native_and_transported",
                        "source {id} is listed as both native and transported"
                    )
                    .with_sources([*id]));
                }
                let mut all = transported.clone();
                all.extend(native.iter().copied());
                let ids = self.check_listed(&all)?;
                for id in transported.iter() {
                    if self.require_known(*id)?.declaration.transport.is_none() {
                        return Err(invalid!(
                            "evidence_ledger.transport_identity_missing",
                            "a transported claim carries its transport identity (source populations, selection diagram, assumptions)"
                        )
                        .with_sources([*id]));
                    }
                }
                for id in native.iter() {
                    if self.require_known(*id)?.declaration.transport.is_some() {
                        return Err(invalid!(
                            "evidence_ledger.transported_as_native",
                            "a claim carrying a transport identity is not native target data"
                        )
                        .with_sources([*id]));
                    }
                }
                transported.sort_unstable();
                native.sort_unstable();
                self.require_disjoint_groups(native, transported, |_, a, b, shared| {
                    Self::native_and_transported(a, b, shared)
                })?;
                direct = ids;
                precomputed = None;
            }
            SynthesisOperation::EvidenceReuse { of } => {
                let original = self.result(*of).ok_or_else(|| {
                    invalid!("evidence_ledger.unknown_result", "result {of} is not in the ledger")
                })?;
                if original.status != ResultStatus::Valid {
                    return Err(retired!(
                        "evidence_ledger.result_not_valid",
                        "result {of} is {}; recompute it before citing it again",
                        original.status.revalidation().as_str()
                    ));
                }
                claim_class = original.claim_class;
                data_reuse = original.data_reuse;
                dependence_declared = original.dependence_declared;
                direct = Vec::new();
                precomputed = Some((
                    original.mass.clone(),
                    original.observations_digest,
                    original.transport_identities.clone(),
                ));
            }
            SynthesisOperation::RepeatedObservationUse { sources, stage } => {
                if sources.is_empty() || !text_ok(stage) {
                    return Err(invalid!(
                        "evidence_ledger.operation_arity",
                        "repeated use names the sources and the later stage that uses them again"
                    ));
                }
                let ids = self.check_listed(sources)?;
                sources.clone_from(&ids);
                let mass = self.mass_account(&ids)?;
                if let Some(&first) = mass.unknown_sources.first() {
                    return Err(overlap_unknown!(
                        "evidence_ledger.overlap_unknown",
                        "the observations behind {first} are not known, so their repeated use cannot be accounted; declare the row-level observations"
                    )
                    .with_sources([first]));
                }
                data_reuse = true;
                direct = ids;
                precomputed = None;
            }
        }
        let (mass, observations_digest, transport_identities) = match precomputed {
            Some(values) => values,
            None => self.account(&direct)?,
        };
        let mut writer = CanonicalWriter::new("antecedent.evidence_ledger.result");
        writer.word(LEDGER_IDENTITY_VERSION);
        writer.text(&request.label);
        canonical.encode(&mut writer);
        match request.replaces {
            Some(replaced) => {
                writer.word(1);
                digest_word(&mut writer, replaced.digest());
            }
            None => {
                writer.word(0);
            }
        }
        let id = ResultId::from_digest(finish(&writer));
        if self.result_index.contains_key(&id) {
            return Err(invalid!(
                "evidence_ledger.duplicate_result",
                "this combination is already registered as result {id}"
            ));
        }
        Ok(CheckedCombination {
            id,
            label: request.label.clone(),
            operation: canonical,
            mass,
            claim_class,
            observations_digest,
            transport_identities,
            data_reuse,
            dependence_declared,
            replaces: request.replaces,
        })
    }

    /// A covariance with no cross terms states independence: it needs the
    /// observations to be known and disjoint.
    fn require_pairwise_independent_for_diagonal(
        &self,
        ids: &[SourceId],
    ) -> Result<(), LedgerError> {
        for (i, &a) in ids.iter().enumerate() {
            for &b in &ids[i + 1..] {
                match self.overlap(a, b)? {
                    Overlap::Independent => {}
                    Overlap::SharedObservations { shared } => {
                        return Err(double_counted!(
                            "evidence_ledger.diagonal_covariance_shared_rows",
                            "the declared covariance has no cross terms but sources {a} and {b} share {shared} observation(s); supply the cross covariance"
                        )
                        .with_sources([a, b]));
                    }
                    Overlap::Unknown { reason } => return Err(Self::unknown_overlap(a, b, reason)),
                }
            }
        }
        Ok(())
    }

    /// The statement a caller may not make from a result: an uncertainty
    /// interval or posterior. Always refused; the message names the reason.
    ///
    /// # Errors
    /// Always: an unknown result is invalid, a known one is not licensed.
    pub fn request_interval(&self, id: ResultId) -> Result<std::convert::Infallible, LedgerError> {
        let record = self.result(id).ok_or_else(|| {
            invalid!("evidence_ledger.unknown_result", "result {id} is not in the ledger")
        })?;
        Err(not_licensed!(
            "evidence_ledger.interval_not_licensed",
            "result {id} is a {} claim; the ledger licenses no interval, posterior or coverage statement for pooled evidence until its coverage is measured",
            record.claim_class.as_str()
        ))
    }

    /// A result that is currently usable.
    ///
    /// # Errors
    /// An unknown result, or one that is stale, withdrawn or replaced.
    pub fn require_valid(&self, id: ResultId) -> Result<&ResultRecord, LedgerError> {
        let record = self.result(id).ok_or_else(|| {
            invalid!("evidence_ledger.unknown_result", "result {id} is not in the ledger")
        })?;
        if record.status == ResultStatus::Valid {
            Ok(record)
        } else {
            Err(retired!(
                "evidence_ledger.result_not_valid",
                "result {id} is {}: its evidence changed after it was registered",
                record.status.revalidation().as_str()
            ))
        }
    }

    /// The sources a result rests on, resolved through reuse.
    #[must_use]
    pub fn result_sources(&self, id: ResultId) -> Vec<SourceId> {
        let mut current = id;
        for _ in 0..=self.results.len() {
            let Some(record) = self.result(current) else { return Vec::new() };
            match &record.operation {
                SynthesisOperation::EvidenceReuse { of } => current = *of,
                other => {
                    let mut ids = other.direct_sources();
                    ids.sort_unstable();
                    return ids;
                }
            }
        }
        Vec::new()
    }

    /// A declaration for a model or claim derived from a valid result, so
    /// that a later withdrawal of the result's sources reaches it through
    /// lineage.
    ///
    /// # Errors
    /// An unknown or unusable result, or a kind that is not a model or claim.
    pub fn declaration_from_result(
        &self,
        result: ResultId,
        label: impl Into<String>,
        kind: SourceKind,
    ) -> Result<SourceDeclaration, LedgerError> {
        self.require_valid(result)?;
        if !kind.is_model_or_claim() {
            return Err(invalid!(
                "evidence_ledger.invalid_declaration",
                "a result is turned into a fitted model or an external claim"
            ));
        }
        let inputs = self.result_sources(result);
        Ok(SourceDeclaration::new(label, kind)
            .with_lineage(Lineage::DerivedFromUsedData { inputs }))
    }

    // -- classification -----------------------------------------------------------------

    fn source_loss_class(&self, operation: &SynthesisOperation) -> Revalidation {
        let losses = |ids: &[SourceId]| -> (usize, usize) {
            let mut permanent = 0;
            let mut any = 0;
            for id in ids {
                match self.loss_of(*id) {
                    Some((_, Loss::Permanent)) => {
                        permanent += 1;
                        any += 1;
                    }
                    Some((_, Loss::Replaceable)) => any += 1,
                    None => {}
                }
            }
            (permanent, any)
        };
        // One rule, four operations: nothing lost keeps the class, losing the sources a
        // result cannot do without withdraws it, any other loss asks for recomputation.
        let by_loss = |any: usize, remaining: usize, needed: usize| {
            if any == 0 {
                Revalidation::StillValid
            } else if remaining < needed {
                Revalidation::Withdrawn
            } else {
                Revalidation::Recompute
            }
        };
        match operation {
            SynthesisOperation::StatisticalPooling { sources, .. } => {
                let (permanent, any) = losses(sources);
                by_loss(any, sources.len() - permanent, 2)
            }
            SynthesisOperation::BayesianBorrowing { priors, likelihood } => {
                let (perm_l, any_l) = losses(likelihood);
                let (_, any_p) = losses(priors);
                by_loss(any_l + any_p, likelihood.len() - perm_l, 1)
            }
            SynthesisOperation::CausalTransport { transported, native } => {
                let (perm_t, any_t) = losses(transported);
                let (_, any_n) = losses(native);
                by_loss(any_t + any_n, transported.len() - perm_t, 1)
            }
            SynthesisOperation::RepeatedObservationUse { sources, .. } => {
                let (permanent, any) = losses(sources);
                by_loss(any, sources.len() - permanent, 1)
            }
            SynthesisOperation::EvidenceReuse { .. } => Revalidation::StillValid,
        }
    }

    /// Re-classify every result against the current source standings. A
    /// result never returns to valid; withdrawn and replaced are terminal.
    fn reclassify(&mut self, cause: OperationId) -> Vec<ResultRevalidation> {
        for i in 0..self.results.len() {
            let current = self.results[i].status.clone();
            if matches!(current, ResultStatus::Withdrawn { .. } | ResultStatus::Replaced { .. }) {
                continue;
            }
            let class = match &self.results[i].operation {
                SynthesisOperation::EvidenceReuse { of } => {
                    // A reuse takes its parent's class; a replaced parent has a successor,
                    // so the reuse recomputes rather than withdraws.
                    match self.result(*of).map(|r| &r.status) {
                        Some(ResultStatus::Replaced { .. }) => Revalidation::Recompute,
                        Some(status) => status.revalidation(),
                        None => Revalidation::Withdrawn,
                    }
                }
                operation => self.source_loss_class(operation),
            };
            let updated = match (class, &current) {
                (Revalidation::Withdrawn, _) => ResultStatus::Withdrawn { cause },
                (Revalidation::Recompute, ResultStatus::Valid) => {
                    ResultStatus::NeedsRecompute { cause }
                }
                _ => current.clone(),
            };
            self.results[i].status = updated;
        }
        self.source_loss_classification()
    }

    /// Source-loss classification of every result, sorted by id. See
    /// [`Self::standing`] for what this class does and does not say.
    #[must_use]
    pub fn source_loss_classification(&self) -> Vec<ResultRevalidation> {
        let mut out: Vec<ResultRevalidation> = self
            .results
            .iter()
            .map(|r| ResultRevalidation { result: r.id, class: r.status.revalidation() })
            .collect();
        out.sort_by_key(|r| r.result);
        out
    }

    /// Same as [`Self::source_loss_classification`].
    #[must_use]
    pub fn classification(&self) -> Vec<ResultRevalidation> {
        self.source_loss_classification()
    }

    // -- operations ---------------------------------------------------------------------

    fn plan(&self, operation: &LedgerOperation) -> Result<Plan, LedgerError> {
        if self.log.len() >= MAX_OPERATIONS {
            return Err(limit!(
                "evidence_ledger.limit_exceeded",
                "a ledger logs at most {MAX_OPERATIONS} operations"
            ));
        }
        match operation {
            LedgerOperation::Add { source } => {
                if self.sources.len() >= MAX_SOURCES {
                    return Err(limit!(
                        "evidence_ledger.limit_exceeded",
                        "a ledger holds at most {MAX_SOURCES} sources"
                    ));
                }
                let canonical = self.canonical_source(source)?;
                self.check_new_source(&canonical)?;
                let overlaps = self.overlaps_with_existing(&canonical, None);
                Ok(Plan::Add { source: canonical, overlaps })
            }
            LedgerOperation::Update { target, replacement } => {
                self.plan_retire(Retire::Update, *target, Some(replacement), None)
            }
            LedgerOperation::Replace { target, replacement } => {
                self.plan_retire(Retire::Replace, *target, Some(replacement), None)
            }
            LedgerOperation::Supersede { target, replacement } => {
                self.plan_retire(Retire::Supersede, *target, Some(replacement), None)
            }
            LedgerOperation::Withdraw { target, reason } => {
                self.plan_retire(Retire::Withdraw, *target, None, Some(reason))
            }
            LedgerOperation::RevalidateDownstream => Ok(Plan::Revalidate),
            LedgerOperation::Combine { request } => {
                Ok(Plan::Combine(self.check_combination(request)?))
            }
        }
    }

    fn check_new_source(&self, source: &CanonicalSource) -> Result<(), LedgerError> {
        if self.source_index.contains_key(&source.id) {
            return Err(double_counted!(
                "evidence_ledger.duplicate_source",
                "source {} is already in the ledger; the same evidence is not added twice",
                source.id
            )
            .with_sources([source.id]));
        }
        let rows = source.effective.known().map_or(0, <[ObservationId]>::len);
        if self.total_observations + rows > MAX_TOTAL_OBSERVATIONS {
            return Err(limit!(
                "evidence_ledger.limit_exceeded",
                "a ledger holds at most {MAX_TOTAL_OBSERVATIONS} observation ids"
            ));
        }
        if self.sources.len() >= MAX_SOURCES {
            return Err(limit!(
                "evidence_ledger.limit_exceeded",
                "a ledger holds at most {MAX_SOURCES} sources"
            ));
        }
        Ok(())
    }

    #[allow(
        clippy::too_many_lines,
        reason = "a flat sequence of independent retirement checks, one per way a source can be withdrawn or superseded"
    )]
    fn plan_retire(
        &self,
        retire: Retire,
        target: SourceId,
        replacement: Option<&SourceDeclaration>,
        reason: Option<&String>,
    ) -> Result<Plan, LedgerError> {
        let existing = self.require_known(target)?;
        match (retire, &existing.status) {
            (Retire::Withdraw, SourceStatus::Withdrawn { .. }) => {
                return Err(retired!(
                    "evidence_ledger.source_withdrawn",
                    "source {target} is already withdrawn"
                )
                .with_sources([target]));
            }
            (Retire::Withdraw, _) | (_, SourceStatus::Active) => {}
            (_, _) => {
                return Err(retired!(
                    "evidence_ledger.source_superseded",
                    "source {target} was already retired ({}); change its successor instead",
                    existing.status.as_str()
                )
                .with_sources([target]));
            }
        }
        if let Some(reason) = reason {
            if !text_ok(reason) {
                return Err(invalid!(
                    "evidence_ledger.invalid_declaration",
                    "a withdrawal states its reason"
                ));
            }
        }
        let Some(replacement) = replacement else {
            return Ok(Plan::Retire { retire, target, reason: reason.cloned(), replacement: None });
        };
        let canonical = self.canonical_source(replacement)?;
        if canonical
            .declaration
            .lineage
            .parents()
            .iter()
            .any(|p| *p == target || self.lineage_closure(*p).contains(&target))
        {
            return Err(invalid!(
                "evidence_ledger.invalid_declaration",
                "a replacement cannot be derived from the source it retires"
            ));
        }
        let kind = existing.declaration.kind;
        if retire == Retire::Supersede && !kind.is_model_or_claim() {
            return Err(invalid!(
                "evidence_ledger.supersede_kind_mismatch",
                "only a fitted model or an external claim is superseded; replace or update other evidence"
            ));
        }
        if canonical.declaration.kind != kind {
            return Err(match retire {
                Retire::Supersede => invalid!(
                    "evidence_ledger.supersede_kind_mismatch",
                    "a {} cannot supersede a {}",
                    canonical.declaration.kind.as_str(),
                    kind.as_str()
                ),
                _ => invalid!(
                    "evidence_ledger.kind_mismatch",
                    "a {} cannot stand in for a {}",
                    canonical.declaration.kind.as_str(),
                    kind.as_str()
                ),
            });
        }
        if retire == Retire::Update {
            match (existing.effective.known(), canonical.effective.known()) {
                (Some(old), Some(new)) => {
                    if !is_subset(old, new) {
                        return Err(invalid!(
                            "evidence_ledger.update_drops_rows",
                            "an update only adds observations; use replace to correct or drop them"
                        ));
                    }
                }
                _ => {
                    return Err(overlap_unknown!(
                        "evidence_ledger.overlap_unknown",
                        "an update cannot be verified as monotone while either observation set is unknown"
                    )
                    .with_sources([target]));
                }
            }
        }
        if self.source_index.contains_key(&canonical.id) {
            return Err(double_counted!(
                "evidence_ledger.duplicate_source",
                "source {} is already in the ledger",
                canonical.id
            )
            .with_sources([canonical.id]));
        }
        self.check_new_source_bounds(&canonical)?;
        let overlaps = self.overlaps_with_existing(&canonical, Some(target));
        Ok(Plan::Retire { retire, target, reason: None, replacement: Some((canonical, overlaps)) })
    }

    fn check_new_source_bounds(&self, source: &CanonicalSource) -> Result<(), LedgerError> {
        let rows = source.effective.known().map_or(0, <[ObservationId]>::len);
        if self.total_observations + rows > MAX_TOTAL_OBSERVATIONS
            || self.sources.len() >= MAX_SOURCES
        {
            return Err(limit!(
                "evidence_ledger.limit_exceeded",
                "the ledger's source or observation bound would be exceeded"
            ));
        }
        Ok(())
    }

    fn operation_identity(&self, plan: &Plan) -> OperationId {
        let mut writer = CanonicalWriter::new("antecedent.evidence_ledger.operation");
        writer.word(LEDGER_IDENTITY_VERSION);
        digest_word(&mut writer, self.head.digest());
        match plan {
            Plan::Add { source, .. } => {
                writer.text("add");
                digest_word(&mut writer, source.id.digest());
            }
            Plan::Retire { retire, target, reason, replacement } => {
                writer.text(match retire {
                    Retire::Update => "update",
                    Retire::Replace => "replace",
                    Retire::Supersede => "supersede",
                    Retire::Withdraw => "withdraw",
                });
                digest_word(&mut writer, target.digest());
                writer.opt_text(reason.as_deref());
                match replacement {
                    Some((source, _)) => {
                        writer.word(1);
                        digest_word(&mut writer, source.id.digest());
                    }
                    None => {
                        writer.word(0);
                    }
                }
            }
            Plan::Revalidate => {
                writer.text("revalidate_downstream");
            }
            Plan::Combine(checked) => {
                writer.text("combine");
                digest_word(&mut writer, checked.id.digest());
            }
        }
        OperationId::from_digest(finish(&writer))
    }

    fn insert_source(&mut self, source: CanonicalSource, added_by: OperationId) {
        let rows = source.effective.known().map_or(0, <[ObservationId]>::len);
        self.total_observations += rows;
        self.source_index.insert(source.id, self.sources.len());
        self.sources.push(EvidenceSource {
            id: source.id,
            declaration: source.declaration,
            effective: source.effective,
            status: SourceStatus::Active,
            added_by,
        });
    }

    /// Apply one operation. On refusal the ledger is unchanged.
    ///
    /// # Errors
    /// A typed [`LedgerError`] naming the rule and what would license the request.
    pub fn apply(&mut self, operation: LedgerOperation) -> Result<&AppliedOperation, LedgerError> {
        let plan = self.plan(&operation)?;
        let id = self.operation_identity(&plan);
        let mut effect = OperationEffect::default();
        match plan {
            Plan::Add { source, overlaps } => {
                effect.created_source = Some(source.id);
                effect.detected_overlaps = overlaps;
                self.insert_source(source, id);
            }
            Plan::Retire { retire, target, reason, replacement } => {
                let successor = replacement.as_ref().map(|(source, _)| source.id);
                let index = self.source_index[&target];
                self.sources[index].status = match (retire, successor) {
                    (Retire::Withdraw, _) | (_, None) => {
                        SourceStatus::Withdrawn { reason: reason.unwrap_or_default(), by: id }
                    }
                    (Retire::Update, Some(by)) => SourceStatus::Updated { by, operation: id },
                    (Retire::Replace, Some(by)) => SourceStatus::Replaced { by, operation: id },
                    (Retire::Supersede, Some(by)) => SourceStatus::Superseded { by, operation: id },
                };
                effect.retired_source = Some(target);
                if let Some((source, overlaps)) = replacement {
                    effect.created_source = Some(source.id);
                    effect.detected_overlaps = overlaps;
                    self.insert_source(source, id);
                }
                effect.revalidation = self.reclassify(id);
            }
            Plan::Revalidate => {
                effect.revalidation = self.reclassify(id);
            }
            Plan::Combine(checked) => {
                if let Some(replaced) = checked.replaces {
                    let index = self.result_index[&replaced];
                    self.results[index].status =
                        ResultStatus::Replaced { by: checked.id, operation: id };
                    effect.replaced_result = Some(replaced);
                }
                effect.created_result = Some(checked.id);
                self.result_index.insert(checked.id, self.results.len());
                self.results.push(ResultRecord {
                    id: checked.id,
                    label: checked.label,
                    operation: checked.operation,
                    mass: checked.mass,
                    claim_class: checked.claim_class,
                    observations_digest: checked.observations_digest,
                    transport_identities: checked.transport_identities,
                    data_reuse: checked.data_reuse,
                    dependence_declared: checked.dependence_declared,
                    replaces: checked.replaces,
                    created_by: id,
                    status: ResultStatus::Valid,
                });
            }
        }
        self.head = id;
        self.log.push(AppliedOperation { id, operation, effect });
        Ok(&self.log[self.log.len() - 1])
    }

    // -- conveniences -------------------------------------------------------------------

    /// Add a source and return its identity.
    ///
    /// # Errors
    /// A typed refusal.
    ///
    /// # Panics
    /// Never in practice: an applied add always records the source it created.
    pub fn add_source(&mut self, source: SourceDeclaration) -> Result<SourceId, LedgerError> {
        let applied = self.apply(LedgerOperation::Add { source })?;
        Ok(applied.effect.created_source.expect("an applied add records the source it created"))
    }

    /// Update a source with grown content.
    ///
    /// # Errors
    /// A typed refusal.
    pub fn update_source(
        &mut self,
        target: SourceId,
        replacement: SourceDeclaration,
    ) -> Result<&AppliedOperation, LedgerError> {
        self.apply(LedgerOperation::Update { target, replacement })
    }

    /// Replace a source with corrected content.
    ///
    /// # Errors
    /// A typed refusal.
    pub fn replace_source(
        &mut self,
        target: SourceId,
        replacement: SourceDeclaration,
    ) -> Result<&AppliedOperation, LedgerError> {
        self.apply(LedgerOperation::Replace { target, replacement })
    }

    /// Supersede a model or claim.
    ///
    /// # Errors
    /// A typed refusal.
    pub fn supersede_source(
        &mut self,
        target: SourceId,
        replacement: SourceDeclaration,
    ) -> Result<&AppliedOperation, LedgerError> {
        self.apply(LedgerOperation::Supersede { target, replacement })
    }

    /// Withdraw a source.
    ///
    /// # Errors
    /// A typed refusal.
    pub fn withdraw_source(
        &mut self,
        target: SourceId,
        reason: impl Into<String>,
    ) -> Result<&AppliedOperation, LedgerError> {
        self.apply(LedgerOperation::Withdraw { target, reason: reason.into() })
    }

    /// Re-classify downstream results and record it.
    ///
    /// # Errors
    /// Only when the operation log is full.
    pub fn revalidate_downstream(&mut self) -> Result<Vec<ResultRevalidation>, LedgerError> {
        Ok(self.apply(LedgerOperation::RevalidateDownstream)?.effect.revalidation.clone())
    }

    /// Register a combination.
    ///
    /// # Errors
    /// A typed refusal: double counting, unknown overlap, a retired source.
    ///
    /// # Panics
    /// Never in practice: an applied combination always records the result it created.
    pub fn combine(&mut self, request: CombinationRequest) -> Result<ResultId, LedgerError> {
        let applied = self.apply(LedgerOperation::Combine { request })?;
        Ok(applied.effect.created_result.expect("an applied combine records the result it created"))
    }
}
