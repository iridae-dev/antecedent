//! Dependence between evidence sources that share rows, clusters, models,
//! parameters, priors, disturbances, or periods.
//!
//! Each [`DependenceSource`] carries an identity and the factor identities it
//! uses. [`DependenceContract::assess`] reports the relation those sets
//! support: independent when every source has declared its factors and no pair
//! shares one, dependent when a factor identity is shared, and unknown when a
//! source's overlap is undeclared and no shared factor is known. A known
//! shared factor stays dependence when another source's overlap is unknown.
//!
//! An independence claim is [`DependenceContract::try_claim_independent`].
//! When the factors do not support it, the error names the shared factor, or
//! names that overlap is unknown, and [`DependenceError::next_contract`] is
//! the dependent or unknown contract. A covariance combination of a dependent
//! contract requests the cross terms of the shared factors. Disjoint source
//! identities do not establish independence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{collections::BTreeSet, error::Error, fmt, sync::Arc};

const BLANK_CODE: &str = antecedent_core::reason_code!("invalid_argument");
const SHARED_CODE: &str = antecedent_core::reason_code!("evidence_double_counted");
const UNKNOWN_CODE: &str = antecedent_core::reason_code!("evidence_overlap_unknown");

const BLANK_SOURCE_DETAIL: &str = "dependence.blank_source_id";
const BLANK_FACTOR_DETAIL: &str = "dependence.blank_factor_id";
const SHARED_DETAIL: &str = "dependence.shared_factor";
const UNKNOWN_DETAIL: &str = "dependence.overlap_unknown";

const BLANK_SOURCE_MESSAGE: &str = "a source id is blank";
const BLANK_FACTOR_MESSAGE: &str = "a factor id is blank";

/// A kind of factor that two evidence sources can share.
///
/// Declaration order is the order shared factors are named.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum SharedFactorKind {
    /// A row, unit, or draw used as evidence.
    Row,
    /// A cluster the rows belong to.
    Cluster,
    /// A fitted or external model.
    Model,
    /// A parameter the evidence informs or conditions on.
    Parameter,
    /// A prior, including a prior built from other evidence.
    Prior,
    /// A shared disturbance or shock.
    Disturbance,
    /// A time period the evidence covers.
    Period,
}

impl SharedFactorKind {
    /// Every kind, in canonical order.
    pub const ALL: [Self; 7] = [
        Self::Row,
        Self::Cluster,
        Self::Model,
        Self::Parameter,
        Self::Prior,
        Self::Disturbance,
        Self::Period,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Row => "row",
            Self::Cluster => "cluster",
            Self::Model => "model",
            Self::Parameter => "parameter",
            Self::Prior => "prior",
            Self::Disturbance => "disturbance",
            Self::Period => "period",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }
}

/// Identity of one shared factor. Blank text is refused and surrounding whitespace is
/// not part of the identity, so `row-1` and `row-1 ` name the same factor.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SharedFactorId(Arc<str>);

impl SharedFactorId {
    fn try_new(text: &str) -> Result<Self, DependenceError> {
        checked_text(text.trim(), BLANK_FACTOR_DETAIL, BLANK_FACTOR_MESSAGE).map(Self)
    }

    /// Text of this identity.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SharedFactorId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl AsRef<str> for SharedFactorId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// One factor identity of a given kind.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SharedFactor {
    kind: SharedFactorKind,
    id: SharedFactorId,
}

impl SharedFactor {
    /// Kind of factor.
    #[must_use]
    pub const fn kind(&self) -> SharedFactorKind {
        self.kind
    }

    /// Identity within that kind.
    #[must_use]
    pub const fn id(&self) -> &SharedFactorId {
        &self.id
    }
}

impl fmt::Display for SharedFactor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "shared {} `{}`", self.kind.as_str(), self.id.as_str())
    }
}

/// The factors one evidence source uses, grouped by kind.
///
/// An empty set of a kind means the source declared that it uses none of that
/// kind. It is distinct from an undeclared overlap on [`DependenceSource`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SharedFactorSets {
    rows: BTreeSet<SharedFactorId>,
    clusters: BTreeSet<SharedFactorId>,
    models: BTreeSet<SharedFactorId>,
    parameters: BTreeSet<SharedFactorId>,
    priors: BTreeSet<SharedFactorId>,
    disturbances: BTreeSet<SharedFactorId>,
    periods: BTreeSet<SharedFactorId>,
}

impl SharedFactorSets {
    /// No factors of any kind.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            rows: BTreeSet::new(),
            clusters: BTreeSet::new(),
            models: BTreeSet::new(),
            parameters: BTreeSet::new(),
            priors: BTreeSet::new(),
            disturbances: BTreeSet::new(),
            periods: BTreeSet::new(),
        }
    }

    /// Record `id` under `kind`. A repeated id is stored once.
    ///
    /// A blank id leaves the sets unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `id` is empty or only whitespace.
    #[must_use = "a blank factor id is refused"]
    pub fn try_insert(
        &mut self,
        kind: SharedFactorKind,
        id: impl AsRef<str>,
    ) -> Result<(), DependenceError> {
        let id = SharedFactorId::try_new(id.as_ref())?;
        self.ids_mut(kind).insert(id);
        Ok(())
    }

    /// Identities recorded for `kind`, in order.
    #[must_use]
    pub fn ids(&self, kind: SharedFactorKind) -> &BTreeSet<SharedFactorId> {
        match kind {
            SharedFactorKind::Row => &self.rows,
            SharedFactorKind::Cluster => &self.clusters,
            SharedFactorKind::Model => &self.models,
            SharedFactorKind::Parameter => &self.parameters,
            SharedFactorKind::Prior => &self.priors,
            SharedFactorKind::Disturbance => &self.disturbances,
            SharedFactorKind::Period => &self.periods,
        }
    }

    /// Whether every kind is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
            && self.clusters.is_empty()
            && self.models.is_empty()
            && self.parameters.is_empty()
            && self.priors.is_empty()
            && self.disturbances.is_empty()
            && self.periods.is_empty()
    }

    fn ids_mut(&mut self, kind: SharedFactorKind) -> &mut BTreeSet<SharedFactorId> {
        match kind {
            SharedFactorKind::Row => &mut self.rows,
            SharedFactorKind::Cluster => &mut self.clusters,
            SharedFactorKind::Model => &mut self.models,
            SharedFactorKind::Parameter => &mut self.parameters,
            SharedFactorKind::Prior => &mut self.priors,
            SharedFactorKind::Disturbance => &mut self.disturbances,
            SharedFactorKind::Period => &mut self.periods,
        }
    }

    fn contains(&self, factor: &SharedFactor) -> bool {
        self.ids(factor.kind()).contains(factor.id())
    }

    fn accumulate_overlap(&self, other: &Self, into: &mut BTreeSet<SharedFactor>) {
        for kind in SharedFactorKind::ALL {
            for id in self.ids(kind).intersection(other.ids(kind)) {
                into.insert(SharedFactor { kind, id: id.clone() });
            }
        }
    }
}

/// Whether a source listed every factor it uses.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum FactorDisclosure {
    /// Every factor the source uses is listed. Empty sets mean it uses none.
    Complete,
    /// The list may be partial. Absence of an intersection is unknown overlap.
    UnknownOverlap,
}

/// Identity of one evidence source in a dependence contract. Blank text is refused.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DependenceSourceId(Arc<str>);

impl DependenceSourceId {
    fn try_new(text: &str) -> Result<Self, DependenceError> {
        checked_text(text, BLANK_SOURCE_DETAIL, BLANK_SOURCE_MESSAGE).map(Self)
    }

    /// Text of this identity.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DependenceSourceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl AsRef<str> for DependenceSourceId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// One evidence source: its identity and the shared factors it uses.
///
/// The source identity does not decide dependence. Two sources with different
/// identities are dependent when they list the same factor, and unknown when
/// either overlap is undeclared.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependenceSource {
    id: DependenceSourceId,
    factors: SharedFactorSets,
    disclosure: FactorDisclosure,
}

impl DependenceSource {
    /// A source whose factor lists are complete.
    ///
    /// # Errors
    ///
    /// Returns an error when `id` is empty or only whitespace.
    #[must_use = "a blank source id is refused"]
    pub fn try_declared(
        id: impl AsRef<str>,
        factors: SharedFactorSets,
    ) -> Result<Self, DependenceError> {
        Ok(Self {
            id: DependenceSourceId::try_new(id.as_ref())?,
            factors,
            disclosure: FactorDisclosure::Complete,
        })
    }

    /// A source whose overlap is unknown.
    ///
    /// `known` is a lower bound. Factors it shares with another source are
    /// dependence. Factors it does not list are not evidence of independence.
    ///
    /// # Errors
    ///
    /// Returns an error when `id` is empty or only whitespace.
    #[must_use = "a blank source id is refused"]
    pub fn try_with_unknown_overlap(
        id: impl AsRef<str>,
        known: SharedFactorSets,
    ) -> Result<Self, DependenceError> {
        Ok(Self {
            id: DependenceSourceId::try_new(id.as_ref())?,
            factors: known,
            disclosure: FactorDisclosure::UnknownOverlap,
        })
    }

    /// Source identity.
    #[must_use]
    pub const fn id(&self) -> &DependenceSourceId {
        &self.id
    }

    /// Factors this source is known to use.
    #[must_use]
    pub const fn factors(&self) -> &SharedFactorSets {
        &self.factors
    }

    /// Whether the factor lists are complete.
    #[must_use]
    pub const fn disclosure(&self) -> FactorDisclosure {
        self.disclosure
    }
}

/// The relation an assessed contract reports.
///
/// The factors themselves live on [`DependenceContract`]. This status is only
/// the verdict.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum DependenceStatus {
    /// Every source declared its factors, and no pair shares one.
    Independent,
    /// At least one factor identity appears in two sources. Sources whose overlap is
    /// undeclared stay on the contract, so the known factors are a lower bound.
    Dependent,
    /// No shared factor is known, and at least one source left its overlap undeclared.
    Unknown,
}

/// The dependence contract supported by a set of evidence sources.
///
/// Produced by [`Self::assess`] or by a successful [`Self::try_claim_independent`].
/// An unsupported independence claim does not produce an independent contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DependenceContract {
    status: DependenceStatus,
    factors: Box<[SharedFactor]>,
    unknown_sources: Box<[DependenceSourceId]>,
}

impl DependenceContract {
    /// The dependence the sources support.
    ///
    /// A shared factor is dependence, including when another source left its
    /// overlap unknown. Unknown overlap with no known shared factor is
    /// [`DependenceStatus::Unknown`]. Independence requires complete disclosure
    /// and pairwise disjoint factor sets. Source identities are not the criterion.
    /// Fewer than two sources are independent when every disclosure is complete.
    #[must_use]
    pub fn assess(sources: &[DependenceSource]) -> Self {
        let mut shared = BTreeSet::new();
        for (index, left) in sources.iter().enumerate() {
            for right in sources.iter().skip(index + 1) {
                left.factors.accumulate_overlap(&right.factors, &mut shared);
            }
        }
        let mut unknown = BTreeSet::new();
        for source in sources {
            if matches!(source.disclosure, FactorDisclosure::UnknownOverlap) {
                unknown.insert(source.id.clone());
            }
        }
        if !shared.is_empty() {
            let factors = shared.into_iter().collect::<Box<[SharedFactor]>>();
            return Self {
                status: DependenceStatus::Dependent,
                factors,
                unknown_sources: unknown.into_iter().collect(),
            };
        }
        if unknown.is_empty() {
            Self {
                status: DependenceStatus::Independent,
                factors: empty_slice(),
                unknown_sources: empty_slice(),
            }
        } else {
            Self {
                status: DependenceStatus::Unknown,
                factors: empty_slice(),
                unknown_sources: unknown.into_iter().collect(),
            }
        }
    }

    /// Claim the sources are independent.
    ///
    /// Succeeds only when [`Self::assess`] is [`DependenceStatus::Independent`].
    /// Otherwise the error carries that assessment as the next contract.
    ///
    /// # Errors
    ///
    /// Returns an error naming each shared factor, or naming that overlap is
    /// unknown. [`DependenceError::next_contract`] is the dependent or unknown
    /// contract, and that contract is never independent.
    #[must_use = "an independence claim the factors do not support is a failed attempt"]
    pub fn try_claim_independent(sources: &[DependenceSource]) -> Result<Self, DependenceError> {
        let contract = Self::assess(sources);
        match contract.status {
            DependenceStatus::Independent => Ok(contract),
            DependenceStatus::Dependent => Err(DependenceError::shared(sources, contract)),
            DependenceStatus::Unknown => Err(DependenceError::unknown_contract(contract)),
        }
    }

    /// The covariance terms a combination of this contract has to request.
    ///
    /// A dependent contract requests the cross terms of its shared factors; when
    /// other sources left their overlap undeclared the combination is marked
    /// incomplete ([`CovarianceCombination::has_unknown_overlap`]) because those
    /// sources may add cross terms nobody listed. An independent contract is
    /// diagonal. Unknown overlap has no diagonal combination.
    ///
    /// # Errors
    ///
    /// Returns an error when the status is [`DependenceStatus::Unknown`]. The
    /// error names that overlap is unknown, and its next contract is this
    /// unknown contract.
    #[must_use = "unknown overlap does not license a diagonal covariance"]
    pub fn covariance_combination(&self) -> Result<CovarianceCombination, DependenceError> {
        match self.status {
            DependenceStatus::Independent => Ok(CovarianceCombination::diagonal()),
            DependenceStatus::Dependent => Ok(CovarianceCombination::cross_terms(
                self.factors.clone(),
                self.unknown_sources.clone(),
            )),
            DependenceStatus::Unknown => Err(DependenceError::unknown_contract(self.clone())),
        }
    }

    /// Verdict of this contract.
    #[must_use]
    pub const fn status(&self) -> DependenceStatus {
        self.status
    }

    /// Whether the factors support independence.
    #[must_use]
    pub const fn is_independent(&self) -> bool {
        matches!(self.status, DependenceStatus::Independent)
    }

    /// Whether a shared factor was found.
    #[must_use]
    pub const fn is_dependent(&self) -> bool {
        matches!(self.status, DependenceStatus::Dependent)
    }

    /// Whether overlap is unknown and no shared factor is known.
    #[must_use]
    pub const fn is_unknown(&self) -> bool {
        matches!(self.status, DependenceStatus::Unknown)
    }

    /// Shared factors, in canonical order. Empty unless the status is dependent.
    #[must_use]
    pub fn shared_factors(&self) -> &[SharedFactor] {
        &self.factors
    }

    /// Sources whose overlap is undeclared, in identity order.
    ///
    /// Empty when the status is independent. Non-empty when the status is unknown, and
    /// also when it is dependent but another source left its overlap undeclared.
    #[must_use]
    pub fn unknown_sources(&self) -> &[DependenceSourceId] {
        &self.unknown_sources
    }
}

/// A refused dependence request.
///
/// An unsupported independence claim stores the contract the factors do
/// support. A blank identity has no contract.
#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use = "the next legal result is the dependent or unknown contract"]
pub struct DependenceError {
    code: &'static str,
    detail: &'static str,
    message: String,
    next: Option<DependenceContract>,
}

impl DependenceError {
    /// Registered reason code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail, `dependence.<rule>`.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }

    /// What was refused, including the shared factor or that overlap is unknown.
    #[must_use]
    pub fn message(&self) -> &str {
        self.message.as_str()
    }

    /// The contract the factors support, when this error is a failed independence claim.
    ///
    /// Present for a shared factor and for unknown overlap. That contract is
    /// dependent or unknown. Absent for a blank identity and for an incomplete
    /// covariance combination.
    #[must_use]
    pub const fn next_contract(&self) -> Option<&DependenceContract> {
        self.next.as_ref()
    }

    /// Take the supported contract. See [`Self::next_contract`].
    #[must_use]
    pub fn into_contract(self) -> Option<DependenceContract> {
        self.next
    }

    /// The refusal as `(code, detail, explanation)`.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, &'static str, &str) {
        (self.code, self.detail, self.message.as_str())
    }

    fn blank(detail: &'static str, message: &'static str) -> Self {
        Self { code: BLANK_CODE, detail, message: message.to_owned(), next: None }
    }

    fn shared(sources: &[DependenceSource], contract: DependenceContract) -> Self {
        let message = shared_message(sources, contract.shared_factors());
        Self { code: SHARED_CODE, detail: SHARED_DETAIL, message, next: Some(contract) }
    }

    fn unknown_contract(contract: DependenceContract) -> Self {
        let message = unknown_message(contract.unknown_sources());
        Self { code: UNKNOWN_CODE, detail: UNKNOWN_DETAIL, message, next: Some(contract) }
    }
}

impl fmt::Display for DependenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.detail, self.message)
    }
}

impl Error for DependenceError {}

/// Which covariance terms a combination requests.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CovarianceTerms {
    /// No shared factor. Off-diagonal blocks are zero.
    Diagonal,
    /// A shared factor exists. The combination requests its cross terms.
    CrossTerms,
}

/// A covariance combination licensed by a dependence contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CovarianceCombination {
    terms: CovarianceTerms,
    factors: Box<[SharedFactor]>,
    unknown_overlap: Box<[DependenceSourceId]>,
}

impl CovarianceCombination {
    /// Diagonal or cross terms.
    #[must_use]
    pub const fn terms(&self) -> CovarianceTerms {
        self.terms
    }

    /// Whether the combination has to include cross terms.
    #[must_use]
    pub const fn requires_cross_terms(&self) -> bool {
        matches!(self.terms, CovarianceTerms::CrossTerms)
    }

    /// Shared factors whose cross terms are requested. Empty when diagonal.
    #[must_use]
    pub fn factors(&self) -> &[SharedFactor] {
        match self.terms {
            CovarianceTerms::Diagonal => &[],
            CovarianceTerms::CrossTerms => &self.factors,
        }
    }

    /// Sources whose overlap is undeclared: the cross terms of [`Self::factors`] are
    /// then a lower bound and the covariance is incomplete. Empty when complete.
    #[must_use]
    pub fn unknown_overlap(&self) -> &[DependenceSourceId] {
        &self.unknown_overlap
    }

    /// Whether the covariance misses cross terms with sources of undeclared overlap.
    #[must_use]
    pub fn has_unknown_overlap(&self) -> bool {
        !self.unknown_overlap.is_empty()
    }

    /// The combination, only when it is complete.
    ///
    /// # Errors
    ///
    /// Returns an `evidence_overlap_unknown` error naming the sources with unknown
    /// overlap when [`Self::has_unknown_overlap`]. A consumer that needs the full
    /// covariance calls this before using [`Self::factors`].
    pub fn require_complete(&self) -> Result<&Self, DependenceError> {
        if self.has_unknown_overlap() {
            let message = format!(
                "covariance is incomplete: {}",
                unknown_message(&self.unknown_overlap).replacen(
                    "claimed independence is not supported: ",
                    "",
                    1
                )
            );
            Err(DependenceError { code: UNKNOWN_CODE, detail: UNKNOWN_DETAIL, message, next: None })
        } else {
            Ok(self)
        }
    }

    #[must_use]
    fn diagonal() -> Self {
        Self {
            terms: CovarianceTerms::Diagonal,
            factors: empty_slice(),
            unknown_overlap: empty_slice(),
        }
    }

    #[must_use]
    fn cross_terms(
        factors: Box<[SharedFactor]>,
        unknown_overlap: Box<[DependenceSourceId]>,
    ) -> Self {
        Self { terms: CovarianceTerms::CrossTerms, factors, unknown_overlap }
    }
}

fn checked_text(
    text: &str,
    detail: &'static str,
    message: &'static str,
) -> Result<Arc<str>, DependenceError> {
    if antecedent_core::is_blank(text) {
        Err(DependenceError::blank(detail, message))
    } else {
        Ok(Arc::from(text))
    }
}

fn empty_slice<T>() -> Box<[T]> {
    Vec::new().into_boxed_slice()
}

fn shared_message(sources: &[DependenceSource], factors: &[SharedFactor]) -> String {
    let mut message = String::from("claimed independence is not supported: ");
    for (index, factor) in factors.iter().enumerate() {
        if index > 0 {
            message.push_str(", ");
        }
        message.push_str(&factor.to_string());
        if let Some((left, right)) = witness(sources, factor) {
            message.push_str(" between `");
            message.push_str(left.as_str());
            message.push_str("` and `");
            message.push_str(right.as_str());
            message.push('`');
        }
    }
    message
}

fn unknown_message(sources: &[DependenceSourceId]) -> String {
    let mut message = String::from("claimed independence is not supported: overlap is unknown");
    if sources.is_empty() {
        return message;
    }
    message.push_str(" for ");
    for (index, id) in sources.iter().enumerate() {
        if index > 0 {
            message.push_str(", ");
        }
        message.push('`');
        message.push_str(id.as_str());
        message.push('`');
    }
    message
}

fn witness<'a>(
    sources: &'a [DependenceSource],
    factor: &SharedFactor,
) -> Option<(&'a DependenceSourceId, &'a DependenceSourceId)> {
    for (index, left) in sources.iter().enumerate() {
        if left.factors.contains(factor) {
            for right in sources.iter().skip(index + 1) {
                if right.factors.contains(factor) {
                    return Some((&left.id, &right.id));
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        CovarianceTerms, DependenceContract, DependenceSource, DependenceStatus, SharedFactorKind,
        SharedFactorSets,
    };

    const KINDS: [SharedFactorKind; 7] = [
        SharedFactorKind::Row,
        SharedFactorKind::Cluster,
        SharedFactorKind::Model,
        SharedFactorKind::Parameter,
        SharedFactorKind::Prior,
        SharedFactorKind::Disturbance,
        SharedFactorKind::Period,
    ];

    fn sets(entries: &[(SharedFactorKind, &str)]) -> SharedFactorSets {
        let mut factors = SharedFactorSets::empty();
        for &(kind, id) in entries {
            factors.try_insert(kind, id).expect("factor id");
        }
        factors
    }

    fn declared(id: &str, entries: &[(SharedFactorKind, &str)]) -> DependenceSource {
        DependenceSource::try_declared(id, sets(entries)).expect("source id")
    }

    fn unknown(id: &str, entries: &[(SharedFactorKind, &str)]) -> DependenceSource {
        DependenceSource::try_with_unknown_overlap(id, sets(entries)).expect("source id")
    }

    #[test]
    fn duplicate_sources_are_dependent() {
        let original = declared(
            "original",
            &[(SharedFactorKind::Row, "row-1"), (SharedFactorKind::Row, "row-2")],
        );
        let copy =
            declared("copy", &[(SharedFactorKind::Row, "row-1"), (SharedFactorKind::Row, "row-2")]);
        assert_ne!(original.id().as_str(), copy.id().as_str());
        let contract = DependenceContract::assess(&[original.clone(), copy.clone()]);
        assert!(contract.is_dependent());
        assert!(!contract.is_independent());
        assert_eq!(contract.shared_factors().len(), 2);
        assert_eq!(contract.shared_factors().first().expect("row").id().as_str(), "row-1");
        assert_eq!(DependenceContract::assess(&[copy.clone(), original.clone()]), contract);

        let partial = declared("partial", &[(SharedFactorKind::Row, "row-2")]);
        let overlap = DependenceContract::assess(&[original.clone(), partial]);
        assert!(overlap.is_dependent());
        assert_eq!(overlap.shared_factors().len(), 1);
        assert_eq!(overlap.shared_factors().first().expect("row").id().as_str(), "row-2");

        let err = DependenceContract::try_claim_independent(&[original, copy])
            .expect_err("duplicate rows");
        assert_eq!(err.code(), "evidence_double_counted");
        assert!(err.message().contains("row"));
        assert!(err.message().contains("row-1"));
        let next = err.next_contract().expect("dependent contract");
        assert_eq!(next, &contract);
        assert!(next.is_dependent());
        assert!(!next.is_independent());
        let combination = next.covariance_combination().expect("cross terms");
        assert!(combination.requires_cross_terms());
        assert_eq!(combination.terms(), CovarianceTerms::CrossTerms);
    }

    #[test]
    fn prior_or_external_model_reuse_is_dependent_when_source_ids_differ() {
        let study = declared(
            "study",
            &[(SharedFactorKind::Row, "row-1"), (SharedFactorKind::Prior, "prior:power")],
        );
        let posterior = declared(
            "posterior",
            &[(SharedFactorKind::Row, "row-2"), (SharedFactorKind::Prior, "prior:power")],
        );
        assert_ne!(study.id().as_str(), posterior.id().as_str());
        let through_prior = DependenceContract::assess(&[study, posterior]);
        assert!(through_prior.is_dependent());
        assert!(through_prior.shared_factors().iter().all(|factor| {
            factor.kind() == SharedFactorKind::Prior && factor.id().as_str() == "prior:power"
        }));

        let fit = declared(
            "fit",
            &[(SharedFactorKind::Row, "row-3"), (SharedFactorKind::Model, "model:external")],
        );
        let claim = declared(
            "external-claim",
            &[(SharedFactorKind::Row, "row-4"), (SharedFactorKind::Model, "model:external")],
        );
        assert_ne!(fit.id().as_str(), claim.id().as_str());
        let through_model = DependenceContract::assess(&[fit.clone(), claim.clone()]);
        assert!(through_model.is_dependent());
        assert!(through_model.shared_factors().iter().all(|factor| {
            factor.kind() == SharedFactorKind::Model && factor.id().as_str() == "model:external"
        }));
        let err =
            DependenceContract::try_claim_independent(&[fit, claim]).expect_err("shared model");
        assert!(err.message().contains("model"));
        assert!(err.message().contains("model:external"));
        let next = err.next_contract().expect("dependent contract");
        assert_eq!(next, &through_model);
        assert!(next.is_dependent());
        assert!(!next.is_independent());
    }

    #[test]
    fn unknown_overlap_is_not_independence() {
        let known = declared("known", &[(SharedFactorKind::Row, "row-1")]);
        let opaque = unknown("opaque", &[]);
        assert_ne!(known.id().as_str(), opaque.id().as_str());
        let contract = DependenceContract::assess(&[known.clone(), opaque.clone()]);
        assert!(contract.is_unknown());
        assert!(!contract.is_independent());
        assert!(!contract.is_dependent());
        assert!(contract.shared_factors().is_empty());
        assert_eq!(contract.unknown_sources().first().expect("opaque").as_str(), "opaque",);

        let err = DependenceContract::try_claim_independent(&[known.clone(), opaque])
            .expect_err("unknown overlap");
        assert_eq!(err.code(), "evidence_overlap_unknown");
        assert_eq!(err.detail(), "dependence.overlap_unknown");
        assert!(err.message().contains("overlap is unknown"));
        let next = err.next_contract().expect("unknown contract");
        assert_eq!(next, &contract);
        assert!(next.is_unknown());
        assert!(!next.is_independent());
        let covariance = next.covariance_combination().expect_err("diagonal is refused");
        let covariance_next = covariance.next_contract().expect("unknown contract");
        assert!(covariance_next.is_unknown());
        assert!(!covariance_next.is_independent());

        let alone = DependenceContract::assess(&[unknown("alone", &[])]);
        assert!(alone.is_unknown());
        assert!(!alone.is_independent());

        let partial = unknown("partial", &[(SharedFactorKind::Row, "row-1")]);
        let shared = DependenceContract::assess(&[known, partial]);
        assert!(shared.is_dependent());
        assert!(!shared.is_unknown());
        assert!(!shared.is_independent());
        // The undeclared source stays on the dependent contract and marks the covariance
        // incomplete.
        assert_eq!(
            shared
                .unknown_sources()
                .iter()
                .map(super::DependenceSourceId::as_str)
                .collect::<Vec<_>>(),
            ["partial"]
        );
        let combination = shared.covariance_combination().expect("cross terms");
        assert!(combination.requires_cross_terms());
        assert!(combination.has_unknown_overlap());
        let refused = combination.require_complete().expect_err("incomplete covariance");
        assert_eq!(refused.detail(), "dependence.overlap_unknown");
        assert!(refused.message().contains("incomplete"));
        assert!(refused.message().contains("overlap is unknown for `partial`"));
        let complete = DependenceContract::assess(&[
            declared("a", &[(SharedFactorKind::Row, "row-1")]),
            declared("b", &[(SharedFactorKind::Row, "row-1")]),
        ])
        .covariance_combination()
        .expect("cross terms");
        assert!(!complete.has_unknown_overlap());
        assert!(complete.require_complete().is_ok());
    }

    #[test]
    fn covariance_combination_requests_cross_terms_for_a_shared_factor() {
        let left = declared("left", &[(SharedFactorKind::Cluster, "site-1")]);
        let middle = declared(
            "middle",
            &[(SharedFactorKind::Cluster, "site-1"), (SharedFactorKind::Period, "2020")],
        );
        let right = declared("right", &[(SharedFactorKind::Period, "2020")]);
        let contract = DependenceContract::assess(&[left, middle, right]);
        assert!(contract.is_dependent());
        let combination = contract.covariance_combination().expect("cross terms");
        assert!(combination.requires_cross_terms());
        assert_eq!(combination.terms(), CovarianceTerms::CrossTerms);
        assert!(combination.factors().iter().any(|factor| {
            factor.kind() == SharedFactorKind::Cluster && factor.id().as_str() == "site-1"
        }));
        assert!(combination.factors().iter().any(|factor| {
            factor.kind() == SharedFactorKind::Period && factor.id().as_str() == "2020"
        }));

        let separate_a = declared("a", &[(SharedFactorKind::Cluster, "site-a")]);
        let separate_b = declared("b", &[(SharedFactorKind::Cluster, "site-b")]);
        let independent = DependenceContract::assess(&[separate_a, separate_b]);
        assert!(independent.is_independent());
        let diagonal = independent.covariance_combination().expect("diagonal");
        assert!(!diagonal.requires_cross_terms());
        assert_eq!(diagonal.terms(), CovarianceTerms::Diagonal);
        assert!(diagonal.factors().is_empty());
    }

    #[test]
    fn disjoint_source_ids_do_not_establish_independence() {
        let left = declared(
            "source-a",
            &[(SharedFactorKind::Parameter, "theta"), (SharedFactorKind::Disturbance, "shock")],
        );
        let right = declared("source-b", &[(SharedFactorKind::Disturbance, "shock")]);
        assert_ne!(left.id().as_str(), right.id().as_str());
        let shared = DependenceContract::assess(&[left.clone(), right.clone()]);
        assert!(shared.is_dependent());
        assert!(!shared.is_independent());
        let err = DependenceContract::try_claim_independent(&[left, right])
            .expect_err("shared disturbance");
        assert!(err.message().contains("disturbance"));
        assert!(err.message().contains("shock"));
        let next = err.into_contract().expect("dependent contract");
        assert_eq!(next, shared);
        assert!(next.is_dependent());
        assert!(!next.is_independent());

        let opaque = unknown("source-c", &[]);
        let other = declared("source-d", &[(SharedFactorKind::Row, "row-9")]);
        assert_ne!(opaque.id().as_str(), other.id().as_str());
        let undeclared = DependenceContract::assess(&[opaque, other]);
        assert!(undeclared.is_unknown());
        assert!(!undeclared.is_independent());

        let one = declared("source-e", &[(SharedFactorKind::Parameter, "alpha")]);
        let two = declared("source-f", &[(SharedFactorKind::Parameter, "beta")]);
        assert_ne!(one.id().as_str(), two.id().as_str());
        let supported =
            DependenceContract::try_claim_independent(&[one, two]).expect("disjoint factors");
        assert!(supported.is_independent());
        assert_eq!(supported.status(), DependenceStatus::Independent);
        let diagonal = supported.covariance_combination().expect("diagonal");
        assert!(!diagonal.requires_cross_terms());
    }

    #[test]
    fn unsupported_independence_claim_is_a_failed_attempt() {
        let left = declared("left", &[(SharedFactorKind::Parameter, "theta")]);
        let right = declared("right", &[(SharedFactorKind::Parameter, "theta")]);
        let assessed = DependenceContract::assess(&[left.clone(), right.clone()]);
        let err = DependenceContract::try_claim_independent(&[left, right])
            .expect_err("shared parameter");
        assert_eq!(err.detail(), "dependence.shared_factor");
        assert_eq!(err.refusal().0, "evidence_double_counted");
        assert!(err.message().contains("parameter"));
        assert!(err.message().contains("theta"));
        assert!(antecedent_core::reason_code::is_runtime_refusal(err.code()));
        let next = err.into_contract().expect("dependent contract");
        assert_eq!(next, assessed);
        assert!(next.is_dependent());
        assert!(!next.is_independent());

        let known = declared("known", &[(SharedFactorKind::Period, "2021")]);
        let opaque = unknown("opaque", &[]);
        let assessed_unknown = DependenceContract::assess(&[known.clone(), opaque.clone()]);
        let err = DependenceContract::try_claim_independent(&[known, opaque])
            .expect_err("unknown overlap");
        assert_eq!(err.detail(), "dependence.overlap_unknown");
        assert!(err.message().contains("overlap is unknown"));
        assert!(antecedent_core::reason_code::is_runtime_refusal(err.code()));
        let next = err.next_contract().expect("unknown contract");
        assert_eq!(next, &assessed_unknown);
        assert!(next.is_unknown());
        assert!(!next.is_independent());

        let supported = DependenceContract::try_claim_independent(&[
            declared("fresh", &[(SharedFactorKind::Period, "2021")]),
            declared("separate", &[(SharedFactorKind::Period, "2022")]),
        ])
        .expect("supported independence");
        assert!(supported.is_independent());
        assert_eq!(supported.status(), DependenceStatus::Independent);
    }

    #[test]
    fn each_shared_factor_kind_is_dependence() {
        assert_eq!(SharedFactorKind::ALL, KINDS);
        assert!(SharedFactorKind::from_name("not-a-factor").is_none());
        for kind in SharedFactorKind::ALL {
            assert_eq!(SharedFactorKind::from_name(kind.as_str()), Some(kind));
            let left = declared("left", &[(kind, "shared")]);
            let right = declared("right", &[(kind, "shared"), (kind, "only-right")]);
            assert_ne!(left.id().as_str(), right.id().as_str());
            let contract = DependenceContract::assess(&[left, right]);
            assert!(contract.is_dependent(), "{}", kind.as_str());
            assert!(!contract.is_independent(), "{}", kind.as_str());
            assert!(
                contract
                    .shared_factors()
                    .iter()
                    .any(|factor| { factor.kind() == kind && factor.id().as_str() == "shared" })
            );
            assert!(
                contract.shared_factors().iter().all(|factor| factor.id().as_str() == "shared")
            );
        }
    }

    #[test]
    fn factor_ids_are_compared_after_trimming() {
        let left = declared("left", &[(SharedFactorKind::Row, "row-1")]);
        let right = declared("right", &[(SharedFactorKind::Row, " row-1\t")]);
        let contract = DependenceContract::assess(&[left, right]);
        assert!(contract.is_dependent());
        assert_eq!(contract.shared_factors().len(), 1);
        assert_eq!(contract.shared_factors()[0].id().as_str(), "row-1");
        let mut factors = SharedFactorSets::empty();
        factors.try_insert(SharedFactorKind::Row, "row-1").expect("factor");
        factors.try_insert(SharedFactorKind::Row, "row-1 ").expect("factor");
        assert_eq!(factors.ids(SharedFactorKind::Row).len(), 1);
    }

    #[test]
    fn blank_source_and_factor_ids_are_refused() {
        let source = DependenceSource::try_declared("  ", SharedFactorSets::empty())
            .expect_err("blank source");
        assert_eq!(source.detail(), "dependence.blank_source_id");
        assert_eq!(source.code(), "invalid_argument");
        assert!(source.message().contains("blank"));
        assert!(source.into_contract().is_none());
        assert!(DependenceSource::try_declared("", SharedFactorSets::empty()).is_err());

        let mut factors = SharedFactorSets::empty();
        let factor = factors.try_insert(SharedFactorKind::Row, " \t").expect_err("blank factor");
        assert_eq!(factor.detail(), "dependence.blank_factor_id");
        assert!(factor.next_contract().is_none());
        assert!(factors.ids(SharedFactorKind::Row).is_empty());
        factors.try_insert(SharedFactorKind::Row, "row 1").expect("inner space");
        assert_eq!(
            factors.ids(SharedFactorKind::Row).iter().next().expect("row").as_str(),
            "row 1",
        );
    }
}
