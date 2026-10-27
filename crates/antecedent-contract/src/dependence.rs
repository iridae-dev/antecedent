//! Dependence between evidence sources that share rows, clusters, models,
//! parameters, priors, disturbances, or periods.
//!
//! Each [`DependenceSource`] carries an identity and the factor identities it
//! uses. [`DependenceContract::assess`] reports the relation those sets
//! support: independent when every source has declared its factors and no pair
//! shares a dependence factor, dependent when a dependence factor is shared,
//! and unknown when a source's overlap is undeclared and no shared dependence
//! factor is known. A known shared factor stays dependence when another
//! source's overlap is unknown.
//!
//! A parameter id is not a dependence factor. Independent studies that inform
//! one parameter, with disjoint rows and no other shared factor, are
//! independent. Factor ids are compared after trimming, case-folding, dropping
//! invisible format characters, and composing common Latin accents. An
//! assessment indexes factors rather than comparing sources pairwise, so its
//! cost is linear (up to ordering) in the number of declared factors.
//!
//! An independence claim is [`DependenceContract::try_claim_independent`].
//! When the factors do not support it, the error names the shared factor, or
//! names that overlap is unknown, and [`DependenceError::next_contract`] is
//! the dependent or unknown contract. A covariance combination of a dependent
//! contract requests the cross terms of the shared factors. Disjoint source
//! identities do not establish independence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
    sync::Arc,
};

use antecedent_core::{Refused, impl_refused, named_variants};

const KIND_COUNT: usize = SharedFactorKind::ALL.len();

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
    /// A parameter the evidence informs.
    ///
    /// Sharing this id is not dependence when the studies are otherwise
    /// independent. Independent studies inform one parameter. A shared row,
    /// cluster, model, prior, disturbance, or period is still dependence, and
    /// the parameter is then omitted from the shared-factor report.
    Parameter,
    /// A prior, including a prior built from other evidence.
    Prior,
    /// A shared disturbance or shock.
    Disturbance,
    /// A time period the evidence covers.
    Period,
}

named_variants! {
    SharedFactorKind {
        Row => "row",
        Cluster => "cluster",
        Model => "model",
        Parameter => "parameter",
        Prior => "prior",
        Disturbance => "disturbance",
        Period => "period",
    }
}

impl SharedFactorKind {
    /// Position of this kind in [`Self::ALL`].
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Identity of one shared factor.
///
/// Blank text is refused. Comparison trims surrounding whitespace, folds case,
/// drops invisible format characters, and composes common Latin letter and
/// combining-mark pairs, so `Row-1`, ` row-1 `, and `ro\u{200B}w-1` name the
/// same factor. The stored text is that normalized form.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SharedFactorId(Arc<str>);

impl SharedFactorId {
    fn try_new(text: &str) -> Result<Self, DependenceError> {
        let folded = fold_identity(text);
        let trimmed = folded.trim();
        if antecedent_core::is_blank(trimmed) {
            return Err(DependenceError::blank(BLANK_FACTOR_DETAIL, BLANK_FACTOR_MESSAGE));
        }
        Ok(Self(Arc::from(trimmed)))
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
    by_kind: [BTreeSet<SharedFactorId>; KIND_COUNT],
}

impl SharedFactorSets {
    /// No factors of any kind.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            by_kind: [
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
                BTreeSet::new(),
            ],
        }
    }

    /// Record `id` under `kind`. A repeated id is stored once.
    ///
    /// A refused id leaves the sets unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when `id` is empty or only whitespace.
    #[must_use = "a refused factor id is not recorded"]
    pub fn try_insert(
        &mut self,
        kind: SharedFactorKind,
        id: impl AsRef<str>,
    ) -> Result<(), DependenceError> {
        let id = SharedFactorId::try_new(id.as_ref())?;
        self.by_kind[kind.index()].insert(id);
        Ok(())
    }

    /// Identities recorded for `kind`, in order.
    #[must_use]
    pub fn ids(&self, kind: SharedFactorKind) -> &BTreeSet<SharedFactorId> {
        &self.by_kind[kind.index()]
    }

    /// Number of factors recorded across every kind.
    #[must_use]
    pub fn len(&self) -> usize {
        self.by_kind.iter().map(BTreeSet::len).sum()
    }

    /// Whether every kind is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.by_kind.iter().all(BTreeSet::is_empty)
    }

    fn contains(&self, factor: &SharedFactor) -> bool {
        self.ids(factor.kind()).contains(factor.id())
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
    /// Every source declared its factors, and no pair shares a dependence factor.
    ///
    /// A shared parameter does not count.
    Independent,
    /// At least one dependence factor appears in two sources. A parameter does
    /// not. Sources whose overlap is undeclared stay on the contract, so the
    /// known factors are a lower bound.
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
    /// A shared dependence factor is dependence, including when another source
    /// left its overlap unknown. A shared parameter is not that factor: studies
    /// with the same parameter id, disjoint rows, and no other shared factor are
    /// independent. Unknown overlap with no known shared dependence factor is
    /// [`DependenceStatus::Unknown`]. Independence requires complete disclosure
    /// and pairwise disjoint dependence factors. Source identities are not the
    /// criterion. Fewer than two sources are independent when every disclosure
    /// is complete.
    ///
    /// # Errors
    ///
    /// Never refuses today; the `Result` is kept so a future refusal is not a breaking change.
    ///
    /// Shared factors come from an index of factor to number of sources naming it, so the cost
    /// does not grow with the square of the number of sources.
    #[must_use = "an assessment that is not read does nothing"]
    pub fn assess(sources: &[DependenceSource]) -> Result<Self, DependenceError> {
        let mut holders: BTreeMap<SharedFactor, usize> = BTreeMap::new();
        for source in sources {
            for kind in SharedFactorKind::ALL {
                // A parameter is the quantity independent studies inform.
                if matches!(kind, SharedFactorKind::Parameter) {
                    continue;
                }
                for id in source.factors.ids(kind) {
                    *holders.entry(SharedFactor { kind, id: id.clone() }).or_insert(0) += 1;
                }
            }
        }
        let shared: BTreeSet<SharedFactor> = holders
            .into_iter()
            .filter(|&(_, count)| count >= 2)
            .map(|(factor, _)| factor)
            .collect();
        let mut unknown = BTreeSet::new();
        for source in sources {
            if matches!(source.disclosure, FactorDisclosure::UnknownOverlap) {
                unknown.insert(source.id.clone());
            }
        }
        if !shared.is_empty() {
            let factors = shared.into_iter().collect::<Box<[SharedFactor]>>();
            return Ok(Self {
                status: DependenceStatus::Dependent,
                factors,
                unknown_sources: unknown.into_iter().collect(),
            });
        }
        if unknown.is_empty() {
            Ok(Self {
                status: DependenceStatus::Independent,
                factors: empty_slice(),
                unknown_sources: empty_slice(),
            })
        } else {
            Ok(Self {
                status: DependenceStatus::Unknown,
                factors: empty_slice(),
                unknown_sources: unknown.into_iter().collect(),
            })
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
        let contract = Self::assess(sources)?;
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
        self.as_refusal().into_parts()
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

impl_refused!(DependenceError);

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

/// Canonical spelling of a factor id.
///
/// Three steps, in order. Invisible format characters (zero-width space and
/// joiners, directional marks and isolates, the word joiner, soft hyphen,
/// byte-order mark, and variation selectors) are dropped. The rest is lowercased
/// with the standard library's Unicode lowercase. Then a lowercase Latin letter
/// followed by a grave, acute, circumflex, tilde, diaeresis, ring, cedilla, or
/// caron combining mark is composed into the single precomposed letter, so the
/// decomposed and composed spellings of those letters agree.
///
/// This is not full Unicode normalization. The workspace carries no
/// normalization table, so decompositions outside that Latin set, and
/// compatibility forms such as full-width letters, stay distinct. Callers spell
/// such an id in one form.
fn fold_identity(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars().filter(|ch| !is_invisible_format(*ch)).flat_map(char::to_lowercase) {
        if let Some(composed) = out.chars().next_back().and_then(|base| compose(base, ch)) {
            out.pop();
            out.push(composed);
        } else {
            out.push(ch);
        }
    }
    out
}

const fn is_invisible_format(ch: char) -> bool {
    matches!(
        ch,
        '\u{00AD}'
            | '\u{061C}'
            | '\u{180E}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206F}'
            | '\u{FE00}'..='\u{FE0F}'
            | '\u{FEFF}'
    )
}

/// Precomposed lowercase Latin letter for `base` followed by `mark`.
const fn compose(base: char, mark: char) -> Option<char> {
    Some(match (mark, base) {
        ('\u{0300}', 'a') => 'à',
        ('\u{0300}', 'e') => 'è',
        ('\u{0300}', 'i') => 'ì',
        ('\u{0300}', 'o') => 'ò',
        ('\u{0300}', 'u') => 'ù',
        ('\u{0301}', 'a') => 'á',
        ('\u{0301}', 'e') => 'é',
        ('\u{0301}', 'i') => 'í',
        ('\u{0301}', 'o') => 'ó',
        ('\u{0301}', 'u') => 'ú',
        ('\u{0301}', 'y') => 'ý',
        ('\u{0301}', 'c') => 'ć',
        ('\u{0301}', 'n') => 'ń',
        ('\u{0301}', 's') => 'ś',
        ('\u{0301}', 'z') => 'ź',
        ('\u{0302}', 'a') => 'â',
        ('\u{0302}', 'e') => 'ê',
        ('\u{0302}', 'i') => 'î',
        ('\u{0302}', 'o') => 'ô',
        ('\u{0302}', 'u') => 'û',
        ('\u{0303}', 'a') => 'ã',
        ('\u{0303}', 'n') => 'ñ',
        ('\u{0303}', 'o') => 'õ',
        ('\u{0308}', 'a') => 'ä',
        ('\u{0308}', 'e') => 'ë',
        ('\u{0308}', 'i') => 'ï',
        ('\u{0308}', 'o') => 'ö',
        ('\u{0308}', 'u') => 'ü',
        ('\u{0308}', 'y') => 'ÿ',
        ('\u{030A}', 'a') => 'å',
        ('\u{0327}', 'c') => 'ç',
        ('\u{030C}', 'c') => 'č',
        ('\u{030C}', 'e') => 'ě',
        ('\u{030C}', 'n') => 'ň',
        ('\u{030C}', 'r') => 'ř',
        ('\u{030C}', 's') => 'š',
        ('\u{030C}', 'z') => 'ž',
        _ => return None,
    })
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

    fn contract_of(sources: &[DependenceSource]) -> DependenceContract {
        DependenceContract::assess(sources).expect("within the source cap")
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
        let contract = contract_of(&[original.clone(), copy.clone()]);
        assert!(contract.is_dependent());
        assert!(!contract.is_independent());
        assert_eq!(contract.shared_factors().len(), 2);
        assert_eq!(contract.shared_factors().first().expect("row").id().as_str(), "row-1");
        assert_eq!(contract_of(&[copy.clone(), original.clone()]), contract);

        let partial = declared("partial", &[(SharedFactorKind::Row, "row-2")]);
        let overlap = contract_of(&[original.clone(), partial]);
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
        let through_prior = contract_of(&[study, posterior]);
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
        let through_model = contract_of(&[fit.clone(), claim.clone()]);
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
        let contract = contract_of(&[known.clone(), opaque.clone()]);
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

        let alone = contract_of(&[unknown("alone", &[])]);
        assert!(alone.is_unknown());
        assert!(!alone.is_independent());

        let partial = unknown("partial", &[(SharedFactorKind::Row, "row-1")]);
        let shared = contract_of(&[known, partial]);
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
        let complete = contract_of(&[
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
        let contract = contract_of(&[left, middle, right]);
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
        let independent = contract_of(&[separate_a, separate_b]);
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
        let shared = contract_of(&[left.clone(), right.clone()]);
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
        let undeclared = contract_of(&[opaque, other]);
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
        let left = declared("left", &[(SharedFactorKind::Row, "row-1")]);
        let right = declared("right", &[(SharedFactorKind::Row, "row-1")]);
        let assessed = contract_of(&[left.clone(), right.clone()]);
        let err =
            DependenceContract::try_claim_independent(&[left, right]).expect_err("shared row");
        assert_eq!(err.detail(), "dependence.shared_factor");
        assert_eq!(err.refusal().0, "evidence_double_counted");
        assert!(err.message().contains("row"));
        assert!(err.message().contains("row-1"));
        assert!(antecedent_core::reason_code::is_runtime_refusal(err.code()));
        let next = err.into_contract().expect("dependent contract");
        assert_eq!(next, assessed);
        assert!(next.is_dependent());
        assert!(!next.is_independent());

        let known = declared("known", &[(SharedFactorKind::Period, "2021")]);
        let opaque = unknown("opaque", &[]);
        let assessed_unknown = contract_of(&[known.clone(), opaque.clone()]);
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
            // A shared parameter is not dependence. The dedicated test covers it.
            if matches!(kind, SharedFactorKind::Parameter) {
                continue;
            }
            let left = declared("left", &[(kind, "shared")]);
            let right = declared("right", &[(kind, "shared"), (kind, "only-right")]);
            assert_ne!(left.id().as_str(), right.id().as_str());
            let contract = contract_of(&[left, right]);
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
    fn factor_ids_are_compared_after_trimming_and_case_folding() {
        let left = declared("left", &[(SharedFactorKind::Row, "Row-1")]);
        let right = declared("right", &[(SharedFactorKind::Row, " row-1\t")]);
        let contract = contract_of(&[left, right]);
        assert!(contract.is_dependent());
        assert_eq!(contract.shared_factors().len(), 1);
        assert_eq!(contract.shared_factors()[0].id().as_str(), "row-1");
        let mut factors = SharedFactorSets::empty();
        factors.try_insert(SharedFactorKind::Row, "row-1").expect("factor");
        factors.try_insert(SharedFactorKind::Row, "row-1 ").expect("factor");
        factors.try_insert(SharedFactorKind::Row, "ROW-1").expect("factor");
        assert_eq!(factors.ids(SharedFactorKind::Row).len(), 1);
        assert_eq!(
            factors.ids(SharedFactorKind::Row).iter().next().expect("row").as_str(),
            "row-1"
        );
    }

    #[test]
    fn padded_cased_zero_width_and_decomposed_ids_collide() {
        let spellings = [
            "caf\u{e9}-1",
            "  CAF\u{c9}-1  ",
            "cafe\u{301}-1",
            "caf\u{200B}e\u{301}-1",
            "\u{FEFF}Caf\u{200D}\u{e9}-1\u{2060}",
        ];
        let mut factors = SharedFactorSets::empty();
        for spelling in spellings {
            factors.try_insert(SharedFactorKind::Row, spelling).expect("factor");
        }
        assert_eq!(factors.ids(SharedFactorKind::Row).len(), 1);
        assert_eq!(
            factors.ids(SharedFactorKind::Row).iter().next().expect("row").as_str(),
            "caf\u{e9}-1"
        );
        let left = declared("left", &[(SharedFactorKind::Row, "Ro\u{200B}w-1")]);
        let right = declared("right", &[(SharedFactorKind::Row, " ROW-1 ")]);
        assert!(contract_of(&[left, right]).is_dependent());
        let mut blank = SharedFactorSets::empty();
        assert!(blank.try_insert(SharedFactorKind::Row, " \u{200B}\u{FEFF} ").is_err());
    }

    #[test]
    fn shared_parameter_of_independent_studies_is_not_dependence() {
        let left = declared(
            "study-a",
            &[(SharedFactorKind::Parameter, "Theta"), (SharedFactorKind::Row, "row-a")],
        );
        let right = declared(
            "study-b",
            &[(SharedFactorKind::Parameter, " theta "), (SharedFactorKind::Row, "row-b")],
        );
        assert_ne!(left.id().as_str(), right.id().as_str());
        let left_parameter = left.factors().ids(SharedFactorKind::Parameter);
        let right_parameter = right.factors().ids(SharedFactorKind::Parameter);
        assert_eq!(left_parameter.iter().next().expect("parameter").as_str(), "theta");
        assert_eq!(right_parameter.iter().next().expect("parameter").as_str(), "theta");
        let contract = contract_of(&[left.clone(), right.clone()]);
        assert!(contract.is_independent());
        assert!(!contract.is_dependent());
        assert!(contract.shared_factors().is_empty());
        assert_eq!(contract.status(), DependenceStatus::Independent);
        let claimed =
            DependenceContract::try_claim_independent(&[left, right]).expect("independent");
        assert!(claimed.is_independent());
        let diagonal = claimed.covariance_combination().expect("diagonal");
        assert!(!diagonal.requires_cross_terms());

        let shared_row = contract_of(&[
            declared(
                "study-c",
                &[(SharedFactorKind::Parameter, "theta"), (SharedFactorKind::Row, "row-1")],
            ),
            declared(
                "study-d",
                &[(SharedFactorKind::Parameter, "THETA"), (SharedFactorKind::Row, "row-1")],
            ),
        ]);
        assert!(shared_row.is_dependent());
        assert!(shared_row.shared_factors().iter().all(|factor| {
            factor.kind() == SharedFactorKind::Row && factor.id().as_str() == "row-1"
        }));
    }

    #[test]
    fn many_sources_and_long_ids_are_assessed_without_a_cap() {
        let sources: Vec<DependenceSource> =
            (0..2_000).map(|index| declared(&index.to_string(), &[])).collect();
        let contract = DependenceContract::assess(&sources).expect("no source cap");
        assert!(contract.is_independent());
        let mut shared: Vec<DependenceSource> = (0..300)
            .map(|index| {
                let row = format!("r{index}");
                declared(&format!("s{index}"), &[(SharedFactorKind::Row, row.as_str())])
            })
            .collect();
        shared.push(declared("tail", &[(SharedFactorKind::Row, "r7")]));
        assert!(DependenceContract::assess(&shared).expect("assessed").is_dependent());
        let mut sets = SharedFactorSets::empty();
        let long = "x".repeat(10_000);
        sets.try_insert(SharedFactorKind::Row, &long).expect("long ids are carried");
        for index in 0..5_000 {
            sets.try_insert(SharedFactorKind::Row, format!("row-{index}")).expect("no factor cap");
        }
        assert_eq!(sets.len(), 5_001);
        assert!(DependenceSource::try_declared(&long, SharedFactorSets::empty()).is_ok());
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
