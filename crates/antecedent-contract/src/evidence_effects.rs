// SPDX-License-Identifier: MIT OR Apache-2.0

//! Operators that change what the evidence ledger counts, recorded before any
//! numerical combination.
//!
//! Pooling, hierarchical borrowing and updating, transport, bridges, frozen
//! reuse, repeated observation use, mixtures, marginalization, and envelopes
//! are separate [`EvidenceOperator`]s. Each [`EvidenceEffect`] is one of those
//! operators applied to ledger subjects ([`SourceId`] and [`ResultId`]).
//! [`EvidenceEffectLedger::try_combine`] admits a numerical combination only
//! after that effect has been recorded. Until then the refusal's next attempt
//! is to record the effect, then combine. A pooling effect also consults a
//! [`DependenceContract`] and the evidence ledger's observation overlap.
//! Sources that share rows are refused as an independent pool, or recorded
//! with [`EvidenceEffect::dependent_pooling`] and then combined.
//!
//! One operator does not stand in for another. A mixture is not a pool, a
//! bridge is not transport, and an envelope is not a marginal. The five
//! operators the evidence ledger already names as a [`SynthesisKind`] still
//! need their own effect here; bridge, mixture, marginalization, and envelope
//! extend the ledger and have no synthesis kind. This module does not pool,
//! update a posterior, transport a functional, or compute an envelope.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use antecedent_core::{CanonicalWriter, NextAttempt, Refused, SemanticDigest, impl_refused};

use crate::dependence::{DependenceContract, DependenceError, DependenceSource, SharedFactorKind};
use crate::evidence_ledger::{
    EvidenceLedger, LedgerError, MAX_COMBINATION_SOURCES, MAX_RESULTS, MAX_TEXT_BYTES, Overlap,
    ResultId, SourceId, SynthesisKind,
};
use antecedent_core::named_variants;

const IDENTITY_TAG: &str = "antecedent.contract.evidence_effect.v1";

/// Version word written into every effect identity.
pub const EFFECT_IDENTITY_VERSION: u64 = 1;

/// Most distinct effects one ledger records.
///
/// The same bound as the evidence ledger's results.
pub const MAX_RECORDED_EFFECTS: usize = MAX_RESULTS;

const NOT_RECORDED_DETAIL: &str = "evidence_effects.effect_not_recorded";
const NOT_INTERCHANGEABLE_DETAIL: &str = "evidence_effects.operators_not_interchangeable";
const TOO_FEW_DETAIL: &str = "evidence_effects.too_few_subjects";
const TOO_MANY_DETAIL: &str = "evidence_effects.too_many_sources";
const DUPLICATE_DETAIL: &str = "evidence_effects.duplicate_source";
const SHARED_DETAIL: &str = "evidence_effects.shared_source";
const BLANK_DETAIL: &str = "evidence_effects.blank_text";
const TEXT_DETAIL: &str = "evidence_effects.text_too_long";
const LIMIT_DETAIL: &str = "evidence_effects.limit";
const NOT_CONSULTED_DETAIL: &str = "evidence_effects.dependence_not_consulted";
const SHARED_ROWS_DETAIL: &str = "evidence_effects.shared_rows";
const DEPENDENT_SOURCES_DETAIL: &str = "evidence_effects.dependent_sources";
const OVERLAP_UNKNOWN_DETAIL: &str = "evidence_effects.overlap_unknown";
const DEPENDENCE_ABSENT_DETAIL: &str = "evidence_effects.dependence_not_found";
const LEDGER_OVERLAP_DETAIL: &str = "evidence_effects.ledger_overlap";

const TOO_MANY_MESSAGE: &str = "a combination names at most 64 sources";
const _: () = assert!(MAX_COMBINATION_SOURCES == 64);
const DUPLICATE_MESSAGE: &str = "a source is listed twice";
const BLANK_MESSAGE: &str = "a label is blank";
const TEXT_MESSAGE: &str = "a label is longer than the ledger allows";
const LIMIT_MESSAGE: &str = "the effect ledger cannot record another effect";

/// One evidence operator. None stands in for another.
///
/// Declaration order is the order of [`Self::ALL`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum EvidenceOperator {
    /// Combine summaries of separate sources into one pooled summary.
    Pooling,
    /// Borrow from a hierarchical prior, or update that hierarchy with likelihood.
    ///
    /// Borrowing and updating are this operator. Neither is pooling.
    HierarchicalBorrowing,
    /// Carry claims into a target population alongside native target evidence.
    Transport,
    /// Map a source claim onto a different scientific question.
    ///
    /// A bridge is not transport.
    Bridge,
    /// Cite an existing result again without updating it.
    FrozenReuse,
    /// Use the same observations in a later stage of one workflow.
    RepeatedObservationUse,
    /// Mix component laws under declared weights.
    ///
    /// A mixture is not a pool.
    Mixture,
    /// Integrate a joint result down to a marginal.
    Marginalization,
    /// Keep a set or assumption range.
    ///
    /// An envelope is not a marginal.
    Envelope,
}

named_variants! {
    EvidenceOperator {
        Pooling => "pooling",
        HierarchicalBorrowing => "hierarchical_borrowing",
        Transport => "transport",
        Bridge => "bridge",
        FrozenReuse => "frozen_reuse",
        RepeatedObservationUse => "repeated_observation_use",
        Mixture => "mixture",
        Marginalization => "marginalization",
        Envelope => "envelope",
    }
}

impl EvidenceOperator {
    /// Ledger synthesis this operator extends, when the ledger already names it.
    ///
    /// Bridge, mixture, marginalization, and envelope have no synthesis kind.
    /// A synthesis kind is not a recorded effect and does not admit numerical
    /// combination.
    #[must_use = "bridge, mixture, marginalization, and envelope have no synthesis kind"]
    pub const fn synthesis_kind(self) -> Option<SynthesisKind> {
        match self {
            Self::Pooling => Some(SynthesisKind::StatisticalPooling),
            Self::HierarchicalBorrowing => Some(SynthesisKind::BayesianBorrowing),
            Self::Transport => Some(SynthesisKind::CausalTransport),
            Self::FrozenReuse => Some(SynthesisKind::EvidenceReuse),
            Self::RepeatedObservationUse => Some(SynthesisKind::RepeatedObservationUse),
            Self::Bridge | Self::Mixture | Self::Marginalization | Self::Envelope => None,
        }
    }

    /// Whether a recorded effect of `self` may be combined as `requested`.
    ///
    /// Only the same operator licenses the combination.
    #[must_use]
    pub const fn licenses(self, requested: Self) -> bool {
        self as u8 == requested as u8
    }

    /// Refuse to combine a recorded effect of `self` as `requested`.
    ///
    /// `None` means the operators are the same. That is not an admission:
    /// numerical combination still waits on [`EvidenceEffectLedger::try_combine`].
    ///
    /// The error names the next attempt: record `requested`, then combine.
    #[must_use = "a different operator is a failed attempt"]
    pub fn substitution_refusal(self, requested: Self) -> Option<EvidenceEffectError> {
        if self.licenses(requested) {
            None
        } else {
            Some(EvidenceEffectError::not_interchangeable(self, requested))
        }
    }
}

impl fmt::Display for EvidenceOperator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The attempt a refused numerical combination names: record the effect, then
/// combine.
///
/// [`EvidenceEffectError::requested_operator`] is the effect that attempt
/// records.
pub const RECORD_THEN_COMBINE: NextAttempt =
    NextAttempt::new("record_then_combine", "record the effect, then combine");

/// The attempt a refused pool names when the sources are dependent: record the
/// pool as dependent, then combine.
pub const RECORD_DEPENDENT_POOLING: NextAttempt = NextAttempt::new(
    "record_dependent_pooling",
    "record the pooling effect as dependent, then combine",
);

/// Identity of one recorded effect.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EffectId(SemanticDigest);

impl EffectId {
    /// Wrap a digest.
    #[must_use]
    pub const fn from_digest(digest: SemanticDigest) -> Self {
        Self(digest)
    }

    /// The underlying digest.
    #[must_use]
    pub const fn digest(self) -> SemanticDigest {
        self.0
    }
}

impl fmt::Display for EffectId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0.to_hex())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum EffectBody {
    Pooling { sources: Box<[SourceId]>, dependent: bool },
    HierarchicalBorrowing { priors: Box<[SourceId]>, likelihood: Box<[SourceId]> },
    Transport { transported: Box<[SourceId]>, native: Box<[SourceId]> },
    Bridge { source: SourceId, contract: Arc<str> },
    FrozenReuse(ResultId),
    RepeatedObservationUse { sources: Box<[SourceId]>, stage: Arc<str> },
    Mixture(Box<[SourceId]>),
    Marginalization(ResultId),
    Envelope(Box<[SourceId]>),
}

/// One operator applied to evidence-ledger subjects.
///
/// Constructed only in a form the ledger can record. Source order does not
/// change the effect. Numerical combination of the effect waits until
/// [`EvidenceEffectLedger::record`] has stored it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceEffect {
    body: EffectBody,
}

impl EvidenceEffect {
    /// Pool `sources`. At least two, each listed once.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than two sources are named, a source is
    /// listed twice, or the list exceeds [`MAX_COMBINATION_SOURCES`].
    #[must_use = "an effect the ledger cannot record is refused"]
    pub fn pooling(
        sources: impl IntoIterator<Item = SourceId>,
    ) -> Result<Self, EvidenceEffectError> {
        let sources = canonical_sources(sources, 2, "pooling needs at least two sources")?;
        Ok(Self { body: EffectBody::Pooling { sources, dependent: false } })
    }

    /// Pool `sources` and record that the pool is dependent.
    ///
    /// Use this when the sources share rows or another dependence factor.
    /// [`EvidenceEffectLedger::try_combine`] admits it only after that
    /// dependence was consulted. An independent pool of the same sources is a
    /// different effect.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than two sources are named, a source is
    /// listed twice, or the list exceeds [`MAX_COMBINATION_SOURCES`].
    #[must_use = "an effect the ledger cannot record is refused"]
    pub fn dependent_pooling(
        sources: impl IntoIterator<Item = SourceId>,
    ) -> Result<Self, EvidenceEffectError> {
        let sources = canonical_sources(sources, 2, "pooling needs at least two sources")?;
        Ok(Self { body: EffectBody::Pooling { sources, dependent: true } })
    }

    /// Hierarchically borrow `priors` into `likelihood`, or update that hierarchy.
    ///
    /// A source cannot be both a prior and a likelihood. This is not a pool.
    ///
    /// # Errors
    ///
    /// Returns an error when either list is empty, a source is repeated, a
    /// source is in both lists, or either list exceeds [`MAX_COMBINATION_SOURCES`].
    #[must_use = "an effect the ledger cannot record is refused"]
    pub fn hierarchical_borrowing(
        priors: impl IntoIterator<Item = SourceId>,
        likelihood: impl IntoIterator<Item = SourceId>,
    ) -> Result<Self, EvidenceEffectError> {
        let priors = canonical_sources(
            priors,
            1,
            "hierarchical borrowing needs a prior and a likelihood source",
        )?;
        let likelihood = canonical_sources(
            likelihood,
            1,
            "hierarchical borrowing needs a prior and a likelihood source",
        )?;
        require_disjoint(&priors, &likelihood, "a source cannot be both a prior and a likelihood")?;
        Ok(Self { body: EffectBody::HierarchicalBorrowing { priors, likelihood } })
    }

    /// Transport `transported` claims onto `native` target evidence.
    ///
    /// A source cannot be both transported and native. This is not a bridge.
    ///
    /// # Errors
    ///
    /// Returns an error when either list is empty, a source is repeated, a
    /// source is in both lists, or either list exceeds [`MAX_COMBINATION_SOURCES`].
    #[must_use = "an effect the ledger cannot record is refused"]
    pub fn transport(
        transported: impl IntoIterator<Item = SourceId>,
        native: impl IntoIterator<Item = SourceId>,
    ) -> Result<Self, EvidenceEffectError> {
        let transported = canonical_sources(
            transported,
            1,
            "transport needs a transported source and a native source",
        )?;
        let native = canonical_sources(
            native,
            1,
            "transport needs a transported source and a native source",
        )?;
        require_disjoint(&transported, &native, "a source cannot be both transported and native")?;
        Ok(Self { body: EffectBody::Transport { transported, native } })
    }

    /// Bridge `source` under `contract`.
    ///
    /// `contract` is the bridge's own name. Transport does not record it.
    ///
    /// # Errors
    ///
    /// Returns an error when `contract` is blank or longer than [`MAX_TEXT_BYTES`].
    #[must_use = "an effect the ledger cannot record is refused"]
    pub fn bridge(
        source: SourceId,
        contract: impl AsRef<str>,
    ) -> Result<Self, EvidenceEffectError> {
        let contract = checked_text(contract.as_ref())?;
        Ok(Self { body: EffectBody::Bridge { source, contract } })
    }

    /// Cite `result` again, frozen. The citation does not update it.
    #[must_use = "the effect has to be recorded before combination"]
    pub fn frozen_reuse(result: ResultId) -> Self {
        Self { body: EffectBody::FrozenReuse(result) }
    }

    /// Use `sources` again at `stage`.
    ///
    /// # Errors
    ///
    /// Returns an error when no source is named, a source is listed twice, the
    /// list exceeds [`MAX_COMBINATION_SOURCES`], or `stage` is blank or longer
    /// than [`MAX_TEXT_BYTES`].
    #[must_use = "an effect the ledger cannot record is refused"]
    pub fn repeated_observation_use(
        sources: impl IntoIterator<Item = SourceId>,
        stage: impl AsRef<str>,
    ) -> Result<Self, EvidenceEffectError> {
        let sources = canonical_sources(sources, 1, "repeated observation use needs a source")?;
        let stage = checked_text(stage.as_ref())?;
        Ok(Self { body: EffectBody::RepeatedObservationUse { sources, stage } })
    }

    /// Mix `components`. At least two. This is not a pool of those sources.
    ///
    /// # Errors
    ///
    /// Returns an error when fewer than two components are named, a source is
    /// listed twice, or the list exceeds [`MAX_COMBINATION_SOURCES`].
    #[must_use = "an effect the ledger cannot record is refused"]
    pub fn mixture(
        components: impl IntoIterator<Item = SourceId>,
    ) -> Result<Self, EvidenceEffectError> {
        let components =
            canonical_sources(components, 2, "a mixture needs at least two components")?;
        Ok(Self { body: EffectBody::Mixture(components) })
    }

    /// Marginalize the joint `result`. This is not an envelope.
    #[must_use = "the effect has to be recorded before combination"]
    pub fn marginalization(result: ResultId) -> Self {
        Self { body: EffectBody::Marginalization(result) }
    }

    /// Envelope `sources`. This is not a marginal of those sources.
    ///
    /// # Errors
    ///
    /// Returns an error when no source is named, a source is listed twice, or
    /// the list exceeds [`MAX_COMBINATION_SOURCES`].
    #[must_use = "an effect the ledger cannot record is refused"]
    pub fn envelope(
        sources: impl IntoIterator<Item = SourceId>,
    ) -> Result<Self, EvidenceEffectError> {
        let sources = canonical_sources(sources, 1, "an envelope needs a source")?;
        Ok(Self { body: EffectBody::Envelope(sources) })
    }

    /// The operator this effect records.
    #[must_use]
    pub const fn operator(&self) -> EvidenceOperator {
        match &self.body {
            EffectBody::Pooling { .. } => EvidenceOperator::Pooling,
            EffectBody::HierarchicalBorrowing { .. } => EvidenceOperator::HierarchicalBorrowing,
            EffectBody::Transport { .. } => EvidenceOperator::Transport,
            EffectBody::Bridge { .. } => EvidenceOperator::Bridge,
            EffectBody::FrozenReuse(_) => EvidenceOperator::FrozenReuse,
            EffectBody::RepeatedObservationUse { .. } => EvidenceOperator::RepeatedObservationUse,
            EffectBody::Mixture(_) => EvidenceOperator::Mixture,
            EffectBody::Marginalization(_) => EvidenceOperator::Marginalization,
            EffectBody::Envelope(_) => EvidenceOperator::Envelope,
        }
    }

    /// Identity of this effect. Source order is already canonical.
    #[must_use]
    pub fn identity(&self) -> EffectId {
        let mut writer = CanonicalWriter::new(IDENTITY_TAG);
        writer.word(EFFECT_IDENTITY_VERSION).text(self.operator().as_str());
        match &self.body {
            EffectBody::Pooling { sources, dependent } => {
                writer.word(u64::from(*dependent));
                write_sources(&mut writer, sources);
            }
            EffectBody::Mixture(sources) | EffectBody::Envelope(sources) => {
                write_sources(&mut writer, sources);
            }
            EffectBody::HierarchicalBorrowing { priors, likelihood } => {
                write_sources(&mut writer, priors);
                write_sources(&mut writer, likelihood);
            }
            EffectBody::Transport { transported, native } => {
                write_sources(&mut writer, transported);
                write_sources(&mut writer, native);
            }
            EffectBody::Bridge { source, contract } => {
                writer.bytes(source.digest().as_bytes()).text(contract);
            }
            EffectBody::FrozenReuse(result) | EffectBody::Marginalization(result) => {
                writer.bytes(result.digest().as_bytes());
            }
            EffectBody::RepeatedObservationUse { sources, stage } => {
                write_sources(&mut writer, sources);
                writer.text(stage);
            }
        }
        EffectId(SemanticDigest::from_bytes(writer.finish()))
    }

    /// Sources of a pooling effect.
    #[must_use = "only a pooling effect has pooling sources"]
    pub fn pooling_sources(&self) -> Option<&[SourceId]> {
        match &self.body {
            EffectBody::Pooling { sources, .. } => Some(sources.as_ref()),
            _ => None,
        }
    }

    /// Prior sources of a hierarchical borrowing effect.
    #[must_use = "only hierarchical borrowing has prior sources"]
    pub fn borrowing_priors(&self) -> Option<&[SourceId]> {
        match &self.body {
            EffectBody::HierarchicalBorrowing { priors, .. } => Some(priors.as_ref()),
            _ => None,
        }
    }

    /// Likelihood sources of a hierarchical borrowing effect.
    #[must_use = "only hierarchical borrowing has likelihood sources"]
    pub fn borrowing_likelihood(&self) -> Option<&[SourceId]> {
        match &self.body {
            EffectBody::HierarchicalBorrowing { likelihood, .. } => Some(likelihood.as_ref()),
            _ => None,
        }
    }

    /// Transported sources of a transport effect.
    #[must_use = "only transport has transported sources"]
    pub fn transported_sources(&self) -> Option<&[SourceId]> {
        match &self.body {
            EffectBody::Transport { transported, .. } => Some(transported.as_ref()),
            _ => None,
        }
    }

    /// Native sources of a transport effect.
    #[must_use = "only transport has native sources"]
    pub fn native_sources(&self) -> Option<&[SourceId]> {
        match &self.body {
            EffectBody::Transport { native, .. } => Some(native.as_ref()),
            _ => None,
        }
    }

    /// Source claim of a bridge effect.
    #[must_use = "only a bridge has a bridge source"]
    pub const fn bridge_source(&self) -> Option<SourceId> {
        match &self.body {
            EffectBody::Bridge { source, .. } => Some(*source),
            _ => None,
        }
    }

    /// Contract name of a bridge effect.
    #[must_use = "only a bridge has a bridge contract"]
    pub fn bridge_contract(&self) -> Option<&str> {
        match &self.body {
            EffectBody::Bridge { contract, .. } => Some(contract.as_ref()),
            _ => None,
        }
    }

    /// Result cited by a frozen reuse.
    #[must_use = "only frozen reuse cites a frozen result"]
    pub const fn frozen_result(&self) -> Option<ResultId> {
        match &self.body {
            EffectBody::FrozenReuse(result) => Some(*result),
            _ => None,
        }
    }

    /// Sources of a repeated observation use.
    #[must_use = "only repeated observation use has those sources"]
    pub fn repeated_sources(&self) -> Option<&[SourceId]> {
        match &self.body {
            EffectBody::RepeatedObservationUse { sources, .. } => Some(sources.as_ref()),
            _ => None,
        }
    }

    /// Stage name of a repeated observation use.
    #[must_use = "only repeated observation use has a stage"]
    pub fn repeated_stage(&self) -> Option<&str> {
        match &self.body {
            EffectBody::RepeatedObservationUse { stage, .. } => Some(stage.as_ref()),
            _ => None,
        }
    }

    /// Components of a mixture.
    #[must_use = "only a mixture has mixture components"]
    pub fn mixture_components(&self) -> Option<&[SourceId]> {
        match &self.body {
            EffectBody::Mixture(components) => Some(components.as_ref()),
            _ => None,
        }
    }

    /// Joint result a marginalization integrates.
    #[must_use = "only marginalization has a marginalized result"]
    pub const fn marginalized_result(&self) -> Option<ResultId> {
        match &self.body {
            EffectBody::Marginalization(result) => Some(*result),
            _ => None,
        }
    }

    /// Sources of an envelope.
    #[must_use = "only an envelope has envelope sources"]
    pub fn envelope_sources(&self) -> Option<&[SourceId]> {
        match &self.body {
            EffectBody::Envelope(sources) => Some(sources.as_ref()),
            _ => None,
        }
    }

    /// Whether this pool was recorded as dependent.
    #[must_use = "only a dependent pool records dependence"]
    pub const fn is_dependent_pool(&self) -> bool {
        match &self.body {
            EffectBody::Pooling { dependent, .. } => *dependent,
            _ => false,
        }
    }
}

/// An effect stored on the ledger, with the identity combination checks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecordedEffect {
    id: EffectId,
    effect: EvidenceEffect,
}

impl RecordedEffect {
    /// Identity of the recorded effect.
    #[must_use]
    pub const fn id(&self) -> EffectId {
        self.id
    }

    /// The effect that was recorded.
    #[must_use]
    pub const fn effect(&self) -> &EvidenceEffect {
        &self.effect
    }
}

/// Admission to run a numerical combination of one recorded effect.
///
/// The admission carries no estimate, posterior, interval, or envelope bound.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NumericalAdmission {
    effect: EffectId,
    operator: EvidenceOperator,
    dependent: bool,
}

impl NumericalAdmission {
    /// The recorded effect this admission is for.
    #[must_use]
    pub const fn effect_id(self) -> EffectId {
        self.effect
    }

    /// The operator that was recorded and combined.
    #[must_use]
    pub const fn operator(self) -> EvidenceOperator {
        self.operator
    }

    /// Synthesis kind to account on the existing ledger, when this operator has one.
    #[must_use = "bridge, mixture, marginalization, and envelope have no synthesis kind"]
    pub const fn synthesis_kind(self) -> Option<SynthesisKind> {
        self.operator.synthesis_kind()
    }

    /// Whether the admitted pool was recorded as dependent.
    #[must_use]
    pub const fn is_dependent(self) -> bool {
        self.dependent
    }
}

/// One ledger source and the dependence factors it declares.
///
/// [`EvidenceEffectLedger::consult`] pairs these with observation overlap on
/// the evidence ledger. The source id is the ledger id. The dependence source
/// names the factors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PooledSource {
    source: SourceId,
    dependence: DependenceSource,
}

impl PooledSource {
    /// Pair `source` with the factors in `dependence`.
    #[must_use]
    pub const fn new(source: SourceId, dependence: DependenceSource) -> Self {
        Self { source, dependence }
    }

    /// Evidence-ledger identity.
    #[must_use]
    pub const fn source(&self) -> SourceId {
        self.source
    }

    /// Factors this source declares.
    #[must_use]
    pub const fn dependence(&self) -> &DependenceSource {
        &self.dependence
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StoredOverlap {
    left: SourceId,
    right: SourceId,
    overlap: Overlap,
}

/// Effects recorded before numerical combination.
///
/// This extends the evidence ledger's subjects. It does not replace
/// [`crate::evidence_ledger::EvidenceLedger`] and it does not compute numbers.
/// Record the effect, then combine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceEffectLedger {
    recorded: Vec<RecordedEffect>,
    consulted: Vec<PooledSource>,
    overlaps: Vec<StoredOverlap>,
}

impl EvidenceEffectLedger {
    /// An empty effect ledger.
    #[must_use]
    pub fn new() -> Self {
        Self { recorded: Vec::new(), consulted: Vec::new(), overlaps: Vec::new() }
    }

    /// Consult `sources` against `ledger` before a pool is recorded or combined.
    ///
    /// Stores the dependence factors and the pairwise observation overlap.
    /// A later consult replaces this one. Pooling reads it from
    /// [`Self::record`] and [`Self::try_combine`].
    ///
    /// # Errors
    ///
    /// Returns an error when more than [`MAX_COMBINATION_SOURCES`] sources are
    /// named, a source id is repeated, the dependence assessment is refused, or
    /// a source is not on `ledger`.
    #[must_use = "pooling is refused until dependence and ledger overlap are consulted"]
    pub fn consult(
        &mut self,
        sources: &[PooledSource],
        ledger: &EvidenceLedger,
    ) -> Result<(), EvidenceEffectError> {
        if sources.len() > MAX_COMBINATION_SOURCES {
            return Err(EvidenceEffectError::invalid(TOO_MANY_DETAIL, TOO_MANY_MESSAGE));
        }
        let mut seen = BTreeSet::new();
        for source in sources {
            if !seen.insert(source.source()) {
                return Err(EvidenceEffectError::shared(DUPLICATE_DETAIL, DUPLICATE_MESSAGE));
            }
        }
        let dependence: Vec<DependenceSource> =
            sources.iter().map(|source| source.dependence().clone()).collect();
        if let Err(error) = DependenceContract::assess(&dependence) {
            return Err(EvidenceEffectError::from_dependence(&error));
        }
        let mut overlaps = Vec::new();
        for (index, left) in sources.iter().enumerate() {
            for right in sources.iter().skip(index + 1) {
                let overlap = ledger
                    .overlap(left.source(), right.source())
                    .map_err(|error| EvidenceEffectError::ledger_overlap(&error))?;
                overlaps.push(StoredOverlap {
                    left: left.source(),
                    right: right.source(),
                    overlap,
                });
            }
        }
        self.consulted = sources.to_vec();
        self.overlaps = overlaps;
        Ok(())
    }

    /// Effects in the order they were first recorded.
    #[must_use = "recorded effects are what combination checks"]
    pub fn recorded(&self) -> &[RecordedEffect] {
        &self.recorded
    }

    /// The effect stored at `id`.
    #[must_use = "a missing effect has not been recorded"]
    pub fn get(&self, id: EffectId) -> Option<&EvidenceEffect> {
        self.recorded.iter().find_map(|entry| (entry.id == id).then_some(&entry.effect))
    }

    /// Record `effect`. Recording the same effect again returns its identity.
    ///
    /// # Errors
    ///
    /// Returns an error when the ledger already holds [`MAX_RECORDED_EFFECTS`]
    /// distinct effects. A pooling effect is also refused when dependence and
    /// ledger overlap have not been consulted, when an independent pool shares
    /// rows or another dependence factor, or when a dependent pool was
    /// requested for sources that are independent.
    #[must_use = "recording the effect is what later combination checks"]
    pub fn record(&mut self, effect: EvidenceEffect) -> Result<EffectId, EvidenceEffectError> {
        let id = effect.identity();
        if self.get(id).is_some() {
            return Ok(id);
        }
        self.check_pooling(&effect)?;
        if self.recorded.len() >= MAX_RECORDED_EFFECTS {
            return Err(EvidenceEffectError::limit());
        }
        self.recorded.push(RecordedEffect { id, effect });
        Ok(id)
    }

    /// Admit numerical combination of `effect`.
    ///
    /// Succeeds only when this ledger already holds that effect. A different
    /// operator on the same subjects is not that effect.
    ///
    /// # Errors
    ///
    /// Returns an error when the effect is not recorded. The next attempt is
    /// to record it, then combine. When a different operator is already
    /// recorded on the same subjects, the error names that the operators are
    /// not interchangeable and the next attempt is to record this effect,
    /// then combine. A recorded pool is refused when dependence was not
    /// consulted, when the sources share rows and the pool is not recorded as
    /// dependent, or when overlap is unknown.
    #[must_use = "numerical combination without a recorded effect is refused"]
    pub fn try_combine(
        &self,
        effect: &EvidenceEffect,
    ) -> Result<NumericalAdmission, EvidenceEffectError> {
        let id = effect.identity();
        if self.get(id).is_some() {
            self.check_pooling(effect)?;
            return Ok(NumericalAdmission {
                effect: id,
                operator: effect.operator(),
                dependent: effect.is_dependent_pool(),
            });
        }
        if let Some(other) = different_operator_on_same_subjects(&self.recorded, effect) {
            return Err(EvidenceEffectError::not_interchangeable(
                other.operator(),
                effect.operator(),
            ));
        }
        Err(EvidenceEffectError::not_recorded(effect.operator()))
    }

    /// Combine the recorded effect `id` as `requested`.
    ///
    /// # Errors
    ///
    /// Returns an error when `id` is not on the ledger. The next attempt is
    /// to record `requested`, then combine. When `id` was recorded under a
    /// different operator, the error names that the operators are not
    /// interchangeable and the next attempt is to record `requested`, then
    /// combine. A recorded pool is checked against the consulted dependence
    /// contract and ledger overlap, the same way [`Self::try_combine`] checks.
    #[must_use = "a different operator does not admit this combination"]
    pub fn try_combine_as(
        &self,
        id: EffectId,
        requested: EvidenceOperator,
    ) -> Result<NumericalAdmission, EvidenceEffectError> {
        match self.get(id) {
            None => Err(EvidenceEffectError::not_recorded(requested)),
            Some(effect) if effect.operator().licenses(requested) => {
                self.check_pooling(effect)?;
                Ok(NumericalAdmission {
                    effect: id,
                    operator: requested,
                    dependent: effect.is_dependent_pool(),
                })
            }
            Some(effect) => {
                Err(EvidenceEffectError::not_interchangeable(effect.operator(), requested))
            }
        }
    }

    fn check_pooling(&self, effect: &EvidenceEffect) -> Result<(), EvidenceEffectError> {
        let EffectBody::Pooling { sources, dependent } = &effect.body else {
            return Ok(());
        };
        let mut paired = Vec::new();
        let mut overlaps = Vec::new();
        for (index, source) in sources.iter().enumerate() {
            let found = self.consulted.iter().find(|item| item.source() == *source);
            let Some(found) = found else {
                return Err(EvidenceEffectError::dependence_not_consulted());
            };
            paired.push(found.dependence().clone());
            for other in sources.iter().skip(index + 1) {
                match self.stored_overlap(*source, *other) {
                    Some(overlap) => overlaps.push(overlap),
                    None => return Err(EvidenceEffectError::dependence_not_consulted()),
                }
            }
        }
        let contract = DependenceContract::assess(&paired)
            .map_err(|error| EvidenceEffectError::from_dependence(&error))?;
        let shared_rows = shares_rows(&contract, &overlaps);
        let unknown = overlap_unknown(&contract, &overlaps) && !shared_rows;
        if unknown {
            return Err(EvidenceEffectError::overlap_unknown());
        }
        let dependent_relation = shared_rows || contract.is_dependent();
        if *dependent {
            if dependent_relation { Ok(()) } else { Err(EvidenceEffectError::dependence_absent()) }
        } else if shared_rows {
            Err(EvidenceEffectError::shared_rows())
        } else if contract.is_dependent() {
            Err(EvidenceEffectError::dependent_sources())
        } else {
            Ok(())
        }
    }

    fn stored_overlap(&self, left: SourceId, right: SourceId) -> Option<Overlap> {
        self.overlaps.iter().find_map(|item| {
            let same = (item.left == left && item.right == right)
                || (item.left == right && item.right == left);
            same.then_some(item.overlap)
        })
    }
}

impl Default for EvidenceEffectLedger {
    fn default() -> Self {
        Self::new()
    }
}

/// A refused effect or a refused numerical combination.
///
/// A missing or substituted combination stores the next attempt. A malformed
/// effect does not.
#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use = "the next attempt is to record the effect, then combine"]
pub struct EvidenceEffectError {
    code: &'static str,
    detail: &'static str,
    message: String,
    next: Option<NextAttempt>,
    recorded: Option<EvidenceOperator>,
    requested: Option<EvidenceOperator>,
}

impl EvidenceEffectError {
    /// Registered reason code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail, `evidence_effects.<rule>`.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }

    /// What was refused, including the next attempt when combination was refused.
    #[must_use]
    pub fn message(&self) -> &str {
        self.message.as_str()
    }

    /// The refusal as `(code, detail, explanation)`.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, &'static str, &str) {
        self.as_refusal().into_parts()
    }

    /// The next attempt, when combination was refused.
    ///
    /// Present when the effect was not recorded or another operator was
    /// offered in its place. The attempt is [`RECORD_THEN_COMBINE.sentence()`].
    #[must_use = "the next attempt is to record the effect, then combine"]
    pub const fn next_attempt(&self) -> Option<NextAttempt> {
        self.next
    }

    /// Operator already recorded, when this refusal is a substitution.
    #[must_use = "a substitution names the operator that was recorded"]
    pub const fn recorded_operator(&self) -> Option<EvidenceOperator> {
        self.recorded
    }

    /// Operator the combination asked for, when combination was refused.
    #[must_use = "a refused combination names the operator it asked for"]
    pub const fn requested_operator(&self) -> Option<EvidenceOperator> {
        self.requested
    }

    fn invalid(detail: &'static str, message: &'static str) -> Self {
        Self {
            code: antecedent_core::reason_code!("invalid_argument"),
            detail,
            message: message.to_owned(),
            next: None,
            recorded: None,
            requested: None,
        }
    }

    fn shared(detail: &'static str, message: &'static str) -> Self {
        Self {
            code: antecedent_core::reason_code!("evidence_double_counted"),
            detail,
            message: message.to_owned(),
            next: None,
            recorded: None,
            requested: None,
        }
    }

    fn limit() -> Self {
        Self::invalid(LIMIT_DETAIL, LIMIT_MESSAGE)
    }

    fn from_dependence(error: &DependenceError) -> Self {
        Self {
            code: error.code(),
            detail: error.detail(),
            message: error.message().to_owned(),
            next: None,
            recorded: None,
            requested: Some(EvidenceOperator::Pooling),
        }
    }

    fn ledger_overlap(error: &LedgerError) -> Self {
        Self {
            code: error.code,
            detail: LEDGER_OVERLAP_DETAIL,
            message: error.message.clone(),
            next: None,
            recorded: None,
            requested: Some(EvidenceOperator::Pooling),
        }
    }

    fn dependence_not_consulted() -> Self {
        let next = RECORD_THEN_COMBINE;
        let mut message = String::from(
            "pooling is refused until dependence and ledger overlap are consulted; next attempt: ",
        );
        message.push_str(next.sentence());
        Self {
            code: antecedent_core::reason_code!("cell_not_licensed"),
            detail: NOT_CONSULTED_DETAIL,
            message,
            next: Some(next),
            recorded: None,
            requested: Some(EvidenceOperator::Pooling),
        }
    }

    fn shared_rows() -> Self {
        Self::dependent_pool(SHARED_ROWS_DETAIL, "pooling sources share rows")
    }

    fn dependent_sources() -> Self {
        Self::dependent_pool(DEPENDENT_SOURCES_DETAIL, "pooling sources are dependent")
    }

    fn dependent_pool(detail: &'static str, prefix: &str) -> Self {
        let next = RECORD_DEPENDENT_POOLING;
        let mut message = String::from(prefix);
        message.push_str("; next attempt: ");
        message.push_str(next.sentence());
        Self {
            code: antecedent_core::reason_code!("evidence_double_counted"),
            detail,
            message,
            next: Some(next),
            recorded: None,
            requested: Some(EvidenceOperator::Pooling),
        }
    }

    fn overlap_unknown() -> Self {
        Self {
            code: antecedent_core::reason_code!("evidence_overlap_unknown"),
            detail: OVERLAP_UNKNOWN_DETAIL,
            message: String::from("pooling sources have unknown overlap"),
            next: None,
            recorded: None,
            requested: Some(EvidenceOperator::Pooling),
        }
    }

    fn dependence_absent() -> Self {
        Self::invalid(
            DEPENDENCE_ABSENT_DETAIL,
            "a dependent pool is refused when the sources are independent",
        )
    }

    fn not_recorded(operator: EvidenceOperator) -> Self {
        let next = RECORD_THEN_COMBINE;
        let mut message = String::from("numerical combination of `");
        message.push_str(operator.as_str());
        message.push_str("` is refused until the effect is recorded; next attempt: ");
        message.push_str(RECORD_THEN_COMBINE.sentence());
        Self {
            code: antecedent_core::reason_code!("cell_not_licensed"),
            detail: NOT_RECORDED_DETAIL,
            message,
            next: Some(next),
            recorded: None,
            requested: Some(operator),
        }
    }

    fn not_interchangeable(recorded: EvidenceOperator, requested: EvidenceOperator) -> Self {
        let next = RECORD_THEN_COMBINE;
        let mut message = String::from(substitution_sentence(recorded, requested));
        message.push_str(" (`");
        message.push_str(recorded.as_str());
        message.push_str("` is not `");
        message.push_str(requested.as_str());
        message.push_str("`); next attempt: ");
        message.push_str(RECORD_THEN_COMBINE.sentence());
        Self {
            code: antecedent_core::reason_code!("distribution_meaning_mismatch"),
            detail: NOT_INTERCHANGEABLE_DETAIL,
            message,
            next: Some(next),
            recorded: Some(recorded),
            requested: Some(requested),
        }
    }
}

impl fmt::Display for EvidenceEffectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.detail, self.message)
    }
}

impl Error for EvidenceEffectError {}

impl_refused!(EvidenceEffectError);

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
enum SubjectKey {
    Source(SourceId),
    Result(ResultId),
}

fn shares_rows(contract: &DependenceContract, overlaps: &[Overlap]) -> bool {
    let factor_rows =
        contract.shared_factors().iter().any(|factor| factor.kind() == SharedFactorKind::Row);
    let ledger_rows =
        overlaps.iter().any(|overlap| matches!(overlap, Overlap::SharedObservations { .. }));
    factor_rows || ledger_rows
}

fn overlap_unknown(contract: &DependenceContract, overlaps: &[Overlap]) -> bool {
    if contract.is_unknown() {
        return true;
    }
    overlaps.iter().any(|overlap| matches!(overlap, Overlap::Unknown { .. }))
}

fn write_sources(writer: &mut CanonicalWriter, sources: &[SourceId]) {
    writer.word(sources.len() as u64);
    for source in sources {
        writer.bytes(source.digest().as_bytes());
    }
}

fn canonical_sources(
    sources: impl IntoIterator<Item = SourceId>,
    minimum: usize,
    too_few_message: &'static str,
) -> Result<Box<[SourceId]>, EvidenceEffectError> {
    let mut sources: Vec<SourceId> = sources.into_iter().collect();
    if sources.len() < minimum {
        return Err(EvidenceEffectError::invalid(TOO_FEW_DETAIL, too_few_message));
    }
    if sources.len() > MAX_COMBINATION_SOURCES {
        return Err(EvidenceEffectError::invalid(TOO_MANY_DETAIL, TOO_MANY_MESSAGE));
    }
    sources.sort_unstable();
    let mut index = 1;
    while index < sources.len() {
        if sources[index] == sources[index - 1] {
            return Err(EvidenceEffectError::shared(DUPLICATE_DETAIL, DUPLICATE_MESSAGE));
        }
        index += 1;
    }
    Ok(sources.into_boxed_slice())
}

fn require_disjoint(
    left: &[SourceId],
    right: &[SourceId],
    message: &'static str,
) -> Result<(), EvidenceEffectError> {
    let mut left_index = 0;
    let mut right_index = 0;
    while left_index < left.len() && right_index < right.len() {
        if left[left_index] == right[right_index] {
            return Err(EvidenceEffectError::shared(SHARED_DETAIL, message));
        }
        if left[left_index] < right[right_index] {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    Ok(())
}

fn checked_text(text: &str) -> Result<Arc<str>, EvidenceEffectError> {
    if antecedent_core::is_blank(text) {
        return Err(EvidenceEffectError::invalid(BLANK_DETAIL, BLANK_MESSAGE));
    }
    if text.len() > MAX_TEXT_BYTES {
        return Err(EvidenceEffectError::invalid(TEXT_DETAIL, TEXT_MESSAGE));
    }
    Ok(Arc::from(text))
}

const fn substitution_sentence(
    recorded: EvidenceOperator,
    requested: EvidenceOperator,
) -> &'static str {
    match (recorded, requested) {
        (EvidenceOperator::Mixture, EvidenceOperator::Pooling)
        | (EvidenceOperator::Pooling, EvidenceOperator::Mixture) => "a mixture is not a pool",
        (EvidenceOperator::Bridge, EvidenceOperator::Transport)
        | (EvidenceOperator::Transport, EvidenceOperator::Bridge) => "a bridge is not transport",
        (EvidenceOperator::Envelope, EvidenceOperator::Marginalization)
        | (EvidenceOperator::Marginalization, EvidenceOperator::Envelope) => {
            "an envelope is not a marginal"
        }
        (
            EvidenceOperator::Pooling
            | EvidenceOperator::HierarchicalBorrowing
            | EvidenceOperator::Transport
            | EvidenceOperator::Bridge
            | EvidenceOperator::FrozenReuse
            | EvidenceOperator::RepeatedObservationUse
            | EvidenceOperator::Mixture
            | EvidenceOperator::Marginalization
            | EvidenceOperator::Envelope,
            EvidenceOperator::Pooling
            | EvidenceOperator::HierarchicalBorrowing
            | EvidenceOperator::Transport
            | EvidenceOperator::Bridge
            | EvidenceOperator::FrozenReuse
            | EvidenceOperator::RepeatedObservationUse
            | EvidenceOperator::Mixture
            | EvidenceOperator::Marginalization
            | EvidenceOperator::Envelope,
        ) => "two effects are not interchangeable",
    }
}

fn insert_sources(key: &mut BTreeSet<SubjectKey>, sources: &[SourceId]) {
    for source in sources {
        key.insert(SubjectKey::Source(*source));
    }
}

fn subject_key(effect: &EvidenceEffect) -> BTreeSet<SubjectKey> {
    let mut key = BTreeSet::new();
    match &effect.body {
        EffectBody::Pooling { sources, .. }
        | EffectBody::Mixture(sources)
        | EffectBody::Envelope(sources)
        | EffectBody::RepeatedObservationUse { sources, .. } => insert_sources(&mut key, sources),
        EffectBody::HierarchicalBorrowing { priors, likelihood } => {
            insert_sources(&mut key, priors);
            insert_sources(&mut key, likelihood);
        }
        EffectBody::Transport { transported, native } => {
            insert_sources(&mut key, transported);
            insert_sources(&mut key, native);
        }
        EffectBody::Bridge { source, .. } => {
            key.insert(SubjectKey::Source(*source));
        }
        EffectBody::FrozenReuse(result) | EffectBody::Marginalization(result) => {
            key.insert(SubjectKey::Result(*result));
        }
    }
    key
}

fn different_operator_on_same_subjects<'a>(
    recorded: &'a [RecordedEffect],
    requested: &EvidenceEffect,
) -> Option<&'a EvidenceEffect> {
    let requested_key = subject_key(requested);
    for entry in recorded {
        if entry.effect.operator().licenses(requested.operator()) {
            continue;
        }
        if subject_key(&entry.effect) == requested_key {
            return Some(&entry.effect);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        EvidenceEffect, EvidenceEffectLedger, EvidenceOperator, PooledSource,
        RECORD_DEPENDENT_POOLING, RECORD_THEN_COMBINE,
    };
    use crate::dependence::{DependenceSource, SharedFactorKind, SharedFactorSets};
    use crate::evidence_ledger::{
        EvidenceLedger, MAX_COMBINATION_SOURCES, MAX_TEXT_BYTES, ObservationId, ResultId,
        SourceDeclaration, SourceId, SourceKind, SynthesisKind,
    };
    use antecedent_core::SemanticDigest;

    const fn source_at(index: u16) -> SourceId {
        let pair = index.to_le_bytes();
        let mut bytes = [0u8; 32];
        bytes[0] = pair[0];
        bytes[1] = pair[1];
        SourceId::from_digest(SemanticDigest::from_bytes(bytes))
    }

    const fn result_at(index: u16) -> ResultId {
        let pair = index.to_le_bytes();
        let mut bytes = [0u8; 32];
        bytes[30] = pair[0];
        bytes[31] = pair[1];
        ResultId::from_digest(SemanticDigest::from_bytes(bytes))
    }

    const fn sentence(recorded: EvidenceOperator, requested: EvidenceOperator) -> &'static str {
        match (recorded, requested) {
            (EvidenceOperator::Mixture, EvidenceOperator::Pooling)
            | (EvidenceOperator::Pooling, EvidenceOperator::Mixture) => "a mixture is not a pool",
            (EvidenceOperator::Bridge, EvidenceOperator::Transport)
            | (EvidenceOperator::Transport, EvidenceOperator::Bridge) => {
                "a bridge is not transport"
            }
            (EvidenceOperator::Envelope, EvidenceOperator::Marginalization)
            | (EvidenceOperator::Marginalization, EvidenceOperator::Envelope) => {
                "an envelope is not a marginal"
            }
            (
                EvidenceOperator::Pooling
                | EvidenceOperator::HierarchicalBorrowing
                | EvidenceOperator::Transport
                | EvidenceOperator::Bridge
                | EvidenceOperator::FrozenReuse
                | EvidenceOperator::RepeatedObservationUse
                | EvidenceOperator::Mixture
                | EvidenceOperator::Marginalization
                | EvidenceOperator::Envelope,
                EvidenceOperator::Pooling
                | EvidenceOperator::HierarchicalBorrowing
                | EvidenceOperator::Transport
                | EvidenceOperator::Bridge
                | EvidenceOperator::FrozenReuse
                | EvidenceOperator::RepeatedObservationUse
                | EvidenceOperator::Mixture
                | EvidenceOperator::Marginalization
                | EvidenceOperator::Envelope,
            ) => "two effects are not interchangeable",
        }
    }

    fn independent_sources(ledger: &mut EvidenceLedger, rows: &[(&str, u32)]) -> Vec<PooledSource> {
        rows.iter()
            .map(|&(label, row)| {
                let observations = [ObservationId::from_key("rows", &row.to_le_bytes())];
                let declaration = SourceDeclaration::new(label, SourceKind::ObservationSet)
                    .with_observations(observations);
                let source = ledger.add_source(declaration).expect("source");
                let mut factors = SharedFactorSets::empty();
                factors.try_insert(SharedFactorKind::Row, label).expect("row");
                factors.try_insert(SharedFactorKind::Parameter, "theta").expect("parameter");
                let dependence =
                    DependenceSource::try_declared(label, factors).expect("dependence");
                PooledSource::new(source, dependence)
            })
            .collect()
    }

    fn sample_effects(left: SourceId, right: SourceId) -> Vec<EvidenceEffect> {
        vec![
            EvidenceEffect::pooling([left, right]).expect("pooling"),
            EvidenceEffect::hierarchical_borrowing([source_at(3)], [source_at(4)])
                .expect("hierarchical borrowing"),
            EvidenceEffect::transport([source_at(5)], [source_at(6)]).expect("transport"),
            EvidenceEffect::bridge(source_at(7), "surrogate-to-endpoint").expect("bridge"),
            EvidenceEffect::frozen_reuse(result_at(8)),
            EvidenceEffect::repeated_observation_use([source_at(9)], "validation")
                .expect("repeated"),
            EvidenceEffect::mixture([source_at(10), source_at(11)]).expect("mixture"),
            EvidenceEffect::marginalization(result_at(12)),
            EvidenceEffect::envelope([source_at(13), source_at(14)]).expect("envelope"),
        ]
    }

    #[test]
    fn each_operator_records_an_effect_before_combination() {
        let forward = EvidenceEffect::pooling([source_at(1), source_at(2)]).expect("forward");
        let backward = EvidenceEffect::pooling([source_at(2), source_at(1)]).expect("backward");
        assert_eq!(forward, backward);
        assert_eq!(forward.identity(), backward.identity());

        assert_eq!(
            EvidenceOperator::Pooling.synthesis_kind(),
            Some(SynthesisKind::StatisticalPooling)
        );
        assert_eq!(
            EvidenceOperator::HierarchicalBorrowing.synthesis_kind(),
            Some(SynthesisKind::BayesianBorrowing)
        );
        assert_eq!(
            EvidenceOperator::Transport.synthesis_kind(),
            Some(SynthesisKind::CausalTransport)
        );
        assert_eq!(
            EvidenceOperator::FrozenReuse.synthesis_kind(),
            Some(SynthesisKind::EvidenceReuse)
        );
        assert_eq!(
            EvidenceOperator::RepeatedObservationUse.synthesis_kind(),
            Some(SynthesisKind::RepeatedObservationUse)
        );
        assert_eq!(EvidenceOperator::Bridge.synthesis_kind(), None);
        assert_eq!(EvidenceOperator::Mixture.synthesis_kind(), None);
        assert_eq!(EvidenceOperator::Marginalization.synthesis_kind(), None);
        assert_eq!(EvidenceOperator::Envelope.synthesis_kind(), None);
        assert!(EvidenceOperator::from_name("statistical_pooling").is_none());
        assert!(EvidenceOperator::from_name("marginal").is_none());

        let mut evidence = EvidenceLedger::new();
        let pooled = independent_sources(&mut evidence, &[("study-a", 1), ("study-b", 2)]);
        let effects = sample_effects(pooled[0].source(), pooled[1].source());
        let mut ledger = EvidenceEffectLedger::new();
        ledger.consult(&pooled, &evidence).expect("consult");
        assert_eq!(effects.len(), EvidenceOperator::ALL.len());
        let mut seen = vec![false; EvidenceOperator::ALL.len()];
        for effect in &effects {
            let operator = effect.operator();
            assert_eq!(EvidenceOperator::from_name(operator.as_str()), Some(operator));
            let index =
                EvidenceOperator::ALL.iter().position(|item| *item == operator).expect("listed");
            assert!(!seen[index], "{}", operator.as_str());
            seen[index] = true;

            let missing = ledger.try_combine(effect).expect_err("before record");
            assert_eq!(missing.code(), "cell_not_licensed");
            assert_eq!(missing.detail(), "evidence_effects.effect_not_recorded");
            assert!(missing.message().contains(RECORD_THEN_COMBINE.sentence()));
            assert_eq!(missing.requested_operator().expect("requested"), operator);

            let id = ledger.record(effect.clone()).expect("record");
            assert_eq!(ledger.record(effect.clone()).expect("idempotent"), id);
            let admission = ledger.try_combine(effect).expect("after record");
            assert_eq!(admission.effect_id(), id);
            assert_eq!(admission.operator(), operator);
            assert_eq!(admission.synthesis_kind(), operator.synthesis_kind());
            let same = ledger.try_combine_as(id, operator).expect("same operator");
            assert_eq!(same.effect_id(), id);
        }
        assert!(seen.iter().all(|flag| *flag));
        assert_eq!(ledger.recorded().len(), EvidenceOperator::ALL.len());

        let pool = &effects[0];
        assert!(pool.pooling_sources().is_some());
        assert!(pool.mixture_components().is_none());
        let mixture = &effects[6];
        assert!(mixture.mixture_components().is_some());
        assert!(mixture.pooling_sources().is_none());
        assert!(effects[3].bridge_source().is_some());
        assert!(effects[3].transported_sources().is_none());
        assert!(effects[2].transported_sources().is_some());
        assert!(effects[2].bridge_contract().is_none());
        assert!(effects[8].envelope_sources().is_some());
        assert!(effects[8].marginalized_result().is_none());
        assert!(effects[7].marginalized_result().is_some());
        assert!(effects[7].envelope_sources().is_none());
        assert!(effects[1].borrowing_priors().is_some());
        assert!(effects[1].borrowing_likelihood().is_some());
        assert!(effects[4].frozen_result().is_some());
        assert_eq!(effects[5].repeated_stage(), Some("validation"));
    }

    #[test]
    fn combination_without_a_recorded_effect_is_refused() {
        let mut evidence = EvidenceLedger::new();
        let pooled = independent_sources(&mut evidence, &[("left", 1), ("right", 2)]);
        let effect =
            EvidenceEffect::pooling([pooled[0].source(), pooled[1].source()]).expect("pool");
        let mut ledger = EvidenceEffectLedger::new();
        let err = ledger.try_combine(&effect).expect_err("no effect");
        assert_eq!(err.code(), "cell_not_licensed");
        assert_eq!(err.detail(), "evidence_effects.effect_not_recorded");
        assert_eq!(
            err.refusal(),
            ("cell_not_licensed", "evidence_effects.effect_not_recorded", err.message(),)
        );
        assert!(err.message().contains("refused until the effect is recorded"));
        assert!(err.message().contains(RECORD_THEN_COMBINE.sentence()));
        assert_eq!(err.next_attempt().expect("next").to_string(), RECORD_THEN_COMBINE.sentence());
        assert_eq!(err.requested_operator().expect("requested"), EvidenceOperator::Pooling);
        assert!(err.recorded_operator().is_none());
        assert_eq!(err.requested_operator(), Some(EvidenceOperator::Pooling));
        assert!(antecedent_core::reason_code::is_runtime_refusal(err.code()));
        assert!(ledger.recorded().is_empty());

        let missing_dependence = ledger.record(effect.clone()).expect_err("not consulted");
        assert_eq!(missing_dependence.detail(), "evidence_effects.dependence_not_consulted");
        assert!(missing_dependence.message().contains("consulted"));
        assert_eq!(
            missing_dependence.requested_operator().expect("requested"),
            EvidenceOperator::Pooling
        );

        ledger.consult(&pooled, &evidence).expect("consult");
        let id = ledger.record(effect.clone()).expect("record");
        let admission = ledger.try_combine(&effect).expect("recorded");
        assert_eq!(admission.effect_id(), id);
        assert_eq!(admission.operator(), EvidenceOperator::Pooling);
        assert!(!admission.is_dependent());
        assert_eq!(admission.synthesis_kind(), Some(SynthesisKind::StatisticalPooling));
    }

    #[test]
    fn mixture_is_not_a_pool_and_bridge_is_not_transport() {
        let mut evidence = EvidenceLedger::new();
        let pooled = independent_sources(&mut evidence, &[("mix-a", 10), ("mix-b", 11)]);
        let sources = [pooled[0].source(), pooled[1].source()];
        let mixture = EvidenceEffect::mixture(sources).expect("mixture");
        let pool = EvidenceEffect::pooling(sources).expect("pool");
        assert_ne!(mixture, pool);
        assert_ne!(mixture.identity(), pool.identity());
        assert!(mixture.pooling_sources().is_none());
        assert!(pool.mixture_components().is_none());
        assert!(!EvidenceOperator::Mixture.licenses(EvidenceOperator::Pooling));
        assert!(!EvidenceOperator::Bridge.licenses(EvidenceOperator::Transport));
        assert!(!EvidenceOperator::Envelope.licenses(EvidenceOperator::Marginalization));

        let mut ledger = EvidenceEffectLedger::new();
        let mixture_id = ledger.record(mixture).expect("record mixture");
        let err = ledger.try_combine(&pool).expect_err("pool");
        assert_eq!(err.code(), "distribution_meaning_mismatch");
        assert_eq!(err.detail(), "evidence_effects.operators_not_interchangeable");
        assert!(err.message().contains("a mixture is not a pool"));
        assert!(err.message().contains(RECORD_THEN_COMBINE.sentence()));
        assert_eq!(err.recorded_operator(), Some(EvidenceOperator::Mixture));
        assert_eq!(err.requested_operator(), Some(EvidenceOperator::Pooling));
        assert_eq!(err.requested_operator().expect("requested"), EvidenceOperator::Pooling);
        assert!(antecedent_core::reason_code::is_runtime_refusal(err.code()));

        let as_pool =
            ledger.try_combine_as(mixture_id, EvidenceOperator::Pooling).expect_err("as pool");
        assert!(as_pool.message().contains("a mixture is not a pool"));
        assert_eq!(as_pool.requested_operator().expect("requested"), EvidenceOperator::Pooling);

        let bridge = EvidenceEffect::bridge(source_at(3), "endpoint").expect("bridge");
        let bridge_id = ledger.record(bridge).expect("record bridge");
        let as_transport = ledger
            .try_combine_as(bridge_id, EvidenceOperator::Transport)
            .expect_err("bridge as transport");
        assert!(as_transport.message().contains("a bridge is not transport"));
        assert_eq!(as_transport.recorded_operator(), Some(EvidenceOperator::Bridge));
        assert_eq!(as_transport.requested_operator(), Some(EvidenceOperator::Transport));
        assert_eq!(
            as_transport.requested_operator().expect("requested"),
            EvidenceOperator::Transport
        );

        let envelope = EvidenceEffect::envelope([source_at(4)]).expect("envelope");
        let envelope_id = ledger.record(envelope).expect("record envelope");
        let as_marginal = ledger
            .try_combine_as(envelope_id, EvidenceOperator::Marginalization)
            .expect_err("envelope as marginal");
        assert!(as_marginal.message().contains("an envelope is not a marginal"));
        assert_eq!(as_marginal.recorded_operator(), Some(EvidenceOperator::Envelope));
        assert_eq!(as_marginal.requested_operator(), Some(EvidenceOperator::Marginalization));
        assert_eq!(
            as_marginal.requested_operator().expect("requested"),
            EvidenceOperator::Marginalization
        );

        let reused = EvidenceEffect::frozen_reuse(result_at(1));
        let marginal = EvidenceEffect::marginalization(result_at(1));
        assert_ne!(reused.identity(), marginal.identity());
        ledger.record(reused).expect("record reuse");
        let not_marginal = ledger.try_combine(&marginal).expect_err("reuse is not a marginal");
        assert!(not_marginal.message().contains("two effects are not interchangeable"));
        assert!(not_marginal.message().contains(RECORD_THEN_COMBINE.sentence()));
        assert_eq!(not_marginal.recorded_operator(), Some(EvidenceOperator::FrozenReuse));
        assert_eq!(not_marginal.requested_operator(), Some(EvidenceOperator::Marginalization));

        ledger.consult(&pooled, &evidence).expect("consult");
        let pool_id = ledger.record(pool.clone()).expect("record pool");
        let admission = ledger.try_combine(&pool).expect("pool recorded");
        assert!(!admission.is_dependent());
        assert_eq!(admission.effect_id(), pool_id);
        assert_ne!(admission.effect_id(), mixture_id);
    }

    #[test]
    fn operators_are_not_interchangeable() {
        for recorded in EvidenceOperator::ALL {
            for requested in EvidenceOperator::ALL {
                let refusal = recorded.substitution_refusal(requested);
                if recorded.licenses(requested) {
                    assert!(refusal.is_none());
                    continue;
                }
                let err = refusal.expect("distinct operators");
                assert!(err.message().contains(RECORD_THEN_COMBINE.sentence()));
                assert_eq!(err.recorded_operator(), Some(recorded));
                assert_eq!(err.requested_operator(), Some(requested));
                assert_eq!(err.requested_operator().expect("requested"), requested);
                assert!(err.message().contains(sentence(recorded, requested)), "{}", err.message(),);
            }
        }
    }

    #[test]
    fn malformed_effects_are_refused() {
        let few = EvidenceEffect::pooling([source_at(1)]).expect_err("one source");
        assert_eq!(few.detail(), "evidence_effects.too_few_subjects");
        assert_eq!(few.code(), "invalid_argument");
        assert!(few.next_attempt().is_none());

        let twice = EvidenceEffect::mixture([source_at(1), source_at(1)]).expect_err("twice");
        assert_eq!(twice.code(), "evidence_double_counted");
        assert_eq!(twice.detail(), "evidence_effects.duplicate_source");

        let shared = EvidenceEffect::hierarchical_borrowing([source_at(1)], [source_at(1)])
            .expect_err("prior and likelihood");
        assert_eq!(shared.detail(), "evidence_effects.shared_source");
        assert_eq!(shared.code(), "evidence_double_counted");

        let native = EvidenceEffect::transport([source_at(2)], [source_at(2)]).expect_err("native");
        assert_eq!(native.detail(), "evidence_effects.shared_source");
        assert!(native.message().contains("transported and native"));

        let blank = EvidenceEffect::bridge(source_at(1), " ").expect_err("blank");
        assert_eq!(blank.detail(), "evidence_effects.blank_text");
        assert!(blank.next_attempt().is_none());

        let long = "y".repeat(MAX_TEXT_BYTES + 1);
        let oversized =
            EvidenceEffect::repeated_observation_use([source_at(1)], &long).expect_err("long");
        assert_eq!(oversized.detail(), "evidence_effects.text_too_long");

        let many: Vec<SourceId> =
            (0..=u16::try_from(MAX_COMBINATION_SOURCES).expect("bound")).map(source_at).collect();
        let too_many = EvidenceEffect::envelope(many).expect_err("too many");
        assert_eq!(too_many.detail(), "evidence_effects.too_many_sources");
    }

    fn study(
        ledger: &mut EvidenceLedger,
        label: &str,
        rows: &[u32],
        factor_row: &str,
    ) -> PooledSource {
        let observations: Vec<ObservationId> =
            rows.iter().map(|row| ObservationId::from_key("rows", &row.to_le_bytes())).collect();
        let declaration = SourceDeclaration::new(label, SourceKind::ObservationSet)
            .with_observations(observations);
        let source = ledger.add_source(declaration).expect("source");
        let mut factors = SharedFactorSets::empty();
        factors.try_insert(SharedFactorKind::Row, factor_row).expect("row");
        factors.try_insert(SharedFactorKind::Parameter, "Theta").expect("parameter");
        let dependence = DependenceSource::try_declared(label, factors).expect("dependence");
        PooledSource::new(source, dependence)
    }

    #[test]
    fn pooling_shared_rows_are_refused_or_recorded_as_dependent() {
        let mut evidence = EvidenceLedger::new();
        let left = study(&mut evidence, "ledger-a", &[1, 2], "factor-a");
        let right = study(&mut evidence, "ledger-b", &[2, 3], "factor-b");
        let pool = EvidenceEffect::pooling([left.source(), right.source()]).expect("pool");
        let dependent =
            EvidenceEffect::dependent_pooling([left.source(), right.source()]).expect("dependent");
        assert_ne!(pool.identity(), dependent.identity());
        let mut ledger = EvidenceEffectLedger::new();
        ledger.consult(&[left, right], &evidence).expect("consult");

        let refused = ledger.record(pool.clone()).expect_err("shared ledger rows");
        assert_eq!(refused.detail(), "evidence_effects.shared_rows");
        assert_eq!(refused.code(), "evidence_double_counted");
        assert!(refused.message().contains("share rows"));
        assert_eq!(refused.next_attempt(), Some(RECORD_DEPENDENT_POOLING));
        assert!(ledger.recorded().is_empty());

        let id = ledger.record(dependent.clone()).expect("record dependent");
        let admission = ledger.try_combine(&dependent).expect("combine dependent");
        assert!(admission.is_dependent());
        assert_eq!(admission.effect_id(), id);
        let still = ledger.try_combine(&pool).expect_err("independent pool was not recorded");
        assert_eq!(still.detail(), "evidence_effects.effect_not_recorded");

        let factor_left = study(&mut evidence, "factor-left", &[10], "shared-row");
        let factor_right = study(&mut evidence, "factor-right", &[20], "shared-row");
        let factor_pool = EvidenceEffect::pooling([factor_left.source(), factor_right.source()])
            .expect("factor pool");
        ledger.consult(&[factor_left, factor_right], &evidence).expect("consult factors");
        let factor_refusal = ledger.record(factor_pool).expect_err("shared factor row");
        assert_eq!(factor_refusal.detail(), "evidence_effects.shared_rows");

        let opaque = ledger_source_without_rows(&mut evidence, "opaque");
        let known = study(&mut evidence, "known-rows", &[30], "known-row");
        let unknown_pool =
            EvidenceEffect::pooling([opaque.source(), known.source()]).expect("unknown pool");
        ledger.consult(&[opaque, known], &evidence).expect("consult unknown");
        let unknown = ledger.record(unknown_pool).expect_err("unknown overlap");
        assert_eq!(unknown.detail(), "evidence_effects.overlap_unknown");
        assert_eq!(unknown.code(), "evidence_overlap_unknown");
    }

    fn ledger_source_without_rows(ledger: &mut EvidenceLedger, label: &str) -> PooledSource {
        let source = ledger
            .add_source(SourceDeclaration::new(label, SourceKind::ObservationSet))
            .expect("source");
        let dependence =
            DependenceSource::try_declared(label, SharedFactorSets::empty()).expect("dependence");
        PooledSource::new(source, dependence)
    }

    #[test]
    fn shared_parameter_does_not_make_an_independent_pool_dependent() {
        let mut evidence = EvidenceLedger::new();
        let pooled = independent_sources(&mut evidence, &[("study-a", 1), ("study-b", 2)]);
        let effect =
            EvidenceEffect::pooling([pooled[0].source(), pooled[1].source()]).expect("pool");
        let dependent = EvidenceEffect::dependent_pooling([pooled[0].source(), pooled[1].source()])
            .expect("dependent");
        let mut ledger = EvidenceEffectLedger::new();
        ledger.consult(&pooled, &evidence).expect("consult");
        ledger.record(effect.clone()).expect("independent studies");
        let admission = ledger.try_combine(&effect).expect("combine");
        assert!(!admission.is_dependent());
        assert!(!effect.is_dependent_pool());
        let false_dependence = ledger.record(dependent).expect_err("no dependence to record");
        assert_eq!(false_dependence.detail(), "evidence_effects.dependence_not_found");

        let shared = pooled
            .iter()
            .map(|source| {
                let mut factors = SharedFactorSets::empty();
                factors.try_insert(SharedFactorKind::Row, "same-row").expect("row");
                let dependence =
                    DependenceSource::try_declared(source.dependence().id().as_str(), factors)
                        .expect("dependence");
                PooledSource::new(source.source(), dependence)
            })
            .collect::<Vec<_>>();
        ledger.consult(&shared, &evidence).expect("consult shared rows");
        let after = ledger.try_combine(&effect).expect_err("combine checks dependence");
        assert_eq!(after.detail(), "evidence_effects.shared_rows");
        assert_eq!(after.next_attempt(), Some(RECORD_DEPENDENT_POOLING));
    }

    #[test]
    fn consult_refuses_more_sources_than_the_combination_cap() {
        let mut sources = Vec::new();
        for index in 0..=MAX_COMBINATION_SOURCES {
            let dependence =
                DependenceSource::try_declared(index.to_string(), SharedFactorSets::empty())
                    .expect("source");
            let id = source_at(u16::try_from(index).expect("index"));
            sources.push(PooledSource::new(id, dependence));
        }
        let mut ledger = EvidenceEffectLedger::new();
        let refusal =
            ledger.consult(&sources, &EvidenceLedger::new()).expect_err("over the source cap");
        assert_eq!(refusal.detail(), "evidence_effects.too_many_sources");
        assert!(refusal.message().contains(&MAX_COMBINATION_SOURCES.to_string()));
    }
}
