// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checked inference realizations.
//!
//! [`OrdinaryBayesianLikelihood`], [`ScoreFrequentist`], [`GeneralizedBayes`], and
//! [`PartialIdentificationEnvelope`] are different objects. No conversion turns one into
//! another. A prior is a belief inside a posterior, not a license for observational
//! identification of a causal effect.
//!
//! [`Prior::as_observational_identification`] refuses that license and names the two
//! legal next attempts: attach a named identifying restriction on a separate
//! [`AssumptionBranch`], or report an [`OrdinaryBayesianLikelihood`] posterior under
//! the declared likelihood. [`OrdinaryBayesianLikelihood::with_identifying_restriction`]
//! does the first. The restriction enters only the branch identity. The posterior
//! identity stays the likelihood and the prior.
//!
//! [`GeneralizedBayes`] records a loss, a prior, and an explicit learning rate `w > 0`,
//! the weight in Bissiri, Holmes and Walker (2016, arXiv 1306.6430), Sections 2 and 3.
//! Construction does not minimize their cumulative loss, does not calibrate `w`, and
//! does not rewrite self-information loss into an ordinary likelihood. A
//! [`PartialIdentificationEnvelope`] is a set. [`PosteriorSample::accept_envelope`]
//! refuses it.

use std::fmt;
use std::sync::Arc;

use crate::identity::{CanonicalWriter, SemanticDigest};
use crate::text::is_blank;

/// Identity tag for an ordinary Bayesian likelihood posterior.
pub const ORDINARY_BAYESIAN_LIKELIHOOD_TAG: &str =
    "antecedent.inference_realization.ordinary_bayesian_likelihood.v1";

/// Identity tag for a score-based frequentist realization.
pub const SCORE_FREQUENTIST_TAG: &str = "antecedent.inference_realization.score_frequentist.v1";

/// Identity tag for a generalized-Bayes realization.
pub const GENERALIZED_BAYES_TAG: &str = "antecedent.inference_realization.generalized_bayes.v1";

/// Identity tag for a partial-identification envelope.
pub const PARTIAL_IDENTIFICATION_ENVELOPE_TAG: &str =
    "antecedent.inference_realization.partial_identification_envelope.v1";

/// Identity tag for the assumption branch that holds an identifying restriction.
pub const ASSUMPTION_BRANCH_TAG: &str = "antecedent.inference_realization.assumption_branch.v1";

/// A declaration name was empty or whitespace.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, thiserror::Error)]
#[non_exhaustive]
pub enum DeclarationError {
    /// The name was empty or whitespace.
    #[error("declaration name must be non-blank")]
    BlankName,
}

/// Declared likelihood of an ordinary Bayesian posterior.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct LikelihoodModel {
    name: Arc<str>,
}

impl LikelihoodModel {
    /// Likelihood named `name`.
    ///
    /// # Errors
    ///
    /// [`DeclarationError::BlankName`] when `name` is empty or whitespace.
    #[must_use = "a blank likelihood name is refused"]
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, DeclarationError> {
        Ok(Self { name: declared_name(name)? })
    }

    /// Declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for LikelihoodModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Declared score or estimating equation.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ScoreModel {
    name: Arc<str>,
}

impl ScoreModel {
    /// Score named `name`.
    ///
    /// # Errors
    ///
    /// [`DeclarationError::BlankName`] when `name` is empty or whitespace.
    #[must_use = "a blank score name is refused"]
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, DeclarationError> {
        Ok(Self { name: declared_name(name)? })
    }

    /// Declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for ScoreModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Declared loss of a generalized posterior.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct LossModel {
    name: Arc<str>,
}

impl LossModel {
    /// Loss named `name`.
    ///
    /// # Errors
    ///
    /// [`DeclarationError::BlankName`] when `name` is empty or whitespace.
    #[must_use = "a blank loss name is refused"]
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, DeclarationError> {
        Ok(Self { name: declared_name(name)? })
    }

    /// Declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for LossModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Prior belief used by a posterior realization.
///
/// The prior narrows that posterior. It does not identify a causal effect from
/// the observations. [`Self::as_observational_identification`] is the refused attempt.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Prior {
    name: Arc<str>,
}

impl Prior {
    /// Prior named `name`.
    ///
    /// # Errors
    ///
    /// [`DeclarationError::BlankName`] when `name` is empty or whitespace.
    #[must_use = "a blank prior name is refused"]
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, DeclarationError> {
        Ok(Self { name: declared_name(name)? })
    }

    /// Declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Refuse this prior as observational identification of a causal effect.
    ///
    /// The attempt has no success value. The refusal names the two legal next
    /// attempts: attach a named restriction with
    /// [`OrdinaryBayesianLikelihood::with_identifying_restriction`], or report an
    /// [`OrdinaryBayesianLikelihood`] under the declared likelihood.
    #[must_use = "a prior alone is not observational identification"]
    pub fn as_observational_identification(&self) -> PriorIdentificationRefusal {
        PriorIdentificationRefusal {
            prior: self.clone(),
            next_attempts: NextIdentificationAttempt::ALL,
        }
    }
}

impl fmt::Display for Prior {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Identifying restriction attached on an assumption branch.
///
/// The restriction is not a prior and is not part of the posterior identity.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct IdentifyingRestriction {
    name: Arc<str>,
}

impl IdentifyingRestriction {
    /// Restriction named `name`.
    ///
    /// # Errors
    ///
    /// [`DeclarationError::BlankName`] when `name` is empty or whitespace.
    #[must_use = "a blank restriction name is refused"]
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, DeclarationError> {
        Ok(Self { name: declared_name(name)? })
    }

    /// Declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for IdentifyingRestriction {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Learning rate `w` of a generalized posterior.
///
/// Finite and strictly positive. Zero, negative zero, negatives, and non-finite
/// values are refused. The value is stored by its IEEE-754 bit pattern.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct LearningRate {
    bits: u64,
}

impl LearningRate {
    /// Rate `value` when it is finite and strictly positive.
    #[must_use = "a non-finite or non-positive value is not a learning rate"]
    pub const fn try_new(value: f64) -> Option<Self> {
        if value.is_finite() && value > 0.0 { Some(Self { bits: value.to_bits() }) } else { None }
    }

    /// The rate as an `f64`.
    #[must_use]
    pub const fn get(self) -> f64 {
        f64::from_bits(self.bits)
    }

    /// IEEE-754 bit pattern of [`Self::get`].
    #[must_use]
    pub const fn to_bits(self) -> u64 {
        self.bits
    }
}

impl fmt::Display for LearningRate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rate = self.get();
        write!(formatter, "{rate}")
    }
}

/// Which of the four realizations a value is.
///
/// The variants do not share a result type. Matching on this kind does not
/// convert one realization into another.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RealizationKind {
    /// [`OrdinaryBayesianLikelihood`].
    OrdinaryBayesianLikelihood,
    /// [`ScoreFrequentist`].
    ScoreFrequentist,
    /// [`GeneralizedBayes`].
    GeneralizedBayes,
    /// [`PartialIdentificationEnvelope`].
    PartialIdentificationEnvelope,
}

impl RealizationKind {
    /// The four realizations, in report order.
    pub const ALL: [Self; 4] = [
        Self::OrdinaryBayesianLikelihood,
        Self::ScoreFrequentist,
        Self::GeneralizedBayes,
        Self::PartialIdentificationEnvelope,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OrdinaryBayesianLikelihood => "ordinary_bayesian_likelihood",
            Self::ScoreFrequentist => "score_frequentist",
            Self::GeneralizedBayes => "generalized_bayes",
            Self::PartialIdentificationEnvelope => "partial_identification_envelope",
        }
    }
}

impl fmt::Display for RealizationKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Ordinary Bayesian posterior: a declared likelihood and a prior.
///
/// The posterior is not an identified causal effect. A learning rate is not an
/// ingredient. Self-information loss with a learning rate stays [`GeneralizedBayes`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct OrdinaryBayesianLikelihood {
    likelihood: LikelihoodModel,
    prior: Prior,
}

impl OrdinaryBayesianLikelihood {
    /// Posterior under `likelihood` with `prior`.
    #[must_use]
    pub const fn new(likelihood: LikelihoodModel, prior: Prior) -> Self {
        Self { likelihood, prior }
    }

    /// Realization kind. Always [`RealizationKind::OrdinaryBayesianLikelihood`].
    #[must_use]
    pub const fn kind(&self) -> RealizationKind {
        RealizationKind::OrdinaryBayesianLikelihood
    }

    /// Declared likelihood.
    #[must_use]
    pub const fn likelihood(&self) -> &LikelihoodModel {
        &self.likelihood
    }

    /// Prior. This is not an identifying restriction.
    #[must_use]
    pub const fn prior(&self) -> &Prior {
        &self.prior
    }

    /// Attach `restriction` on a separate assumption branch.
    ///
    /// The returned posterior is this value. Its identity does not read the
    /// restriction. The restriction's identity is [`AssumptionBranch::identity`].
    #[must_use = "the identifying restriction is recorded on a separate assumption branch"]
    pub const fn with_identifying_restriction(
        self,
        restriction: IdentifyingRestriction,
    ) -> RestrictedPosterior {
        RestrictedPosterior { posterior: self, assumption_branch: AssumptionBranch { restriction } }
    }

    /// Digest of the likelihood and the prior.
    ///
    /// An identifying restriction is not an input.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(ORDINARY_BAYESIAN_LIKELIHOOD_TAG);
        writer.text(self.likelihood.name()).text(self.prior.name());
        SemanticDigest::from_bytes(writer.finish())
    }
}

/// Score-based frequentist realization.
///
/// A score estimate is not a posterior and not the large-`w` limit of
/// [`GeneralizedBayes`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ScoreFrequentist {
    score: ScoreModel,
}

impl ScoreFrequentist {
    /// Realization of `score`.
    #[must_use]
    pub const fn new(score: ScoreModel) -> Self {
        Self { score }
    }

    /// Realization kind. Always [`RealizationKind::ScoreFrequentist`].
    #[must_use]
    pub const fn kind(&self) -> RealizationKind {
        RealizationKind::ScoreFrequentist
    }

    /// Declared score.
    #[must_use]
    pub const fn score(&self) -> &ScoreModel {
        &self.score
    }

    /// Digest of the score name.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(SCORE_FREQUENTIST_TAG);
        writer.text(self.score.name());
        SemanticDigest::from_bytes(writer.finish())
    }
}

/// Generalized posterior: a loss, a prior, and an explicit learning rate.
///
/// Bissiri, Holmes and Walker (2016), Sections 2 and 3, update beliefs by
/// minimizing a cumulative loss with weight `w` on the loss-to-data. This type
/// records those three ingredients. It does not compute the minimizer.
///
/// Section 2.1 recovers ordinary Bayes from self-information loss. That rewrite
/// is not applied here, including when `w = 1`. Section 3 does not choose `w`:
/// annealing, unit information, hierarchical loss, probability matching, and
/// conjugate prior-sample-size calibration are not run. The result is not a
/// [`ScoreFrequentist`] estimate and not a [`PartialIdentificationEnvelope`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct GeneralizedBayes {
    loss: LossModel,
    prior: Prior,
    learning_rate: LearningRate,
}

impl GeneralizedBayes {
    /// Generalized posterior under `loss` and `prior` at `learning_rate`.
    ///
    /// # Errors
    ///
    /// [`MissingLearningRate`] when `learning_rate` is absent. A rate must be
    /// built with [`LearningRate::try_new`] before it can be supplied.
    #[must_use = "a missing learning rate is not a generalized posterior"]
    pub fn try_new(
        loss: LossModel,
        prior: Prior,
        learning_rate: Option<LearningRate>,
    ) -> Result<Self, MissingLearningRate> {
        let Some(learning_rate) = learning_rate else {
            return Err(MissingLearningRate);
        };
        Ok(Self { loss, prior, learning_rate })
    }

    /// Realization kind. Always [`RealizationKind::GeneralizedBayes`].
    #[must_use]
    pub const fn kind(&self) -> RealizationKind {
        RealizationKind::GeneralizedBayes
    }

    /// Declared loss.
    #[must_use]
    pub const fn loss(&self) -> &LossModel {
        &self.loss
    }

    /// Prior used in the generalized update.
    #[must_use]
    pub const fn prior(&self) -> &Prior {
        &self.prior
    }

    /// Explicit learning rate `w`.
    #[must_use]
    pub const fn learning_rate(&self) -> LearningRate {
        self.learning_rate
    }

    /// Digest of the loss, the prior, and the learning-rate bits.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(GENERALIZED_BAYES_TAG);
        writer.text(self.loss.name()).text(self.prior.name()).word(self.learning_rate.to_bits());
        SemanticDigest::from_bytes(writer.finish())
    }
}

/// Generalized Bayes was attempted without a learning rate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, thiserror::Error)]
#[error("generalized Bayes requires an explicit learning rate")]
pub struct MissingLearningRate;

/// Partial-identification set or structural envelope.
///
/// The value is a set. It has no posterior weights and is not a draw.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PartialIdentificationEnvelope {
    name: Arc<str>,
}

impl PartialIdentificationEnvelope {
    /// Envelope named `name`.
    ///
    /// # Errors
    ///
    /// [`DeclarationError::BlankName`] when `name` is empty or whitespace.
    #[must_use = "a blank envelope name is refused"]
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, DeclarationError> {
        Ok(Self { name: declared_name(name)? })
    }

    /// Realization kind. Always [`RealizationKind::PartialIdentificationEnvelope`].
    #[must_use]
    pub const fn kind(&self) -> RealizationKind {
        RealizationKind::PartialIdentificationEnvelope
    }

    /// Declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Digest of the envelope name.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(PARTIAL_IDENTIFICATION_ENVELOPE_TAG);
        writer.text(&self.name);
        SemanticDigest::from_bytes(writer.finish())
    }
}

/// Assumption branch that holds one identifying restriction.
///
/// The branch identity is not the posterior identity. Changing the restriction
/// changes this digest and leaves the likelihood-and-prior digest unchanged.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct AssumptionBranch {
    restriction: IdentifyingRestriction,
}

impl AssumptionBranch {
    /// Restriction recorded on this branch.
    #[must_use]
    pub const fn restriction(&self) -> &IdentifyingRestriction {
        &self.restriction
    }

    /// Digest of the restriction alone.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(ASSUMPTION_BRANCH_TAG);
        writer.text(self.restriction.name());
        SemanticDigest::from_bytes(writer.finish())
    }
}

/// Posterior under a declared likelihood, with an identifying restriction beside it.
///
/// [`Self::posterior`] is the ordinary Bayesian posterior. [`Self::assumption_branch`]
/// holds the restriction. The two identities differ even when the prior and the
/// restriction share a name.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct RestrictedPosterior {
    posterior: OrdinaryBayesianLikelihood,
    assumption_branch: AssumptionBranch,
}

impl RestrictedPosterior {
    /// Posterior under the declared likelihood.
    #[must_use]
    pub const fn posterior(&self) -> &OrdinaryBayesianLikelihood {
        &self.posterior
    }

    /// Identifying restriction, recorded separately from the posterior.
    #[must_use]
    pub const fn assumption_branch(&self) -> &AssumptionBranch {
        &self.assumption_branch
    }
}

/// Legal next attempt after a prior was refused as observational identification.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum NextIdentificationAttempt {
    /// Attach the named restriction on a separate assumption branch.
    AttachRestrictionOnAssumptionBranch,
    /// Report a posterior under the declared likelihood, not an identified causal effect.
    ReportPosteriorUnderDeclaredLikelihood,
}

impl NextIdentificationAttempt {
    /// Both legal next attempts, in the order the refusal names them.
    pub const ALL: [Self; 2] =
        [Self::AttachRestrictionOnAssumptionBranch, Self::ReportPosteriorUnderDeclaredLikelihood];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AttachRestrictionOnAssumptionBranch => "attach_restriction_on_assumption_branch",
            Self::ReportPosteriorUnderDeclaredLikelihood => {
                "report_posterior_under_declared_likelihood"
            }
        }
    }
}

impl fmt::Display for NextIdentificationAttempt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Refusal to treat a prior as observational identification.
///
/// [`Self::next_attempts`] names the two legal next attempts. The message states
/// the same pair in prose.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PriorIdentificationRefusal {
    prior: Prior,
    next_attempts: [NextIdentificationAttempt; 2],
}

impl PriorIdentificationRefusal {
    /// Prior that was offered as identification.
    #[must_use]
    pub const fn prior(&self) -> &Prior {
        &self.prior
    }

    /// Legal next attempts, in name order.
    #[must_use]
    pub const fn next_attempts(&self) -> [NextIdentificationAttempt; 2] {
        self.next_attempts
    }
}

impl fmt::Display for PriorIdentificationRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "prior {} does not license observational identification; next attempt: attach the \
             identifying restriction on a separate assumption branch, or report the result as a \
             posterior under the declared likelihood rather than as an identified causal effect",
            self.prior.name()
        )
    }
}

impl std::error::Error for PriorIdentificationRefusal {}

/// What a posterior sample was accepted from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PosteriorSampleKind {
    /// Sample from an [`OrdinaryBayesianLikelihood`].
    OrdinaryBayesianLikelihood,
    /// Sample from a [`GeneralizedBayes`] posterior at this learning rate.
    GeneralizedBayes(LearningRate),
}

impl PosteriorSampleKind {
    /// Realization the sample was accepted from.
    #[must_use]
    pub const fn realization(self) -> RealizationKind {
        match self {
            Self::OrdinaryBayesianLikelihood => RealizationKind::OrdinaryBayesianLikelihood,
            Self::GeneralizedBayes(_) => RealizationKind::GeneralizedBayes,
        }
    }
}

/// Acceptance of an ordinary or generalized posterior as a source of draws.
///
/// The value records that acceptance and the realization identity. It does not
/// contain numeric draws. A score realization and a partial-identification
/// envelope are refused: accepting either is a [`PosteriorSampleRefusal`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct PosteriorSample {
    kind: PosteriorSampleKind,
    identity: SemanticDigest,
}

impl PosteriorSample {
    /// Accept a draw from an ordinary Bayesian likelihood posterior.
    #[must_use]
    pub fn accept_ordinary(realization: &OrdinaryBayesianLikelihood) -> Self {
        Self {
            kind: PosteriorSampleKind::OrdinaryBayesianLikelihood,
            identity: realization.identity(),
        }
    }

    /// Accept a draw from a generalized posterior, retaining its learning rate.
    #[must_use]
    pub fn accept_generalized(realization: &GeneralizedBayes) -> Self {
        Self {
            kind: PosteriorSampleKind::GeneralizedBayes(realization.learning_rate()),
            identity: realization.identity(),
        }
    }

    /// Refuse a score realization as a posterior sample.
    ///
    /// # Errors
    ///
    /// Always. A score estimate is not a posterior draw. The refusal names
    /// [`SampleNextAttempt::ReportAsScoreEstimate`].
    #[must_use = "a score realization is not a posterior sample"]
    pub fn accept_score(realization: &ScoreFrequentist) -> Result<Self, PosteriorSampleRefusal> {
        Err(PosteriorSampleRefusal {
            kind: RefusedSampleKind::ScoreFrequentist,
            name: Arc::clone(&realization.score.name),
            identity: realization.identity(),
        })
    }

    /// Refuse an envelope as a posterior sample.
    ///
    /// # Errors
    ///
    /// Always. An envelope is a set, not a draw. The refusal names
    /// [`SampleNextAttempt::ReportAsIdentifiedSet`].
    #[must_use = "an envelope is not a posterior sample"]
    pub fn accept_envelope(
        realization: &PartialIdentificationEnvelope,
    ) -> Result<Self, PosteriorSampleRefusal> {
        Err(PosteriorSampleRefusal {
            kind: RefusedSampleKind::PartialIdentificationEnvelope,
            name: Arc::clone(&realization.name),
            identity: realization.identity(),
        })
    }

    /// Which posterior the sample was accepted from.
    #[must_use]
    pub const fn kind(self) -> PosteriorSampleKind {
        self.kind
    }

    /// Identity of the realization the sample was accepted from.
    #[must_use]
    pub const fn identity(self) -> SemanticDigest {
        self.identity
    }
}

/// Realization that cannot be a posterior sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RefusedSampleKind {
    /// [`ScoreFrequentist`].
    ScoreFrequentist,
    /// [`PartialIdentificationEnvelope`].
    PartialIdentificationEnvelope,
}

impl RefusedSampleKind {
    /// Matching [`RealizationKind`].
    #[must_use]
    pub const fn realization(self) -> RealizationKind {
        match self {
            Self::ScoreFrequentist => RealizationKind::ScoreFrequentist,
            Self::PartialIdentificationEnvelope => RealizationKind::PartialIdentificationEnvelope,
        }
    }
}

/// What to report instead of a posterior sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SampleNextAttempt {
    /// Keep the score realization as a score estimate.
    ReportAsScoreEstimate,
    /// Keep the envelope as an identified set.
    ReportAsIdentifiedSet,
}

impl SampleNextAttempt {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReportAsScoreEstimate => "report_as_score_estimate",
            Self::ReportAsIdentifiedSet => "report_as_identified_set",
        }
    }
}

/// Refusal to accept a non-posterior realization as a posterior sample.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PosteriorSampleRefusal {
    kind: RefusedSampleKind,
    name: Arc<str>,
    identity: SemanticDigest,
}

impl PosteriorSampleRefusal {
    /// Realization that was offered as a sample.
    #[must_use]
    pub const fn kind(&self) -> RefusedSampleKind {
        self.kind
    }

    /// Declared name of that realization.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Identity of the refused realization. Acceptance does not relabel it.
    #[must_use]
    pub const fn identity(&self) -> SemanticDigest {
        self.identity
    }

    /// Legal report of this realization.
    #[must_use]
    pub const fn next_attempt(&self) -> SampleNextAttempt {
        match self.kind {
            RefusedSampleKind::ScoreFrequentist => SampleNextAttempt::ReportAsScoreEstimate,
            RefusedSampleKind::PartialIdentificationEnvelope => {
                SampleNextAttempt::ReportAsIdentifiedSet
            }
        }
    }
}

impl fmt::Display for PosteriorSampleRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            RefusedSampleKind::ScoreFrequentist => write!(
                formatter,
                "score-based frequentist realization {} is not a posterior sample; report it as \
                 a score estimate",
                self.name
            ),
            RefusedSampleKind::PartialIdentificationEnvelope => write!(
                formatter,
                "partial-identification envelope {} is not a posterior sample; report it as an \
                 identified set",
                self.name
            ),
        }
    }
}

impl std::error::Error for PosteriorSampleRefusal {}

fn declared_name(name: impl Into<Arc<str>>) -> Result<Arc<str>, DeclarationError> {
    let name = name.into();
    if is_blank(&name) { Err(DeclarationError::BlankName) } else { Ok(name) }
}

#[cfg(test)]
mod tests {
    use super::{
        DeclarationError, GeneralizedBayes, IdentifyingRestriction, LearningRate, LikelihoodModel,
        LossModel, MissingLearningRate, NextIdentificationAttempt, OrdinaryBayesianLikelihood,
        PartialIdentificationEnvelope, PosteriorSample, PosteriorSampleKind, Prior,
        RealizationKind, RefusedSampleKind, SampleNextAttempt, ScoreFrequentist, ScoreModel,
    };

    fn prior(name: &str) -> Prior {
        Prior::new(name).unwrap()
    }

    fn likelihood(name: &str) -> LikelihoodModel {
        LikelihoodModel::new(name).unwrap()
    }

    #[test]
    fn prior_alone_does_not_identify() {
        let prior = prior("effect_prior");
        let refusal = prior.as_observational_identification();
        assert_eq!(refusal.prior(), &prior);
        assert_eq!(refusal.next_attempts(), NextIdentificationAttempt::ALL);
        assert_eq!(
            refusal.next_attempts()[0],
            NextIdentificationAttempt::AttachRestrictionOnAssumptionBranch
        );
        assert_eq!(
            refusal.next_attempts()[1],
            NextIdentificationAttempt::ReportPosteriorUnderDeclaredLikelihood
        );
        let message = refusal.to_string();
        assert!(message.contains("assumption branch"));
        assert!(message.contains("declared likelihood"));
        assert!(message.contains("identified causal effect"));
        assert!(message.contains("effect_prior"));
    }

    #[test]
    fn prior_plus_named_restriction_is_posterior_with_separate_branch() {
        let posterior = OrdinaryBayesianLikelihood::new(likelihood("gaussian"), prior("effect"));
        let unrestricted = posterior.identity();
        let restriction = IdentifyingRestriction::new("exclusion").unwrap();
        let identified = posterior.clone().with_identifying_restriction(restriction);
        assert_eq!(identified.posterior().identity(), unrestricted);
        assert_eq!(identified.posterior().prior().name(), "effect");
        assert_eq!(identified.posterior().likelihood().name(), "gaussian");
        assert_eq!(identified.assumption_branch().restriction().name(), "exclusion");
        assert_ne!(identified.posterior().identity(), identified.assumption_branch().identity());

        let other = IdentifyingRestriction::new("monotonicity").unwrap();
        let alternative = posterior.with_identifying_restriction(other);
        assert_eq!(alternative.posterior().identity(), unrestricted);
        assert_ne!(
            identified.assumption_branch().identity(),
            alternative.assumption_branch().identity()
        );

        let shared_name = OrdinaryBayesianLikelihood::new(likelihood("gaussian"), prior("effect"))
            .with_identifying_restriction(IdentifyingRestriction::new("effect").unwrap());
        assert_eq!(
            shared_name.posterior().prior().name(),
            shared_name.assumption_branch().restriction().name()
        );
        assert_ne!(shared_name.posterior().identity(), shared_name.assumption_branch().identity());
    }

    #[test]
    fn missing_learning_rate_refuses_generalized_bayes() {
        let loss = LossModel::new("absolute_deviation").unwrap();
        let location = prior("location");
        let missing = GeneralizedBayes::try_new(loss.clone(), location.clone(), None);
        assert_eq!(missing.unwrap_err(), MissingLearningRate);
        assert!(LearningRate::try_new(0.0).is_none());
        assert!(LearningRate::try_new(-0.0).is_none());
        assert!(LearningRate::try_new(-1.0).is_none());
        assert!(LearningRate::try_new(f64::NAN).is_none());
        assert!(LearningRate::try_new(f64::INFINITY).is_none());

        let rate = LearningRate::try_new(0.25).unwrap();
        let generalized = GeneralizedBayes::try_new(loss, location, Some(rate)).unwrap();
        assert_eq!(generalized.learning_rate(), rate);
        assert_eq!(generalized.learning_rate().get().to_bits(), 0.25_f64.to_bits());
        assert_eq!(generalized.kind(), RealizationKind::GeneralizedBayes);

        let unit = LearningRate::try_new(1.0).unwrap();
        let self_information = GeneralizedBayes::try_new(
            LossModel::new("self_information").unwrap(),
            prior("theta"),
            Some(unit),
        )
        .unwrap();
        let ordinary =
            OrdinaryBayesianLikelihood::new(likelihood("self_information"), prior("theta"));
        assert_ne!(self_information.kind(), ordinary.kind());
        assert_ne!(self_information.identity(), ordinary.identity());
        let sample = PosteriorSample::accept_generalized(&self_information);
        assert_eq!(sample.kind(), PosteriorSampleKind::GeneralizedBayes(unit));
        assert_eq!(sample.identity(), self_information.identity());
    }

    #[test]
    fn envelope_is_not_a_posterior_sample() {
        let envelope = PartialIdentificationEnvelope::new("structural_envelope").unwrap();
        let refused = PosteriorSample::accept_envelope(&envelope).unwrap_err();
        assert_eq!(refused.kind(), RefusedSampleKind::PartialIdentificationEnvelope);
        assert_eq!(refused.kind().realization(), RealizationKind::PartialIdentificationEnvelope);
        assert_eq!(refused.name(), "structural_envelope");
        assert_eq!(refused.identity(), envelope.identity());
        assert_eq!(refused.next_attempt(), SampleNextAttempt::ReportAsIdentifiedSet);
        let message = refused.to_string();
        assert!(message.contains("not a posterior sample"));
        assert!(message.contains("identified set"));

        let ordinary = OrdinaryBayesianLikelihood::new(likelihood("gaussian"), prior("flat"));
        let sample = PosteriorSample::accept_ordinary(&ordinary);
        assert_eq!(sample.kind(), PosteriorSampleKind::OrdinaryBayesianLikelihood);
        assert_eq!(sample.kind().realization(), RealizationKind::OrdinaryBayesianLikelihood);
        assert_eq!(sample.identity(), ordinary.identity());
        assert_ne!(sample.identity(), envelope.identity());
    }

    #[test]
    fn score_is_not_a_posterior_sample() {
        let score = ScoreFrequentist::new(ScoreModel::new("aipw").unwrap());
        let refused = PosteriorSample::accept_score(&score).unwrap_err();
        assert_eq!(refused.kind(), RefusedSampleKind::ScoreFrequentist);
        assert_eq!(refused.name(), "aipw");
        assert_eq!(refused.identity(), score.identity());
        assert_eq!(refused.next_attempt(), SampleNextAttempt::ReportAsScoreEstimate);
        assert_eq!(score.kind(), RealizationKind::ScoreFrequentist);
    }

    #[test]
    fn realization_identities_stay_distinct() {
        let name = "shared";
        let ordinary = OrdinaryBayesianLikelihood::new(likelihood(name), prior(name));
        let score = ScoreFrequentist::new(ScoreModel::new(name).unwrap());
        let generalized = GeneralizedBayes::try_new(
            LossModel::new(name).unwrap(),
            prior(name),
            Some(LearningRate::try_new(1.0).unwrap()),
        )
        .unwrap();
        let envelope = PartialIdentificationEnvelope::new(name).unwrap();
        let identities =
            [ordinary.identity(), score.identity(), generalized.identity(), envelope.identity()];
        let mut index = 0;
        while index < identities.len() {
            let mut other = index + 1;
            while other < identities.len() {
                assert_ne!(identities[index], identities[other]);
                other += 1;
            }
            index += 1;
        }
        assert_eq!(ordinary.kind(), RealizationKind::OrdinaryBayesianLikelihood);
        assert_eq!(envelope.kind(), RealizationKind::PartialIdentificationEnvelope);
    }

    #[test]
    fn blank_names_are_refused() {
        assert_eq!(Prior::new("  ").unwrap_err(), DeclarationError::BlankName);
        assert_eq!(IdentifyingRestriction::new("").unwrap_err(), DeclarationError::BlankName);
        assert_eq!(LikelihoodModel::new("\n").unwrap_err(), DeclarationError::BlankName);
        assert_eq!(LossModel::new("\t").unwrap_err(), DeclarationError::BlankName);
        assert_eq!(ScoreModel::new("").unwrap_err(), DeclarationError::BlankName);
        assert_eq!(
            PartialIdentificationEnvelope::new(" ").unwrap_err(),
            DeclarationError::BlankName
        );
    }

    #[test]
    fn realization_kind_names_differ() {
        for (index, kind) in RealizationKind::ALL.iter().enumerate() {
            for other in RealizationKind::ALL.iter().skip(index + 1) {
                assert_ne!(kind, other);
                assert_ne!(kind.as_str(), other.as_str());
            }
        }
    }
}
