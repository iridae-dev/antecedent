// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adapter from a frequentist estimate and covariance to a summary observation model.
//!
//! A [`FrequentistSummary`] stays a frequentist summary. An interval and a
//! covariance summary are different objects from a Bayesian likelihood and from
//! a Bayesian law. [`FrequentistSummary::as_bayesian_likelihood`] refuses that
//! reading and names the next attempt.
//!
//! An estimate and its covariance may enter a [`SummaryObservationModel`] when
//! a sampling approximation, a selection record, a dependence record, and an
//! origin are all present. Those four names are an [`AssumptionChecklist`].
//! A checklist is not a likelihood. The permitted use is a likelihood update,
//! checked by [`SummaryObservationModel::validate_use`], and the admission
//! carries the estimate and the covariance. Without those numbers the model
//! is refused and stays a checklist. The same model read as a Bayesian law is
//! refused. An interval cannot take this path: the four names do not turn
//! coverage into an observation model.
//!
//! [`FrequentistSummary::estimate`] and [`FrequentistSummary::covariance`] read
//! those stored numbers back. An interval has neither. This module does not
//! form a posterior, multiply a likelihood, or estimate a covariance.

use std::error::Error;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use antecedent_core::text::is_blank;
use antecedent_core::{CanonicalWriter, SemanticDigest};

/// Identity tag for a frequentist summary.
pub const FREQUENTIST_SUMMARY_TAG: &str = "antecedent.contract.frequentist_summary.v1";

/// Identity tag for a summary observation model.
pub const SUMMARY_OBSERVATION_TAG: &str = "antecedent.contract.summary_observation.v1";

const BLANK_DETAIL: &str = "summary_observation.blank_record";
const NON_FINITE_DETAIL: &str = "summary_observation.non_finite_value";
const SHAPE_DETAIL: &str = "summary_observation.covariance_shape";
const NOT_LIKELIHOOD_DETAIL: &str = "summary_observation.not_bayesian_likelihood";
const INTERVAL_DETAIL: &str = "summary_observation.interval_stays_frequentist";
const NOT_LAW_DETAIL: &str = "summary_observation.not_bayesian_law";
const CHECKLIST_DETAIL: &str = "summary_observation.checklist_not_likelihood";

const MISMATCH_CODE: &str = antecedent_core::reason_code!("distribution_meaning_mismatch");
const INVALID_CODE: &str = antecedent_core::reason_code!("invalid_argument");

/// What a frequentist summary is.
///
/// Adaptation does not change this kind. Declaration order is [`Self::ALL`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum FrequentistSummaryKind {
    /// Point estimate together with its covariance.
    EstimateCovariance,
    /// Coverage interval or region.
    Interval,
}

impl FrequentistSummaryKind {
    /// Both summaries, in declaration order.
    pub const ALL: [Self; 2] = [Self::EstimateCovariance, Self::Interval];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EstimateCovariance => "estimate_covariance",
            Self::Interval => "interval",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not a frequentist summary"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }
}

impl fmt::Display for FrequentistSummaryKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Square covariance stored with a frequentist estimate, row-major.
///
/// A scalar variance converts through [`From<f64>`] into a one-by-one matrix.
/// [`Self::matrix`] owns a square matrix of any order. Equality follows
/// `to_bits`, so the value read back is the value that was stored.
#[derive(Clone, Debug)]
pub struct SummaryCovariance {
    dimension: usize,
    entries: Vec<f64>,
}

impl SummaryCovariance {
    /// Row-major square matrix with `dimension` rows.
    ///
    /// # Errors
    ///
    /// [`SummaryObservationError`] when `dimension` is zero, `entries` has a
    /// length other than `dimension * dimension`, or any entry is non-finite.
    /// The error has no next attempt.
    #[must_use = "a covariance that is not a finite square matrix is refused"]
    pub fn matrix(dimension: usize, entries: Vec<f64>) -> Result<Self, SummaryObservationError> {
        let Some(cells) = dimension.checked_mul(dimension) else {
            return Err(SummaryObservationError::shape());
        };
        if dimension == 0 || entries.len() != cells {
            return Err(SummaryObservationError::shape());
        }
        if entries.iter().any(|entry| !entry.is_finite()) {
            return Err(SummaryObservationError::non_finite("covariance"));
        }
        Ok(Self { dimension, entries })
    }

    /// Side length of the square matrix. A scalar variance has length one.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Row-major entries.
    #[must_use]
    pub fn entries(&self) -> &[f64] {
        &self.entries
    }
}

impl From<f64> for SummaryCovariance {
    fn from(variance: f64) -> Self {
        Self { dimension: 1, entries: vec![variance] }
    }
}

impl PartialEq for SummaryCovariance {
    fn eq(&self, other: &Self) -> bool {
        self.dimension == other.dimension && same_bits(&self.entries, &other.entries)
    }
}

impl Eq for SummaryCovariance {}

impl Hash for SummaryCovariance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dimension.hash(state);
        for entry in &self.entries {
            entry.to_bits().hash(state);
        }
    }
}

/// Estimate payload, or an interval that carries neither number.
#[derive(Clone, Debug)]
enum SummaryBody {
    EstimateCovariance { estimate: f64, covariance: SummaryCovariance },
    Interval,
}

impl PartialEq for SummaryBody {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::EstimateCovariance { estimate: left_estimate, covariance: left_covariance },
                Self::EstimateCovariance { estimate: right_estimate, covariance: right_covariance },
            ) => {
                left_estimate.to_bits() == right_estimate.to_bits()
                    && left_covariance == right_covariance
            }
            (Self::Interval, Self::Interval) => true,
            (Self::EstimateCovariance { .. }, Self::Interval)
            | (Self::Interval, Self::EstimateCovariance { .. }) => false,
        }
    }
}

impl Eq for SummaryBody {}

impl Hash for SummaryBody {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::EstimateCovariance { estimate, covariance } => {
                0_u8.hash(state);
                estimate.to_bits().hash(state);
                covariance.hash(state);
            }
            Self::Interval => 1_u8.hash(state),
        }
    }
}

/// A frequentist estimate/covariance or a frequentist interval.
///
/// The kind is fixed at construction. An estimate stores its value and its
/// covariance. An interval stores neither, so it cannot be read as a posterior.
/// Offering the summary as a Bayesian likelihood returns
/// [`FrequentistSummary::as_bayesian_likelihood`] and leaves this value a
/// frequentist summary.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct FrequentistSummary {
    name: Arc<str>,
    body: SummaryBody,
}

impl FrequentistSummary {
    /// Estimate `estimate` with covariance `covariance`, named `name`.
    ///
    /// `covariance` is a scalar variance (`f64`) or a [`SummaryCovariance`]
    /// built by [`SummaryCovariance::matrix`].
    ///
    /// # Errors
    ///
    /// [`SummaryObservationError`] when `name` is empty or whitespace, or when
    /// `estimate` or a covariance entry is non-finite. The error has no next
    /// attempt.
    #[must_use = "a blank name or a non-finite value is refused"]
    pub fn estimate_covariance(
        name: impl Into<Arc<str>>,
        estimate: f64,
        covariance: impl Into<SummaryCovariance>,
    ) -> Result<Self, SummaryObservationError> {
        let name = declared(name, "summary")?;
        let estimate = finite(estimate, "estimate")?;
        let covariance = covariance.into();
        if covariance.entries.iter().any(|entry| !entry.is_finite()) {
            return Err(SummaryObservationError::non_finite("covariance"));
        }
        Ok(Self { name, body: SummaryBody::EstimateCovariance { estimate, covariance } })
    }

    /// Coverage interval named `name`.
    ///
    /// # Errors
    ///
    /// [`SummaryObservationError`] when `name` is empty or whitespace. The
    /// error has no next attempt.
    #[must_use = "a blank summary name is refused"]
    pub fn interval(name: impl Into<Arc<str>>) -> Result<Self, SummaryObservationError> {
        Ok(Self { name: declared(name, "summary")?, body: SummaryBody::Interval })
    }

    /// Frequentist kind. Adaptation does not replace it.
    #[must_use]
    pub const fn kind(&self) -> FrequentistSummaryKind {
        match &self.body {
            SummaryBody::EstimateCovariance { .. } => FrequentistSummaryKind::EstimateCovariance,
            SummaryBody::Interval => FrequentistSummaryKind::Interval,
        }
    }

    /// Declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Point estimate stored with an estimate-covariance summary.
    ///
    /// An interval returns [`None`]. That absence is not a posterior.
    #[must_use]
    pub const fn estimate(&self) -> Option<f64> {
        match &self.body {
            SummaryBody::EstimateCovariance { estimate, .. } => Some(*estimate),
            SummaryBody::Interval => None,
        }
    }

    /// Covariance stored with an estimate-covariance summary.
    ///
    /// An interval returns [`None`].
    #[must_use]
    pub const fn covariance(&self) -> Option<&SummaryCovariance> {
        match &self.body {
            SummaryBody::EstimateCovariance { covariance, .. } => Some(covariance),
            SummaryBody::Interval => None,
        }
    }

    /// Digest of the kind, the name, and any stored estimate and covariance.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(FREQUENTIST_SUMMARY_TAG);
        write_summary(&mut writer, self);
        SemanticDigest::from_bytes(writer.finish())
    }

    /// Refuse this summary as a Bayesian likelihood.
    ///
    /// An interval's next attempt is [`SummaryNextAttempt::ReportAsFrequentistInterval`].
    /// A covariance summary's next attempt is
    /// [`SummaryNextAttempt::AdaptEstimateCovariance`]. Neither attempt retags
    /// this summary or forms a posterior.
    #[must_use = "a frequentist summary is not a Bayesian likelihood"]
    pub fn as_bayesian_likelihood(&self) -> SummaryObservationError {
        let kind = self.kind();
        let next = match kind {
            FrequentistSummaryKind::Interval => SummaryNextAttempt::ReportAsFrequentistInterval,
            FrequentistSummaryKind::EstimateCovariance => {
                SummaryNextAttempt::AdaptEstimateCovariance
            }
        };
        let prefix = match kind {
            FrequentistSummaryKind::Interval => "frequentist interval ",
            FrequentistSummaryKind::EstimateCovariance => "frequentist covariance summary ",
        };
        let suffix = match kind {
            FrequentistSummaryKind::Interval => {
                "stays a frequentist summary and is not a Bayesian likelihood and is not a Bayesian law"
            }
            FrequentistSummaryKind::EstimateCovariance => {
                "stays a frequentist summary and is not a Bayesian likelihood"
            }
        };
        SummaryObservationError::meaning(
            NOT_LIKELIHOOD_DETAIL,
            named(prefix, self.name(), suffix, next),
            next,
            kind,
        )
    }
}

/// Sampling approximation under which an estimate's covariance is read.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SamplingApproximation {
    name: Arc<str>,
}

impl SamplingApproximation {
    /// Approximation named `name`.
    ///
    /// # Errors
    ///
    /// [`SummaryObservationError`] when `name` is empty or whitespace.
    #[must_use = "a blank sampling approximation is refused"]
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, SummaryObservationError> {
        Ok(Self { name: declared(name, "sampling approximation")? })
    }

    /// Declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Selection or sampling assumption of the estimate.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Selection {
    name: Arc<str>,
}

impl Selection {
    /// Selection named `name`.
    ///
    /// # Errors
    ///
    /// [`SummaryObservationError`] when `name` is empty or whitespace.
    #[must_use = "a blank selection record is refused"]
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, SummaryObservationError> {
        Ok(Self { name: declared(name, "selection")? })
    }

    /// Declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Dependence among the sampling units of the estimate.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Dependence {
    name: Arc<str>,
}

impl Dependence {
    /// Dependence named `name`.
    ///
    /// # Errors
    ///
    /// [`SummaryObservationError`] when `name` is empty or whitespace.
    #[must_use = "a blank dependence record is refused"]
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, SummaryObservationError> {
        Ok(Self { name: declared(name, "dependence")? })
    }

    /// Declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Origin of the estimate and covariance.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Origin {
    name: Arc<str>,
}

impl Origin {
    /// Origin named `name`.
    ///
    /// # Errors
    ///
    /// [`SummaryObservationError`] when `name` is empty or whitespace.
    #[must_use = "a blank origin is refused"]
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, SummaryObservationError> {
        Ok(Self { name: declared(name, "origin")? })
    }

    /// Declared name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Named assumptions recorded beside an estimate.
///
/// The four names are a checklist. [`Self::as_likelihood`] refuses to read
/// them as a likelihood. A likelihood needs the estimate and its covariance,
/// which a [`SummaryObservationModel`] stores and a
/// [`LikelihoodUpdateAdmission`] returns.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AssumptionChecklist {
    sampling_approximation: SamplingApproximation,
    selection: Selection,
    dependence: Dependence,
    origin: Origin,
}

impl AssumptionChecklist {
    /// Checklist of the four named records.
    #[must_use]
    pub const fn new(
        sampling_approximation: SamplingApproximation,
        selection: Selection,
        dependence: Dependence,
        origin: Origin,
    ) -> Self {
        Self { sampling_approximation, selection, dependence, origin }
    }

    /// Sampling approximation name.
    #[must_use]
    pub const fn sampling_approximation(&self) -> &SamplingApproximation {
        &self.sampling_approximation
    }

    /// Selection name.
    #[must_use]
    pub const fn selection(&self) -> &Selection {
        &self.selection
    }

    /// Dependence name.
    #[must_use]
    pub const fn dependence(&self) -> &Dependence {
        &self.dependence
    }

    /// Origin name.
    #[must_use]
    pub const fn origin(&self) -> &Origin {
        &self.origin
    }

    /// Refuse this checklist as a likelihood.
    ///
    /// The names are not an estimate and not a covariance. The next attempt is
    /// [`SummaryNextAttempt::SupplyEstimateCovariance`].
    #[must_use = "a checklist of names is not a likelihood"]
    pub fn as_likelihood(&self) -> SummaryObservationError {
        let next = SummaryNextAttempt::SupplyEstimateCovariance;
        let mut message = String::from(
            "a checklist of sampling approximation, selection, dependence, and origin is not a \
             likelihood",
        );
        message.push_str("; next attempt: ");
        message.push_str(next.phrase());
        SummaryObservationError::checklist(message)
    }
}

/// Observation model adapted from an estimate and its covariance.
///
/// The wrapped [`FrequentistSummary`] stays
/// [`FrequentistSummaryKind::EstimateCovariance`]. The sampling approximation,
/// selection, dependence, and origin are an [`AssumptionChecklist`], not a
/// likelihood. A likelihood update is the permitted use and carries the
/// estimate and the covariance. A Bayesian law is not a permitted use.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct SummaryObservationModel {
    summary: FrequentistSummary,
    checklist: AssumptionChecklist,
}

impl SummaryObservationModel {
    /// Adapt `summary` when it is an estimate and covariance and the four
    /// records are present.
    ///
    /// An interval is refused even when all four records are supplied. The
    /// next attempt is to report that interval as a frequentist coverage
    /// summary.
    ///
    /// # Errors
    ///
    /// [`SummaryObservationError`] when `summary` is an interval.
    #[must_use = "a frequentist interval stays a frequentist summary"]
    pub fn try_adapt(
        summary: FrequentistSummary,
        sampling_approximation: SamplingApproximation,
        selection: Selection,
        dependence: Dependence,
        origin: Origin,
    ) -> Result<Self, SummaryObservationError> {
        if summary.kind() == FrequentistSummaryKind::Interval {
            let next = SummaryNextAttempt::ReportAsFrequentistInterval;
            return Err(SummaryObservationError::meaning(
                INTERVAL_DETAIL,
                named(
                    "frequentist interval ",
                    summary.name(),
                    "stays a frequentist summary and is not a summary observation model",
                    next,
                ),
                next,
                summary.kind(),
            ));
        }
        Ok(Self {
            summary,
            checklist: AssumptionChecklist::new(
                sampling_approximation,
                selection,
                dependence,
                origin,
            ),
        })
    }

    /// Frequentist estimate and covariance this model was adapted from.
    #[must_use]
    pub const fn summary(&self) -> &FrequentistSummary {
        &self.summary
    }

    /// Named assumptions. This checklist is not a likelihood.
    #[must_use = "a checklist of names is not a likelihood"]
    pub const fn checklist(&self) -> &AssumptionChecklist {
        &self.checklist
    }

    /// Sampling approximation recorded on this model.
    #[must_use]
    pub const fn sampling_approximation(&self) -> &SamplingApproximation {
        self.checklist.sampling_approximation()
    }

    /// Selection recorded on this model.
    #[must_use]
    pub const fn selection(&self) -> &Selection {
        self.checklist.selection()
    }

    /// Dependence recorded on this model.
    #[must_use]
    pub const fn dependence(&self) -> &Dependence {
        self.checklist.dependence()
    }

    /// Origin recorded on this model.
    #[must_use]
    pub const fn origin(&self) -> &Origin {
        self.checklist.origin()
    }

    /// Digest of the frequentist summary and the four records.
    ///
    /// The frequentist summary contributes its kind, name, and stored estimate
    /// and covariance. This digest is not [`FrequentistSummary::identity`].
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(SUMMARY_OBSERVATION_TAG);
        write_summary(&mut writer, &self.summary);
        writer.text(self.checklist.sampling_approximation().name());
        writer.text(self.checklist.selection().name());
        writer.text(self.checklist.dependence().name());
        writer.text(self.checklist.origin().name());
        SemanticDigest::from_bytes(writer.finish())
    }

    /// Admit `requested` when it is a likelihood update of this model.
    ///
    /// [`SummaryUse::LikelihoodUpdate`] returns an admission whose summary kind
    /// is still [`FrequentistSummaryKind::EstimateCovariance`] and which carries
    /// the estimate and the covariance. The checklist of names is not that
    /// likelihood. [`SummaryUse::BayesianLaw`] is refused. The next attempt is
    /// [`SummaryNextAttempt::UseAsLikelihoodUpdate`]. When the estimate or the
    /// covariance is absent the error is the checklist refusal.
    ///
    /// # Errors
    ///
    /// [`SummaryObservationError`] when `requested` is a Bayesian law.
    #[must_use = "only a likelihood update of the summary observation model is permitted"]
    pub fn validate_use(
        &self,
        requested: SummaryUse,
    ) -> Result<LikelihoodUpdateAdmission, SummaryObservationError> {
        match requested {
            SummaryUse::LikelihoodUpdate => {
                match (self.summary.estimate(), self.summary.covariance()) {
                    (Some(estimate), Some(covariance)) => Ok(LikelihoodUpdateAdmission {
                        summary_kind: self.summary.kind(),
                        summary_name: Arc::clone(&self.summary.name),
                        summary_identity: self.summary.identity(),
                        model_identity: self.identity(),
                        estimate,
                        covariance: covariance.clone(),
                    }),
                    _ => Err(self.checklist.as_likelihood()),
                }
            }
            SummaryUse::BayesianLaw => {
                let next = SummaryNextAttempt::UseAsLikelihoodUpdate;
                Err(SummaryObservationError::meaning(
                    NOT_LAW_DETAIL,
                    named(
                        "summary observation model ",
                        self.summary.name(),
                        "is not a Bayesian law",
                        next,
                    ),
                    next,
                    self.summary.kind(),
                ))
            }
        }
    }
}

/// How a summary observation model may be used.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum SummaryUse {
    /// Use the model as the likelihood of an update.
    LikelihoodUpdate,
    /// Read the frequentist summary as a Bayesian law.
    BayesianLaw,
}

impl SummaryUse {
    /// Both uses, in declaration order.
    pub const ALL: [Self; 2] = [Self::LikelihoodUpdate, Self::BayesianLaw];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LikelihoodUpdate => "likelihood_update",
            Self::BayesianLaw => "bayesian_law",
        }
    }
}

impl fmt::Display for SummaryUse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Admission to use a summary observation model as a likelihood update.
///
/// The frequentist summary kind and identity are the estimate and covariance
/// the model was adapted from. The estimate and covariance are the likelihood.
/// A name list is not.
#[derive(Clone, Debug)]
pub struct LikelihoodUpdateAdmission {
    summary_kind: FrequentistSummaryKind,
    summary_name: Arc<str>,
    summary_identity: SemanticDigest,
    model_identity: SemanticDigest,
    estimate: f64,
    covariance: SummaryCovariance,
}

impl PartialEq for LikelihoodUpdateAdmission {
    fn eq(&self, other: &Self) -> bool {
        self.summary_kind == other.summary_kind
            && self.summary_name == other.summary_name
            && self.summary_identity == other.summary_identity
            && self.model_identity == other.model_identity
            && self.estimate.to_bits() == other.estimate.to_bits()
            && self.covariance == other.covariance
    }
}

impl Eq for LikelihoodUpdateAdmission {}

impl Hash for LikelihoodUpdateAdmission {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.summary_kind.hash(state);
        self.summary_name.hash(state);
        self.summary_identity.hash(state);
        self.model_identity.hash(state);
        self.estimate.to_bits().hash(state);
        self.covariance.hash(state);
    }
}

impl LikelihoodUpdateAdmission {
    /// Kind of the frequentist summary. An admitted update keeps it.
    #[must_use]
    pub const fn summary_kind(&self) -> FrequentistSummaryKind {
        self.summary_kind
    }

    /// Name of the frequentist summary.
    #[must_use]
    pub fn summary_name(&self) -> &str {
        &self.summary_name
    }

    /// Identity of the frequentist summary.
    #[must_use]
    pub const fn summary_identity(&self) -> SemanticDigest {
        self.summary_identity
    }

    /// Identity of the summary observation model.
    #[must_use]
    pub const fn model_identity(&self) -> SemanticDigest {
        self.model_identity
    }

    /// Point estimate of the likelihood.
    #[must_use]
    pub const fn estimate(&self) -> f64 {
        self.estimate
    }

    /// Covariance of the likelihood.
    #[must_use]
    pub const fn covariance(&self) -> &SummaryCovariance {
        &self.covariance
    }
}

/// Legal next attempt after a summary was refused as a Bayesian reading.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum SummaryNextAttempt {
    /// Keep the interval as a frequentist coverage summary.
    ReportAsFrequentistInterval,
    /// Record the four fields and adapt the estimate and covariance.
    AdaptEstimateCovariance,
    /// Use the summary observation model as the likelihood of the update.
    UseAsLikelihoodUpdate,
    /// Supply the estimate and its covariance. A name list is not a likelihood.
    SupplyEstimateCovariance,
}

impl SummaryNextAttempt {
    /// Every next attempt, in declaration order.
    pub const ALL: [Self; 4] = [
        Self::ReportAsFrequentistInterval,
        Self::AdaptEstimateCovariance,
        Self::UseAsLikelihoodUpdate,
        Self::SupplyEstimateCovariance,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReportAsFrequentistInterval => "report_as_frequentist_interval",
            Self::AdaptEstimateCovariance => "adapt_estimate_covariance",
            Self::UseAsLikelihoodUpdate => "use_as_likelihood_update",
            Self::SupplyEstimateCovariance => "supply_estimate_covariance",
        }
    }

    /// Wording of the next attempt.
    #[must_use]
    pub const fn phrase(self) -> &'static str {
        match self {
            Self::ReportAsFrequentistInterval => {
                "report the interval as a frequentist coverage summary"
            }
            Self::AdaptEstimateCovariance => {
                "record sampling approximation, selection, dependence, and origin on a summary \
                 observation model"
            }
            Self::UseAsLikelihoodUpdate => {
                "use the summary observation model as the likelihood of the update"
            }
            Self::SupplyEstimateCovariance => "supply the estimate and its covariance",
        }
    }

    /// Parse the spelling of [`Self::as_str`].
    #[must_use = "an unknown name is not a next attempt"]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|attempt| attempt.as_str() == name)
    }
}

impl fmt::Display for SummaryNextAttempt {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.phrase())
    }
}

/// A refused summary record or a refused Bayesian reading.
///
/// A Bayesian reading stores [`Self::next_attempt`]. A blank record does not.
#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use = "using a frequentist summary as a Bayesian likelihood is refused"]
pub struct SummaryObservationError {
    code: &'static str,
    detail: &'static str,
    message: String,
    next: Option<SummaryNextAttempt>,
    summary_kind: Option<FrequentistSummaryKind>,
}

impl SummaryObservationError {
    /// Registered reason code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail, `summary_observation.<rule>`.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }

    /// What was refused, including the next attempt when a reading was refused.
    #[must_use]
    pub fn message(&self) -> &str {
        self.message.as_str()
    }

    /// The refusal as `(code, detail, explanation)`.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, &'static str, &str) {
        (self.code, self.detail, self.message.as_str())
    }

    /// The next attempt, when a Bayesian reading or an interval adaptation was refused.
    #[must_use = "a refused Bayesian reading names its next attempt"]
    pub const fn next_attempt(&self) -> Option<SummaryNextAttempt> {
        self.next
    }

    /// Frequentist kind of the summary that was offered, when the refusal has one.
    ///
    /// A blank record has none. A refused reading keeps the frequentist kind.
    #[must_use = "a refused reading keeps the frequentist kind"]
    pub const fn summary_kind(&self) -> Option<FrequentistSummaryKind> {
        self.summary_kind
    }

    fn blank(field: &'static str) -> Self {
        Self {
            code: INVALID_CODE,
            detail: BLANK_DETAIL,
            message: format!("{field} is blank"),
            next: None,
            summary_kind: None,
        }
    }

    fn non_finite(field: &'static str) -> Self {
        Self {
            code: INVALID_CODE,
            detail: NON_FINITE_DETAIL,
            message: format!("{field} is not finite"),
            next: None,
            summary_kind: None,
        }
    }

    fn shape() -> Self {
        Self {
            code: INVALID_CODE,
            detail: SHAPE_DETAIL,
            message: String::from("covariance is not a square matrix"),
            next: None,
            summary_kind: None,
        }
    }

    fn checklist(message: String) -> Self {
        Self {
            code: MISMATCH_CODE,
            detail: CHECKLIST_DETAIL,
            message,
            next: Some(SummaryNextAttempt::SupplyEstimateCovariance),
            summary_kind: None,
        }
    }

    fn meaning(
        detail: &'static str,
        message: String,
        next: SummaryNextAttempt,
        kind: FrequentistSummaryKind,
    ) -> Self {
        Self { code: MISMATCH_CODE, detail, message, next: Some(next), summary_kind: Some(kind) }
    }
}

impl fmt::Display for SummaryObservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.detail, self.message)
    }
}

impl Error for SummaryObservationError {}

fn declared(
    name: impl Into<Arc<str>>,
    field: &'static str,
) -> Result<Arc<str>, SummaryObservationError> {
    let name = name.into();
    if is_blank(&name) { Err(SummaryObservationError::blank(field)) } else { Ok(name) }
}

fn named(prefix: &str, name: &str, suffix: &str, next: SummaryNextAttempt) -> String {
    let mut message = String::from(prefix);
    message.push('`');
    message.push_str(name);
    message.push_str("` ");
    message.push_str(suffix);
    message.push_str("; next attempt: ");
    message.push_str(next.phrase());
    message
}

fn finite(value: f64, field: &'static str) -> Result<f64, SummaryObservationError> {
    if value.is_finite() { Ok(value) } else { Err(SummaryObservationError::non_finite(field)) }
}

fn same_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left_entry, right_entry)| left_entry.to_bits() == right_entry.to_bits())
}

fn write_summary(writer: &mut CanonicalWriter, summary: &FrequentistSummary) {
    writer.text(summary.kind().as_str());
    writer.text(summary.name());
    if let SummaryBody::EstimateCovariance { estimate, covariance } = &summary.body {
        writer.real(*estimate);
        writer.reals(covariance.entries());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Dependence, FrequentistSummary, FrequentistSummaryKind, Origin, SamplingApproximation,
        Selection, SummaryCovariance, SummaryNextAttempt, SummaryObservationModel, SummaryUse,
    };

    struct ObservationRecords {
        sampling: SamplingApproximation,
        selection: Selection,
        dependence: Dependence,
        origin: Origin,
    }

    fn records() -> ObservationRecords {
        ObservationRecords {
            sampling: SamplingApproximation::new("normal_approximation").expect("sampling"),
            selection: Selection::new("complete_cases").expect("selection"),
            dependence: Dependence::new("cluster_dependence").expect("dependence"),
            origin: Origin::new("trial_estimator").expect("origin"),
        }
    }

    fn estimate(name: &str) -> FrequentistSummary {
        FrequentistSummary::estimate_covariance(name, 1.25_f64, 0.25_f64).expect("estimate")
    }

    fn interval(name: &str) -> FrequentistSummary {
        FrequentistSummary::interval(name).expect("interval")
    }

    #[test]
    fn frequentist_interval_is_not_a_bayesian_likelihood() {
        let interval = interval("ate");
        let identity = interval.identity();
        let refusal = interval.as_bayesian_likelihood();

        assert_eq!(interval.kind(), FrequentistSummaryKind::Interval);
        assert_eq!(interval.identity(), identity);
        assert_eq!(refusal.summary_kind(), Some(FrequentistSummaryKind::Interval));
        assert_eq!(refusal.code(), "distribution_meaning_mismatch");
        assert_eq!(refusal.detail(), "summary_observation.not_bayesian_likelihood");
        assert_eq!(refusal.next_attempt(), Some(SummaryNextAttempt::ReportAsFrequentistInterval));
        assert_eq!(
            refusal.next_attempt().expect("next").as_str(),
            "report_as_frequentist_interval"
        );
        assert_eq!(
            refusal.next_attempt().expect("next").to_string(),
            SummaryNextAttempt::ReportAsFrequentistInterval.phrase()
        );
        let message = refusal.message();
        assert!(message.contains("frequentist interval `ate`"));
        assert!(message.contains("stays a frequentist summary"));
        assert!(message.contains("not a Bayesian likelihood"));
        assert!(message.contains("not a Bayesian law"));
        assert!(message.contains(SummaryNextAttempt::ReportAsFrequentistInterval.phrase()));
        assert_eq!(
            refusal.refusal(),
            (
                "distribution_meaning_mismatch",
                "summary_observation.not_bayesian_likelihood",
                message,
            )
        );
        assert!(antecedent_core::reason_code::is_runtime_refusal(refusal.code()));
        assert_ne!(interval.kind().as_str(), "bayesian_law");
        assert_ne!(interval.identity(), estimate("ate").identity());
    }

    #[test]
    fn covariance_summary_is_not_a_bayesian_likelihood() {
        let summary = estimate("ate");
        let identity = summary.identity();
        let refusal = summary.as_bayesian_likelihood();

        assert_eq!(summary.kind(), FrequentistSummaryKind::EstimateCovariance);
        assert_eq!(summary.identity(), identity);
        assert_eq!(refusal.summary_kind(), Some(FrequentistSummaryKind::EstimateCovariance));
        assert_eq!(refusal.code(), "distribution_meaning_mismatch");
        assert_eq!(refusal.detail(), "summary_observation.not_bayesian_likelihood");
        assert_eq!(refusal.next_attempt(), Some(SummaryNextAttempt::AdaptEstimateCovariance));
        assert_eq!(refusal.next_attempt().expect("next").as_str(), "adapt_estimate_covariance");
        let message = refusal.message();
        assert!(message.contains("frequentist covariance summary `ate`"));
        assert!(message.contains("stays a frequentist summary"));
        assert!(message.contains("not a Bayesian likelihood"));
        assert!(message.contains("sampling approximation"));
        assert!(message.contains("selection"));
        assert!(message.contains("dependence"));
        assert!(message.contains("origin"));
        assert!(message.contains(SummaryNextAttempt::AdaptEstimateCovariance.phrase()));
        assert!(antecedent_core::reason_code::is_runtime_refusal(refusal.code()));
        assert_ne!(refusal.next_attempt(), Some(SummaryNextAttempt::ReportAsFrequentistInterval));
    }

    #[test]
    fn interval_stays_frequentist_when_adaptation_is_refused() {
        let interval = interval("risk_difference");
        let identity = interval.identity();
        let fields = records();
        let refusal = SummaryObservationModel::try_adapt(
            interval.clone(),
            fields.sampling,
            fields.selection,
            fields.dependence,
            fields.origin,
        )
        .expect_err("interval");

        assert_eq!(interval.kind(), FrequentistSummaryKind::Interval);
        assert_eq!(interval.identity(), identity);
        assert_eq!(refusal.summary_kind(), Some(FrequentistSummaryKind::Interval));
        assert_eq!(refusal.detail(), "summary_observation.interval_stays_frequentist");
        assert_eq!(refusal.code(), "distribution_meaning_mismatch");
        assert_eq!(refusal.next_attempt(), Some(SummaryNextAttempt::ReportAsFrequentistInterval));
        assert!(refusal.message().contains("stays a frequentist summary"));
        assert!(refusal.message().contains("not a summary observation model"));
        assert!(refusal.message().contains("risk_difference"));
        assert!(antecedent_core::reason_code::is_runtime_refusal(refusal.code()));
        assert_ne!(refusal.detail(), interval.as_bayesian_likelihood().detail());
    }

    #[test]
    fn estimate_covariance_adapts_with_required_fields() {
        let summary = estimate("ate");
        let summary_identity = summary.identity();
        let fields = records();
        let model = SummaryObservationModel::try_adapt(
            summary.clone(),
            fields.sampling.clone(),
            fields.selection.clone(),
            fields.dependence.clone(),
            fields.origin.clone(),
        )
        .expect("adapt");

        assert_eq!(model.summary(), &summary);
        assert_eq!(model.summary().kind(), FrequentistSummaryKind::EstimateCovariance);
        assert_eq!(model.summary().identity(), summary_identity);
        assert_eq!(model.sampling_approximation(), &fields.sampling);
        assert_eq!(model.selection(), &fields.selection);
        assert_eq!(model.dependence(), &fields.dependence);
        assert_eq!(model.origin(), &fields.origin);
        assert_eq!(model.sampling_approximation().name(), "normal_approximation");
        assert_eq!(model.selection().name(), "complete_cases");
        assert_eq!(model.dependence().name(), "cluster_dependence");
        assert_eq!(model.origin().name(), "trial_estimator");
        assert_ne!(model.identity(), summary_identity);

        let again = SummaryObservationModel::try_adapt(
            summary,
            fields.sampling,
            fields.selection,
            fields.dependence,
            fields.origin,
        )
        .expect("again");
        assert_eq!(again.identity(), model.identity());

        let other_origin = Origin::new("registry_estimator").expect("origin");
        let shifted = SummaryObservationModel::try_adapt(
            estimate("ate"),
            SamplingApproximation::new("normal_approximation").expect("sampling"),
            Selection::new("complete_cases").expect("selection"),
            Dependence::new("cluster_dependence").expect("dependence"),
            other_origin,
        )
        .expect("shifted");
        assert_ne!(shifted.identity(), model.identity());
        assert_eq!(shifted.summary().kind(), FrequentistSummaryKind::EstimateCovariance);
        assert_eq!(shifted.summary().identity(), summary_identity);
    }

    #[test]
    fn likelihood_update_is_permitted_and_bayesian_law_is_refused() {
        let summary = estimate("ate");
        let fields = records();
        let model = SummaryObservationModel::try_adapt(
            summary.clone(),
            fields.sampling,
            fields.selection,
            fields.dependence,
            fields.origin,
        )
        .expect("adapt");
        let before = model.identity();
        let admission = model.validate_use(SummaryUse::LikelihoodUpdate).expect("update");

        assert_eq!(admission.summary_kind(), FrequentistSummaryKind::EstimateCovariance);
        assert_eq!(admission.summary_name(), "ate");
        assert_eq!(admission.summary_identity(), summary.identity());
        assert_eq!(admission.model_identity(), before);
        assert_eq!(admission.estimate().to_bits(), 1.25_f64.to_bits());
        assert_eq!(
            admission.covariance().entries().first().copied().map(f64::to_bits),
            Some(0.25_f64.to_bits())
        );
        let checklist = model.checklist().as_likelihood();
        assert_eq!(checklist.detail(), "summary_observation.checklist_not_likelihood");
        assert!(checklist.message().contains("not a likelihood"));
        assert!(checklist.message().contains("checklist"));
        assert_eq!(checklist.next_attempt(), Some(SummaryNextAttempt::SupplyEstimateCovariance));
        assert!(checklist.summary_kind().is_none());
        assert_ne!(admission.summary_identity(), admission.model_identity());
        assert_ne!(admission.summary_kind(), FrequentistSummaryKind::Interval);
        assert_eq!(model.summary().kind(), FrequentistSummaryKind::EstimateCovariance);
        assert_eq!(model.identity(), before);

        let refusal = model.validate_use(SummaryUse::BayesianLaw).expect_err("law");
        assert_eq!(refusal.summary_kind(), Some(FrequentistSummaryKind::EstimateCovariance));
        assert_eq!(refusal.code(), "distribution_meaning_mismatch");
        assert_eq!(refusal.detail(), "summary_observation.not_bayesian_law");
        assert_eq!(refusal.next_attempt(), Some(SummaryNextAttempt::UseAsLikelihoodUpdate));
        assert_eq!(refusal.next_attempt().expect("next").as_str(), "use_as_likelihood_update");
        assert!(refusal.message().contains("not a Bayesian law"));
        assert!(refusal.message().contains(SummaryNextAttempt::UseAsLikelihoodUpdate.phrase()));
        assert!(refusal.message().contains("ate"));
        assert!(antecedent_core::reason_code::is_runtime_refusal(refusal.code()));
        assert_eq!(model.summary().kind(), summary.kind());
        assert_eq!(model.summary().identity(), summary.identity());
        assert_eq!(model.identity(), before);
        assert_ne!(SummaryUse::LikelihoodUpdate, SummaryUse::BayesianLaw);
    }

    #[test]
    fn blank_records_are_refused() {
        let cases = [
            (
                FrequentistSummary::estimate_covariance(" ", 0.0_f64, 1.0_f64).err(),
                "summary is blank",
            ),
            (FrequentistSummary::interval("\n").err(), "summary is blank"),
            (SamplingApproximation::new("").err(), "sampling approximation is blank"),
            (Selection::new("\t").err(), "selection is blank"),
            (Dependence::new("  ").err(), "dependence is blank"),
            (Origin::new("").err(), "origin is blank"),
        ];
        for (refusal, expected) in cases {
            let refusal = refusal.expect("blank");
            assert_eq!(refusal.code(), "invalid_argument");
            assert_eq!(refusal.detail(), "summary_observation.blank_record");
            assert_eq!(refusal.message(), expected);
            assert!(refusal.next_attempt().is_none());
            assert!(refusal.summary_kind().is_none());
            assert!(antecedent_core::reason_code::is_runtime_refusal(refusal.code()));
        }
    }

    #[test]
    fn reads_estimate_and_covariance_and_interval_refuses_bayesian_likelihood() {
        let summary =
            FrequentistSummary::estimate_covariance("ate", 1.5_f64, 0.04_f64).expect("estimate");
        let covariance = summary.covariance().expect("covariance");

        assert_eq!(summary.kind(), FrequentistSummaryKind::EstimateCovariance);
        assert_eq!(summary.name(), "ate");
        assert_eq!(summary.estimate().map(f64::to_bits), Some(1.5_f64.to_bits()));
        assert_eq!(covariance.dimension(), 1);
        assert_eq!(
            covariance.entries().first().map(|entry| entry.to_bits()),
            Some(0.04_f64.to_bits()),
        );
        assert_eq!(
            FrequentistSummary::estimate_covariance("ate", 1.5_f64, 0.04_f64)
                .expect("again")
                .identity(),
            summary.identity(),
        );
        assert_ne!(
            FrequentistSummary::estimate_covariance("ate", 1.5_f64, 0.09_f64)
                .expect("other")
                .identity(),
            summary.identity(),
        );

        let matrix =
            SummaryCovariance::matrix(2, vec![1.0_f64, 0.2_f64, 0.3_f64, 4.0_f64]).expect("matrix");
        let joint =
            FrequentistSummary::estimate_covariance("joint", -0.5_f64, matrix).expect("joint");
        let stored = joint.covariance().expect("covariance");
        assert_eq!(joint.estimate().map(f64::to_bits), Some((-0.5_f64).to_bits()));
        assert_eq!(stored.dimension(), 2);
        assert_eq!(stored.entries().len(), 4);
        assert_eq!(stored.entries().get(1).map(|entry| entry.to_bits()), Some(0.2_f64.to_bits()));
        assert_eq!(stored.entries().get(2).map(|entry| entry.to_bits()), Some(0.3_f64.to_bits()));
        assert_eq!(stored.entries().get(3).map(|entry| entry.to_bits()), Some(4.0_f64.to_bits()));

        let interval = interval("ate");
        assert!(interval.estimate().is_none());
        assert!(interval.covariance().is_none());
        let refusal = interval.as_bayesian_likelihood();
        assert_eq!(interval.kind(), FrequentistSummaryKind::Interval);
        assert_eq!(refusal.summary_kind(), Some(FrequentistSummaryKind::Interval));
        assert_eq!(refusal.detail(), "summary_observation.not_bayesian_likelihood");
        assert_eq!(refusal.next_attempt(), Some(SummaryNextAttempt::ReportAsFrequentistInterval));
        assert!(refusal.message().contains("not a Bayesian likelihood"));
        assert!(refusal.message().contains("not a Bayesian law"));
    }

    #[test]
    fn non_finite_estimate_and_misshapen_covariance_are_refused() {
        let cases = [
            (
                FrequentistSummary::estimate_covariance("ate", f64::NAN, 1.0_f64).err(),
                "summary_observation.non_finite_value",
                "estimate is not finite",
            ),
            (
                FrequentistSummary::estimate_covariance("ate", 1.0_f64, f64::INFINITY).err(),
                "summary_observation.non_finite_value",
                "covariance is not finite",
            ),
            (
                SummaryCovariance::matrix(0, vec![]).err(),
                "summary_observation.covariance_shape",
                "covariance is not a square matrix",
            ),
            (
                SummaryCovariance::matrix(2, vec![1.0_f64, 0.0_f64, 0.0_f64]).err(),
                "summary_observation.covariance_shape",
                "covariance is not a square matrix",
            ),
            (
                SummaryCovariance::matrix(1, vec![f64::NEG_INFINITY]).err(),
                "summary_observation.non_finite_value",
                "covariance is not finite",
            ),
        ];
        for (refusal, detail, message) in cases {
            let refusal = refusal.expect("refused");
            assert_eq!(refusal.code(), "invalid_argument");
            assert_eq!(refusal.detail(), detail);
            assert_eq!(refusal.message(), message);
            assert!(refusal.next_attempt().is_none());
            assert!(refusal.summary_kind().is_none());
            assert!(antecedent_core::reason_code::is_runtime_refusal(refusal.code()));
        }
    }

    #[test]
    fn next_attempt_names_differ() {
        assert_eq!(SummaryNextAttempt::ALL.len(), 4);
        assert_eq!(FrequentistSummaryKind::ALL.len(), 2);
        assert_eq!(SummaryUse::ALL.len(), 2);
        for (index, attempt) in SummaryNextAttempt::ALL.iter().enumerate() {
            assert_eq!(SummaryNextAttempt::from_name(attempt.as_str()), Some(*attempt));
            for other in SummaryNextAttempt::ALL.iter().skip(index + 1) {
                assert_ne!(attempt.as_str(), other.as_str());
                assert_ne!(attempt.phrase(), other.phrase());
            }
        }
        for kind in FrequentistSummaryKind::ALL {
            assert_eq!(FrequentistSummaryKind::from_name(kind.as_str()), Some(kind));
        }
        assert!(SummaryNextAttempt::from_name("bayesian_law").is_none());
        assert!(FrequentistSummaryKind::from_name("bayesian_likelihood").is_none());
    }
}
