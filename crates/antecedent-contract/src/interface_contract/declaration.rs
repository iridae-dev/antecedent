//! The interface contract declaration: ports, supports, operations,
//! uncertainty meaning, calibration labels and the canonical identity.
//!
//! The contract is data. It holds no estimator input and runs no estimator; it
//! states what a component produces and consumes in the vocabulary the rest of
//! the contract crate already owns (the scientific quantity, the distribution
//! meaning, the measurement model identity, the evidence-ledger source
//! identity, the composition stage), so a connection can be decided from two
//! declarations before anything is executed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fmt::Write as _;

use antecedent_core::identity::{CanonicalWriter, SemanticDigest};

use super::refusal::{
    InterfaceRefusal, blank_field, calibration_record_blank, calibration_without_interval,
    duplicate_port, inherited_input_unknown, invalid_digest, invalid_quantity, invalid_support,
    missing_operations, point_only_with_uncertainty, posterior_without_posterior_meaning,
    uncertainty_meaning_required, uncertainty_meaning_unused, unknown_provenance_stage,
};
use crate::composition_provenance::CompositionStage;
use crate::evidence_ledger::SourceId;
use crate::quantity::meaning_tag;
use crate::{DistributionMeaning, ExternalCapability, ScientificQuantity};
use antecedent_core::named_variants;

/// Version word written into every interface contract identity. Bump only
/// when the canonical encoding below changes.
pub const INTERFACE_CONTRACT_VERSION: u64 = 1;
// The sizes below are the defaults that decoders of untrusted contract records apply to their
// own input. Validating a declaration is linear in its size, so the contract itself imposes no
// count or length bound; memory is the only limit.

/// Default bound on produced ports, and on consumed ports, that a decoder of untrusted input
/// applies to one contract record.
pub const MAX_INTERFACE_PORTS: usize = 256;
/// Default bound on an identity or name, in bytes, that a decoder of untrusted input applies.
pub const MAX_INTERFACE_TEXT_BYTES: usize = 4096;
/// Default bound on measurement models, evidence sources, support levels or provenance stages
/// per port that a decoder of untrusted input applies.
pub const MAX_PORT_SET_ITEMS: usize = 1024;

/// Domain tag of the canonical contract stream.
pub(super) const TAG_CONTRACT: &str = "antecedent.interface_contract";

/// An operation a produced port offers or a consumed port requires. No
/// operation implies another: a mean does not determine a law, draws do not
/// determine a covariance, a posterior is not an interval.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ContractOperation {
    /// One point value.
    Point,
    /// The mean of the quantity.
    Mean,
    /// The law of the quantity (evaluable distribution).
    Law,
    /// Independent or marginal draws of the quantity.
    Draws,
    /// Draws aligned with other quantities of the same joint law.
    JointDraws,
    /// A covariance with other quantities of a declared joint law.
    Covariance,
    /// An interval.
    Interval,
    /// A posterior distribution.
    Posterior,
}

named_variants! {
    ContractOperation {
        Point => "point",
        Mean => "mean",
        Law => "law",
        Draws => "draws",
        JointDraws => "joint_draws",
        Covariance => "covariance",
        Interval => "interval",
        Posterior => "posterior",
    }
}

impl ContractOperation {
    /// Canonical tag word of the operation.
    #[must_use]
    pub const fn tag(self) -> u64 {
        match self {
            Self::Point => 1,
            Self::Mean => 2,
            Self::Law => 3,
            Self::Draws => 4,
            Self::JointDraws => 5,
            Self::Covariance => 6,
            Self::Interval => 7,
            Self::Posterior => 8,
        }
    }

    /// Whether the operation carries uncertainty whose meaning must be stated.
    #[must_use]
    pub const fn carries_uncertainty(self) -> bool {
        !matches!(self, Self::Point | Self::Mean)
    }

    /// The contract operation a provider capability discloses, when it
    /// discloses one. Sampling is draws; a distribution function, quantile or
    /// log density is a law; a posterior update is a posterior. Conditional,
    /// interventional, predictive, factor and utility capabilities answer
    /// other questions and map to no operation here.
    #[must_use]
    pub const fn from_capability(capability: ExternalCapability) -> Option<Self> {
        match capability {
            ExternalCapability::Sample => Some(Self::Draws),
            ExternalCapability::Cdf
            | ExternalCapability::Quantile
            | ExternalCapability::LogProbability => Some(Self::Law),
            ExternalCapability::Mean => Some(Self::Mean),
            ExternalCapability::Covariance => Some(Self::Covariance),
            ExternalCapability::Update => Some(Self::Posterior),
            _ => None,
        }
    }
}

/// The operations a set of provider capabilities discloses.
#[must_use]
pub fn operations_of_capabilities(
    capabilities: &[ExternalCapability],
) -> BTreeSet<ContractOperation> {
    capabilities.iter().filter_map(|c| ContractOperation::from_capability(*c)).collect()
}

/// The inferential paradigm an uncertainty statement belongs to.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UncertaintyParadigm {
    /// A distribution of belief given the data (posterior, posterior predictive).
    Posterior,
    /// The repeated-sampling distribution of an estimator.
    Sampling,
    /// A resampling distribution conditional on one data snapshot.
    Resampling,
    /// The empirical distribution of observed outcomes.
    Empirical,
}

named_variants! {
    UncertaintyParadigm {
        Posterior => "posterior",
        Sampling => "sampling",
        Resampling => "resampling",
        Empirical => "empirical",
    }
}

/// What a covariance, law, draws, interval or posterior means.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct UncertaintyMeaning {
    /// What the randomness represents.
    pub meaning: DistributionMeaning,
}

impl UncertaintyMeaning {
    /// Every distribution meaning an uncertainty statement can carry.
    pub const ALL_MEANINGS: [DistributionMeaning; 7] = [
        DistributionMeaning::ParameterPosterior,
        DistributionMeaning::CausalFunctionalPosterior,
        DistributionMeaning::PosteriorPredictive,
        DistributionMeaning::InterventionalPredictive,
        DistributionMeaning::EstimatorSampling,
        DistributionMeaning::Bootstrap,
        DistributionMeaning::EmpiricalOutcome,
    ];

    /// A meaning.
    #[must_use]
    pub const fn new(meaning: DistributionMeaning) -> Self {
        Self { meaning }
    }

    /// The paradigm the meaning belongs to.
    #[must_use]
    pub const fn paradigm(self) -> UncertaintyParadigm {
        match self.meaning {
            DistributionMeaning::ParameterPosterior
            | DistributionMeaning::CausalFunctionalPosterior
            | DistributionMeaning::PosteriorPredictive
            | DistributionMeaning::InterventionalPredictive => UncertaintyParadigm::Posterior,
            DistributionMeaning::EstimatorSampling => UncertaintyParadigm::Sampling,
            DistributionMeaning::Bootstrap => UncertaintyParadigm::Resampling,
            DistributionMeaning::EmpiricalOutcome => UncertaintyParadigm::Empirical,
        }
    }

    /// Stable `snake_case` name of the meaning.
    #[must_use]
    pub const fn name(self) -> &'static str {
        meaning_tag(self.meaning)
    }

    /// Parse the name written by [`Self::name`].
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL_MEANINGS.into_iter().find(|meaning| meaning_tag(*meaning) == name).map(Self::new)
    }
}

/// What calibration a produced quantity holds. The label is a declaration:
/// the contract layer carries it and never upgrades it; the record is
/// verified by the artifact that holds it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CalibrationStatus {
    /// No calibration statement.
    None,
    /// A point value; no uncertainty statement is made or licensed.
    PointOnly,
    /// An uncertainty statement is made whose coverage has not been measured.
    Unmeasured,
    /// Coverage was measured, in the record with this identity.
    Measured {
        /// Identity of the measured calibration record.
        record_id: String,
    },
    /// Declared to be inherited from a consumed port of the same component;
    /// resolved against the upstream producer when a composition is checked. The
    /// output must keep the upstream quantity and uncertainty coordinates; the
    /// declaration asserts that the component forwards the calibrated quantity.
    Inherited {
        /// Name of the consumed port the calibration is inherited from.
        input: String,
    },
}

impl CalibrationStatus {
    /// Stable `snake_case` label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PointOnly => "point_only",
            Self::Unmeasured => "unmeasured",
            Self::Measured { .. } => "measured",
            Self::Inherited { .. } => "inherited",
        }
    }

    /// Whether the status is a measured record.
    #[must_use]
    pub const fn is_measured(&self) -> bool {
        matches!(self, Self::Measured { .. })
    }
}

/// What a consumed port requires of the calibration of its input.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum CalibrationRequirement {
    /// No calibration is required; the connection still reports the label held.
    NotRequired,
    /// The input must hold a measured calibration record.
    Measured,
}

named_variants! {
    CalibrationRequirement {
        NotRequired => "not_required",
        Measured => "measured",
    }
}

/// The evidence a produced quantity rests on, in evidence-ledger terms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvidenceBasis {
    /// The producer does not declare its evidence; overlap with other inputs
    /// is not computable.
    Undeclared,
    /// The evidence sources (ledger source identities) the value rests on.
    Declared(BTreeSet<SourceId>),
}

impl EvidenceBasis {
    /// The declared sources, or `None` when undeclared.
    #[must_use]
    pub const fn sources(&self) -> Option<&BTreeSet<SourceId>> {
        match self {
            Self::Undeclared => None,
            Self::Declared(sources) => Some(sources),
        }
    }
}

/// What a consumed port requires of the dependence of its input.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DependenceRequirement {
    /// No requirement.
    Unconstrained,
    /// The input declares the evidence sources it rests on.
    DeclaredEvidence,
    /// As [`Self::DeclaredEvidence`], and the evidence is disjoint from that of
    /// every other input of the component that requires the same.
    DisjointEvidence,
}

named_variants! {
    DependenceRequirement {
        Unconstrained => "unconstrained",
        DeclaredEvidence => "declared_evidence",
        DisjointEvidence => "disjoint_evidence",
    }
}

impl DependenceRequirement {
    /// Whether the evidence of the input must be declared.
    #[must_use]
    pub const fn requires_declared(self) -> bool {
        !matches!(self, Self::Unconstrained)
    }
}

/// The values a quantity takes, or a consumer accepts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SupportDomain {
    /// Any finite real.
    Real,
    /// A real that is zero or positive.
    NonNegative,
    /// Zero or one.
    Binary,
    /// A closed interval; both bounds stored as `f64` bit patterns.
    Interval {
        /// Lower bound bits.
        lower_bits: u64,
        /// Upper bound bits.
        upper_bits: u64,
    },
    /// A finite set of named levels, sorted and distinct.
    Categorical {
        /// The levels.
        levels: Vec<String>,
    },
}

impl SupportDomain {
    /// A closed interval `[lower, upper]`.
    #[must_use]
    pub fn interval(lower: f64, upper: f64) -> Self {
        Self::Interval { lower_bits: lower.to_bits(), upper_bits: upper.to_bits() }
    }

    /// Stable kind name.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::NonNegative => "non_negative",
            Self::Binary => "binary",
            Self::Interval { .. } => "interval",
            Self::Categorical { .. } => "categorical",
        }
    }

    /// A short human-readable rendering for refusals.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Interval { lower_bits, upper_bits } => {
                format!("interval[{},{}]", f64::from_bits(*lower_bits), f64::from_bits(*upper_bits))
            }
            Self::Categorical { levels } => format!("categorical{{{}}}", levels.join(",")),
            other => other.kind().to_owned(),
        }
    }

    /// Whether the declaration is well formed: finite ordered interval bounds,
    /// sorted distinct non-blank levels.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        match self {
            Self::Real | Self::NonNegative | Self::Binary => true,
            Self::Interval { lower_bits, upper_bits } => {
                let (lower, upper) = (f64::from_bits(*lower_bits), f64::from_bits(*upper_bits));
                lower.is_finite() && upper.is_finite() && lower <= upper
            }
            Self::Categorical { levels } => {
                !levels.is_empty()
                    && levels.iter().all(|l| !l.trim().is_empty())
                    && levels.windows(2).all(|pair| pair[0] < pair[1])
            }
        }
    }

    /// Whether a consumer that accepts `self` accepts every value of
    /// `produced`. Containment only: a binary quantity is accepted by a real
    /// or non-negative domain and by an interval that holds zero and one; a
    /// categorical domain accepts a categorical domain whose levels it lists;
    /// nothing else is inferred.
    #[must_use]
    pub fn accepts(&self, produced: &Self) -> bool {
        let bounds = |support: &Self| match support {
            Self::Interval { lower_bits, upper_bits } => {
                Some((f64::from_bits(*lower_bits), f64::from_bits(*upper_bits)))
            }
            _ => None,
        };
        match (self, produced) {
            (Self::Real, Self::Real | Self::NonNegative | Self::Binary | Self::Interval { .. })
            | (Self::NonNegative, Self::NonNegative | Self::Binary)
            | (Self::Binary, Self::Binary) => true,
            (Self::NonNegative, Self::Interval { .. }) => {
                bounds(produced).is_some_and(|(lower, _)| lower >= 0.0)
            }
            (Self::Interval { .. }, Self::Binary) => {
                bounds(self).is_some_and(|(lower, upper)| lower <= 0.0 && upper >= 1.0)
            }
            (Self::Interval { .. }, Self::Interval { .. }) => {
                match (bounds(self), bounds(produced)) {
                    (Some((lower, upper)), Some((p_lower, p_upper))) => {
                        lower <= p_lower && p_upper <= upper
                    }
                    _ => false,
                }
            }
            (Self::Categorical { levels }, Self::Categorical { levels: produced_levels }) => {
                produced_levels.iter().all(|level| levels.contains(level))
            }
            _ => false,
        }
    }
}

/// A quantity a component produces.
#[derive(Clone, Debug, PartialEq)]
pub struct ProducedPort {
    /// Port name, unique among the component's produced ports.
    pub name: String,
    /// The full semantic coordinate of the quantity (units, population, regime,
    /// functional, conditioning, transform).
    pub quantity: ScientificQuantity,
    /// The estimand or semantic identity the value answers.
    pub semantic_identity: String,
    /// The values the quantity takes.
    pub support: SupportDomain,
    /// What the component offers.
    pub operations: BTreeSet<ContractOperation>,
    /// What the offered covariance, law, draws, interval or posterior means.
    pub uncertainty: Option<UncertaintyMeaning>,
    /// What calibration the quantity holds.
    pub calibration: CalibrationStatus,
    /// Identities (hex) of the measurement models the quantity is
    /// established under.
    pub measurement_models: BTreeSet<String>,
    /// The evidence the value rests on.
    pub evidence: EvidenceBasis,
    /// Composition stage names whose provenance the component supplies.
    pub provenance: BTreeSet<String>,
}

impl ProducedPort {
    /// A port with no uncertainty, calibration `none`, no measurement models,
    /// undeclared evidence and no provenance.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        quantity: ScientificQuantity,
        semantic_identity: impl Into<String>,
        support: SupportDomain,
        operations: impl IntoIterator<Item = ContractOperation>,
    ) -> Self {
        Self {
            name: name.into(),
            quantity,
            semantic_identity: semantic_identity.into(),
            support,
            operations: operations.into_iter().collect(),
            uncertainty: None,
            calibration: CalibrationStatus::None,
            measurement_models: BTreeSet::new(),
            evidence: EvidenceBasis::Undeclared,
            provenance: BTreeSet::new(),
        }
    }

    /// State what the offered uncertainty means.
    #[must_use]
    pub fn with_uncertainty(mut self, meaning: DistributionMeaning) -> Self {
        self.uncertainty = Some(UncertaintyMeaning::new(meaning));
        self
    }

    /// State the calibration label.
    #[must_use]
    pub fn with_calibration(mut self, calibration: CalibrationStatus) -> Self {
        self.calibration = calibration;
        self
    }

    /// Declare the measurement models (hex identities).
    #[must_use]
    pub fn with_measurement_models<S: Into<String>>(
        mut self,
        models: impl IntoIterator<Item = S>,
    ) -> Self {
        self.measurement_models = models.into_iter().map(Into::into).collect();
        self
    }

    /// Declare the evidence sources.
    #[must_use]
    pub fn with_evidence(mut self, sources: impl IntoIterator<Item = SourceId>) -> Self {
        let sources: BTreeSet<_> = sources.into_iter().collect();
        self.evidence = if sources.is_empty() {
            EvidenceBasis::Undeclared
        } else {
            EvidenceBasis::Declared(sources)
        };
        self
    }

    /// Declare the provenance stages supplied.
    #[must_use]
    pub fn with_provenance<S: Into<String>>(mut self, stages: impl IntoIterator<Item = S>) -> Self {
        self.provenance = stages.into_iter().map(Into::into).collect();
        self
    }
}

/// A quantity a component consumes.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedPort {
    /// Port name, unique among the component's consumed ports.
    pub name: String,
    /// The full semantic coordinate the component requires, including the
    /// population and regime it is valid for.
    pub quantity: ScientificQuantity,
    /// The estimand or semantic identity the component requires.
    pub semantic_identity: String,
    /// The values the component accepts.
    pub support: SupportDomain,
    /// What the component requires.
    pub operations: BTreeSet<ContractOperation>,
    /// What the required covariance, law, draws, interval or posterior must mean.
    pub uncertainty: Option<UncertaintyMeaning>,
    /// What the component requires of the input's calibration.
    pub calibration: CalibrationRequirement,
    /// Identities (hex) of the measurement models the input must be
    /// established under; the set must match exactly.
    pub measurement_models: BTreeSet<String>,
    /// What the component requires of the input's evidence.
    pub dependence: DependenceRequirement,
    /// Composition stage names whose provenance must accompany the input.
    pub provenance: BTreeSet<String>,
}

impl ConsumedPort {
    /// A port with no uncertainty, no calibration requirement, no
    /// measurement models, unconstrained dependence and no provenance.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        quantity: ScientificQuantity,
        semantic_identity: impl Into<String>,
        support: SupportDomain,
        operations: impl IntoIterator<Item = ContractOperation>,
    ) -> Self {
        Self {
            name: name.into(),
            quantity,
            semantic_identity: semantic_identity.into(),
            support,
            operations: operations.into_iter().collect(),
            uncertainty: None,
            calibration: CalibrationRequirement::NotRequired,
            measurement_models: BTreeSet::new(),
            dependence: DependenceRequirement::Unconstrained,
            provenance: BTreeSet::new(),
        }
    }

    /// State what the required uncertainty must mean.
    #[must_use]
    pub fn with_uncertainty(mut self, meaning: DistributionMeaning) -> Self {
        self.uncertainty = Some(UncertaintyMeaning::new(meaning));
        self
    }

    /// Require a calibration status.
    #[must_use]
    pub fn with_calibration(mut self, requirement: CalibrationRequirement) -> Self {
        self.calibration = requirement;
        self
    }

    /// Declare the measurement models (hex identities) required.
    #[must_use]
    pub fn with_measurement_models<S: Into<String>>(
        mut self,
        models: impl IntoIterator<Item = S>,
    ) -> Self {
        self.measurement_models = models.into_iter().map(Into::into).collect();
        self
    }

    /// Require a dependence status.
    #[must_use]
    pub fn with_dependence(mut self, requirement: DependenceRequirement) -> Self {
        self.dependence = requirement;
        self
    }

    /// Require provenance stages.
    #[must_use]
    pub fn with_provenance<S: Into<String>>(mut self, stages: impl IntoIterator<Item = S>) -> Self {
        self.provenance = stages.into_iter().map(Into::into).collect();
        self
    }
}

/// The port-level facts a mechanism, bridge or measurement-model declaration
/// does not itself state but an interface contract needs: the support of the
/// quantity, the identity of the estimand it answers, what its uncertainty
/// means and the measurement models it is established under.
///
/// The adapters that project those declarations to an [`InterfaceContract`]
/// take these facts explicitly. Nothing is inferred: a support that is not
/// supplied is [`SupportDomain::Real`], no uncertainty meaning is stated, and
/// no measurement model is claimed.
#[derive(Clone, Debug, PartialEq)]
pub struct PortFacts {
    /// The values the quantity takes (produced) or the consumer accepts.
    pub support: SupportDomain,
    /// The estimand or semantic identity; `None` names the variable.
    pub semantic_identity: Option<String>,
    /// What the offered or required uncertainty means.
    pub uncertainty: Option<DistributionMeaning>,
    /// Identities (hex) of the measurement models the value is established
    /// under (produced) or must be established under (consumed).
    pub measurement_models: BTreeSet<String>,
}

impl PortFacts {
    /// Facts with the given support, no stated estimand, uncertainty meaning or
    /// measurement model.
    #[must_use]
    pub fn new(support: SupportDomain) -> Self {
        Self {
            support,
            semantic_identity: None,
            uncertainty: None,
            measurement_models: BTreeSet::new(),
        }
    }

    /// State the estimand or semantic identity.
    #[must_use]
    pub fn with_semantic_identity(mut self, identity: impl Into<String>) -> Self {
        self.semantic_identity = Some(identity.into());
        self
    }

    /// State what the offered or required uncertainty means.
    #[must_use]
    pub fn with_uncertainty(mut self, meaning: DistributionMeaning) -> Self {
        self.uncertainty = Some(meaning);
        self
    }

    /// State the measurement models (hex identities).
    #[must_use]
    pub fn with_measurement_models<S: Into<String>>(
        mut self,
        models: impl IntoIterator<Item = S>,
    ) -> Self {
        self.measurement_models = models.into_iter().map(Into::into).collect();
        self
    }

    /// A produced port for `quantity` under these facts. The calibration is
    /// `calibration` when stated, otherwise `unmeasured` for an offering that
    /// carries uncertainty and `point_only` for one that does not; it is never
    /// measured by default.
    pub(crate) fn produced(
        &self,
        name: &str,
        quantity: &ScientificQuantity,
        operations: BTreeSet<ContractOperation>,
        provenance: impl IntoIterator<Item = &'static str>,
        calibration: Option<CalibrationStatus>,
        evidence: EvidenceBasis,
    ) -> ProducedPort {
        let carries = uncertainty_ops(&operations);
        let mut port = ProducedPort::new(
            name,
            quantity.clone(),
            self.semantic_identity.clone().unwrap_or_else(|| quantity.variable_id.clone()),
            self.support.clone(),
            operations,
        )
        .with_measurement_models(self.measurement_models.iter().cloned())
        .with_provenance(provenance);
        if let Some(meaning) = self.uncertainty {
            port = port.with_uncertainty(meaning);
        }
        port = port.with_calibration(calibration.unwrap_or(if carries {
            CalibrationStatus::Unmeasured
        } else {
            CalibrationStatus::PointOnly
        }));
        port.evidence = evidence;
        port
    }

    /// A consumed port for `quantity` under these facts.
    pub(crate) fn consumed(
        &self,
        name: &str,
        quantity: &ScientificQuantity,
        operations: BTreeSet<ContractOperation>,
        dependence: DependenceRequirement,
    ) -> ConsumedPort {
        let mut port = ConsumedPort::new(
            name,
            quantity.clone(),
            self.semantic_identity.clone().unwrap_or_else(|| quantity.variable_id.clone()),
            self.support.clone(),
            operations,
        )
        .with_measurement_models(self.measurement_models.iter().cloned())
        .with_dependence(dependence);
        if let Some(meaning) = self.uncertainty {
            port = port.with_uncertainty(meaning);
        }
        port
    }
}

/// What one component produces and consumes.
#[derive(Clone, Debug, PartialEq)]
pub struct InterfaceContract {
    /// Stable component identity.
    pub component: String,
    /// Quantities the component produces.
    pub produces: Vec<ProducedPort>,
    /// Quantities the component consumes.
    pub consumes: Vec<ConsumedPort>,
}

fn check_text(field: &str, text: &str) -> Result<(), InterfaceRefusal> {
    if text.trim().is_empty() {
        return Err(blank_field(field));
    }
    Ok(())
}

/// Whether `text` is 64 lowercase hex characters. Declared model identities are canonical
/// identity text, so uppercase is refused (unlike externally supplied digests).
pub(super) fn is_lower_hex_digest(text: &str) -> bool {
    antecedent_core::digest_from_lower_hex(text).is_some()
}

fn check_models(field: &str, models: &BTreeSet<String>) -> Result<(), InterfaceRefusal> {
    if models.iter().any(|m| !is_lower_hex_digest(m)) {
        return Err(invalid_digest(field));
    }
    Ok(())
}

fn check_stages(stages: &BTreeSet<String>) -> Result<(), InterfaceRefusal> {
    if let Some(stage) = stages.iter().find(|s| CompositionStage::from_name(s).is_none()) {
        return Err(unknown_provenance_stage(stage));
    }
    Ok(())
}

fn check_common(
    port: &str,
    quantity: &ScientificQuantity,
    semantic_identity: &str,
    support: &SupportDomain,
    operations: &BTreeSet<ContractOperation>,
) -> Result<(), InterfaceRefusal> {
    check_text("port.name", port)?;
    quantity.validate().map_err(|_| invalid_quantity(port))?;
    check_text("port.semantic_identity", semantic_identity)?;
    if !support.is_well_formed() {
        return Err(invalid_support(port));
    }
    if operations.is_empty() {
        return Err(missing_operations(port));
    }
    Ok(())
}

fn uncertainty_ops(operations: &BTreeSet<ContractOperation>) -> bool {
    operations.iter().copied().any(ContractOperation::carries_uncertainty)
}

fn check_uncertainty(
    port: &str,
    operations: &BTreeSet<ContractOperation>,
    uncertainty: Option<UncertaintyMeaning>,
) -> Result<(), InterfaceRefusal> {
    match (uncertainty_ops(operations), uncertainty) {
        (true, None) => Err(uncertainty_meaning_required(port)),
        (false, Some(_)) => Err(uncertainty_meaning_unused(port)),
        (true, Some(meaning)) => {
            if operations.contains(&ContractOperation::Posterior)
                && meaning.paradigm() != UncertaintyParadigm::Posterior
            {
                return Err(posterior_without_posterior_meaning(port));
            }
            Ok(())
        }
        (false, None) => Ok(()),
    }
}

impl ProducedPort {
    fn validate(&self, consumed: &BTreeSet<&str>) -> Result<(), InterfaceRefusal> {
        check_common(
            &self.name,
            &self.quantity,
            &self.semantic_identity,
            &self.support,
            &self.operations,
        )?;
        check_uncertainty(&self.name, &self.operations, self.uncertainty)?;
        check_models("measurement_models", &self.measurement_models)?;
        check_stages(&self.provenance)?;
        match &self.calibration {
            CalibrationStatus::None | CalibrationStatus::Unmeasured => {}
            CalibrationStatus::PointOnly => {
                if uncertainty_ops(&self.operations) {
                    return Err(point_only_with_uncertainty(&self.name));
                }
            }
            CalibrationStatus::Measured { record_id } => {
                check_text("calibration.record_id", record_id)
                    .map_err(|_| calibration_record_blank(&self.name))?;
                let interval_or_posterior = self.operations.contains(&ContractOperation::Interval)
                    || self.operations.contains(&ContractOperation::Posterior);
                if !interval_or_posterior {
                    return Err(calibration_without_interval(&self.name));
                }
            }
            CalibrationStatus::Inherited { input } => {
                if !consumed.contains(input.as_str()) {
                    return Err(inherited_input_unknown(&self.name, input));
                }
            }
        }
        Ok(())
    }
}

impl ConsumedPort {
    fn validate(&self) -> Result<(), InterfaceRefusal> {
        check_common(
            &self.name,
            &self.quantity,
            &self.semantic_identity,
            &self.support,
            &self.operations,
        )?;
        check_uncertainty(&self.name, &self.operations, self.uncertainty)?;
        check_models("measurement_models", &self.measurement_models)?;
        check_stages(&self.provenance)?;
        if self.calibration == CalibrationRequirement::Measured {
            let interval_or_posterior = self.operations.contains(&ContractOperation::Interval)
                || self.operations.contains(&ContractOperation::Posterior);
            if !interval_or_posterior {
                return Err(calibration_without_interval(&self.name));
            }
        }
        Ok(())
    }
}

impl InterfaceContract {
    /// Check the declaration is well formed: identities, quantities, supports,
    /// operations, uncertainty meanings, calibration labels and identities.
    ///
    /// # Errors
    /// The first defect, as an `interface_contract_invalid` refusal that
    /// names the declaration that would make the contract well formed.
    pub fn validate(&self) -> Result<(), InterfaceRefusal> {
        check_text("component", &self.component)?;
        let mut consumed = BTreeSet::new();
        for port in &self.consumes {
            if !consumed.insert(port.name.as_str()) {
                return Err(duplicate_port(&port.name));
            }
        }
        let mut produced = BTreeSet::new();
        for port in &self.produces {
            if !produced.insert(port.name.as_str()) {
                return Err(duplicate_port(&port.name));
            }
        }
        for port in &self.consumes {
            port.validate()?;
        }
        for port in &self.produces {
            port.validate(&consumed)?;
        }
        Ok(())
    }

    /// The produced port with this name.
    #[must_use]
    pub fn produced(&self, name: &str) -> Option<&ProducedPort> {
        self.produces.iter().find(|port| port.name == name)
    }

    /// The consumed port with this name.
    #[must_use]
    pub fn consumed(&self, name: &str) -> Option<&ConsumedPort> {
        self.consumes.iter().find(|port| port.name == name)
    }

    /// The canonical identity of the declaration: a [`CanonicalWriter`] stream
    /// under the tag `antecedent.interface_contract`, encoded as in
    /// `spec/canonicalization-and-identity.md`. Declaration order of ports does
    /// not affect it; a display label of a quantity does.
    ///
    /// The identity is defined for any declaration; a consumer validates the
    /// declaration separately.
    #[must_use]
    pub fn canonical_identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(TAG_CONTRACT);
        writer.word(INTERFACE_CONTRACT_VERSION).text(&self.component);
        let mut produced: Vec<&ProducedPort> = self.produces.iter().collect();
        produced.sort_by(|a, b| a.name.cmp(&b.name));
        writer.word(produced.len() as u64);
        for port in produced {
            write_produced(&mut writer, port);
        }
        let mut consumed: Vec<&ConsumedPort> = self.consumes.iter().collect();
        consumed.sort_by(|a, b| a.name.cmp(&b.name));
        writer.word(consumed.len() as u64);
        for port in consumed {
            write_consumed(&mut writer, port);
        }
        SemanticDigest::from_bytes(writer.finish())
    }
}

fn write_support(writer: &mut CanonicalWriter, support: &SupportDomain) {
    match support {
        SupportDomain::Real => {
            writer.word(1);
        }
        SupportDomain::NonNegative => {
            writer.word(2);
        }
        SupportDomain::Binary => {
            writer.word(3);
        }
        SupportDomain::Interval { lower_bits, upper_bits } => {
            writer.word(4).word(*lower_bits).word(*upper_bits);
        }
        SupportDomain::Categorical { levels } => {
            writer.word(5).texts(levels);
        }
    }
}

fn write_operations(writer: &mut CanonicalWriter, operations: &BTreeSet<ContractOperation>) {
    writer.word(operations.len() as u64);
    for operation in operations {
        writer.word(operation.tag());
    }
}

fn write_uncertainty(writer: &mut CanonicalWriter, uncertainty: Option<UncertaintyMeaning>) {
    match uncertainty {
        None => {
            writer.word(0);
        }
        Some(meaning) => {
            writer.word(1).text(meaning.name()).text(meaning.paradigm().as_str());
        }
    }
}

fn write_set(writer: &mut CanonicalWriter, set: &BTreeSet<String>) {
    writer.word(set.len() as u64);
    for item in set {
        writer.text(item);
    }
}

fn write_produced(writer: &mut CanonicalWriter, port: &ProducedPort) {
    writer.text(&port.name);
    port.quantity.write_canonical(writer);
    writer.text(&port.semantic_identity);
    write_support(writer, &port.support);
    write_operations(writer, &port.operations);
    write_uncertainty(writer, port.uncertainty);
    match &port.calibration {
        CalibrationStatus::None => {
            writer.word(1);
        }
        CalibrationStatus::PointOnly => {
            writer.word(2);
        }
        CalibrationStatus::Unmeasured => {
            writer.word(3);
        }
        CalibrationStatus::Measured { record_id } => {
            writer.word(4).text(record_id);
        }
        CalibrationStatus::Inherited { input } => {
            writer.word(5).text(input);
        }
    }
    write_set(writer, &port.measurement_models);
    match &port.evidence {
        EvidenceBasis::Undeclared => {
            writer.word(0);
        }
        EvidenceBasis::Declared(sources) => {
            writer.word(1).word(sources.len() as u64);
            for source in sources {
                writer.text(&source.to_hex());
            }
        }
    }
    write_set(writer, &port.provenance);
}

fn write_consumed(writer: &mut CanonicalWriter, port: &ConsumedPort) {
    writer.text(&port.name);
    port.quantity.write_canonical(writer);
    writer.text(&port.semantic_identity);
    write_support(writer, &port.support);
    write_operations(writer, &port.operations);
    write_uncertainty(writer, port.uncertainty);
    writer.word(match port.calibration {
        CalibrationRequirement::NotRequired => 0,
        CalibrationRequirement::Measured => 1,
    });
    write_set(writer, &port.measurement_models);
    writer.word(match port.dependence {
        DependenceRequirement::Unconstrained => 0,
        DependenceRequirement::DeclaredEvidence => 1,
        DependenceRequirement::DisjointEvidence => 2,
    });
    write_set(writer, &port.provenance);
}

/// A short rendering of a set of operations for refusals.
pub(super) fn operations_text(operations: &BTreeSet<ContractOperation>) -> String {
    let mut text = String::new();
    for (index, operation) in operations.iter().enumerate() {
        if index > 0 {
            text.push(',');
        }
        let _ = write!(text, "{}", operation.as_str());
    }
    if text.is_empty() { "none".to_owned() } else { text }
}
