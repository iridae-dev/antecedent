//! Typed measurement models: how an observed indicator relates to the
//! scientific quantity it is offered for.
//!
//! [`crate::MeasurementSpec`] is a flag inside a variable schema: a
//! description and a `noisy` bit. It states that a column is measured with
//! error; it does not say with what error, for which population, regime and
//! arm, or what may be concluded about the construct. A [`MeasurementModel`]
//! is that statement, kept separate from the schema so no schema digest moves.
//!
//! # Kinds
//!
//! Six kinds are distinct in type and in refusal ([`MeasurementKind`]): an
//! observed column, a latent scientific construct, a noisy or misclassified
//! indicator, a probabilistic feature assignment, a learned vector
//! representation and a derived variable. A model relates one *indicator* of
//! a non-latent kind to one *construct* of the latent scientific kind; offering
//! a proxy, a learned representation or a derived column *as* the construct is
//! [`MeasurementModelError::ProxyOfferedAsConstruct`], not a silent cast.
//!
//! # Binding
//!
//! A model is bound to a population, a regime and an arm
//! ([`MeasurementBinding`]). Whether the error depends on the treatment arm,
//! the population or time is an explicit [`ErrorStructure`]; the
//! non-differential route is never reused for a model declared differential.
//!
//! # Representations
//!
//! A representation has a coordinate system and a version
//! ([`RepresentationVersion`]). Coordinates change only through an explicit
//! [`RepresentationMapping`]; a mapping carries *values* and no intervention
//! meaning unless it declares evidence that it preserves it
//! ([`MappingSemantics::InterventionPreserving`]).
//!
//! Numerics live in `antecedent-stats` and `antecedent-estimate`; contracts
//! and refusal codes in `antecedent-contract`; the wire in `antecedent-io-core`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::identity::{CanonicalWriter, SemanticDigest};

/// Format version of the canonical measurement-model encoding.
pub const MEASUREMENT_MODEL_FORMAT: u16 = 1;

const TAG_MEASUREMENT: &str = "antecedent.measurement_model.measurement.v1";
const TAG_POPULATION: &str = "antecedent.measurement_model.population.v1";
const TAG_REGIME: &str = "antecedent.measurement_model.regime.v1";
const TAG_ARM: &str = "antecedent.measurement_model.arm.v1";
const TAG_REPRESENTATION: &str = "antecedent.measurement_model.representation.v1";
const TAG_EVIDENCE: &str = "antecedent.measurement_model.evidence.v1";
const TAG_MODEL: &str = "antecedent.measurement_model.model.v1";
const TAG_DATA: &str = "antecedent.measurement_model.data.v1";

/// What kind of quantity a variable is, relative to a measurement.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum MeasurementKind {
    /// Recorded without measurement error; a model for it is not needed.
    Observed,
    /// The scientific construct itself; never recorded directly here.
    LatentScientific,
    /// A recorded value that differs from the construct by error or
    /// misclassification.
    NoisyMisclassified,
    /// A categorical label (or a probability over labels) assigned by a model
    /// from features, characterised by a confusion model.
    ProbabilisticFeatureAssignment,
    /// A learned vector representation: its coordinates have no scientific
    /// meaning of their own.
    LearnedVectorRepresentation,
    /// A deterministic function of other variables.
    Derived,
}

impl MeasurementKind {
    /// Every kind.
    pub const ALL: [Self; 6] = [
        Self::Observed,
        Self::LatentScientific,
        Self::NoisyMisclassified,
        Self::ProbabilisticFeatureAssignment,
        Self::LearnedVectorRepresentation,
        Self::Derived,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Observed => "observed",
            Self::LatentScientific => "latent_scientific",
            Self::NoisyMisclassified => "noisy_misclassified",
            Self::ProbabilisticFeatureAssignment => "probabilistic_feature_assignment",
            Self::LearnedVectorRepresentation => "learned_vector_representation",
            Self::Derived => "derived",
        }
    }

    /// Parse a stable name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    const fn tag(self) -> u64 {
        match self {
            Self::Observed => 1,
            Self::LatentScientific => 2,
            Self::NoisyMisclassified => 3,
            Self::ProbabilisticFeatureAssignment => 4,
            Self::LearnedVectorRepresentation => 5,
            Self::Derived => 6,
        }
    }
}

/// The bounded correction route a model licenses.
///
/// The set is closed by the routes' frozen records; a new route is a new
/// record and a deliberate breaking change, so matches over it are exhaustive.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum MeasurementRoute {
    /// Classical additive continuous error with declared or validated variance.
    ClassicalContinuousError,
    /// Binary misclassification with sensitivity and specificity.
    BinaryMisclassification,
    /// Replicated measurements: the error variance is estimated from replicates.
    RepeatedMeasurements,
    /// Assigned categorical labels with a declared confusion matrix.
    ProbabilisticCategorical,
    /// A latent-variable posterior supplied by an external provider.
    ExternalLatentPosterior,
}

impl MeasurementRoute {
    /// Every route.
    pub const ALL: [Self; 5] = [
        Self::ClassicalContinuousError,
        Self::BinaryMisclassification,
        Self::RepeatedMeasurements,
        Self::ProbabilisticCategorical,
        Self::ExternalLatentPosterior,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ClassicalContinuousError => "classical_continuous_error",
            Self::BinaryMisclassification => "binary_misclassification",
            Self::RepeatedMeasurements => "repeated_measurements",
            Self::ProbabilisticCategorical => "probabilistic_categorical",
            Self::ExternalLatentPosterior => "external_latent_posterior",
        }
    }

    /// Parse a stable name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|route| route.as_str() == name)
    }

    const fn tag(self) -> u64 {
        match self {
            Self::ClassicalContinuousError => 1,
            Self::BinaryMisclassification => 2,
            Self::RepeatedMeasurements => 3,
            Self::ProbabilisticCategorical => 4,
            Self::ExternalLatentPosterior => 5,
        }
    }

    /// Indicator kinds this route accepts.
    #[must_use]
    pub const fn accepted_indicator_kinds(self) -> &'static [MeasurementKind] {
        match self {
            Self::ClassicalContinuousError
            | Self::BinaryMisclassification
            | Self::RepeatedMeasurements => &[MeasurementKind::NoisyMisclassified],
            Self::ProbabilisticCategorical => &[MeasurementKind::ProbabilisticFeatureAssignment],
            Self::ExternalLatentPosterior => {
                &[MeasurementKind::NoisyMisclassified, MeasurementKind::LearnedVectorRepresentation]
            }
        }
    }
}

/// Which factor the measurement error is allowed to depend on.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum DifferentialAxis {
    /// The error differs by treatment arm.
    Arm,
    /// The error differs by population.
    Population,
    /// The error differs over time.
    Time,
}

impl DifferentialAxis {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Arm => "arm",
            Self::Population => "population",
            Self::Time => "time",
        }
    }

    const fn tag(self) -> u64 {
        match self {
            Self::Arm => 1,
            Self::Population => 2,
            Self::Time => 3,
        }
    }
}

/// Whether the error is independent of arm, population and time, or declared
/// to depend on one of them.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ErrorStructure {
    /// The error does not depend on arm, outcome, population or time.
    NonDifferential,
    /// The error depends on the named axis.
    Differential(DifferentialAxis),
}

impl ErrorStructure {
    fn write(self, stream: &mut CanonicalWriter) {
        match self {
            Self::NonDifferential => {
                stream.word(1).word(0);
            }
            Self::Differential(axis) => {
                stream.word(2).word(axis.tag());
            }
        }
    }
}

/// The arm a model applies to.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ArmBinding {
    /// All arms, with arm-specific parameters when declared differential.
    AllArms,
    /// Exactly the named arm; data from another arm is refused.
    Arm(String),
}

/// Population, regime and arm a measurement model was established for.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MeasurementBinding {
    /// Stable population identity.
    pub population_id: String,
    /// Stable regime identity (`observational` is explicit).
    pub regime_id: String,
    /// Arm scope.
    pub arm: ArmBinding,
}

/// Coordinate system and version of an indicator's representation.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RepresentationVersion {
    /// Stable coordinate-system identity (for a scalar sensor, its identity).
    pub coordinate_system_id: String,
    /// Version of that coordinate system.
    pub version: u32,
}

/// What an explicit representation mapping carries across versions.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum MappingSemantics {
    /// Values map across; causal or intervention meaning does not.
    CoordinateRelabeling,
    /// The declaring evidence establishes that interventions on the source
    /// representation correspond to interventions on the target.
    InterventionPreserving {
        /// Evidence identity that establishes the correspondence.
        evidence_id: String,
    },
}

/// An explicit mapping between two representation versions.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RepresentationMapping {
    /// Source representation.
    pub from: RepresentationVersion,
    /// Target representation.
    pub to: RepresentationVersion,
    /// Stable identity of the mapping function and its fit.
    pub mapping_id: String,
    /// What the mapping carries.
    pub semantics: MappingSemantics,
}

impl RepresentationMapping {
    /// Whether this mapping carries intervention meaning.
    #[must_use]
    pub fn carries_intervention_meaning(&self) -> bool {
        matches!(&self.semantics, MappingSemantics::InterventionPreserving { evidence_id } if !evidence_id.trim().is_empty())
    }
}

/// An indicator or a construct.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct MeasurementVariable {
    /// Stable variable identity, not a display label.
    pub id: String,
    /// Kind of quantity.
    pub kind: MeasurementKind,
    /// Declared unit identity (`dimensionless`, a level set, or a coordinate system).
    pub units: String,
}

/// Sensitivity and specificity of one binary indicator, for one arm or shared.
#[derive(Clone, Debug, PartialEq)]
pub struct BinaryArmRates {
    /// The arm these rates apply to; `None` for rates shared by all arms.
    pub arm_id: Option<String>,
    /// `P(observed = 1 | true = 1)`.
    pub sensitivity: f64,
    /// `P(observed = 0 | true = 0)`.
    pub specificity: f64,
    /// Variance of the sensitivity estimate (zero when declared exact).
    pub sensitivity_variance: f64,
    /// Variance of the specificity estimate (zero when declared exact).
    pub specificity_variance: f64,
}

/// Reference to an external latent-variable posterior object.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct PosteriorReference {
    /// Provider identity.
    pub provider_id: String,
    /// Object identity within the provider.
    pub object_id: String,
    /// Provider version.
    pub version_id: String,
    /// Input snapshot.
    pub snapshot_id: String,
    /// Exact request fingerprint.
    pub request_id: String,
}

/// The numerical parameters of one route.
#[derive(Clone, Debug, PartialEq)]
pub enum MeasurementParameters {
    /// Classical additive error.
    ClassicalError {
        /// Error variance of one measurement.
        error_variance: f64,
        /// Standard error of that variance (zero when declared exact).
        error_variance_standard_error: f64,
        /// Smallest implied reliability the correction accepts, in `[0, 1)`.
        min_reliability: f64,
    },
    /// Binary misclassification.
    Binary {
        /// One shared entry when non-differential; one entry per arm when
        /// differential by arm.
        rates: Vec<BinaryArmRates>,
        /// Smallest Youden index the correction accepts, in `(0, 1)`.
        min_youden: f64,
    },
    /// Replicated measurements; the error variance is estimated from the data.
    Repeated {
        /// Fewest units that must carry at least two replicates.
        min_replicated_units: u32,
        /// Smallest implied reliability the correction accepts, in `[0, 1)`.
        min_reliability: f64,
    },
    /// Assigned categorical labels with a declared confusion matrix.
    Categorical {
        /// True-category level names, in the order of the confusion axes.
        levels: Vec<String>,
        /// Row-major `[observed][true]` column-stochastic confusion matrix.
        confusion: Vec<f64>,
        /// Validation units per true category; `None` when declared exact.
        column_counts: Option<Vec<u64>>,
        /// Largest 1-norm condition number the inversion accepts.
        max_condition: f64,
    },
    /// An external latent posterior.
    ExternalPosterior {
        /// Which provider object supplies the posterior.
        reference: PosteriorReference,
        /// Whether the posterior conditions on the outcome (so its draws are
        /// proper imputations for the outcome model).
        conditioned_on_outcome: bool,
        /// Fewest joint draws the route accepts.
        min_draws: u32,
    },
}

impl MeasurementParameters {
    /// The route these parameters belong to.
    #[must_use]
    pub const fn route(&self) -> MeasurementRoute {
        match self {
            Self::ClassicalError { .. } => MeasurementRoute::ClassicalContinuousError,
            Self::Binary { .. } => MeasurementRoute::BinaryMisclassification,
            Self::Repeated { .. } => MeasurementRoute::RepeatedMeasurements,
            Self::Categorical { .. } => MeasurementRoute::ProbabilisticCategorical,
            Self::ExternalPosterior { .. } => MeasurementRoute::ExternalLatentPosterior,
        }
    }

    fn write(&self, stream: &mut CanonicalWriter) {
        match self {
            Self::ClassicalError {
                error_variance,
                error_variance_standard_error,
                min_reliability,
            } => {
                stream
                    .word(1)
                    .real(*error_variance)
                    .real(*error_variance_standard_error)
                    .real(*min_reliability);
            }
            Self::Binary { rates, min_youden } => {
                stream.word(2).real(*min_youden).word(rates.len() as u64);
                for arm in rates {
                    stream
                        .opt_text(arm.arm_id.as_deref())
                        .real(arm.sensitivity)
                        .real(arm.specificity)
                        .real(arm.sensitivity_variance)
                        .real(arm.specificity_variance);
                }
            }
            Self::Repeated { min_replicated_units, min_reliability } => {
                stream.word(3).word(u64::from(*min_replicated_units)).real(*min_reliability);
            }
            Self::Categorical { levels, confusion, column_counts, max_condition } => {
                stream.word(4).texts(levels).reals(confusion).real(*max_condition);
                match column_counts {
                    Some(counts) => {
                        stream.word(1).word(counts.len() as u64);
                        for count in counts {
                            stream.word(*count);
                        }
                    }
                    None => {
                        stream.word(0);
                    }
                }
            }
            Self::ExternalPosterior { reference, conditioned_on_outcome, min_draws } => {
                stream
                    .word(5)
                    .text(&reference.provider_id)
                    .text(&reference.object_id)
                    .text(&reference.version_id)
                    .text(&reference.snapshot_id)
                    .text(&reference.request_id)
                    .word(u64::from(*conditioned_on_outcome))
                    .word(u64::from(*min_draws));
            }
        }
    }
}

/// Where the model's parameters came from.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum ParameterEvidence {
    /// Declared by a named party without a sample behind it here.
    Declared {
        /// Who declared the parameters.
        declarer: String,
    },
    /// Estimated from a validation sample against a reference measurement.
    ValidationSample {
        /// Stable sample identity.
        sample_id: String,
        /// Validation units.
        units: u64,
    },
    /// Estimated from the analysis sample's own replicates.
    ReplicateSample {
        /// Stable sample identity.
        sample_id: String,
    },
    /// Supplied by an external provider; carries no native trust.
    ExternalProvider {
        /// Provider identity.
        provider_id: String,
    },
}

impl ParameterEvidence {
    fn write(&self, stream: &mut CanonicalWriter) {
        match self {
            Self::Declared { declarer } => {
                stream.word(1).text(declarer);
            }
            Self::ValidationSample { sample_id, units } => {
                stream.word(2).text(sample_id).word(*units);
            }
            Self::ReplicateSample { sample_id } => {
                stream.word(3).text(sample_id);
            }
            Self::ExternalProvider { provider_id } => {
                stream.word(4).text(provider_id);
            }
        }
    }
}

/// Why a measurement model is not well formed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MeasurementModelError {
    /// A required identity or label is blank.
    BlankField(&'static str),
    /// The indicator is recorded without error; no error model applies.
    ObservedIndicator,
    /// A proxy, learned representation or derived column was offered as the
    /// scientific construct.
    ProxyOfferedAsConstruct {
        /// What was offered.
        offered: MeasurementKind,
    },
    /// The indicator's kind is not one the route accepts.
    IndicatorKindNotAccepted {
        /// The route.
        route: MeasurementRoute,
        /// The indicator's kind.
        kind: MeasurementKind,
    },
    /// The parameters belong to a different route than declared.
    RouteParameterMismatch,
    /// A parameter is outside its range.
    InvalidParameter(&'static str),
    /// The declared error structure disagrees with the parameter layout
    /// (for example shared rates declared differential, or arm rates declared
    /// non-differential).
    StructureParameterMismatch,
}

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

fn probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn variance(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

/// A typed relationship between one indicator and one construct.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementModel {
    /// Human-readable label; never part of the semantic identity.
    pub model_id: String,
    /// The bounded correction route.
    pub route: MeasurementRoute,
    /// The recorded quantity.
    pub indicator: MeasurementVariable,
    /// The scientific quantity the indicator is offered for.
    pub construct: MeasurementVariable,
    /// Population, regime and arm the parameters were established for.
    pub binding: MeasurementBinding,
    /// Whether the error is declared differential.
    pub structure: ErrorStructure,
    /// The route's numerical parameters.
    pub parameters: MeasurementParameters,
    /// Coordinate system and version of the indicator.
    pub representation: RepresentationVersion,
    /// Where the parameters came from.
    pub evidence: ParameterEvidence,
}

/// The separately frozen identities of one model.
///
/// Measurement, population, regime, arm, representation and evidence
/// identities are distinct, domain-tagged digests so a change in one is
/// visible as that change and in no other; the model identity binds all six.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MeasurementIdentities {
    /// Route, indicator and construct, structure and parameters.
    pub measurement: SemanticDigest,
    /// Population identity.
    pub population: SemanticDigest,
    /// Regime identity.
    pub regime: SemanticDigest,
    /// Arm scope (all arms, or one named arm).
    pub arm: SemanticDigest,
    /// Representation coordinate system and version.
    pub representation: SemanticDigest,
    /// Provenance of the parameters.
    pub evidence: SemanticDigest,
    /// All six together, in the order of the fields above.
    pub model: SemanticDigest,
}

impl MeasurementModel {
    /// Check the model is well formed.
    ///
    /// # Errors
    /// Each refusal is a distinct [`MeasurementModelError`].
    pub fn validate(&self) -> Result<(), MeasurementModelError> {
        for (name, value) in [
            ("indicator.id", &self.indicator.id),
            ("indicator.units", &self.indicator.units),
            ("construct.id", &self.construct.id),
            ("construct.units", &self.construct.units),
            ("binding.population_id", &self.binding.population_id),
            ("binding.regime_id", &self.binding.regime_id),
            ("representation.coordinate_system_id", &self.representation.coordinate_system_id),
        ] {
            if blank(value) {
                return Err(MeasurementModelError::BlankField(name));
            }
        }
        if let ArmBinding::Arm(arm) = &self.binding.arm {
            if blank(arm) {
                return Err(MeasurementModelError::BlankField("binding.arm"));
            }
        }
        if self.indicator.kind == MeasurementKind::Observed {
            return Err(MeasurementModelError::ObservedIndicator);
        }
        if self.construct.kind != MeasurementKind::LatentScientific {
            return Err(MeasurementModelError::ProxyOfferedAsConstruct {
                offered: self.construct.kind,
            });
        }
        if !self.route.accepted_indicator_kinds().contains(&self.indicator.kind) {
            return Err(MeasurementModelError::IndicatorKindNotAccepted {
                route: self.route,
                kind: self.indicator.kind,
            });
        }
        if self.parameters.route() != self.route {
            return Err(MeasurementModelError::RouteParameterMismatch);
        }
        self.validate_parameters()
    }

    fn validate_parameters(&self) -> Result<(), MeasurementModelError> {
        let floor_ok = |v: f64| v.is_finite() && (0.0..1.0).contains(&v);
        match &self.parameters {
            MeasurementParameters::ClassicalError {
                error_variance,
                error_variance_standard_error,
                min_reliability,
            } => {
                if !variance(*error_variance) || !variance(*error_variance_standard_error) {
                    return Err(MeasurementModelError::InvalidParameter(
                        "error variance and its standard error must be finite and non-negative",
                    ));
                }
                if !floor_ok(*min_reliability) {
                    return Err(MeasurementModelError::InvalidParameter(
                        "minimum reliability must lie in [0, 1)",
                    ));
                }
                if self.structure != ErrorStructure::NonDifferential {
                    return Err(MeasurementModelError::StructureParameterMismatch);
                }
            }
            MeasurementParameters::Repeated { min_replicated_units, min_reliability } => {
                if *min_replicated_units < 2 || !floor_ok(*min_reliability) {
                    return Err(MeasurementModelError::InvalidParameter(
                        "repeated measurements need at least two replicated units and a reliability floor in [0, 1)",
                    ));
                }
                if self.structure != ErrorStructure::NonDifferential {
                    return Err(MeasurementModelError::StructureParameterMismatch);
                }
            }
            MeasurementParameters::Binary { rates, min_youden } => {
                if !(min_youden.is_finite() && *min_youden > 0.0 && *min_youden < 1.0) {
                    return Err(MeasurementModelError::InvalidParameter(
                        "minimum Youden index must lie in (0, 1)",
                    ));
                }
                for arm in rates {
                    if !(probability(arm.sensitivity) && probability(arm.specificity)) {
                        return Err(MeasurementModelError::InvalidParameter(
                            "sensitivity and specificity must lie in [0, 1]",
                        ));
                    }
                    if !(variance(arm.sensitivity_variance) && variance(arm.specificity_variance))
                    {
                        return Err(MeasurementModelError::InvalidParameter(
                            "rate variances must be finite and non-negative",
                        ));
                    }
                    if arm.arm_id.as_deref().is_some_and(blank) {
                        return Err(MeasurementModelError::BlankField("rates.arm_id"));
                    }
                }
                let shared = rates.len() == 1 && rates[0].arm_id.is_none();
                let per_arm = rates.len() >= 2
                    && rates.iter().all(|arm| arm.arm_id.is_some())
                    && {
                        let mut ids: Vec<&str> =
                            rates.iter().filter_map(|arm| arm.arm_id.as_deref()).collect();
                        ids.sort_unstable();
                        ids.windows(2).all(|pair| pair[0] != pair[1])
                    };
                match self.structure {
                    ErrorStructure::NonDifferential if shared => {}
                    ErrorStructure::Differential(DifferentialAxis::Arm) if per_arm => {}
                    _ => return Err(MeasurementModelError::StructureParameterMismatch),
                }
            }
            MeasurementParameters::Categorical { levels, confusion, column_counts, max_condition } => {
                let k = levels.len();
                if !(2..=16).contains(&k)
                    || confusion.len() != k * k
                    || levels.iter().any(|level| blank(level))
                {
                    return Err(MeasurementModelError::InvalidParameter(
                        "a confusion model needs 2 to 16 named levels and a square matrix",
                    ));
                }
                let mut seen: Vec<&str> = levels.iter().map(String::as_str).collect();
                seen.sort_unstable();
                if seen.windows(2).any(|pair| pair[0] == pair[1]) {
                    return Err(MeasurementModelError::InvalidParameter("level names repeat"));
                }
                if !confusion.iter().all(|v| probability(*v)) {
                    return Err(MeasurementModelError::InvalidParameter(
                        "confusion entries must be probabilities",
                    ));
                }
                for t in 0..k {
                    let sum: f64 = (0..k).map(|j| confusion[j * k + t]).sum();
                    if (sum - 1.0).abs() > 1e-9 {
                        return Err(MeasurementModelError::InvalidParameter(
                            "each confusion column must sum to one",
                        ));
                    }
                }
                if let Some(counts) = column_counts {
                    if counts.len() != k || counts.contains(&0) {
                        return Err(MeasurementModelError::InvalidParameter(
                            "validation counts need one positive count per true level",
                        ));
                    }
                }
                if !(max_condition.is_finite() && *max_condition >= 1.0) {
                    return Err(MeasurementModelError::InvalidParameter(
                        "maximum condition number must be at least one",
                    ));
                }
                if self.structure != ErrorStructure::NonDifferential {
                    return Err(MeasurementModelError::StructureParameterMismatch);
                }
            }
            MeasurementParameters::ExternalPosterior { reference, min_draws, .. } => {
                for (name, value) in [
                    ("reference.provider_id", &reference.provider_id),
                    ("reference.object_id", &reference.object_id),
                    ("reference.version_id", &reference.version_id),
                    ("reference.snapshot_id", &reference.snapshot_id),
                    ("reference.request_id", &reference.request_id),
                ] {
                    if blank(value) {
                        return Err(MeasurementModelError::BlankField(name));
                    }
                }
                if *min_draws < 2 {
                    return Err(MeasurementModelError::InvalidParameter(
                        "a posterior route needs at least two joint draws",
                    ));
                }
                if self.structure != ErrorStructure::NonDifferential {
                    return Err(MeasurementModelError::StructureParameterMismatch);
                }
            }
        }
        Ok(())
    }

    fn write_variable(stream: &mut CanonicalWriter, variable: &MeasurementVariable) {
        stream.text(&variable.id).word(variable.kind.tag()).text(&variable.units);
    }

    /// The separately frozen identities. Floats enter as bit patterns; the
    /// display label `model_id` is excluded.
    #[must_use]
    pub fn identities(&self) -> MeasurementIdentities {
        let mut measurement = CanonicalWriter::new(TAG_MEASUREMENT);
        measurement.word(u64::from(MEASUREMENT_MODEL_FORMAT)).word(self.route.tag());
        Self::write_variable(&mut measurement, &self.indicator);
        Self::write_variable(&mut measurement, &self.construct);
        self.structure.write(&mut measurement);
        self.parameters.write(&mut measurement);

        let mut population = CanonicalWriter::new(TAG_POPULATION);
        population.text(&self.binding.population_id);
        let mut regime = CanonicalWriter::new(TAG_REGIME);
        regime.text(&self.binding.regime_id);
        let mut arm = CanonicalWriter::new(TAG_ARM);
        match &self.binding.arm {
            ArmBinding::AllArms => {
                arm.word(1);
            }
            ArmBinding::Arm(name) => {
                arm.word(2).text(name);
            }
        }

        let mut representation = CanonicalWriter::new(TAG_REPRESENTATION);
        representation
            .text(&self.representation.coordinate_system_id)
            .word(u64::from(self.representation.version));

        let mut evidence = CanonicalWriter::new(TAG_EVIDENCE);
        self.evidence.write(&mut evidence);

        let measurement = SemanticDigest::from_bytes(measurement.finish());
        let population = SemanticDigest::from_bytes(population.finish());
        let regime = SemanticDigest::from_bytes(regime.finish());
        let arm = SemanticDigest::from_bytes(arm.finish());
        let representation = SemanticDigest::from_bytes(representation.finish());
        let evidence = SemanticDigest::from_bytes(evidence.finish());
        let mut model = CanonicalWriter::new(TAG_MODEL);
        for digest in [&measurement, &population, &regime, &arm, &representation, &evidence] {
            model.bytes(digest.as_bytes());
        }
        MeasurementIdentities {
            measurement,
            population,
            regime,
            arm,
            representation,
            evidence,
            model: SemanticDigest::from_bytes(model.finish()),
        }
    }
}

/// Digest of the data a corrected functional was computed from, for binding a
/// result to its inputs.
///
/// `parts` is a sequence of named numeric blocks; each is framed by its name
/// and a count-prefixed sequence of bit patterns.
#[must_use]
pub fn measurement_data_digest(parts: &[(&str, &[f64])]) -> SemanticDigest {
    let mut stream = CanonicalWriter::new(TAG_DATA);
    stream.word(parts.len() as u64);
    for (name, values) in parts {
        stream.text(name).reals(values);
    }
    SemanticDigest::from_bytes(stream.finish())
}

/// What a corrected functional reports about its uncertainty.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MeasurementInference {
    /// A point only.
    PointOnly,
    /// A point with a delta-method standard error that includes the
    /// measurement-parameter uncertainty. The standard error is an
    /// *uncalibrated* estimate of an asymptotic standard deviation: no
    /// coverage has been measured and no interval is licensed.
    UncalibratedStandardError,
}

impl MeasurementInference {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PointOnly => "point_only",
            Self::UncalibratedStandardError => "uncalibrated_standard_error",
        }
    }
}

/// Whether the uncertainty of the measurement parameters is in the reported
/// variance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ParameterUncertainty {
    /// The parameters were estimated and their sampling variance (or the
    /// between-imputation variance of the posterior draws) enters the
    /// variance.
    Propagated,
    /// The parameters were declared exact; the variance is conditional on
    /// them.
    DeclaredExact,
    /// The parameters are uncertain but that uncertainty is not representable
    /// from what was supplied (for example a provider's hyperparameters seen
    /// through per-unit posterior means). The variance is conditional on them.
    NotPropagated,
}

impl ParameterUncertainty {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Propagated => "propagated",
            Self::DeclaredExact => "declared_exact",
            Self::NotPropagated => "not_propagated",
        }
    }

    /// Parse a stable name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::Propagated, Self::DeclaredExact, Self::NotPropagated]
            .into_iter()
            .find(|value| value.as_str() == name)
    }
}

/// One named part of a variance.
#[derive(Clone, Debug, PartialEq)]
pub struct VarianceComponent {
    /// Stable component name.
    pub name: String,
    /// Variance contribution.
    pub variance: f64,
}

/// A corrected functional bound to the model that produced it.
///
/// No interval is carried: coverage of the whole measurement-plus-estimation
/// method is unmeasured, so the interval route stays closed.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedFunctionalRecord {
    /// Model identity the result is bound to.
    pub model_identity: SemanticDigest,
    /// Route that produced the result.
    pub route: MeasurementRoute,
    /// Stable functional name (for example `linear_coefficient`).
    pub functional_id: String,
    /// Corrected point value.
    pub point: f64,
    /// The uncorrected value of the same functional, when one exists.
    pub naive_point: Option<f64>,
    /// Delta-method standard error, uncalibrated.
    pub standard_error: Option<f64>,
    /// Named variance parts that sum to the reported variance.
    pub variance_components: Vec<VarianceComponent>,
    /// Rows or units the result is computed from.
    pub rows: u64,
    /// Digest of the data the result is computed from.
    pub data_identity: SemanticDigest,
    /// What the uncertainty means.
    pub inference: MeasurementInference,
    /// Whether the measurement parameters' uncertainty is in the variance.
    pub parameter_uncertainty: ParameterUncertainty,
}

impl CorrectedFunctionalRecord {
    /// Whether the record carries an interval. Always false: the interval
    /// route is closed until the whole method has a measured coverage record.
    #[must_use]
    pub const fn has_interval(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> MeasurementModel {
        MeasurementModel {
            model_id: "bp-cuff".into(),
            route: MeasurementRoute::BinaryMisclassification,
            indicator: MeasurementVariable {
                id: "schema:flag".into(),
                kind: MeasurementKind::NoisyMisclassified,
                units: "binary".into(),
            },
            construct: MeasurementVariable {
                id: "construct:disease".into(),
                kind: MeasurementKind::LatentScientific,
                units: "binary".into(),
            },
            binding: MeasurementBinding {
                population_id: "clinic-a".into(),
                regime_id: "observational".into(),
                arm: ArmBinding::AllArms,
            },
            structure: ErrorStructure::NonDifferential,
            parameters: MeasurementParameters::Binary {
                rates: vec![BinaryArmRates {
                    arm_id: None,
                    sensitivity: 0.9,
                    specificity: 0.8,
                    sensitivity_variance: 0.0001,
                    specificity_variance: 0.0004,
                }],
                min_youden: 0.1,
            },
            representation: RepresentationVersion {
                coordinate_system_id: "scalar".into(),
                version: 1,
            },
            evidence: ParameterEvidence::ValidationSample {
                sample_id: "val-2024".into(),
                units: 500,
            },
        }
    }

    #[test]
    fn a_well_formed_model_validates() {
        assert_eq!(model().validate(), Ok(()));
    }

    #[test]
    fn kinds_stay_distinct_in_refusals() {
        for offered in [
            MeasurementKind::NoisyMisclassified,
            MeasurementKind::ProbabilisticFeatureAssignment,
            MeasurementKind::LearnedVectorRepresentation,
            MeasurementKind::Derived,
            MeasurementKind::Observed,
        ] {
            let mut m = model();
            m.construct.kind = offered;
            assert_eq!(
                m.validate(),
                Err(MeasurementModelError::ProxyOfferedAsConstruct { offered })
            );
        }
        let mut observed = model();
        observed.indicator.kind = MeasurementKind::Observed;
        assert_eq!(observed.validate(), Err(MeasurementModelError::ObservedIndicator));
        let mut vector = model();
        vector.indicator.kind = MeasurementKind::LearnedVectorRepresentation;
        assert_eq!(
            vector.validate(),
            Err(MeasurementModelError::IndicatorKindNotAccepted {
                route: MeasurementRoute::BinaryMisclassification,
                kind: MeasurementKind::LearnedVectorRepresentation,
            })
        );
        let names: Vec<&str> = MeasurementKind::ALL.iter().map(|k| k.as_str()).collect();
        let mut dedup = names.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(dedup.len(), 6);
        for kind in MeasurementKind::ALL {
            assert_eq!(MeasurementKind::from_name(kind.as_str()), Some(kind));
        }
    }

    #[test]
    fn shared_rates_cannot_be_declared_differential_and_vice_versa() {
        let mut shared_as_diff = model();
        shared_as_diff.structure = ErrorStructure::Differential(DifferentialAxis::Arm);
        assert_eq!(
            shared_as_diff.validate(),
            Err(MeasurementModelError::StructureParameterMismatch)
        );
        let mut per_arm = model();
        per_arm.structure = ErrorStructure::Differential(DifferentialAxis::Arm);
        per_arm.parameters = MeasurementParameters::Binary {
            rates: vec![
                BinaryArmRates {
                    arm_id: Some("treated".into()),
                    sensitivity: 0.9,
                    specificity: 0.8,
                    sensitivity_variance: 0.0,
                    specificity_variance: 0.0,
                },
                BinaryArmRates {
                    arm_id: Some("control".into()),
                    sensitivity: 0.7,
                    specificity: 0.95,
                    sensitivity_variance: 0.0,
                    specificity_variance: 0.0,
                },
            ],
            min_youden: 0.1,
        };
        assert_eq!(per_arm.validate(), Ok(()));
        let mut not_declared = per_arm.clone();
        not_declared.structure = ErrorStructure::NonDifferential;
        assert_eq!(
            not_declared.validate(),
            Err(MeasurementModelError::StructureParameterMismatch)
        );
        // Differential by population has no parameter layout in this route.
        let mut by_population = per_arm;
        by_population.structure = ErrorStructure::Differential(DifferentialAxis::Population);
        assert_eq!(
            by_population.validate(),
            Err(MeasurementModelError::StructureParameterMismatch)
        );
    }

    #[test]
    fn route_and_parameters_must_agree() {
        let mut m = model();
        m.route = MeasurementRoute::ClassicalContinuousError;
        assert_eq!(m.validate(), Err(MeasurementModelError::RouteParameterMismatch));
    }

    #[test]
    fn identities_are_separately_sensitive_and_label_free() {
        let base = model();
        let id = base.identities();
        let mut relabeled = base.clone();
        relabeled.model_id = "another label".into();
        assert_eq!(relabeled.identities(), id);

        let mut population = base.clone();
        population.binding.population_id = "clinic-b".into();
        let p = population.identities();
        assert_ne!(p.population, id.population);
        assert_ne!(p.model, id.model);
        assert_eq!(p.measurement, id.measurement);
        assert_eq!(p.regime, id.regime);
        assert_eq!(p.arm, id.arm);
        assert_eq!(p.representation, id.representation);
        assert_eq!(p.evidence, id.evidence);

        // Changing only the regime moves only the regime digest (and the model).
        let mut regime = base.clone();
        regime.binding.regime_id = "do(a=1)".into();
        let r = regime.identities();
        assert_ne!(r.regime, id.regime);
        assert_ne!(r.model, id.model);
        assert_eq!(r.population, id.population);
        assert_eq!(r.arm, id.arm);
        assert_eq!(r.measurement, id.measurement);
        assert_eq!(r.representation, id.representation);
        assert_eq!(r.evidence, id.evidence);

        // Changing only the arm scope moves only the arm digest (and the model).
        let mut arm = base.clone();
        arm.binding.arm = ArmBinding::Arm("treated".into());
        let a = arm.identities();
        assert_ne!(a.arm, id.arm);
        assert_ne!(a.model, id.model);
        assert_eq!(a.population, id.population);
        assert_eq!(a.regime, id.regime);
        assert_eq!(a.measurement, id.measurement);
        assert_eq!(a.representation, id.representation);
        assert_eq!(a.evidence, id.evidence);

        // The three digests are domain separated: the same text in two roles differs.
        let mut same_text = base.clone();
        same_text.binding.population_id = "x".into();
        same_text.binding.regime_id = "x".into();
        let t = same_text.identities();
        assert_ne!(t.population, t.regime);

        let mut version = base.clone();
        version.representation.version = 2;
        let v = version.identities();
        assert_ne!(v.representation, id.representation);
        assert_eq!(v.measurement, id.measurement);
        assert_ne!(v.model, id.model);

        // Swapping sensitivity and specificity is a different measurement.
        let mut swapped = base.clone();
        swapped.parameters = MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: 0.8,
                specificity: 0.9,
                sensitivity_variance: 0.0004,
                specificity_variance: 0.0001,
            }],
            min_youden: 0.1,
        };
        let s = swapped.identities();
        assert_ne!(s.measurement, id.measurement);
        assert_eq!(s.population, id.population);
        assert_eq!(s.regime, id.regime);
        assert_eq!(s.arm, id.arm);

        let mut evidence = base;
        evidence.evidence = ParameterEvidence::Declared { declarer: "lab".into() };
        assert_ne!(evidence.identities().evidence, id.evidence);
    }

    #[test]
    fn mappings_carry_intervention_meaning_only_with_evidence() {
        let from = RepresentationVersion { coordinate_system_id: "emb".into(), version: 1 };
        let to = RepresentationVersion { coordinate_system_id: "emb".into(), version: 2 };
        let relabel = RepresentationMapping {
            from: from.clone(),
            to: to.clone(),
            mapping_id: "procrustes".into(),
            semantics: MappingSemantics::CoordinateRelabeling,
        };
        assert!(!relabel.carries_intervention_meaning());
        let preserving = RepresentationMapping {
            from,
            to,
            mapping_id: "procrustes".into(),
            semantics: MappingSemantics::InterventionPreserving { evidence_id: "exp-9".into() },
        };
        assert!(preserving.carries_intervention_meaning());
    }

    #[test]
    fn data_digest_binds_names_and_bit_patterns() {
        let a = measurement_data_digest(&[("y", &[1.0, 2.0]), ("w", &[3.0])]);
        let b = measurement_data_digest(&[("y", &[1.0, 2.0]), ("w", &[3.0])]);
        let c = measurement_data_digest(&[("y", &[1.0, 2.0]), ("w", &[3.0 + f64::EPSILON * 4.0])]);
        let d = measurement_data_digest(&[("w", &[1.0, 2.0]), ("y", &[3.0])]);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }
}
