//! Portable measurement model and corrected-functional result.
//!
//! The artifact carries the typed [`MeasurementModel`], its separately frozen
//! identities and, optionally, one corrected functional bound to the model's
//! identity together with the *sufficient inputs* of its computation
//! ([`MeasurementReplay`]): moments for a linear fit, counts for a binary
//! correction, observed-class summaries for a categorical one, per-imputation
//! fits for a posterior route. A consumer that holds only the bytes and an
//! independently retained model identity recomputes the point and the
//! standard error with the pure functions of `antecedent-stats` and refuses
//! any artifact whose stored numbers differ.
//!
//! Floats travel as IEEE-754 bit patterns. The artifact has no interval field:
//! no interval route is open for a measurement-corrected functional.
//!
//! A consumer holding the original data can additionally check
//! [`MeasurementModelArtifact::covers_data_identity`].

use antecedent_core::identity::SemanticDigest;
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, CorrectedFunctionalRecord, DifferentialAxis, ErrorStructure,
    MeasurementInference, MeasurementKind, MeasurementModel, MeasurementParameters,
    MeasurementRoute, MeasurementVariable, ParameterEvidence, ParameterUncertainty,
    PosteriorReference, RepresentationVersion, VarianceComponent,
};
use antecedent_stats::measurement_error::{
    BinaryErrorRates, ConfusionMatrix, LinearMoments, ObservedClassSummary, TwoByTwoCounts,
    corrected_categories, corrected_prevalence, corrected_risk_difference_differential,
    corrected_risk_difference_nondifferential, rubin_combine,
};
use serde::{Deserialize, Serialize};

use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;

/// Maximum artifact bytes accepted before decode allocation.
pub const MAX_MEASUREMENT_MODEL_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
/// Metadata format version.
pub const MEASUREMENT_MODEL_ARTIFACT_VERSION: u16 = 1;
/// Most per-imputation fits one artifact carries.
pub const MAX_MEASUREMENT_IMPUTATIONS: usize = 10_000;
const MAX_DIM: usize = 66;
const ARTIFACT_KIND: &str = crate::artifact_kinds::MEASUREMENT_MODEL;
const META_SECTION: &str = "measurement_model_meta";
/// Relative tolerance of the replay comparison.
const REPLAY_TOLERANCE: f64 = 1e-10;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VariableWire {
    id: String,
    kind: String,
    units: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingWire {
    population_id: String,
    regime_id: String,
    /// `None` binds all arms.
    arm: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RepresentationWire {
    coordinate_system_id: String,
    version: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StructureWire {
    /// `non_differential` or `differential`.
    kind: String,
    /// Axis name when differential.
    axis: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BinaryRatesWire {
    arm_id: Option<String>,
    sensitivity_bits: u64,
    specificity_bits: u64,
    sensitivity_variance_bits: u64,
    specificity_variance_bits: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PosteriorReferenceWire {
    provider_id: String,
    object_id: String,
    version_id: String,
    snapshot_id: String,
    request_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ParametersWire {
    ClassicalError {
        error_variance_bits: u64,
        error_variance_standard_error_bits: u64,
        min_reliability_bits: u64,
    },
    Binary {
        rates: Vec<BinaryRatesWire>,
        min_youden_bits: u64,
    },
    Repeated {
        min_replicated_units: u32,
        min_reliability_bits: u64,
    },
    Categorical {
        levels: Vec<String>,
        confusion_bits: Vec<u64>,
        column_counts: Option<Vec<u64>>,
        max_condition_bits: u64,
    },
    ExternalPosterior {
        reference: PosteriorReferenceWire,
        conditioned_on_outcome: bool,
        min_draws: u32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EvidenceWire {
    Declared { declarer: String },
    ValidationSample { sample_id: String, units: u64 },
    ReplicateSample { sample_id: String },
    ExternalProvider { provider_id: String },
}

/// The model on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementModelWire {
    /// Display label; not part of any identity.
    label: String,
    route: String,
    indicator: VariableWire,
    construct: VariableWire,
    binding: BindingWire,
    structure: StructureWire,
    parameters: ParametersWire,
    representation: RepresentationWire,
    evidence: EvidenceWire,
}

/// The separately frozen identities, as lowercase hex.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementIdentityWire {
    /// Route, indicator and construct, structure and parameters.
    pub measurement: String,
    /// Population, regime and arm.
    pub binding: String,
    /// Representation coordinate system and version.
    pub representation: String,
    /// Parameter provenance.
    pub evidence: String,
    /// All four together.
    pub model: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComponentWire {
    name: String,
    variance_bits: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordWire {
    model_identity: String,
    route: String,
    functional_id: String,
    point_bits: u64,
    naive_point_bits: Option<u64>,
    standard_error_bits: Option<u64>,
    components: Vec<ComponentWire>,
    rows: u64,
    data_identity: String,
    inference: String,
    parameter_uncertainty: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MomentsWire {
    n: u64,
    dim: u32,
    gram_bits: Vec<u64>,
    cross_bits: Vec<u64>,
    meat_bits: Vec<u64>,
    naive_meat_bits: Vec<u64>,
    influence_cross_bits: Vec<u64>,
    influence_variance_bits: u64,
    error_variance_bits: u64,
    mean_inverse_replicates_bits: u64,
    reliability_floor_bits: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClassWire {
    count: u64,
    mean_bits: u64,
    variance_bits: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReplayWire {
    Linear {
        moments: MomentsWire,
        coefficient_index: u32,
    },
    BinaryPrevalence {
        events: u64,
        total: u64,
        arm_id: Option<String>,
    },
    BinaryRiskDifference {
        exposed_arm_id: String,
        unexposed_arm_id: String,
        exposed_events: u64,
        exposed_total: u64,
        unexposed_events: u64,
        unexposed_total: u64,
    },
    Categorical {
        classes: Vec<ClassWire>,
        /// `proportion`, `outcome_mean` or `outcome_contrast`.
        functional: String,
        first: String,
        second: Option<String>,
    },
    MultipleImputation {
        coefficients_bits: Vec<Vec<u64>>,
        covariances_bits: Vec<Vec<u64>>,
        coefficient_index: u32,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultWire {
    record: RecordWire,
    replay: ReplayWire,
    /// Trust label of an external posterior; `None` for native routes.
    provider_trust: Option<String>,
}

/// Stored metadata of a measurement-model artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementModelMetadata {
    /// Metadata format version.
    pub version: u16,
    /// The model.
    pub model: MeasurementModelWire,
    /// The model's separately frozen identities.
    pub identities: MeasurementIdentityWire,
    result: Option<ResultWire>,
    /// Always false: this artifact makes no interval claim.
    pub has_interval: bool,
}

// ---------------------------------------------------------------------------
// Public replay input
// ---------------------------------------------------------------------------

/// A categorical functional named by level.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CategoricalReplayFunctional {
    /// Proportion of one true category.
    Proportion(String),
    /// Mean outcome in one true category.
    OutcomeMean(String),
    /// Difference of outcome means `first - second`.
    OutcomeContrast {
        /// Minuend.
        first: String,
        /// Subtrahend.
        second: String,
    },
}

/// The sufficient inputs of a corrected functional.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum MeasurementReplay {
    /// A linear coefficient from sufficient moments.
    Linear {
        /// Sufficient moments.
        moments: LinearMoments,
        /// Coefficient index (1 is the latent regressor).
        coefficient_index: usize,
    },
    /// Corrected prevalence from observed counts.
    BinaryPrevalence {
        /// Observed positives.
        events: u64,
        /// Units observed.
        total: u64,
        /// Arm whose rates apply, for a differential model.
        arm_id: Option<String>,
    },
    /// Corrected risk difference from the observed two-by-two counts.
    BinaryRiskDifference {
        /// Arm identity of the exposed group.
        exposed_arm_id: String,
        /// Arm identity of the unexposed group.
        unexposed_arm_id: String,
        /// Observed counts.
        table: TwoByTwoCounts,
    },
    /// A functional of corrected categories from observed-class summaries.
    Categorical {
        /// Observed-label summaries in level order.
        classes: Vec<ObservedClassSummary>,
        /// Which functional the record reports.
        functional: CategoricalReplayFunctional,
    },
    /// A coefficient combined over outcome-conditioned imputations.
    MultipleImputation {
        /// Per-imputation coefficients.
        coefficients: Vec<Vec<f64>>,
        /// Per-imputation covariances, row-major.
        covariances: Vec<Vec<f64>>,
        /// Coefficient index (1 is the latent regressor).
        coefficient_index: usize,
    },
}

/// A corrected functional with the inputs needed to recompute it.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementResult {
    /// The result, bound to the model identity.
    pub record: CorrectedFunctionalRecord,
    /// Sufficient inputs.
    pub replay: MeasurementReplay,
    /// Trust label of an external posterior (`externally_attested` or
    /// `exact_request_verified`); required for the posterior route.
    pub provider_trust: Option<String>,
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

fn bits_vec(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn reals_vec(bits: &[u64]) -> Vec<f64> {
    bits.iter().map(|b| f64::from_bits(*b)).collect()
}

fn convert(message: impl Into<String>) -> IoError {
    IoError::Convert(message.into())
}

fn refused(message: impl Into<String>) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("measurement_binding_mismatch"),
        message: message.into(),
    }
}

fn axis_name(axis: DifferentialAxis) -> &'static str {
    axis.as_str()
}

fn axis_from_name(name: &str) -> Result<DifferentialAxis, IoError> {
    match name {
        "arm" => Ok(DifferentialAxis::Arm),
        "population" => Ok(DifferentialAxis::Population),
        "time" => Ok(DifferentialAxis::Time),
        other => Err(convert(format!("unknown differential axis `{other}`"))),
    }
}

fn variable_to_wire(variable: &MeasurementVariable) -> VariableWire {
    VariableWire {
        id: variable.id.clone(),
        kind: variable.kind.as_str().to_owned(),
        units: variable.units.clone(),
    }
}

fn variable_from_wire(wire: &VariableWire) -> Result<MeasurementVariable, IoError> {
    Ok(MeasurementVariable {
        id: wire.id.clone(),
        kind: MeasurementKind::from_name(&wire.kind)
            .ok_or_else(|| convert(format!("unknown measurement kind `{}`", wire.kind)))?,
        units: wire.units.clone(),
    })
}

/// Encode a model to its wire form.
#[must_use]
pub fn model_to_wire(model: &MeasurementModel) -> MeasurementModelWire {
    let structure = match model.structure {
        ErrorStructure::NonDifferential => {
            StructureWire { kind: "non_differential".into(), axis: None }
        }
        ErrorStructure::Differential(axis) => {
            StructureWire { kind: "differential".into(), axis: Some(axis_name(axis).into()) }
        }
        // `ErrorStructure` is non-exhaustive: an unknown structure is carried as
        // an unrecognized kind that loading refuses.
        _ => StructureWire { kind: "unrecognized".into(), axis: None },
    };
    let parameters = match &model.parameters {
        MeasurementParameters::ClassicalError {
            error_variance,
            error_variance_standard_error,
            min_reliability,
        } => ParametersWire::ClassicalError {
            error_variance_bits: error_variance.to_bits(),
            error_variance_standard_error_bits: error_variance_standard_error.to_bits(),
            min_reliability_bits: min_reliability.to_bits(),
        },
        MeasurementParameters::Binary { rates, min_youden } => ParametersWire::Binary {
            rates: rates
                .iter()
                .map(|arm| BinaryRatesWire {
                    arm_id: arm.arm_id.clone(),
                    sensitivity_bits: arm.sensitivity.to_bits(),
                    specificity_bits: arm.specificity.to_bits(),
                    sensitivity_variance_bits: arm.sensitivity_variance.to_bits(),
                    specificity_variance_bits: arm.specificity_variance.to_bits(),
                })
                .collect(),
            min_youden_bits: min_youden.to_bits(),
        },
        MeasurementParameters::Repeated { min_replicated_units, min_reliability } => {
            ParametersWire::Repeated {
                min_replicated_units: *min_replicated_units,
                min_reliability_bits: min_reliability.to_bits(),
            }
        }
        MeasurementParameters::Categorical { levels, confusion, column_counts, max_condition } => {
            ParametersWire::Categorical {
                levels: levels.clone(),
                confusion_bits: bits_vec(confusion),
                column_counts: column_counts.clone(),
                max_condition_bits: max_condition.to_bits(),
            }
        }
        MeasurementParameters::ExternalPosterior { reference, conditioned_on_outcome, min_draws } => {
            ParametersWire::ExternalPosterior {
                reference: PosteriorReferenceWire {
                    provider_id: reference.provider_id.clone(),
                    object_id: reference.object_id.clone(),
                    version_id: reference.version_id.clone(),
                    snapshot_id: reference.snapshot_id.clone(),
                    request_id: reference.request_id.clone(),
                },
                conditioned_on_outcome: *conditioned_on_outcome,
                min_draws: *min_draws,
            }
        }
    };
    let evidence = match &model.evidence {
        ParameterEvidence::Declared { declarer } => {
            EvidenceWire::Declared { declarer: declarer.clone() }
        }
        ParameterEvidence::ValidationSample { sample_id, units } => {
            EvidenceWire::ValidationSample { sample_id: sample_id.clone(), units: *units }
        }
        ParameterEvidence::ReplicateSample { sample_id } => {
            EvidenceWire::ReplicateSample { sample_id: sample_id.clone() }
        }
        ParameterEvidence::ExternalProvider { provider_id } => {
            EvidenceWire::ExternalProvider { provider_id: provider_id.clone() }
        }
        // Non-exhaustive: an unknown provenance is recorded as a declaration
        // by an unnamed party, which a consumer sees as a different identity.
        _ => EvidenceWire::Declared { declarer: "unrecognized".into() },
    };
    MeasurementModelWire {
        label: model.model_id.clone(),
        route: model.route.as_str().to_owned(),
        indicator: variable_to_wire(&model.indicator),
        construct: variable_to_wire(&model.construct),
        binding: BindingWire {
            population_id: model.binding.population_id.clone(),
            regime_id: model.binding.regime_id.clone(),
            arm: match &model.binding.arm {
                ArmBinding::AllArms => None,
                ArmBinding::Arm(arm) => Some(arm.clone()),
                // Non-exhaustive: an unknown scope is bound to no arm name, so
                // it can never match data.
                _ => Some(String::new()),
            },
        },
        structure,
        parameters,
        representation: RepresentationWire {
            coordinate_system_id: model.representation.coordinate_system_id.clone(),
            version: model.representation.version,
        },
        evidence,
    }
}

/// Decode a model from its wire form and validate it.
///
/// # Errors
/// An unknown name or a model that fails validation refuses.
pub fn model_from_wire(wire: &MeasurementModelWire) -> Result<MeasurementModel, IoError> {
    let structure = match (wire.structure.kind.as_str(), wire.structure.axis.as_deref()) {
        ("non_differential", None) => ErrorStructure::NonDifferential,
        ("differential", Some(axis)) => ErrorStructure::Differential(axis_from_name(axis)?),
        _ => return Err(convert("unrecognized error structure")),
    };
    let parameters = match &wire.parameters {
        ParametersWire::ClassicalError {
            error_variance_bits,
            error_variance_standard_error_bits,
            min_reliability_bits,
        } => MeasurementParameters::ClassicalError {
            error_variance: f64::from_bits(*error_variance_bits),
            error_variance_standard_error: f64::from_bits(*error_variance_standard_error_bits),
            min_reliability: f64::from_bits(*min_reliability_bits),
        },
        ParametersWire::Binary { rates, min_youden_bits } => MeasurementParameters::Binary {
            rates: rates
                .iter()
                .map(|r| BinaryArmRates {
                    arm_id: r.arm_id.clone(),
                    sensitivity: f64::from_bits(r.sensitivity_bits),
                    specificity: f64::from_bits(r.specificity_bits),
                    sensitivity_variance: f64::from_bits(r.sensitivity_variance_bits),
                    specificity_variance: f64::from_bits(r.specificity_variance_bits),
                })
                .collect(),
            min_youden: f64::from_bits(*min_youden_bits),
        },
        ParametersWire::Repeated { min_replicated_units, min_reliability_bits } => {
            MeasurementParameters::Repeated {
                min_replicated_units: *min_replicated_units,
                min_reliability: f64::from_bits(*min_reliability_bits),
            }
        }
        ParametersWire::Categorical { levels, confusion_bits, column_counts, max_condition_bits } => {
            if levels.len() > 16 || confusion_bits.len() > 16 * 16 {
                return Err(IoError::TooLarge);
            }
            MeasurementParameters::Categorical {
                levels: levels.clone(),
                confusion: reals_vec(confusion_bits),
                column_counts: column_counts.clone(),
                max_condition: f64::from_bits(*max_condition_bits),
            }
        }
        ParametersWire::ExternalPosterior { reference, conditioned_on_outcome, min_draws } => {
            MeasurementParameters::ExternalPosterior {
                reference: PosteriorReference {
                    provider_id: reference.provider_id.clone(),
                    object_id: reference.object_id.clone(),
                    version_id: reference.version_id.clone(),
                    snapshot_id: reference.snapshot_id.clone(),
                    request_id: reference.request_id.clone(),
                },
                conditioned_on_outcome: *conditioned_on_outcome,
                min_draws: *min_draws,
            }
        }
    };
    let evidence = match &wire.evidence {
        EvidenceWire::Declared { declarer } => ParameterEvidence::Declared { declarer: declarer.clone() },
        EvidenceWire::ValidationSample { sample_id, units } => {
            ParameterEvidence::ValidationSample { sample_id: sample_id.clone(), units: *units }
        }
        EvidenceWire::ReplicateSample { sample_id } => {
            ParameterEvidence::ReplicateSample { sample_id: sample_id.clone() }
        }
        EvidenceWire::ExternalProvider { provider_id } => {
            ParameterEvidence::ExternalProvider { provider_id: provider_id.clone() }
        }
    };
    let model = MeasurementModel {
        model_id: wire.label.clone(),
        route: MeasurementRoute::from_name(&wire.route)
            .ok_or_else(|| convert(format!("unknown measurement route `{}`", wire.route)))?,
        indicator: variable_from_wire(&wire.indicator)?,
        construct: variable_from_wire(&wire.construct)?,
        binding: antecedent_core::measurement_model::MeasurementBinding {
            population_id: wire.binding.population_id.clone(),
            regime_id: wire.binding.regime_id.clone(),
            arm: match &wire.binding.arm {
                None => ArmBinding::AllArms,
                Some(arm) => ArmBinding::Arm(arm.clone()),
            },
        },
        structure,
        parameters,
        representation: RepresentationVersion {
            coordinate_system_id: wire.representation.coordinate_system_id.clone(),
            version: wire.representation.version,
        },
        evidence,
    };
    model.validate().map_err(|error| convert(format!("invalid measurement model: {error:?}")))?;
    Ok(model)
}

fn identities_to_wire(model: &MeasurementModel) -> MeasurementIdentityWire {
    let ids = model.identities();
    MeasurementIdentityWire {
        measurement: ids.measurement.to_hex(),
        binding: ids.binding.to_hex(),
        representation: ids.representation.to_hex(),
        evidence: ids.evidence.to_hex(),
        model: ids.model.to_hex(),
    }
}

fn digest_from_hex(hex: &str) -> Result<SemanticDigest, IoError> {
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(convert("a digest is 64 hexadecimal characters"));
    }
    let mut bytes = [0_u8; 32];
    for (i, slot) in bytes.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .map_err(|_| convert("a digest is 64 hexadecimal characters"))?;
    }
    Ok(SemanticDigest::from_bytes(bytes))
}

fn record_to_wire(record: &CorrectedFunctionalRecord) -> RecordWire {
    RecordWire {
        model_identity: record.model_identity.to_hex(),
        route: record.route.as_str().to_owned(),
        functional_id: record.functional_id.clone(),
        point_bits: record.point.to_bits(),
        naive_point_bits: record.naive_point.map(f64::to_bits),
        standard_error_bits: record.standard_error.map(f64::to_bits),
        components: record
            .variance_components
            .iter()
            .map(|c| ComponentWire { name: c.name.clone(), variance_bits: c.variance.to_bits() })
            .collect(),
        rows: record.rows,
        data_identity: record.data_identity.to_hex(),
        inference: record.inference.as_str().to_owned(),
        parameter_uncertainty: record.parameter_uncertainty.as_str().to_owned(),
    }
}

fn record_from_wire(wire: &RecordWire) -> Result<CorrectedFunctionalRecord, IoError> {
    let inference = match wire.inference.as_str() {
        "point_only" => MeasurementInference::PointOnly,
        "uncalibrated_standard_error" => MeasurementInference::UncalibratedStandardError,
        other => return Err(convert(format!("unknown inference label `{other}`"))),
    };
    if (inference == MeasurementInference::PointOnly) != wire.standard_error_bits.is_none() {
        return Err(convert("the inference label disagrees with the presence of a standard error"));
    }
    Ok(CorrectedFunctionalRecord {
        model_identity: digest_from_hex(&wire.model_identity)?,
        route: MeasurementRoute::from_name(&wire.route)
            .ok_or_else(|| convert(format!("unknown measurement route `{}`", wire.route)))?,
        functional_id: wire.functional_id.clone(),
        point: f64::from_bits(wire.point_bits),
        naive_point: wire.naive_point_bits.map(f64::from_bits),
        standard_error: wire.standard_error_bits.map(f64::from_bits),
        variance_components: wire
            .components
            .iter()
            .map(|c| VarianceComponent {
                name: c.name.clone(),
                variance: f64::from_bits(c.variance_bits),
            })
            .collect(),
        rows: wire.rows,
        data_identity: digest_from_hex(&wire.data_identity)?,
        inference,
        parameter_uncertainty: ParameterUncertainty::from_name(&wire.parameter_uncertainty)
            .ok_or_else(|| {
                convert(format!("unknown parameter uncertainty `{}`", wire.parameter_uncertainty))
            })?,
    })
}

fn moments_to_wire(m: &LinearMoments) -> MomentsWire {
    MomentsWire {
        n: m.n as u64,
        dim: u32::try_from(m.dim).unwrap_or(u32::MAX),
        gram_bits: bits_vec(&m.gram),
        cross_bits: bits_vec(&m.cross),
        meat_bits: bits_vec(&m.meat),
        naive_meat_bits: bits_vec(&m.naive_meat),
        influence_cross_bits: bits_vec(&m.influence_cross),
        influence_variance_bits: m.influence_variance.to_bits(),
        error_variance_bits: m.error_variance.to_bits(),
        mean_inverse_replicates_bits: m.mean_inverse_replicates.to_bits(),
        reliability_floor_bits: m.reliability_floor.to_bits(),
    }
}

fn moments_from_wire(w: &MomentsWire) -> Result<LinearMoments, IoError> {
    let dim = w.dim as usize;
    if dim < 2 || dim > MAX_DIM {
        return Err(IoError::TooLarge);
    }
    Ok(LinearMoments {
        n: usize::try_from(w.n).map_err(|_| IoError::TooLarge)?,
        dim,
        gram: reals_vec(&w.gram_bits),
        cross: reals_vec(&w.cross_bits),
        meat: reals_vec(&w.meat_bits),
        naive_meat: reals_vec(&w.naive_meat_bits),
        influence_cross: reals_vec(&w.influence_cross_bits),
        influence_variance: f64::from_bits(w.influence_variance_bits),
        error_variance: f64::from_bits(w.error_variance_bits),
        mean_inverse_replicates: f64::from_bits(w.mean_inverse_replicates_bits),
        reliability_floor: f64::from_bits(w.reliability_floor_bits),
    })
}

fn replay_to_wire(replay: &MeasurementReplay) -> ReplayWire {
    match replay {
        MeasurementReplay::Linear { moments, coefficient_index } => ReplayWire::Linear {
            moments: moments_to_wire(moments),
            coefficient_index: u32::try_from(*coefficient_index).unwrap_or(u32::MAX),
        },
        MeasurementReplay::BinaryPrevalence { events, total, arm_id } => {
            ReplayWire::BinaryPrevalence {
                events: *events,
                total: *total,
                arm_id: arm_id.clone(),
            }
        }
        MeasurementReplay::BinaryRiskDifference { exposed_arm_id, unexposed_arm_id, table } => {
            ReplayWire::BinaryRiskDifference {
                exposed_arm_id: exposed_arm_id.clone(),
                unexposed_arm_id: unexposed_arm_id.clone(),
                exposed_events: table.exposed_events,
                exposed_total: table.exposed_total,
                unexposed_events: table.unexposed_events,
                unexposed_total: table.unexposed_total,
            }
        }
        MeasurementReplay::Categorical { classes, functional } => {
            let (kind, first, second) = match functional {
                CategoricalReplayFunctional::Proportion(level) => {
                    ("proportion", level.clone(), None)
                }
                CategoricalReplayFunctional::OutcomeMean(level) => {
                    ("outcome_mean", level.clone(), None)
                }
                CategoricalReplayFunctional::OutcomeContrast { first, second } => {
                    ("outcome_contrast", first.clone(), Some(second.clone()))
                }
            };
            ReplayWire::Categorical {
                classes: classes
                    .iter()
                    .map(|c| ClassWire {
                        count: c.count,
                        mean_bits: c.mean.to_bits(),
                        variance_bits: c.variance.to_bits(),
                    })
                    .collect(),
                functional: kind.to_owned(),
                first,
                second,
            }
        }
        MeasurementReplay::MultipleImputation { coefficients, covariances, coefficient_index } => {
            ReplayWire::MultipleImputation {
                coefficients_bits: coefficients.iter().map(|c| bits_vec(c)).collect(),
                covariances_bits: covariances.iter().map(|c| bits_vec(c)).collect(),
                coefficient_index: u32::try_from(*coefficient_index).unwrap_or(u32::MAX),
            }
        }
    }
}

fn replay_from_wire(wire: &ReplayWire) -> Result<MeasurementReplay, IoError> {
    Ok(match wire {
        ReplayWire::Linear { moments, coefficient_index } => MeasurementReplay::Linear {
            moments: moments_from_wire(moments)?,
            coefficient_index: *coefficient_index as usize,
        },
        ReplayWire::BinaryPrevalence { events, total, arm_id } => {
            MeasurementReplay::BinaryPrevalence {
                events: *events,
                total: *total,
                arm_id: arm_id.clone(),
            }
        }
        ReplayWire::BinaryRiskDifference {
            exposed_arm_id,
            unexposed_arm_id,
            exposed_events,
            exposed_total,
            unexposed_events,
            unexposed_total,
        } => MeasurementReplay::BinaryRiskDifference {
            exposed_arm_id: exposed_arm_id.clone(),
            unexposed_arm_id: unexposed_arm_id.clone(),
            table: TwoByTwoCounts {
                exposed_events: *exposed_events,
                exposed_total: *exposed_total,
                unexposed_events: *unexposed_events,
                unexposed_total: *unexposed_total,
            },
        },
        ReplayWire::Categorical { classes, functional, first, second } => {
            if classes.len() > 16 {
                return Err(IoError::TooLarge);
            }
            let functional = match (functional.as_str(), second) {
                ("proportion", None) => CategoricalReplayFunctional::Proportion(first.clone()),
                ("outcome_mean", None) => CategoricalReplayFunctional::OutcomeMean(first.clone()),
                ("outcome_contrast", Some(second)) => CategoricalReplayFunctional::OutcomeContrast {
                    first: first.clone(),
                    second: second.clone(),
                },
                _ => return Err(convert("unrecognized categorical functional")),
            };
            MeasurementReplay::Categorical {
                classes: classes
                    .iter()
                    .map(|c| ObservedClassSummary {
                        count: c.count,
                        mean: f64::from_bits(c.mean_bits),
                        variance: f64::from_bits(c.variance_bits),
                    })
                    .collect(),
                functional,
            }
        }
        ReplayWire::MultipleImputation { coefficients_bits, covariances_bits, coefficient_index } => {
            if coefficients_bits.len() > MAX_MEASUREMENT_IMPUTATIONS
                || covariances_bits.len() > MAX_MEASUREMENT_IMPUTATIONS
                || coefficients_bits.iter().any(|c| c.len() > MAX_DIM)
                || covariances_bits.iter().any(|c| c.len() > MAX_DIM * MAX_DIM)
            {
                return Err(IoError::TooLarge);
            }
            MeasurementReplay::MultipleImputation {
                coefficients: coefficients_bits.iter().map(|c| reals_vec(c)).collect(),
                covariances: covariances_bits.iter().map(|c| reals_vec(c)).collect(),
                coefficient_index: *coefficient_index as usize,
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------

/// Numbers recomputed from a replay input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayedValues {
    /// Corrected point value.
    pub point: f64,
    /// Uncorrected value of the same functional, when one exists.
    pub naive_point: Option<f64>,
    /// Delta-method variance.
    pub variance: f64,
    /// Rows or units.
    pub rows: u64,
    /// Functional name the record must carry.
    pub functional_id: &'static str,
}

fn find_arm<'a>(rates: &'a [BinaryArmRates], arm: &str) -> Option<&'a BinaryArmRates> {
    rates.iter().find(|r| r.arm_id.as_deref() == Some(arm))
}

fn error_rates(entry: &BinaryArmRates) -> Result<BinaryErrorRates, IoError> {
    BinaryErrorRates::new(
        entry.sensitivity,
        entry.specificity,
        entry.sensitivity_variance,
        entry.specificity_variance,
    )
    .map_err(|e| convert(e.to_string()))
}

fn level_index(levels: &[String], level: &str) -> Result<usize, IoError> {
    levels
        .iter()
        .position(|l| l == level)
        .ok_or_else(|| convert(format!("`{level}` is not a category of the model")))
}

/// Recompute a corrected functional from the model and its sufficient inputs.
///
/// This is the independent consumer's computation: it uses only the pure
/// functions of `antecedent-stats`, never an estimator.
///
/// # Errors
/// A replay input that does not belong to the model's route or structure, or a
/// correction the model's declared bounds refuse.
pub fn replay_values(
    model: &MeasurementModel,
    replay: &MeasurementReplay,
) -> Result<ReplayedValues, IoError> {
    match (replay, &model.parameters) {
        (
            MeasurementReplay::Linear { moments, coefficient_index },
            MeasurementParameters::ClassicalError { error_variance, min_reliability, .. },
        ) => {
            if moments.error_variance.to_bits() != error_variance.to_bits()
                || moments.reliability_floor.to_bits() != min_reliability.to_bits()
            {
                return Err(refused(
                    "the moments were computed under different measurement parameters than the model declares",
                ));
            }
            linear_values(moments, *coefficient_index)
        }
        (
            MeasurementReplay::Linear { moments, coefficient_index },
            MeasurementParameters::Repeated { min_reliability, .. },
        ) => {
            if moments.reliability_floor.to_bits() != min_reliability.to_bits() {
                return Err(refused(
                    "the moments were computed under a different reliability floor than the model declares",
                ));
            }
            linear_values(moments, *coefficient_index)
        }
        (
            MeasurementReplay::Linear { moments, coefficient_index },
            MeasurementParameters::ExternalPosterior { conditioned_on_outcome: false, .. },
        ) => {
            if moments.error_variance != 0.0 {
                return Err(refused("a posterior-mean substitution carries no error variance"));
            }
            linear_values(moments, *coefficient_index)
        }
        (
            MeasurementReplay::MultipleImputation { coefficients, covariances, coefficient_index },
            MeasurementParameters::ExternalPosterior {
                conditioned_on_outcome: true, min_draws, ..
            },
        ) => {
            if coefficients.len() < *min_draws as usize {
                return Err(refused("fewer imputations than the model declares"));
            }
            let combined =
                rubin_combine(coefficients, covariances).map_err(|e| convert(e.to_string()))?;
            let d = combined.dim;
            if *coefficient_index >= d {
                return Err(convert("coefficient index outside the design"));
            }
            Ok(ReplayedValues {
                point: combined.coefficients[*coefficient_index],
                naive_point: None,
                variance: combined.covariance[coefficient_index * d + coefficient_index],
                rows: 0,
                functional_id: "linear_coefficient",
            })
        }
        (
            MeasurementReplay::BinaryPrevalence { events, total, arm_id },
            MeasurementParameters::Binary { rates, min_youden },
        ) => {
            let entry = match model.structure {
                ErrorStructure::NonDifferential => rates.first(),
                ErrorStructure::Differential(DifferentialAxis::Arm) => {
                    arm_id.as_deref().and_then(|arm| find_arm(rates, arm))
                }
                _ => None,
            }
            .ok_or_else(|| refused("no rates apply to the replayed arm under the declared structure"))?;
            let result = corrected_prevalence(*events, *total, &error_rates(entry)?, *min_youden)
                .map_err(|e| convert(e.to_string()))?;
            Ok(ReplayedValues {
                point: result.corrected,
                naive_point: Some(result.observed),
                variance: result.variance,
                rows: *total,
                functional_id: "latent_prevalence",
            })
        }
        (
            MeasurementReplay::BinaryRiskDifference { exposed_arm_id, unexposed_arm_id, table },
            MeasurementParameters::Binary { rates, min_youden },
        ) => {
            let result = match model.structure {
                ErrorStructure::NonDifferential => {
                    let shared = rates
                        .first()
                        .ok_or_else(|| convert("a non-differential model has shared rates"))?;
                    corrected_risk_difference_nondifferential(table, &error_rates(shared)?, *min_youden)
                }
                ErrorStructure::Differential(DifferentialAxis::Arm) => {
                    let exposed = find_arm(rates, exposed_arm_id)
                        .ok_or_else(|| refused("no rates for the exposed arm"))?;
                    let unexposed = find_arm(rates, unexposed_arm_id)
                        .ok_or_else(|| refused("no rates for the unexposed arm"))?;
                    corrected_risk_difference_differential(
                        table,
                        &error_rates(exposed)?,
                        &error_rates(unexposed)?,
                        *min_youden,
                    )
                }
                _ => return Err(refused("differential structure without a bounded route")),
            }
            .map_err(|e| convert(e.to_string()))?;
            Ok(ReplayedValues {
                point: result.corrected,
                naive_point: Some(result.naive),
                variance: result.variance,
                rows: table.exposed_total + table.unexposed_total,
                functional_id: "risk_difference",
            })
        }
        (
            MeasurementReplay::Categorical { classes, functional },
            MeasurementParameters::Categorical { levels, confusion, column_counts, max_condition },
        ) => {
            let matrix = ConfusionMatrix::new(levels.len(), confusion.clone(), column_counts.clone())
                .map_err(|e| convert(e.to_string()))?;
            let result = corrected_categories(classes, &matrix, *max_condition)
                .map_err(|e| convert(e.to_string()))?;
            let k = levels.len();
            let rows: u64 = classes.iter().map(|c| c.count).sum();
            let (point, naive, variance, id) = match functional {
                CategoricalReplayFunctional::Proportion(level) => {
                    let i = level_index(levels, level)?;
                    (
                        result.proportions[i],
                        result.observed_proportions[i],
                        result.proportion_covariance[i * k + i],
                        "category_proportion",
                    )
                }
                CategoricalReplayFunctional::OutcomeMean(level) => {
                    let i = level_index(levels, level)?;
                    (
                        result.outcome_means[i],
                        classes[i].mean,
                        result.outcome_mean_covariance[i * k + i],
                        "category_outcome_mean",
                    )
                }
                CategoricalReplayFunctional::OutcomeContrast { first, second } => {
                    let i = level_index(levels, first)?;
                    let j = level_index(levels, second)?;
                    let c = &result.outcome_mean_covariance;
                    (
                        result.outcome_means[i] - result.outcome_means[j],
                        classes[i].mean - classes[j].mean,
                        c[i * k + i] + c[j * k + j] - 2.0 * c[i * k + j],
                        "category_outcome_contrast",
                    )
                }
            };
            Ok(ReplayedValues {
                point,
                naive_point: Some(naive),
                variance,
                rows,
                functional_id: id,
            })
        }
        _ => Err(refused(
            "the replay input does not belong to the route and structure the model declares",
        )),
    }
}

fn linear_values(moments: &LinearMoments, index: usize) -> Result<ReplayedValues, IoError> {
    let fit = moments.solve().map_err(|e| convert(e.to_string()))?;
    if index >= fit.dim {
        return Err(convert("coefficient index outside the design"));
    }
    Ok(ReplayedValues {
        point: fit.coefficients[index],
        naive_point: Some(fit.naive_coefficients[index]),
        variance: fit.covariance[index * fit.dim + index],
        rows: fit.n as u64,
        functional_id: "linear_coefficient",
    })
}

fn near(stored: f64, replayed: f64) -> bool {
    if !(stored.is_finite() && replayed.is_finite()) {
        return false;
    }
    (stored - replayed).abs() <= REPLAY_TOLERANCE * (1.0 + stored.abs().max(replayed.abs()))
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

/// A validated measurement model with an optional corrected-functional result.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementModelArtifact {
    metadata: MeasurementModelMetadata,
}

fn validate_metadata(meta: &MeasurementModelMetadata) -> Result<(), IoError> {
    if meta.version != MEASUREMENT_MODEL_ARTIFACT_VERSION {
        return Err(IoError::UnsupportedVersion { version: u32::from(meta.version) });
    }
    if meta.has_interval {
        return Err(convert("a measurement-model artifact carries no interval"));
    }
    let model = model_from_wire(&meta.model)?;
    if identities_to_wire(&model) != meta.identities {
        return Err(refused(
            "the stored identities do not match the model; a semantic field was changed",
        ));
    }
    let Some(result) = &meta.result else {
        return Ok(());
    };
    let record = record_from_wire(&result.record)?;
    let ids = model.identities();
    if record.model_identity != ids.model {
        return Err(refused("the result is bound to a different measurement model"));
    }
    if record.route != model.route {
        return Err(refused("the result's route is not the model's route"));
    }
    if record.has_interval() {
        return Err(convert("a measurement-model result carries no interval"));
    }
    match (&model.parameters, &result.provider_trust) {
        (MeasurementParameters::ExternalPosterior { .. }, Some(label))
            if label == "externally_attested" || label == "exact_request_verified" => {}
        (MeasurementParameters::ExternalPosterior { .. }, _) => {
            return Err(refused(
                "an external-posterior result states how far the provider was trusted",
            ));
        }
        (_, None) => {}
        (_, Some(_)) => return Err(convert("only an external-posterior result carries a trust label")),
    }
    let replay = replay_from_wire(&result.replay)?;
    let values = replay_values(&model, &replay)?;
    let rows_ok = values.rows == 0 || values.rows == record.rows;
    // A posterior-mean substitution has no uncorrected counterpart: the fit on
    // the posterior means *is* the correction, so its record carries none. The
    // replay of its moments still computes the same-design naive solve, which
    // is not a second value of the functional and is not compared.
    let substitution = matches!(
        &model.parameters,
        MeasurementParameters::ExternalPosterior { conditioned_on_outcome: false, .. }
    );
    let naive_ok = if substitution {
        record.naive_point.is_none()
    } else {
        record.naive_point.is_some() == values.naive_point.is_some()
            && !record.naive_point.zip(values.naive_point).is_some_and(|(a, b)| !near(a, b))
    };
    if record.functional_id != values.functional_id
        || !rows_ok
        || !near(record.point, values.point)
        || !naive_ok
    {
        return Err(refused(
            "the stored result is not the recomputation of its inputs under the model",
        ));
    }
    if let Some(se) = record.standard_error {
        if !near(se * se, values.variance) {
            return Err(refused(
                "the stored standard error is not the recomputation of its inputs under the model",
            ));
        }
    }
    Ok(())
}

impl MeasurementModelArtifact {
    /// Bind a model and an optional result.
    ///
    /// # Errors
    /// An invalid model, a result bound to another model, or a result that is
    /// not the recomputation of its inputs refuses.
    pub fn new(
        model: &MeasurementModel,
        result: Option<&MeasurementResult>,
    ) -> Result<Self, IoError> {
        let metadata = MeasurementModelMetadata {
            version: MEASUREMENT_MODEL_ARTIFACT_VERSION,
            model: model_to_wire(model),
            identities: identities_to_wire(model),
            result: result.map(|r| ResultWire {
                record: record_to_wire(&r.record),
                replay: replay_to_wire(&r.replay),
                provider_trust: r.provider_trust.clone(),
            }),
            has_interval: false,
        };
        validate_metadata(&metadata)?;
        Ok(Self { metadata })
    }

    /// Stored metadata.
    #[must_use]
    pub fn metadata(&self) -> &MeasurementModelMetadata {
        &self.metadata
    }

    /// The model.
    ///
    /// # Errors
    /// None for a validated artifact; a decode failure refuses.
    pub fn model(&self) -> Result<MeasurementModel, IoError> {
        model_from_wire(&self.metadata.model)
    }

    /// The model identity digest as lowercase hex.
    #[must_use]
    pub fn model_identity_hex(&self) -> &str {
        &self.metadata.identities.model
    }

    /// The stored result, when there is one.
    ///
    /// # Errors
    /// None for a validated artifact; a decode failure refuses.
    pub fn result(&self) -> Result<Option<MeasurementResult>, IoError> {
        let Some(wire) = &self.metadata.result else {
            return Ok(None);
        };
        Ok(Some(MeasurementResult {
            record: record_from_wire(&wire.record)?,
            replay: replay_from_wire(&wire.replay)?,
            provider_trust: wire.provider_trust.clone(),
        }))
    }

    /// Whether the stored result was computed from data with this identity
    /// (see [`antecedent_core::measurement_model::measurement_data_digest`]).
    #[must_use]
    pub fn covers_data_identity(&self, identity: &SemanticDigest) -> bool {
        self.metadata
            .result
            .as_ref()
            .is_some_and(|r| r.record.data_identity == identity.to_hex())
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// A blank artifact id, an invalid state or an oversized payload refuses.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        validate_metadata(&self.metadata)?;
        if artifact_id.trim().is_empty() {
            return Err(convert("missing artifact id"));
        }
        let meta_bytes = to_cbor(&self.metadata)?;
        crate::encode_container(
            &crate::ArtifactContainer {
                kind: ARTIFACT_KIND,
                note: "measurement_model",
                max_bytes: MAX_MEASUREMENT_MODEL_ARTIFACT_BYTES,
                artifact_id,
                library_version: env!("CARGO_PKG_VERSION"),
            },
            &[crate::OutSection {
                id: META_SECTION,
                content_type: crate::CONTENT_CBOR,
                bytes: &meta_bytes,
            }],
        )
        .map_err(|error| error.into_io("unsupported measurement model layout"))
    }

    /// Load, check against an independently retained model identity, and
    /// recompute the stored result from its inputs.
    ///
    /// `expected_model_identity` is the 64-character hex model identity the
    /// consumer retained; a swapped sensitivity or specificity, population,
    /// regime, arm or representation version is a different identity.
    ///
    /// # Errors
    /// Oversized, truncated, corrupt, incompatible or semantically changed
    /// artifacts, an identity that is not the expected one, and a result that
    /// does not recompute refuse.
    pub fn from_bytes(bytes: &[u8], expected_model_identity: &str) -> Result<Self, IoError> {
        if bytes.len() > MAX_MEASUREMENT_MODEL_ARTIFACT_BYTES {
            return Err(IoError::TooLarge);
        }
        digest_from_hex(expected_model_identity)?;
        let parts = crate::decode_container(
            ARTIFACT_KIND,
            MAX_MEASUREMENT_MODEL_ARTIFACT_BYTES,
            &[META_SECTION],
            bytes,
        )
        .map_err(|error| error.into_io("unsupported measurement model layout"))?;
        let Ok([meta_bytes]) = <[Vec<u8>; 1]>::try_from(parts) else {
            return Err(convert("unsupported measurement model layout"));
        };
        let metadata: MeasurementModelMetadata = from_cbor(&meta_bytes)?;
        if metadata.identities.model != expected_model_identity {
            return Err(refused(
                "the artifact's model identity differs from the identity the consumer retained",
            ));
        }
        validate_metadata(&metadata)?;
        Ok(Self { metadata })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::measurement_model::{
        MeasurementBinding, measurement_data_digest,
    };

    fn binary_model() -> MeasurementModel {
        MeasurementModel {
            model_id: "screen".into(),
            route: MeasurementRoute::BinaryMisclassification,
            indicator: MeasurementVariable {
                id: "schema:flag".into(),
                kind: MeasurementKind::NoisyMisclassified,
                units: "binary".into(),
            },
            construct: MeasurementVariable {
                id: "construct:condition".into(),
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
                    sensitivity_variance: 0.0004,
                    specificity_variance: 0.0009,
                }],
                min_youden: 0.1,
            },
            representation: RepresentationVersion {
                coordinate_system_id: "scalar".into(),
                version: 1,
            },
            evidence: ParameterEvidence::ValidationSample {
                sample_id: "val".into(),
                units: 400,
            },
        }
    }

    fn rd_result(model: &MeasurementModel) -> MeasurementResult {
        // Truth 0.4 vs 0.2 under Se 0.9 / Sp 0.8: observed 0.48 and 0.34.
        let table = TwoByTwoCounts {
            exposed_events: 480,
            exposed_total: 1000,
            unexposed_events: 340,
            unexposed_total: 1000,
        };
        let replay = MeasurementReplay::BinaryRiskDifference {
            exposed_arm_id: "treated".into(),
            unexposed_arm_id: "control".into(),
            table,
        };
        let values = replay_values(model, &replay).unwrap();
        MeasurementResult {
            record: CorrectedFunctionalRecord {
                model_identity: model.identities().model,
                route: model.route,
                functional_id: values.functional_id.into(),
                point: values.point,
                naive_point: values.naive_point,
                standard_error: Some(values.variance.sqrt()),
                variance_components: vec![VarianceComponent {
                    name: "total".into(),
                    variance: values.variance,
                }],
                rows: values.rows,
                data_identity: measurement_data_digest(&[(
                    "table",
                    &[480.0, 1000.0, 340.0, 1000.0],
                )]),
                inference: MeasurementInference::UncalibratedStandardError,
                parameter_uncertainty: ParameterUncertainty::Propagated,
            },
            replay,
            provider_trust: None,
        }
    }

    #[test]
    fn replay_matches_the_independent_closed_form() {
        let model = binary_model();
        let result = rd_result(&model);
        // Hand calculation: corrected = (0.48 - 0.34) / 0.7 = 0.2.
        assert!((result.record.point - 0.2).abs() < 1e-12);
        assert!((result.record.naive_point.unwrap() - 0.14).abs() < 1e-12);
        // Variance = [0.48*0.52/1000 + 0.34*0.66/1000]/0.49 + 0.04*(0.0004+0.0009)/0.49.
        let expect = (0.48 * 0.52 / 1000.0 + 0.34 * 0.66 / 1000.0) / 0.49
            + 0.2 * 0.2 * (0.0004 + 0.0009) / 0.49;
        let se = result.record.standard_error.unwrap();
        assert!((se * se - expect).abs() < 1e-12);
    }

    #[test]
    fn artifact_round_trips_and_an_independent_consumer_recomputes_it() {
        let model = binary_model();
        let result = rd_result(&model);
        let artifact = MeasurementModelArtifact::new(&model, Some(&result)).unwrap();
        let bytes = artifact.to_bytes("measurement-1").unwrap();
        let identity = model.identities().model.to_hex();
        let loaded = MeasurementModelArtifact::from_bytes(&bytes, &identity).unwrap();
        assert_eq!(loaded.model().unwrap(), model);
        let back = loaded.result().unwrap().unwrap();
        assert_eq!(back.record, result.record);
        assert_eq!(back.replay, result.replay);
        assert!(!loaded.metadata().has_interval);
        assert!(loaded.covers_data_identity(&result.record.data_identity));
        // Deterministic bytes.
        assert_eq!(bytes, artifact.to_bytes("measurement-1").unwrap());
    }

    fn prevalence_result(model: &MeasurementModel) -> MeasurementResult {
        // Truth 0.3 under Se 0.9 / Sp 0.8: observed 410 of 1000.
        let replay = MeasurementReplay::BinaryPrevalence {
            events: 410,
            total: 1000,
            arm_id: None,
        };
        let values = replay_values(model, &replay).unwrap();
        MeasurementResult {
            record: CorrectedFunctionalRecord {
                model_identity: model.identities().model,
                route: model.route,
                functional_id: values.functional_id.into(),
                point: values.point,
                naive_point: values.naive_point,
                standard_error: Some(values.variance.sqrt()),
                variance_components: vec![],
                rows: values.rows,
                data_identity: measurement_data_digest(&[("events", &[410.0, 1000.0])]),
                inference: MeasurementInference::UncalibratedStandardError,
                parameter_uncertainty: ParameterUncertainty::Propagated,
            },
            replay,
            provider_trust: None,
        }
    }

    #[test]
    fn a_swapped_sensitivity_and_specificity_is_refused() {
        let model = binary_model();
        let result = prevalence_result(&model);
        assert!((result.record.point - 0.3).abs() < 1e-12);
        // Retained identity of the declared model.
        let identity = model.identities().model.to_hex();
        let mut swapped = model.clone();
        swapped.parameters = MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: 0.8,
                specificity: 0.9,
                sensitivity_variance: 0.0009,
                specificity_variance: 0.0004,
            }],
            min_youden: 0.1,
        };
        // 1. A consumer's expected identity catches the swapped model.
        let forged = MeasurementModelArtifact::new(&swapped, None).unwrap();
        let bytes = forged.to_bytes("m").unwrap();
        let err = MeasurementModelArtifact::from_bytes(&bytes, &identity).unwrap_err();
        assert_eq!(err.reason_code(), Some("measurement_binding_mismatch"));
        // 2. The result of the original model cannot be attached to the swapped model.
        let err = MeasurementModelArtifact::new(&swapped, Some(&result)).unwrap_err();
        assert_eq!(err.reason_code(), Some("measurement_binding_mismatch"));
        // 3. Resealing: swapped model, original numbers, identity patched to the
        // swapped model; the prevalence (0.41 + 0.9 - 1) / 0.7 differs from 0.3.
        let mut patched = result.clone();
        patched.record.model_identity = swapped.identities().model;
        let err = MeasurementModelArtifact::new(&swapped, Some(&patched)).unwrap_err();
        assert_eq!(err.reason_code(), Some("measurement_binding_mismatch"));
    }

    #[test]
    fn population_and_representation_changes_are_different_identities() {
        let model = binary_model();
        let artifact = MeasurementModelArtifact::new(&model, None).unwrap();
        let bytes = artifact.to_bytes("m").unwrap();
        let mut moved = model.clone();
        moved.binding.population_id = "clinic-b".into();
        let err = MeasurementModelArtifact::from_bytes(&bytes, &moved.identities().model.to_hex())
            .unwrap_err();
        assert_eq!(err.reason_code(), Some("measurement_binding_mismatch"));
        let mut versioned = model.clone();
        versioned.representation.version = 2;
        assert_ne!(versioned.identities().model, model.identities().model);
        let err =
            MeasurementModelArtifact::from_bytes(&bytes, &versioned.identities().model.to_hex())
                .unwrap_err();
        assert_eq!(err.reason_code(), Some("measurement_binding_mismatch"));
        assert!(MeasurementModelArtifact::from_bytes(&bytes, "not-a-digest").is_err());
    }

    #[test]
    fn a_stored_point_that_does_not_recompute_is_refused() {
        let model = binary_model();
        let mut result = rd_result(&model);
        result.record.point += 0.01;
        let err = MeasurementModelArtifact::new(&model, Some(&result)).unwrap_err();
        assert_eq!(err.reason_code(), Some("measurement_binding_mismatch"));
        let mut result = rd_result(&model);
        result.record.standard_error = Some(result.record.standard_error.unwrap() * 1.1);
        assert!(MeasurementModelArtifact::new(&model, Some(&result)).is_err());
    }

    #[test]
    fn a_differential_replay_is_not_accepted_for_a_shared_rate_model() {
        let model = binary_model();
        let mut result = rd_result(&model);
        // Same counts, but declared to use a per-arm arm id the shared model lacks.
        result.replay = MeasurementReplay::BinaryPrevalence {
            events: 480,
            total: 1000,
            arm_id: Some("treated".into()),
        };
        assert!(MeasurementModelArtifact::new(&model, Some(&result)).is_err());
    }

    #[test]
    fn linear_replay_uses_only_the_moments() {
        use antecedent_stats::measurement_error::{ErrorVariance, LinearErrorData};
        // Full factorial design: X in {-1, 1}, U in {-0.5, 0.5}, E in {-0.7, 0.7}.
        let mut y = Vec::new();
        let mut w = Vec::new();
        for sx in [-1.0, 1.0] {
            for su in [-0.5, 0.5] {
                for se in [-0.7, 0.7] {
                    w.push(sx + su);
                    y.push(2.0 + 3.0 * sx + se);
                }
            }
        }
        let error = ErrorVariance::known(0.25, 0.0).unwrap();
        let moments = LinearMoments::accumulate(
            &LinearErrorData {
                y: &y,
                w: &w,
                covariates: &[],
                covariate_count: 0,
                replicate_counts: None,
            },
            &error,
            0.1,
        )
        .unwrap();
        let fit = moments.solve().unwrap();
        assert!((fit.coefficients[1] - 3.0).abs() < 1e-12);
        let model = MeasurementModel {
            model_id: "lab".into(),
            route: MeasurementRoute::ClassicalContinuousError,
            indicator: MeasurementVariable {
                id: "schema:w".into(),
                kind: MeasurementKind::NoisyMisclassified,
                units: "mg".into(),
            },
            construct: MeasurementVariable {
                id: "construct:x".into(),
                kind: MeasurementKind::LatentScientific,
                units: "mg".into(),
            },
            binding: MeasurementBinding {
                population_id: "cohort".into(),
                regime_id: "observational".into(),
                arm: ArmBinding::AllArms,
            },
            structure: ErrorStructure::NonDifferential,
            parameters: MeasurementParameters::ClassicalError {
                error_variance: 0.25,
                error_variance_standard_error: 0.0,
                min_reliability: 0.1,
            },
            representation: RepresentationVersion {
                coordinate_system_id: "scalar".into(),
                version: 1,
            },
            evidence: ParameterEvidence::Declared { declarer: "lab".into() },
        };
        let replay = MeasurementReplay::Linear { moments, coefficient_index: 1 };
        let values = replay_values(&model, &replay).unwrap();
        assert!((values.point - 3.0).abs() < 1e-12);
        assert!((values.naive_point.unwrap() - 2.4).abs() < 1e-12);
        // A model with a different declared error variance is not this fit.
        let mut other = model;
        other.parameters = MeasurementParameters::ClassicalError {
            error_variance: 0.3,
            error_variance_standard_error: 0.0,
            min_reliability: 0.1,
        };
        assert!(replay_values(&other, &replay).is_err());
    }
}
