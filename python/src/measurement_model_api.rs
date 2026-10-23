//! Python bridge for corrected functionals under a declared measurement model.
//!
//! Python states the model, the analysis binding and the outcome-side data as
//! typed declarations. Every check, correction, identity and refusal is
//! Rust's: a refusal comes back as structured JSON for the Python layer to
//! raise as its own typed exception, and a native failure that is not a
//! scientific refusal (a malformed declaration that never reached the
//! checks) is an input error.
//!
//! Trust is never taken from Python. A provider posterior reaches the checks
//! with an external attestation or with a verification receipt the library
//! issued; a claim of native trust is forwarded so the contract refuses it by
//! name.

use antecedent::analysis::continuous_measurement::{
    CorrectionMethod, ErrorLocation, OutcomeLink, SimexExtrapolant,
};
use antecedent::analysis::transported_measurement::{
    ErrorParameter, InvarianceDeclaration, PopulationErrorStructure, TransportBasis,
    TransportReceipt, transport_measurement_estimate, verify_measurement_transport,
};
use antecedent::{
    CategoricalTarget, ConsumedMeasurement, ContinuousErrorInput, ContinuousMeasurementEstimate,
    MeasurementAnalysisBinding, MeasurementDetail, MeasurementEstimate, MeasurementInput,
    MeasurementModelRefusal, PreparedMeasurementModel, StudyBuilder,
    consume_measurement_model_artifact,
};
use antecedent_contract::measurement::MeasurementOperation;
use antecedent_contract::{
    ExternalCapability, ExternalPosteriorKind, ExternalScientificObject, ExternalTrustState,
    PosteriorProviderContract, ProviderObjectIdentity, ScientificQuantity, capability_from_name,
};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, DifferentialAxis, ErrorStructure, MappingSemantics,
    MeasurementBinding, MeasurementKind, MeasurementModel, MeasurementParameters, MeasurementRoute,
    MeasurementVariable, ParameterEvidence, PosteriorReference, RepresentationMapping,
    RepresentationVersion,
};
use antecedent_estimate::measurement::PosteriorCombination;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_stats::measurement_error::{ObservedClassSummary, TwoByTwoCounts};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Map, Value as Json, json};

use crate::provider_api::PyProviderVerificationReceipt;
use crate::transport_common::execution_context;

/// Largest declaration this bridge parses, in bytes of JSON text.
const MAX_DECLARATION_BYTES: usize = 64 * 1024 * 1024;

// ---------------------------------------------------------------------------
// JSON access: malformed declarations are input errors, never refusals.
// ---------------------------------------------------------------------------

pub(crate) fn parse(text: &str, what: &str) -> PyResult<Json> {
    crate::bridge::check_declared_size(text, MAX_DECLARATION_BYTES, what)?;
    serde_json::from_str(text)
        .map_err(|error| crate::value_err(format!("invalid {what} declaration: {error}")))
}

pub(crate) fn object<'a>(value: &'a Json, what: &str) -> PyResult<&'a Map<String, Json>> {
    value.as_object().ok_or_else(|| crate::value_err(format!("{what} must be an object")))
}

pub(crate) fn member<'a>(map: &'a Map<String, Json>, key: &str) -> PyResult<&'a Json> {
    map.get(key).ok_or_else(|| crate::value_err(format!("missing field `{key}`")))
}

pub(crate) fn text(map: &Map<String, Json>, key: &str) -> PyResult<String> {
    member(map, key)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| crate::value_err(format!("field `{key}` must be a string")))
}

pub(crate) fn opt_text(map: &Map<String, Json>, key: &str) -> PyResult<Option<String>> {
    match map.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(crate::value_err(format!("field `{key}` must be a string or null"))),
    }
}

pub(crate) fn real(map: &Map<String, Json>, key: &str) -> PyResult<f64> {
    member(map, key)?
        .as_f64()
        .ok_or_else(|| crate::value_err(format!("field `{key}` must be a number")))
}

pub(crate) fn unsigned(map: &Map<String, Json>, key: &str) -> PyResult<u64> {
    member(map, key)?
        .as_u64()
        .ok_or_else(|| crate::value_err(format!("field `{key}` must be a non-negative integer")))
}

pub(crate) fn small(map: &Map<String, Json>, key: &str) -> PyResult<u32> {
    u32::try_from(unsigned(map, key)?)
        .map_err(|_| crate::value_err(format!("field `{key}` does not fit in 32 bits")))
}

pub(crate) fn index(map: &Map<String, Json>, key: &str) -> PyResult<usize> {
    usize::try_from(unsigned(map, key)?)
        .map_err(|_| crate::value_err(format!("field `{key}` does not fit in an index")))
}

pub(crate) fn flag(map: &Map<String, Json>, key: &str) -> PyResult<bool> {
    member(map, key)?
        .as_bool()
        .ok_or_else(|| crate::value_err(format!("field `{key}` must be a boolean")))
}

pub(crate) fn list<'a>(map: &'a Map<String, Json>, key: &str) -> PyResult<&'a Vec<Json>> {
    member(map, key)?
        .as_array()
        .ok_or_else(|| crate::value_err(format!("field `{key}` must be a list")))
}

pub(crate) fn reals(map: &Map<String, Json>, key: &str) -> PyResult<Vec<f64>> {
    list(map, key)?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| crate::value_err(format!("`{key}` holds a non-number"))))
        .collect()
}

pub(crate) fn texts(map: &Map<String, Json>, key: &str) -> PyResult<Vec<String>> {
    list(map, key)?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| crate::value_err(format!("`{key}` holds a non-string")))
        })
        .collect()
}

fn small_list(map: &Map<String, Json>, key: &str) -> PyResult<Vec<u32>> {
    list(map, key)?
        .iter()
        .map(|v| {
            v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| {
                crate::value_err(format!("`{key}` holds a value that is not a 32-bit count"))
            })
        })
        .collect()
}

pub(crate) fn present<'a>(map: &'a Map<String, Json>, key: &str) -> Option<&'a Json> {
    match map.get(key) {
        None | Some(Json::Null) => None,
        Some(value) => Some(value),
    }
}

// ---------------------------------------------------------------------------
// Refusals as data.
// ---------------------------------------------------------------------------

fn refusal_json(refusal: &MeasurementModelRefusal) -> String {
    json!({
        "code": refusal.code,
        "stage": "measurement_model",
        "detail": refusal.detail,
        "message": refusal.message,
        "offending": Json::Null,
        "remedy": refusal.remedy,
    })
    .to_string()
}

fn declaration_refusal(code: &'static str, detail: &str, message: &str) -> MeasurementModelRefusal {
    MeasurementModelRefusal {
        code,
        detail: detail.to_owned(),
        message: message.to_owned(),
        remedy: None,
    }
}

// ---------------------------------------------------------------------------
// Declarations.
// ---------------------------------------------------------------------------

fn parse_variable(value: &Json, what: &str) -> PyResult<MeasurementVariable> {
    let map = object(value, what)?;
    let name = text(map, "kind")?;
    let kind = MeasurementKind::from_name(&name)
        .ok_or_else(|| crate::value_err(format!("unknown measurement kind `{name}`")))?;
    Ok(MeasurementVariable { id: text(map, "id")?, kind, units: text(map, "units")? })
}

fn parse_representation(value: &Json) -> PyResult<RepresentationVersion> {
    let map = object(value, "representation")?;
    Ok(RepresentationVersion {
        coordinate_system_id: text(map, "coordinate_system_id")?,
        version: small(map, "version")?,
    })
}

fn parse_structure(name: &str) -> PyResult<ErrorStructure> {
    match name {
        "non_differential" => Ok(ErrorStructure::NonDifferential),
        "differential_arm" => Ok(ErrorStructure::Differential(DifferentialAxis::Arm)),
        "differential_population" => Ok(ErrorStructure::Differential(DifferentialAxis::Population)),
        "differential_time" => Ok(ErrorStructure::Differential(DifferentialAxis::Time)),
        other => Err(crate::value_err(format!("unknown error structure `{other}`"))),
    }
}

fn structure_name(structure: ErrorStructure) -> &'static str {
    match structure {
        ErrorStructure::NonDifferential => "non_differential",
        ErrorStructure::Differential(DifferentialAxis::Arm) => "differential_arm",
        ErrorStructure::Differential(DifferentialAxis::Population) => "differential_population",
        ErrorStructure::Differential(DifferentialAxis::Time) => "differential_time",
        _ => "unrecognized",
    }
}

fn parse_mapping(value: &Json) -> PyResult<RepresentationMapping> {
    let map = object(value, "representation mapping")?;
    let semantics = object(member(map, "semantics")?, "mapping semantics")?;
    let semantics = match text(semantics, "kind")?.as_str() {
        "coordinate_relabeling" => MappingSemantics::CoordinateRelabeling,
        "intervention_preserving" => MappingSemantics::InterventionPreserving {
            evidence_id: text(semantics, "evidence_id")?,
        },
        other => {
            return Err(crate::value_err(format!("unknown mapping semantics `{other}`")));
        }
    };
    Ok(RepresentationMapping {
        from: parse_representation(member(map, "from")?)?,
        to: parse_representation(member(map, "to")?)?,
        mapping_id: text(map, "mapping_id")?,
        semantics,
    })
}

fn parse_reference(value: &Json) -> PyResult<PosteriorReference> {
    let map = object(value, "posterior reference")?;
    Ok(PosteriorReference {
        provider_id: text(map, "provider_id")?,
        object_id: text(map, "object_id")?,
        version_id: text(map, "version_id")?,
        snapshot_id: text(map, "snapshot_id")?,
        request_id: text(map, "request_id")?,
    })
}

fn parse_parameters(value: &Json) -> PyResult<MeasurementParameters> {
    let map = object(value, "parameters")?;
    match text(map, "kind")?.as_str() {
        "classical_error" => Ok(MeasurementParameters::ClassicalError {
            error_variance: real(map, "error_variance")?,
            error_variance_standard_error: real(map, "error_variance_standard_error")?,
            min_reliability: real(map, "min_reliability")?,
        }),
        "binary" => {
            let mut rates = Vec::new();
            for entry in list(map, "rates")? {
                let entry = object(entry, "rates entry")?;
                rates.push(BinaryArmRates {
                    arm_id: opt_text(entry, "arm_id")?,
                    sensitivity: real(entry, "sensitivity")?,
                    specificity: real(entry, "specificity")?,
                    sensitivity_variance: real(entry, "sensitivity_variance")?,
                    specificity_variance: real(entry, "specificity_variance")?,
                });
            }
            Ok(MeasurementParameters::Binary { rates, min_youden: real(map, "min_youden")? })
        }
        "repeated" => Ok(MeasurementParameters::Repeated {
            min_replicated_units: small(map, "min_replicated_units")?,
            min_reliability: real(map, "min_reliability")?,
        }),
        "categorical" => {
            let column_counts = match present(map, "column_counts") {
                None => None,
                Some(Json::Array(items)) => Some(
                    items
                        .iter()
                        .map(|v| {
                            v.as_u64().ok_or_else(|| {
                                crate::value_err("`column_counts` holds a non-count")
                            })
                        })
                        .collect::<PyResult<Vec<u64>>>()?,
                ),
                Some(_) => {
                    return Err(crate::value_err("`column_counts` must be a list or null"));
                }
            };
            Ok(MeasurementParameters::Categorical {
                levels: texts(map, "levels")?,
                confusion: reals(map, "confusion")?,
                column_counts,
                max_condition: real(map, "max_condition")?,
            })
        }
        "external_posterior" => Ok(MeasurementParameters::ExternalPosterior {
            reference: parse_reference(member(map, "reference")?)?,
            conditioned_on_outcome: flag(map, "conditioned_on_outcome")?,
            min_draws: small(map, "min_draws")?,
        }),
        other => Err(crate::value_err(format!("unknown measurement parameters `{other}`"))),
    }
}

fn parse_evidence(value: &Json) -> PyResult<ParameterEvidence> {
    let map = object(value, "parameter evidence")?;
    match text(map, "kind")?.as_str() {
        "declared" => Ok(ParameterEvidence::Declared { declarer: text(map, "declarer")? }),
        "validation_sample" => Ok(ParameterEvidence::ValidationSample {
            sample_id: text(map, "sample_id")?,
            units: unsigned(map, "units")?,
        }),
        "replicate_sample" => {
            Ok(ParameterEvidence::ReplicateSample { sample_id: text(map, "sample_id")? })
        }
        "external_provider" => {
            Ok(ParameterEvidence::ExternalProvider { provider_id: text(map, "provider_id")? })
        }
        other => Err(crate::value_err(format!("unknown parameter evidence `{other}`"))),
    }
}

pub(crate) fn parse_model(value: &Json) -> PyResult<MeasurementModel> {
    let map = object(value, "measurement model")?;
    let route_name = text(map, "route")?;
    let route = MeasurementRoute::from_name(&route_name)
        .ok_or_else(|| crate::value_err(format!("unknown measurement route `{route_name}`")))?;
    let binding = object(member(map, "binding")?, "model binding")?;
    let arm = match opt_text(binding, "arm")? {
        Some(arm) => ArmBinding::Arm(arm),
        None => ArmBinding::AllArms,
    };
    Ok(MeasurementModel {
        model_id: text(map, "model_id")?,
        route,
        indicator: parse_variable(member(map, "indicator")?, "indicator")?,
        construct: parse_variable(member(map, "construct")?, "construct")?,
        binding: MeasurementBinding {
            population_id: text(binding, "population_id")?,
            regime_id: text(binding, "regime_id")?,
            arm,
        },
        structure: parse_structure(&text(map, "structure")?)?,
        parameters: parse_parameters(member(map, "parameters")?)?,
        representation: parse_representation(member(map, "representation")?)?,
        evidence: parse_evidence(member(map, "evidence")?)?,
    })
}

fn representation_json(version: &RepresentationVersion) -> Json {
    json!({"coordinate_system_id": version.coordinate_system_id, "version": version.version})
}

fn variable_json(variable: &MeasurementVariable) -> Json {
    json!({"id": variable.id, "kind": variable.kind.as_str(), "units": variable.units})
}

fn parameters_json(parameters: &MeasurementParameters) -> Json {
    match parameters {
        MeasurementParameters::ClassicalError {
            error_variance,
            error_variance_standard_error,
            min_reliability,
        } => json!({
            "kind": "classical_error",
            "error_variance": error_variance,
            "error_variance_standard_error": error_variance_standard_error,
            "min_reliability": min_reliability,
        }),
        MeasurementParameters::Binary { rates, min_youden } => json!({
            "kind": "binary",
            "min_youden": min_youden,
            "rates": rates.iter().map(|r| json!({
                "arm_id": r.arm_id,
                "sensitivity": r.sensitivity,
                "specificity": r.specificity,
                "sensitivity_variance": r.sensitivity_variance,
                "specificity_variance": r.specificity_variance,
            })).collect::<Vec<_>>(),
        }),
        MeasurementParameters::Repeated { min_replicated_units, min_reliability } => json!({
            "kind": "repeated",
            "min_replicated_units": min_replicated_units,
            "min_reliability": min_reliability,
        }),
        MeasurementParameters::Categorical { levels, confusion, column_counts, max_condition } => {
            json!({
                "kind": "categorical",
                "levels": levels,
                "confusion": confusion,
                "column_counts": column_counts,
                "max_condition": max_condition,
            })
        }
        MeasurementParameters::ExternalPosterior {
            reference,
            conditioned_on_outcome,
            min_draws,
        } => {
            json!({
                "kind": "external_posterior",
                "reference": {
                    "provider_id": reference.provider_id,
                    "object_id": reference.object_id,
                    "version_id": reference.version_id,
                    "snapshot_id": reference.snapshot_id,
                    "request_id": reference.request_id,
                },
                "conditioned_on_outcome": conditioned_on_outcome,
                "min_draws": min_draws,
            })
        }
    }
}

fn evidence_json(evidence: &ParameterEvidence) -> Json {
    match evidence {
        ParameterEvidence::Declared { declarer } => {
            json!({"kind": "declared", "declarer": declarer})
        }
        ParameterEvidence::ValidationSample { sample_id, units } => {
            json!({"kind": "validation_sample", "sample_id": sample_id, "units": units})
        }
        ParameterEvidence::ReplicateSample { sample_id } => {
            json!({"kind": "replicate_sample", "sample_id": sample_id})
        }
        ParameterEvidence::ExternalProvider { provider_id } => {
            json!({"kind": "external_provider", "provider_id": provider_id})
        }
        _ => json!({"kind": "unrecognized"}),
    }
}

pub(crate) fn model_json(model: &MeasurementModel) -> Json {
    let arm = match &model.binding.arm {
        ArmBinding::Arm(arm) => Json::String(arm.clone()),
        _ => Json::Null,
    };
    json!({
        "model_id": model.model_id,
        "route": model.route.as_str(),
        "indicator": variable_json(&model.indicator),
        "construct": variable_json(&model.construct),
        "binding": {
            "population_id": model.binding.population_id,
            "regime_id": model.binding.regime_id,
            "arm": arm,
        },
        "structure": structure_name(model.structure),
        "parameters": parameters_json(&model.parameters),
        "representation": representation_json(&model.representation),
        "evidence": evidence_json(&model.evidence),
    })
}

fn parse_analysis_binding(value: &Json) -> PyResult<MeasurementAnalysisBinding> {
    let map = object(value, "analysis binding")?;
    let representation_mapping = match present(map, "representation_mapping") {
        None => None,
        Some(mapping) => Some(parse_mapping(mapping)?),
    };
    Ok(MeasurementAnalysisBinding {
        construct_id: text(map, "construct_id")?,
        population_id: text(map, "population_id")?,
        regime_id: text(map, "regime_id")?,
        arm_id: opt_text(map, "arm_id")?,
        representation: parse_representation(member(map, "representation")?)?,
        representation_mapping,
        assumed_structure: parse_structure(&text(map, "assumed_structure")?)?,
    })
}

fn parse_target(value: &Json) -> PyResult<CategoricalTarget> {
    let map = object(value, "categorical target")?;
    match text(map, "kind")?.as_str() {
        "proportion" => Ok(CategoricalTarget::Proportion(text(map, "level")?)),
        "outcome_mean" => Ok(CategoricalTarget::OutcomeMean(text(map, "level")?)),
        "outcome_contrast" => Ok(CategoricalTarget::OutcomeContrast {
            first: text(map, "first")?,
            second: text(map, "second")?,
        }),
        other => Err(crate::value_err(format!("unknown categorical target `{other}`"))),
    }
}

fn parse_input(value: &Json) -> PyResult<MeasurementInput> {
    let map = object(value, "measurement data")?;
    match text(map, "kind")?.as_str() {
        "classical_linear" => {
            let replicate_counts = match present(map, "replicate_counts") {
                None => None,
                Some(_) => Some(small_list(map, "replicate_counts")?),
            };
            Ok(MeasurementInput::ClassicalLinear {
                outcome: reals(map, "outcome")?,
                regressor: reals(map, "regressor")?,
                covariates: reals(map, "covariates")?,
                covariate_count: index(map, "covariate_count")?,
                replicate_counts,
            })
        }
        "repeated_linear" => Ok(MeasurementInput::RepeatedLinear {
            outcome: reals(map, "outcome")?,
            replicate_values: reals(map, "replicate_values")?,
            replicate_counts: small_list(map, "replicate_counts")?,
            covariates: reals(map, "covariates")?,
            covariate_count: index(map, "covariate_count")?,
        }),
        "binary_prevalence" => Ok(MeasurementInput::BinaryPrevalence {
            events: unsigned(map, "events")?,
            total: unsigned(map, "total")?,
        }),
        "binary_risk_difference" => Ok(MeasurementInput::BinaryRiskDifference {
            exposed_arm_id: text(map, "exposed_arm_id")?,
            unexposed_arm_id: text(map, "unexposed_arm_id")?,
            table: TwoByTwoCounts {
                exposed_events: unsigned(map, "exposed_events")?,
                exposed_total: unsigned(map, "exposed_total")?,
                unexposed_events: unsigned(map, "unexposed_events")?,
                unexposed_total: unsigned(map, "unexposed_total")?,
            },
        }),
        "categorical" => {
            let mut classes = Vec::new();
            for entry in list(map, "classes")? {
                let entry = object(entry, "class summary")?;
                classes.push(ObservedClassSummary {
                    count: unsigned(entry, "count")?,
                    mean: real(entry, "mean")?,
                    variance: real(entry, "variance")?,
                });
            }
            Ok(MeasurementInput::Categorical {
                classes,
                target: parse_target(member(map, "target")?)?,
            })
        }
        "latent_posterior" => Ok(MeasurementInput::LatentPosterior {
            outcome: reals(map, "outcome")?,
            draws: reals(map, "draws")?,
            draw_count: index(map, "draw_count")?,
            covariates: reals(map, "covariates")?,
            covariate_count: index(map, "covariate_count")?,
        }),
        other => Err(crate::value_err(format!("unknown measurement data kind `{other}`"))),
    }
}

fn parse_continuous_method(value: &Json) -> PyResult<CorrectionMethod> {
    let map = object(value, "correction method")?;
    match text(map, "kind")?.as_str() {
        "regression_calibration" => Ok(CorrectionMethod::RegressionCalibration),
        "simex" => Ok(CorrectionMethod::Simex {
            lambdas: reals(map, "lambdas")?,
            replicates: small(map, "replicates")?,
            extrapolant: match text(map, "extrapolant")?.as_str() {
                "linear" => SimexExtrapolant::Linear,
                "quadratic" => SimexExtrapolant::Quadratic,
                other => {
                    return Err(crate::value_err(format!("unknown SIMEX extrapolant `{other}`")));
                }
            },
        }),
        other => Err(crate::value_err(format!("unknown correction method `{other}`"))),
    }
}

fn parse_continuous_input(value: &Json) -> PyResult<ContinuousErrorInput> {
    let map = object(value, "continuous measurement data")?;
    let location = match text(map, "location")?.as_str() {
        "treatment" => ErrorLocation::Treatment,
        "covariate" => {
            ErrorLocation::Covariate { treatment_column: index(map, "treatment_column")? }
        }
        "outcome" => ErrorLocation::Outcome,
        other => return Err(crate::value_err(format!("unknown error location `{other}`"))),
    };
    let link = match text(map, "link")?.as_str() {
        "identity" => OutcomeLink::Identity,
        "logit" => OutcomeLink::Logit,
        other => return Err(crate::value_err(format!("unknown outcome link `{other}`"))),
    };
    Ok(ContinuousErrorInput {
        outcome: reals(map, "outcome")?,
        regressor: reals(map, "regressor")?,
        covariates: reals(map, "covariates")?,
        covariate_count: index(map, "covariate_count")?,
        replicate_values: match present(map, "replicate_values") {
            Some(_) => Some(reals(map, "replicate_values")?),
            None => None,
        },
        replicate_counts: match present(map, "replicate_counts") {
            Some(_) => Some(small_list(map, "replicate_counts")?),
            None => None,
        },
        spec: antecedent::analysis::continuous_measurement::ContinuousCorrectionSpec {
            location,
            link,
            method: parse_continuous_method(member(map, "method")?)?,
            bootstrap_replicates: small(map, "bootstrap_replicates")?,
            seed: unsigned(map, "seed")?,
        },
    })
}

fn operation_from_name(name: &str) -> PyResult<MeasurementOperation> {
    [
        MeasurementOperation::LinearAdjustedEffect,
        MeasurementOperation::LinearFunctionalOfCoefficients,
        MeasurementOperation::LatentPrevalence,
        MeasurementOperation::RiskDifference,
        MeasurementOperation::CategoryOutcomeMeans,
        MeasurementOperation::ConditionOnConstruct,
        MeasurementOperation::InterveneOnRepresentation,
        MeasurementOperation::TransportToOtherPopulation,
        MeasurementOperation::NonlinearOutcomeModel,
        MeasurementOperation::Interval,
    ]
    .into_iter()
    .find(|operation| operation.as_str() == name)
    .ok_or_else(|| crate::value_err(format!("unknown measurement operation `{name}`")))
}

/// The provider object a posterior route names, and what is claimed for it.
struct ProviderDeclaration {
    object: ExternalScientificObject,
    claimed_trust: String,
    attestor: Option<String>,
}

fn parse_provider(value: &Json) -> PyResult<ProviderDeclaration> {
    let map = object(value, "provider object")?;
    let identity = object(member(map, "identity")?, "provider identity")?;
    let mut quantities = Vec::new();
    for entry in list(map, "quantities")? {
        let wire: ScientificQuantityWire = serde_json::from_value(entry.clone())
            .map_err(|error| crate::value_err(format!("invalid provider quantity: {error}")))?;
        quantities.push(ScientificQuantity::try_from(wire).map_err(crate::value_err)?);
    }
    let mut capabilities: Vec<ExternalCapability> = Vec::new();
    for name in texts(map, "capabilities")? {
        capabilities.push(
            capability_from_name(&name)
                .ok_or_else(|| crate::value_err(format!("unknown capability `{name}`")))?,
        );
    }
    let kind = match text(map, "kind")?.as_str() {
        "parameter" => ExternalPosteriorKind::Parameter,
        "functional" => ExternalPosteriorKind::Functional,
        "causal_functional" => ExternalPosteriorKind::CausalFunctional,
        "structural_model" => ExternalPosteriorKind::StructuralModel,
        other => return Err(crate::value_err(format!("unknown posterior kind `{other}`"))),
    };
    let object = ExternalScientificObject::Posterior(PosteriorProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: text(identity, "provider_id")?,
            object_id: text(identity, "object_id")?,
            version_id: text(identity, "version_id")?,
            snapshot_id: text(identity, "snapshot_id")?,
            request_id: text(identity, "request_id")?,
        },
        kind,
        quantities,
        capabilities,
    });
    Ok(ProviderDeclaration {
        object,
        claimed_trust: text(map, "trust")?,
        attestor: opt_text(map, "attestor")?,
    })
}

/// The trust state the declaration claims. A receipt the library issued
/// stands for exact-request verification; an attestation is an external
/// party's assertion; a claim of native trust is forwarded and refused.
fn trust_of(
    provider: &ProviderDeclaration,
    receipt: Option<&PyProviderVerificationReceipt>,
) -> Result<ExternalTrustState, MeasurementModelRefusal> {
    match provider.claimed_trust.as_str() {
        "exact_request_verified" => match receipt {
            Some(receipt) => {
                Ok(ExternalTrustState::ExactRequestVerified(Box::new(receipt.inner.clone())))
            }
            None => Err(declaration_refusal(
                antecedent_core::reason_code!("measurement_posterior_not_trusted"),
                "measurement_model.posterior_receipt",
                "exact-request verification needs the verification receipt the library issued; \
                 a label is not a receipt",
            )),
        },
        "externally_attested" => {
            let attestor = provider.attestor.as_deref().unwrap_or("");
            ExternalTrustState::attest(&provider.object, attestor).map_err(|error| {
                declaration_refusal(
                    antecedent_core::reason_code!("measurement_posterior_not_trusted"),
                    "measurement_model.posterior_declaration",
                    &format!("the provider declaration cannot be attested: {error:?}"),
                )
            })
        }
        "native_licensed" => Ok(ExternalTrustState::NativeLicensed),
        _ => Err(declaration_refusal(
            antecedent_core::reason_code!("measurement_posterior_not_trusted"),
            "measurement_model.posterior_declaration",
            "a provider posterior is externally attested or exact-request verified; \
             nothing else is a trust state a caller can state",
        )),
    }
}

// ---------------------------------------------------------------------------
// Results as data.
// ---------------------------------------------------------------------------

fn record_json(record: &antecedent_core::measurement_model::CorrectedFunctionalRecord) -> Json {
    json!({
        "model_identity": record.model_identity.to_hex(),
        "route": record.route.as_str(),
        "functional_id": record.functional_id,
        "point": record.point,
        "naive_point": record.naive_point,
        "standard_error": record.standard_error,
        "variance_components": record.variance_components.iter().map(|c| json!({
            "name": c.name,
            "variance": c.variance,
        })).collect::<Vec<_>>(),
        "rows": record.rows,
        "data_identity": record.data_identity.to_hex(),
        "inference": record.inference.as_str(),
        "parameter_uncertainty": record.parameter_uncertainty.as_str(),
    })
}

fn estimate_json(estimate: &MeasurementEstimate) -> Json {
    let mut summary = record_json(estimate.record());
    summary["operation"] = json!(estimate.operation().as_str());
    summary["provider_trust"] = json!(
        estimate
            .provider_trust()
            .map(antecedent_contract::measurement::LatentPosteriorTrust::as_str)
    );
    summary["representation_mapping_id"] = json!(estimate.representation_mapping_id());
    summary["interval"] = Json::Null;
    summary
}

fn detail_json(estimate: &MeasurementEstimate) -> Json {
    match estimate.detail() {
        MeasurementDetail::Linear(effect) => json!({
            "kind": "linear",
            "dimension": effect.fit.dim,
            "rows": effect.fit.n,
            "coefficients": effect.fit.coefficients,
            "naive_coefficients": effect.fit.naive_coefficients,
            "covariance": effect.fit.covariance,
            "covariance_given_error_variance": effect.fit.covariance_given_error_variance,
            "naive_covariance": effect.fit.naive_covariance,
            "reliability": effect.fit.reliability,
            "error_variance": effect.fit.error_variance,
            "components": effect.components.as_ref().map(|c| json!({
                "units": c.units,
                "observations": c.observations,
                "grand_mean": c.grand_mean,
                "within_variance": c.within_variance,
                "between_variance": c.between_variance,
                "intraclass_correlation": c.intraclass_correlation,
            })),
        }),
        MeasurementDetail::PosteriorLinear(effect) => json!({
            "kind": "posterior_linear",
            "combination": match effect.combination {
                PosteriorCombination::MultipleImputation => "multiple_imputation",
                PosteriorCombination::PosteriorMeanSubstitution => "posterior_mean_substitution",
            },
            "dimension": effect.coefficients.len(),
            "coefficients": effect.coefficients,
            "covariance": effect.covariance,
        }),
        MeasurementDetail::BinaryPrevalence(corrected) => json!({
            "kind": "binary_prevalence",
            "events": corrected.events,
            "total": corrected.total,
        }),
        MeasurementDetail::BinaryRiskDifference(corrected) => json!({
            "kind": "binary_risk_difference",
            "differential": corrected.differential,
        }),
        MeasurementDetail::Categorical(corrected, _) => json!({
            "kind": "categorical",
            "levels": corrected.levels,
        }),
        _ => json!({"kind": "unrecognized"}),
    }
}

fn continuous_json(estimate: &ContinuousMeasurementEstimate) -> Json {
    let effect = estimate.effect();
    let summary = &effect.summary;
    let mut out = record_json(estimate.record());
    out["method"] = json!(summary.method.as_str());
    out["link"] = json!(summary.link.as_str());
    out["location"] = json!(summary.location.as_str());
    out["treatment_column"] = match summary.location {
        ErrorLocation::Covariate { treatment_column } => json!(treatment_column),
        _ => Json::Null,
    };
    out["error_variance"] = json!(summary.error_variance);
    out["error_variance_standard_error"] = json!(summary.error_variance_standard_error);
    out["reliability"] = json!(summary.reliability);
    out["bootstrap_replicates"] = json!(summary.bootstrap_replicates);
    out["failed_bootstrap_replicates"] = json!(summary.failed_bootstrap_replicates);
    out["seed"] = json!(summary.seed);
    out["coefficients"] = json!(effect.coefficients);
    out["naive_coefficients"] = json!(effect.naive_coefficients);
    out["coefficient_index"] = json!(effect.coefficient_index);
    out["simex"] = effect.simex.as_ref().map_or(Json::Null, |fit| {
        json!({
            "lambdas": fit.lambdas,
            "means": fit.lambda_means,
            "extrapolant": fit.extrapolant.as_str(),
            "replicates": fit.replicates,
        })
    });
    out["interval"] = Json::Null;
    out
}

fn route_card_json(prepared: &PreparedMeasurementModel) -> Json {
    let card = prepared.route_card();
    json!({
        "route": card.route.as_str(),
        "assumptions": card.assumptions,
        "allowed": card.allowed.iter().map(|o| o.as_str()).collect::<Vec<_>>(),
        "refused": card.refused.iter().map(|(operation, why)| json!({
            "operation": operation.as_str(),
            "why": why,
        })).collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------------------
// Handles.
// ---------------------------------------------------------------------------

/// A checked model, binding and data. Every method that changes it returns a
/// new handle and leaves this one untouched.
#[pyclass(name = "PreparedMeasurementHandle", skip_from_py_object)]
pub(crate) struct PreparedMeasurementHandle {
    inner: PreparedMeasurementModel,
}

/// A corrected functional bound to its model, with the inputs to replay it.
#[pyclass(name = "MeasurementEstimateHandle", skip_from_py_object)]
pub(crate) struct MeasurementEstimateHandle {
    inner: MeasurementEstimate,
}

/// A consumed artifact: the model, the stored record and the replayed values.
#[pyclass(name = "ConsumedMeasurementHandle", skip_from_py_object)]
pub(crate) struct ConsumedMeasurementHandle {
    inner: ConsumedMeasurement,
}

type Prepared =
    (Option<PreparedMeasurementHandle>, Option<MeasurementEstimateHandle>, Option<String>);

fn refreshed(outcome: Result<antecedent::MeasurementRefresh, MeasurementModelRefusal>) -> Prepared {
    match outcome {
        Ok(done) => (
            Some(PreparedMeasurementHandle { inner: done.prepared }),
            Some(MeasurementEstimateHandle { inner: done.estimate }),
            None,
        ),
        Err(refusal) => (None, None, Some(refusal_json(&refusal))),
    }
}

#[pymethods]
impl PreparedMeasurementHandle {
    /// The checked model as a declaration.
    fn model_json(&self) -> String {
        model_json(self.inner.model()).to_string()
    }

    /// The analysis binding the model was checked against.
    fn binding_json(&self) -> String {
        let binding = self.inner.binding();
        json!({
            "construct_id": binding.construct_id,
            "population_id": binding.population_id,
            "regime_id": binding.regime_id,
            "arm_id": binding.arm_id,
            "representation": representation_json(&binding.representation),
            "assumed_structure": structure_name(binding.assumed_structure),
            "representation_mapping_id":
                binding.representation_mapping.as_ref().map(|m| m.mapping_id.clone()),
        })
        .to_string()
    }

    /// The separately frozen identities and the operation, as JSON.
    fn identities_json(&self) -> String {
        let identities = self.inner.identities();
        json!({
            "measurement": identities.measurement.to_hex(),
            "population": identities.population.to_hex(),
            "regime": identities.regime.to_hex(),
            "arm": identities.arm.to_hex(),
            "representation": identities.representation.to_hex(),
            "evidence": identities.evidence.to_hex(),
            "model": identities.model.to_hex(),
            "operation": self.inner.operation().as_str(),
            "provider_trust": self.inner.provider_trust().map(antecedent_contract::measurement::LatentPosteriorTrust::as_str),
        })
        .to_string()
    }

    /// Assumptions, allowed operations and refused operations of the route.
    fn route_card_json(&self) -> String {
        route_card_json(&self.inner).to_string()
    }

    /// The refusal of `operation` for this binding, or `None` when licensed.
    fn check_operation(&self, operation: &str) -> PyResult<Option<String>> {
        let operation = operation_from_name(operation)?;
        Ok(self.inner.check_operation(operation).err().map(|r| refusal_json(&r)))
    }

    /// Always the typed refusal: no coverage record measures the method.
    fn interval(&self) -> String {
        match self.inner.interval() {
            Err(refusal) => refusal_json(&refusal),
            Ok(never) => match never {},
        }
    }

    /// Form the corrected functional from the retained data.
    #[pyo3(signature = (*, cancel=None))]
    fn estimate(
        &self,
        py: Python<'_>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<MeasurementEstimateHandle>, Option<String>)> {
        let inner = self.inner.clone();
        let outcome = crate::detach_catch(py, move || {
            let ctx = execution_context(0, None, cancel);
            Ok(inner.estimate(&ctx))
        })?;
        Ok(match outcome {
            Ok(estimate) => (Some(MeasurementEstimateHandle { inner: estimate }), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        })
    }

    /// Correct one coefficient for continuous classical measurement error
    /// (regression calibration or SIMEX, linear or logistic outcome model).
    /// Returns the result as JSON or the structured refusal.
    #[pyo3(signature = (input_json, *, cancel=None))]
    fn estimate_continuous(
        &self,
        py: Python<'_>,
        input_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<String>, Option<String>)> {
        let input = parse_continuous_input(&parse(input_json, "continuous measurement data")?)?;
        let inner = self.inner.clone();
        let outcome = crate::detach_catch(py, move || {
            let ctx = execution_context(0, None, cancel);
            Ok(inner.estimate_continuous(input, &ctx))
        })?;
        Ok(match outcome {
            Ok(estimate) => (Some(continuous_json(&estimate).to_string()), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        })
    }

    /// Recompute with new outcome-only data of the same question. A refusal
    /// leaves this handle as it was.
    #[pyo3(signature = (data_json, *, cancel=None))]
    fn refresh(
        &self,
        py: Python<'_>,
        data_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Prepared> {
        let input = parse_input(&parse(data_json, "measurement data")?)?;
        let inner = self.inner.clone();
        let outcome = crate::detach_catch(py, move || {
            let ctx = execution_context(0, None, cancel);
            Ok(inner.refresh(input, &ctx))
        })?;
        Ok(refreshed(outcome))
    }

    /// Rebuild the model with new validation parameters and evidence and
    /// recompute on the retained data. A refusal leaves this handle as it was.
    #[pyo3(signature = (parameters_json, evidence_json, *, cancel=None))]
    fn with_validation_parameters(
        &self,
        py: Python<'_>,
        parameters_json: &str,
        evidence_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Prepared> {
        let parameters = parse_parameters(&parse(parameters_json, "parameters")?)?;
        let evidence = parse_evidence(&parse(evidence_json, "parameter evidence")?)?;
        let inner = self.inner.clone();
        let outcome = crate::detach_catch(py, move || {
            let ctx = execution_context(0, None, cancel);
            Ok(inner.with_validation_parameters(parameters, evidence, &ctx))
        })?;
        Ok(refreshed(outcome))
    }
}

#[pymethods]
impl MeasurementEstimateHandle {
    /// The record, its operation, trust label and mapping, as JSON.
    fn summary_json(&self) -> String {
        estimate_json(&self.inner).to_string()
    }

    /// The estimator's own detail for the route (coefficients, covariance,
    /// components), as JSON.
    fn detail_json(&self) -> String {
        detail_json(&self.inner).to_string()
    }

    /// Always the typed refusal: no coverage record measures the method.
    fn interval(&self) -> String {
        match self.inner.interval() {
            Err(refusal) => refusal_json(&refusal),
            Ok(never) => match never {},
        }
    }

    /// Value and variance of a linear functional of the corrected
    /// coefficient vector, or the refusal.
    fn downstream_linear_functional(
        &self,
        weights: Vec<f64>,
    ) -> (Option<(f64, f64)>, Option<String>) {
        match self.inner.downstream_linear_functional(&weights) {
            Ok(pair) => (Some(pair), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        }
    }

    /// The artifact bytes, or the refusal.
    fn export(&self, py: Python<'_>, artifact_id: &str) -> (Option<Py<PyBytes>>, Option<String>) {
        match self.inner.export(artifact_id) {
            Ok(bytes) => (Some(PyBytes::new(py, &bytes).unbind()), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        }
    }
}

#[pymethods]
impl ConsumedMeasurementHandle {
    /// The model, the stored record and the recomputed values, as JSON.
    fn summary_json(&self) -> String {
        let consumed = &self.inner;
        json!({
            "model": model_json(consumed.model()),
            "model_identity": consumed.model().identities().model.to_hex(),
            "record": consumed.record().map(record_json),
            "replayed": consumed.replayed().map(|r| json!({
                "point": r.point,
                "naive_point": r.naive_point,
                "variance": r.variance,
                "standard_error": r.variance.sqrt(),
                "rows": r.rows,
                "functional_id": r.functional_id,
            })),
            "provider_trust": consumed.provider_trust(),
        })
        .to_string()
    }

    /// Whether the stored result was computed from data with this identity.
    fn covers_data_identity(&self, identity_hex: &str) -> PyResult<bool> {
        let digest = antecedent_contract::evidence_ledger::parse_digest_hex(identity_hex)
            .ok_or_else(|| crate::value_err("a data identity is 64 hexadecimal characters"))?;
        Ok(self.inner.covers_data_identity(&digest))
    }

    /// The refusal when the stored result may not be used for `binding` and
    /// `operation`, or `None`.
    fn check_use(&self, binding_json: &str, operation: &str) -> PyResult<Option<String>> {
        let binding = parse_analysis_binding(&parse(binding_json, "analysis binding")?)?;
        let operation = operation_from_name(operation)?;
        Ok(self.inner.check_use(&binding, operation).err().map(|r| refusal_json(&r)))
    }
}

// ---------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------

/// Prepare a corrected functional under a declared measurement model.
///
/// Returns the handle or the structured refusal; a malformed declaration is
/// an input error. `provider_json` is required by the posterior route.
#[pyfunction]
#[pyo3(signature = (model_json, binding_json, data_json, provider_json=None, receipt=None, *, cancel=None))]
fn measurement_model_prepare(
    py: Python<'_>,
    model_json: &str,
    binding_json: &str,
    data_json: &str,
    provider_json: Option<&str>,
    receipt: Option<PyRef<'_, PyProviderVerificationReceipt>>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<PreparedMeasurementHandle>, Option<String>)> {
    let model = parse_model(&parse(model_json, "measurement model")?)?;
    let binding = parse_analysis_binding(&parse(binding_json, "analysis binding")?)?;
    let input = parse_input(&parse(data_json, "measurement data")?)?;
    let provider = match provider_json {
        Some(text) => Some(parse_provider(&parse(text, "provider object")?)?),
        None => None,
    };
    let trust = match &provider {
        Some(declaration) => match trust_of(declaration, receipt.as_deref()) {
            Ok(trust) => Some(trust),
            Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
        },
        None => None,
    };
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        let supplied = match (&provider, &trust) {
            (Some(declaration), Some(trust)) => Some((&declaration.object, trust)),
            _ => None,
        };
        Ok(StudyBuilder::measurement_model(model, binding, input, supplied, &ctx))
    })?;
    Ok(match outcome {
        Ok(inner) => (Some(PreparedMeasurementHandle { inner }), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    })
}

/// Independently consume a measurement-model artifact against the model
/// identity the consumer retained.
#[pyfunction]
fn measurement_model_consume(
    artifact: &[u8],
    expected_model_identity: &str,
) -> (Option<ConsumedMeasurementHandle>, Option<String>) {
    match consume_measurement_model_artifact(artifact, expected_model_identity) {
        Ok(inner) => (Some(ConsumedMeasurementHandle { inner }), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// The identities of a model declaration without preparing anything.
#[pyfunction]
fn measurement_model_identities(model_json: &str) -> PyResult<String> {
    let model = parse_model(&parse(model_json, "measurement model")?)?;
    let identities = model.identities();
    Ok(json!({
        "measurement": identities.measurement.to_hex(),
        "population": identities.population.to_hex(),
            "regime": identities.regime.to_hex(),
            "arm": identities.arm.to_hex(),
        "representation": identities.representation.to_hex(),
        "evidence": identities.evidence.to_hex(),
        "model": identities.model.to_hex(),
    })
    .to_string())
}

// ---------------------------------------------------------------------------
// Transport of a measurement model under a declared invariance premise.
// ---------------------------------------------------------------------------

fn parse_error_parameter(name: &str) -> PyResult<ErrorParameter> {
    [
        ErrorParameter::Sensitivity,
        ErrorParameter::Specificity,
        ErrorParameter::ConfusionMatrix,
        ErrorParameter::ErrorVariance,
    ]
    .into_iter()
    .find(|parameter| parameter.as_str() == name)
    .ok_or_else(|| crate::value_err(format!("unknown error parameter `{name}`")))
}

fn parse_error_structure(value: &Json) -> PyResult<PopulationErrorStructure> {
    let map = object(value, "population error structure")?;
    match text(map, "kind")?.as_str() {
        "invariant" => {
            let parameters = texts(map, "invariant")?
                .iter()
                .map(|name| parse_error_parameter(name))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PopulationErrorStructure::Invariant(InvarianceDeclaration::new(
                &text(map, "source_population")?,
                &text(map, "target_population")?,
                &text(map, "declarer")?,
                &parameters,
            )))
        }
        "differential" => {
            Ok(PopulationErrorStructure::Differential { declarer: text(map, "declarer")? })
        }
        other => Err(crate::value_err(format!("unknown population error structure `{other}`"))),
    }
}

fn parse_models(text_json: &str) -> PyResult<Vec<MeasurementModel>> {
    match parse(text_json, "measurement models")? {
        Json::Array(items) => items.iter().map(parse_model).collect(),
        _ => Err(crate::value_err("measurement models must be a list")),
    }
}

fn parse_optional_structure(text_json: Option<&str>) -> PyResult<Option<PopulationErrorStructure>> {
    match text_json {
        None => Ok(None),
        Some(text_json) => {
            parse_error_structure(&parse(text_json, "population error structure")?).map(Some)
        }
    }
}

fn transport_receipt_json(receipt: &TransportReceipt) -> Json {
    let basis = match &receipt.basis {
        TransportBasis::Established => json!({"kind": "established"}),
        TransportBasis::DeclaredInvariance(declaration) => json!({
            "kind": "declared_invariance",
            "declarer": declaration.declarer,
            "source_population": declaration.source_population,
            "target_population": declaration.target_population,
            "invariant": declaration.invariant.iter().map(|p| p.as_str()).collect::<Vec<_>>(),
        }),
        TransportBasis::DifferentialEstablished { declarer } => {
            json!({"kind": "differential_established", "declarer": declarer})
        }
    };
    json!({
        "route": "measurement_model",
        "identity": receipt.identity.to_hex(),
        "source_model": receipt.source_model.to_hex(),
        "source_population": receipt.source_population,
        "target_population": receipt.target_population,
        "basis": basis,
        "carried": receipt.carried(),
        "support": if receipt.carried() { "extrapolative" } else { "supported" },
        "evidence": evidence_json(&receipt.evidence),
        "assumptions": receipt.assumptions(),
    })
}

/// Transport the measurement model of another population to the analysis population
/// and form the corrected functional on the re-bound model: `(estimate handle, receipt
/// json, refusal json)`.
#[pyfunction]
#[pyo3(signature = (models_json, structure_json, binding_json, data_json, provider_json=None, receipt=None, *, cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "one flat native entry mirroring the Rust facade call"
)]
fn measurement_model_transport(
    py: Python<'_>,
    models_json: &str,
    structure_json: Option<&str>,
    binding_json: &str,
    data_json: &str,
    provider_json: Option<&str>,
    receipt: Option<PyRef<'_, PyProviderVerificationReceipt>>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<MeasurementEstimateHandle>, Option<String>, Option<String>)> {
    let models = parse_models(models_json)?;
    let structure = parse_optional_structure(structure_json)?;
    let binding = parse_analysis_binding(&parse(binding_json, "analysis binding")?)?;
    let input = parse_input(&parse(data_json, "measurement data")?)?;
    let provider = match provider_json {
        Some(text) => Some(parse_provider(&parse(text, "provider object")?)?),
        None => None,
    };
    let trust = match &provider {
        Some(declaration) => match trust_of(declaration, receipt.as_deref()) {
            Ok(trust) => Some(trust),
            Err(refusal) => return Ok((None, None, Some(refusal_json(&refusal)))),
        },
        None => None,
    };
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        let supplied = match (&provider, &trust) {
            (Some(declaration), Some(trust)) => Some((&declaration.object, trust)),
            _ => None,
        };
        Ok(transport_measurement_estimate(
            &models,
            structure.as_ref(),
            binding,
            input,
            supplied,
            &ctx,
        ))
    })?;
    Ok(match outcome {
        Ok(done) => {
            let receipt = transport_receipt_json(done.receipt()).to_string();
            (
                Some(MeasurementEstimateHandle { inner: done.estimate().clone() }),
                Some(receipt),
                None,
            )
        }
        Err(refusal) => (None, None, Some(refusal_json(&refusal))),
    })
}

/// Re-derive the transport from the retained source models and declaration and require
/// the retained identity: `(receipt json, refusal json)`.
#[pyfunction]
fn measurement_model_transport_verify(
    models_json: &str,
    structure_json: Option<&str>,
    binding_json: &str,
    expected_identity: &str,
) -> PyResult<(Option<String>, Option<String>)> {
    let models = parse_models(models_json)?;
    let structure = parse_optional_structure(structure_json)?;
    let binding = parse_analysis_binding(&parse(binding_json, "analysis binding")?)?;
    Ok(
        match verify_measurement_transport(&models, structure.as_ref(), &binding, expected_identity)
        {
            Ok(receipt) => (Some(transport_receipt_json(&receipt).to_string()), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        },
    )
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedMeasurementHandle>()?;
    module.add_class::<MeasurementEstimateHandle>()?;
    module.add_class::<ConsumedMeasurementHandle>()?;
    module.add_function(wrap_pyfunction!(measurement_model_prepare, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_model_consume, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_model_identities, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_model_transport, module)?)?;
    module.add_function(wrap_pyfunction!(measurement_model_transport_verify, module)?)?;
    Ok(())
}
