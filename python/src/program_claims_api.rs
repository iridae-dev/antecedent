//! Bounded Python bridge for program-bound external claims and native response claims.
//!
//! Python builds the declarations; every identity, check and refusal rule is
//! Rust's. A fallible call returns `(value, refusal_json)` so the Python layer
//! can raise the refusal as its own exception type.
//!
//! A Python response view keeps no private native payload, so a native claim is
//! built from the view's public projection (treatment, outcome, grid, means,
//! support labels and identification status). That projection retains no draws,
//! so a claim built here supplies a mean source and never a joint law.

use std::fmt::Display;
use std::sync::Arc;

use antecedent::analysis::native_claims::{
    NativeDecisionInput, NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use antecedent_core::{
    AssumptionSet, CausalResponse, CausalSchema, CausalSchemaBuilder, ContinuousDomain,
    ExternalProgramClaim, ExternalRefusal, GridSpec, IdentificationStatus, ProgramBinding,
    ResponseFunctional, ResponseIdentification, ResponseUncertainty, ResponseValue,
    ScientificQuantity, SupportRegion, SupportReport, SupportStatus,
    check_external_against_program,
};
use antecedent_design::decision_artifact::contract_from_json_refusal;
use antecedent_design::decision_contract::SourceRepresentation;
use antecedent_io::distribution_artifact::DistributionCalibration;
use antecedent_io::external_binding_wire::{
    ContractWire, MAX_BINDING_WIRE_BYTES, RefusalWire, ResponseWire, bind_response_to_program,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;

use crate::external_api::PyExternalClaimArtifact;

fn malformed(error: impl Display) -> PyErr {
    PyValueError::new_err(format!("invalid program declaration: {error}"))
}

fn check_size(parts: &[&str]) -> PyResult<()> {
    if parts.iter().any(|part| part.len() > MAX_BINDING_WIRE_BYTES) {
        return Err(PyValueError::new_err("program declaration is too large"));
    }
    Ok(())
}

fn refusal_json(refusal: ExternalRefusal) -> PyResult<String> {
    serde_json::to_string(&RefusalWire::from(refusal)).map_err(malformed)
}

fn declared_refusal(
    stage: &'static str,
    detail: &str,
    offending: Option<String>,
    expected: Option<String>,
    supplied: Option<String>,
    remedy: &'static str,
) -> ExternalRefusal {
    ExternalRefusal {
        code: antecedent_core::reason_code!("invalid_argument"),
        stage,
        detail: detail.to_owned(),
        offending,
        expected,
        supplied,
        capability: None,
        remedy: Some(remedy),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingWire {
    graph_id: String,
    contract_id: String,
    treatment_id: String,
    outcome_id: String,
    population_id: String,
    intervention_kind: String,
    horizon: u32,
    dose_grid: Vec<f64>,
    dose_units: String,
    outcome_units: String,
    functional_id: String,
    transform_id: String,
}

impl From<BindingWire> for ProgramBinding {
    fn from(wire: BindingWire) -> Self {
        Self {
            graph_id: wire.graph_id,
            contract_id: wire.contract_id,
            treatment_id: wire.treatment_id,
            outcome_id: wire.outcome_id,
            population_id: wire.population_id,
            intervention_kind: wire.intervention_kind,
            horizon: wire.horizon,
            dose_grid: wire.dose_grid,
            dose_units: wire.dose_units,
            outcome_units: wire.outcome_units,
            functional_id: wire.functional_id,
            transform_id: wire.transform_id,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClaimWire {
    contract_id: String,
    graph_id: String,
    declared_identity: String,
    treatment_id: String,
    outcome_id: String,
    population_id: String,
    doses: Vec<f64>,
    dose_units: String,
    quantities: Vec<ScientificQuantityWire>,
}

fn parse_binding(json: &str) -> PyResult<ProgramBinding> {
    serde_json::from_str::<BindingWire>(json).map(ProgramBinding::from).map_err(malformed)
}

// The refusal is the cold path of a once-per-claim check; boxing would not pay.
#[allow(clippy::result_large_err)]
fn claim_from(wire: ClaimWire) -> Result<ExternalProgramClaim, ExternalRefusal> {
    let quantities = wire
        .quantities
        .into_iter()
        .map(|quantity| {
            ScientificQuantity::try_from(quantity).map_err(|why| {
                declared_refusal(
                    "bind",
                    "program_binding.malformed_quantity",
                    Some("quantities".to_owned()),
                    None,
                    Some(why.to_owned()),
                    "declare each coordinate as a valid scientific quantity",
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ExternalProgramClaim {
        contract_id: wire.contract_id,
        graph_id: wire.graph_id,
        declared_identity: wire.declared_identity,
        treatment_id: wire.treatment_id,
        outcome_id: wire.outcome_id,
        population_id: wire.population_id,
        doses: wire.doses,
        dose_units: wire.dose_units,
        quantities,
    })
}

/// The durable identity of a program binding, or the refusal of an invalid one.
#[pyfunction]
fn program_binding_identity(binding_json: &str) -> PyResult<(Option<String>, Option<String>)> {
    check_size(&[binding_json])?;
    let binding = parse_binding(binding_json)?;
    match binding.validate() {
        Ok(()) => Ok((Some(binding.identity()), None)),
        Err(refusal) => Ok((None, Some(refusal_json(refusal)?))),
    }
}

/// Check a declared claim against the program; returns `{identity, coordinates}`.
#[pyfunction]
fn check_external_program(
    binding_json: &str,
    claim_json: &str,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(&[binding_json, claim_json])?;
    let binding = parse_binding(binding_json)?;
    let wire: ClaimWire = serde_json::from_str(claim_json).map_err(malformed)?;
    match claim_from(wire).and_then(|claim| check_external_against_program(&binding, &claim)) {
        Ok(check) => Ok((
            Some(
                serde_json::json!({
                    "identity": check.identity,
                    "coordinates": check.coordinates,
                })
                .to_string(),
            ),
            None,
        )),
        Err(refusal) => Ok((None, Some(refusal_json(refusal)?))),
    }
}

/// Bind a response under its contract and to the program it answers.
#[pyfunction]
fn bind_external_to_program(
    binding_json: &str,
    claim_json: &str,
    contract_json: &str,
    response_json: &str,
    causal_contract_id: &str,
) -> PyResult<(Option<PyExternalClaimArtifact>, Option<String>)> {
    check_size(&[binding_json, claim_json, contract_json, response_json])?;
    let binding = parse_binding(binding_json)?;
    let wire: ClaimWire = serde_json::from_str(claim_json).map_err(malformed)?;
    let contract: ContractWire = serde_json::from_str(contract_json).map_err(malformed)?;
    let response: ResponseWire = serde_json::from_str(response_json).map_err(malformed)?;
    let declared = match claim_from(wire) {
        Ok(declared) => declared,
        Err(refusal) => return Ok((None, Some(refusal_json(refusal)?))),
    };
    match bind_response_to_program(&contract, &response, causal_contract_id, &binding, &declared) {
        Ok(artifact) => Ok((Some(PyExternalClaimArtifact { artifact }), None)),
        Err(refusal) => Ok((None, Some(serde_json::to_string(&refusal).map_err(malformed)?))),
    }
}

/// The public projection of a Python response view that a native claim is built from.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectionWire {
    treatment: String,
    outcome: String,
    grid: Vec<f64>,
    means: Vec<f64>,
    identification: String,
    has_envelope: bool,
    support_status: String,
    point_status: Option<Vec<String>>,
    provenance_id: String,
}

fn projection_refusal(detail: &str, supplied: &str) -> ExternalRefusal {
    declared_refusal(
        "bind",
        detail,
        None,
        None,
        Some(supplied.to_owned()),
        "project a response result that this library produced",
    )
}

// The refusal is the cold path of a once-per-claim check; boxing would not pay.
#[allow(clippy::result_large_err)]
fn support_label(name: &str) -> Result<SupportStatus, ExternalRefusal> {
    SupportStatus::from_name(name)
        .ok_or_else(|| projection_refusal("native_claims.unknown_support_label", name))
}

// The refusal is the cold path of a once-per-claim check; boxing would not pay.
#[allow(clippy::result_large_err)]
fn response_from(
    projection: ProjectionWire,
) -> Result<(CausalResponse, CausalSchema), ExternalRefusal> {
    let schema = CausalSchemaBuilder::new()
        .continuous(projection.treatment.as_str())
        .treatment()
        .continuous(projection.outcome.as_str())
        .outcome()
        .build()
        .map_err(|error| {
            projection_refusal("native_claims.invalid_variables", &error.to_string())
        })?;
    let id_of = |name: &str| {
        schema.id_of(name).map_err(|error| {
            projection_refusal("native_claims.invalid_variables", &error.to_string())
        })
    };
    let (treatment, outcome) = (id_of(&projection.treatment)?, id_of(&projection.outcome)?);
    let status = IdentificationStatus::from_name(&projection.identification).ok_or_else(|| {
        projection_refusal("native_claims.unknown_identification", &projection.identification)
    })?;
    let summary = support_label(&projection.support_status)?;
    let labels = projection
        .point_status
        .as_deref()
        .map(|names| names.iter().map(|name| support_label(name)).collect::<Result<Vec<_>, _>>())
        .transpose()?;
    let low = projection.grid.first().copied().unwrap_or(0.0);
    let high = projection.grid.last().copied().unwrap_or(0.0);
    let value = ResponseValue::Surface {
        grid: Arc::from(projection.grid.clone()),
        dimension: 1,
        mean: Arc::from(projection.means),
    };
    let estimate = match status {
        _ if projection.has_envelope => ResponseIdentification::PartiallyIdentified(value),
        IdentificationStatus::PartiallyIdentified => {
            ResponseIdentification::PartiallyIdentified(value)
        }
        IdentificationStatus::GraphDependent => {
            ResponseIdentification::GraphDependent(vec![(0, value)])
        }
        IdentificationStatus::NotIdentified => {
            ResponseIdentification::Unidentified { certificate: Arc::from("not_identified") }
        }
        _ => ResponseIdentification::PointIdentified(value),
    };
    let response = CausalResponse {
        estimand: ResponseFunctional::MeanCurve {
            outcome,
            treatment: ContinuousDomain::new(
                treatment,
                GridSpec::Values(Arc::from(projection.grid)),
            ),
        },
        identification_status: status,
        estimate,
        uncertainty: ResponseUncertainty::None,
        support: SupportReport {
            status: summary,
            query_region: SupportRegion {
                minima: Arc::from(vec![low]),
                maxima: Arc::from(vec![high]),
            },
            diagnostics: vec![],
            warnings: vec![],
            point_status: labels.map(Arc::from),
        },
        assumptions: AssumptionSet::new(),
        provenance_id: Arc::from(projection.provenance_id),
        horizon_identification: None,
        interaction_structurally_zero: false,
    };
    Ok((response, schema))
}

// The refusal is the cold path of a once-per-claim check; boxing would not pay.
#[allow(clippy::result_large_err)]
fn calibration_from(name: &str) -> Result<DistributionCalibration, ExternalRefusal> {
    match name {
        "unmeasured" => Ok(DistributionCalibration::Unmeasured),
        "point_only" => Ok(DistributionCalibration::PointOnly),
        other => Err(declared_refusal(
            "bind",
            "native_claims.calibration_not_licensed",
            None,
            Some("unmeasured,point_only".to_owned()),
            Some(other.to_owned()),
            "a calibration status comes from a measured calibration record, not from the caller",
        )),
    }
}

fn snake_case<T: serde::Serialize>(value: &T) -> PyResult<String> {
    match serde_json::to_value(value).map_err(malformed)? {
        serde_json::Value::String(name) => Ok(name),
        other => Err(malformed(format!("expected a name, found {other}"))),
    }
}

fn quantity_wires(coordinates: &[ScientificQuantity]) -> Vec<ScientificQuantityWire> {
    coordinates.iter().map(ScientificQuantityWire::from).collect()
}

const fn representation_name(representation: SourceRepresentation) -> &'static str {
    match representation {
        SourceRepresentation::Mean => "mean",
        SourceRepresentation::MeanAndCovariance => "mean_and_covariance",
        SourceRepresentation::Cdf => "cdf",
        SourceRepresentation::QuantileFunction => "quantile_function",
        SourceRepresentation::MarginalDraws => "marginal_draws",
        SourceRepresentation::JointDraws => "joint_draws",
    }
}

fn describe(input: &NativeDecisionInput) -> PyResult<serde_json::Value> {
    let withheld: Vec<serde_json::Value> = input
        .withheld
        .iter()
        .map(|item| {
            serde_json::json!({
                "index": item.index,
                "coordinate": ScientificQuantityWire::from(&item.coordinate),
                "status": item.status.as_str(),
            })
        })
        .collect();
    let labels: Vec<&str> = input.point_status.iter().map(|status| status.as_str()).collect();
    let mut body = serde_json::json!({
        "representation": representation_name(input.representation),
        "coordinates": quantity_wires(&input.coordinates),
        "point_status": labels,
        "withheld": withheld,
        "trust": snake_case(&input.trust)?,
        "calibration": snake_case(&input.calibration)?,
        "program_identity": input.program_identity,
    });
    let source = match &input.source {
        NativeDecisionSource::Mean(source) => serde_json::json!({
            "mean_source": {
                "means": source.means,
                "provider_id": source.provider_id,
                "snapshot_id": source.snapshot_id,
                "causal_contract_id": source.causal_contract_id,
                "rng_id": source.rng_id,
            }
        }),
        NativeDecisionSource::JointLaw(artifact) => serde_json::json!({
            "joint_law": { "shape": artifact.shape(), "draws": artifact.draws() }
        }),
    };
    if let (Some(target), serde_json::Value::Object(extra)) = (body.as_object_mut(), source) {
        target.extend(extra);
    }
    Ok(body)
}

/// A native response with its scientific coordinates, support and provenance.
#[pyclass(name = "NativeResponseClaim", skip_from_py_object)]
pub(crate) struct PyNativeResponseClaim {
    claim: NativeResponseClaim,
}

#[pymethods]
impl PyNativeResponseClaim {
    #[getter]
    fn coordinates_json(&self) -> PyResult<String> {
        serde_json::to_string(&quantity_wires(self.claim.coordinates())).map_err(malformed)
    }

    #[getter]
    fn means(&self) -> Vec<f64> {
        self.claim.means().to_vec()
    }

    #[getter]
    fn point_status(&self) -> Vec<String> {
        self.claim.point_status().iter().map(|status| status.as_str().to_owned()).collect()
    }

    #[getter]
    fn support_status(&self) -> String {
        self.claim.support_status().as_str().to_owned()
    }

    #[getter]
    fn trust(&self) -> PyResult<String> {
        snake_case(&self.claim.trust())
    }

    #[getter]
    fn calibration(&self) -> PyResult<String> {
        snake_case(&self.claim.calibration())
    }

    #[getter]
    fn program_identity(&self) -> String {
        self.claim.program_identity().to_owned()
    }

    #[getter]
    fn provenance_id(&self) -> String {
        self.claim.provenance_id().to_owned()
    }

    #[getter]
    fn has_joint_law(&self) -> bool {
        self.claim.has_joint_law()
    }

    /// The decision source a contract needs, or the refusal when the claim does
    /// not supply the law it asks for (a mean never becomes an outcome law).
    fn decision_source(&self, contract_json: &str) -> PyResult<(Option<String>, Option<String>)> {
        check_size(&[contract_json])?;
        let contract = match contract_from_json_refusal(contract_json) {
            Ok(contract) => contract,
            Err(refusal) => return Ok((None, Some(refusal_json(refusal)?))),
        };
        let requirement = match contract.source_requirement() {
            Ok(Some(requirement)) => requirement,
            Ok(None) => {
                let refusal = declared_refusal(
                    "evaluate",
                    "native_claims.no_source_requirement",
                    None,
                    None,
                    None,
                    "declare a criterion that names the source it reads",
                );
                return Ok((None, Some(refusal_json(refusal)?)));
            }
            Err(error) => return Ok((None, Some(refusal_json(error.to_refusal())?))),
        };
        match self.claim.decision_source(&requirement) {
            Ok(input) => Ok((Some(describe(&input)?.to_string()), None)),
            Err(refusal) => Ok((None, Some(refusal_json(refusal)?))),
        }
    }
}

/// Build a native response claim from a response view's projection and a program.
#[pyfunction]
fn native_response_claim(
    projection_json: &str,
    binding_json: &str,
    snapshot_id: &str,
    rng_id: &str,
    calibration: &str,
) -> PyResult<(Option<PyNativeResponseClaim>, Option<String>)> {
    check_size(&[projection_json, binding_json])?;
    let projection: ProjectionWire = serde_json::from_str(projection_json).map_err(malformed)?;
    let program = parse_binding(binding_json)?;
    let built = calibration_from(calibration).and_then(|calibration| {
        let (response, schema) = response_from(projection)?;
        let context = NativeResponseContext {
            program,
            snapshot_id: snapshot_id.to_owned(),
            rng_id: rng_id.to_owned(),
            calibration,
        };
        NativeResponseClaim::from_response(&response, &schema, &context)
    });
    match built {
        Ok(claim) => Ok((Some(PyNativeResponseClaim { claim }), None)),
        Err(refusal) => Ok((None, Some(refusal_json(refusal)?))),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNativeResponseClaim>()?;
    m.add_function(wrap_pyfunction!(program_binding_identity, m)?)?;
    m.add_function(wrap_pyfunction!(check_external_program, m)?)?;
    m.add_function(wrap_pyfunction!(bind_external_to_program, m)?)?;
    m.add_function(wrap_pyfunction!(native_response_claim, m)?)
}
