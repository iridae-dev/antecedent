//! Bounded Python bridge for program-bound external claims and native response claims.
//!
//! Python builds the declarations; every identity, check and refusal rule is
//! Rust's. A fallible call returns `(value, refusal_json)` so the Python layer
//! can raise the refusal as its own exception type.
//!
//! Native trust is issued only from opaque checked execution state, never caller projections.

use std::fmt::Display;

use antecedent::analysis::native_claims::{
    NativeDecisionInput, NativeDecisionSource, NativeResponseClaim, NativeResponseContext,
};
use antecedent_core::{
    ExternalProgramClaim, ExternalRefusal, ProgramBinding, ResponseUncertainty, ScientificQuantity,
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
pub(crate) struct ClaimWire {
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

pub(crate) fn parse_binding(json: &str) -> PyResult<ProgramBinding> {
    serde_json::from_str::<BindingWire>(json).map(ProgramBinding::from).map_err(malformed)
}

// The refusal is the cold path of a once-per-claim check; boxing would not pay.
#[allow(clippy::result_large_err)]
pub(crate) fn claim_from(wire: ClaimWire) -> Result<ExternalProgramClaim, ExternalRefusal> {
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
            "joint_law": { "shape": artifact.shape() }
        }),
    };
    if let (Some(target), serde_json::Value::Object(extra)) = (body.as_object_mut(), source) {
        target.extend(extra);
    }
    Ok(body)
}

/// A native response with its scientific coordinates, support and provenance.
#[pyclass(name = "NativeResponseClaim", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyNativeResponseClaim {
    pub(crate) claim: NativeResponseClaim,
    pub(crate) authority: std::sync::Arc<crate::response_api::NativeResponseAuthority>,
}

#[pymethods]
impl PyNativeResponseClaim {
    #[getter]
    fn source_evidence(&self) -> PyResult<crate::source_evidence_api::PySourceEvidence> {
        crate::source_evidence_api::PySourceEvidence::from_claim(self)
    }
    #[getter]
    fn execution_program_id(&self) -> Option<String> {
        self.authority
            .result
            .executed_contract
            .as_ref()
            .and_then(|contract| contract.identities.program)
            .map(|id| id.to_hex())
    }
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

    fn joint_artifact(
        &self,
        contract_json: &str,
    ) -> PyResult<(Option<crate::distribution_api::PyJointDistributionArtifact>, Option<String>)>
    {
        check_size(&[contract_json])?;
        let contract = match contract_from_json_refusal(contract_json) {
            Ok(contract) => contract,
            Err(refusal) => return Ok((None, Some(refusal_json(refusal)?))),
        };
        let requirement = match contract.source_requirement() {
            Ok(Some(requirement)) => requirement,
            Ok(None) => {
                return Ok((
                    None,
                    Some(refusal_json(declared_refusal(
                        "evaluate",
                        "native_claims.no_source_requirement",
                        None,
                        None,
                        None,
                        "declare a source requirement",
                    ))?),
                ));
            }
            Err(error) => return Ok((None, Some(refusal_json(error.to_refusal())?))),
        };
        match self.claim.decision_source(&requirement) {
            Ok(input) => match input.source {
                NativeDecisionSource::JointLaw(artifact) => Ok((
                    Some(crate::distribution_api::PyJointDistributionArtifact {
                        artifact: *artifact,
                        source_claim: Some(self.clone()),
                    }),
                    None,
                )),
                NativeDecisionSource::Mean(_) => Ok((None, None)),
            },
            Err(refusal) => Ok((None, Some(refusal_json(refusal)?))),
        }
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

fn native_contract_id(graph: &str, status: &str, premises: Option<&str>) -> PyResult<String> {
    antecedent_io::external_binding_wire::native_contract_identity(graph, status, premises)
        .map_err(|error| malformed(format!("{error:?}")))
}

/// Build a claim from producer-issued native state and verify its public projection.
#[pyfunction]
#[pyo3(signature=(raw,projection_json,binding_json,snapshot_id=None,rng_id=None,calibration=None,premises_json=None))]
fn native_response_claim(
    raw: Option<PyRef<'_, crate::response_api::ResponseAnalysisResult>>,
    projection_json: &str,
    binding_json: &str,
    snapshot_id: Option<&str>,
    rng_id: Option<&str>,
    calibration: Option<&str>,
    premises_json: Option<&str>,
) -> PyResult<(Option<PyNativeResponseClaim>, Option<String>)> {
    check_size(&[projection_json, binding_json])?;
    let supplied: serde_json::Value = serde_json::from_str(projection_json).map_err(malformed)?;
    let program = parse_binding(binding_json)?;
    let refusal = |detail: &str| {
        declared_refusal(
            "bind",
            detail,
            None,
            None,
            None,
            "use the unchanged response and provenance issued by an actual native execution",
        )
    };
    let Some(raw) = raw else {
        return Ok((None, Some(refusal_json(refusal("native_claims.native_state_unavailable"))?)));
    };
    let Some(authority) = &raw.authority else {
        return Ok((None, Some(refusal_json(refusal("native_claims.native_state_unavailable"))?)));
    };
    if supplied != raw.claim_projection() {
        return Ok((None, Some(refusal_json(refusal("native_claims.projection_mismatch"))?)));
    }
    let Some(executed) = &authority.result.executed_contract else {
        return Ok((None, Some(refusal_json(refusal("native_claims.native_state_unavailable"))?)));
    };
    let actual_snapshot = executed.identities.data_snapshot.to_hex();
    let actual_rng = format!("native:seed:{}", authority.context.rng.master_seed());
    let actual_graph = format!("graph:{}", executed.identities.identification);
    if snapshot_id.is_some_and(|id| id != actual_snapshot)
        || rng_id.is_some_and(|id| id != actual_rng)
    {
        return Ok((None, Some(refusal_json(refusal("native_claims.provenance_mismatch"))?)));
    }
    if program.graph_id != actual_graph {
        return Ok((None, Some(refusal_json(refusal("native_claims.program_mismatch"))?)));
    }
    if program.contract_id
        != native_contract_id(
            &actual_graph,
            authority.response.identification_status.as_str(),
            premises_json,
        )?
    {
        return Ok((None, Some(refusal_json(refusal("native_claims.contract_mismatch"))?)));
    }
    let actual_calibration = if matches!(authority.response.uncertainty, ResponseUncertainty::None)
    {
        "point_only"
    } else {
        "unmeasured"
    };
    if calibration.is_some_and(|value| value != actual_calibration) {
        return Ok((None, Some(refusal_json(refusal("native_claims.calibration_not_licensed"))?)));
    }
    let context = NativeResponseContext {
        program,
        snapshot_id: actual_snapshot,
        rng_id: actual_rng,
        calibration: calibration_from(actual_calibration)
            .map_err(|_| malformed("invalid native calibration"))?,
    };
    match NativeResponseClaim::from_response(&authority.response, &authority.schema, &context) {
        Ok(claim) => Ok((
            Some(PyNativeResponseClaim { claim, authority: std::sync::Arc::clone(authority) }),
            None,
        )),
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
