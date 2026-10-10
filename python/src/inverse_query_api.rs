//! Bounded Python bridge for F7: the generalized finite inverse decision query.
//!
//! Python builds the declarations (a decision contract, a query, forward claims);
//! the typed functional engine, the feasibility fields, the artifact and every
//! refusal are Rust's. A refusal comes back as structured JSON for the Python layer
//! to raise as its own exception type, keeping the engine's own detail (for example
//! `decision_evaluation.joint_law_required` or
//! `decision_evaluation.mean_source_insufficient`).

use antecedent::analysis::inverse_query::{
    AtomEvidence, Comparison, EnumeratedForwardValue, ForwardClaim, ForwardEvidence, IdentifiedSet,
    IntervalRegion, InverseQueryArtifact, InverseQueryError, InverseQueryIdentity,
    InverseQueryWire, MAX_INVERSE_QUERY_ARTIFACT_BYTES, MeanSource, StructuralAtom,
    finite_enumeration_baseline,
};
use antecedent_core::{ExternalRefusal, ScientificQuantity, reason_code};
use antecedent_design::decision_artifact::{contract_from_json_refusal, contract_to_json};
use antecedent_io::IoError;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::json;

use crate::CausalSerializationError;
use crate::distribution_api::PyJointDistributionArtifact;

fn serialization(error: impl std::fmt::Display) -> PyErr {
    CausalSerializationError::new_err(error.to_string())
}

fn check_size(text: &str) -> PyResult<()> {
    if text.len() > MAX_INVERSE_QUERY_ARTIFACT_BYTES {
        return Err(PyValueError::new_err("inverse query declaration is too large"));
    }
    Ok(())
}

fn refusal_json(value: &ExternalRefusal) -> String {
    json!({
        "code": value.code,
        "stage": value.stage,
        "detail": value.detail,
        "offending": value.offending,
        "expected": value.expected,
        "supplied": value.supplied,
        "remedy": value.remedy,
    })
    .to_string()
}

fn plain_refusal(code: &'static str, detail: &str, offending: Option<String>) -> String {
    refusal_json(&ExternalRefusal {
        code,
        stage: "inverse_query",
        detail: detail.to_owned(),
        offending,
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    })
}

fn declaration_refusal(message: &str) -> String {
    plain_refusal(
        reason_code!("invalid_argument"),
        "inverse_query.invalid_declaration",
        Some(message.to_owned()),
    )
}

fn query_refusal(error: &InverseQueryError) -> String {
    refusal_json(&error.to_refusal())
}

// `*code` is required: `plain_refusal` needs `&'static str`, and auto-deref would shorten it.
#[allow(clippy::explicit_auto_deref)]
fn io_refusal(error: &IoError) -> String {
    match error {
        IoError::Refused { code, message } => {
            let (detail, text) = message
                .split_once(": ")
                .unwrap_or(("functional_inverse_query.wrong_contract", message.as_str()));
            plain_refusal(*code, detail, Some(text.to_owned()))
        }
        other => plain_refusal(
            reason_code!("invalid_argument"),
            "functional_inverse_query.invalid_artifact",
            Some(other.to_string()),
        ),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MeansIn {
    coordinates: Vec<ScientificQuantityWire>,
    means: Vec<f64>,
    provider_id: String,
    snapshot_id: String,
    causal_contract_id: String,
    rng_id: String,
}

/// A native law, or a JSON mean grid.
fn claim_from(object: &Bound<'_, PyAny>) -> PyResult<ForwardClaim> {
    if let Ok(mean) =
        object.extract::<PyRef<'_, crate::source_evidence_api::PySourceBackedMeanClaim>>()
    {
        return Ok(ForwardClaim::Means(mean.means.clone()));
    }
    if let Ok(claim) = object.extract::<PyRef<'_, crate::external_api::PyExternalClaimArtifact>>() {
        return antecedent_design::composition_verifiers::mean_source_of(&claim.artifact)
            .map(ForwardClaim::Means)
            .map_err(serialization);
    }
    if let Ok(claim) =
        object.extract::<PyRef<'_, crate::program_claims_api::PyNativeResponseClaim>>()
    {
        use antecedent::analysis::native_claims::NativeDecisionSource;
        use antecedent_design::decision_contract::{SourceRepresentation, SourceRequirement};
        let mode = if claim.claim.has_joint_law() {
            SourceRepresentation::JointDraws
        } else {
            SourceRepresentation::Mean
        };
        let source = claim
            .claim
            .decision_source(&SourceRequirement {
                any_of: vec![mode],
                sampled_needs_error_receipt: false,
            })
            .map_err(|cause| PyValueError::new_err(format!("{}: {}", cause.code, cause.detail)))?;
        return Ok(match source.source {
            NativeDecisionSource::Mean(source) => ForwardClaim::Means(source),
            NativeDecisionSource::JointLaw(law) => ForwardClaim::Law(law),
        });
    }
    if let Ok(law) = object.extract::<PyRef<'_, PyJointDistributionArtifact>>() {
        return Ok(ForwardClaim::Law(Box::new(law.inner().clone())));
    }
    if let Ok(text) = object.extract::<String>() {
        check_size(&text)?;
        let means: MeansIn = serde_json::from_str(&text)
            .map_err(|e| PyValueError::new_err(format!("invalid mean claim: {e}")))?;
        let coordinates = means
            .coordinates
            .into_iter()
            .map(|wire| {
                ScientificQuantity::try_from(wire)
                    .map_err(|e| PyValueError::new_err(String::from(e)))
            })
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(ForwardClaim::Means(MeanSource {
            coordinates,
            means: means.means,
            provider_id: means.provider_id,
            snapshot_id: means.snapshot_id,
            causal_contract_id: means.causal_contract_id,
            rng_id: means.rng_id,
        }));
    }
    Err(PyValueError::new_err("a forward claim is a native law artifact or a mean grid"))
}

/// One scenario or set member: `(id, probability, kind, payload)` where `kind` is
/// `evaluated` (payload a native law), `unidentified` (no payload) or `unevaluated`
/// (payload the reason).
fn atom_from(object: &Bound<'_, PyAny>) -> PyResult<StructuralAtom> {
    let (id, probability, kind, payload): (String, Option<f64>, String, Option<Bound<'_, PyAny>>) =
        object.extract()?;
    let evidence = match (kind.as_str(), payload) {
        ("evaluated", Some(law)) => match claim_from(&law)? {
            ForwardClaim::Law(artifact) => AtomEvidence::Evaluated(artifact),
            ForwardClaim::Means(_) => {
                return Err(PyValueError::new_err("a scenario or set member needs a law"));
            }
        },
        ("unidentified", _) => AtomEvidence::Unidentified,
        ("unevaluated", Some(reason)) => AtomEvidence::Unevaluated(reason.extract::<String>()?),
        _ => return Err(PyValueError::new_err("unknown scenario state")),
    };
    Ok(StructuralAtom { id, probability, evidence })
}

fn atoms_from(objects: &[Bound<'_, PyAny>]) -> PyResult<Vec<StructuralAtom>> {
    objects.iter().map(atom_from).collect()
}

#[pyclass(name = "InverseQueryArtifact", skip_from_py_object)]
pub(crate) struct PyInverseQueryArtifact {
    artifact: InverseQueryArtifact,
    source_claims: Vec<crate::program_claims_api::PyNativeResponseClaim>,
    external_sources: Vec<antecedent_design::source_evidence::SourceEvidence>,
}

impl PyInverseQueryArtifact {
    fn with_sources(&self) -> PyResult<InverseQueryArtifact> {
        if self.source_claims.is_empty() && self.external_sources.is_empty() {
            return Ok(self.artifact.clone());
        }
        if self.source_claims.len() + self.external_sources.len() > 8 {
            return Err(crate::with_reason_code(
                crate::value_err(
                    "functional_inverse_query.bounds_exceeded: too many original source artifacts",
                ),
                antecedent_core::reason_code!("inverse_functional_unsupported"),
            ));
        }
        let contract = contract_to_json(&self.artifact.query().contract).map_err(serialization)?;
        let mut sources = self
            .source_claims
            .iter()
            .map(|claim| {
                crate::source_evidence_api::PySourceEvidence::from_claim(claim)?
                    .inner
                    .project(&contract, &self.artifact.query().grid_order)
                    .map_err(|cause| crate::recalc_api::invalid(cause.detail, cause.to_string()))
            })
            .collect::<PyResult<Vec<_>>>()?;
        sources.extend(
            self.external_sources
                .iter()
                .map(|source| source.project(&contract, &self.artifact.query().grid_order))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|cause| crate::recalc_api::invalid(cause.detail, cause.to_string()))?,
        );
        self.artifact.clone().with_source_evidence(sources).map_err(serialization)
    }
}

type Built = (Option<PyInverseQueryArtifact>, Option<String>);
#[pymethods]
impl PyInverseQueryArtifact {
    #[getter]
    fn source_evidence(&self) -> PyResult<Vec<crate::source_evidence_api::PySourceEvidence>> {
        Ok(self
            .with_sources()?
            .source_evidence()
            .iter()
            .cloned()
            .map(|inner| crate::source_evidence_api::PySourceEvidence { inner, resolution: None })
            .collect())
    }
    /// Evaluate a query on forward evidence through the shared functional engine and
    /// seal it. Returns the artifact or a structured refusal.
    #[staticmethod]
    #[pyo3(signature = (contract_json, query_json, point=None, interval_region=None, identified_set=None, scenarios=None))]
    fn build(
        contract_json: &str,
        query_json: &str,
        point: Option<Bound<'_, PyAny>>,
        interval_region: Option<(Bound<'_, PyAny>, Bound<'_, PyAny>, bool)>,
        identified_set: Option<(Vec<Bound<'_, PyAny>>, bool)>,
        scenarios: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Built> {
        check_size(contract_json)?;
        check_size(query_json)?;
        let contract = match contract_from_json_refusal(contract_json) {
            Ok(contract) => contract,
            Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
        };
        let wire: InverseQueryWire = match serde_json::from_str(query_json) {
            Ok(wire) => wire,
            Err(error) => return Ok((None, Some(declaration_refusal(&error.to_string())))),
        };
        let mut source_claims = Vec::new();
        let mut external_sources = Vec::new();
        let mut retain = |value: &Bound<'_, PyAny>| -> PyResult<()> {
            if let Ok(claim) =
                value.extract::<PyRef<'_, crate::external_api::PyExternalClaimArtifact>>()
            {
                let evidence =
                    crate::source_evidence_api::PySourceEvidence::from_external(&claim)?.inner;
                if !external_sources.iter().any(
                    |held: &antecedent_design::source_evidence::SourceEvidence| {
                        held.original_bytes() == evidence.original_bytes()
                    },
                ) {
                    external_sources.push(evidence);
                }
            }
            let claim = value
                .extract::<PyRef<'_, crate::program_claims_api::PyNativeResponseClaim>>()
                .ok()
                .map(|claim| (*claim).clone())
                .or_else(|| {
                    value
                        .extract::<PyRef<'_, crate::source_evidence_api::PySourceBackedMeanClaim>>()
                        .ok()
                        .map(|mean| mean.original.clone())
                })
                .or_else(|| {
                    value
                        .extract::<PyRef<'_, PyJointDistributionArtifact>>()
                        .ok()
                        .and_then(|law| law.source_claim.clone())
                });
            if let Some(claim) = claim {
                if !source_claims.iter().any(
                    |held: &crate::program_claims_api::PyNativeResponseClaim| {
                        std::sync::Arc::ptr_eq(&held.authority, &claim.authority)
                    },
                ) {
                    source_claims.push(claim);
                }
            }
            Ok(())
        };
        if let Some(point) = &point {
            retain(point)?;
        }
        if let Some((lower, upper, _)) = &interval_region {
            retain(lower)?;
            retain(upper)?;
        }
        for atom in identified_set
            .as_ref()
            .into_iter()
            .flat_map(|(atoms, _)| atoms)
            .chain(scenarios.as_ref().into_iter().flatten())
        {
            let (_, _, kind, payload): (String, Option<f64>, String, Option<Bound<'_, PyAny>>) =
                atom.extract()?;
            if kind == "evaluated" {
                if let Some(payload) = payload {
                    retain(&payload)?;
                }
            }
        }
        let evidence = ForwardEvidence {
            point: point.as_ref().map(claim_from).transpose()?,
            interval_region: interval_region
                .map(|(lower, upper, bound)| {
                    Ok::<_, PyErr>(IntervalRegion {
                        lower: claim_from(&lower)?,
                        upper: claim_from(&upper)?,
                        endpoints_bound_functional: bound,
                    })
                })
                .transpose()?,
            identified_set: identified_set
                .map(|(members, exhaustive)| {
                    Ok::<_, PyErr>(IdentifiedSet { members: atoms_from(&members)?, exhaustive })
                })
                .transpose()?,
            scenarios: scenarios.map(|atoms| atoms_from(&atoms)).transpose()?,
        };
        match InverseQueryArtifact::new(wire.into_query(contract), evidence) {
            Ok(artifact) => Ok((Some(Self { artifact, source_claims, external_sources }), None)),
            Err(error) => Ok((None, Some(query_refusal(&error)))),
        }
    }

    /// Consume an artifact by re-evaluation; `expected_identity_json` is an identity
    /// the consumer retained independently of the bytes.
    #[staticmethod]
    #[pyo3(signature = (data, expected_identity_json=None))]
    fn consume(data: &[u8], expected_identity_json: Option<&str>) -> PyResult<Built> {
        let expected: Option<InverseQueryIdentity> = match expected_identity_json {
            Some(text) => {
                check_size(text)?;
                match serde_json::from_str(text) {
                    Ok(identity) => Some(identity),
                    Err(error) => {
                        return Ok((None, Some(declaration_refusal(&error.to_string()))));
                    }
                }
            }
            None => None,
        };
        match InverseQueryArtifact::from_bytes(data, expected.as_ref()) {
            Ok(artifact) => Ok((
                Some(Self { artifact, source_claims: Vec::new(), external_sources: Vec::new() }),
                None,
            )),
            Err(error) => Ok((None, Some(io_refusal(&error)))),
        }
    }

    /// The per-action status table with the distinct feasibility fields, the
    /// selection and the exhaustive flag, as JSON.
    #[getter]
    fn result_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.artifact.result_wire()).map_err(serialization)
    }

    /// The identity digests (premises and data kept apart), as JSON.
    #[getter]
    fn identity_json(&self) -> PyResult<String> {
        serde_json::to_string(self.with_sources()?.identity()).map_err(serialization)
    }

    /// The declared query (grid, scope, constraints, selection, tolerance), as JSON.
    #[getter]
    fn query_json(&self) -> PyResult<String> {
        serde_json::to_string(&InverseQueryWire::from_query(self.artifact.query()))
            .map_err(serialization)
    }

    /// The decision contract declaring the actions, as normalized JSON.
    #[getter]
    fn contract_json(&self) -> PyResult<String> {
        contract_to_json(&self.artifact.query().contract).map_err(serialization)
    }

    /// Serialize through the bounded sectioned container.
    fn export<'py>(&self, py: Python<'py>, artifact_id: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.with_sources()?.to_bytes(artifact_id).map_err(serialization)?;
        Ok(PyBytes::new(py, &bytes))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselinePointIn {
    id: String,
    value: Option<f64>,
    supported: bool,
}

/// The 2.2 finite-enumeration baseline: classify each enumerated action by its
/// forward mean against `target`. Returns `[[id, status], ...]` JSON or a refusal.
#[pyfunction]
fn inverse_query_baseline(
    points_json: &str,
    target: f64,
    comparison: &str,
    tolerance: f64,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(points_json)?;
    let comparison = match comparison {
        "at_least" => Comparison::AtLeast,
        "at_most" => Comparison::AtMost,
        other => return Err(PyValueError::new_err(format!("unknown comparison `{other}`"))),
    };
    let points: Vec<BaselinePointIn> = match serde_json::from_str(points_json) {
        Ok(points) => points,
        Err(error) => return Ok((None, Some(declaration_refusal(&error.to_string())))),
    };
    let points: Vec<EnumeratedForwardValue> = points
        .into_iter()
        .map(|p| EnumeratedForwardValue { id: p.id, value: p.value, supported: p.supported })
        .collect();
    match finite_enumeration_baseline(&points, target, comparison, tolerance) {
        Ok(classes) => {
            let rows: Vec<_> =
                classes.into_iter().map(|(id, status)| json!([id, status.as_str()])).collect();
            Ok((Some(serde_json::Value::Array(rows).to_string()), None))
        }
        Err(error) => Ok((None, Some(query_refusal(&error)))),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInverseQueryArtifact>()?;
    m.add_function(wrap_pyfunction!(inverse_query_baseline, m)?)
}
