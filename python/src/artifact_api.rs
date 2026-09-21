//! Python bridge for standalone format-0.5 causal wire artifacts.

use antecedent_io::executed_functional_labels;
use antecedent_io::{
    CausalPayloadWire, CausalQueryWire, CausalResponseWire, ExternalEstimateAttach,
    InterferenceEstimateWire, TransportEffectEstimateWire, TransportIdentificationWire,
    decode_analysis_result_artifact, decode_causal_payload_artifact,
    encode_causal_payload_artifact, encode_external_estimate_claim, parse_digest_hex,
    payload_digest,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::CausalSerializationError;

#[pyclass(skip_from_py_object)]
struct DecodedCausalArtifact {
    #[pyo3(get)]
    artifact_id: String,
    #[pyo3(get)]
    format_major: u16,
    #[pyo3(get)]
    format_minor: u16,
    #[pyo3(get)]
    payload_kind: String,
    #[pyo3(get)]
    variable_names: Vec<String>,
    #[pyo3(get)]
    payload_json: String,
    #[pyo3(get)]
    contract_json: Option<String>,
}

fn serialization_error(error: impl std::fmt::Display) -> PyErr {
    CausalSerializationError::new_err(error.to_string())
}

fn parse<T: DeserializeOwned>(json: &str) -> PyResult<T> {
    serde_json::from_str(json).map_err(serialization_error)
}

fn json<T: Serialize>(value: &T) -> PyResult<String> {
    serde_json::to_string(value).map_err(serialization_error)
}

fn parse_payload(kind: &str, payload_json: &str) -> PyResult<CausalPayloadWire> {
    Ok(match kind {
        "query" => {
            let wire = parse::<CausalQueryWire>(payload_json)?;
            let domain =
                antecedent_io::causal_query_from_wire(&wire).map_err(serialization_error)?;
            CausalPayloadWire::Query(Box::new(
                antecedent_io::causal_query_to_wire(&domain).map_err(serialization_error)?,
            ))
        }
        "static_result" => CausalPayloadWire::StaticResult(Box::new(parse::<
            antecedent_io::StaticResultWire,
        >(payload_json)?)),
        "response_result" => {
            let wire = parse::<CausalResponseWire>(payload_json)?;
            let domain =
                antecedent_io::causal_response_from_wire(&wire).map_err(serialization_error)?;
            CausalPayloadWire::ResponseResult(Box::new(
                antecedent_io::causal_response_to_wire(&domain).map_err(serialization_error)?,
            ))
        }
        "transport_identification" => {
            let wire = parse::<TransportIdentificationWire>(payload_json)?;
            let domain = antecedent_io::transport_identification_from_wire(&wire);
            CausalPayloadWire::TransportIdentification(Box::new(
                antecedent_io::transport_identification_to_wire(&domain),
            ))
        }
        "transport_estimate" => {
            let wire = parse::<TransportEffectEstimateWire>(payload_json)?;
            let domain =
                antecedent_io::transport_effect_from_wire(&wire).map_err(serialization_error)?;
            CausalPayloadWire::TransportEstimate(Box::new(
                antecedent_io::transport_effect_to_wire(&domain).map_err(serialization_error)?,
            ))
        }
        "interference_estimate" => {
            let wire = parse::<InterferenceEstimateWire>(payload_json)?;
            let domain = antecedent_io::interference_estimate_from_wire(&wire);
            CausalPayloadWire::InterferenceEstimate(Box::new(
                antecedent_io::interference_estimate_to_wire(&domain),
            ))
        }
        _ => {
            return Err(CausalSerializationError::new_err(format!(
                "unknown causal artifact payload kind {kind:?}"
            )));
        }
    })
}

fn payload_json(payload: &CausalPayloadWire) -> PyResult<String> {
    match payload {
        CausalPayloadWire::Query(value) => json(value),
        CausalPayloadWire::ResponseResult(value) => json(value),
        CausalPayloadWire::StaticResult(value) => json(value),
        CausalPayloadWire::TransportIdentification(value) => json(value),
        CausalPayloadWire::TransportEstimate(value) => json(value),
        CausalPayloadWire::InterferenceEstimate(value) => json(value),
    }
}

#[pyfunction]
fn encode_causal_artifact<'py>(
    py: Python<'py>,
    payload_kind: &str,
    variable_names: Vec<String>,
    payload_json: &str,
    artifact_id: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let artifact = if payload_kind == "analysis_result" {
        let result = parse::<antecedent_io::AnalysisResultWire>(payload_json)?;
        antecedent_io::encode_analysis_result_artifact(&result, variable_names, artifact_id)
    } else {
        let payload = parse_payload(payload_kind, payload_json)?;
        encode_causal_payload_artifact(&payload, variable_names, artifact_id)
    }
    .map_err(serialization_error)?;
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes).map_err(serialization_error)?;
    // Return real Python `bytes`, not the `list[int]` PyO3 would produce from
    // a bare `Vec<u8>` return type -- callers (and the `antecedent.artifacts`
    // wire format) depend on `isinstance(..., bytes)`.
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction]
fn decode_causal_artifact(bytes: &[u8]) -> PyResult<DecodedCausalArtifact> {
    let Ok((artifact, header, payload)) = decode_causal_payload_artifact(bytes) else {
        let (artifact, header, payload) =
            decode_analysis_result_artifact(bytes).map_err(serialization_error)?;
        let contract_json = match antecedent_io::decode_analysis_result_contract(&artifact) {
            Ok(Some(contract)) => Some(json(&contract)?),
            Ok(None) => None,
            Err(err) => return Err(serialization_error(err)),
        };
        return Ok(DecodedCausalArtifact {
            artifact_id: artifact.manifest.artifact_id,
            format_major: artifact.manifest.format_version.major,
            format_minor: artifact.manifest.format_version.minor,
            payload_kind: "analysis_result".into(),
            variable_names: header.variable_names,
            payload_json: json(&payload)?,
            contract_json,
        });
    };
    Ok(DecodedCausalArtifact {
        artifact_id: artifact.manifest.artifact_id,
        format_major: artifact.manifest.format_version.major,
        format_minor: artifact.manifest.format_version.minor,
        payload_kind: match header.payload_kind {
            antecedent_io::CausalPayloadKind::Query => "query",
            antecedent_io::CausalPayloadKind::ResponseResult => "response_result",
            antecedent_io::CausalPayloadKind::StaticResult => "static_result",
            antecedent_io::CausalPayloadKind::TransportIdentification => "transport_identification",
            antecedent_io::CausalPayloadKind::TransportEstimate => "transport_estimate",
            antecedent_io::CausalPayloadKind::InterferenceEstimate => "interference_estimate",
        }
        .into(),
        variable_names: header.variable_names,
        payload_json: payload_json(&payload)?,
        contract_json: None,
    })
}

#[pyfunction]
fn accept_analysis_result_contract(
    bytes: &[u8],
) -> PyResult<std::collections::HashMap<String, String>> {
    let consumed = antecedent_io::consume_analysis_result(bytes).map_err(serialization_error)?;
    let mut out = std::collections::HashMap::new();
    out.insert("recognized".into(), consumed.acceptance.recognized.to_string());
    out.insert("verified_references".into(), consumed.acceptance.verified_references.to_string());
    out.insert(
        "accepts_as_verified_program".into(),
        consumed.acceptance.accepts_as_verified_program().to_string(),
    );
    if let Some(restriction) = &consumed.acceptance.restriction {
        out.insert("restriction".into(), restriction.to_string());
    }
    out.insert(
        "unresolved".into(),
        consumed
            .acceptance
            .unresolved
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(","),
    );
    if let Some(contract) = &consumed.contract {
        out.insert("program".into(), antecedent_io::digest_hex(&contract.identities.program));
        out.insert("target".into(), antecedent_io::digest_hex(&contract.identities.target));
        out.insert("graph_class".into(), contract.graph_class.clone());
        out.insert("variable_names".into(), contract.target.schema.variable_names().join(","));
        for (key, value) in executed_functional_labels(&contract.target.query) {
            out.insert(key, value);
        }
    }
    Ok(out)
}

#[pyfunction(name = "encode_external_estimate_claim")]
#[pyo3(signature = (*, learner, config, payload, names, treatment, outcome, identifier, status, confounders, identification=None, data_snapshot=None, snapshot_payload=None, scalar_value=None))]
#[allow(clippy::too_many_arguments)]
fn encode_external_estimate_claim_py<'py>(
    py: Python<'py>,
    learner: &str,
    config: &[u8],
    payload: &[u8],
    names: Vec<String>,
    treatment: &str,
    outcome: &str,
    identifier: &str,
    status: &str,
    confounders: Vec<String>,
    identification: Option<&str>,
    data_snapshot: Option<&str>,
    snapshot_payload: Option<&[u8]>,
    scalar_value: Option<f64>,
) -> PyResult<Bound<'py, PyBytes>> {
    let identification =
        identification.map(parse_digest_hex).transpose().map_err(serialization_error)?;
    let data_snapshot = match data_snapshot {
        Some(hex) => parse_digest_hex(hex).map_err(serialization_error)?,
        None => payload_digest("external_estimate.data_snapshot", snapshot_payload.unwrap_or(b"")),
    };
    let bytes = encode_external_estimate_claim(&ExternalEstimateAttach {
        learner: learner.into(),
        config: config.to_vec(),
        payload: payload.to_vec(),
        names,
        treatment: treatment.into(),
        outcome: outcome.into(),
        confounders,
        identifier: identifier.into(),
        status: status.into(),
        identification,
        data_snapshot,
        scalar_value,
    })
    .map_err(serialization_error)?;
    Ok(PyBytes::new(py, &bytes))
}

/// A verified parent claim and its immutable provider-independent predictor.
#[pyclass(frozen, skip_from_py_object)]
struct FittedEffectModel {
    model: antecedent_estimate::FittedEffect,
    bytes: Vec<u8>,
    #[pyo3(get)]
    features: Vec<String>,
    #[pyo3(get)]
    parent_claim: String,
}

#[pymethods]
impl FittedEffectModel {
    #[staticmethod]
    fn load(bytes: &[u8]) -> PyResult<Self> {
        let consumed =
            antecedent_io::consume_analysis_result(bytes).map_err(serialization_error)?;
        if !consumed.acceptance.accepts_as_verified_program()
            || !consumed.acceptance.verified_references
        {
            return Err(serialization_error("fitted prediction requires a verified parent claim"));
        }
        let claim =
            consumed.contract.as_ref().and_then(|c| c.claim.as_ref()).ok_or_else(|| {
                serialization_error("fitted prediction requires an executed claim")
            })?;
        let model = consumed
            .body
            .fitted_effect
            .ok_or_else(|| serialization_error("this result has no portable fitted effect"))?;
        model.validate().map_err(serialization_error)?;
        let features = model
            .features
            .iter()
            .map(|v| consumed.header.variable_names[*v as usize].clone())
            .collect();
        Ok(Self {
            model,
            bytes: bytes.to_vec(),
            features,
            parent_claim: antecedent_io::digest_hex(&claim.claim_id),
        })
    }

    /// Predict from NumPy feature columns without copying them into Python lists.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        columns: Vec<PyReadonlyArray1<'py, f64>>,
        nrows: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let ids: Vec<_> =
            self.model.features.iter().map(|v| antecedent_core::VariableId::from_raw(*v)).collect();
        // The read-only guards in `columns` outlive the detached closure that borrows the slices.
        let slices: Vec<&[f64]> = columns
            .iter()
            .map(|column| {
                column.as_slice().map_err(|_| {
                    PyValueError::new_err("prediction columns must be contiguous float64 arrays")
                })
            })
            .collect::<PyResult<_>>()?;
        let values = py.detach(|| {
            self.model
                .predict(
                    &ids,
                    &slices,
                    nrows,
                    &antecedent_core::ExecutionContext::production_default(0),
                )
                .map_err(serialization_error)
        })?;
        Ok(PyArray1::from_vec(py, values))
    }

    fn export<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.bytes)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DecodedCausalArtifact>()?;
    m.add_class::<FittedEffectModel>()?;
    m.add_function(wrap_pyfunction!(encode_causal_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(decode_causal_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(accept_analysis_result_contract, m)?)?;
    m.add_function(wrap_pyfunction!(encode_external_estimate_claim_py, m)?)?;
    Ok(())
}
