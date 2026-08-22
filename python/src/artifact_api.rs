//! Python bridge for standalone format-0.4 causal wire artifacts.

use antecedent_io::{
    CausalPayloadWire, CausalQueryWire, CausalResponseWire, InterferenceEstimateWire,
    TransportEffectEstimateWire, TransportIdentificationWire, decode_causal_payload_artifact,
    encode_causal_payload_artifact,
};
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
    let payload = parse_payload(payload_kind, payload_json)?;
    let artifact = encode_causal_payload_artifact(&payload, variable_names, artifact_id)
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
    let (artifact, header, payload) =
        decode_causal_payload_artifact(bytes).map_err(serialization_error)?;
    Ok(DecodedCausalArtifact {
        artifact_id: artifact.manifest.artifact_id,
        format_major: artifact.manifest.format_version.major,
        format_minor: artifact.manifest.format_version.minor,
        payload_kind: match header.payload_kind {
            antecedent_io::CausalPayloadKind::Query => "query",
            antecedent_io::CausalPayloadKind::ResponseResult => "response_result",
            antecedent_io::CausalPayloadKind::TransportIdentification => "transport_identification",
            antecedent_io::CausalPayloadKind::TransportEstimate => "transport_estimate",
            antecedent_io::CausalPayloadKind::InterferenceEstimate => "interference_estimate",
        }
        .into(),
        variable_names: header.variable_names,
        payload_json: payload_json(&payload)?,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DecodedCausalArtifact>()?;
    m.add_function(wrap_pyfunction!(encode_causal_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(decode_causal_artifact, m)?)?;
    Ok(())
}
