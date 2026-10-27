//! Python bridge for standalone format-0.5 causal wire artifacts.

use crate::NumericColumn;
use crate::bridge::serialization as serialization_error;
use antecedent_io::executed_functional_labels;
use antecedent_io::{
    CausalPayloadWire, CausalQueryWire, CausalResponseWire, ExternalEstimateAttach,
    InterferenceEstimateWire, TransportEffectEstimateWire, TransportIdentificationWire,
    decode_analysis_result_artifact, decode_causal_payload_artifact,
    encode_causal_payload_artifact, encode_external_estimate_claim, parse_digest_hex,
};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Serialize;
use serde::de::DeserializeOwned;

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
            let mut canonical =
                antecedent_io::causal_response_to_wire(&domain).map_err(serialization_error)?;
            canonical.coordinates = wire.coordinates;
            CausalPayloadWire::ResponseResult(Box::new(canonical))
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
            return Err(crate::serialization_err(format!(
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
#[pyo3(signature = (payload_kind, variable_names, payload_json, artifact_id, contract_json=None))]
fn encode_causal_artifact<'py>(
    py: Python<'py>,
    payload_kind: &str,
    variable_names: Vec<String>,
    payload_json: &str,
    artifact_id: &str,
    contract_json: Option<&str>,
) -> PyResult<Bound<'py, PyBytes>> {
    if contract_json.is_some() && payload_kind != "analysis_result" {
        return Err(crate::serialization_err("a contract belongs to an analysis_result artifact"));
    }
    let artifact = if payload_kind == "analysis_result" {
        let result = parse::<antecedent_io::AnalysisResultWire>(payload_json)?;
        let contract =
            contract_json.map(parse::<antecedent_io::AnalysisResultContractWire>).transpose()?;
        antecedent_io::encode_analysis_result_artifact_with_contract(
            &result,
            variable_names,
            artifact_id,
            contract.as_ref(),
        )
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

/// A provider result or external callback output, read structurally.
///
/// The payload is the declared header: a `verified_extension` trust here is a
/// producer's claim, believed only by the registry that holds its evidence.
fn decode_external_frame(bytes: &[u8]) -> PyResult<Option<DecodedCausalArtifact>> {
    use antecedent_io::external_callback_artifact::ExternalCallbackArtifact;
    use antecedent_io::provider_envelope::{inspect_provider_envelope, is_provider_envelope};

    let stable = antecedent_io::STABLE_FORMAT;
    if is_provider_envelope(bytes) {
        let envelope = inspect_provider_envelope(bytes).map_err(serialization_error)?;
        return Ok(Some(DecodedCausalArtifact {
            artifact_id: envelope.header.artifact_id(),
            format_major: stable.major,
            format_minor: stable.minor,
            payload_kind: "provider_result".into(),
            variable_names: Vec::new(),
            payload_json: json(&envelope.header)?,
            contract_json: None,
        }));
    }
    if ExternalCallbackArtifact::recognizes(bytes) {
        let callback = ExternalCallbackArtifact::inspect(bytes).map_err(serialization_error)?;
        let payload = serde_json::json!({
            "request_digest": antecedent_core::to_hex(&callback.request_digest),
            "provider_digest": antecedent_core::to_hex(&callback.provider_digest),
            "policy": callback.policy,
            "identity": callback.identity,
            "claim_blake3": antecedent_core::blake3_hex(&callback.artifact),
        });
        return Ok(Some(DecodedCausalArtifact {
            artifact_id: callback.artifact_id(),
            format_major: stable.major,
            format_minor: stable.minor,
            payload_kind: "external_callback".into(),
            variable_names: Vec::new(),
            payload_json: json(&payload)?,
            contract_json: None,
        }));
    }
    Ok(None)
}

/// Inspect and checksum a container without invoking any scientific consumer.
#[pyfunction]
fn inspect_artifact_container(bytes: &[u8]) -> PyResult<String> {
    inspect_container_metadata(bytes, 0)
}

fn inspect_container_metadata(bytes: &[u8], depth: usize) -> PyResult<String> {
    use antecedent_io::container::EncodedArtifact;
    use antecedent_io::reader::ArtifactReader;
    use std::io::Cursor;
    if depth > 8 {
        return Err(serialization_error("artifact framing exceeds the inspection depth limit"));
    }
    if bytes.starts_with(antecedent_io::bayesian_prior_recalc_artifact::BOUND_PRIOR_MAGIC) {
        let source =
            antecedent_io::bayesian_prior_recalc_artifact::BoundBayesianPriorWire::decode(bytes)
                .map_err(serialization_error)?;
        // Framing and retained metadata only. Posterior authority requires the
        // scientific source replay performed by the Bayesian prior consumer.
        return json(&serde_json::json!({
            "manifest": {"artifact_kind": {"other": "bound_bayesian_prior_source_v1"}},
            "metadata": {
                "source_snapshot": blake3::Hash::from(source.source_snapshot).to_hex().to_string(),
                "columns": source.columns.len(), "draws": source.draws,
                "seed": source.seed, "posterior_bytes": source.posterior.len(),
            },
        }));
    }
    if bytes.starts_with(b"ANTSTALEBASE1") {
        return json(&serde_json::json!({
            "manifest": {"artifact_kind": {"other": "staleness_baseline_v1"}},
            "metadata": {"encoded_payload_bytes": bytes.len() - 13, "validation": "requires_consumer"},
        }));
    }
    if bytes.starts_with(b"ANTECEDENT-") {
        let delimiter = bytes
            .iter()
            .position(|byte| *byte == 1)
            .ok_or_else(|| serialization_error("artifact frame has no version delimiter"))?;
        let marker = std::str::from_utf8(&bytes[..delimiter]).map_err(serialization_error)?;
        if !marker.bytes().all(|byte| byte.is_ascii_uppercase() || byte == b'-') {
            return Err(serialization_error("invalid artifact frame marker"));
        }
        let payload = &bytes[delimiter + 1..];
        // Temporal extension frames contain their container directly. Other
        // released frames carry the schema names and an inner container as CBOR.
        if marker == "ANTECEDENT-TRANSPORT-VIEW" {
            let metadata: serde_json::Value =
                serde_json::from_slice(payload).map_err(serialization_error)?;
            if !metadata.is_object() {
                return Err(serialization_error("transported view artifact is not a JSON object"));
            }
            return json(&serde_json::json!({
                "manifest": {"artifact_kind": {"other": marker}, "frame_version": 1},
                "metadata": {"envelope": metadata},
            }));
        }
        if matches!(marker, "ANTECEDENT-PERIOD-REFRESH" | "ANTECEDENT-PERIOD-REFRESH-SUMMARY") {
            // These frames prefix a raw scientific record, rather than a schema tuple.
            // Inspection recognizes framing only. Parsing row blocks and verifying the
            // seal belong to the independent consumer, keeping this metadata bounded.
            return json(&serde_json::json!({
                "manifest": {"artifact_kind": {"other": marker}, "frame_version": 1},
                "metadata": {"encoded_payload_bytes": payload.len(), "validation": "requires_consumer"},
            }));
        }
        let (names, inner) =
            if marker.starts_with("ANTECEDENT-TEMPORAL-")
                || matches!(marker, "ANTECEDENT-LEARNED-TRIAL" | "ANTECEDENT-SCENARIO-COVARIANCE")
            {
                (Vec::<String>::new(), payload.to_vec())
            } else {
                antecedent_io::convert::from_cbor::<(Vec<String>, Vec<u8>)>(payload)
                    .map_err(serialization_error)?
            };
        let inspected: serde_json::Value =
            serde_json::from_str(&inspect_container_metadata(&inner, depth + 1)?)
                .map_err(serialization_error)?;
        return json(&serde_json::json!({
            "manifest": {"artifact_kind": {"other": marker}, "frame_version": 1},
            "metadata": {"variable_names": names, "inner": inspected},
        }));
    }
    if !bytes.starts_with(antecedent_io::container::MAGIC) {
        if antecedent_io::array_fit_artifact::inspect_array_fit_header(bytes).map_err(serialization_error)? {
            return json(&serde_json::json!({
                "manifest": {"artifact_kind": {"other": "array_fit_replay_v1"}},
                "metadata": {"encoded_payload_bytes": bytes.len(), "validation": "requires_independent_refit"},
            }));
        }
        let metadata: serde_json::Value =
            antecedent_io::convert::from_cbor(bytes).map_err(serialization_error)?;
        let kind = metadata
            .get("required_features")
            .and_then(|v| v.as_array())
            .and_then(|features| features.first())
            .and_then(|v| v.as_str())
            .or_else(|| metadata.get("format").and_then(serde_json::Value::as_str))
            .or_else(|| {
                metadata
                    .get("version")
                    .filter(|version| version.is_number())
                    .map(|_| "unregistered_cbor")
            })
            .ok_or_else(|| {
                serialization_error("CBOR artifact has no format or required feature discriminator")
            })?;
        return json(&serde_json::json!({
            "manifest": {"artifact_kind": {"other": kind}, "format_version": metadata.get("version")},
            "metadata": {"body": metadata},
        }));
    }
    let reader = ArtifactReader::open_seek(Cursor::new(bytes)).map_err(serialization_error)?;
    let wanted = reader
        .manifest()
        .sections
        .iter()
        .filter(|section| section.id.contains("meta") || section.id.contains("header"))
        .map(|section| section.id.as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut cursor = Cursor::new(bytes);
    let artifact =
        EncodedArtifact::read_selective(&mut cursor, &wanted).map_err(serialization_error)?;
    if cursor.position() != bytes.len() as u64 {
        return Err(serialization_error("trailing bytes after artifact container"));
    }
    let metadata = artifact
        .sections
        .iter()
        .filter_map(|section| {
            antecedent_io::convert::from_cbor::<serde_json::Value>(&section.data)
                .ok()
                .map(|value| (section.id.clone(), value))
        })
        .collect::<serde_json::Map<_, _>>();
    json(&serde_json::json!({"manifest": artifact.manifest, "metadata": metadata}))
}

#[pyfunction]
fn decode_causal_artifact(bytes: &[u8]) -> PyResult<DecodedCausalArtifact> {
    if let Some(external) = decode_external_frame(bytes)? {
        return Ok(external);
    }
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
        out.insert("program".into(), antecedent_core::to_hex(&contract.identities.program));
        out.insert("target".into(), antecedent_core::to_hex(&contract.identities.target));
        out.insert("graph_class".into(), contract.graph_class.clone());
        out.insert("variable_names".into(), contract.target.schema.variable_names().join(","));
        for (key, value) in executed_functional_labels(&contract.target.query) {
            out.insert(key, value);
        }
    }
    Ok(out)
}

#[pyfunction(name = "encode_external_estimate_claim")]
#[pyo3(signature = (*, learner, config, payload, names, treatment, outcome, identifier, status, confounders, identification=None, data_snapshot=None, snapshot_payload=None, scalar_value=None, value_types=None, contrast=None, modifiers=None))]
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
    value_types: Option<Vec<String>>,
    contrast: Option<(f64, f64)>,
    modifiers: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyBytes>> {
    let value_types = value_types
        .unwrap_or_default()
        .iter()
        .map(|name| match name.as_str() {
            "continuous" => Ok(antecedent_core::ValueType::Continuous),
            "binary" => Ok(antecedent_core::ValueType::Binary),
            "count" => Ok(antecedent_core::ValueType::Count),
            other => Err(serialization_error(format!(
                "unsupported value type `{other}` (use continuous, binary or count)"
            ))),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let identification =
        identification.map(parse_digest_hex).transpose().map_err(serialization_error)?;
    let data_snapshot = match data_snapshot {
        Some(hex) => parse_digest_hex(hex).map_err(serialization_error)?,
        None => antecedent_core::CanonicalWriter::digest_bytes(
            "antecedent.external_estimate.data_snapshot.v1",
            snapshot_payload.unwrap_or(b""),
        ),
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
        value_types,
        contrast,
        modifiers: modifiers.unwrap_or_default(),
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
            .as_deref()
            .cloned()
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
            parent_claim: antecedent_core::to_hex(&claim.claim_id),
        })
    }

    /// Predict from NumPy feature columns without copying them into Python lists.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        columns: Vec<NumericColumn<'py>>,
        nrows: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let ids: Vec<_> =
            self.model.features.iter().map(|v| antecedent_core::VariableId::from_raw(*v)).collect();
        // Contiguous float64 columns are borrowed; any other column is widened once. Either way the
        // values outlive the detached closure that borrows the slices.
        let owned: Vec<std::borrow::Cow<'_, [f64]>> = columns
            .iter()
            .enumerate()
            .map(|(index, column)| column.values(&format!("prediction column {index}")))
            .collect::<PyResult<_>>()?;
        let slices: Vec<&[f64]> = owned.iter().map(|column| &column[..]).collect();
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
    m.add_function(wrap_pyfunction!(inspect_artifact_container, m)?)?;
    m.add_function(wrap_pyfunction!(accept_analysis_result_contract, m)?)?;
    m.add_function(wrap_pyfunction!(encode_external_estimate_claim_py, m)?)?;
    Ok(())
}
