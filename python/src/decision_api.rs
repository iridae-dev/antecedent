//! Bounded Python bridge for 2.3 decision contracts.
//!
//! Python builds the contract declaration; the identity, validation, evaluation,
//! artifacts and refusals are Rust's. A refusal comes back as structured JSON
//! for the Python layer to raise as its own exception type.

use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, MAX_DECISION_ARTIFACT_BYTES,
    contract_from_json, contract_from_json_refusal, contract_to_json, mean_result_to_json,
    result_to_json, source_digest,
};
use antecedent_design::decision_eval::{
    DecisionEvalError, MeanSource, evaluate_contract, evaluate_contract_on_means,
};
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use antecedent_core::{ExternalRefusal, ScientificQuantity};

use crate::CausalSerializationError;
use crate::distribution_api::PyJointDistributionArtifact;

fn serialization(error: impl std::fmt::Display) -> PyErr {
    CausalSerializationError::new_err(error.to_string())
}

fn refusal_json(value: &ExternalRefusal) -> String {
    serde_json::json!({
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

fn check_size(json: &str) -> PyResult<()> {
    if json.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(PyValueError::new_err("decision declaration is too large"));
    }
    Ok(())
}

fn io_refusal(error: &IoError, stage: &'static str) -> String {
    let (code, detail, offending) = match error {
        IoError::Refused { code, message } => {
            (*code, message.split(':').next().unwrap_or("decision_artifact").to_owned(), None)
        }
        other => (
            antecedent_core::reason_code!("decision_contract_unsatisfied"),
            "decision_contract.invalid_declaration".to_owned(),
            Some(other.to_string()),
        ),
    };
    refusal_json(&ExternalRefusal {
        code,
        stage,
        detail,
        offending,
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    })
}

fn eval_refusal(error: &DecisionEvalError) -> String {
    refusal_json(&error.to_refusal())
}

/// Validate a contract declaration; returns its normalized JSON (with the
/// recomputed identity) or a structured refusal.
#[pyfunction]
fn decision_contract_normalize(contract_json: &str) -> PyResult<(Option<String>, Option<String>)> {
    check_size(contract_json)?;
    match contract_from_json_refusal(contract_json) {
        Ok(contract) => Ok((Some(contract_to_json(&contract).map_err(serialization)?), None)),
        Err(refusal) => Ok((None, Some(refusal_json(&refusal)))),
    }
}

/// Evaluate a contract on aligned joint draws; returns the result JSON or a refusal.
#[pyfunction]
fn evaluate_decision(
    contract_json: &str,
    source: &PyJointDistributionArtifact,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(contract_json)?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
    };
    match evaluate_contract(&contract, source.inner()) {
        Ok(result) => {
            Ok((Some(result_to_json(&result, source.inner()).map_err(serialization)?), None))
        }
        Err(error) => Ok((None, Some(eval_refusal(&error)))),
    }
}

/// Evaluate a contract on a mean-only source (one mean per coordinate, such as an
/// external response grid); returns the result JSON or a refusal. The result's
/// `lineage` is empty: the caller supplies the derivation of the means.
#[pyfunction]
fn evaluate_decision_means(
    contract_json: &str,
    coordinates_json: &str,
    means: Vec<f64>,
    provider_id: &str,
    snapshot_id: &str,
    causal_contract_id: &str,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(contract_json)?;
    check_size(coordinates_json)?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
    };
    let wires: Vec<ScientificQuantityWire> = serde_json::from_str(coordinates_json)
        .map_err(|e| PyValueError::new_err(format!("invalid coordinates: {e}")))?;
    let coordinates = wires
        .into_iter()
        .map(|wire| {
            ScientificQuantity::try_from(wire).map_err(|e| PyValueError::new_err(String::from(e)))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let source = MeanSource {
        coordinates,
        means,
        provider_id: provider_id.to_owned(),
        snapshot_id: snapshot_id.to_owned(),
        causal_contract_id: causal_contract_id.to_owned(),
        rng_id: "none:mean_grid".to_owned(),
    };
    match evaluate_contract_on_means(&contract, &source) {
        Ok(result) => {
            Ok((Some(mean_result_to_json(&result, &source).map_err(serialization)?), None))
        }
        Err(error) => Ok((None, Some(eval_refusal(&error)))),
    }
}

/// Export a contract artifact.
#[pyfunction]
fn export_decision_contract<'py>(
    py: Python<'py>,
    contract_json: &str,
    artifact_id: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    check_size(contract_json)?;
    let contract = contract_from_json(contract_json).map_err(serialization)?;
    let artifact = DecisionContractArtifact::new(contract).map_err(serialization)?;
    Ok(PyBytes::new(py, &artifact.to_bytes(artifact_id).map_err(serialization)?))
}

/// Load a contract artifact under the consumer's retained identity.
#[pyfunction]
fn load_decision_contract(data: &[u8], expected_identity: &str) -> PyResult<String> {
    let artifact =
        DecisionContractArtifact::from_bytes(data, expected_identity).map_err(serialization)?;
    contract_to_json(artifact.contract()).map_err(serialization)
}

/// Evaluate a contract on a source and export the bound result artifact.
#[pyfunction]
fn export_decision_result<'py>(
    py: Python<'py>,
    contract_json: &str,
    source: &PyJointDistributionArtifact,
    artifact_id: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    check_size(contract_json)?;
    let contract = contract_from_json(contract_json).map_err(serialization)?;
    let result = evaluate_contract(&contract, source.inner()).map_err(serialization_eval)?;
    let artifact = DecisionResultArtifact::new(result, source.inner());
    Ok(PyBytes::new(py, &artifact.to_bytes(artifact_id).map_err(serialization)?))
}

fn serialization_eval(error: DecisionEvalError) -> PyErr {
    serialization(format!("{error:?}"))
}

/// Replay a stored result: load it under this contract and source, recompute,
/// and require an exact match. Returns the result JSON, or a refusal.
#[pyfunction]
fn replay_decision_result(
    data: &[u8],
    contract_json: &str,
    source: &PyJointDistributionArtifact,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(contract_json)?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
    };
    let identity = contract.identity().map_err(|e| serialization(format!("{e:?}")))?;
    let outcome =
        DecisionResultArtifact::from_bytes(data, &identity, &source_digest(source.inner()))
            .and_then(|artifact| artifact.replay(&contract, source.inner()).map(|()| artifact));
    match outcome {
        Ok(artifact) => Ok((
            Some(result_to_json(artifact.result(), source.inner()).map_err(serialization)?),
            None,
        )),
        Err(error) => Ok((None, Some(io_refusal(&error, "replay")))),
    }
}

/// Validate a derivation chain given as `[id, stage, [parent ids]]` rows and
/// return every link with its Merkle digest and its parents' digests, as JSON.
///
/// Python cannot hash, so a chain it extends (a decision over an external claim)
/// gets its digests here.
#[pyfunction]
fn composition_lineage(links_json: &str) -> PyResult<String> {
    check_size(links_json)?;
    let rows: Vec<(String, String, Vec<String>)> =
        serde_json::from_str(links_json).map_err(serialization)?;
    let parents: Vec<Vec<&str>> =
        rows.iter().map(|(_, _, parents)| parents.iter().map(String::as_str).collect()).collect();
    let borrowed: Vec<(&str, &str, &[&str])> = rows
        .iter()
        .zip(&parents)
        .map(|((id, stage, _), parents)| (id.as_str(), stage.as_str(), parents.as_slice()))
        .collect();
    let wire =
        antecedent_io::external_claim_artifact::lineage_wire(&borrowed).map_err(serialization)?;
    serde_json::to_string(&wire).map_err(serialization)
}

/// BLAKE3 digest of a source's aligned draws, for retaining alongside a result.
#[pyfunction]
fn decision_source_digest(source: &PyJointDistributionArtifact) -> String {
    source_digest(source.inner())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decision_contract_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_decision, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_decision_means, m)?)?;
    m.add_function(wrap_pyfunction!(export_decision_contract, m)?)?;
    m.add_function(wrap_pyfunction!(load_decision_contract, m)?)?;
    m.add_function(wrap_pyfunction!(export_decision_result, m)?)?;
    m.add_function(wrap_pyfunction!(replay_decision_result, m)?)?;
    m.add_function(wrap_pyfunction!(composition_lineage, m)?)?;
    m.add_function(wrap_pyfunction!(decision_source_digest, m)?)
}
