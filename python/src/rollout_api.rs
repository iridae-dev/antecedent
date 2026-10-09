//! Bounded finite source-state, terminal-decision and ranking artifact bridge.
//! Historical numerical replay preserves source standing and issues no native authority.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::{CausalSerializationError, detach_catch, value_err, with_reason_code};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn invalid_rollout_request(error: serde_json::Error) -> PyErr {
    with_reason_code(
        value_err(format!("rollout.invalid_request: {error}")),
        antecedent_core::reason_code!("invalid_argument"),
    )
}

fn serialization(error: impl std::fmt::Display) -> PyErr {
    CausalSerializationError::new_err(error.to_string())
}
fn check_binding_size(json: &str) -> PyResult<()> {
    if json.len() > 16 * 1024 * 1024 {
        return Err(PyValueError::new_err("rollout binding exceeds its JSON byte bound"));
    }
    Ok(())
}

/// Bind and independently check the original finite source law and ranking.
#[pyfunction]
#[pyo3(signature=(source, ranking, binding_json, artifact_id, source_evidence=None))]
fn export_rollout(
    py: Python<'_>,
    source: &[u8],
    ranking: &[u8],
    binding_json: &str,
    artifact_id: &str,
    source_evidence: Option<&[u8]>,
) -> PyResult<(String, Vec<u8>)> {
    check_binding_size(binding_json)?;
    if source.len() > antecedent_io::distribution_artifact::MAX_DISTRIBUTION_ARTIFACT_BYTES
        || ranking.len()
            > antecedent_design::design_ranking_artifact::MAX_DESIGN_RANKING_ARTIFACT_BYTES
    {
        return Err(PyValueError::new_err("rollout source or ranking exceeds its byte bound"));
    }
    if source_evidence.is_some_and(|bytes| {
        bytes.len() > antecedent_design::source_evidence::MAX_SOURCE_EVIDENCE_BYTES
    }) {
        return Err(with_reason_code(
            value_err("rollout.source_evidence_mismatch"),
            antecedent_core::reason_code!("invalid_argument"),
        ));
    }
    let source_evidence = source_evidence.map(<[u8]>::to_vec);
    let source = source.to_vec();
    let ranking = ranking.to_vec();
    let binding: antecedent_design::rollout_artifact::RolloutExpectation =
        serde_json::from_str(binding_json).map_err(invalid_rollout_request)?;
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || {
        let artifact =
            antecedent_design::rollout_artifact::RolloutArtifact::new(source, ranking, binding)
                .map_err(rollout_error)?;
        let artifact = if let Some(bytes) = source_evidence {
            artifact.with_source_evidence(bytes).map_err(rollout_error)?
        } else {
            artifact
        };
        let bytes = artifact.to_bytes(&artifact_id).map_err(rollout_error)?;
        Ok((rollout_json(&artifact)?, bytes))
    })
}

fn rollout_error(error: antecedent_io::error::IoError) -> PyErr {
    if let antecedent_io::error::IoError::Refused { code, message } = error {
        with_reason_code(value_err(message), code)
    } else {
        serialization(error)
    }
}

fn rollout_json(
    artifact: &antecedent_design::rollout_artifact::RolloutArtifact,
) -> PyResult<String> {
    let chain = artifact.provenance_chain().map_err(rollout_error)?;
    serde_json::to_string(&serde_json::json!({
        "identity": artifact.digest,
        "expectation": artifact.binding,
        "source_evidence": {
            "source_verified":true,"verification_scope":"original finite source law and ranking replay; source standing retained",
            "resolved_source_digest":artifact.binding.source_digest,
            "unresolved_source_digests":antecedent_design::design_ranking_artifact::DesignRankingArtifactWire::from_bytes(&artifact.ranking_artifact).map_err(serialization)?.source_digests.iter().filter(|digest| *digest != &artifact.binding.source_digest).collect::<Vec<_>>(),
            "source":artifact.binding.source,"state":artifact.binding.state,
            "original_diagnostics":artifact.diagnostic_evidence().map_err(rollout_error)?.map(|evidence| evidence.summary()),
            "native_row_origin_verified":false,
            "lineage":chain.links().iter().map(|link|serde_json::json!({
                "id":link.id,"stage":link.stage.as_str(),"parents":link.parents,"digest":chain.digest_of(&link.id).expect("validated link"),
                "parent_digests":link.parents.iter().map(|id|chain.digest_of(id).expect("validated parent")).collect::<Vec<_>>()
            })).collect::<Vec<_>>()
        },
        "native_execution_authority": false,
    }))
    .map_err(serialization)
}

/// Replay source projection and ranking under independent full expected inputs.
#[pyfunction]
fn consume_rollout(py: Python<'_>, artifact: &[u8], expectation_json: &str) -> PyResult<String> {
    check_binding_size(expectation_json)?;
    if artifact.len() > antecedent_design::rollout_artifact::MAX_ROLLOUT_BYTES {
        return Err(PyValueError::new_err("rollout artifact exceeds its byte bound"));
    }
    let artifact = artifact.to_vec();
    let expected: antecedent_design::rollout_artifact::RolloutExpectation =
        serde_json::from_str(expectation_json).map_err(invalid_rollout_request)?;
    detach_catch(py, move || {
        let artifact = antecedent_design::rollout_artifact::consume(&artifact, &expected)
            .map_err(rollout_error)?;
        rollout_json(&artifact)
    })
}

/// Original ranking bytes after complete source-state/terminal replay.
#[pyfunction]
fn rollout_ranking(artifact: &[u8], expectation_json: &str) -> PyResult<Vec<u8>> {
    check_binding_size(expectation_json)?;
    if artifact.len() > antecedent_design::rollout_artifact::MAX_ROLLOUT_BYTES {
        return Err(PyValueError::new_err("rollout artifact exceeds its byte bound"));
    }
    let expected = serde_json::from_str(expectation_json).map_err(invalid_rollout_request)?;
    Ok(antecedent_design::rollout_artifact::consume(artifact, &expected)
        .map_err(rollout_error)?
        .ranking_artifact)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(export_rollout, m)?)?;
    m.add_function(wrap_pyfunction!(rollout_ranking, m)?)?;
    m.add_function(wrap_pyfunction!(consume_rollout, m)?)
}
