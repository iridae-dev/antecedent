//! Bounded Python bridge for the fixed-population temporal counterfactual and the closed
//! transported path-specific route.
//!
//! Python builds the declaration (graph, mechanism fit, factual unit histories, two named action
//! histories); abduction, action, prediction, the shared-abduction receipt, the artifact and
//! every refusal are Rust's. A refusal comes back as structured JSON, retaining the refusing
//! witness, for the Python layer to raise as its own exception type.

use std::collections::BTreeMap;

use crate::bridge::check_declared_size;
use antecedent::analysis::temporal_counterfactual::{
    GeneralTemporalCounterfactual, GeneralTemporalCounterfactualArtifactError,
    GeneralTemporalIdentity, GeneralTemporalRequestWire, PopulationRole, RegimeFactorKey,
    TemporalCounterfactual, TemporalCounterfactualArtifactError, TemporalCounterfactualIdentity,
    TemporalCounterfactualRequestWire, TransportedCounterfactualPrerequisites,
    transported_path_specific,
};
use pyo3::prelude::*;

use crate::{
    CausalCancelledError, detach_catch, py_execution_context, value_err, with_reason_code,
};

/// Largest request / identity JSON accepted.
const MAX_JSON_BYTES: usize = crate::bridge::MAX_SEALED_ARTIFACT_BYTES;

type EvaluatePayload = (Option<String>, Option<Vec<u8>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);

fn refusal_json(error: &TemporalCounterfactualArtifactError) -> PyResult<Option<String>> {
    let Some(refusal) = error.refusal() else {
        return Ok(None);
    };
    Ok(Some(
        serde_json::to_string(&refusal)
            .map_err(|cause| crate::serialization_err(cause.to_string()))?,
    ))
}

/// A non-refusal failure: cancellation, a bound, or a corrupt / other-version artifact.
fn failure(error: &TemporalCounterfactualArtifactError) -> PyErr {
    match error {
        TemporalCounterfactualArtifactError::Cancelled => {
            CausalCancelledError::new_err("cancelled during temporal_counterfactual")
        }
        TemporalCounterfactualArtifactError::LimitsExceeded(what) => with_reason_code(
            value_err(format!("temporal_counterfactual.limits_exceeded: {what}")),
            antecedent_core::reason_code!("invalid_argument"),
        ),
        other => crate::serialization_err(other.to_string()),
    }
}

fn check_size(text: &str) -> PyResult<()> {
    check_declared_size(text, MAX_JSON_BYTES, "temporal counterfactual")
}

/// Evaluate both action histories against one abduced history per unit:
/// `(report json, artifact bytes, refusal json)`.
#[pyfunction]
#[pyo3(signature = (request_json, artifact_id, *, seed=0))]
fn evaluate_temporal_counterfactual(
    py: Python<'_>,
    request_json: &str,
    artifact_id: &str,
    seed: u64,
) -> PyResult<EvaluatePayload> {
    check_size(request_json)?;
    let request: TemporalCounterfactualRequestWire =
        serde_json::from_str(request_json).map_err(|e| {
            with_reason_code(
                value_err(format!("temporal_counterfactual.invalid_request: {e}")),
                antecedent_core::reason_code!("invalid_argument"),
            )
        })?;
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || {
        match TemporalCounterfactual::evaluate(&request, &py_execution_context(seed, 1)) {
            Ok(done) => {
                let bytes = done.export(&artifact_id).map_err(|e| failure(&e))?;
                let report = serde_json::to_string(&done.report())
                    .map_err(|e| crate::serialization_err(e.to_string()))?;
                Ok((Some(report), Some(bytes), None))
            }
            Err(error) => match refusal_json(&error)? {
                Some(json) => Ok((None, None, Some(json))),
                None => Err(failure(&error)),
            },
        }
    })
}

/// Consume an artifact by replaying both worlds: `(report json, refusal json)`.
///
/// `expected_identity_json` is an identity the caller retained independently; a changed action
/// time, unit history, snapshot or mechanism fit is refused against it even when resealed.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity_json=None, *, seed=0))]
fn consume_temporal_counterfactual_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    expected_identity_json: Option<String>,
    seed: u64,
) -> PyResult<ConsumePayload> {
    let expected: Option<TemporalCounterfactualIdentity> = match expected_identity_json {
        Some(text) => {
            check_size(&text)?;
            Some(serde_json::from_str(&text).map_err(|e| {
                with_reason_code(
                    value_err(format!("temporal_counterfactual.invalid_identity: {e}")),
                    antecedent_core::reason_code!("invalid_argument"),
                )
            })?)
        }
        None => None,
    };
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, 1);
        match TemporalCounterfactual::consume(&artifact, expected.as_ref(), &ctx) {
            Ok(done) => {
                let report = serde_json::to_string(&done.report())
                    .map_err(|e| crate::serialization_err(e.to_string()))?;
                Ok((Some(report), None))
            }
            Err(error) => match refusal_json(&error)? {
                Some(json) => Ok((None, Some(json))),
                None => Err(failure(&error)),
            },
        }
    })
}

fn general_refusal_json(
    error: &GeneralTemporalCounterfactualArtifactError,
) -> PyResult<Option<String>> {
    let Some(refusal) = error.refusal() else {
        return Ok(None);
    };
    Ok(Some(
        serde_json::to_string(&refusal)
            .map_err(|cause| crate::serialization_err(cause.to_string()))?,
    ))
}

fn general_failure(error: &GeneralTemporalCounterfactualArtifactError) -> PyErr {
    match error {
        GeneralTemporalCounterfactualArtifactError::Cancelled => {
            CausalCancelledError::new_err("cancelled during temporal_counterfactual")
        }
        GeneralTemporalCounterfactualArtifactError::LimitsExceeded(what) => with_reason_code(
            value_err(format!("temporal_counterfactual.limits_exceeded: {what}")),
            antecedent_core::reason_code!("invalid_argument"),
        ),
        other => crate::serialization_err(other.to_string()),
    }
}

/// Evaluate every named action history of any horizon against one abduced history per unit:
/// `(report json, artifact bytes, refusal json)`.
#[pyfunction]
#[pyo3(signature = (request_json, artifact_id, *, seed=0))]
fn evaluate_general_temporal_counterfactual(
    py: Python<'_>,
    request_json: &str,
    artifact_id: &str,
    seed: u64,
) -> PyResult<EvaluatePayload> {
    check_size(request_json)?;
    let request: GeneralTemporalRequestWire = serde_json::from_str(request_json).map_err(|e| {
        with_reason_code(
            value_err(format!("temporal_counterfactual.invalid_request: {e}")),
            antecedent_core::reason_code!("invalid_argument"),
        )
    })?;
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || {
        match GeneralTemporalCounterfactual::evaluate(&request, &py_execution_context(seed, 1)) {
            Ok(done) => {
                let bytes = done.export(&artifact_id).map_err(|e| general_failure(&e))?;
                let report = serde_json::to_string(&done.report())
                    .map_err(|e| crate::serialization_err(e.to_string()))?;
                Ok((Some(report), Some(bytes), None))
            }
            Err(error) => match general_refusal_json(&error)? {
                Some(json) => Ok((None, None, Some(json))),
                None => Err(general_failure(&error)),
            },
        }
    })
}

/// Consume a general artifact by replaying every world: `(report json, refusal json)`.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity_json=None, *, seed=0))]
fn consume_general_temporal_counterfactual_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    expected_identity_json: Option<String>,
    seed: u64,
) -> PyResult<ConsumePayload> {
    let expected: Option<GeneralTemporalIdentity> = match expected_identity_json {
        Some(text) => {
            check_size(&text)?;
            Some(serde_json::from_str(&text).map_err(|e| {
                with_reason_code(
                    value_err(format!("temporal_counterfactual.invalid_identity: {e}")),
                    antecedent_core::reason_code!("invalid_argument"),
                )
            })?)
        }
        None => None,
    };
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, 1);
        match GeneralTemporalCounterfactual::consume(&artifact, expected.as_ref(), &ctx) {
            Ok(done) => {
                let report = serde_json::to_string(&done.report())
                    .map_err(|e| crate::serialization_err(e.to_string()))?;
                Ok((Some(report), None))
            }
            Err(error) => match general_refusal_json(&error)? {
                Some(json) => Ok((None, Some(json))),
                None => Err(general_failure(&error)),
            },
        }
    })
}

fn role_of(tag: &str) -> PyResult<PopulationRole> {
    match tag {
        "source" => Ok(PopulationRole::Source),
        "target" => Ok(PopulationRole::Target),
        other => Err(value_err(format!(
            "transported_counterfactual.invalid_factor: role must be \"source\" or \"target\", \
             got {other:?}"
        ))),
    }
}

/// The closed transported path-specific route: always the live refusal, as JSON.
///
/// Reports every missing prerequisite gate and every required regime factor absent from the
/// supplied evidence map. Nothing is evaluated whatever is supplied.
#[pyfunction]
fn transported_path_specific_refusal(
    transport_license: bool,
    fixed_population_license: bool,
    cross_population_assumptions: bool,
    required_factors: Vec<(String, String)>,
    supplied_factors: Vec<(String, String, String)>,
) -> PyResult<String> {
    let key = |(role, regime): &(String, String)| -> PyResult<RegimeFactorKey> {
        Ok(RegimeFactorKey { role: role_of(role)?, regime: regime.clone() })
    };
    let required = required_factors.iter().map(key).collect::<PyResult<Vec<_>>>()?;
    let mut supplied: BTreeMap<RegimeFactorKey, String> = BTreeMap::new();
    for (role, regime, evidence) in &supplied_factors {
        supplied.insert(key(&(role.clone(), regime.clone()))?, evidence.clone());
    }
    let prerequisites = TransportedCounterfactualPrerequisites {
        transport_license,
        fixed_population_license,
        cross_population_assumptions,
    };
    let refusal = match transported_path_specific(&prerequisites, &required, &supplied) {
        Err(refusal) => refusal,
        Ok(never) => match never {},
    };
    Ok(serde_json::json!({
        "code": refusal.refusal.code,
        "stage": refusal.refusal.stage,
        "detail": refusal.refusal.detail,
        "offending": refusal.refusal.offending,
        "expected": refusal.refusal.expected,
        "supplied": refusal.refusal.supplied,
        "remedy": refusal.refusal.remedy,
        "missing_gates": refusal.missing_gates.iter().map(|g| g.name()).collect::<Vec<_>>(),
        "missing_factors": refusal.missing_factors.iter().map(RegimeFactorKey::label).collect::<Vec<_>>(),
    })
    .to_string())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_temporal_counterfactual, m)?)?;
    m.add_function(wrap_pyfunction!(consume_temporal_counterfactual_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_general_temporal_counterfactual, m)?)?;
    m.add_function(wrap_pyfunction!(consume_general_temporal_counterfactual_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(transported_path_specific_refusal, m)?)
}
