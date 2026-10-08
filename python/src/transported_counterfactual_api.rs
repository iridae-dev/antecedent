//! Bounded Python bridge for the 2.3.0 A5 transported static path-specific counterfactual.
//!
//! Python builds the declaration (the additive-noise structural equations, the source and
//! target covariate laws, the selection diagram, the edge assignment, the declared premises and
//! the supplied evidence); the covariate-selected evaluation, the checked and declared
//! derivation, the artifact and every refusal, including the two-model non-recoverability
//! witness, are Rust's. A refusal comes back as structured JSON, retaining the witness and the
//! missing regime factors, for the Python layer to raise as its own exception type.

use antecedent::analysis::transported_counterfactual::{
    TransportedCounterfactual, TransportedCounterfactualArtifactError,
    TransportedCounterfactualIdentity, TransportedCounterfactualRequestWire,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{CausalSerializationError, detach_catch, value_err, with_reason_code};

/// Largest request / identity JSON accepted.
const MAX_JSON_BYTES: usize = 64 * 1024 * 1024;

type EvaluatePayload = (Option<String>, Option<Vec<u8>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);

fn refusal_json(error: &TransportedCounterfactualArtifactError) -> Option<String> {
    let refusal = error.refusal()?;
    Some(
        serde_json::json!({
            "code": refusal.code,
            "stage": refusal.stage,
            "detail": refusal.detail,
            "offending": refusal.offending,
            "expected": serde_json::Value::Null,
            "supplied": serde_json::Value::Null,
            "remedy": refusal.remedy,
            "witness": refusal.witness,
            "missing_factors": refusal.missing_factors,
        })
        .to_string(),
    )
}

/// A non-refusal failure: a bound, or a corrupt / other-version artifact.
fn failure(error: &TransportedCounterfactualArtifactError) -> PyErr {
    match error {
        TransportedCounterfactualArtifactError::LimitsExceeded(what) => with_reason_code(
            value_err(format!("transported_counterfactual.limits_exceeded: {what}")),
            antecedent_core::reason_code!("invalid_argument"),
        ),
        other => CausalSerializationError::new_err(other.to_string()),
    }
}

fn check_size(text: &str) -> PyResult<()> {
    if text.len() > MAX_JSON_BYTES {
        return Err(PyValueError::new_err("transported counterfactual declaration is too large"));
    }
    Ok(())
}

/// Evaluate the transported contrast and seal it:
/// `(report json, artifact bytes, refusal json)`.
#[pyfunction]
fn evaluate_transported_counterfactual(
    py: Python<'_>,
    request_json: &str,
    artifact_id: &str,
) -> PyResult<EvaluatePayload> {
    check_size(request_json)?;
    let request: TransportedCounterfactualRequestWire = serde_json::from_str(request_json)
        .map_err(|e| {
            with_reason_code(
                value_err(format!("transported_counterfactual.invalid_request: {e}")),
                antecedent_core::reason_code!("invalid_argument"),
            )
        })?;
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || match TransportedCounterfactual::evaluate(&request) {
        Ok(done) => {
            let bytes = done.export(&artifact_id).map_err(|e| failure(&e))?;
            let report = serde_json::to_string(&done.report())
                .map_err(|e| CausalSerializationError::new_err(e.to_string()))?;
            Ok((Some(report), Some(bytes), None))
        }
        Err(error) => match refusal_json(&error) {
            Some(json) => Ok((None, None, Some(json))),
            None => Err(failure(&error)),
        },
    })
}

/// Consume an artifact by recomputing the contrast: `(report json, refusal json)`.
///
/// `expected_identity_json` is an identity the caller retained independently; a changed
/// premise, law, coefficient, selection or assignment is refused against it even when resealed.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity_json=None))]
fn consume_transported_counterfactual_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    expected_identity_json: Option<String>,
) -> PyResult<ConsumePayload> {
    let expected: Option<TransportedCounterfactualIdentity> = match expected_identity_json {
        Some(text) => {
            check_size(&text)?;
            Some(serde_json::from_str(&text).map_err(|e| {
                with_reason_code(
                    value_err(format!("transported_counterfactual.invalid_identity: {e}")),
                    antecedent_core::reason_code!("invalid_argument"),
                )
            })?)
        }
        None => None,
    };
    detach_catch(py, move || {
        match TransportedCounterfactual::consume(&artifact, expected.as_ref()) {
            Ok(done) => {
                let report = serde_json::to_string(&done.report())
                    .map_err(|e| CausalSerializationError::new_err(e.to_string()))?;
                Ok((Some(report), None))
            }
            Err(error) => match refusal_json(&error) {
                Some(json) => Ok((None, Some(json))),
                None => Err(failure(&error)),
            },
        }
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_transported_counterfactual, m)?)?;
    m.add_function(wrap_pyfunction!(consume_transported_counterfactual_artifact, m)?)
}
