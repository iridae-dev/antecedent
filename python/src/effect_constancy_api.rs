//! Bounded Python bridge for the F18 `EffectConstancy` test.
//!
//! Python builds the declaration (partitions, dependence, multiplicity family, level); the
//! frozen null, the shared-estimand check, the heterogeneity statistic, the Holm adjustment, the
//! identity digests, the artifact and every refusal are Rust's. A refusal comes back as
//! structured JSON for the Python layer to raise as its own exception type; corruption and
//! unknown versions raise `CausalSerializationError`.

use antecedent::analysis::effect_constancy::{
    EffectConstancy, EffectConstancyArtifactError, EffectConstancyIdentity,
    EffectConstancyRequestWire,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{CausalSerializationError, detach_catch, value_err, with_reason_code};

/// Largest request / identity JSON accepted (a 1024-partition covariance is about 25 MB).
const MAX_JSON_BYTES: usize = 64 * 1024 * 1024;

type EvaluatePayload = (Option<String>, Option<Vec<u8>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);

fn refusal_json(error: &EffectConstancyArtifactError, stage: &'static str) -> Option<String> {
    let (code, detail, text) = error.refusal()?;
    let offending = match error {
        EffectConstancyArtifactError::IdentityMismatch { field } => Some(*field),
        _ => None,
    };
    Some(
        serde_json::json!({
            "code": code,
            "stage": stage,
            "detail": detail,
            "offending": offending,
            "expected": serde_json::Value::Null,
            "supplied": serde_json::Value::Null,
            "remedy": serde_json::Value::Null,
            "message": text,
        })
        .to_string(),
    )
}

/// A non-refusal failure: a bound is a value error, everything else is a serialization error.
fn failure(error: &EffectConstancyArtifactError) -> PyErr {
    match error {
        EffectConstancyArtifactError::LimitsExceeded(what) => with_reason_code(
            value_err(format!("effect_constancy.limits_exceeded: {what}")),
            antecedent_core::reason_code!("invalid_argument"),
        ),
        other => CausalSerializationError::new_err(other.to_string()),
    }
}

fn check_size(text: &str) -> PyResult<()> {
    if text.len() > MAX_JSON_BYTES {
        return Err(PyValueError::new_err("effect constancy declaration is too large"));
    }
    Ok(())
}

/// Run the test on a declared request: `(report json, artifact bytes, refusal json)`.
#[pyfunction]
fn evaluate_effect_constancy(
    py: Python<'_>,
    request_json: &str,
    artifact_id: &str,
) -> PyResult<EvaluatePayload> {
    check_size(request_json)?;
    let request: EffectConstancyRequestWire = serde_json::from_str(request_json)
        .map_err(|e| value_err(format!("effect_constancy.invalid_request: {e}")))?;
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || match EffectConstancy::evaluate(&request) {
        Ok(done) => {
            let bytes = done.export(&artifact_id).map_err(|e| failure(&e))?;
            let report = serde_json::to_string(&done.report())
                .map_err(|e| CausalSerializationError::new_err(e.to_string()))?;
            Ok((Some(report), Some(bytes), None))
        }
        Err(error) => match refusal_json(&error, "test") {
            Some(json) => Ok((None, None, Some(json))),
            None => Err(failure(&error)),
        },
    })
}

/// Consume an artifact by recomputation: `(report json, refusal json)`.
///
/// `expected_identity_json` is an identity the caller retained independently; a changed
/// partition identity, estimand, covariance, null or multiplicity family is refused against it.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity_json=None))]
fn consume_effect_constancy_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    expected_identity_json: Option<String>,
) -> PyResult<ConsumePayload> {
    let expected: Option<EffectConstancyIdentity> = match expected_identity_json {
        Some(text) => {
            check_size(&text)?;
            Some(serde_json::from_str(&text).map_err(|e| {
                with_reason_code(
                    value_err(format!("effect_constancy.invalid_identity: {e}")),
                    antecedent_core::reason_code!("invalid_argument"),
                )
            })?)
        }
        None => None,
    };
    detach_catch(py, move || match EffectConstancy::consume(&artifact, expected.as_ref()) {
        Ok(done) => {
            let report = serde_json::to_string(&done.report())
                .map_err(|e| CausalSerializationError::new_err(e.to_string()))?;
            Ok((Some(report), None))
        }
        Err(error) => match refusal_json(&error, "consume") {
            Some(json) => Ok((None, Some(json))),
            None => Err(failure(&error)),
        },
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_effect_constancy, m)?)?;
    m.add_function(wrap_pyfunction!(consume_effect_constancy_artifact, m)?)
}
