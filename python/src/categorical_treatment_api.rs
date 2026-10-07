//! Bounded Python bridge for the B4 categorical-treatment fit.
//!
//! Python supplies the outcome, adjustment columns, level labels and a small JSON declaration
//! (snapshot, adjustment names, level set, scale, reference, minimum level rows, pairs,
//! monotonicity direction, covariance kind); the dummy-coded regression, the sparse/absent-level
//! refusals, the full covariance, the Holm family, the monotonicity test, the identity digests,
//! the artifact and every refusal are Rust's. A refusal comes back as structured JSON for the
//! Python layer to raise as its own exception type; corruption and unknown versions raise
//! `CausalSerializationError`.

use antecedent::analysis::categorical_treatment::{
    CategoricalSpecWire, CategoricalTreatment, CategoricalTreatmentArtifactError,
    CategoricalTreatmentIdentity, CategoricalTreatmentRequest,
};
use antecedent::analysis::vector_treatment::AdjustmentColumnRequest;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{CausalSerializationError, detach_catch, value_err, with_reason_code};

/// Largest declaration / identity JSON accepted.
const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;

type EvaluatePayload = (Option<String>, Option<Vec<u8>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    row_snapshot: String,
    adjustment: Vec<String>,
    spec: CategoricalSpecWire,
}

fn refusal_json(error: &CategoricalTreatmentArtifactError, stage: &'static str) -> Option<String> {
    let (code, detail, text) = error.refusal()?;
    let offending = match error {
        CategoricalTreatmentArtifactError::IdentityMismatch { field } => Some(*field),
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
fn failure(error: &CategoricalTreatmentArtifactError) -> PyErr {
    match error {
        CategoricalTreatmentArtifactError::LimitsExceeded(what) => with_reason_code(
            value_err(format!("categorical_treatment.limits_exceeded: {what}")),
            antecedent_core::reason_code!("invalid_argument"),
        ),
        other => CausalSerializationError::new_err(other.to_string()),
    }
}

fn check_size(text: &str) -> PyResult<()> {
    if text.len() > MAX_JSON_BYTES {
        return Err(PyValueError::new_err("categorical treatment declaration is too large"));
    }
    Ok(())
}

/// Fit the categorical regime: `(report json, artifact bytes, refusal json)`.
///
/// `adjustment_columns[j]` is the column of `declaration.adjustment[j]`.
#[pyfunction]
fn evaluate_categorical_treatment(
    py: Python<'_>,
    outcome: Vec<f64>,
    adjustment_columns: Vec<Vec<f64>>,
    levels: Vec<String>,
    declaration_json: &str,
    artifact_id: &str,
) -> PyResult<EvaluatePayload> {
    check_size(declaration_json)?;
    let declared: Declaration = serde_json::from_str(declaration_json)
        .map_err(|e| value_err(format!("categorical_treatment.invalid_request: {e}")))?;
    if declared.adjustment.len() != adjustment_columns.len() {
        return Err(with_reason_code(
            value_err(
                "categorical_treatment.invalid_request: a column is missing for a declared name",
            ),
            antecedent_core::reason_code!("invalid_argument"),
        ));
    }
    let request = CategoricalTreatmentRequest {
        outcome,
        row_snapshot: declared.row_snapshot,
        adjustment: declared
            .adjustment
            .into_iter()
            .zip(adjustment_columns)
            .map(|(name, values)| AdjustmentColumnRequest { name, values })
            .collect(),
        levels,
        spec: declared.spec,
    };
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || match CategoricalTreatment::evaluate(&request) {
        Ok(done) => {
            let bytes = done.export(&artifact_id).map_err(|e| failure(&e))?;
            let report = serde_json::to_string(&done.report())
                .map_err(|e| CausalSerializationError::new_err(e.to_string()))?;
            Ok((Some(report), Some(bytes), None))
        }
        Err(error) => match refusal_json(&error, "fit") {
            Some(json) => Ok((None, None, Some(json))),
            None => Err(failure(&error)),
        },
    })
}

/// Consume an artifact by recomputation: `(report json, refusal json)`.
///
/// `expected_identity_json` is an identity the caller retained independently; a changed level
/// order, reference, scale, family, covariance kind, design, null or result is refused against
/// it.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity_json=None))]
fn consume_categorical_treatment_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    expected_identity_json: Option<String>,
) -> PyResult<ConsumePayload> {
    let expected: Option<CategoricalTreatmentIdentity> = match expected_identity_json {
        Some(text) => {
            check_size(&text)?;
            Some(serde_json::from_str(&text).map_err(|e| {
                with_reason_code(
                    value_err(format!("categorical_treatment.invalid_identity: {e}")),
                    antecedent_core::reason_code!("invalid_argument"),
                )
            })?)
        }
        None => None,
    };
    detach_catch(py, move || match CategoricalTreatment::consume(&artifact, expected.as_ref()) {
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
    m.add_function(wrap_pyfunction!(evaluate_categorical_treatment, m)?)?;
    m.add_function(wrap_pyfunction!(consume_categorical_treatment_artifact, m)?)
}
