//! Bounded Python bridge for the 2.3 B1 dose-grid functional row.
//!
//! Python declares the request (design, one named functional, grid, bandwidth and its
//! range, minimum local effective sample size, claims) and passes the dose/outcome table.
//! Per-dose support, the local-quadratic fit, the pointwise standard errors and intervals
//! (calibration unmeasured), the closed simultaneous band, the digests, the artifact and
//! every refusal are Rust's. A refusal comes back as structured JSON for the Python layer
//! to raise as its own exception type; corruption and unknown versions raise
//! `CausalSerializationError`.

use antecedent::analysis::dose_grid_functional::{
    DoseGridArtifactError, DoseGridConsumeLimits, DoseGridFunctional, DoseGridRequestWire,
    dose_support_table,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::{CausalSerializationError, detach_catch, value_err};

/// Largest request JSON accepted (the table travels separately).
const MAX_JSON_BYTES: usize = 8 * 1024 * 1024;

type EvaluatePayload = (Option<String>, Option<Py<PyBytes>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);

fn refusal_json(error: &DoseGridArtifactError, stage: &'static str) -> Option<String> {
    if matches!(error, DoseGridArtifactError::Undecodable(_)) {
        return None;
    }
    let (code, detail) = error.refusal();
    let message = match error {
        DoseGridArtifactError::Refused { message, .. } => message.clone(),
        other => other.to_string(),
    };
    Some(
        serde_json::json!({
            "code": code,
            "stage": stage,
            "detail": detail,
            "offending": serde_json::Value::Null,
            "expected": serde_json::Value::Null,
            "supplied": serde_json::Value::Null,
            "remedy": serde_json::Value::Null,
            "message": message,
        })
        .to_string(),
    )
}

fn serialization(error: &DoseGridArtifactError) -> PyErr {
    CausalSerializationError::new_err(error.to_string())
}

/// The report: the premises with the table elided (its length is `result.n_rows`), the
/// result and both digests.
fn report_json(done: &DoseGridFunctional) -> PyResult<String> {
    let mut request = serde_json::to_value(done.request())
        .map_err(|e| CausalSerializationError::new_err(e.to_string()))?;
    if let Some(map) = request.as_object_mut() {
        map.remove("dose");
        map.remove("outcome");
    }
    let (premises, data) = done.digests();
    Ok(serde_json::json!({
        "request": request,
        "result": done.result(),
        "premises_digest": premises,
        "data_digest": data,
    })
    .to_string())
}

/// Run the row and seal it: `(report json, artifact bytes, refusal json)`.
///
/// `request_json` carries every premise with empty `dose` and `outcome`; the table is
/// passed as `dose` and `outcome`.
#[pyfunction]
fn evaluate_dose_grid_functional(
    py: Python<'_>,
    request_json: &str,
    dose: Vec<f64>,
    outcome: Vec<f64>,
) -> PyResult<EvaluatePayload> {
    if request_json.len() > MAX_JSON_BYTES {
        return Err(value_err("dose_grid.invalid_request: the declaration is too large"));
    }
    let mut request: DoseGridRequestWire = serde_json::from_str(request_json)
        .map_err(|e| value_err(format!("dose_grid.invalid_request: {e}")))?;
    request.dose = dose;
    request.outcome = outcome;
    let payload = detach_catch(py, move || match DoseGridFunctional::evaluate(request) {
        Ok(done) => {
            let bytes = done.export().map_err(|e| serialization(&e))?;
            Ok((Some(report_json(&done)?), Some(bytes), None))
        }
        Err(error) => match refusal_json(&error, "dose_grid") {
            Some(json) => Ok((None, None, Some(json))),
            None => Err(serialization(&error)),
        },
    })?;
    let (report, bytes, refusal) = payload;
    Ok((report, bytes.map(|b| PyBytes::new(py, &b).unbind()), refusal))
}

/// Consume an artifact by recomputation: `(report json, refusal json)`.
#[pyfunction]
#[pyo3(signature = (artifact, *, max_rows=100_000, max_grid=1_024))]
fn consume_dose_grid_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    max_rows: usize,
    max_grid: usize,
) -> PyResult<ConsumePayload> {
    let limits = DoseGridConsumeLimits { max_rows, max_grid };
    detach_catch(py, move || match DoseGridFunctional::consume(&artifact, limits) {
        Ok(done) => Ok((Some(report_json(&done)?), None)),
        Err(error) => match refusal_json(&error, "consume") {
            Some(json) => Ok((None, Some(json))),
            None => Err(serialization(&error)),
        },
    })
}

/// Per-dose support labels without refusing: `(labels json, refusal json)`.
#[pyfunction]
fn dose_support_labels(
    py: Python<'_>,
    dose: Vec<f64>,
    points: Vec<f64>,
    bandwidth: f64,
    minimum_local_ess: f64,
) -> PyResult<ConsumePayload> {
    detach_catch(py, move || {
        match dose_support_table(&dose, &points, bandwidth, minimum_local_ess) {
            Ok(table) => {
                let json = serde_json::to_string(&table)
                    .map_err(|e| CausalSerializationError::new_err(e.to_string()))?;
                Ok((Some(json), None))
            }
            Err(error) => match refusal_json(&error, "dose_grid") {
                Some(json) => Ok((None, Some(json))),
                None => Err(serialization(&error)),
            },
        }
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_dose_grid_functional, m)?)?;
    m.add_function(wrap_pyfunction!(consume_dose_grid_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(dose_support_labels, m)?)
}
