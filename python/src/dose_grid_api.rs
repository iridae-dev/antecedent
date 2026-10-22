//! Bounded Python bridge for the dose-grid functional row.
//!
//! Python declares the request (design, one named functional, grid, bandwidth and its
//! range, minimum local effective sample size, claims) and passes the dose/outcome table.
//! Per-dose support, the local-quadratic fit, the pointwise standard errors and intervals
//! (calibration unmeasured), the closed simultaneous band, the digests, the artifact and
//! every refusal are Rust's. A refusal comes back as structured JSON for the Python layer
//! to raise as its own exception type; corruption and unknown versions raise
//! `CausalSerializationError`.

use crate::bridge::{RefusalJson, check_declared_size, serialization};
use antecedent::analysis::bayesian_posterior_row::{
    BayesianPosteriorRow, PosteriorRowArtifactError, PosteriorRowConsumeLimits,
    PosteriorRowRequestWire,
};
use antecedent::analysis::dose_grid_functional::{
    DoseGridArtifactError, DoseGridConsumeLimits, DoseGridFunctional, DoseGridRequestWire,
    dose_support_table,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::{detach_catch, value_err};

/// Largest request JSON accepted (the table travels separately).
const MAX_JSON_BYTES: usize = crate::bridge::MAX_SPEC_DECLARATION_BYTES;

type EvaluatePayload = (Option<String>, Option<Py<PyBytes>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);
type ObservedEvaluatePayload = (
    Option<String>,
    Option<Py<PyBytes>>,
    Option<String>,
    Py<crate::inspect_receipt_api::PyExecutionFacts>,
);
type ObservedConsumePayload =
    (Option<String>, Option<String>, Py<crate::inspect_receipt_api::PyExecutionFacts>);

fn dose_context(
    threads: Option<u32>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<antecedent_core::ExecutionContext> {
    if threads == Some(0) {
        return Err(value_err("threads must be positive"));
    }
    let mut ctx = antecedent_core::ExecutionContext::production(
        0,
        threads.unwrap_or_else(antecedent_core::default_user_threads),
    );
    ctx.set_memory_limit(max_bytes);
    crate::apply_cancel(&mut ctx, cancel);
    Ok(ctx)
}

fn refusal_json(error: &DoseGridArtifactError, stage: &'static str) -> Option<String> {
    if matches!(error, DoseGridArtifactError::Refused { code, .. } if *code == antecedent_core::reason_code!("transport_budget_cancel"))
    {
        return None;
    }
    if matches!(error, DoseGridArtifactError::Undecodable(_)) {
        return None;
    }
    let (code, detail) = error.refusal();
    let message = match error {
        DoseGridArtifactError::Refused { message, .. } => message.clone(),
        other => other.to_string(),
    };
    Some(RefusalJson::new(code, stage, &detail, &message).json())
}

fn execution_error(error: &DoseGridArtifactError) -> PyErr {
    if let DoseGridArtifactError::Refused { code, detail, message } = error {
        if *code == antecedent_core::reason_code!("transport_budget_cancel") {
            return if detail == "dose_grid.cancelled" {
                crate::CausalCancelledError::new_err(message.clone())
            } else {
                crate::CausalResourceError::new_err(message.clone())
            };
        }
    }
    serialization(error)
}

/// The report: the premises with the table elided (its length is `result.n_rows`), the
/// result and both digests.
fn report_json(done: &DoseGridFunctional) -> PyResult<String> {
    let mut request = serde_json::to_value(done.request())
        .map_err(|e| crate::serialization_err(e.to_string()))?;
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
#[pyo3(signature=(request_json, dose, outcome, *, threads=None, max_bytes=None, cancel=None))]
fn evaluate_dose_grid_functional(
    py: Python<'_>,
    request_json: &str,
    dose: Vec<f64>,
    outcome: Vec<f64>,
    threads: Option<u32>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<EvaluatePayload> {
    check_declared_size(request_json, MAX_JSON_BYTES, "dose grid")?;
    let mut request: DoseGridRequestWire = serde_json::from_str(request_json)
        .map_err(|e| value_err(format!("dose_grid.invalid_request: {e}")))?;
    request.dose = dose;
    request.outcome = outcome;
    let ctx = dose_context(threads, max_bytes, cancel)?;
    let payload =
        detach_catch(py, move || match DoseGridFunctional::evaluate_with_context(request, &ctx) {
            Ok(done) => {
                let bytes = done.export().map_err(|e| serialization(&e))?;
                Ok((Some(report_json(&done)?), Some(bytes), None))
            }
            Err(error) => match refusal_json(&error, "dose_grid") {
                Some(json) => Ok((None, None, Some(json))),
                None => Err(execution_error(&error)),
            },
        })?;
    let (report, bytes, refusal) = payload;
    Ok((report, bytes.map(|b| PyBytes::new(py, &b).unbind()), refusal))
}

/// Consume an artifact by recomputation: `(report json, refusal json)`.
#[pyfunction]
#[pyo3(signature = (artifact, *, max_rows=100_000, max_grid=1_024, threads=None, max_bytes=None, cancel=None))]
fn consume_dose_grid_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    max_rows: usize,
    max_grid: usize,
    threads: Option<u32>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ConsumePayload> {
    let limits = DoseGridConsumeLimits { max_rows, max_grid };
    let ctx = dose_context(threads, max_bytes, cancel)?;
    detach_catch(py, move || {
        match DoseGridFunctional::consume_with_context(&artifact, limits, &ctx) {
            Ok(done) => Ok((Some(report_json(&done)?), None)),
            Err(error) => match refusal_json(&error, "consume") {
                Some(json) => Ok((None, Some(json))),
                None => Err(execution_error(&error)),
            },
        }
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
                    .map_err(|e| crate::serialization_err(e.to_string()))?;
                Ok((Some(json), None))
            }
            Err(error) => match refusal_json(&error, "dose_grid") {
                Some(json) => Ok((None, Some(json))),
                None => Err(execution_error(&error)),
            },
        }
    })
}

fn posterior_refusal_json(
    error: &PosteriorRowArtifactError,
    stage: &'static str,
) -> Option<String> {
    if matches!(
        error,
        PosteriorRowArtifactError::Undecodable(_)
            | PosteriorRowArtifactError::Refused { code: "transport_budget_cancel", .. }
    ) {
        return None;
    }
    let (code, detail) = error.refusal();
    let message = match error {
        PosteriorRowArtifactError::Refused { message, .. } => message.clone(),
        other => other.to_string(),
    };
    Some(RefusalJson::new(code, stage, &detail, &message).json())
}

fn posterior_execution_error(error: &PosteriorRowArtifactError) -> PyErr {
    if let PosteriorRowArtifactError::Refused { code: "transport_budget_cancel", detail, message } =
        error
    {
        if detail.ends_with(".cancelled") {
            return crate::CausalCancelledError::new_err(message.clone());
        }
        return crate::CausalResourceError::new_err(message.clone());
    }
    serialization(error)
}

/// The posterior report: the premises with the table elided, the retained posterior and both
/// digests.
fn posterior_report_json(done: &BayesianPosteriorRow) -> PyResult<String> {
    let mut request = serde_json::to_value(done.request())
        .map_err(|e| crate::serialization_err(e.to_string()))?;
    if let Some(map) = request.get_mut("dose_grid").and_then(|value| value.as_object_mut()) {
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

/// Run the unlicensed Bayesian dose-grid posterior and seal it:
/// `(report json, artifact bytes, refusal json)`.
///
/// `request_json` is the `dose_grid` request with empty `dose` and `outcome`; the table is
/// passed as `dose` and `outcome`.
#[pyfunction]
#[pyo3(signature = (request_json, dose, outcome, *, threads=None, max_bytes=None, cancel=None))]
fn evaluate_dose_grid_posterior(
    py: Python<'_>,
    request_json: &str,
    dose: Vec<f64>,
    outcome: Vec<f64>,
    threads: Option<u32>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ObservedEvaluatePayload> {
    check_declared_size(request_json, MAX_JSON_BYTES, "dose grid posterior")?;
    let mut request: PosteriorRowRequestWire = serde_json::from_str(request_json)
        .map_err(|e| value_err(format!("dose_grid.invalid_request: {e}")))?;
    let PosteriorRowRequestWire::DoseGrid(wire) = &mut request;
    wire.dose = dose;
    wire.outcome = outcome;
    let ctx = dose_context(threads, max_bytes, cancel)?;
    let (payload, facts) = detach_catch(py, move || {
        let attempt = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
            let done = BayesianPosteriorRow::evaluate_with_context(request, &ctx)?;
            let bytes = done.export()?;
            let report = posterior_report_json(&done)
                .map_err(|error| PosteriorRowArtifactError::Undecodable(error.to_string()))?;
            Ok::<_, PosteriorRowArtifactError>((report, bytes))
        });
        let payload = match attempt.result {
            Ok((report, bytes)) => Ok((Some(report), Some(bytes), None)),
            Err(error) => match posterior_refusal_json(&error, "dose_grid") {
                Some(json) => Ok((None, None, Some(json))),
                None => Err(posterior_execution_error(&error)),
            },
        };
        Ok((payload, attempt.facts))
    })?;
    let ((report, bytes, refusal), facts) =
        crate::inspect_receipt_api::finish_observed(py, payload, facts)?;
    Ok((report, bytes.map(|b| PyBytes::new(py, &b).unbind()), refusal, facts))
}

/// Consume a dose-grid posterior artifact by recomputation: `(report json, refusal json)`.
#[pyfunction]
#[pyo3(signature = (artifact, *, max_rows=100_000, max_grid=1_024, threads=None, max_bytes=None, cancel=None))]
fn consume_dose_grid_posterior_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    max_rows: usize,
    max_grid: usize,
    threads: Option<u32>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ObservedConsumePayload> {
    let limits = PosteriorRowConsumeLimits { max_rows, max_points: max_grid };
    let ctx = dose_context(threads, max_bytes, cancel)?;
    let (payload, facts) = detach_catch(py, move || {
        let attempt = antecedent_core::execution_attempt::observe_with_context(&ctx, || {
            let done = BayesianPosteriorRow::consume_with_context(&artifact, limits, &ctx)?;
            posterior_report_json(&done)
                .map_err(|error| PosteriorRowArtifactError::Undecodable(error.to_string()))
        });
        let payload = match attempt.result {
            Ok(report) => Ok((Some(report), None)),
            Err(error) => match posterior_refusal_json(&error, "consume") {
                Some(json) => Ok((None, Some(json))),
                None => Err(posterior_execution_error(&error)),
            },
        };
        Ok((payload, attempt.facts))
    })?;
    let ((report, refusal), facts) =
        crate::inspect_receipt_api::finish_observed(py, payload, facts)?;
    Ok((report, refusal, facts))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_dose_grid_functional, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_dose_grid_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(consume_dose_grid_posterior_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(consume_dose_grid_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(dose_support_labels, m)?)
}
