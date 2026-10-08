//! Python bridge of the 2.3 B3 source-target mechanism discrepancy diagnostic.
//!
//! Python declares the two populations (a measurement contract and either raw rows or summary
//! statistics); the OLS fits, the Wald statistic, the p-value, the Holm breakdown, the minimal
//! detectable differences, the artifact, its identity digests and every refusal are Rust's. A
//! refusal comes back as structured JSON (`code`, `detail`, `message`) for the Python layer to
//! raise as its own typed exception. Non-rejection never certifies invariance and the test's
//! Type I error and power are unmeasured.
use antecedent::analysis::mechanism_discrepancy::{
    MechanismDiscrepancyIdentity, MechanismDiscrepancyRefusal, MechanismDiscrepancyRequestWire,
    MechanismMeasurement, ParentSpec, PopulationSample, consume_mechanism_discrepancy,
    run_mechanism_discrepancy, summarize_sample,
};
use antecedent_core::reason_code;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::json;

fn refusal_json(refusal: &MechanismDiscrepancyRefusal) -> String {
    json!({"code": refusal.code, "detail": refusal.detail, "message": refusal.message}).to_string()
}

fn invalid_json(detail: &str, message: &str) -> String {
    json!({
        "code": reason_code!("invalid_argument"),
        "detail": detail,
        "message": message,
    })
    .to_string()
}

/// Sufficient statistics (`n`, `X'X`, `X'y`, `y'y`) of one population's raw rows; returns the
/// summary JSON or a structured refusal.
#[pyfunction]
#[pyo3(signature = (
    label, node, node_unit, parents, protocol_id, outcome, parent_values
))]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
fn mechanism_discrepancy_summarize(
    label: String,
    node: String,
    node_unit: String,
    parents: Vec<(String, String)>,
    protocol_id: String,
    outcome: Vec<f64>,
    parent_values: Vec<Vec<f64>>,
) -> (Option<String>, Option<String>) {
    let sample = PopulationSample {
        label,
        measurement: MechanismMeasurement {
            node,
            node_unit,
            parents: parents.into_iter().map(|(name, unit)| ParentSpec { name, unit }).collect(),
            protocol_id,
        },
        outcome,
        parent_values,
        unit_ids: Vec::new(),
    };
    match summarize_sample(&sample) {
        Ok(summary) => (
            Some(
                json!({
                    "n": summary.n,
                    "xtx": summary.xtx,
                    "xty": summary.xty,
                    "yty": summary.yty,
                })
                .to_string(),
            ),
            None,
        ),
        Err(error) => {
            (None, Some(refusal_json(&MechanismDiscrepancyRefusal::from_estimation(&error))))
        }
    }
}

/// Run and seal the diagnostic of a declared request; returns the report JSON and the artifact
/// bytes, or a structured refusal.
#[pyfunction]
#[pyo3(signature = (request_json, source_ids, target_ids, artifact_id))]
#[doc(hidden)]
#[allow(clippy::type_complexity)]
fn mechanism_discrepancy_run<'py>(
    py: Python<'py>,
    request_json: &str,
    source_ids: Vec<String>,
    target_ids: Vec<String>,
    artifact_id: &str,
) -> (Option<String>, Option<Bound<'py, PyBytes>>, Option<String>) {
    let request: MechanismDiscrepancyRequestWire = match serde_json::from_str(request_json) {
        Ok(request) => request,
        Err(error) => {
            return (
                None,
                None,
                Some(invalid_json("mechanism_discrepancy.invalid_request", &error.to_string())),
            );
        }
    };
    let artifact = match run_mechanism_discrepancy(&request, &source_ids, &target_ids) {
        Ok(artifact) => artifact,
        Err(error) => {
            return (
                None,
                None,
                Some(refusal_json(&MechanismDiscrepancyRefusal::from_artifact_error(&error))),
            );
        }
    };
    let report = serde_json::to_string(&artifact.report());
    match (report, artifact.to_bytes(artifact_id)) {
        (Ok(report), Ok(bytes)) => (Some(report), Some(PyBytes::new(py, &bytes)), None),
        (_, Err(error)) => (
            None,
            None,
            Some(refusal_json(&MechanismDiscrepancyRefusal::from_artifact_error(&error))),
        ),
        (Err(error), _) => (
            None,
            None,
            Some(invalid_json("mechanism_discrepancy.invalid_artifact", &error.to_string())),
        ),
    }
}

/// Consume artifact bytes by recomputation; returns the report JSON or a structured refusal.
#[pyfunction]
#[pyo3(signature = (data, expected_identity_json=None))]
#[doc(hidden)]
fn mechanism_discrepancy_consume(
    data: &[u8],
    expected_identity_json: Option<&str>,
) -> (Option<String>, Option<String>) {
    let expected: Option<MechanismDiscrepancyIdentity> = match expected_identity_json {
        Some(text) => match serde_json::from_str(text) {
            Ok(identity) => Some(identity),
            Err(error) => {
                return (
                    None,
                    Some(invalid_json(
                        "mechanism_discrepancy.invalid_identity",
                        &error.to_string(),
                    )),
                );
            }
        },
        None => None,
    };
    match consume_mechanism_discrepancy(data, expected.as_ref()) {
        Ok(artifact) => match serde_json::to_string(&artifact.report()) {
            Ok(report) => (Some(report), None),
            Err(error) => (
                None,
                Some(invalid_json("mechanism_discrepancy.invalid_artifact", &error.to_string())),
            ),
        },
        Err(error) => {
            (None, Some(refusal_json(&MechanismDiscrepancyRefusal::from_artifact_error(&error))))
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(mechanism_discrepancy_summarize, module)?)?;
    module.add_function(wrap_pyfunction!(mechanism_discrepancy_run, module)?)?;
    module.add_function(wrap_pyfunction!(mechanism_discrepancy_consume, module)?)?;
    Ok(())
}
