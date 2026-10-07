//! Bounded Python bridge for the B4 compact runtime export.
//!
//! Python declares the coefficient vector, covariance, basis, per-input support and refusal mask
//! as JSON; the validation, canonical ordering, the identity, the container, the independent
//! verifier and every refusal (out of support, masked, tampered, wrong identity) are Rust's. A
//! refusal comes back as structured JSON for the Python layer to raise as its own exception
//! type.
//!
//! The export evaluates a point prediction and a model-based standard error only; the standard
//! error is labelled `model_based_sqrt_phi_v_phi` with calibration unmeasured and is never an
//! interval.

use std::collections::BTreeMap;

use antecedent::analysis::compact_export::{
    CompactExport, ExportInput, ExportLimits, ExportQuery, ExportRefusal, ExportSpec, MaskRegion,
    PointWithSe, QueryValue, TermSpec,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::{detach_catch, value_err};

/// Largest declaration / query JSON accepted.
const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;

type BuildPayload = (Option<String>, Option<Vec<u8>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecJson {
    response: ScientificQuantityWire,
    inputs: Vec<ExportInput>,
    terms: Vec<TermSpec>,
    coefficients: Vec<f64>,
    covariance: Vec<f64>,
    mask: Vec<MaskRegion>,
}

/// One query value; a `null` number is the non-finite sentinel and is refused by the export.
#[derive(serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum QueryJson {
    Number(Option<f64>),
    Level(String),
}

fn refusal_value(error: &ExportRefusal, stage: &'static str) -> serde_json::Value {
    serde_json::json!({
        "code": error.code,
        "stage": stage,
        "detail": error.detail,
        "offending": error.subject,
        "expected": serde_json::Value::Null,
        "supplied": serde_json::Value::Null,
        "remedy": serde_json::Value::Null,
        "message": error.to_string(),
    })
}

fn refusal_json(error: &ExportRefusal, stage: &'static str) -> String {
    refusal_value(error, stage).to_string()
}

fn point_value(point: &PointWithSe) -> serde_json::Value {
    serde_json::json!({
        "point": point.point,
        "model_based_se": point.model_based_se,
        "se_basis": point.se_basis,
        "calibration": point.calibration,
        "export_identity": point.export_identity,
    })
}

fn check_size(text: &str) -> PyResult<()> {
    if text.len() > MAX_JSON_BYTES {
        return Err(PyValueError::new_err("compact export declaration is too large"));
    }
    Ok(())
}

fn body_json(export: &CompactExport) -> PyResult<String> {
    serde_json::to_string(&serde_json::json!({
        "identity": export.identity(),
        "body": export.body(),
    }))
    .map_err(|e| value_err(format!("compact_export.encode: {e}")))
}

fn limits_of(max_bytes: Option<usize>) -> ExportLimits {
    let mut limits = ExportLimits::default();
    if let Some(max_bytes) = max_bytes {
        limits.max_bytes = max_bytes;
    }
    limits
}

/// Build and seal an export: `(summary json, artifact bytes, refusal json)`.
#[pyfunction]
fn build_compact_export(
    py: Python<'_>,
    spec_json: &str,
    artifact_id: &str,
) -> PyResult<BuildPayload> {
    check_size(spec_json)?;
    let parsed: SpecJson = serde_json::from_str(spec_json)
        .map_err(|e| value_err(format!("compact_export.invalid_request: {e}")))?;
    let spec = ExportSpec {
        response: parsed.response,
        inputs: parsed.inputs,
        terms: parsed.terms,
        coefficients: parsed.coefficients,
        covariance: parsed.covariance,
        mask: parsed.mask,
    };
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || {
        let built = CompactExport::build(spec).and_then(|export| {
            let bytes = export.to_bytes(&artifact_id)?;
            Ok((export, bytes))
        });
        match built {
            Ok((export, bytes)) => Ok((Some(body_json(&export)?), Some(bytes), None)),
            Err(error) => Ok((None, None, Some(refusal_json(&error, "build")))),
        }
    })
}

/// Independently verify an artifact against the identity the caller retained:
/// `(summary json, refusal json)`.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity, max_bytes=None))]
fn consume_compact_export(
    py: Python<'_>,
    artifact: Vec<u8>,
    expected_identity: String,
    max_bytes: Option<usize>,
) -> PyResult<ConsumePayload> {
    let limits = limits_of(max_bytes);
    detach_catch(py, move || match CompactExport::consume(&artifact, &limits, &expected_identity) {
        Ok(export) => Ok((Some(body_json(&export)?), None)),
        Err(error) => Ok((None, Some(refusal_json(&error, "consume")))),
    })
}

/// Verify, then evaluate every query: `(results json, refusal json)`.
///
/// `queries_json` is a JSON array of `{variable_id: {"number": x} | {"level": name}}` objects.
/// The result is a JSON array with one `{"ok": {...}}` or `{"refusal": {...}}` per query, in
/// order; a verification refusal comes back in the second slot instead.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity, queries_json, max_bytes=None))]
fn evaluate_compact_export(
    py: Python<'_>,
    artifact: Vec<u8>,
    expected_identity: String,
    queries_json: &str,
    max_bytes: Option<usize>,
) -> PyResult<ConsumePayload> {
    check_size(queries_json)?;
    let parsed: Vec<BTreeMap<String, QueryJson>> = serde_json::from_str(queries_json)
        .map_err(|e| value_err(format!("compact_export.invalid_query: {e}")))?;
    let queries: Vec<ExportQuery> = parsed
        .into_iter()
        .map(|values| ExportQuery {
            values: values
                .into_iter()
                .map(|(name, value)| {
                    let value = match value {
                        QueryJson::Number(x) => QueryValue::Number(x.unwrap_or(f64::NAN)),
                        QueryJson::Level(level) => QueryValue::Level(level),
                    };
                    (name, value)
                })
                .collect(),
        })
        .collect();
    let limits = limits_of(max_bytes);
    detach_catch(py, move || match CompactExport::consume(&artifact, &limits, &expected_identity) {
        Ok(export) => {
            let results: Vec<serde_json::Value> = queries
                .iter()
                .map(|query| match export.evaluate(query) {
                    Ok(point) => serde_json::json!({ "ok": point_value(&point) }),
                    Err(error) => {
                        serde_json::json!({ "refusal": refusal_value(&error, "evaluate") })
                    }
                })
                .collect();
            Ok((Some(serde_json::Value::Array(results).to_string()), None))
        }
        Err(error) => Ok((None, Some(refusal_json(&error, "consume")))),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_compact_export, m)?)?;
    m.add_function(wrap_pyfunction!(consume_compact_export, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_compact_export, m)?)
}
