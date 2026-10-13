//! Python bridge for decision reliability.
//!
//! Python builds the draws carrier (utilities with stream identities and declared
//! dependence) and a request as JSON; the validation of every declaration, the Monte
//! Carlo error, the ranking diagnostics, the selection-aware evaluation, the artifact and
//! every refusal are Rust's. A refusal comes back as structured JSON for the Python layer
//! to raise as a typed exception. Nothing here estimates dependence: it is declared or it
//! refuses, and no ranking probability or interval is published.

use antecedent::analysis::decision_reliability::{
    CalibrationCoordinate, DecisionDraws, DecisionReliabilityExpectation,
    DecisionReliabilityInspection, DecisionReliabilityRefusal, PreparedDecisionReliability,
    PublishedRoute, ReliabilityRequest, StreamId, consume_decision_reliability_artifact,
};
use antecedent_design::decision_reliability::{
    RankingRequest, ReliabilityError, license_route, rank,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Value as Json, json};

use crate::PyCancellationToken;
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);
/// Exported bytes, report JSON and inspection JSON.
type AssessParts = (Py<PyBytes>, String, String);

fn refusal_json(refusal: &DecisionReliabilityRefusal) -> String {
    json!({
        "code": refusal.code,
        "stage": "decision_reliability",
        "detail": refusal.detail,
        "message": refusal.message,
        "offending": Json::Null,
        "remedy": refusal.remedy,
    })
    .to_string()
}

fn error_json(error: &ReliabilityError) -> String {
    let refusal = error.to_refusal();
    json!({
        "code": refusal.code,
        "stage": "decision_reliability",
        "detail": refusal.detail,
        "message": error.to_string(),
        "offending": refusal.offending,
        "remedy": refusal.remedy,
    })
    .to_string()
}

fn invalid_json(message: String) -> String {
    json!({
        "code": antecedent_core::reason_code!("invalid_argument"),
        "stage": "decision_reliability",
        "detail": "decision_reliability.invalid_input",
        "message": message,
        "offending": Json::Null,
        "remedy": Json::Null,
    })
    .to_string()
}

fn parse<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> Result<T, String> {
    serde_json::from_str(text).map_err(|error| invalid_json(format!("{what}: {error}")))
}

fn inspection_json(inspection: &DecisionReliabilityInspection) -> String {
    json!({
        "decided_id": inspection.decided_id,
        "standing": inspection.standing,
        "source_uncertainty": inspection.source_uncertainty,
        "monte_carlo_error": inspection.monte_carlo_error,
        "selection_uncertainty": inspection.selection_uncertainty,
        "closed_routes": inspection.closed_routes,
        "coordinate": inspection.coordinate,
    })
    .to_string()
}

fn parse_expectation(text: &str) -> Result<DecisionReliabilityExpectation, String> {
    let value: Json = parse(text, "expectation")?;
    let opt_text = |key: &str| value.get(key).and_then(Json::as_str).map(str::to_owned);
    let stream = |key: &str| -> Result<Option<StreamId>, String> {
        match value.get(key) {
            None | Some(Json::Null) => Ok(None),
            Some(inner) => serde_json::from_value(inner.clone())
                .map(Some)
                .map_err(|error| invalid_json(format!("{key}: {error}"))),
        }
    };
    Ok(DecisionReliabilityExpectation {
        draws_digest: opt_text("draws_digest"),
        candidate_set_digest: opt_text("candidate_set_digest"),
        selection_stream: stream("selection_stream")?,
        validation_stream: stream("validation_stream")?,
        utility_unit: opt_text("utility_unit"),
        decided_id: opt_text("decided_id"),
    })
}

/// Assess the reliability of a decision and export `decision_reliability_v1`.
#[pyfunction]
#[pyo3(signature = (draws_json, validation_json, request_json, artifact_id, seed=0, cancel=None))]
fn decision_reliability_assess(
    py: Python<'_>,
    draws_json: &str,
    validation_json: Option<&str>,
    request_json: &str,
    artifact_id: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> Refused<AssessParts> {
    let draws: DecisionDraws = match parse(draws_json, "draws") {
        Ok(value) => value,
        Err(refusal) => return (None, Some(refusal)),
    };
    let validation: Option<DecisionDraws> = match validation_json.map(|t| parse(t, "validation")) {
        Some(Ok(value)) => Some(value),
        Some(Err(refusal)) => return (None, Some(refusal)),
        None => None,
    };
    let request: ReliabilityRequest = match parse(request_json, "request") {
        Ok(value) => value,
        Err(refusal) => return (None, Some(refusal)),
    };
    let ctx = execution_context(seed, None, cancel);
    let prepared = match PreparedDecisionReliability::assess(draws, validation, request, &ctx) {
        Ok(value) => value,
        Err(refusal) => return (None, Some(refusal_json(&refusal))),
    };
    let bytes = match prepared.export(artifact_id, &ctx) {
        Ok(value) => value,
        Err(refusal) => return (None, Some(refusal_json(&refusal))),
    };
    let report = match serde_json::to_string(prepared.report()) {
        Ok(value) => value,
        Err(error) => return (None, Some(invalid_json(error.to_string()))),
    };
    (Some((PyBytes::new(py, &bytes).unbind(), report, inspection_json(&prepared.inspect()))), None)
}

/// Independently consume `decision_reliability_v1` bytes by recomputation.
#[pyfunction]
#[pyo3(signature = (artifact, expected_json=None, seed=0, cancel=None))]
fn decision_reliability_consume(
    artifact: &[u8],
    expected_json: Option<&str>,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> Refused<(String, String)> {
    let expected = match expected_json.map(parse_expectation) {
        Some(Ok(value)) => value,
        Some(Err(refusal)) => return (None, Some(refusal)),
        None => DecisionReliabilityExpectation::default(),
    };
    let ctx = execution_context(seed, None, cancel);
    match consume_decision_reliability_artifact(artifact, &expected, &ctx) {
        Ok(prepared) => match serde_json::to_string(prepared.report()) {
            Ok(report) => (Some((report, inspection_json(&prepared.inspect()))), None),
            Err(error) => (None, Some(invalid_json(error.to_string()))),
        },
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// Ranking statements over a finite candidate set.
#[pyfunction]
#[pyo3(signature = (draws_json, request_json, seed=0, cancel=None))]
fn decision_reliability_rank(
    draws_json: &str,
    request_json: &str,
    seed: u64,
    cancel: Option<PyCancellationToken>,
) -> Refused<String> {
    let draws: DecisionDraws = match parse(draws_json, "draws") {
        Ok(value) => value,
        Err(refusal) => return (None, Some(refusal)),
    };
    let request: RankingRequest = match parse(request_json, "ranking request") {
        Ok(value) => value,
        Err(refusal) => return (None, Some(refusal)),
    };
    let ctx = execution_context(seed, None, cancel);
    match rank(&draws, &request, &ctx) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(text) => (Some(text), None),
            Err(error) => (None, Some(invalid_json(error.to_string()))),
        },
        Err(error) => (None, Some(error_json(&error))),
    }
}

/// Digests of a carrier (draws and candidate set), after validating its declarations.
#[pyfunction]
fn decision_reliability_digests(draws_json: &str) -> Refused<(String, String)> {
    let draws: DecisionDraws = match parse(draws_json, "draws") {
        Ok(value) => value,
        Err(refusal) => return (None, Some(refusal)),
    };
    match draws.validate() {
        Ok(()) => (Some((draws.digest_hex(), draws.candidate_set_digest())), None),
        Err(error) => (None, Some(error_json(&error))),
    }
}

/// Ask for a published ranking probability or interval route at a coordinate. Every
/// route is closed until a coverage record measures it, so this returns its refusal.
#[pyfunction]
fn decision_reliability_publish(route: &str, coordinate_json: &str) -> Option<String> {
    let route: PublishedRoute = match serde_json::from_value(Json::String(route.to_owned())) {
        Ok(value) => value,
        Err(error) => return Some(invalid_json(format!("route: {error}"))),
    };
    let coordinate: CalibrationCoordinate = match parse(coordinate_json, "coordinate") {
        Ok(value) => value,
        Err(refusal) => return Some(refusal),
    };
    license_route(route, &coordinate).err().map(|error| error_json(&error))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decision_reliability_assess, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_consume, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_rank, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_digests, module)?)?;
    module.add_function(wrap_pyfunction!(decision_reliability_publish, module)?)?;
    Ok(())
}
