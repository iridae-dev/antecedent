//! Python surface for consuming previous-release artifacts.
//!
//! A previous-release artifact carries wide score row ids, the score-row digest it
//! retained, a carried scientific-target digest and a recalculation receipt. It is
//! adopted only when both digests are already the current digests and the receipt
//! claims no derived-stage reuse in a fresh process. A route-record artifact is
//! carried record by record onto the current methods, each with its own outcome or
//! break. Nothing here widens a stale digest.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    reason_code, AssumptionSet, Conditioning, GraphFamily, InferenceSelection, Intervention,
    InterventionWorld, ModelAssumptions, ObservationUnit, OutcomeQuantity, QuantityUnits,
    ScientificTarget, ScientificWorldId, TargetForm, TargetPopulation, TimeFrame, Value,
    VariableId, WorldAction, WorldBinding,
};
use antecedent_io::recalc_receipt_artifact::{
    RecalcReceiptArtifact, RecalcReceiptArtifactError,
};
use antecedent_io::release_migration::{
    consume_previous_release_artifact, consume_previous_route_records, current_score_row_digest,
    narrow_score_row_digest, CurrentMethod, MethodAssignment, PreviousReleaseError,
    RouteRecordError, RouteRecordOutcome,
};
use pyo3::prelude::*;
use serde_json::{json, Value as Json};

/// What a previous-release artifact adopted: score-row digest, target digest, boundary
/// name and the `(stage, status)` rows in receipt order.
type Adopted = (Vec<u8>, Vec<u8>, String, Vec<(String, String)>);

/// One current method record: id, route, estimator, design, inference, status.
type MethodRow = (String, String, String, String, String, String);

fn invalid(message: impl AsRef<str>) -> PyErr {
    crate::value_err_code(reason_code!("invalid_argument"), message)
}

/// The scientific target the artifact's carried digest is checked against: the mean of
/// variable 1 when variable 0 is set to `level` for the whole observed population, at
/// the decision time.
fn declared_target(level: f64) -> PyResult<ScientificTarget> {
    if !level.is_finite() {
        return Err(invalid("the declared intervention level must be finite"));
    }
    let fail = |error: &dyn std::fmt::Display| invalid(error.to_string());
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("subject").map_err(|e| fail(&e))?,
        OutcomeQuantity::mean(VariableId::from_raw(1), QuantityUnits::dimensionless())
            .map_err(|e| fail(&e))?,
        Conditioning::unconditional(),
    )
    .map_err(|e| fail(&e))?;
    let action =
        WorldAction::at_decision([Intervention::set(VariableId::from_raw(0), Value::f64(level))])
            .map_err(|e| fail(&e))?;
    let world = InterventionWorld::new(
        ScientificWorldId::from_raw(1),
        action,
        false,
        binding,
        TimeFrame::static_decision(),
    )
    .map_err(|e| fail(&e))?;
    Ok(ScientificTarget::new(
        TargetForm::World(world),
        ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new()),
        InferenceSelection::Unspecified,
    ))
}

fn release_err(error: &PreviousReleaseError) -> PyErr {
    match error {
        PreviousReleaseError::ArtifactNotDecoded => crate::serialization_err(error.to_string()),
        _ => crate::unsupported_message(error.to_string()),
    }
}

fn receipt_err(error: &RecalcReceiptArtifactError) -> PyErr {
    crate::serialization_err(error.to_string())
}

/// Score-row digest of eight-byte row ids, as the current reader computes it.
#[pyfunction]
fn current_score_row_digest_bytes(row_ids: Vec<u64>) -> Vec<u8> {
    current_score_row_digest(&row_ids).to_vec()
}

/// Score-row digest of four-byte row ids, the encoding a previous release retained.
#[pyfunction]
fn narrow_score_row_digest_bytes(row_ids: Vec<u32>) -> Vec<u8> {
    narrow_score_row_digest(&row_ids).to_vec()
}

/// Identity of the scientific target set to `level`, which a carried digest must equal.
#[pyfunction]
fn declared_target_digest(level: f64) -> PyResult<Vec<u8>> {
    Ok(declared_target(level)?.identity().as_bytes().to_vec())
}

/// Adopt a previous-release artifact against the target set to `level`.
#[pyfunction]
fn consume_previous_release(artifact: Vec<u8>, level: f64) -> PyResult<Adopted> {
    let target = declared_target(level)?;
    let adopted =
        consume_previous_release_artifact(&artifact, &target).map_err(|e| release_err(&e))?;
    Ok((
        adopted.score_row_digest.to_vec(),
        adopted.target_digest.to_vec(),
        adopted.boundary_name,
        adopted.stage_status,
    ))
}

fn assignment_json(assignment: &MethodAssignment) -> Json {
    json!({
        "method": assignment.method,
        "estimator": assignment.estimator,
        "status": assignment.status,
        "axes": assignment.axes.iter().map(|axis| axis.as_str()).collect::<Vec<_>>(),
        "narrowed": assignment.narrowed,
    })
}

fn outcome_json(outcome: &RouteRecordOutcome) -> Json {
    match outcome {
        RouteRecordOutcome::RefusalKept => json!({"outcome": "refusal_kept"}),
        RouteRecordOutcome::Retired => json!({"outcome": "retired"}),
        RouteRecordOutcome::OneToOne { method, unserved_axes } => json!({
            "outcome": "one_to_one",
            "methods": [assignment_json(method)],
            "unserved_axes": unserved_axes.iter().map(|axis| axis.as_str()).collect::<Vec<_>>(),
        }),
        RouteRecordOutcome::Split { methods, unserved_axes } => json!({
            "outcome": "split",
            "methods": methods.iter().map(assignment_json).collect::<Vec<_>>(),
            "unserved_axes": unserved_axes.iter().map(|axis| axis.as_str()).collect::<Vec<_>>(),
        }),
    }
}

fn break_json(error: &RouteRecordError) -> Json {
    let (kind, methods): (&str, &[String]) = match error {
        RouteRecordError::UnknownStatus { .. } => ("unknown_status", &[]),
        RouteRecordError::RouteWithoutCurrentMethod { .. } => ("route_without_current_method", &[]),
        RouteRecordError::LicenseNotCarried { methods, .. } => ("license_not_carried", methods),
    };
    json!({"outcome": "break", "kind": kind, "unlicensed_methods": methods, "message": error.to_string()})
}

/// Carry every record of a previous-release route artifact onto the current methods.
///
/// Returns a JSON list in artifact order; each element has the record `id` and either an
/// outcome (`one_to_one`, `split`, `retired`, `refusal_kept`) or a `break` that names its
/// remedy. One break does not hide the other records.
#[pyfunction]
fn consume_previous_routes(artifact: Vec<u8>, methods: Vec<MethodRow>) -> PyResult<String> {
    let current: Vec<CurrentMethod> = methods
        .into_iter()
        .map(|(id, route, estimator, design, inference, status)| CurrentMethod {
            id,
            route,
            estimator,
            design,
            inference,
            status,
        })
        .collect();
    let migrated =
        consume_previous_route_records(&artifact, &current).map_err(|e| release_err(&e))?;
    let rows: Vec<Json> = migrated
        .iter()
        .map(|record| {
            let mut value = match &record.result {
                Ok(outcome) => outcome_json(outcome),
                Err(error) => break_json(error),
            };
            value["id"] = Json::from(record.id.clone());
            value
        })
        .collect();
    Ok(Json::Array(rows).to_string())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(current_score_row_digest_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(narrow_score_row_digest_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(declared_target_digest, m)?)?;
    m.add_function(wrap_pyfunction!(consume_previous_release, m)?)?;
    m.add_function(wrap_pyfunction!(consume_previous_routes, m)?)
}
