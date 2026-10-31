//! Python bridge for transported responses: a value carried to a target population with the
//! receipt that says how, the decision made on one, and the assumption range of its point.
//!
//! Python declares the value, the coordinates and the transport declaration; the receipt
//! checks, the admissibility rules, the decision, the assumption range and every refusal are
//! Rust's. A refusal comes back as structured JSON for the Python layer to raise as a typed
//! exception. A transported value is never native target data, and nothing here supplies a law,
//! an interval or a calibration claim for it.

use antecedent::analysis::cohort_decision::{CohortExpectation, consume_cohort_decision_source};
use antecedent::analysis::inverse_sources::{
    TransportDeclaration, TransportedResponse, TransportedRoute, TransportedValue,
};
use antecedent::analysis::transport_sensitivity::{
    TransportRoute, TransportSensitivityEntryRefusal, TransportSensitivityRequest, ViolationFamily,
    transport_range, transport_sensitivity_artifact,
};
use antecedent::analysis::{BoundDecisionContract, SupportPolicy, TransportedDemand};
use antecedent_contract::ScientificQuantity;
use antecedent_core::SupportStatus;
use antecedent_design::composition_boundary::UnsupportedActionPolicy;
use antecedent_design::decision_robust_artifact::admissible_contract_from_json_refusal;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use crate::bridge::{RefusalJson, check_declared_size, external_refusal_json};
use crate::distribution_api::PyJointDistributionArtifact;
use crate::prepared_api::PyPreparedAnalysis;
use crate::sensitivity_decision_api::PySensitivityArtifact;

type Refused<T> = (Option<T>, Option<String>);

const MAX_JSON_BYTES: usize = crate::bridge::MAX_DECLARED_ARTIFACT_BYTES;

fn check_size(text: &str, what: &str) -> PyResult<()> {
    check_declared_size(text, MAX_JSON_BYTES, what)
}

/// A transported response a decision, an inverse query or a scene reads.
#[pyclass(name = "TransportedResponseHandle", skip_from_py_object)]
pub(crate) struct TransportedResponseHandle {
    pub(crate) inner: TransportedResponse,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclarationIn {
    route: TransportedRoute,
    source_identity: String,
    source_populations: Vec<String>,
    target_population: String,
    transport_identities: Vec<String>,
    assumptions: Vec<String>,
    support: String,
    #[serde(default)]
    uncalibrated_standard_error: Option<f64>,
    provider_id: String,
    snapshot_id: String,
    causal_contract_id: String,
}

fn declaration_of(text: &str) -> PyResult<TransportDeclaration> {
    check_size(text, "transport declaration")?;
    let wire: DeclarationIn = serde_json::from_str(text)
        .map_err(|e| crate::value_err(format!("invalid transport declaration: {e}")))?;
    let support = SupportStatus::from_name(&wire.support)
        .ok_or_else(|| crate::value_err(format!("unknown support status {:?}", wire.support)))?;
    Ok(TransportDeclaration {
        route: wire.route,
        source_identity: wire.source_identity,
        source_populations: wire.source_populations,
        target_population: wire.target_population,
        transport_identities: wire.transport_identities,
        assumptions: wire.assumptions,
        support,
        uncalibrated_standard_error: wire.uncalibrated_standard_error,
        provider_id: wire.provider_id,
        snapshot_id: wire.snapshot_id,
        causal_contract_id: wire.causal_contract_id,
    })
}

pub(crate) fn quantities_of(text: &str, what: &str) -> PyResult<Vec<ScientificQuantity>> {
    check_size(text, what)?;
    let wires: Vec<ScientificQuantityWire> =
        serde_json::from_str(text).map_err(|e| crate::value_err(format!("invalid {what}: {e}")))?;
    wires
        .into_iter()
        .map(|wire| ScientificQuantity::try_from(wire).map_err(crate::value_err))
        .collect()
}

pub(crate) fn quantity_of(text: &str, what: &str) -> PyResult<ScientificQuantity> {
    check_size(text, what)?;
    let wire: ScientificQuantityWire =
        serde_json::from_str(text).map_err(|e| crate::value_err(format!("invalid {what}: {e}")))?;
    ScientificQuantity::try_from(wire).map_err(crate::value_err)
}

fn value_json(response: &TransportedResponse) -> PyResult<Json> {
    let coordinates: Vec<Json> = response
        .coordinates()
        .iter()
        .map(|q| {
            serde_json::to_value(ScientificQuantityWire::from(q))
                .map_err(|error| crate::serialization_err(error.to_string()))
        })
        .collect::<PyResult<_>>()?;
    Ok(match response.value() {
        TransportedValue::Means(mean) => json!({
            "kind": "point",
            "coordinates": coordinates,
            "means": mean.means,
            "provider_id": mean.provider_id,
            "snapshot_id": mean.snapshot_id,
            "causal_contract_id": mean.causal_contract_id,
        }),
        TransportedValue::Law(_) => json!({"kind": "law", "coordinates": coordinates}),
    })
}

#[pymethods]
impl TransportedResponseHandle {
    /// The receipt (route, populations, transport identities, assumptions, support label,
    /// representation, uncalibrated standard error) as JSON.
    fn receipt_json(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.receipt())
            .map_err(|error| crate::serialization_err(error.to_string()))
    }

    /// The value (coordinates and means for a point, coordinates for a law) as JSON.
    fn value_json(&self) -> PyResult<String> {
        Ok(value_json(&self.inner)?.to_string())
    }

    /// Canonical identity of the receipt.
    fn identity(&self) -> String {
        self.inner.receipt().canonical_identity()
    }
}

fn transported_refusal_json(refusal: &antecedent::analysis::TransportedRefusal) -> String {
    external_refusal_json(&refusal.to_refusal())
}

/// A transported point: one mean per coordinate.
#[pyfunction]
fn transported_response_declared_point(
    coordinates_json: &str,
    means: Vec<f64>,
    declaration_json: &str,
) -> PyResult<Refused<TransportedResponseHandle>> {
    let coordinates = quantities_of(coordinates_json, "transported coordinates")?;
    let declaration = declaration_of(declaration_json)?;
    Ok(match TransportedResponse::declared_point(coordinates, means, &declaration) {
        Ok(inner) => (Some(TransportedResponseHandle { inner }), None),
        Err(refusal) => (None, Some(transported_refusal_json(&refusal))),
    })
}

/// A transported law: aligned draws under a declared transport model.
#[pyfunction]
fn transported_response_declared_law(
    law: PyRef<'_, PyJointDistributionArtifact>,
    declaration_json: &str,
) -> PyResult<Refused<TransportedResponseHandle>> {
    let declaration = declaration_of(declaration_json)?;
    Ok(match TransportedResponse::declared_law(law.inner().clone(), &declaration) {
        Ok(inner) => (Some(TransportedResponseHandle { inner }), None),
        Err(refusal) => (None, Some(transported_refusal_json(&refusal))),
    })
}

/// The transported response of a cohort-decision source built under a declared covariate
/// shift, re-derived from the source's artifact bytes under the retained identity.
#[pyfunction]
fn transported_response_from_shifted_cohort(
    artifact: &[u8],
    expected_identity: &str,
) -> Refused<TransportedResponseHandle> {
    let expectation = CohortExpectation {
        identity: expected_identity.to_owned(),
        parent_claim: None,
        population_id: None,
        outcome_units: None,
        contrast_regime: None,
        cohort_digest: None,
    };
    let source = match consume_cohort_decision_source(artifact, &expectation) {
        Ok(source) => source,
        Err(error) => {
            return (
                None,
                Some(
                    RefusalJson::new(error.code, "consume", error.detail, &error.to_string())
                        .offending(Some(&error.subject))
                        .remedy(Some(error.remedy))
                        .json(),
                ),
            );
        }
    };
    match TransportedResponse::from_shifted_cohort(&source) {
        Ok(inner) => (Some(TransportedResponseHandle { inner }), None),
        Err(refusal) => (None, Some(transported_refusal_json(&refusal))),
    }
}

fn parse_demand(name: &str) -> PyResult<TransportedDemand> {
    match name {
        "expected_utility" => Ok(TransportedDemand::ExpectedUtility),
        "law" => Ok(TransportedDemand::Law),
        "interval" => Ok(TransportedDemand::Interval),
        other => Err(crate::value_err(format!("unknown transported demand {other:?}"))),
    }
}

fn parse_unsupported(name: &str) -> PyResult<UnsupportedActionPolicy> {
    match name {
        "compare_supported" => Ok(UnsupportedActionPolicy::CompareSupported),
        "require_all" => Ok(UnsupportedActionPolicy::RequireAllActions),
        other => Err(crate::value_err(format!("unknown unsupported-action policy {other:?}"))),
    }
}

/// Decide an admissible contract on a transported response, bound to the identities of the
/// causal contract of a prepared analysis.
///
/// Returns the decision JSON (the support-aware decision, the receipt, the assumptions and the
/// bound identity) or a structured refusal.
#[pyfunction]
fn transported_decision_evaluate(
    prepared: PyRef<'_, PyPreparedAnalysis>,
    declaration_json: &str,
    response: PyRef<'_, TransportedResponseHandle>,
    demand: &str,
    unsupported: &str,
    weakest_support: &str,
) -> PyResult<Refused<String>> {
    check_size(declaration_json, "decision declaration")?;
    let demand = parse_demand(demand)?;
    let policy = SupportPolicy {
        unsupported: parse_unsupported(unsupported)?,
        weakest_support: SupportStatus::from_name(weakest_support).ok_or_else(|| {
            crate::value_err(format!("unknown support status {weakest_support:?}"))
        })?,
    };
    let admissible = match admissible_contract_from_json_refusal(declaration_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(external_refusal_json(&refusal)))),
    };
    let causal = prepared.inner.inspect().map_err(crate::py_err)?;
    let bound = match BoundDecisionContract::bind(&causal, admissible) {
        Ok(bound) => bound,
        Err(refusal) => return Ok((None, Some(external_refusal_json(&refusal)))),
    };
    Ok(match bound.evaluate_transported(&response.inner, demand, &policy) {
        Ok(decided) => {
            let receipt = serde_json::to_value(decided.receipt())
                .map_err(|error| crate::serialization_err(error.to_string()))?;
            let inner = decided.decision();
            let inspection = decided.inspect();
            (
                Some(
                    json!({
                        "decision": crate::composition_api::supported_json(&inner.decision),
                        "receipt": receipt,
                        "assumptions": inner.assumptions,
                        "inference": inner.inference,
                        "native_target_data": inspection.native_target_data,
                        "identity": decided.identity(),
                        "decision_identity": inner.identity,
                        "bound_identity": decided.bound_identity(),
                        "causal_target": decided.causal_target(),
                        "causal_identification": decided.causal_identification(),
                    })
                    .to_string(),
                ),
                None,
            )
        }
        Err(refusal) => (None, Some(external_refusal_json(&refusal))),
    })
}

fn entry_refusal_json(error: &TransportSensitivityEntryRefusal) -> String {
    RefusalJson::new(
        error.reason_code(),
        "transport_sensitivity",
        error.detail(),
        &error.to_string(),
    )
    .remedy(Some(error.remedy()))
    .json()
}

/// The assumption range of one coordinate of a transported point and the sealed sensitivity
/// artifact of the decision between acting on it and keeping a reference value.
///
/// Returns `(artifact, range json, refusal json)`.
#[pyfunction]
#[pyo3(signature = (response, family, bound, grid_points, reference_json, reference_value, *, coordinate=0, route=None))]
#[allow(clippy::too_many_arguments)] // Mirrors the Python signature.
fn transport_sensitivity_build(
    response: PyRef<'_, TransportedResponseHandle>,
    family: &str,
    bound: f64,
    grid_points: usize,
    reference_json: &str,
    reference_value: f64,
    coordinate: usize,
    route: Option<&str>,
) -> PyResult<(Option<PySensitivityArtifact>, Option<String>, Option<String>)> {
    let family = match family {
        "additive_shift" => ViolationFamily::AdditiveShift { bound },
        "proportional_shift" => ViolationFamily::ProportionalShift { bound },
        other => return Err(crate::value_err(format!("unknown violation family {other:?}"))),
    };
    let route = match route {
        None => None,
        Some(name) => Some(
            TransportRoute::ALL
                .into_iter()
                .find(|candidate| candidate.as_str() == name)
                .ok_or_else(|| crate::value_err(format!("unknown transport route {name:?}")))?,
        ),
    };
    let reference = quantity_of(reference_json, "reference quantity")?;
    let request = TransportSensitivityRequest { coordinate, route, family, grid_points };
    let range = match transport_range(&response.inner, &request) {
        Ok(range) => range,
        Err(error) => return Ok((None, None, Some(entry_refusal_json(&error)))),
    };
    let artifact = match transport_sensitivity_artifact(
        &response.inner,
        &request,
        &reference,
        reference_value,
    ) {
        Ok(artifact) => artifact,
        Err(error) => return Ok((None, None, Some(entry_refusal_json(&error)))),
    };
    let (low, high) = range.range_at_bound();
    let summary = json!({
        "route": range.route().as_str(),
        "premise": range.premise().as_str(),
        "family": range.family().scale(),
        "bound": range.family().bound(),
        "point_value": range.point_value(),
        "grid": range.grid(),
        "lower": range.lower(),
        "upper": range.upper(),
        "range_at_bound": [low, high],
        "reference_value": reference_value,
        "reference_reached_at": range.reference_reached_at(reference_value),
        "statement": range.statement(),
        "identity": range.identity().to_hex(),
    });
    Ok((
        Some(PySensitivityArtifact { artifact, source: std::sync::Mutex::new(None), recipe: None }),
        Some(summary.to_string()),
        None,
    ))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TransportedResponseHandle>()?;
    module.add_function(wrap_pyfunction!(transported_response_declared_point, module)?)?;
    module.add_function(wrap_pyfunction!(transported_response_declared_law, module)?)?;
    module.add_function(wrap_pyfunction!(transported_response_from_shifted_cohort, module)?)?;
    module.add_function(wrap_pyfunction!(transported_decision_evaluate, module)?)?;
    module.add_function(wrap_pyfunction!(transport_sensitivity_build, module)?)?;
    Ok(())
}
