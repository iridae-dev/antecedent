//! Python bridge for the forward sources of an inverse query and the associational
//! observational-scenario route.
//!
//! A forward source is a claim about the response of each declared action, adapted from a
//! composed mechanism model, a cohort-mean contrast or a transported response. Python
//! declares the actions' coordinates and queries and the transport receipt; the replay of every
//! composition query, the license of what the source can answer, the engine, the sealed
//! artifact and every refusal are Rust's. A refusal comes back as structured JSON for the
//! Python layer to raise as a typed exception.
//!
//! No source supplies an interval region, and a source's own standard error never becomes a
//! feasibility statement.

use antecedent::analysis::cohort_decision::{CohortExpectation, consume_cohort_decision_source};
use antecedent::analysis::inverse_query::{InverseQueryWire, InverseResultWire};
use antecedent::analysis::inverse_sources::{
    ComposedActionQuery, ComposedSourceRefusal, InverseDemand, LicensedForwardSource,
    ObservationalScenarioQuery, ObservationalScenarioResult, OutcomeEvent, ScenarioCell,
    ScenarioCondition, TransportReceipt, TransportedInputPolicy, cohort_source, evaluate,
    evaluate_observational, inspect_observational, inspect_origin, seal, seal_per_action,
    transported_source, verify_observational,
};
use antecedent::analysis::mechanism_composition::CompositionQueryWire;
use antecedent_core::reason_code;
use antecedent_design::decision_artifact::contract_from_json_refusal;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use crate::bridge::{RefusalJson, check_declared_size, external_refusal_json};
use crate::distribution_api::PyJointDistributionArtifact;
use crate::inverse_query_api::PyInverseQueryArtifact;
use crate::mechanism_composition_api::PreparedCompositionHandle;
use crate::transport_common::execution_context;
use crate::transported_response_api::TransportedResponseHandle;

type Refused<T> = (Option<T>, Option<String>);

const MAX_JSON_BYTES: usize = crate::bridge::MAX_DECLARED_ARTIFACT_BYTES;

fn check_size(text: &str, what: &str) -> PyResult<()> {
    check_declared_size(text, MAX_JSON_BYTES, what)
}

fn policy_of(accepts_transported: bool) -> TransportedInputPolicy {
    if accepts_transported {
        TransportedInputPolicy::AcceptsTransported
    } else {
        TransportedInputPolicy::NativeTargetOnly
    }
}

/// A forward source that passed its own checks, with the origin it carries.
#[pyclass(name = "ForwardSourceHandle", skip_from_py_object)]
pub(crate) struct ForwardSourceHandle {
    inner: LicensedForwardSource,
}

fn origin_json(source: &LicensedForwardSource) -> Json {
    let origin = source.origin();
    let inspection = inspect_origin(origin);
    json!({
        "origin": serde_json::to_value(origin).unwrap_or(Json::Null),
        "kind": inspection.kind,
        "representation": inspection.representation,
        "support": inspection.support,
        "interval_region": inspection.interval_region,
        "transported": inspection.transported,
        "source_populations": inspection.source_populations,
        "target_population": inspection.target_population,
        "assumed_mechanisms": inspection.assumed_mechanisms,
        "claim_digest": inspection.claim_digest,
        "source_identity": origin.source_identity,
        "provider_id": origin.provider_id,
        "per_action": source.action_claims().is_some(),
    })
}

#[pymethods]
impl ForwardSourceHandle {
    /// The origin and its plain-data inspection as JSON.
    fn origin_json(&self) -> String {
        origin_json(&self.inner).to_string()
    }
}

/// A composed-source refusal as JSON: the composition's own refusal (code, detail, offending,
/// expected, supplied, remedy) is the payload, with every refusal of the composition under
/// `refusals` and the refused action under `action_id`. The payload's detail is
/// `inverse_forward_source.origin_invalid` only when the replay of the exported query did not
/// reproduce its premises.
fn composed_refusal_json(refusal: &ComposedSourceRefusal) -> String {
    let all = refusal.refusals();
    let mut payload: Json =
        serde_json::from_str(&external_refusal_json(&refusal.to_refusal())).unwrap_or(Json::Null);
    if let Json::Object(map) = &mut payload {
        let each: Vec<Json> = all
            .iter()
            .filter_map(|r| serde_json::from_str(&external_refusal_json(r)).ok())
            .collect();
        map.insert("refusals".to_owned(), Json::Array(each));
        if let Some(action) = refusal.action_id() {
            map.insert("action_id".to_owned(), Json::String(action.to_owned()));
        }
    }
    payload.to_string()
}

fn declaration_refusal(detail: &str, message: &str) -> String {
    RefusalJson::new(reason_code!("invalid_argument"), "inverse_query", detail, message).json()
}

// ---------------------------------------------------------------------------
// Sources.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionIn {
    action_id: String,
    coordinate: ScientificQuantityWire,
    query: CompositionQueryWire,
    #[serde(default)]
    level_values: Vec<(String, f64)>,
}

/// A composition's replayed forward source: one query per declared action, each sealed and
/// consumed again before its result is read.
#[pyfunction]
#[pyo3(signature = (handle, actions_json, transport_json, accepts_transported, *, cancel=None))]
fn forward_source_composed(
    py: Python<'_>,
    handle: PyRef<'_, PreparedCompositionHandle>,
    actions_json: &str,
    transport_json: Option<&str>,
    accepts_transported: bool,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<ForwardSourceHandle>> {
    check_size(actions_json, "composed actions")?;
    let wires: Vec<ActionIn> = serde_json::from_str(actions_json)
        .map_err(|e| crate::value_err(format!("invalid composed actions: {e}")))?;
    let actions = wires
        .into_iter()
        .map(|wire| {
            Ok(ComposedActionQuery {
                action_id: wire.action_id,
                coordinate: antecedent_contract::ScientificQuantity::try_from(wire.coordinate)
                    .map_err(crate::value_err)?,
                query: wire.query,
                level_values: wire.level_values,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let transport: Option<TransportReceipt> = match transport_json {
        None => None,
        Some(text) => {
            check_size(text, "transport receipt")?;
            Some(
                serde_json::from_str(text)
                    .map_err(|e| crate::value_err(format!("invalid transport receipt: {e}")))?,
            )
        }
    };
    let inner = handle.inner.clone();
    let policy = policy_of(accepts_transported);
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(inner.inverse_forward_source_typed(&actions, transport, policy, &ctx))
    })?;
    Ok(match outcome {
        Ok(inner) => (Some(ForwardSourceHandle { inner }), None),
        Err(refusal) => (None, Some(composed_refusal_json(&refusal))),
    })
}

/// A cohort-mean contrast as a forward source, re-derived from the cohort source's artifact
/// bytes under the identity the caller retained.
#[pyfunction]
fn forward_source_cohort(
    artifact: &[u8],
    expected_identity: &str,
    accepts_transported: bool,
) -> Refused<ForwardSourceHandle> {
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
    match cohort_source(&source, policy_of(accepts_transported)) {
        Ok(inner) => (Some(ForwardSourceHandle { inner }), None),
        Err(refusal) => (None, Some(external_refusal_json(&refusal.to_refusal()))),
    }
}

/// A transported response as a forward source.
#[pyfunction]
fn forward_source_transported(
    response: PyRef<'_, TransportedResponseHandle>,
    accepts_transported: bool,
) -> Refused<ForwardSourceHandle> {
    match transported_source(&response.inner, policy_of(accepts_transported)) {
        Ok(inner) => (Some(ForwardSourceHandle { inner }), None),
        Err(refusal) => (None, Some(external_refusal_json(&refusal.to_refusal()))),
    }
}

// ---------------------------------------------------------------------------
// Evaluate and seal.
// ---------------------------------------------------------------------------

fn query_of(
    contract_json: &str,
    query_json: &str,
) -> PyResult<Result<antecedent::analysis::inverse_query::InverseQuery, String>> {
    check_size(contract_json, "inverse query contract")?;
    check_size(query_json, "inverse query")?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok(Err(external_refusal_json(&refusal))),
    };
    let wire: InverseQueryWire = match serde_json::from_str(query_json) {
        Ok(wire) => wire,
        Err(error) => {
            return Ok(Err(declaration_refusal(
                "inverse_query.invalid_declaration",
                &error.to_string(),
            )));
        }
    };
    Ok(Ok(wire.into_query(contract)))
}

/// Evaluate an inverse query on a source: the result table beside the source's origin.
#[pyfunction]
fn forward_source_evaluate(
    source: PyRef<'_, ForwardSourceHandle>,
    contract_json: &str,
    query_json: &str,
    interval_region: bool,
) -> PyResult<Refused<String>> {
    let query = match query_of(contract_json, query_json)? {
        Ok(query) => query,
        Err(refusal) => return Ok((None, Some(refusal))),
    };
    let demand = InverseDemand { interval_region };
    Ok(match evaluate(&query, &source.inner, demand) {
        Ok(sourced) => {
            let result = InverseResultWire::from(&sourced.result);
            (
                Some(
                    json!({
                        "result": serde_json::to_value(&result).unwrap_or(Json::Null),
                        "source": origin_json(&source.inner),
                    })
                    .to_string(),
                ),
                None,
            )
        }
        Err(refusal) => (None, Some(external_refusal_json(&refusal.to_refusal()))),
    })
}

/// Evaluate and seal a query on a source that has one claim for every action, binding the
/// origin into the artifact.
#[pyfunction]
fn forward_source_seal(
    source: PyRef<'_, ForwardSourceHandle>,
    contract_json: &str,
    query_json: &str,
    interval_region: bool,
) -> PyResult<Refused<PyInverseQueryArtifact>> {
    let query = match query_of(contract_json, query_json)? {
        Ok(query) => query,
        Err(refusal) => return Ok((None, Some(refusal))),
    };
    let demand = InverseDemand { interval_region };
    Ok(match seal(query, &source.inner, demand) {
        Ok(artifact) => (Some(PyInverseQueryArtifact::from_sealed(artifact)), None),
        Err(refusal) => (None, Some(external_refusal_json(&refusal.to_refusal()))),
    })
}

/// Evaluate and seal one single-action artifact per answered action.
#[pyfunction]
fn forward_source_seal_per_action(
    source: PyRef<'_, ForwardSourceHandle>,
    contract_json: &str,
    query_json: &str,
    interval_region: bool,
) -> PyResult<Refused<Vec<(String, PyInverseQueryArtifact)>>> {
    let query = match query_of(contract_json, query_json)? {
        Ok(query) => query,
        Err(refusal) => return Ok((None, Some(refusal))),
    };
    let demand = InverseDemand { interval_region };
    Ok(match seal_per_action(&query, &source.inner, demand) {
        Ok(sealed) => (
            Some(
                sealed
                    .into_iter()
                    .map(|(action, artifact)| {
                        (action, PyInverseQueryArtifact::from_sealed(artifact))
                    })
                    .collect(),
            ),
            None,
        ),
        Err(refusal) => (None, Some(external_refusal_json(&refusal.to_refusal()))),
    })
}

// ---------------------------------------------------------------------------
// Observational scenarios.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConditionIn {
    coordinate: ScientificQuantityWire,
    lower: f64,
    upper: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioIn {
    id: String,
    conditions: Vec<ConditionIn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EventIn {
    direction: String,
    threshold: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioQueryIn {
    outcome: ScientificQuantityWire,
    event: EventIn,
    scenarios: Vec<ScenarioIn>,
}

fn scenario_query_of(text: &str) -> PyResult<ObservationalScenarioQuery> {
    check_size(text, "scenario query")?;
    let wire: ScenarioQueryIn = serde_json::from_str(text)
        .map_err(|e| crate::value_err(format!("invalid scenario query: {e}")))?;
    let quantity = |wire: ScientificQuantityWire| {
        antecedent_contract::ScientificQuantity::try_from(wire).map_err(crate::value_err)
    };
    let event = match wire.event.direction.as_str() {
        "at_least" => OutcomeEvent::AtLeast(wire.event.threshold),
        "at_most" => OutcomeEvent::AtMost(wire.event.threshold),
        other => {
            return Err(crate::value_err(format!("unknown outcome event direction {other:?}")));
        }
    };
    let scenarios = wire
        .scenarios
        .into_iter()
        .map(|scenario| {
            Ok(ScenarioCell {
                id: scenario.id,
                conditions: scenario
                    .conditions
                    .into_iter()
                    .map(|c| {
                        Ok(ScenarioCondition {
                            coordinate: quantity(c.coordinate)?,
                            lower: c.lower,
                            upper: c.upper,
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(ObservationalScenarioQuery { outcome: quantity(wire.outcome)?, event, scenarios })
}

/// An associational answer: it ranks no action and is not an intervention effect.
#[pyclass(name = "ObservationalScenarioHandle", skip_from_py_object)]
pub(crate) struct ObservationalScenarioHandle {
    query: ObservationalScenarioQuery,
    result: ObservationalScenarioResult,
}

#[pymethods]
impl ObservationalScenarioHandle {
    /// Every value of the result and its inspection, as JSON.
    fn summary_json(&self) -> String {
        let inspection = inspect_observational(&self.result);
        json!({
            "standing": inspection.standing,
            "ranks_actions": inspection.ranks_actions,
            "is_intervention_effect": inspection.is_intervention_effect,
            "event_mass": self.result.event_mass,
            "covered_event_mass": self.result.covered_event_mass,
            "uncovered_event_mass": inspection.uncovered_event_mass,
            "effective_event_draws": self.result.effective_event_draws,
            "n_draws": self.result.n_draws,
            "exact": inspection.exact,
            "standard_error": self.result.standard_error,
            "snapshot_id": self.result.snapshot_id,
            "provider_id": self.result.provider_id,
            "scope_note": inspection.scope_note,
            "identity": self.result.identity,
            "scenarios": self.result.scenarios.iter().map(|s| json!({
                "id": s.id,
                "mass": s.mass,
                "joint_mass": s.joint_mass,
                "event_given_scenario": s.event_given_scenario,
                "scenario_given_event": s.scenario_given_event,
                "lift": s.lift,
                "scenario_draws": s.scenario_draws,
                "joint_draws": s.joint_draws,
            })).collect::<Vec<_>>(),
        })
        .to_string()
    }

    /// Re-derive the stored result from a law and the declared query; `None` when it replays,
    /// the refusal JSON otherwise.
    fn verify(&self, law: PyRef<'_, PyJointDistributionArtifact>) -> Option<String> {
        verify_observational(&self.result, &self.query, law.inner())
            .err()
            .map(|error| external_refusal_json(&error.to_refusal()))
    }
}

/// Answer an observational-scenario question on an aligned observational law.
#[pyfunction]
fn observational_scenarios_evaluate(
    law: PyRef<'_, PyJointDistributionArtifact>,
    query_json: &str,
) -> PyResult<Refused<ObservationalScenarioHandle>> {
    let query = scenario_query_of(query_json)?;
    Ok(match evaluate_observational(&query, law.inner()) {
        Ok(result) => (Some(ObservationalScenarioHandle { query, result }), None),
        Err(error) => (None, Some(external_refusal_json(&error.to_refusal()))),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ForwardSourceHandle>()?;
    module.add_class::<ObservationalScenarioHandle>()?;
    module.add_function(wrap_pyfunction!(forward_source_composed, module)?)?;
    module.add_function(wrap_pyfunction!(forward_source_cohort, module)?)?;
    module.add_function(wrap_pyfunction!(forward_source_transported, module)?)?;
    module.add_function(wrap_pyfunction!(forward_source_evaluate, module)?)?;
    module.add_function(wrap_pyfunction!(forward_source_seal, module)?)?;
    module.add_function(wrap_pyfunction!(forward_source_seal_per_action, module)?)?;
    module.add_function(wrap_pyfunction!(observational_scenarios_evaluate, module)?)?;
    Ok(())
}
