//! Python bridge for scientific interface contracts.
//!
//! Python declares contracts as JSON in the artifact's wire layout (ports sorted
//! by name; the Python layer sorts them); the validation, the connection and
//! composition decisions, the identities and the artifact are Rust's. Refusals
//! come back as structured JSON for the Python layer to raise as typed
//! exceptions; a malformed JSON document is an input error.
//!
//! A connection never raises a calibration label and nothing here produces an
//! interval or posterior.

use antecedent::analysis::interface_contract::{
    CalibrationStatus, ComponentConnection, ComposedInterfaces, Connection, EvidenceBasis,
    InterfaceContract, check_interface_composition_with_ledger,
};
use antecedent::analysis::{
    InterfaceContractRefusal, check_interface_composition, check_interface_connection,
    consume_interface_contract_artifact, export_interface_contract,
    inspect_interface_contract_artifact,
};
use antecedent_io::interface_contract_artifact::{
    ContractWire, contract_from_wire, contract_to_wire,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Value as Json, json};

use crate::evidence_ledger_api::EvidenceLedgerHandle;
use crate::measurement_model_api::parse;

fn refusal_value(refusal: &InterfaceContractRefusal) -> Json {
    json!({
        "code": refusal.code,
        "stage": "interface_contract",
        "detail": refusal.detail,
        "message": refusal.message,
        "offending": refusal.offending,
        "expected": refusal.expected,
        "supplied": refusal.supplied,
        "remedy": refusal.remedy,
    })
}

fn refusal_json(refusal: &InterfaceContractRefusal) -> String {
    refusal_value(refusal).to_string()
}

fn refusals_json(refusals: &[InterfaceContractRefusal]) -> String {
    Json::Array(refusals.iter().map(refusal_value).collect()).to_string()
}

pub(crate) fn contract_of(text: &str, what: &str) -> PyResult<InterfaceContract> {
    let value = parse(text, what)?;
    let wire: ContractWire = serde_json::from_value(value)
        .map_err(|error| crate::value_err(format!("invalid {what} declaration: {error}")))?;
    contract_from_wire(&wire)
        .map_err(|error| crate::value_err(format!("invalid {what} declaration: {error}")))
}

pub(crate) fn connection_json(connection: &Connection) -> String {
    let calibration = match &connection.calibration {
        CalibrationStatus::Measured { record_id } => {
            json!({"status": "measured", "record_id": record_id, "input": null})
        }
        CalibrationStatus::Inherited { input } => {
            json!({"status": "inherited", "record_id": null, "input": input})
        }
        other => json!({"status": other.label(), "record_id": null, "input": null}),
    };
    let evidence = match &connection.evidence {
        EvidenceBasis::Undeclared => json!({"declared": false, "sources": []}),
        EvidenceBasis::Declared(sources) => json!({
            "declared": true,
            "sources": sources.iter().map(antecedent_contract::evidence_ledger::SourceId::to_hex).collect::<Vec<_>>(),
        }),
    };
    json!({
        "producer": connection.producer,
        "produced_port": connection.produced_port,
        "consumer": connection.consumer,
        "consumed_port": connection.consumed_port,
        "operations": connection.operations.iter().map(|o| o.as_str()).collect::<Vec<_>>(),
        "uncertainty": connection.uncertainty.map(|u| json!({
            "meaning": u.name(),
            "paradigm": u.paradigm().as_str(),
        })),
        "calibration": calibration,
        "evidence": evidence,
        "identity": connection.identity.to_hex(),
    })
    .to_string()
}

pub(crate) type Checked = (Option<String>, Option<String>);

/// The canonical identity of a declared contract, or its refusal.
#[pyfunction]
fn interface_contract_identity(contract_json: &str) -> PyResult<Checked> {
    let contract = contract_of(contract_json, "interface contract")?;
    Ok(match contract.validate() {
        Ok(()) => (Some(contract.canonical_identity().to_hex()), None),
        Err(refusal) => (
            None,
            Some(
                json!({
                    "code": refusal.code,
                    "stage": "interface_contract",
                    "detail": refusal.detail,
                    "message": refusal.remedy.unwrap_or_default(),
                    "offending": refusal.offending,
                    "expected": refusal.expected,
                    "supplied": refusal.supplied,
                    "remedy": refusal.remedy,
                })
                .to_string(),
            ),
        ),
    })
}

/// Decide one connection from the declarations alone; the refusals are a
/// JSON array.
#[pyfunction]
fn interface_contract_check_connection(
    producer_json: &str,
    produced_port: &str,
    consumer_json: &str,
    consumed_port: &str,
) -> PyResult<Checked> {
    let producer = contract_of(producer_json, "producer contract")?;
    let consumer = contract_of(consumer_json, "consumer contract")?;
    Ok(match check_interface_connection(&producer, produced_port, &consumer, consumed_port) {
        Ok(connection) => (Some(connection_json(&connection)), None),
        Err(refusals) => (None, Some(refusals_json(&refusals))),
    })
}

pub(crate) fn components_of(components_json: &str) -> PyResult<Vec<InterfaceContract>> {
    let components_value = parse(components_json, "components")?;
    let items = components_value
        .as_array()
        .ok_or_else(|| crate::value_err("components must be an array"))?;
    let mut components = Vec::with_capacity(items.len());
    for item in items {
        components.push(contract_of(&item.to_string(), "component contract")?);
    }
    Ok(components)
}

pub(crate) fn connections_of(connections_json: &str) -> PyResult<Vec<ComponentConnection>> {
    let connections_value = parse(connections_json, "connections")?;
    let entries = connections_value
        .as_array()
        .ok_or_else(|| crate::value_err("connections must be an array"))?;
    let mut connections = Vec::with_capacity(entries.len());
    for entry in entries {
        let parts: Vec<&str> = entry
            .as_array()
            .map(|a| a.iter().filter_map(Json::as_str).collect())
            .unwrap_or_default();
        let [producer, produced, consumer, consumed] = parts[..] else {
            return Err(crate::value_err(
                "a connection is [producer, produced_port, consumer, consumed_port]",
            ));
        };
        connections.push(ComponentConnection::new(producer, produced, consumer, consumed));
    }
    Ok(connections)
}

pub(crate) fn composed_checked(
    decided: Result<ComposedInterfaces, Vec<InterfaceContractRefusal>>,
) -> Checked {
    match decided {
        Ok(composed) => (
            Some(
                json!({
                    "order": composed.order,
                    "identity": composed.identity.to_hex(),
                    "connections": composed
                        .connections
                        .iter()
                        .map(|c| serde_json::from_str::<Json>(&connection_json(c)).unwrap_or(Json::Null))
                        .collect::<Vec<_>>(),
                })
                .to_string(),
            ),
            None,
        ),
        Err(refusals) => (None, Some(refusals_json(&refusals))),
    }
}

/// Check a composition: `components_json` is an array of contracts and
/// `connections_json` an array of `[producer, port, consumer, port]`.
#[pyfunction]
fn interface_contract_check_composition(
    components_json: &str,
    connections_json: &str,
) -> PyResult<Checked> {
    let components = components_of(components_json)?;
    let connections = connections_of(connections_json)?;
    Ok(composed_checked(check_interface_composition(&components, &connections)))
}

/// Check a composition against an evidence ledger: a requirement of disjoint evidence is decided
/// from the ledger's row-level observations, so sources that share observations are refused and a
/// source whose observations are not computable is unknown, never disjoint.
#[pyfunction]
fn interface_contract_check_composition_with_ledger(
    components_json: &str,
    connections_json: &str,
    ledger: PyRef<'_, EvidenceLedgerHandle>,
) -> PyResult<Checked> {
    let components = components_of(components_json)?;
    let connections = connections_of(connections_json)?;
    Ok(composed_checked(check_interface_composition_with_ledger(
        &components,
        &connections,
        ledger.ledger(),
    )))
}

type Exported = (Option<(Py<PyBytes>, String, String)>, Option<String>);

/// Seal a contract as `interface_contract_v1`.
#[pyfunction]
fn interface_contract_export(
    py: Python<'_>,
    contract_json: &str,
    artifact_id: &str,
) -> PyResult<Exported> {
    let contract = contract_of(contract_json, "interface contract")?;
    Ok(match export_interface_contract(&contract, artifact_id) {
        Ok(done) => (
            Some((PyBytes::new(py, &done.bytes).unbind(), done.identity, done.contract_identity)),
            None,
        ),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    })
}

/// Consume an artifact against a retained identity; the result is a JSON
/// document with the contract, its identity and the artifact identity.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None))]
fn interface_contract_consume(artifact: &[u8], expected_identity: Option<&str>) -> Checked {
    match consume_interface_contract_artifact(artifact, expected_identity) {
        Ok(done) => (
            Some(
                json!({
                    "contract": serde_json::to_value(contract_to_wire(&done.contract))
                        .unwrap_or(Json::Null),
                    "contract_identity": done.contract_identity,
                    "identity": done.identity,
                })
                .to_string(),
            ),
            None,
        ),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// Read the stored metadata without replaying it.
#[pyfunction]
fn interface_contract_inspect(artifact: &[u8]) -> Checked {
    match inspect_interface_contract_artifact(artifact) {
        Ok(meta) => (Some(serde_json::to_value(&meta).unwrap_or(Json::Null).to_string()), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(interface_contract_identity, module)?)?;
    module.add_function(wrap_pyfunction!(interface_contract_check_connection, module)?)?;
    module.add_function(wrap_pyfunction!(interface_contract_check_composition, module)?)?;
    module.add_function(wrap_pyfunction!(
        interface_contract_check_composition_with_ledger,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(interface_contract_export, module)?)?;
    module.add_function(wrap_pyfunction!(interface_contract_consume, module)?)?;
    module.add_function(wrap_pyfunction!(interface_contract_inspect, module)?)?;
    Ok(())
}
