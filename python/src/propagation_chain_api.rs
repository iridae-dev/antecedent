//! Python bridge for uncertainty propagated through a chain of transformations.
//!
//! Python hands over the declaration as the record wire (JSON with IEEE bit
//! patterns), and the rule, the arithmetic, the identity, the replay and every
//! refusal are Rust's. A refusal comes back as structured JSON, with its
//! registered code, namespaced detail and next attempt, for the Python layer
//! to raise as a typed exception.

use antecedent::analysis::propagation_chain::{
    ChainRefusal, PropagationChain, compile_propagation_chain, consume_propagation_chain,
    independent_product_table,
};
use antecedent_io::propagation_chain_record::{ChainChangeWire, PropagationChainRecordWire};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::json;

use crate::PyCancellationToken;
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);
/// Exported record bytes, the record as JSON and the lowered program identity.
type ChainParts = (Py<PyBytes>, String, String);

const STAGE: &str = "propagate_chain";

fn refusal_json(refusal: &ChainRefusal) -> String {
    json!({
        "code": refusal.code(),
        "stage": STAGE,
        "detail": refusal.detail(),
        "message": refusal.to_string(),
        "offending": refusal.stage(),
        "next_attempt": refusal.next_attempt().name(),
        "remedy": refusal.next_attempt().sentence(),
    })
    .to_string()
}

fn invalid_json(message: String) -> String {
    json!({
        "code": antecedent_core::reason_code!("invalid_argument"),
        "stage": STAGE,
        "detail": "propagation_chain.invalid_request",
        "message": message,
        "offending": null,
        "next_attempt": "fix_chain_declaration",
        "remedy": antecedent_contract::propagation_chain::FIX_CHAIN_DECLARATION.sentence(),
    })
    .to_string()
}

fn chain_parts(py: Python<'_>, chain: &PropagationChain) -> Refused<ChainParts> {
    let bytes = match chain.export() {
        Ok(bytes) => bytes,
        Err(refusal) => return (None, Some(refusal_json(&refusal))),
    };
    match serde_json::to_string(chain.record()) {
        Ok(record) => (
            Some((
                PyBytes::new(py, &bytes).unbind(),
                record,
                chain.lowered_program_identity().to_hex(),
            )),
            None,
        ),
        Err(error) => (None, Some(invalid_json(error.to_string()))),
    }
}

fn parse_record(record_json: &str) -> Result<PropagationChainRecordWire, String> {
    serde_json::from_str(record_json).map_err(|error| invalid_json(error.to_string()))
}

/// Check a declared chain, compute every stage and seal the record.
#[pyfunction]
#[pyo3(signature = (record_json, cancel=None))]
fn propagation_chain_seal(
    py: Python<'_>,
    record_json: &str,
    cancel: Option<PyCancellationToken>,
) -> Refused<ChainParts> {
    let draft = match parse_record(record_json) {
        Ok(draft) => draft,
        Err(refusal) => return (None, Some(refusal)),
    };
    let ctx = execution_context(0, None, cancel);
    match PropagationChain::seal(draft, &ctx) {
        Ok(chain) => chain_parts(py, &chain),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// Append a change (a withdrawn source, or re-weighted rows) to a sealed
/// record and return the resealed record.
#[pyfunction]
#[pyo3(signature = (record_json, change_json, cancel=None))]
fn propagation_chain_change(
    py: Python<'_>,
    record_json: &str,
    change_json: &str,
    cancel: Option<PyCancellationToken>,
) -> Refused<ChainParts> {
    let mut draft = match parse_record(record_json) {
        Ok(record) => record,
        Err(refusal) => return (None, Some(refusal)),
    };
    let change: ChainChangeWire = match serde_json::from_str(change_json) {
        Ok(change) => change,
        Err(error) => return (None, Some(invalid_json(error.to_string()))),
    };
    draft.changes.push(change);
    let ctx = execution_context(0, None, cancel);
    match PropagationChain::seal(draft, &ctx) {
        Ok(chain) => chain_parts(py, &chain),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// Independently consume exported record bytes: the rule is checked again and
/// every stage replayed to identical bits. Returns the replayed record's
/// results as JSON.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None, cancel=None))]
fn propagation_chain_consume(
    artifact: &[u8],
    expected_identity: Option<&str>,
    cancel: Option<PyCancellationToken>,
) -> Refused<String> {
    let ctx = execution_context(0, None, cancel);
    match consume_propagation_chain(artifact, expected_identity, &ctx) {
        Ok(consumed) => (
            Some(
                json!({
                    "identity": consumed.identity,
                    "label": consumed.label,
                    "output": consumed.output,
                    "results": consumed.results,
                })
                .to_string(),
            ),
            None,
        ),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// Compile a declared chain without executing it and view every layer of its
/// lowered program as JSON.
#[pyfunction]
#[pyo3(signature = (record_json, cancel=None))]
fn propagation_chain_plan_view(
    record_json: &str,
    cancel: Option<PyCancellationToken>,
) -> Refused<String> {
    let draft = match parse_record(record_json) {
        Ok(draft) => draft,
        Err(refusal) => return (None, Some(refusal)),
    };
    let ctx = execution_context(0, None, cancel);
    match compile_propagation_chain(draft, &ctx) {
        Ok(plan) => (Some(plan.plan_view().to_json().to_string()), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// The stages that depend on a source of a declared chain, from the compiled plan.
#[pyfunction]
fn propagation_chain_dependents(record_json: &str, source: &str) -> Refused<Vec<String>> {
    let draft = match parse_record(record_json) {
        Ok(draft) => draft,
        Err(refusal) => return (None, Some(refusal)),
    };
    let ctx = execution_context(0, None, None);
    match compile_propagation_chain(draft, &ctx) {
        Ok(plan) => {
            (Some(plan.dependents_of(source).into_iter().map(str::to_owned).collect()), None)
        }
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// The independent product of separately drawn columns as one aligned table.
#[pyfunction]
fn propagation_chain_product_table(
    columns: Vec<Vec<f64>>,
    max_rows: usize,
) -> Refused<Vec<Vec<f64>>> {
    match independent_product_table(&columns, max_rows) {
        Ok(table) => (Some(table), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(propagation_chain_seal, module)?)?;
    module.add_function(wrap_pyfunction!(propagation_chain_change, module)?)?;
    module.add_function(wrap_pyfunction!(propagation_chain_consume, module)?)?;
    module.add_function(wrap_pyfunction!(propagation_chain_plan_view, module)?)?;
    module.add_function(wrap_pyfunction!(propagation_chain_dependents, module)?)?;
    module.add_function(wrap_pyfunction!(propagation_chain_product_table, module)?)?;
    Ok(())
}
