//! Python bridge for uncertainty propagated through a chain of transformations.
//!
//! Python hands over the declaration as the record wire (JSON with IEEE bit
//! patterns), and the rule, the arithmetic, the identity, the replay and every
//! refusal are Rust's. A refusal comes back as structured JSON, with its
//! registered code, namespaced detail and next attempt, for the Python layer
//! to raise as a typed exception.

use antecedent::analysis::propagation_chain::{
    ChainRefusal, FamilyDependence, FamilyEstimate, LawMeaning, PropagationChain, SharedFactorKind,
    compile_propagation_chain, consume_propagation_chain, family_table, independent_product_table,
};
use antecedent_contract::evidence_ledger::SourceId;
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

/// One family estimate: id, estimate, standard error, law meaning, `(kind, id)` factors,
/// whether the factor list may be partial, and the authorizing ledger source as hex.
type FamilyEstimateArg = (String, f64, f64, String, Vec<(String, String)>, bool, Option<String>);
/// Chain sources as record-wire JSON, the joint covariance (row-major) and the
/// `(chain source, ledger source hex)` authority bindings.
type FamilyTableParts = (Vec<String>, Vec<f64>, Vec<(String, String)>);

fn family_estimates(estimates: Vec<FamilyEstimateArg>) -> Result<Vec<FamilyEstimate>, String> {
    let mut out = Vec::with_capacity(estimates.len());
    for (id, estimate, standard_error, meaning, factors, unknown_overlap, authority) in estimates {
        let meaning = LawMeaning::from_name(&meaning).ok_or_else(|| {
            format!("estimate `{id}` has meaning `{meaning}`; expected `sampling` or `posterior`")
        })?;
        let mut built = FamilyEstimate::new(&id, estimate, standard_error, meaning);
        for (kind, factor) in &factors {
            let kind = SharedFactorKind::from_name(kind)
                .ok_or_else(|| format!("estimate `{id}` names an unknown factor kind `{kind}`"))?;
            built = built.with_factor(kind, factor);
        }
        if unknown_overlap {
            built = built.with_unknown_overlap();
        }
        if let Some(hex) = authority {
            let source = SourceId::from_hex(&hex).ok_or_else(|| {
                format!("estimate `{id}` names `{hex}`, which is not a source identity")
            })?;
            built = built.with_authority(source);
        }
        out.push(built);
    }
    Ok(out)
}

/// Turn family estimates into one aligned joint-draw table with the dependence they share.
///
/// `dependence` is `independent`, `covariance` (with the row-major `covariance`) or
/// `ledger_overlap` (with the `ledger` that holds the authorizing sources).
#[pyfunction]
#[pyo3(signature = (alignment, estimates, dependence, covariance=None, ledger=None, rows=1597, seed=0))]
fn propagation_chain_family_table(
    py: Python<'_>,
    alignment: &str,
    estimates: Vec<FamilyEstimateArg>,
    dependence: &str,
    covariance: Option<Vec<f64>>,
    ledger: Option<PyRef<'_, crate::evidence_ledger_api::EvidenceLedgerHandle>>,
    rows: usize,
    seed: u64,
) -> PyResult<Refused<FamilyTableParts>> {
    // The Python arguments are already extracted by PyO3; honor a scoped
    // cancellation before converting them into owned native family values.
    let ctx = crate::py_execution_context(seed, 1);
    if ctx.cancellation.is_cancelled() {
        return Err(crate::CausalCancelledError::new_err("family table cancelled"));
    }
    let estimates = match family_estimates(estimates) {
        Ok(estimates) => estimates,
        Err(message) => return Ok((None, Some(invalid_json(message)))),
    };
    let dependence = match (dependence, covariance) {
        ("independent", None) => FamilyDependence::Independent,
        ("covariance", Some(matrix)) => FamilyDependence::Covariance(matrix),
        ("ledger_overlap", None) => FamilyDependence::LedgerOverlap,
        (other, _) => {
            return Ok((
                None,
                Some(invalid_json(format!(
                    "dependence `{other}` with this covariance argument is not one of \
                     `independent`, `covariance` (with a covariance) or `ledger_overlap`"
                ))),
            ));
        }
    };
    if ctx.cancellation.is_cancelled() {
        return Err(crate::CausalCancelledError::new_err("family table cancelled"));
    }
    let alignment = alignment.to_owned();
    let ledger = ledger.as_ref().map(|handle| handle.ledger().clone());
    crate::detach_catch(py, move || {
        let table_result = antecedent_core::numerical_admission::with_context(&ctx, || {
            family_table(&alignment, &estimates, &dependence, ledger.as_ref(), rows, seed)
        });
        if ctx.cancellation.is_cancelled() {
            return Err(crate::CausalCancelledError::new_err("family table cancelled"));
        }
        let table = match table_result {
            Ok(table) => table,
            Err(refusal) if refusal.detail() == "propagation_chain.family_table_work_limit" => {
                return Err(crate::CausalResourceError::new_err(refusal.to_string()));
            }
            Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
        };
        let mut sources = Vec::with_capacity(table.sources.len());
        for source in &table.sources {
            match serde_json::to_string(source) {
                Ok(wire) => sources.push(wire),
                Err(error) => return Ok((None, Some(invalid_json(error.to_string())))),
            }
        }
        let bindings = table
            .authority
            .bindings()
            .iter()
            .map(|(chain_source, ledger_source)| (chain_source.clone(), ledger_source.to_hex()))
            .collect();
        Ok((Some((sources, table.covariance, bindings)), None))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(propagation_chain_seal, module)?)?;
    module.add_function(wrap_pyfunction!(propagation_chain_change, module)?)?;
    module.add_function(wrap_pyfunction!(propagation_chain_consume, module)?)?;
    module.add_function(wrap_pyfunction!(propagation_chain_plan_view, module)?)?;
    module.add_function(wrap_pyfunction!(propagation_chain_dependents, module)?)?;
    module.add_function(wrap_pyfunction!(propagation_chain_product_table, module)?)?;
    module.add_function(wrap_pyfunction!(propagation_chain_family_table, module)?)?;
    Ok(())
}
