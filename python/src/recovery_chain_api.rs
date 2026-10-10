//! Bounded Python bridge for the 2.3 B2 ordered-response recovery row.
//!
//! Python names the m-graph (nodes, directed edges, bidirected edges, which a refusal
//! reports) and the roles of the two partially observed variables. The decision, the
//! checked plan, the verified nonrecoverability witness, the exact evaluation, the
//! digests, the artifact and every refusal are Rust's. A refusal comes back as structured
//! JSON for the Python layer to raise as its own exception type; corruption and unknown
//! versions raise `CausalSerializationError`.

use antecedent::analysis::recovery_chain::{
    ChainLawWire, ChainPartial, ChainPlanWire, ChainQueryWire, ChainRecoveredWire,
    ChainRecoveryDecision, ChainRecoveryDetail, ChainRecoveryError, ChainRecoveryQuery,
    ChainWitnessCheckWire, ChainWitnessWire, RecoveryChain, RecoveryChainArtifactError,
    RecoveryChainArtifactWire,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::transport_common::execution_context;
use crate::{CausalSerializationError, detach_catch, value_err};

type DecidePayload = (Option<String>, Option<Py<PyBytes>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);
type Roles = (String, String, String);

fn name_of(names: &[String], raw: u32) -> String {
    names.get(raw as usize).cloned().unwrap_or_else(|| format!("v{raw}"))
}

fn names_of(names: &[String], raw: &[u32]) -> Vec<String> {
    raw.iter().map(|r| name_of(names, *r)).collect()
}

fn route_refusal_json(error: &ChainRecoveryError) -> String {
    let mut message = error.message.clone();
    if let Some(receipt) = &error.receipt {
        message = format!("{message}; {}", receipt.summary());
    }
    serde_json::json!({
        "code": error.reason_code(),
        "stage": "recovery_chain",
        "detail": error.detail.detail(),
        "offending": serde_json::Value::Null,
        "remedy": serde_json::Value::Null,
        "message": message,
    })
    .to_string()
}

/// `None` for corruption (a serialization error), otherwise the refusal JSON.
fn artifact_refusal_json(error: &RecoveryChainArtifactError) -> Option<String> {
    match error {
        RecoveryChainArtifactError::Undecodable(_) => None,
        RecoveryChainArtifactError::Route(inner) => Some(route_refusal_json(inner)),
        other => {
            let (code, detail) = other.refusal();
            Some(
                serde_json::json!({
                    "code": code,
                    "stage": "consume",
                    "detail": detail,
                    "offending": serde_json::Value::Null,
                    "remedy": serde_json::Value::Null,
                    "message": other.to_string(),
                })
                .to_string(),
            )
        }
    }
}

fn model_json(
    model: &[antecedent::analysis::recovery_chain::ChainMechanismWire],
    names: &[String],
) -> Vec<serde_json::Value> {
    model
        .iter()
        .map(|m| {
            serde_json::json!({
                "node": name_of(names, m.node),
                "parents": names_of(names, &m.parents),
                "p_one_times_60": m.numerators,
            })
        })
        .collect()
}

struct ReportParts<'a> {
    names: &'a [String],
    query: &'a ChainQueryWire,
    outcome: &'static str,
    plan: Option<&'a ChainPlanWire>,
    observed: Option<&'a ChainLawWire>,
    recovered: Option<&'a ChainRecoveredWire>,
    witness: Option<&'a ChainWitnessWire>,
    check: Option<&'a ChainWitnessCheckWire>,
    digests: Option<(&'a str, &'a str)>,
}

fn report_json(parts: &ReportParts<'_>) -> String {
    let names = parts.names;
    let roles = |r: (u32, u32, u32)| {
        serde_json::json!({
            "variable": name_of(names, r.0),
            "response": name_of(names, r.1),
            "proxy": name_of(names, r.2),
        })
    };
    let witness = parts.witness.map(|w| {
        serde_json::json!({
            "self_censoring_edge": [name_of(names, w.edge.0), name_of(names, w.edge.1)],
            "first_model": model_json(&w.first, names),
            "second_model": model_json(&w.second, names),
        })
    });
    let check = parts.check.map(|c| {
        serde_json::json!({
            "observed_cells_equal": c.observed_cells,
            "differing_target_cell": [c.differing_cell.0, c.differing_cell.1],
            "target_masses": [c.masses.0, c.masses.1],
            "denominator": c.denominator,
        })
    });
    let (reason, detail) = if parts.outcome == "nonrecoverable" {
        (Some("transport_proven_non_transportable"), Some("recovery_chain.nonrecoverable_witness"))
    } else {
        (None, None)
    };
    serde_json::json!({
        "outcome": parts.outcome,
        "route": "ordered_response_recovery_not_mar_ipcw",
        "query": {"first": roles(parts.query.first), "second": roles(parts.query.second)},
        "plan": parts.plan,
        "observed": parts.observed,
        "recovered": parts.recovered,
        "witness": witness,
        "witness_check": check,
        "premises_digest": parts.digests.map(|d| d.0),
        "data_digest": parts.digests.map(|d| d.1),
        "reason": reason,
        "detail": detail,
        // Exact laws carry no sampling uncertainty: the route is point-only.
        "interval": {"available": false, "status": "point_only"},
    })
    .to_string()
}

fn report_of_wire(wire: &RecoveryChainArtifactWire) -> String {
    report_json(&ReportParts {
        names: &wire.variable_names,
        query: &wire.query,
        outcome: if wire.outcome == "recovered" { "recovered" } else { "nonrecoverable" },
        plan: wire.plan.as_ref(),
        observed: wire.observed.as_ref(),
        recovered: wire.recovered.as_ref(),
        witness: wire.witness.as_ref(),
        check: wire.witness_check.as_ref(),
        digests: Some((&wire.premises_digest, &wire.data_digest)),
    })
}

fn build_graph(
    nodes: &[String],
    directed: &[(String, String)],
    bidirected: &[(String, String)],
) -> PyResult<Admg> {
    let index =
        |name: &str| {
            nodes.iter().position(|n| n == name).and_then(|i| u32::try_from(i).ok()).ok_or_else(
                || value_err(format!("recovery_chain.invalid_query: unknown node {name}")),
            )
        };
    let mut distinct = nodes.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != nodes.len() {
        return Err(value_err("recovery_chain.invalid_query: node names must be distinct"));
    }
    let mut graph = Admg::empty();
    for i in 0..nodes.len() {
        let raw = u32::try_from(i).map_err(|_| value_err("the graph is too large"))?;
        graph
            .add_node(NodeRef::Static(VariableId::from_raw(raw)))
            .map_err(|e| value_err(e.to_string()))?;
    }
    for (from, to) in directed {
        graph
            .insert_directed(DenseNodeId::from_raw(index(from)?), DenseNodeId::from_raw(index(to)?))
            .map_err(|e| value_err(format!("recovery_chain.invalid_query: {e}")))?;
    }
    for (a, b) in bidirected {
        graph
            .insert_bidirected(DenseNodeId::from_raw(index(a)?), DenseNodeId::from_raw(index(b)?))
            .map_err(|e| value_err(format!("recovery_chain.invalid_query: {e}")))?;
    }
    Ok(graph)
}

fn build_query(nodes: &[String], first: &Roles, second: &Roles) -> PyResult<ChainRecoveryQuery> {
    let id = |name: &str| {
        nodes
            .iter()
            .position(|n| n == name)
            .and_then(|i| u32::try_from(i).ok())
            .map(VariableId::from_raw)
            .ok_or_else(|| value_err(format!("recovery_chain.invalid_query: unknown node {name}")))
    };
    let part = |(x, r, p): &Roles| {
        Ok::<_, PyErr>(ChainPartial { variable: id(x)?, response: id(r)?, proxy: id(p)? })
    };
    Ok(ChainRecoveryQuery { first: part(first)?, second: part(second)? })
}

/// Decide, and when `law_json` is given evaluate, one ordered-response recovery:
/// `(report json, artifact bytes, refusal json)`.
///
/// A nonrecoverable decision is a result (the report carries the verified witness) and is
/// exported; a recovered decision without a law has no artifact. Refusals (unsupported
/// mechanism, invalid query, positivity, invalid observed law, budget) come back as JSON.
#[pyfunction]
#[pyo3(signature = (nodes, directed, bidirected, first, second, law_json=None, *, seed=0, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn recover_recovery_chain(
    py: Python<'_>,
    nodes: Vec<String>,
    directed: Vec<(String, String)>,
    bidirected: Vec<(String, String)>,
    first: Roles,
    second: Roles,
    law_json: Option<String>,
    seed: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<DecidePayload> {
    let graph = build_graph(&nodes, &directed, &bidirected)?;
    let query = build_query(&nodes, &first, &second)?;
    let law: Option<ChainLawWire> = law_json
        .map(|text| {
            serde_json::from_str(&text)
                .map_err(|e| value_err(format!("recovery_chain.invalid_observed_law: {e}")))
        })
        .transpose()?;
    let payload = detach_catch(py, move || {
        let ctx: ExecutionContext = execution_context(seed, memory_bytes, cancel);
        let step = || -> Result<(String, Option<Vec<u8>>), ChainRecoveryError> {
            let decided = RecoveryChain::decide(&graph, &query, &nodes, &ctx)?;
            let evaluate = matches!(decided.decision(), ChainRecoveryDecision::Recovered(_));
            let decided = match &law {
                Some(wire) if evaluate => decided.recover(wire.to_law()?)?,
                _ => decided,
            };
            match decided.export() {
                Ok(bytes) => {
                    let wire = RecoveryChainArtifactWire::decode(&bytes).map_err(|e| {
                        ChainRecoveryError::new(
                            ChainRecoveryDetail::InvalidDerivation,
                            e.to_string(),
                        )
                    })?;
                    Ok((report_of_wire(&wire), Some(bytes)))
                }
                // A recovered decision with no law: report the plan only.
                Err(_) => {
                    let ChainRecoveryDecision::Recovered(plan) = decided.decision() else {
                        unreachable!("only a recovered decision lacks an artifact");
                    };
                    let plan = ChainPlanWire::of_plan(plan);
                    let query = ChainQueryWire::from_query(&query);
                    let report = report_json(&ReportParts {
                        names: &nodes,
                        query: &query,
                        outcome: "recovered",
                        plan: Some(&plan),
                        observed: None,
                        recovered: None,
                        witness: None,
                        check: None,
                        digests: None,
                    });
                    Ok((report, None))
                }
            }
        };
        match step() {
            Ok((report, bytes)) => Ok((Some(report), bytes, None)),
            Err(error) => Ok((None, None, Some(route_refusal_json(&error)))),
        }
    })?;
    let (report, bytes, refusal) = payload;
    Ok((report, bytes.map(|b| PyBytes::new(py, &b).unbind()), refusal))
}

/// Consume an artifact by re-deciding, re-evaluating and re-verifying:
/// `(report json, refusal json)`. The names are the verified mapping the artifact stores.
#[pyfunction]
#[pyo3(signature = (artifact, *, seed=0, memory_bytes=None, cancel=None))]
fn consume_recovery_chain_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    seed: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ConsumePayload> {
    detach_catch(py, move || {
        let ctx = execution_context(seed, memory_bytes, cancel);
        match RecoveryChainArtifactWire::consume_typed(&artifact, &ctx) {
            Ok(consumed) => Ok((Some(report_of_wire(&consumed.wire)), None)),
            Err(error) => match artifact_refusal_json(&error) {
                Some(json) => Ok((None, Some(json))),
                None => Err(CausalSerializationError::new_err(error.to_string())),
            },
        }
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(recover_recovery_chain, m)?)?;
    m.add_function(wrap_pyfunction!(consume_recovery_chain_artifact, m)?)
}
