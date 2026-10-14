//! Python bridge for transport across a discovered equivalence class.
//!
//! Python declares a static discovery contract as JSON (variables are positions in the supplied
//! columns), the searched columns, a mean-curve transport query and the selection targets. The
//! discovery, the enumeration of the class, every member's structural verdict and every refusal
//! are Rust's. A scientific refusal comes back as structured JSON for the Python layer to raise
//! as a typed exception; a failure of the discovery algorithm itself raises
//! `CausalDiscoveryError`; a malformed query raises the shared value error.
//!
//! The result is labelled `structural_envelope`: the structural verdict of each member of the
//! class, never a point, an interval or a probability over members.

use std::sync::Arc;

use antecedent::analysis::discovered_transport::{
    DEFAULT_ENVELOPE_MEMBER_BOUND, DiscoveredTransportError, DiscoveredTransportReport,
    DiscoveredTransportRun, EnvelopeStatus, MemberVerdict, discover_and_transport, formula_kind,
    transport_discovered_structure,
};
use antecedent_core::{
    ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_discovery::selective::{DiscoveryOutcome, SelectiveDiscoveryError};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use serde_json::{Map, Value as Json, json};

use crate::bridge::RefusalJson;
use crate::selective_discovery_api as sd;

const STAGE: &str = "discovered_transport";

type Reply = (Option<String>, Option<String>);

fn hex(bytes: &[u8; 32]) -> String {
    antecedent_discovery::selective::digest_hex(bytes)
}

fn status_label(status: EnvelopeStatus) -> &'static str {
    match status {
        EnvelopeStatus::IdentifiedInEveryMember => "identified_in_every_member",
        EnvelopeStatus::NotIdentifiedInEveryMember => "not_identified_in_every_member",
        EnvelopeStatus::MembersDisagree => "members_disagree",
        EnvelopeStatus::Partial => "partial",
    }
}

fn pair_json(a: VariableId, b: VariableId, reason: &str) -> Json {
    json!({ "a": a.raw(), "b": b.raw(), "reason": reason })
}

fn envelope_json(report: &DiscoveredTransportReport) -> Json {
    let envelope = report.envelope();
    let members: Vec<Json> = envelope
        .members()
        .iter()
        .map(|member| {
            let verdict = match &member.verdict {
                MemberVerdict::Identified { rule, formula } => {
                    json!({ "kind": "identified", "rule": rule, "formula": formula_kind(formula) })
                }
                MemberVerdict::NotCertified { reason } => {
                    json!({ "kind": "not_certified", "reason": reason })
                }
                MemberVerdict::MissingEvidence { reason } => {
                    json!({ "kind": "missing_evidence", "reason": reason })
                }
            };
            let edges: Vec<Json> = member
                .directed_edges
                .iter()
                .map(|(parent, child)| json!([parent.raw(), child.raw()]))
                .collect();
            json!({ "directed_edges": edges, "verdict": verdict })
        })
        .collect();
    let unresolved: Vec<Json> =
        envelope.unresolved().iter().map(|p| pair_json(p.a, p.b, p.reason)).collect();
    json!({
        "kind": report.kind(),
        "status": status_label(envelope.status()),
        "formulas_agree": envelope.formulas_agree(),
        "enumeration_complete": envelope.enumeration_complete(),
        "member_bound": envelope.member_bound(),
        "identified_members": envelope.identified_members(),
        "members": members,
        "unresolved": unresolved,
        "assumptions": report.assumptions(),
        "selection_targets": report.selection_targets().iter().map(|t| t.raw()).collect::<Vec<_>>(),
        "structure_identity": hex(report.structure_identity()),
        "result_identity": hex(report.result_identity()),
        "identity": hex(report.identity()),
    })
}

/// A refusal as JSON, or the shared value error for a failure that is not a scientific refusal.
fn refusal_of(error: &DiscoveredTransportError) -> PyResult<String> {
    match error {
        DiscoveredTransportError::Refused(refusal) => {
            let external =
                antecedent::analysis::discovered_transport::discovered_transport_refusal(refusal);
            let unresolved: Vec<Json> =
                refusal.unresolved.iter().map(|p| pair_json(p.a, p.b, p.reason)).collect();
            let unknown: Vec<u32> =
                refusal.unknown_selection_targets.iter().map(|t| t.raw()).collect();
            Ok(RefusalJson::new(external.code, STAGE, &external.detail, &refusal.remedy)
                .offending(external.offending.as_deref())
                .remedy(external.remedy)
                .field("unresolved", Json::Array(unresolved))
                .field("exhausted_search_bounds", json!(refusal.exhausted_search_bounds))
                .field("unknown_selection_targets", json!(unknown))
                .json())
        }
        other => Err(crate::value_err(other.to_string())),
    }
}

fn query_of(text: &str) -> PyResult<TransportQuery> {
    let parsed: Json = serde_json::from_str(text)
        .map_err(|e| crate::value_err(format!("the query is not valid JSON: {e}")))?;
    let map: &Map<String, Json> =
        parsed.as_object().ok_or_else(|| crate::value_err("a query is a JSON object"))?;
    let id = |key: &str| -> PyResult<VariableId> {
        map.get(key)
            .and_then(Json::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .map(VariableId::from_raw)
            .ok_or_else(|| crate::value_err(format!("query field `{key}` must be a variable")))
    };
    let text_of = |key: &str| -> PyResult<String> {
        map.get(key)
            .and_then(Json::as_str)
            .map(str::to_owned)
            .ok_or_else(|| crate::value_err(format!("query field `{key}` must be text")))
    };
    let grid: Vec<f64> = map
        .get("grid")
        .and_then(Json::as_array)
        .ok_or_else(|| crate::value_err("query field `grid` must be a list of numbers"))?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| crate::value_err("query field `grid` must be numbers")))
        .collect::<PyResult<_>>()?;
    let experiments: Vec<VariableId> = match map.get("source_experiments") {
        None | Some(Json::Null) => Vec::new(),
        Some(Json::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .map(VariableId::from_raw)
                    .ok_or_else(|| crate::value_err("source_experiments are variables"))
            })
            .collect::<PyResult<_>>()?,
        Some(_) => return Err(crate::value_err("`source_experiments` must be a list")),
    };
    let query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: id("outcome")?,
            treatment: ContinuousDomain::new(id("treatment")?, GridSpec::Values(Arc::from(grid))),
        }),
        text_of("source_population")?,
        text_of("target_population")?,
        experiments,
    );
    query.validate().map_err(|e| crate::value_err(e.to_string()))?;
    Ok(query)
}

/// The transport verdict across the class a retained outcome returned.
pub(crate) fn across_outcome(
    outcome: &DiscoveryOutcome,
    query_json: &str,
    targets: &[u32],
    member_bound: Option<usize>,
) -> PyResult<Reply> {
    let query = query_of(query_json)?;
    let ids: Vec<VariableId> = targets.iter().copied().map(VariableId::from_raw).collect();
    let bound = member_bound.unwrap_or(DEFAULT_ENVELOPE_MEMBER_BOUND);
    match transport_discovered_structure(outcome, &ids, &query, bound) {
        Ok(report) => Ok((Some(envelope_json(&report).to_string()), None)),
        Err(error) => Ok((None, Some(refusal_of(&error)?))),
    }
}

/// What a discovery-and-transport request returns: the envelope, the refusal, and the
/// discovery (report and outcome) either way.
type RunReply = (Option<String>, Option<String>, Option<String>, Option<String>);

/// Discover on the supplied columns, then request the transport verdict across the class.
///
/// Returns `(envelope, transport_refusal, discovery, discovery_refusal)`. Exactly one of
/// `envelope` and `transport_refusal` is set when discovery ran; `discovery` carries the run
/// report and the retained outcome in both cases.
#[pyfunction]
#[pyo3(signature = (contract_json, names, columns, query_json, selection_targets, member_bound=None, *, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one request: contract, data, query, targets, bound")]
fn discovered_transport_run(
    py: Python<'_>,
    contract_json: &str,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    query_json: &str,
    selection_targets: Vec<u32>,
    member_bound: Option<usize>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<RunReply> {
    let parsed = sd::parse_json(contract_json)?;
    let map = sd::contract_map(&parsed, "static")?;
    let seed = sd::get_u64(map, "seed")?;
    let contract = sd::static_contract(map, names.len())?;
    let query = query_of(query_json)?;
    let ids: Vec<VariableId> = selection_targets.into_iter().map(VariableId::from_raw).collect();
    let bound = member_bound.unwrap_or(DEFAULT_ENVELOPE_MEMBER_BOUND);
    let batch = crate::columns_to_batch(&names, &columns)?;
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let ran: Result<DiscoveredTransportRun, SelectiveDiscoveryError> =
        crate::detach_catch(py, move || {
            let (data, _): (antecedent_data::TabularData, _) =
                crate::discovery_api::tabular_from_batch(&batch)?;
            Ok(discover_and_transport(&contract, &data, &ids, &query, bound, &ctx))
        })?;
    match ran {
        Err(error) => match sd::error_json(&error) {
            Some(refusal) => Ok((None, None, None, Some(refusal))),
            None => Err(crate::CausalDiscoveryError::new_err(error.to_string())),
        },
        Ok(run) => {
            let discovery = json!({
                "report": serde_json::from_str::<Json>(&sd::report_json(&run.discovery))
                    .unwrap_or(Json::Null),
                "outcome": serde_json::from_str::<Json>(&sd::outcome_json(&run.outcome))
                    .unwrap_or(Json::Null),
            })
            .to_string();
            match &run.transport {
                Ok(report) => {
                    Ok((Some(envelope_json(report).to_string()), None, Some(discovery), None))
                }
                Err(error) => Ok((None, Some(refusal_of(error)?), Some(discovery), None)),
            }
        }
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(discovered_transport_run, module)?)?;
    Ok(())
}
