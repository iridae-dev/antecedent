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
use antecedent::analysis::discovered_transport_estimate::{
    DiscoveredTransportEstimate, MemberOutcome, TransportAnswer, TransportEvidence,
    estimate_discovered_transport, verify_discovered_transport_estimate,
};
use antecedent::analysis::discovered_transport_temporal::{
    DiscoveredTemporalTransportReport, TemporalClass, TemporalHorizon,
    estimate_discovered_temporal_transport, transport_discovered_temporal_structure,
    transport_temporal_class,
};
use antecedent_core::{
    ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_discovery::selective::{DiscoveryOutcome, SelectiveDiscoveryError};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
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
            let pairs = |edges: &[(VariableId, VariableId)]| -> Vec<Json> {
                edges.iter().map(|(a, b)| json!([a.raw(), b.raw()])).collect()
            };
            json!({
                "directed_edges": edges,
                "bidirected_edges": pairs(&member.bidirected_edges),
                "confounded_edges": pairs(&member.confounded_edges),
                "verdict": verdict,
            })
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
                .field(
                    "member_reasons",
                    Json::Array(
                        refusal
                            .member_reasons
                            .iter()
                            .map(|(index, reason)| json!({ "member": index, "reason": reason }))
                            .collect(),
                    ),
                )
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

fn answer_json(answer: &TransportAnswer) -> Json {
    json!({
        "kind": answer.kind.label(),
        "levels": answer
            .levels
            .iter()
            .map(|l| json!({ "level": l.level, "low": l.low, "high": l.high }))
            .collect::<Vec<_>>(),
        "contrast": answer.contrast.map(|(low, high)| json!({ "low": low, "high": high })),
        "expressions_agree": answer.expressions_agree,
        "interval": answer.interval.map(|i| json!({
            "low": i.low,
            "high": i.high,
            "coverage_target": i.coverage_target,
            "calibration": i.calibration,
        })),
        "interval_reason": answer.interval_reason,
    })
}

fn estimate_json(estimate: &DiscoveredTransportEstimate) -> Json {
    let members: Vec<Json> = estimate
        .members()
        .iter()
        .enumerate()
        .map(|(index, member)| match member {
            MemberOutcome::Estimated(e) => json!({
                "member": index,
                "status": "estimated",
                "expression": e.expression,
                "means": e.means,
                "contrast": e.contrast,
            }),
            MemberOutcome::Unidentified { verdict } => {
                json!({ "member": index, "status": "unidentified", "detail": verdict })
            }
            MemberOutcome::Unestimable { detail } => {
                json!({ "member": index, "status": "unestimable", "detail": detail })
            }
        })
        .collect();
    json!({
        "levels": estimate.levels(),
        "members": members,
        "answer": answer_json(estimate.answer()),
        "report_identity": hex(estimate.report_identity()),
        "identity": hex(estimate.identity()),
    })
}

/// `(estimate, transport_refusal, discovery, discovery_refusal, artifact)`.
type EstimateReply =
    (Option<String>, Option<String>, Option<String>, Option<String>, Option<Py<PyBytes>>);

/// Discover on the source columns, enumerate the class, and estimate the transported response
/// from the source and target columns.
///
/// Exactly one of `estimate` and `transport_refusal` is set when discovery ran; `discovery`
/// carries the run report and outcome in both cases, and `artifact` the exported estimate.
#[pyfunction]
#[pyo3(signature = (contract_json, names, source_columns, target_columns, query_json, selection_targets, member_bound=None, bootstrap_replicates=0, *, threads=None, cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "one request: contract, two tables, query, targets, bounds"
)]
fn discovered_transport_estimate_run(
    py: Python<'_>,
    contract_json: &str,
    names: Vec<String>,
    source_columns: Vec<PyReadonlyArray1<'_, f64>>,
    target_columns: Vec<PyReadonlyArray1<'_, f64>>,
    query_json: &str,
    selection_targets: Vec<u32>,
    member_bound: Option<usize>,
    bootstrap_replicates: u32,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<EstimateReply> {
    type Ran = Result<
        (
            DiscoveredTransportRun,
            Option<Result<(DiscoveredTransportEstimate, Vec<u8>), DiscoveredTransportError>>,
        ),
        SelectiveDiscoveryError,
    >;
    let parsed = sd::parse_json(contract_json)?;
    let map = sd::contract_map(&parsed, "static")?;
    let seed = sd::get_u64(map, "seed")?;
    let contract = sd::static_contract(map, names.len())?;
    let query = query_of(query_json)?;
    let ids: Vec<VariableId> = selection_targets.into_iter().map(VariableId::from_raw).collect();
    let bound = member_bound.unwrap_or(DEFAULT_ENVELOPE_MEMBER_BOUND);
    let source_batch = crate::columns_to_batch(&names, &source_columns)?;
    let target_batch = crate::columns_to_batch(&names, &target_columns)?;
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let ran: Ran = crate::detach_catch(py, move || {
        let (source, _): (antecedent_data::TabularData, _) =
            crate::discovery_api::tabular_from_batch(&source_batch)?;
        let (target, _): (antecedent_data::TabularData, _) =
            crate::discovery_api::tabular_from_batch(&target_batch)?;
        let run = match discover_and_transport(&contract, &source, &ids, &query, bound, &ctx) {
            Ok(run) => run,
            Err(error) => return Ok(Err(error)),
        };
        let estimated = run.transport.as_ref().ok().map(|report| {
            let mut evidence = TransportEvidence::new(&source, &target);
            evidence.bootstrap_replicates = bootstrap_replicates;
            estimate_discovered_transport(report, &query, &evidence, &ctx).and_then(|estimate| {
                let bytes = estimate
                    .export()
                    .map_err(|e| DiscoveredTransportError::Estimation(e.to_string()))?;
                Ok((estimate, bytes))
            })
        });
        Ok(Ok((run, estimated)))
    })?;
    let (run, estimated) = match ran {
        Err(error) => {
            return match sd::error_json(&error) {
                Some(refusal) => Ok((None, None, None, Some(refusal), None)),
                None => Err(crate::CausalDiscoveryError::new_err(error.to_string())),
            };
        }
        Ok(pair) => pair,
    };
    let discovery = json!({
        "report": serde_json::from_str::<Json>(&sd::report_json(&run.discovery)).unwrap_or(Json::Null),
        "outcome": serde_json::from_str::<Json>(&sd::outcome_json(&run.outcome)).unwrap_or(Json::Null),
    })
    .to_string();
    let refusal_of_run = |error: &DiscoveredTransportError| -> PyResult<Reply> {
        Ok((None, Some(refusal_of(error)?)))
    };
    match (&run.transport, estimated) {
        (Err(error), _) => {
            let (_, refusal) = refusal_of_run(error)?;
            Ok((None, refusal, Some(discovery), None, None))
        }
        (Ok(_), Some(Ok((estimate, bytes)))) => Ok((
            Some(estimate_json(&estimate).to_string()),
            None,
            Some(discovery),
            None,
            Some(PyBytes::new(py, &bytes).unbind()),
        )),
        (Ok(_), Some(Err(error))) => {
            let (_, refusal) = refusal_of_run(&error)?;
            Ok((None, refusal, Some(discovery), None, None))
        }
        (Ok(_), None) => Err(crate::value_err("the transport request produced no result")),
    }
}

/// Replay an exported estimate: consume every member grid and recompute the answer.
///
/// Returns the recomputed answer as JSON; a malformed or forged artifact raises the shared
/// value error.
#[pyfunction]
#[pyo3(signature = (artifact, *, seed=0))]
fn verify_discovered_transport_estimate_artifact(artifact: &[u8], seed: u64) -> PyResult<String> {
    let ctx = crate::py_execution_context_cancel(seed, crate::resolve_user_threads(None), None);
    let verified = verify_discovered_transport_estimate(
        artifact,
        antecedent_expr::ExactEvaluationLimits::default(),
        &ctx,
    )
    .map_err(|e| crate::value_err(e.to_string()))?;
    Ok(json!({
        "answer": answer_json(&verified.answer),
        "report_identity": hex(&verified.report_identity),
        "members": verified.members,
    })
    .to_string())
}

fn lagged_edges_json(
    edges: &[antecedent::analysis::discovered_transport_temporal::LaggedEdge],
) -> Vec<Json> {
    edges.iter().map(|e| json!([e.from.raw(), e.to.raw(), e.lag])).collect()
}

fn temporal_envelope_json(report: &DiscoveredTemporalTransportReport) -> Json {
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
            json!({
                "directed_edges": lagged_edges_json(&member.directed_edges),
                "bidirected_edges": lagged_edges_json(&member.bidirected_edges),
                "confounded_edges": lagged_edges_json(&member.confounded_edges),
                "verdict": verdict,
            })
        })
        .collect();
    json!({
        "kind": report.kind(),
        "status": status_label(envelope.status()),
        "formulas_agree": envelope.formulas_agree(),
        "enumeration_complete": envelope.enumeration_complete(),
        "member_bound": envelope.member_bound(),
        "identified_members": envelope.identified_members(),
        "members": members,
        "unresolved": envelope.unresolved().iter().map(|p| pair_json(p.a, p.b, p.reason)).collect::<Vec<_>>(),
        "assumptions": report.assumptions(),
        "selection_targets": report.selection_targets().iter().map(|t| t.raw()).collect::<Vec<_>>(),
        "history": envelope.history(),
        "horizon_steps": envelope.horizon_steps(),
        "max_lag": envelope.max_lag(),
        "structure_identity": hex(report.structure_identity()),
        "result_identity": report.result_identity().map(hex),
        "identity": hex(report.identity()),
    })
}

/// The temporal transport verdict across the class a retained temporal outcome returned.
pub(crate) fn across_temporal_outcome(
    outcome: &DiscoveryOutcome,
    horizon_steps: u32,
    history: Option<u32>,
    query_json: &str,
    targets: &[u32],
    member_bound: Option<usize>,
) -> PyResult<Reply> {
    let query = query_of(query_json)?;
    let ids: Vec<VariableId> = targets.iter().copied().map(VariableId::from_raw).collect();
    let bound = member_bound.unwrap_or(DEFAULT_ENVELOPE_MEMBER_BOUND);
    match transport_discovered_temporal_structure(
        outcome,
        TemporalHorizon { horizon_steps, history },
        &ids,
        &query,
        bound,
    ) {
        Ok(report) => Ok((Some(temporal_envelope_json(&report).to_string()), None)),
        Err(error) => Ok((None, Some(refusal_of(&error)?))),
    }
}

/// The temporal transport verdict across a temporal graph held in Python (a `TemporalDag`,
/// `TemporalCpdag` or `TemporalPag`).
#[pyfunction]
#[pyo3(signature = (graph, horizon_steps, query_json, selection_targets, history=None, member_bound=None, exhausted_search_bounds=0))]
fn discovered_temporal_graph_transport(
    graph: &Bound<'_, PyAny>,
    horizon_steps: u32,
    query_json: &str,
    selection_targets: Vec<u32>,
    history: Option<u32>,
    member_bound: Option<usize>,
    exhausted_search_bounds: usize,
) -> PyResult<Reply> {
    let query = query_of(query_json)?;
    let ids: Vec<VariableId> = selection_targets.into_iter().map(VariableId::from_raw).collect();
    let bound = member_bound.unwrap_or(DEFAULT_ENVELOPE_MEMBER_BOUND);
    let horizon = TemporalHorizon { horizon_steps, history };
    let run = |class: TemporalClass<'_>| {
        transport_temporal_class(class, exhausted_search_bounds, horizon, &ids, &query, bound)
    };
    let report = if let Ok(dag) = graph.extract::<crate::graphs::TemporalDag>() {
        run(TemporalClass::Dag(&dag.dag))
    } else if let Ok(cpdag) = graph.extract::<crate::graphs::TemporalCpdag>() {
        run(TemporalClass::Cpdag(&cpdag.cpdag))
    } else if let Ok(pag) = graph.extract::<crate::graphs::TemporalPag>() {
        run(TemporalClass::Pag(&pag.pag))
    } else {
        return Err(crate::value_err("graph must be a TemporalDag, TemporalCpdag or TemporalPag"));
    };
    match report {
        Ok(report) => Ok((Some(temporal_envelope_json(&report).to_string()), None)),
        Err(error) => Ok((None, Some(refusal_of(&error)?))),
    }
}

/// Discover on the source series, unroll the class at the horizon, and estimate the transported
/// response from the source and target series. Returns
/// `(estimate, transport_refusal, discovery, discovery_refusal, artifact)` like the static run.
#[pyfunction]
#[pyo3(signature = (contract_json, names, source_columns, target_columns, query_json, selection_targets, horizon_steps, history=None, member_bound=None, bootstrap_replicates=0, block_length=None, *, threads=None, cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "one request: contract, two series, query, horizon, bounds"
)]
fn discovered_temporal_transport_estimate_run(
    py: Python<'_>,
    contract_json: &str,
    names: Vec<String>,
    source_columns: Vec<PyReadonlyArray1<'_, f64>>,
    target_columns: Vec<PyReadonlyArray1<'_, f64>>,
    query_json: &str,
    selection_targets: Vec<u32>,
    horizon_steps: u32,
    history: Option<u32>,
    member_bound: Option<usize>,
    bootstrap_replicates: u32,
    block_length: Option<u32>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<EstimateReply> {
    type Ran = Result<
        (
            antecedent_discovery::selective::DiscoveryOutcome,
            antecedent_discovery::selective::SelectiveRunReport,
            Result<DiscoveredTemporalTransportReport, DiscoveredTransportError>,
            Option<Result<(DiscoveredTransportEstimate, Vec<u8>), DiscoveredTransportError>>,
        ),
        SelectiveDiscoveryError,
    >;
    let parsed = sd::parse_json(contract_json)?;
    let map = sd::contract_map(&parsed, "temporal")?;
    let seed = sd::get_u64(map, "seed")?;
    let contract = sd::temporal_contract(map, names.len())?;
    let query = query_of(query_json)?;
    let ids: Vec<VariableId> = selection_targets.into_iter().map(VariableId::from_raw).collect();
    let bound = member_bound.unwrap_or(DEFAULT_ENVELOPE_MEMBER_BOUND);
    let source_batch = crate::columns_to_batch(&names, &source_columns)?;
    let target_batch = crate::columns_to_batch(&names, &target_columns)?;
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let ran: Ran = crate::detach_catch(py, move || {
        let (series, _): (antecedent_data::TimeSeriesData, _) =
            crate::discovery_api::series_from_batch(&source_batch)?;
        let (source, _): (antecedent_data::TabularData, _) =
            crate::discovery_api::tabular_from_batch(&source_batch)?;
        let (target, _): (antecedent_data::TabularData, _) =
            crate::discovery_api::tabular_from_batch(&target_batch)?;
        let mut adapter = antecedent_discovery::selective::TemporalSelectiveDiscovery::new();
        let discovery = match adapter.run(&contract, &series, &ctx) {
            Ok(report) => report,
            Err(error) => return Ok(Err(error)),
        };
        let Some(outcome) = adapter.outcome().cloned() else {
            return Ok(Err(SelectiveDiscoveryError::refused(
                antecedent_core::reason_code!("invalid_argument"),
                "selective_discovery.nothing_retained",
                "the discovery run retained no outcome",
            )));
        };
        let transport = transport_discovered_temporal_structure(
            &outcome,
            TemporalHorizon { horizon_steps, history },
            &ids,
            &query,
            bound,
        );
        let estimated = transport.as_ref().ok().map(|report| {
            let mut evidence = TransportEvidence::new(&source, &target);
            evidence.bootstrap_replicates = bootstrap_replicates;
            evidence.block_length = block_length;
            estimate_discovered_temporal_transport(report, &query, &evidence, &ctx).and_then(
                |estimate| {
                    let bytes = estimate
                        .export()
                        .map_err(|e| DiscoveredTransportError::Estimation(e.to_string()))?;
                    Ok((estimate, bytes))
                },
            )
        });
        Ok(Ok((outcome, discovery, transport, estimated)))
    })?;
    let (outcome, discovery, transport, estimated) = match ran {
        Err(error) => {
            return match sd::error_json(&error) {
                Some(refusal) => Ok((None, None, None, Some(refusal), None)),
                None => Err(crate::CausalDiscoveryError::new_err(error.to_string())),
            };
        }
        Ok(parts) => parts,
    };
    let discovery_json = json!({
        "report": serde_json::from_str::<Json>(&sd::report_json(&discovery)).unwrap_or(Json::Null),
        "outcome": serde_json::from_str::<Json>(&sd::outcome_json(&outcome)).unwrap_or(Json::Null),
    })
    .to_string();
    match (transport, estimated) {
        (Err(error), _) | (Ok(_), Some(Err(error))) => {
            Ok((None, Some(refusal_of(&error)?), Some(discovery_json), None, None))
        }
        (Ok(_), Some(Ok((estimate, bytes)))) => Ok((
            Some(estimate_json(&estimate).to_string()),
            None,
            Some(discovery_json),
            None,
            Some(PyBytes::new(py, &bytes).unbind()),
        )),
        (Ok(_), None) => Err(crate::value_err("the transport request produced no result")),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(discovered_transport_run, module)?)?;
    module.add_function(wrap_pyfunction!(discovered_transport_estimate_run, module)?)?;
    module.add_function(wrap_pyfunction!(discovered_temporal_graph_transport, module)?)?;
    module.add_function(wrap_pyfunction!(discovered_temporal_transport_estimate_run, module)?)?;
    module
        .add_function(wrap_pyfunction!(verify_discovered_transport_estimate_artifact, module)?)?;
    Ok(())
}
