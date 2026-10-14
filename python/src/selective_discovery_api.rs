//! Python bridge for selective discovery.
//!
//! Python declares a contract as JSON (variables are positions in the supplied columns) and
//! passes the searched columns. The statistic store, the stage plan, the algorithms, the
//! work counters, the portable state and every refusal are Rust's. A refusal comes back as
//! structured JSON for the Python layer to raise as a typed exception; a failure of the
//! discovery algorithm itself (data, test, graph, cancellation) raises `CausalDiscoveryError`.
//!
//! No recovery, error-control or interval statement is produced.

use antecedent_core::{Lag, VariableId};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::constraints::{CiMaskType, DiscoveryConstraints};
use antecedent_discovery::result::LaggedLink;
use antecedent_discovery::selective::{
    CiTestSpec, DiscoveredStructure, DiscoveredStructureRefusal, DiscoveryOutcome, DiscoveryPlan,
    NativeResult, PortableDiscoveryState, SelectiveDiscoveryError, SelectiveRunReport,
    StageDecision, StaticAlgorithm, StaticDiscoveryContract, StaticSelectiveDiscovery,
    TemporalAlgorithm, TemporalDiscoveryContract, TemporalSelectiveDiscovery, WorkUnits,
    request_downstream_interval, request_recovery_claim, request_score_search_reuse,
    request_selection_diagram,
};
use antecedent_io::selective_discovery_artifact::{
    SelectiveDiscoveryArtifact, SelectiveDiscoveryArtifactError,
};
use antecedent_stats::{FdrAdjustment, MultipleTestingMethod, SignificanceMethod};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Map, Value as Json, json};

const STAGE: &str = "selective_discovery";

// ---------------------------------------------------------------------------
// Refusals as data.
// ---------------------------------------------------------------------------

fn refusal_json(code: &str, stage: &str, detail: &str, message: &str, remedy: &str) -> String {
    json!({
        "code": code,
        "stage": stage,
        "detail": detail,
        "message": message,
        "offending": Json::Null,
        "remedy": remedy,
    })
    .to_string()
}

fn error_json(error: &SelectiveDiscoveryError) -> Option<String> {
    match error {
        SelectiveDiscoveryError::Refused { code, detail, remedy } => {
            Some(refusal_json(code, STAGE, detail, remedy, remedy))
        }
        _ => None,
    }
}

fn artifact_refusal(error: &SelectiveDiscoveryArtifactError) -> String {
    match error.refusal() {
        Some((code, detail, message)) => refusal_json(code, STAGE, &detail, &message, &message),
        None => refusal_json(
            antecedent_core::reason_code!("invalid_argument"),
            STAGE,
            "selective_discovery.artifact_invalid",
            &error.to_string(),
            "supply an intact selective_discovery_v1 artifact",
        ),
    }
}

fn structure_refusal(refusal: &DiscoveredStructureRefusal) -> String {
    let edges: Vec<Json> = refusal
        .unresolved_edges
        .iter()
        .map(|u| {
            json!({
                "a": u.edge.a.label(),
                "b": u.edge.b.label(),
                "at_a": u.edge.at_a.as_str(),
                "at_b": u.edge.at_b.as_str(),
                "reason": u.reason,
            })
        })
        .collect();
    let offending: Vec<String> = refusal
        .unresolved_edges
        .iter()
        .map(|u| format!("{}~{}:{}", u.edge.a.label(), u.edge.b.label(), u.reason))
        .collect();
    json!({
        "code": refusal.code,
        "stage": "bind",
        "detail": refusal.detail,
        "message": refusal.remedy,
        "offending": if offending.is_empty() { Json::Null } else { Json::String(offending.join(";")) },
        "remedy": refusal.remedy,
        "unresolved_edges": edges,
        "bounded_search_receipts": refusal.bounded_search_receipts,
    })
    .to_string()
}

fn discovery_failure(error: &SelectiveDiscoveryError) -> PyErr {
    crate::CausalDiscoveryError::new_err(error.to_string())
}

type Reply = PyResult<(Option<String>, Option<String>)>;

fn reply(result: Result<String, SelectiveDiscoveryError>) -> Reply {
    match result {
        Ok(json) => Ok((Some(json), None)),
        Err(error) => match error_json(&error) {
            Some(refusal) => Ok((None, Some(refusal))),
            None => Err(discovery_failure(&error)),
        },
    }
}

// ---------------------------------------------------------------------------
// Declarations.
// ---------------------------------------------------------------------------

fn input_err(message: impl AsRef<str>) -> PyErr {
    crate::value_err(message)
}

fn field<'a>(map: &'a Map<String, Json>, key: &str) -> Option<&'a Json> {
    map.get(key).filter(|v| !v.is_null())
}

fn get_u64(map: &Map<String, Json>, key: &str) -> PyResult<u64> {
    field(map, key)
        .and_then(Json::as_u64)
        .ok_or_else(|| input_err(format!("field `{key}` must be an unsigned integer")))
}

fn opt_usize(map: &Map<String, Json>, key: &str) -> PyResult<Option<usize>> {
    match field(map, key) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| input_err(format!("field `{key}` must be an unsigned integer"))),
    }
}

fn opt_u32(map: &Map<String, Json>, key: &str) -> PyResult<Option<u32>> {
    match opt_usize(map, key)? {
        None => Ok(None),
        Some(n) => {
            u32::try_from(n).map(Some).map_err(|_| input_err(format!("field `{key}` is too large")))
        }
    }
}

fn get_text(map: &Map<String, Json>, key: &str) -> PyResult<String> {
    field(map, key)
        .and_then(Json::as_str)
        .map(str::to_owned)
        .ok_or_else(|| input_err(format!("field `{key}` must be a string")))
}

fn id_of(value: &Json, what: &str) -> PyResult<u32> {
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| input_err(format!("{what} must be an unsigned integer")))
}

fn links(value: Option<&Json>) -> PyResult<Option<std::sync::Arc<[LaggedLink]>>> {
    let Some(value) = value.filter(|v| !v.is_null()) else { return Ok(None) };
    let items = value.as_array().ok_or_else(|| input_err("links must be a list"))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let parts = item
            .as_array()
            .filter(|p| p.len() == 4)
            .ok_or_else(|| input_err("a link is [source, source_lag, target, target_lag]"))?;
        out.push(LaggedLink {
            source: VariableId::from_raw(id_of(&parts[0], "link source")?),
            source_lag: Lag::from_raw(id_of(&parts[1], "link source lag")?),
            target: VariableId::from_raw(id_of(&parts[2], "link target")?),
            target_lag: Lag::from_raw(id_of(&parts[3], "link target lag")?),
        });
    }
    Ok(Some(std::sync::Arc::from(out)))
}

fn apply_constraints(
    constraints: &mut DiscoveryConstraints,
    map: &Map<String, Json>,
) -> PyResult<()> {
    if let Some(alpha) = field(map, "alpha") {
        constraints.alpha = alpha.as_f64().ok_or_else(|| input_err("`alpha` must be a number"))?;
    }
    if let Some(n) = opt_usize(map, "max_cond_size")? {
        constraints.max_cond_size = n;
    }
    if map.contains_key("max_parents") {
        constraints.max_parents = opt_usize(map, "max_parents")?;
    }
    if let Some(n) = opt_u32(map, "min_lag")? {
        constraints.temporal.min_lag = Lag::from_raw(n);
    }
    if let Some(n) = opt_u32(map, "max_lag")? {
        constraints.temporal.max_lag = Lag::from_raw(n);
    }
    if let Some(tag) = field(map, "mask_type").and_then(Json::as_str) {
        constraints.mask_type = CiMaskType::parse(tag).map_err(|e| input_err(e.to_string()))?;
    }
    if let Some(sig) = field(map, "significance") {
        let sig = sig.as_object().ok_or_else(|| input_err("`significance` must be an object"))?;
        constraints.significance = match get_text(sig, "kind")?.as_str() {
            "analytic" => SignificanceMethod::Analytic,
            "block_shuffle" => SignificanceMethod::BlockShuffle {
                replicates: opt_u32(sig, "replicates")?
                    .ok_or_else(|| input_err("block_shuffle needs `replicates`"))?,
                block_size: opt_usize(sig, "block_size")?
                    .ok_or_else(|| input_err("block_shuffle needs `block_size`"))?,
            },
            other => return Err(input_err(format!("unknown significance kind {other:?}"))),
        };
    }
    if let Some(required) = links(map.get("required"))? {
        constraints.required = required;
    }
    if let Some(forbidden) = links(map.get("forbidden"))? {
        constraints.forbidden = forbidden;
    }
    if let Some(tiers) = field(map, "tiers") {
        let tiers = tiers.as_array().ok_or_else(|| input_err("`tiers` must be a list"))?;
        let mut out: Vec<std::sync::Arc<[VariableId]>> = Vec::with_capacity(tiers.len());
        for tier in tiers {
            let members = tier.as_array().ok_or_else(|| input_err("a tier must be a list"))?;
            let ids = members
                .iter()
                .map(|m| id_of(m, "tier member").map(VariableId::from_raw))
                .collect::<PyResult<Vec<_>>>()?;
            out.push(std::sync::Arc::from(ids));
        }
        constraints.tiers = std::sync::Arc::from(out);
    }
    Ok(())
}

fn fdr_of(
    map: &Map<String, Json>,
    current: Option<FdrAdjustment>,
) -> PyResult<Option<FdrAdjustment>> {
    match map.get("fdr") {
        None => Ok(current),
        Some(Json::Null) => Ok(None),
        Some(Json::Object(obj)) => {
            let method = match get_text(obj, "method")?.as_str() {
                "bh" => MultipleTestingMethod::BenjaminiHochberg,
                "by" => MultipleTestingMethod::BenjaminiYekutieli,
                "bonferroni" => MultipleTestingMethod::Bonferroni,
                "holm" => MultipleTestingMethod::Holm,
                other => return Err(input_err(format!("unknown multiplicity method {other:?}"))),
            };
            let exclude = obj
                .get("exclude_contemporaneous")
                .and_then(Json::as_bool)
                .ok_or_else(|| input_err("fdr needs `exclude_contemporaneous`"))?;
            Ok(Some(FdrAdjustment { method, exclude_contemporaneous: exclude }))
        }
        Some(_) => Err(input_err("`fdr` must be null or an object")),
    }
}

fn populations(map: &Map<String, Json>) -> PyResult<(Option<String>, Vec<String>)> {
    let population = match field(map, "population") {
        None => None,
        Some(v) => Some(
            v.as_str().map(str::to_owned).ok_or_else(|| input_err("`population` must be text"))?,
        ),
    };
    let additional = match field(map, "additional_populations") {
        None => Vec::new(),
        Some(v) => v
            .as_array()
            .ok_or_else(|| input_err("`additional_populations` must be a list"))?
            .iter()
            .map(|p| {
                p.as_str().map(str::to_owned).ok_or_else(|| input_err("a population must be text"))
            })
            .collect::<PyResult<Vec<_>>>()?,
    };
    Ok((population, additional))
}

fn variables_of(n: usize) -> PyResult<Vec<VariableId>> {
    let n = u32::try_from(n).map_err(|_| input_err("too many variables"))?;
    Ok((0..n).map(VariableId::from_raw).collect())
}

fn static_contract(map: &Map<String, Json>, n: usize) -> PyResult<StaticDiscoveryContract> {
    let algorithm = match get_text(map, "algorithm")?.as_str() {
        "pc" => StaticAlgorithm::Pc,
        "fci" => StaticAlgorithm::Fci,
        other => return Err(input_err(format!("unknown static algorithm {other:?}"))),
    };
    let mut contract = StaticDiscoveryContract::new(
        algorithm,
        variables_of(n)?,
        CiTestSpec::named(get_text(map, "ci_test")?),
        get_u64(map, "seed")?,
    );
    apply_constraints(&mut contract.constraints, map)?;
    contract.fdr = fdr_of(map, contract.fdr)?;
    if let Some(nodes) = opt_usize(map, "possible_d_sep_max_nodes")? {
        contract.possible_d_sep_max_nodes = nodes;
    }
    if let Some(budget) = field(map, "discriminating_paths").and_then(Json::as_object) {
        if let Some(n) = opt_usize(budget, "max_paths_per_edge")? {
            contract.discriminating_path_budget.max_paths_per_edge = n;
        }
        if let Some(n) = opt_usize(budget, "max_len")? {
            contract.discriminating_path_budget.max_len = n;
        }
    }
    let (population, additional) = populations(map)?;
    if let Some(p) = population {
        contract.population = p;
    }
    contract.additional_populations = additional;
    Ok(contract)
}

fn temporal_contract(map: &Map<String, Json>, n: usize) -> PyResult<TemporalDiscoveryContract> {
    let algorithm = match get_text(map, "algorithm")?.as_str() {
        "pcmci" => TemporalAlgorithm::Pcmci,
        "pcmci_plus" => TemporalAlgorithm::PcmciPlus,
        other => return Err(input_err(format!("unknown temporal algorithm {other:?}"))),
    };
    let mut contract = TemporalDiscoveryContract::new(
        algorithm,
        variables_of(n)?,
        CiTestSpec::named(get_text(map, "ci_test")?),
        get_u64(map, "seed")?,
    );
    apply_constraints(&mut contract.constraints, map)?;
    contract.fdr = fdr_of(map, contract.fdr)?;
    let (population, additional) = populations(map)?;
    if let Some(p) = population {
        contract.population = p;
    }
    contract.additional_populations = additional;
    Ok(contract)
}

// ---------------------------------------------------------------------------
// Results as JSON.
// ---------------------------------------------------------------------------

fn hex(bytes: &[u8; 32]) -> String {
    antecedent_discovery::selective::digest_hex(bytes)
}

fn node_json(node: antecedent_discovery::selective::StructureNode) -> Json {
    json!({ "kind": node.kind, "variable": node.variable, "aux": node.aux })
}

fn structure_json(structure: &DiscoveredStructure) -> Json {
    let edges: Vec<Json> = structure
        .edges()
        .iter()
        .map(|e| {
            json!({
                "a": node_json(e.a),
                "b": node_json(e.b),
                "at_a": e.at_a.as_str(),
                "at_b": e.at_b.as_str(),
                "middle": e.middle,
                "unresolved_reason": e.unresolved_reason(),
            })
        })
        .collect();
    json!({
        "class": structure.class().as_str(),
        "nodes": structure.nodes().iter().copied().map(node_json).collect::<Vec<_>>(),
        "edges": edges,
        "identity": hex(&structure.identity()),
        "fully_directed": structure.is_fully_directed(),
    })
}

fn link_json(link: &LaggedLink, statistic: f64, p_value: f64, adjusted: Option<f64>) -> Json {
    json!({
        "source": link.source.raw(),
        "source_lag": link.source_lag.raw(),
        "target": link.target.raw(),
        "target_lag": link.target_lag.raw(),
        "statistic": statistic,
        "p_value": p_value,
        "adjusted_p_value": adjusted,
    })
}

fn evidence_json(native: &NativeResult) -> Vec<Json> {
    let scored: Vec<_> = match native {
        NativeResult::StaticCpdag(r) => r.evidence.links.to_vec(),
        NativeResult::StaticPag(r) => r.evidence.links.to_vec(),
        NativeResult::TemporalDag(r) => r.evidence.links.to_vec(),
        NativeResult::TemporalCpdag(r) => r.evidence.links.to_vec(),
        _ => Vec::new(),
    };
    scored.iter().map(|s| link_json(&s.link, s.statistic, s.p_value, s.adjusted_p_value)).collect()
}

fn outcome_json(outcome: &DiscoveryOutcome) -> String {
    let unresolved: Vec<Json> = outcome
        .structure
        .unresolved_edges()
        .iter()
        .map(|u| {
            json!({
                "a": node_json(u.edge.a),
                "b": node_json(u.edge.b),
                "at_a": u.edge.at_a.as_str(),
                "at_b": u.edge.at_b.as_str(),
                "reason": u.reason,
            })
        })
        .collect();
    json!({
        "structure": structure_json(&outcome.structure),
        "unresolved_edges": unresolved,
        "evidence": evidence_json(&outcome.native),
        "diagnostics": outcome.diagnostics,
        "bounded_search_receipts": outcome.bounded_search_receipts,
        "search_incomplete": outcome.search_incomplete,
        "orientation_conflict": outcome.orientation_conflict,
        "structure_identity": hex(&outcome.structure_identity),
        "result_identity": hex(&outcome.result_identity),
    })
    .to_string()
}

fn work_json(work: &WorkUnits) -> Json {
    json!({
        "ci_tests_attempted": work.ci_tests_attempted,
        "ci_tests_completed": work.ci_tests_completed,
        "ci_tests_reused": work.ci_tests_reused,
        "ci_tests_consumed": work.ci_tests_consumed,
        "search_passes": work.search_passes,
        "orientation_runs": work.orientation_runs,
        "orientation_rounds": work.orientation_rounds,
        "orientations_applied": work.orientations_applied,
        "score_evaluations": work.score_evaluations,
        "algorithm_executions": work.algorithm_executions,
    })
}

fn plan_json(plan: &DiscoveryPlan) -> Json {
    let entries: Vec<Json> = plan
        .entries()
        .iter()
        .map(|e| {
            let (status, because): (&str, Vec<&str>) = match &e.decision {
                StageDecision::Reused => ("reused", Vec::new()),
                StageDecision::Recomputed { because } => ("recomputed", because.clone()),
            };
            json!({
                "stage": e.stage.as_str(),
                "status": status,
                "because": because,
                "identity": e.identity.to_hex(),
            })
        })
        .collect();
    json!({ "stages": entries, "pure_reuse": plan.is_pure_reuse() })
}

fn report_json(report: &SelectiveRunReport) -> String {
    json!({
        "plan": plan_json(&report.plan),
        "work": work_json(&report.work),
        "outcome_changed": report.outcome_changed,
        "retained_statistics": report.retained_statistics,
    })
    .to_string()
}

fn state_json(state: &PortableDiscoveryState) -> String {
    json!({
        "route": state.route,
        "contract_identity": hex(&state.contract_identity),
        "data_identity": hex(&state.data_identity),
        "rows": state.rows,
        "receipts": state.receipts.len(),
        "receipts_identity": hex(&state.receipts_identity),
        "graph_identity": hex(&state.graph_identity()),
        "result_identity": hex(&state.result_identity),
        "search_incomplete": state.search_incomplete,
        "work": work_json(&state.work),
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Handle and entry points.
// ---------------------------------------------------------------------------

enum Adapter {
    Static(StaticSelectiveDiscovery),
    Temporal(TemporalSelectiveDiscovery),
}

impl Adapter {
    fn outcome(&self) -> Option<&DiscoveryOutcome> {
        match self {
            Self::Static(a) => a.outcome(),
            Self::Temporal(a) => a.outcome(),
        }
    }

    fn export_state(&self) -> Result<PortableDiscoveryState, SelectiveDiscoveryError> {
        match self {
            Self::Static(a) => a.export_state(),
            Self::Temporal(a) => a.export_state(),
        }
    }

    fn retained(&self) -> usize {
        match self {
            Self::Static(a) => a.retained_statistics(),
            Self::Temporal(a) => a.retained_statistics(),
        }
    }
}

/// One selective-discovery adapter and the state it retains.
#[pyclass(name = "SelectiveDiscoveryHandle", skip_from_py_object)]
pub(crate) struct SelectiveDiscoveryHandle {
    inner: Option<Adapter>,
    kind: &'static str,
}

fn kind_of(kind: &str) -> PyResult<&'static str> {
    match kind {
        "static" => Ok("static"),
        "temporal" => Ok("temporal"),
        other => Err(input_err(format!("kind must be `static` or `temporal`, got {other:?}"))),
    }
}

fn contract_map<'a>(parsed: &'a Json, kind: &str) -> PyResult<&'a Map<String, Json>> {
    let map = parsed.as_object().ok_or_else(|| input_err("a contract is a JSON object"))?;
    if get_text(map, "kind")? != kind {
        return Err(input_err(format!("the contract is not a {kind} contract")));
    }
    Ok(map)
}

fn parse_json(text: &str) -> PyResult<Json> {
    serde_json::from_str(text).map_err(|e| input_err(format!("contract is not valid JSON: {e}")))
}

/// What a run or plan returns inside the detached closure.
type Ran = (Adapter, Result<String, SelectiveDiscoveryError>);

#[pymethods]
impl SelectiveDiscoveryHandle {
    #[new]
    fn new(kind: &str) -> PyResult<Self> {
        let kind = kind_of(kind)?;
        let inner = match kind {
            "static" => Adapter::Static(StaticSelectiveDiscovery::new()),
            _ => Adapter::Temporal(TemporalSelectiveDiscovery::new()),
        };
        Ok(Self { inner: Some(inner), kind })
    }

    /// Discover, reusing retained statistics where the plan allows. Returns the run report
    /// JSON or a refusal.
    #[pyo3(signature = (contract_json, names, columns, *, threads=None, cancel=None))]
    fn run(
        &mut self,
        py: Python<'_>,
        contract_json: &str,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> Reply {
        self.drive(py, contract_json, &names, &columns, threads, cancel, false)
    }

    /// What `run` would reuse, without executing.
    #[pyo3(signature = (contract_json, names, columns, *, threads=None, cancel=None))]
    fn plan(
        &mut self,
        py: Python<'_>,
        contract_json: &str,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> Reply {
        self.drive(py, contract_json, &names, &columns, threads, cancel, true)
    }

    /// The retained outcome as JSON, or `None` before any run.
    fn outcome_json(&self) -> Option<String> {
        self.inner.as_ref().and_then(Adapter::outcome).map(outcome_json)
    }

    /// Statistics currently retained.
    fn retained_statistics(&self) -> usize {
        self.inner.as_ref().map_or(0, Adapter::retained)
    }

    /// The portable state summary and the artifact bytes with identity, or a refusal.
    #[allow(clippy::type_complexity, reason = "native triple of summary, bytes and identity")]
    fn export(
        &self,
        py: Python<'_>,
        artifact_id: &str,
    ) -> PyResult<(Option<(String, Py<PyBytes>, String)>, Option<String>)> {
        let Some(adapter) = self.inner.as_ref() else {
            return Ok((None, None));
        };
        let state = match adapter.export_state() {
            Ok(state) => state,
            Err(error) => {
                return match error_json(&error) {
                    Some(refusal) => Ok((None, Some(refusal))),
                    None => Err(discovery_failure(&error)),
                };
            }
        };
        if artifact_id.trim().is_empty() {
            return Err(input_err("artifact_id must be non-empty"));
        }
        match SelectiveDiscoveryArtifact::seal(&state) {
            Ok((artifact, _)) => match SelectiveDiscoveryArtifact::to_bytes(&state, artifact_id) {
                Ok(bytes) => Ok((
                    Some((
                        state_json(&state),
                        PyBytes::new(py, &bytes).unbind(),
                        artifact.identity().to_owned(),
                    )),
                    None,
                )),
                Err(error) => Ok((None, Some(artifact_refusal(&error)))),
            },
            Err(error) => Ok((None, Some(artifact_refusal(&error)))),
        }
    }

    /// Refusal of a recovery or error-control claim for the retained structure.
    fn request_recovery_claim(&self) -> String {
        self.closed_claim(|o| request_recovery_claim(o).map(|_| ()))
    }

    /// Refusal of an interval for a downstream estimate resting on the retained structure.
    fn request_downstream_interval(&self) -> String {
        self.closed_claim(|o| request_downstream_interval(o).map(|_| ()))
    }

    /// Refusal of a selection diagram built from the retained structure.
    fn request_selection_diagram(&self, targets: Vec<u32>) -> String {
        let Some(outcome) = self.inner.as_ref().and_then(Adapter::outcome) else {
            return refusal_json(
                antecedent_core::reason_code!("invalid_argument"),
                STAGE,
                "selective_discovery.nothing_retained",
                "no discovery has run",
                "run discovery before requesting a selection diagram",
            );
        };
        let ids: Vec<VariableId> = targets.into_iter().map(VariableId::from_raw).collect();
        match request_selection_diagram(outcome, &ids) {
            Err(refusal) => structure_refusal(&refusal),
            Ok(never) => match never {},
        }
    }
}

impl SelectiveDiscoveryHandle {
    fn closed_claim(
        &self,
        request: impl FnOnce(&DiscoveryOutcome) -> Result<(), SelectiveDiscoveryError>,
    ) -> String {
        let Some(outcome) = self.inner.as_ref().and_then(Adapter::outcome) else {
            return refusal_json(
                antecedent_core::reason_code!("invalid_argument"),
                STAGE,
                "selective_discovery.nothing_retained",
                "no discovery has run",
                "run discovery before making a claim request",
            );
        };
        match request(outcome) {
            Err(error) => error_json(&error).unwrap_or_default(),
            Ok(()) => String::new(),
        }
    }

    #[allow(clippy::too_many_arguments, reason = "mirrors the pyfunction signature")]
    fn drive(
        &mut self,
        py: Python<'_>,
        contract_json: &str,
        names: &[String],
        columns: &[PyReadonlyArray1<'_, f64>],
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        plan_only: bool,
    ) -> Reply {
        let parsed = parse_json(contract_json)?;
        let map = contract_map(&parsed, self.kind)?;
        let seed = get_u64(map, "seed")?;
        let batch = crate::columns_to_batch(names, columns)?;
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let adapter = self.inner.take().ok_or_else(|| input_err("handle is unavailable"))?;
        let n = names.len();
        let outcome: PyResult<Ran> = match adapter {
            Adapter::Static(mut adapter) => {
                let contract = match static_contract(map, n) {
                    Ok(c) => c,
                    Err(e) => {
                        self.inner = Some(Adapter::Static(adapter));
                        return Err(e);
                    }
                };
                crate::detach_catch(py, move || {
                    let (data, _): (TabularData, _) =
                        crate::discovery_api::tabular_from_batch(&batch)?;
                    let result = if plan_only {
                        adapter.plan(&contract, &data, &ctx).map(|p| plan_json(&p).to_string())
                    } else {
                        adapter.run(&contract, &data, &ctx).map(|r| report_json(&r))
                    };
                    Ok((Adapter::Static(adapter), result))
                })
            }
            Adapter::Temporal(mut adapter) => {
                let contract = match temporal_contract(map, n) {
                    Ok(c) => c,
                    Err(e) => {
                        self.inner = Some(Adapter::Temporal(adapter));
                        return Err(e);
                    }
                };
                crate::detach_catch(py, move || {
                    let (data, _): (TimeSeriesData, _) =
                        crate::discovery_api::series_from_batch(&batch)?;
                    let result = if plan_only {
                        adapter.plan(&contract, &data, &ctx).map(|p| plan_json(&p).to_string())
                    } else {
                        adapter.run(&contract, &data, &ctx).map(|r| report_json(&r))
                    };
                    Ok((Adapter::Temporal(adapter), result))
                })
            }
        };
        match outcome {
            Ok((adapter, result)) => {
                self.inner = Some(adapter);
                reply(result)
            }
            Err(error) => {
                // The detached call failed before returning the adapter (an input or
                // conversion failure inside the closure): start from no retained state.
                self.inner = Some(match self.kind {
                    "static" => Adapter::Static(StaticSelectiveDiscovery::new()),
                    _ => Adapter::Temporal(TemporalSelectiveDiscovery::new()),
                });
                Err(error)
            }
        }
    }
}

type Resumed = (Option<SelectiveDiscoveryHandle>, Option<String>, Option<String>);

/// Resume from artifact bytes: the algorithm is replayed over the receipts only.
///
/// `columns` is `None` when the consumer has no raw data, which is refused.
#[pyfunction]
#[pyo3(signature = (kind, contract_json, names, columns, artifact, expected_identity=None, *, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one resume request")]
fn selective_discovery_resume(
    py: Python<'_>,
    kind: &str,
    contract_json: &str,
    names: Vec<String>,
    columns: Option<Vec<PyReadonlyArray1<'_, f64>>>,
    artifact: &[u8],
    expected_identity: Option<&str>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Resumed> {
    let kind = kind_of(kind)?;
    let parsed = parse_json(contract_json)?;
    let map = contract_map(&parsed, kind)?;
    let seed = get_u64(map, "seed")?;
    let state = match SelectiveDiscoveryArtifact::from_bytes(artifact, expected_identity) {
        Ok(a) => a.state().clone(),
        Err(error) => return Ok((None, None, Some(artifact_refusal(&error)))),
    };
    let batch = match columns {
        Some(columns) => Some(crate::columns_to_batch(&names, &columns)?),
        None => None,
    };
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let n = names.len();
    let result: Result<(Adapter, SelectiveRunReport), SelectiveDiscoveryError> = if kind == "static"
    {
        let contract = static_contract(map, n)?;
        crate::detach_catch(py, move || {
            let data = match &batch {
                Some(b) => Some(crate::discovery_api::tabular_from_batch(b)?.0),
                None => None,
            };
            Ok(StaticSelectiveDiscovery::resume(&contract, data.as_ref(), &state, &ctx)
                .map(|(a, r)| (Adapter::Static(a), r)))
        })?
    } else {
        let contract = temporal_contract(map, n)?;
        crate::detach_catch(py, move || {
            let data = match &batch {
                Some(b) => Some(crate::discovery_api::series_from_batch(b)?.0),
                None => None,
            };
            Ok(TemporalSelectiveDiscovery::resume(&contract, data.as_ref(), &state, &ctx)
                .map(|(a, r)| (Adapter::Temporal(a), r)))
        })?
    };
    match result {
        Ok((adapter, report)) => Ok((
            Some(SelectiveDiscoveryHandle { inner: Some(adapter), kind }),
            Some(report_json(&report)),
            None,
        )),
        Err(error) => match error_json(&error) {
            Some(refusal) => Ok((None, None, Some(refusal))),
            None => Err(discovery_failure(&error)),
        },
    }
}

/// The stored metadata of an artifact, readable without raw data; nothing is replayed.
#[pyfunction]
fn selective_discovery_inspect(artifact: &[u8]) -> (Option<String>, Option<String>) {
    match SelectiveDiscoveryArtifact::inspect(artifact) {
        Ok(meta) => match serde_json::to_string(&meta) {
            Ok(text) => (Some(text), None),
            Err(error) => (
                None,
                Some(refusal_json(
                    antecedent_core::reason_code!("invalid_argument"),
                    STAGE,
                    "selective_discovery.artifact_invalid",
                    &error.to_string(),
                    "supply an intact selective_discovery_v1 artifact",
                )),
            ),
        },
        Err(error) => (None, Some(artifact_refusal(&error))),
    }
}

/// Validate an artifact and return its identity, checking an independently retained identity.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None))]
fn selective_discovery_consume(
    artifact: &[u8],
    expected_identity: Option<&str>,
) -> (Option<String>, Option<String>) {
    match SelectiveDiscoveryArtifact::from_bytes(artifact, expected_identity) {
        Ok(a) => (
            Some(
                json!({ "identity": a.identity(), "state": serde_json::from_str::<Json>(&state_json(a.state())).unwrap_or(Json::Null) })
                    .to_string(),
            ),
            None,
        ),
        Err(error) => (None, Some(artifact_refusal(&error))),
    }
}

/// The refusal of score-based search reuse.
#[pyfunction]
fn selective_discovery_score_search_refusal() -> String {
    match request_score_search_reuse() {
        Err(error) => error_json(&error).unwrap_or_default(),
        Ok(never) => match never {},
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<SelectiveDiscoveryHandle>()?;
    module.add_function(wrap_pyfunction!(selective_discovery_resume, module)?)?;
    module.add_function(wrap_pyfunction!(selective_discovery_inspect, module)?)?;
    module.add_function(wrap_pyfunction!(selective_discovery_consume, module)?)?;
    module.add_function(wrap_pyfunction!(selective_discovery_score_search_refusal, module)?)?;
    Ok(())
}
