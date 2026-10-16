//! Python bridge for selective discovery.
//!
//! Python declares a contract as JSON (variables are positions in the supplied columns) and
//! passes the searched columns. The statistic store, the stage plan, the algorithms, the
//! work counters, the portable state and every refusal are Rust's. A refusal comes back as
//! structured JSON for the Python layer to raise as a typed exception; a failure of the
//! discovery algorithm itself (data, test, graph, cancellation) raises `CausalDiscoveryError`.
//!
//! No recovery, error-control or interval statement is produced.

use antecedent::analysis::per_population_discovery::{
    PerPopulationDiscovery, PerPopulationReport, PopulationRows, add_per_population_claims,
    request_selection_targets_from_differences,
};
use antecedent::analysis::sample_split_analysis::{
    AnalysisBinding, SplitDiscovery, SplitRule, WHOLE_METHOD_CALIBRATION, discover_on_split,
    split_evidence_sources,
};
use antecedent::analysis::sample_split_effect::{
    SplitEffect, estimate_effect_on_split, measure_source,
};
use antecedent::analysis::sample_split_parent_set_effect::{
    ParentSetEffect, ParentSetOptions, Transformation, estimate_parent_set_effect,
    measure_source as measure_parent_set_source,
};
use antecedent::analysis::selective_discovery::{
    DeclaredCompletion, DiscoveryClaimKeys, DiscoverySnapshot, add_discovery_claims,
    discovered_graph_source, discovery_data_source, discovery_rows_source,
};
use antecedent_contract::composition_verification::{CompositionGraph, GraphError};
use antecedent_contract::evidence_ledger::{SourceDeclaration, SourceId, parse_digest_hex};
use antecedent_core::{Lag, VariableId};
use antecedent_data::{
    OwnedColumnarStorage, TableView, TabularData, TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::constraints::{BackgroundKnowledge, CiMaskType, DiscoveryConstraints};
use antecedent_discovery::result::LaggedLink;
use antecedent_discovery::selective::{
    CiTestSpec, DiscoveredStructure, DiscoveredStructureRefusal, DiscoveryOutcome, DiscoveryPlan,
    NativeResult, PortableDiscoveryState, SelectiveDiscoveryError, SelectiveRunReport,
    StageDecision, StaticAlgorithm, StaticDiscoveryContract, StaticSelectiveDiscovery,
    TemporalAlgorithm, TemporalDiscoveryContract, TemporalSelectiveDiscovery, WorkUnits,
    request_downstream_interval, request_recovery_claim, request_score_search_reuse,
    request_selection_diagram,
};
use antecedent_discovery::selective::{DataIdentity, StageInputs};
use antecedent_io::selective_discovery_artifact::{
    SelectiveDiscoveryArtifact, SelectiveDiscoveryArtifactError,
};
use antecedent_stats::{FdrAdjustment, MultipleTestingMethod, SignificanceMethod};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Map, Value as Json, json};

use crate::evidence_ledger_api::declaration_json;

const STAGE: &str = "selective_discovery";

// ---------------------------------------------------------------------------
// Refusals as data.
// ---------------------------------------------------------------------------

pub(crate) fn refusal_json(
    code: &str,
    stage: &str,
    detail: &str,
    message: &str,
    remedy: &str,
) -> String {
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

pub(crate) fn error_json(error: &SelectiveDiscoveryError) -> Option<String> {
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

pub(crate) fn input_err(message: impl AsRef<str>) -> PyErr {
    crate::value_err(message)
}

fn field<'a>(map: &'a Map<String, Json>, key: &str) -> Option<&'a Json> {
    map.get(key).filter(|v| !v.is_null())
}

pub(crate) fn get_u64(map: &Map<String, Json>, key: &str) -> PyResult<u64> {
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
    background_knowledge(map)?.apply_to(constraints);
    Ok(())
}

/// Read the background knowledge of a wire object: `required` and `forbidden` links
/// `[source, source_lag, target, target_lag]`, `tiers` (lists of variable positions) and
/// `max_parents`; an absent key states nothing.
///
/// The one parser of background knowledge: the selective contracts and the ordinary discovery
/// entry points both read it here, so the same statement means the same thing on either path.
pub(crate) fn background_knowledge(map: &Map<String, Json>) -> PyResult<BackgroundKnowledge> {
    let tiers: std::sync::Arc<[std::sync::Arc<[VariableId]>]> = match field(map, "tiers") {
        None => std::sync::Arc::from([]),
        Some(tiers) => {
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
            std::sync::Arc::from(out)
        }
    };
    Ok(BackgroundKnowledge {
        required: links(map.get("required"))?.unwrap_or_else(|| std::sync::Arc::from([])),
        forbidden: links(map.get("forbidden"))?.unwrap_or_else(|| std::sync::Arc::from([])),
        tiers,
        max_parents: opt_usize(map, "max_parents")?,
    })
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

pub(crate) fn static_contract(
    map: &Map<String, Json>,
    n: usize,
) -> PyResult<StaticDiscoveryContract> {
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

pub(crate) fn hex(bytes: &[u8; 32]) -> String {
    antecedent_discovery::selective::digest_hex(bytes)
}

pub(crate) fn node_json(node: antecedent_discovery::selective::StructureNode) -> Json {
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

pub(crate) fn outcome_json(outcome: &DiscoveryOutcome) -> String {
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

pub(crate) fn work_json(work: &WorkUnits) -> Json {
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

pub(crate) fn plan_json(plan: &DiscoveryPlan) -> Json {
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

pub(crate) fn report_value(report: &SelectiveRunReport) -> Json {
    json!({
        "plan": plan_json(&report.plan),
        "work": work_json(&report.work),
        "outcome_changed": report.outcome_changed,
        "retained_statistics": report.retained_statistics,
    })
}

pub(crate) fn report_json(report: &SelectiveRunReport) -> String {
    report_value(report).to_string()
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

    fn inputs(&self) -> Option<&StageInputs> {
        match self {
            Self::Static(a) => a.inputs(),
            Self::Temporal(a) => a.inputs(),
        }
    }

    fn data_identity(&self) -> Option<&DataIdentity> {
        match self {
            Self::Static(a) => a.data_identity(),
            Self::Temporal(a) => a.data_identity(),
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

pub(crate) fn contract_map<'a>(parsed: &'a Json, kind: &str) -> PyResult<&'a Map<String, Json>> {
    let map = parsed.as_object().ok_or_else(|| input_err("a contract is a JSON object"))?;
    if get_text(map, "kind")? != kind {
        return Err(input_err(format!("the contract is not a {kind} contract")));
    }
    Ok(map)
}

pub(crate) fn parse_json(text: &str) -> PyResult<Json> {
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

    /// Add this run's data, statistics and structure (and a declared completion) to a
    /// composition graph. Returns `{"graph", "keys"}` JSON, or a refusal of the graph.
    #[pyo3(signature = (graph_json, prefix, completion_label=None, completion_identity=None))]
    fn add_claims(
        &self,
        graph_json: Option<&str>,
        prefix: &str,
        completion_label: Option<&str>,
        completion_identity: Option<&str>,
    ) -> Reply {
        let adapter = self.inner.as_ref();
        let (Some(inputs), Some(data), Some(outcome)) = (
            adapter.and_then(Adapter::inputs),
            adapter.and_then(Adapter::data_identity),
            adapter.and_then(Adapter::outcome),
        ) else {
            return Err(input_err("run discovery before adding its claims to a graph"));
        };
        let completion = declared_completion(completion_label, completion_identity)?;
        let Some(mut graph) = claim_graph(graph_json)? else {
            return Ok((None, Some(claim_graph_refusal(graph_json)?)));
        };
        let snapshot = DiscoverySnapshot { inputs, data, outcome, completion: completion.as_ref() };
        match add_discovery_claims(&mut graph, prefix, &snapshot) {
            Ok(keys) => Ok((Some(claims_json(&graph, &[(None, keys)])), None)),
            Err(error) => {
                Ok((None, Some(crate::composition_verification_api::graph_refusal(&error))))
            }
        }
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

    /// The structural transport verdict of a mean-curve query across the retained class.
    ///
    /// Returns `(envelope_json, refusal_json)`; a PAG, a temporal structure, an exhausted
    /// search bound, an orientation conflict and a target outside the structure are refusals.
    #[pyo3(signature = (query_json, selection_targets, member_bound=None))]
    fn transport_across_class(
        &self,
        query_json: &str,
        selection_targets: Vec<u32>,
        member_bound: Option<usize>,
    ) -> PyResult<(Option<String>, Option<String>)> {
        let Some(outcome) = self.inner.as_ref().and_then(Adapter::outcome) else {
            return Ok((
                None,
                Some(refusal_json(
                    antecedent_core::reason_code!("invalid_argument"),
                    STAGE,
                    "selective_discovery.nothing_retained",
                    "no discovery has run",
                    "run discovery before requesting transport across its class",
                )),
            ));
        };
        crate::discovered_transport_api::across_outcome(
            outcome,
            query_json,
            &selection_targets,
            member_bound,
        )
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

// ---------------------------------------------------------------------------
// Per-population discovery.
// ---------------------------------------------------------------------------

fn per_population_report_json(report: &PerPopulationReport) -> String {
    let populations: Vec<Json> = report
        .populations
        .iter()
        .map(|p| {
            json!({
                "population": p.population,
                "rows": p.data_identity.rows,
                "data_identity": hex(&p.data_identity.digest),
                "columns": p
                    .data_identity
                    .columns
                    .iter()
                    .map(|(variable, digest)| json!({ "variable": variable.raw(), "digest": hex(digest) }))
                    .collect::<Vec<_>>(),
                "report": report_value(&p.report),
                "structure_identity": hex(&p.structure_identity),
                "result_identity": hex(&p.result_identity),
                "retained_statistics": p.retained_statistics,
                "receipts_identity": p.receipts_identity.as_ref().map(hex),
                "search_incomplete": p.search_incomplete,
                "unresolved_edges": p.unresolved_edges,
            })
        })
        .collect();
    let edges: Vec<Json> = report
        .edges
        .iter()
        .map(|edge| {
            let marks: Vec<Json> = edge
                .marks
                .iter()
                .map(|(population, marks)| {
                    json!({
                        "population": population,
                        "marks": marks.as_ref().map(|(at_a, at_b)| {
                            json!({ "at_a": at_a.as_str(), "at_b": at_b.as_str() })
                        }),
                    })
                })
                .collect();
            json!({
                "a": node_json(edge.a),
                "b": node_json(edge.b),
                "marks": marks,
                "agrees": edge.agrees(),
            })
        })
        .collect();
    json!({
        "kind": "descriptive_comparison",
        "populations": populations,
        "edges": edges,
        "differing_edges": report.differing_edges().len(),
        "identity": hex(&report.identity),
    })
    .to_string()
}

/// Per-population discovery: one retained statistic store per population, nothing pooled.
#[pyclass(name = "PerPopulationDiscoveryHandle", skip_from_py_object)]
pub(crate) struct PerPopulationDiscoveryHandle {
    inner: Option<PerPopulationDiscovery>,
}

#[pymethods]
impl PerPopulationDiscoveryHandle {
    #[new]
    fn new() -> Self {
        Self { inner: Some(PerPopulationDiscovery::new()) }
    }

    /// Run the contract on each population's columns separately. `columns[i]` are the
    /// searched columns (in `names` order) of `populations[i]`. Returns the report JSON or a
    /// refusal.
    #[pyo3(signature = (contract_json, populations, names, columns, *, threads=None, cancel=None))]
    #[allow(clippy::too_many_arguments, reason = "one request: contract, labels, data")]
    fn run(
        &mut self,
        py: Python<'_>,
        contract_json: &str,
        populations: Vec<String>,
        names: Vec<String>,
        columns: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> Reply {
        let parsed = parse_json(contract_json)?;
        let map = contract_map(&parsed, "static")?;
        let seed = get_u64(map, "seed")?;
        let contract = static_contract(map, names.len())?;
        if populations.len() != columns.len() {
            return Err(input_err("give one set of columns per population"));
        }
        let batches = columns
            .iter()
            .map(|set| crate::columns_to_batch(&names, set))
            .collect::<PyResult<Vec<_>>>()?;
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let mut driver = self.inner.take().ok_or_else(|| input_err("handle is unavailable"))?;
        let ran: PyResult<(PerPopulationDiscovery, Result<String, SelectiveDiscoveryError>)> =
            crate::detach_catch(py, move || {
                let mut tables: Vec<TabularData> = Vec::with_capacity(batches.len());
                for batch in &batches {
                    let (data, _): (TabularData, _) =
                        crate::discovery_api::tabular_from_batch(batch)?;
                    tables.push(data);
                }
                let rows: Vec<PopulationRows<'_>> = populations
                    .iter()
                    .zip(&tables)
                    .map(|(population, data)| PopulationRows { population, data })
                    .collect();
                let result = driver.run(&contract, &rows, &ctx).map(per_population_report_json);
                Ok((driver, result))
            });
        match ran {
            Ok((driver, result)) => {
                self.inner = Some(driver);
                reply(result)
            }
            Err(error) => {
                self.inner = Some(PerPopulationDiscovery::new());
                Err(error)
            }
        }
    }

    /// The last report in which every population succeeded, as JSON.
    fn report_json(&self) -> Option<String> {
        self.inner.as_ref().and_then(PerPopulationDiscovery::report).map(per_population_report_json)
    }

    /// Add every population's discovery stages of the last committed report to a composition
    /// graph under `{prefix}.{population}`. Returns `{"graph", "keys"}` JSON, or a refusal of
    /// the graph.
    fn add_claims(&self, graph_json: Option<&str>, prefix: &str) -> Reply {
        let driver = self.inner.as_ref().ok_or_else(|| input_err("handle is unavailable"))?;
        let Some(mut graph) = claim_graph(graph_json)? else {
            return Ok((None, Some(claim_graph_refusal(graph_json)?)));
        };
        match add_per_population_claims(&mut graph, prefix, driver) {
            Ok(keys) => {
                let keyed: Vec<(Option<String>, DiscoveryClaimKeys)> =
                    keys.into_iter().map(|(label, k)| (Some(label), k)).collect();
                Ok((Some(claims_json(&graph, &keyed)), None))
            }
            Err(error) => {
                Ok((None, Some(crate::composition_verification_api::graph_refusal(&error))))
            }
        }
    }

    /// One population's retained outcome as JSON.
    fn outcome_json(&self, population: &str) -> Option<String> {
        self.inner
            .as_ref()
            .and_then(|d| d.adapter(population))
            .and_then(StaticSelectiveDiscovery::outcome)
            .map(outcome_json)
    }

    /// Statistics retained for one population.
    fn retained_statistics(&self, population: &str) -> usize {
        self.inner
            .as_ref()
            .and_then(|d| d.adapter(population))
            .map_or(0, StaticSelectiveDiscovery::retained_statistics)
    }

    /// Refusal of selection targets read from the edges that differ between populations.
    fn request_selection_targets(&self, candidates: Vec<u32>) -> String {
        let Some(report) = self.inner.as_ref().and_then(PerPopulationDiscovery::report) else {
            return refusal_json(
                antecedent_core::reason_code!("invalid_argument"),
                STAGE,
                "selective_discovery.nothing_retained",
                "no per-population discovery has run",
                "run per-population discovery before requesting selection targets",
            );
        };
        let ids: Vec<VariableId> = candidates.into_iter().map(VariableId::from_raw).collect();
        match request_selection_targets_from_differences(report, &ids) {
            Err(refusal) => structure_refusal(&refusal),
            Ok(never) => match never {},
        }
    }
}

// ---------------------------------------------------------------------------
// Discovery followed by analysis on a row split.
// ---------------------------------------------------------------------------

fn rule_of(text: &str) -> PyResult<SplitRule> {
    let parsed = parse_json(text)?;
    let map = parsed.as_object().ok_or_else(|| input_err("a split rule is a JSON object"))?;
    match get_text(map, "kind")?.as_str() {
        "explicit" => {
            let rows = field(map, "discovery_rows")
                .and_then(Json::as_array)
                .ok_or_else(|| input_err("field `discovery_rows` must be a list"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| input_err("discovery rows are row indices"))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(SplitRule::Explicit { discovery_rows: rows })
        }
        "seeded_permutation" => Ok(SplitRule::SeededPermutation {
            seed: get_u64(map, "seed")?,
            discovery_fraction: field(map, "discovery_fraction")
                .and_then(Json::as_f64)
                .ok_or_else(|| input_err("field `discovery_fraction` must be a number"))?,
        }),
        other => Err(input_err(format!("unknown split rule {other:?}"))),
    }
}

fn rule_value(rule: &SplitRule) -> Json {
    match rule {
        SplitRule::Explicit { discovery_rows } => {
            let mut rows = discovery_rows.clone();
            rows.sort_unstable();
            json!({ "kind": "explicit", "discovery_rows": rows })
        }
        SplitRule::SeededPermutation { seed, discovery_fraction } => json!({
            "kind": "seeded_permutation",
            "seed": seed,
            "discovery_fraction": discovery_fraction,
        }),
    }
}

fn binding_json(binding: &AnalysisBinding) -> String {
    json!({
        "receipt_identity": hex(&binding.receipt_identity),
        "structure_identity": hex(&binding.structure_identity),
        "result_identity": hex(&binding.result_identity),
        "analysis_data_identity": hex(&binding.analysis_data_identity),
        "identity": hex(&binding.identity),
        "conditional_on_selected_graph": binding.conditional_on_selected_graph(),
        "whole_method_calibration": binding.whole_method_calibration(),
    })
    .to_string()
}

fn columns_of(py: Python<'_>, data: &TabularData) -> PyResult<Vec<Py<PyArray1<f64>>>> {
    data.schema()
        .variables()
        .iter()
        .map(|spec| {
            let values = data.float64_values(spec.id).map_err(|e| input_err(e.to_string()))?;
            Ok(PyArray1::from_vec(py, values).unbind())
        })
        .collect()
}

/// Rebuild `data` with the observation weights and the analysis mask the caller supplied
/// (`mask[i]` true keeps row `i`). Either may be absent; a length that differs from the row
/// count, or weights that are not a finite non-negative measure, are input errors.
fn table_with_rows(
    data: TabularData,
    weights: Option<Vec<f64>>,
    mask: Option<Vec<bool>>,
) -> PyResult<TabularData> {
    if weights.is_none() && mask.is_none() {
        return Ok(data);
    }
    let n = data.row_count();
    let storage = data.storage();
    let mask = mask
        .map(|keep| {
            if keep.len() != n {
                return Err(input_err(format!("mask has {} entries for {n} rows", keep.len())));
            }
            let mut bytes = vec![0u8; n.div_ceil(8)];
            for (i, kept) in keep.iter().enumerate() {
                if *kept {
                    bytes[i / 8] |= 1 << (i % 8);
                }
            }
            ValidityBitmap::from_bytes(bytes, n).map_err(|e| input_err(e.to_string()))
        })
        .transpose()?;
    let weights = weights
        .map(|w| {
            if w.len() == n {
                Ok(std::sync::Arc::<[f64]>::from(w))
            } else {
                Err(input_err(format!("weights has {} entries for {n} rows", w.len())))
            }
        })
        .transpose()?;
    OwnedColumnarStorage::try_new(
        storage.schema().clone(),
        storage.columns().to_vec(),
        mask.or_else(|| storage.analysis_mask().cloned()),
        weights.or_else(|| storage.weights().map(std::sync::Arc::from)),
    )
    .map(TabularData::new)
    .map_err(|e| input_err(e.to_string()))
}

/// A row split, the discovery run on its discovery part and the receipt tying them together.
#[pyclass(name = "SampleSplitDiscoveryHandle", skip_from_py_object)]
pub(crate) struct SampleSplitDiscoveryHandle {
    inner: SplitDiscovery,
    names: Vec<String>,
}

#[pymethods]
impl SampleSplitDiscoveryHandle {
    /// The split receipt as JSON.
    fn receipt_json(&self) -> String {
        let receipt = self.inner.receipt();
        json!({
            "rule": rule_value(&receipt.rule),
            "n_rows": receipt.n_rows,
            "masked_out_rows": receipt.masked_out_rows,
            "incomplete_rows": receipt.incomplete_rows,
            "n_discovery": receipt.n_discovery,
            "n_analysis": receipt.n_analysis,
            "discovery_rows_identity": hex(&receipt.discovery_rows_identity),
            "analysis_rows_identity": hex(&receipt.analysis_rows_identity),
            "mask_identity": receipt.mask_identity.as_ref().map(hex),
            "weights": receipt.weights.as_ref().map(|w| json!({
                "source": hex(&w.source),
                "discovery": hex(&w.discovery),
                "analysis": hex(&w.analysis),
                "discovery_effective_sample_size": w.discovery_effective_sample_size,
                "analysis_effective_sample_size": w.analysis_effective_sample_size,
            })),
            "source_data_identity": hex(&receipt.source_data.digest),
            "discovery_data_identity": hex(&receipt.discovery_data.digest),
            "analysis_data_identity": hex(&receipt.analysis_data.digest),
            "identity": hex(&receipt.identity),
            "conditional_on_selected_graph": true,
            "whole_method_calibration": WHOLE_METHOD_CALIBRATION,
        })
        .to_string()
    }

    /// The original row index of each discovery row, in the part's row order.
    fn discovery_row_ids(&self) -> Vec<usize> {
        self.inner.discovery_row_ids().to_vec()
    }

    /// The original row index of each analysis row, in the part's row order.
    fn analysis_row_ids(&self) -> Vec<usize> {
        self.inner.analysis_row_ids().to_vec()
    }

    /// The analysis rows' weights, unnormalized; `None` for an unweighted source.
    fn analysis_weights(&self) -> Option<Vec<f64>> {
        self.inner.analysis_weights().map(<[f64]>::to_vec)
    }

    /// The discovery outcome as JSON.
    fn outcome_json(&self) -> String {
        outcome_json(self.inner.outcome())
    }

    /// The discovery part's columns, in `names` order.
    fn discovery_columns(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray1<f64>>>> {
        columns_of(py, self.inner.discovery_rows())
    }

    /// The analysis part's columns, in `names` order.
    fn analysis_columns(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray1<f64>>>> {
        columns_of(py, self.inner.analysis_rows())
    }

    /// The two parts as evidence-ledger source declarations `(discovery, analysis)`, keyed
    /// by `snapshot_id` and row index; or a refusal when the receipt does not verify.
    fn evidence_sources(&self, snapshot_id: &[u8]) -> Reply {
        match split_evidence_sources(self.inner.receipt(), snapshot_id) {
            Ok((discovery, analysis)) => Ok((
                Some(
                    json!({
                        "discovery": declaration_json(&discovery),
                        "analysis": declaration_json(&analysis),
                    })
                    .to_string(),
                ),
                None,
            )),
            Err(error) => match error_json(&error) {
                Some(refusal) => Ok((None, Some(refusal))),
                None => Err(discovery_failure(&error)),
            },
        }
    }

    /// Re-derive the partition from the rule; a refusal when the receipt does not hold.
    fn verify(&self) -> Option<String> {
        self.inner.verify().err().and_then(|e| error_json(&e))
    }

    /// Replay the split on the supplied source table (columns, weights, mask) and require it to
    /// be the table the receipt describes; a refusal when it is not.
    fn verify_source(
        &self,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        weights: Option<Vec<f64>>,
        mask: Option<Vec<bool>>,
    ) -> PyResult<Option<String>> {
        if names != self.names {
            return Err(input_err(
                "the source table must carry the discovery table's columns in the same order",
            ));
        }
        let batch = crate::columns_to_batch(&names, &columns)?;
        let (data, _): (TabularData, _) = crate::discovery_api::tabular_from_batch(&batch)?;
        let data = table_with_rows(data, weights, mask)?;
        match self.inner.verify_against_source(&data) {
            Ok(()) => Ok(None),
            Err(error) => match error_json(&error) {
                Some(refusal) => Ok(Some(refusal)),
                None => Err(discovery_failure(&error)),
            },
        }
    }

    /// Check that the supplied columns (and weights) are exactly the analysis part; returns the
    /// binding JSON or a refusal.
    fn check_analysis_table(
        &self,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        weights: Option<Vec<f64>>,
    ) -> PyResult<(Option<String>, Option<String>)> {
        if names != self.names {
            return Err(input_err(
                "the analysis table must carry the discovery table's columns in the same order",
            ));
        }
        let batch = crate::columns_to_batch(&names, &columns)?;
        let (data, _): (TabularData, _) = crate::discovery_api::tabular_from_batch(&batch)?;
        // A split of a masked source gives each part the all-included mask of rows that passed
        // it, so the table to check carries that mask too.
        let mask = self.inner.receipt().mask_identity.map(|_| vec![true; data.row_count()]);
        let data = table_with_rows(data, weights, mask)?;
        match self.inner.check_analysis_table(&data) {
            Ok(binding) => Ok((Some(binding_json(&binding)), None)),
            Err(error) => match error_json(&error) {
                Some(refusal) => Ok((None, Some(refusal))),
                None => Err(discovery_failure(&error)),
            },
        }
    }

    /// Refusal of an interval for an analysis that reused the rows discovery saw.
    fn request_unsplit_interval(&self) -> String {
        match self.inner.request_unsplit_interval() {
            Err(error) => error_json(&error).unwrap_or_default(),
            Ok(never) => match never {},
        }
    }
}

type Split = (Option<SampleSplitDiscoveryHandle>, Option<String>, Option<String>);

/// Split the rows by the declared rule and discover on the discovery part.
///
/// Returns `(handle, discovery_report_json, refusal)`.
#[pyfunction]
#[pyo3(signature = (contract_json, names, columns, rule_json, *, weights=None, mask=None, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one request: contract, data, weights, mask, rule")]
fn sample_split_discover(
    py: Python<'_>,
    contract_json: &str,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    rule_json: &str,
    weights: Option<Vec<f64>>,
    mask: Option<Vec<bool>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Split> {
    let parsed = parse_json(contract_json)?;
    let map = contract_map(&parsed, "static")?;
    let seed = get_u64(map, "seed")?;
    let contract = static_contract(map, names.len())?;
    let rule = rule_of(rule_json)?;
    let batch = crate::columns_to_batch(&names, &columns)?;
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let result: Result<SplitDiscovery, SelectiveDiscoveryError> =
        crate::detach_catch(py, move || {
            let (data, _): (TabularData, _) = crate::discovery_api::tabular_from_batch(&batch)?;
            let data = table_with_rows(data, weights, mask)?;
            Ok(discover_on_split(&contract, &data, rule, &ctx))
        })?;
    match result {
        Ok(split) => {
            let report = report_json(split.report());
            Ok((Some(SampleSplitDiscoveryHandle { inner: split, names }), Some(report), None))
        }
        Err(error) => match error_json(&error) {
            Some(refusal) => Ok((None, None, Some(refusal))),
            None => Err(discovery_failure(&error)),
        },
    }
}

/// The refusal an io-level split-effect failure carries, as the structured JSON the Python
/// layer raises; any other io failure is an error, not a refusal.
fn effect_refusal(error: antecedent_io::IoError) -> PyResult<String> {
    match error {
        antecedent_io::IoError::Refused { code, message } => {
            let detail = message.split_once(": ").map_or("refused", |(d, _)| d);
            Ok(refusal_json(code, STAGE, detail, &message, &message))
        }
        other => Err(crate::transport_common::error(other)),
    }
}

fn effect_json(estimate: &SplitEffect) -> String {
    let basis = estimate.keyed_basis();
    let (lower, upper) = estimate.interval();
    json!({
        "effect": estimate.effect(),
        "lower": lower,
        "upper": upper,
        "level": estimate.level(),
        "standard_error": estimate.standard_error(),
        "analysis_rows": estimate.analysis_rows(),
        "discovery_rows": estimate.discovery_rows(),
        "adjustment_set": estimate.adjustment_set(),
        "receipt_identity": hex(estimate.receipt_identity()),
        "structure_identity": hex(estimate.structure_identity()),
        "basis": basis_json(&estimate.keyed_basis()),
    })
    .to_string()
}

fn parent_set_effect_json(estimate: &ParentSetEffect) -> String {
    let reported = estimate.reported();
    let unresolved: Vec<Json> = estimate
        .unresolved_edges()
        .iter()
        .map(|edge| json!({ "a": edge.a, "b": edge.b, "reason": edge.reason }))
        .collect();
    json!({
        "scalar": reported.name,
        "effect": reported.point,
        "lower": reported.lower,
        "upper": reported.upper,
        "level": estimate.level(),
        "standard_error": reported.standard_error,
        "coefficient": estimate.coefficient(),
        "transformation": estimate.transformation().name(),
        "analysis_rows": estimate.analysis_rows(),
        "discovery_rows": estimate.discovery_rows(),
        "treatment_parents": estimate.treatment_parents(),
        "treatment_children": estimate.treatment_children(),
        "unresolved_edges": unresolved,
        "edge_count": estimate.edge_count(),
        "acceptance": estimate.acceptance(),
        "receipt_identity": hex(estimate.receipt_identity()),
        "structure_identity": hex(estimate.structure_identity()),
        "basis": basis_json(&estimate.keyed_basis()),
    })
    .to_string()
}

type Effect = (Option<String>, Option<Vec<u8>>, Option<String>);

/// The linear effect of a treatment on an outcome after PC discovery on one half of a seeded
/// row split and parent adjustment on the other half.
///
/// Returns `(estimate_json, source_bytes, refusal)`.
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, split_seed, seed, *, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one request: data, roles and seeds")]
fn sample_split_effect(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    treatment: u32,
    outcome: u32,
    split_seed: u64,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Effect> {
    let batch = crate::columns_to_batch(&names, &columns)?;
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let result = crate::detach_catch(py, move || {
        let (data, _): (TabularData, _) = crate::discovery_api::tabular_from_batch(&batch)?;
        Ok(estimate_effect_on_split(
            &data,
            VariableId::from_raw(treatment),
            VariableId::from_raw(outcome),
            split_seed,
            &ctx,
        ))
    })?;
    match result {
        Ok(estimate) => {
            Ok((Some(effect_json(&estimate)), Some(estimate.source_artifact().to_vec()), None))
        }
        Err(error) => Ok((None, None, Some(effect_refusal(error)?))),
    }
}

type Measured = (Option<crate::measured_inference_api::NativeMeasuredInference>, Option<String>);

/// Replay a split-effect source and attest its interval from the coverage records.
///
/// Returns `(measured, refusal)`; a refusal is `cell_not_licensed` when no current attesting
/// record covers the construction at this analysis row count.
#[pyfunction]
#[pyo3(signature = (source, seed, *, threads=None, cancel=None))]
fn sample_split_effect_measure(
    py: Python<'_>,
    source: Vec<u8>,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Measured> {
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let result = crate::detach_catch(py, move || Ok(measure_source(&source, &ctx)))?;
    match result {
        Ok(measured) => Ok((
            Some(crate::measured_inference_api::NativeMeasuredInference::from_io(measured)?),
            None,
        )),
        Err(error) => Ok((None, Some(effect_refusal(error)?))),
    }
}

/// The effect of a treatment on an outcome after PC discovery on one half of a seeded row split,
/// accepted on the treatment's parent set, for a declared linear or log-linear study.
///
/// The transformation is parsed before any data are read. Returns
/// `(estimate_json, source_bytes, refusal)`.
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, outcome, split_seed, seed, transformation, max_degree, log_shift=None, *, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one request: data, roles, declarations and seeds")]
fn sample_split_parent_set_effect(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    treatment: u32,
    outcome: u32,
    split_seed: u64,
    seed: u64,
    transformation: &str,
    max_degree: u32,
    log_shift: Option<f64>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Effect> {
    let transformation = match Transformation::parse(transformation) {
        Ok(declared) => declared,
        Err(error) => return Ok((None, None, Some(effect_refusal(error)?))),
    };
    let options = ParentSetOptions { transformation, max_degree, log_shift };
    let batch = crate::columns_to_batch(&names, &columns)?;
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let result = crate::detach_catch(py, move || {
        let (data, _): (TabularData, _) = crate::discovery_api::tabular_from_batch(&batch)?;
        Ok(estimate_parent_set_effect(
            &data,
            VariableId::from_raw(treatment),
            VariableId::from_raw(outcome),
            options,
            split_seed,
            &ctx,
        ))
    })?;
    match result {
        Ok(estimate) => Ok((
            Some(parent_set_effect_json(&estimate)),
            Some(estimate.source_artifact().to_vec()),
            None,
        )),
        Err(error) => Ok((None, None, Some(effect_refusal(error)?))),
    }
}

/// Replay a parent-set split-effect source and attest its interval from the coverage records.
///
/// Returns `(measured, refusal)`; a refusal is `cell_not_licensed` when no current attesting
/// record covers the construction at this variable count, degree and analysis row count.
#[pyfunction]
#[pyo3(signature = (source, seed, *, threads=None, cancel=None))]
fn sample_split_parent_set_effect_measure(
    py: Python<'_>,
    source: Vec<u8>,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Measured> {
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let result = crate::detach_catch(py, move || Ok(measure_parent_set_source(&source, &ctx)))?;
    match result {
        Ok(measured) => Ok((
            Some(crate::measured_inference_api::NativeMeasuredInference::from_io(measured)?),
            None,
        )),
        Err(error) => Ok((None, Some(effect_refusal(error)?))),
    }
}

fn declared_completion(
    label: Option<&str>,
    identity: Option<&str>,
) -> PyResult<Option<DeclaredCompletion>> {
    match (label, identity) {
        (None, None) => Ok(None),
        (Some(label), Some(identity)) => {
            let digest = parse_digest_hex(identity)
                .ok_or_else(|| input_err("a completion identity is 64 hexadecimal characters"))?;
            Ok(Some(DeclaredCompletion { label: label.to_owned(), identity: *digest.as_bytes() }))
        }
        _ => Err(input_err("a declared completion needs both a label and an identity")),
    }
}

/// The graph to add to: empty when none is given, `None` when the declaration does not
/// validate (the refusal is rendered by [`claim_graph_refusal`]).
fn claim_graph(graph_json: Option<&str>) -> PyResult<Option<CompositionGraph>> {
    Ok(parsed_graph(graph_json)?.ok())
}

fn parsed_graph(graph_json: Option<&str>) -> PyResult<Result<CompositionGraph, GraphError>> {
    match graph_json {
        None => Ok(Ok(CompositionGraph::new())),
        Some(text) => {
            let value: Json = serde_json::from_str(text)
                .map_err(|e| input_err(format!("graph is not valid JSON: {e}")))?;
            crate::composition_verification_api::parse_graph(&value, "graph")
        }
    }
}

fn claim_graph_refusal(graph_json: Option<&str>) -> PyResult<String> {
    match parsed_graph(graph_json)? {
        Err(error) => Ok(crate::composition_verification_api::graph_refusal(&error)),
        Ok(_) => Err(input_err("the graph validated")),
    }
}

fn claims_json(graph: &CompositionGraph, keys: &[(Option<String>, DiscoveryClaimKeys)]) -> String {
    let wire = |k: &DiscoveryClaimKeys| {
        json!({
            "data": k.data,
            "statistics": k.statistics,
            "graph": k.graph,
            "completion": k.completion,
            "downstream_anchor": k.downstream_anchor(),
        })
    };
    let keys = match keys {
        [(None, only)] => wire(only),
        many => Json::Object(
            many.iter().map(|(label, k)| (label.clone().unwrap_or_default(), wire(k))).collect(),
        ),
    };
    json!({
        "graph": crate::composition_verification_api::graph_wire(graph),
        "keys": keys,
    })
    .to_string()
}

fn declaration_text(declaration: &SourceDeclaration) -> String {
    declaration_json(declaration).to_string()
}

/// Declare a contiguous row range of a dataset as an evidence source (JSON declaration).
#[pyfunction]
fn selective_discovery_data_source(
    label: &str,
    snapshot_id: &[u8],
    start: u64,
    stop: u64,
) -> PyResult<String> {
    if stop < start {
        return Err(input_err("a row range ends at or after its start"));
    }
    Ok(declaration_text(&discovery_data_source(label, snapshot_id, start..stop)))
}

/// Declare an explicit set of rows as an evidence source (JSON declaration).
#[pyfunction]
fn selective_discovery_rows_source(label: &str, snapshot_id: &[u8], rows: Vec<usize>) -> String {
    declaration_text(&discovery_rows_source(label, snapshot_id, &rows))
}

/// Declare a discovered graph as a fitted-model source derived from used data sources.
#[pyfunction]
fn selective_discovery_graph_source(label: &str, data_sources: Vec<String>) -> PyResult<String> {
    let ids = data_sources
        .iter()
        .map(|id| {
            SourceId::from_hex(id)
                .ok_or_else(|| input_err("a source id is 64 hexadecimal characters"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(declaration_text(&discovered_graph_source(label, ids)))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(selective_discovery_data_source, module)?)?;
    module.add_function(wrap_pyfunction!(selective_discovery_rows_source, module)?)?;
    module.add_function(wrap_pyfunction!(selective_discovery_graph_source, module)?)?;
    module.add_class::<SelectiveDiscoveryHandle>()?;
    module.add_function(wrap_pyfunction!(selective_discovery_resume, module)?)?;
    module.add_function(wrap_pyfunction!(selective_discovery_inspect, module)?)?;
    module.add_function(wrap_pyfunction!(selective_discovery_consume, module)?)?;
    module.add_function(wrap_pyfunction!(selective_discovery_score_search_refusal, module)?)?;
    module.add_class::<PerPopulationDiscoveryHandle>()?;
    module.add_class::<SampleSplitDiscoveryHandle>()?;
    module.add_function(wrap_pyfunction!(sample_split_discover, module)?)?;
    module.add_function(wrap_pyfunction!(sample_split_effect, module)?)?;
    module.add_function(wrap_pyfunction!(sample_split_effect_measure, module)?)?;
    module.add_function(wrap_pyfunction!(sample_split_parent_set_effect, module)?)?;
    module.add_function(wrap_pyfunction!(sample_split_parent_set_effect_measure, module)?)?;
    Ok(())
}
