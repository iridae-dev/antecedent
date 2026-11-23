//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

mod static_routes;
mod temporal_routes;

use crate::NumericColumn;
use crate::compiled_route_api::{DiscoverInput, RouteRecord};
use crate::*;
use antecedent::discovery::BackgroundKnowledge;
use antecedent::{DiscoveredStructure, DiscoverySpec, GraphRef};
use pyo3::prelude::*;
use pyo3::types::PyAny;

fn endpoint_name(e: Endpoint) -> &'static str {
    match e {
        Endpoint::Tail => "tail",
        Endpoint::Arrow => "arrow",
        Endpoint::Circle => "circle",
        Endpoint::Conflict => "conflict",
    }
}

fn node_ref_parts(names: &[String], node: NodeRef) -> (String, u32) {
    match node {
        NodeRef::Lagged { variable, lag } => (
            names
                .get(variable.as_usize())
                .cloned()
                .unwrap_or_else(|| format!("var{}", variable.raw())),
            lag.raw(),
        ),
        // Discovery graphs are never unfolded; an unfolded slot reads as its history lag.
        NodeRef::Unfolded { variable, offset } => (
            names
                .get(variable.as_usize())
                .cloned()
                .unwrap_or_else(|| format!("var{}", variable.raw())),
            u32::try_from(-i64::from(offset)).unwrap_or(0),
        ),
        NodeRef::Static(variable) | NodeRef::Context { variable, .. } => (
            names
                .get(variable.as_usize())
                .cloned()
                .unwrap_or_else(|| format!("var{}", variable.raw())),
            0,
        ),
    }
}

fn graph_edge_from_marked(names: &[String], nodes: &[NodeRef], edge: MarkedEdge) -> GraphEdge {
    let (source, source_lag) = node_ref_parts(names, nodes[edge.a.as_usize()]);
    let (target, target_lag) = node_ref_parts(names, nodes[edge.b.as_usize()]);
    GraphEdge {
        source,
        source_lag,
        target,
        target_lag,
        at_source: endpoint_name(edge.at_a).to_string(),
        at_target: endpoint_name(edge.at_b).to_string(),
    }
}

fn cpdag_graph_edges(names: &[String], cpdag: &TemporalCpdag) -> Vec<GraphEdge> {
    cpdag.edges().into_iter().map(|e| graph_edge_from_marked(names, cpdag.nodes(), e)).collect()
}

fn static_cpdag_graph_edges(names: &[String], cpdag: &Cpdag) -> Vec<GraphEdge> {
    cpdag.edges().into_iter().map(|e| graph_edge_from_marked(names, cpdag.nodes(), e)).collect()
}

fn static_dag_graph_edges(names: &[String], dag: &Dag) -> Vec<GraphEdge> {
    dag.edges().map(|e| graph_edge_from_marked(names, dag.nodes(), e)).collect()
}

fn pag_graph_edges(names: &[String], pag: &TemporalPag) -> PyResult<Vec<GraphEdge>> {
    let nodes = pag.nodes();
    let mut out = Vec::new();
    for i in 0..pag.node_count() {
        let a = DenseNodeId::from_raw(crate::fit_u32(i)?);
        for (b, at_a, at_b) in pag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            out.push(graph_edge_from_marked(
                names,
                nodes,
                MarkedEdge { a, b, at_a, at_b, middle: MiddleMark::Empty },
            ));
        }
    }
    Ok(out)
}

fn static_pag_graph_edges(names: &[String], pag: &Pag) -> PyResult<Vec<GraphEdge>> {
    let nodes = pag.nodes();
    let mut out = Vec::new();
    for i in 0..pag.node_count() {
        let a = DenseNodeId::from_raw(crate::fit_u32(i)?);
        for (b, at_a, at_b) in pag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            out.push(graph_edge_from_marked(
                names,
                nodes,
                MarkedEdge { a, b, at_a, at_b, middle: MiddleMark::Empty },
            ));
        }
    }
    Ok(out)
}

fn static_pag_definite_directed_count(pag: &Pag) -> u64 {
    let mut directed = 0u64;
    for i in 0..pag.node_count() {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "node ids are u32 by construction (DenseNodeId), so node positions fit u32"
        )]
        let a = DenseNodeId::from_raw(i as u32);
        for (b, at_a, at_b) in pag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            if matches!(
                (at_a, at_b),
                (Endpoint::Tail, Endpoint::Arrow) | (Endpoint::Arrow, Endpoint::Tail)
            ) {
                directed += 1;
            }
        }
    }
    directed
}

/// Edges with a circle endpoint: the PAG-appropriate analogue of a CPDAG's undirected count.
fn circle_edge_count(edges: &[GraphEdge]) -> u64 {
    edges.iter().filter(|e| e.at_source == "circle" || e.at_target == "circle").count() as u64
}

/// One discovered lagged link for Python.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct DiscoveredLink {
    #[pyo3(get)]
    pub(crate) source: String,
    #[pyo3(get)]
    pub(crate) source_lag: u32,
    #[pyo3(get)]
    pub(crate) target: String,
    #[pyo3(get)]
    pub(crate) target_lag: u32,
    #[pyo3(get)]
    pub(crate) statistic: f64,
    #[pyo3(get)]
    pub(crate) p_value: f64,
    /// Benjamini–Hochberg adjusted p-value when FDR ran; otherwise `None`.
    #[pyo3(get)]
    pub(crate) adjusted_p_value: Option<f64>,
    /// Conditioning set recorded for this exact `(source, source_lag) -> target` pair,
    /// as `(variable_name, lag)` pairs. Empty when the discovery run never recorded one
    /// for this pair.
    ///
    /// Sourced from the discovery algorithm's separating-set map (keyed uniquely per
    /// link, one set per key — never a list of alternative sets in the current PC /
    /// PCMCI-family implementations, so this is exposed as a single set rather than a
    /// list of sets). Semantics differ by algorithm and by whether the link survived:
    ///
    /// - PC / GES / FCI / RFCI / PCMCI+ / J-PCMCI+: a non-empty set here would mean the
    ///   pair was found conditionally independent and removed, which is mutually
    ///   exclusive with the link being retained (present in `links`) — so a *retained*
    ///   link from these algorithms always reports an empty conditioning set here.
    /// - Plain PCMCI: the PC1 phase records this set purely to prune conditioning
    ///   candidates (`pa(target)`), not to test-and-remove the exact `(source,
    ///   source_lag) -> target` link — MCI re-tests every candidate pair with its own
    ///   (different) conditioning set regardless of the PC1 outcome. So a *retained*
    ///   plain-PCMCI link can legitimately show a non-empty set here; it documents which
    ///   candidate parents were pruned during PC1, not a separation certificate for the
    ///   retained link's own MCI test.
    /// - LPCMCI: the scored-link and separating-set accumulators are reconciled against
    ///   the final PAG before being returned, so every entry in `links` corresponds to
    ///   an edge in `graph_edges`, and a non-empty set here is populated only for a pair
    ///   actually separated in the final PAG.
    #[pyo3(get)]
    pub(crate) conditioning_set: Vec<(String, u32)>,
}

#[pymethods]
impl DiscoveredLink {
    fn __repr__(&self) -> String {
        format!(
            "DiscoveredLink({}{} -> {}{}, p={:.3})",
            self.source,
            lagged_node_suffix(self.source_lag),
            self.target,
            lagged_node_suffix(self.target_lag),
            self.p_value
        )
    }
}

/// Format a lag as ``"[t]"`` (lag 0) or ``"[t-N]"`` (lag > 0), for `__repr__` use.
fn lagged_node_suffix(lag: u32) -> String {
    if lag == 0 { "[t]".to_string() } else { format!("[t-{lag}]") }
}

/// Coarse-grained PCMCI discovery result (single boundary crossing).
///
/// Field set is the stable Rust↔Python temporal discovery schema for .
#[pyclass]
pub(crate) struct PcmciDiscoveryResult {
    #[pyo3(get)]
    pub(crate) links: Vec<DiscoveredLink>,
    #[pyo3(get)]
    pub(crate) algorithm_id: String,
    #[pyo3(get)]
    pub(crate) algorithm_config: String,
    #[pyo3(get)]
    pub(crate) ci_tests: u64,
    #[pyo3(get)]
    pub(crate) links_retained: u64,
    #[pyo3(get)]
    pub(crate) pending_edge_count: u64,
    #[pyo3(get)]
    pub(crate) lagged_frame_bytes: u64,
    #[pyo3(get)]
    pub(crate) worker_threads: u32,
    #[pyo3(get)]
    pub(crate) ci_name: String,
    #[pyo3(get)]
    pub(crate) cpdag_nodes: u64,
    #[pyo3(get)]
    pub(crate) cpdag_directed_edges: u64,
    /// Undirected edges of a CPDAG result; always 0 for PAG results (FCI, RFCI, LPCMCI),
    /// which have circle endpoints instead (see `pag_circle_edges`).
    #[pyo3(get)]
    pub(crate) cpdag_undirected_edges: u64,
    /// Edges of a PAG result with at least one circle endpoint; `None` for non-PAG results.
    #[pyo3(get)]
    pub(crate) pag_circle_edges: Option<u64>,
    /// Oriented graph body (CPDAG/PAG marks); empty for lagged-only PCMCI.
    #[pyo3(get)]
    pub(crate) graph_edges: Vec<GraphEdge>,
    /// The schema's variable names, in column order. Includes variables the
    /// algorithm found no adjacency for, which no edge list can name.
    #[pyo3(get)]
    pub(crate) variable_names: Vec<String>,
    /// Hex identity of the background knowledge the run was given (required and forbidden
    /// links, tiers, parent limit); `None` when it was given none. The same digest the
    /// selective contracts declare, so it changes exactly when the knowledge does.
    #[pyo3(get)]
    pub(crate) background_knowledge: Option<String>,
    /// The Rust review artifact and variable names this discovery produced.
    pub(crate) review: Option<Arc<(ReviewSlot, Vec<String>)>>,
    /// Row accounting of a run that reads table rows (NOTEARS), as JSON: rows used and
    /// dropped, the missing count per variable and the dropped-row digest.
    pub(crate) row_accounting_json: Option<String>,
    /// The compiled discovery route this result came from.
    pub(crate) route: Option<RouteRecord>,
}

/// Discovery review artifact retained so acceptance runs the Rust review gate.
pub(crate) enum ReviewSlot {
    Dag(DagReview),
    Cpdag(CpdagReview),
    Pag(Pag, PagReview),
    TemporalGraph(TemporalGraphReview),
    TemporalCpdag(TemporalCpdagReview),
    TemporalPag(TemporalPag, TemporalPagReview),
}

impl PcmciDiscoveryResult {
    fn with_review(mut self, slot: ReviewSlot, names: &[String]) -> Self {
        self.review = Some(Arc::new((slot, names.to_vec())));
        self
    }

    fn with_row_accounting(mut self, accounting: &antecedent::data::RowAccounting) -> Self {
        self.row_accounting_json =
            Some(crate::missing_report::accounting_json(accounting).to_string());
        self
    }

    /// Record the circle-bearing edges of a PAG result, counted from the edges it returns.
    fn with_pag_circle_edges(mut self, count: u64) -> Self {
        self.pag_circle_edges = Some(count);
        self
    }

    fn with_knowledge(mut self, identity: Option<String>) -> Self {
        self.background_knowledge = identity;
        self
    }

    fn with_route(mut self, route: RouteRecord) -> Self {
        self.route = Some(route);
        self
    }
}

/// Graph object for an accepted structure, with the schema's variable names.
pub(crate) fn accepted_graph_object(
    py: Python<'_>,
    accepted: &AcceptedGraph,
    names: &[String],
) -> PyResult<Py<PyAny>> {
    let names = names.to_vec();
    match accepted.graph_ref() {
        GraphRef::Dag(g) => Ok(Py::new(py, graphs::Dag { dag: g.clone(), names })?.into_any()),
        GraphRef::Cpdag(g) => {
            Ok(Py::new(py, graphs::Cpdag { cpdag: g.clone(), names })?.into_any())
        }
        GraphRef::Pag(g) => Ok(Py::new(py, graphs::Pag { pag: g.clone(), names })?.into_any()),
        GraphRef::TemporalDag(g) => {
            Ok(Py::new(py, graphs::TemporalDag { dag: g.clone(), names })?.into_any())
        }
        GraphRef::TemporalCpdag(g) => {
            Ok(Py::new(py, graphs::TemporalCpdag { cpdag: g.clone(), names })?.into_any())
        }
        GraphRef::TemporalPag(g) => {
            Ok(Py::new(py, graphs::TemporalPag { pag: g.clone(), names })?.into_any())
        }
        _ => Err(crate::value_err("accepted structure has no holdable graph object")),
    }
}

#[pymethods]
impl PcmciDiscoveryResult {
    /// Row accounting of a run that reads table rows (NOTEARS) as a mapping, else `None`:
    /// `rows_total`, `rows_kept`, `rows_dropped`, `missing_by_variable`, `reasons`,
    /// `dropped_rows` and `dropped_digest`.
    #[getter]
    fn row_accounting(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        self.row_accounting_json
            .as_deref()
            .map(|json| Ok(py.import("json")?.call_method1("loads", (json,))?.unbind()))
            .transpose()
    }

    /// The compiled discovery route this result came from.
    #[getter]
    fn route(&self, py: Python<'_>) -> PyResult<crate::compiled_route_api::RouteHandle> {
        crate::compiled_route_api::route_object(py, self.route.as_ref())
    }

    /// The contract the discovery executed under: `query_kind="Discover"`,
    /// `structure_source="discovered"` and the inference mode the method declares.
    #[getter]
    fn contract(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.contract()
    }

    /// The claim sealed over the discovery execution.
    #[getter]
    fn claim(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.claim()
    }

    /// Cheap inspection record of the compiled discovery study.
    fn inspect(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.inspect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PcmciDiscoveryResult(algorithm={:?}, links={}, ci_tests={})",
            self.algorithm_id,
            self.links.len(),
            self.ci_tests
        )
    }

    /// Accept this discovery through the Rust review gate and return the structure.
    ///
    /// `accept_discovered=True` auto-accepts what the review allows (pending
    /// directed edges; class marks stay class information). `False` raises
    /// `ReviewRequired` while anything is pending.
    #[pyo3(signature = (*, accept_discovered=true))]
    fn accepted_graph(&self, py: Python<'_>, accept_discovered: bool) -> PyResult<Py<PyAny>> {
        let (slot, names) = self
            .review
            .as_deref()
            .ok_or_else(|| crate::value_err("this discovery result retained no review"))?;
        let accepted = match slot {
            ReviewSlot::Dag(r) => crate::accept_dag_review(r.clone(), accept_discovered),
            ReviewSlot::Cpdag(r) => crate::accept_cpdag_review(r.clone(), accept_discovered),
            ReviewSlot::Pag(g, r) => {
                crate::accept_pag_review(g.clone(), r.clone(), accept_discovered)
            }
            ReviewSlot::TemporalGraph(r) => {
                crate::accept_temporal_graph_review(r.clone(), accept_discovered)
            }
            ReviewSlot::TemporalCpdag(r) => {
                crate::accept_temporal_cpdag_review(r.clone(), accept_discovered)
            }
            ReviewSlot::TemporalPag(g, r) => {
                crate::accept_temporal_pag_review(g.clone(), r.clone(), accept_discovered)
            }
        }
        .map_err(py_err)?;
        accepted_graph_object(py, &accepted, names)
    }
}

pub(crate) fn series_from_batch(
    batch: &crate::ColumnBatch,
) -> PyResult<(TimeSeriesData, Vec<VariableId>)> {
    let loaded = batch.loaded()?;
    let series = series_from_tabular(loaded.data)?;
    let variables: Vec<VariableId> = series.schema().variables().iter().map(|v| v.id).collect();
    Ok((series, variables))
}

/// Refuse unit lengths that are empty, contain a zero, or do not add up to the `rows` of the
/// pooled columns.
///
/// # Errors
///
/// A typed `invalid_argument` refusal.
pub(crate) fn check_unit_lengths(rows: usize, unit_lengths: &[usize]) -> PyResult<()> {
    if unit_lengths.is_empty()
        || unit_lengths.contains(&0)
        || unit_lengths.iter().try_fold(0usize, |total, length| total.checked_add(*length))
            != Some(rows)
    {
        return Err(crate::refusal(
            antecedent::reason::reason_code!("invalid_argument"),
            format!(
                "unit_lengths must be positive and add up to the {rows} rows of the pooled \
                 columns; got {unit_lengths:?}"
            ),
        ));
    }
    Ok(())
}

/// The per-unit series of a panel ingested as one batch per unit window.
///
/// Every unit becomes its own series, so lagged discovery builds each unit's lag windows inside
/// that unit and never takes a lagged parent across a unit boundary. The unit batches were
/// copied straight from the pooled NumPy columns ([`crate::columns_to_ingested`]), so no unit
/// is copied a second time here.
pub(crate) fn panel_units_from_unit_batches(
    units: &[crate::ColumnBatch],
) -> PyResult<(Vec<TimeSeriesData>, Vec<VariableId>)> {
    let mut series_units = Vec::with_capacity(units.len());
    let mut variables = Vec::new();
    for unit in units {
        let (series, ids) = series_from_batch(unit)?;
        variables = ids;
        series_units.push(series);
    }
    Ok((series_units, variables))
}

/// A series' or a panel's ingested columns as a tabular table, for the static searches.
///
/// # Errors
///
/// A panel (`unit_lengths`) applies to the temporal searches only.
pub(crate) fn tabular_from_ingested(
    ingested: crate::IngestedColumns,
) -> PyResult<(antecedent::prelude::TabularData, Vec<VariableId>)> {
    match ingested {
        crate::IngestedColumns::Series(batch) => tabular_from_batch(&batch),
        crate::IngestedColumns::Panel(_) => {
            Err(crate::value_err("unit_lengths applies to the temporal searches only"))
        }
    }
}

/// Read the `background_knowledge` argument of a direct entry point: the wire JSON of
/// [`crate::selective_discovery_api::background_knowledge`], or nothing.
pub(crate) fn knowledge_arg(text: Option<&str>) -> PyResult<BackgroundKnowledge> {
    let Some(text) = text else { return Ok(BackgroundKnowledge::default()) };
    let parsed: serde_json::Value =
        serde_json::from_str(text).map_err(|e| value_err(format!("background_knowledge: {e}")))?;
    let map = parsed
        .as_object()
        .ok_or_else(|| value_err("background_knowledge must be a JSON object"))?;
    crate::selective_discovery_api::background_knowledge(map)
}

/// The hex identity recorded on a result for the knowledge a run was given.
fn knowledge_identity(knowledge: &BackgroundKnowledge) -> Option<String> {
    (!knowledge.is_empty()).then(|| antecedent::identity::to_hex(&knowledge.identity()))
}

/// A facade discovery failure as a Python exception. Background knowledge the search cannot
/// honour (a contradiction, an unsearched variable, a parent limit the algorithm cannot
/// enforce) is a typed refusal with its registered reason code, never a generic failure.
fn discovery_err(error: antecedent::CausalError) -> PyErr {
    match error {
        antecedent::CausalError::Discovery(antecedent::discovery::DiscoveryError::Knowledge(
            fault,
        )) => crate::refusal(fault.reason_code(), fault.message()),
        other => py_err(other),
    }
}

/// Run `spec` over `input` through the compiled discovery route and take the engine
/// result `pick` reads from the structure it returned.
pub(crate) fn compiled_search<T>(
    spec: DiscoverySpec,
    input: DiscoverInput,
    variables: &[VariableId],
    ctx: &antecedent::prelude::ExecutionContext,
    pick: impl FnOnce(DiscoveredStructure) -> Option<T>,
) -> PyResult<(T, RouteRecord)> {
    let (structure, route) = crate::compiled_route_api::run_discovery(spec, input, variables, ctx)
        .map_err(discovery_err)?;
    let result = pick(structure).ok_or_else(|| {
        crate::value_err("the compiled search returned a structure class its method does not state")
    })?;
    Ok((result, route))
}

/// The series input of a temporal search: one series, or the units of a panel.
fn series_input(units: &[TimeSeriesData], panel: bool) -> DiscoverInput {
    if panel {
        DiscoverInput::Panel(units.to_vec())
    } else {
        DiscoverInput::Series(units[0].clone())
    }
}

type SharedCi = std::sync::Arc<dyn antecedent::inference::ConditionalIndependence + Send + Sync>;

/// Shared NumPy→series discovery preamble (CI resolve + GIL release).
///
/// `unit_lengths` marks the columns as the row-concatenated units of a panel; `f` then gets one
/// series per unit and `panel = true`, otherwise exactly one series and `panel = false`.
fn run_series_discovery<F>(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    unit_lengths: Option<Vec<usize>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    f: F,
) -> PyResult<PcmciDiscoveryResult>
where
    F: FnOnce(
            &[TimeSeriesData],
            bool,
            &[VariableId],
            SharedCi,
            String,
            &antecedent::prelude::ExecutionContext,
            &[String],
        ) -> PyResult<PcmciDiscoveryResult>
        + Send,
{
    if unit_lengths.is_some() && weights.is_some() {
        // Observation weights are aligned to the end of one series; pooled units have no such
        // alignment, so a weighted test over a panel would weight the wrong rows.
        return Err(crate::refusal(
            antecedent::reason::reason_code!("option_not_applicable"),
            "observation weights are defined per row of one series and cannot be combined with a \
             panel (unit_lengths)",
        ));
    }
    let ingested = crate::columns_to_ingested(&names, &columns, unit_lengths.as_deref())?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    drop(columns);
    let threads = if is_callback { Some(1) } else { threads };
    detach_catch(py, move || {
        let (units, variables, panel) = match ingested {
            crate::IngestedColumns::Panel(batches) => {
                let (units, variables) = panel_units_from_unit_batches(&batches)?;
                (units, variables, true)
            }
            crate::IngestedColumns::Series(batch) => {
                let (series, variables) = series_from_batch(&batch)?;
                (vec![series], variables, false)
            }
        };
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        f(&units, panel, &variables, ci_impl, ci_name, &ctx, &names)
    })
}

pub(crate) fn tabular_from_batch(
    batch: &crate::ColumnBatch,
) -> PyResult<(antecedent::prelude::TabularData, Vec<VariableId>)> {
    let loaded = batch.loaded()?;
    let tabular = loaded.data;
    let variables: Vec<VariableId> = tabular.schema().variables().iter().map(|v| v.id).collect();
    Ok((tabular, variables))
}

/// Resolve a `LaggedParent` conditioning set to Python-friendly `(variable_name, lag)` pairs.
fn conditioning_set_parts(names: &[String], set: &[LaggedParent]) -> Vec<(String, u32)> {
    set.iter()
        .map(|&(variable, lag)| {
            let name = names
                .get(variable.as_usize())
                .cloned()
                .unwrap_or_else(|| format!("var{}", variable.raw()));
            (name, lag.raw())
        })
        .collect()
}

fn discovered_links(
    names: &[String],
    links: &[ScoredLink],
    sepsets: &PcSepsets,
) -> Vec<DiscoveredLink> {
    links
        .iter()
        .map(|s| {
            let key = (s.link.source, s.link.source_lag, s.link.target, s.link.target_lag);
            let conditioning_set =
                sepsets.get(&key).map(|set| conditioning_set_parts(names, set)).unwrap_or_default();
            DiscoveredLink {
                source: names
                    .get(s.link.source.as_usize())
                    .cloned()
                    .unwrap_or_else(|| format!("var{}", s.link.source.raw())),
                source_lag: s.link.source_lag.raw(),
                target: names
                    .get(s.link.target.as_usize())
                    .cloned()
                    .unwrap_or_else(|| format!("var{}", s.link.target.raw())),
                target_lag: s.link.target_lag.raw(),
                statistic: s.statistic,
                p_value: s.p_value,
                adjusted_p_value: s.adjusted_p_value,
                conditioning_set,
            }
        })
        .collect()
}

fn discovery_result_fields(
    names: &[String],
    links: &[ScoredLink],
    sepsets: &PcSepsets,
    algorithm_id: &str,
    algorithm_config: &str,
    performance: &DiscoveryPerformanceRecord,
    pending_edge_count: u64,
    ci_name: String,
    cpdag_nodes: u64,
    cpdag_directed_edges: u64,
    cpdag_undirected_edges: u64,
    graph_edges: Vec<GraphEdge>,
) -> PcmciDiscoveryResult {
    PcmciDiscoveryResult {
        links: discovered_links(names, links, sepsets),
        algorithm_id: algorithm_id.to_string(),
        algorithm_config: algorithm_config.to_string(),
        ci_tests: performance.ci_tests,
        links_retained: performance.links_retained,
        pending_edge_count,
        lagged_frame_bytes: performance.lagged_frame_bytes,
        worker_threads: performance.worker_threads,
        ci_name,
        cpdag_nodes,
        cpdag_directed_edges,
        cpdag_undirected_edges,
        pag_circle_edges: None,
        graph_edges,
        variable_names: names.to_vec(),
        background_knowledge: None,
        review: None,
        row_accounting_json: None,
        route: None,
    }
}

/// Pack a static CPDAG discovery result into the shared Python DTO.
fn pcmci_result_from_static_cpdag(
    names: &[String],
    links: &[ScoredLink],
    sepsets: &PcSepsets,
    algorithm_id: &str,
    algorithm_config: &str,
    performance: &DiscoveryPerformanceRecord,
    pending_edges: usize,
    pending_undirected: usize,
    cpdag: &Cpdag,
    ci_name: String,
) -> PcmciDiscoveryResult {
    let directed = cpdag.directed_edge_count() as u64;
    let undirected = cpdag.undirected_edge_count() as u64;
    let pending = pending_edges as u64 + pending_undirected as u64;
    let graph_edges = static_cpdag_graph_edges(names, cpdag);
    discovery_result_fields(
        names,
        links,
        sepsets,
        algorithm_id,
        algorithm_config,
        performance,
        pending,
        ci_name,
        cpdag.node_count() as u64,
        directed,
        undirected,
        graph_edges,
    )
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    temporal_routes::register_first(m)?;
    static_routes::register(m)?;
    temporal_routes::register_remaining(m)?;
    crate::structure_stability_api::register(m)?;
    Ok(())
}
