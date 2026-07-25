//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

use crate::*;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
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

fn pag_graph_edges(names: &[String], pag: &TemporalPag) -> Vec<GraphEdge> {
    let nodes = pag.nodes();
    let mut out = Vec::new();
    for i in 0..pag.node_count() {
        let a = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
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
    out
}

fn static_pag_graph_edges(names: &[String], pag: &Pag) -> Vec<GraphEdge> {
    let nodes = pag.nodes();
    let mut out = Vec::new();
    for i in 0..pag.node_count() {
        let a = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
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
    out
}

fn static_pag_definite_directed_count(pag: &Pag) -> u64 {
    let mut directed = 0u64;
    for i in 0..pag.node_count() {
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
    #[pyo3(get)]
    pub(crate) cpdag_undirected_edges: u64,
    /// Oriented graph body (CPDAG/PAG marks); empty for lagged-only PCMCI.
    #[pyo3(get)]
    pub(crate) graph_edges: Vec<GraphEdge>,
}

pub(crate) fn series_from_batch(
    batch: &RecordBatch,
) -> PyResult<(TimeSeriesData, Vec<VariableId>)> {
    let loaded = tabular_from_record_batch(batch).map_err(py_err)?;
    let tabular = loaded.data;
    let n = tabular.row_count();
    let series = TimeSeriesData::try_new(
        tabular.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .map_err(py_err)?;
    let variables: Vec<VariableId> = series.schema().variables().iter().map(|v| v.id).collect();
    Ok((series, variables))
}

pub(crate) fn tabular_from_batch(
    batch: &RecordBatch,
) -> PyResult<(antecedent_data::TabularData, Vec<VariableId>)> {
    let loaded = tabular_from_record_batch(batch).map_err(py_err)?;
    let tabular = loaded.data;
    let variables: Vec<VariableId> = tabular.schema().variables().iter().map(|v| v.id).collect();
    Ok((tabular, variables))
}

fn discovered_links(names: &[String], links: &[ScoredLink]) -> Vec<DiscoveredLink> {
    links
        .iter()
        .map(|s| DiscoveredLink {
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
        })
        .collect()
}

fn discovery_result_fields(
    names: &[String],
    links: &[ScoredLink],
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
        links: discovered_links(names, links),
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
        graph_edges,
    }
}

/// Pack a static CPDAG discovery result into the shared Python DTO.
fn pcmci_result_from_static_cpdag(
    names: &[String],
    links: &[ScoredLink],
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

/// Run lagged PCMCI discovery.
///
/// NumPy columns in, structured link list out once. Batch CI only (no per-query Python loop
/// unless `ci` is an explicit slow-path callable — ).
/// `ci` selects a named test (default `parcorr`) or a Python batch callable.
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=1))]
fn discover_pcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints::default(),
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_pcmci(&series, &variables, &params, &ctx).map_err(py_err)?;
        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            result.review.pending_edges.len() as u64,
            ci_name,
            0,
            0,
            0,
            Vec::new(),
        ))
    })
}

/// Run static PC discovery over tabular (non-temporal) columns.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=1))]
fn discover_pc(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc: false,
            max_subset: None,
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_pc(&data, &variables, &params, &ctx).map_err(py_err)?;
        Ok(pcmci_result_from_static_cpdag(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            result.review.pending_edges.len(),
            result.review.pending_undirected.len(),
            &result.evidence.graph,
            ci_name,
        ))
    })
}

/// Run GES discovery over tabular columns → CPDAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=1, screen_pc=false, max_subset=None))]
fn discover_ges(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: u32,
    screen_pc: bool,
    max_subset: Option<usize>,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc,
            max_subset,
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_ges(&data, &variables, &params, &ctx).map_err(py_err)?;
        Ok(pcmci_result_from_static_cpdag(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            result.review.pending_edges.len(),
            result.review.pending_undirected.len(),
            &result.evidence.graph,
            ci_name,
        ))
    })
}

/// Run DirectLiNGAM discovery over tabular columns → DAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, prune_threshold=0.05, seed=1, max_cond_size=8, threads=1))]
fn discover_lingam(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    prune_threshold: f64,
    seed: u64,
    max_cond_size: usize,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha: 0.05,
            max_cond_size,
            fdr: None,
            ci: Arc::new(PartialCorrelation),
            screen_pc: false,
            max_subset: None,
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_lingam(&data, &variables, &params, prune_threshold, &ctx)
            .map_err(py_err)?;

        let dag = &result.evidence.graph;
        let directed = dag.edges().count() as u64;
        let pending = result.review.pending_edges.len() as u64;
        let graph_edges = static_dag_graph_edges(&names, dag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            "direct_lingam".into(),
            dag.node_count() as u64,
            directed,
            0,
            graph_edges,
        ))
    })
}

/// Run NOTEARS discovery over tabular columns → DAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, l1=0.1, threshold=0.3, standardize=true, seed=1, max_cond_size=8, threads=1))]
fn discover_notears(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    l1: f64,
    threshold: f64,
    standardize: bool,
    seed: u64,
    max_cond_size: usize,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha: 0.05,
            max_cond_size,
            fdr: None,
            ci: Arc::new(PartialCorrelation),
            screen_pc: false,
            max_subset: None,
        };
        let ctx = py_execution_context(seed, threads);
        let result =
            facade_discover_notears(&data, &variables, &params, l1, threshold, standardize, &ctx)
                .map_err(py_err)?;

        let dag = &result.discovery.evidence.graph;
        let directed = dag.edges().count() as u64;
        let pending = result.discovery.review.pending_edges.len() as u64;
        let graph_edges = static_dag_graph_edges(&names, dag);

        Ok(discovery_result_fields(
            &names,
            &result.discovery.evidence.links,
            result.discovery.algorithm.id.as_ref(),
            result.discovery.algorithm.config.as_ref(),
            &result.discovery.performance,
            pending,
            "notears".into(),
            dag.node_count() as u64,
            directed,
            0,
            graph_edges,
        ))
    })
}

/// Run classic static FCI discovery over tabular columns → PAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=1))]
fn discover_fci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc: false,
            max_subset: None,
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_fci(&data, &variables, &params, &ctx).map_err(py_err)?;

        let pag = &result.evidence.graph;
        let pending = result.review.pending_circles.len() as u64;
        let directed = static_pag_definite_directed_count(pag);
        let graph_edges = static_pag_graph_edges(&names, pag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            pag.node_count() as u64,
            directed,
            pending,
            graph_edges,
        ))
    })
}

/// Run classic static RFCI discovery over tabular columns → PAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=1))]
fn discover_rfci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc: false,
            max_subset: None,
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_rfci(&data, &variables, &params, &ctx).map_err(py_err)?;

        let pag = &result.evidence.graph;
        let pending = result.review.pending_circles.len() as u64;
        let directed = static_pag_definite_directed_count(pag);
        let graph_edges = static_pag_graph_edges(&names, pag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            pag.node_count() as u64,
            directed,
            pending,
            graph_edges,
        ))
    })
}

/// Run PCMCI+ discovery returning links plus oriented temporal CPDAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=1))]
fn discover_pcmci_plus(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints::default(),
        };
        let ctx = py_execution_context(seed, threads);
        let result =
            facade_discover_pcmci_plus(&series, &variables, &params, &ctx).map_err(py_err)?;

        let cpdag = &result.evidence.graph;
        let directed = cpdag.directed_edge_count() as u64;
        let undirected = cpdag.undirected_edge_count() as u64;
        let pending = result.review.pending_edges.len() as u64
            + result.review.pending_undirected.len() as u64;
        let graph_edges = cpdag_graph_edges(&names, cpdag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            cpdag.node_count() as u64,
            directed,
            undirected,
            graph_edges,
        ))
    })
}

/// Run LPCMCI discovery returning links plus temporal PAG summary (no per-edge GIL).
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=1))]
fn discover_lpcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints::default(),
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_lpcmci(&series, &variables, &params, &ctx).map_err(py_err)?;

        let pag = &result.evidence.graph;
        let pending = result.review.pending_circles.len() as u64;
        let directed = pag_definite_directed_edge_count(pag);
        let graph_edges = pag_graph_edges(&names, pag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            pag.node_count() as u64,
            directed,
            pending, // undirected field reused as circle-pending count
            graph_edges,
        ))
    })
}

/// J-PCMCI+ over multiple environments (one GIL crossing).
///
/// `env_columns` is a list of column batches (each env: same `names` order).
/// Optional `context_names` lists observed context columns (must appear in `names`);
/// remaining names are treated as system variables.
#[pyfunction]
#[pyo3(signature = (
    names,
    env_columns,
    *,
    max_lag=1,
    alpha=0.05,
    fdr=true,
    seed=1,
    ci=None,
    weights=None,
    threads=1,
    context_names=None,
    include_space_dummy=true,
    include_time_dummy=false,
    space_dummy_ci="scalar",
    time_dummy_encoding="integer",
    time_dummy_ci="scalar",
))]
fn discover_jpcmci_plus(
    py: Python<'_>,
    names: Vec<String>,
    env_columns: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: u32,
    context_names: Option<Vec<String>>,
    include_space_dummy: bool,
    include_time_dummy: bool,
    space_dummy_ci: &str,
    time_dummy_encoding: &str,
    time_dummy_ci: &str,
) -> PyResult<PcmciDiscoveryResult> {
    if env_columns.is_empty() {
        return Err(PyValueError::new_err("discover_jpcmci_plus needs ≥1 environment"));
    }
    let mut batches = Vec::with_capacity(env_columns.len());
    for cols in &env_columns {
        batches.push(columns_to_batch(&names, cols)?);
    }
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    let context_names = context_names.unwrap_or_default();
    let threads = if is_callback { 1 } else { threads };
    drop(env_columns);

    detach_catch(py, move || {
        let mut series_list = Vec::with_capacity(batches.len());
        let mut all_variables = Vec::new();
        for (i, batch) in batches.iter().enumerate() {
            let (series, vars) = series_from_batch(batch)?;
            if i == 0 {
                all_variables = vars;
            }
            series_list.push(series);
        }
        let multi = MultiEnvironmentData::try_new(Arc::from(series_list)).map_err(py_err)?;

        let mut context_ids = Vec::new();
        for cname in &context_names {
            let Some(idx) = names.iter().position(|n| n == cname) else {
                return Err(PyValueError::new_err(format!(
                    "context_names entry '{cname}' not found in names"
                )));
            };
            context_ids.push(all_variables[idx]);
        }
        let system: Vec<VariableId> =
            all_variables.iter().copied().filter(|v| !context_ids.contains(v)).collect();
        if system.is_empty() {
            return Err(PyValueError::new_err(
                "discover_jpcmci_plus needs ≥1 system variable after excluding context_names",
            ));
        }

        let space_dummy_ci = match space_dummy_ci {
            "scalar" | "scalar_one_hot" | "one_hot" => SpaceDummyCiMode::ScalarOneHot,
            "multivariate" | "multivariate_block" | "block" => SpaceDummyCiMode::MultivariateBlock,
            other => {
                return Err(PyValueError::new_err(format!(
                    "space_dummy_ci must be 'scalar' or 'multivariate', got '{other}'"
                )));
            }
        };
        let time_dummy_encoding = match time_dummy_encoding {
            "integer" | "integer_index" | "index" => TimeDummyEncoding::IntegerIndex,
            "one_hot" | "onehot" | "oh" => TimeDummyEncoding::OneHot,
            other => {
                return Err(PyValueError::new_err(format!(
                    "time_dummy_encoding must be 'integer' or 'one_hot', got '{other}'"
                )));
            }
        };
        let time_dummy_ci = match time_dummy_ci {
            "scalar" | "scalar_one_hot" | "one_hot" => TimeDummyCiMode::ScalarOneHot,
            "multivariate" | "multivariate_block" | "block" => TimeDummyCiMode::MultivariateBlock,
            other => {
                return Err(PyValueError::new_err(format!(
                    "time_dummy_ci must be 'scalar' or 'multivariate', got '{other}'"
                )));
            }
        };
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints {
                context_variables: Arc::from(context_ids),
                include_space_dummy,
                include_time_dummy,
                space_dummy_ci,
                time_dummy_encoding,
                time_dummy_ci,
                ..MultiDatasetConstraints::default()
            },
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_jpcmci_plus(&multi, &system, &params, &ctx).map_err(py_err)?;
        let cpdag = &result.evidence.graph;
        let graph_edges = cpdag_graph_edges(&names, cpdag);
        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            result.review.pending_undirected.len() as u64,
            ci_name,
            cpdag.node_count() as u64,
            cpdag.directed_edge_count() as u64,
            cpdag.undirected_edge_count() as u64,
            graph_edges,
        ))
    })
}

/// Two-regime half-split assignment (opt-in helper for RPCMCI; not applied by default).
#[pyfunction(name = "two_regime_half_split")]
fn two_regime_half_split_py(series_len: usize) -> Vec<u32> {
    two_regime_half_split(series_len).regimes.iter().map(|r| r.raw()).collect()
}

/// RPCMCI with caller-supplied regimes (required; no silent half-split).
#[pyfunction]
#[pyo3(signature = (names, columns, *, regimes, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=1))]
fn discover_rpcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    regimes: Vec<u32>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: u32,
) -> PyResult<RpcmciDiscoverySummary> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, _ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };
    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        if regimes.len() != series.row_count() {
            return Err(PyValueError::new_err(format!(
                "regimes length {} != series length {}",
                regimes.len(),
                series.row_count()
            )));
        }
        let assign = RegimeAssignment::try_new(
            regimes.into_iter().map(RegimeId::from_raw).collect::<Vec<_>>(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints::default(),
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_rpcmci(&series, &variables, &assign, &params, None, &ctx)
            .map_err(py_err)?;
        let mut regime_ids = Vec::new();
        let mut directed = Vec::new();
        let mut undirected = Vec::new();
        for (rid, g) in result.graphs.graphs.iter() {
            regime_ids.push(rid.raw());
            directed.push(g.directed_edge_count() as u64);
            undirected.push(g.undirected_edge_count() as u64);
        }
        Ok(RpcmciDiscoverySummary {
            algorithm: result.algorithm.id.to_string(),
            n_regimes: regime_ids.len() as u64,
            regime_ids,
            directed_edges: directed,
            undirected_edges: undirected,
        })
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(discover_pcmci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pcmci_plus, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pc, m)?)?;
    m.add_function(wrap_pyfunction!(discover_ges, m)?)?;
    m.add_function(wrap_pyfunction!(discover_lingam, m)?)?;
    m.add_function(wrap_pyfunction!(discover_notears, m)?)?;
    m.add_function(wrap_pyfunction!(discover_fci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_rfci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_lpcmci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_jpcmci_plus, m)?)?;
    m.add_function(wrap_pyfunction!(discover_rpcmci, m)?)?;
    m.add_function(wrap_pyfunction!(two_regime_half_split_py, m)?)?;
    Ok(())
}
