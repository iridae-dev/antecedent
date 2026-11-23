//! Static graph discovery routes.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Run static PC discovery over tabular (non-temporal) columns.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=None, cancel=None, background_knowledge=None))]
fn discover_pc(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    background_knowledge: Option<String>,
) -> PyResult<PcmciDiscoveryResult> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let knowledge = knowledge_arg(background_knowledge.as_deref())?;
    let identity = knowledge_identity(&knowledge);
    let batch = numeric_columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { Some(1) } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc: false,
            max_subset: None,
            knowledge,
        };
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let (result, route) = compiled_search(
            DiscoverySpec::pc(params, ci_name.clone()),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::StaticCpdag(result) => Some(result),
                _ => None,
            },
        )?;
        let slot = ReviewSlot::Cpdag(result.review.clone());
        Ok(pcmci_result_from_static_cpdag(
            &names,
            &result.evidence.links,
            &result.sepsets,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            result.review.pending_edges.len(),
            result.review.pending_undirected.len(),
            &result.evidence.graph,
            ci_name,
        )
        .with_knowledge(identity)
        .with_review(slot, &names)
        .with_route(route))
    })
}

/// Run GES discovery over tabular columns → CPDAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=None, cancel=None, screen_pc=false, max_subset=None, background_knowledge=None))]
fn discover_ges(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    screen_pc: bool,
    max_subset: Option<usize>,
    background_knowledge: Option<String>,
) -> PyResult<PcmciDiscoveryResult> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let knowledge = knowledge_arg(background_knowledge.as_deref())?;
    let identity = knowledge_identity(&knowledge);
    let batch = numeric_columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { Some(1) } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc,
            max_subset,
            knowledge,
        };
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let (result, route) = compiled_search(
            DiscoverySpec::ges(params, ci_name.clone()),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::StaticCpdag(result) => Some(result),
                _ => None,
            },
        )?;
        let slot = ReviewSlot::Cpdag(result.review.clone());
        Ok(pcmci_result_from_static_cpdag(
            &names,
            &result.evidence.links,
            &result.sepsets,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            result.review.pending_edges.len(),
            result.review.pending_undirected.len(),
            &result.evidence.graph,
            ci_name,
        )
        .with_knowledge(identity)
        .with_review(slot, &names)
        .with_route(route))
    })
}

/// Run DirectLiNGAM discovery over tabular columns → DAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, prune_threshold=0.05, seed=1, max_cond_size=8, threads=None, cancel=None))]
fn discover_lingam(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    prune_threshold: f64,
    seed: u64,
    max_cond_size: usize,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PcmciDiscoveryResult> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let batch = numeric_columns_to_batch(&names, &columns)?;
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
            knowledge: BackgroundKnowledge::default(),
        };
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let (result, route) = compiled_search(
            DiscoverySpec::lingam(params, prune_threshold, "parcorr"),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::StaticDag(result) => Some(result),
                _ => None,
            },
        )?;

        let dag = &result.evidence.graph;
        let directed = dag.edges().count() as u64;
        let pending = result.review.pending_edges.len() as u64;
        let graph_edges = static_dag_graph_edges(&names, dag);

        let slot = ReviewSlot::Dag(result.review.clone());
        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            &result.sepsets,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            "direct_lingam".into(),
            dag.node_count() as u64,
            directed,
            0,
            graph_edges,
        )
        .with_review(slot, &names)
        .with_route(route))
    })
}

/// Run NOTEARS discovery over tabular columns → DAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, l1=0.1, threshold=0.3, standardize=true, seed=1, max_cond_size=8, threads=None, cancel=None, missing=None))]
fn discover_notears(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    l1: f64,
    threshold: f64,
    standardize: bool,
    seed: u64,
    max_cond_size: usize,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    missing: Option<String>,
) -> PyResult<PcmciDiscoveryResult> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let missing = missing.as_deref().map(crate::missing_report::parse_policy).transpose()?;
    let batch = numeric_columns_to_batch(&names, &columns)?;
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
            knowledge: BackgroundKnowledge::default(),
        };
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let (result, route) = compiled_search(
            DiscoverySpec::notears(params, l1, threshold, standardize, missing, "parcorr"),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::Notears(result) => Some(result),
                _ => None,
            },
        )?;

        let dag = &result.discovery.evidence.graph;
        let directed = dag.edges().count() as u64;
        let pending = result.discovery.review.pending_edges.len() as u64;
        let graph_edges = static_dag_graph_edges(&names, dag);

        Ok(discovery_result_fields(
            &names,
            &result.discovery.evidence.links,
            &result.discovery.sepsets,
            result.discovery.algorithm.id.as_ref(),
            result.discovery.algorithm.config.as_ref(),
            &result.discovery.performance,
            pending,
            "notears".into(),
            dag.node_count() as u64,
            directed,
            0,
            graph_edges,
        )
        .with_review(ReviewSlot::Dag(result.discovery.review.clone()), &names)
        .with_row_accounting(&result.row_accounting)
        .with_route(route))
    })
}

/// Run classic static FCI discovery over tabular columns → PAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=None, cancel=None, background_knowledge=None))]
fn discover_fci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    background_knowledge: Option<String>,
) -> PyResult<PcmciDiscoveryResult> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let knowledge = knowledge_arg(background_knowledge.as_deref())?;
    let identity = knowledge_identity(&knowledge);
    let batch = numeric_columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { Some(1) } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc: false,
            max_subset: None,
            knowledge,
        };
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let (result, route) = compiled_search(
            DiscoverySpec::fci(params, ci_name.clone()),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::StaticPag(result) => Some(result),
                _ => None,
            },
        )?;

        let pag = &result.evidence.graph;
        let pending = result.review.pending_circles.len() as u64;
        let directed = static_pag_definite_directed_count(pag);
        let graph_edges = static_pag_graph_edges(&names, pag)?;
        let circle_edges = circle_edge_count(&graph_edges);

        let slot = ReviewSlot::Pag(result.evidence.graph.clone(), result.review.clone());
        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            &result.sepsets,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            pag.node_count() as u64,
            directed,
            0,
            graph_edges,
        )
        .with_pag_circle_edges(circle_edges)
        .with_knowledge(identity)
        .with_review(slot, &names)
        .with_route(route))
    })
}

/// Run classic static RFCI discovery over tabular columns → PAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=None, cancel=None, background_knowledge=None))]
fn discover_rfci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    background_knowledge: Option<String>,
) -> PyResult<PcmciDiscoveryResult> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let knowledge = knowledge_arg(background_knowledge.as_deref())?;
    let identity = knowledge_identity(&knowledge);
    let batch = numeric_columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { Some(1) } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc: false,
            max_subset: None,
            knowledge,
        };
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let (result, route) = compiled_search(
            DiscoverySpec::rfci(params, ci_name.clone()),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::StaticPag(result) => Some(result),
                _ => None,
            },
        )?;

        let pag = &result.evidence.graph;
        let pending = result.review.pending_circles.len() as u64;
        let directed = static_pag_definite_directed_count(pag);
        let graph_edges = static_pag_graph_edges(&names, pag)?;
        let circle_edges = circle_edge_count(&graph_edges);

        let slot = ReviewSlot::Pag(result.evidence.graph.clone(), result.review.clone());
        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            &result.sepsets,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            pag.node_count() as u64,
            directed,
            0,
            graph_edges,
        )
        .with_pag_circle_edges(circle_edges)
        .with_knowledge(identity)
        .with_review(slot, &names)
        .with_route(route))
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(discover_pc, m)?)?;
    m.add_function(wrap_pyfunction!(discover_ges, m)?)?;
    m.add_function(wrap_pyfunction!(discover_lingam, m)?)?;
    m.add_function(wrap_pyfunction!(discover_notears, m)?)?;
    m.add_function(wrap_pyfunction!(discover_fci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_rfci, m)?)?;
    Ok(())
}
