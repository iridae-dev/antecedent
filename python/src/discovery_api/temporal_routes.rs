//! Temporal graph discovery routes.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Run lagged PCMCI discovery.
///
/// NumPy columns in, structured link list out once. Batch CI only (no per-query Python loop
/// unless `ci` is an explicit slow-path callable — ).
/// `ci` selects a named test (default `parcorr`) or a Python batch callable.
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=None, cancel=None, max_cond_size=2, unit_lengths=None, background_knowledge=None))]
fn discover_pcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    max_cond_size: usize,
    unit_lengths: Option<Vec<usize>>,
    background_knowledge: Option<String>,
) -> PyResult<PcmciDiscoveryResult> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let knowledge = knowledge_arg(background_knowledge.as_deref())?;
    let identity = knowledge_identity(&knowledge);
    run_series_discovery(
        py,
        names,
        columns,
        seed,
        ci,
        weights,
        unit_lengths,
        threads,
        cancel,
        |units, panel, variables, ci_impl, ci_name, ctx, names| {
            let params = DiscoverParams {
                max_lag,
                alpha,
                fdr: fdr.then(FdrAdjustment::bh),
                ci: ci_impl,
                multi_dataset: MultiDatasetConstraints::default(),
                max_cond_size,
                knowledge,
            };
            let (result, route) = compiled_search(
                DiscoverySpec::pcmci(params, ci_name.clone()),
                series_input(units, panel),
                variables,
                ctx,
                |structure| match structure {
                    DiscoveredStructure::TemporalDag(result) => Some(result),
                    _ => None,
                },
            )?;
            let slot = ReviewSlot::TemporalGraph(result.review.clone());
            Ok(discovery_result_fields(
                names,
                &result.evidence.links,
                &result.sepsets,
                result.algorithm.id.as_ref(),
                result.algorithm.config.as_ref(),
                &result.performance,
                result.review.pending_edges.len() as u64,
                ci_name,
                0,
                0,
                0,
                Vec::new(),
            )
            .with_knowledge(identity)
            .with_review(slot, names)
            .with_route(route))
        },
    )
}

/// Run PCMCI+ discovery returning links plus oriented temporal CPDAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=None, cancel=None, max_cond_size=2, unit_lengths=None, background_knowledge=None))]
fn discover_pcmci_plus(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    max_cond_size: usize,
    unit_lengths: Option<Vec<usize>>,
    background_knowledge: Option<String>,
) -> PyResult<PcmciDiscoveryResult> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let knowledge = knowledge_arg(background_knowledge.as_deref())?;
    let identity = knowledge_identity(&knowledge);
    run_series_discovery(
        py,
        names,
        columns,
        seed,
        ci,
        weights,
        unit_lengths,
        threads,
        cancel,
        |units, panel, variables, ci_impl, ci_name, ctx, names| {
            let params = DiscoverParams {
                max_lag,
                alpha,
                fdr: fdr.then(FdrAdjustment::bh),
                ci: ci_impl,
                multi_dataset: MultiDatasetConstraints::default(),
                max_cond_size,
                knowledge,
            };
            let (result, route) = compiled_search(
                DiscoverySpec::pcmci_plus(params, ci_name.clone()),
                series_input(units, panel),
                variables,
                ctx,
                |structure| match structure {
                    DiscoveredStructure::TemporalCpdag(result) => Some(result),
                    _ => None,
                },
            )?;
            let cpdag = &result.evidence.graph;
            let directed = cpdag.directed_edge_count() as u64;
            let undirected = cpdag.undirected_edge_count() as u64;
            let pending = result.review.pending_edges.len() as u64
                + result.review.pending_undirected.len() as u64;
            let graph_edges = cpdag_graph_edges(names, cpdag);
            Ok(discovery_result_fields(
                names,
                &result.evidence.links,
                &result.sepsets,
                result.algorithm.id.as_ref(),
                result.algorithm.config.as_ref(),
                &result.performance,
                pending,
                ci_name,
                cpdag.node_count() as u64,
                directed,
                undirected,
                graph_edges,
            )
            .with_knowledge(identity)
            .with_review(ReviewSlot::TemporalCpdag(result.review.clone()), names)
            .with_route(route))
        },
    )
}

/// Run LPCMCI discovery returning links plus temporal PAG summary (no per-edge GIL).
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=None, cancel=None, max_cond_size=2, unit_lengths=None))]
fn discover_lpcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    max_cond_size: usize,
    unit_lengths: Option<Vec<usize>>,
) -> PyResult<PcmciDiscoveryResult> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    run_series_discovery(
        py,
        names,
        columns,
        seed,
        ci,
        weights,
        unit_lengths,
        threads,
        cancel,
        |units, panel, variables, ci_impl, ci_name, ctx, names| {
            let params = DiscoverParams {
                max_lag,
                alpha,
                fdr: fdr.then(FdrAdjustment::bh),
                ci: ci_impl,
                multi_dataset: MultiDatasetConstraints::default(),
                max_cond_size,
                knowledge: BackgroundKnowledge::default(),
            };
            let (result, route) = compiled_search(
                DiscoverySpec::lpcmci(params, ci_name.clone()),
                series_input(units, panel),
                variables,
                ctx,
                |structure| match structure {
                    DiscoveredStructure::TemporalPag(result) => Some(result),
                    _ => None,
                },
            )?;
            let pag = &result.evidence.graph;
            let pending = result.review.pending_circles.len() as u64;
            let directed = pag_definite_directed_edge_count(pag);
            let graph_edges = pag_graph_edges(names, pag)?;
            let circle_edges = circle_edge_count(&graph_edges);
            let slot =
                ReviewSlot::TemporalPag(result.evidence.graph.clone(), result.review.clone());
            Ok(discovery_result_fields(
                names,
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
            .with_review(slot, names)
            .with_route(route))
        },
    )
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
    threads=None,
    cancel=None,
    context_names=None,
    include_space_dummy=true,
    include_time_dummy=false,
    space_dummy_ci="scalar",
    time_dummy_encoding="integer",
    time_dummy_ci="scalar",
    max_cond_size=2,
))]
fn discover_jpcmci_plus(
    py: Python<'_>,
    names: Vec<String>,
    env_columns: Vec<Vec<NumericColumn<'_>>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    context_names: Option<Vec<String>>,
    include_space_dummy: bool,
    include_time_dummy: bool,
    space_dummy_ci: &str,
    time_dummy_encoding: &str,
    time_dummy_ci: &str,
    max_cond_size: usize,
) -> PyResult<PcmciDiscoveryResult> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    if env_columns.is_empty() {
        return Err(crate::value_err("discover_jpcmci_plus needs ≥1 environment"));
    }
    let mut batches = Vec::with_capacity(env_columns.len());
    for cols in &env_columns {
        batches.push(numeric_columns_to_batch(&names, cols)?);
    }
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    let context_names = context_names.unwrap_or_default();
    let threads = if is_callback { Some(1) } else { threads };
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
                return Err(crate::value_err(format!(
                    "context_names entry '{cname}' not found in names"
                )));
            };
            context_ids.push(all_variables[idx]);
        }
        let system: Vec<VariableId> =
            all_variables.iter().copied().filter(|v| !context_ids.contains(v)).collect();
        if system.is_empty() {
            return Err(crate::value_err(
                "discover_jpcmci_plus needs ≥1 system variable after excluding context_names",
            ));
        }

        let (space_dummy_ci, time_dummy_encoding, time_dummy_ci) =
            parse_dummy_ci_modes(space_dummy_ci, time_dummy_encoding, time_dummy_ci)?;
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
            max_cond_size,
            knowledge: BackgroundKnowledge::default(),
        };
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let (result, route) = compiled_search(
            DiscoverySpec::jpcmci_plus(params, ci_name.clone()),
            DiscoverInput::MultiEnv(multi),
            &system,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::TemporalCpdag(result) => Some(result),
                _ => None,
            },
        )?;
        let cpdag = &result.evidence.graph;
        let graph_edges = cpdag_graph_edges(&names, cpdag);
        let slot = ReviewSlot::TemporalCpdag(result.review.clone());
        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            &result.sepsets,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            result.review.pending_undirected.len() as u64,
            ci_name,
            cpdag.node_count() as u64,
            cpdag.directed_edge_count() as u64,
            cpdag.undirected_edge_count() as u64,
            graph_edges,
        )
        .with_review(slot, &names)
        .with_route(route))
    })
}

/// Two-regime half-split assignment (opt-in helper for RPCMCI; not applied by default).
#[pyfunction(name = "two_regime_half_split")]
fn two_regime_half_split_py(series_len: usize) -> Vec<u32> {
    two_regime_half_split(series_len).regimes.iter().map(|r| r.raw()).collect()
}

/// RPCMCI with caller-supplied regimes (required; no silent half-split).
#[pyfunction]
#[pyo3(signature = (names, columns, *, regimes, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=None, cancel=None, max_cond_size=2))]
fn discover_rpcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    regimes: Vec<u32>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    max_cond_size: usize,
) -> PyResult<RpcmciDiscoverySummary> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let batch = numeric_columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    drop(columns);
    let threads = if is_callback { Some(1) } else { threads };
    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        if regimes.len() != series.row_count() {
            return Err(crate::value_err(format!(
                "regimes length {} != series length {}",
                regimes.len(),
                series.row_count()
            )));
        }
        let assign = RegimeAssignment::try_new(
            regimes.into_iter().map(RegimeId::from_raw).collect::<Vec<_>>(),
        )
        .map_err(|e| crate::value_err(e.to_string()))?;
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints::default(),
            max_cond_size,
            knowledge: BackgroundKnowledge::default(),
        };
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let (result, _route) = compiled_search(
            DiscoverySpec::rpcmci(params, assign, None, ci_name),
            DiscoverInput::Series(series),
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::Regimes(result) => Some(result),
                _ => None,
            },
        )?;
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

/// RPCMCI discovery accepted through the Rust review gate.
///
/// Returns the accepted structure as a `TemporalDag` or `TemporalCpdag`. A
/// single accepted graph exists only when discovery found one regime; more
/// regimes raise `ReviewRequired` (each regime's CPDAG is reviewed separately).
/// `accept_discovered=False` leaves pending undirected edges under review.
#[pyfunction]
#[pyo3(signature = (names, columns, *, regimes, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=None, cancel=None, max_cond_size=2, accept_discovered=true))]
fn accept_rpcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    regimes: Vec<u32>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    max_cond_size: usize,
    accept_discovered: bool,
) -> PyResult<Py<PyAny>> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let batch = numeric_columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    drop(columns);
    let threads = if is_callback { Some(1) } else { threads };
    let out_names = names.clone();
    let accepted = detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        if regimes.len() != series.row_count() {
            return Err(crate::value_err(format!(
                "regimes length {} != series length {}",
                regimes.len(),
                series.row_count()
            )));
        }
        let assign = RegimeAssignment::try_new(
            regimes.into_iter().map(RegimeId::from_raw).collect::<Vec<_>>(),
        )
        .map_err(|e| crate::value_err(e.to_string()))?;
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints::default(),
            max_cond_size,
            knowledge: BackgroundKnowledge::default(),
        };
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let (result, _route) = compiled_search(
            DiscoverySpec::rpcmci(params, assign, None, ci_name),
            DiscoverInput::Series(series),
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::Regimes(result) => Some(result),
                _ => None,
            },
        )?;
        crate::accept_rpcmci_review(&result, accept_discovered).map_err(py_err)
    })?;
    match accepted.graph_ref() {
        GraphRef::TemporalDag(dag) => {
            Ok(Py::new(py, graphs::TemporalDag { dag: dag.clone(), names: out_names })?.into_any())
        }
        GraphRef::TemporalCpdag(cpdag) => {
            Ok(Py::new(py, graphs::TemporalCpdag { cpdag: cpdag.clone(), names: out_names })?
                .into_any())
        }
        _ => Err(crate::value_err("RPCMCI acceptance produced no temporal graph")),
    }
}

pub(super) fn register_first(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(accept_rpcmci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pcmci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pcmci_plus, m)?)?;
    Ok(())
}

pub(super) fn register_remaining(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(discover_lpcmci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_jpcmci_plus, m)?)?;
    m.add_function(wrap_pyfunction!(discover_rpcmci, m)?)?;
    m.add_function(wrap_pyfunction!(two_regime_half_split_py, m)?)?;
    Ok(())
}
