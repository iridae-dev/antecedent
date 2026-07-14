//! `PyO3` bindings — Phase 0–7: Arrow load, `analyze_ate` (incl. Bayesian),
//! `analyze`, `discover_pcmci`, `discover_pcmci_plus`, GCM fit/sample/CF.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs)]
#![allow(unsafe_code)] // required by PyO3
#![allow(
    clippy::doc_markdown,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::cast_possible_truncation
)]

use std::sync::Arc;

use arrow_array::{Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use causal::{
    BayesianConfig, CausalAnalysis, InferenceMode, JpcmciPlus, RefuteSuite, Rpcmci,
    TemporalLinearPredictor, TemporalMediationEstimator, counterfactual_ite, fit_gcm, sample_do,
    two_regime_half_split,
};
use causal_core::{
    AverageEffectQuery, CausalRng, ExecutionContext, Intervention, Lag, MediationContrast,
    MediationQuery, TemporalEffectQuery, TemporalPolicy, Value, VariableId,
};
use causal_data::{
    MultiEnvironmentData, SamplingRegularity, TableView, TimeIndex, TimeSeriesData,
    tabular_from_record_batch,
};
use causal_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, Lpcmci, Pcmci, PcmciPlus, TemporalConstraints,
};
use causal_expr::{CausalExprArena, IdentifiedEstimand};
use causal_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};
use causal_stats::ci_from_name;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn py_err(e: impl ToString) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Result of loading columns into the Rust data layer.
#[pyclass]
struct ArrowLoadInfo {
    #[pyo3(get)]
    row_count: usize,
    #[pyo3(get)]
    column_count: usize,
    #[pyo3(get)]
    bytes_copied: u64,
    #[pyo3(get)]
    diagnostic_count: usize,
}

/// Coarse-grained ATE analysis result (single boundary crossing).
#[pyclass]
struct AteAnalysisResult {
    #[pyo3(get)]
    ate: f64,
    #[pyo3(get)]
    se_analytic: f64,
    #[pyo3(get)]
    se_bootstrap: Option<f64>,
    #[pyo3(get)]
    adjustment_set: Vec<String>,
    #[pyo3(get)]
    identification_status: String,
    #[pyo3(get)]
    refutation_passed: bool,
    #[pyo3(get)]
    refutation_count: usize,
    #[pyo3(get)]
    assumption_count: usize,
    #[pyo3(get)]
    derivation_step_count: usize,
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    estimator_id: String,
    #[pyo3(get)]
    overlap_ess: Option<f64>,
    #[pyo3(get)]
    overlap_propensity_min: Option<f64>,
    /// Posterior mean of the primary effect (Bayesian path).
    #[pyo3(get)]
    posterior_effect_mean: Option<f64>,
    /// Posterior SD of the primary effect.
    #[pyo3(get)]
    posterior_effect_sd: Option<f64>,
    /// 2.5% quantile of the primary effect.
    #[pyo3(get)]
    posterior_q025: Option<f64>,
    /// 97.5% quantile of the primary effect.
    #[pyo3(get)]
    posterior_q975: Option<f64>,
    /// Number of posterior draws.
    #[pyo3(get)]
    posterior_n_draws: Option<usize>,
    /// Empirical P(effect < 0).
    #[pyo3(get)]
    posterior_p_below_zero: Option<f64>,
    /// Inference backend id (e.g. laplace / conjugate_gaussian).
    #[pyo3(get)]
    posterior_backend: Option<String>,
}

fn columns_to_batch(
    names: &[String],
    columns: &[PyReadonlyArray1<'_, f64>],
) -> PyResult<RecordBatch> {
    if names.len() != columns.len() {
        return Err(PyValueError::new_err("names and columns must have the same length"));
    }
    if columns.is_empty() {
        return Err(PyValueError::new_err("at least one column required"));
    }
    let n = columns[0].as_array().len();
    for col in columns {
        if col.as_array().len() != n {
            return Err(PyValueError::new_err("column length mismatch"));
        }
    }
    let fields: Vec<Field> =
        names.iter().map(|nm| Field::new(nm, DataType::Float64, true)).collect();
    let schema = Schema::new(fields);
    // Contiguous copy from NumPy buffers (no Option-per-element intermediate).
    let arrays: Vec<Arc<dyn arrow_array::Array>> = columns
        .iter()
        .map(|c| {
            let slice = c.as_array();
            let values: Vec<f64> = slice.iter().copied().collect();
            Arc::new(Float64Array::from(values)) as Arc<dyn arrow_array::Array>
        })
        .collect();
    RecordBatch::try_new(Arc::new(schema), arrays).map_err(py_err)
}

/// Load float64 NumPy columns (copied into Arrow, then into library-owned storage).
#[pyfunction]
fn load_float64_columns(
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
) -> PyResult<ArrowLoadInfo> {
    let batch = columns_to_batch(&names, &columns)?;
    let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
    Ok(ArrowLoadInfo {
        row_count: loaded.data.row_count(),
        column_count: loaded.data.schema().len(),
        bytes_copied: loaded.bytes_copied,
        diagnostic_count: loaded.diagnostics.len(),
    })
}

/// Run static ATE: identify → estimate → optional refute (Phase 4; DESIGN.md §21.2).
///
/// `identifier`/`estimator` select the identification strategy and estimator; leaving both
/// `None` preserves the Phase 0–3 default (`backdoor.adjustment` + `linear.adjustment.ate`).
/// See [`causal::CausalAnalysisBuilder::identifier`] and
/// [`causal::CausalAnalysisBuilder::estimator`] for the supported ids.
///
/// Crosses the Python boundary once: NumPy columns + edge list in, structured
/// summary out. No per-row callbacks. Releases the GIL during native work.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    edges,
    treatment,
    outcome,
    *,
    identifier=None,
    estimator=None,
    inference=None,
    n_draws=1000,
    prior_scale=10.0,
    refute=true,
    seed=1,
    bootstrap=50
))]
fn analyze_ate(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    identifier: Option<String>,
    estimator: Option<String>,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    refute: bool,
    seed: u64,
    bootstrap: u32,
) -> PyResult<AteAnalysisResult> {
    let batch = columns_to_batch(&names, &columns)?;
    // Drop NumPy borrows before releasing the GIL.
    drop(columns);

    py.allow_threads(move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;

        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut dag = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data
                .schema()
                .id_of(from)
                .map_err(|e| PyValueError::new_err(format!("edge from: {e}")))?;
            let to_id = data
                .schema()
                .id_of(to)
                .map_err(|e| PyValueError::new_err(format!("edge to: {e}")))?;
            dag.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }

        let query = AverageEffectQuery::binary_ate(t_id, y_id);
        let suite = if refute { RefuteSuite::PlaceboAndRcc } else { RefuteSuite::None };
        let mut builder = CausalAnalysis::builder()
            .data(data)
            .graph(dag)
            .query(query)
            .refute(suite)
            .bootstrap_replicates(bootstrap);
        if let Some(id) = identifier {
            builder = builder.identifier(id);
        }
        if let Some(est) = estimator {
            builder = builder.estimator(est);
        }
        if let Some(mode) = inference {
            let cfg = match mode.to_ascii_lowercase().as_str() {
                "bayesian" | "bayesian.laplace" | "laplace" => {
                    BayesianConfig::laplace().n_draws(n_draws).prior_scale(prior_scale)
                }
                "bayesian.conjugate" | "conjugate" => {
                    BayesianConfig::conjugate().n_draws(n_draws).prior_scale(prior_scale)
                }
                "frequentist" => {
                    builder = builder.inference(InferenceMode::Frequentist);
                    let analysis = builder.build().map_err(py_err)?;
                    let ctx = ExecutionContext::for_tests(seed);
                    let result = analysis.run(&ctx).map_err(py_err)?;
                    return Ok(ate_result_from_analysis(&names, result));
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown inference mode {other:?}; use frequentist|bayesian|conjugate"
                    )));
                }
            };
            builder = builder.inference(InferenceMode::Bayesian(cfg)).refute(RefuteSuite::None);
        }
        let analysis = builder.build().map_err(py_err)?;
        let ctx = ExecutionContext::for_tests(seed);
        let result = analysis.run(&ctx).map_err(py_err)?;
        Ok(ate_result_from_analysis(&names, result))
    })
}

fn ate_result_from_analysis(
    names: &[String],
    result: causal::CausalAnalysisResult,
) -> AteAnalysisResult {
    let adjustment_set: Vec<String> = result
        .estimand
        .adjustment_set
        .iter()
        .map(|id| names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("var{}", id.raw())))
        .collect();

    let refutation_passed =
        result.refutations.is_empty() || result.refutations.iter().all(|r| r.passed);
    let estimator_id = result.logical_plan.estimator.as_deref().unwrap_or("").to_string();
    let overlap_ess = result.estimate.overlap_report.as_ref().map(|r| r.ess);
    let overlap_propensity_min = result.estimate.overlap_report.as_ref().map(|r| r.propensity_min);

    let (
        posterior_effect_mean,
        posterior_effect_sd,
        posterior_q025,
        posterior_q975,
        posterior_n_draws,
        posterior_p_below_zero,
        posterior_backend,
    ) = if let Some(post) = result.posterior.as_ref() {
        let eq = post.effect_column().unwrap_or(0);
        (
            Some(post.summaries.mean[eq]),
            Some(post.summaries.sd[eq]),
            Some(post.summaries.q025[eq]),
            Some(post.summaries.q975[eq]),
            Some(post.draws.n_draws),
            post.probability_below(0.0).ok(),
            Some(post.diagnostics.backend_id.to_string()),
        )
    } else {
        (None, None, None, None, None, None, None)
    };

    AteAnalysisResult {
        ate: result.estimate.ate,
        se_analytic: result.estimate.se_analytic,
        se_bootstrap: result.estimate.se_bootstrap,
        adjustment_set,
        identification_status: format!("{:?}", result.identification.status),
        refutation_passed,
        refutation_count: result.refutations.len(),
        assumption_count: result.estimate.assumptions.len(),
        derivation_step_count: result.identification.derivation.steps.len(),
        method: result.estimand.method.to_string(),
        estimator_id,
        overlap_ess,
        overlap_propensity_min,
        posterior_effect_mean,
        posterior_effect_sd,
        posterior_q025,
        posterior_q975,
        posterior_n_draws,
        posterior_p_below_zero,
        posterior_backend,
    }
}

/// One discovered lagged link for Python.
#[pyclass]
#[derive(Clone)]
struct DiscoveredLink {
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    source_lag: u32,
    #[pyo3(get)]
    target: String,
    #[pyo3(get)]
    target_lag: u32,
    #[pyo3(get)]
    statistic: f64,
    #[pyo3(get)]
    p_value: f64,
    /// Benjamini–Hochberg adjusted p-value when FDR ran; otherwise `None`.
    #[pyo3(get)]
    adjusted_p_value: Option<f64>,
}

/// Coarse-grained PCMCI discovery result (single boundary crossing).
///
/// Field set is the stable Rust↔Python temporal discovery schema for Phase 2.
#[pyclass]
struct PcmciDiscoveryResult {
    #[pyo3(get)]
    links: Vec<DiscoveredLink>,
    #[pyo3(get)]
    algorithm_id: String,
    #[pyo3(get)]
    algorithm_config: String,
    #[pyo3(get)]
    ci_tests: u64,
    #[pyo3(get)]
    links_retained: u64,
    #[pyo3(get)]
    pending_edge_count: u64,
    #[pyo3(get)]
    lagged_frame_bytes: u64,
    #[pyo3(get)]
    worker_threads: u32,
    #[pyo3(get)]
    ci_name: String,
    #[pyo3(get)]
    cpdag_nodes: u64,
    #[pyo3(get)]
    cpdag_directed_edges: u64,
    #[pyo3(get)]
    cpdag_undirected_edges: u64,
}

fn resolve_ci(
    ci: &str,
    weights: Option<Vec<f64>>,
) -> PyResult<Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>> {
    let key = ci.trim().to_ascii_lowercase();
    if matches!(key.as_str(), "weighted_parcorr" | "weighted_partial_corr") {
        let Some(w) = weights else {
            return Err(PyValueError::new_err("weights required when ci='weighted_parcorr'"));
        };
        return Ok(Arc::new(causal_stats::WeightedPartialCorrelation::new(w)));
    }
    if weights.is_some() {
        return Err(PyValueError::new_err(
            "observation weights are only supported when ci='weighted_parcorr'",
        ));
    }
    ci_from_name(ci).map_err(py_err)
}

fn series_from_batch(batch: &RecordBatch) -> PyResult<(TimeSeriesData, Vec<VariableId>)> {
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

fn discovered_links(
    names: &[String],
    links: &[causal_discovery::ScoredLink],
) -> Vec<DiscoveredLink> {
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
    links: &[causal_discovery::ScoredLink],
    algorithm_id: &str,
    algorithm_config: &str,
    performance: &causal_discovery::DiscoveryPerformanceRecord,
    pending_edge_count: u64,
    ci_name: String,
    cpdag_nodes: u64,
    cpdag_directed_edges: u64,
    cpdag_undirected_edges: u64,
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
    }
}

/// Run lagged PCMCI discovery.
///
/// NumPy columns in, structured link list out once. No per-query Python callbacks.
/// `ci` selects the conditional-independence test by name (default `parcorr`).
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci="parcorr", weights=None))]
fn discover_pcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: &str,
    weights: Option<Vec<f64>>,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let ci_name = ci.to_string();
    let ci_impl = resolve_ci(ci, weights)?;
    drop(columns);

    py.allow_threads(move || {
        let (series, variables) = series_from_batch(&batch)?;
        let pcmci = Pcmci::new()
            .with_fdr(fdr)
            .with_constraints(DiscoveryConstraints {
                temporal: TemporalConstraints {
                    max_lag: Lag::from_raw(max_lag),
                    min_lag: Lag::from_raw(1),
                },
                alpha,
                max_cond_size: 2,
                ..DiscoveryConstraints::default()
            })
            .with_ci(ci_impl);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(seed);
        let result = pcmci.run(&series, &variables, &mut ws, &ctx).map_err(py_err)?;
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
        ))
    })
}

/// Run PCMCI+ discovery returning links plus oriented temporal CPDAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci="parcorr", weights=None))]
fn discover_pcmci_plus(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: &str,
    weights: Option<Vec<f64>>,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let ci_name = ci.to_string();
    let ci_impl = resolve_ci(ci, weights)?;
    drop(columns);

    py.allow_threads(move || {
        let (series, variables) = series_from_batch(&batch)?;
        let plus = PcmciPlus::new()
            .with_fdr(fdr)
            .with_constraints(DiscoveryConstraints {
                temporal: TemporalConstraints {
                    max_lag: Lag::from_raw(max_lag),
                    min_lag: Lag::CONTEMPORANEOUS,
                },
                alpha,
                max_cond_size: 2,
                ..DiscoveryConstraints::default()
            })
            .with_ci(ci_impl);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(seed);
        let result = plus.run(&series, &variables, &mut ws, &ctx).map_err(py_err)?;

        let cpdag = &result.evidence.graph;
        let directed = cpdag.directed_edge_count() as u64;
        let undirected = cpdag.undirected_edge_count() as u64;
        let pending = result.review.pending_edges.len() as u64
            + result.review.pending_undirected.len() as u64;

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
        ))
    })
}

/// Run LPCMCI discovery returning links plus temporal PAG summary (no per-edge GIL).
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci="parcorr", weights=None))]
fn discover_lpcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: &str,
    weights: Option<Vec<f64>>,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let ci_name = ci.to_string();
    let ci_impl = resolve_ci(ci, weights)?;
    drop(columns);

    py.allow_threads(move || {
        let (series, variables) = series_from_batch(&batch)?;
        let alg = Lpcmci::new()
            .with_fdr(fdr)
            .with_constraints(DiscoveryConstraints {
                temporal: TemporalConstraints {
                    max_lag: Lag::from_raw(max_lag),
                    min_lag: Lag::CONTEMPORANEOUS,
                },
                alpha,
                max_cond_size: 2,
                ..DiscoveryConstraints::default()
            })
            .with_ci(ci_impl);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(seed);
        let result = alg.run(&series, &variables, &mut ws, &ctx).map_err(py_err)?;

        let pag = &result.evidence.graph;
        let pending = result.review.pending_circles.len() as u64;
        // Count definite directed edges for summary.
        let mut directed = 0u64;
        for i in 0..pag.node_count() {
            let a = DenseNodeId::from_raw(i as u32);
            for (b, at_a, at_b) in pag.neighbors(a) {
                if b.raw() < a.raw() {
                    continue;
                }
                if matches!(
                    (at_a, at_b),
                    (
                        causal_graph::Endpoint::Tail,
                        causal_graph::Endpoint::Arrow
                    ) | (
                        causal_graph::Endpoint::Arrow,
                        causal_graph::Endpoint::Tail
                    )
                ) {
                    directed += 1;
                }
            }
        }

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
        ))
    })
}

/// J-PCMCI+ over multiple environments (one GIL crossing).
///
/// `env_columns` is a list of column batches (each env: same `names` order).
#[pyfunction]
#[pyo3(signature = (names, env_columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci="parcorr", weights=None))]
fn discover_jpcmci_plus(
    py: Python<'_>,
    names: Vec<String>,
    env_columns: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: &str,
    weights: Option<Vec<f64>>,
) -> PyResult<PcmciDiscoveryResult> {
    if env_columns.is_empty() {
        return Err(PyValueError::new_err("discover_jpcmci_plus needs ≥1 environment"));
    }
    let mut batches = Vec::with_capacity(env_columns.len());
    for cols in &env_columns {
        batches.push(columns_to_batch(&names, cols)?);
    }
    let ci_name = ci.to_string();
    let ci_impl = resolve_ci(ci, weights)?;
    drop(env_columns);

    py.allow_threads(move || {
        let mut series_list = Vec::with_capacity(batches.len());
        let mut variables = Vec::new();
        for (i, batch) in batches.iter().enumerate() {
            let (series, vars) = series_from_batch(batch)?;
            if i == 0 {
                variables = vars;
            }
            series_list.push(series);
        }
        let multi = MultiEnvironmentData::try_new(Arc::from(series_list)).map_err(py_err)?;
        let alg = JpcmciPlus::new()
            .with_fdr(fdr)
            .with_constraints(DiscoveryConstraints {
                temporal: TemporalConstraints {
                    max_lag: Lag::from_raw(max_lag),
                    min_lag: Lag::CONTEMPORANEOUS,
                },
                alpha,
                max_cond_size: 2,
                ..DiscoveryConstraints::default()
            })
            .with_ci(ci_impl);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(seed);
        let result = alg.run(&multi, &variables, &mut ws, &ctx).map_err(py_err)?;
        let cpdag = &result.evidence.graph;
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
        ))
    })
}

/// RPCMCI with half-split regimes (one GIL crossing).
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci="parcorr", weights=None))]
fn discover_rpcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: &str,
    weights: Option<Vec<f64>>,
) -> PyResult<RpcmciDiscoverySummary> {
    let batch = columns_to_batch(&names, &columns)?;
    let ci_impl = resolve_ci(ci, weights)?;
    drop(columns);
    py.allow_threads(move || {
        let (series, variables) = series_from_batch(&batch)?;
        let assign = two_regime_half_split(series.row_count());
        let plus = PcmciPlus::new()
            .with_fdr(fdr)
            .with_constraints(DiscoveryConstraints {
                temporal: TemporalConstraints {
                    max_lag: Lag::from_raw(max_lag),
                    min_lag: Lag::CONTEMPORANEOUS,
                },
                alpha,
                max_cond_size: 2,
                ..DiscoveryConstraints::default()
            })
            .with_ci(ci_impl);
        let alg = Rpcmci::new().with_min_regime_len(40).with_pcmci_plus(plus);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(seed);
        let result = alg.run(&series, &variables, &assign, &mut ws, &ctx).map_err(py_err)?;
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

/// Mediation effect surface summary (total / direct / mediated).
#[pyfunction]
#[pyo3(signature = (names, columns, treatment, mediator, outcome, *, seed=1))]
fn mediation_effects_summary(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    treatment: String,
    mediator: String,
    outcome: String,
    seed: u64,
) -> PyResult<MediationEffectsSummary> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    py.allow_threads(move || {
        let (series, _) = series_from_batch(&batch)?;
        let id = |nm: &str| {
            series
                .schema()
                .id_of(nm)
                .map_err(|e| PyValueError::new_err(format!("unknown variable {nm}: {e}")))
        };
        let t = id(&treatment)?;
        let m = id(&mediator)?;
        let y = id(&outcome)?;
        let q = MediationQuery::binary(t, y, [m], MediationContrast::Total);
        let mut arena = CausalExprArena::new();
        let functional = arena.frontdoor_ate(
            t,
            y,
            &[m],
            Value::f64(1.0),
            Value::f64(0.0),
        );
        let estimand = IdentifiedEstimand::frontdoor(
            "temporal_mediation.total",
            Arc::from([m]),
            functional,
        );
        let ctx = ExecutionContext::for_tests(seed);
        let surface = TemporalMediationEstimator::new()
            .effect_surface(&series, &estimand, &q, &ctx)
            .map_err(py_err)?;
        Ok(MediationEffectsSummary {
            total: surface.total,
            direct: surface.direct,
            mediated: surface.mediated,
        })
    })
}

/// Intervene+predict summary (mean predicted outcome under do(parent=level)).
#[pyfunction]
#[pyo3(signature = (names, columns, target, parent, *, parent_lag=1, level=1.0))]
fn predict_intervened_summary(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    target: String,
    parent: String,
    parent_lag: u32,
    level: f64,
) -> PyResult<PredictSummary> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    py.allow_threads(move || {
        let (series, _) = series_from_batch(&batch)?;
        let id = |nm: &str| {
            series
                .schema()
                .id_of(nm)
                .map_err(|e| PyValueError::new_err(format!("unknown variable {nm}: {e}")))
        };
        let y = id(&target)?;
        let x = id(&parent)?;
        let pred = TemporalLinearPredictor::fit(
            &series,
            y,
            [causal_data::LaggedColumn { variable: x, lag: Lag::from_raw(parent_lag) }],
        )
        .map_err(py_err)?;
        let yhat = pred.predict_intervened(&series, x, level).map_err(py_err)?;
        let mean = yhat.iter().sum::<f64>() / yhat.len().max(1) as f64;
        Ok(PredictSummary { mean_prediction: mean, n: yhat.len() as u64 })
    })
}

/// RPCMCI summary (typed regimes, no single-graph collapse).
#[pyclass]
struct RpcmciDiscoverySummary {
    #[pyo3(get)]
    algorithm: String,
    #[pyo3(get)]
    n_regimes: u64,
    #[pyo3(get)]
    regime_ids: Vec<u32>,
    #[pyo3(get)]
    directed_edges: Vec<u64>,
    #[pyo3(get)]
    undirected_edges: Vec<u64>,
}

/// Mediation effects summary.
#[pyclass]
struct MediationEffectsSummary {
    #[pyo3(get)]
    total: f64,
    #[pyo3(get)]
    direct: f64,
    #[pyo3(get)]
    mediated: f64,
}

/// Prediction summary under intervention.
#[pyclass]
struct PredictSummary {
    #[pyo3(get)]
    mean_prediction: f64,
    #[pyo3(get)]
    n: u64,
}

/// Unified analysis result (static or temporal).
#[pyclass]
#[derive(Clone)]
struct AnalysisResult {
    #[pyo3(get)]
    ate: f64,
    #[pyo3(get)]
    se_analytic: f64,
    #[pyo3(get)]
    se_bootstrap: Option<f64>,
    #[pyo3(get)]
    plan_id: String,
    #[pyo3(get)]
    modality: String,
    #[pyo3(get)]
    peak_memory_bytes: Option<u64>,
    #[pyo3(get)]
    identification_status: String,
    #[pyo3(get)]
    method: String,
}

/// Run temporal effect analysis with a supplied lagged edge list.
///
/// `edges` are `(source, source_lag, target, target_lag)` with lags ≥ 0.
/// `treatment_lag` is the pulse offset as a non-negative lag (pulse at `-treatment_lag`).
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    edges,
    treatment,
    outcome,
    *,
    treatment_lag=1,
    horizon_steps=1,
    active_level=1.0,
    seed=1,
    bootstrap=0
))]
fn analyze(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, u32, String, u32)>,
    treatment: String,
    outcome: String,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    seed: u64,
    bootstrap: u32,
) -> PyResult<AnalysisResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    py.allow_threads(move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let tabular = loaded.data;
        let n = tabular.row_count();
        let series = TimeSeriesData::try_new(
            tabular.storage().clone(),
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .map_err(py_err)?;

        let name_to_id = |nm: &str| -> PyResult<VariableId> {
            series
                .schema()
                .id_of(nm)
                .map_err(|e| PyValueError::new_err(format!("unknown variable {nm}: {e}")))
        };
        let t_id = name_to_id(&treatment)?;
        let y_id = name_to_id(&outcome)?;

        let mut g = TemporalDag::empty();
        for (src, slag, tgt, tlag) in &edges {
            let s =
                ensure_lagged(&mut g, name_to_id(src)?, Lag::from_raw(*slag)).map_err(py_err)?;
            let t =
                ensure_lagged(&mut g, name_to_id(tgt)?, Lag::from_raw(*tlag)).map_err(py_err)?;
            g.insert_directed(s, t).map_err(py_err)?;
        }

        let pulse_at = -i32::try_from(treatment_lag)
            .map_err(|_| PyValueError::new_err("treatment_lag too large"))?;
        let q = TemporalEffectQuery::pulse(t_id, y_id, active_level)
            .with_policy(TemporalPolicy::pulse(pulse_at))
            .with_horizon_steps(horizon_steps);

        let analysis = CausalAnalysis::builder()
            .series(series)
            .temporal_graph(g)
            .temporal_query(q)
            .bootstrap_replicates(bootstrap)
            .build()
            .map_err(py_err)?;
        let ctx = ExecutionContext::for_tests(seed);
        let result = analysis.run(&ctx).map_err(py_err)?;
        Ok(AnalysisResult {
            ate: result.estimate.ate,
            se_analytic: result.estimate.se_analytic,
            se_bootstrap: result.estimate.se_bootstrap,
            plan_id: result.logical_plan.plan_id.to_string(),
            modality: format!("{:?}", result.logical_plan.data_classification),
            peak_memory_bytes: result.physical_plan.estimated_peak_memory_bytes,
            identification_status: format!("{:?}", result.identification.status),
            method: result.estimand.method.to_string(),
        })
    })
}

/// Result of a GCM counterfactual ITE (single boundary crossing).
#[pyclass]
struct GcmIteResult {
    #[pyo3(get)]
    mean_ite: f64,
    #[pyo3(get)]
    n_units: usize,
    #[pyo3(get)]
    noise_inference: String,
    #[pyo3(get)]
    n_assignments: usize,
}

/// Interventional sample summary under hard `do` (means only; no per-draw GIL).
#[pyclass]
struct GcmSampleResult {
    #[pyo3(get)]
    column_means: Vec<f64>,
    #[pyo3(get)]
    n_draws: usize,
    #[pyo3(get)]
    n_nodes: usize,
}

/// Fit a linear-Gaussian GCM and return mean ITE under hard interventions.
///
/// Crosses the Python boundary once: NumPy columns + edges in, summary out.
#[pyfunction]
fn gcm_counterfactual_ite(
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    active: f64,
    control: f64,
    seed: u64,
) -> PyResult<GcmIteResult> {
    Python::with_gil(|_py| {
        let batch = columns_to_batch(&names, &columns)?;
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let n_assignments = fitted.assignments.len();
        let ctx = ExecutionContext::for_tests(seed);
        let ite = counterfactual_ite(fitted.model, &data, t_id, y_id, active, control, &ctx)
            .map_err(py_err)?;
        Ok(GcmIteResult {
            mean_ite: ite.mean_ite,
            n_units: ite.unit_effects.len(),
            noise_inference: format!("{:?}", ite.noise_inference),
            n_assignments,
        })
    })
}

/// Fit GCM and return interventional column means under hard `do(treatment=value)`.
#[pyfunction]
fn gcm_sample_do(
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    do_value: f64,
    n_draws: usize,
    seed: u64,
) -> PyResult<GcmSampleResult> {
    Python::with_gil(|_py| {
        let batch = columns_to_batch(&names, &columns)?;
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let ctx = ExecutionContext::for_tests(seed);
        let mut rng = CausalRng::from_seed(seed);
        let samples = sample_do(
            &fitted.model,
            &[Intervention::set(t_id, Value::f64(do_value))],
            n_draws,
            &mut rng,
            &ctx,
        )
        .map_err(py_err)?;
        let mut means = Vec::with_capacity(samples.n_nodes);
        for i in 0..samples.n_nodes {
            let col = samples.column(i).map_err(py_err)?;
            let m = col.iter().sum::<f64>() / col.len().max(1) as f64;
            means.push(m);
        }
        Ok(GcmSampleResult {
            column_means: means,
            n_draws: samples.n_rows,
            n_nodes: samples.n_nodes,
        })
    })
}

/// Python module `causal._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_float64_columns, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate, m)?)?;
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pcmci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pcmci_plus, m)?)?;
    m.add_function(wrap_pyfunction!(discover_lpcmci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_jpcmci_plus, m)?)?;
    m.add_function(wrap_pyfunction!(discover_rpcmci, m)?)?;
    m.add_function(wrap_pyfunction!(mediation_effects_summary, m)?)?;
    m.add_function(wrap_pyfunction!(predict_intervened_summary, m)?)?;
    m.add_function(wrap_pyfunction!(gcm_counterfactual_ite, m)?)?;
    m.add_function(wrap_pyfunction!(gcm_sample_do, m)?)?;
    m.add_class::<ArrowLoadInfo>()?;
    m.add_class::<AteAnalysisResult>()?;
    m.add_class::<AnalysisResult>()?;
    m.add_class::<DiscoveredLink>()?;
    m.add_class::<PcmciDiscoveryResult>()?;
    m.add_class::<RpcmciDiscoverySummary>()?;
    m.add_class::<MediationEffectsSummary>()?;
    m.add_class::<PredictSummary>()?;
    m.add_class::<GcmIteResult>()?;
    m.add_class::<GcmSampleResult>()?;
    m.add("__version__", causal_core::VERSION)?;
    Ok(())
}
