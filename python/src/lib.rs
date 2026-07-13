//! PyO3 bindings — Phase 0–2: Arrow load, `analyze_ate`, `discover_pcmci`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs)]
#![allow(unsafe_code)] // required by PyO3

use std::sync::Arc;

use arrow_array::{Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use causal_analysis::{CausalAnalysis, RefuteSuite};
use causal_core::{AverageEffectQuery, ExecutionContext, Lag, VariableId};
use causal_data::{
    SamplingRegularity, TableView, TimeIndex, TimeSeriesData, tabular_from_record_batch,
};
use causal_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, Pcmci, TemporalConstraints,
};
use causal_graph::{Dag, DenseNodeId};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    RecordBatch::try_new(Arc::new(schema), arrays).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Load float64 NumPy columns (copied into Arrow, then into library-owned storage).
#[pyfunction]
fn load_float64_columns(
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
) -> PyResult<ArrowLoadInfo> {
    let batch = columns_to_batch(&names, &columns)?;
    let loaded =
        tabular_from_record_batch(&batch).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(ArrowLoadInfo {
        row_count: loaded.data.row_count(),
        column_count: loaded.data.schema().len(),
        bytes_copied: loaded.bytes_copied,
        diagnostic_count: loaded.diagnostics.len(),
    })
}

/// Run static ATE: identify (backdoor) → linear estimate → optional refute.
///
/// Crosses the Python boundary once: NumPy columns + edge list in, structured
/// summary out. No per-row callbacks. Releases the GIL during native work.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment, outcome, *, refute=true, seed=1, bootstrap=50))]
fn analyze_ate(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    refute: bool,
    seed: u64,
    bootstrap: u32,
) -> PyResult<AteAnalysisResult> {
    let batch = columns_to_batch(&names, &columns)?;
    // Drop NumPy borrows before releasing the GIL.
    drop(columns);

    py.allow_threads(move || {
        let loaded =
            tabular_from_record_batch(&batch).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let data = loaded.data;
        let t_id =
            data.schema().id_of(&treatment).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let y_id =
            data.schema().id_of(&outcome).map_err(|e| PyValueError::new_err(e.to_string()))?;

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
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }

        let query = AverageEffectQuery::binary_ate(t_id, y_id);
        let suite = if refute { RefuteSuite::PlaceboAndRcc } else { RefuteSuite::None };
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(dag)
            .query(query)
            .refute(suite)
            .bootstrap_replicates(bootstrap)
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let ctx = ExecutionContext::for_tests(seed);
        let result = analysis.run(&ctx).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let adjustment_set: Vec<String> = result
            .estimand
            .adjustment_set
            .iter()
            .map(|id| {
                names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("var{}", id.raw()))
            })
            .collect();

        let refutation_passed =
            result.refutations.is_empty() || result.refutations.iter().all(|r| r.passed);

        Ok(AteAnalysisResult {
            ate: result.estimate.ate,
            se_analytic: result.estimate.se_analytic,
            se_bootstrap: result.estimate.se_bootstrap,
            adjustment_set,
            identification_status: format!("{:?}", result.identification.status),
            refutation_passed,
            refutation_count: result.refutations.len(),
            assumption_count: result.estimate.assumptions.len(),
            derivation_step_count: result.identification.derivation.steps.len(),
        })
    })
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
}

/// Coarse-grained PCMCI discovery result (single boundary crossing).
#[pyclass]
struct PcmciDiscoveryResult {
    #[pyo3(get)]
    links: Vec<DiscoveredLink>,
    #[pyo3(get)]
    algorithm_id: String,
    #[pyo3(get)]
    ci_tests: u64,
    #[pyo3(get)]
    links_retained: u64,
}

/// Run lagged PCMCI discovery.
///
/// NumPy columns in, structured link list out once. No per-query Python callbacks.
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1))]
fn discover_pcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);

    py.allow_threads(move || {
        let loaded =
            tabular_from_record_batch(&batch).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let tabular = loaded.data;
        let n = tabular.row_count();
        let series = TimeSeriesData::try_new(
            tabular.storage().clone(),
            TimeIndex {
                regularity: SamplingRegularity::Regular { interval_ns: 1 },
                length: n,
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let variables: Vec<VariableId> = series
            .schema()
            .variables()
            .iter()
            .map(|v| v.id)
            .collect();
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
            });
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(seed);
        let result = pcmci
            .run(&series, &variables, &mut ws, &ctx)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let links: Vec<DiscoveredLink> = result
            .evidence
            .links
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
            })
            .collect();

        Ok(PcmciDiscoveryResult {
            links,
            algorithm_id: result.algorithm.id.to_string(),
            ci_tests: result.performance.ci_tests,
            links_retained: result.performance.links_retained,
        })
    })
}

/// Python module `causal._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_float64_columns, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pcmci, m)?)?;
    m.add_class::<ArrowLoadInfo>()?;
    m.add_class::<AteAnalysisResult>()?;
    m.add_class::<DiscoveredLink>()?;
    m.add_class::<PcmciDiscoveryResult>()?;
    m.add("__version__", causal_core::VERSION)?;
    Ok(())
}
