//! Discovery stability validators bound from `causal-validate::stability`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_arguments)]

use std::sync::Arc;

use arrow_array::RecordBatch;
use causal::discovery::RegimeAssignment;
use causal::discovery_defaults::{jpcmci_constraints, pcmci_constraints, resolve_ci};
use causal_core::{Lag, RegimeId, VariableId};
use causal_data::{EnvHoldoutSplit, MultiEnvironmentData, TableView};
use causal_discovery::{
    DiscoveryWorkspace, JpcmciPlus, MultiDatasetConstraints, Pcmci, PcmciPlus, Rpcmci,
};
use causal_validate::{
    AlphaThresholdSensitivity, BlockBootstrapStability, CiTestSensitivity, EnvironmentHoldout,
    FalsePositiveCheck, LagWindowSensitivity, NullTransform, OrientationStability, RegimeStability,
    SyntheticNullCalibration,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{
    CausalValidateError, columns_to_batch, detach_catch, py_err, py_execution_context,
    series_from_batch,
};

fn py_validate(e: causal_validate::ValidationError) -> PyErr {
    CausalValidateError::new_err(e.to_string())
}

fn pcmci_from_params(max_lag: u32, alpha: f64, fdr: bool, ci: &str) -> PyResult<Pcmci> {
    let ci_impl = resolve_ci(ci, None).map_err(py_err)?;
    Ok(Pcmci::new()
        .with_fdr(fdr)
        .with_constraints(pcmci_constraints(max_lag, alpha))
        .with_ci(ci_impl))
}

fn pcmci_plus_from_params(max_lag: u32, alpha: f64, fdr: bool, ci: &str) -> PyResult<PcmciPlus> {
    let ci_impl = resolve_ci(ci, None).map_err(py_err)?;
    let mut constraints = pcmci_constraints(max_lag, alpha);
    constraints.temporal.min_lag = Lag::CONTEMPORANEOUS;
    Ok(PcmciPlus::new().with_fdr(fdr).with_constraints(constraints).with_ci(ci_impl))
}

fn link_dict(
    py: Python<'_>,
    names: &[String],
    link: causal_discovery::LaggedLink,
) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    let src = names
        .get(link.source.as_usize())
        .cloned()
        .unwrap_or_else(|| format!("var{}", link.source.raw()));
    let tgt = names
        .get(link.target.as_usize())
        .cloned()
        .unwrap_or_else(|| format!("var{}", link.target.raw()));
    d.set_item("source", src)?;
    d.set_item("source_lag", link.source_lag.raw())?;
    d.set_item("target", tgt)?;
    d.set_item("target_lag", link.target_lag.raw())?;
    Ok(d.unbind())
}

fn discovery_stability_dict(
    py: Python<'_>,
    names: &[String],
    report: &causal_validate::DiscoveryStabilityReport,
) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    let mut freqs = Vec::with_capacity(report.frequencies.len());
    for ls in report.frequencies.iter() {
        let entry = PyDict::new(py);
        entry.set_item("link", link_dict(py, names, ls.link)?)?;
        entry.set_item("frequency", ls.frequency)?;
        freqs.push(entry);
    }
    d.set_item("frequencies", freqs)?;
    d.set_item("replicates", report.replicates)?;
    d.set_item("block_size", report.block_size)?;
    Ok(d.unbind())
}

/// Block-bootstrap PCMCI link-frequency stability.
#[pyfunction]
#[pyo3(signature = (
    names, columns, *, max_lag=1, alpha=0.05, fdr=false, ci="parcorr",
    replicates=20, block_size=20, seed=1, threads=1
))]
fn validate_pcmci_block_bootstrap(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    ci: &str,
    replicates: u32,
    block_size: usize,
    seed: u64,
    threads: u32,
) -> PyResult<Py<PyDict>> {
    let batch = columns_to_batch(&names, &columns)?;
    let ci = ci.to_string();
    drop(columns);
    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let pcmci = pcmci_from_params(max_lag, alpha, fdr, &ci)?;
        let checker = BlockBootstrapStability { pcmci, replicates, block_size };
        let mut ws = DiscoveryWorkspace::default();
        let ctx = py_execution_context(seed, threads);
        let report = checker.run(&series, &variables, &mut ws, &ctx).map_err(py_validate)?;
        Python::attach(|py| discovery_stability_dict(py, &names, &report))
    })
}

/// Surrogate false-positive check (column permute or phase-randomize + PCMCI).
#[pyfunction]
#[pyo3(signature = (
    names, columns, *, max_lag=1, alpha=0.05, fdr=false, ci="parcorr",
    transform="permute", replicates=20, seed=1, threads=1
))]
fn validate_pcmci_false_positive(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    ci: &str,
    transform: &str,
    replicates: u32,
    seed: u64,
    threads: u32,
) -> PyResult<Py<PyDict>> {
    let batch = columns_to_batch(&names, &columns)?;
    let ci = ci.to_string();
    let transform = transform.to_ascii_lowercase();
    drop(columns);
    detach_catch(py, move || {
        let null = match transform.as_str() {
            "permute" | "column_permute" => NullTransform::ColumnPermute,
            "phase" | "phase_randomize" => NullTransform::PhaseRandomize,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown transform={other:?}; use permute|phase"
                )));
            }
        };
        let (series, variables) = series_from_batch(&batch)?;
        let pcmci = pcmci_from_params(max_lag, alpha, fdr, &ci)?;
        let checker = FalsePositiveCheck::new(pcmci, null, replicates);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = py_execution_context(seed, threads);
        let report = checker.run(&series, &variables, &mut ws, &ctx).map_err(py_validate)?;
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item(
                "method",
                match report.method {
                    NullTransform::ColumnPermute => "permute",
                    NullTransform::PhaseRandomize => "phase",
                },
            )?;
            d.set_item("replicates", report.replicates)?;
            d.set_item("mean_edge_count", report.mean_edge_count)?;
            d.set_item("empirical_fpr", report.empirical_fpr)?;
            d.set_item("passed", report.passed)?;
            Ok(d.unbind())
        })
    })
}

/// Alpha-threshold sensitivity grid for PCMCI.
#[pyfunction]
#[pyo3(signature = (
    names, columns, alphas, *, max_lag=1, fdr=false, ci="parcorr", seed=1, threads=1
))]
fn validate_pcmci_alpha_sensitivity(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    alphas: Vec<f64>,
    max_lag: u32,
    fdr: bool,
    ci: &str,
    seed: u64,
    threads: u32,
) -> PyResult<Py<PyDict>> {
    let batch = columns_to_batch(&names, &columns)?;
    let ci = ci.to_string();
    drop(columns);
    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let base_alpha = alphas.first().copied().unwrap_or(0.05);
        let pcmci = pcmci_from_params(max_lag, base_alpha, fdr, &ci)?;
        let checker = AlphaThresholdSensitivity::new(pcmci, alphas);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = py_execution_context(seed, threads);
        let report = checker.run(&series, &variables, &mut ws, &ctx).map_err(py_validate)?;
        Python::attach(|py| discovery_stability_dict(py, &names, &report))
    })
}

/// Max-lag window sensitivity grid for PCMCI.
#[pyfunction]
#[pyo3(signature = (
    names, columns, max_lags, *, alpha=0.05, fdr=false, ci="parcorr", seed=1, threads=1
))]
fn validate_pcmci_lag_sensitivity(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lags: Vec<u32>,
    alpha: f64,
    fdr: bool,
    ci: &str,
    seed: u64,
    threads: u32,
) -> PyResult<Py<PyDict>> {
    let batch = columns_to_batch(&names, &columns)?;
    let ci = ci.to_string();
    drop(columns);
    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let base_lag = max_lags.first().copied().unwrap_or(1);
        let pcmci = pcmci_from_params(base_lag, alpha, fdr, &ci)?;
        let checker = LagWindowSensitivity::new(pcmci, max_lags);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = py_execution_context(seed, threads);
        let report = checker.run(&series, &variables, &mut ws, &ctx).map_err(py_validate)?;
        Python::attach(|py| discovery_stability_dict(py, &names, &report))
    })
}

/// CI-test sensitivity grid for PCMCI.
#[pyfunction]
#[pyo3(signature = (
    names, columns, ci_names, *, max_lag=1, alpha=0.05, fdr=false, seed=1, threads=1
))]
fn validate_pcmci_ci_sensitivity(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    ci_names: Vec<String>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    threads: u32,
) -> PyResult<Py<PyDict>> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let base_ci = ci_names.first().map_or("parcorr", String::as_str);
        let pcmci = pcmci_from_params(max_lag, alpha, fdr, base_ci)?;
        let names_arc: Arc<[Arc<str>]> =
            Arc::from(ci_names.into_iter().map(Arc::<str>::from).collect::<Vec<_>>());
        let checker = CiTestSensitivity::new(pcmci, names_arc);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = py_execution_context(seed, threads);
        let report = checker.run(&series, &variables, &mut ws, &ctx).map_err(py_validate)?;
        Python::attach(|py| discovery_stability_dict(py, &names, &report))
    })
}

/// PCMCI+ contemporaneous orientation stability under block bootstrap.
#[pyfunction]
#[pyo3(signature = (
    names, columns, *, max_lag=1, alpha=0.05, fdr=false, ci="parcorr",
    replicates=20, block_size=20, seed=1, threads=1
))]
fn validate_pcmci_plus_orientation(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    ci: &str,
    replicates: u32,
    block_size: usize,
    seed: u64,
    threads: u32,
) -> PyResult<Py<PyDict>> {
    let batch = columns_to_batch(&names, &columns)?;
    let ci = ci.to_string();
    drop(columns);
    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let pcmci_plus = pcmci_plus_from_params(max_lag, alpha, fdr, &ci)?;
        let checker = OrientationStability { pcmci_plus, replicates, block_size };
        let mut ws = DiscoveryWorkspace::default();
        let ctx = py_execution_context(seed, threads);
        let report = checker.run(&series, &variables, &mut ws, &ctx).map_err(py_validate)?;
        Python::attach(|py| {
            let d = PyDict::new(py);
            let mut directed = Vec::with_capacity(report.directed.len());
            for ls in report.directed.iter() {
                let entry = PyDict::new(py);
                entry.set_item("link", link_dict(py, &names, ls.link)?)?;
                entry.set_item("frequency", ls.frequency)?;
                directed.push(entry);
            }
            let mut undirected = Vec::with_capacity(report.undirected.len());
            for u in report.undirected.iter() {
                let entry = PyDict::new(py);
                let a = names
                    .get(u.a.as_usize())
                    .cloned()
                    .unwrap_or_else(|| format!("var{}", u.a.raw()));
                let b = names
                    .get(u.b.as_usize())
                    .cloned()
                    .unwrap_or_else(|| format!("var{}", u.b.raw()));
                entry.set_item("a", a)?;
                entry.set_item("b", b)?;
                entry.set_item("frequency", u.frequency)?;
                undirected.push(entry);
            }
            d.set_item("directed", directed)?;
            d.set_item("undirected", undirected)?;
            d.set_item("conflict_rate", report.conflict_rate)?;
            d.set_item("replicates", report.replicates)?;
            d.set_item("block_size", report.block_size)?;
            Ok(d.unbind())
        })
    })
}

/// Synthetic-null FPR calibration for PCMCI (independent Gaussian noise).
#[pyfunction]
#[pyo3(signature = (
    *, max_lag=1, alpha=0.05, fdr=false, ci="parcorr",
    n_sim=20, n_obs=100, n_vars=3, seed=1, threads=1
))]
fn validate_synthetic_null_calibration(
    py: Python<'_>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    ci: &str,
    n_sim: u32,
    n_obs: usize,
    n_vars: usize,
    seed: u64,
    threads: u32,
) -> PyResult<Py<PyDict>> {
    let ci = ci.to_string();
    detach_catch(py, move || {
        let pcmci = pcmci_from_params(max_lag, alpha, fdr, &ci)?;
        let checker = SyntheticNullCalibration::new(pcmci, alpha, n_sim, n_obs, n_vars);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = py_execution_context(seed, threads);
        let report = checker.run(&mut ws, &ctx).map_err(py_validate)?;
        Python::attach(|py| {
            let d = PyDict::new(py);
            d.set_item("alpha", report.alpha)?;
            d.set_item("n_sim", report.n_sim)?;
            d.set_item("empirical_fpr", report.empirical_fpr)?;
            d.set_item("se", report.se)?;
            d.set_item("within_band", report.within_band)?;
            d.set_item("band_tol", report.band_tol)?;
            Ok(d.unbind())
        })
    })
}

fn multi_env_from_batches(
    names: &[String],
    env_batches: &[RecordBatch],
) -> PyResult<MultiEnvironmentData> {
    let mut series_list = Vec::with_capacity(env_batches.len());
    for batch in env_batches {
        let (series, _) = series_from_batch(batch)?;
        series_list.push(series);
    }
    let _ = names;
    MultiEnvironmentData::try_new(Arc::from(series_list)).map_err(py_err)
}

/// Environment-holdout link agreement under J-PCMCI+.
#[pyfunction]
#[pyo3(signature = (
    names, env_columns, *, max_lag=1, alpha=0.05, fdr=false, ci="parcorr",
    n_discovery=1, seed=1, threads=1
))]
fn validate_environment_holdout(
    py: Python<'_>,
    names: Vec<String>,
    env_columns: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    ci: &str,
    n_discovery: usize,
    seed: u64,
    threads: u32,
) -> PyResult<Py<PyDict>> {
    if env_columns.is_empty() {
        return Err(PyValueError::new_err("env_columns needs ≥1 environment"));
    }
    let mut batches = Vec::with_capacity(env_columns.len());
    for cols in &env_columns {
        batches.push(columns_to_batch(&names, cols)?);
    }
    let ci = ci.to_string();
    drop(env_columns);
    detach_catch(py, move || {
        let multi = multi_env_from_batches(&names, &batches)?;
        let split = EnvHoldoutSplit::try_prefix(multi.env_count(), n_discovery).map_err(py_err)?;
        let ci_impl = resolve_ci(&ci, None).map_err(py_err)?;
        let jpcmci = JpcmciPlus::new()
            .with_fdr(fdr)
            .with_constraints(jpcmci_constraints(
                max_lag,
                alpha,
                MultiDatasetConstraints::default(),
            ))
            .with_ci(ci_impl);
        let checker = EnvironmentHoldout::new(jpcmci, split);
        let variables: Vec<VariableId> =
            (0..names.len() as u32).map(VariableId::from_raw).collect();
        let mut ws = DiscoveryWorkspace::default();
        let ctx = py_execution_context(seed, threads);
        let report = checker.run(&multi, &variables, &mut ws, &ctx).map_err(py_validate)?;
        Python::attach(|py| {
            let d = PyDict::new(py);
            let disc: Vec<_> = report
                .discovery_links
                .iter()
                .map(|l| link_dict(py, &names, *l))
                .collect::<PyResult<_>>()?;
            let hold: Vec<_> = report
                .holdout_links
                .iter()
                .map(|l| link_dict(py, &names, *l))
                .collect::<PyResult<_>>()?;
            d.set_item("discovery_links", disc)?;
            d.set_item("holdout_links", hold)?;
            d.set_item("shared_frequency", report.shared_frequency)?;
            d.set_item("jaccard", report.jaccard)?;
            Ok(d.unbind())
        })
    })
}

/// Per-regime block-bootstrap stability under fixed RPCMCI labels.
#[pyfunction]
#[pyo3(signature = (
    names, columns, regimes, *, max_lag=1, alpha=0.05, fdr=false, ci="parcorr",
    replicates=10, block_size=20, seed=1, threads=1
))]
fn validate_regime_stability(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    regimes: Vec<u32>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    ci: &str,
    replicates: u32,
    block_size: usize,
    seed: u64,
    threads: u32,
) -> PyResult<Py<PyDict>> {
    let batch = columns_to_batch(&names, &columns)?;
    let ci = ci.to_string();
    drop(columns);
    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        if regimes.len() != series.row_count() {
            return Err(PyValueError::new_err("regimes length must match series length"));
        }
        let assignment = RegimeAssignment::try_new(
            regimes.into_iter().map(RegimeId::from_raw).collect::<Vec<_>>(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let pcmci_plus = pcmci_plus_from_params(max_lag, alpha, fdr, &ci)?;
        let rpcmci = Rpcmci::new().with_pcmci_plus(pcmci_plus).with_alternating_iters(0);
        let mut checker = RegimeStability::new(rpcmci, assignment);
        checker.replicates = replicates;
        checker.block_size = block_size;
        let mut ws = DiscoveryWorkspace::default();
        let ctx = py_execution_context(seed, threads);
        let report = checker.run(&series, &variables, &mut ws, &ctx).map_err(py_validate)?;
        Python::attach(|py| {
            let d = PyDict::new(py);
            let per = PyDict::new(py);
            for (rid, sub) in &report.per_regime {
                per.set_item(rid.raw(), discovery_stability_dict(py, &names, sub)?)?;
            }
            d.set_item("per_regime", per)?;
            d.set_item("replicates", report.replicates)?;
            d.set_item("block_size", report.block_size)?;
            Ok(d.unbind())
        })
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_pcmci_block_bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(validate_pcmci_false_positive, m)?)?;
    m.add_function(wrap_pyfunction!(validate_pcmci_alpha_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(validate_pcmci_lag_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(validate_pcmci_ci_sensitivity, m)?)?;
    m.add_function(wrap_pyfunction!(validate_pcmci_plus_orientation, m)?)?;
    m.add_function(wrap_pyfunction!(validate_synthetic_null_calibration, m)?)?;
    m.add_function(wrap_pyfunction!(validate_environment_holdout, m)?)?;
    m.add_function(wrap_pyfunction!(validate_regime_stability, m)?)?;
    Ok(())
}
