//! Bayesian graph-posterior discovery bindings.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::discovery::{
    BayesianDiscoverParams, CiSoftWeight, GraphMcmcSchedule, GraphPosterior as RustGraphPosterior,
    StaticDiscoverParams, discover_ci_screened_posterior as facade_discover_ci_screened,
    discover_dbn_posterior as facade_discover_dbn,
    discover_exact_dag_posterior as facade_discover_exact,
    discover_order_mcmc as facade_discover_order_mcmc,
    discover_structure_mcmc as facade_discover_structure_mcmc,
};
use antecedent::discovery_defaults::resolve_ci;
use antecedent_discovery::has_edge;
use antecedent_prob::InferenceDiagnostics;
use antecedent_stats::{FdrAdjustment, PartialCorrelation};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

use crate::{
    columns_to_batch, detach_catch, py_err, py_execution_context, py_msg, series_from_batch,
    tabular_from_batch,
};

/// Columnar graph posterior returned by Bayesian discovery engines.
#[pyclass(name = "GraphPosterior")]
pub struct PyGraphPosterior {
    #[pyo3(get)]
    names: Vec<String>,
    #[pyo3(get)]
    n_vars: usize,
    #[pyo3(get)]
    n_graphs: usize,
    #[pyo3(get)]
    weights: Vec<f64>,
    #[pyo3(get)]
    adjacency: Vec<u64>,
    #[pyo3(get)]
    edge_marginals: Vec<f64>,
    #[pyo3(get)]
    orientation_marginals: Vec<f64>,
    #[pyo3(get)]
    ess: f64,
    #[pyo3(get)]
    rejected_invalid: u64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    lagged_edge_marginals: Option<Vec<f64>>,
    #[pyo3(get)]
    max_lag: Option<u32>,
    #[pyo3(get)]
    lag_masks: Option<Vec<u64>>,
}

impl PyGraphPosterior {
    fn from_rust(names: Vec<String>, post: RustGraphPosterior) -> Self {
        Self {
            names,
            n_vars: post.n_vars,
            n_graphs: post.n_graphs,
            weights: post.weights.as_ref().to_vec(),
            adjacency: post.adjacency.as_ref().to_vec(),
            edge_marginals: post.edge_marginals.as_ref().to_vec(),
            orientation_marginals: post.orientation_marginals.as_ref().to_vec(),
            ess: post.ess,
            rejected_invalid: post.rejected_invalid,
            converged: post.diagnostics.converged,
            lagged_edge_marginals: post.lagged_edge_marginals.as_ref().map(|v| v.as_ref().to_vec()),
            max_lag: post.max_lag,
            lag_masks: post.lag_masks.as_ref().map(|v| v.as_ref().to_vec()),
        }
    }

    pub(crate) fn to_rust(&self) -> PyResult<RustGraphPosterior> {
        let mut post = RustGraphPosterior::new(
            self.n_vars,
            self.weights.clone(),
            self.adjacency.clone(),
            self.edge_marginals.clone(),
            self.orientation_marginals.clone(),
            self.ess,
            InferenceDiagnostics::analytic("from_atoms"),
            self.rejected_invalid,
        )
        .map_err(py_msg)?;
        if let (Some(lagged), Some(max_lag)) = (&self.lagged_edge_marginals, self.max_lag) {
            post = post.with_lagged_marginals(max_lag, lagged.clone()).map_err(py_msg)?;
        }
        if let Some(lag_masks) = &self.lag_masks {
            post = post.with_lag_masks(lag_masks.clone()).map_err(py_msg)?;
        }
        Ok(post.with_algorithm("from_atoms"))
    }
}

#[pymethods]
impl PyGraphPosterior {
    /// Summary dict compatible with weighted-graph envelope consumers.
    fn to_weighted_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("n_samples", self.n_graphs)?;
        d.set_item("weights", self.weights.clone())?;
        d.set_item("graph_keys", self.adjacency.clone())?;
        d.set_item("edge_marginals", self.edge_marginals.clone())?;
        d.set_item("orientation_marginals", self.orientation_marginals.clone())?;
        d.set_item("ess", self.ess)?;
        d.set_item("names", self.names.clone())?;
        Ok(d)
    }

    /// Edge marginal as an `n_vars × n_vars` nested list (row-major from→to).
    fn edge_marginal_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let n = self.n_vars;
        let rows = PyList::empty(py);
        for i in 0..n {
            let row = PyList::empty(py);
            for j in 0..n {
                row.append(self.edge_marginals[i * n + j])?;
            }
            rows.append(row)?;
        }
        Ok(rows)
    }

    fn __repr__(&self) -> String {
        format!(
            "GraphPosterior(n_vars={}, n_graphs={}, ess={:.3}, converged={})",
            self.n_vars, self.n_graphs, self.ess, self.converged
        )
    }

    /// Build a posterior from explicit atoms (known-truth fixtures / replay).
    #[classmethod]
    #[pyo3(signature = (
        names,
        weights,
        adjacency,
        *,
        edge_marginals=None,
        orientation_marginals=None,
        lagged_edge_marginals=None,
        lag_masks=None,
        max_lag=None,
        ess=None,
    ))]
    fn from_atoms(
        _cls: &Bound<'_, pyo3::types::PyType>,
        names: Vec<String>,
        weights: Vec<f64>,
        adjacency: Vec<u64>,
        edge_marginals: Option<Vec<f64>>,
        orientation_marginals: Option<Vec<f64>>,
        lagged_edge_marginals: Option<Vec<f64>>,
        lag_masks: Option<Vec<u64>>,
        max_lag: Option<u32>,
        ess: Option<f64>,
    ) -> PyResult<Self> {
        let n_vars = names.len();
        if adjacency.len() != weights.len() {
            return Err(PyValueError::new_err("adjacency/weights length mismatch"));
        }
        let cell = n_vars.saturating_mul(n_vars);
        let edge_marginals = edge_marginals.unwrap_or_else(|| {
            let mut marginals = vec![0.0; cell];
            for (weight, mask) in weights.iter().zip(&adjacency) {
                for from in 0..n_vars {
                    for to in 0..n_vars {
                        if from != to && has_edge(*mask, n_vars, from, to) {
                            marginals[from * n_vars + to] += *weight;
                        }
                    }
                }
            }
            marginals
        });
        let orientation_marginals = orientation_marginals.unwrap_or_else(|| edge_marginals.clone());
        let ess = ess.unwrap_or_else(|| 1.0 / weights.iter().map(|w| w * w).sum::<f64>());
        let mut post = RustGraphPosterior::new(
            n_vars,
            weights,
            adjacency,
            edge_marginals,
            orientation_marginals,
            ess,
            InferenceDiagnostics::analytic("from_atoms"),
            0,
        )
        .map_err(py_msg)?;
        match (lagged_edge_marginals, max_lag) {
            (Some(lagged), Some(lag)) => {
                post = post.with_lagged_marginals(lag, lagged).map_err(py_msg)?;
            }
            (None, None) => {}
            _ => {
                return Err(PyValueError::new_err(
                    "lagged_edge_marginals and max_lag must be supplied together",
                ));
            }
        }
        if let Some(lag_masks) = lag_masks {
            post = post.with_lag_masks(lag_masks).map_err(py_msg)?;
        }
        Ok(Self::from_rust(names, post.with_algorithm("from_atoms")))
    }
}

fn bayesian_params() -> BayesianDiscoverParams {
    BayesianDiscoverParams::default()
}

fn schedule_from_args(n_chains: u32, n_warmup: u32, n_draws: u32, thin: u32) -> GraphMcmcSchedule {
    GraphMcmcSchedule { n_chains, n_warmup, n_draws, thin }
}

fn parse_soft_weight(name: &str) -> PyResult<CiSoftWeight> {
    match name.to_ascii_lowercase().as_str() {
        "none" | "" => Ok(CiSoftWeight::None),
        "bayes_factor" | "bayesfactor" => Ok(CiSoftWeight::BayesFactor),
        "posterior_dependence" | "posteriordependence" => Ok(CiSoftWeight::PosteriorDependence),
        other => Err(PyValueError::new_err(format!(
            "unknown soft_weight {other:?}; expected none|bayes_factor|posterior_dependence"
        ))),
    }
}

/// Exact DAG posterior enumeration (`n ≤ 6` hard limit, Gaussian BIC).
///
/// Larger graphs: use `discover_order_mcmc`, `discover_structure_mcmc`, or
/// `discover_ci_screened_posterior`.
#[pyfunction]
#[pyo3(signature = (names, columns, *, seed=1, threads=1))]
fn discover_exact_dag_posterior(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    seed: u64,
    threads: u32,
) -> PyResult<PyGraphPosterior> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, threads);
        let post =
            facade_discover_exact(&data, &variables, &bayesian_params(), &ctx).map_err(py_err)?;
        Ok(PyGraphPosterior::from_rust(names, post))
    })
}

/// Order MCMC DAG posterior (Gaussian BIC).
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    *,
    n_chains=4,
    n_warmup=500,
    n_draws=1000,
    thin=1,
    require_diagnostics_gate=true,
    seed=1,
    threads=1
))]
fn discover_order_mcmc(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    n_chains: u32,
    n_warmup: u32,
    n_draws: u32,
    thin: u32,
    require_diagnostics_gate: bool,
    seed: u64,
    threads: u32,
) -> PyResult<PyGraphPosterior> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, threads);
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, thin);
        let post = facade_discover_order_mcmc(
            &data,
            &variables,
            &bayesian_params(),
            &schedule,
            require_diagnostics_gate,
            &ctx,
        )
        .map_err(py_err)?;
        Ok(PyGraphPosterior::from_rust(names, post))
    })
}

/// Structure MCMC DAG posterior (Gaussian BIC).
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    *,
    n_chains=4,
    n_warmup=500,
    n_draws=1000,
    thin=1,
    seed=1,
    threads=1
))]
fn discover_structure_mcmc(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    n_chains: u32,
    n_warmup: u32,
    n_draws: u32,
    thin: u32,
    seed: u64,
    threads: u32,
) -> PyResult<PyGraphPosterior> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, threads);
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, thin);
        let post =
            facade_discover_structure_mcmc(&data, &variables, &bayesian_params(), &schedule, &ctx)
                .map_err(py_err)?;
        Ok(PyGraphPosterior::from_rust(names, post))
    })
}

/// CI-screened candidate-edge posterior (PC skeleton → structure MCMC).
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    *,
    alpha=0.05,
    fdr=true,
    ci=None,
    max_cond_size=2,
    soft_weight="none",
    n_chains=2,
    n_warmup=300,
    n_draws=600,
    thin=1,
    seed=1,
    threads=1
))]
fn discover_ci_screened_posterior(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    alpha: f64,
    fdr: bool,
    ci: Option<String>,
    max_cond_size: usize,
    soft_weight: &str,
    n_chains: u32,
    n_warmup: u32,
    n_draws: u32,
    thin: u32,
    seed: u64,
    threads: u32,
) -> PyResult<PyGraphPosterior> {
    let soft = parse_soft_weight(soft_weight)?;
    let ci_name = ci.unwrap_or_else(|| "parcorr".to_string());
    let ci_impl = if ci_name.eq_ignore_ascii_case("parcorr") {
        std::sync::Arc::new(PartialCorrelation)
            as std::sync::Arc<dyn antecedent_stats::ConditionalIndependence + Send + Sync>
    } else {
        resolve_ci(&ci_name, None).map_err(py_err)?
    };
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, threads);
        let screen = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc: false,
            max_subset: None,
        };
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, thin);
        let post = facade_discover_ci_screened(
            &data,
            &variables,
            &bayesian_params(),
            &screen,
            &schedule,
            soft,
            &ctx,
        )
        .map_err(py_err)?;
        Ok(PyGraphPosterior::from_rust(names, post))
    })
}

/// Bounded-lag DBN template posterior from time-series columns (Gaussian BIC).
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    *,
    max_lag=1,
    force_mcmc=false,
    n_chains=2,
    n_warmup=200,
    n_draws=400,
    thin=1,
    seed=1,
    threads=1
))]
fn discover_dbn_posterior(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    force_mcmc: bool,
    n_chains: u32,
    n_warmup: u32,
    n_draws: u32,
    thin: u32,
    seed: u64,
    threads: u32,
) -> PyResult<PyGraphPosterior> {
    let _ = thin; // DBN engine schedule has no thin; kept for API symmetry.
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let ctx = py_execution_context(seed, threads);
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, 1);
        let post = facade_discover_dbn(
            &series,
            &variables,
            &bayesian_params(),
            max_lag,
            force_mcmc,
            &schedule,
            &ctx,
        )
        .map_err(py_err)?;
        Ok(PyGraphPosterior::from_rust(names, post))
    })
}

/// Register Bayesian discovery types and functions on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphPosterior>()?;
    m.add_function(wrap_pyfunction!(discover_exact_dag_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(discover_order_mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(discover_structure_mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(discover_ci_screened_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(discover_dbn_posterior, m)?)?;
    Ok(())
}
