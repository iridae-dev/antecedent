//! Bayesian graph-posterior discovery bindings.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::discovery::{
    BayesianDiscoverParams, CiSoftWeight, GraphMcmcSchedule, GraphPosterior as RustGraphPosterior,
    SamplingSummary, StaticDiscoverParams,
};
use antecedent::discovery_defaults::resolve_ci;
use antecedent::{DiscoveredStructure, DiscoverySpec};
use antecedent_discovery::adjacency_mask::{
    adjacency_mask_from_admg, adjacency_mask_from_cpdag, adjacency_masks_from_pag, has_edge,
    mask_is_dag,
};
use antecedent_discovery::{
    AdjacencyMask, GraphPosteriorAtomKind, MAX_POSTERIOR_NODES, n_directed_edges,
};
use antecedent_graph::Cpdag as RustCpdag;
use antecedent_prob::InferenceDiagnostics;
use antecedent_stats::{FdrAdjustment, PartialCorrelation};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule};

use crate::compiled_route_api::{DiscoverInput, RouteRecord};
use crate::discovery_api::compiled_search;
use crate::{
    columns_to_batch, detach_catch, py_err, py_execution_context, py_msg, series_from_batch,
    tabular_from_batch,
};

#[path = "bayesian_temporal_atoms.rs"]
mod temporal_atoms;

/// The posterior a compiled structure-posterior search returned.
fn posterior_of(structure: DiscoveredStructure) -> Option<RustGraphPosterior> {
    match structure {
        DiscoveredStructure::Posterior(posterior) => Some(posterior),
        _ => None,
    }
}

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
    adjacency: Vec<AdjacencyMask>,
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
    /// Per-atom lagged latent-confounder masks for temporal ADMGs.
    #[pyo3(get)]
    lag_bidirected_masks: Option<Vec<u64>>,
    /// Structure-learning algorithm that produced the atoms (`None` if untagged).
    #[pyo3(get)]
    algorithm: Option<String>,
    /// Class of every atom: `Dag`, `Cpdag`, `Pag`, or `Admg`.
    #[pyo3(get)]
    atom_kind: String,
    /// Optional PAG circle-endpoint masks, same length as `adjacency`.
    mark_masks: Option<Vec<AdjacencyMask>>,
    /// Monte-Carlo standard error of each edge marginal (`None` for exact posteriors).
    #[pyo3(get)]
    edge_mcse: Option<Vec<f64>>,
    /// Split R-hat of each edge indicator across chains (`None` for exact posteriors).
    #[pyo3(get)]
    edge_rhat: Option<Vec<f64>>,
    /// Bulk effective sample size of each edge indicator across chains (`None` for exact
    /// posteriors).
    #[pyo3(get)]
    edge_ess: Option<Vec<f64>>,
    /// Backend identifier carried from the producing diagnostics.
    backend_id: String,
    /// The compiled discovery route this posterior came from (`None` for supplied atoms).
    route: Option<RouteRecord>,
}

/// Width of the packed lag masks.
const MASK_BITS: usize = u64::BITS as usize;

/// Python `int` holding the packed edge bits of `mask` (bit layout `edge_bit`).
fn mask_to_pyint(py: Python<'_>, mask: AdjacencyMask) -> PyResult<Bound<'_, PyAny>> {
    py.import("builtins")?
        .getattr("int")?
        .call_method1("from_bytes", (PyBytes::new(py, &mask.to_le_bytes()), "little"))
}

/// Packed edge bits from a non-negative Python `int` below `2**256`.
fn mask_from_pyint(value: &Bound<'_, PyAny>) -> PyResult<AdjacencyMask> {
    let refuse = || {
        crate::value_err(
            "adjacency masks must be non-negative integers below 2**256 (at most 16 variables)",
        )
    };
    let bytes: Vec<u8> =
        value.call_method1("to_bytes", (32, "little")).map_err(|_| refuse())?.extract()?;
    let bytes: [u8; 32] = bytes.try_into().map_err(|_| refuse())?;
    Ok(AdjacencyMask::from_le_bytes(bytes))
}

/// Every edge bit of an `n_vars`-variable graph.
fn full_edge_set(n_vars: usize) -> AdjacencyMask {
    (0..n_directed_edges(n_vars)).fold(AdjacencyMask::EMPTY, |set, bit| {
        set.with_bit(u32::try_from(bit).unwrap_or(u32::MAX), true)
    })
}

/// Arguments of [`PyGraphPosterior::from_atom_inputs`]: explicit posterior atoms.
struct AtomInputs {
    names: Vec<String>,
    weights: Vec<f64>,
    adjacency: Vec<AdjacencyMask>,
    edge_marginals: Option<Vec<f64>>,
    orientation_marginals: Option<Vec<f64>>,
    lagged_edge_marginals: Option<Vec<f64>>,
    lag_masks: Option<Vec<u64>>,
    lag_bidirected_masks: Option<Vec<u64>>,
    max_lag: Option<u32>,
    ess: Option<f64>,
    atom_kind: GraphPosteriorAtomKind,
    mark_masks: Option<Vec<AdjacencyMask>>,
}

impl PyGraphPosterior {
    fn from_rust(names: Vec<String>, post: RustGraphPosterior) -> Self {
        Self {
            names,
            n_vars: post.n_vars,
            n_graphs: post.n_graphs,
            weights: post.weights.as_ref().to_vec(),
            adjacency: post.atoms.as_ref().to_vec(),
            edge_marginals: post.edge_marginals.as_ref().to_vec(),
            orientation_marginals: post.orientation_marginals.as_ref().to_vec(),
            ess: post.ess,
            rejected_invalid: post.rejected_invalid,
            converged: post.diagnostics.converged,
            lagged_edge_marginals: post.lagged_edge_marginals.as_ref().map(|v| v.as_ref().to_vec()),
            max_lag: post.max_lag,
            lag_masks: post.lag_masks.as_ref().map(|v| v.as_ref().to_vec()),
            lag_bidirected_masks: post.lag_bidirected_masks.as_ref().map(|v| v.as_ref().to_vec()),
            algorithm: post.algorithm.as_deref().map(str::to_owned),
            atom_kind: post.atom_kind.as_str().to_owned(),
            mark_masks: post.mark_atoms.as_ref().map(|v| v.as_ref().to_vec()),
            edge_mcse: post.edge_mcse.as_ref().map(|v| v.as_ref().to_vec()),
            edge_rhat: post.edge_rhat.as_ref().map(|v| v.as_ref().to_vec()),
            edge_ess: post.edge_ess.as_ref().map(|v| v.as_ref().to_vec()),
            backend_id: post.diagnostics.backend_id.to_string(),
            route: None,
        }
    }

    fn with_route(mut self, route: RouteRecord) -> Self {
        self.route = Some(route);
        self
    }

    /// Rebuild the engine posterior for a study that consumes this handle.
    ///
    /// Replay keeps the producer's convergence verdict and algorithm tag, so a
    /// non-converged MCMC posterior is not relabelled as analytic truth and the
    /// plan record names the real structure-learning algorithm.
    pub(crate) fn to_rust(&self) -> PyResult<RustGraphPosterior> {
        let mut diagnostics = InferenceDiagnostics::analytic(self.backend_id.as_str());
        diagnostics.converged = self.converged;
        let mut post = RustGraphPosterior::from_atoms(
            self.n_vars,
            self.weights.clone(),
            self.adjacency.clone(),
            self.edge_marginals.clone(),
            self.orientation_marginals.clone(),
            SamplingSummary { ess: self.ess, diagnostics, rejected_invalid: self.rejected_invalid },
        )
        .map_err(py_msg)?;
        if let (Some(lagged), Some(max_lag)) = (&self.lagged_edge_marginals, self.max_lag) {
            post = post.with_lagged_marginals(max_lag, lagged.clone()).map_err(py_msg)?;
        }
        if let Some(lag_masks) = &self.lag_masks {
            post = post.with_lag_masks(lag_masks.clone()).map_err(py_msg)?;
        }
        post = post.with_atom_kind(parse_atom_kind(&self.atom_kind)?);
        if let Some(masks) = &self.lag_bidirected_masks {
            post = post.with_lag_bidirected_masks(masks.clone()).map_err(py_msg)?;
        }
        if let Some(mark_masks) = &self.mark_masks {
            post = post.with_mark_atoms(mark_masks.clone()).map_err(py_msg)?;
        }
        if let Some(mcse) = &self.edge_mcse {
            post = post.with_edge_mcse(mcse.clone()).map_err(py_msg)?;
        }
        if let (Some(rhat), Some(ess)) = (&self.edge_rhat, &self.edge_ess) {
            post = post.with_edge_chain_diagnostics(rhat.clone(), ess.clone()).map_err(py_msg)?;
        }
        Ok(match &self.algorithm {
            Some(algorithm) => post.with_algorithm(algorithm.as_str()),
            None => post,
        })
    }

    /// Refuse a posterior whose variable order does not match the data columns.
    ///
    /// Atom masks are positional, so a reordered table would silently permute the
    /// structure; this mirrors the column-order check applied to supplied graphs.
    pub(crate) fn require_bound_to(&self, names: &[String]) -> PyResult<()> {
        if self.names != names || self.n_vars != names.len() {
            return Err(crate::value_err(
                "GraphPosterior variable names must match data column names and order",
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl PyGraphPosterior {
    /// The compiled discovery route this posterior came from; `None` for supplied atoms.
    #[getter]
    fn route(&self, py: Python<'_>) -> PyResult<crate::compiled_route_api::RouteHandle> {
        crate::compiled_route_api::route_object(py, self.route.as_ref())
    }

    /// The contract the posterior search executed under (`query_kind="Discover"`,
    /// `inference="bayesian"`).
    #[getter]
    fn contract(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.contract()
    }

    /// The claim sealed over the posterior search.
    #[getter]
    fn claim(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.claim()
    }

    /// Accept the structure posterior this search returned, recording where it came from:
    /// the provenance (method, inference, class, query, bounds, data snapshot, lowered
    /// program, claim and review) the acceptance binds to the posterior, and its atom count.
    fn accept_posterior(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.accept_posterior()
    }

    /// Cheap inspection record of the compiled posterior study.
    fn inspect(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.inspect()
    }

    /// Packed adjacency of every atom: bit `from * (n_vars - 1) + (to if to < from else to - 1)`
    /// of the Python `int` is the edge `from -> to`.
    #[getter]
    fn adjacency<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.adjacency.iter().map(|&mask| mask_to_pyint(py, mask)).collect()
    }

    /// Optional PAG circle-endpoint masks, same packing as `adjacency`.
    #[getter]
    fn mark_masks<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        self.mark_masks
            .as_ref()
            .map(|masks| masks.iter().map(|&mask| mask_to_pyint(py, mask)).collect())
            .transpose()
    }

    /// Summary dict compatible with weighted-graph envelope consumers.
    fn to_weighted_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("n_samples", self.n_graphs)?;
        d.set_item("weights", self.weights.clone())?;
        d.set_item("graph_keys", self.adjacency.iter().map(|m| m.key()).collect::<Vec<_>>())?;
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
            "GraphPosterior(n_vars={}, n_graphs={}, atom_kind={}, ess={:.3}, converged={})",
            self.n_vars, self.n_graphs, self.atom_kind, self.ess, self.converged
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
        atom_kind="Dag",
        mark_masks=None,
    ))]
    fn from_atoms(
        _cls: &Bound<'_, pyo3::types::PyType>,
        names: Vec<String>,
        weights: Vec<f64>,
        adjacency: Vec<Bound<'_, PyAny>>,
        edge_marginals: Option<Vec<f64>>,
        orientation_marginals: Option<Vec<f64>>,
        lagged_edge_marginals: Option<Vec<f64>>,
        lag_masks: Option<Vec<u64>>,
        max_lag: Option<u32>,
        ess: Option<f64>,
        atom_kind: &str,
        mark_masks: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<Self> {
        let adjacency = adjacency.iter().map(mask_from_pyint).collect::<PyResult<Vec<_>>>()?;
        let mark_masks = mark_masks
            .map(|masks| masks.iter().map(mask_from_pyint).collect::<PyResult<Vec<_>>>())
            .transpose()?;
        Self::from_atom_inputs(AtomInputs {
            names,
            weights,
            adjacency,
            edge_marginals,
            orientation_marginals,
            lagged_edge_marginals,
            lag_masks,
            lag_bidirected_masks: None,
            max_lag,
            ess,
            atom_kind: parse_atom_kind(atom_kind)?,
            mark_masks,
        })
    }

    /// Pack typed graphs into a posterior. Every atom must share one class.
    #[classmethod]
    #[pyo3(signature = (names, weights, graphs, *, ess=None))]
    fn from_graphs(
        _cls: &Bound<'_, pyo3::types::PyType>,
        names: Vec<String>,
        weights: Vec<f64>,
        graphs: Vec<Bound<'_, PyAny>>,
        ess: Option<f64>,
    ) -> PyResult<Self> {
        if graphs.len() != weights.len() {
            return Err(crate::value_err("graphs/weights length mismatch"));
        }
        if graphs.is_empty() {
            return Err(crate::value_err("from_graphs requires at least one atom"));
        }
        if let Some(temporal) = temporal_atoms::from_graphs(&names, &weights, &graphs, ess) {
            return temporal;
        }
        let (atom_kind, adjacency, mark_masks) = pack_static_atoms(&names, &graphs)?;
        Self::from_atom_inputs(AtomInputs {
            names,
            weights,
            adjacency,
            edge_marginals: None,
            orientation_marginals: None,
            lagged_edge_marginals: None,
            lag_masks: None,
            lag_bidirected_masks: None,
            max_lag: None,
            ess,
            atom_kind,
            mark_masks,
        })
    }
}

impl PyGraphPosterior {
    /// Validate explicit atoms and build the posterior handle.
    fn from_atom_inputs(inputs: AtomInputs) -> PyResult<Self> {
        let AtomInputs {
            names,
            weights,
            adjacency,
            edge_marginals,
            orientation_marginals,
            lagged_edge_marginals,
            lag_masks,
            lag_bidirected_masks,
            max_lag,
            ess,
            atom_kind: kind,
            mark_masks,
        } = inputs;
        let n_vars = names.len();
        if adjacency.len() != weights.len() {
            return Err(crate::value_err("adjacency/weights length mismatch"));
        }
        if n_vars == 0 {
            return Err(crate::value_err("from_atoms requires at least one variable"));
        }
        if n_vars > MAX_POSTERIOR_NODES {
            return Err(crate::value_err(format!(
                "from_atoms supports at most {MAX_POSTERIOR_NODES} variables"
            )));
        }
        let all_edges = full_edge_set(n_vars);
        for (index, mask) in adjacency.iter().enumerate() {
            if !all_edges.contains_all(*mask) {
                return Err(crate::value_err(format!(
                    "adjacency atom {index} sets edge bits beyond {n_vars} variables"
                )));
            }
            if kind == GraphPosteriorAtomKind::Dag && !mask_is_dag(*mask, n_vars) {
                return Err(crate::value_err(format!("adjacency atom {index} is not a DAG")));
            }
        }
        // `ess` and the default marginals below assume normalized mass; the
        // engine constructor only rejects non-positive totals.
        let total: f64 = weights.iter().sum();
        if total.is_finite() && (total - 1.0).abs() > 1e-9 {
            return Err(crate::value_err(format!(
                "posterior weights must sum to 1 (got {total}); normalize the atoms first"
            )));
        }
        let cell = n_vars.saturating_mul(n_vars);
        if lag_masks.is_some() && max_lag.is_none() {
            return Err(crate::value_err("lag_masks require max_lag"));
        }
        if let (Some(masks), Some(lag)) = (&lag_masks, max_lag) {
            let lag_bits = usize::try_from(lag).map_err(py_msg)?.saturating_mul(cell);
            if lag_bits > MASK_BITS {
                return Err(crate::value_err("lag_masks require max_lag * n_vars^2 <= 64 bits"));
            }
            for (index, mask) in masks.iter().enumerate() {
                if lag_bits < MASK_BITS && (mask >> lag_bits) != 0 {
                    return Err(crate::value_err(format!(
                        "lag mask atom {index} sets bits beyond max_lag={lag} and {n_vars} variables"
                    )));
                }
            }
        }
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
        let mut post = RustGraphPosterior::from_atoms(
            n_vars,
            weights,
            adjacency,
            edge_marginals,
            orientation_marginals,
            SamplingSummary {
                ess,
                diagnostics: InferenceDiagnostics::analytic("from_atoms"),
                rejected_invalid: 0,
            },
        )
        .map_err(py_msg)?;
        match (lagged_edge_marginals, max_lag) {
            (Some(lagged), Some(lag)) => {
                post = post.with_lagged_marginals(lag, lagged).map_err(py_msg)?;
            }
            (None, None) => {}
            _ => {
                return Err(crate::value_err(
                    "lagged_edge_marginals and max_lag must be supplied together",
                ));
            }
        }
        if let Some(lag_masks) = lag_masks {
            post = post.with_lag_masks(lag_masks).map_err(py_msg)?;
        }
        post = post.with_atom_kind(kind);
        if let Some(masks) = lag_bidirected_masks {
            post = post.with_lag_bidirected_masks(masks).map_err(py_msg)?;
        }
        if let Some(mark_masks) = mark_masks {
            post = post.with_mark_atoms(mark_masks).map_err(py_msg)?;
        }
        Ok(Self::from_rust(names, post.with_algorithm("from_atoms")))
    }
}

fn parse_atom_kind(name: &str) -> PyResult<GraphPosteriorAtomKind> {
    match name {
        "Dag" | "dag" => Ok(GraphPosteriorAtomKind::Dag),
        "Cpdag" | "cpdag" => Ok(GraphPosteriorAtomKind::Cpdag),
        "Pag" | "pag" => Ok(GraphPosteriorAtomKind::Pag),
        "Admg" | "admg" => Ok(GraphPosteriorAtomKind::Admg),
        "TemporalAdmg" | "temporal_admg" => Ok(GraphPosteriorAtomKind::TemporalAdmg),
        other => Err(crate::value_err(format!(
            "atom_kind must be Dag, Cpdag, Pag, or Admg; got {other:?}"
        ))),
    }
}

fn require_graph_names(graph_names: &[String], names: &[String], kind: &str) -> PyResult<()> {
    if graph_names != names {
        return Err(crate::value_err(format!(
            "{kind} variable names must match GraphPosterior names and order"
        )));
    }
    Ok(())
}

fn pack_static_atoms(
    names: &[String],
    graphs: &[Bound<'_, PyAny>],
) -> PyResult<(GraphPosteriorAtomKind, Vec<AdjacencyMask>, Option<Vec<AdjacencyMask>>)> {
    let first = &graphs[0];
    if first.extract::<crate::graphs::Dag>().is_ok() {
        let mut adjacency = Vec::with_capacity(graphs.len());
        for graph in graphs {
            let dag = graph.extract::<crate::graphs::Dag>()?;
            require_graph_names(&dag.names, names, "Dag")?;
            adjacency
                .push(adjacency_mask_from_cpdag(&RustCpdag::from_dag(&dag.dag)).map_err(py_msg)?);
        }
        return Ok((GraphPosteriorAtomKind::Dag, adjacency, None));
    }
    if first.extract::<crate::graphs::Cpdag>().is_ok() {
        let mut adjacency = Vec::with_capacity(graphs.len());
        for graph in graphs {
            let cpdag = graph.extract::<crate::graphs::Cpdag>()?;
            require_graph_names(&cpdag.names, names, "Cpdag")?;
            adjacency.push(adjacency_mask_from_cpdag(&cpdag.cpdag).map_err(py_msg)?);
        }
        return Ok((GraphPosteriorAtomKind::Cpdag, adjacency, None));
    }
    if first.extract::<crate::graphs::Pag>().is_ok() {
        let mut adjacency = Vec::with_capacity(graphs.len());
        let mut marks = Vec::with_capacity(graphs.len());
        for graph in graphs {
            let pag = graph.extract::<crate::graphs::Pag>()?;
            require_graph_names(&pag.names, names, "Pag")?;
            let (adj, mark) = adjacency_masks_from_pag(&pag.pag).map_err(py_msg)?;
            adjacency.push(adj);
            marks.push(mark);
        }
        return Ok((GraphPosteriorAtomKind::Pag, adjacency, Some(marks)));
    }
    if first.extract::<crate::graphs::Admg>().is_ok() {
        let mut adjacency = Vec::with_capacity(graphs.len());
        for graph in graphs {
            let admg = graph.extract::<crate::graphs::Admg>()?;
            require_graph_names(&admg.names, names, "Admg")?;
            adjacency.push(adjacency_mask_from_admg(&admg.admg).map_err(py_msg)?);
        }
        return Ok((GraphPosteriorAtomKind::Admg, adjacency, None));
    }
    Err(crate::value_err(
        "from_graphs accepts Dag, Cpdag, Pag, Admg, TemporalDag, TemporalCpdag, TemporalPag, or TemporalAdmg atoms",
    ))
}

fn bayesian_params() -> BayesianDiscoverParams {
    BayesianDiscoverParams::default()
}

/// Default Bayesian parameters with an optional cap on the number of parents of any node.
fn bayesian_params_capped(max_parents: Option<usize>) -> BayesianDiscoverParams {
    let mut params = bayesian_params();
    params.prior.constraints.max_parents = max_parents;
    params
}

/// The local score a posterior search runs under: Gaussian noise (the default, invariant
/// within a Markov equivalence class) or Laplace noise (which orients what non-Gaussian noise
/// orients).
fn score_family_of(noise: &str) -> PyResult<antecedent_state::GraphScoreFamily> {
    match noise.to_ascii_lowercase().as_str() {
        "gaussian" => Ok(antecedent_state::GraphScoreFamily::GaussianBic),
        "laplace" => Ok(antecedent_state::GraphScoreFamily::LaplaceBic),
        other => {
            Err(crate::value_err(format!("unknown noise {other:?}; expected gaussian|laplace")))
        }
    }
}

/// The search itself, or the posterior over its Markov equivalence classes.
fn over_classes(spec: DiscoverySpec, equivalence_classes: bool) -> DiscoverySpec {
    if equivalence_classes { spec.equivalence_classes() } else { spec }
}

fn schedule_from_args(n_chains: u32, n_warmup: u32, n_draws: u32, thin: u32) -> GraphMcmcSchedule {
    GraphMcmcSchedule { n_chains, n_warmup, n_draws, thin }
}

fn parse_soft_weight(name: &str) -> PyResult<CiSoftWeight> {
    match name.to_ascii_lowercase().as_str() {
        "none" | "" => Ok(CiSoftWeight::None),
        "bayes_factor" | "bayesfactor" => Ok(CiSoftWeight::BayesFactor),
        "posterior_dependence" | "posteriordependence" => Ok(CiSoftWeight::PosteriorDependence),
        other => Err(crate::value_err(format!(
            "unknown soft_weight {other:?}; expected none|bayes_factor|posterior_dependence"
        ))),
    }
}

/// Exact DAG posterior enumeration (`n ≤ 6` hard limit, Gaussian BIC).
///
/// Larger graphs: use `discover_order_mcmc`, `discover_structure_mcmc`, or
/// `discover_ci_screened_posterior`.
#[pyfunction]
#[pyo3(signature = (names, columns, *, seed=1, threads=None, equivalence_classes=false))]
fn discover_exact_dag_posterior(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    seed: u64,
    threads: Option<u32>,
    equivalence_classes: bool,
) -> PyResult<PyGraphPosterior> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let (post, route) = compiled_search(
            over_classes(
                DiscoverySpec::exact_dag_posterior(bayesian_params()),
                equivalence_classes,
            ),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            posterior_of,
        )?;
        Ok(PyGraphPosterior::from_rust(names, post).with_route(route))
    })
}

/// Exact posterior over PAG classes of the Gaussian maximal ancestral graphs of at most four
/// variables (latent confounding allowed), scored by BIC.
#[pyfunction]
#[pyo3(signature = (names, columns, *, seed=1, threads=None))]
fn discover_exact_pag_posterior(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<PyGraphPosterior> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let (post, route) = compiled_search(
            DiscoverySpec::exact_pag_posterior(bayesian_params()),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            posterior_of,
        )?;
        Ok(PyGraphPosterior::from_rust(names, post).with_route(route))
    })
}

/// Metropolis posterior over PAG classes of the Gaussian maximal ancestral graphs of two to
/// eight variables (latent confounding allowed), by single-pair link moves.
#[pyfunction]
#[pyo3(signature = (names, columns, *, n_chains=4, n_warmup=500, n_draws=2000, thin=1, seed=1, threads=None))]
fn discover_pag_mcmc(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    n_chains: u32,
    n_warmup: u32,
    n_draws: u32,
    thin: u32,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<PyGraphPosterior> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, thin);
        let (post, route) = compiled_search(
            DiscoverySpec::pag_mcmc(bayesian_params(), schedule),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            posterior_of,
        )?;
        Ok(PyGraphPosterior::from_rust(names, post).with_route(route))
    })
}

/// Exact edge marginals of the DAG posterior (Gaussian BIC) for up to 12 variables.
///
/// Returns the row-major `P(i -> j)` (`from * n + to`) and the log normalizing constant. No
/// graph atoms exist at this size.
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_parents=None, seed=1, threads=None))]
fn discover_exact_dag_marginals(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_parents: Option<usize>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<(Vec<f64>, f64)> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let (marginals, _route) = compiled_search(
            DiscoverySpec::exact_dag_marginals(bayesian_params_capped(max_parents)),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::EdgeMarginals(marginals) => Some(marginals),
                _ => None,
            },
        )?;
        Ok((marginals.edge_marginals.to_vec(), marginals.log_evidence))
    })
}

/// Order MCMC DAG posterior (Gaussian BIC), up to 16 variables.
///
/// `max_parents` caps the number of parents of any node (`None`: no cap).
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
    max_parents=None,
    seed=1,
    threads=None,
    equivalence_classes=false,
    noise="gaussian"
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
    max_parents: Option<usize>,
    seed: u64,
    threads: Option<u32>,
    equivalence_classes: bool,
    noise: &str,
) -> PyResult<PyGraphPosterior> {
    let family = score_family_of(noise)?;
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, thin);
        let mut params = bayesian_params_capped(max_parents);
        params.score_family = family;
        let (post, route) = compiled_search(
            over_classes(
                DiscoverySpec::order_mcmc(params, schedule, require_diagnostics_gate),
                equivalence_classes,
            ),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            posterior_of,
        )?;
        Ok(PyGraphPosterior::from_rust(names, post).with_route(route))
    })
}

/// Structure MCMC DAG posterior (Gaussian BIC), up to 16 variables.
///
/// `max_parents` caps the number of parents of any node (`None`: no cap).
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    *,
    n_chains=4,
    n_warmup=500,
    n_draws=1000,
    thin=1,
    max_parents=None,
    seed=1,
    threads=None,
    equivalence_classes=false
))]
fn discover_structure_mcmc(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    n_chains: u32,
    n_warmup: u32,
    n_draws: u32,
    thin: u32,
    max_parents: Option<usize>,
    seed: u64,
    threads: Option<u32>,
    equivalence_classes: bool,
) -> PyResult<PyGraphPosterior> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, thin);
        let (post, route) = compiled_search(
            over_classes(
                DiscoverySpec::structure_mcmc(bayesian_params_capped(max_parents), schedule),
                equivalence_classes,
            ),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            posterior_of,
        )?;
        Ok(PyGraphPosterior::from_rust(names, post).with_route(route))
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
    threads=None,
    equivalence_classes=false
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
    threads: Option<u32>,
    equivalence_classes: bool,
) -> PyResult<PyGraphPosterior> {
    let soft = parse_soft_weight(soft_weight)?;
    let ci_name = ci.unwrap_or_else(|| "parcorr".to_string());
    let ci_label = ci_name.to_ascii_lowercase();
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
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let screen = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc: false,
            max_subset: None,
            knowledge: antecedent::discovery::BackgroundKnowledge::default(),
        };
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, thin);
        let (post, route) = compiled_search(
            over_classes(
                DiscoverySpec::ci_screened_posterior(
                    bayesian_params(),
                    screen,
                    schedule,
                    soft,
                    ci_label,
                ),
                equivalence_classes,
            ),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            posterior_of,
        )?;
        Ok(PyGraphPosterior::from_rust(names, post).with_route(route))
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
    threads=None,
    unit_lengths=None,
    equivalence_classes=false
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
    threads: Option<u32>,
    unit_lengths: Option<Vec<usize>>,
    equivalence_classes: bool,
) -> PyResult<PyGraphPosterior> {
    let _ = thin; // DBN engine schedule has no thin; kept for API symmetry.
    let ingested = crate::columns_to_ingested(&names, &columns, unit_lengths.as_deref())?;
    drop(columns);
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, 1);
        let (input, variables) = match ingested {
            // `unit_lengths` marks the columns as a panel's row-concatenated units: each unit's
            // lag windows are built inside that unit.
            crate::IngestedColumns::Panel(batches) => {
                let (units, variables) =
                    crate::discovery_api::panel_units_from_unit_batches(&batches)?;
                (DiscoverInput::Panel(units), variables)
            }
            crate::IngestedColumns::Series(batch) => {
                let (series, variables) = series_from_batch(&batch)?;
                (DiscoverInput::Series(series), variables)
            }
        };
        let (post, route) = compiled_search(
            over_classes(
                DiscoverySpec::dbn_posterior(bayesian_params(), max_lag, force_mcmc, schedule),
                equivalence_classes,
            ),
            input,
            &variables,
            &ctx,
            posterior_of,
        )?;
        Ok(PyGraphPosterior::from_rust(names, post).with_route(route))
    })
}

/// Register Bayesian discovery types and functions on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGraphPosterior>()?;
    m.add_function(wrap_pyfunction!(discover_exact_dag_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(discover_exact_dag_marginals, m)?)?;
    m.add_function(wrap_pyfunction!(discover_exact_pag_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pag_mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(discover_order_mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(discover_structure_mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(discover_ci_screened_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(discover_dbn_posterior, m)?)?;
    Ok(())
}
