//! Bayesian graph-posterior discovery bindings.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::NumericColumn;
use antecedent::discovery::adjacency_mask::{
    adjacency_mask_from_admg, adjacency_mask_from_cpdag, adjacency_masks_from_pag, has_edge,
    mask_is_dag,
};
use antecedent::discovery::{
    AdjacencyMask, BayesianDiscoverParams, CiSoftWeight, EdgePriors, GraphMcmcSchedule,
    GraphPosterior as RustGraphPosterior, GraphPosteriorAtomKind, InferenceDiagnostics,
    SamplingSummary, SoftTiers, StaticDiscoverParams, n_directed_edges,
};
use antecedent::discovery_defaults::resolve_ci;
use antecedent::graph::Cpdag as RustCpdag;
use antecedent::inference::{FdrAdjustment, PartialCorrelation};
use antecedent::{DiscoveredStructure, DiscoverySpec};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyModule};

use crate::compiled_route_api::{DiscoverInput, RouteRecord};
use crate::discovery_api::compiled_search;
use crate::{
    detach_catch, numeric_columns_to_batch, py_err, py_execution_context, py_msg,
    series_from_batch, tabular_from_batch,
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
    /// Accepted provenance for a pooled search or an explicitly scored/capped exact search.
    accepted: Option<antecedent::analysis::structure_route::AcceptedPosterior>,
}

/// Width of the packed lag masks.
const MASK_BITS: usize = u64::BITS as usize;

/// Python `int` holding the packed edge bits of `mask` (bit layout `edge_bit`).
fn mask_to_pyint(py: Python<'_>, mask: AdjacencyMask) -> PyResult<Bound<'_, PyAny>> {
    py.import("builtins")?
        .getattr("int")?
        .call_method1("from_bytes", (PyBytes::new(py, &mask.to_le_bytes()), "little"))
}

/// Packed edge bits from a non-negative Python `int`.
fn mask_from_pyint(value: &Bound<'_, PyAny>) -> PyResult<AdjacencyMask> {
    let refuse = || crate::value_err("adjacency masks must be non-negative integers");
    let bits: usize = value.call_method0("bit_length").map_err(|_| refuse())?.extract()?;
    let len = bits.div_ceil(8).max(32);
    let bytes: Vec<u8> =
        value.call_method1("to_bytes", (len, "little")).map_err(|_| refuse())?.extract()?;
    Ok(AdjacencyMask::from_le_bytes(&bytes))
}

/// Most variables of a supplied atom posterior. The default edge marginals are `n_vars²` cells,
/// so the count is bounded before any edge set is sized by it.
const MAX_ATOM_VARIABLES: usize = 4096;

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
            n_vars: post.n_vars(),
            n_graphs: post.n_graphs(),
            weights: post.weights().as_ref().to_vec(),
            adjacency: post.atoms().as_ref().to_vec(),
            edge_marginals: post.edge_marginals().as_ref().to_vec(),
            orientation_marginals: post.orientation_marginals().as_ref().to_vec(),
            ess: post.ess(),
            rejected_invalid: post.rejected_invalid(),
            converged: post.diagnostics().converged,
            lagged_edge_marginals: post.lagged_edge_marginals().map(|v| v.as_ref().to_vec()),
            max_lag: post.max_lag(),
            lag_masks: post
                .lag_masks()
                .and_then(|v| v.iter().map(AdjacencyMask::to_word).collect()),
            lag_bidirected_masks: post
                .lag_bidirected_masks()
                .and_then(|v| v.iter().map(AdjacencyMask::to_word).collect()),
            algorithm: post.algorithm().map(ToString::to_string),
            atom_kind: post.atom_kind().as_str().to_owned(),
            mark_masks: post.mark_atoms().map(|v| v.as_ref().to_vec()),
            edge_mcse: post.edge_mcse().map(|v| v.as_ref().to_vec()),
            edge_rhat: post.edge_rhat().map(|v| v.as_ref().to_vec()),
            edge_ess: post.edge_ess().map(|v| v.as_ref().to_vec()),
            backend_id: post.diagnostics().backend_id.to_string(),
            route: None,
            accepted: None,
        }
    }

    fn with_accepted(
        mut self,
        accepted: Option<antecedent::analysis::structure_route::AcceptedPosterior>,
    ) -> Self {
        self.accepted = accepted;
        self
    }

    /// The accepted pooled posterior, for a study that consumes this handle.
    pub(crate) fn pooled_accepted(
        &self,
    ) -> Option<antecedent::analysis::structure_route::AcceptedPosterior> {
        self.accepted.as_ref().filter(|value| value.provenance().pooling.is_some()).cloned()
    }

    pub(crate) fn configured_exact(
        &self,
    ) -> Option<&antecedent::analysis::structure_route::AcceptedPosterior> {
        self.accepted.as_ref().filter(|value| value.provenance().pooling.is_none())
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
    /// Whether the atoms carry lagged edge marginals (a dynamic-network posterior).
    pub(crate) fn is_temporal(&self) -> bool {
        self.lagged_edge_marginals.is_some() && self.max_lag.is_some()
    }

    /// Variable names of the posterior's nodes, in node order.
    pub(crate) fn names(&self) -> &[String] {
        &self.names
    }

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
        self.adjacency.iter().map(|mask| mask_to_pyint(py, mask.clone())).collect()
    }

    /// Optional PAG circle-endpoint masks, same packing as `adjacency`.
    #[getter]
    fn mark_masks<'py>(&self, py: Python<'py>) -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
        self.mark_masks
            .as_ref()
            .map(|masks| masks.iter().map(|mask| mask_to_pyint(py, mask.clone())).collect())
            .transpose()
    }

    /// Summary dict compatible with weighted-graph envelope consumers.
    fn to_weighted_samples<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("n_samples", self.n_graphs)?;
        d.set_item("weights", self.weights.clone())?;
        d.set_item(
            "graph_keys",
            self.adjacency
                .iter()
                .map(antecedent::discovery::AdjacencyMask::key)
                .collect::<Vec<_>>(),
        )?;
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

    /// Accepted provenance of a configured exact search or pooled posterior, including
    /// the pooling rule and imputation count for pooled results.
    #[getter]
    fn provenance(&self) -> Option<std::collections::HashMap<String, String>> {
        self.accepted
            .as_ref()
            .map(|accepted| crate::compiled_route_api::provenance_map(accepted.provenance()))
    }

    /// The equal-weight mixture over imputations of each imputation's structure posterior.
    ///
    /// The posteriors must be over the same variables in the same order, with DAG atoms. The
    /// pooled mass of a graph is the mean of its per-imputation masses.
    #[classmethod]
    fn pool(
        _cls: &Bound<'_, pyo3::types::PyType>,
        posteriors: Vec<PyRef<'_, PyGraphPosterior>>,
    ) -> PyResult<Self> {
        let first = posteriors
            .first()
            .ok_or_else(|| crate::value_err("pooling needs at least one posterior"))?;
        let names = first.names.clone();
        let mut inner = Vec::with_capacity(posteriors.len());
        for posterior in &posteriors {
            if posterior.configured_exact().is_some() {
                return Err(crate::value_err(
                    "pooling exact structure posteriors supports the uncapped Gaussian BIC search",
                ));
            }
            if posterior.names != names {
                return Err(crate::value_err(
                    "the imputations' posteriors must be over the same variables in the same order",
                ));
            }
            inner.push(posterior.to_rust()?);
        }
        let pooled = RustGraphPosterior::pooled_over_imputations(&inner).map_err(py_msg)?;
        Ok(Self::from_rust(names, pooled))
    }

    /// Pack typed graphs into a posterior. Every atom must share one class.
    /// ADMG atoms must be bow-free: this encoding cannot retain a directed and
    /// bidirected edge on the same pair. An explicit ADMG graph retains bows.
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
        if n_vars > MAX_ATOM_VARIABLES {
            return Err(crate::value_err(format!(
                "from_atoms supports at most {MAX_ATOM_VARIABLES} variables, got {n_vars}"
            )));
        }
        let edge_bits = n_directed_edges(n_vars);
        for (index, mask) in adjacency.iter().enumerate() {
            if mask.set_bits().any(|bit| bit as usize >= edge_bits) {
                return Err(crate::value_err(format!(
                    "adjacency atom {index} sets edge bits beyond {n_vars} variables"
                )));
            }
            if kind == GraphPosteriorAtomKind::Dag && !mask_is_dag(mask, n_vars) {
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
                        if from != to && has_edge(mask, n_vars, from, to) {
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
            adjacency.push(
                adjacency_mask_from_admg(&admg.admg)
                    .map_err(|error| crate::value_err(error.to_string()))?,
            );
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

pub(crate) fn exact_score_family(score: &str) -> PyResult<antecedent::state::GraphScoreFamily> {
    use antecedent::state::GraphScoreFamily;
    match score {
        "gaussian_bic" => Ok(GraphScoreFamily::GaussianBic),
        "gaussian_signed_uniform" => Ok(GraphScoreFamily::GaussianSignedUniform),
        other => Err(crate::value_err(format!(
            "unknown exact DAG score {other:?}; use gaussian_bic or gaussian_signed_uniform"
        ))),
    }
}

/// Per-edge prior probabilities and soft tiers of a posterior search, by variable name.
///
/// `edge_priors` lists `(from, to, probability)`; `edge_prior_default` is the probability of the
/// unlisted edges. `soft_tiers` lists variable names, earliest tier first, and
/// `soft_tier_penalty` is the log-penalty charged to a present edge against that order.
#[derive(Clone, Debug, Default)]
pub(crate) struct PriorArgs {
    edge_priors: Option<Vec<(String, String, f64)>>,
    edge_prior_default: Option<f64>,
    soft_tiers: Option<Vec<Vec<String>>>,
    soft_tier_penalty: Option<f64>,
}

impl PriorArgs {
    /// Read the arguments from the mapping a `StructurePrior` forwards with a prepared study.
    pub(crate) fn from_dict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        const KEYS: &[&str] =
            &["edge_priors", "edge_prior_default", "soft_tiers", "soft_tier_penalty"];
        for (key, _) in dict.iter() {
            let key: String = key.extract()?;
            if !KEYS.contains(&key.as_str()) {
                return Err(crate::value_err(format!(
                    "unknown structure prior member {key:?}; expected one of {KEYS:?}"
                )));
            }
        }
        let item = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
            Ok(dict.get_item(key)?.filter(|value| !value.is_none()))
        };
        Ok(Self {
            edge_priors: item("edge_priors")?.map(|value| value.extract()).transpose()?,
            edge_prior_default: item("edge_prior_default")?
                .map(|value| value.extract())
                .transpose()?,
            soft_tiers: item("soft_tiers")?.map(|value| value.extract()).transpose()?,
            soft_tier_penalty: item("soft_tier_penalty")?
                .map(|value| value.extract())
                .transpose()?,
        })
    }

    /// The default Bayesian parameters under these prior arguments, for a search over the
    /// variables `names` of a table or series (`variables[i]` is the variable of `names[i]`).
    pub(crate) fn params_over(
        self,
        names: &[String],
        variables: &[antecedent::prelude::VariableId],
    ) -> PyResult<BayesianDiscoverParams> {
        let mut params = bayesian_params();
        self.apply(&mut params, names, variables)?;
        Ok(params)
    }

    /// Set the per-edge map and soft tiers on `params`, resolving names against `names`
    /// (`variables[i]` is the variable of `names[i]`).
    fn apply(
        self,
        params: &mut BayesianDiscoverParams,
        names: &[String],
        variables: &[antecedent::prelude::VariableId],
    ) -> PyResult<()> {
        let resolve = |name: &str| -> PyResult<antecedent::prelude::VariableId> {
            names
                .iter()
                .position(|candidate| candidate == name)
                .map(|index| variables[index])
                .ok_or_else(|| crate::value_err(format!("unknown variable {name:?} in the prior")))
        };
        if self.edge_priors.is_some() || self.edge_prior_default.is_some() {
            let mut edges = Vec::new();
            for (from, to, probability) in self.edge_priors.unwrap_or_default() {
                edges.push((resolve(&from)?, resolve(&to)?, probability));
            }
            let mut priors = EdgePriors::new(edges);
            priors.default = self.edge_prior_default;
            params.prior = params.prior.clone().with_edge_priors(priors);
        }
        match (self.soft_tiers, self.soft_tier_penalty) {
            (None, None) => {}
            (Some(tiers), Some(penalty)) => {
                let mut resolved: Vec<std::sync::Arc<[antecedent::prelude::VariableId]>> =
                    Vec::new();
                for tier in tiers {
                    let members =
                        tier.iter().map(|name| resolve(name)).collect::<PyResult<Vec<_>>>()?;
                    resolved.push(members.into());
                }
                params.prior = params
                    .prior
                    .clone()
                    .with_soft_tiers(SoftTiers::with_log_penalty(resolved, penalty));
            }
            _ => {
                return Err(crate::value_err("soft tiers need both the tiers and a penalty"));
            }
        }
        Ok(())
    }
}

/// The local score a posterior search runs under: Gaussian noise (the default, invariant
/// within a Markov equivalence class) or Laplace noise (which orients what non-Gaussian noise
/// orients).
fn score_family_of(noise: &str) -> PyResult<antecedent::state::GraphScoreFamily> {
    match noise.to_ascii_lowercase().as_str() {
        "gaussian" => Ok(antecedent::state::GraphScoreFamily::GaussianBic),
        "laplace" => Ok(antecedent::state::GraphScoreFamily::LaplaceBic),
        other => {
            Err(crate::value_err(format!("unknown noise {other:?}; expected gaussian|laplace")))
        }
    }
}

/// The search itself, or the posterior over its Markov equivalence classes.
fn over_classes(spec: DiscoverySpec, equivalence_classes: bool) -> DiscoverySpec {
    if equivalence_classes { spec.equivalence_classes() } else { spec }
}

fn schedule_from_args(
    n_chains: u32,
    n_warmup: u32,
    n_draws: u32,
    thin: u32,
) -> PyResult<GraphMcmcSchedule> {
    crate::bridge::check_mcmc_schedule(n_chains, n_warmup, n_draws, thin)?;
    Ok(GraphMcmcSchedule { n_chains, n_warmup, n_draws, thin })
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
#[pyo3(signature = (
    names,
    columns,
    *,
    score="gaussian_bic",
    max_parents=None,
    seed=1,
    threads=None,
    equivalence_classes=false,
    edge_priors=None,
    edge_prior_default=None,
    soft_tiers=None,
    soft_tier_penalty=None,
))]
fn discover_exact_dag_posterior(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    score: &str,
    max_parents: Option<usize>,
    seed: u64,
    threads: Option<u32>,
    equivalence_classes: bool,
    edge_priors: Option<Vec<(String, String, f64)>>,
    edge_prior_default: Option<f64>,
    soft_tiers: Option<Vec<Vec<String>>>,
    soft_tier_penalty: Option<f64>,
) -> PyResult<PyGraphPosterior> {
    let family = exact_score_family(score)?;
    let batch = numeric_columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let mut params = bayesian_params_capped(max_parents);
        params.score_family = family;
        PriorArgs { edge_priors, edge_prior_default, soft_tiers, soft_tier_penalty }.apply(
            &mut params,
            &names,
            &variables,
        )?;
        let (post, route) = compiled_search(
            over_classes(DiscoverySpec::exact_dag_posterior(params), equivalence_classes),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            posterior_of,
        )?;
        let configured =
            family != antecedent::state::GraphScoreFamily::GaussianBic || max_parents.is_some();
        let accepted = configured.then(|| route.accepted_posterior()).transpose()?;
        Ok(PyGraphPosterior::from_rust(names, post).with_route(route).with_accepted(accepted))
    })
}

/// Exact posterior over PAG classes of the Gaussian maximal ancestral graphs of at most four
/// variables (latent confounding allowed), scored by BIC.
#[pyfunction]
#[pyo3(signature = (names, columns, *, seed=1, threads=None))]
fn discover_exact_pag_posterior(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<PyGraphPosterior> {
    let batch = numeric_columns_to_batch(&names, &columns)?;
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
/// 31 variables (`PAG_MCMC_MAX_NODES`; latent confounding allowed), by single-pair link moves.
#[pyfunction]
#[pyo3(signature = (names, columns, *, n_chains=4, n_warmup=500, n_draws=2000, thin=1, seed=1, threads=None))]
fn discover_pag_mcmc(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    n_chains: u32,
    n_warmup: u32,
    n_draws: u32,
    thin: u32,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<PyGraphPosterior> {
    let batch = numeric_columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, thin)?;
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
#[pyo3(signature = (
    names,
    columns,
    *,
    score="gaussian_bic",
    max_parents=None,
    seed=1,
    threads=None,
    edge_priors=None,
    edge_prior_default=None,
    soft_tiers=None,
    soft_tier_penalty=None,
))]
fn discover_exact_dag_marginals(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    score: &str,
    max_parents: Option<usize>,
    seed: u64,
    threads: Option<u32>,
    edge_priors: Option<Vec<(String, String, f64)>>,
    edge_prior_default: Option<f64>,
    soft_tiers: Option<Vec<Vec<String>>>,
    soft_tier_penalty: Option<f64>,
) -> PyResult<(Vec<f64>, f64)> {
    let family = exact_score_family(score)?;
    let batch = numeric_columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let mut params = bayesian_params_capped(max_parents);
        params.score_family = family;
        PriorArgs { edge_priors, edge_prior_default, soft_tiers, soft_tier_penalty }.apply(
            &mut params,
            &names,
            &variables,
        )?;
        let (marginals, _route) = compiled_search(
            DiscoverySpec::exact_dag_marginals(params),
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

/// The completed tables of an imputed run as native column batches over one variable list.
fn imputed_tables(
    names: &[String],
    tables: Vec<Vec<NumericColumn<'_>>>,
) -> PyResult<Vec<crate::ColumnBatch>> {
    if tables.is_empty() {
        return Err(crate::value_err("imputations needs at least one completed table"));
    }
    tables.into_iter().map(|columns| numeric_columns_to_batch(names, &columns)).collect()
}

/// Exact DAG posterior (at most six variables) pooled across imputations: the equal-weight mixture of the
/// per-imputation exact posteriors, tagged `pooled_imputations(...)`.
#[pyfunction]
#[pyo3(signature = (
    names,
    tables,
    *,
    seed=1,
    threads=None,
    edge_priors=None,
    edge_prior_default=None,
    soft_tiers=None,
    soft_tier_penalty=None,
))]
fn discover_pooled_exact_dag_posterior(
    py: Python<'_>,
    names: Vec<String>,
    tables: Vec<Vec<NumericColumn<'_>>>,
    seed: u64,
    threads: Option<u32>,
    edge_priors: Option<Vec<(String, String, f64)>>,
    edge_prior_default: Option<f64>,
    soft_tiers: Option<Vec<Vec<String>>>,
    soft_tier_penalty: Option<f64>,
) -> PyResult<PyGraphPosterior> {
    let batches = imputed_tables(&names, tables)?;
    detach_catch(py, move || {
        let loaded = batches.iter().map(tabular_from_batch).collect::<PyResult<Vec<_>>>()?;
        let variables = loaded[0].1.clone();
        let data: Vec<_> = loaded.into_iter().map(|(table, _)| table).collect();
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let mut params = bayesian_params();
        PriorArgs { edge_priors, edge_prior_default, soft_tiers, soft_tier_penalty }.apply(
            &mut params,
            &names,
            &variables,
        )?;
        let mut posteriors = Vec::with_capacity(data.len());
        for table in data {
            let (post, _route) = compiled_search(
                DiscoverySpec::exact_dag_posterior(params.clone()),
                DiscoverInput::Tabular(table),
                &variables,
                &ctx,
                posterior_of,
            )?;
            posteriors.push(post);
        }
        let pooled = RustGraphPosterior::pooled_over_imputations(&posteriors).map_err(py_msg)?;
        Ok(PyGraphPosterior::from_rust(names, pooled))
    })
}

/// Exact DAG edge marginals (seven to fifteen variables) pooled across imputations: the means of the per-imputation
/// marginals and the log of the mean of the per-imputation normalizers.
#[pyfunction]
#[pyo3(signature = (
    names,
    tables,
    *,
    seed=1,
    threads=None,
    edge_priors=None,
    edge_prior_default=None,
    soft_tiers=None,
    soft_tier_penalty=None,
))]
fn discover_pooled_exact_dag_marginals(
    py: Python<'_>,
    names: Vec<String>,
    tables: Vec<Vec<NumericColumn<'_>>>,
    seed: u64,
    threads: Option<u32>,
    edge_priors: Option<Vec<(String, String, f64)>>,
    edge_prior_default: Option<f64>,
    soft_tiers: Option<Vec<Vec<String>>>,
    soft_tier_penalty: Option<f64>,
) -> PyResult<(Vec<f64>, f64)> {
    let batches = imputed_tables(&names, tables)?;
    detach_catch(py, move || {
        let loaded = batches.iter().map(tabular_from_batch).collect::<PyResult<Vec<_>>>()?;
        let variables = loaded[0].1.clone();
        let data: Vec<_> = loaded.into_iter().map(|(table, _)| table).collect();
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let mut params = bayesian_params();
        PriorArgs { edge_priors, edge_prior_default, soft_tiers, soft_tier_penalty }.apply(
            &mut params,
            &names,
            &variables,
        )?;
        let mut parts = Vec::with_capacity(data.len());
        for table in data {
            let (marginals, _route) = compiled_search(
                DiscoverySpec::exact_dag_marginals(params.clone()),
                DiscoverInput::Tabular(table),
                &variables,
                &ctx,
                |structure| match structure {
                    DiscoveredStructure::EdgeMarginals(marginals) => Some(marginals),
                    _ => None,
                },
            )?;
            parts.push(marginals);
        }
        let pooled = antecedent::discovery::ExactDagMarginals::pooled_over_imputations(&parts)
            .map_err(py_msg)?;
        Ok((pooled.edge_marginals.to_vec(), pooled.log_evidence))
    })
}

fn setting<'py>(
    settings: &Bound<'py, PyDict>,
    key: &str,
    algorithm: &str,
) -> PyResult<Bound<'py, PyAny>> {
    settings
        .get_item(key)?
        .filter(|value| !value.is_none())
        .ok_or_else(|| crate::value_err(format!("{algorithm} needs the setting {key:?}")))
}

/// A discovery search as a function of the searched variables.
type SearchBuilder =
    Box<dyn Fn(&[antecedent::prelude::VariableId]) -> PyResult<DiscoverySpec> + Send>;

/// Any posterior search pooled across imputations: the search runs, through the compiled
/// route, on every completed table (a table for the static searches, a series or panel for the
/// DBN posterior) and the posteriors are mixed with equal weight. `settings` carries the
/// search's own options (the fields of its configuration); the per-imputation Monte-Carlo
/// diagnostics stay in the pooled posterior's diagnostics.
#[pyfunction]
#[pyo3(signature = (
    algorithm,
    names,
    tables,
    settings,
    *,
    seed=1,
    threads=None,
    unit_lengths=None,
    edge_priors=None,
    edge_prior_default=None,
    soft_tiers=None,
    soft_tier_penalty=None,
))]
fn discover_pooled_posterior(
    py: Python<'_>,
    algorithm: &str,
    names: Vec<String>,
    tables: Vec<Vec<NumericColumn<'_>>>,
    settings: &Bound<'_, PyDict>,
    seed: u64,
    threads: Option<u32>,
    unit_lengths: Option<Vec<usize>>,
    edge_priors: Option<Vec<(String, String, f64)>>,
    edge_prior_default: Option<f64>,
    soft_tiers: Option<Vec<Vec<String>>>,
    soft_tier_penalty: Option<f64>,
) -> PyResult<PyGraphPosterior> {
    if tables.is_empty() {
        return Err(crate::value_err("imputations needs at least one completed table"));
    }
    let prior = PriorArgs { edge_priors, edge_prior_default, soft_tiers, soft_tier_penalty };
    let get = |key: &str| setting(settings, key, algorithm);
    // The search, as a function of the searched variables.
    let build: SearchBuilder = match algorithm {
        "exact_dag_posterior" => {
            if !settings.is_empty() {
                return Err(crate::value_err(
                    "pooling exact structure posteriors supports the uncapped Gaussian BIC search",
                ));
            }
            let prior = prior.clone();
            let names = names.clone();
            Box::new(move |variables| {
                let params = prior.clone().params_over(&names, variables)?;
                Ok(DiscoverySpec::exact_dag_posterior(params))
            })
        }
        "order_mcmc" | "structure_mcmc" => {
            let schedule = schedule_from_args(
                get("n_chains")?.extract()?,
                get("n_warmup")?.extract()?,
                get("n_draws")?.extract()?,
                get("thin")?.extract()?,
            )?;
            let max_parents: Option<usize> = settings
                .get_item("max_parents")?
                .filter(|value| !value.is_none())
                .map(|value| value.extract())
                .transpose()?;
            let order = algorithm == "order_mcmc";
            let (gate, family) = if order {
                (
                    get("require_diagnostics_gate")?.extract::<bool>()?,
                    score_family_of(&get("noise")?.extract::<String>()?)?,
                )
            } else {
                (false, antecedent::state::GraphScoreFamily::GaussianBic)
            };
            let prior = prior.clone();
            let names = names.clone();
            Box::new(move |variables| {
                let mut params = bayesian_params_capped(max_parents);
                params.score_family = family;
                prior.clone().apply(&mut params, &names, variables)?;
                Ok(if order {
                    DiscoverySpec::order_mcmc(params, schedule, gate)
                } else {
                    DiscoverySpec::structure_mcmc(params, schedule)
                })
            })
        }
        "ci_screened_posterior" => {
            let max_cond_size: usize = get("max_cond_size")?.extract()?;
            crate::bridge::check_count(
                "max_cond_size",
                max_cond_size as u64,
                crate::bridge::MAX_CONDITIONING,
            )?;
            let soft = parse_soft_weight(&get("soft_weight")?.extract::<String>()?)?;
            let ci_name: String = settings
                .get_item("ci")?
                .filter(|value| !value.is_none())
                .map(|value| value.extract())
                .transpose()?
                .unwrap_or_else(|| "parcorr".to_owned());
            let ci_label = ci_name.to_ascii_lowercase();
            let ci_impl = if ci_name.eq_ignore_ascii_case("parcorr") {
                std::sync::Arc::new(PartialCorrelation)
                    as std::sync::Arc<
                        dyn antecedent::inference::ConditionalIndependence + Send + Sync,
                    >
            } else {
                resolve_ci(&ci_name, None).map_err(py_err)?
            };
            let screen = StaticDiscoverParams {
                alpha: get("alpha")?.extract()?,
                max_cond_size,
                fdr: get("fdr")?
                    .extract::<bool>()?
                    .then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
                ci: ci_impl,
                screen_pc: false,
                max_subset: None,
                knowledge: antecedent::discovery::BackgroundKnowledge::default(),
            };
            let schedule = schedule_from_args(
                get("n_chains")?.extract()?,
                get("n_warmup")?.extract()?,
                get("n_draws")?.extract()?,
                get("thin")?.extract()?,
            )?;
            let prior = prior.clone();
            let names = names.clone();
            Box::new(move |variables| {
                let params = prior.clone().params_over(&names, variables)?;
                Ok(DiscoverySpec::ci_screened_posterior(
                    params,
                    screen.clone(),
                    schedule,
                    soft,
                    ci_label.clone(),
                ))
            })
        }
        "dbn_posterior" => {
            let max_lag: u32 = get("max_lag")?.extract()?;
            let force_mcmc: bool = get("force_mcmc")?.extract()?;
            let schedule = schedule_from_args(
                get("n_chains")?.extract()?,
                get("n_warmup")?.extract()?,
                get("n_draws")?.extract()?,
                1,
            )?;
            let prior = prior.clone();
            let names = names.clone();
            Box::new(move |variables| {
                let params = prior.clone().params_over(&names, variables)?;
                Ok(DiscoverySpec::dbn_posterior(params, max_lag, force_mcmc, schedule))
            })
        }
        other => {
            return Err(crate::value_err(format!(
                "{other:?} is not a posterior search that pools across imputations"
            )));
        }
    };
    let temporal = algorithm == "dbn_posterior";
    let mut tabular = Vec::new();
    let mut ingested = Vec::new();
    for columns in tables {
        if temporal {
            ingested.push(crate::columns_to_ingested(&names, &columns, unit_lengths.as_deref())?);
        } else {
            tabular.push(numeric_columns_to_batch(&names, &columns)?);
        }
    }
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let mut inputs: Vec<(DiscoverInput, Vec<antecedent::prelude::VariableId>)> = Vec::new();
        for batch in &tabular {
            let (data, variables) = tabular_from_batch(batch)?;
            inputs.push((DiscoverInput::Tabular(data), variables));
        }
        for columns in ingested {
            inputs.push(match columns {
                crate::IngestedColumns::Panel(batches) => {
                    let (units, variables) =
                        crate::discovery_api::panel_units_from_unit_batches(&batches)?;
                    (DiscoverInput::Panel(units), variables)
                }
                crate::IngestedColumns::Series(batch) => {
                    let (series, variables) = series_from_batch(&batch)?;
                    (DiscoverInput::Series(series), variables)
                }
            });
        }
        let mut posteriors = Vec::with_capacity(inputs.len());
        let mut searched: Option<(DiscoverySpec, Vec<antecedent::prelude::VariableId>)> = None;
        let mut completed_tables = Vec::new();
        for (input, variables) in inputs {
            let spec = build(&variables)?;
            if let DiscoverInput::Tabular(data) = &input {
                completed_tables.push(data.clone());
            }
            if searched.is_none() {
                searched = Some((spec.clone(), variables.clone()));
            }
            let (post, _route) = compiled_search(spec, input, &variables, &ctx, posterior_of)?;
            posteriors.push(post);
        }
        let pooled = RustGraphPosterior::pooled_over_imputations(&posteriors).map_err(py_msg)?;
        // The pooled tabular search is accepted with its provenance: the search and its
        // declared structure prior, the pooling and the tables it ran on.
        let accepted = match searched {
            Some((spec, variables)) if completed_tables.len() == posteriors.len() => {
                let query = spec
                    .query(variables)
                    .map_err(|error| py_err(antecedent::CausalError::from(error)))?;
                antecedent::analysis::structure_route::AcceptedPosterior::pooled(
                    &query,
                    pooled.clone(),
                    &completed_tables,
                    None,
                )
                .ok()
            }
            _ => None,
        };
        Ok(PyGraphPosterior::from_rust(names, pooled).with_accepted(accepted))
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
    noise="gaussian",
    edge_priors=None,
    edge_prior_default=None,
    soft_tiers=None,
    soft_tier_penalty=None
))]
fn discover_order_mcmc(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
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
    edge_priors: Option<Vec<(String, String, f64)>>,
    edge_prior_default: Option<f64>,
    soft_tiers: Option<Vec<Vec<String>>>,
    soft_tier_penalty: Option<f64>,
) -> PyResult<PyGraphPosterior> {
    let family = score_family_of(noise)?;
    let batch = numeric_columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, thin)?;
        let mut params = bayesian_params_capped(max_parents);
        params.score_family = family;
        PriorArgs { edge_priors, edge_prior_default, soft_tiers, soft_tier_penalty }.apply(
            &mut params,
            &names,
            &variables,
        )?;
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
    equivalence_classes=false,
    edge_priors=None,
    edge_prior_default=None,
    soft_tiers=None,
    soft_tier_penalty=None
))]
fn discover_structure_mcmc(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    n_chains: u32,
    n_warmup: u32,
    n_draws: u32,
    thin: u32,
    max_parents: Option<usize>,
    seed: u64,
    threads: Option<u32>,
    equivalence_classes: bool,
    edge_priors: Option<Vec<(String, String, f64)>>,
    edge_prior_default: Option<f64>,
    soft_tiers: Option<Vec<Vec<String>>>,
    soft_tier_penalty: Option<f64>,
) -> PyResult<PyGraphPosterior> {
    let batch = numeric_columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, thin)?;
        let mut params = bayesian_params_capped(max_parents);
        PriorArgs { edge_priors, edge_prior_default, soft_tiers, soft_tier_penalty }.apply(
            &mut params,
            &names,
            &variables,
        )?;
        let (post, route) = compiled_search(
            over_classes(DiscoverySpec::structure_mcmc(params, schedule), equivalence_classes),
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
    equivalence_classes=false,
    edge_priors=None,
    edge_prior_default=None,
    soft_tiers=None,
    soft_tier_penalty=None
))]
fn discover_ci_screened_posterior(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
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
    edge_priors: Option<Vec<(String, String, f64)>>,
    edge_prior_default: Option<f64>,
    soft_tiers: Option<Vec<Vec<String>>>,
    soft_tier_penalty: Option<f64>,
) -> PyResult<PyGraphPosterior> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let soft = parse_soft_weight(soft_weight)?;
    let ci_name = ci.unwrap_or_else(|| "parcorr".to_string());
    let ci_label = ci_name.to_ascii_lowercase();
    let ci_impl = if ci_name.eq_ignore_ascii_case("parcorr") {
        std::sync::Arc::new(PartialCorrelation)
            as std::sync::Arc<dyn antecedent::inference::ConditionalIndependence + Send + Sync>
    } else {
        resolve_ci(&ci_name, None).map_err(py_err)?
    };
    let batch = numeric_columns_to_batch(&names, &columns)?;
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
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, thin)?;
        let mut params = bayesian_params();
        PriorArgs { edge_priors, edge_prior_default, soft_tiers, soft_tier_penalty }.apply(
            &mut params,
            &names,
            &variables,
        )?;
        let (post, route) = compiled_search(
            over_classes(
                DiscoverySpec::ci_screened_posterior(params, screen, schedule, soft, ci_label),
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

/// CI screen of every variable, then exact edge marginals of the focus variables' skeleton
/// neighbourhood (at most `max_variables`, itself at most 15) with the screen as a per-edge prior.
///
/// Returns the selected variable names, the row-major `P(i -> j)` over them, the log normalizing
/// constant and the statement that the marginals are conditional on the screen.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    focus,
    *,
    max_variables=12,
    in_probability=0.5,
    out_probability=0.02,
    alpha=0.05,
    fdr=true,
    ci=None,
    max_cond_size=2,
    seed=1,
    threads=None,
    edge_priors=None,
    edge_prior_default=None,
    soft_tiers=None,
    soft_tier_penalty=None
))]
fn discover_screened_exact_marginals(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    focus: Vec<String>,
    max_variables: u32,
    in_probability: f64,
    out_probability: f64,
    alpha: f64,
    fdr: bool,
    ci: Option<String>,
    max_cond_size: usize,
    seed: u64,
    threads: Option<u32>,
    edge_priors: Option<Vec<(String, String, f64)>>,
    edge_prior_default: Option<f64>,
    soft_tiers: Option<Vec<Vec<String>>>,
    soft_tier_penalty: Option<f64>,
) -> PyResult<(Vec<String>, Vec<f64>, f64, String)> {
    crate::bridge::check_count(
        "max_cond_size",
        max_cond_size as u64,
        crate::bridge::MAX_CONDITIONING,
    )?;
    let ci_name = ci.unwrap_or_else(|| "parcorr".to_string());
    let ci_label = ci_name.to_ascii_lowercase();
    let ci_impl = if ci_name.eq_ignore_ascii_case("parcorr") {
        std::sync::Arc::new(PartialCorrelation)
            as std::sync::Arc<dyn antecedent::inference::ConditionalIndependence + Send + Sync>
    } else {
        resolve_ci(&ci_name, None).map_err(py_err)?
    };
    let batch = numeric_columns_to_batch(&names, &columns)?;
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
        let focus_ids = focus
            .iter()
            .map(|name| {
                names
                    .iter()
                    .position(|candidate| candidate == name)
                    .map(|index| variables[index])
                    .ok_or_else(|| crate::value_err(format!("unknown focus variable {name:?}")))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mut params = bayesian_params();
        PriorArgs { edge_priors, edge_prior_default, soft_tiers, soft_tier_penalty }.apply(
            &mut params,
            &names,
            &variables,
        )?;
        let (marginals, _route) = compiled_search(
            DiscoverySpec::screened_exact_marginals(
                params,
                screen,
                focus_ids,
                max_variables,
                in_probability,
                out_probability,
                ci_label,
            ),
            DiscoverInput::Tabular(data),
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::EdgeMarginals(marginals) => Some(marginals),
                _ => None,
            },
        )?;
        let selected = marginals
            .variables
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .map(|variable| names[variable.as_usize()].clone())
            .collect();
        let statement = marginals.conditional_on_screen.as_deref().unwrap_or_default().to_owned();
        Ok((selected, marginals.edge_marginals.to_vec(), marginals.log_evidence, statement))
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
    equivalence_classes=false,
    edge_priors=None,
    edge_prior_default=None,
    soft_tiers=None,
    soft_tier_penalty=None
))]
fn discover_dbn_posterior(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
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
    edge_priors: Option<Vec<(String, String, f64)>>,
    edge_prior_default: Option<f64>,
    soft_tiers: Option<Vec<Vec<String>>>,
    soft_tier_penalty: Option<f64>,
) -> PyResult<PyGraphPosterior> {
    let _ = thin; // DBN engine schedule has no thin; kept for API symmetry.
    let ingested = crate::columns_to_ingested(&names, &columns, unit_lengths.as_deref())?;
    drop(columns);
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let schedule = schedule_from_args(n_chains, n_warmup, n_draws, 1)?;
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
        let mut params = bayesian_params();
        PriorArgs { edge_priors, edge_prior_default, soft_tiers, soft_tier_penalty }.apply(
            &mut params,
            &names,
            &variables,
        )?;
        let (post, route) = compiled_search(
            over_classes(
                DiscoverySpec::dbn_posterior(params, max_lag, force_mcmc, schedule),
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
    m.add_function(wrap_pyfunction!(discover_pooled_exact_dag_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pooled_exact_dag_marginals, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pooled_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(discover_exact_dag_marginals, m)?)?;
    m.add_function(wrap_pyfunction!(discover_exact_pag_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pag_mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(discover_order_mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(discover_structure_mcmc, m)?)?;
    m.add_function(wrap_pyfunction!(discover_ci_screened_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(discover_screened_exact_marginals, m)?)?;
    m.add_function(wrap_pyfunction!(discover_dbn_posterior, m)?)?;
    Ok(())
}
