//! Sampling distribution of a frequentist structure search for the Python bindings.
//!
//! The search runs through the compiled discovery route wrapped in a stability search: it is
//! re-run on resamples drawn under the declared sampling design (independent rows for a
//! table, circular blocks for a series, whole units for a panel) and the result tabulates
//! how often every node pair was adjacent, carried every pair of endpoint marks, and how
//! often every whole structure came back. The frequencies are a bootstrap distribution of
//! the search, not a posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_arguments, reason = "one keyword per option of the wrapped search")]

use std::sync::Arc;

use antecedent::discovery::{
    BackgroundKnowledge, DiscoverParams, MultiDatasetConstraints, StaticDiscoverParams,
    StructureStability,
};
use antecedent::discovery_defaults::resolve_ci;
use antecedent::{DiscoveredStructure, DiscoverySpec};
use antecedent_core::VariableId;
use antecedent_discovery::selective::{Mark, StructureEdge, StructureNode};
use antecedent_stats::{ConditionalIndependence, FdrAdjustment, PartialCorrelation};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::compiled_route_api::{DiscoverInput, RouteRecord};
use crate::discovery_api::{
    compiled_search, knowledge_arg, panel_units_from_unit_batches, tabular_from_ingested,
};
use crate::{GraphEdge, columns_to_batch, detach_catch, py_err};

/// How often one pair of endpoint marks appeared on a node pair.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct StabilityMark {
    /// Mark at the source: `tail` | `arrow` | `circle` | `conflict`.
    #[pyo3(get)]
    pub(crate) at_source: String,
    /// Mark at the target.
    #[pyo3(get)]
    pub(crate) at_target: String,
    /// Share of the successful resamples that carried this pair of marks on the node pair.
    #[pyo3(get)]
    pub(crate) frequency: f64,
}

/// How often a node pair was adjacent.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct StabilityEdge {
    #[pyo3(get)]
    pub(crate) source: String,
    #[pyo3(get)]
    pub(crate) source_lag: u32,
    #[pyo3(get)]
    pub(crate) target: String,
    #[pyo3(get)]
    pub(crate) target_lag: u32,
    /// Share of the successful resamples whose structure joined the pair.
    #[pyo3(get)]
    pub(crate) frequency: f64,
    /// Whether the full-sample structure joins the pair.
    #[pyo3(get)]
    pub(crate) in_reference: bool,
    /// Every pair of marks seen on the node pair, most frequent first.
    #[pyo3(get)]
    pub(crate) marks: Vec<StabilityMark>,
}

/// How often the search returned one whole structure.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct StabilityStructure {
    /// `count` over the successful resamples.
    #[pyo3(get)]
    pub(crate) frequency: f64,
    /// Successful resamples that returned the structure.
    #[pyo3(get)]
    pub(crate) count: u32,
    /// The structure's marked edges.
    #[pyo3(get)]
    pub(crate) edges: Vec<GraphEdge>,
}

/// The tabulated sampling distribution of one structure search.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct StructureStabilityResult {
    /// `bootstrap_stability`.
    #[pyo3(get)]
    algorithm_id: String,
    /// The search that was re-run.
    #[pyo3(get)]
    base_algorithm: String,
    /// Class of every structure tabulated: `cpdag` (a DAG search's graphs are held as fully
    /// directed CPDAGs), `pag`, `temporal_dag`, `temporal_cpdag` or `temporal_pag`.
    #[pyo3(get)]
    structure_class: String,
    #[pyo3(get)]
    variable_names: Vec<String>,
    #[pyo3(get)]
    replicates_requested: u32,
    #[pyo3(get)]
    replicates_succeeded: u32,
    /// Resamples on which the search failed; kept as failed mass, never redrawn.
    #[pyo3(get)]
    replicates_failed: u32,
    /// Share of the successful resamples that returned exactly the full-sample structure.
    #[pyo3(get)]
    reference_frequency: f64,
    /// The structure the search returned on the full sample.
    #[pyo3(get)]
    reference_edges: Vec<GraphEdge>,
    #[pyo3(get)]
    edges: Vec<StabilityEdge>,
    #[pyo3(get)]
    structures: Vec<StabilityStructure>,
    route: Option<RouteRecord>,
}

#[pymethods]
impl StructureStabilityResult {
    /// The compiled discovery route this result came from.
    #[getter]
    fn route(&self, py: Python<'_>) -> PyResult<crate::compiled_route_api::RouteHandle> {
        crate::compiled_route_api::route_object(py, self.route.as_ref())
    }

    /// The contract the search executed under (`query_kind="Discover"`).
    #[getter]
    fn contract(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.contract()
    }

    /// The claim sealed over the stability search.
    #[getter]
    fn claim(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.claim()
    }

    /// Cheap inspection record of the compiled stability study.
    fn inspect(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.inspect()
    }

    /// Share of the successful resamples in which `source` and `target` were adjacent in either
    /// direction, or `0` for a pair no structure joined. Lags are `0` for a static search.
    #[pyo3(signature = (source, target, *, source_lag=0, target_lag=0))]
    fn adjacency_frequency(
        &self,
        source: &str,
        target: &str,
        source_lag: u32,
        target_lag: u32,
    ) -> f64 {
        let same =
            |name: &str, lag: u32, other: &str, other_lag: u32| name == other && lag == other_lag;
        self.edges
            .iter()
            .find(|e| {
                same(&e.source, e.source_lag, source, source_lag)
                    && same(&e.target, e.target_lag, target, target_lag)
                    || same(&e.source, e.source_lag, target, target_lag)
                        && same(&e.target, e.target_lag, source, source_lag)
            })
            .map_or(0.0, |e| e.frequency)
    }

    fn __repr__(&self) -> String {
        format!(
            "StructureStabilityResult(base={:?}, replicates={}/{}, structures={})",
            self.base_algorithm,
            self.replicates_succeeded,
            self.replicates_requested,
            self.structures.len()
        )
    }
}

const fn mark_name(mark: Mark) -> &'static str {
    match mark {
        Mark::Tail => "tail",
        Mark::Arrow => "arrow",
        Mark::Circle => "circle",
        Mark::Conflict => "conflict",
    }
}

/// The variable name and lag a canonical node stands for.
fn node_parts(names: &[String], node: StructureNode) -> (String, u32) {
    let name = usize::try_from(node.variable)
        .ok()
        .and_then(|index| names.get(index))
        .cloned()
        .unwrap_or_else(|| format!("var{}", node.variable));
    let lag = match node.kind {
        1 => u32::try_from(node.aux).unwrap_or(0),
        2 => u32::try_from(-node.aux).unwrap_or(0),
        _ => 0,
    };
    (name, lag)
}

fn graph_edge(names: &[String], edge: &StructureEdge) -> GraphEdge {
    let (source, source_lag) = node_parts(names, edge.a);
    let (target, target_lag) = node_parts(names, edge.b);
    GraphEdge {
        source,
        source_lag,
        target,
        target_lag,
        at_source: mark_name(edge.at_a).to_owned(),
        at_target: mark_name(edge.at_b).to_owned(),
    }
}

fn result_of(
    names: Vec<String>,
    base: &str,
    table: &StructureStability,
    route: RouteRecord,
) -> StructureStabilityResult {
    let edges = table
        .edges
        .iter()
        .map(|selection| {
            let (source, source_lag) = node_parts(&names, selection.a);
            let (target, target_lag) = node_parts(&names, selection.b);
            StabilityEdge {
                source,
                source_lag,
                target,
                target_lag,
                frequency: selection.frequency,
                in_reference: selection.in_reference,
                marks: selection
                    .marks
                    .iter()
                    .map(|mark| StabilityMark {
                        at_source: mark_name(mark.at_a).to_owned(),
                        at_target: mark_name(mark.at_b).to_owned(),
                        frequency: mark.frequency,
                    })
                    .collect(),
            }
        })
        .collect();
    let structures = table
        .structures
        .iter()
        .map(|selection| StabilityStructure {
            frequency: selection.frequency,
            count: selection.count,
            edges: selection
                .structure
                .edges()
                .iter()
                .map(|edge| graph_edge(&names, edge))
                .collect(),
        })
        .collect();
    let reference_edges =
        table.reference.edges().iter().map(|edge| graph_edge(&names, edge)).collect();
    StructureStabilityResult {
        algorithm_id: "bootstrap_stability".to_owned(),
        base_algorithm: base.to_owned(),
        structure_class: table.class.as_str().to_owned(),
        variable_names: names,
        replicates_requested: table.requested,
        replicates_succeeded: table.succeeded,
        replicates_failed: table.failed,
        reference_frequency: table.reference_frequency,
        reference_edges,
        edges,
        structures,
        route: Some(route),
    }
}

type SharedCi = Arc<dyn ConditionalIndependence + Send + Sync>;

fn named_ci(name: Option<&str>) -> PyResult<(SharedCi, String)> {
    let name = name.unwrap_or("parcorr");
    let label = name.to_ascii_lowercase();
    let ci: SharedCi = if label == "parcorr" {
        Arc::new(PartialCorrelation)
    } else {
        resolve_ci(name, None).map_err(py_err)?
    };
    Ok((ci, label))
}

/// Sampling distribution of a structure search over `replicates` resamples.
///
/// `base` is `pc`, `fci`, `rfci`, `ges`, `direct_lingam`, `notears`, `pcmci`, `pcmci_plus` or
/// `lpcmci`.
/// The static searches read independent rows and resample rows; the temporal searches read one
/// series (resampled in circular blocks) or, with `unit_lengths`, the row-concatenated units
/// of a panel (resampled as whole units). The conditional-independence test is a named test.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    *,
    base="pc",
    replicates=200,
    alpha=0.05,
    fdr=true,
    seed=1,
    ci=None,
    max_cond_size=2,
    max_lag=1,
    screen_pc=false,
    max_subset=None,
    prune_threshold=0.05,
    l1=0.1,
    threshold=0.3,
    standardize=true,
    missing=None,
    threads=None,
    cancel=None,
    unit_lengths=None,
    background_knowledge=None
))]
fn discover_bootstrap_stability(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    base: &str,
    replicates: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<String>,
    max_cond_size: usize,
    max_lag: u32,
    screen_pc: bool,
    max_subset: Option<usize>,
    prune_threshold: f64,
    l1: f64,
    threshold: f64,
    standardize: bool,
    missing: Option<String>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    unit_lengths: Option<Vec<usize>>,
    background_knowledge: Option<String>,
) -> PyResult<StructureStabilityResult> {
    let temporal = matches!(base, "pcmci" | "pcmci_plus" | "lpcmci");
    if !temporal && !matches!(base, "pc" | "fci" | "rfci" | "ges" | "direct_lingam" | "notears") {
        return Err(crate::value_err(format!(
            "unknown base {base:?}; expected pc|fci|rfci|ges|direct_lingam|notears|pcmci|pcmci_plus|lpcmci"
        )));
    }
    if !temporal && unit_lengths.is_some() {
        return Err(crate::value_err("unit_lengths applies to the temporal searches only"));
    }
    let takes_knowledge = !matches!(base, "direct_lingam" | "notears" | "lpcmci");
    if !takes_knowledge && background_knowledge.is_some() {
        return Err(crate::value_err(format!(
            "{base} takes no background knowledge; remove background_knowledge"
        )));
    }
    let knowledge = if takes_knowledge {
        knowledge_arg(background_knowledge.as_deref())?
    } else {
        BackgroundKnowledge::default()
    };
    let missing = missing.as_deref().map(crate::missing_report::parse_policy).transpose()?;
    let (ci_impl, ci_label) = named_ci(ci.as_deref())?;
    let ingested = crate::columns_to_ingested(&names, &columns, unit_lengths.as_deref())?;
    drop(columns);
    let base = base.to_owned();

    detach_catch(py, move || {
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let static_params = |ci: SharedCi, alpha: f64, fdr: bool| StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci,
            screen_pc,
            max_subset,
            knowledge: knowledge.clone(),
        };
        let (spec, input, variables) = if temporal {
            let (input, variables) = match ingested {
                crate::IngestedColumns::Panel(batches) => {
                    let (units, variables) = panel_units_from_unit_batches(&batches)?;
                    (DiscoverInput::Panel(units), variables)
                }
                crate::IngestedColumns::Series(batch) => {
                    let (series, variables) = crate::series_from_batch(&batch)?;
                    (DiscoverInput::Series(series), variables)
                }
            };
            let params = DiscoverParams {
                max_lag,
                alpha,
                fdr: fdr.then(FdrAdjustment::bh),
                ci: ci_impl,
                multi_dataset: MultiDatasetConstraints::default(),
                max_cond_size,
                knowledge: knowledge.clone(),
            };
            let spec = match base.as_str() {
                "pcmci" => DiscoverySpec::pcmci(params, ci_label),
                "lpcmci" => DiscoverySpec::lpcmci(params, ci_label),
                _ => DiscoverySpec::pcmci_plus(params, ci_label),
            };
            (spec, input, variables)
        } else {
            let (data, variables) = tabular_from_ingested(ingested)?;
            let spec = match base.as_str() {
                "pc" => DiscoverySpec::pc(static_params(ci_impl, alpha, fdr), ci_label),
                "fci" => DiscoverySpec::fci(static_params(ci_impl, alpha, fdr), ci_label),
                "rfci" => DiscoverySpec::rfci(static_params(ci_impl, alpha, fdr), ci_label),
                "ges" => DiscoverySpec::ges(static_params(ci_impl, alpha, fdr), ci_label),
                "direct_lingam" => DiscoverySpec::lingam(
                    static_params(Arc::new(PartialCorrelation), 0.05, false),
                    prune_threshold,
                    "parcorr",
                ),
                _ => DiscoverySpec::notears(
                    static_params(Arc::new(PartialCorrelation), 0.05, false),
                    l1,
                    threshold,
                    standardize,
                    missing,
                    "parcorr",
                ),
            };
            (spec, DiscoverInput::Tabular(data), variables)
        };
        let (table, route) = compiled_search(
            spec.bootstrap_stability(replicates),
            input,
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::Stability(table) => Some(table),
                _ => None,
            },
        )?;
        Ok(result_of(names, &base, &table, route))
    })
}

/// One table per labelled regime of a regime-PCMCI stability search.
#[pyclass]
pub(crate) struct RegimeStabilityResult {
    /// Regime ids of the full-sample run, in table order.
    #[pyo3(get)]
    regimes: Vec<u32>,
    /// The stability table of each regime.
    #[pyo3(get)]
    tables: Vec<StructureStabilityResult>,
}

#[pymethods]
impl RegimeStabilityResult {
    /// The table of `regime`.
    fn table(&self, regime: u32) -> PyResult<StructureStabilityResult> {
        self.regimes
            .iter()
            .position(|&r| r == regime)
            .map(|at| self.tables[at].clone())
            .ok_or_else(|| crate::value_err(format!("no table for regime {regime}")))
    }

    fn __repr__(&self) -> String {
        format!("RegimeStabilityResult(regimes={:?})", self.regimes)
    }
}

/// Sampling distribution of J-PCMCI+ over circular-block resamples of every environment.
///
/// `env_columns[e]` holds the columns of environment `e` in the order of `names`;
/// `context_names` lists the context variables, excluded from the searched system.
#[pyfunction]
#[pyo3(signature = (
    names,
    env_columns,
    *,
    replicates=200,
    max_lag=1,
    alpha=0.05,
    fdr=true,
    seed=1,
    ci=None,
    threads=None,
    cancel=None,
    context_names=None,
    max_cond_size=2
))]
fn discover_jpcmci_plus_stability(
    py: Python<'_>,
    names: Vec<String>,
    env_columns: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
    replicates: u32,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<String>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    context_names: Option<Vec<String>>,
    max_cond_size: usize,
) -> PyResult<StructureStabilityResult> {
    if env_columns.is_empty() {
        return Err(crate::value_err("J-PCMCI+ stability needs at least one environment"));
    }
    let mut batches = Vec::with_capacity(env_columns.len());
    for cols in &env_columns {
        batches.push(columns_to_batch(&names, cols)?);
    }
    let (ci_impl, ci_label) = named_ci(ci.as_deref())?;
    let context_names = context_names.unwrap_or_default();
    drop(env_columns);
    detach_catch(py, move || {
        let mut series_list = Vec::with_capacity(batches.len());
        let mut all_variables = Vec::new();
        for (index, batch) in batches.iter().enumerate() {
            let (series, variables) = crate::series_from_batch(batch)?;
            if index == 0 {
                all_variables = variables;
            }
            series_list.push(series);
        }
        let multi = antecedent_data::MultiEnvironmentData::try_new(Arc::from(series_list))
            .map_err(py_err)?;
        let mut context_ids = Vec::new();
        for name in &context_names {
            let Some(at) = names.iter().position(|n| n == name) else {
                return Err(crate::value_err(format!("context_names entry '{name}' not found")));
            };
            context_ids.push(all_variables[at]);
        }
        let system: Vec<VariableId> =
            all_variables.iter().copied().filter(|v| !context_ids.contains(v)).collect();
        if system.is_empty() {
            return Err(crate::value_err("J-PCMCI+ stability needs a system variable"));
        }
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints {
                context_variables: Arc::from(context_ids),
                ..MultiDatasetConstraints::default()
            },
            max_cond_size,
            knowledge: BackgroundKnowledge::default(),
        };
        let ctx =
            crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
        let (table, route) = compiled_search(
            DiscoverySpec::jpcmci_plus(params, ci_label).bootstrap_stability(replicates),
            DiscoverInput::MultiEnv(multi),
            &system,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::Stability(table) => Some(table),
                _ => None,
            },
        )?;
        Ok(result_of(names, "jpcmci_plus", &table, route))
    })
}

/// Sampling distribution of regime PCMCI over circular-block resamples that carry their
/// regime labels, one table per regime.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    *,
    regimes,
    replicates=200,
    max_lag=1,
    alpha=0.05,
    fdr=true,
    seed=1,
    ci=None,
    threads=None,
    cancel=None,
    max_cond_size=2,
    min_regime_len=None
))]
fn discover_rpcmci_stability(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    regimes: Vec<u32>,
    replicates: u32,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<String>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
    max_cond_size: usize,
    min_regime_len: Option<usize>,
) -> PyResult<RegimeStabilityResult> {
    let (ci_impl, ci_label) = named_ci(ci.as_deref())?;
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (series, variables) = crate::series_from_batch(&batch)?;
        let labels: Vec<antecedent_core::RegimeId> =
            regimes.iter().map(|&r| antecedent_core::RegimeId::from_raw(r)).collect();
        let assignment = antecedent::discovery::RegimeAssignment::try_new(labels)
            .map_err(|error| py_err(antecedent::CausalError::from(error)))?;
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
        let (tables, route) = compiled_search(
            DiscoverySpec::rpcmci(params, assignment, min_regime_len, ci_label)
                .bootstrap_stability(replicates),
            DiscoverInput::Series(series),
            &variables,
            &ctx,
            |structure| match structure {
                DiscoveredStructure::RegimeStability(tables) => Some(tables),
                _ => None,
            },
        )?;
        let regime_ids = tables.iter().map(|(regime, _)| regime.raw()).collect();
        let tables = tables
            .iter()
            .map(|(_, table)| result_of(names.clone(), "rpcmci", table, route.clone()))
            .collect();
        Ok(RegimeStabilityResult { regimes: regime_ids, tables })
    })
}

/// Register the stability search and its result types on the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StructureStabilityResult>()?;
    m.add_class::<StabilityEdge>()?;
    m.add_class::<StabilityMark>()?;
    m.add_class::<StabilityStructure>()?;
    m.add_class::<RegimeStabilityResult>()?;
    m.add_function(wrap_pyfunction!(discover_bootstrap_stability, m)?)?;
    m.add_function(wrap_pyfunction!(discover_jpcmci_plus_stability, m)?)?;
    m.add_function(wrap_pyfunction!(discover_rpcmci_stability, m)?)?;
    Ok(())
}
