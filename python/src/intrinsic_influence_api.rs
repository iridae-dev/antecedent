//! Python bridge for intrinsic causal influence (variance attribution to noise terms).
//!
//! Python supplies the columns, the edges and the settings; the mechanism fit, the nested
//! simulation, the Shapley allocation, every refusal and the three-component uncertainty report
//! are Rust's. The result crosses as JSON.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::gcm::{
    ComponentStatus, IntrinsicAllocation, IntrinsicInfluenceQuery, IntrinsicInfluenceResult,
    NoiseIndependence, ShapleyMethod, UncertaintyComponent, attribute_intrinsic_influence, fit_gcm,
};
use antecedent_core::{ComponentId, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_model::CompiledCausalModel;
use pyo3::prelude::*;

use crate::{detach_catch, py_err, py_execution_context};

fn source_name(names: &[String], id: ComponentId) -> String {
    names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("V{}", id.raw()))
}

fn status_json(status: &ComponentStatus) -> serde_json::Value {
    match status {
        ComponentStatus::Quantified { stderr, ci_low, ci_high, level, df } => serde_json::json!({
            "status": "quantified",
            "stderr": stderr,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "level": level,
            "df": df,
        }),
        ComponentStatus::NotApplicable { reason } => {
            serde_json::json!({ "status": "not_applicable", "stderr": 0.0, "reason": reason })
        }
        ComponentStatus::Unquantified { reason } => {
            serde_json::json!({ "status": "unquantified", "reason": reason })
        }
        _ => serde_json::json!({ "status": "unquantified", "reason": "unrecognized_status" }),
    }
}

fn component_name(component: UncertaintyComponent) -> &'static str {
    match component {
        UncertaintyComponent::PermutationSampling => "permutation_sampling",
        UncertaintyComponent::Simulation => "simulation",
        UncertaintyComponent::MechanismFit => "mechanism_fit",
        _ => "unrecognized_component",
    }
}

fn result_json(result: &IntrinsicInfluenceResult, names: &[String]) -> String {
    let sources: Vec<serde_json::Value> = result
        .sources
        .iter()
        .map(|s| {
            let uncertainty: Vec<serde_json::Value> = s
                .uncertainty
                .iter()
                .map(|u| {
                    let mut entry = status_json(&u.status);
                    entry["component"] = component_name(u.component).into();
                    entry
                })
                .collect();
            serde_json::json!({
                "source": source_name(names, s.source),
                "contribution": s.contribution,
                "uncertainty": uncertainty,
            })
        })
        .collect();
    let shapley = match result.method.shapley {
        ShapleyMethod::Exact => serde_json::json!({ "kind": "exact" }),
        ShapleyMethod::PermutationSampling { n_permutations, seed } => serde_json::json!({
            "kind": "permutation_sampling",
            "n_permutations": n_permutations,
            "seed": seed,
        }),
        _ => serde_json::json!({ "kind": "unrecognized" }),
    };
    serde_json::json!({
        "target": names.get(result.target.as_usize()).cloned()
            .unwrap_or_else(|| format!("V{}", result.target.raw())),
        "sources": sources,
        "total_variance": result.total_variance,
        "total_variance_stderr": result.total_variance_stderr,
        "allocation_residual": result.allocation_residual,
        "allocation_residual_stderr": result.allocation_residual_stderr,
        "level": result.level,
        "method": {
            "game": result.method.game,
            "estimator": result.method.estimator,
            "shapley": shapley,
            "outer_draws": result.method.outer_draws,
            "inner_draws": result.method.inner_draws,
            "replicates": result.method.replicates,
            "fit_bootstrap_replicates": result.method.fit_bootstrap_replicates,
        },
        "budget": {
            "coalition_evaluations": result.budget.coalition_evaluations,
            "rows_replayed": result.budget.rows_replayed,
            "permutations": result.budget.permutations,
            "exact_coalitions": result.budget.exact_coalitions,
            "refits": result.budget.refits,
        },
    })
    .to_string()
}

/// Settings of one attribution, as Python passes them.
pub(crate) struct IntrinsicOptions {
    pub(crate) independent_noise: bool,
    pub(crate) bidirected_edges: Option<Vec<(String, String)>>,
    pub(crate) allocation: IntrinsicAllocation,
    pub(crate) outer_draws: usize,
    pub(crate) inner_draws: usize,
    pub(crate) replicates: usize,
    pub(crate) fit_bootstrap: u32,
    pub(crate) level: f64,
    pub(crate) max_sources: usize,
    pub(crate) seed: u64,
    pub(crate) threads: Option<u32>,
    pub(crate) cancel: Option<crate::PyCancellationToken>,
}

pub(crate) fn allocation_of(
    allocation: &str,
    n_permutations: usize,
) -> PyResult<IntrinsicAllocation> {
    match allocation {
        "auto" => Ok(IntrinsicAllocation::Auto),
        "exact" => Ok(IntrinsicAllocation::Exact),
        "permutation" => Ok(IntrinsicAllocation::Permutation { n_permutations }),
        other => Err(crate::value_err(format!(
            "allocation must be 'auto', 'exact' or 'permutation', got {other:?}"
        ))),
    }
}

/// Attribute the variance of `target` under an already fitted `model`; `edges` are the model's
/// directed edges by name (needed only to place declared bidirected edges), and `data` the rows
/// the mechanism-fit bootstrap resamples.
pub(crate) fn attribute(
    model: &CompiledCausalModel,
    data: &TabularData,
    names: &[String],
    edges: &[(String, String)],
    target: &str,
    options: IntrinsicOptions,
) -> PyResult<String> {
    let target_id: VariableId = data.schema().id_of(target).map_err(py_err)?;
    let structure = options
        .bidirected_edges
        .as_ref()
        .map(|pairs| -> PyResult<Admg> {
            let directed: Vec<(&str, &str)> =
                edges.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
            let mut admg = Admg::from_named_edges(data.schema(), &directed).map_err(py_err)?;
            for (a, b) in pairs {
                let dense = |name: &String| -> PyResult<DenseNodeId> {
                    let id = data.schema().id_of(name).map_err(py_err)?;
                    Ok(DenseNodeId::from_raw(id.raw()))
                };
                admg.insert_bidirected(dense(a)?, dense(b)?).map_err(py_err)?;
            }
            Ok(admg)
        })
        .transpose()?;
    let noise = match (&structure, options.independent_noise) {
        (Some(admg), _) => NoiseIndependence::Structure(admg),
        (None, true) => NoiseIndependence::Declared,
        (None, false) => NoiseIndependence::Undeclared,
    };
    let query = IntrinsicInfluenceQuery::new(target_id)
        .with_allocation(options.allocation)
        .with_simulation(options.outer_draws, options.inner_draws, options.replicates)
        .with_fit_bootstrap(options.fit_bootstrap)
        .with_level(options.level)
        .with_max_sources(options.max_sources)
        .with_seed(options.seed);
    let mut ctx = py_execution_context(options.seed, crate::resolve_user_threads(options.threads));
    crate::apply_cancel(&mut ctx, options.cancel);
    let result =
        attribute_intrinsic_influence(model, noise, Some(data), &query, &ctx).map_err(py_err)?;
    Ok(result_json(&result, names))
}

/// Intrinsic causal influence of the noise terms upstream of `target` on its variance.
///
/// The mechanisms are fitted to the data on `edges`; the attribution is then Rust's nested Monte
/// Carlo Shapley. Returns the result as JSON.
#[pyfunction]
#[pyo3(
    name = "attribute_intrinsic_influence",
    signature = (names, columns, edges, target, *, independent_noise=false, bidirected_edges=None, allocation="auto", n_permutations=200, outer_draws=200, inner_draws=40, replicates=4, fit_bootstrap=0, level=0.95, max_sources=64, seed=0, threads=None, cancel=None)
)]
fn attribute_intrinsic_influence_py(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    target: String,
    independent_noise: bool,
    bidirected_edges: Option<Vec<(String, String)>>,
    allocation: &str,
    n_permutations: usize,
    outer_draws: usize,
    inner_draws: usize,
    replicates: usize,
    fit_bootstrap: u32,
    level: f64,
    max_sources: usize,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let allocation = allocation_of(allocation, n_permutations)?;
    let (data, _) = crate::tabular_from_py_columns(names.clone(), columns)?;
    detach_catch(py, move || {
        let graph = crate::dag_from_named_edges(data.schema(), &edges)?;
        let fitted = fit_gcm(graph, &data).map_err(py_err)?;
        let options = IntrinsicOptions {
            independent_noise,
            bidirected_edges,
            allocation,
            outer_draws,
            inner_draws,
            replicates,
            fit_bootstrap,
            level,
            max_sources,
            seed,
            threads,
            cancel,
        };
        attribute(&fitted.model, &data, &names, &edges, &target, options)
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(attribute_intrinsic_influence_py, m)?)
}
