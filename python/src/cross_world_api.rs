//! Python bindings for cross-world edge contrasts on a fixed Markovian DAG.
//!
//! The Python layer owns names and argument shapes; the check, the coupled
//! abduction-action-prediction operation and the artifact consumer are the Rust
//! route's, so a Python answer is the Rust answer bit for bit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::cross_world::{
    CrossWorldEffect, CrossWorldOptions, consume_cross_world_artifact, evaluate_cross_world_effect,
};
use antecedent::gcm::NestedOutcomeMechanism;
use antecedent_core::CrossWorldQuery;
use antecedent_data::TableView;
use pyo3::prelude::*;

use crate::{
    dag_from_named_edges, detach_catch, py_err, py_execution_context, tabular_from_py_columns,
};

/// A facade failure as its Python class: a reason-coded refusal is the registered
/// unsupported class (a malformed query the value class) with its `reason_code`,
/// and `variable <id>` is spelled with the variable's name.
fn refused(error: antecedent::CausalError, names: &[String]) -> PyErr {
    let text = error.to_string();
    let Some((code, rest)) = antecedent_core::reason_code::split_prefix(&text) else {
        return py_err(error);
    };
    let mut words: Vec<String> = rest.split(' ').map(str::to_string).collect();
    for i in 1..words.len() {
        if words[i - 1] == "variable" {
            if let Some(name) = words[i].parse::<usize>().ok().and_then(|k| names.get(k)) {
                words[i].clone_from(name);
            }
        }
    }
    let message = words.join(" ");
    if !antecedent_core::reason_code::is_runtime_refusal(code) {
        return py_err(error);
    }
    if code == "invalid_argument" {
        return crate::with_reason_code(crate::value_err(message), code);
    }
    crate::refusal(code, message)
}

/// A malformed query (an unknown variable name, an edge the graph lacks) is the
/// documented `cross_world.invalid_query` detail under `invalid_argument`, the
/// same as the Rust route, not a schema error.
fn invalid_query(message: impl std::fmt::Display) -> PyErr {
    crate::with_reason_code(
        crate::value_err(format!("cross_world.invalid_query: {message}")),
        "invalid_argument",
    )
}

fn mechanism_of(tag: &str) -> PyResult<NestedOutcomeMechanism> {
    match tag {
        "linear_gaussian" => Ok(NestedOutcomeMechanism::LinearGaussian),
        "non_separable_basis" => Ok(NestedOutcomeMechanism::NonSeparableBasis),
        other => Err(crate::value_err(format!(
            "unknown mechanism {other:?}; use \"linear_gaussian\" or \"non_separable_basis\""
        ))),
    }
}

/// The result as JSON (point, per-unit effects, witness, query text) and the
/// exported artifact bytes.
fn payload(effect: &CrossWorldEffect) -> PyResult<(String, Vec<u8>)> {
    let artifact = effect.export_artifact().map_err(py_err)?;
    let json = serde_json::json!({
        "point": effect.point,
        "point_bits": format!("{:016x}", effect.point.to_bits()),
        "unit_effects": effect.unit_effects.as_ref(),
        "data_digest": effect.data_digest,
        "independently_verified": effect.independently_verified,
        "witness": effect.witness,
        "query_text": effect.query.canonical_text(),
        "mechanism": match effect.mechanism {
            NestedOutcomeMechanism::LinearGaussian => "linear_gaussian",
            NestedOutcomeMechanism::NonSeparableBasis => "non_separable_basis",
        },
    });
    Ok((json.to_string(), artifact))
}

/// Evaluate a path-specific edge contrast: `(json, artifact bytes)`.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment, outcome, control, active, intervened_edges, *, mechanism="linear_gaussian", interval_requested=false, seed=0))]
#[allow(clippy::too_many_arguments, reason = "one native entry mirrors the Python signature")]
fn evaluate_cross_world_edge_contrast(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    control: f64,
    active: f64,
    intervened_edges: Vec<(String, String)>,
    mechanism: &str,
    interval_requested: bool,
    seed: u64,
) -> PyResult<(String, Vec<u8>)> {
    let mechanism = mechanism_of(mechanism)?;
    let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
    detach_catch(py, move || {
        let id = |name: &str| {
            data.schema()
                .id_of(name)
                .map_err(|_| invalid_query(format!("unknown variable {name:?}")))
        };
        for (a, b) in edges.iter().chain(&intervened_edges) {
            id(a)?;
            id(b)?;
        }
        let graph = dag_from_named_edges(data.schema(), &edges)?;
        let pair = |(a, b): &(String, String)| Ok((id(a)?, id(b)?));
        let graph_edges = edges.iter().map(pair).collect::<PyResult<Vec<_>>>()?;
        let chosen = intervened_edges.iter().map(pair).collect::<PyResult<Vec<_>>>()?;
        let query = CrossWorldQuery::path_specific(
            id(&treatment)?,
            id(&outcome)?,
            control,
            active,
            &graph_edges,
            &chosen,
        )
        .map_err(invalid_query)?;
        let effect = evaluate_cross_world_effect(
            graph,
            &data,
            &query,
            CrossWorldOptions { mechanism, interval_requested },
            &py_execution_context(seed, 1),
        )
        .map_err(|e| refused(e, &names))?;
        payload(&effect)
    })
}

/// Consume a cross-world artifact by recomputing it: `(json, artifact bytes)`.
#[pyfunction]
#[pyo3(signature = (artifact, *, seed=0))]
fn consume_cross_world_edge_contrast_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    seed: u64,
) -> PyResult<(String, Vec<u8>)> {
    detach_catch(py, move || {
        let effect = consume_cross_world_artifact(&artifact, &py_execution_context(seed, 1))
            .map_err(|e| crate::CausalSerializationError::new_err(e.to_string()))?;
        payload(&effect)
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(evaluate_cross_world_edge_contrast, module)?)?;
    module.add_function(wrap_pyfunction!(consume_cross_world_edge_contrast_artifact, module)?)?;
    Ok(())
}
