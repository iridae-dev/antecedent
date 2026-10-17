//! Closed-form intervention programs from Python.
//!
//! A program is the JSON of `antecedent.intervention_programs`: transforms and assignments
//! whose expressions name variables. It is parsed by the library
//! (`antecedent_io::policy_program_artifact`), compiled by the same overlay and executed by
//! the same per-node executor as every Rust route: forward sampling, counterfactual execution
//! on abduced noise, and the temporal rollout that reads earlier steps. Nothing here
//! evaluates an expression.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent::gcm::{
    TemporalPlan, counterfactual_policy as facade_counterfactual_policy,
    sample_do as facade_sample_do, sample_temporal_policy as facade_sample_temporal_policy,
};
use antecedent_core::{
    CausalRng, Intervention, InterventionSequence, SequencedIntervention, TemporalPolicy,
};
use antecedent_data::TableView;
use antecedent_io::IoError;
use antecedent_io::policy_program_artifact as artifact;
use antecedent_model::ValueBatch;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::gcm_api::{PyFittedGcm, value_batch_to_sample_result};
use crate::{
    GcmSampleResult, detach_catch, py_err, py_execution_context, py_msg, refusal,
    resolve_user_threads, value_err,
};

/// A library refusal keeps its registered reason code; a malformed program is a value error.
fn io_err(error: IoError) -> PyErr {
    match error {
        IoError::Refused { code, message } => refusal(code, message),
        IoError::Convert(message) => value_err(message),
        other => py_msg(other),
    }
}

fn programs_of(fitted: &PyFittedGcm, text: &str) -> PyResult<Vec<Intervention>> {
    let schema = fitted.data.schema();
    artifact::interventions_from_named_json(text, &|name| {
        schema.id_of(name).ok().map(antecedent_core::VariableId::raw)
    })
    .map_err(io_err)
}

fn index_resolver(names: &[String]) -> impl Fn(&str) -> Option<u32> + '_ {
    move |name| names.iter().position(|n| n == name).and_then(|i| u32::try_from(i).ok())
}

/// Sample the fitted model under a program (forward ancestral sampling).
#[pyfunction]
#[pyo3(signature = (fitted, policies, n, seed, threads=None))]
pub(crate) fn sample_policy(
    py: Python<'_>,
    fitted: PyRef<'_, PyFittedGcm>,
    policies: &str,
    n: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<GcmSampleResult> {
    let interventions = programs_of(&fitted, policies)?;
    let inner = Arc::clone(&fitted.inner);
    let batch = detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        let mut rng = CausalRng::from_seed(seed);
        facade_sample_do(&inner.model, &interventions, n, &mut rng, &ctx).map_err(py_err)
    })?;
    value_batch_to_sample_result(py, batch)
}

/// Replay the fitted model's factual rows under a program on abduced noise.
///
/// Returns the counterfactual value of each requested outcome per row.
#[pyfunction]
#[pyo3(signature = (fitted, policies, outcomes, seed, threads=None))]
pub(crate) fn counterfactual_policy<'py>(
    py: Python<'py>,
    fitted: PyRef<'py, PyFittedGcm>,
    policies: &str,
    outcomes: Vec<String>,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<Bound<'py, PyDict>> {
    let interventions = programs_of(&fitted, policies)?;
    let ids = outcomes
        .iter()
        .map(|name| fitted.data.schema().id_of(name).map_err(py_err))
        .collect::<PyResult<Vec<_>>>()?;
    let inner = Arc::clone(&fitted.inner);
    let data = Arc::clone(&fitted.data);
    let columns = detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        let result =
            facade_counterfactual_policy(inner.model.clone(), &data, &interventions, &ids, &ctx)
                .map_err(py_err)?;
        ids.iter()
            .map(|id| {
                let dense = inner
                    .model
                    .dense_of(*id)
                    .ok_or_else(|| value_err(format!("outcome {id} is not in the model")))?;
                Ok((0..result.n_units).map(|unit| result.get(0, dense, unit)).collect::<Vec<_>>())
            })
            .collect::<PyResult<Vec<_>>>()
    })?;
    let out = PyDict::new(py);
    for (name, column) in outcomes.iter().zip(columns) {
        out.set_item(name, PyArray1::from_vec(py, column))?;
    }
    Ok(out)
}

/// Roll the fitted model out over time under a program active from step `start`; a lagged
/// read sees up to `window` earlier steps (`history`, most recent first, supplies steps
/// before time zero).
#[pyfunction]
#[pyo3(signature = (fitted, policies, n_rows, n_steps, window, seed, start=0, history=None, threads=None))]
#[allow(clippy::too_many_arguments, reason = "one keyword per rollout parameter")]
pub(crate) fn sample_temporal_policy(
    py: Python<'_>,
    fitted: PyRef<'_, PyFittedGcm>,
    policies: &str,
    n_rows: usize,
    n_steps: usize,
    window: u32,
    seed: u64,
    start: i32,
    history: Option<Vec<HashMap<String, Vec<f64>>>>,
    threads: Option<u32>,
) -> PyResult<Vec<GcmSampleResult>> {
    let interventions = programs_of(&fitted, policies)?;
    let inner = Arc::clone(&fitted.inner);
    let n_nodes = inner.model.n_nodes();
    let mut initial = Vec::new();
    for step in history.iter().flatten() {
        let mut values = vec![0.0; n_rows.saturating_mul(n_nodes)];
        let mut seen = 0usize;
        for (name, column) in step {
            let id = fitted.data.schema().id_of(name).map_err(py_err)?;
            let dense = inner
                .model
                .dense_of(id)
                .ok_or_else(|| value_err(format!("history variable {name:?} is not in the model")))?
                .as_usize();
            if column.len() != n_rows {
                return Err(value_err(format!(
                    "history column {name:?} has {} values for {n_rows} rows",
                    column.len()
                )));
            }
            values[dense * n_rows..(dense + 1) * n_rows].copy_from_slice(column);
            seen += 1;
        }
        if seen != n_nodes {
            return Err(value_err(format!(
                "a history step names {seen} of the model's {n_nodes} variables; supply every one"
            )));
        }
        initial.push(ValueBatch { n_rows, n_nodes, values: Arc::from(values) });
    }
    let batches = detach_catch(py, move || {
        let last = i32::try_from(n_steps).unwrap_or(i32::MAX);
        let schedule = InterventionSequence::new(
            interventions
                .into_iter()
                .map(|iv| SequencedIntervention::new(iv, TemporalPolicy::sustained(start, last)))
                .collect::<Vec<_>>(),
        );
        let plan = TemporalPlan {
            sequence: &schedule,
            n_steps,
            n_rows,
            window,
            initial_history: &initial,
        };
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        let mut rng = CausalRng::from_seed(seed);
        facade_sample_temporal_policy(&inner.model, &plan, &mut rng, &ctx)
            .map(|path| path.steps)
            .map_err(py_err)
    })?;
    batches.into_iter().map(|batch| value_batch_to_sample_result(py, batch)).collect()
}

/// Encode a program (variables spelled by `names`) as a versioned portable artifact.
#[pyfunction]
pub(crate) fn encode_policy_program<'py>(
    py: Python<'py>,
    names: Vec<String>,
    policies: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let interventions = artifact::interventions_from_named_json(policies, &index_resolver(&names))
        .map_err(io_err)?;
    let bytes = artifact::encode_policy_program(&interventions).map_err(io_err)?;
    Ok(PyBytes::new(py, &bytes))
}

/// Decode and verify a program artifact back to named-variable program text.
#[pyfunction]
pub(crate) fn decode_policy_program(names: Vec<String>, data: &[u8]) -> PyResult<String> {
    let interventions = artifact::decode_policy_program(data).map_err(io_err)?;
    artifact::named_json_from_interventions(&interventions, &names).map_err(io_err)
}

/// Identity (hex BLAKE3) of a program.
#[pyfunction]
pub(crate) fn policy_program_identity(names: Vec<String>, policies: &str) -> PyResult<String> {
    let interventions = artifact::interventions_from_named_json(policies, &index_resolver(&names))
        .map_err(io_err)?;
    let digest = artifact::policy_program_identity(&interventions).map_err(io_err)?;
    Ok(blake3::Hash::from_bytes(digest).to_hex().to_string())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sample_policy, m)?)?;
    m.add_function(wrap_pyfunction!(counterfactual_policy, m)?)?;
    m.add_function(wrap_pyfunction!(sample_temporal_policy, m)?)?;
    m.add_function(wrap_pyfunction!(encode_policy_program, m)?)?;
    m.add_function(wrap_pyfunction!(decode_policy_program, m)?)?;
    m.add_function(wrap_pyfunction!(policy_program_identity, m)?)
}
