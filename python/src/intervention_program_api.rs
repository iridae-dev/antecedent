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
    PolicySupport, PolicyValue, PreparedPolicy, RouteDecision, TemporalPlan,
    counterfactual_policy as facade_counterfactual_policy,
    counterfactual_temporal_policy as facade_counterfactual_temporal_policy,
    policy_support as facade_policy_support, policy_value as facade_policy_value,
    sample_do as facade_sample_do, sample_temporal_policy as facade_sample_temporal_policy,
};
use antecedent_core::{
    CausalRng, Intervention, InterventionSequence, SequencedIntervention, TemporalPolicy,
};
use antecedent_data::TableView;
use antecedent_io::IoError;
use antecedent_io::policy_program_artifact as artifact;
use antecedent_model::ValueBatch;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyDictMethods};

use crate::gcm_api::{PyFittedGcm, value_batch_to_sample_result};
use crate::{
    GcmSampleResult, detach_catch, py_err, py_execution_context, py_msg, refusal,
    resolve_user_threads, tabular_from_numpy, value_err,
};

/// A library refusal keeps its registered reason code; a malformed program is a value error.
pub(crate) fn io_err(error: IoError) -> PyErr {
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

pub(crate) fn index_resolver(names: &[String]) -> impl Fn(&str) -> Option<u32> + '_ {
    move |name| names.iter().position(|n| n == name).and_then(|i| u32::try_from(i).ok())
}

fn run_sample(
    py: Python<'_>,
    fitted: &PyFittedGcm,
    interventions: Vec<Intervention>,
    n: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<GcmSampleResult> {
    let inner = Arc::clone(&fitted.inner);
    let batch = detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        let mut rng = CausalRng::from_seed(seed);
        facade_sample_do(&inner.model, &interventions, n, &mut rng, &ctx).map_err(py_err)
    })?;
    value_batch_to_sample_result(py, batch)
}

fn run_counterfactual<'py>(
    py: Python<'py>,
    fitted: &PyFittedGcm,
    interventions: Vec<Intervention>,
    outcomes: &[String],
    seed: u64,
    threads: Option<u32>,
) -> PyResult<Bound<'py, PyDict>> {
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

/// One route's decision as `{licensed, assumption, reason_code, detail, message}`.
fn decision_dict<'py>(py: Python<'py>, decision: &RouteDecision) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("licensed", decision.is_licensed())?;
    out.set_item("assumption", decision.assumption())?;
    let (code, detail, message) = match decision {
        RouteDecision::Refused { code, detail, message } => {
            (Some(code.as_str()), Some(detail.as_str()), Some(message.as_str()))
        }
        _ => (None, None, None),
    };
    out.set_item("reason_code", code)?;
    out.set_item("detail", detail)?;
    out.set_item("message", message)?;
    Ok(out)
}

fn support_dict<'py>(py: Python<'py>, support: &PolicySupport) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("simulation", decision_dict(py, &support.simulation)?)?;
    out.set_item("counterfactual", decision_dict(py, &support.counterfactual)?)?;
    out.set_item("identification", decision_dict(py, &support.identification)?)?;
    Ok(out)
}

/// A policy value as `{outcome, n_draws, policy_mean, ..., support}`.
fn value_dict<'py>(
    py: Python<'py>,
    outcome: &str,
    value: &PolicyValue,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("outcome", outcome)?;
    out.set_item("n_draws", value.n_draws)?;
    out.set_item("policy_mean", value.policy_mean)?;
    out.set_item("baseline_mean", value.baseline_mean)?;
    out.set_item("mean_difference", value.mean_difference)?;
    out.set_item("policy_simulation_stderr", value.policy_simulation_stderr)?;
    out.set_item("difference_simulation_stderr", value.difference_simulation_stderr)?;
    out.set_item("support", support_dict(py, &value.support)?)?;
    Ok(out)
}

fn run_value<'py>(
    py: Python<'py>,
    fitted: &PyFittedGcm,
    interventions: Vec<Intervention>,
    outcome: &str,
    n: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<Bound<'py, PyDict>> {
    let id = fitted.data.schema().id_of(outcome).map_err(py_err)?;
    let inner = Arc::clone(&fitted.inner);
    let value = detach_catch(py, move || {
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        facade_policy_value(&inner.model, &interventions, id, n, &ctx).map_err(py_err)
    })?;
    value_dict(py, outcome, &value)
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
    run_sample(py, &fitted, programs_of(&fitted, policies)?, n, seed, threads)
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
    run_counterfactual(py, &fitted, programs_of(&fitted, policies)?, &outcomes, seed, threads)
}

/// The licence of each route (simulation, counterfactual execution, observational
/// identification) for a program on the fitted model, with the reason code of every refusal.
#[pyfunction]
pub(crate) fn policy_support<'py>(
    py: Python<'py>,
    fitted: PyRef<'py, PyFittedGcm>,
    policies: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let interventions = programs_of(&fitted, policies)?;
    support_dict(py, &facade_policy_support(&fitted.inner.model, &interventions))
}

/// Model-based value of a program on one outcome, against the same noise unintervened.
#[pyfunction]
#[pyo3(signature = (fitted, policies, outcome, n, seed, threads=None))]
pub(crate) fn policy_value<'py>(
    py: Python<'py>,
    fitted: PyRef<'py, PyFittedGcm>,
    policies: &str,
    outcome: &str,
    n: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<Bound<'py, PyDict>> {
    run_value(py, &fitted, programs_of(&fitted, policies)?, outcome, n, seed, threads)
}

/// A program compiled and licensed once against one fitted model, rerun on any route without
/// being compiled again: the retained handle of the policy workflow. Every method calls the
/// same executor as the free functions; `value` runs it on the retained compiled overlay.
#[pyclass(name = "PreparedPolicy", frozen)]
pub struct PyPreparedPolicy {
    fitted: Py<PyFittedGcm>,
    core: PreparedPolicy,
    identity: Option<String>,
}

#[pymethods]
impl PyPreparedPolicy {
    #[new]
    fn new(py: Python<'_>, fitted: Py<PyFittedGcm>, policies: &str) -> PyResult<Self> {
        let interventions = programs_of(&fitted.bind(py).borrow(), policies)?;
        // A runtime-bound program has no identity; every other program does.
        let identity = artifact::policy_program_identity(&interventions)
            .ok()
            .map(|digest| blake3::Hash::from_bytes(digest).to_hex().to_string());
        let core = PreparedPolicy::prepare(&fitted.bind(py).borrow().inner.model, interventions)
            .map_err(py_err)?;
        Ok(Self { fitted, core, identity })
    }

    /// Hex identity of the program, or `None` for a program that cannot leave the process.
    #[getter]
    fn identity(&self) -> Option<String> {
        self.identity.clone()
    }

    /// How many times the program was compiled: one, however often it is rerun or rebound.
    #[getter]
    fn program_compilations(&self) -> u64 {
        self.core.program_compilations()
    }

    /// How many fitted models the program's overlay and licence were resolved against.
    #[getter]
    fn model_bindings(&self) -> u64 {
        self.core.model_bindings()
    }

    /// The licence of each route, fixed when the policy was prepared.
    fn support<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        support_dict(py, self.core.support())
    }

    /// The same program bound to another fitted model (for example one refitted on refreshed
    /// data) without compiling the program again.
    fn rebind(&self, py: Python<'_>, fitted: Py<PyFittedGcm>) -> PyResult<Self> {
        // The program names variables by position in the model it was prepared on.
        if fitted.bind(py).borrow().names != self.fitted.bind(py).borrow().names {
            return Err(value_err(
                "a policy rebinds only to a model over the same variables in the same order",
            ));
        }
        let core = self.core.rebind(&fitted.bind(py).borrow().inner.model).map_err(py_err)?;
        Ok(Self { fitted, core, identity: self.identity.clone() })
    }

    #[pyo3(signature = (n, seed, threads=None))]
    fn sample(
        &self,
        py: Python<'_>,
        n: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<GcmSampleResult> {
        let fitted = self.fitted.bind(py).borrow();
        run_sample(py, &fitted, self.core.interventions().to_vec(), n, seed, threads)
    }

    #[pyo3(signature = (outcomes, seed, threads=None))]
    fn counterfactual<'py>(
        &self,
        py: Python<'py>,
        outcomes: Vec<String>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let fitted = self.fitted.bind(py).borrow();
        run_counterfactual(
            py,
            &fitted,
            self.core.interventions().to_vec(),
            &outcomes,
            seed,
            threads,
        )
    }

    #[pyo3(signature = (outcome, n, seed, threads=None))]
    fn value<'py>(
        &self,
        py: Python<'py>,
        outcome: &str,
        n: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let fitted = self.fitted.bind(py).borrow();
        let id = fitted.data.schema().id_of(outcome).map_err(py_err)?;
        let inner = Arc::clone(&fitted.inner);
        let core = self.core.clone();
        let value = detach_catch(py, move || {
            let ctx = py_execution_context(seed, resolve_user_threads(threads));
            core.value(&inner.model, id, n, &ctx).map_err(py_err)
        })?;
        value_dict(py, outcome, &value)
    }
}

/// Earlier steps a lagged read may see, most recent first, as batches over every model variable.
fn initial_history(
    fitted: &PyFittedGcm,
    history: Option<Vec<HashMap<String, Vec<f64>>>>,
    n_rows: usize,
) -> PyResult<Vec<ValueBatch>> {
    let n_nodes = fitted.inner.model.n_nodes();
    let mut initial = Vec::new();
    for step in history.iter().flatten() {
        let mut values = vec![0.0; n_rows.saturating_mul(n_nodes)];
        let mut seen = 0usize;
        for (name, column) in step {
            let id = fitted.data.schema().id_of(name).map_err(py_err)?;
            let dense = fitted
                .inner
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
    Ok(initial)
}

/// The program as a schedule active from step `start` to the end of the rollout.
fn sustained_schedule(
    interventions: Vec<Intervention>,
    start: i32,
    n_steps: usize,
) -> InterventionSequence {
    let last = i32::try_from(n_steps).unwrap_or(i32::MAX);
    InterventionSequence::new(
        interventions
            .into_iter()
            .map(|iv| SequencedIntervention::new(iv, TemporalPolicy::sustained(start, last)))
            .collect::<Vec<_>>(),
    )
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
    let initial = initial_history(&fitted, history, n_rows)?;
    let batches = detach_catch(py, move || {
        let schedule = sustained_schedule(interventions, start, n_steps);
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

/// Replay a factual panel under a program that may read earlier steps.
///
/// `steps[t]` maps every model variable to the factual values of the same units at step `t`.
/// Each step's exogenous noise is abduced from its table and replayed through the same
/// rollout executor as `sample_temporal_policy`, so a lagged read sees the counterfactual
/// earlier steps (`history`, most recent first, supplies steps before the panel starts).
#[pyfunction]
#[pyo3(signature = (fitted, policies, steps, window, seed, start=0, history=None, threads=None))]
#[allow(clippy::too_many_arguments, reason = "one keyword per rollout parameter")]
pub(crate) fn counterfactual_temporal_policy(
    py: Python<'_>,
    fitted: PyRef<'_, PyFittedGcm>,
    policies: &str,
    steps: Vec<Bound<'_, PyDict>>,
    window: u32,
    seed: u64,
    start: i32,
    history: Option<Vec<HashMap<String, Vec<f64>>>>,
    threads: Option<u32>,
) -> PyResult<Vec<GcmSampleResult>> {
    let interventions = programs_of(&fitted, policies)?;
    let tables = steps
        .iter()
        .map(|step| {
            let columns = fitted
                .names
                .iter()
                .map(|name| {
                    step.get_item(name)?
                        .ok_or_else(|| {
                            value_err(format!("a panel step does not hold variable {name:?}"))
                        })?
                        .extract::<PyReadonlyArray1<'_, f64>>()
                        .map_err(PyErr::from)
                })
                .collect::<PyResult<Vec<_>>>()?;
            if step.len() != columns.len() {
                return Err(value_err(
                    "a panel step names variables the model does not have; supply exactly the \
                     model's variables",
                ));
            }
            tabular_from_numpy(&fitted.names, &columns)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let n_rows = tables.first().map_or(0, TableView::row_count);
    if n_rows == 0 || tables.iter().any(|table| table.row_count() != n_rows) {
        return Err(value_err("every panel step needs the same, non-zero number of units"));
    }
    let n_steps = tables.len();
    let initial = initial_history(&fitted, history, n_rows)?;
    let inner = Arc::clone(&fitted.inner);
    let batches = detach_catch(py, move || {
        let schedule = sustained_schedule(interventions, start, n_steps);
        let plan = TemporalPlan {
            sequence: &schedule,
            n_steps,
            n_rows,
            window,
            initial_history: &initial,
        };
        let ctx = py_execution_context(seed, resolve_user_threads(threads));
        facade_counterfactual_temporal_policy(inner.model.clone(), &tables, &plan, &ctx)
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
    m.add_function(wrap_pyfunction!(counterfactual_temporal_policy, m)?)?;
    m.add_function(wrap_pyfunction!(policy_support, m)?)?;
    m.add_function(wrap_pyfunction!(policy_value, m)?)?;
    m.add_class::<PyPreparedPolicy>()?;
    m.add_function(wrap_pyfunction!(encode_policy_program, m)?)?;
    m.add_function(wrap_pyfunction!(decode_policy_program, m)?)?;
    m.add_function(wrap_pyfunction!(policy_program_identity, m)?)
}
