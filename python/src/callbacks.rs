//! Slow-path Python callback bridges.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::design::{DesignError, Utility};
use antecedent::gcm::{CompiledCausalModel, DynamicMechanism, MechanismSlot};
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_data::TableView;
use antecedent_graph::DenseNodeId;
use antecedent_model::{MechanismWorkspace, ModelError, ParentBatch};
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependence,
    ConditionalIndependenceTest, PreparedCiTest, StatsError,
};
use antecedent_validate::{
    CustomEffectValidator, RefutationProblem, RefutationReport, ValidationError,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::interrupt::callback_failure;

/// A mechanism callback must return exactly one value per row; extra values would hide a shape
/// bug in the user's function.
fn check_len(what: &str, got: usize, expected: usize) -> Result<(), ModelError> {
    if got == expected {
        Ok(())
    } else {
        Err(ModelError::Shape {
            message: format!("{what} returned {got} values; expected exactly {expected}"),
        })
    }
}

/// Whether the user's `sample_noise` takes an `rng` keyword (a NumPy `Generator` seeded from the
/// run's stream). Without it the draws come from the user's own source and `seed=` does not
/// govern them.
fn accepts_rng(obj: &Bound<'_, PyAny>) -> Result<bool, ModelError> {
    let py = obj.py();
    let unsupported = |e: PyErr| ModelError::Unsupported {
        message: callback_failure("could not inspect sample_noise", e),
    };
    let method = obj.getattr("sample_noise").map_err(unsupported)?;
    let Ok(signature) = py.import("inspect").and_then(|m| m.call_method1("signature", (&method,)))
    else {
        return Ok(false);
    };
    let parameters = signature.getattr("parameters").map_err(unsupported)?;
    parameters.contains("rng").map_err(unsupported)
}

/// Python callable CI test: `(columns, queries) -> list[(statistic, p_value)]`.
pub struct PyConditionalIndependence {
    callback: Py<PyAny>,
}

impl PyConditionalIndependence {
    pub fn new(callback: Py<PyAny>) -> Self {
        Self { callback }
    }
}

impl ConditionalIndependenceTest for PyConditionalIndependence {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let bound = prepared.bind_request(request);
        Python::attach(|py| -> Result<CiBatchResult, StatsError> {
            let cols = PyList::empty(py);
            for col in bound.columns {
                let arr = PyArray1::from_slice(py, col);
                cols.append(arr).map_err(|e| StatsError::Backend(e.to_string()))?;
            }
            let queries = PyList::empty(py);
            for q in bound.queries {
                let z: Vec<usize> = bound.z_flat[q.z_start..q.z_start + q.z_len].to_vec();
                queries.append((q.x, q.y, z)).map_err(|e| StatsError::Backend(e.to_string()))?;
            }
            let out = self.callback.bind(py).call1((cols, queries)).map_err(|e| {
                StatsError::Backend(callback_failure("Python CI callback failed", e))
            })?;
            let seq = out.cast::<PyList>().map_err(|_| StatsError::Shape {
                message: "Python CI callback must return a list of (statistic, p_value)",
            })?;
            if seq.len() != bound.queries.len() {
                return Err(StatsError::Backend(format!(
                    "Python CI callback returned {} results for {} queries",
                    seq.len(),
                    bound.queries.len()
                )));
            }
            let mut results = Vec::with_capacity(seq.len());
            for item in seq.iter() {
                let (stat, p): (f64, f64) = item.extract().map_err(|_| StatsError::Shape {
                    message: "each CI result must be (statistic: float, p_value: float)",
                })?;
                results.push(CiResult { statistic: stat, p_value: p, df: f64::NAN, ci: None });
            }
            Ok(CiBatchResult { results })
        })
    }
}

/// Python mechanism wrapper: `sample_noise(n) -> array`, `evaluate(parents, noise) -> array`.
pub struct PyDynamicMechanism {
    obj: Py<PyAny>,
}

impl PyDynamicMechanism {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
    }
}

impl DynamicMechanism for PyDynamicMechanism {
    fn sample_noise_column(
        &self,
        n_rows: usize,
        rng: &mut CausalRng,
        output: &mut [f64],
    ) -> Result<(), ModelError> {
        Python::attach(|py| -> Result<(), ModelError> {
            let bound = self.obj.bind(py);
            let out = (if accepts_rng(bound)? {
                // The run's seeded stream governs the user's draws through a NumPy generator.
                let generator = py
                    .import("numpy.random")
                    .and_then(|m| m.call_method1("default_rng", (rng.next_u64(),)))
                    .map_err(|e| ModelError::Unsupported {
                        message: callback_failure("failed to seed sample_noise rng", e),
                    })?;
                let kwargs = PyDict::new(py);
                kwargs
                    .set_item("rng", generator)
                    .and_then(|()| bound.call_method("sample_noise", (n_rows,), Some(&kwargs)))
            } else {
                bound.call_method1("sample_noise", (n_rows,))
            })
            .map_err(|e| ModelError::Unsupported {
                message: callback_failure("Python sample_noise failed", e),
            })?;
            let arr: PyReadonlyArray1<'_, f64> = out.extract().map_err(|e| ModelError::Shape {
                message: format!("sample_noise must return float64 ndarray: {e}"),
            })?;
            let slice = arr.as_slice().map_err(|_| ModelError::Shape {
                message: "sample_noise ndarray must be contiguous".into(),
            })?;
            check_len("sample_noise", slice.len(), n_rows)?;
            output[..n_rows].copy_from_slice(slice);
            Ok(())
        })
    }

    fn evaluate_column(
        &self,
        parents: ParentBatch<'_>,
        noise: &[f64],
        output: &mut [f64],
        _workspace: &mut MechanismWorkspace,
    ) -> Result<(), ModelError> {
        let n = parents.n_rows;
        Python::attach(|py| -> Result<(), ModelError> {
            let parent_cols = PyList::empty(py);
            for p in 0..parents.n_parents {
                let col =
                    parents.column(p).map_err(|e| ModelError::Shape { message: e.to_string() })?;
                parent_cols.append(PyArray1::from_slice(py, col)).map_err(|e| {
                    ModelError::Unsupported {
                        message: format!("failed to build parent columns: {e}"),
                    }
                })?;
            }
            let noise_arr = PyArray1::from_slice(py, &noise[..n]);
            let out =
                self.obj.bind(py).call_method1("evaluate", (parent_cols, noise_arr)).map_err(
                    |e| ModelError::Unsupported {
                        message: callback_failure("Python evaluate failed", e),
                    },
                )?;
            let arr: PyReadonlyArray1<'_, f64> = out.extract().map_err(|e| ModelError::Shape {
                message: format!("evaluate must return float64 ndarray: {e}"),
            })?;
            let slice = arr.as_slice().map_err(|_| ModelError::Shape {
                message: "evaluate ndarray must be contiguous".into(),
            })?;
            check_len("evaluate", slice.len(), n)?;
            output[..n].copy_from_slice(slice);
            Ok(())
        })
    }

    fn infer_noise_column(
        &self,
        value: &[f64],
        parents: ParentBatch<'_>,
        output: &mut [f64],
    ) -> Result<(), ModelError> {
        let n = parents.n_rows;
        let py_result = Python::attach(|py| -> Option<Result<(), ModelError>> {
            let bound = self.obj.bind(py);
            match bound.hasattr("infer_noise") {
                Ok(true) => {}
                Ok(false) => return None,
                Err(e) => {
                    return Some(Err(ModelError::Unsupported {
                        message: callback_failure("Python infer_noise lookup failed", e),
                    }));
                }
            }
            Some((|| {
                let parent_cols = PyList::empty(py);
                for p in 0..parents.n_parents {
                    let col = parents
                        .column(p)
                        .map_err(|e| ModelError::Shape { message: e.to_string() })?;
                    parent_cols.append(PyArray1::from_slice(py, col)).map_err(|e| {
                        ModelError::Unsupported {
                            message: format!("failed to build parent columns: {e}"),
                        }
                    })?;
                }
                let value_arr = PyArray1::from_slice(py, &value[..n]);
                let out =
                    bound.call_method1("infer_noise", (value_arr, parent_cols)).map_err(|e| {
                        ModelError::Unsupported {
                            message: callback_failure("Python infer_noise failed", e),
                        }
                    })?;
                let arr: PyReadonlyArray1<'_, f64> =
                    out.extract().map_err(|e| ModelError::Shape {
                        message: format!("infer_noise must return float64 ndarray: {e}"),
                    })?;
                let slice = arr.as_slice().map_err(|_| ModelError::Shape {
                    message: "infer_noise ndarray must be contiguous".into(),
                })?;
                check_len("infer_noise", slice.len(), n)?;
                output[..n].copy_from_slice(slice);
                Ok(())
            })())
        });
        if let Some(r) = py_result {
            r
        } else {
            // Additive default: noise = y − f(pa, 0).
            let zeros = vec![0.0; n];
            let mut mean = vec![0.0; n];
            let mut ws = MechanismWorkspace::default();
            self.evaluate_column(parents, &zeros, &mut mean, &mut ws)?;
            for i in 0..n {
                output[i] = value[i] - mean[i];
            }
            Ok(())
        }
    }

    fn log_prob_column(
        &self,
        values: &[f64],
        parents: ParentBatch<'_>,
        output: &mut [f64],
    ) -> Result<(), ModelError> {
        let n = parents.n_rows;
        let py_result = Python::attach(|py| -> Option<Result<(), ModelError>> {
            let bound = self.obj.bind(py);
            match bound.hasattr("log_prob") {
                Ok(true) => {}
                Ok(false) => return None,
                Err(e) => {
                    return Some(Err(ModelError::Unsupported {
                        message: callback_failure("Python log_prob lookup failed", e),
                    }));
                }
            }
            Some((|| {
                let parent_cols = PyList::empty(py);
                for p in 0..parents.n_parents {
                    let col = parents
                        .column(p)
                        .map_err(|e| ModelError::Shape { message: e.to_string() })?;
                    parent_cols.append(PyArray1::from_slice(py, col)).map_err(|e| {
                        ModelError::Unsupported {
                            message: format!("failed to build parent columns: {e}"),
                        }
                    })?;
                }
                let value_arr = PyArray1::from_slice(py, &values[..n]);
                let out =
                    bound.call_method1("log_prob", (value_arr, parent_cols)).map_err(|e| {
                        ModelError::Unsupported {
                            message: callback_failure("Python log_prob failed", e),
                        }
                    })?;
                let arr: PyReadonlyArray1<'_, f64> =
                    out.extract().map_err(|e| ModelError::Shape {
                        message: format!("log_prob must return float64 ndarray: {e}"),
                    })?;
                let slice = arr.as_slice().map_err(|_| ModelError::Shape {
                    message: "log_prob ndarray must be contiguous".into(),
                })?;
                check_len("log_prob", slice.len(), n)?;
                output[..n].copy_from_slice(slice);
                Ok(())
            })())
        });
        if let Some(r) = py_result {
            r
        } else {
            let mut resid = vec![0.0; n];
            self.infer_noise_column(values, parents, &mut resid)?;
            let log_norm = -0.5 * (2.0 * std::f64::consts::PI).ln();
            for i in 0..n {
                output[i] = log_norm - 0.5 * resid[i] * resid[i];
            }
            Ok(())
        }
    }
}

/// Python utility: `utility(actions, outcomes) -> flat ndarray` length `n_a * n_o`.
pub struct PyUtility {
    callback: Py<PyAny>,
}

impl PyUtility {
    pub fn new(callback: Py<PyAny>) -> Self {
        Self { callback }
    }
}

impl Utility<f64, f64> for PyUtility {
    fn evaluate_batch(
        &self,
        actions: &[f64],
        outcomes: &[f64],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        let expected = actions.len().saturating_mul(outcomes.len());
        if out.len() < expected {
            return Err(DesignError::Shape(format!(
                "utility out buffer {} < {expected}",
                out.len()
            )));
        }
        Python::attach(|py| -> Result<(), DesignError> {
            let a = PyArray1::from_slice(py, actions);
            let o = PyArray1::from_slice(py, outcomes);
            let got =
                self.callback.bind(py).call1((a, o)).map_err(|err| DesignError::Callback {
                    name: "utility".into(),
                    message: callback_failure("utility failed", err),
                })?;
            let arr = match got.extract::<PyReadonlyArray1<'_, f64>>() {
                Ok(arr) => arr,
                Err(err) => {
                    return Err(DesignError::Callback {
                        name: "utility".into(),
                        message: err.to_string(),
                    });
                }
            };
            let slice = arr.as_slice().map_err(|_| DesignError::Callback {
                name: "utility".into(),
                message: "utility return must be contiguous float64 ndarray".into(),
            })?;
            if slice.len() != expected {
                return Err(DesignError::Callback {
                    name: "utility".into(),
                    message: format!(
                        "utility returned {} values; expected exactly {expected}",
                        slice.len()
                    ),
                });
            }
            out[..expected].copy_from_slice(slice);
            Ok(())
        })
    }
}

/// Python custom validator callable.
///
/// Signature: `fn(*, ate, se_analytic, method, adjustment_set) -> dict`
/// with keys `passed`, optional `refuted_ate`, `comparison`, `informative`, `failure_condition`.
pub struct PyCustomValidator {
    name: String,
    callback: Py<PyAny>,
}

impl PyCustomValidator {
    pub fn new(name: impl Into<String>, callback: Py<PyAny>) -> Self {
        Self { name: name.into(), callback }
    }
}

impl CustomEffectValidator for PyCustomValidator {
    fn name(&self) -> &str {
        &self.name
    }

    fn validate(
        &self,
        problem: &RefutationProblem<'_>,
        _ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        Python::attach(|py| -> Result<RefutationReport, ValidationError> {
            let py_err = |e: PyErr| {
                ValidationError::data_msg(callback_failure(
                    &format!("Python validator `{}` failed", self.name),
                    e,
                ))
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("ate", problem.original.ate).map_err(py_err)?;
            kwargs.set_item("se_analytic", problem.original.se_analytic).map_err(py_err)?;
            kwargs.set_item("method", problem.estimand.method.to_string()).map_err(py_err)?;
            let adj: Vec<String> = problem
                .estimand
                .adjustment_set
                .iter()
                .map(|v| {
                    problem.data.schema().get(*v).map_or_else(
                        |_| format!("V{}", v.raw()),
                        |variable| variable.name.to_string(),
                    )
                })
                .collect();
            kwargs.set_item("adjustment_set", adj).map_err(py_err)?;
            let out = self.callback.bind(py).call((), Some(&kwargs)).map_err(py_err)?;
            let dict = out.cast::<PyDict>().map_err(|_| {
                ValidationError::data_msg(format!("validator `{}` must return a dict", self.name))
            })?;
            let passed: bool = dict
                .get_item("passed")
                .map_err(py_err)?
                .ok_or_else(|| ValidationError::data_msg("validator dict missing `passed`"))?
                .extract()
                .map_err(py_err)?;
            let refuted_ate: f64 = dict
                .get_item("refuted_ate")
                .map_err(py_err)?
                .map(|v| v.extract())
                .transpose()
                .map_err(py_err)?
                .unwrap_or(problem.original.ate);
            let comparison: f64 = dict
                .get_item("comparison")
                .map_err(py_err)?
                .map(|v| v.extract())
                .transpose()
                .map_err(py_err)?
                .unwrap_or(if passed { 1.0 } else { 0.0 });
            let informative: bool = dict
                .get_item("informative")
                .map_err(py_err)?
                .map(|v| v.extract())
                .transpose()
                .map_err(py_err)?
                .unwrap_or(true);
            let failure_condition: Option<String> = dict
                .get_item("failure_condition")
                .map_err(py_err)?
                .map(|v| v.extract())
                .transpose()
                .map_err(py_err)?;
            Ok(RefutationReport::new(
                Arc::from(self.name.as_str()),
                problem.original.ate,
                refuted_ate,
                comparison,
                informative,
                passed,
                failure_condition.map(Arc::from),
                0,
            ))
        })
    }
}

/// Resolve `ci` as either a built-in name (`str`) or a Python callable.
///
/// When `ci` is `None`, defaults to partial correlation.
/// Returns `(impl, label, is_callback)`.
pub fn resolve_ci_arg(
    ci: Option<&Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
) -> PyResult<(Arc<dyn ConditionalIndependence + Send + Sync>, String, bool)> {
    let Some(ci) = ci else {
        let impl_ =
            antecedent::discovery_defaults::resolve_ci("parcorr", weights).map_err(|e| {
                crate::with_reason_code(
                    crate::CausalCompileError::new_err(e.to_string()),
                    antecedent_core::reason_code!("invalid_argument"),
                )
            })?;
        return Ok((impl_, "parcorr".into(), false));
    };
    if let Ok(name) = ci.extract::<&str>() {
        let impl_ = antecedent::discovery_defaults::resolve_ci(name, weights).map_err(|e| {
            crate::with_reason_code(
                crate::CausalCompileError::new_err(e.to_string()),
                antecedent_core::reason_code!("invalid_argument"),
            )
        })?;
        return Ok((impl_, name.to_string(), false));
    }
    if ci.is_callable() {
        let cb = PyConditionalIndependence::new(ci.clone().unbind());
        return Ok((Arc::new(cb), "python.callback".into(), true));
    }
    Err(crate::with_reason_code(
        crate::CausalCompileError::new_err(
            "ci must be a str CI name (e.g. 'parcorr') or a callable batch test",
        ),
        antecedent_core::reason_code!("invalid_argument"),
    ))
}

/// Qualified name of a Python callable: `module.qualname` (the class's for a callable object).
fn qualified_name(item: &Bound<'_, PyAny>) -> PyResult<String> {
    let owner =
        if item.hasattr("__qualname__")? { item.clone() } else { item.get_type().into_any() };
    let qualname: String = owner.getattr("__qualname__")?.extract()?;
    let module: Option<String> = owner
        .getattr("__module__")
        .ok()
        .and_then(|module| module.extract::<Option<String>>().ok())
        .flatten();
    Ok(match module {
        Some(module) => format!("{module}.{qualname}"),
        None => qualname,
    })
}

/// Parse caller validators into named custom validators.
///
/// A list names each callable by its qualified name (`module.qualname`); a
/// mapping supplies the names. Names are the attested identity of each
/// validator's evidence on a claim, so they must be unique.
pub fn parse_validators(
    validators: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Arc<dyn CustomEffectValidator>>> {
    let Some(obj) = validators else {
        return Ok(Vec::new());
    };
    let named: Vec<(String, Bound<'_, PyAny>)> = if let Ok(mapping) = obj.cast::<PyDict>() {
        mapping
            .iter()
            .map(|(name, item)| Ok((name.extract::<String>()?, item)))
            .collect::<PyResult<_>>()?
    } else {
        let list = obj.cast::<PyList>().map_err(|_| {
            PyValueError::new_err(
                "validators must be a list of callables or a {name: callable} mapping",
            )
        })?;
        list.iter().map(|item| Ok((qualified_name(&item)?, item))).collect::<PyResult<_>>()?
    };
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(named.len());
    for (name, item) in named {
        if !item.is_callable() {
            return Err(PyValueError::new_err(format!("validator {name:?} is not callable")));
        }
        if !seen.insert(name.clone()) {
            return Err(PyValueError::new_err(format!(
                "validator name {name:?} is not unique; pass a {{name: callable}} mapping to name \
                 each validator's attested evidence"
            )));
        }
        out.push(
            Arc::new(PyCustomValidator::new(name, item.unbind())) as Arc<dyn CustomEffectValidator>
        );
    }
    Ok(out)
}

/// Overlay Python mechanism wrappers onto a fitted GCM store.
#[allow(
    clippy::cast_possible_truncation,
    reason = "node ids are u32 by construction (DenseNodeId), so node positions fit u32"
)]
pub fn apply_mechanism_wrappers(
    model: &CompiledCausalModel,
    names: &[String],
    wrappers: &Bound<'_, PyDict>,
) -> PyResult<CompiledCausalModel> {
    let mut store = model.mechanisms.clone();
    for (key, val) in wrappers.iter() {
        let name: String = key.extract()?;
        let idx = names.iter().position(|n| n == &name).ok_or_else(|| {
            PyValueError::new_err(format!("unknown mechanism wrapper node {name}"))
        })?;
        let slot = MechanismSlot::Dynamic {
            id: Arc::from(name.as_str()),
            mechanism: Arc::new(PyDynamicMechanism::new(val.unbind())),
        };
        store = store
            .with_replaced(DenseNodeId::from_raw(idx as u32), slot)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    Ok(model.clone().with_mechanisms(store))
}

/// Progress sink that calls a Python `(fraction: float, stage: str) -> None`.
pub struct PyProgressSink {
    callback: Py<PyAny>,
}

impl PyProgressSink {
    pub fn new(callback: Py<PyAny>) -> Self {
        Self { callback }
    }
}

impl antecedent_core::ProgressSink for PyProgressSink {
    fn report(&self, fraction: f64, stage: &str) {
        Python::attach(|py| {
            // `ProgressSink::report` returns `()` and is invoked from hot per-replicate
            // / per-sample loops (bootstrap, Shapley, discovery search) across several
            // crates, so propagating a `Result` here would mean threading a fallible
            // signature through every call site outside the files this change owns.
            // Surface the failure the same way an uncaught Python exception normally
            // would instead of silently discarding it: print the traceback to stderr
            // (this also sets `sys.last_*`), so a broken `on_progress` callback is
            // diagnosable rather than appearing to just stop firing.
            if let Err(err) = self.callback.bind(py).call1((fraction, stage)) {
                err.print(py);
            }
        });
    }
}

/// Build an optional [`ProgressSink`] from a Python callable.
pub fn progress_sink_from_py(
    on_progress: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<std::sync::Arc<dyn antecedent_core::ProgressSink>>> {
    match on_progress {
        None => Ok(None),
        Some(cb) => {
            if !cb.is_callable() {
                return Err(PyValueError::new_err("on_progress must be callable"));
            }
            Ok(Some(std::sync::Arc::new(PyProgressSink::new(cb.clone().unbind()))
                as std::sync::Arc<dyn antecedent_core::ProgressSink>))
        }
    }
}

/// Stage-result sink that calls a Python `(stage: str, payload: dict) -> None`.
pub struct PyStageResultSink {
    callback: Py<PyAny>,
}

impl PyStageResultSink {
    pub fn new(callback: Py<PyAny>) -> Self {
        Self { callback }
    }
}

impl antecedent::StageResultSink for PyStageResultSink {
    fn on_stage(&self, event: &antecedent::StageEvent) {
        Python::attach(|py| {
            let payload = match event {
                antecedent::StageEvent::Identify { identification, estimand } => {
                    let d = PyDict::new(py);
                    let _ = d.set_item("status", format!("{:?}", identification.status));
                    let _ = d.set_item("method", estimand.method.as_ref());
                    d
                }
                antecedent::StageEvent::Point { estimate }
                | antecedent::StageEvent::Uncertainty { estimate } => {
                    let d = PyDict::new(py);
                    let _ = d.set_item("ate", estimate.ate);
                    let _ = d.set_item("se_analytic", estimate.se_analytic);
                    let _ = d.set_item("se_bootstrap", estimate.se_bootstrap);
                    d
                }
                antecedent::StageEvent::Validate { refutations, predictive_checks } => {
                    let d = PyDict::new(py);
                    let _ = d.set_item("n_refutations", refutations.len());
                    let _ = d.set_item("n_predictive_checks", predictive_checks.len());
                    d
                }
            };
            // See `PyProgressSink::report` above: `StageResultSink::on_stage` also
            // returns `()`, so surface a failing `on_stage` callback via PyO3's
            // traceback-to-stderr path rather than discarding it silently.
            if let Err(err) = self.callback.bind(py).call1((event.stage_id(), payload)) {
                err.print(py);
            }
        });
    }
}

/// Build an optional [`antecedent::StageResultSink`] from a Python callable.
pub fn stage_sink_from_py(
    on_stage: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<std::sync::Arc<dyn antecedent::StageResultSink>>> {
    match on_stage {
        None => Ok(None),
        Some(cb) => {
            if !cb.is_callable() {
                return Err(PyValueError::new_err("on_stage must be callable"));
            }
            Ok(Some(std::sync::Arc::new(PyStageResultSink::new(cb.clone().unbind()))
                as std::sync::Arc<dyn antecedent::StageResultSink>))
        }
    }
}
