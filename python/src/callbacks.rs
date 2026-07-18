//! Slow-path Python callback bridges (DESIGN.md §25.4).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal::{CompiledCausalModel, DynamicMechanism, MechanismSlot, Utility};
use causal_core::{CausalRng, ExecutionContext};
use causal_graph::DenseNodeId;
use causal_model::{MechanismWorkspace, ModelError, ParentBatch};
use causal_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependence,
    ConditionalIndependenceTest, PreparedCiTest, StatsError,
};
use causal_validate::{CustomEffectValidator, RefutationProblem, RefutationReport, ValidationError};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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
                queries
                    .append((q.x, q.y, z))
                    .map_err(|e| StatsError::Backend(e.to_string()))?;
            }
            let out = self
                .callback
                .bind(py)
                .call1((cols, queries))
                .map_err(|e| StatsError::Backend(format!("Python CI callback failed: {e}")))?;
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
                results.push(CiResult {
                    statistic: stat,
                    p_value: p,
                    df: f64::NAN,
                    ci: None,
                });
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
        _rng: &mut CausalRng,
        output: &mut [f64],
    ) -> Result<(), ModelError> {
        Python::attach(|py| -> Result<(), ModelError> {
            let out = self
                .obj
                .bind(py)
                .call_method1("sample_noise", (n_rows,))
                .map_err(|e| ModelError::Unsupported {
                    message: format!("Python sample_noise failed: {e}"),
                })?;
            let arr: PyReadonlyArray1<'_, f64> = out.extract().map_err(|e| ModelError::Shape {
                message: format!("sample_noise must return float64 ndarray: {e}"),
            })?;
            let slice = arr.as_slice().map_err(|_| ModelError::Shape {
                message: "sample_noise ndarray must be contiguous".into(),
            })?;
            if slice.len() < n_rows {
                return Err(ModelError::Shape {
                    message: "sample_noise returned too few values".into(),
                });
            }
            output[..n_rows].copy_from_slice(&slice[..n_rows]);
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
                let col = parents.column(p).map_err(|e| ModelError::Shape {
                    message: e.to_string(),
                })?;
                parent_cols
                    .append(PyArray1::from_slice(py, col))
                    .map_err(|e| ModelError::Unsupported {
                        message: format!("failed to build parent columns: {e}"),
                    })?;
            }
            let noise_arr = PyArray1::from_slice(py, &noise[..n]);
            let out = self
                .obj
                .bind(py)
                .call_method1("evaluate", (parent_cols, noise_arr))
                .map_err(|e| ModelError::Unsupported {
                    message: format!("Python evaluate failed: {e}"),
                })?;
            let arr: PyReadonlyArray1<'_, f64> = out.extract().map_err(|e| ModelError::Shape {
                message: format!("evaluate must return float64 ndarray: {e}"),
            })?;
            let slice = arr.as_slice().map_err(|_| ModelError::Shape {
                message: "evaluate ndarray must be contiguous".into(),
            })?;
            if slice.len() < n {
                return Err(ModelError::Shape {
                    message: "evaluate returned too few values".into(),
                });
            }
            output[..n].copy_from_slice(&slice[..n]);
            Ok(())
        })
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
    fn evaluate_batch(&self, actions: &[f64], outcomes: &[f64], out: &mut [f64]) {
        let expected = actions.len().saturating_mul(outcomes.len());
        if out.len() < expected {
            out.fill(f64::NAN);
            return;
        }
        let result = Python::attach(|py| -> PyResult<()> {
            let a = PyArray1::from_slice(py, actions);
            let o = PyArray1::from_slice(py, outcomes);
            let got = self.callback.bind(py).call1((a, o))?;
            let arr: PyReadonlyArray1<'_, f64> = got.extract()?;
            let slice = arr.as_slice().map_err(|_| {
                PyValueError::new_err("utility return must be contiguous float64 ndarray")
            })?;
            if slice.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "utility returned {} values; expected {expected}",
                    slice.len()
                )));
            }
            out[..expected].copy_from_slice(&slice[..expected]);
            Ok(())
        });
        if result.is_err() {
            out[..expected].fill(f64::NAN);
        }
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
                ValidationError::data_msg(format!("Python validator `{}` failed: {e}", self.name))
            };
            let kwargs = PyDict::new(py);
            kwargs.set_item("ate", problem.original.ate).map_err(py_err)?;
            kwargs
                .set_item("se_analytic", problem.original.se_analytic)
                .map_err(py_err)?;
            kwargs
                .set_item("method", problem.estimand.method.to_string())
                .map_err(py_err)?;
            let adj: Vec<String> = problem
                .estimand
                .adjustment_set
                .iter()
                .map(|v| format!("V{}", v.raw()))
                .collect();
            kwargs.set_item("adjustment_set", adj).map_err(py_err)?;
            let out = self
                .callback
                .bind(py)
                .call((), Some(&kwargs))
                .map_err(py_err)?;
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
            Ok(RefutationReport {
                refuter: Arc::from(self.name.as_str()),
                original_ate: problem.original.ate,
                refuted_ate,
                comparison,
                informative,
                passed,
                failure_condition: failure_condition.map(Arc::from),
                replicates: 0,
            })
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
        let impl_ = causal::resolve_ci("parcorr", weights)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((impl_, "parcorr".into(), false));
    };
    if let Ok(name) = ci.extract::<&str>() {
        let impl_ = causal::resolve_ci(name, weights)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        return Ok((impl_, name.to_string(), false));
    }
    if ci.is_callable() {
        let cb = PyConditionalIndependence::new(ci.clone().unbind());
        return Ok((Arc::new(cb), "python.callback".into(), true));
    }
    Err(PyValueError::new_err(
        "ci must be a str CI name (e.g. 'parcorr') or a callable batch test",
    ))
}

/// Parse optional validator callables into custom validators.
pub fn parse_validators(
    validators: Option<&Bound<'_, PyAny>>,
) -> PyResult<Vec<Arc<dyn CustomEffectValidator>>> {
    let Some(obj) = validators else {
        return Ok(Vec::new());
    };
    let list = obj.cast::<PyList>().map_err(|_| {
        PyValueError::new_err("validators must be a list of callables")
    })?;
    let mut out = Vec::with_capacity(list.len());
    for (i, item) in list.iter().enumerate() {
        if !item.is_callable() {
            return Err(PyValueError::new_err(format!(
                "validators[{i}] is not callable"
            )));
        }
        let name = format!("python.validator.{i}");
        out.push(Arc::new(PyCustomValidator::new(name, item.unbind()))
            as Arc<dyn CustomEffectValidator>);
    }
    Ok(out)
}

/// Overlay Python mechanism wrappers onto a fitted GCM store.
pub fn apply_mechanism_wrappers(
    model: &CompiledCausalModel,
    names: &[String],
    wrappers: &Bound<'_, PyDict>,
) -> PyResult<CompiledCausalModel> {
    let mut store = model.mechanisms.clone();
    for (key, val) in wrappers.iter() {
        let name: String = key.extract()?;
        let idx = names
            .iter()
            .position(|n| n == &name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown mechanism wrapper node {name}")))?;
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
