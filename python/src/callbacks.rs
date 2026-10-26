//! Slow-path Python callback bridges.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, OnceLock};

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
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList};

use crate::interrupt::callback_failure;

/// `inspect.signature` and `numpy.random.default_rng`, resolved once per process.
///
/// Each mechanism still asks `signature` about its own `sample_noise`. The
/// callable objects are not imported again on the next draw.
static INSPECT_SIGNATURE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static NUMPY_DEFAULT_RNG: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

fn inspect_signature(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let func = INSPECT_SIGNATURE.get_or_try_init(py, || -> PyResult<_> {
        Ok(py.import("inspect")?.getattr("signature")?.unbind())
    })?;
    Ok(func.bind(py).clone())
}

fn numpy_default_rng(py: Python<'_>, seed: u64) -> PyResult<Bound<'_, PyAny>> {
    let func = NUMPY_DEFAULT_RNG.get_or_try_init(py, || -> PyResult<_> {
        Ok(py.import("numpy.random")?.getattr("default_rng")?.unbind())
    })?;
    func.bind(py).call1((seed,))
}

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

/// The parent columns as a list of float64 arrays, one per parent.
fn parent_list<'py>(
    py: Python<'py>,
    parents: ParentBatch<'_>,
) -> Result<Bound<'py, PyList>, ModelError> {
    let list = PyList::empty(py);
    for p in 0..parents.n_parents {
        let col = parents.column(p).map_err(|e| ModelError::Shape { message: e.to_string() })?;
        list.append(PyArray1::from_slice(py, col)).map_err(|e| ModelError::Unsupported {
            message: format!("failed to build parent columns: {e}"),
        })?;
    }
    Ok(list)
}

/// The typed error of a user method `name` that raised.
fn method_failure(name: &str, err: PyErr) -> ModelError {
    ModelError::Unsupported { message: callback_failure(&format!("Python {name} failed"), err) }
}

/// Copy the float64 ndarray `out` returned by the user's `name` into `output[..n]`, requiring
/// a contiguous array of exactly `n` values.
fn copy_column_result(
    out: &Bound<'_, PyAny>,
    name: &str,
    n: usize,
    output: &mut [f64],
) -> Result<(), ModelError> {
    let arr: PyReadonlyArray1<'_, f64> = out.extract().map_err(|e| ModelError::Shape {
        message: format!("{name} must return float64 ndarray: {e}"),
    })?;
    let slice = arr
        .as_slice()
        .map_err(|_| ModelError::Shape { message: format!("{name} ndarray must be contiguous") })?;
    check_len(name, slice.len(), n)?;
    output[..n].copy_from_slice(slice);
    Ok(())
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
    let Ok(signature_of) = inspect_signature(py) else {
        return Ok(false);
    };
    let Ok(signature) = signature_of.call1((&method,)) else {
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
                // The conditioning set converts straight from the request's flat buffer.
                let z: &[usize] = &bound.z_flat[q.z_start..q.z_start + q.z_len];
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
                results.push(callback_ci_result(stat, p)?);
            }
            Ok(CiBatchResult { results })
        })
    }
}

/// Callback evidence is consumed directly by graph search; NaN or out-of-range
/// probabilities must not silently choose the dependent/independent branch.
fn callback_ci_result(statistic: f64, p_value: f64) -> Result<CiResult, StatsError> {
    if !statistic.is_finite() || !p_value.is_finite() || !(0.0..=1.0).contains(&p_value) {
        return Err(StatsError::Backend(
            "Python CI callback must return a finite statistic and a finite p_value in [0, 1]"
                .to_owned(),
        ));
    }
    Ok(CiResult { statistic, p_value, df: f64::NAN, ci: None })
}

#[cfg(test)]
mod ci_callback_tests {
    use super::callback_ci_result;

    #[test]
    fn callback_evidence_rejects_invalid_numbers_before_graph_search() {
        for (statistic, probability) in [
            (f64::NAN, 0.5),
            (f64::INFINITY, 0.5),
            (0.0, f64::NAN),
            (0.0, f64::INFINITY),
            (0.0, -0.01),
            (0.0, 1.01),
        ] {
            assert!(callback_ci_result(statistic, probability).is_err());
        }
        for probability in [0.0, 0.5, 1.0] {
            let result = callback_ci_result(-2.0, probability).expect("valid evidence");
            assert_eq!(result.statistic, -2.0);
            assert_eq!(result.p_value, probability);
        }
    }
}

/// Python mechanism wrapper: `sample_noise(n) -> array`, `evaluate(parents, noise) -> array`.
pub struct PyDynamicMechanism {
    obj: Py<PyAny>,
    /// Whether `sample_noise` accepts `rng`. Computed once per mechanism.
    accepts_rng: OnceLock<bool>,
}

impl PyDynamicMechanism {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj, accepts_rng: OnceLock::new() }
    }

    fn accepts_rng(&self, py: Python<'_>) -> Result<bool, ModelError> {
        if let Some(accepts) = self.accepts_rng.get() {
            return Ok(*accepts);
        }
        let accepts = accepts_rng(self.obj.bind(py))?;
        let _ = self.accepts_rng.set(accepts);
        Ok(accepts)
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
            let out = (if self.accepts_rng(py)? {
                // The run's seeded stream governs the user's draws through a NumPy generator.
                let generator =
                    numpy_default_rng(py, rng.next_u64()).map_err(|e| ModelError::Unsupported {
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
            copy_column_result(&out, "sample_noise", n_rows, output)
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
            let parent_cols = parent_list(py, parents)?;
            let noise_arr = PyArray1::from_slice(py, &noise[..n]);
            let out = self
                .obj
                .bind(py)
                .call_method1("evaluate", (parent_cols, noise_arr))
                .map_err(|e| method_failure("evaluate", e))?;
            copy_column_result(&out, "evaluate", n, output)
        })
    }

    fn supports_noise_inference(&self) -> bool {
        // An object that declares `supports_noise_inference = False` opts out of abduction;
        // one that does not declare it keeps the additive default.
        Python::attach(|py| {
            self.obj
                .bind(py)
                .getattr("supports_noise_inference")
                .and_then(|flag| flag.is_truthy())
                .unwrap_or(true)
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
                let parent_cols = parent_list(py, parents)?;
                let value_arr = PyArray1::from_slice(py, &value[..n]);
                let out = bound
                    .call_method1("infer_noise", (value_arr, parent_cols))
                    .map_err(|e| method_failure("infer_noise", e))?;
                copy_column_result(&out, "infer_noise", n, output)
            })())
        });
        if let Some(r) = py_result {
            r
        } else {
            // Additive default: noise = y − f(pa, 0).
            let zeros = vec![0.0; n];
            let mut ws = MechanismWorkspace::default();
            self.evaluate_column(parents, &zeros, &mut output[..n], &mut ws)?;
            for (slot, &observed) in output[..n].iter_mut().zip(value) {
                *slot = observed - *slot;
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
                let parent_cols = parent_list(py, parents)?;
                let value_arr = PyArray1::from_slice(py, &values[..n]);
                let out = bound
                    .call_method1("log_prob", (value_arr, parent_cols))
                    .map_err(|e| method_failure("log_prob", e))?;
                copy_column_result(&out, "log_prob", n, output)
            })())
        });
        if let Some(r) = py_result {
            r
        } else {
            // Convention: the inferred additive noise is scored as a standard normal draw.
            // A mechanism whose noise is not N(0, 1) should define `log_prob` itself.
            let mut resid = vec![0.0; n];
            self.infer_noise_column(values, parents, &mut resid)?;
            antecedent_model::standard_normal_log_density(&resid, &mut output[..n])
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
            kwargs.set_item("se_analytic", problem.original.se_analytic()).map_err(py_err)?;
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
            crate::value_err("validators must be a list of callables or a {name: callable} mapping")
        })?;
        list.iter().map(|item| Ok((qualified_name(&item)?, item))).collect::<PyResult<_>>()?
    };
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(named.len());
    for (name, item) in named {
        if !item.is_callable() {
            return Err(crate::value_err(format!("validator {name:?} is not callable")));
        }
        if !seen.insert(name.clone()) {
            return Err(crate::value_err(format!(
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
        let idx = names
            .iter()
            .position(|n| n == &name)
            .ok_or_else(|| crate::value_err(format!("unknown mechanism wrapper node {name}")))?;
        let slot = MechanismSlot::Dynamic {
            id: Arc::from(name.as_str()),
            mechanism: Arc::new(PyDynamicMechanism::new(val.unbind())),
        };
        store = store
            .with_replaced(DenseNodeId::from_raw(idx as u32), slot)
            .map_err(|e| crate::value_err(e.to_string()))?;
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
            if let Err(err) = self.callback.bind(py).call1((fraction, stage)) {
                if let Err(unscoped) = crate::interrupt::observer_failure(err) {
                    // Outside an executing run no Result boundary owns this observer.
                    // Keep the historical diagnostic without leaking it into another run.
                    unscoped.print(py);
                }
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
                return Err(crate::value_err("on_progress must be callable"));
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
                    let _ = d.set_item("se_analytic", estimate.se_analytic());
                    let _ = d.set_item("se_bootstrap", estimate.se_bootstrap());
                    d
                }
                antecedent::StageEvent::Validate { refutations, predictive_checks } => {
                    let d = PyDict::new(py);
                    let _ = d.set_item("n_refutations", refutations.len());
                    let _ = d.set_item("n_predictive_checks", predictive_checks.len());
                    d
                }
            };
            if let Err(err) = self.callback.bind(py).call1((event.stage_id(), payload)) {
                if let Err(unscoped) = crate::interrupt::observer_failure(err) {
                    unscoped.print(py);
                }
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
                return Err(crate::value_err("on_stage must be callable"));
            }
            Ok(Some(std::sync::Arc::new(PyStageResultSink::new(cb.clone().unbind()))
                as std::sync::Arc<dyn antecedent::StageResultSink>))
        }
    }
}
