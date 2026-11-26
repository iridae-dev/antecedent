//! Row-isolated Python provider factories for the native AIPW cross-fit.
use std::borrow::Cow;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use antecedent::estimate::external_learner::{
    DesignView, FittedPredictor, IsolatedFactory, LearnError, LearnerCapabilities, LearnerFactory,
    LearnerProvenance, PredictionTask, TargetView,
};
use antecedent::prelude::ExecutionContext;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ndarray::Array2};
use pyo3::prelude::*;
use pyo3::types::PyAny;

type PythonReceipt = (u64, Py<PyAny>, Py<PyAny>);
mod declared;
mod joint_cells;
mod smoothed_dose;
mod trial_aipw;

#[derive(Default)]
struct CallbackState {
    error: Mutex<Option<PyErr>>,
    receipts: Mutex<Vec<PythonReceipt>>,
    next: AtomicU64,
}

impl CallbackState {
    fn fail(&self, error: PyErr) -> LearnError {
        let mut slot = self.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if slot.is_none() {
            *slot = Some(error);
        }
        LearnError::Backend("external learner callback failed".into())
    }

    fn finish<T>(&self, result: PyResult<T>) -> PyResult<(T, Vec<PythonReceipt>)> {
        if let Some(error) =
            self.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
        {
            return Err(error);
        }
        let payload = result?;
        let receipts = std::mem::take(
            &mut *self.receipts.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        Ok((payload, receipts))
    }
}

fn learner_weights(py: Python<'_>, learner: &Py<PyAny>, expected: &str) -> PyResult<bool> {
    let bound = learner.bind(py);
    let task: String = bound.getattr("task")?.extract()?;
    if task != expected {
        return Err(crate::value_err("nuisance learner task mismatch"));
    }
    bound.getattr("supports_weights")?.extract()
}

struct PythonFactory {
    learner: Py<PyAny>,
    task: PredictionTask,
    weights: bool,
    state: Arc<CallbackState>,
}

fn design_columns<'py>(py: Python<'py>, x: DesignView<'_>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let elements = x
        .nrows()
        .checked_mul(x.ncols())
        .ok_or_else(|| crate::value_err("callback design size overflow"))?;
    let mut values = Vec::with_capacity(elements);
    for col in 0..x.ncols() {
        for row in 0..x.nrows() {
            values.push(x.get(row, col).map_err(|error| crate::value_err(error.to_string()))?);
        }
    }
    let columns = Array2::from_shape_vec((x.ncols(), x.nrows()), values)
        .map_err(|error| crate::value_err(error.to_string()))?;
    Ok(PyArray2::from_owned_array(py, columns))
}

fn callback_bytes(x: DesignView<'_>, vectors: usize) -> Result<u64, LearnError> {
    x.ncols()
        .checked_add(vectors)
        .and_then(|n| n.checked_mul(x.nrows()))
        .and_then(|n| n.checked_mul(size_of::<f64>()))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(LearnError::Shape { message: "callback data size overflow" })
}

fn fit_columns<'py>(
    py: Python<'py>,
    learner: &Bound<'py, PyAny>,
    x: DesignView<'_>,
    y: TargetView<'_>,
    weights: Option<&[f64]>,
    bytes: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let columns = design_columns(py, x)?;
    let target = PyArray1::from_slice(py, y.values());
    let weights = weights.map(|weights| PyArray1::from_slice(py, weights));
    learner.call_method1("_fit_columns", (columns, target, weights, bytes))
}

fn admission_error(error: impl std::fmt::Display, ctx: &ExecutionContext) -> PyErr {
    if ctx.cancellation.is_cancelled() {
        crate::py_err(antecedent::CausalError::Cancelled { stage: "nuisance" })
    } else {
        crate::value_err(error.to_string())
    }
}

fn callback_context(
    seed: u64,
    cancel: Option<crate::PyCancellationToken>,
    limit: Option<u64>,
) -> PyResult<ExecutionContext> {
    let mut ctx = crate::py_execution_context_cancel(seed, 1, cancel);
    // Keep the caller's shared admission pool, but never run Python providers concurrently.
    ctx.parallelism.max_threads = std::num::NonZeroU32::MIN;
    match (ctx.memory_limit_bytes(), limit) {
        (Some(existing), Some(requested)) if requested < existing => {
            return Err(crate::value_err(
                "a smaller per-call memory limit cannot replace an active shared execution budget",
            ));
        }
        (None, Some(requested)) => ctx = ctx.with_memory_limit(Some(requested)),
        _ => {}
    }
    Ok(ctx)
}

impl LearnerFactory for PythonFactory {
    fn task(&self) -> PredictionTask {
        self.task
    }
    fn capabilities(&self) -> LearnerCapabilities {
        LearnerCapabilities {
            regression: self.task == PredictionTask::Regression,
            binary_probability: self.task == PredictionTask::BinaryProbability,
            sample_weights: self.weights,
            ..LearnerCapabilities::none()
        }
    }
    fn fit(
        &self,
        x: DesignView<'_>,
        y: TargetView<'_>,
        weights: Option<&[f64]>,
        ctx: &ExecutionContext,
    ) -> Result<Box<dyn FittedPredictor>, LearnError> {
        if ctx.cancellation.is_cancelled() {
            return Err(LearnError::Cancelled);
        }
        let bytes = callback_bytes(x, 1 + usize::from(weights.is_some()))?;
        let _hold = ctx.acquire_admission(0, bytes).map_err(LearnError::Admission)?;
        let fitted = Python::attach(|py| {
            fit_columns(py, self.learner.bind(py), x, y, weights, bytes).map(Bound::unbind)
        })
        .map_err(|e| self.state.fail(e))?;
        if ctx.cancellation.is_cancelled() {
            return Err(LearnError::Cancelled);
        }
        Ok(Box::new(PythonPredictor {
            fitted,
            state: Arc::clone(&self.state),
            id: self.state.next.fetch_add(1, Ordering::Relaxed),
        }))
    }
}

struct PythonPredictor {
    fitted: Py<PyAny>,
    state: Arc<CallbackState>,
    id: u64,
}
impl FittedPredictor for PythonPredictor {
    fn predict(
        &self,
        x: DesignView<'_>,
        out: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<(), LearnError> {
        if ctx.cancellation.is_cancelled() {
            return Err(LearnError::Cancelled);
        }
        let bytes = callback_bytes(x, 1)?;
        let _hold = ctx.acquire_admission(0, bytes).map_err(LearnError::Admission)?;
        Python::attach(|py| -> PyResult<()> {
            let fitted = self.fitted.bind(py);
            let result =
                fitted.call_method1("_predict_columns", (design_columns(py, x)?, bytes))?;
            let estimate = result.getattr("estimate")?;
            let array = estimate.extract::<PyReadonlyArray1<'_, f64>>()?;
            let values = array.as_slice().map_err(|error| crate::value_err(error.to_string()))?;
            if values.len() != out.len() {
                return Err(crate::value_err(
                    "provider prediction length does not match held-out rows",
                ));
            }
            out.copy_from_slice(values);
            let fit = fitted.getattr("fit_receipt")?.unbind();
            self.state.receipts.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push((
                self.id,
                fit,
                result.unbind(),
            ));
            Ok(())
        })
        .map_err(|e| self.state.fail(e))?;
        if ctx.cancellation.is_cancelled() {
            return Err(LearnError::Cancelled);
        }
        Ok(())
    }
    fn provenance(&self) -> LearnerProvenance {
        LearnerProvenance {
            spec: "external_provider".into(),
            implementation: format!("python_receipt:{}", self.id),
            version: "1".into(),
        }
    }
}

/// Cross-fit two explicitly registered Python nuisance learners. Returns predictions,
/// fold provenance and the individual host-sealed fit and prediction receipts.
#[pyfunction]
#[pyo3(signature = (outcome, treatment, design, y, t, fold_ids, *, weights=None, seed=1, cancel=None, memory_limit_bytes=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[allow(
    deprecated,
    reason = "the Python binding of the deprecated direct path; the Python function warns"
)]
fn external_aipw_crossfit(
    py: Python<'_>,
    outcome: Py<PyAny>,
    treatment: Py<PyAny>,
    design: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    t: PyReadonlyArray1<'_, f64>,
    fold_ids: PyReadonlyArray1<'_, u16>,
    weights: Option<PyReadonlyArray1<'_, f64>>,
    seed: u64,
    cancel: Option<crate::PyCancellationToken>,
    memory_limit_bytes: Option<u64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<String>>, Vec<PythonReceipt>)> {
    let design = design.as_array();
    let (n, cols) = design.dim();
    if n == 0 || cols == 0 {
        return Err(crate::value_err("design must be a nonempty rectangular matrix"));
    }
    let ctx = callback_context(seed, cancel, memory_limit_bytes)?;
    let bytes = [
        design.len(),
        y.as_array().len(),
        t.as_array().len(),
        weights.as_ref().map_or(0, |w| w.as_array().len()),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|v| v.checked_mul(size_of::<f64>()))
    .and_then(|v| {
        fold_ids
            .as_array()
            .len()
            .checked_mul(size_of::<u16>())
            .and_then(|folds| v.checked_add(folds))
    })
    .and_then(|v| u64::try_from(v).ok())
    .ok_or_else(|| crate::value_err("design size overflow"))?;
    let _input_hold = ctx.acquire_admission(0, bytes).map_err(|e| admission_error(e, &ctx))?;
    let mut column_major = Vec::with_capacity(design.len());
    for column in design.columns() {
        column_major.extend(column.iter().copied());
    }
    let y = copy_numeric_array(&y);
    let t = copy_numeric_array(&t);
    let fold_ids = fold_ids.as_array().iter().copied().collect::<Vec<_>>();
    let weights = weights.as_ref().map(copy_numeric_array);
    let outcome_weights = learner_weights(py, &outcome, "regression")?;
    let treatment_weights = learner_weights(py, &treatment, "binary_probability")?;
    let state = Arc::new(CallbackState::default());
    let result = py.detach(|| {
        let outcome = PythonFactory {
            learner: outcome,
            task: PredictionTask::Regression,
            weights: outcome_weights,
            state: Arc::clone(&state),
        };
        let treatment = PythonFactory {
            learner: treatment,
            task: PredictionTask::BinaryProbability,
            weights: treatment_weights,
            state: Arc::clone(&state),
        };
        let view = DesignView::from_column_major(&column_major, n, cols)
            .map_err(antecedent::estimate::external_learner::AipwCrossFitError::from)?;
        let folds = fold_ids.iter().copied().max().map_or(0, |v| usize::from(v) + 1);
        antecedent::estimate::external_learner::cross_fit_aipw_with_factories(
            &IsolatedFactory::new(&outcome),
            &IsolatedFactory::new(&treatment),
            view,
            view,
            &y,
            &t,
            (folds, fold_ids),
            &ctx,
            weights.as_deref(),
        )
    });
    let ((mu0, mu1, e, record), receipts) = state.finish(result.map_err(|e| match e {
        antecedent::estimate::external_learner::AipwCrossFitError::Cancelled => {
            crate::py_err(antecedent::CausalError::Cancelled { stage: "nuisance" })
        }
        other => crate::value_err(other.to_string()),
    }))?;
    let provenance = record
        .model_provenance
        .into_iter()
        .map(|p| vec![p.spec, p.implementation, p.version])
        .collect();
    Ok((mu0, mu1, e, provenance, receipts))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(external_aipw_crossfit, module)?)?;
    module.add_function(wrap_pyfunction!(declared::declared_learner_ate, module)?)?;
    module
        .add_function(wrap_pyfunction!(smoothed_dose::smoothed_dose_learner_transport, module)?)?;
    module.add_function(wrap_pyfunction!(trial_aipw::trial_aipw_learner_transport, module)?)?;
    module.add_function(wrap_pyfunction!(joint_cells::joint_cells_learner_fit, module)?)
}

fn copy_numeric_array(values: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    values.as_array().iter().copied().collect()
}

fn learner_table(
    names: &[String],
    columns: &[PyReadonlyArray1<'_, f64>],
) -> PyResult<antecedent::prelude::TabularData> {
    let values = columns
        .iter()
        .map(|column| match column.as_slice() {
            Ok(values) => Cow::Borrowed(values),
            Err(_) => Cow::Owned(copy_numeric_array(column)),
        })
        .collect::<Vec<_>>();
    antecedent::prelude::TabularData::from_f64_columns(
        names.iter().zip(&values).map(|(name, column)| (name.as_str(), column.as_ref())),
    )
    .map_err(crate::py_err)
}

fn input_bytes(
    columns: &[PyReadonlyArray1<'_, f64>],
    weights: Option<&PyReadonlyArray1<'_, f64>>,
) -> PyResult<u64> {
    columns
        .iter()
        .try_fold(weights.map_or(0, |w| w.as_array().len()), |values, column| {
            let copies = if column.as_slice().is_ok() { 1 } else { 2 };
            values.checked_add(column.as_array().len().checked_mul(copies)?)
        })
        .and_then(|n| n.checked_mul(size_of::<f64>()))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| crate::value_err("input size overflow"))
}
