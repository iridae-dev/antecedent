//! Row-isolated Python provider factories for the native AIPW cross-fit.
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use antecedent::estimate::external_learner::{
    DesignView, FittedPredictor, IsolatedFactory, LearnError, LearnerCapabilities, LearnerFactory,
    LearnerProvenance, PredictionTask, TargetView,
};
use antecedent::prelude::ExecutionContext;
use pyo3::prelude::*;
use pyo3::types::PyAny;

type PythonReceipt = (u64, Py<PyAny>, Py<PyAny>);

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
}

struct PythonFactory {
    learner: Py<PyAny>,
    task: PredictionTask,
    weights: bool,
    state: Arc<CallbackState>,
}

fn rows(x: DesignView<'_>) -> Result<Vec<Vec<f64>>, LearnError> {
    (0..x.nrows()).map(|r| (0..x.ncols()).map(|c| x.get(r, c)).collect()).collect()
}

fn callback_bytes(x: DesignView<'_>, vectors: usize) -> Result<u64, LearnError> {
    x.ncols()
        .checked_add(vectors)
        .and_then(|n| n.checked_mul(x.nrows()))
        // Charge for Rust copies, Python float/list objects and NumPy materialization.
        .and_then(|n| n.checked_mul(64))
        .and_then(|n| x.nrows().checked_mul(96).and_then(|rows| n.checked_add(rows)))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(LearnError::Shape { message: "callback data size overflow" })
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
        let _hold = ctx
            .acquire_admission(0, callback_bytes(x, 1 + usize::from(weights.is_some()))?)
            .map_err(LearnError::Admission)?;
        let matrix = rows(x)?;
        let target = y.values().to_vec();
        let weights = weights.map(<[f64]>::to_vec);
        let fitted = Python::attach(|py| {
            self.learner.bind(py).call_method1("fit", (matrix, target, weights)).map(Bound::unbind)
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
        let _hold =
            ctx.acquire_admission(0, callback_bytes(x, 0)?).map_err(LearnError::Admission)?;
        let matrix = rows(x)?;
        Python::attach(|py| -> PyResult<()> {
            let fitted = self.fitted.bind(py);
            let result = fitted.call_method1("predict", (matrix,))?;
            let values: Vec<f64> = result.getattr("estimate")?.call_method0("tolist")?.extract()?;
            if values.len() != out.len() {
                return Err(crate::value_err(
                    "provider prediction length does not match held-out rows",
                ));
            }
            out.copy_from_slice(&values);
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
fn external_aipw_crossfit(
    py: Python<'_>,
    outcome: Py<PyAny>,
    treatment: Py<PyAny>,
    design: Vec<Vec<f64>>,
    y: Vec<f64>,
    t: Vec<f64>,
    fold_ids: Vec<u16>,
    weights: Option<Vec<f64>>,
    seed: u64,
    cancel: Option<crate::PyCancellationToken>,
    memory_limit_bytes: Option<u64>,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<String>>, Vec<PythonReceipt>)> {
    if design.is_empty()
        || design[0].is_empty()
        || design.iter().any(|row| row.len() != design[0].len())
    {
        return Err(crate::value_err(
            "design must be a nonempty rectangular matrix",
        ));
    }
    let n = design.len();
    let cols = design[0].len();
    let ctx = callback_context(seed, cancel, memory_limit_bytes)?;
    let bytes = cols
        .checked_add(3)
        .and_then(|v| v.checked_mul(n))
        .and_then(|v| v.checked_mul(8))
        .and_then(|v| u64::try_from(v).ok())
        .ok_or_else(|| crate::value_err("design size overflow"))?;
    let _input_hold = ctx.acquire_admission(0, bytes).map_err(|e| admission_error(e, &ctx))?;
    let mut column_major = Vec::with_capacity(
        n.checked_mul(cols)
            .ok_or_else(|| crate::value_err("design size overflow"))?,
    );
    for c in 0..cols {
        for row in &design {
            column_major.push(row[c]);
        }
    }
    let task_and_weights = |obj: &Py<PyAny>, expected: &str| -> PyResult<bool> {
        let bound = obj.bind(py);
        let task: String = bound.getattr("task")?.extract()?;
        if task != expected {
            return Err(crate::value_err("nuisance learner task mismatch"));
        }
        bound.getattr("supports_weights")?.extract()
    };
    let outcome_weights = task_and_weights(&outcome, "regression")?;
    let treatment_weights = task_and_weights(&treatment, "binary_probability")?;
    let state = Arc::new(CallbackState {
        error: Mutex::new(None),
        receipts: Mutex::new(Vec::new()),
        next: AtomicU64::new(0),
    });
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
    if let Some(error) =
        state.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
    {
        return Err(error);
    }
    let (mu0, mu1, e, record) = result.map_err(|e| match e {
        antecedent::estimate::external_learner::AipwCrossFitError::Cancelled => {
            crate::py_err(antecedent::CausalError::Cancelled { stage: "nuisance" })
        }
        other => crate::value_err(other.to_string()),
    })?;
    let provenance = record
        .model_provenance
        .into_iter()
        .map(|p| vec![p.spec, p.implementation, p.version])
        .collect();
    let receipts = std::mem::take(
        &mut *state.receipts.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    Ok((mu0, mu1, e, provenance, receipts))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(external_aipw_crossfit, module)?)?;
    module.add_function(wrap_pyfunction!(external_dml_ate, module)?)
}

/// Identify a static DAG adjustment and fit its DML AIPW score using Python providers.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment_name, outcome_name, outcome, treatment, *, folds=5, weights=None, seed=1, cancel=None, method="dml", score="aipw", memory_limit_bytes=None))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn external_dml_ate(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
    edges: Vec<(String, String)>,
    treatment_name: String,
    outcome_name: String,
    outcome: Py<PyAny>,
    treatment: Py<PyAny>,
    folds: usize,
    weights: Option<Vec<f64>>,
    seed: u64,
    cancel: Option<crate::PyCancellationToken>,
    method: &str,
    score: &str,
    memory_limit_bytes: Option<u64>,
) -> PyResult<(
    f64,
    f64,
    Vec<u64>,
    Vec<u32>,
    Vec<u32>,
    Vec<Vec<String>>,
    String,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Vec<PythonReceipt>,
)> {
    use antecedent::data::TableView;
    if names.len() != columns.len() || names.is_empty() {
        return Err(crate::value_err(
            "one data column is required per name",
        ));
    }
    let task_and_weights = |obj: &Py<PyAny>, expected: &str| -> PyResult<bool> {
        let bound = obj.bind(py);
        let task: String = bound.getattr("task")?.extract()?;
        if task != expected {
            return Err(crate::value_err("nuisance learner task mismatch"));
        }
        bound.getattr("supports_weights")?.extract()
    };
    let outcome_weights = task_and_weights(&outcome, "regression")?;
    let treatment_weights = task_and_weights(&treatment, "binary_probability")?;
    let state = Arc::new(CallbackState {
        error: Mutex::new(None),
        receipts: Mutex::new(Vec::new()),
        next: AtomicU64::new(0),
    });
    let ctx = callback_context(seed, cancel, memory_limit_bytes)?;
    let result: PyResult<_> = py.detach(|| {
        let bytes = columns.iter().try_fold(0usize, |sum, column| sum.checked_add(column.len()))
            .and_then(|n| n.checked_mul(16)).and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| crate::value_err("input size overflow"))?;
        let _input_hold = ctx.acquire_admission(0, bytes).map_err(|e| admission_error(e, &ctx))?;
        let data = antecedent::prelude::TabularData::from_f64_columns(
            names.iter().zip(&columns).map(|(name, column)| (name.as_str(), column.as_slice())),
        )
        .map_err(crate::py_err)?;
        let t = data.schema().id_of(&treatment_name).map_err(crate::py_err)?;
        let y = data.schema().id_of(&outcome_name).map_err(crate::py_err)?;
        let dag = crate::dag_from_named_edges(data.schema(), &edges)?;
        let query = antecedent::query::AverageEffectQuery::binary_ate(t, y);
        let identification = antecedent::Study::tabular(data.clone())
            .graph(dag)
            .query(query.clone())
            .identify_structure_within(&ctx)
            .map_err(crate::py_err)?;
        if !identification.is_identified() {
            return Err(crate::value_err(
                "DML requires an identified adjustment estimand",
            ));
        }
        let estimand = identification.estimands().first().ok_or_else(|| {
            crate::value_err(
                "identification produced no adjustment estimand",
            )
        })?;
        let assumptions = match &identification {
            antecedent::Identification::Point { result, .. } => result.required_assumptions.clone(),
            _ => {
                return Err(crate::value_err(
                    "DML requires point identification",
                ));
            }
        };
        let sampling = weights
            .map(antecedent::estimate::SamplingWeights::new)
            .transpose()
            .map_err(crate::py_err)?;
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
        let (problem, effect) = match method {
            "dml" => {
                let score = antecedent::estimate::DmlScore::parse(score).map_err(crate::py_err)?;
                let mut estimator =
                    antecedent::estimate::DmlAte::new().with_folds(folds).with_score(score);
                if let Some(weights) = sampling {
                    estimator = estimator.with_sampling_weights(weights);
                }
                let problem = estimator.prepare(&data, estimand, &query).map_err(crate::py_err)?;
                let effect = estimator
                    .fit_with_factories(&problem, &ctx, assumptions, &outcome, &treatment)
                    .map_err(crate::py_err)?;
                (problem, effect)
            }
            "aipw" => {
                if score != "aipw" {
                    return Err(crate::value_err(
                        "AIPW requires the AIPW score",
                    ));
                }
                if folds != 5 {
                    return Err(crate::value_err("AIPW uses five folds"));
                }
                let mut estimator =
                    antecedent::estimate::AipwAte::new().with_bootstrap_replicates(0);
                if let Some(weights) = sampling {
                    estimator = estimator.with_sampling_weights(weights);
                }
                let problem = estimator.prepare(&data, estimand, &query).map_err(crate::py_err)?;
                let effect = estimator
                    .fit_with_factories(&problem, &ctx, assumptions, &outcome, &treatment)
                    .map_err(crate::py_err)?;
                (problem, effect)
            }
            "dr_learner" => {
                if score != "aipw" { return Err(crate::value_err("DR learner requires the AIPW score")); }
                let mut estimator = antecedent::estimate::DrLearner::new().with_folds(folds);
                if let Some(weights) = sampling { estimator = estimator.with_sampling_weights(weights); }
                let problem = estimator.prepare(&data, estimand, &query).map_err(crate::py_err)?;
                let effect = estimator.fit_with_factories(&problem, &ctx, assumptions, &outcome, &treatment).map_err(crate::py_err)?;
                (problem, effect)
            }
            "causal_forest" => {
                if score != "aipw" || folds != 5 || sampling.is_some() {
                    return Err(crate::value_err("causal forest requires its five-fold AIPW score and does not support sampling weights"));
                }
                let estimator = antecedent::estimate::CausalForest::new();
                let problem = estimator.prepare(&data, estimand, &query).map_err(crate::py_err)?;
                let effect = estimator.fit_with_factories(&problem, &ctx, assumptions, &outcome, &treatment).map_err(crate::py_err)?;
                (problem, effect)
            }
            _ => return Err(crate::value_err("unknown external estimator")),
        };
        let row_ids = problem.row_index.iter().map(|row| row.get()).collect();
        let adjustment = problem.adjustment_set.iter().map(|id| id.raw()).collect();
        let fold_ids = effect.crossfit.fold_assignment.as_deref().unwrap_or(&[]).to_vec();
        let provenance = effect
            .crossfit
            .learner_provenance
            .iter()
            .map(|p| vec![p.spec.clone(), p.implementation.clone(), p.version.clone()])
            .collect();
        let assumptions = serde_json::to_string(&antecedent::io::assumptions_to_wire(
            &effect.provenance.assumptions,
        ))
        .map_err(|error| crate::value_err(error.to_string()))?;
        Ok((
            effect.ate,
            effect.se_analytic(),
            row_ids,
            fold_ids,
            adjustment,
            provenance,
            assumptions,
            effect.heterogeneity.cate.as_deref().map(<[f64]>::to_vec),
            effect.heterogeneity.cate_se.as_deref().map(<[f64]>::to_vec),
        ))
    });
    if let Some(error) =
        state.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
    {
        return Err(error);
    }
    let (estimate, se, row_ids, fold_ids, adjustment, provenance, assumptions, cate, cate_se) =
        result?;
    let receipts = std::mem::take(
        &mut *state.receipts.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    Ok((
        estimate,
        se,
        row_ids,
        fold_ids,
        adjustment,
        provenance,
        assumptions,
        cate,
        cate_se,
        receipts,
    ))
}
