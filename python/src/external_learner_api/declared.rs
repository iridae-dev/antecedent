//! Python learners declared on a compiled study.
//!
//! [`declared_learner_ate`] declares the two registered Python providers as the study's learner
//! factories and runs the study through the compiler: the static average-effect route over the
//! declared DAG fits the configured DML, AIPW, DR-learner or causal-forest estimator with them,
//! through the same `fit_with_factories` kernel the direct `external_dml_ate` calls. The
//! factories are the binding's row-isolated provider callbacks, so every fit and prediction
//! still returns its provider receipt.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex, atomic::AtomicU64};

use antecedent::analysis::declared_learners::LearnerFactories;
use antecedent::estimate::external_learner::PredictionTask;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use super::{CallbackState, PythonFactory, PythonReceipt, admission_error, callback_context};

/// The configured estimator a declared-learner study runs.
#[allow(
    clippy::large_enum_variant,
    reason = "one estimator is configured per call and moved into the study"
)]
enum Configured {
    Dml(antecedent::estimate::DmlAte),
    Aipw(antecedent::estimate::AipwAte),
    DrLearner(antecedent::estimate::DrLearner),
    CausalForest(antecedent::estimate::CausalForest),
}

impl Configured {
    /// The estimator `method` names, configured as `external_dml_ate` configures it.
    fn new(
        method: &str,
        score: &str,
        folds: usize,
        sampling: Option<antecedent::estimate::SamplingWeights>,
    ) -> PyResult<Self> {
        Ok(match method {
            "dml" => {
                let score = antecedent::estimate::DmlScore::parse(score).map_err(crate::py_err)?;
                let mut estimator =
                    antecedent::estimate::DmlAte::new().with_folds(folds).with_score(score);
                if let Some(weights) = sampling {
                    estimator = estimator.with_sampling_weights(weights);
                }
                Self::Dml(estimator)
            }
            "aipw" => {
                if score != "aipw" {
                    return Err(crate::value_err("AIPW requires the AIPW score"));
                }
                if folds != 5 {
                    return Err(crate::value_err("AIPW uses five folds"));
                }
                let mut estimator =
                    antecedent::estimate::AipwAte::new().with_bootstrap_replicates(0);
                if let Some(weights) = sampling {
                    estimator = estimator.with_sampling_weights(weights);
                }
                Self::Aipw(estimator)
            }
            "dr_learner" => {
                if score != "aipw" {
                    return Err(crate::value_err("DR learner requires the AIPW score"));
                }
                let mut estimator = antecedent::estimate::DrLearner::new().with_folds(folds);
                if let Some(weights) = sampling {
                    estimator = estimator.with_sampling_weights(weights);
                }
                Self::DrLearner(estimator)
            }
            "causal_forest" => {
                if score != "aipw" || folds != 5 || sampling.is_some() {
                    return Err(crate::value_err(
                        "causal forest requires its five-fold AIPW score and does not support \
                         sampling weights",
                    ));
                }
                Self::CausalForest(antecedent::estimate::CausalForest::new())
            }
            _ => return Err(crate::value_err("unknown external estimator")),
        })
    }

    fn spec(&self) -> antecedent::EstimatorSpec {
        match self {
            Self::Dml(estimator) => estimator.clone().into(),
            Self::Aipw(estimator) => estimator.clone().into(),
            Self::DrLearner(estimator) => estimator.clone().into(),
            Self::CausalForest(estimator) => estimator.clone().into(),
        }
    }

    /// The analysed row ids and adjustment columns the estimator prepares on the compiled
    /// estimand. Preparation selects rows and builds the design; it fits nothing.
    fn rows_and_adjustment(
        &self,
        data: &antecedent::prelude::TabularData,
        estimand: &antecedent::prelude::IdentifiedEstimand,
        query: &antecedent::query::AverageEffectQuery,
    ) -> PyResult<(Vec<u64>, Vec<u32>)> {
        let problem = match self {
            Self::Dml(estimator) => estimator.prepare(data, estimand, query),
            Self::Aipw(estimator) => estimator.prepare(data, estimand, query),
            Self::DrLearner(estimator) => estimator.prepare(data, estimand, query),
            Self::CausalForest(estimator) => estimator.prepare(data, estimand, query),
        }
        .map_err(crate::py_err)?;
        Ok((
            problem.row_index.iter().map(|row| row.get()).collect(),
            problem.adjustment_set.iter().map(|id| id.raw()).collect(),
        ))
    }
}

/// What `declared_learner_ate` returns: the `external_dml_ate` tuple followed by the
/// out-of-fold propensity of each analysed row, when the fit keeps an AIPW score table.
type DeclaredAte = (
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
    Option<Vec<f64>>,
);

/// Run the compiled static average-effect study with two registered Python providers declared
/// as its learner factories under `identity`.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment_name, outcome_name, outcome, treatment, *, identity, folds=5, weights=None, seed=1, cancel=None, method="dml", score="aipw", memory_limit_bytes=None))]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(super) fn declared_learner_ate(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<numpy::PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment_name: String,
    outcome_name: String,
    outcome: Py<PyAny>,
    treatment: Py<PyAny>,
    identity: String,
    folds: usize,
    weights: Option<numpy::PyReadonlyArray1<'_, f64>>,
    seed: u64,
    cancel: Option<crate::PyCancellationToken>,
    method: &str,
    score: &str,
    memory_limit_bytes: Option<u64>,
) -> PyResult<DeclaredAte> {
    use antecedent::data::TableView;
    if names.len() != columns.len() || names.is_empty() {
        return Err(crate::value_err("one data column is required per name"));
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
    let _input_hold = ctx
        .acquire_admission(0, super::input_bytes(&columns, weights.as_ref())?)
        .map_err(|e| admission_error(e, &ctx))?;
    let data = super::learner_table(&names, &columns)?;
    let sampling = weights
        .as_ref()
        .map(super::copy_numeric_array)
        .map(antecedent::estimate::SamplingWeights::new)
        .transpose()
        .map_err(crate::py_err)?;
    let configured = Configured::new(method, score, folds, sampling)?;
    let result: PyResult<_> = py.detach(|| {
        let t = data.schema().id_of(&treatment_name).map_err(crate::py_err)?;
        let y = data.schema().id_of(&outcome_name).map_err(crate::py_err)?;
        let dag = crate::dag_from_named_edges(data.schema(), &edges)?;
        let query = antecedent::query::AverageEffectQuery::binary_ate(t, y);
        let factories = LearnerFactories::new(
            Arc::new(PythonFactory {
                learner: outcome,
                task: PredictionTask::Regression,
                weights: outcome_weights,
                state: Arc::clone(&state),
            }),
            Arc::new(PythonFactory {
                learner: treatment,
                task: PredictionTask::BinaryProbability,
                weights: treatment_weights,
                state: Arc::clone(&state),
            }),
            identity,
        )
        .map_err(crate::py_err)?;
        let study = antecedent::Study::tabular(data.clone())
            .graph(dag)
            .query(query.clone())
            .estimator(configured.spec())
            .learner_factories(factories)
            .refute(antecedent::RefuteSuite::None)
            .build()
            .map_err(crate::py_err)?;
        let compiled = study.run(&ctx).map_err(crate::py_err)?;
        let effect = compiled
            .estimate()
            .as_effect()
            .ok_or_else(|| crate::value_err("the compiled study returned no effect estimate"))?;
        let (row_ids, adjustment) =
            configured.rows_and_adjustment(&data, &compiled.estimand, &query)?;
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
        let propensity = effect.scores.table.as_ref().and_then(|table| {
            let n = table.n_rows;
            (table.propensities.len() == 2 * n).then(|| table.propensities[n..].to_vec())
        });
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
            propensity,
        ))
    });
    if let Some(error) =
        state.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take()
    {
        return Err(error);
    }
    let (estimate, se, rows, fold_ids, adjustment, provenance, assumptions, cate, cate_se, e) =
        result?;
    let receipts = std::mem::take(
        &mut *state.receipts.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    Ok((
        estimate,
        se,
        rows,
        fold_ids,
        adjustment,
        provenance,
        assumptions,
        cate,
        cate_se,
        receipts,
        e,
    ))
}
