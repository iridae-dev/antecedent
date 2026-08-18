//! Compile-once / re-estimate-many [`PreparedStudy`] Python OO surface.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, PreparedStudy, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery,
};
use antecedent_data::{TableView, tabular_from_record_batch};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::response_api::{ResponseAnalysisResult, response_result};
use crate::{
    AteAnalysisResult, ate_result_from_analysis, columns_to_batch, dag_from_named_edges,
    detach_catch, py_err, py_execution_context_ext, suite_from_refute,
};

/// Durable prepare-once / estimate-many handle for static ATE on a supplied DAG.
#[pyclass(name = "PreparedAnalysis")]
pub struct PyPreparedAnalysis {
    /// Arc so per-click estimate/refute detach with a refcount bump, not a
    /// deep `PreparedStudy` clone; `refresh` clones-on-write to swap data.
    inner: Arc<PreparedStudy>,
    names: Vec<String>,
    /// Last estimate result retained for second-click refute.
    last: Option<antecedent::StudyResult>,
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Compile once from tabular columns + DAG edges (static AverageEffect).
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        edges,
        treatment,
        outcome,
        *,
        control_level=0.0,
        active_level=1.0,
        identifier=None,
        estimator=None,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        refute=None,
        seed=1,
        bootstrap=50,
        threads=1,
        latency=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        edges: Vec<(String, String)>,
        treatment: String,
        outcome: String,
        control_level: f64,
        active_level: f64,
        identifier: Option<String>,
        estimator: Option<String>,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        bootstrap: u32,
        threads: u32,
        latency: Option<String>,
    ) -> PyResult<Self> {
        let batch = columns_to_batch(&names, &columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };
        drop(columns);

        detach_catch(py, move || {
            let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
            let data = loaded.data;
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
            let mut builder = Study::tabular(data)
                .graph(dag)
                .query(query)
                .refute(suite)
                .bootstrap_replicates(bootstrap);
            if let Some(mode) = latency_mode {
                builder = builder.latency_mode(mode);
            }
            if let Some(id) = identifier {
                builder = builder.identifier(
                    id.parse::<antecedent::IdentifierId>()
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                );
            }
            if let Some(est) = estimator {
                builder = builder.estimator(
                    est.parse::<antecedent::EstimatorId>()
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                );
            }
            if let Some(mode) = inference.as_deref() {
                builder = apply_inference(builder, mode, n_draws, prior_scale)?;
            }
            let analysis = builder.build().map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let prepared = analysis.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None })
        })
    }

    /// Compile once from tabular columns + DAG edges (static ResponseCurve).
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        edges,
        treatment,
        outcome,
        grid,
        *,
        identifier=None,
        estimator=None,
        seed=1,
        threads=1,
        latency=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        edges: Vec<(String, String)>,
        treatment: String,
        outcome: String,
        grid: Vec<f64>,
        identifier: Option<String>,
        estimator: Option<String>,
        seed: u64,
        threads: u32,
        latency: Option<String>,
    ) -> PyResult<Self> {
        let batch = columns_to_batch(&names, &columns)?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };
        drop(columns);

        detach_catch(py, move || {
            let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
            let data = loaded.data;
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let query = CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: y_id,
                treatment: ContinuousDomain::new(t_id, GridSpec::Values(grid.into())),
            }));
            let mut builder = Study::tabular(data)
                .graph(dag)
                .query(query)
                .refute(antecedent::RefuteSuite::None)
                .bootstrap_replicates(0);
            if let Some(mode) = latency_mode {
                builder = builder.latency_mode(mode);
            }
            if let Some(id) = identifier {
                builder = builder.identifier(
                    id.parse::<antecedent::IdentifierId>()
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                );
            }
            if let Some(est) = estimator {
                builder = builder.estimator(
                    est.parse::<antecedent::EstimatorId>()
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                );
            }
            let analysis = builder.build().map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let prepared = analysis.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None })
        })
    }

    /// Re-estimate on new columns (same schema) without recompiling.
    #[pyo3(signature = (names, columns, *, seed=1, threads=1))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        seed: u64,
        threads: u32,
    ) -> PyResult<AteAnalysisResult> {
        if names != self.names {
            return Err(PyValueError::new_err(
                "prepared estimate requires the same column names (order) as prepare",
            ));
        }
        let batch = columns_to_batch(&names, &columns)?;
        drop(columns);
        let inner = Arc::clone(&self.inner);
        let out_names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = inner.estimate(&loaded.data, &ctx).map_err(py_err)?;
            let mapped = ate_result_from_analysis(&out_names, result.clone(), false)?;
            Ok((mapped, result))
        })?;
        self.last = Some(result);
        Ok(mapped)
    }

    /// Re-estimate a prepared ResponseCurve (same schema) without recompiling.
    #[pyo3(signature = (names, columns, *, seed=1, threads=1))]
    fn estimate_response(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        seed: u64,
        threads: u32,
    ) -> PyResult<ResponseAnalysisResult> {
        if names != self.names {
            return Err(PyValueError::new_err(
                "prepared estimate requires the same column names (order) as prepare",
            ));
        }
        let batch = columns_to_batch(&names, &columns)?;
        drop(columns);
        let inner = Arc::clone(&self.inner);
        let out_names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = inner.estimate(&loaded.data, &ctx).map_err(py_err)?;
            let mapped = response_from_study(&out_names, &result)?;
            Ok((mapped, result))
        })?;
        self.last = Some(result);
        Ok(mapped)
    }

    /// Replace retained data and re-estimate (same schema).
    #[pyo3(signature = (names, columns, *, seed=1, threads=1))]
    fn refresh(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        seed: u64,
        threads: u32,
    ) -> PyResult<AteAnalysisResult> {
        if names != self.names {
            return Err(PyValueError::new_err(
                "prepared refresh requires the same column names (order) as prepare",
            ));
        }
        let batch = columns_to_batch(&names, &columns)?;
        drop(columns);
        let mut inner = (*self.inner).clone();
        let out_names = self.names.clone();
        let (updated, mapped, result) = detach_catch(py, move || {
            let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = inner.refresh(loaded.data, &ctx).map_err(py_err)?;
            let mapped = ate_result_from_analysis(&out_names, result.clone(), false)?;
            Ok((inner, mapped, result))
        })?;
        self.inner = Arc::new(updated);
        self.last = Some(result);
        Ok(mapped)
    }

    /// Replace retained data and re-estimate a prepared ResponseCurve.
    #[pyo3(signature = (names, columns, *, seed=1, threads=1))]
    fn refresh_response(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        seed: u64,
        threads: u32,
    ) -> PyResult<ResponseAnalysisResult> {
        if names != self.names {
            return Err(PyValueError::new_err(
                "prepared refresh requires the same column names (order) as prepare",
            ));
        }
        let batch = columns_to_batch(&names, &columns)?;
        drop(columns);
        let mut inner = (*self.inner).clone();
        let out_names = self.names.clone();
        let (updated, mapped, result) = detach_catch(py, move || {
            let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = inner.refresh(loaded.data, &ctx).map_err(py_err)?;
            let mapped = response_from_study(&out_names, &result)?;
            Ok((inner, mapped, result))
        })?;
        self.inner = Arc::new(updated);
        self.last = Some(result);
        Ok(mapped)
    }

    /// Second-click refute against the last estimate (same schema data).
    #[pyo3(signature = (names, columns, suite, *, seed=1, threads=1, cancel=None))]
    fn refute(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        suite: Bound<'_, PyAny>,
        seed: u64,
        threads: u32,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<AteAnalysisResult> {
        if names != self.names {
            return Err(PyValueError::new_err(
                "prepared refute requires the same column names (order) as prepare",
            ));
        }
        let prior = self
            .last
            .clone()
            .ok_or_else(|| PyValueError::new_err("call estimate/refresh before refute"))?;
        let batch = columns_to_batch(&names, &columns)?;
        let refute_suite = suite_from_refute(Some(&suite))?;
        let cancel_token = cancel.map(|c| c.inner);
        drop(columns);
        let inner = Arc::clone(&self.inner);
        let out_names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                cancel_token,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = inner.refute(&prior, &loaded.data, refute_suite, &ctx).map_err(py_err)?;
            let mapped = ate_result_from_analysis(&out_names, result.clone(), false)?;
            Ok((mapped, result))
        })?;
        self.last = Some(result);
        Ok(mapped)
    }

    #[getter]
    fn names(&self) -> Vec<String> {
        self.names.clone()
    }

    /// Physical-plan highlights retained from prepare (no recompile).
    fn plan_summary(&self) -> std::collections::HashMap<String, String> {
        let rec = &self.inner.plan().record;
        let mut out = std::collections::HashMap::new();
        out.insert("plan_id".into(), rec.plan_id.to_string());
        if let Some(b) = rec.estimated_peak_memory_bytes {
            out.insert("estimated_peak_memory_bytes".into(), b.to_string());
        }
        if let Some(b) = rec.workspace_bytes {
            out.insert("workspace_bytes".into(), b.to_string());
        }
        if let Some(b) = rec.batch_size {
            out.insert("batch_size".into(), b.to_string());
        }
        out.insert("worker_threads".into(), rec.worker_threads.to_string());
        out.insert("expected_python_crossings".into(), rec.expected_python_crossings.to_string());
        out.insert("deterministic_reductions".into(), rec.deterministic_reductions.to_string());
        let kernels: Vec<String> =
            rec.kernels.iter().map(|(name, k)| format!("{name}:{k:?}")).collect();
        out.insert("kernels".into(), kernels.join(","));
        out
    }
}

fn response_from_study(
    names: &[String],
    result: &antecedent::StudyResult,
) -> PyResult<ResponseAnalysisResult> {
    let response = result.response.clone().ok_or_else(|| {
        PyValueError::new_err("prepared response estimate did not carry a response payload")
    })?;
    let name_of = |id: antecedent_core::VariableId| {
        names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("var{}", id.raw()))
    };
    let treatments = vec![name_of(result.treatment)];
    let outcomes = vec![name_of(result.outcome)];
    let adjustment_set = result.estimand.adjustment_set.iter().copied().map(name_of).collect();
    response_result(response, treatments, outcomes, adjustment_set)
}

fn apply_inference(
    builder: antecedent::StudyBuilder,
    mode: &str,
    n_draws: usize,
    prior_scale: f64,
) -> PyResult<antecedent::StudyBuilder> {
    match mode.to_ascii_lowercase().as_str() {
        "bayesian" | "bayesian.laplace" | "laplace" => {
            let cfg = BayesianConfig::laplace().n_draws(n_draws).prior_scale(prior_scale);
            Ok(builder.inference(InferenceMode::Bayesian(cfg)))
        }
        "bayesian.conjugate" | "conjugate" => {
            let cfg = BayesianConfig::conjugate().n_draws(n_draws).prior_scale(prior_scale);
            Ok(builder.inference(InferenceMode::Bayesian(cfg)))
        }
        "bayesian.hmc" | "hmc" => {
            let cfg = BayesianConfig::hmc().n_draws(n_draws).prior_scale(prior_scale);
            Ok(builder.inference(InferenceMode::Bayesian(cfg)))
        }
        "frequentist" => Ok(builder.inference(InferenceMode::Frequentist)),
        other => Err(PyValueError::new_err(format!(
            "unknown inference mode {other:?}; use frequentist|bayesian|conjugate|hmc"
        ))),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPreparedAnalysis>()?;
    Ok(())
}
