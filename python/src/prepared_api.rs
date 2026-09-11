//! Compile-once / re-estimate-many [`PreparedStudy`] Python OO surface.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::discovery::{
    BayesianDiscoverParams, GraphMcmcSchedule, discover_dbn_posterior, discover_exact_dag_posterior,
};
use antecedent::{BayesianConfig, EstimatorId, IdentifierId, InferenceMode, PreparedStudy, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ContinuousDomain, GridSpec,
    Intervention, InterventionalDistributionQuery, MediationContrast, MediationQuery,
    PathSpecificEffectQuery, ResponseFunctional, ResponseQuery, TemporalResponseSpec, Value,
};
use antecedent_data::{TableView, TabularData};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use crate::response_api::{
    ResponseAnalysisResult, attach_study_response_meta, build_functional, response_result,
};
use crate::temporal_api::{TemporalClassGraph, bind_temporal_class};
use crate::{
    AteAnalysisResult, ate_result_from_analysis, dag_from_named_edges, detach_catch, graphs,
    py_err, py_execution_context_ext, py_msg, require_named_graph_order, series_from_tabular,
    suite_from_refute, tabular_from_arrow_c_objs, tabular_from_numpy, tabular_from_py_columns,
    temporal_dag_from_schema_edges,
};

fn require_prepared_names(expected: &[String], names: &[String], op: &str) -> PyResult<()> {
    if names != expected {
        return Err(PyValueError::new_err(format!(
            "prepared {op} requires the same column names (order) as prepare"
        )));
    }
    Ok(())
}

/// Durable prepare-once / estimate-many handle for static ATE on a supplied DAG.
#[pyclass(name = "PreparedAnalysis")]
pub struct PyPreparedAnalysis {
    /// Arc so per-click estimate/refute detach with a refcount bump, not a
    /// deep `PreparedStudy` clone; `refresh` clones-on-write to swap data.
    inner: Arc<PreparedStudy>,
    names: Vec<String>,
    /// Last estimate result retained for second-click refute.
    last: Option<antecedent::StudyResult>,
    /// When true, estimate/refresh clicks use series data (`estimate_series`).
    series: bool,
}

impl PyPreparedAnalysis {
    pub(crate) fn from_study(prepared: PreparedStudy, names: Vec<String>) -> Self {
        Self { inner: Arc::new(prepared), names, last: None, series: false }
    }

    fn finish_ate_estimate(
        &mut self,
        py: Python<'_>,
        data: TabularData,
        seed: u64,
        threads: u32,
    ) -> PyResult<AteAnalysisResult> {
        let inner = Arc::clone(&self.inner);
        let out_names = self.names.clone();
        let series = self.series;
        let (mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = if series {
                let series_data = series_from_tabular(data)?;
                inner.estimate_series(&series_data, &ctx).map_err(py_err)?
            } else {
                inner.estimate(&data, &ctx).map_err(py_err)?
            };
            let mapped = ate_result_from_analysis(&out_names, result.clone(), false)?;
            Ok((mapped, result))
        })?;
        self.last = Some(result);
        Ok(mapped)
    }

    fn finish_response_estimate(
        &mut self,
        py: Python<'_>,
        data: TabularData,
        seed: u64,
        threads: u32,
    ) -> PyResult<ResponseAnalysisResult> {
        let inner = Arc::clone(&self.inner);
        let out_names = self.names.clone();
        let series = self.series;
        let (mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = if series {
                let series_data = series_from_tabular(data)?;
                inner.estimate_series(&series_data, &ctx).map_err(py_err)?
            } else {
                inner.estimate(&data, &ctx).map_err(py_err)?
            };
            let mapped = response_from_study(&out_names, &result)?;
            Ok((mapped, result))
        })?;
        self.last = Some(result);
        Ok(mapped)
    }

    fn finish_ate_refresh(
        &mut self,
        py: Python<'_>,
        data: TabularData,
        seed: u64,
        threads: u32,
    ) -> PyResult<AteAnalysisResult> {
        let mut inner = (*self.inner).clone();
        let out_names = self.names.clone();
        let series = self.series;
        let (updated, mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = if series {
                let series_data = series_from_tabular(data)?;
                inner.refresh_series(series_data, &ctx).map_err(py_err)?
            } else {
                inner.refresh(data, &ctx).map_err(py_err)?
            };
            let mapped = ate_result_from_analysis(&out_names, result.clone(), false)?;
            Ok((inner, mapped, result))
        })?;
        self.inner = Arc::new(updated);
        self.last = Some(result);
        Ok(mapped)
    }

    fn finish_response_refresh(
        &mut self,
        py: Python<'_>,
        data: TabularData,
        seed: u64,
        threads: u32,
    ) -> PyResult<ResponseAnalysisResult> {
        let mut inner = (*self.inner).clone();
        let out_names = self.names.clone();
        let series = self.series;
        let (updated, mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = if series {
                let series_data = series_from_tabular(data)?;
                inner.refresh_series(series_data, &ctx).map_err(py_err)?
            } else {
                inner.refresh(data, &ctx).map_err(py_err)?
            };
            let mapped = response_from_study(&out_names, &result)?;
            Ok((inner, mapped, result))
        })?;
        self.inner = Arc::new(updated);
        self.last = Some(result);
        Ok(mapped)
    }

    fn finish_ate_refute(
        &mut self,
        py: Python<'_>,
        data: TabularData,
        suite: Bound<'_, PyAny>,
        seed: u64,
        threads: u32,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<AteAnalysisResult> {
        let prior = self
            .last
            .clone()
            .ok_or_else(|| PyValueError::new_err("call estimate/refresh before refute"))?;
        let refute_suite = suite_from_refute(Some(&suite))?;
        let cancel_token = cancel.map(|c| c.inner);
        let inner = Arc::clone(&self.inner);
        let out_names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                threads,
                cancel_token,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = inner.refute(&prior, &data, refute_suite, &ctx).map_err(py_err)?;
            let mapped = ate_result_from_analysis(&out_names, result.clone(), false)?;
            Ok((mapped, result))
        })?;
        self.last = Some(result);
        Ok(mapped)
    }
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
        accepted=false,
        outcome_functional=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
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
        accepted: bool,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };

        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let mut query =
                AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
            if let Some(functional) = functional {
                query = query.with_outcome_functional(functional);
            }
            let mut builder = if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(dag))
            } else {
                Study::tabular(data).graph(dag)
            }
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
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Compile once for AverageEffect on a CoDetermined / Unknown tier background.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        tiers,
        within_tier,
        treatment,
        outcome,
        *,
        control_level=0.0,
        active_level=1.0,
        estimator=None,
        refute=None,
        seed=1,
        bootstrap=0,
        threads=1,
        latency=None,
        outcome_functional=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_tiered(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        tiers: Vec<Vec<String>>,
        within_tier: String,
        treatment: String,
        outcome: String,
        control_level: f64,
        active_level: f64,
        estimator: Option<String>,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        bootstrap: u32,
        threads: u32,
        latency: Option<String>,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };
        detach_catch(py, move || {
            let within = crate::parse_within_tier(Some(within_tier.as_str()))?;
            let named: Vec<Vec<&str>> =
                tiers.iter().map(|tier| tier.iter().map(String::as_str).collect()).collect();
            let background =
                antecedent_graph::TieredBackground::from_named(data.schema(), &named, within)
                    .map_err(py_err)?;
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let mut query =
                AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
            if let Some(functional) = functional {
                query = query.with_outcome_functional(functional);
            }
            let mut builder = Study::tabular(data)
                .tiered_background(background)
                .map_err(py_err)?
                .query(query)
                .refute(suite)
                .bootstrap_replicates(bootstrap);
            if let Some(mode) = latency_mode {
                builder = builder.latency_mode(mode);
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
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Compile once for AverageEffect on a supplied PAG (generalized adjustment).
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        graph,
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
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_pag(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: graphs::Pag,
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
        accepted: bool,
    ) -> PyResult<Self> {
        require_named_graph_order(&graph.names, &names, "Pag")?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };
        let pag = graph.pag;

        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
            let mut builder = if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(pag))
            } else {
                Study::tabular(data).graph(pag)
            }
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
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Compile once for AverageEffect on a supplied CPDAG (MEC envelope).
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        graph,
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
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_cpdag(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: graphs::Cpdag,
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
        accepted: bool,
    ) -> PyResult<Self> {
        require_named_graph_order(&graph.names, &names, "Cpdag")?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };
        let cpdag = graph.cpdag;

        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
            let mut builder = if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(cpdag))
            } else {
                Study::tabular(data).graph(cpdag)
            }
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
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Compile once for ResponseCurve / InterventionResponse on a supplied PAG.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        graph,
        kind,
        treatments,
        outcomes,
        *,
        grid=None,
        intervention_kinds=None,
        intervention_parameters=None,
        identifier=None,
        estimator=None,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        seed=1,
        threads=1,
        latency=None,
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_pag_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: graphs::Pag,
        kind: String,
        treatments: Vec<String>,
        outcomes: Vec<String>,
        grid: Option<Vec<f64>>,
        intervention_kinds: Option<Vec<String>>,
        intervention_parameters: Option<Vec<Vec<f64>>>,
        identifier: Option<String>,
        estimator: Option<String>,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        seed: u64,
        threads: u32,
        latency: Option<String>,
        accepted: bool,
    ) -> PyResult<Self> {
        require_named_graph_order(&graph.names, &names, "Pag")?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let latency_mode = parse_latency(latency)?;
        let pag = graph.pag;
        detach_catch(py, move || {
            prepare_class_response(
                data,
                names,
                ClassResponseGraph::Pag(pag),
                kind,
                treatments,
                outcomes,
                grid,
                intervention_kinds,
                intervention_parameters,
                identifier,
                estimator,
                inference,
                n_draws,
                prior_scale,
                seed,
                threads,
                latency_mode,
                accepted,
            )
        })
    }

    /// Compile once for ResponseCurve / InterventionResponse on a supplied CPDAG.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        graph,
        kind,
        treatments,
        outcomes,
        *,
        grid=None,
        intervention_kinds=None,
        intervention_parameters=None,
        identifier=None,
        estimator=None,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        seed=1,
        threads=1,
        latency=None,
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_cpdag_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: graphs::Cpdag,
        kind: String,
        treatments: Vec<String>,
        outcomes: Vec<String>,
        grid: Option<Vec<f64>>,
        intervention_kinds: Option<Vec<String>>,
        intervention_parameters: Option<Vec<Vec<f64>>>,
        identifier: Option<String>,
        estimator: Option<String>,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        seed: u64,
        threads: u32,
        latency: Option<String>,
        accepted: bool,
    ) -> PyResult<Self> {
        require_named_graph_order(&graph.names, &names, "Cpdag")?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let latency_mode = parse_latency(latency)?;
        let cpdag = graph.cpdag;
        detach_catch(py, move || {
            prepare_class_response(
                data,
                names,
                ClassResponseGraph::Cpdag(cpdag),
                kind,
                treatments,
                outcomes,
                grid,
                intervention_kinds,
                intervention_parameters,
                identifier,
                estimator,
                inference,
                n_draws,
                prior_scale,
                seed,
                threads,
                latency_mode,
                accepted,
            )
        })
    }

    /// Compile once for AverageEffect on a supplied ADMG (general ID + FunctionalEffect).
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        graph,
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
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_admg(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: graphs::Admg,
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
        accepted: bool,
    ) -> PyResult<Self> {
        require_named_graph_order(&graph.names, &names, "Admg")?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };
        let admg = graph.admg;

        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
            let mut builder = if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(admg))
            } else {
                Study::tabular(data).graph(admg)
            }
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
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Identification reads names and graph only; no dummy statistical fit.
    #[staticmethod]
    #[pyo3(signature = (names, edges, kind, treatments, outcomes, *, mediators=Vec::new(),
        contrast="mediated", control_level=0.0, active_level=1.0, at=None, direction=None,
        order=1, scale="identity", weighting="observed"))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn identify_existing(
        names: Vec<String>,
        edges: Vec<(String, String)>,
        kind: &str,
        treatments: Vec<String>,
        outcomes: Vec<String>,
        mediators: Vec<String>,
        contrast: &str,
        control_level: f64,
        active_level: f64,
        at: Option<Vec<f64>>,
        direction: Option<Vec<f64>>,
        order: u8,
        scale: &str,
        weighting: &str,
    ) -> PyResult<(String, String, Vec<String>, String)> {
        let empty: &[f64] = &[];
        let data = TabularData::from_f64_columns(names.iter().map(|name| (name.as_str(), empty)))
            .map_err(py_err)?;
        let dag = dag_from_named_edges(data.schema(), &edges)?;
        let query = if kind == "mediation" || kind == "counterfactual" {
            if treatments.len() != 1 || outcomes.len() != 1 {
                return Err(PyValueError::new_err("one treatment/outcome required"));
            }
            static_kind_query(
                data.schema(),
                kind,
                &treatments[0],
                &outcomes[0],
                &mediators,
                contrast,
                control_level,
                active_level,
            )?
        } else {
            let ts = crate::response_api::resolve_names(data.schema(), &treatments)?;
            let ys = crate::response_api::resolve_names(data.schema(), &outcomes)?;
            CausalQuery::Response(ResponseQuery::new(build_functional(
                kind,
                &ts,
                &ys,
                None,
                at,
                direction,
                None,
                None,
                order,
                crate::response_api::parse_scale(scale)?,
                crate::response_api::parse_weighting(weighting)?,
            )?))
        };
        let id = antecedent::identify_dag(&dag, &query).map_err(py_err)?;
        let adjustment = id
            .estimands()
            .first()
            .map(|e| e.adjustment_set.iter().map(|v| names[v.as_usize()].clone()).collect())
            .unwrap_or_default();
        let method = id.estimands().first().map(|e| e.method.to_string()).unwrap_or_default();
        Ok((format!("{:?}", id.status()), method, adjustment, id.strategy().as_str().to_owned()))
    }

    /// Static mediation and counterfactuals retain the same staged result axes.
    #[staticmethod]
    #[pyo3(signature = (names, columns, edges, kind, treatment, outcome, *, mediators=Vec::new(),
        contrast="mediated", control_level=0.0, active_level=1.0, refute=None,
        bootstrap=0, accepted=false, seed=1, threads=1))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_static_kind(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        kind: &str,
        treatment: String,
        outcome: String,
        mediators: Vec<String>,
        contrast: &str,
        control_level: f64,
        active_level: f64,
        refute: Option<Bound<'_, PyAny>>,
        bootstrap: u32,
        accepted: bool,
        seed: u64,
        threads: u32,
    ) -> PyResult<Self> {
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let query = static_kind_query(
            data.schema(),
            kind,
            &treatment,
            &outcome,
            &mediators,
            contrast,
            control_level,
            active_level,
        )?;
        let suite = suite_from_refute(refute.as_ref())?;
        detach_catch(py, move || {
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let builder = if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(dag))
            } else {
                Study::tabular(data).graph(dag)
            };
            let study = builder
                .query(query)
                .refute(suite)
                .bootstrap_replicates(bootstrap)
                .build()
                .map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let prepared = study.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Freeze identification for a complete-data static derivative.
    #[staticmethod]
    #[pyo3(signature = (names, columns, edges, kind, treatments, outcomes, *, at=None,
        direction=None, order=1, scale="identity", weighting="observed", bandwidth=None,
        accepted=false, seed=1, threads=1))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_derivative(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        kind: String,
        treatments: Vec<String>,
        outcomes: Vec<String>,
        at: Option<Vec<f64>>,
        direction: Option<Vec<f64>>,
        order: u8,
        scale: &str,
        weighting: &str,
        bandwidth: Option<f64>,
        accepted: bool,
        seed: u64,
        threads: u32,
    ) -> PyResult<Self> {
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let scale = crate::response_api::parse_scale(scale)?;
        let weighting = crate::response_api::parse_weighting(weighting)?;
        detach_catch(py, move || {
            let ts = crate::response_api::resolve_names(data.schema(), &treatments)?;
            let ys = crate::response_api::resolve_names(data.schema(), &outcomes)?;
            let functional = build_functional(
                &kind, &ts, &ys, None, at, direction, None, None, order, scale, weighting,
            )?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let builder = if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(dag))
            } else {
                Study::tabular(data).graph(dag)
            };
            let study = builder
                .query(CausalQuery::Response(ResponseQuery::new(functional)))
                .response_options(antecedent_estimate::ContinuousResponseOptions {
                    bandwidth,
                    ..Default::default()
                })
                .refute(antecedent::RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let prepared = study.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
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
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        seed=1,
        threads=1,
        latency=None,
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        treatment: String,
        outcome: String,
        grid: Vec<f64>,
        identifier: Option<String>,
        estimator: Option<String>,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        seed: u64,
        threads: u32,
        latency: Option<String>,
        accepted: bool,
    ) -> PyResult<Self> {
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };

        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let query = CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: y_id,
                treatment: ContinuousDomain::new(t_id, GridSpec::Values(grid.into())),
            }));
            let mut builder = if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(dag))
            } else {
                Study::tabular(data).graph(dag)
            }
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
            builder = apply_inference(
                builder,
                inference.as_deref().unwrap_or("frequentist"),
                n_draws,
                prior_scale,
            )?;
            let analysis = builder.build().map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let prepared = analysis.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Compile once for a temporal ResponseCurve / InterventionResponse (series data).
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        edges,
        kind,
        treatments,
        outcomes,
        *,
        grid=None,
        intervention_kinds=None,
        intervention_parameters=None,
        horizons,
        policy=crate::temporal_license::DEFAULT_POLICY,
        treatment_lag=crate::temporal_license::DEFAULT_TREATMENT_LAG,
        max_history_lag=None,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        seed=1,
        threads=1,
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_temporal_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, u32, String, u32)>,
        kind: String,
        treatments: Vec<String>,
        outcomes: Vec<String>,
        grid: Option<Vec<f64>>,
        intervention_kinds: Option<Vec<String>>,
        intervention_parameters: Option<Vec<Vec<f64>>>,
        horizons: Vec<u32>,
        policy: &str,
        treatment_lag: u32,
        max_history_lag: Option<u32>,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        seed: u64,
        threads: u32,
        accepted: bool,
    ) -> PyResult<Self> {
        let (tabular, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let policy = policy.to_ascii_lowercase();
        detach_catch(py, move || {
            let series = series_from_tabular(tabular)?;
            let dag = temporal_dag_from_schema_edges(series.schema(), &edges)?;
            let treatment_ids: Vec<_> = treatments
                .iter()
                .map(|n| series.schema().id_of(n).map_err(py_err))
                .collect::<PyResult<_>>()?;
            let outcome_ids: Vec<_> = outcomes
                .iter()
                .map(|n| series.schema().id_of(n).map_err(py_err))
                .collect::<PyResult<_>>()?;
            let functional = build_functional(
                &kind,
                &treatment_ids,
                &outcome_ids,
                grid,
                None,
                None,
                intervention_kinds,
                intervention_parameters,
                1,
                antecedent_core::DerivativeScale::Identity,
                antecedent_core::DerivativeWeighting::Observed,
            )?;
            let temporal_policy = crate::temporal_license::policy_at_lag(policy, treatment_lag)?;
            let temporal = TemporalResponseSpec::new(horizons, temporal_policy, max_history_lag)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let query =
                CausalQuery::Response(ResponseQuery::new(functional).with_temporal(temporal));
            let mut builder = Study::series(series);
            builder = if accepted {
                builder.graph(antecedent::AcceptedGraph::from(dag))
            } else {
                builder.graph(dag)
            };
            builder =
                builder.query(query).refute(antecedent::RefuteSuite::None).bootstrap_replicates(0);
            builder = apply_inference(
                builder,
                inference.as_deref().unwrap_or("frequentist"),
                n_draws,
                prior_scale,
            )?;
            let analysis = builder.build().map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let prepared = analysis.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: true })
        })
    }

    /// Compile once for licensed Pulse / single-step Sustained on a TemporalDag.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        edges,
        treatment,
        outcome,
        *,
        policy="pulse",
        window=None,
        treatment_lag=1,
        horizon_steps=1,
        active_level=1.0,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        refute=None,
        seed=1,
        bootstrap=0,
        threads=1,
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_temporal_effect(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, u32, String, u32)>,
        treatment: String,
        outcome: String,
        policy: &str,
        window: Option<(i32, i32)>,
        treatment_lag: u32,
        horizon_steps: u32,
        active_level: f64,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        bootstrap: u32,
        threads: u32,
        accepted: bool,
    ) -> PyResult<Self> {
        let (tabular, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let policy = policy.to_ascii_lowercase();
        let suite = suite_from_refute(refute.as_ref())?;
        detach_catch(py, move || {
            let series = series_from_tabular(tabular)?;
            let t_id = series.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = series.schema().id_of(&outcome).map_err(py_err)?;
            let dag = temporal_dag_from_schema_edges(series.schema(), &edges)?;
            let mut q = crate::temporal_api::temporal_query_from_policy(
                &policy,
                t_id,
                y_id,
                treatment_lag,
                horizon_steps,
                active_level,
            )?;
            if let Some((from, until)) = window {
                if policy != "sustained" {
                    return Err(PyValueError::new_err("window requires policy='sustained'"));
                }
                q = q.with_policy(antecedent_core::TemporalPolicy::sustained(from, until));
            }
            let mut builder = Study::series(series);
            builder = if accepted {
                builder.graph(antecedent::AcceptedGraph::from(dag))
            } else {
                builder.graph(dag)
            };
            builder = builder.temporal_query(q).refute(suite).bootstrap_replicates(bootstrap);
            builder = crate::temporal_api::apply_temporal_inference(
                builder,
                inference.as_deref(),
                n_draws,
                prior_scale,
                None,
            )?;
            let analysis = builder.build().map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let prepared = analysis.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: true })
        })
    }

    /// Compile once for Pulse / single-step Sustained on a TemporalCpdag.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        graph,
        treatment,
        outcome,
        *,
        policy="pulse",
        window=None,
        treatment_lag=1,
        horizon_steps=1,
        active_level=1.0,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        refute=None,
        seed=1,
        bootstrap=0,
        threads=1,
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_temporal_cpdag_effect(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: graphs::TemporalCpdag,
        treatment: String,
        outcome: String,
        policy: &str,
        window: Option<(i32, i32)>,
        treatment_lag: u32,
        horizon_steps: u32,
        active_level: f64,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        bootstrap: u32,
        threads: u32,
        accepted: bool,
    ) -> PyResult<Self> {
        require_named_graph_order(&graph.names, &names, "TemporalCpdag")?;
        prepare_temporal_class_effect(
            py,
            names,
            columns,
            TemporalClassGraph::Cpdag(graph.cpdag),
            treatment,
            outcome,
            policy,
            window,
            treatment_lag,
            horizon_steps,
            active_level,
            inference,
            n_draws,
            prior_scale,
            refute,
            seed,
            bootstrap,
            threads,
            accepted,
        )
    }

    /// Compile once for Pulse / single-step Sustained on a TemporalPag.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        graph,
        treatment,
        outcome,
        *,
        policy="pulse",
        window=None,
        treatment_lag=1,
        horizon_steps=1,
        active_level=1.0,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        refute=None,
        seed=1,
        bootstrap=0,
        threads=1,
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_temporal_pag_effect(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: graphs::TemporalPag,
        treatment: String,
        outcome: String,
        policy: &str,
        window: Option<(i32, i32)>,
        treatment_lag: u32,
        horizon_steps: u32,
        active_level: f64,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        bootstrap: u32,
        threads: u32,
        accepted: bool,
    ) -> PyResult<Self> {
        require_named_graph_order(&graph.names, &names, "TemporalPag")?;
        prepare_temporal_class_effect(
            py,
            names,
            columns,
            TemporalClassGraph::Pag(graph.pag),
            treatment,
            outcome,
            policy,
            window,
            treatment_lag,
            horizon_steps,
            active_level,
            inference,
            n_draws,
            prior_scale,
            refute,
            seed,
            bootstrap,
            threads,
            accepted,
        )
    }

    /// Compile once for licensed TemporalMediationEffect on a TemporalDag.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        edges,
        treatment,
        mediator,
        outcome,
        *,
        contrast="mediated",
        control_level=0.0,
        active_level=1.0,
        horizons=None,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        refute=None,
        seed=1,
        bootstrap=0,
        threads=1,
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_temporal_mediation(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, u32, String, u32)>,
        treatment: String,
        mediator: String,
        outcome: String,
        contrast: &str,
        control_level: f64,
        active_level: f64,
        horizons: Option<Vec<u32>>,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        bootstrap: u32,
        threads: u32,
        accepted: bool,
    ) -> PyResult<Self> {
        let (tabular, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let contrast = contrast.to_string();
        let suite = suite_from_refute(refute.as_ref())?;
        detach_catch(py, move || {
            let series = series_from_tabular(tabular)?;
            let t_id = series.schema().id_of(&treatment).map_err(py_err)?;
            let m_id = series.schema().id_of(&mediator).map_err(py_err)?;
            let y_id = series.schema().id_of(&outcome).map_err(py_err)?;
            let contrast = match contrast.to_ascii_lowercase().as_str() {
                "total" => MediationContrast::Total,
                "direct" => MediationContrast::Direct,
                "mediated" | "indirect" => MediationContrast::Mediated,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown mediation contrast {other:?}; use total|direct|mediated"
                    )));
                }
            };
            let mut q = MediationQuery::binary(t_id, y_id, [m_id], contrast);
            q.control = Intervention::set(t_id, Value::f64(control_level));
            q.active = Intervention::set(t_id, Value::f64(active_level));
            if let Some(hs) = horizons {
                q = q.with_horizons(hs).map_err(py_msg)?;
            }
            let dag = temporal_dag_from_schema_edges(series.schema(), &edges)?;
            let mut builder = Study::series(series);
            builder = if accepted {
                builder.graph(antecedent::AcceptedGraph::from(dag))
            } else {
                builder.graph(dag)
            };
            builder = builder
                .query(CausalQuery::Mediation(q))
                .refute(suite)
                .bootstrap_replicates(bootstrap);
            builder = apply_inference(
                builder,
                inference.as_deref().unwrap_or("frequentist"),
                n_draws,
                prior_scale,
            )?;
            let analysis = builder.build().map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let prepared = analysis.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: true })
        })
    }

    /// Compile once for licensed AverageEffect × graph_posterior × Bayesian or Frequentist.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        treatment,
        outcome,
        *,
        control_level=0.0,
        active_level=1.0,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        refute=None,
        seed=1,
        bootstrap=0,
        threads=1,
        posterior=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_graph_posterior_ate(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        treatment: String,
        outcome: String,
        control_level: f64,
        active_level: f64,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        bootstrap: u32,
        threads: u32,
        posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
    ) -> PyResult<Self> {
        let supplied = posterior
            .map(|bound| {
                let posterior = bound.borrow();
                posterior.require_bound_to(&names)?;
                posterior.to_rust()
            })
            .transpose()?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let gp = if let Some(gp) = supplied {
                gp
            } else {
                let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
                discover_exact_dag_posterior(&data, &vars, &BayesianDiscoverParams::default(), &ctx)
                    .map_err(py_err)?
            };
            let mut builder = Study::tabular(data)
                .graph_posterior(gp)
                .query(query)
                .refute(suite)
                .bootstrap_replicates(bootstrap);
            builder = apply_inference(
                builder,
                inference.as_deref().unwrap_or("conjugate"),
                n_draws,
                prior_scale,
            )?;
            let analysis = builder.build().map_err(py_err)?;
            let prepared = analysis.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Compile once for licensed Pulse/Sustained × DBN graph_posterior × Bayesian.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        treatment,
        outcome,
        *,
        policy="pulse",
        treatment_lag=1,
        horizon_steps=1,
        active_level=1.0,
        max_lag=1,
        force_mcmc=false,
        n_chains=2,
        n_warmup=200,
        mcmc_draws=400,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        refute=None,
        seed=1,
        threads=1,
        posterior=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_dbn_posterior_temporal(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        treatment: String,
        outcome: String,
        policy: &str,
        treatment_lag: u32,
        horizon_steps: u32,
        active_level: f64,
        max_lag: u32,
        force_mcmc: bool,
        n_chains: u32,
        n_warmup: u32,
        mcmc_draws: u32,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        threads: u32,
        posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
    ) -> PyResult<Self> {
        let supplied = posterior
            .map(|bound| {
                let posterior = bound.borrow();
                posterior.require_bound_to(&names)?;
                posterior.to_rust()
            })
            .transpose()?;
        let (tabular, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let policy = policy.to_ascii_lowercase();
        let suite = suite_from_refute(refute.as_ref())?;
        detach_catch(py, move || {
            let series = series_from_tabular(tabular)?;
            let t_id = series.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = series.schema().id_of(&outcome).map_err(py_err)?;
            let q = crate::temporal_api::temporal_query_from_policy(
                &policy,
                t_id,
                y_id,
                treatment_lag,
                horizon_steps,
                active_level,
            )?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let gp = if let Some(gp) = supplied {
                gp
            } else {
                let vars: Vec<_> = series.schema().variables().iter().map(|v| v.id).collect();
                let schedule =
                    GraphMcmcSchedule { n_chains, n_warmup, n_draws: mcmc_draws, thin: 1 };
                discover_dbn_posterior(
                    &series,
                    &vars,
                    &BayesianDiscoverParams::default(),
                    max_lag,
                    force_mcmc,
                    &schedule,
                    &ctx,
                )
                .map_err(py_err)?
            };
            let mut builder = Study::series(series)
                .graph_posterior(gp)
                .temporal_query(q)
                .refute(suite)
                .bootstrap_replicates(0);
            builder = crate::temporal_api::apply_temporal_inference(
                builder,
                Some(inference.as_deref().unwrap_or("conjugate")),
                n_draws,
                prior_scale,
                None,
            )?;
            let analysis = builder.build().map_err(py_err)?;
            let prepared = analysis.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: true })
        })
    }

    /// Compile once from tabular columns + DAG edges (static InterventionResponse).
    ///
    /// Reuses `response_api::build_functional`'s `"intervention_response"` branch
    /// (the same construction `analyze_response` uses) so a prepared handle's
    /// `ResponseFunctional::InterventionResponse` is built identically to the
    /// one-shot `analyze()` path. The generic `CausalQuery::Response(_)` branch on
    /// `PreparedStudy` (see `analysis/prepared.rs`) already caches identification
    /// for any response functional on a supplied `Dag`, so no new prepare-time
    /// machinery is needed beyond constructing the query.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        edges,
        outcome,
        treatments,
        intervention_kinds,
        intervention_parameters,
        *,
        estimator=None,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        outcome_functional=None,
        refute=None,
        seed=1,
        threads=1,
        latency=None,
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_intervention_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        outcome: String,
        treatments: Vec<String>,
        intervention_kinds: Vec<String>,
        intervention_parameters: Vec<Vec<f64>>,
        estimator: Option<String>,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        threads: u32,
        latency: Option<String>,
        accepted: bool,
    ) -> PyResult<Self> {
        let outcome_functional =
            crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };

        detach_catch(py, move || {
            let treatment_ids = crate::response_api::resolve_names(data.schema(), &treatments)?;
            let outcome_ids = crate::response_api::resolve_names(data.schema(), &[outcome])?;
            let functional = crate::response_api::build_functional(
                "intervention_response",
                &treatment_ids,
                &outcome_ids,
                None,
                None,
                None,
                Some(intervention_kinds),
                Some(intervention_parameters),
                1,
                antecedent_core::DerivativeScale::Identity,
                antecedent_core::DerivativeWeighting::Observed,
            )?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let mut response_query = ResponseQuery::new(functional);
            if let Some(functional) = outcome_functional {
                response_query = response_query.with_outcome_functional(functional);
            }
            let query = CausalQuery::Response(response_query);
            let mut builder = if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(dag))
            } else {
                Study::tabular(data).graph(dag)
            }
            .query(query)
            .refute(suite)
            .bootstrap_replicates(0);
            if let Some(mode) = latency_mode {
                builder = builder.latency_mode(mode);
            }
            if let Some(est) = estimator {
                builder = builder.estimator(
                    est.parse::<antecedent::EstimatorId>()
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                );
            }
            builder = apply_inference(
                builder,
                inference.as_deref().unwrap_or("frequentist"),
                n_draws,
                prior_scale,
            )?;
            let analysis = builder.build().map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let prepared = analysis.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Compile once for joint InterventionResponse on a CoDetermined tier closure.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        tiers,
        within_tier,
        outcome,
        treatments,
        intervention_kinds,
        intervention_parameters,
        *,
        outcome_functional=None,
        refute=None,
        seed=1,
        threads=1,
        latency=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_tiered_intervention_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        tiers: Vec<Vec<String>>,
        within_tier: String,
        outcome: String,
        treatments: Vec<String>,
        intervention_kinds: Vec<String>,
        intervention_parameters: Vec<Vec<f64>>,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        threads: u32,
        latency: Option<String>,
    ) -> PyResult<Self> {
        let outcome_functional =
            crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };

        detach_catch(py, move || {
            let within = crate::parse_within_tier(Some(within_tier.as_str()))?;
            let named: Vec<Vec<&str>> =
                tiers.iter().map(|tier| tier.iter().map(String::as_str).collect()).collect();
            let background =
                antecedent_graph::TieredBackground::from_named(data.schema(), &named, within)
                    .map_err(py_err)?;
            let treatment_ids = crate::response_api::resolve_names(data.schema(), &treatments)?;
            let outcome_ids = crate::response_api::resolve_names(data.schema(), &[outcome])?;
            let functional = crate::response_api::build_functional(
                "intervention_response",
                &treatment_ids,
                &outcome_ids,
                None,
                None,
                None,
                Some(intervention_kinds),
                Some(intervention_parameters),
                1,
                antecedent_core::DerivativeScale::Identity,
                antecedent_core::DerivativeWeighting::Observed,
            )?;
            let mut response_query = ResponseQuery::new(functional);
            if let Some(functional) = outcome_functional {
                response_query = response_query.with_outcome_functional(functional);
            }
            let query = CausalQuery::Response(response_query);
            let mut builder = Study::tabular(data)
                .tiered_background(background)
                .map_err(py_err)?
                .query(query)
                .estimator(antecedent::EstimatorId::CellAipw)
                .refute(suite)
                .bootstrap_replicates(0);
            if let Some(mode) = latency_mode {
                builder = builder.latency_mode(mode);
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
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Compile once from tabular columns + DAG edges (static ConditionalEffect).
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        edges,
        treatment,
        outcome,
        modifier,
        *,
        control_level=0.0,
        active_level=1.0,
        refute=None,
        inference=None,
        n_draws=1000,
        prior_scale=10.0,
        seed=1,
        bootstrap=50,
        threads=1,
        latency=None,
        accepted=false,
        outcome_functional=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_conditional(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        treatment: String,
        outcome: String,
        modifier: String,
        control_level: f64,
        active_level: f64,
        refute: Option<Bound<'_, PyAny>>,
        inference: Option<String>,
        n_draws: usize,
        prior_scale: f64,
        seed: u64,
        bootstrap: u32,
        threads: u32,
        latency: Option<String>,
        accepted: bool,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };

        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let w_id = data.schema().id_of(&modifier).map_err(py_err)?;
            let mut inner =
                AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level)
                    .with_effect_modifiers([w_id]);
            if let Some(functional) = functional {
                inner = inner.with_outcome_functional(functional);
            }
            let cq = ConditionalEffectQuery::try_new(inner)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let mut builder = if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(dag))
            } else {
                Study::tabular(data).graph(dag)
            }
            .query(CausalQuery::ConditionalEffect(cq))
            .refute(suite)
            .bootstrap_replicates(bootstrap);
            if let Some(mode) = latency_mode {
                builder = builder.latency_mode(mode);
            }
            builder = apply_inference(
                builder,
                inference.as_deref().unwrap_or("frequentist"),
                n_draws,
                prior_scale,
            )?;
            let analysis = builder.build().map_err(py_err)?;
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let prepared = analysis.prepare(&ctx).map_err(py_err)?;
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Compile once for ConditionalEffect on a supplied CPDAG.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        graph,
        treatment,
        outcome,
        modifier,
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
        accepted=false,
        outcome_functional=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_cpdag_conditional(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: graphs::Cpdag,
        treatment: String,
        outcome: String,
        modifier: String,
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
        accepted: bool,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        require_named_graph_order(&graph.names, &names, "Cpdag")?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        let latency_mode = parse_latency(latency)?;
        let cpdag = graph.cpdag;
        detach_catch(py, move || {
            prepare_class_conditional(
                data,
                names,
                ClassResponseGraph::Cpdag(cpdag),
                treatment,
                outcome,
                modifier,
                control_level,
                active_level,
                identifier,
                estimator,
                inference,
                n_draws,
                prior_scale,
                suite,
                seed,
                bootstrap,
                threads,
                latency_mode,
                accepted,
                functional,
            )
        })
    }

    /// Compile once for ConditionalEffect on a supplied PAG.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        graph,
        treatment,
        outcome,
        modifier,
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
        accepted=false,
        outcome_functional=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_pag_conditional(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: graphs::Pag,
        treatment: String,
        outcome: String,
        modifier: String,
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
        accepted: bool,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<Self> {
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        require_named_graph_order(&graph.names, &names, "Pag")?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let suite = suite_from_refute(refute.as_ref())?;
        let latency_mode = parse_latency(latency)?;
        let pag = graph.pag;
        detach_catch(py, move || {
            prepare_class_conditional(
                data,
                names,
                ClassResponseGraph::Pag(pag),
                treatment,
                outcome,
                modifier,
                control_level,
                active_level,
                identifier,
                estimator,
                inference,
                n_draws,
                prior_scale,
                suite,
                seed,
                bootstrap,
                threads,
                latency_mode,
                accepted,
                functional,
            )
        })
    }

    /// Compile once from tabular columns + DAG edges (static PathSpecificEffect).
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
        path_nodes=None,
        max_paths=64,
        max_len=16,
        refute=None,
        seed=1,
        bootstrap=50,
        threads=1,
        latency=None,
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_path_specific(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        treatment: String,
        outcome: String,
        control_level: f64,
        active_level: f64,
        path_nodes: Option<Vec<String>>,
        max_paths: usize,
        max_len: usize,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        bootstrap: u32,
        threads: u32,
        latency: Option<String>,
        accepted: bool,
    ) -> PyResult<Self> {
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };

        let suite = suite_from_refute(refute.as_ref())?;
        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let mut query = PathSpecificEffectQuery::binary(t_id, y_id)
                .with_max_paths(max_paths)
                .with_max_len(max_len);
            query.control = Intervention::set(t_id, Value::f64(control_level));
            query.active = Intervention::set(t_id, Value::f64(active_level));
            if let Some(nodes) = path_nodes {
                let mut ids = Vec::with_capacity(nodes.len());
                for name in &nodes {
                    ids.push(data.schema().id_of(name).map_err(py_err)?);
                }
                query = query.with_path_nodes(ids);
            }
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let mut builder = if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(dag))
            } else {
                Study::tabular(data).graph(dag)
            }
            .query(CausalQuery::PathSpecific(query))
            .identifier(IdentifierId::PathSpecificNatural)
            .estimator(EstimatorId::FunctionalEffect)
            .refute(suite)
            .bootstrap_replicates(bootstrap);
            if let Some(mode) = latency_mode {
                builder = builder.latency_mode(mode);
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
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Compile once from tabular columns + DAG edges (static InterventionalDistribution).
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        edges,
        outcome,
        interventions,
        *,
        conditioning=None,
        refute=None,
        seed=1,
        threads=1,
        latency=None,
        accepted=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_distribution(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        outcome: String,
        interventions: std::collections::HashMap<String, f64>,
        conditioning: Option<Vec<String>>,
        refute: Option<Bound<'_, PyAny>>,
        seed: u64,
        threads: u32,
        latency: Option<String>,
        accepted: bool,
    ) -> PyResult<Self> {
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let latency_mode = match latency.as_deref() {
            None => None,
            Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "unknown latency={s:?}; use interactive|standard|report"
                ))
            })?),
        };

        let suite = suite_from_refute(refute.as_ref())?;
        detach_catch(py, move || {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let mut ivs = Vec::with_capacity(interventions.len());
            for (name, level) in &interventions {
                let id = data.schema().id_of(name).map_err(py_err)?;
                ivs.push(Intervention::set(id, Value::f64(*level)));
            }
            let mut query = InterventionalDistributionQuery::new(y_id, ivs);
            if let Some(cond) = conditioning {
                let mut z = Vec::with_capacity(cond.len());
                for name in &cond {
                    z.push(data.schema().id_of(name).map_err(py_err)?);
                }
                query = query.with_conditioning(z);
            }
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let mut builder = if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(dag))
            } else {
                Study::tabular(data).graph(dag)
            }
            .query(CausalQuery::Distribution(query))
            .identifier(IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalDistribution)
            .refute(suite);
            if let Some(mode) = latency_mode {
                builder = builder.latency_mode(mode);
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
            Ok(Self { inner: Arc::new(prepared), names, last: None, series: false })
        })
    }

    /// Weighted-mean retarget from the frozen score table. Does not refit.
    #[pyo3(signature = (weights, depends_on, *, seed=1, threads=1))]
    fn retarget(
        &mut self,
        py: Python<'_>,
        weights: Vec<f64>,
        depends_on: Vec<String>,
        seed: u64,
        threads: u32,
    ) -> PyResult<AteAnalysisResult> {
        let inner = Arc::clone(&self.inner);
        let out_names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                threads,
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let schema_ids = depends_on
                .iter()
                .map(|name| {
                    out_names
                        .iter()
                        .position(|n| n == name)
                        .map(|i| antecedent_core::VariableId::from_raw(i as u32))
                        .ok_or_else(|| PyValueError::new_err(format!("unknown depends_on {name}")))
                })
                .collect::<PyResult<Vec<_>>>()?;
            let result = inner.retarget(&weights, &schema_ids, &ctx).map_err(py_err)?;
            let mapped = ate_result_from_analysis(&out_names, result.clone(), false)?;
            Ok((mapped, result))
        })?;
        self.last = Some(result);
        Ok(mapped)
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
        require_prepared_names(&self.names, &names, "estimate")?;
        let data = tabular_from_numpy(&names, &columns)?;
        drop(columns);
        self.finish_ate_estimate(py, data, seed, threads)
    }

    #[pyo3(signature = (names, columns, *, seed=1, threads=1))]
    fn estimate_arrow_c(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: u32,
    ) -> PyResult<AteAnalysisResult> {
        require_prepared_names(&self.names, &names, "estimate")?;
        let (data, _) = tabular_from_arrow_c_objs(py, names, columns)?;
        self.finish_ate_estimate(py, data, seed, threads)
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
        require_prepared_names(&self.names, &names, "estimate")?;
        let data = tabular_from_numpy(&names, &columns)?;
        drop(columns);
        self.finish_response_estimate(py, data, seed, threads)
    }

    #[pyo3(signature = (names, columns, *, seed=1, threads=1))]
    fn estimate_response_arrow_c(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: u32,
    ) -> PyResult<ResponseAnalysisResult> {
        require_prepared_names(&self.names, &names, "estimate")?;
        let (data, _) = tabular_from_arrow_c_objs(py, names, columns)?;
        self.finish_response_estimate(py, data, seed, threads)
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
        require_prepared_names(&self.names, &names, "refresh")?;
        let data = tabular_from_numpy(&names, &columns)?;
        drop(columns);
        self.finish_ate_refresh(py, data, seed, threads)
    }

    #[pyo3(signature = (names, columns, *, seed=1, threads=1))]
    fn refresh_arrow_c(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: u32,
    ) -> PyResult<AteAnalysisResult> {
        require_prepared_names(&self.names, &names, "refresh")?;
        let (data, _) = tabular_from_arrow_c_objs(py, names, columns)?;
        self.finish_ate_refresh(py, data, seed, threads)
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
        require_prepared_names(&self.names, &names, "refresh")?;
        let data = tabular_from_numpy(&names, &columns)?;
        drop(columns);
        self.finish_response_refresh(py, data, seed, threads)
    }

    #[pyo3(signature = (names, columns, *, seed=1, threads=1))]
    fn refresh_response_arrow_c(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: u32,
    ) -> PyResult<ResponseAnalysisResult> {
        require_prepared_names(&self.names, &names, "refresh")?;
        let (data, _) = tabular_from_arrow_c_objs(py, names, columns)?;
        self.finish_response_refresh(py, data, seed, threads)
    }

    /// Export the retained full posterior or response without refitting.
    #[pyo3(signature = (*, artifact_id="prepared-result", payload="result"))]
    fn export_artifact<'py>(
        &self,
        py: Python<'py>,
        artifact_id: &str,
        payload: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let result = self
            .last
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("estimate before exporting an artifact"))?;
        let bytes = if payload == "query" {
            let query = antecedent_io::CausalPayloadWire::Query(Box::new(
                antecedent_io::causal_query_to_wire(self.inner.query()).map_err(py_err)?,
            ));
            let artifact = antecedent_io::encode_causal_payload_artifact(
                &query,
                self.names.clone(),
                artifact_id,
            )
            .map_err(py_err)?;
            let mut bytes = Vec::new();
            artifact.write_to(&mut bytes).map_err(py_err)?;
            bytes
        } else if payload != "result" {
            return Err(PyValueError::new_err("payload must be 'query' or 'result'"));
        } else if let Some(post) = &result.posterior {
            antecedent_io::encode_causal_posterior_bytes(post, artifact_id).map_err(py_err)?
        } else if let Some(response) = &result.response {
            let payload = antecedent_io::CausalPayloadWire::ResponseResult(Box::new(
                antecedent_io::causal_response_to_wire(response).map_err(py_err)?,
            ));
            let artifact = antecedent_io::encode_causal_payload_artifact(
                &payload,
                self.names.clone(),
                artifact_id,
            )
            .map_err(py_err)?;
            let mut bytes = Vec::new();
            artifact.write_to(&mut bytes).map_err(py_err)?;
            bytes
        } else if result.mediation.is_some() || result.counterfactual.is_some() {
            let (control_level, active_level) = match self.inner.query() {
                CausalQuery::Mediation(q) => (hard_value(&q.control), hard_value(&q.active)),
                CausalQuery::Counterfactual(q) => {
                    (hard_value(&q.control), q.interventions.first().and_then(hard_value))
                }
                _ => unreachable!(),
            };
            let wire = antecedent_io::StaticResultWire {
                identification: antecedent_io::identification_to_wire(&result.identification)
                    .map_err(py_err)?,
                estimate: result.estimate.ate,
                standard_error: result.estimate.se_bootstrap.or_else(|| {
                    result.estimate.se_analytic.is_finite().then_some(result.estimate.se_analytic)
                }),
                assumptions: antecedent_io::assumptions_to_wire(&result.estimate.assumptions),
                support: result
                    .diagnostics
                    .iter()
                    .filter(|d| d.code.contains("support") || d.code.contains("overlap"))
                    .map(antecedent_io::diagnostic_to_wire)
                    .collect(),
                diagnostics: result
                    .diagnostics
                    .iter()
                    .map(antecedent_io::diagnostic_to_wire)
                    .collect(),
                refutations: result
                    .refutations
                    .iter()
                    .map(antecedent_io::refutation_to_wire)
                    .collect(),
                unit_effects: result.counterfactual.as_ref().map(|c| c.unit_effects.to_vec()),
                mediation: result
                    .mediation
                    .as_ref()
                    .map(|m| [m.total.unwrap(), m.direct.unwrap(), m.mediated.unwrap()]),
                control_level: control_level
                    .ok_or_else(|| PyValueError::new_err("missing control"))?,
                active_level: active_level
                    .ok_or_else(|| PyValueError::new_err("missing active"))?,
            };
            let payload = antecedent_io::CausalPayloadWire::StaticResult(Box::new(wire));
            let artifact = antecedent_io::encode_causal_payload_artifact(
                &payload,
                self.names.clone(),
                artifact_id,
            )
            .map_err(py_err)?;
            let mut bytes = Vec::new();
            artifact.write_to(&mut bytes).map_err(py_err)?;
            bytes
        } else {
            return Err(PyValueError::new_err(
                "retained result has no posterior or response artifact payload",
            ));
        };
        Ok(pyo3::types::PyBytes::new(py, &bytes))
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
        require_prepared_names(&self.names, &names, "refute")?;
        let data = tabular_from_numpy(&names, &columns)?;
        drop(columns);
        self.finish_ate_refute(py, data, suite, seed, threads, cancel)
    }

    #[pyo3(signature = (names, columns, suite, *, seed=1, threads=1, cancel=None))]
    fn refute_arrow_c(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        suite: Bound<'_, PyAny>,
        seed: u64,
        threads: u32,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<AteAnalysisResult> {
        require_prepared_names(&self.names, &names, "refute")?;
        let (data, _) = tabular_from_arrow_c_objs(py, names, columns)?;
        self.finish_ate_refute(py, data, suite, seed, threads, cancel)
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
        out.insert("structure_source".into(), self.inner.structure_source().as_str().to_string());
        if let Some(status) = self.inner.support_status() {
            out.insert("evidence_status".into(), status.as_str().to_string());
            if let Some(reason) = status.allowlist_reason() {
                out.insert("allowlist_reason".into(), reason.to_string());
            }
            if let Some(parent) = status.allowlist_parent() {
                out.insert("allowlist_parent".into(), parent.to_string());
            }
        }
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

pub(crate) fn response_from_study(
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
    Ok(attach_study_response_meta(
        response_result(
            response,
            treatments,
            outcomes,
            adjustment_set,
            names,
            result.support_status,
        )?,
        crate::identification_details::analysis_to_json(result, names)?,
        format!("{:?}", result.identification.status),
        result.logical_plan.identifier.as_deref().map(str::to_owned),
        result.diagnostics.iter().map(|d| format!("{}: {}", d.code, d.message)).collect(),
    ))
}

fn parse_latency(latency: Option<String>) -> PyResult<Option<antecedent::LatencyMode>> {
    match latency.as_deref() {
        None => Ok(None),
        Some(s) => Ok(Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
            PyValueError::new_err(format!("unknown latency={s:?}; use interactive|standard|report"))
        })?)),
    }
}

enum ClassResponseGraph {
    Pag(antecedent_graph::Pag),
    Cpdag(antecedent_graph::Cpdag),
}

#[allow(clippy::too_many_arguments)]
fn prepare_class_response(
    data: TabularData,
    names: Vec<String>,
    graph: ClassResponseGraph,
    kind: String,
    treatments: Vec<String>,
    outcomes: Vec<String>,
    grid: Option<Vec<f64>>,
    intervention_kinds: Option<Vec<String>>,
    intervention_parameters: Option<Vec<Vec<f64>>>,
    identifier: Option<String>,
    estimator: Option<String>,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    seed: u64,
    threads: u32,
    latency_mode: Option<antecedent::LatencyMode>,
    accepted: bool,
) -> PyResult<PyPreparedAnalysis> {
    let treatment_ids = crate::response_api::resolve_names(data.schema(), &treatments)?;
    let outcome_ids = crate::response_api::resolve_names(data.schema(), &outcomes)?;
    let functional = build_functional(
        &kind,
        &treatment_ids,
        &outcome_ids,
        grid,
        None,
        None,
        intervention_kinds,
        intervention_parameters,
        1,
        antecedent_core::DerivativeScale::Identity,
        antecedent_core::DerivativeWeighting::Observed,
    )?;
    let query = CausalQuery::Response(ResponseQuery::new(functional));
    let mut builder = match graph {
        ClassResponseGraph::Pag(pag) => {
            if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(pag))
            } else {
                Study::tabular(data).graph(pag)
            }
        }
        ClassResponseGraph::Cpdag(cpdag) => {
            if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(cpdag))
            } else {
                Study::tabular(data).graph(cpdag)
            }
        }
    }
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
    builder = apply_inference(
        builder,
        inference.as_deref().unwrap_or("frequentist"),
        n_draws,
        prior_scale,
    )?;
    let analysis = builder.build().map_err(py_err)?;
    let ctx = py_execution_context_ext(
        seed,
        threads,
        None,
        None,
        Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
    );
    let prepared = analysis.prepare(&ctx).map_err(py_err)?;
    Ok(PyPreparedAnalysis { inner: Arc::new(prepared), names, last: None, series: false })
}

#[allow(clippy::too_many_arguments)]
fn prepare_class_conditional(
    data: TabularData,
    names: Vec<String>,
    graph: ClassResponseGraph,
    treatment: String,
    outcome: String,
    modifier: String,
    control_level: f64,
    active_level: f64,
    identifier: Option<String>,
    estimator: Option<String>,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    suite: antecedent::RefuteSuite,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    latency_mode: Option<antecedent::LatencyMode>,
    accepted: bool,
    outcome_functional: Option<antecedent_core::OutcomeFunctional>,
) -> PyResult<PyPreparedAnalysis> {
    let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
    let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
    let w_id = data.schema().id_of(&modifier).map_err(py_err)?;
    let mut inner = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level)
        .with_effect_modifiers([w_id]);
    if let Some(functional) = outcome_functional {
        inner = inner.with_outcome_functional(functional);
    }
    let cq =
        ConditionalEffectQuery::try_new(inner).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut builder = match graph {
        ClassResponseGraph::Pag(pag) => {
            if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(pag))
            } else {
                Study::tabular(data).graph(pag)
            }
        }
        ClassResponseGraph::Cpdag(cpdag) => {
            if accepted {
                Study::tabular(data).graph(antecedent::AcceptedGraph::from(cpdag))
            } else {
                Study::tabular(data).graph(cpdag)
            }
        }
    }
    .query(CausalQuery::ConditionalEffect(cq))
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
    builder = apply_inference(
        builder,
        inference.as_deref().unwrap_or("frequentist"),
        n_draws,
        prior_scale,
    )?;
    let analysis = builder.build().map_err(py_err)?;
    let ctx = py_execution_context_ext(
        seed,
        threads,
        None,
        None,
        Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
    );
    let prepared = analysis.prepare(&ctx).map_err(py_err)?;
    Ok(PyPreparedAnalysis { inner: Arc::new(prepared), names, last: None, series: false })
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

fn static_kind_query(
    schema: &antecedent_core::CausalSchema,
    kind: &str,
    treatment: &str,
    outcome: &str,
    mediators: &[String],
    contrast: &str,
    control: f64,
    active: f64,
) -> PyResult<CausalQuery> {
    let t = schema.id_of(treatment).map_err(py_err)?;
    let y = schema.id_of(outcome).map_err(py_err)?;
    match kind {
        "mediation" => {
            let contrast = match contrast {
                "total" => MediationContrast::Total,
                "direct" => MediationContrast::Direct,
                "mediated" => MediationContrast::Mediated,
                "natural_direct" => MediationContrast::NaturalDirect,
                "natural_indirect" => MediationContrast::NaturalIndirect,
                _ => return Err(PyValueError::new_err("unknown static mediation contrast")),
            };
            let ms = crate::response_api::resolve_names(schema, mediators)?;
            let mut q = MediationQuery::binary(t, y, Arc::from(ms), contrast);
            q.control = Intervention::set(t, Value::f64(control));
            q.active = Intervention::set(t, Value::f64(active));
            Ok(CausalQuery::Mediation(q))
        }
        "counterfactual" => Ok(CausalQuery::Counterfactual(
            antecedent_core::CounterfactualQuery::new(
                y,
                Arc::from([Intervention::set(t, Value::f64(active))]),
            )
            .with_control(Intervention::set(t, Value::f64(control))),
        )),
        _ => Err(PyValueError::new_err("unsupported static staged kind")),
    }
}

#[allow(clippy::too_many_arguments)]
fn prepare_temporal_class_effect(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    graph: TemporalClassGraph,
    treatment: String,
    outcome: String,
    policy: &str,
    window: Option<(i32, i32)>,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    accepted: bool,
) -> PyResult<PyPreparedAnalysis> {
    let (tabular, _) = tabular_from_py_columns(py, names.clone(), columns)?;
    let policy = policy.to_ascii_lowercase();
    let suite = suite_from_refute(refute.as_ref())?;
    detach_catch(py, move || {
        let series = series_from_tabular(tabular)?;
        let t_id = series.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = series.schema().id_of(&outcome).map_err(py_err)?;
        let mut q = crate::temporal_api::temporal_query_from_policy(
            &policy,
            t_id,
            y_id,
            treatment_lag,
            horizon_steps,
            active_level,
        )?;
        if let Some((from, until)) = window {
            if policy != "sustained" {
                return Err(PyValueError::new_err("window requires policy='sustained'"));
            }
            q = q.with_policy(antecedent_core::TemporalPolicy::sustained(from, until));
        }
        let mut builder = bind_temporal_class(Study::series(series), graph, accepted)
            .temporal_query(q)
            .refute(suite)
            .bootstrap_replicates(bootstrap);
        builder = crate::temporal_api::apply_temporal_inference(
            builder,
            inference.as_deref(),
            n_draws,
            prior_scale,
            None,
        )?;
        let analysis = builder.build().map_err(py_err)?;
        let ctx = py_execution_context_ext(
            seed,
            threads,
            None,
            None,
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        );
        let prepared = analysis.prepare(&ctx).map_err(py_err)?;
        Ok(PyPreparedAnalysis { inner: Arc::new(prepared), names, last: None, series: true })
    })
}

fn hard_value(intervention: &Intervention) -> Option<f64> {
    match intervention {
        Intervention::Set { value, .. } => value.as_f64(),
        _ => None,
    }
}
