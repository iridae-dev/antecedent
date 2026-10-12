//! Graph-posterior and DBN prepare routes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalQuery, ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery,
    TemporalResponseSpec,
};
use antecedent_data::TableView;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::{
    PrepareRequest, PyPreparedAnalysis, SeriesGraphQuery, conditional_query, configured_spec,
    dbn_posterior, exact_dag_posterior, finish_series_graph_posterior,
    finish_static_graph_posterior, one_series, parse_response_options, response_query,
    static_ate_query, take_supplied_posterior, temporal_effect_query, temporal_mediation_query,
};
use crate::py_err;
use crate::response_api::build_functional;

#[pymethods]
impl PyPreparedAnalysis {
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
        estimator_config=None,
        posterior=None,
        seed=1,
        threads=None,
        options=None,
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
        estimator_config: Option<&Bound<'_, PyDict>>,
        posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let spec = configured_spec(estimator_config, None, &req.opts)?;
        let supplied = take_supplied_posterior(posterior, &req.names)?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let query = CausalQuery::AverageEffect(static_ate_query(
                &data,
                &treatment,
                &outcome,
                control_level,
                active_level,
                &req.opts,
            )?);
            let ctx = req.opts.ctx(seed, threads);
            let gp = exact_dag_posterior(supplied, &data, &ctx)?;
            finish_static_graph_posterior(data, req, gp, query, spec, None, &ctx)
        })
    }

    /// Compile once for licensed ConditionalEffect × graph_posterior.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        treatment,
        outcome,
        modifier,
        *,
        control_level=0.0,
        active_level=1.0,
        outcome_functional=None,
        posterior=None,
        seed=1,
        threads=None,
        options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_graph_posterior_conditional(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        treatment: String,
        outcome: String,
        modifier: String,
        control_level: f64,
        active_level: f64,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
        posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let supplied = take_supplied_posterior(posterior, &req.names)?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let query = conditional_query(
                &data,
                &treatment,
                &outcome,
                &modifier,
                control_level,
                active_level,
                functional,
                &req.opts,
            )?;
            let ctx = req.opts.ctx(seed, threads);
            let gp = exact_dag_posterior(supplied, &data, &ctx)?;
            finish_static_graph_posterior(data, req, gp, query, None, None, &ctx)
        })
    }

    /// Compile once for licensed static ResponseCurve × graph_posterior.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        treatment,
        outcome,
        grid,
        *,
        response_options=None,
        posterior=None,
        seed=1,
        threads=None,
        options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_graph_posterior_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        treatment: String,
        outcome: String,
        grid: Vec<f64>,
        response_options: Option<Bound<'_, PyDict>>,
        posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let response_options = parse_response_options(response_options.as_ref())?;
        let supplied = take_supplied_posterior(posterior, &req.names)?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let query = response_query(
                ResponseFunctional::MeanCurve {
                    outcome: y_id,
                    treatment: ContinuousDomain::new(t_id, GridSpec::Values(grid.into())),
                },
                &req.opts,
            );
            let ctx = req.opts.ctx(seed, threads);
            let gp = exact_dag_posterior(supplied, &data, &ctx)?;
            finish_static_graph_posterior(data, req, gp, query, None, response_options, &ctx)
        })
    }

    /// Compile once for licensed one-coordinate InterventionResponse × graph_posterior.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        outcome,
        treatments,
        intervention_kinds,
        intervention_parameters,
        *,
        posterior=None,
        seed=1,
        threads=None,
        options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_graph_posterior_intervention_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        treatments: Vec<String>,
        intervention_kinds: Vec<String>,
        intervention_parameters: Vec<Vec<f64>>,
        posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let supplied = take_supplied_posterior(posterior, &req.names)?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
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
            let query = response_query(functional, &req.opts);
            let ctx = req.opts.ctx(seed, threads);
            let gp = exact_dag_posterior(supplied, &data, &ctx)?;
            finish_static_graph_posterior(data, req, gp, query, None, None, &ctx)
        })
    }

    /// Compile once for licensed Pulse/Sustained × DBN graph_posterior.
    ///
    /// Bayesian mixes per-atom posteriors; Frequentist mixes atom estimates with
    /// a shared circular-block SE from the replicate budget (`0` withholds it).
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        treatment,
        outcome,
        *,
        policy="pulse",
        window=None,
        treatment_lag=1,
        horizon_steps=1,
        control_level=0.0,
        active_level=1.0,
        max_history_lag=None,
        max_lag=1,
        force_mcmc=false,
        n_chains=2,
        n_warmup=200,
        mcmc_draws=400,
        posterior=None,
        frame=None,
        seed=1,
        threads=None,
        options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_dbn_posterior_temporal(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        treatment: String,
        outcome: String,
        policy: &str,
        window: Option<(i32, i32)>,
        treatment_lag: u32,
        horizon_steps: u32,
        control_level: f64,
        active_level: f64,
        max_history_lag: Option<u32>,
        max_lag: u32,
        force_mcmc: bool,
        n_chains: u32,
        n_warmup: u32,
        mcmc_draws: u32,
        posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
        frame: Option<Bound<'_, PyDict>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let supplied = take_supplied_posterior(posterior, &req.names)?;
        let input = req.load_frame(py, columns, frame.as_ref())?;
        let policy = policy.to_ascii_lowercase();
        req.off_gil(py, move |req| {
            let series = one_series(input)?;
            let t_id = series.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = series.schema().id_of(&outcome).map_err(py_err)?;
            let q = temporal_effect_query(
                &policy,
                window,
                t_id,
                y_id,
                treatment_lag,
                horizon_steps,
                control_level,
                active_level,
                max_history_lag,
                &req.opts,
            )?;
            let ctx = req.opts.ctx(seed, threads);
            let gp = dbn_posterior(
                supplied, &series, max_lag, force_mcmc, n_chains, n_warmup, mcmc_draws, &ctx,
            )?;
            finish_series_graph_posterior(series, req, gp, SeriesGraphQuery::Temporal(q), &ctx)
        })
    }

    /// Compile once for licensed TemporalMediationEffect × DBN graph_posterior.
    ///
    /// Bayesian mixes per-atom posteriors; single-horizon Frequentist mixes atom
    /// estimates with a shared circular-block SE from the replicate budget
    /// (`0` withholds the SE).
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        treatment,
        mediator,
        outcome,
        *,
        contrast="mediated",
        control_level=0.0,
        active_level=1.0,
        horizons=None,
        max_lag=1,
        force_mcmc=false,
        n_chains=2,
        n_warmup=200,
        mcmc_draws=400,
        posterior=None,
        frame=None,
        seed=1,
        threads=None,
        options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_dbn_posterior_mediation(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        treatment: String,
        mediator: String,
        outcome: String,
        contrast: &str,
        control_level: f64,
        active_level: f64,
        horizons: Option<Vec<u32>>,
        max_lag: u32,
        force_mcmc: bool,
        n_chains: u32,
        n_warmup: u32,
        mcmc_draws: u32,
        posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
        frame: Option<Bound<'_, PyDict>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let supplied = take_supplied_posterior(posterior, &req.names)?;
        let input = req.load_frame(py, columns, frame.as_ref())?;
        let contrast = contrast.to_string();
        req.off_gil(py, move |req| {
            let series = one_series(input)?;
            let q = temporal_mediation_query(
                series.schema(),
                &treatment,
                &mediator,
                &outcome,
                &contrast,
                control_level,
                active_level,
                horizons,
                &req.opts,
            )?;
            let ctx = req.opts.ctx(seed, threads);
            let gp = dbn_posterior(
                supplied, &series, max_lag, force_mcmc, n_chains, n_warmup, mcmc_draws, &ctx,
            )?;
            finish_series_graph_posterior(series, req, gp, SeriesGraphQuery::Mediation(q), &ctx)
        })
    }

    /// Compile once for licensed temporal ResponseCurve / InterventionResponse
    /// × DBN or temporal-class graph_posterior.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
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
        max_lag=1,
        force_mcmc=false,
        n_chains=2,
        n_warmup=200,
        mcmc_draws=400,
        posterior=None,
        frame=None,
        seed=1,
        threads=None,
        options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_dbn_posterior_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
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
        max_lag: u32,
        force_mcmc: bool,
        n_chains: u32,
        n_warmup: u32,
        mcmc_draws: u32,
        posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
        frame: Option<Bound<'_, PyDict>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let supplied = take_supplied_posterior(posterior, &req.names)?;
        let input = req.load_frame(py, columns, frame.as_ref())?;
        let policy = policy.to_ascii_lowercase();
        req.off_gil(py, move |req| {
            let series = one_series(input)?;
            let schema = series.schema();
            let treatment_ids: Vec<_> = treatments
                .iter()
                .map(|n| schema.id_of(n).map_err(py_err))
                .collect::<PyResult<_>>()?;
            let outcome_ids: Vec<_> = outcomes
                .iter()
                .map(|n| schema.id_of(n).map_err(py_err))
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
            let origin = -i32::try_from(treatment_lag)
                .map_err(|_| crate::value_err("treatment_lag does not fit in i32"))?;
            let functional = crate::response_api::wrap_temporal_sequence_steps(functional, origin)?;
            let temporal = TemporalResponseSpec::new(horizons, temporal_policy, max_history_lag)
                .map_err(|e| crate::value_err(e.to_string()))?;
            let mut response = ResponseQuery::new(functional).with_temporal(temporal);
            if let Some(population) = req.opts.target_population.clone() {
                response.target_population = population;
            }
            let ctx = req.opts.ctx(seed, threads);
            let gp = dbn_posterior(
                supplied, &series, max_lag, force_mcmc, n_chains, n_warmup, mcmc_draws, &ctx,
            )?;
            finish_series_graph_posterior(
                series,
                req,
                gp,
                SeriesGraphQuery::Response(response),
                &ctx,
            )
        })
    }
}
