//! Graph-posterior and DBN prepare routes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AnomalyAttributionQuery, CausalQuery, ChangeAttributionQuery, ContinuousDomain, GridSpec,
    Intervention, InterventionalDistributionQuery, PathSpecificEffectQuery, PopulationSelector,
    ResponseFunctional, ResponseQuery, TemporalResponseSpec, Value,
};
use antecedent_data::TableView;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::{
    PrepareRequest, PyPreparedAnalysis, SeriesGraphQuery, conditional_query, configured_spec,
    dbn_posterior, exact_dag_posterior, finish_series_graph_posterior,
    finish_static_graph_posterior, one_series, parse_response_options, response_query,
    static_ate_query, static_kind_query, take_supplied_posterior, temporal_effect_query, temporal_mediation_query,
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

    /// Compile once for a mediation, path-restricted effect, natural direct effect,
    /// interventional distribution or anomaly / change attribution over a graph posterior.
    ///
    /// `kind` is `mediation`, `nested_counterfactual`, `path_specific`, `distribution`,
    /// `anomaly_attribution` or `change_attribution`; the other arguments are read by the kind
    /// that names them. Every posterior atom is answered by the explicit executor of its own
    /// structure and the atoms are mixed by posterior probability.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        kind,
        *,
        treatment=None,
        outcome=None,
        mediators=Vec::new(),
        contrast="mediated",
        control_level=0.0,
        active_level=1.0,
        path_nodes=None,
        max_paths=64,
        max_len=16,
        interventions=None,
        conditioning=None,
        targets=Vec::new(),
        max_units=0,
        reference=None,
        baseline_start=0,
        baseline_end=0,
        comparison_start=0,
        comparison_end=0,
        posterior=None,
        seed=1,
        threads=None,
        options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_graph_posterior_kind(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        kind: String,
        treatment: Option<String>,
        outcome: Option<String>,
        mediators: Vec<String>,
        contrast: &str,
        control_level: f64,
        active_level: f64,
        path_nodes: Option<Vec<String>>,
        max_paths: usize,
        max_len: usize,
        interventions: Option<std::collections::HashMap<String, f64>>,
        conditioning: Option<Vec<String>>,
        targets: Vec<String>,
        max_units: usize,
        reference: Option<(f64, f64)>,
        baseline_start: usize,
        baseline_end: usize,
        comparison_start: usize,
        comparison_end: usize,
        posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let supplied = take_supplied_posterior(posterior, &req.names)?;
        let data = req.load_columns(py, columns)?;
        let contrast = contrast.to_owned();
        let need = |value: &Option<String>, what: &str| -> PyResult<String> {
            value.clone().ok_or_else(|| crate::value_err(format!("{kind} requires a {what}")))
        };
        let query = match kind.as_str() {
            "mediation" | "nested_counterfactual" => {
                let mut query = static_kind_query(
                    data.schema(),
                    &kind,
                    &need(&treatment, "treatment")?,
                    &need(&outcome, "outcome")?,
                    &mediators,
                    &contrast,
                    control_level,
                    active_level,
                )?;
                if let (CausalQuery::Mediation(q), Some(population)) =
                    (&mut query, req.opts.target_population.clone())
                {
                    q.target_population = population;
                }
                query
            }
            "path_specific" => {
                let t_id = data.schema().id_of(&need(&treatment, "treatment")?).map_err(py_err)?;
                let y_id = data.schema().id_of(&need(&outcome, "outcome")?).map_err(py_err)?;
                let mut query = PathSpecificEffectQuery::binary(t_id, y_id)
                    .with_max_paths(max_paths)
                    .with_max_len(max_len);
                query.control = Intervention::set(t_id, Value::f64(control_level));
                query.active = Intervention::set(t_id, Value::f64(active_level));
                if let Some(population) = req.opts.target_population.clone() {
                    query.target_population = population;
                }
                if let Some(nodes) = &path_nodes {
                    let mut ids = Vec::with_capacity(nodes.len());
                    for name in nodes {
                        ids.push(data.schema().id_of(name).map_err(py_err)?);
                    }
                    query = query.with_path_nodes(ids);
                }
                CausalQuery::PathSpecific(query)
            }
            "distribution" => {
                let y_id = data.schema().id_of(&need(&outcome, "outcome")?).map_err(py_err)?;
                let mut ivs = Vec::new();
                for (name, level) in interventions.as_ref().into_iter().flatten() {
                    let id = data.schema().id_of(name).map_err(py_err)?;
                    ivs.push(Intervention::set(id, Value::f64(*level)));
                }
                let mut query = InterventionalDistributionQuery::new(y_id, ivs);
                if let Some(population) = req.opts.target_population.clone() {
                    query.target_population = population;
                }
                if let Some(cond) = &conditioning {
                    let mut z = Vec::with_capacity(cond.len());
                    for name in cond {
                        z.push(data.schema().id_of(name).map_err(py_err)?);
                    }
                    query = query.with_conditioning(z);
                }
                CausalQuery::Distribution(query)
            }
            "anomaly_attribution" => {
                let ids = targets
                    .iter()
                    .map(|name| crate::graph_build::schema_var_id(data.schema(), name))
                    .collect::<PyResult<Vec<_>>>()?;
                let mut query = AnomalyAttributionQuery::new(ids, max_units);
                if let Some((center, scale)) = reference {
                    query = query
                        .with_reference(antecedent_core::AnomalyReference::fixed(center, scale));
                }
                CausalQuery::AnomalyAttribution(query)
            }
            "change_attribution" => {
                let outcome_id = crate::graph_build::schema_var_id(
                    data.schema(),
                    &need(&outcome, "outcome")?,
                )?;
                CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
                    outcome_id,
                    PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
                    PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
                ))
            }
            other => {
                return Err(crate::value_err(format!(
                    "unsupported graph-posterior query kind {other:?}"
                )));
            }
        };
        req.off_gil(py, move |req| {
            let ctx = req.opts.ctx(seed, threads);
            let gp = exact_dag_posterior(supplied, &data, &ctx)?;
            finish_static_graph_posterior(data, req, gp, query, None, None, &ctx)
        })
    }

    /// Compile once for a trial-to-target TransportQuery over a graph posterior: every atom is a
    /// selection diagram with the declared selection targets, decided by sID on its own graph.
    #[staticmethod]
    #[pyo3(signature = (names, columns, selections, source_population, target_population,
        source_experiments, kind, treatments, outcomes, trial, selection_probability,
        treatment_probability, *, grid=None, at=None, direction=None, order=1, scale="identity",
        weighting="observed", posterior=None, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_graph_posterior_transport(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        selections: Vec<String>,
        source_population: String,
        target_population: String,
        source_experiments: Vec<String>,
        kind: String,
        treatments: Vec<String>,
        outcomes: Vec<String>,
        trial: String,
        selection_probability: String,
        treatment_probability: String,
        grid: Option<Vec<f64>>,
        at: Option<Vec<f64>>,
        direction: Option<Vec<f64>>,
        order: u8,
        scale: &str,
        weighting: &str,
        posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_prior("a transport query")?;
        let supplied = take_supplied_posterior(posterior, &req.names)?;
        let data = req.load_columns(py, columns)?;
        let (scale, weighting) = (scale.to_owned(), weighting.to_owned());
        req.off_gil(py, move |mut req| {
            use crate::transport_interference_api::{ResponseArgs, schema_ids, transport_query};
            let schema = data.schema().clone();
            let query = transport_query(
                ResponseArgs {
                    kind,
                    treatments,
                    outcomes,
                    grid,
                    at,
                    direction,
                    order,
                    scale,
                    weighting,
                },
                source_population,
                target_population,
                &source_experiments,
                |names| schema_ids(&schema, names),
            )?;
            let column = |name: &str| crate::graph_build::schema_var_id(&schema, name);
            let trial_spec = antecedent::TransportTrialSpec {
                trial: column(&trial)?,
                selection_probability: column(&selection_probability)?,
                treatment_probability: column(&treatment_probability)?,
            };
            let ctx = req.opts.ctx(seed, threads);
            let gp = exact_dag_posterior(supplied, &data, &ctx)?;
            req.refuse_prior("a graph-posterior mixture")?;
            let builder = antecedent::Study::tabular(data)
                .graph_posterior(gp)
                .query(CausalQuery::Transport(query))
                .selection_targets(schema_ids(&schema, &selections)?)
                .transport_trial(trial_spec);
            let builder = req.opts.apply_budget_for(builder, false);
            let builder = req.opts.apply_inference(builder)?;
            req.compile_with(builder, false, &ctx)
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
