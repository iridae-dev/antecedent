//! Response and temporal prepare routes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent_core::{
    CausalQuery, ContinuousDomain, FeedbackDeclaration, GridSpec, InitialLaw, ResponseFunctional,
    ResponseQuery, TemporalResponseSpec,
};
use antecedent_data::TableView;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::{
    PrepareRequest, PyPreparedAnalysis, bind_temporal_class, extract_temporal_class,
    parse_estimator, parse_identifier, parse_response_options, response_query,
    temporal_effect_query, temporal_mediation_query, with_graph, with_response_options,
};
use crate::response_api::build_functional;
use crate::{dag_from_named_edges, py_err, temporal_dag_from_schema_edges};

/// Declared feedback mechanism in the Python facade's argument order: delay
/// ticks, level, autoregression, input gain, innovation variance, initial law
/// name (`None` is the contemporaneous observed law), and initial state.
type FeedbackArgs = (u32, f64, f64, f64, f64, Option<String>, f64);

fn feedback_declaration(args: FeedbackArgs) -> PyResult<FeedbackDeclaration> {
    let (delay_ticks, level, autoregression, input_gain, innovation_variance, law, initial_state) =
        args;
    let initial_law = match law {
        None => InitialLaw::contemporaneous_observed(),
        Some(name) => InitialLaw::named(name).map_err(|e| crate::value_err(e.to_string()))?,
    };
    Ok(FeedbackDeclaration {
        delay_ticks,
        level,
        autoregression,
        input_gain,
        innovation_variance,
        initial_law,
        initial_state,
    })
}

#[pymethods]
impl PyPreparedAnalysis {
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
        accepted=false,
        response_options=None,
        seed=1,
        threads=None,
        options=None,
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
        accepted: bool,
        response_options: Option<Bound<'_, PyDict>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let response_options = parse_response_options(response_options.as_ref())?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let query = response_query(
                ResponseFunctional::MeanCurve {
                    outcome: y_id,
                    treatment: ContinuousDomain::new(t_id, GridSpec::Values(grid.into())),
                },
                &req.opts,
            );
            let mut builder = req.opts.apply(with_response_options(
                with_graph(Study::tabular(data), dag, accepted, req.opts.discovery_algorithm())
                    .query(query),
                response_options,
            ));
            if let Some(id) = parse_identifier(identifier)? {
                builder = builder.identifier(id);
            }
            if let Some(est) = parse_estimator(estimator)? {
                builder = builder.estimator(est);
            }
            req.finish(builder, false)
        })
    }

    /// Compile once for a temporal ResponseCurve / InterventionResponse.
    ///
    /// Series columns, or a panel / multi-environment / event `frame` of the
    /// same schema. Frequentist omitted `bootstrap` is the builder's replicate
    /// count behind the joint circular-block bands; `0` publishes no band and
    /// warns `estimate.temporal_response.band_withheld`.
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
        feedback=None,
        accepted=false,
        class_graph=None,
        class_prior_ordered=None,
        class_prior_pairs=None,
        max_completions=None,
        observation_kind=None,
        latent=None,
        observed=None,
        censoring=None,
        event=None,
        lower=None,
        upper=None,
        indicator=None,
        assumption_kind=None,
        assumption_variables=Vec::new(),
        structural_model=None,
        frame=None,
        seed=1,
        threads=None,
        options=None,
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
        feedback: Option<FeedbackArgs>,
        accepted: bool,
        class_graph: Option<Bound<'_, PyAny>>,
        class_prior_ordered: Option<Vec<f64>>,
        class_prior_pairs: Option<Vec<(u64, f64)>>,
        max_completions: Option<usize>,
        observation_kind: Option<String>,
        latent: Option<String>,
        observed: Option<String>,
        censoring: Option<String>,
        event: Option<String>,
        lower: Option<String>,
        upper: Option<String>,
        indicator: Option<String>,
        assumption_kind: Option<String>,
        assumption_variables: Vec<String>,
        structural_model: Option<String>,
        frame: Option<Bound<'_, PyDict>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let class_graph = extract_temporal_class(class_graph, &req.names)?;
        let input = req.load_frame(py, columns, frame.as_ref())?;
        let policy = policy.to_ascii_lowercase();
        req.off_gil(py, move |mut req| {
            let data = input.materialize()?;
            let schema = data.schema().clone();
            let dag = temporal_dag_from_schema_edges(&schema, &edges)?;
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
            let mut temporal =
                TemporalResponseSpec::new(horizons, temporal_policy, max_history_lag)
                    .map_err(|e| crate::value_err(e.to_string()))?;
            if let Some(feedback) = feedback {
                temporal = temporal
                    .with_feedback(feedback_declaration(feedback)?)
                    .map_err(|e| crate::value_err(e.to_string()))?;
            }
            let observation =
                observation_kind.map(|kind| crate::observation_api::ObservationArgs {
                    kind,
                    latent: latent.unwrap_or_default(),
                    observed,
                    censoring,
                    event,
                    lower,
                    upper,
                    indicator,
                    assumption_kind: assumption_kind.unwrap_or_default(),
                    assumption_variables,
                    structural_model,
                });
            let mut response = ResponseQuery::new(functional).with_temporal(temporal);
            if let Some(population) = req.opts.target_population.clone() {
                response.target_population = population;
            }
            let query = CausalQuery::Response(crate::observation_api::maybe_with_observation(
                response,
                &schema,
                observation.as_ref(),
            )?);
            let mut builder = data.builder()?;
            builder = if let Some(graph) = class_graph {
                bind_temporal_class(builder, graph, accepted, req.opts.discovery_algorithm())
            } else {
                with_graph(builder, dag, accepted, req.opts.discovery_algorithm())
            };
            builder = req.opts.apply(builder.query(query));
            builder = req.opts.apply_inference(builder)?;
            builder = crate::temporal_api::apply_class_prior(
                builder,
                class_prior_ordered,
                class_prior_pairs,
            )?;
            builder = crate::temporal_api::apply_max_completions(builder, max_completions);
            req.compile(builder, true)
        })
    }

    /// Compile once for Pulse / Sustained on a TemporalDag (lagged `edges`) or a
    /// TemporalCpdag / TemporalPag (`class_graph`).
    ///
    /// Series columns, or a panel / multi-environment / event `frame` of the
    /// same schema; the study builder licenses or refuses the combination.
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
        control_level=0.0,
        active_level=1.0,
        max_history_lag=None,
        modifier=None,
        accepted=false,
        class_graph=None,
        class_prior_ordered=None,
        class_prior_pairs=None,
        max_completions=None,
        frame=None,
        seed=1,
        threads=None,
        options=None,
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
        control_level: f64,
        active_level: f64,
        max_history_lag: Option<u32>,
        modifier: Option<String>,
        accepted: bool,
        class_graph: Option<Bound<'_, PyAny>>,
        class_prior_ordered: Option<Vec<f64>>,
        class_prior_pairs: Option<Vec<(u64, f64)>>,
        max_completions: Option<usize>,
        frame: Option<Bound<'_, PyDict>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let class_graph = extract_temporal_class(class_graph, &req.names)?;
        let input = req.load_weighted_series(py, columns, frame.as_ref())?;
        let policy = policy.to_ascii_lowercase();
        req.off_gil(py, move |mut req| {
            let data = input.materialize()?;
            let schema = data.schema().clone();
            let t_id = schema.id_of(&treatment).map_err(py_err)?;
            let y_id = schema.id_of(&outcome).map_err(py_err)?;
            let mut q = temporal_effect_query(
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
            if let Some(modifier) = modifier.as_deref() {
                let modifier_id = schema.id_of(modifier).map_err(py_err)?;
                q = q.with_effect_modifiers([modifier_id]);
            }
            let mut builder = data.builder()?;
            builder = if let Some(graph) = class_graph {
                bind_temporal_class(builder, graph, accepted, req.opts.discovery_algorithm())
            } else {
                with_graph(
                    builder,
                    temporal_dag_from_schema_edges(&schema, &edges)?,
                    accepted,
                    req.opts.discovery_algorithm(),
                )
            };
            builder = req.opts.apply(builder.temporal_query(q));
            builder = req.opts.apply_inference(builder)?;
            builder = crate::temporal_api::apply_class_prior(
                builder,
                class_prior_ordered,
                class_prior_pairs,
            )?;
            builder = crate::temporal_api::apply_max_completions(builder, max_completions);
            req.compile(builder, true)
        })
    }

    /// Compile once for TemporalMediationEffect on a TemporalDag or TemporalCpdag.
    ///
    /// Series columns, or a panel / multi-environment / event `frame`; the
    /// study builder licenses or refuses the combination.
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
        accepted=false,
        class_graph=None,
        class_prior_ordered=None,
        class_prior_pairs=None,
        max_completions=None,
        frame=None,
        seed=1,
        threads=None,
        options=None,
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
        accepted: bool,
        class_graph: Option<Bound<'_, PyAny>>,
        class_prior_ordered: Option<Vec<f64>>,
        class_prior_pairs: Option<Vec<(u64, f64)>>,
        max_completions: Option<usize>,
        frame: Option<Bound<'_, PyDict>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_prior("temporal mediation")?;
        let class_graph = extract_temporal_class(class_graph, &req.names)?;
        let input = req.load_frame(py, columns, frame.as_ref())?;
        let contrast = contrast.to_string();
        req.off_gil(py, move |mut req| {
            let data = input.materialize()?;
            let schema = data.schema().clone();
            let q = temporal_mediation_query(
                &schema,
                &treatment,
                &mediator,
                &outcome,
                &contrast,
                control_level,
                active_level,
                horizons,
                &req.opts,
            )?;
            let mut builder = data.builder()?;
            builder = if let Some(graph) = class_graph {
                bind_temporal_class(builder, graph, accepted, req.opts.discovery_algorithm())
            } else {
                with_graph(
                    builder,
                    temporal_dag_from_schema_edges(&schema, &edges)?,
                    accepted,
                    req.opts.discovery_algorithm(),
                )
            };
            builder = req.opts.apply(builder.query(CausalQuery::Mediation(q)));
            builder = req.opts.apply_inference(builder)?;
            builder = crate::temporal_api::apply_class_prior(
                builder,
                class_prior_ordered,
                class_prior_pairs,
            )?;
            builder = crate::temporal_api::apply_max_completions(builder, max_completions);
            req.compile(builder, true)
        })
    }
}
