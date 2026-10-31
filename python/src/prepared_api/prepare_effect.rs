//! Static and class average-effect prepare routes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{EstimatorId, IdentifierId, Study};
use antecedent_core::{AverageEffectQuery, CausalQuery, Intervention, ResponseQuery, Value};
use antecedent_data::{TableView, TabularData};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use std::sync::Arc;

use super::{
    PrepareRequest, PyPreparedAnalysis, StaticClassGraph, apply_estimator, configured_spec,
    parse_estimator, parse_identifier, parse_response_options, response_query, static_ate_query,
    static_kind_query, with_graph, with_response_options,
};
use crate::response_api::build_functional;
use crate::{dag_from_named_edges, py_err};

#[pymethods]
impl PyPreparedAnalysis {
    /// Compile once from tabular columns + DAG edges (static AverageEffect).
    ///
    /// `rd.sharp` is selected by the estimator id or by the running-variable
    /// triple (loose or inside `estimator_config`); an omitted identifier then
    /// resolves to the matching `rd.sharp` identifier.
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
        estimator_config=None,
        outcome_functional=None,
        running_variable=None,
        cutoff=None,
        bandwidth=None,
        accepted=false,
        seed=1,
        threads=None,
        options=None,
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
        estimator_config: Option<&Bound<'_, PyDict>>,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
        running_variable: Option<String>,
        cutoff: Option<f64>,
        bandwidth: Option<f64>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let parsed_estimator = crate::estimator_config::parse_estimator_config(
            estimator_config,
            estimator.as_deref(),
            req.opts.ambient_bootstrap(),
        )?;
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let data = req.load_columns_recording(py, columns)?;

        req.off_gil(py, move |req| {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let mut query =
                AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
            if let Some(pop) = req.opts.target_population.clone() {
                query = query.with_target_population(pop);
            }
            if let Some(functional) = functional {
                query = query.with_outcome_functional(functional);
            }
            let (merged_rv, merged_cutoff, merged_bandwidth) =
                crate::estimator_config::merge_rd_triple(
                    running_variable,
                    cutoff,
                    bandwidth,
                    parsed_estimator.rd_running_variable,
                    parsed_estimator.rd_cutoff,
                    parsed_estimator.rd_bandwidth,
                )?;
            let rd_ids = crate::ate_api::parse_rd_config(
                estimator.as_deref(),
                merged_rv.as_deref(),
                merged_cutoff,
                merged_bandwidth,
                |rv| data.schema().id_of(rv).map_err(py_err),
            )?;
            let mut builder = req.opts.apply_budget_for(
                with_graph(Study::tabular(data), dag, accepted, req.opts.discovery_algorithm())
                    .query(query),
                parsed_estimator.spec.is_some(),
            );
            let mut identifier = parse_identifier(identifier)?;
            let mut estimator_id = parse_estimator(estimator)?;
            if let Some((rv_id, cut, bw)) = rd_ids {
                builder = builder.rd_design(crate::ate_api::rd_design(
                    rv_id,
                    cut,
                    bw,
                    parsed_estimator.rd_se_kind,
                ));
                // The running-variable design is the rd.sharp strategy; an
                // omitted identifier or estimator names that strategy.
                identifier = identifier.or(Some(IdentifierId::RdSharp));
                estimator_id = estimator_id.or(Some(EstimatorId::RdSharp));
            }
            req.refuse_conflicting_selection(identifier, estimator_id)?;
            if let Some(id) = identifier {
                builder = builder.identifier(id);
            }
            if let Some(spec) = parsed_estimator.spec {
                builder = builder.estimator(spec);
            } else if let Some(est) = estimator_id {
                builder = builder.estimator(est);
            }
            req.finish(builder, false)
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
        estimator_config=None,
        outcome_functional=None,
        seed=1,
        threads=None,
        options=None,
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
        estimator_config: Option<&Bound<'_, PyDict>>,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let spec = configured_spec(estimator_config, estimator.as_deref(), &req.opts)?;
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let within = crate::parse_within_tier(Some(within_tier.as_str()))?;
            let named: Vec<Vec<&str>> =
                tiers.iter().map(|tier| tier.iter().map(String::as_str).collect()).collect();
            let background =
                antecedent_graph::TieredBackground::from_named(data.schema(), &named, within)
                    .map_err(py_err)?;
            let query = static_ate_query(
                &data,
                &treatment,
                &outcome,
                control_level,
                active_level,
                &req.opts,
            )?;
            let query = match functional {
                Some(functional) => query.with_outcome_functional(functional),
                None => query,
            };
            let builder =
                Study::tabular(data).tiered_background(background).map_err(py_err)?.query(query);
            let builder = apply_estimator(
                req.opts.apply_budget_for(builder, spec.is_some()),
                estimator,
                spec,
            )?;
            req.finish(builder, false)
        })
    }

    /// Compile once for AverageEffect on a supplied PAG, CPDAG, or ADMG.
    ///
    /// A PAG uses the generalized-adjustment envelope, a CPDAG the MEC
    /// envelope, and a bidirected ADMG general ID with the functional plug-in.
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
        estimator_config=None,
        outcome_functional=None,
        accepted=false,
        seed=1,
        threads=None,
        options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_class_ate(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: Bound<'_, PyAny>,
        treatment: String,
        outcome: String,
        control_level: f64,
        active_level: f64,
        identifier: Option<String>,
        estimator: Option<String>,
        estimator_config: Option<&Bound<'_, PyDict>>,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let spec = configured_spec(estimator_config, estimator.as_deref(), &req.opts)?;
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let graph = StaticClassGraph::extract(&graph, &req.names)?;
        if matches!(graph, StaticClassGraph::Admg(_)) {
            req.refuse_prior("an ADMG general-ID average effect")?;
        }
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let query = static_ate_query(
                &data,
                &treatment,
                &outcome,
                control_level,
                active_level,
                &req.opts,
            )?;
            let query = match functional {
                Some(functional) => query.with_outcome_functional(functional),
                None => query,
            };
            let builder = graph
                .bind(Study::tabular(data), accepted, req.opts.discovery_algorithm())
                .query(query);
            let mut builder = req.opts.apply_budget_for(builder, spec.is_some());
            if let Some(id) = parse_identifier(identifier)? {
                builder = builder.identifier(id);
            }
            let builder = apply_estimator(builder, estimator, spec)?;
            req.finish(builder, false)
        })
    }

    /// Compile once for ResponseCurve / InterventionResponse on a supplied PAG or CPDAG.
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
        response_options=None,
        accepted=false,
        seed=1,
        threads=None,
        options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_class_response(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: Bound<'_, PyAny>,
        kind: String,
        treatments: Vec<String>,
        outcomes: Vec<String>,
        grid: Option<Vec<f64>>,
        intervention_kinds: Option<Vec<String>>,
        intervention_parameters: Option<Vec<Vec<f64>>>,
        identifier: Option<String>,
        estimator: Option<String>,
        response_options: Option<Bound<'_, PyDict>>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_prior("a Cpdag/Pag response envelope")?;
        let response_options = parse_response_options(response_options.as_ref())?;
        let graph = StaticClassGraph::extract(&graph, &req.names)?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
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
            let query = response_query(functional, &req.opts);
            let mut builder = req.opts.apply(with_response_options(
                graph
                    .bind(Study::tabular(data), accepted, req.opts.discovery_algorithm())
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
        let query =
            if kind == "mediation" || kind == "counterfactual" || kind == "nested_counterfactual" {
                if treatments.len() != 1 || outcomes.len() != 1 {
                    return Err(crate::value_err("one treatment/outcome required"));
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
        // The compiled study over the schema-only table: a bare DAG is an explicit structure.
        let id =
            Study::tabular(data).graph(dag).query(query).identify_structure().map_err(py_err)?;
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
        contrast="mediated", control_level=0.0, active_level=1.0, extra_outcomes=Vec::new(),
        joint_active=Vec::new(), joint_control=Vec::new(), accepted=false, seed=1,
        threads=None, options=None))]
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
        extra_outcomes: Vec<String>,
        joint_active: Vec<(String, f64)>,
        joint_control: Vec<(String, f64)>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        if kind == "counterfactual" || kind == "nested_counterfactual" {
            req.refuse_prior("a counterfactual query")?;
        }
        let data = req.load_columns(py, columns)?;
        let mut query = static_kind_query(
            data.schema(),
            kind,
            &treatment,
            &outcome,
            &mediators,
            contrast,
            control_level,
            active_level,
        )?;
        if kind == "counterfactual" {
            query = widen_counterfactual(
                query,
                data.schema(),
                &extra_outcomes,
                &joint_active,
                &joint_control,
            )?;
        } else if !extra_outcomes.is_empty()
            || !joint_active.is_empty()
            || !joint_control.is_empty()
        {
            return Err(crate::value_err(
                "additional outcomes and joint interventions apply to a counterfactual only",
            ));
        }
        if let (CausalQuery::Mediation(q), Some(population)) =
            (&mut query, req.opts.target_population.clone())
        {
            q.target_population = population;
        } else {
            req.refuse_population("a counterfactual unit-effect query")?;
        }
        req.off_gil(py, move |req| {
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let builder = req.opts.apply(
                with_graph(Study::tabular(data), dag, accepted, req.opts.discovery_algorithm())
                    .query(query),
            );
            req.finish(builder, false)
        })
    }

    /// Freeze identification for a complete-data static derivative.
    #[staticmethod]
    #[pyo3(signature = (names, columns, edges, kind, treatments, outcomes, *, at=None,
        direction=None, order=1, scale="identity", weighting="observed", response_options=None,
        accepted=false, seed=1, threads=None, options=None))]
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
        response_options: Option<Bound<'_, PyDict>>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_prior("a response derivative")?;
        let response_options = parse_response_options(response_options.as_ref())?;
        let data = req.load_columns(py, columns)?;
        let scale = crate::response_api::parse_scale(scale)?;
        let weighting = crate::response_api::parse_weighting(weighting)?;
        req.off_gil(py, move |req| {
            let ts = crate::response_api::resolve_names(data.schema(), &treatments)?;
            let ys = crate::response_api::resolve_names(data.schema(), &outcomes)?;
            let functional = build_functional(
                &kind, &ts, &ys, None, at, direction, None, None, order, scale, weighting,
            )?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let builder = with_response_options(
                with_graph(Study::tabular(data), dag, accepted, req.opts.discovery_algorithm())
                    .query(response_query(functional, &req.opts)),
                response_options,
            );
            req.finish_applied(builder, false)
        })
    }
}

/// Widen a single-outcome counterfactual with further outcomes and a joint baseline.
///
/// `joint_active` adds simultaneous interventions to the active world and `joint_control`
/// the matching ones to the control world; the query's own validation requires both worlds
/// to intervene on the same variables.
fn widen_counterfactual(
    query: CausalQuery,
    schema: &antecedent_core::CausalSchema,
    extra_outcomes: &[String],
    joint_active: &[(String, f64)],
    joint_control: &[(String, f64)],
) -> PyResult<CausalQuery> {
    let CausalQuery::Counterfactual(mut cf) = query else {
        return Ok(query);
    };
    if !extra_outcomes.is_empty() {
        let mut outcomes = cf.outcomes.to_vec();
        for name in extra_outcomes {
            let id = schema.id_of(name).map_err(py_err)?;
            if outcomes.contains(&id) {
                return Err(crate::value_err(format!("outcome {name} is listed twice")));
            }
            outcomes.push(id);
        }
        cf.outcomes = Arc::from(outcomes);
    }
    if joint_active.is_empty() != joint_control.is_empty() {
        return Err(crate::value_err(
            "a joint baseline needs further interventions in both the active and control worlds",
        ));
    }
    if !joint_active.is_empty() {
        let mut active = cf.interventions.to_vec();
        let mut control = cf.control_world();
        for (name, level) in joint_active {
            let id = schema.id_of(name).map_err(py_err)?;
            active.push(Intervention::set(id, Value::f64(*level)));
        }
        for (name, level) in joint_control {
            let id = schema.id_of(name).map_err(py_err)?;
            control.push(Intervention::set(id, Value::f64(*level)));
        }
        cf.interventions = Arc::from(active);
        cf = cf.with_joint_control(control);
    }
    cf.validate().map_err(|error| crate::value_err(error.to_string()))?;
    Ok(CausalQuery::Counterfactual(cf))
}
