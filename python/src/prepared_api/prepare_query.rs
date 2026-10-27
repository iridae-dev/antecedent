//! Intervention, conditional, transport, and interference prepare routes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{EstimatorId, IdentifierId, Study};
use antecedent_core::{
    CausalQuery, Intervention, InterventionalDistributionQuery, PathSpecificEffectQuery, Value,
};
use antecedent_data::TableView;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::{
    conditional_query, intervention_response_query, parse_estimator, parse_identifier, with_graph,
    PrepareRequest, PyPreparedAnalysis, StaticClassGraph,
};
use crate::{dag_from_named_edges, graphs, py_err, py_msg, require_named_graph_order};

#[pymethods]
impl PyPreparedAnalysis {
    /// Compile once from tabular columns + DAG edges (static InterventionResponse).
    ///
    /// Reuses `response_api::build_functional`'s `"intervention_response"`
    /// branch so a prepared handle's functional is built identically to the
    /// one-shot path; the generic response branch on `PreparedStudy` caches
    /// identification for any response functional on a supplied `Dag`.
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
        identifier=None,
        estimator=None,
        outcome_functional=None,
        accepted=false,
        seed=1,
        threads=None,
        options=None,
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
        identifier: Option<String>,
        estimator: Option<String>,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_prior("an InterventionResponse")?;
        let outcome_functional =
            crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let query = intervention_response_query(
                &data,
                &outcome,
                &treatments,
                intervention_kinds,
                intervention_parameters,
                outcome_functional,
                &req.opts,
            )?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let mut builder = req.opts.apply(
                with_graph(Study::tabular(data), dag, accepted, req.opts.discovery_algorithm())
                    .query(query),
            );
            if let Some(id) = parse_identifier(identifier)? {
                builder = builder.identifier(id);
            }
            if let Some(est) = parse_estimator(estimator)? {
                builder = builder.estimator(est);
            }
            req.finish(builder, false)
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
        seed=1,
        threads=None,
        options=None,
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
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let outcome_functional =
            crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let within = crate::parse_within_tier(Some(within_tier.as_str()))?;
            let named: Vec<Vec<&str>> =
                tiers.iter().map(|tier| tier.iter().map(String::as_str).collect()).collect();
            let background =
                antecedent_graph::TieredBackground::from_named(data.schema(), &named, within)
                    .map_err(py_err)?;
            let query = intervention_response_query(
                &data,
                &outcome,
                &treatments,
                intervention_kinds,
                intervention_parameters,
                outcome_functional,
                &req.opts,
            )?;
            let builder =
                Study::tabular(data).tiered_background(background).map_err(py_err)?.query(query);
            // The Frequentist joint cell is cell.aipw; a Bayesian declaration runs
            // response.bayesian over the same closure set (unlicensed).
            let builder = if req.opts.inference.is_bayesian() {
                builder
            } else {
                builder.estimator(antecedent::EstimatorId::CellAipw)
            };
            req.finish_applied(builder, false)
        })
    }

    /// Compile once for ConditionalEffect on a supplied DAG, CPDAG, or PAG.
    ///
    /// A class structure (`graph`) uses generalized adjustment per completion;
    /// otherwise `edges` is the DAG.
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        edges,
        treatment,
        outcome,
        modifiers,
        *,
        graph=None,
        control_level=0.0,
        active_level=1.0,
        identifier=None,
        estimator=None,
        outcome_functional=None,
        accepted=false,
        seed=1,
        threads=None,
        options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_conditional(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        treatment: String,
        outcome: String,
        modifiers: Vec<String>,
        graph: Option<Bound<'_, PyAny>>,
        control_level: f64,
        active_level: f64,
        identifier: Option<String>,
        estimator: Option<String>,
        outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        if graph.is_some() {
            req.refuse_prior("a Cpdag/Pag conditional envelope")?;
        }
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let class = graph.map(|g| StaticClassGraph::extract(&g, &req.names)).transpose()?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let query = conditional_query(
                &data,
                &treatment,
                &outcome,
                &modifiers,
                control_level,
                active_level,
                functional,
                &req.opts,
            )?;
            let mut builder = if let Some(class) = class {
                class.bind(Study::tabular(data), accepted, req.opts.discovery_algorithm())
            } else {
                let dag = dag_from_named_edges(data.schema(), &edges)?;
                with_graph(Study::tabular(data), dag, accepted, req.opts.discovery_algorithm())
            };
            builder = req.opts.apply(builder.query(query));
            if let Some(id) = parse_identifier(identifier)? {
                builder = builder.identifier(id);
            }
            if let Some(est) = parse_estimator(estimator)? {
                builder = builder.estimator(est);
            }
            req.finish(builder, false)
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
        accepted=false,
        seed=1,
        threads=None,
        options=None,
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
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_prior("a path-specific effect")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let mut query = PathSpecificEffectQuery::binary(t_id, y_id)
                .with_max_paths(max_paths)
                .with_max_len(max_len);
            query.control = Intervention::set(t_id, Value::f64(control_level));
            query.active = Intervention::set(t_id, Value::f64(active_level));
            if let Some(population) = req.opts.target_population.clone() {
                query.target_population = population;
            }
            if let Some(nodes) = path_nodes {
                let mut ids = Vec::with_capacity(nodes.len());
                for name in &nodes {
                    ids.push(data.schema().id_of(name).map_err(py_err)?);
                }
                query = query.with_path_nodes(ids);
            }
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let builder =
                with_graph(Study::tabular(data), dag, accepted, req.opts.discovery_algorithm())
                    .query(CausalQuery::PathSpecific(query))
                    .identifier(IdentifierId::PathSpecificNatural)
                    .estimator(EstimatorId::FunctionalEffect);
            req.finish_applied(builder, false)
        })
    }

    /// Compile once from tabular columns + DAG edges or a supplied ADMG
    /// (static InterventionalDistribution).
    #[staticmethod]
    #[pyo3(signature = (
        names,
        columns,
        edges,
        outcome,
        interventions,
        *,
        graph=None,
        conditioning=None,
        accepted=false,
        seed=1,
        threads=None,
        options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_distribution(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        outcome: String,
        interventions: std::collections::HashMap<String, f64>,
        graph: Option<Bound<'_, PyAny>>,
        conditioning: Option<Vec<String>>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_prior("an interventional distribution")?;
        let class = graph.map(|g| StaticClassGraph::extract(&g, &req.names)).transpose()?;
        if matches!(&class, Some(g) if !matches!(g, StaticClassGraph::Admg(_))) {
            return Err(crate::value_err(
                "prepare_distribution graph must be an Admg; Pag/Cpdag remain refused",
            ));
        }
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let mut ivs = Vec::with_capacity(interventions.len());
            for (name, level) in &interventions {
                let id = data.schema().id_of(name).map_err(py_err)?;
                ivs.push(Intervention::set(id, Value::f64(*level)));
            }
            let mut query = InterventionalDistributionQuery::new(y_id, ivs);
            if let Some(population) = req.opts.target_population.clone() {
                query.target_population = population;
            }
            if let Some(cond) = conditioning {
                let mut z = Vec::with_capacity(cond.len());
                for name in &cond {
                    z.push(data.schema().id_of(name).map_err(py_err)?);
                }
                query = query.with_conditioning(z);
            }
            let builder = if let Some(class) = class {
                class.bind(Study::tabular(data), accepted, req.opts.discovery_algorithm())
            } else {
                let dag = dag_from_named_edges(data.schema(), &edges)?;
                with_graph(Study::tabular(data), dag, accepted, req.opts.discovery_algorithm())
            };
            let builder = builder
                .query(CausalQuery::Distribution(query))
                .identifier(IdentifierId::GeneralId)
                .estimator(EstimatorId::FunctionalDistribution);
            req.finish_applied(builder, false)
        })
    }

    /// Freeze a single-source transport study on a selection-diagram `Admg`.
    ///
    /// The selection diagram (the graph plus `selections`) and the trial,
    /// selection-probability and treatment-probability columns freeze at
    /// prepare; every click reads those columns from the clicked data.
    #[staticmethod]
    #[pyo3(signature = (names, columns, graph, selections, source_population, target_population,
        source_experiments, kind, treatments, outcomes, trial, selection_probability,
        treatment_probability, *, grid=None, at=None, direction=None, order=1, scale="identity",
        weighting="observed", accepted=false, seed=1, threads=None, options=None, catalog=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_transport(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: Bound<'_, PyAny>,
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
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
        catalog: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_prior("a transport query")?;
        let admg = graph
            .extract::<graphs::Admg>()
            .map_err(|_| crate::value_err("TransportQuery requires graph=Admg(...)"))?;
        require_named_graph_order(&admg.names, &req.names, "Admg")?;
        let catalog = catalog
            .map(|c| crate::transport_interference_api::parse_catalog(c, &admg))
            .transpose()?;
        let admg = admg.aligned_to_names(&req.names)?;
        let data = req.load_columns(py, columns)?;
        let (scale, weighting) = (scale.to_owned(), weighting.to_owned());
        req.off_gil(py, move |req| {
            use crate::transport_interference_api::{schema_ids, transport_query, ResponseArgs};
            let schema = data.schema().clone();
            let mut query = transport_query(
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
            if let Some(catalog) = catalog {
                query = query.with_catalog(catalog).map_err(|e| crate::value_err(e.to_string()))?;
            }
            let column = |name: &str| crate::graph_build::schema_var_id(&schema, name);
            let trial_spec = antecedent::TransportTrialSpec {
                trial: column(&trial)?,
                selection_probability: column(&selection_probability)?,
                treatment_probability: column(&treatment_probability)?,
            };
            let builder =
                with_graph(Study::tabular(data), admg, accepted, req.opts.discovery_algorithm())
                    .query(CausalQuery::Transport(query))
                    .selection_targets(schema_ids(&schema, &selections)?)
                    .transport_trial(trial_spec);
            req.finish_applied(builder, false)
        })
    }

    /// Freeze a randomized-interference study: the fixed unit network and the
    /// realized assignment are the design; the unit table is the data.
    #[staticmethod]
    #[pyo3(signature = (names, columns, edges, outcome, network, realized_assignment,
        assignment_kind, assignment_probabilities, treated, clusters, treated_clusters, exposure,
        from_level, to_level, *, probability_draws=10_000, low_probability=0.0,
        high_probability=0.0, realized_saturation=Vec::new(),
        propensity_from=Vec::new(), propensity_to=Vec::new(), propensity_provenance="known".to_string(),
        assume_network_exchangeability=false, accepted=false, seed=1, threads=None,
        options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_interference(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        outcome: String,
        network: Vec<(u64, u64, f64)>,
        realized_assignment: Vec<bool>,
        assignment_kind: String,
        assignment_probabilities: Vec<f64>,
        treated: usize,
        clusters: Vec<u32>,
        treated_clusters: usize,
        exposure: String,
        from_level: (f64, f64),
        to_level: (f64, f64),
        probability_draws: u32,
        low_probability: f64,
        high_probability: f64,
        realized_saturation: Vec<f64>,
        propensity_from: Vec<f64>,
        propensity_to: Vec<f64>,
        propensity_provenance: String,
        assume_network_exchangeability: bool,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_prior("an interference query")?;
        req.refuse_population("an interference exposure contrast")?;
        let data = req.load_columns(py, columns)?;
        req.off_gil(py, move |req| {
            use crate::transport_interference_api::{interference_query, InterferenceArgs};
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            let query = interference_query(
                InterferenceArgs {
                    assignment_kind,
                    assignment_probabilities,
                    treated,
                    clusters,
                    treated_clusters,
                    low_probability,
                    high_probability,
                    realized_saturation,
                    propensity_from,
                    propensity_to,
                    propensity_provenance,
                    assume_network_exchangeability,
                    exposure,
                    from_level,
                    to_level,
                    probability_draws,
                },
                outcome_id,
            )?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let network = antecedent::estimate::NetworkData::try_new(
                data.clone(),
                network
                    .into_iter()
                    .map(|(from, to, weight)| {
                        antecedent::estimate::NetworkEdge::new(from, to, weight)
                    })
                    .collect::<Vec<_>>(),
            )
            .map_err(py_err)?;
            let builder =
                with_graph(Study::tabular(data), dag, accepted, req.opts.discovery_algorithm())
                    .query(CausalQuery::Interference(query))
                    .interference(antecedent::InterferenceSpec {
                        network,
                        assignment: Arc::from(realized_assignment),
                    });
            req.finish_applied(builder, false)
        })
    }

    /// Freeze a graphless Bernoulli, complete, stratified, or cluster randomized ITT study.
    #[staticmethod]
    #[pyo3(signature = (names, columns, outcome, realized_assignment, assignment_probabilities,
        assignment_units, outcome_units, treatment_arms, design_kind, treated_units=None, blocks=None,
        treated_per_row=None, *, treated_clusters=None, fixed_cuped=None, ancova_covariates=None, periods=None, received_treatment=None, treatment_on_treated=false, exact_randomization_test=false,
        second_factor_assignment=None, factorial_cell_counts=None, second_factor_arms=None,
        multi_arm_labels=None, multi_arm_assignment=None, multi_arm_probabilities=None,
        accepted=false, seed=1, threads=None,
        options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_randomized_effect(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        outcome: String,
        realized_assignment: Vec<bool>,
        assignment_probabilities: Vec<f64>,
        assignment_units: Vec<String>,
        outcome_units: Vec<String>,
        treatment_arms: (String, String),
        design_kind: String,
        treated_units: Option<usize>,
        blocks: Option<Vec<String>>,
        treated_per_row: Option<Vec<usize>>,
        treated_clusters: Option<usize>,
        fixed_cuped: Option<(String, f64)>,
        ancova_covariates: Option<Vec<String>>,
        periods: Option<Vec<String>>,
        received_treatment: Option<Vec<bool>>,
        treatment_on_treated: bool,
        exact_randomization_test: bool,
        second_factor_assignment: Option<Vec<bool>>,
        factorial_cell_counts: Option<[usize; 4]>,
        second_factor_arms: Option<(String, String)>,
        multi_arm_labels: Option<Vec<String>>,
        multi_arm_assignment: Option<Vec<usize>>,
        multi_arm_probabilities: Option<Vec<Vec<f64>>>,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        req.refuse_design("a randomized ITT query")?;
        let data = req.load_weighted_columns(py, columns)?;
        req.off_gil(py, move |req| {
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            let design = match design_kind.as_str() {
                "bernoulli" => antecedent_core::RandomizationDesign::Bernoulli,
                "complete" => antecedent_core::RandomizationDesign::Complete {
                    treated_units: treated_units
                        .ok_or_else(|| py_msg("complete randomization requires treated_units"))?,
                },
                "stratified" => antecedent_core::RandomizationDesign::Stratified {
                    blocks: blocks
                        .ok_or_else(|| py_msg("stratified randomization requires blocks"))?
                        .into_iter()
                        .map(Arc::<str>::from)
                        .collect::<Vec<_>>()
                        .into(),
                    treated_per_row: treated_per_row
                        .ok_or_else(|| py_msg("stratified randomization requires treated_per_row"))?
                        .into(),
                },
                "cluster" => antecedent_core::RandomizationDesign::Cluster {
                    treated_clusters: treated_clusters
                        .ok_or_else(|| py_msg("cluster randomization requires treated_clusters"))?,
                },
                "switchback" => antecedent_core::RandomizationDesign::Switchback {
                    periods: periods
                        .ok_or_else(|| py_msg("switchback randomization requires periods"))?
                        .into_iter()
                        .map(Arc::<str>::from)
                        .collect::<Vec<_>>()
                        .into(),
                },
                "factorial_2x2" => {
                    let second = second_factor_assignment
                        .ok_or_else(|| py_msg("factorial second-factor assignment is required"))?;
                    let counts = factorial_cell_counts
                        .ok_or_else(|| py_msg("factorial four-cell counts are required"))?;
                    let arms = second_factor_arms
                        .ok_or_else(|| py_msg("factorial second-factor labels are required"))?;
                    antecedent_core::RandomizationDesign::Factorial2x2 {
                        second_factor_assignment: second.into(),
                        cell_counts: counts,
                        second_factor_arms: (Arc::<str>::from(arms.0), Arc::<str>::from(arms.1)),
                    }
                }
                "multi_arm" => antecedent_core::RandomizationDesign::MultiArm {
                    arms: multi_arm_labels
                        .ok_or_else(|| py_msg("multi-arm labels are required"))?
                        .into_iter()
                        .map(Arc::<str>::from)
                        .collect::<Vec<_>>()
                        .into(),
                    assignment: multi_arm_assignment
                        .ok_or_else(|| py_msg("multi-arm assignment is required"))?
                        .into(),
                    probabilities: multi_arm_probabilities
                        .ok_or_else(|| py_msg("multi-arm probability rows are required"))?
                        .into(),
                },
                _ => {
                    return Err(py_msg(
                        "design_kind must be bernoulli, complete, stratified, cluster, switchback, factorial_2x2, or multi_arm",
                    ));
                }
            };
            let mut query = antecedent_core::RandomizedEffectQuery::with_design(
                design,
                outcome_id,
                realized_assignment,
                assignment_probabilities,
                assignment_units.into_iter().map(Arc::<str>::from).collect::<Vec<_>>(),
                outcome_units.into_iter().map(Arc::<str>::from).collect::<Vec<_>>(),
                (Arc::<str>::from(treatment_arms.0), Arc::<str>::from(treatment_arms.1)),
            );
            if let Some((covariate, coefficient)) = fixed_cuped {
                let covariate_id = crate::graph_build::schema_var_id(data.schema(), &covariate)?;
                query = query.with_fixed_cuped(covariate_id, coefficient);
            }
            if let Some(covariates) = ancova_covariates {
                if !covariates.is_empty() {
                    let ids = covariates
                        .iter()
                        .map(|name| crate::graph_build::schema_var_id(data.schema(), name))
                        .collect::<PyResult<Vec<_>>>()?;
                    query = query.with_ancova(ids);
                }
            }
            if let Some(received) = received_treatment {
                query = if treatment_on_treated {
                    query.with_treatment_on_treated(received)
                } else {
                    query.with_received_treatment(received)
                };
            } else if treatment_on_treated {
                return Err(py_msg("treatment-on-treated requires observed treatment receipt"));
            }
            if exact_randomization_test {
                query = query.with_exact_randomization_test();
            }
            query
                .validate()
                .map_err(|e| py_err(antecedent::CausalError::rejected(e)))?;
            let _ = accepted;
            let builder = Study::tabular(data).query(CausalQuery::RandomizedEffect(query));
            req.finish_applied(builder, false)
        })
    }
}
