//! Temporal attachments reuse native response and unfolded cross-world scientific owners.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalQuery, CausalSchema, Intervention, InterventionalDistributionQuery,
    PathSpecificEffectQuery, ResponseQuery, TemporalPolicy, TemporalResponseSpec, Value,
};
use antecedent_data::TableView;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use super::{
    PrepareRequest, PyPreparedAnalysis, bind_temporal_class, dbn_posterior, extract_temporal_class,
    one_series, parse_response_options, static_kind_query, take_supplied_posterior, with_graph,
    with_response_options,
};
use crate::{graphs, py_err, require_named_graph_order};

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct TemporalGeometry {
    horizons: Vec<u32>,
    policy: String,
    treatment_lag: u32,
    max_history_lag: Option<u32>,
    window: Option<(i32, i32)>,
}

impl TemporalGeometry {
    fn response_spec(self) -> PyResult<TemporalResponseSpec> {
        let policy = match self.window {
            Some((from, until)) if self.policy == "sustained" => {
                TemporalPolicy::sustained(from, until)
            }
            Some(_) => return Err(crate::value_err("a temporal window requires sustained policy")),
            None => crate::temporal_license::policy_at_lag(self.policy, self.treatment_lag)?,
        };
        TemporalResponseSpec::new(self.horizons, policy, self.max_history_lag)
            .map_err(antecedent::CausalError::from)
            .map_err(py_err)
    }

    fn require_unfolded_geometry(&self) -> PyResult<()> {
        if self.horizons.as_slice() != [1]
            || self.policy != "pulse"
            || self.treatment_lag != 1
            || self.max_history_lag.is_some()
            || self.window.is_some()
        {
            return Err(crate::refusal(
                antecedent_core::reason_code!("option_not_applicable"),
                "this unfolded cross-world target has one lag and one horizon; temporal geometry cannot change its scientific target",
            ));
        }
        Ok(())
    }
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct TemporalTarget {
    kind: String,
    treatments: Vec<String>,
    outcomes: Vec<String>,
    at: Option<Vec<f64>>,
    direction: Option<Vec<f64>>,
    order: u8,
    scale: String,
    weighting: String,
    control_level: f64,
    active_level: f64,
    mediators: Vec<String>,
    path_nodes: Option<Vec<String>>,
    max_paths: usize,
    max_len: usize,
    interventions: Vec<(String, f64)>,
    conditioning: Vec<String>,
}

impl TemporalTarget {
    fn query(
        self,
        schema: &CausalSchema,
        temporal: TemporalGeometry,
        opts: &crate::prepared_options::PrepareOptions,
    ) -> PyResult<CausalQuery> {
        if matches!(
            self.kind.as_str(),
            "counterfactual" | "nested_counterfactual" | "path_specific" | "distribution"
        ) {
            temporal.require_unfolded_geometry()?;
            return self.unfolded_query(schema, opts);
        }
        let treatments = crate::response_api::resolve_names(schema, &self.treatments)?;
        let outcomes = crate::response_api::resolve_names(schema, &self.outcomes)?;
        let functional = crate::response_api::build_functional(
            &self.kind,
            &treatments,
            &outcomes,
            None,
            self.at,
            self.direction,
            None,
            None,
            self.order,
            crate::response_api::parse_scale(&self.scale)?,
            crate::response_api::parse_weighting(&self.weighting)?,
        )?;
        let mut response = ResponseQuery::new(functional).with_temporal(temporal.response_spec()?);
        if let Some(population) = opts.target_population.clone() {
            response.target_population = population;
        }
        Ok(CausalQuery::Response(response))
    }

    fn unfolded_query(
        self,
        schema: &CausalSchema,
        opts: &crate::prepared_options::PrepareOptions,
    ) -> PyResult<CausalQuery> {
        let outcome = self
            .outcomes
            .first()
            .ok_or_else(|| crate::value_err("temporal target requires an outcome"))?;
        let y = schema.id_of(outcome).map_err(py_err)?;
        let mut query = if self.kind == "distribution" {
            let interventions = self
                .interventions
                .into_iter()
                .map(|(name, value)| {
                    schema
                        .id_of(&name)
                        .map(|id| Intervention::set(id, Value::f64(value)))
                        .map_err(py_err)
                })
                .collect::<PyResult<Vec<_>>>()?;
            // An unconditional law has no conditioning variables. The response
            // coordinate parser deliberately requires a nonempty response axis.
            let conditioning = self
                .conditioning
                .iter()
                .map(|name| schema.id_of(name).map_err(py_err))
                .collect::<PyResult<Vec<_>>>()?;
            CausalQuery::Distribution(
                InterventionalDistributionQuery::new(y, interventions)
                    .with_conditioning(conditioning),
            )
        } else {
            let treatment = self
                .treatments
                .first()
                .ok_or_else(|| crate::value_err("temporal target requires a treatment"))?;
            if self.kind == "path_specific" {
                let t = schema.id_of(treatment).map_err(py_err)?;
                let mut path = PathSpecificEffectQuery::binary(t, y)
                    .with_max_paths(self.max_paths)
                    .with_max_len(self.max_len);
                path.control = Intervention::set(t, Value::f64(self.control_level));
                path.active = Intervention::set(t, Value::f64(self.active_level));
                if let Some(nodes) = self.path_nodes {
                    path =
                        path.with_path_nodes(crate::response_api::resolve_names(schema, &nodes)?);
                }
                CausalQuery::PathSpecific(path)
            } else {
                static_kind_query(
                    schema,
                    &self.kind,
                    treatment,
                    outcome,
                    &self.mediators,
                    "mediated",
                    self.control_level,
                    self.active_level,
                )?
            }
        };
        if let Some(population) = opts.target_population.clone() {
            match &mut query {
                CausalQuery::PathSpecific(path) => path.target_population = population,
                CausalQuery::Distribution(distribution) => {
                    distribution.target_population = population
                }
                _ => {
                    return Err(crate::refusal(
                        antecedent_core::reason_code!("option_not_applicable"),
                        "this unfolded target does not declare a target population",
                    ));
                }
            }
        }
        Ok(query)
    }
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct DbnRequest {
    max_lag: u32,
    force_mcmc: bool,
    n_chains: u32,
    n_warmup: u32,
    n_draws: u32,
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Retain the actual temporal target; compile/seal work remains in PrepareRequest.
    #[staticmethod]
    #[pyo3(signature = (names, columns, graph, query_spec, *, options=None, seed=1, threads=None, frame=None, accepted=false))]
    #[expect(
        clippy::too_many_arguments,
        reason = "private native entry mirrors the shared frame, execution controls and owned-column boundary"
    )]
    fn prepare_temporal_query(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        graph: Option<Bound<'_, PyAny>>,
        query_spec: &Bound<'_, PyDict>,
        options: Option<Bound<'_, PyDict>>,
        seed: u64,
        threads: Option<u32>,
        frame: Option<Bound<'_, PyDict>>,
        accepted: bool,
    ) -> PyResult<Self> {
        let req = PrepareRequest::parse(names, options.as_ref(), seed, threads)?;
        let target: TemporalTarget = query_spec
            .get_item("target")?
            .ok_or_else(|| crate::value_err("temporal target is required"))?
            .extract()?;
        let geometry: TemporalGeometry = query_spec
            .get_item("temporal")?
            .ok_or_else(|| crate::value_err("temporal geometry is required"))?
            .extract()?;
        let response_options_value = query_spec.get_item("response_options")?;
        let response_options = parse_response_options(
            response_options_value
                .as_ref()
                .filter(|value| !value.is_none())
                .map(|value| value.cast::<PyDict>())
                .transpose()?,
        )?;
        if response_options.is_some() {
            return Err(crate::refusal(
                antecedent_core::reason_code!("option_not_applicable"),
                "the native unfolded temporal estimator does not use static response smoothing options",
            ));
        }
        let discovery = query_spec
            .get_item("discovery")?
            .filter(|value| !value.is_none())
            .map(|value| value.extract::<DbnRequest>())
            .transpose()?;
        let posterior = query_spec
            .get_item("posterior")?
            .filter(|value| !value.is_none())
            .map(|value| value.cast_into::<crate::bayesian::PyGraphPosterior>())
            .transpose()?;
        let supplied = take_supplied_posterior(posterior, &req.names)?;
        let class_prior_ordered = query_spec
            .get_item("class_prior_ordered")?
            .filter(|value| !value.is_none())
            .map(|value| value.extract::<Vec<f64>>())
            .transpose()?;
        let class_prior_pairs = query_spec
            .get_item("class_prior_pairs")?
            .filter(|value| !value.is_none())
            .map(|value| value.extract::<Vec<(u64, f64)>>())
            .transpose()?;
        let max_completions = query_spec
            .get_item("max_completions")?
            .filter(|value| !value.is_none())
            .map(|value| value.extract::<usize>())
            .transpose()?;
        let dag = graph.as_ref().map(|value| value.extract::<graphs::TemporalDag>()).transpose();
        let (dag, class) = match dag {
            Ok(Some(dag)) => {
                require_named_graph_order(&dag.names, &req.names, "TemporalDag")?;
                (Some(dag.dag), None)
            }
            _ => (None, extract_temporal_class(graph.clone(), &req.names)?),
        };
        let input = req.load_frame(py, columns, frame.as_ref())?;
        req.off_gil(py, move |req| {
            if discovery.is_some() || supplied.is_some() {
                let series = one_series(input)?;
                let query = target.query(series.schema(), geometry, &req.opts)?;
                let ctx = req.opts.ctx(seed, threads);
                let gp = if let Some(gp) = supplied {
                    gp
                } else {
                    let settings = discovery
                        .ok_or_else(|| crate::value_err("posterior discovery is required"))?;
                    dbn_posterior(
                        None,
                        &series,
                        settings.max_lag,
                        settings.force_mcmc,
                        settings.n_chains,
                        settings.n_warmup,
                        settings.n_draws,
                        &ctx,
                    )?
                };
                let builder = with_response_options(
                    antecedent::Study::series(series).graph_posterior(gp).query(query),
                    response_options,
                );
                let builder = crate::temporal_api::apply_class_prior(
                    builder,
                    class_prior_ordered,
                    class_prior_pairs,
                )?;
                let builder = crate::temporal_api::apply_max_completions(builder, max_completions);
                let builder = req.opts.apply(builder);
                return req.finish_with(builder, true, &ctx);
            }
            let data = input.materialize()?;
            let query = target.query(data.schema(), geometry, &req.opts)?;
            let builder = data.builder()?;
            let builder = match (dag, class) {
                (Some(dag), None) => {
                    with_graph(builder, dag, accepted, req.opts.discovery_algorithm())
                }
                (None, Some(class)) => {
                    bind_temporal_class(builder, class, accepted, req.opts.discovery_algorithm())
                }
                _ => {
                    return Err(crate::value_err(
                        "TemporalQuery requires a temporal graph or graph posterior",
                    ));
                }
            };
            let builder = with_response_options(builder.query(query), response_options);
            let builder = crate::temporal_api::apply_class_prior(
                builder,
                class_prior_ordered,
                class_prior_pairs,
            )?;
            let builder = crate::temporal_api::apply_max_completions(builder, max_completions);
            req.finish_applied(builder, true)
        })
    }
}
