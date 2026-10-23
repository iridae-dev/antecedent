//! Query construction and graph binding shared by prepare routes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{EstimatorId, IdentifierId};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchema, ConditionalEffectQuery, Intervention,
    MediationContrast, MediationQuery, ResponseFunctional, ResponseQuery, Value,
};
use antecedent_data::{TableView, TabularData};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::prepared_options::PrepareOptions;
use crate::temporal_api::TemporalClassGraph;
use crate::{graphs, py_err, py_msg, require_named_graph_order};

pub(super) fn parse_identifier(identifier: Option<String>) -> PyResult<Option<IdentifierId>> {
    identifier.map(|id| id.parse::<IdentifierId>().map_err(unknown_strategy_err)).transpose()
}

/// An unknown estimator or identifier id, as a reason-coded `ValueError`.
pub(super) fn unknown_strategy_err(err: antecedent::strategy_table::UnknownStrategy) -> PyErr {
    crate::with_reason_code(
        crate::value_err(err.to_string()),
        antecedent_core::reason_code!("unknown_strategy"),
    )
}

pub(super) fn parse_estimator(estimator: Option<String>) -> PyResult<Option<EstimatorId>> {
    estimator.map(|id| id.parse::<EstimatorId>().map_err(unknown_strategy_err)).transpose()
}

/// Bind a supplied or accepted structure; an accepted one keeps its discovery algorithm.
pub(super) fn with_graph<G>(
    builder: antecedent::StudyBuilder,
    graph: G,
    accepted: bool,
    discovery_algorithm: Option<&str>,
) -> antecedent::StudyBuilder
where
    G: antecedent::IntoGraphInput,
    antecedent::AcceptedGraph: From<G>,
{
    if accepted {
        let accepted = antecedent::AcceptedGraph::from(graph);
        builder.graph(match discovery_algorithm {
            Some(algorithm) => accepted.with_discovery_algorithm(algorithm),
            None => accepted,
        })
    } else {
        builder.graph(graph)
    }
}

/// Configured estimator from `estimator_config`, inheriting the ambient replicate count.
pub(super) fn configured_spec(
    estimator_config: Option<&Bound<'_, PyDict>>,
    estimator: Option<&str>,
    opts: &PrepareOptions,
) -> PyResult<Option<antecedent::EstimatorSpec>> {
    let parsed = crate::estimator_config::parse_estimator_config(
        estimator_config,
        estimator,
        opts.ambient_bootstrap(),
    )?;
    if parsed.rd_running_variable.is_some()
        || parsed.rd_cutoff.is_some()
        || parsed.rd_bandwidth.is_some()
        || parsed.rd_se_kind.is_some()
    {
        return Err(crate::refusal(
            antecedent_core::reason_code!("option_not_applicable"),
            "the rd.sharp design identifies from a running variable on \
             a supplied Dag; this structure route has no running-variable identifier",
        ));
    }
    Ok(parsed.spec)
}

/// Select a configured estimator, or an estimator id, on the builder.
pub(super) fn apply_estimator(
    builder: antecedent::StudyBuilder,
    estimator: Option<String>,
    spec: Option<antecedent::EstimatorSpec>,
) -> PyResult<antecedent::StudyBuilder> {
    Ok(match (spec, parse_estimator(estimator)?) {
        (Some(spec), _) => builder.estimator(spec),
        (None, Some(id)) => builder.estimator(id),
        (None, None) => builder,
    })
}

/// Static average-effect query carrying the declared target population.
pub(super) fn static_ate_query(
    data: &TabularData,
    treatment: &str,
    outcome: &str,
    control_level: f64,
    active_level: f64,
    opts: &PrepareOptions,
) -> PyResult<AverageEffectQuery> {
    let t_id = data.schema().id_of(treatment).map_err(py_err)?;
    let y_id = data.schema().id_of(outcome).map_err(py_err)?;
    let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
    Ok(match opts.target_population.clone() {
        Some(population) => query.with_target_population(population),
        None => query,
    })
}

/// Pulse / Sustained query from a policy name, optional sustained window, and
/// the declared target population.
#[allow(clippy::too_many_arguments)]
pub(super) fn temporal_effect_query(
    policy: &str,
    window: Option<(i32, i32)>,
    t_id: antecedent_core::VariableId,
    y_id: antecedent_core::VariableId,
    treatment_lag: u32,
    horizon_steps: u32,
    control_level: f64,
    active_level: f64,
    max_history_lag: Option<u32>,
    opts: &PrepareOptions,
) -> PyResult<antecedent_core::TemporalEffectQuery> {
    let mut q = crate::temporal_api::temporal_query_from_policy(
        policy,
        t_id,
        y_id,
        treatment_lag,
        horizon_steps,
        active_level,
    )?;
    if let Some((from, until)) = window {
        if policy != "sustained" {
            return Err(crate::value_err("window requires policy='sustained'"));
        }
        q = q.with_policy(antecedent_core::TemporalPolicy::sustained(from, until));
    }
    q.control = Intervention::set(t_id, Value::f64(control_level));
    q.max_history_lag = max_history_lag;
    if let Some(population) = opts.target_population.clone() {
        q.target_population = population;
    }
    Ok(q)
}

/// Temporal mediation query over treatment, one mediator, and outcome, with
/// the declared target population.
#[allow(clippy::too_many_arguments)]
pub(super) fn temporal_mediation_query(
    schema: &CausalSchema,
    treatment: &str,
    mediator: &str,
    outcome: &str,
    contrast: &str,
    control_level: f64,
    active_level: f64,
    horizons: Option<Vec<u32>>,
    opts: &PrepareOptions,
) -> PyResult<MediationQuery> {
    let t_id = schema.id_of(treatment).map_err(py_err)?;
    let m_id = schema.id_of(mediator).map_err(py_err)?;
    let y_id = schema.id_of(outcome).map_err(py_err)?;
    let contrast = match contrast.to_ascii_lowercase().as_str() {
        "total" => MediationContrast::Total,
        "direct" => MediationContrast::Direct,
        "mediated" | "indirect" => MediationContrast::Mediated,
        other => {
            return Err(crate::value_err(format!(
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
    if let Some(population) = opts.target_population.clone() {
        q.target_population = population;
    }
    Ok(q)
}

/// Conditional-effect query with one modifier and the declared population.
#[allow(clippy::too_many_arguments)]
pub(super) fn conditional_query(
    data: &TabularData,
    treatment: &str,
    outcome: &str,
    modifier: &str,
    control_level: f64,
    active_level: f64,
    functional: Option<antecedent_core::OutcomeFunctional>,
    opts: &PrepareOptions,
) -> PyResult<CausalQuery> {
    let w_id = data.schema().id_of(modifier).map_err(py_err)?;
    let mut inner = static_ate_query(data, treatment, outcome, control_level, active_level, opts)?
        .with_effect_modifiers([w_id]);
    if let Some(functional) = functional {
        inner = inner.with_outcome_functional(functional);
    }
    Ok(CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(inner).map_err(|e| crate::value_err(e.to_string()))?,
    ))
}

/// Static InterventionResponse query from encoded intervention specs.
pub(super) fn intervention_response_query(
    data: &TabularData,
    outcome: &str,
    treatments: &[String],
    intervention_kinds: Vec<String>,
    intervention_parameters: Vec<Vec<f64>>,
    outcome_functional: Option<antecedent_core::OutcomeFunctional>,
    opts: &PrepareOptions,
) -> PyResult<CausalQuery> {
    let treatment_ids = crate::response_api::resolve_names(data.schema(), treatments)?;
    let outcome_ids = crate::response_api::resolve_names(data.schema(), &[outcome.to_string()])?;
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
    let mut query = response_query(functional, opts);
    if let (CausalQuery::Response(response), Some(functional)) = (&mut query, outcome_functional) {
        *response = response.clone().with_outcome_functional(functional);
    }
    Ok(query)
}

/// Response query carrying the declared target population.
pub(super) fn response_query(functional: ResponseFunctional, opts: &PrepareOptions) -> CausalQuery {
    let mut query = ResponseQuery::new(functional);
    if let Some(population) = opts.target_population.clone() {
        query.target_population = population;
    }
    CausalQuery::Response(query)
}

/// Continuous-response estimator options from a `response_options` mapping.
///
/// Every [`antecedent_estimate::ContinuousResponseOptions`] field has a key; an
/// omitted key keeps the estimator default and an unknown key is refused, so
/// no option is dropped. `None` leaves the builder without response options.
pub(super) fn parse_response_options(
    options: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<antecedent_estimate::ContinuousResponseOptions>> {
    const KEYS: &[&str] = &[
        "folds",
        "nuisance_basis",
        "nuisance_lambda",
        "bandwidth",
        "minimum_local_ess",
        "confidence_level",
        "simultaneous_replicates",
        "multiplier_seed",
        "export_row_diagnostics",
    ];
    let Some(dict) = options else {
        return Ok(None);
    };
    for (key, _) in dict.iter() {
        let key: String = key.extract()?;
        if !KEYS.contains(&key.as_str()) {
            return Err(crate::value_err(format!(
                "unknown response option {key:?}; expected one of {KEYS:?}"
            )));
        }
    }
    let item = |key: &str| -> PyResult<Option<Bound<'_, PyAny>>> {
        Ok(dict.get_item(key)?.filter(|value| !value.is_none()))
    };
    let mut parsed = antecedent_estimate::ContinuousResponseOptions::default();
    if let Some(value) = item("folds")? {
        parsed.folds = value.extract()?;
    }
    if let Some(value) = item("nuisance_basis")? {
        parsed.nuisance_basis = value.extract()?;
    }
    if let Some(value) = item("nuisance_lambda")? {
        parsed.nuisance_lambda = value.extract()?;
    }
    parsed.bandwidth = item("bandwidth")?.map(|value| value.extract()).transpose()?;
    if let Some(value) = item("minimum_local_ess")? {
        parsed.minimum_local_ess = value.extract()?;
    }
    if let Some(value) = item("confidence_level")? {
        parsed.confidence_level = value.extract()?;
    }
    parsed.simultaneous_replicates =
        item("simultaneous_replicates")?.map(|value| value.extract()).transpose()?;
    if let Some(value) = item("multiplier_seed")? {
        parsed.multiplier_seed = value.extract()?;
    }
    if let Some(value) = item("export_row_diagnostics")? {
        parsed.export_row_diagnostics = value.extract()?;
    }
    Ok(Some(parsed))
}

/// Set parsed response options on the builder, when any were supplied.
pub(super) fn with_response_options(
    builder: antecedent::StudyBuilder,
    options: Option<antecedent_estimate::ContinuousResponseOptions>,
) -> antecedent::StudyBuilder {
    match options {
        Some(options) => builder.response_options(options),
        None => builder,
    }
}

/// A supplied static class structure (PAG / CPDAG) or bidirected ADMG.
pub(super) enum StaticClassGraph {
    Pag(antecedent_graph::Pag),
    Cpdag(antecedent_graph::Cpdag),
    Admg(antecedent_graph::Admg),
}

impl StaticClassGraph {
    pub(super) fn extract(graph: &Bound<'_, PyAny>, names: &[String]) -> PyResult<Self> {
        if let Ok(g) = graph.extract::<graphs::Pag>() {
            require_named_graph_order(&g.names, names, "Pag")?;
            Ok(Self::Pag(g.pag))
        } else if let Ok(g) = graph.extract::<graphs::Cpdag>() {
            require_named_graph_order(&g.names, names, "Cpdag")?;
            Ok(Self::Cpdag(g.cpdag))
        } else if let Ok(g) = graph.extract::<graphs::Admg>() {
            require_named_graph_order(&g.names, names, "Admg")?;
            Ok(Self::Admg(g.aligned_to_names(names)?))
        } else {
            Err(crate::value_err("graph must be a Pag, Cpdag, or Admg"))
        }
    }

    pub(super) fn bind(
        self,
        builder: antecedent::StudyBuilder,
        accepted: bool,
        discovery_algorithm: Option<&str>,
    ) -> antecedent::StudyBuilder {
        match self {
            Self::Pag(g) => with_graph(builder, g, accepted, discovery_algorithm),
            Self::Cpdag(g) => with_graph(builder, g, accepted, discovery_algorithm),
            Self::Admg(g) => with_graph(builder, g, accepted, discovery_algorithm),
        }
    }
}

/// Bind a TemporalCpdag / TemporalPag class; an accepted one keeps its discovery algorithm.
pub(super) fn bind_temporal_class(
    builder: antecedent::StudyBuilder,
    graph: TemporalClassGraph,
    accepted: bool,
    discovery_algorithm: Option<&str>,
) -> antecedent::StudyBuilder {
    match graph {
        TemporalClassGraph::Cpdag(g) => with_graph(builder, g, accepted, discovery_algorithm),
        TemporalClassGraph::Pag(g) => with_graph(builder, g, accepted, discovery_algorithm),
        TemporalClassGraph::Admg(g) => with_graph(builder, g, accepted, discovery_algorithm),
    }
}

pub(super) fn static_kind_query(
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
                _ => return Err(crate::value_err("unknown static mediation contrast")),
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
        "nested_counterfactual" => {
            if mediators.len() != 1 {
                return Err(crate::value_err(
                    "nested_counterfactual requires exactly one mediator",
                ));
            }
            let mediator = schema.id_of(&mediators[0]).map_err(py_err)?;
            let q = antecedent_core::NestedCounterfactualQuery::with_levels(
                t, mediator, y, control, active,
            )
            .map_err(|error| crate::value_err(error.to_string()))?;
            Ok(CausalQuery::NestedCounterfactual(q))
        }
        _ => Err(crate::value_err("unsupported static staged kind")),
    }
}

pub(super) fn extract_temporal_class(
    graph: Option<Bound<'_, PyAny>>,
    names: &[String],
) -> PyResult<Option<TemporalClassGraph>> {
    let Some(graph) = graph else {
        return Ok(None);
    };
    if let Ok(g) = graph.extract::<graphs::TemporalCpdag>() {
        require_named_graph_order(&g.names, names, "TemporalCpdag")?;
        Ok(Some(TemporalClassGraph::Cpdag(g.cpdag)))
    } else if let Ok(g) = graph.extract::<graphs::TemporalPag>() {
        require_named_graph_order(&g.names, names, "TemporalPag")?;
        Ok(Some(TemporalClassGraph::Pag(g.pag)))
    } else if let Ok(g) = graph.extract::<graphs::TemporalAdmg>() {
        require_named_graph_order(&g.names, names, "TemporalAdmg")?;
        Ok(Some(TemporalClassGraph::Admg(g.graph)))
    } else {
        Err(crate::value_err("class_graph requires TemporalCpdag, TemporalPag or TemporalAdmg"))
    }
}
