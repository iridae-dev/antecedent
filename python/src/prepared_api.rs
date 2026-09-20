//! Compile-once / re-estimate-many [`PreparedStudy`] Python OO surface.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::discovery::{
    BayesianDiscoverParams, GraphMcmcSchedule, discover_dbn_posterior, discover_exact_dag_posterior,
};
use antecedent::{CausalContract, EstimatorId, IdentifierId, PreparedStudy, Study};
use antecedent_core::{
    AnomalyAttributionQuery, AverageEffectQuery, CausalQuery, CausalSchema, ChangeAttributionQuery,
    ConditionalEffectQuery, ContinuousDomain, GridSpec, Intervention,
    InterventionalDistributionQuery, MediationContrast, MediationQuery, PathSpecificEffectQuery,
    PopulationSelector, ResponseFunctional, ResponseQuery, TemporalResponseSpec, Value,
};
use antecedent_data::{TableView, TabularData, TimeSeriesData};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

use crate::prepared_options::PrepareOptions;
use crate::response_api::{
    ResponseAnalysisResult, attach_study_response_meta, build_functional, response_result,
};
use crate::temporal_api::TemporalClassGraph;
use crate::{
    AteAnalysisResult, ate_result_from_analysis, dag_from_named_edges, detach_catch, graphs,
    py_err, py_execution_context_ext, py_msg, require_named_graph_order, series_from_tabular,
    suite_from_refute, tabular_from_arrow_c_objs, tabular_from_numpy, tabular_from_py_columns,
    temporal_dag_from_schema_edges,
};

/// Execution controls for one estimate or refresh click.
///
/// Cancellation and progress ride on the execution context. A stage sink is
/// set on a private clone of the prepared study for this click only, so it is
/// never retained by the handle, a snapshot, or an exported claim.
struct ClickControls {
    cancel: Option<antecedent_core::CancellationToken>,
    progress: Option<Arc<dyn antecedent_core::ProgressSink>>,
    stage: Option<Arc<dyn antecedent::StageResultSink>>,
}

impl ClickControls {
    fn parse(
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<&Bound<'_, PyAny>>,
        on_stage: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            cancel: cancel.map(|token| token.inner),
            progress: crate::callbacks::progress_sink_from_py(on_progress)?,
            stage: crate::callbacks::stage_sink_from_py(on_stage)?,
        })
    }

    fn ctx(&self, seed: u64, threads: Option<u32>) -> antecedent_core::ExecutionContext {
        py_execution_context_ext(
            seed,
            crate::resolve_user_threads(threads),
            self.cancel.clone(),
            self.progress.clone(),
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        )
    }

    /// The study this click executes: the handle itself, or a stage-streaming clone.
    fn study(&self, inner: &Arc<PreparedStudy>) -> PyResult<Arc<PreparedStudy>> {
        match &self.stage {
            None => Ok(Arc::clone(inner)),
            Some(sink) => {
                refuse_unstaged(inner)?;
                let mut staged = (**inner).clone();
                staged.set_stage_sink(Some(Arc::clone(sink)));
                Ok(Arc::new(staged))
            }
        }
    }

    /// Owned copy for a click that replaces retained data.
    fn owned_study(&self, inner: &Arc<PreparedStudy>) -> PyResult<PreparedStudy> {
        let mut owned = (**inner).clone();
        if let Some(sink) = &self.stage {
            refuse_unstaged(inner)?;
            owned.set_stage_sink(Some(Arc::clone(sink)));
        }
        Ok(owned)
    }
}

fn map_ate(names: &[String], result: &antecedent::StudyResult) -> PyResult<AteAnalysisResult> {
    ate_result_from_analysis(names, result.clone(), false)
}

fn refuse_unstaged(inner: &PreparedStudy) -> PyResult<()> {
    if inner.streams_stages() {
        return Ok(());
    }
    Err(crate::refusal(
        antecedent_core::reason_code!("stage_stream_unavailable"),
        "on_stage streams identify, estimate_point, \
         uncertainty and validate for one identification and one scalar effect whose point is \
         fitted before its uncertainty; this route fits its point and uncertainty together or \
         mixes several identifications, so it has no such stages (use on_progress)",
    ))
}

/// Strip a click-only stage sink before a study is retained.
fn unstaged(mut study: PreparedStudy) -> PreparedStudy {
    study.set_stage_sink(None);
    study
}

fn parse_identifier(identifier: Option<String>) -> PyResult<Option<IdentifierId>> {
    identifier.map(|id| id.parse::<IdentifierId>().map_err(unknown_strategy_err)).transpose()
}

/// An unknown estimator or identifier id, as a reason-coded `ValueError`.
fn unknown_strategy_err(err: antecedent::strategy_table::UnknownStrategy) -> PyErr {
    crate::with_reason_code(
        PyValueError::new_err(err.to_string()),
        antecedent_core::reason_code!("unknown_strategy"),
    )
}

fn parse_estimator(estimator: Option<String>) -> PyResult<Option<EstimatorId>> {
    estimator.map(|id| id.parse::<EstimatorId>().map_err(unknown_strategy_err)).transpose()
}

/// Bind a supplied or accepted structure; an accepted one keeps its discovery algorithm.
fn with_graph<G>(
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
fn configured_spec(
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
fn apply_estimator(
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
fn static_ate_query(
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
fn temporal_effect_query(
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
            return Err(PyValueError::new_err("window requires policy='sustained'"));
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
fn temporal_mediation_query(
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
    if let Some(population) = opts.target_population.clone() {
        q.target_population = population;
    }
    Ok(q)
}

/// Conditional-effect query with one modifier and the declared population.
#[allow(clippy::too_many_arguments)]
fn conditional_query(
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
        ConditionalEffectQuery::try_new(inner).map_err(|e| PyValueError::new_err(e.to_string()))?,
    ))
}

/// Static InterventionResponse query from encoded intervention specs.
fn intervention_response_query(
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
fn response_query(functional: ResponseFunctional, opts: &PrepareOptions) -> CausalQuery {
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
fn parse_response_options(
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
            return Err(PyValueError::new_err(format!(
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
fn with_response_options(
    builder: antecedent::StudyBuilder,
    options: Option<antecedent_estimate::ContinuousResponseOptions>,
) -> antecedent::StudyBuilder {
    match options {
        Some(options) => builder.response_options(options),
        None => builder,
    }
}

/// A supplied static class structure (PAG / CPDAG) or bidirected ADMG.
enum StaticClassGraph {
    Pag(antecedent_graph::Pag),
    Cpdag(antecedent_graph::Cpdag),
    Admg(antecedent_graph::Admg),
}

impl StaticClassGraph {
    fn extract(graph: &Bound<'_, PyAny>, names: &[String]) -> PyResult<Self> {
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
            Err(PyValueError::new_err("graph must be a Pag, Cpdag, or Admg"))
        }
    }

    fn bind(
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
fn bind_temporal_class(
    builder: antecedent::StudyBuilder,
    graph: TemporalClassGraph,
    accepted: bool,
    discovery_algorithm: Option<&str>,
) -> antecedent::StudyBuilder {
    match graph {
        TemporalClassGraph::Cpdag(g) => with_graph(builder, g, accepted, discovery_algorithm),
        TemporalClassGraph::Pag(g) => with_graph(builder, g, accepted, discovery_algorithm),
    }
}

/// Data partitions of a panel, multi-environment, or event frame, ingested under the GIL.
enum FrameInput {
    Series(TabularData),
    Panel { units: Vec<TabularData>, unit_ids: Vec<u32> },
    MultiEnv(Vec<TabularData>),
    Events { data: TabularData, event_times_ns: Vec<i64>, align_interval_ns: u64 },
}

/// Materialized frame data, built without the GIL.
enum FrameData {
    Series(TimeSeriesData),
    Panel(antecedent_data::PanelData),
    MultiEnv(antecedent_data::MultiEnvironmentData),
    Events(antecedent_data::EventData, u64),
}

impl FrameInput {
    fn ingest(
        py: Python<'_>,
        names: &[String],
        columns: Vec<Bound<'_, PyAny>>,
        frame: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let Some(frame) = frame else {
            return Ok(Self::Series(tabular_from_py_columns(py, names.to_vec(), columns)?.0));
        };
        let kind: String = frame
            .get_item("kind")?
            .ok_or_else(|| PyValueError::new_err("frame requires kind"))?
            .extract()?;
        let partitions = |key: &str| -> PyResult<Vec<TabularData>> {
            let parts: Vec<Vec<Bound<'_, PyAny>>> = frame
                .get_item(key)?
                .ok_or_else(|| PyValueError::new_err(format!("{kind} frame requires {key}")))?
                .extract()?;
            parts
                .into_iter()
                .map(|cols| Ok(tabular_from_py_columns(py, names.to_vec(), cols)?.0))
                .collect()
        };
        match kind.as_str() {
            "panel" => {
                let units = partitions("unit_columns")?;
                let unit_ids: Vec<u32> = frame
                    .get_item("unit_ids")?
                    .ok_or_else(|| PyValueError::new_err("panel frame requires unit_ids"))?
                    .extract()?;
                if units.is_empty() || units.len() != unit_ids.len() {
                    return Err(PyValueError::new_err(
                        "panel frame needs one unit id per non-empty unit",
                    ));
                }
                Ok(Self::Panel { units, unit_ids })
            }
            "multi_env" => {
                let envs = partitions("env_columns")?;
                if envs.is_empty() {
                    return Err(PyValueError::new_err("multi_env frame needs ≥1 environment"));
                }
                Ok(Self::MultiEnv(envs))
            }
            "events" => {
                let event_times_ns: Vec<i64> = frame
                    .get_item("event_times_ns")?
                    .ok_or_else(|| PyValueError::new_err("events frame requires event_times_ns"))?
                    .extract()?;
                let align_interval_ns: u64 = frame
                    .get_item("align_interval_ns")?
                    .ok_or_else(|| {
                        PyValueError::new_err("events frame requires align_interval_ns")
                    })?
                    .extract()?;
                let (data, _) = tabular_from_py_columns(py, names.to_vec(), columns)?;
                Ok(Self::Events { data, event_times_ns, align_interval_ns })
            }
            other => Err(PyValueError::new_err(format!(
                "unknown frame kind {other:?}; use panel|multi_env|events"
            ))),
        }
    }

    fn materialize(self) -> PyResult<FrameData> {
        match self {
            Self::Series(data) => Ok(FrameData::Series(series_from_tabular(data)?)),
            Self::Panel { units, unit_ids } => {
                let units = units
                    .into_iter()
                    .zip(unit_ids)
                    .map(|(data, unit_id)| {
                        Ok(antecedent_data::PanelUnit {
                            unit_id,
                            series: series_from_tabular(data)?,
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                Ok(FrameData::Panel(
                    antecedent_data::PanelData::try_new(Arc::from(units)).map_err(py_err)?,
                ))
            }
            Self::MultiEnv(envs) => {
                let series =
                    envs.into_iter().map(series_from_tabular).collect::<PyResult<Vec<_>>>()?;
                Ok(FrameData::MultiEnv(
                    antecedent_data::MultiEnvironmentData::try_new(Arc::from(series))
                        .map_err(py_err)?,
                ))
            }
            Self::Events { data, event_times_ns, align_interval_ns } => {
                let events = antecedent_data::EventData::try_new(
                    data.storage().clone(),
                    Arc::from(event_times_ns),
                )
                .map_err(py_err)?;
                Ok(FrameData::Events(events, align_interval_ns))
            }
        }
    }
}

impl FrameData {
    fn schema(&self) -> &CausalSchema {
        match self {
            Self::Series(data) => data.schema(),
            Self::Panel(data) => data.schema(),
            Self::MultiEnv(data) => data.schema(),
            Self::Events(data, _) => data.schema(),
        }
    }

    fn builder(self) -> PyResult<antecedent::StudyBuilder> {
        Ok(match self {
            Self::Series(data) => Study::series(data),
            Self::Panel(data) => Study::panel(data),
            Self::MultiEnv(data) => Study::series_multi(data),
            Self::Events(data, align) => Study::events(&data, align).map_err(py_err)?,
        })
    }

    /// Bind this frame to `study` (same modality, schema and regularity) and execute.
    fn bind_and_run(
        self,
        study: &mut PreparedStudy,
        ctx: &antecedent_core::ExecutionContext,
    ) -> PyResult<antecedent::StudyResult> {
        let result = match self {
            Self::Series(data) => study.refresh_series(data, ctx),
            Self::Panel(data) => study.refresh_panel(data, ctx),
            Self::MultiEnv(data) => study.refresh_multi_env(data, ctx),
            Self::Events(data, align) => {
                let aligned = data
                    .align_to_grid(align)
                    .map_err(|e| py_msg(format!("event align_to_grid: {e}")))?;
                study.refresh_series(aligned, ctx)
            }
        };
        result.map_err(py_err)
    }
}

fn contract_to_map(
    contract: &CausalContract,
    schema: &CausalSchema,
    query: &CausalQuery,
    registry: Option<&antecedent_core::PopulationRegistry>,
) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    out.insert("target".into(), contract.identities.target.to_hex());
    out.insert("identification".into(), contract.identities.identification.to_hex());
    if let Some(product) = contract.identities.identification_product {
        out.insert("identification_product".into(), product.to_hex());
    }
    if let Some(program) = contract.identities.program {
        out.insert("program".into(), program.to_hex());
    }
    out.insert("inference_binding".into(), contract.identities.inference_binding.to_hex());
    out.insert("observation".into(), contract.identities.observation.to_hex());
    out.insert("data_snapshot".into(), contract.identities.data_snapshot.to_hex());
    out.insert("graph_class".into(), contract.graph_class.as_str().to_string());
    out.insert("structure_source".into(), contract.structure_source.as_str().to_string());
    out.insert("accepted_version".into(), contract.accepted_version.to_string());
    if let Some(algorithm) = &contract.discovery_algorithm {
        out.insert("discovery_algorithm".into(), algorithm.to_string());
    }
    out.insert(
        "accepted_variable_binding".into(),
        if contract.accepted_variable_names.is_some() { "explicit" } else { "unbound" }.into(),
    );
    if let Some(status) = contract.support_status {
        out.insert("matrix_status".into(), status.as_str().to_string());
    }
    if let Some(support) = contract.reasoning.support.as_ref() {
        if let Some(coordinate) = &support.matrix_coordinate {
            out.insert("matrix_coordinate".into(), coordinate.to_string());
        }
        out.insert(
            "empirical_support".into(),
            support.empirical.label(std::string::ToString::to_string),
        );
    }
    out.insert(
        "assumptions".into(),
        contract.reasoning.assumptions.label(|slot| {
            slot.obligations.iter().map(|o| o.id.to_string()).collect::<Vec<_>>().join(",")
        }),
    );
    out.insert(
        "identification_status".into(),
        contract.reasoning.identification.label(|slot| slot.status.as_str().to_string()),
    );
    if let Some(slot) = contract.reasoning.identification.as_ref() {
        out.insert("identified_mass".into(), slot.identified_mass.to_string());
        out.insert("unidentified_mass".into(), slot.unidentified_mass.to_string());
        out.insert("unevaluable_mass".into(), slot.unevaluable_mass.to_string());
        out.insert("incomplete_search_mass".into(), slot.incomplete_search_mass.to_string());
        out.insert("full_mass_scope".into(), slot.full_mass_scope.to_string());
        out.insert("search_capped".into(), slot.search_capped.to_string());
    }
    if let Some(slot) = contract.reasoning.uncertainty.as_ref() {
        let components: Vec<_> = slot.components.iter().map(|c| serde_json::json!({
            "source": c.source.as_str(), "target": c.target.to_string(), "omitted": c.omitted
        })).collect();
        out.insert("uncertainty_components".into(), serde_json::json!(components).to_string());
    }
    if let Some(slot) = contract.reasoning.assumptions.as_ref() {
        let obligations: Vec<_> = slot.obligations.iter().map(|o| serde_json::json!({
            "id": o.id.to_string(), "scope": o.scope.as_str(), "kind": o.kind.as_str(), "status": o.status.as_str()
        })).collect();
        out.insert("assumption_obligations".into(), serde_json::json!(obligations).to_string());
    }
    out.insert("uncertainty".into(), contract.reasoning.uncertainty.label(|_| "available".into()));
    out.insert(
        "variable_names".into(),
        schema
            .variables()
            .iter()
            .map(|variable| variable.name.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    if let Ok(wire) = antecedent_io::causal_query_to_wire_with_registry(query, registry) {
        for (key, value) in antecedent_io::executed_functional_labels(&wire) {
            out.insert(key, value);
        }
    }
    out
}

/// Flat inspect keys for the score-reuse and target-weights identities.
fn insert_reuse_identities(
    report: &mut std::collections::HashMap<String, String>,
    score_reuse: Option<antecedent_core::SemanticDigest>,
    target_weights: Option<antecedent_core::SemanticDigest>,
) {
    if let Some(digest) = score_reuse {
        report.insert("score_reuse".into(), digest.to_hex());
    }
    if let Some(digest) = target_weights {
        report.insert("target_weights".into(), digest.to_hex());
    }
}

fn require_prepared_names(expected: &[String], names: &[String], op: &str) -> PyResult<()> {
    if names != expected {
        return Err(crate::with_reason_code(
            PyValueError::new_err(format!(
                "prepared {op} requires the same column names (order) as prepare"
            )),
            antecedent_core::reason_code!("schema_mismatch"),
        ));
    }
    Ok(())
}

fn take_supplied_posterior(
    posterior: Option<Bound<'_, crate::bayesian::PyGraphPosterior>>,
    names: &[String],
) -> PyResult<Option<antecedent::discovery::GraphPosterior>> {
    posterior
        .map(|bound| {
            let posterior = bound.borrow();
            posterior.require_bound_to(names)?;
            posterior.to_rust()
        })
        .transpose()
}

fn finished_prepared(
    prepared: PreparedStudy,
    names: Vec<String>,
    series: bool,
) -> PyPreparedAnalysis {
    PyPreparedAnalysis {
        inner: Arc::new(prepared),
        names,
        last: None,
        last_study: None,
        last_seed: 1,
        last_threads: 1,
        borrowed_bytes: None,
        series,
    }
}

fn exact_dag_posterior(
    supplied: Option<antecedent::discovery::GraphPosterior>,
    data: &TabularData,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<antecedent::discovery::GraphPosterior> {
    if let Some(gp) = supplied {
        return Ok(gp);
    }
    let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
    discover_exact_dag_posterior(data, &vars, &BayesianDiscoverParams::default(), ctx)
        .map_err(py_err)
}

fn dbn_posterior(
    supplied: Option<antecedent::discovery::GraphPosterior>,
    series: &TimeSeriesData,
    max_lag: u32,
    force_mcmc: bool,
    n_chains: u32,
    n_warmup: u32,
    mcmc_draws: u32,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<antecedent::discovery::GraphPosterior> {
    if let Some(gp) = supplied {
        return Ok(gp);
    }
    let vars: Vec<_> = series.schema().variables().iter().map(|v| v.id).collect();
    let schedule = GraphMcmcSchedule { n_chains, n_warmup, n_draws: mcmc_draws, thin: 1 };
    discover_dbn_posterior(
        series,
        &vars,
        &BayesianDiscoverParams::default(),
        max_lag,
        force_mcmc,
        &schedule,
        ctx,
    )
    .map_err(py_err)
}

#[allow(clippy::too_many_arguments)]
fn finish_static_graph_posterior(
    data: TabularData,
    names: Vec<String>,
    gp: antecedent::discovery::GraphPosterior,
    query: CausalQuery,
    spec: Option<antecedent::EstimatorSpec>,
    response_options: Option<antecedent_estimate::ContinuousResponseOptions>,
    mut opts: PrepareOptions,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<PyPreparedAnalysis> {
    opts.refuse_prior_transfer("a graph-posterior mixture")?;
    if spec.is_some() && opts.inference.is_bayesian() {
        return Err(crate::refusal(
            antecedent_core::reason_code!("option_not_applicable"),
            "a Bayesian graph-posterior mixture fits \
             Bayesian g-computation per atom; a configured Frequentist estimator does not apply",
        ));
    }
    let builder = with_response_options(
        Study::tabular(data).graph_posterior(gp).query(query),
        response_options,
    );
    let builder = apply_estimator(opts.apply_budget_for(builder, spec.is_some()), None, spec)?;
    let analysis = opts.apply_inference(builder)?.build().map_err(py_err)?;
    Ok(finished_prepared(analysis.prepare(ctx).map_err(py_err)?, names, false))
}

enum SeriesGraphQuery {
    Temporal(antecedent_core::TemporalEffectQuery),
    Mediation(MediationQuery),
    Response(ResponseQuery),
}

fn finish_series_graph_posterior(
    series: TimeSeriesData,
    names: Vec<String>,
    gp: antecedent::discovery::GraphPosterior,
    query: SeriesGraphQuery,
    mut opts: PrepareOptions,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<PyPreparedAnalysis> {
    opts.refuse_prior_transfer("a DBN graph-posterior mixture")?;
    let mut builder = Study::series(series).graph_posterior(gp);
    builder = match query {
        SeriesGraphQuery::Temporal(q) => builder.temporal_query(q),
        SeriesGraphQuery::Mediation(q) => builder.query(CausalQuery::Mediation(q)),
        SeriesGraphQuery::Response(q) => builder.query(CausalQuery::Response(q)),
    };
    let analysis = opts.apply_inference(opts.apply(builder))?.build().map_err(py_err)?;
    Ok(finished_prepared(analysis.prepare(ctx).map_err(py_err)?, names, true))
}

/// Durable prepare-once / estimate-many handle for static ATE on a supplied DAG.
#[pyclass(name = "PreparedAnalysis")]
pub struct PyPreparedAnalysis {
    /// Arc so per-click estimate/refute detach with a refcount bump, not a
    /// deep `PreparedStudy` clone; `refresh` clones-on-write to swap data.
    inner: Arc<PreparedStudy>,
    names: Vec<String>,
    /// Last estimate result retained for second-click refute.
    last: Option<Arc<antecedent::StudyResult>>,
    last_study: Option<Arc<PreparedStudy>>,
    last_seed: u64,
    last_threads: u32,
    borrowed_bytes: Option<u64>,
    /// When true, estimate/refresh clicks use series data (`estimate_series`).
    series: bool,
}

impl PyPreparedAnalysis {
    pub(crate) fn from_study(prepared: PreparedStudy, names: Vec<String>) -> Self {
        Self {
            inner: Arc::new(prepared),
            names,
            last: None,
            last_study: None,
            last_seed: 1,
            last_threads: 1,
            borrowed_bytes: None,
            series: false,
        }
    }

    /// Execute one click on new data, optionally replacing the retained data.
    ///
    /// The executed study (with the click's data bound) becomes the snapshot
    /// source for exports and second-click refutes; a refresh also replaces the
    /// handle's retained study. A click-only stage sink is stripped first.
    fn click<T: Send>(
        &mut self,
        py: Python<'_>,
        data: FrameInput,
        refresh: bool,
        seed: u64,
        threads: Option<u32>,
        controls: ClickControls,
        map: fn(&[String], &antecedent::StudyResult) -> PyResult<T>,
    ) -> PyResult<T> {
        let mut study = controls.owned_study(&self.inner)?;
        let out_names = self.names.clone();
        let series = self.series;
        let (mapped, result, bound) = detach_catch(py, move || {
            let ctx = controls.ctx(seed, threads);
            let result = match data {
                // Both estimate and refresh bind the click's data to the executed
                // copy, so its snapshot re-executes on exactly that data.
                FrameInput::Series(data) if !series => study.refresh(data, &ctx).map_err(py_err)?,
                frame => frame.materialize()?.bind_and_run(&mut study, &ctx)?,
            };
            let mapped = map(&out_names, &result)?;
            Ok((mapped, result, Arc::new(unstaged(study))))
        })?;
        if refresh {
            self.inner = Arc::clone(&bound);
            self.borrowed_bytes = None;
        }
        self.last_study = Some(bound);
        self.last = Some(Arc::new(result));
        self.last_seed = seed;
        self.last_threads = crate::resolve_user_threads(threads);
        Ok(mapped)
    }

    /// Execute the retained data.
    fn click_bound<T: Send>(
        &mut self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        controls: ClickControls,
        map: fn(&[String], &antecedent::StudyResult) -> PyResult<T>,
    ) -> PyResult<T> {
        let borrowed_bytes = self.borrowed_bytes;
        let study = controls.study(&self.inner)?;
        let names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let ctx = controls.ctx(seed, threads);
            let mut result = study.estimate_retained(&ctx).map_err(py_err)?;
            result.performance.bytes_borrowed = borrowed_bytes;
            let mapped = map(&names, &result)?;
            Ok((mapped, result))
        })?;
        self.last_study = Some(Arc::clone(&self.inner));
        self.last = Some(Arc::new(result));
        self.last_seed = seed;
        self.last_threads = crate::resolve_user_threads(threads);
        Ok(mapped)
    }

    /// A design-defined population cannot be reweighted.
    ///
    /// A transport study's target population is the selection diagram's
    /// target rows under known selection probabilities, and an interference
    /// contrast is a finite-population design estimand; neither has a score
    /// table whose rows a weight vector could redeclare.
    fn refuse_design_retarget(&self) -> PyResult<()> {
        let route = match self.inner.query() {
            CausalQuery::Transport(_) => "a TransportQuery study",
            CausalQuery::Interference(_) => "an InterferenceQuery study",
            CausalQuery::AnomalyAttribution(_) => "an AnomalyAttribution study",
            CausalQuery::ChangeAttribution(_) => "a ChangeAttribution study",
            _ => return Ok(()),
        };
        Err(crate::refusal(
            antecedent_core::reason_code!("population_not_estimable"),
            format!(
                "{route} estimates the population its design defines; a row-weight retarget has \
                 no score table to reweight and would redeclare the design's own target"
            ),
        ))
    }

    fn finish_ate_refute(
        &mut self,
        py: Python<'_>,
        data: TabularData,
        suite: Bound<'_, PyAny>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<AteAnalysisResult> {
        let prior = self
            .last
            .clone()
            .ok_or_else(|| PyValueError::new_err("call estimate/refresh before refute"))?;
        let refute_suite = suite_from_refute(Some(&suite))?;
        let cancel_token = cancel.map(|c| c.inner);
        let bound = self.last_study.clone().unwrap_or_else(|| Arc::clone(&self.inner));
        let inner = Arc::clone(&bound);
        let out_names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                crate::resolve_user_threads(threads),
                cancel_token,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let result = inner.refute(&prior, &data, refute_suite, &ctx).map_err(py_err)?;
            let mapped = ate_result_from_analysis(&out_names, result.clone(), false)?;
            Ok((mapped, result))
        })?;
        self.last_study = Some(bound);
        self.last = Some(Arc::new(result));
        self.last_seed = seed;
        self.last_threads = crate::resolve_user_threads(threads);
        Ok(mapped)
    }
}

#[pymethods]
impl PyPreparedAnalysis {
    /// Inspection of the frozen execution, using the executed reasoning slots.
    fn execution_contract(&self) -> PyResult<std::collections::HashMap<String, String>> {
        let result = self
            .last
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("estimate before inspecting an execution"))?;
        let inner = self.last_study.as_ref().unwrap_or(&self.inner);
        let ctx = py_execution_context_ext(
            self.last_seed,
            self.last_threads,
            None,
            None,
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        );
        // The contract the execution ran under, with the identities the export
        // carries: a retarget moves the target population and its weighting
        // identity, and a second-click refute moves the validation suite.
        let (mut contract, claim) = inner.execution_contract(result, &ctx).map_err(py_err)?;
        contract.reasoning = claim.reasoning;
        let mut report =
            contract_to_map(&contract, inner.schema(), inner.query(), inner.population_registry());
        insert_reuse_identities(
            &mut report,
            inner.score_reuse_identity().map_err(py_err)?,
            result.row_weights.as_ref().map(|binding| binding.target_weights),
        );
        report.insert("claim_id".into(), claim.claim_id.to_hex());
        report.insert("calibration_status".into(), claim.calibration.status.to_string());
        if let Some(record_id) = &claim.calibration.record_id {
            report.insert("calibration_record_id".into(), record_id.to_string());
        }
        if let Some(reason) = &claim.calibration.reason {
            report.insert("calibration_reason".into(), reason.to_string());
        }
        if let Some(scope_n) = claim.calibration.scope_n {
            report.insert("calibration_scope_n".into(), scope_n.to_string());
        }
        if let Some(dep) = &claim.calibration.scope_dependence {
            report.insert("calibration_scope_dependence".into(), dep.to_string());
        }
        if let Some(sha) = &claim.calibration.calibration_sha {
            report.insert("calibration_sha".into(), sha.to_string());
        }
        Ok(report)
    }

    /// Frozen reference to the last execution, sharing immutable Rust resources.
    fn snapshot(&self) -> PyResult<Self> {
        let last = self
            .last
            .clone()
            .ok_or_else(|| PyValueError::new_err("estimate before taking a result snapshot"))?;
        Ok(Self {
            inner: self.last_study.clone().unwrap_or_else(|| Arc::clone(&self.inner)),
            names: self.names.clone(),
            last: Some(last),
            last_study: self.last_study.clone(),
            last_seed: self.last_seed,
            last_threads: self.last_threads,
            borrowed_bytes: self.borrowed_bytes,
            series: self.series,
        })
    }

    /// Execute the retained data without crossing cell buffers through Python again.
    ///
    /// `cancel` / `on_progress` apply to this click; `on_stage` streams its
    /// stages where the route has them and is refused where it does not.
    #[pyo3(signature = (*, seed=1, threads=None, cancel=None, on_progress=None, on_stage=None))]
    fn estimate_bound(
        &mut self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<AteAnalysisResult> {
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        self.click_bound(py, seed, threads, controls, map_ate)
    }

    #[pyo3(signature = (*, seed=1, threads=None, cancel=None, on_progress=None, on_stage=None))]
    fn estimate_response_bound(
        &mut self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<ResponseAnalysisResult> {
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        self.click_bound(py, seed, threads, controls, response_from_study)
    }

    /// Execute on a panel, multi-environment, or event frame of the prepared schema.
    ///
    /// `refresh=True` also replaces the retained data. Returns the ATE-shaped
    /// result, or the response result when `response=True`.
    #[pyo3(signature = (
        names, columns, frame, *, response=false, refresh=false, seed=1, threads=None,
        cancel=None, on_progress=None, on_stage=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn estimate_frame(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        frame: Bound<'_, PyDict>,
        response: bool,
        refresh: bool,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        require_prepared_names(&self.names, &names, "estimate")?;
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        let data = FrameInput::ingest(py, &names, columns, Some(&frame))?;
        if response {
            let out =
                self.click(py, data, refresh, seed, threads, controls, response_from_study)?;
            Ok(Py::new(py, out)?.into_any())
        } else {
            let out = self.click(py, data, refresh, seed, threads, controls, map_ate)?;
            Ok(Py::new(py, out)?.into_any())
        }
    }

    /// Whether estimates of this handle stream progressive stages (`on_stage`).
    fn streams_stages(&self) -> bool {
        self.inner.streams_stages()
    }

    /// Re-bind caller custom validators by their attested names.
    ///
    /// The validators are in-process callbacks: a loaded claim carries their
    /// attested results, never the callables, so a rebound study must present
    /// exactly the names its claims attest.
    fn rebind_validators(&mut self, validators: Bound<'_, PyAny>) -> PyResult<()> {
        let parsed = crate::callbacks::parse_validators(Some(&validators))?;
        let mut study = (*self.inner).clone();
        study.rebind_custom_validators(parsed).map_err(py_err)?;
        self.inner = Arc::new(study);
        Ok(())
    }

    /// Attested custom-validator names frozen at prepare, in order.
    fn validator_names(&self) -> Vec<String> {
        self.inner.custom_validator_names().iter().map(|name| (*name).to_string()).collect()
    }

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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        let parsed_estimator = crate::estimator_config::parse_estimator_config(
            estimator_config,
            estimator.as_deref(),
            opts.ambient_bootstrap(),
        )?;
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let (data, borrowed_bytes) = tabular_from_py_columns(py, names.clone(), columns)?;

        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let mut query =
                AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
            if let Some(pop) = opts.target_population.clone() {
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
            let mut builder = opts.apply_budget_for(
                with_graph(Study::tabular(data), dag, accepted, opts.discovery_algorithm())
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
            if let Some(id) = identifier {
                builder = builder.identifier(id);
            }
            if let Some(spec) = parsed_estimator.spec {
                builder = builder.estimator(spec);
            } else if let Some(est) = estimator_id {
                builder = builder.estimator(est);
            }
            let analysis = opts.apply_inference(builder)?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            let mut out = finished_prepared(prepared, names, false);
            out.borrowed_bytes = borrowed_bytes;
            Ok(out)
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        let spec = configured_spec(estimator_config, estimator.as_deref(), &opts)?;
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let within = crate::parse_within_tier(Some(within_tier.as_str()))?;
            let named: Vec<Vec<&str>> =
                tiers.iter().map(|tier| tier.iter().map(String::as_str).collect()).collect();
            let background =
                antecedent_graph::TieredBackground::from_named(data.schema(), &named, within)
                    .map_err(py_err)?;
            let query =
                static_ate_query(&data, &treatment, &outcome, control_level, active_level, &opts)?;
            let query = match functional {
                Some(functional) => query.with_outcome_functional(functional),
                None => query,
            };
            let builder =
                Study::tabular(data).tiered_background(background).map_err(py_err)?.query(query);
            let builder =
                apply_estimator(opts.apply_budget_for(builder, spec.is_some()), estimator, spec)?;
            let analysis = opts.apply_inference(builder)?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        let spec = configured_spec(estimator_config, estimator.as_deref(), &opts)?;
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let graph = StaticClassGraph::extract(&graph, &names)?;
        if matches!(graph, StaticClassGraph::Admg(_)) {
            opts.refuse_prior_transfer("an ADMG general-ID average effect")?;
        }
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let query =
                static_ate_query(&data, &treatment, &outcome, control_level, active_level, &opts)?;
            let query = match functional {
                Some(functional) => query.with_outcome_functional(functional),
                None => query,
            };
            let builder =
                graph.bind(Study::tabular(data), accepted, opts.discovery_algorithm()).query(query);
            let mut builder = opts.apply_budget_for(builder, spec.is_some());
            if let Some(id) = parse_identifier(identifier)? {
                builder = builder.identifier(id);
            }
            let builder = apply_estimator(builder, estimator, spec)?;
            let analysis = opts.apply_inference(builder)?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        opts.refuse_prior_transfer("a Cpdag/Pag response envelope")?;
        let response_options = parse_response_options(response_options.as_ref())?;
        let graph = StaticClassGraph::extract(&graph, &names)?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
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
            let query = response_query(functional, &opts);
            let mut builder = opts.apply(with_response_options(
                graph.bind(Study::tabular(data), accepted, opts.discovery_algorithm()).query(query),
                response_options,
            ));
            if let Some(id) = parse_identifier(identifier)? {
                builder = builder.identifier(id);
            }
            if let Some(est) = parse_estimator(estimator)? {
                builder = builder.estimator(est);
            }
            let analysis = opts.apply_inference(builder)?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
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
        contrast="mediated", control_level=0.0, active_level=1.0, accepted=false, seed=1,
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
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        if kind == "counterfactual" {
            opts.refuse_prior_transfer("a counterfactual query")?;
        }
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
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
        if let (CausalQuery::Mediation(q), Some(population)) =
            (&mut query, opts.target_population.clone())
        {
            q.target_population = population;
        } else {
            opts.refuse_population("a counterfactual unit-effect query")?;
        }
        detach_catch(py, move || {
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let builder = opts.apply(
                with_graph(Study::tabular(data), dag, accepted, opts.discovery_algorithm())
                    .query(query),
            );
            let study = opts.apply_inference(builder)?.build().map_err(py_err)?;
            let prepared = study.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        opts.refuse_prior_transfer("a response derivative")?;
        let response_options = parse_response_options(response_options.as_ref())?;
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
            let builder = with_response_options(
                with_graph(Study::tabular(data), dag, accepted, opts.discovery_algorithm())
                    .query(response_query(functional, &opts)),
                response_options,
            );
            let study = opts.apply_inference(opts.apply(builder))?.build().map_err(py_err)?;
            let prepared = study.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        let response_options = parse_response_options(response_options.as_ref())?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let query = response_query(
                ResponseFunctional::MeanCurve {
                    outcome: y_id,
                    treatment: ContinuousDomain::new(t_id, GridSpec::Values(grid.into())),
                },
                &opts,
            );
            let mut builder = opts.apply(with_response_options(
                with_graph(Study::tabular(data), dag, accepted, opts.discovery_algorithm())
                    .query(query),
                response_options,
            ));
            if let Some(id) = parse_identifier(identifier)? {
                builder = builder.identifier(id);
            }
            if let Some(est) = parse_estimator(estimator)? {
                builder = builder.estimator(est);
            }
            let analysis = opts.apply_inference(builder)?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        let class_graph = extract_temporal_class(class_graph, &names)?;
        let input = FrameInput::ingest(py, &names, columns, frame.as_ref())?;
        let policy = policy.to_ascii_lowercase();
        detach_catch(py, move || {
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
                .map_err(|_| PyValueError::new_err("treatment_lag does not fit in i32"))?;
            let functional = crate::response_api::wrap_temporal_sequence_steps(functional, origin)?;
            let temporal = TemporalResponseSpec::new(horizons, temporal_policy, max_history_lag)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
            if let Some(population) = opts.target_population.clone() {
                response.target_population = population;
            }
            let query = CausalQuery::Response(crate::observation_api::maybe_with_observation(
                response,
                &schema,
                observation.as_ref(),
            )?);
            let mut builder = data.builder()?;
            builder = if let Some(graph) = class_graph {
                bind_temporal_class(builder, graph, accepted, opts.discovery_algorithm())
            } else {
                with_graph(builder, dag, accepted, opts.discovery_algorithm())
            };
            builder = opts.apply(builder.query(query));
            builder = opts.apply_inference(builder)?;
            builder = crate::temporal_api::apply_class_prior(
                builder,
                class_prior_ordered,
                class_prior_pairs,
            )?;
            builder = crate::temporal_api::apply_max_completions(builder, max_completions);
            let analysis = builder.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, true))
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        let class_graph = extract_temporal_class(class_graph, &names)?;
        let input = FrameInput::ingest(py, &names, columns, frame.as_ref())?;
        let policy = policy.to_ascii_lowercase();
        detach_catch(py, move || {
            let data = input.materialize()?;
            let schema = data.schema().clone();
            let t_id = schema.id_of(&treatment).map_err(py_err)?;
            let y_id = schema.id_of(&outcome).map_err(py_err)?;
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
                &opts,
            )?;
            let mut builder = data.builder()?;
            builder = if let Some(graph) = class_graph {
                bind_temporal_class(builder, graph, accepted, opts.discovery_algorithm())
            } else {
                with_graph(
                    builder,
                    temporal_dag_from_schema_edges(&schema, &edges)?,
                    accepted,
                    opts.discovery_algorithm(),
                )
            };
            builder = opts.apply(builder.temporal_query(q));
            builder = opts.apply_inference(builder)?;
            builder = crate::temporal_api::apply_class_prior(
                builder,
                class_prior_ordered,
                class_prior_pairs,
            )?;
            builder = crate::temporal_api::apply_max_completions(builder, max_completions);
            let analysis = builder.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, true))
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        opts.refuse_prior_transfer("temporal mediation")?;
        let class_graph = extract_temporal_class(class_graph, &names)?;
        let input = FrameInput::ingest(py, &names, columns, frame.as_ref())?;
        let contrast = contrast.to_string();
        detach_catch(py, move || {
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
                &opts,
            )?;
            let mut builder = data.builder()?;
            builder = if let Some(graph) = class_graph {
                bind_temporal_class(builder, graph, accepted, opts.discovery_algorithm())
            } else {
                with_graph(
                    builder,
                    temporal_dag_from_schema_edges(&schema, &edges)?,
                    accepted,
                    opts.discovery_algorithm(),
                )
            };
            builder = opts.apply(builder.query(CausalQuery::Mediation(q)));
            builder = opts.apply_inference(builder)?;
            builder = crate::temporal_api::apply_class_prior(
                builder,
                class_prior_ordered,
                class_prior_pairs,
            )?;
            builder = crate::temporal_api::apply_max_completions(builder, max_completions);
            let analysis = builder.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, true))
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
        let opts = PrepareOptions::parse(options.as_ref())?;
        let spec = configured_spec(estimator_config, None, &opts)?;
        let supplied = take_supplied_posterior(posterior, &names)?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let query = CausalQuery::AverageEffect(static_ate_query(
                &data,
                &treatment,
                &outcome,
                control_level,
                active_level,
                &opts,
            )?);
            let ctx = opts.ctx(seed, threads);
            let gp = exact_dag_posterior(supplied, &data, &ctx)?;
            finish_static_graph_posterior(data, names, gp, query, spec, None, opts, &ctx)
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
        let opts = PrepareOptions::parse(options.as_ref())?;
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let supplied = take_supplied_posterior(posterior, &names)?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let query = conditional_query(
                &data,
                &treatment,
                &outcome,
                &modifier,
                control_level,
                active_level,
                functional,
                &opts,
            )?;
            let ctx = opts.ctx(seed, threads);
            let gp = exact_dag_posterior(supplied, &data, &ctx)?;
            finish_static_graph_posterior(data, names, gp, query, None, None, opts, &ctx)
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
        let opts = PrepareOptions::parse(options.as_ref())?;
        let response_options = parse_response_options(response_options.as_ref())?;
        let supplied = take_supplied_posterior(posterior, &names)?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let query = response_query(
                ResponseFunctional::MeanCurve {
                    outcome: y_id,
                    treatment: ContinuousDomain::new(t_id, GridSpec::Values(grid.into())),
                },
                &opts,
            );
            let ctx = opts.ctx(seed, threads);
            let gp = exact_dag_posterior(supplied, &data, &ctx)?;
            finish_static_graph_posterior(
                data,
                names,
                gp,
                query,
                None,
                response_options,
                opts,
                &ctx,
            )
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
        let opts = PrepareOptions::parse(options.as_ref())?;
        let supplied = take_supplied_posterior(posterior, &names)?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
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
            let query = response_query(functional, &opts);
            let ctx = opts.ctx(seed, threads);
            let gp = exact_dag_posterior(supplied, &data, &ctx)?;
            finish_static_graph_posterior(data, names, gp, query, None, None, opts, &ctx)
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
        let opts = PrepareOptions::parse(options.as_ref())?;
        let supplied = take_supplied_posterior(posterior, &names)?;
        let input = FrameInput::ingest(py, &names, columns, frame.as_ref())?;
        let policy = policy.to_ascii_lowercase();
        detach_catch(py, move || {
            let series = match input.materialize()? {
                FrameData::Series(series) => series,
                FrameData::Events(events, align) => events
                    .align_to_grid(align)
                    .map_err(|e| py_msg(format!("event align_to_grid: {e}")))?,
                _ => {
                    return Err(crate::refusal(
                        antecedent_core::reason_code!("data_modality_not_licensed"),
                        "a DBN graph-posterior mixture is \
                         licensed on one series or one event stream",
                    ));
                }
            };
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
                &opts,
            )?;
            let ctx = opts.ctx(seed, threads);
            let gp = dbn_posterior(
                supplied, &series, max_lag, force_mcmc, n_chains, n_warmup, mcmc_draws, &ctx,
            )?;
            finish_series_graph_posterior(
                series,
                names,
                gp,
                SeriesGraphQuery::Temporal(q),
                opts,
                &ctx,
            )
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
        let opts = PrepareOptions::parse(options.as_ref())?;
        let supplied = take_supplied_posterior(posterior, &names)?;
        let input = FrameInput::ingest(py, &names, columns, frame.as_ref())?;
        let contrast = contrast.to_string();
        detach_catch(py, move || {
            let series = match input.materialize()? {
                FrameData::Series(series) => series,
                FrameData::Events(events, align) => events
                    .align_to_grid(align)
                    .map_err(|e| py_msg(format!("event align_to_grid: {e}")))?,
                _ => {
                    return Err(crate::refusal(
                        antecedent_core::reason_code!("data_modality_not_licensed"),
                        "a DBN graph-posterior mixture is \
                         licensed on one series or one event stream",
                    ));
                }
            };
            let q = temporal_mediation_query(
                series.schema(),
                &treatment,
                &mediator,
                &outcome,
                &contrast,
                control_level,
                active_level,
                horizons,
                &opts,
            )?;
            let ctx = opts.ctx(seed, threads);
            let gp = dbn_posterior(
                supplied, &series, max_lag, force_mcmc, n_chains, n_warmup, mcmc_draws, &ctx,
            )?;
            finish_series_graph_posterior(
                series,
                names,
                gp,
                SeriesGraphQuery::Mediation(q),
                opts,
                &ctx,
            )
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
        let opts = PrepareOptions::parse(options.as_ref())?;
        let supplied = take_supplied_posterior(posterior, &names)?;
        let input = FrameInput::ingest(py, &names, columns, frame.as_ref())?;
        let policy = policy.to_ascii_lowercase();
        detach_catch(py, move || {
            let series = match input.materialize()? {
                FrameData::Series(series) => series,
                FrameData::Events(events, align) => events
                    .align_to_grid(align)
                    .map_err(|e| py_msg(format!("event align_to_grid: {e}")))?,
                _ => {
                    return Err(crate::refusal(
                        antecedent_core::reason_code!("data_modality_not_licensed"),
                        "a DBN graph-posterior mixture is \
                         licensed on one series or one event stream",
                    ));
                }
            };
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
                .map_err(|_| PyValueError::new_err("treatment_lag does not fit in i32"))?;
            let functional = crate::response_api::wrap_temporal_sequence_steps(functional, origin)?;
            let temporal = TemporalResponseSpec::new(horizons, temporal_policy, max_history_lag)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let mut response = ResponseQuery::new(functional).with_temporal(temporal);
            if let Some(population) = opts.target_population.clone() {
                response.target_population = population;
            }
            let ctx = opts.ctx(seed, threads);
            let gp = dbn_posterior(
                supplied, &series, max_lag, force_mcmc, n_chains, n_warmup, mcmc_draws, &ctx,
            )?;
            finish_series_graph_posterior(
                series,
                names,
                gp,
                SeriesGraphQuery::Response(response),
                opts,
                &ctx,
            )
        })
    }

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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        opts.refuse_prior_transfer("an InterventionResponse")?;
        let outcome_functional =
            crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let query = intervention_response_query(
                &data,
                &outcome,
                &treatments,
                intervention_kinds,
                intervention_parameters,
                outcome_functional,
                &opts,
            )?;
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let mut builder = opts.apply(
                with_graph(Study::tabular(data), dag, accepted, opts.discovery_algorithm())
                    .query(query),
            );
            if let Some(id) = parse_identifier(identifier)? {
                builder = builder.identifier(id);
            }
            if let Some(est) = parse_estimator(estimator)? {
                builder = builder.estimator(est);
            }
            let analysis = opts.apply_inference(builder)?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        let outcome_functional =
            crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
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
                &opts,
            )?;
            let builder = Study::tabular(data)
                .tiered_background(background)
                .map_err(py_err)?
                .query(query)
                .estimator(antecedent::EstimatorId::CellAipw);
            let analysis = opts.apply_inference(opts.apply(builder))?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
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
        modifier,
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
        modifier: String,
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        if graph.is_some() {
            opts.refuse_prior_transfer("a Cpdag/Pag conditional envelope")?;
        }
        let functional = crate::ate_api::parse_outcome_functional(outcome_functional.as_ref())?;
        let class = graph.map(|g| StaticClassGraph::extract(&g, &names)).transpose()?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let query = conditional_query(
                &data,
                &treatment,
                &outcome,
                &modifier,
                control_level,
                active_level,
                functional,
                &opts,
            )?;
            let mut builder = if let Some(class) = class {
                class.bind(Study::tabular(data), accepted, opts.discovery_algorithm())
            } else {
                let dag = dag_from_named_edges(data.schema(), &edges)?;
                with_graph(Study::tabular(data), dag, accepted, opts.discovery_algorithm())
            };
            builder = opts.apply(builder.query(query));
            if let Some(id) = parse_identifier(identifier)? {
                builder = builder.identifier(id);
            }
            if let Some(est) = parse_estimator(estimator)? {
                builder = builder.estimator(est);
            }
            let analysis = opts.apply_inference(builder)?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        opts.refuse_prior_transfer("a path-specific effect")?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let mut query = PathSpecificEffectQuery::binary(t_id, y_id)
                .with_max_paths(max_paths)
                .with_max_len(max_len);
            query.control = Intervention::set(t_id, Value::f64(control_level));
            query.active = Intervention::set(t_id, Value::f64(active_level));
            if let Some(population) = opts.target_population.clone() {
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
                with_graph(Study::tabular(data), dag, accepted, opts.discovery_algorithm())
                    .query(CausalQuery::PathSpecific(query))
                    .identifier(IdentifierId::PathSpecificNatural)
                    .estimator(EstimatorId::FunctionalEffect);
            let analysis = opts.apply_inference(opts.apply(builder))?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        opts.refuse_prior_transfer("an interventional distribution")?;
        let class = graph.map(|g| StaticClassGraph::extract(&g, &names)).transpose()?;
        if matches!(&class, Some(g) if !matches!(g, StaticClassGraph::Admg(_))) {
            return Err(PyValueError::new_err(
                "prepare_distribution graph must be an Admg; Pag/Cpdag remain refused",
            ));
        }
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let mut ivs = Vec::with_capacity(interventions.len());
            for (name, level) in &interventions {
                let id = data.schema().id_of(name).map_err(py_err)?;
                ivs.push(Intervention::set(id, Value::f64(*level)));
            }
            let mut query = InterventionalDistributionQuery::new(y_id, ivs);
            if let Some(population) = opts.target_population.clone() {
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
                class.bind(Study::tabular(data), accepted, opts.discovery_algorithm())
            } else {
                let dag = dag_from_named_edges(data.schema(), &edges)?;
                with_graph(Study::tabular(data), dag, accepted, opts.discovery_algorithm())
            };
            let builder = builder
                .query(CausalQuery::Distribution(query))
                .identifier(IdentifierId::GeneralId)
                .estimator(EstimatorId::FunctionalDistribution);
            let analysis = opts.apply_inference(opts.apply(builder))?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
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
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        opts.refuse_prior_transfer("a transport query")?;
        let admg = graph
            .extract::<graphs::Admg>()
            .map_err(|_| PyValueError::new_err("TransportQuery requires graph=Admg(...)"))?;
        require_named_graph_order(&admg.names, &names, "Admg")?;
        let catalog = catalog
            .map(|c| crate::transport_interference_api::parse_catalog(c, &admg))
            .transpose()?;
        let admg = admg.aligned_to_names(&names)?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let (scale, weighting) = (scale.to_owned(), weighting.to_owned());
        detach_catch(py, move || {
            use crate::transport_interference_api::{ResponseArgs, schema_ids, transport_query};
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
                query = query
                    .with_catalog(catalog)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
            let column = |name: &str| crate::graph_build::schema_var_id(&schema, name);
            let trial_spec = antecedent::TransportTrialSpec {
                trial: column(&trial)?,
                selection_probability: column(&selection_probability)?,
                treatment_probability: column(&treatment_probability)?,
            };
            let builder =
                with_graph(Study::tabular(data), admg, accepted, opts.discovery_algorithm())
                    .query(CausalQuery::Transport(query))
                    .selection_targets(schema_ids(&schema, &selections)?)
                    .transport_trial(trial_spec);
            let analysis = opts.apply_inference(opts.apply(builder))?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
        })
    }

    /// Freeze a randomized-interference study: the fixed unit network and the
    /// realized assignment are the design; the unit table is the data.
    #[staticmethod]
    #[pyo3(signature = (names, columns, edges, outcome, network, realized_assignment,
        assignment_kind, assignment_probabilities, treated, clusters, treated_clusters, exposure,
        from_level, to_level, *, probability_draws=10_000, accepted=false, seed=1, threads=None,
        options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_interference(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        outcome: String,
        network: Vec<(u32, u32, f64)>,
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
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        opts.refuse_prior_transfer("an interference query")?;
        opts.refuse_population("an interference exposure contrast")?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            use crate::transport_interference_api::{InterferenceArgs, interference_query};
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            let query = interference_query(
                InterferenceArgs {
                    assignment_kind,
                    assignment_probabilities,
                    treated,
                    clusters,
                    treated_clusters,
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
                    .map(|(from, to, weight)| antecedent::estimate::NetworkEdge {
                        from,
                        to,
                        weight,
                    })
                    .collect::<Vec<_>>(),
            )
            .map_err(py_err)?;
            let builder =
                with_graph(Study::tabular(data), dag, accepted, opts.discovery_algorithm())
                    .query(CausalQuery::Interference(query))
                    .interference(antecedent::InterferenceSpec {
                        network,
                        assignment: Arc::from(realized_assignment),
                    });
            let analysis = opts.apply_inference(opts.apply(builder))?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
        })
    }

    /// Freeze GCM anomaly scores on a supplied explicit Dag.
    ///
    /// Compile assigns `gcm.parametric` / `gcm.fit`. Do not set identifier or
    /// estimator here.
    #[staticmethod]
    #[pyo3(signature = (names, columns, edges, targets, max_units, *, accepted=false, seed=1,
        threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_anomaly_attribution(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        targets: Vec<String>,
        max_units: usize,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        opts.refuse_prior_transfer("an anomaly attribution query")?;
        opts.refuse_population("an anomaly attribution query")?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let target_ids = targets
                .iter()
                .map(|name| crate::graph_build::schema_var_id(data.schema(), name))
                .collect::<PyResult<Vec<_>>>()?;
            let query = AnomalyAttributionQuery::new(target_ids, max_units);
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let builder =
                with_graph(Study::tabular(data), dag, accepted, opts.discovery_algorithm())
                    .query(CausalQuery::AnomalyAttribution(query));
            let analysis = opts.apply_inference(opts.apply(builder))?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
        })
    }

    /// Freeze GCM distribution-change Shapley on a supplied explicit Dag.
    ///
    /// Compile assigns `gcm.parametric` / `gcm.fit`. Do not set identifier or
    /// estimator here.
    #[staticmethod]
    #[pyo3(signature = (names, columns, edges, outcome, baseline_start, baseline_end,
        comparison_start, comparison_end, *, accepted=false, seed=1, threads=None, options=None))]
    #[allow(clippy::too_many_arguments)]
    fn prepare_change_attribution(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        outcome: String,
        baseline_start: usize,
        baseline_end: usize,
        comparison_start: usize,
        comparison_end: usize,
        accepted: bool,
        seed: u64,
        threads: Option<u32>,
        options: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut opts = PrepareOptions::parse(options.as_ref())?;
        opts.refuse_prior_transfer("a change attribution query")?;
        opts.refuse_population("a change attribution query")?;
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        detach_catch(py, move || {
            let outcome_id = crate::graph_build::schema_var_id(data.schema(), &outcome)?;
            let query = ChangeAttributionQuery::new(
                outcome_id,
                PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
                PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
            );
            let dag = dag_from_named_edges(data.schema(), &edges)?;
            let builder =
                with_graph(Study::tabular(data), dag, accepted, opts.discovery_algorithm())
                    .query(CausalQuery::ChangeAttribution(query));
            let analysis = opts.apply_inference(opts.apply(builder))?.build().map_err(py_err)?;
            let prepared = analysis.prepare(&opts.ctx(seed, threads)).map_err(py_err)?;
            Ok(finished_prepared(prepared, names, false))
        })
    }

    /// Weighted-mean retarget from the frozen score table. Does not refit.
    #[pyo3(signature = (weights, depends_on, *, seed=1, threads=None))]
    fn retarget(
        &mut self,
        py: Python<'_>,
        weights: Vec<f64>,
        depends_on: Vec<String>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<AteAnalysisResult> {
        self.refuse_design_retarget()?;
        // Retarget the scores of the execution this call follows: after
        // `estimate(data)` that is the handle refreshed on `data`, not the one
        // frozen at prepare.
        let bound = self.last_study.clone().unwrap_or_else(|| Arc::clone(&self.inner));
        let inner = Arc::clone(&bound);
        let out_names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                crate::resolve_user_threads(threads),
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
        self.last_study = Some(bound);
        self.last = Some(Arc::new(result));
        self.last_seed = seed;
        self.last_threads = crate::resolve_user_threads(threads);
        Ok(mapped)
    }

    /// Re-execute the row-weight retarget an exported contract carries.
    ///
    /// Refuses with `row_weights_bound_to_snapshot` unless this handle holds
    /// the data snapshot and score table the weights were bound to.
    #[pyo3(signature = (artifact, *, seed=1, threads=None))]
    fn reexecute_retarget(
        &mut self,
        py: Python<'_>,
        artifact: Vec<u8>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<AteAnalysisResult> {
        self.refuse_design_retarget()?;
        let bound = self.last_study.clone().unwrap_or_else(|| Arc::clone(&self.inner));
        let inner = Arc::clone(&bound);
        let out_names = self.names.clone();
        let (mapped, result) = detach_catch(py, move || {
            let ctx = py_execution_context_ext(
                seed,
                crate::resolve_user_threads(threads),
                None,
                None,
                Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
            );
            let consumed = antecedent_io::consume_analysis_result(&artifact).map_err(py_err)?;
            let section = consumed
                .contract
                .and_then(|contract| contract.target_weights)
                .ok_or_else(|| {
                    PyValueError::new_err("artifact carries no row-weight retarget to re-execute")
                })?;
            let result = inner.reexecute_retarget(&section, &ctx).map_err(py_err)?;
            let mapped = ate_result_from_analysis(&out_names, result.clone(), false)?;
            Ok((mapped, result))
        })?;
        self.last_study = Some(bound);
        self.last = Some(Arc::new(result));
        self.last_seed = seed;
        self.last_threads = crate::resolve_user_threads(threads);
        Ok(mapped)
    }

    /// Re-estimate on new columns (same schema) without recompiling.
    #[pyo3(signature = (names, columns, *, seed=1, threads=None, cancel=None, on_progress=None, on_stage=None))]
    #[allow(clippy::too_many_arguments)]
    fn estimate(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<AteAnalysisResult> {
        require_prepared_names(&self.names, &names, "estimate")?;
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        let data = FrameInput::Series(tabular_from_py_columns(py, names, columns)?.0);
        self.click(py, data, false, seed, threads, controls, map_ate)
    }

    /// Re-estimate a prepared response (same schema) without recompiling.
    #[pyo3(signature = (names, columns, *, seed=1, threads=None, cancel=None, on_progress=None, on_stage=None))]
    #[allow(clippy::too_many_arguments)]
    fn estimate_response(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<ResponseAnalysisResult> {
        require_prepared_names(&self.names, &names, "estimate")?;
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        let data = FrameInput::Series(tabular_from_py_columns(py, names, columns)?.0);
        self.click(py, data, false, seed, threads, controls, response_from_study)
    }

    /// Replace retained data and re-estimate (same schema).
    #[pyo3(signature = (names, columns, *, seed=1, threads=None, cancel=None, on_progress=None, on_stage=None))]
    #[allow(clippy::too_many_arguments)]
    fn refresh(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<AteAnalysisResult> {
        require_prepared_names(&self.names, &names, "refresh")?;
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        let data = FrameInput::Series(tabular_from_py_columns(py, names, columns)?.0);
        self.click(py, data, true, seed, threads, controls, map_ate)
    }

    /// Replace retained data and re-estimate a prepared response.
    #[pyo3(signature = (names, columns, *, seed=1, threads=None, cancel=None, on_progress=None, on_stage=None))]
    #[allow(clippy::too_many_arguments)]
    fn refresh_response(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
        on_progress: Option<Bound<'_, PyAny>>,
        on_stage: Option<Bound<'_, PyAny>>,
    ) -> PyResult<ResponseAnalysisResult> {
        require_prepared_names(&self.names, &names, "refresh")?;
        let controls = ClickControls::parse(cancel, on_progress.as_ref(), on_stage.as_ref())?;
        let data = FrameInput::Series(tabular_from_py_columns(py, names, columns)?.0);
        self.click(py, data, true, seed, threads, controls, response_from_study)
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
                antecedent_io::causal_query_to_wire_with_registry(
                    self.inner.query(),
                    self.inner.population_registry(),
                )
                .map_err(py_err)?,
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
        } else if result.mediation_grid.is_some()
            || result.structural_response.is_some()
            || (result.posterior.is_some() && result.response.is_some())
        {
            let wire = composite_result_wire(
                result,
                self.inner.query(),
                self.inner.population_registry(),
                self.inner.temporal_identification(),
                artifact_id,
            )?;
            let artifact = antecedent_io::encode_analysis_result_artifact(
                &wire,
                self.names.clone(),
                artifact_id,
            )
            .map_err(py_err)?;
            let mut bytes = Vec::new();
            artifact.write_to(&mut bytes).map_err(py_err)?;
            bytes
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
                identification: antecedent_io::identification_to_wire_with_registry(
                    &result.identification,
                    self.inner.population_registry(),
                )
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
                unit_extrapolative: result
                    .counterfactual
                    .as_ref()
                    .and_then(|c| c.unit_extrapolative.as_ref())
                    .map(|flags| flags.to_vec()),
                mediation: result
                    .mediation
                    .as_ref()
                    .map(|m| {
                        Ok::<_, PyErr>([
                            m.total.ok_or_else(|| {
                                PyValueError::new_err("mediation total is unavailable")
                            })?,
                            m.direct.ok_or_else(|| {
                                PyValueError::new_err("mediation direct is unavailable")
                            })?,
                            m.mediated.ok_or_else(|| {
                                PyValueError::new_err("mediation mediated is unavailable")
                            })?,
                        ])
                    })
                    .transpose()?,
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

    /// Export the last estimate as an `analysis_result` with a contract section.
    #[pyo3(signature = (*, artifact_id="prepared-contract"))]
    fn export_contracted_artifact<'py>(
        &self,
        py: Python<'py>,
        artifact_id: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let result = self.last.as_ref().ok_or_else(|| {
            PyValueError::new_err("estimate before exporting a contracted artifact")
        })?;
        let ctx = crate::py_execution_context_ext(
            self.last_seed,
            self.last_threads,
            None,
            None,
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        );
        let bytes = self
            .last_study
            .as_ref()
            .unwrap_or(&self.inner)
            .encode_contracted_result(result, artifact_id, &ctx)
            .map_err(py_err)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Second-click refute against the last estimate (same schema data).
    #[pyo3(signature = (names, columns, suite, *, seed=1, threads=None, cancel=None))]
    fn refute(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        suite: Bound<'_, PyAny>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<AteAnalysisResult> {
        require_prepared_names(&self.names, &names, "refute")?;
        let data = tabular_from_numpy(&names, &columns)?;
        drop(columns);
        self.finish_ate_refute(py, data, suite, seed, threads, cancel)
    }

    #[pyo3(signature = (names, columns, suite, *, seed=1, threads=None, cancel=None))]
    fn refute_arrow_c(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        suite: Bound<'_, PyAny>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<AteAnalysisResult> {
        require_prepared_names(&self.names, &names, "refute")?;
        let (data, _) = tabular_from_arrow_c_objs(py, names, columns)?;
        self.finish_ate_refute(py, data, suite, seed, threads, cancel)
    }

    /// Whether the last retarget's weights changed the target population.
    ///
    /// A constant weight vector re-reports the prepared population, so the
    /// handle is not bound to the retarget's data snapshot.
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

    /// Cheap inspect: four slots without using cached identification.
    fn inspect(&self) -> PyResult<std::collections::HashMap<String, String>> {
        let contract = self.inner.inspect().map_err(py_err)?;
        Ok(contract_to_map(
            &contract,
            self.inner.schema(),
            self.inner.query(),
            self.inner.population_registry(),
        ))
    }

    /// Domain-separated contract identities and four reasoning slots.
    fn contract(&self) -> PyResult<std::collections::HashMap<String, String>> {
        let contract = self.inner.contract().map_err(py_err)?;
        let mut report = contract_to_map(
            &contract,
            self.inner.schema(),
            self.inner.query(),
            self.inner.population_registry(),
        );
        insert_reuse_identities(
            &mut report,
            self.inner.score_reuse_identity().map_err(py_err)?,
            None,
        );
        Ok(report)
    }

    /// Transformation preview retaining all frozen contract input identities.
    fn preview_transform(
        &self,
        intent: String,
    ) -> PyResult<std::collections::HashMap<String, String>> {
        let intent = parse_transform_intent(&intent)?;
        let report = self.inner.preview_transform(intent).map_err(py_err)?;
        Ok(transform_report_map(&report))
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
            result.structural_response.as_ref(),
            treatments,
            outcomes,
            adjustment_set,
            names,
            result.support_status,
        )?,
        crate::identification_details::analysis_to_json(result, names)?,
        result.logical_plan.identifier.as_deref().map(str::to_owned),
        result.diagnostics.iter().map(|d| format!("{}: {}", d.code, d.message)).collect(),
    ))
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

fn extract_temporal_class(
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
    } else {
        Err(PyValueError::new_err("class_graph requires TemporalCpdag or TemporalPag"))
    }
}

fn hard_value(intervention: &Intervention) -> Option<f64> {
    match intervention {
        Intervention::Set { value, .. } => value.as_f64(),
        _ => None,
    }
}

fn composite_result_wire(
    result: &antecedent::StudyResult,
    query: &CausalQuery,
    registry: Option<&antecedent_core::PopulationRegistry>,
    temporal: Option<&antecedent::analysis::CachedTemporalIdentification>,
    artifact_id: &str,
) -> PyResult<antecedent_io::AnalysisResultWire> {
    let identification =
        antecedent_io::identification_to_wire_with_registry(&result.identification, registry)
            .map_err(py_err)?;
    let temporal_identification = temporal
        .into_iter()
        .flat_map(|cache| cache.by_horizon.iter())
        .map(|entry| {
            let variables = (0..entry.indexer.dense_len())
                .map(|dense| {
                    let key = entry
                        .indexer
                        .key_of(
                            u32::try_from(dense)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?,
                        )
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(antecedent_io::HorizonAdjustmentNodeWire {
                        variable: key.variable.raw(),
                        offset: key.offset,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(antecedent_io::TemporalIdentificationWire {
                horizon: entry.horizon,
                variables,
                identification: antecedent_io::identification_to_wire_with_registry(
                    &entry.identification,
                    registry,
                )
                .map_err(py_err)?,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut identification_variables = temporal_identification
        .iter()
        .find(|entry| entry.identification.query == identification.query)
        .map(|entry| entry.variables.clone());
    if identification_variables.is_none() {
        if let Some(antecedent::AnalysisIdentification {
            identification: antecedent::Identification::TemporalEnvelope { envelope, .. },
            ..
        }) = result.certificate.as_ref()
        {
            if let Some((_, indexer)) =
                envelope.envelope.cases.iter().zip(&envelope.indexers).find(|(case, _)| {
                    case.result.estimands.iter().any(|estimand| {
                        estimand.method == result.estimand.method
                            && estimand.adjustment_set == result.estimand.adjustment_set
                    })
                })
            {
                identification_variables = Some(
                    (0..indexer.dense_len())
                        .map(|dense| {
                            let key = indexer
                                .key_of(u32::try_from(dense).map_err(py_msg)?)
                                .map_err(py_msg)?;
                            Ok(antecedent_io::HorizonAdjustmentNodeWire {
                                variable: key.variable.raw(),
                                offset: key.offset,
                            })
                        })
                        .collect::<PyResult<Vec<_>>>()?,
                );
            }
        }
    }
    let mut wire = antecedent_io::AnalysisResultWire {
        query: antecedent_io::causal_query_to_wire_with_registry(query, registry)
            .map_err(py_err)?,
        identification,
        identification_variables,
        temporal_identification,
        estimate: result.estimate.ate.is_finite().then_some(result.estimate.ate),
        standard_error: result.estimate.se_bootstrap.or_else(|| {
            result.estimate.se_analytic.is_finite().then_some(result.estimate.se_analytic)
        }),
        assumptions: antecedent_io::assumptions_to_wire(&result.estimate.assumptions),
        diagnostics: result.diagnostics.iter().map(antecedent_io::diagnostic_to_wire).collect(),
        refutations: result.refutations.iter().map(antecedent_io::refutation_to_wire).collect(),
        response: None,
        posterior_artifact: None,
        mediation_grid: None,
        structural_response: None,
        unit_effects: None,
        cate: result.estimate.cate.as_ref().map(|v| v.to_vec()),
        outcome_oof_r2: result.estimate.outcome_oof_r2,
        treatment_oof_logloss: result.estimate.treatment_oof_logloss,
        crossfit_folds: result.estimate.crossfit_folds,
        crossfit_seed: result.estimate.crossfit_seed,
        learner_provenance: result
            .estimate
            .learner_provenance
            .iter()
            .map(|p| (p.spec.clone(), p.implementation.clone(), p.version.clone()))
            .collect(),
    };
    result.fill_analysis_result_payloads(&mut wire, artifact_id).map_err(py_err)?;
    Ok(wire)
}

pub(crate) fn parse_transform_intent(intent: &str) -> PyResult<antecedent_core::TransformIntent> {
    let intent = match intent {
        "display_precision" => antecedent_core::TransformIntent::DisplayPrecision,
        "compatible_data_replace" => antecedent_core::TransformIntent::CompatibleDataReplace,
        "retarget" => antecedent_core::TransformIntent::Retarget,
        "filter_display" => antecedent_core::TransformIntent::FilterDisplay,
        "filter_population" => antecedent_core::TransformIntent::FilterPopulation,
        "new_conditional_query" => antecedent_core::TransformIntent::NewConditionalQuery,
        "change_graph" => antecedent_core::TransformIntent::ChangeGraph,
        "change_prior" => antecedent_core::TransformIntent::ChangePrior,
        "change_physical_policy" => antecedent_core::TransformIntent::ChangePhysicalPolicy,
        "average_unweighted_class" => antecedent_core::TransformIntent::AverageUnweightedClass,
        other => {
            return Err(PyValueError::new_err(format!("unknown transform intent {other:?}")));
        }
    };
    Ok(intent)
}
pub(crate) fn transform_report_map(
    report: &antecedent_core::TransformationReport,
) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    out.insert("intent".into(), report.intent.as_str().to_string());
    for identity in report.input_identities.iter() {
        out.insert(format!("input_{}", identity.domain.as_str()), identity.digest.to_hex());
    }
    out.insert("refused".into(), report.refused.to_string());
    if let Some(refusal) = &report.refusal {
        if let Some((code, _)) = antecedent_core::reason_code::split_prefix(refusal) {
            out.insert("refusal_code".into(), code.to_string());
        }
        out.insert("refusal".into(), refusal.to_string());
    }
    out.insert(
        "obligations".into(),
        report.obligations.iter().map(|o| o.id.to_string()).collect::<Vec<_>>().join(","),
    );
    out
}
