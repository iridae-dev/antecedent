//! Batch preparation, execution and completed-result handles.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn parse_latency_mode(latency: Option<&str>) -> PyResult<Option<antecedent::LatencyMode>> {
    match latency {
        None => Ok(None),
        Some(s) => Ok(Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
            crate::value_err(format!("unknown latency={s:?}; use interactive|standard|report"))
        })?)),
    }
}

type AteBatchQuerySpec = (
    String,
    String,
    f64,
    f64,
    Option<antecedent::query::OutcomeFunctional>,
    Option<antecedent::query::TargetPopulation>,
);

fn parse_ate_batch_query_specs(queries: Vec<PyBatchQuery<'_>>) -> PyResult<Vec<AteBatchQuerySpec>> {
    let mut parsed = Vec::with_capacity(queries.len());
    for (treatment, outcome, control, active, functional, population) in queries {
        parsed.push((
            treatment,
            outcome,
            control,
            active,
            parse_outcome_functional(functional.as_ref())?,
            parse_target_population(population.as_ref())?,
        ));
    }
    Ok(parsed)
}

/// The typed estimator configuration of a batch, parsed with the single-query `analyze`
/// parser so both entry points accept the same keys and refuse the same mistakes.
fn parse_batch_estimator_config(
    estimator_config: Option<&Bound<'_, PyDict>>,
    estimator: Option<&str>,
    bootstrap: Option<u32>,
) -> PyResult<Option<antecedent::EstimatorSpec>> {
    let parsed = crate::estimator_config::parse_estimator_config(
        estimator_config,
        estimator,
        bootstrap.unwrap_or(antecedent::StudyBuilder::OMITTED_BOOTSTRAP),
    )?;
    if parsed.rd_running_variable.is_some()
        || parsed.rd_cutoff.is_some()
        || parsed.rd_bandwidth.is_some()
        || parsed.rd_se_kind.is_some()
    {
        return Err(crate::value_err(
            "estimator_config for rd.sharp does not apply to a batch of average effects",
        ));
    }
    Ok(parsed.spec)
}

fn compile_batch_study(
    data: antecedent::prelude::TabularData,
    edges: Vec<(String, String)>,
    identifier: Option<String>,
    estimator: Option<String>,
    configured_spec: Option<antecedent::EstimatorSpec>,
    suite: Option<antecedent::RefuteSuite>,
    bootstrap: Option<u32>,
    latency_mode: Option<antecedent::LatencyMode>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u64>>,
    estimate_rows: Option<Vec<u64>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
) -> PyResult<antecedent::BatchStudy> {
    if tiers.is_some() && !edges.is_empty() {
        return Err(crate::value_err(
            "a batch takes edges or tiers, not both; tiers declare the background and the \
             edges would be dropped",
        ));
    }
    if tiers.is_none() && within_tier.is_some() {
        return Err(crate::value_err("within_tier orders the tiers; declare tiers or drop it"));
    }
    let mut batch = if let Some(tiers) = tiers {
        let within = crate::parse_within_tier(within_tier.as_deref())?;
        let named: Vec<Vec<&str>> =
            tiers.iter().map(|tier| tier.iter().map(String::as_str).collect()).collect();
        let background =
            antecedent::graph::TieredBackground::from_named(data.schema(), &named, within)
                .map_err(py_err)?;
        antecedent::BatchStudy::tiered(data, background)
    } else {
        let dag = dag_from_named_edges(data.schema(), &edges)?;
        antecedent::BatchStudy::new(data, dag)
    };
    // A configured estimator carries its own bootstrap count (the study refuses both), as
    // for a single-query `analyze`.
    if let Some(replicates) = bootstrap.filter(|_| configured_spec.is_none()) {
        batch = batch.bootstrap_replicates(replicates);
    }
    if let Some(suite) = suite {
        batch = batch.refute(suite);
    }
    if let Some(mode) = latency_mode {
        batch = batch.latency_mode(mode);
    }
    if let Some(id) = identifier {
        batch = batch.identifier(
            id.parse::<antecedent::IdentifierId>().map_err(|e| crate::value_err(e.to_string()))?,
        );
    }
    if let Some(spec) = configured_spec {
        batch = batch.estimator_spec(spec);
    } else if let Some(est) = estimator {
        batch = batch.estimator(
            est.parse::<antecedent::EstimatorId>().map_err(|e| crate::value_err(e.to_string()))?,
        );
    }
    if let Some(screen) =
        parse_candidate_screen(screen_id, screen_procedure, screen_rows, estimate_rows)?
    {
        batch = batch.candidate_screen(screen);
    }
    Ok(batch)
}

fn compile_ate_batch(
    data: antecedent::prelude::TabularData,
    edges: Vec<(String, String)>,
    parsed_queries: Vec<AteBatchQuerySpec>,
    identifier: Option<String>,
    estimator: Option<String>,
    configured_spec: Option<antecedent::EstimatorSpec>,
    suite: Option<antecedent::RefuteSuite>,
    bootstrap: Option<u32>,
    latency_mode: Option<antecedent::LatencyMode>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u64>>,
    estimate_rows: Option<Vec<u64>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
) -> PyResult<(antecedent::BatchStudy, Vec<AverageEffectQuery>)> {
    let mut ate_queries = Vec::with_capacity(parsed_queries.len());
    for (treatment, outcome, control, active, functional, population) in &parsed_queries {
        let t_id = data.schema().id_of(treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(outcome).map_err(py_err)?;
        let mut query = AverageEffectQuery::with_levels(t_id, y_id, *control, *active);
        if let Some(functional) = functional.clone() {
            query = query.with_outcome_functional(functional);
        }
        if let Some(population) = population.clone() {
            query = query.with_target_population(population);
        }
        ate_queries.push(query);
    }
    let batch = compile_batch_study(
        data,
        edges,
        identifier,
        estimator,
        configured_spec,
        suite,
        bootstrap,
        latency_mode,
        screen_id,
        screen_procedure,
        screen_rows,
        estimate_rows,
        tiers,
        within_tier,
    )?;
    Ok((batch, ate_queries))
}

struct CellBatchQuerySpec {
    outcome: String,
    treatments: Vec<String>,
    kinds: Vec<String>,
    parameters: Vec<Vec<f64>>,
    functional: Option<antecedent::query::OutcomeFunctional>,
    population: Option<TargetPopulation>,
    observation: crate::observation_api::ObservationArgs,
    assumptions: Vec<crate::observation_api::ObservationArgs>,
}

type PyCellBatchQuery<'py> = (
    String,
    Vec<String>,
    Vec<String>,
    Vec<Vec<f64>>,
    Option<Bound<'py, PyDict>>,
    Option<Bound<'py, PyDict>>,
    Option<Bound<'py, PyDict>>,
    Vec<Bound<'py, PyDict>>,
);

/// Read the established observation kwargs codec; scientific validation remains
/// in the shared observation builders and the native ResponseQuery consumers.
fn cell_observation_args(
    specification: Option<&Bound<'_, PyDict>>,
    outcome: &str,
) -> PyResult<crate::observation_api::ObservationArgs> {
    let text = |name: &str| -> PyResult<Option<String>> {
        specification
            .map(|spec| spec.get_item(name))
            .transpose()?
            .flatten()
            .filter(|value| !value.is_none())
            .map(|value| value.extract())
            .transpose()
    };
    let assumption_variables = specification
        .map(|spec| spec.get_item("assumption_variables"))
        .transpose()?
        .flatten()
        .map(|value| value.extract::<Vec<String>>())
        .transpose()?
        .unwrap_or_default();
    Ok(crate::observation_api::ObservationArgs {
        kind: text("observation_kind")?.unwrap_or_else(|| "complete".into()),
        latent: text("latent")?.unwrap_or_else(|| outcome.to_owned()),
        observed: text("observed")?,
        censoring: text("censoring")?,
        event: text("event")?,
        lower: text("lower")?,
        upper: text("upper")?,
        indicator: text("indicator")?,
        assumption_kind: text("assumption_kind")?.unwrap_or_default(),
        assumption_variables,
        structural_model: text("structural_model")?,
    })
}

fn parse_cell_batch_query_specs(
    queries: Vec<PyCellBatchQuery<'_>>,
) -> PyResult<Vec<CellBatchQuerySpec>> {
    let mut parsed = Vec::with_capacity(queries.len());
    for (
        outcome,
        treatments,
        kinds,
        parameters,
        functional,
        population,
        observation,
        assumptions,
    ) in queries
    {
        let observation = cell_observation_args(observation.as_ref(), &outcome)?;
        let assumptions = assumptions
            .iter()
            .map(|assumption| cell_observation_args(Some(assumption), &outcome))
            .collect::<PyResult<Vec<_>>>()?;
        parsed.push(CellBatchQuerySpec {
            outcome,
            treatments,
            kinds,
            parameters,
            functional: parse_outcome_functional(functional.as_ref())?,
            population: parse_target_population(population.as_ref())?,
            observation,
            assumptions,
        });
    }
    Ok(parsed)
}

fn compile_cell_batch(
    data: antecedent::prelude::TabularData,
    edges: Vec<(String, String)>,
    parsed_queries: Vec<CellBatchQuerySpec>,
    identifier: Option<String>,
    estimator: Option<String>,
    configured_spec: Option<antecedent::EstimatorSpec>,
    suite: Option<antecedent::RefuteSuite>,
    bootstrap: Option<u32>,
    latency_mode: Option<antecedent::LatencyMode>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u64>>,
    estimate_rows: Option<Vec<u64>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
    family_contrast: Option<String>,
) -> PyResult<(antecedent::BatchStudy, Vec<ResponseQuery>)> {
    let mut cell_queries = Vec::with_capacity(parsed_queries.len());
    for declaration in &parsed_queries {
        let treatment_ids =
            crate::response_api::resolve_names(data.schema(), &declaration.treatments)?;
        let outcome_ids = crate::response_api::resolve_names(
            data.schema(),
            std::slice::from_ref(&declaration.outcome),
        )?;
        let built = crate::response_api::build_functional(
            "intervention_response",
            &treatment_ids,
            &outcome_ids,
            None,
            None,
            None,
            Some(declaration.kinds.clone()),
            Some(declaration.parameters.clone()),
            1,
            antecedent::query::DerivativeScale::Identity,
            antecedent::query::DerivativeWeighting::Observed,
        )?;
        let mut query = ResponseQuery::new(built);
        if let Some(functional) = declaration.functional.clone() {
            query = query.with_outcome_functional(functional);
        }
        if let Some(population) = declaration.population.clone() {
            query.target_population = population;
        }
        let latent = data.schema().id_of(&declaration.observation.latent).map_err(py_err)?;
        let observation = crate::observation_api::build_observation(
            data.schema(),
            &declaration.observation,
            latent,
        )?;
        let assumptions = declaration
            .assumptions
            .iter()
            .map(|args| crate::observation_api::build_assumption(data.schema(), args))
            .collect::<PyResult<Vec<_>>>()?;
        query = query.with_observation(observation, assumptions);
        cell_queries.push(query);
    }
    let batch = compile_batch_study(
        data,
        edges,
        identifier,
        estimator,
        configured_spec,
        suite,
        bootstrap,
        latency_mode,
        screen_id,
        screen_procedure,
        screen_rows,
        estimate_rows,
        tiers,
        within_tier,
    )?;
    let contrast = match family_contrast.as_deref() {
        None => None,
        Some(name) => Some(name.parse::<antecedent::CellFamilyContrast>().map_err(py_err)?),
    };
    Ok((batch.family_contrast(contrast), cell_queries))
}

/// Batch static ATE: one table ingest, N average-effect queries.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    edges,
    queries,
    *,
    identifier=None,
    estimator=None,
    estimator_config=None,
    refute=None,
    seed=1,
    bootstrap=None,
    threads=None,
    latency=None,
    screen_id=None,
    screen_procedure=None,
    screen_rows=None,
    estimate_rows=None,
    tiers=None,
    within_tier=None,
))]
fn analyze_ate_many(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    queries: Vec<PyBatchQuery<'_>>,
    identifier: Option<String>,
    estimator: Option<String>,
    estimator_config: Option<Bound<'_, PyDict>>,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: Option<u32>,
    threads: Option<u32>,
    latency: Option<String>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u64>>,
    estimate_rows: Option<Vec<u64>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
) -> PyResult<Vec<AteAnalysisResult>> {
    let data = tabular_from_py_columns(names.clone(), columns)?;
    let suite = refute.as_ref().map(|r| suite_from_refute(Some(r))).transpose()?;
    let latency_mode = parse_latency_mode(latency.as_deref())?;
    let parsed_queries = parse_ate_batch_query_specs(queries)?;
    let configured_spec =
        parse_batch_estimator_config(estimator_config.as_ref(), estimator.as_deref(), bootstrap)?;
    detach_catch(py, move || {
        let (batch, ate_queries) = compile_ate_batch(
            data,
            edges,
            parsed_queries,
            identifier,
            estimator,
            configured_spec,
            suite,
            bootstrap,
            latency_mode,
            screen_id,
            screen_procedure,
            screen_rows,
            estimate_rows,
            tiers,
            within_tier,
        )?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let results = batch.estimate_many(&ate_queries, &ctx).map_err(py_err)?;
        results.into_iter().map(|r| ate_result_from_analysis(&names, &r, false, &ctx)).collect()
    })
}

/// Pair lower and upper interval ends; unequal lengths are a broken result, never truncated.
fn parse_candidate_screen(
    screen_id: Option<String>,
    procedure: Option<String>,
    screen_rows: Option<Vec<u64>>,
    estimate_rows: Option<Vec<u64>>,
) -> PyResult<Option<antecedent::CandidateScreen>> {
    if screen_id.is_none()
        && procedure.is_none()
        && screen_rows.is_none()
        && estimate_rows.is_none()
    {
        return Ok(None);
    }
    let screen_id = screen_id.ok_or_else(|| {
        crate::value_err(
            "candidate screen requires screen_id, procedure, screen_rows, and estimate_rows",
        )
    })?;
    let procedure = match procedure.as_deref().unwrap_or("unrecorded") {
        "max_t" => antecedent::CandidateProcedure::MaxT,
        "bh" => antecedent::CandidateProcedure::BenjaminiHochberg,
        "by" => antecedent::CandidateProcedure::BenjaminiYekutieli,
        "unrecorded" => antecedent::CandidateProcedure::Unrecorded,
        other => {
            return Err(crate::value_err(format!(
                "unknown screen procedure {other:?}; use max_t|bh|by|unrecorded"
            )));
        }
    };
    Ok(Some(antecedent::CandidateScreen {
        screen_id: std::sync::Arc::from(screen_id),
        procedure,
        screen_rows: screen_rows
            .unwrap_or_default()
            .into_iter()
            .map(antecedent::data::GlobalId::new)
            .collect::<Vec<_>>()
            .into(),
        estimate_rows: estimate_rows
            .unwrap_or_default()
            .into_iter()
            .map(antecedent::data::GlobalId::new)
            .collect::<Vec<_>>()
            .into(),
    }))
}

/// Immutable multi-query preparation over one owned native table.
#[pyclass(name = "PreparedBatch")]
pub struct PyPreparedBatch {
    pub(crate) inner: std::sync::Arc<antecedent::PreparedBatch>,
    pub(crate) names: Vec<String>,
}

impl PyPreparedBatch {
    fn new(prepared: antecedent::PreparedBatch, names: Vec<String>) -> Self {
        Self { inner: std::sync::Arc::new(prepared), names }
    }
}

/// One immutable execution, with score ownership independent of later runs.
#[pyclass(name = "CompletedBatch")]
pub struct PyCompletedBatch {
    pub(crate) inner: std::sync::Arc<antecedent::CompletedBatch>,
    pub(crate) names: Vec<String>,
    seed: u64,
    threads: u32,
    metadata_context: antecedent::prelude::ExecutionContext,
}

#[pymethods]
impl PyPreparedBatch {
    fn n_plans(&self) -> usize {
        self.inner.plans().len()
    }

    fn shares_covariates(&self) -> bool {
        self.inner.shared_design().and_then(|d| d.covariate.as_ref()).is_some()
    }

    fn shared_n_folds(&self) -> Option<u32> {
        self.inner.shared_design().map(|d| d.n_folds)
    }

    fn shared_fold_ids(&self) -> Option<Vec<u32>> {
        self.inner.shared_design().map(|d| d.fold_ids.to_vec())
    }

    fn shared_adjustment_set(&self) -> Option<Vec<u32>> {
        self.inner
            .shared_design()
            .and_then(|d| d.covariate.as_ref())
            .map(|c| c.adjustment_set.iter().map(|v| v.raw()).collect())
    }

    fn materialized_plan_count(&self) -> usize {
        self.inner.materialized_plan_count()
    }

    fn retained_score_table_count(&self) -> usize {
        self.inner.retained_score_table_count()
    }

    #[pyo3(signature = (*, seed=1, threads=None, on_stage=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        on_stage: Option<Vec<Bound<'_, PyAny>>>,
    ) -> PyResult<PyCompletedBatch> {
        let inner = std::sync::Arc::clone(&self.inner);
        let sinks = on_stage
            .map(|callbacks| {
                callbacks
                    .iter()
                    .map(|callback| {
                        crate::callbacks::stage_sink_from_py(Some(callback))?.ok_or_else(|| {
                            crate::value_err("batch stage observer must be callable")
                        })
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?;
        let threads = crate::resolve_user_threads(threads);
        let (completed, metadata_context) = detach_catch(py, move || {
            let ctx = py_execution_context(seed, threads);
            let staged = sinks
                .map(|sinks| {
                    inner.with_stage_sinks(
                        sinks.into_iter().map(crate::interrupt::bind_stage_sink).collect(),
                    )
                })
                .transpose()
                .map_err(py_err)?;
            let source = staged.as_ref().unwrap_or(inner.as_ref());
            let completed = source.execute_retained(&ctx).map_err(py_err)?;
            Ok((completed, ctx))
        })?;
        Ok(PyCompletedBatch {
            inner: std::sync::Arc::new(completed),
            names: self.names.clone(),
            seed,
            threads,
            metadata_context,
        })
    }

    #[pyo3(signature = (names, columns, *, seed=1, threads=None))]
    fn rebind(
        &self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Self> {
        if names != self.names {
            return Err(crate::value_err(
                "prepared batch rebind requires its declared column order",
            ));
        }
        let data = tabular_from_py_columns(names.clone(), columns)?;
        let inner = std::sync::Arc::clone(&self.inner);
        let prepared = detach_catch(py, move || {
            let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
            inner.rebind_data(data, &ctx).map_err(py_err)
        })?;
        Ok(Self::new(prepared, names))
    }

    #[pyo3(signature = (*, seed=1, threads=None))]
    fn reconfigure_execution(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Self> {
        let inner = std::sync::Arc::clone(&self.inner);
        let prepared = detach_catch(py, move || {
            let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
            inner.reconfigure_execution(&ctx).map_err(py_err)
        })?;
        Ok(Self::new(prepared, self.names.clone()))
    }
}

#[pymethods]
impl PyCompletedBatch {
    fn bound_table(&self) -> PyResult<crate::data_snapshot::TableSnapshot> {
        Ok(crate::data_snapshot::TableSnapshot::from_native(
            self.inner.bound_table().map_err(py_err)?.clone(),
        ))
    }

    fn preparation(&self) -> PyPreparedBatch {
        PyPreparedBatch::new(self.inner.prepared().clone(), self.names.clone())
    }

    fn members(
        &self,
    ) -> PyResult<Vec<(AteAnalysisResult, crate::prepared_api::PyPreparedAnalysis)>> {
        self.inner
            .prepared()
            .plans()
            .iter()
            .zip(self.inner.results())
            .map(|(study, result)| {
                let raw =
                    ate_result_from_analysis(&self.names, result, false, &self.metadata_context)?;
                let execution = crate::prepared_api::PyPreparedAnalysis::from_completed(
                    std::sync::Arc::new(study.clone()),
                    std::sync::Arc::clone(result),
                    self.names.clone(),
                    self.seed,
                    self.threads,
                );
                Ok((raw, execution))
            })
            .collect()
    }
}

#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    edges,
    queries,
    *,
    identifier=None,
    estimator=None,
    estimator_config=None,
    refute=None,
    seed=1,
    bootstrap=None,
    threads=None,
    latency=None,
    screen_id=None,
    screen_procedure=None,
    screen_rows=None,
    estimate_rows=None,
    tiers=None,
    within_tier=None,
    data_diagnostics=None,
))]
fn prepare_ate_batch(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    queries: Vec<PyBatchQuery<'_>>,
    identifier: Option<String>,
    estimator: Option<String>,
    estimator_config: Option<Bound<'_, PyDict>>,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: Option<u32>,
    threads: Option<u32>,
    latency: Option<String>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u64>>,
    estimate_rows: Option<Vec<u64>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
    data_diagnostics: Option<String>,
) -> PyResult<PyPreparedBatch> {
    let data = tabular_from_py_columns(names.clone(), columns)?;
    let suite = refute.as_ref().map(|r| suite_from_refute(Some(r))).transpose()?;
    let latency_mode = parse_latency_mode(latency.as_deref())?;
    let parsed_queries = parse_ate_batch_query_specs(queries)?;
    let configured_spec =
        parse_batch_estimator_config(estimator_config.as_ref(), estimator.as_deref(), bootstrap)?;
    detach_catch(py, move || {
        let (batch, ate_queries) = compile_ate_batch(
            data,
            edges,
            parsed_queries,
            identifier,
            estimator,
            configured_spec,
            suite,
            bootstrap,
            latency_mode,
            screen_id,
            screen_procedure,
            screen_rows,
            estimate_rows,
            tiers,
            within_tier,
        )?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let batch = match data_diagnostics.as_deref() {
            Some(text) => {
                batch.data_diagnostics(crate::data_diagnostics_api::policy_setting(text)?)
            }
            None => batch,
        };
        let prepared = batch.prepare_deferred(&ate_queries, &ctx).map_err(py_err)?;
        Ok(PyPreparedBatch::new(prepared, names))
    })
}

#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    edges,
    queries,
    *,
    identifier=None,
    estimator=None,
    estimator_config=None,
    refute=None,
    seed=1,
    bootstrap=None,
    threads=None,
    latency=None,
    screen_id=None,
    screen_procedure=None,
    screen_rows=None,
    estimate_rows=None,
    tiers=None,
    within_tier=None,
    family_contrast="cell_minus_control",
    family_posterior_draws=None,
    data_diagnostics=None,
))]
fn prepare_cells_batch(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    queries: Vec<PyCellBatchQuery<'_>>,
    identifier: Option<String>,
    estimator: Option<String>,
    estimator_config: Option<Bound<'_, PyDict>>,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: Option<u32>,
    threads: Option<u32>,
    latency: Option<String>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u64>>,
    estimate_rows: Option<Vec<u64>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
    family_contrast: Option<&str>,
    family_posterior_draws: Option<usize>,
    data_diagnostics: Option<String>,
) -> PyResult<PyPreparedBatch> {
    let data = tabular_from_py_columns(names.clone(), columns)?;
    let suite = refute.as_ref().map(|r| suite_from_refute(Some(r))).transpose()?;
    let latency_mode = parse_latency_mode(latency.as_deref())?;
    let parsed_queries = parse_cell_batch_query_specs(queries)?;
    let configured_spec =
        parse_batch_estimator_config(estimator_config.as_ref(), estimator.as_deref(), bootstrap)?;
    let family_contrast = family_contrast.map(str::to_owned);
    detach_catch(py, move || {
        let (batch, cell_queries) = compile_cell_batch(
            data,
            edges,
            parsed_queries,
            identifier,
            estimator,
            configured_spec,
            suite,
            bootstrap,
            latency_mode,
            screen_id,
            screen_procedure,
            screen_rows,
            estimate_rows,
            tiers,
            within_tier,
            family_contrast,
        )?;
        let batch = batch.family_posterior(family_posterior_draws.map(|draws| {
            antecedent::estimate::score_posterior::ScorePosteriorOptions {
                draws,
                ..antecedent::estimate::score_posterior::ScorePosteriorOptions::default()
            }
        }));
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let batch = match data_diagnostics.as_deref() {
            Some(text) => {
                batch.data_diagnostics(crate::data_diagnostics_api::policy_setting(text)?)
            }
            None => batch,
        };
        let prepared = batch.prepare_cells_deferred(&cell_queries, &ctx).map_err(py_err)?;
        Ok(PyPreparedBatch::new(prepared, names))
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_ate_many, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_ate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_cells_batch, m)?)?;
    m.add_class::<PyPreparedBatch>()?;
    m.add_class::<PyCompletedBatch>()?;
    Ok(())
}
