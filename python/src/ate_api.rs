//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

use crate::graphs;
use crate::*;
use antecedent::AcceptedGraph;
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ResponseQuery, VariableId,
};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

pub(crate) fn parse_rd_config<F>(
    estimator: Option<&str>,
    running_variable: Option<&str>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    resolve_var: F,
) -> PyResult<Option<(VariableId, f64, f64)>>
where
    F: FnOnce(&str) -> PyResult<VariableId>,
{
    let wants_rd = estimator.is_some_and(|e| e.eq_ignore_ascii_case("rd.sharp"))
        || running_variable.is_some()
        || cutoff.is_some()
        || bandwidth.is_some();
    if !wants_rd {
        return Ok(None);
    }
    let (Some(rv), Some(cut), Some(bw)) = (running_variable, cutoff, bandwidth) else {
        return Err(crate::with_reason_code(
            crate::value_err(
                "rd.sharp (or any RD kwargs) requires running_variable, cutoff, and bandwidth",
            ),
            antecedent_core::reason_code!("required_option_missing"),
        ));
    };
    Ok(Some((resolve_var(rv)?, cut, bw)))
}

/// `rd.sharp` design from the resolved triple plus the optional configured SE kind
/// (`None` keeps [`antecedent::RdConfig`]'s HC1 default).
pub(crate) fn rd_design(
    running_variable: VariableId,
    cutoff: f64,
    bandwidth: f64,
    se_kind: Option<antecedent_estimate::AnalyticSeKind>,
) -> antecedent::RdConfig {
    let config = antecedent::RdConfig::new(running_variable, cutoff, bandwidth);
    match se_kind {
        Some(kind) => config.with_se_kind(kind),
        None => config,
    }
}

struct PpcFields {
    prior_ppc_p_value: Option<f64>,
    prior_ppc_observed: Option<f64>,
    prior_ppc_predictive_mean: Option<f64>,
    prior_ppc_predictive_sd: Option<f64>,
    prior_ppc_n_sims: Option<u32>,
    prior_ppc_dispersion_p_value: Option<f64>,
    prior_ppc_noise: Option<String>,
    posterior_ppc_p_value: Option<f64>,
    posterior_ppc_observed: Option<f64>,
    posterior_ppc_predictive_mean: Option<f64>,
    posterior_ppc_predictive_sd: Option<f64>,
    posterior_ppc_n_sims: Option<u32>,
    posterior_ppc_dispersion_p_value: Option<f64>,
    posterior_ppc_noise: Option<String>,
}

/// Stable wire name of how a predictive check's replicates carry observation noise.
fn predictive_noise_name(noise: antecedent_validate::PredictiveNoise) -> &'static str {
    match noise {
        antecedent_validate::PredictiveNoise::Likelihood => "likelihood",
        antecedent_validate::PredictiveNoise::MeanOnlyImproperScale => "mean_only_improper_scale",
    }
}

fn ppc_fields_from_checks(checks: &[antecedent_validate::PredictiveCheckReport]) -> PpcFields {
    let mut fields = PpcFields {
        prior_ppc_p_value: None,
        prior_ppc_observed: None,
        prior_ppc_predictive_mean: None,
        prior_ppc_predictive_sd: None,
        prior_ppc_n_sims: None,
        prior_ppc_dispersion_p_value: None,
        prior_ppc_noise: None,
        posterior_ppc_p_value: None,
        posterior_ppc_observed: None,
        posterior_ppc_predictive_mean: None,
        posterior_ppc_predictive_sd: None,
        posterior_ppc_n_sims: None,
        posterior_ppc_dispersion_p_value: None,
        posterior_ppc_noise: None,
    };
    for pc in checks {
        match pc.kind {
            PredictiveCheckKind::Prior => {
                fields.prior_ppc_p_value = Some(pc.p_value);
                fields.prior_ppc_observed = Some(pc.observed);
                fields.prior_ppc_predictive_mean = Some(pc.predictive_mean);
                fields.prior_ppc_predictive_sd = Some(pc.predictive_sd);
                fields.prior_ppc_n_sims = Some(pc.n_sims);
                fields.prior_ppc_dispersion_p_value = Some(pc.dispersion_p_value);
                fields.prior_ppc_noise = Some(predictive_noise_name(pc.noise).to_owned());
            }
            PredictiveCheckKind::Posterior => {
                fields.posterior_ppc_p_value = Some(pc.p_value);
                fields.posterior_ppc_observed = Some(pc.observed);
                fields.posterior_ppc_predictive_mean = Some(pc.predictive_mean);
                fields.posterior_ppc_predictive_sd = Some(pc.predictive_sd);
                fields.posterior_ppc_n_sims = Some(pc.n_sims);
                fields.posterior_ppc_dispersion_p_value = Some(pc.dispersion_p_value);
                fields.posterior_ppc_noise = Some(predictive_noise_name(pc.noise).to_owned());
            }
        }
    }
    fields
}

/// The encoded posterior artifact of a Bayesian study result, when one was produced.
fn posterior_artifact_from_result(result: &antecedent::StudyResult) -> PyResult<Option<Vec<u8>>> {
    result
        .posterior
        .as_ref()
        .map(|post| encode_causal_posterior_bytes(post, "ate-analysis").map_err(py_err))
        .transpose()
}

/// Prior-sensitivity sweep of a posterior, as the fields of the Python result.
#[derive(Default)]
struct PriorSensitivityFields {
    scales: Option<Vec<f64>>,
    alphas: Option<Vec<f64>>,
    means: Option<Vec<f64>>,
    sds: Option<Vec<f64>>,
    family: Option<String>,
    variance_multipliers: Option<Vec<f64>>,
}

fn prior_sensitivity_from_result(result: &antecedent::StudyResult) -> PriorSensitivityFields {
    let Some(sens) = result.posterior.as_ref().and_then(|post| post.prior_sensitivity.as_ref())
    else {
        return PriorSensitivityFields::default();
    };
    let non_empty = |values: Vec<f64>| if values.is_empty() { None } else { Some(values) };
    PriorSensitivityFields {
        scales: non_empty(sens.prior_scales.iter().copied().collect()),
        alphas: non_empty(sens.alphas.iter().copied().collect()),
        means: Some(sens.effect_means.iter().copied().collect()),
        sds: Some(sens.effect_sds.iter().copied().collect()),
        family: Some(sens.family.as_str().to_string()),
        variance_multipliers: non_empty(sens.variance_multipliers.iter().copied().collect()),
    }
}

/// Source-conflict summary of a posterior, as the fields of the Python result.
#[derive(Default)]
struct ConflictSummaryFields {
    source_ids: Option<Vec<String>>,
    alphas_requested: Option<Vec<f64>>,
    alphas_applied: Option<Vec<f64>>,
}

fn conflict_summary_from_result(result: &antecedent::StudyResult) -> ConflictSummaryFields {
    let Some(cs) = result.posterior.as_ref().and_then(|post| post.conflict_summary.as_ref()) else {
        return ConflictSummaryFields::default();
    };
    ConflictSummaryFields {
        source_ids: Some(cs.source_ids.iter().map(std::string::ToString::to_string).collect()),
        alphas_requested: Some(cs.alphas_requested.iter().copied().collect()),
        alphas_applied: Some(cs.alphas_applied.iter().copied().collect()),
    }
}

// Python batch query: treatment, outcome, control, active, functional specification,
// target population specification.
type PyBatchQuery<'py> =
    (String, String, f64, f64, Option<Bound<'py, PyDict>>, Option<Bound<'py, PyDict>>);

/// Run static ATE: identify → estimate → optional refute .
///
/// Parse optional `target_population` dict from Python (`kind` + fields).
pub(crate) fn parse_target_population(
    spec: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<TargetPopulation>> {
    let Some(d) = spec else {
        return Ok(None);
    };
    let kind: String = d
        .get_item("kind")?
        .ok_or_else(|| crate::value_err("target_population requires 'kind'"))?
        .extract()?;
    let kind = kind.to_ascii_lowercase();
    Ok(Some(match kind.as_str() {
        "all" | "all_observed" => TargetPopulation::AllObserved,
        "treated" => TargetPopulation::Treated,
        "untreated" => TargetPopulation::Untreated,
        "named" => {
            let name: String = d
                .get_item("name")?
                .ok_or_else(|| crate::value_err("named target_population requires 'name'"))?
                .extract()?;
            TargetPopulation::Predicate(PredicateExpr::named(name))
        }
        "rows" => {
            let rows: Vec<usize> = d
                .get_item("rows")?
                .ok_or_else(|| crate::value_err("rows target_population requires 'rows'"))?
                .extract()?;
            TargetPopulation::Predicate(PredicateExpr::rows(rows))
        }
        "custom_distribution" | "custom" => {
            let id: u32 = d
                .get_item("id")?
                .ok_or_else(|| {
                    crate::value_err("custom_distribution target_population requires 'id'")
                })?
                .extract()?;
            TargetPopulation::CustomDistribution(DistributionRef::from_raw(id))
        }
        other => {
            return Err(crate::value_err(format!("unknown target_population kind {other:?}")));
        }
    }))
}

pub(crate) fn parse_outcome_functional(
    spec: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<antecedent_core::OutcomeFunctional>> {
    let Some(d) = spec else {
        return Ok(None);
    };
    let kind: String = d
        .get_item("kind")?
        .ok_or_else(|| crate::value_err("outcome_functional requires 'kind'"))?
        .extract()?;
    Ok(Some(match kind.to_ascii_lowercase().as_str() {
        "mean" => antecedent_core::OutcomeFunctional::Mean,
        "exceedance" => {
            let threshold: f64 = d
                .get_item("threshold")?
                .ok_or_else(|| crate::value_err("exceedance requires 'threshold'"))?
                .extract()?;
            antecedent_core::OutcomeFunctional::exceedance(threshold)
        }
        "exceedance_grid" | "grid" => {
            let thresholds: Vec<f64> = d
                .get_item("thresholds")?
                .ok_or_else(|| crate::value_err("exceedance_grid requires 'thresholds'"))?
                .extract()?;
            antecedent_core::OutcomeFunctional::exceedance_grid(thresholds)
        }
        "quantile" => {
            let tau: f64 = d
                .get_item("tau")?
                .ok_or_else(|| crate::value_err("quantile requires 'tau'"))?
                .extract()?;
            antecedent_core::OutcomeFunctional::quantile(tau)
        }
        other => {
            return Err(crate::value_err(format!("unknown outcome_functional kind {other:?}")));
        }
    }))
}

/// Build a [`PopulationRegistry`] from optional predicate/distribution dicts.
pub(crate) fn parse_population_registry(
    predicates: Option<&Bound<'_, PyDict>>,
    distributions: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<PopulationRegistry>> {
    if predicates.is_none() && distributions.is_none() {
        return Ok(None);
    }
    let mut reg = PopulationRegistry::new();
    if let Some(preds) = predicates {
        for (k, v) in preds.iter() {
            let name: String = k.extract()?;
            let rows: Vec<usize> = v.extract()?;
            reg.insert_predicate(name, rows);
        }
    }
    if let Some(dists) = distributions {
        for (k, v) in dists.iter() {
            let id: u32 = k.extract()?;
            if let Ok(spec) = v.cast::<PyDict>() {
                let weights: Vec<f64> = spec
                    .get_item("weights")?
                    .ok_or_else(|| crate::value_err("missing weights"))?
                    .extract()?;
                let depends: Vec<u32> = spec
                    .get_item("depends_on")?
                    .ok_or_else(|| crate::value_err("missing depends_on"))?
                    .extract()?;
                reg.insert_distribution_with_dependence(
                    DistributionRef::from_raw(id),
                    weights,
                    depends
                        .into_iter()
                        .map(antecedent_core::VariableId::from_raw)
                        .collect::<Vec<_>>(),
                );
            } else {
                let weights: Vec<f64> = v.extract()?;
                reg.insert_distribution(DistributionRef::from_raw(id), weights);
            }
        }
    }
    Ok(Some(reg))
}

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
    Option<antecedent_core::OutcomeFunctional>,
    Option<antecedent_core::TargetPopulation>,
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
    data: antecedent_data::TabularData,
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
            antecedent_graph::TieredBackground::from_named(data.schema(), &named, within)
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
    data: antecedent_data::TabularData,
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
    functional: Option<antecedent_core::OutcomeFunctional>,
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
    data: antecedent_data::TabularData,
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
        let outcome_ids =
            crate::response_api::resolve_names(data.schema(), &[declaration.outcome.clone()])?;
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
            antecedent_core::DerivativeScale::Identity,
            antecedent_core::DerivativeWeighting::Observed,
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
    let (data, _) = tabular_from_py_columns(names.clone(), columns)?;
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

pub(crate) fn ate_result_from_analysis(
    names: &[String],
    result: &antecedent::StudyResult,
    include_posterior_artifact: bool,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<AteAnalysisResult> {
    let effect_authority =
        result.identification.average_effect().zip(result.estimate().as_effect()).map(
            |(query, effect)| crate::scalar_decision_api::NativeEffectAuthority {
                names: names.to_vec(),
                query: query.clone(),
                status: result.identification.status,
                ate: effect.ate,
                has_structural_response: result.structural_response.is_some(),
                support_status: result.support_status(),
                executed_contract: result.executed_contract().copied(),
            },
        );
    let effect = result.estimate().as_effect();
    let certificate_json = crate::identification_details::analysis_to_json(result, names, ctx)?;
    // A single-outcome counterfactual answers with its own unit result; one with several
    // outcomes or a joint baseline answers with contrasts, whose first outcome fills the
    // same fields so the single-outcome surface reads identically.
    let counterfactual_view = match (result.counterfactual(), result.counterfactual_contrasts()) {
        (Some(cf), _) => Some((
            &cf.unit_effects,
            cf.unit_effect_intervals.as_ref(),
            cf.unit_extrapolative.as_ref(),
        )),
        (None, Some(set)) => {
            let primary = set.primary();
            Some((
                &primary.unit_effects,
                primary.unit_effect_intervals.as_ref(),
                primary.unit_extrapolative.as_ref(),
            ))
        }
        (None, None) => None,
    };
    let estimator_id = result.logical_plan.estimator.as_deref().unwrap_or("").to_string();
    let posterior_artifact =
        if include_posterior_artifact { posterior_artifact_from_result(result)? } else { None };
    let sections = crate::shared_study_sections(names, result, estimator_id)?
        .with_posterior_artifact(posterior_artifact);
    let ppc = ppc_fields_from_checks(&result.predictive_checks);
    let PriorSensitivityFields {
        scales: prior_sensitivity_scales,
        alphas: prior_sensitivity_alphas,
        means: prior_sensitivity_means,
        sds: prior_sensitivity_sds,
        family: prior_sensitivity_family,
        variance_multipliers: prior_sensitivity_variance_multipliers,
    } = prior_sensitivity_from_result(result);
    let ConflictSummaryFields {
        source_ids: conflict_source_ids,
        alphas_requested: conflict_alphas_requested,
        alphas_applied: conflict_alphas_applied,
    } = conflict_summary_from_result(result);
    let SharedStudySections {
        identification,
        estimate,
        posterior,
        validation,
        performance,
        identified_set,
        identification_status,
        method,
        adjustment_set,
        estimator_id,
        plan_id,
        modality,
        refutations,
        evidence_status,
        allowlist_reason,
        allowlist_parent,
        structural_weight_basis,
        structural_identified_mass,
        structural_unidentified_mass,
        structural_unevaluable_mass,
        structural_aggregation_policy,
    } = sections;
    let mediation_slices: &[antecedent_estimate::TemporalMediationSlice] =
        result.mediation_grid().map_or(&[], |grid| grid.slices.as_ref());
    let mediation_uncertainty: Vec<_> =
        mediation_slices.iter().map(crate::mediation_uncertainty_projection).collect();

    Ok(AteAnalysisResult {
        structural_weight_basis,
        structural_identified_mass,
        structural_unidentified_mass,
        structural_unevaluable_mass,
        structural_aggregation_policy,
        structural_identified_set: identified_set.set,
        structural_identified_set_interval: identified_set.interval,
        structural_identified_set_interval_level: identified_set.level,
        structural_identified_set_interval_method: identified_set.method,
        structural_identified_set_interval_truncated: identified_set.truncated,
        certificate_json,
        missing_data_json: crate::missing_report::report_json(result.missing_data().as_ref()),
        interval_bindings_json: crate::interval_binding_api::bindings_json(result),
        ate: effect.map_or(f64::NAN, |estimate| estimate.ate),
        se_analytic: effect.map_or(f64::NAN, antecedent_estimate::EffectEstimate::se_analytic),
        se_bootstrap: effect.and_then(antecedent_estimate::EffectEstimate::se_bootstrap),
        bootstrap_replicates_failed: effect
            .and_then(|estimate| estimate.uncertainty.bootstrap.replicates_failed),
        adjustment_set,
        identification_status,
        refutation_passed: validation.passed,
        refutation_ran: validation.ran,
        refutation_count: validation.count,
        refutations,
        policy_value: result.policy_value().map(|value| {
            let mut section = crate::policy_api::PolicyValueSection::from(value);
            section.graphless_support_status = result
                .support_status()
                .map(antecedent::support::CellStatus::as_str)
                .map(str::to_string);
            section
        }),
        continuous_dose_response: result.continuous_dose_response().map(|value| {
            let mut section = crate::policy_api::ContinuousDoseResponseSection::from(value);
            section.graphless_support_status = result
                .support_status()
                .map(antecedent::support::CellStatus::as_str)
                .map(str::to_string);
            section
        }),
        assumption_count: result.identification.required_assumptions.len(),
        derivation_step_count: result.identification.derivation.steps.len(),
        method,
        estimator_id,
        overlap_ess: estimate.overlap_ess,
        overlap_propensity_min: estimate.overlap_propensity_min,
        posterior_effect_mean: posterior.effect_mean,
        posterior_effect_sd: posterior.effect_sd,
        posterior_q025: posterior.q025,
        posterior_q975: posterior.q975,
        posterior_n_draws: posterior.n_draws,
        posterior_p_below_zero: posterior.p_below_zero,
        posterior_backend: posterior.backend.clone(),
        posterior_artifact: posterior.artifact.clone(),
        diagnostics: result
            .diagnostics
            .iter()
            .map(|d| format!("{}: {}", d.code, d.message))
            .collect(),
        route_diagnostics: result
            .diagnostics
            .iter()
            .filter_map(antecedent_core::diagnostic::RouteNativeDiagnostic::from_diagnostic)
            .map(|record| {
                (
                    record.route_name().to_owned(),
                    record.outcome_name().to_owned(),
                    record.failed(),
                    record.unevaluable_mass().get(),
                )
            })
            .collect(),
        provenance_node_count: result.provenance.len(),
        plan_id,
        modality,
        discovery_algorithm: result
            .logical_plan
            .discovery_algorithm
            .as_ref()
            .map(std::string::ToString::to_string),
        structure_source: result.structure_source.as_str().to_string(),
        graph_review_required: result.logical_plan.graph_review_required,
        plan_identifier: result
            .logical_plan
            .identifier
            .as_ref()
            .map(std::string::ToString::to_string),
        plan_estimator: result
            .logical_plan
            .estimator
            .as_ref()
            .map(std::string::ToString::to_string),
        validation_suite: result
            .logical_plan
            .validation_suite
            .as_ref()
            .map(std::string::ToString::to_string),
        peak_memory_bytes: result.physical_plan.estimated_peak_memory_bytes,
        worker_threads: result.physical_plan.worker_threads,
        expected_python_crossings: result.physical_plan.expected_python_crossings,
        prior_ppc_p_value: ppc.prior_ppc_p_value,
        prior_ppc_observed: ppc.prior_ppc_observed,
        prior_ppc_predictive_mean: ppc.prior_ppc_predictive_mean,
        prior_ppc_predictive_sd: ppc.prior_ppc_predictive_sd,
        prior_ppc_n_sims: ppc.prior_ppc_n_sims,
        prior_ppc_dispersion_p_value: ppc.prior_ppc_dispersion_p_value,
        prior_ppc_noise: ppc.prior_ppc_noise,
        posterior_ppc_p_value: ppc.posterior_ppc_p_value,
        posterior_ppc_observed: ppc.posterior_ppc_observed,
        posterior_ppc_predictive_mean: ppc.posterior_ppc_predictive_mean,
        posterior_ppc_predictive_sd: ppc.posterior_ppc_predictive_sd,
        posterior_ppc_n_sims: ppc.posterior_ppc_n_sims,
        posterior_ppc_dispersion_p_value: ppc.posterior_ppc_dispersion_p_value,
        posterior_ppc_noise: ppc.posterior_ppc_noise,
        predictive_replicates: result
            .predictive_replicates()
            .map(crate::predictive_assessment_api::PredictiveReplicateLaws::from),
        prior_sensitivity_scales,
        prior_sensitivity_alphas,
        prior_sensitivity_means,
        prior_sensitivity_sds,
        prior_sensitivity_family,
        prior_sensitivity_variance_multipliers,
        conflict_source_ids,
        conflict_alphas_requested,
        conflict_alphas_applied,
        posterior_unidentified_mass: posterior.unidentified_mass,
        posterior_subsampled_out_mass: posterior.subsampled_out_mass,
        latency_mode: performance.latency_mode.clone(),
        wall_time_ns: result.performance.wall_time_ns,
        bootstrap_replicates_requested: result.performance.bootstrap_replicates_requested,
        bootstrap_replicates_ok: performance.bootstrap_replicates_ok,
        n_draws_effort: result.performance.n_draws,
        cancelled: performance.cancelled,
        early_stopped: result.performance.early_stopped,
        stage_timings: performance.stage_timings.clone(),
        identification,
        estimate,
        posterior,
        validation,
        performance,
        assumptions: effect
            .map_or(&result.identification.required_assumptions, |estimate| {
                &estimate.provenance.assumptions
            })
            .entries
            .iter()
            .map(|r| format!("{:?}", r.assumption))
            .collect(),
        support_diagnostics: result.support_diagnostics().map(|d| d.message.to_string()).collect(),
        unit_effects: counterfactual_view.as_ref().map(|view| view.0.to_vec()),
        unit_effect_intervals: counterfactual_view
            .as_ref()
            .and_then(|view| view.1)
            .map(|i| i.lower.iter().copied().zip(i.upper.iter().copied()).collect()),
        unit_effect_intervals_level: counterfactual_view
            .as_ref()
            .and_then(|view| view.1)
            .map(|i| i.level),
        unit_effect_intervals_method: counterfactual_view
            .as_ref()
            .and_then(|view| view.1)
            .map(|i| i.method.to_owned()),
        unit_extrapolative: counterfactual_view
            .as_ref()
            .and_then(|view| view.2)
            .map(|flags| flags.to_vec()),
        counterfactual_contrasts: result.counterfactual_contrasts().map(|set| {
            set.contrasts
                .iter()
                .map(|contrast| {
                    (
                        names
                            .get(contrast.outcome.as_usize())
                            .cloned()
                            .unwrap_or_else(|| contrast.outcome.as_usize().to_string()),
                        contrast.mean_ite,
                        contrast.standard_error,
                        contrast.unit_effects.to_vec(),
                        contrast.unit_effect_intervals.as_ref().map(|i| {
                            i.lower.iter().copied().zip(i.upper.iter().copied()).collect()
                        }),
                        contrast.unit_extrapolative.as_ref().map(|flags| flags.to_vec()),
                        contrast.unit_effects_homogeneous,
                    )
                })
                .collect()
        }),
        mediation_total: result.mediation().and_then(|m| m.total),
        mediation_direct: result.mediation().and_then(|m| m.direct),
        mediation_mediated: result.mediation().and_then(|m| m.mediated),
        mediation_horizons: mediation_slices.iter().map(|slice| slice.horizon).collect(),
        mediation_effects: mediation_slices.iter().map(|slice| slice.estimate.effect.ate).collect(),
        mediation_totals: mediation_slices
            .iter()
            .map(|slice| slice.estimate.total.unwrap_or(f64::NAN))
            .collect(),
        mediation_directs: mediation_slices
            .iter()
            .map(|slice| slice.estimate.direct.unwrap_or(f64::NAN))
            .collect(),
        mediation_mediated_effects: mediation_slices
            .iter()
            .map(|slice| slice.estimate.mediated.unwrap_or(f64::NAN))
            .collect(),
        mediation_identification_statuses: mediation_slices
            .iter()
            .map(|slice| format!("{:?}", slice.identification_status))
            .collect(),
        mediation_methods: mediation_slices.iter().map(|slice| slice.method.to_string()).collect(),
        mediation_adjustments: mediation_slices
            .iter()
            .map(|slice| {
                slice.adjustment.iter().map(|key| (key.variable.raw(), key.offset)).collect()
            })
            .collect(),
        mediation_uncertainty_kinds: mediation_uncertainty
            .iter()
            .map(|row| row.0.clone())
            .collect(),
        mediation_standard_deviations: mediation_uncertainty.iter().map(|row| row.1).collect(),
        mediation_q025: mediation_uncertainty.iter().map(|row| row.2).collect(),
        mediation_q975: mediation_uncertainty.iter().map(|row| row.3).collect(),
        mediation_identified_lower: mediation_slices
            .iter()
            .map(|slice| slice.identified_set.map(|set| set.lower))
            .collect(),
        mediation_identified_upper: mediation_slices
            .iter()
            .map(|slice| slice.identified_set.map(|set| set.upper))
            .collect(),
        mediation_joint_posterior: result.mediation_grid().map(|grid| grid.joint_posterior),
        transport: result
            .transport()
            .map(crate::transport_interference_api::TransportSection::from_estimate),
        interference: result.interference().map(|estimate| {
            crate::transport_interference_api::InterferenceSection::from_estimate(
                estimate,
                &result.diagnostics,
                result.interference_inference(),
            )
        }),
        randomized_effect: result.randomized_effect().map(|value| {
            let mut section = crate::experiment_api::RandomizedEffectSection::from(value);
            section.graphless_support_status = result
                .support_status()
                .map(antecedent::support::CellStatus::as_str)
                .map(str::to_string);
            section
        }),
        panel_did: result.panel_did().map(|value| {
            let mut section = crate::quasi_api::PanelDidSection::from(value);
            section.graphless_support_status = result
                .support_status()
                .map(antecedent::support::CellStatus::as_str)
                .map(str::to_string);
            section
        }),
        synthetic_control: result.synthetic_control().map(Into::into),
        synthetic_did: result.synthetic_did().map(Into::into),
        local_polynomial_ratio: result.local_polynomial_ratio().map(|value| {
            let mut section = crate::quasi_api::LocalPolynomialRatioSection::from(value);
            section.graphless_support_status = result
                .support_status()
                .map(antecedent::support::CellStatus::as_str)
                .map(str::to_string);
            section
        }),
        survival: result.survival().map(Into::into),
        posterior_row: result.posterior_row().map(Into::into),
        longitudinal_regime: result.longitudinal_regime().map(Into::into),
        sequential_contrast: result.sequential_contrast().map(Into::into),
        history_match: result.history_match().map(Into::into),
        temporal_design: result.temporal_design().map(Into::into),
        iv_bounds: result.iv_bounds().map(Into::into),
        association: result.association().map(Into::into),
        quasi_rollout: result.quasi_rollout().map(Into::into),
        anomaly: result
            .anomaly()
            .map(|scores| crate::gcm_api::anomaly_scores_from_rust(scores.iter().cloned(), names)),
        change_attribution: result.change_attribution().map(|change| {
            crate::gcm_api::change_result_from_rust(change.clone(), names)
                .with_compiled_study_origin()
        }),
        evidence_status,
        allowlist_reason,
        allowlist_parent,
        effect_authority,
    })
}

/// One marked edge from an oriented temporal CPDAG/PAG.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct GraphEdge {
    #[pyo3(get)]
    pub(crate) source: String,
    #[pyo3(get)]
    pub(crate) source_lag: u32,
    #[pyo3(get)]
    pub(crate) target: String,
    #[pyo3(get)]
    pub(crate) target_lag: u32,
    /// Endpoint mark at `source`: `tail` | `arrow` | `circle` | `conflict`.
    #[pyo3(get)]
    pub(crate) at_source: String,
    /// Endpoint mark at `target`: `tail` | `arrow` | `circle` | `conflict`.
    #[pyo3(get)]
    pub(crate) at_target: String,
}

/// Identify-only on a static ADMG (no estimation).
///
/// An ADMG carries bidirected edges, so it is the only static graph type that
/// can state "these two variables share an unmeasured common cause". Without
/// this entry point an unobserved confounder had to be flattened into a DAG
/// before identification, where it looks like an ordinary adjustable node and
/// the effect is reported as identified by adjusting on a variable no study
/// can measure.
#[pyfunction(name = "_identify_ate_admg")]
#[pyo3(signature = (names, graph, treatment, outcome, *, identifier=None))]
fn identify_ate_admg(
    py: Python<'_>,
    names: Vec<String>,
    graph: graphs::Admg,
    treatment: String,
    outcome: String,
    identifier: Option<String>,
) -> PyResult<(String, String, Vec<String>)> {
    detach_catch(py, move || {
        let zeros = [0.0_f64, 1.0];
        let pairs: Vec<(&str, &[f64])> =
            names.iter().map(|n| (n.as_str(), zeros.as_slice())).collect();
        let data = antecedent_data::TabularData::from_f64_columns(pairs).map_err(py_err)?;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let admg = graph.aligned_to_names(&names)?;
        let mut builder = Study::tabular(data)
            .graph(admg)
            .query(AverageEffectQuery::binary_ate(t_id, y_id))
            .refute(RefuteSuite::None);
        if let Some(id) = identifier {
            builder = builder.identifier(
                id.parse::<antecedent::IdentifierId>()
                    .map_err(|e| crate::value_err(e.to_string()))?,
            );
        }
        let analysis = builder.build().map_err(py_err)?;
        let id_res = analysis.identify_only().map_err(py_err)?;
        let status = format!("{:?}", id_res.status);
        let method = id_res.estimands.first().map(|e| e.method.to_string()).unwrap_or_default();
        let adjustment: Vec<String> = id_res
            .estimands
            .first()
            .map(|e| {
                e.adjustment_set
                    .iter()
                    .filter_map(|vid| names.get(vid.as_usize()).cloned())
                    .collect()
            })
            .unwrap_or_default();
        Ok((status, method, adjustment))
    })
}

/// Identify-only on a static DAG (no estimation).
#[pyfunction(name = "_identify_ate")]
#[pyo3(signature = (names, edges, treatment, outcome, *, identifier=None))]
fn identify_ate(
    py: Python<'_>,
    names: Vec<String>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    identifier: Option<String>,
) -> PyResult<(String, String, Vec<String>)> {
    detach_catch(py, move || {
        let zeros = [0.0_f64, 1.0];
        let pairs: Vec<(&str, &[f64])> =
            names.iter().map(|n| (n.as_str(), zeros.as_slice())).collect();
        let data = antecedent_data::TabularData::from_f64_columns(pairs).map_err(py_err)?;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let dag = dag_from_named_edges(data.schema(), &edges)?;
        let mut builder = Study::tabular(data)
            .graph(dag)
            .query(AverageEffectQuery::binary_ate(t_id, y_id))
            .refute(RefuteSuite::None);
        if let Some(id) = identifier {
            builder = builder.identifier(
                id.parse::<antecedent::IdentifierId>()
                    .map_err(|e| crate::value_err(e.to_string()))?,
            );
        }
        let analysis = builder.build().map_err(py_err)?;
        let id_res = analysis.identify_only().map_err(py_err)?;
        let status = format!("{:?}", id_res.status);
        let method = id_res.estimands.first().map(|e| e.method.to_string()).unwrap_or_default();
        let adjustment: Vec<String> = id_res
            .estimands
            .first()
            .map(|e| {
                e.adjustment_set
                    .iter()
                    .filter_map(|vid| names.get(vid.as_usize()).cloned())
                    .collect()
            })
            .unwrap_or_default();
        Ok((status, method, adjustment))
    })
}

fn structure_var_id(names: &[String], name: &str) -> PyResult<VariableId> {
    names
        .iter()
        .position(|n| n == name)
        .and_then(|i| u32::try_from(i).ok())
        .map(VariableId::from_raw)
        .ok_or_else(|| crate::value_err(format!("unknown variable {name:?}")))
}

fn structure_query(
    kind: &str,
    names: &[String],
    treatment: &str,
    outcome: &str,
    modifiers: Option<&[String]>,
    policy: Option<&str>,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    window: Option<(i32, i32)>,
    treatments: Option<Vec<String>>,
) -> PyResult<CausalQuery> {
    let t_id = structure_var_id(names, treatment)?;
    let y_id = structure_var_id(names, outcome)?;
    match kind {
        "average" | "average_effect" => {
            Ok(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(t_id, y_id)))
        }
        "intervention_response" => {
            use antecedent_core::{ResponseFunctional, ResponseQuery};
            let targets = treatments.unwrap_or_else(|| vec![treatment.to_string()]);
            let interventions = targets
                .iter()
                .map(|name| {
                    structure_var_id(names, name).map(|id| Intervention::set(id, Value::f64(1.0)))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(CausalQuery::Response(ResponseQuery::new(
                ResponseFunctional::InterventionResponse {
                    outcome: y_id,
                    interventions: interventions.into(),
                },
            )))
        }
        "response" | "response_curve" => {
            use antecedent_core::{ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery};
            Ok(CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: y_id,
                treatment: ContinuousDomain::new(
                    t_id,
                    GridSpec::Values(std::sync::Arc::from([0.0, 1.0])),
                ),
            })))
        }
        "conditional" | "conditional_effect" => {
            let modifiers = modifiers.filter(|m| !m.is_empty()).ok_or_else(|| {
                crate::value_err("ConditionalEffect identify requires modifiers=")
            })?;
            let w_ids = modifiers
                .iter()
                .map(|name| structure_var_id(names, name))
                .collect::<PyResult<Vec<_>>>()?;
            let inner = AverageEffectQuery::binary_ate(t_id, y_id).with_effect_modifiers(w_ids);
            let cq = ConditionalEffectQuery::try_new(inner)
                .map_err(|e| crate::value_err(e.to_string()))?;
            Ok(CausalQuery::ConditionalEffect(cq))
        }
        "pulse" | "sustained" | "temporal_effect" => {
            let policy = policy.unwrap_or(if kind == "sustained" { "sustained" } else { "pulse" });
            let mut query = crate::temporal_api::temporal_query_from_policy(
                policy,
                t_id,
                y_id,
                treatment_lag,
                horizon_steps,
                active_level,
            )?;
            if let Some((from, until)) = window {
                if !matches!(query.policy, TemporalPolicy::Sustained { .. }) {
                    return Err(crate::value_err("window requires a sustained policy"));
                }
                query.policy = TemporalPolicy::sustained(from, until);
            }
            Ok(CausalQuery::TemporalEffect(query))
        }
        "temporal_response" => {
            use antecedent_core::{
                ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TemporalResponseSpec,
            };
            let mut query = ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: y_id,
                treatment: ContinuousDomain::new(
                    t_id,
                    GridSpec::Values(std::sync::Arc::from([0.0, 1.0])),
                ),
            });
            let temporal_policy =
                crate::temporal_license::policy_at_lag(policy.unwrap_or("pulse"), treatment_lag)?;
            query.temporal = Some(
                TemporalResponseSpec::new(vec![horizon_steps.max(1)], temporal_policy, None)
                    .map_err(|error| crate::value_err(error.to_string()))?,
            );
            Ok(CausalQuery::Response(query))
        }
        "temporal_mediation" => {
            use antecedent_core::{MediationContrast, MediationQuery};
            let mediators = treatments
                .unwrap_or_default()
                .iter()
                .map(|name| structure_var_id(names, name))
                .collect::<PyResult<Vec<_>>>()?;
            let contrast = match policy.unwrap_or("mediated") {
                "total" => MediationContrast::Total,
                "direct" => MediationContrast::Direct,
                "mediated" | "indirect" => MediationContrast::Mediated,
                other => {
                    return Err(crate::value_err(format!(
                        "unknown temporal mediation contrast {other}"
                    )));
                }
            };
            Ok(CausalQuery::Mediation(
                MediationQuery::binary(t_id, y_id, mediators, contrast)
                    .with_horizons([horizon_steps])
                    .map_err(py_msg)?,
            ))
        }
        other => Err(crate::value_err(format!("identify_structure unknown query kind {other:?}"))),
    }
}

fn identification_tuple(
    identification: &antecedent::Identification,
    names: &[String],
) -> (String, String, Vec<String>) {
    let status = format!("{:?}", identification.status());
    let method = identification.strategy().as_str().to_string();
    let indexer = match identification {
        antecedent::Identification::Point { temporal_indexer, .. } => temporal_indexer.as_ref(),
        antecedent::Identification::TemporalEnvelope { envelope, .. } => envelope.indexers.first(),
        _ => None,
    };
    let adjustment = identification
        .estimands()
        .first()
        .map(|e| {
            e.adjustment_set
                .iter()
                .filter_map(|id| {
                    let variable =
                        indexer.and_then(|i| i.key_of(id.raw()).ok()).map_or(*id, |k| k.variable);
                    names.get(variable.as_usize()).cloned()
                })
                .collect()
        })
        .unwrap_or_default();
    (status, method, adjustment)
}

/// Identify without data against a typed graph class.
#[pyfunction(name = "_identify_structure")]
#[pyo3(signature = (
    graph,
    query_kind,
    treatment,
    outcome,
    *,
    identifier=None,
    modifiers=None,
    policy=None,
    treatment_lag=1,
    horizon_steps=1,
    active_level=1.0,
    window=None,
    treatments=None,
    include_details=false,
    horizons=None,
    max_history_lag=None,
    response_grid=None,
    response_steps=None,
))]
#[allow(clippy::too_many_arguments)]
fn identify_structure(
    py: Python<'_>,
    graph: Bound<'_, PyAny>,
    query_kind: String,
    treatment: String,
    outcome: String,
    identifier: Option<String>,
    modifiers: Option<Vec<String>>,
    policy: Option<String>,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    window: Option<(i32, i32)>,
    treatments: Option<Vec<String>>,
    include_details: bool,
    horizons: Option<Vec<u32>>,
    max_history_lag: Option<u32>,
    response_grid: Option<Vec<f64>>,
    response_steps: Option<Vec<(String, String, Vec<f64>)>>,
) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    let dag = graph.extract::<graphs::Dag>().ok();
    let cpdag = graph.extract::<graphs::Cpdag>().ok();
    let pag = graph.extract::<graphs::Pag>().ok();
    let tdag = graph.extract::<graphs::TemporalDag>().ok();
    let tcpdag = graph.extract::<graphs::TemporalCpdag>().ok();
    let tpag = graph.extract::<graphs::TemporalPag>().ok();
    let (summary, details) = detach_catch(py, move || {
        let (structure, names) = if let Some(g) = dag {
            (AcceptedGraph::from(g.dag), g.names)
        } else if let Some(g) = cpdag {
            (AcceptedGraph::from(g.cpdag), g.names)
        } else if let Some(g) = pag {
            (AcceptedGraph::from(g.pag), g.names)
        } else if let Some(g) = tdag {
            (AcceptedGraph::from(g.dag), g.names)
        } else if let Some(g) = tcpdag {
            (AcceptedGraph::from(g.cpdag), g.names)
        } else if let Some(g) = tpag {
            (AcceptedGraph::from(g.pag), g.names)
        } else {
            return Err(crate::value_err(
                "identify_structure requires a Dag, Cpdag, Pag, TemporalDag, TemporalCpdag, or TemporalPag",
            ));
        };
        let mut query = structure_query(
            &query_kind,
            &names,
            &treatment,
            &outcome,
            modifiers.as_deref(),
            policy.as_deref(),
            treatment_lag,
            horizon_steps,
            active_level,
            window,
            treatments,
        )?;
        match &mut query {
            CausalQuery::TemporalEffect(temporal) => {
                temporal.max_history_lag = max_history_lag;
            }
            CausalQuery::Response(response) if response.temporal.is_some() => {
                let Some(temporal) = response.temporal.as_mut() else {
                    return Err(py_msg("temporal response is missing"));
                };
                if let Some(horizons) = horizons {
                    *temporal = antecedent_core::TemporalResponseSpec::new(
                        horizons,
                        temporal.policy.clone(),
                        max_history_lag,
                    )
                    .map_err(py_msg)?;
                } else {
                    temporal.max_history_lag = max_history_lag;
                }
                if let Some(grid) = response_grid {
                    if let antecedent_core::ResponseFunctional::MeanCurve { treatment, .. } =
                        &mut response.functional
                    {
                        treatment.grid = antecedent_core::GridSpec::Values(grid.into());
                    }
                }
                if let Some(steps) = response_steps {
                    let treatment_ids = steps
                        .iter()
                        .map(|(name, _, _)| structure_var_id(&names, name))
                        .collect::<PyResult<Vec<_>>>()?;
                    let kinds = steps.iter().map(|(_, kind, _)| kind.clone()).collect();
                    let parameters = steps.into_iter().map(|(_, _, values)| values).collect();
                    let functional = crate::response_api::build_functional(
                        "intervention_response",
                        &treatment_ids,
                        &[structure_var_id(&names, &outcome)?],
                        None,
                        None,
                        None,
                        Some(kinds),
                        Some(parameters),
                        1,
                        antecedent_core::DerivativeScale::Identity,
                        antecedent_core::DerivativeWeighting::Observed,
                    )?;
                    response.functional = crate::response_api::wrap_temporal_sequence_steps(
                        functional,
                        temporal.treatment_offset().map_err(py_msg)?,
                    )?;
                }
            }
            CausalQuery::Mediation(mediation) => {
                if let Some(horizons) = horizons {
                    *mediation = mediation.clone().with_horizons(horizons).map_err(py_msg)?;
                }
            }
            _ => {}
        }
        // The compiled study over a table with the graph's variables and no rows: identification
        // reads structure and query only, so the answer is the one a study over the data derives.
        let empty: &[f64] = &[];
        let columns = names.iter().map(|name| (name.as_str(), empty));
        let temporal = matches!(
            structure.class(),
            antecedent::GraphClass::TemporalDag
                | antecedent::GraphClass::TemporalCpdag
                | antecedent::GraphClass::TemporalPag
                | antecedent::GraphClass::TemporalAdmg
        );
        let builder = if temporal {
            Study::series(
                antecedent_data::TimeSeriesData::from_f64_columns(columns, 1).map_err(py_err)?,
            )
        } else {
            Study::tabular(antecedent_data::TabularData::from_f64_columns(columns).map_err(py_err)?)
        };
        let mut builder = builder.graph(structure.clone()).query(query.clone());
        if let Some(id) = identifier {
            let strategy = id
                .parse::<antecedent::IdentifierId>()
                .map_err(|e| crate::value_err(e.to_string()))?;
            builder = builder.identifier(strategy);
        }
        let ctx = py_execution_context(0, crate::resolve_user_threads(None));
        let identification = builder.identify_structure_within(&ctx).map_err(py_err)?;
        let details = if include_details {
            Some(crate::identification_details::to_json(
                &identification,
                &query,
                &names,
                structure.class().as_str(),
                None,
                &ctx,
            )?)
        } else {
            None
        };
        Ok((identification_tuple(&identification, &names), details))
    })?;
    match details {
        Some(json) => json.into_py_any(py),
        None => summary.into_py_any(py),
    }
}

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
            .map(antecedent_data::GlobalId::new)
            .collect::<Vec<_>>()
            .into(),
        estimate_rows: estimate_rows
            .unwrap_or_default()
            .into_iter()
            .map(antecedent_data::GlobalId::new)
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
    metadata_context: antecedent_core::ExecutionContext,
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
        let (data, _) = tabular_from_py_columns(names.clone(), columns)?;
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
    let (data, _) = tabular_from_py_columns(names.clone(), columns)?;
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
    let (data, _) = tabular_from_py_columns(names.clone(), columns)?;
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
            antecedent_estimate::score_posterior::ScorePosteriorOptions {
                draws,
                ..antecedent_estimate::score_posterior::ScorePosteriorOptions::default()
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_ate_many, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_ate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_cells_batch, m)?)?;
    m.add_class::<PyPreparedBatch>()?;
    m.add_class::<PyCompletedBatch>()?;
    m.add_function(wrap_pyfunction!(identify_ate, m)?)?;
    m.add_function(wrap_pyfunction!(identify_ate_admg, m)?)?;
    m.add_function(wrap_pyfunction!(identify_structure, m)?)?;
    Ok(())
}
