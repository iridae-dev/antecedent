//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

use crate::*;
use antecedent::{AcceptedGraph, StudyBuilder};
use antecedent_graph::Dag;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

fn bind_dag(builder: StudyBuilder, dag: Dag, accepted: bool) -> StudyBuilder {
    if accepted { builder.graph(AcceptedGraph::from(dag)) } else { builder.graph(dag) }
}

fn parse_rd_config<F>(
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
        return Err(PyValueError::new_err(
            "rd.sharp (or any RD kwargs) requires running_variable, cutoff, and bandwidth",
        ));
    };
    Ok(Some((resolve_var(rv)?, cut, bw)))
}

fn parse_prior_mapping(
    prior_mapping: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<antecedent_io::PriorMapping>> {
    match prior_mapping {
        None => Ok(None),
        Some(d) => Ok(Some(crate::prior_bank::mapping_from_dict(d)?)),
    }
}

fn run_static_ate_from_builder(
    names: &[String],
    mut builder: antecedent::StudyBuilder,
    inference: Option<&str>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<&[u8]>,
    prior_mapping: Option<antecedent_io::PriorMapping>,
    composed_prior: Option<crate::prior_bank::OwnedComposedPrior>,
    seed: u64,
    threads: u32,
    cancel: Option<antecedent_core::CancellationToken>,
    progress: Option<std::sync::Arc<dyn antecedent_core::ProgressSink>>,
    include_posterior_artifact: bool,
    bytes_borrowed: Option<u64>,
) -> PyResult<AteAnalysisResult> {
    if let Some(mode) = inference {
        let mut cfg = match mode.to_ascii_lowercase().as_str() {
            "bayesian" | "bayesian.laplace" | "laplace" => {
                BayesianConfig::laplace().n_draws(n_draws).prior_scale(prior_scale)
            }
            "bayesian.conjugate" | "conjugate" => {
                BayesianConfig::conjugate().n_draws(n_draws).prior_scale(prior_scale)
            }
            "bayesian.hmc" | "hmc" => {
                BayesianConfig::hmc().n_draws(n_draws).prior_scale(prior_scale)
            }
            "frequentist" => {
                builder = builder.inference(InferenceMode::Frequentist);
                let analysis = builder.build().map_err(py_err)?;
                let ctx = py_execution_context_ext(
                    seed,
                    threads,
                    cancel.clone(),
                    progress.clone(),
                    Some(PY_DEFAULT_CACHE_MAX_BYTES),
                );
                let mut result = analysis.run(&ctx).map_err(py_err)?;
                if let Some(n) = bytes_borrowed {
                    result.performance.bytes_borrowed = Some(n);
                }
                return ate_result_from_analysis(names, result, include_posterior_artifact);
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown inference mode {other:?}; use frequentist|bayesian|conjugate|hmc"
                )));
            }
        };
        if let Some(comp) = composed_prior {
            cfg = crate::prior_bank::apply_owned_composed_prior(cfg, comp)?;
        } else if let Some(bytes) = prior_artifact {
            cfg = cfg.prior_from_artifact(bytes.to_vec(), prior_mapping);
        }
        builder = builder.inference(InferenceMode::Bayesian(cfg));
    }
    let analysis = builder.build().map_err(py_err)?;
    let ctx =
        py_execution_context_ext(seed, threads, cancel, progress, Some(PY_DEFAULT_CACHE_MAX_BYTES));
    let mut result = analysis.run(&ctx).map_err(py_err)?;
    if let Some(n) = bytes_borrowed {
        result.performance.bytes_borrowed = Some(n);
    }
    ate_result_from_analysis(names, result, include_posterior_artifact)
}

struct PpcFields {
    prior_ppc_p_value: Option<f64>,
    prior_ppc_observed: Option<f64>,
    prior_ppc_predictive_mean: Option<f64>,
    prior_ppc_predictive_sd: Option<f64>,
    prior_ppc_n_sims: Option<u32>,
    posterior_ppc_p_value: Option<f64>,
    posterior_ppc_observed: Option<f64>,
    posterior_ppc_predictive_mean: Option<f64>,
    posterior_ppc_predictive_sd: Option<f64>,
    posterior_ppc_n_sims: Option<u32>,
}

fn ppc_fields_from_checks(checks: &[antecedent_validate::PredictiveCheckReport]) -> PpcFields {
    let mut fields = PpcFields {
        prior_ppc_p_value: None,
        prior_ppc_observed: None,
        prior_ppc_predictive_mean: None,
        prior_ppc_predictive_sd: None,
        prior_ppc_n_sims: None,
        posterior_ppc_p_value: None,
        posterior_ppc_observed: None,
        posterior_ppc_predictive_mean: None,
        posterior_ppc_predictive_sd: None,
        posterior_ppc_n_sims: None,
    };
    for pc in checks {
        match pc.kind {
            PredictiveCheckKind::Prior => {
                fields.prior_ppc_p_value = Some(pc.p_value);
                fields.prior_ppc_observed = Some(pc.observed);
                fields.prior_ppc_predictive_mean = Some(pc.predictive_mean);
                fields.prior_ppc_predictive_sd = Some(pc.predictive_sd);
                fields.prior_ppc_n_sims = Some(pc.n_sims);
            }
            PredictiveCheckKind::Posterior => {
                fields.posterior_ppc_p_value = Some(pc.p_value);
                fields.posterior_ppc_observed = Some(pc.observed);
                fields.posterior_ppc_predictive_mean = Some(pc.predictive_mean);
                fields.posterior_ppc_predictive_sd = Some(pc.predictive_sd);
                fields.posterior_ppc_n_sims = Some(pc.n_sims);
            }
        }
    }
    fields
}

type PosteriorSummary = (
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<usize>,
    Option<f64>,
    Option<String>,
    Option<Vec<u8>>,
);

fn posterior_summary_from_result(
    result: &antecedent::StudyResult,
    include_artifact: bool,
) -> PyResult<PosteriorSummary> {
    if let Some(post) = result.posterior.as_ref() {
        let eq = post.effect_column().unwrap_or(0);
        let artifact = if include_artifact {
            Some(encode_causal_posterior_bytes(post, "ate-analysis").map_err(py_err)?)
        } else {
            None
        };
        let p_below = post.probability_below(0.0).map_err(py_err)?;
        Ok((
            Some(post.summaries.mean[eq]),
            Some(post.summaries.sd[eq]),
            Some(post.summaries.q025[eq]),
            Some(post.summaries.q975[eq]),
            Some(post.draws.n_draws),
            Some(p_below),
            Some(post.diagnostics.backend_id.to_string()),
            artifact,
        ))
    } else {
        Ok((None, None, None, None, None, None, None, None))
    }
}

fn prior_sensitivity_from_result(result: &antecedent::StudyResult) -> PriorSensitivityFields {
    if let Some(post) = result.posterior.as_ref() {
        if let Some(sens) = post.prior_sensitivity.as_ref() {
            let scales = sens.prior_scales.iter().copied().collect::<Vec<_>>();
            let alphas = sens.alphas.iter().copied().collect::<Vec<_>>();
            return (
                if scales.is_empty() { None } else { Some(scales) },
                if alphas.is_empty() { None } else { Some(alphas) },
                Some(sens.effect_means.iter().copied().collect()),
                Some(sens.effect_sds.iter().copied().collect()),
            );
        }
    }
    (None, None, None, None)
}

fn conflict_summary_from_result(result: &antecedent::StudyResult) -> ConflictSummaryFields {
    if let Some(post) = result.posterior.as_ref() {
        if let Some(cs) = post.conflict_summary.as_ref() {
            return (
                Some(cs.source_ids.iter().map(std::string::ToString::to_string).collect()),
                Some(cs.alphas_requested.iter().copied().collect()),
                Some(cs.alphas_applied.iter().copied().collect()),
            );
        }
    }
    (None, None, None)
}

pub(crate) fn panel_multi_dataset_constraints(
    panel: &PanelData,
    context_names: Vec<String>,
    include_space_dummy: bool,
    include_time_dummy: bool,
    space_dummy_ci: bool,
    time_dummy_encoding: &str,
    time_dummy_ci: bool,
) -> PyResult<MultiDatasetConstraints> {
    let encoding = parse_time_dummy_encoding(time_dummy_encoding)?;
    let space_mode = space_dummy_ci_from_bool(space_dummy_ci);
    let time_mode = time_dummy_ci_from_bool(time_dummy_ci);
    let mut context_ids = Vec::new();
    for cname in context_names {
        context_ids.push(panel.schema().id_of(&cname).map_err(py_err)?);
    }
    Ok(MultiDatasetConstraints {
        context_variables: Arc::from(context_ids),
        include_space_dummy,
        include_time_dummy,
        space_dummy_ci: space_mode,
        time_dummy_encoding: encoding,
        time_dummy_ci: time_mode,
        ..MultiDatasetConstraints::default()
    })
}

/// Run standalone discovery over panel data and return a builder already seeded with the
/// accepted graph via [`Study::panel`] + [`antecedent::StudyBuilder::graph`].
///
/// `pcmci`/`pcmci_plus`/`lpcmci` pool all units into one series first (matches the
/// preprocessing the old lazy `Study::compile()` applied to these algorithms over
/// `PanelData`); `jpcmci_plus` uses the panel's per-unit multi-environment view directly.
pub(crate) fn panel_discovery_builder(
    panel: PanelData,
    algo: &str,
    max_lag: u32,
    alpha: f64,
    max_cond_size: usize,
    fdr_ctrl: FdrControl,
    accept_discovered: bool,
    multi_dataset: MultiDatasetConstraints,
    ci_impl: Arc<dyn antecedent_stats::ConditionalIndependence + Send + Sync>,
    ctx: &antecedent_core::ExecutionContext,
) -> PyResult<antecedent::StudyBuilder> {
    let fdr = fdr_ctrl.adjustment();
    match algo {
        "jpcmci_plus" | "jpcmci+" => {
            let multi = panel.as_multi_env().map_err(py_err)?;
            let vars: Vec<VariableId> = multi.schema().variables().iter().map(|v| v.id).collect();
            let params =
                DiscoverParams { max_lag, alpha, fdr, ci: ci_impl, multi_dataset, max_cond_size };
            let found = facade_discover_jpcmci_plus(&multi, &vars, &params, ctx).map_err(py_err)?;
            let accepted =
                accept_temporal_cpdag_review(found.review, accept_discovered).map_err(py_err)?;
            Ok(Study::panel(panel).graph(accepted))
        }
        "pcmci" => {
            let series = pool_panel_series(&panel)?;
            let vars: Vec<VariableId> = series.schema().variables().iter().map(|v| v.id).collect();
            let params = DiscoverParams {
                max_lag,
                alpha,
                fdr,
                ci: ci_impl,
                multi_dataset: MultiDatasetConstraints::default(),
                max_cond_size,
            };
            let found = facade_discover_pcmci(&series, &vars, &params, ctx).map_err(py_err)?;
            let accepted =
                accept_temporal_graph_review(found.review, accept_discovered).map_err(py_err)?;
            Ok(Study::panel(panel).graph(accepted))
        }
        "pcmci_plus" | "pcmci+" => {
            let series = pool_panel_series(&panel)?;
            let vars: Vec<VariableId> = series.schema().variables().iter().map(|v| v.id).collect();
            let params = DiscoverParams {
                max_lag,
                alpha,
                fdr,
                ci: ci_impl,
                multi_dataset: MultiDatasetConstraints::default(),
                max_cond_size,
            };
            let found = facade_discover_pcmci_plus(&series, &vars, &params, ctx).map_err(py_err)?;
            let accepted =
                accept_temporal_cpdag_review(found.review, accept_discovered).map_err(py_err)?;
            Ok(Study::panel(panel).graph(accepted))
        }
        "lpcmci" => {
            let series = pool_panel_series(&panel)?;
            let vars: Vec<VariableId> = series.schema().variables().iter().map(|v| v.id).collect();
            let params = DiscoverParams {
                max_lag,
                alpha,
                fdr,
                ci: ci_impl,
                multi_dataset: MultiDatasetConstraints::default(),
                max_cond_size,
            };
            let found = facade_discover_lpcmci(&series, &vars, &params, ctx).map_err(py_err)?;
            let accepted = accept_temporal_pag_review(
                found.evidence.graph.clone(),
                found.review,
                accept_discovered,
            )
            .map_err(py_err)?;
            Ok(Study::panel(panel).graph(accepted))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown panel discovery algorithm {other:?}; use jpcmci_plus|pcmci|pcmci_plus|lpcmci"
        ))),
    }
}

/// Run static ATE: identify → estimate → optional refute .
///
/// Parse optional `target_population` dict from Python (`kind` + fields).
fn parse_target_population(spec: Option<&Bound<'_, PyDict>>) -> PyResult<Option<TargetPopulation>> {
    let Some(d) = spec else {
        return Ok(None);
    };
    let kind: String = d
        .get_item("kind")?
        .ok_or_else(|| PyValueError::new_err("target_population requires 'kind'"))?
        .extract()?;
    let kind = kind.to_ascii_lowercase();
    Ok(Some(match kind.as_str() {
        "all" | "all_observed" => TargetPopulation::AllObserved,
        "treated" => TargetPopulation::Treated,
        "untreated" => TargetPopulation::Untreated,
        "named" => {
            let name: String = d
                .get_item("name")?
                .ok_or_else(|| PyValueError::new_err("named target_population requires 'name'"))?
                .extract()?;
            TargetPopulation::Predicate(PredicateExpr::named(name))
        }
        "rows" => {
            let rows: Vec<usize> = d
                .get_item("rows")?
                .ok_or_else(|| PyValueError::new_err("rows target_population requires 'rows'"))?
                .extract()?;
            TargetPopulation::Predicate(PredicateExpr::rows(rows))
        }
        "custom_distribution" | "custom" => {
            let id: u32 = d
                .get_item("id")?
                .ok_or_else(|| {
                    PyValueError::new_err("custom_distribution target_population requires 'id'")
                })?
                .extract()?;
            TargetPopulation::CustomDistribution(DistributionRef::from_raw(id))
        }
        other => {
            return Err(PyValueError::new_err(format!("unknown target_population kind {other:?}")));
        }
    }))
}

/// Build a [`PopulationRegistry`] from optional predicate/distribution dicts.
fn parse_population_registry(
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
            let weights: Vec<f64> = v.extract()?;
            reg.insert_distribution(DistributionRef::from_raw(id), weights);
        }
    }
    Ok(Some(reg))
}

/// `identifier`/`estimator` select the identification strategy and estimator; leaving both
/// `None` preserves the default (`backdoor.adjustment` + `linear.adjustment.ate`).
/// See [`antecedent::StudyBuilder::identifier`] and
/// [`antecedent::StudyBuilder::estimator`] for the supported ids.
///
/// Crosses the Python boundary once: NumPy columns + edge list in, structured
/// summary out. No per-row callbacks. Releases the GIL during native work.
#[pyfunction]
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
    inference=None,
    n_draws=1000,
    prior_scale=10.0,
    prior_artifact=None,
    prior_mapping=None,
    composed_prior=None,
    refute=None,
    validators=None,
    running_variable=None,
    cutoff=None,
    bandwidth=None,
    estimator_config=None,
    seed=1,
    bootstrap=50,
    threads=1,
    target_population=None,
    population_predicates=None,
    population_distributions=None,
    latency=None,
    cancel=None,
    on_progress=None,
    on_stage=None,
    return_posterior_artifact=false,
    accepted=false,
))]
fn analyze_ate(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    control_level: f64,
    active_level: f64,
    identifier: Option<String>,
    estimator: Option<String>,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    prior_mapping: Option<&Bound<'_, PyDict>>,
    composed_prior: Option<&Bound<'_, PyDict>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    estimator_config: Option<Bound<'_, PyDict>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    target_population: Option<Bound<'_, PyDict>>,
    population_predicates: Option<Bound<'_, PyDict>>,
    population_distributions: Option<Bound<'_, PyDict>>,
    latency: Option<String>,
    cancel: Option<PyCancellationToken>,
    on_progress: Option<Bound<'_, PyAny>>,
    on_stage: Option<Bound<'_, PyAny>>,
    return_posterior_artifact: bool,
    accepted: bool,
) -> PyResult<AteAnalysisResult> {
    let pop_spec = parse_target_population(target_population.as_ref())?;
    let registry = parse_population_registry(
        population_predicates.as_ref(),
        population_distributions.as_ref(),
    )?;
    let batch = columns_to_batch(&names, &columns)?;
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
    let prior_mapping = parse_prior_mapping(prior_mapping)?;
    let composed_prior = match composed_prior {
        Some(d) => Some(crate::prior_bank::owned_composed_prior_from_dict(d)?),
        None => None,
    };
    let cancel_token = cancel.map(|c| c.inner);
    let progress = callbacks::progress_sink_from_py(on_progress.as_ref())?;
    let stage_sink = callbacks::stage_sink_from_py(on_stage.as_ref())?;
    // Parsed with the GIL held (dict access requires it); the result is plain owned data
    // that crosses into `detach_catch` via `move` like `estimator`/`bootstrap` already do.
    let latency_mode = match latency.as_deref() {
        None => None,
        Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
            PyValueError::new_err(format!("unknown latency={s:?}; use interactive|standard|report"))
        })?),
    };
    let parsed_estimator_config = crate::estimator_config::parse_estimator_config(
        estimator_config.as_ref(),
        estimator.as_deref(),
        bootstrap,
    )?;
    // Drop NumPy borrows before releasing the GIL.
    drop(columns);

    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;

        let dag = dag_from_named_edges(data.schema(), &edges)?;

        let mut query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
        if let Some(pop) = pop_spec {
            query = query.with_target_population(pop);
        }

        let crate::estimator_config::ParsedEstimatorConfig {
            spec: configured_spec,
            rd_running_variable: configured_rv,
            rd_cutoff: configured_cutoff,
            rd_bandwidth: configured_bandwidth,
        } = parsed_estimator_config;
        let (merged_rv, merged_cutoff, merged_bandwidth) =
            crate::estimator_config::merge_rd_triple(
                running_variable,
                cutoff,
                bandwidth,
                configured_rv,
                configured_cutoff,
                configured_bandwidth,
            )?;
        let rd_ids = parse_rd_config(
            estimator.as_deref(),
            merged_rv.as_deref(),
            merged_cutoff,
            merged_bandwidth,
            |rv| data.schema().id_of(rv).map_err(py_err),
        )?;
        let mut builder = bind_dag(Study::tabular(data), dag, accepted)
            .query(query)
            .refute(suite)
            .custom_validators(custom_validators);
        // A configured estimator already carries its own (default-or-overridden) bootstrap
        // count; combining it with an explicit `StudyBuilder::bootstrap_replicates` call is
        // refused at `build()` time (`CausalError::Conflict`), so skip that call here.
        if configured_spec.is_none() {
            builder = builder.bootstrap_replicates(bootstrap);
        }
        if let Some(mode) = latency_mode {
            builder = builder.latency_mode(mode);
        }
        if let Some(sink) = stage_sink {
            builder = builder.stage_sink(sink);
        }
        if let Some(reg) = registry {
            builder = builder.population_registry(reg);
        }
        // Names at the boundary, ids on the hot path: an unknown strategy name is
        // rejected here, at the call the user made, not deep inside compile().
        if let Some(id) = identifier {
            builder = builder.identifier(
                id.parse::<antecedent::IdentifierId>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
        }
        if let Some(spec) = configured_spec {
            builder = builder.estimator(spec);
        } else if let Some(est) = estimator {
            builder = builder.estimator(
                est.parse::<antecedent::EstimatorId>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
        }
        if let Some((rv_id, cut, bw)) = rd_ids {
            builder = builder.rd_config(rv_id, cut, bw);
        }

        run_static_ate_from_builder(
            &names,
            builder,
            inference.as_deref(),
            n_draws,
            prior_scale,
            prior_artifact.as_deref(),
            prior_mapping,
            composed_prior,
            seed,
            threads,
            cancel_token,
            progress,
            return_posterior_artifact,
            None,
        )
    })
}

/// Static ATE from Arrow C Data Interface column exporters (zero-copy when possible).
#[pyfunction]
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
    inference=None,
    n_draws=1000,
    prior_scale=10.0,
    prior_artifact=None,
    prior_mapping=None,
    composed_prior=None,
    refute=None,
    validators=None,
    running_variable=None,
    cutoff=None,
    bandwidth=None,
    estimator_config=None,
    seed=1,
    bootstrap=50,
    threads=1,
    latency=None,
    cancel=None,
    on_progress=None,
    on_stage=None,
    return_posterior_artifact=false,
    accepted=false,
))]
fn analyze_ate_arrow_c(
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
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    prior_mapping: Option<&Bound<'_, PyDict>>,
    composed_prior: Option<&Bound<'_, PyDict>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    estimator_config: Option<Bound<'_, PyDict>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    latency: Option<String>,
    cancel: Option<PyCancellationToken>,
    on_progress: Option<Bound<'_, PyAny>>,
    on_stage: Option<Bound<'_, PyAny>>,
    return_posterior_artifact: bool,
    accepted: bool,
) -> PyResult<AteAnalysisResult> {
    let (data, bytes_borrowed) = tabular_from_arrow_c_objs(py, names.clone(), columns)?;
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
    let prior_mapping = parse_prior_mapping(prior_mapping)?;
    let composed_prior = match composed_prior {
        Some(d) => Some(crate::prior_bank::owned_composed_prior_from_dict(d)?),
        None => None,
    };
    let cancel_token = cancel.map(|c| c.inner);
    let progress = callbacks::progress_sink_from_py(on_progress.as_ref())?;
    let stage_sink = callbacks::stage_sink_from_py(on_stage.as_ref())?;
    let latency_mode = match latency.as_deref() {
        None => None,
        Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
            PyValueError::new_err(format!("unknown latency={s:?}; use interactive|standard|report"))
        })?),
    };
    let parsed_estimator_config = crate::estimator_config::parse_estimator_config(
        estimator_config.as_ref(),
        estimator.as_deref(),
        bootstrap,
    )?;

    detach_catch(py, move || {
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;

        let dag = dag_from_named_edges(data.schema(), &edges)?;

        let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);

        let crate::estimator_config::ParsedEstimatorConfig {
            spec: configured_spec,
            rd_running_variable: configured_rv,
            rd_cutoff: configured_cutoff,
            rd_bandwidth: configured_bandwidth,
        } = parsed_estimator_config;
        let (merged_rv, merged_cutoff, merged_bandwidth) =
            crate::estimator_config::merge_rd_triple(
                running_variable,
                cutoff,
                bandwidth,
                configured_rv,
                configured_cutoff,
                configured_bandwidth,
            )?;
        let rd_ids = parse_rd_config(
            estimator.as_deref(),
            merged_rv.as_deref(),
            merged_cutoff,
            merged_bandwidth,
            |rv| data.schema().id_of(rv).map_err(py_err),
        )?;
        let mut builder = bind_dag(Study::tabular(data), dag, accepted)
            .query(query)
            .refute(suite)
            .custom_validators(custom_validators);
        if configured_spec.is_none() {
            builder = builder.bootstrap_replicates(bootstrap);
        }
        if let Some(mode) = latency_mode {
            builder = builder.latency_mode(mode);
        }
        if let Some(sink) = stage_sink {
            builder = builder.stage_sink(sink);
        }
        // Names at the boundary, ids on the hot path: an unknown strategy name is
        // rejected here, at the call the user made, not deep inside compile().
        if let Some(id) = identifier {
            builder = builder.identifier(
                id.parse::<antecedent::IdentifierId>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
        }
        if let Some(spec) = configured_spec {
            builder = builder.estimator(spec);
        } else if let Some(est) = estimator {
            builder = builder.estimator(
                est.parse::<antecedent::EstimatorId>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
        }
        if let Some((rv_id, cut, bw)) = rd_ids {
            builder = builder.rd_config(rv_id, cut, bw);
        }

        run_static_ate_from_builder(
            &names,
            builder,
            inference.as_deref(),
            n_draws,
            prior_scale,
            prior_artifact.as_deref(),
            prior_mapping,
            composed_prior,
            seed,
            threads,
            cancel_token,
            progress,
            return_posterior_artifact,
            Some(bytes_borrowed),
        )
    })
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
    refute=None,
    seed=1,
    bootstrap=50,
    threads=1,
    latency=None,
))]
fn analyze_ate_many(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    queries: Vec<(String, String, f64, f64)>,
    identifier: Option<String>,
    estimator: Option<String>,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    latency: Option<String>,
) -> PyResult<Vec<AteAnalysisResult>> {
    let batch = columns_to_batch(&names, &columns)?;
    let suite = suite_from_refute(refute.as_ref())?;
    let latency_mode = match latency.as_deref() {
        None => None,
        Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
            PyValueError::new_err(format!("unknown latency={s:?}; use interactive|standard|report"))
        })?),
    };
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let dag = dag_from_named_edges(data.schema(), &edges)?;
        let mut ate_queries = Vec::with_capacity(queries.len());
        for (treatment, outcome, control, active) in &queries {
            let t_id = data.schema().id_of(treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(outcome).map_err(py_err)?;
            ate_queries.push(AverageEffectQuery::with_levels(t_id, y_id, *control, *active));
        }
        let mut batch =
            antecedent::BatchStudy::new(data, dag).bootstrap_replicates(bootstrap).refute(suite);
        if let Some(mode) = latency_mode {
            batch = batch.latency_mode(mode);
        }
        if let Some(id) = identifier {
            batch = batch.identifier(
                id.parse::<antecedent::IdentifierId>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
        }
        if let Some(est) = estimator {
            batch = batch.estimator(
                est.parse::<antecedent::EstimatorId>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
        }
        let ctx = py_execution_context(seed, threads);
        let results = batch.estimate_many(&ate_queries, &ctx).map_err(py_err)?;
        results.into_iter().map(|r| ate_result_from_analysis(&names, r, false)).collect()
    })
}

/// Shared NumPy → batch / validators / refute preamble for typed-graph ATE entry points.
type AteTabularPreamble =
    (RecordBatch, RefuteSuite, Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>, u32);

fn prepare_ate_tabular_preamble(
    names: &[String],
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    refute: Option<&Bound<'_, PyAny>>,
    validators: Option<&Bound<'_, PyAny>>,
    threads: u32,
) -> PyResult<AteTabularPreamble> {
    let batch = columns_to_batch(names, &columns)?;
    let custom_validators = callbacks::parse_validators(validators)?;
    let suite = suite_from_refute(refute)?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
    drop(columns);
    Ok((batch, suite, custom_validators, threads))
}

/// Static graph input for the shared `analyze_ate_{pag,cpdag,admg}` dispatch helper.
///
/// Local replacement for the old crate-level `GraphInput` enum (removed along with the
/// facade's `Study::builder()` refactor): a bare dispatch tag over the three typed static
/// graph classes these three Python entry points accept, nothing more.
enum StaticGraphInput {
    Pag(Pag),
    Cpdag(Cpdag),
    Admg(antecedent_graph::Admg),
}

fn analyze_ate_typed_graph(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    graph: StaticGraphInput,
    treatment: String,
    outcome: String,
    control_level: f64,
    active_level: f64,
    identifier: Option<String>,
    estimator: Option<String>,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    estimator_config: Option<Bound<'_, PyDict>>,
    latency: Option<String>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    let (batch, suite, custom_validators, threads) = prepare_ate_tabular_preamble(
        &names,
        columns,
        refute.as_ref(),
        validators.as_ref(),
        threads,
    )?;
    let latency_mode = match latency.as_deref() {
        None => None,
        Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
            PyValueError::new_err(format!("unknown latency={s:?}; use interactive|standard|report"))
        })?),
    };
    let parsed_estimator_config = crate::estimator_config::parse_estimator_config(
        estimator_config.as_ref(),
        estimator.as_deref(),
        bootstrap,
    )?;
    detach_catch(py, move || {
        run_ate_with_graph_input(
            &names,
            batch,
            graph,
            treatment,
            outcome,
            control_level,
            active_level,
            identifier,
            estimator,
            inference,
            n_draws,
            prior_scale,
            prior_artifact,
            suite,
            custom_validators,
            running_variable,
            cutoff,
            bandwidth,
            parsed_estimator_config,
            latency_mode,
            seed,
            bootstrap,
            threads,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn run_ate_with_graph_input(
    names: &[String],
    batch: RecordBatch,
    graph: StaticGraphInput,
    treatment: String,
    outcome: String,
    control_level: f64,
    active_level: f64,
    identifier: Option<String>,
    estimator: Option<String>,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    suite: RefuteSuite,
    custom_validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    parsed_estimator_config: crate::estimator_config::ParsedEstimatorConfig,
    latency_mode: Option<antecedent::LatencyMode>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
    let data = loaded.data;
    let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
    let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
    let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
    let crate::estimator_config::ParsedEstimatorConfig {
        spec: configured_spec,
        rd_running_variable: configured_rv,
        rd_cutoff: configured_cutoff,
        rd_bandwidth: configured_bandwidth,
    } = parsed_estimator_config;
    let (merged_rv, merged_cutoff, merged_bandwidth) = crate::estimator_config::merge_rd_triple(
        running_variable,
        cutoff,
        bandwidth,
        configured_rv,
        configured_cutoff,
        configured_bandwidth,
    )?;
    let rd_ids = parse_rd_config(
        estimator.as_deref(),
        merged_rv.as_deref(),
        merged_cutoff,
        merged_bandwidth,
        |rv| data.schema().id_of(rv).map_err(py_err),
    )?;
    let mut builder =
        Study::tabular(data).query(query).refute(suite).custom_validators(custom_validators);
    if configured_spec.is_none() {
        builder = builder.bootstrap_replicates(bootstrap);
    }
    builder = match graph {
        StaticGraphInput::Pag(pag) => builder.graph(pag),
        StaticGraphInput::Cpdag(cpdag) => {
            let accepted = AcceptedGraph::cpdag(cpdag).map_err(py_err)?;
            builder.graph(accepted)
        }
        StaticGraphInput::Admg(admg) => builder.graph(admg),
    };
    // Names at the boundary, ids on the hot path.
    if let Some(id) = identifier {
        builder = builder.identifier(
            id.parse::<antecedent::IdentifierId>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
    }
    if let Some(spec) = configured_spec {
        builder = builder.estimator(spec);
    } else if let Some(est) = estimator {
        builder = builder.estimator(
            est.parse::<antecedent::EstimatorId>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
    }
    if let Some((rv_id, cut, bw)) = rd_ids {
        builder = builder.rd_config(rv_id, cut, bw);
    }
    if let Some(mode) = latency_mode {
        builder = builder.latency_mode(mode);
    }
    run_static_ate_from_builder(
        names,
        builder,
        inference.as_deref(),
        n_draws,
        prior_scale,
        prior_artifact.as_deref(),
        None,
        None,
        seed,
        threads,
        None,
        None,
        false,
        None,
    )
}

/// Static ATE with a typed PAG.
#[pyfunction]
#[pyo3(signature = (
    names, columns, graph, treatment, outcome, *,
    control_level=0.0, active_level=1.0, identifier=None, estimator=None,
    inference=None, n_draws=1000, prior_scale=10.0,
    prior_artifact=None, refute=None, validators=None,
    running_variable=None,
    cutoff=None,
    bandwidth=None,
    estimator_config=None,
    latency=None,
    seed=1, bootstrap=50, threads=1
))]
#[allow(clippy::too_many_arguments)]
fn analyze_ate_pag(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    graph: graphs::Pag,
    treatment: String,
    outcome: String,
    control_level: f64,
    active_level: f64,
    identifier: Option<String>,
    estimator: Option<String>,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    estimator_config: Option<Bound<'_, PyDict>>,
    latency: Option<String>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    analyze_ate_typed_graph(
        py,
        names,
        columns,
        StaticGraphInput::Pag(graph.pag),
        treatment,
        outcome,
        control_level,
        active_level,
        identifier,
        estimator,
        inference,
        n_draws,
        prior_scale,
        prior_artifact,
        refute,
        validators,
        running_variable,
        cutoff,
        bandwidth,
        estimator_config,
        latency,
        seed,
        bootstrap,
        threads,
    )
}

/// Static ATE with a typed CPDAG.
#[pyfunction]
#[pyo3(signature = (
    names, columns, graph, treatment, outcome, *,
    control_level=0.0, active_level=1.0, identifier=None, estimator=None,
    inference=None, n_draws=1000, prior_scale=10.0,
    prior_artifact=None, refute=None, validators=None,
    running_variable=None,
    cutoff=None,
    bandwidth=None,
    estimator_config=None,
    latency=None,
    seed=1, bootstrap=50, threads=1
))]
#[allow(clippy::too_many_arguments)]
fn analyze_ate_cpdag(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    graph: graphs::Cpdag,
    treatment: String,
    outcome: String,
    control_level: f64,
    active_level: f64,
    identifier: Option<String>,
    estimator: Option<String>,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    estimator_config: Option<Bound<'_, PyDict>>,
    latency: Option<String>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    analyze_ate_typed_graph(
        py,
        names,
        columns,
        StaticGraphInput::Cpdag(graph.cpdag),
        treatment,
        outcome,
        control_level,
        active_level,
        identifier,
        estimator,
        inference,
        n_draws,
        prior_scale,
        prior_artifact,
        refute,
        validators,
        running_variable,
        cutoff,
        bandwidth,
        estimator_config,
        latency,
        seed,
        bootstrap,
        threads,
    )
}

/// Static ATE with a typed ADMG.
#[pyfunction]
#[pyo3(signature = (
    names, columns, graph, treatment, outcome, *,
    control_level=0.0, active_level=1.0, identifier=None, estimator=None,
    inference=None, n_draws=1000, prior_scale=10.0,
    prior_artifact=None, refute=None, validators=None,
    running_variable=None,
    cutoff=None,
    bandwidth=None,
    estimator_config=None,
    latency=None,
    seed=1, bootstrap=50, threads=1
))]
#[allow(clippy::too_many_arguments)]
fn analyze_ate_admg(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    graph: graphs::Admg,
    treatment: String,
    outcome: String,
    control_level: f64,
    active_level: f64,
    identifier: Option<String>,
    estimator: Option<String>,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    estimator_config: Option<Bound<'_, PyDict>>,
    latency: Option<String>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    analyze_ate_typed_graph(
        py,
        names,
        columns,
        StaticGraphInput::Admg(graph.admg),
        treatment,
        outcome,
        control_level,
        active_level,
        identifier,
        estimator,
        inference,
        n_draws,
        prior_scale,
        prior_artifact,
        refute,
        validators,
        running_variable,
        cutoff,
        bandwidth,
        estimator_config,
        latency,
        seed,
        bootstrap,
        threads,
    )
}

/// Static ATE via static discovery → DAG (when fully oriented).
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    treatment,
    outcome,
    *,
    algorithm="pc",
    alpha=0.05,
    fdr=true,
    max_cond_size=2,
    prune_threshold=0.0,
    l1=0.1,
    threshold=0.3,
    standardize=true,
    accept_discovered=true,
    control_level=0.0,
    active_level=1.0,
    identifier=None,
    estimator=None,
    inference=None,
    n_draws=1000,
    prior_scale=10.0,
    prior_artifact=None,
    refute=None,
    validators=None,
    ci=None,
    n_chains=2,
    n_warmup=100,
    mcmc_draws=200,
    thin=1,
    soft_weight="none",
    require_diagnostics_gate=true,
    running_variable=None,
    cutoff=None,
    bandwidth=None,
    estimator_config=None,
    seed=1,
    bootstrap=50,
    threads=1
))]
fn analyze_ate_discover(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    treatment: String,
    outcome: String,
    algorithm: &str,
    alpha: f64,
    fdr: bool,
    max_cond_size: usize,
    prune_threshold: f64,
    l1: f64,
    threshold: f64,
    standardize: bool,
    accept_discovered: bool,
    control_level: f64,
    active_level: f64,
    identifier: Option<String>,
    estimator: Option<String>,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    ci: Option<Bound<'_, PyAny>>,
    n_chains: u32,
    n_warmup: u32,
    mcmc_draws: u32,
    thin: u32,
    soft_weight: &str,
    require_diagnostics_gate: bool,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    estimator_config: Option<Bound<'_, PyDict>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    let algo = algorithm.to_ascii_lowercase();
    let batch = columns_to_batch(&names, &columns)?;
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let (ci_impl, _ci_name, is_ci_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    let parsed_estimator_config = crate::estimator_config::parse_estimator_config(
        estimator_config.as_ref(),
        estimator.as_deref(),
        bootstrap,
    )?;
    drop(columns);
    let threads = if is_ci_callback || !custom_validators.is_empty() { 1 } else { threads };
    let soft_weight = soft_weight.to_string();
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
        let fdr_ctrl = if fdr { FdrControl::bh() } else { FdrControl::Off };
        let ctx = py_execution_context(seed, threads);
        let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();

        let crate::estimator_config::ParsedEstimatorConfig {
            spec: configured_spec,
            rd_running_variable: configured_rv,
            rd_cutoff: configured_cutoff,
            rd_bandwidth: configured_bandwidth,
        } = parsed_estimator_config;
        let (merged_rv, merged_cutoff, merged_bandwidth) =
            crate::estimator_config::merge_rd_triple(
                running_variable,
                cutoff,
                bandwidth,
                configured_rv,
                configured_cutoff,
                configured_bandwidth,
            )?;
        let rd_ids = parse_rd_config(
            estimator.as_deref(),
            merged_rv.as_deref(),
            merged_cutoff,
            merged_bandwidth,
            |rv| data.schema().id_of(rv).map_err(py_err),
        )?;
        let soft = match soft_weight.as_str() {
            "none" | "" => antecedent::discovery::CiSoftWeight::None,
            "bayes_factor" | "bf" => antecedent::discovery::CiSoftWeight::BayesFactor,
            "posterior_dependence" | "pd" => {
                antecedent::discovery::CiSoftWeight::PosteriorDependence
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown soft_weight {other:?}; use none|bayes_factor|posterior_dependence"
                )));
            }
        };

        // Bayesian graph-posterior discovery: no discrete graph to review — wire the
        // posterior directly. Frequentist inference + a posterior is refused by the
        // library itself (checked inside `build`/`run`, not here).
        let is_graph_posterior = matches!(
            algo.as_str(),
            "exact_dag_posterior"
                | "exact"
                | "order_mcmc"
                | "structure_mcmc"
                | "ci_screened_posterior"
                | "ci_screened"
        );
        let builder = if is_graph_posterior {
            let params = antecedent::discovery::BayesianDiscoverParams::default();
            let gp = match algo.as_str() {
                "exact_dag_posterior" | "exact" => {
                    antecedent::discovery::discover_exact_dag_posterior(&data, &vars, &params, &ctx)
                        .map_err(py_err)?
                }
                "order_mcmc" => {
                    let schedule = antecedent::discovery::GraphMcmcSchedule {
                        n_chains,
                        n_warmup,
                        n_draws: mcmc_draws,
                        thin,
                    };
                    antecedent::discovery::discover_order_mcmc(
                        &data,
                        &vars,
                        &params,
                        &schedule,
                        require_diagnostics_gate,
                        &ctx,
                    )
                    .map_err(py_err)?
                }
                "structure_mcmc" => {
                    let schedule = antecedent::discovery::GraphMcmcSchedule {
                        n_chains,
                        n_warmup,
                        n_draws: mcmc_draws,
                        thin,
                    };
                    antecedent::discovery::discover_structure_mcmc(
                        &data, &vars, &params, &schedule, &ctx,
                    )
                    .map_err(py_err)?
                }
                _ => {
                    // "ci_screened_posterior" | "ci_screened"
                    let screen = StaticDiscoverParams {
                        alpha,
                        max_cond_size,
                        fdr: fdr_ctrl.adjustment(),
                        ci: ci_impl.clone(),
                        screen_pc: false,
                        max_subset: None,
                    };
                    let schedule = antecedent::discovery::GraphMcmcSchedule {
                        n_chains,
                        n_warmup,
                        n_draws: mcmc_draws,
                        thin,
                    };
                    antecedent::discovery::discover_ci_screened_posterior(
                        &data, &vars, &params, &screen, &schedule, soft, &ctx,
                    )
                    .map_err(py_err)?
                }
            };
            Study::tabular(data).graph_posterior(gp)
        } else {
            let accepted = match algo.as_str() {
                "pc" => {
                    let params = StaticDiscoverParams {
                        alpha,
                        max_cond_size,
                        fdr: fdr_ctrl.adjustment(),
                        ci: ci_impl.clone(),
                        screen_pc: false,
                        max_subset: None,
                    };
                    let found = facade_discover_pc(&data, &vars, &params, &ctx).map_err(py_err)?;
                    accept_cpdag_review(found.review, accept_discovered).map_err(py_err)?
                }
                "ges" => {
                    let params = StaticDiscoverParams {
                        alpha,
                        max_cond_size,
                        fdr: fdr_ctrl.adjustment(),
                        ci: ci_impl.clone(),
                        screen_pc: false,
                        max_subset: None,
                    };
                    let found = facade_discover_ges(&data, &vars, &params, &ctx).map_err(py_err)?;
                    accept_cpdag_review(found.review, accept_discovered).map_err(py_err)?
                }
                "lingam" => {
                    // LiNGAM ignores `params.ci`/`params.fdr` (independence-of-residuals is
                    // internal to the algorithm); a fresh partial-correlation stub satisfies
                    // the required field without invoking a possibly-slow Python `ci=`.
                    let params = StaticDiscoverParams {
                        alpha,
                        max_cond_size,
                        fdr: None,
                        ci: Arc::new(antecedent_stats::PartialCorrelation),
                        screen_pc: false,
                        max_subset: None,
                    };
                    let found =
                        facade_discover_lingam(&data, &vars, &params, prune_threshold, &ctx)
                            .map_err(py_err)?;
                    accept_dag_review(found.review, accept_discovered).map_err(py_err)?
                }
                "notears" => {
                    // NOTEARS ignores `params.ci`/`params.fdr` (continuous-SEM solver); see
                    // the `lingam` arm above for why a stub CI is passed here.
                    let params = StaticDiscoverParams {
                        alpha,
                        max_cond_size,
                        fdr: None,
                        ci: Arc::new(antecedent_stats::PartialCorrelation),
                        screen_pc: false,
                        max_subset: None,
                    };
                    let found = facade_discover_notears(
                        &data,
                        &vars,
                        &params,
                        l1,
                        threshold,
                        standardize,
                        &ctx,
                    )
                    .map_err(py_err)?;
                    accept_dag_review(found.discovery.review, accept_discovered).map_err(py_err)?
                }
                "fci" => {
                    let params = StaticDiscoverParams {
                        alpha,
                        max_cond_size,
                        fdr: fdr_ctrl.adjustment(),
                        ci: ci_impl.clone(),
                        screen_pc: false,
                        max_subset: None,
                    };
                    let found = facade_discover_fci(&data, &vars, &params, &ctx).map_err(py_err)?;
                    accept_pag_review(found.evidence.graph.clone(), found.review, accept_discovered)
                        .map_err(py_err)?
                }
                "rfci" => {
                    let params = StaticDiscoverParams {
                        alpha,
                        max_cond_size,
                        fdr: fdr_ctrl.adjustment(),
                        ci: ci_impl.clone(),
                        screen_pc: false,
                        max_subset: None,
                    };
                    let found =
                        facade_discover_rfci(&data, &vars, &params, &ctx).map_err(py_err)?;
                    accept_pag_review(found.evidence.graph.clone(), found.review, accept_discovered)
                        .map_err(py_err)?
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown static discovery algorithm {other:?}; use pc|ges|lingam|notears|\
                         fci|rfci|exact_dag_posterior|order_mcmc|structure_mcmc|\
                         ci_screened_posterior"
                    )));
                }
            };
            Study::tabular(data).graph(accepted)
        };
        let mut builder = builder.query(query).refute(suite).custom_validators(custom_validators);
        // A configured estimator already carries its own (default-or-overridden) bootstrap
        // count; combining it with an explicit `StudyBuilder::bootstrap_replicates` call is
        // refused at `build()` time (`CausalError::Conflict`), so skip that call here — same
        // rule `analyze_ate` follows for the static-graph path.
        if configured_spec.is_none() {
            builder = builder.bootstrap_replicates(bootstrap);
        }
        // Names at the boundary, ids on the hot path: an unknown strategy name is
        // rejected here, at the call the user made, not deep inside compile().
        if let Some(id) = identifier {
            builder = builder.identifier(
                id.parse::<antecedent::IdentifierId>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
        }
        if let Some(spec) = configured_spec {
            builder = builder.estimator(spec);
        } else if let Some(est) = estimator {
            builder = builder.estimator(
                est.parse::<antecedent::EstimatorId>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
        }
        if let Some((rv_id, cut, bw)) = rd_ids {
            builder = builder.rd_config(rv_id, cut, bw);
        }

        run_static_ate_from_builder(
            &names,
            builder,
            inference.as_deref(),
            n_draws,
            prior_scale,
            prior_artifact.as_deref(),
            None,
            None,
            seed,
            threads,
            None,
            None,
            false,
            None,
        )
    })
}

/// Interventional distribution via ID/IDC + functional distribution estimator.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    edges,
    outcome,
    interventions,
    *,
    conditioning=None,
    seed=1,
    threads=1
))]

fn analyze_distribution(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    outcome: String,
    interventions: std::collections::HashMap<String, f64>,
    conditioning: Option<Vec<String>>,
    seed: u64,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let mut ivs = Vec::with_capacity(interventions.len());
        for (name, level) in &interventions {
            let id = data.schema().id_of(name).map_err(py_err)?;
            ivs.push(Intervention::set(id, Value::f64(*level)));
        }
        let mut query = InterventionalDistributionQuery::new(y_id, ivs);
        if let Some(cond) = conditioning {
            let mut z = Vec::with_capacity(cond.len());
            for name in &cond {
                z.push(data.schema().id_of(name).map_err(py_err)?);
            }
            query = query.with_conditioning(z);
        }
        let dag = dag_from_named_edges(data.schema(), &edges)?;
        let analysis = Study::tabular(data)
            .graph(dag)
            .query(CausalQuery::Distribution(query))
            .identifier(IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalDistribution)
            .build()
            .map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        ate_result_from_analysis(&names, result, false)
    })
}

/// Path-specific natural effect via ID + functional effect estimator.
#[pyfunction]
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
    seed=1,
    bootstrap=50,
    threads=1
))]
fn analyze_path_specific(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    control_level: f64,
    active_level: f64,
    path_nodes: Option<Vec<String>>,
    max_paths: usize,
    max_len: usize,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let mut query = PathSpecificEffectQuery::binary(t_id, y_id)
            .with_max_paths(max_paths)
            .with_max_len(max_len);
        // Override levels via control/active interventions.
        query.control = Intervention::set(t_id, Value::f64(control_level));
        query.active = Intervention::set(t_id, Value::f64(active_level));
        if let Some(nodes) = path_nodes {
            let mut ids = Vec::with_capacity(nodes.len());
            for name in &nodes {
                ids.push(data.schema().id_of(name).map_err(py_err)?);
            }
            query = query.with_path_nodes(ids);
        }
        let dag = dag_from_named_edges(data.schema(), &edges)?;
        let analysis = Study::tabular(data)
            .graph(dag)
            .query(CausalQuery::PathSpecific(query))
            .identifier(IdentifierId::PathSpecificNatural)
            .estimator(EstimatorId::FunctionalEffect)
            .bootstrap_replicates(bootstrap)
            .build()
            .map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        ate_result_from_analysis(&names, result, false)
    })
}

pub(crate) fn ate_result_from_analysis(
    names: &[String],
    result: antecedent::StudyResult,
    include_posterior_artifact: bool,
) -> PyResult<AteAnalysisResult> {
    let adjustment_set: Vec<String> = result
        .estimand
        .adjustment_set
        .iter()
        .map(|id| names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("var{}", id.raw())))
        .collect();

    let refutation_ran = !result.refutations.is_empty();
    let refutation_passed = if refutation_ran {
        result.refutations.iter().all(|r| r.passed)
    } else {
        // Do not claim pass when no validators ran (e.g. refute=False or empty suite).
        false
    };
    let estimator_id = result.logical_plan.estimator.as_deref().unwrap_or("").to_string();
    let overlap_ess = result.estimate.overlap_report.as_ref().and_then(|r| r.ess);
    let overlap_propensity_min = result.estimate.overlap_report.as_ref().map(|r| r.propensity_min);

    let (
        posterior_effect_mean,
        posterior_effect_sd,
        posterior_q025,
        posterior_q975,
        posterior_n_draws,
        posterior_p_below_zero,
        posterior_backend,
        posterior_artifact,
    ) = posterior_summary_from_result(&result, include_posterior_artifact)?;
    let ppc = ppc_fields_from_checks(&result.predictive_checks);
    let (
        prior_sensitivity_scales,
        prior_sensitivity_alphas,
        prior_sensitivity_means,
        prior_sensitivity_sds,
    ) = prior_sensitivity_from_result(&result);
    let (conflict_source_ids, conflict_alphas_requested, conflict_alphas_applied) =
        conflict_summary_from_result(&result);
    let posterior_unidentified_mass = result.posterior.as_ref().map(|p| p.unidentified_mass);

    // Values shared between an existing flat field and its new nested-section
    // counterpart are computed once here, then cloned into the section so the
    // flat field and the section can never drift apart.
    let identification_status = format!("{:?}", result.identification.status);
    let method = result.estimand.method.to_string();
    let refutations: Vec<RefutationReportView> =
        result.refutations.iter().map(RefutationReportView::from).collect();
    let plan_id = result.logical_plan.plan_id.to_string();
    let modality = format!("{:?}", result.logical_plan.data_classification);
    let latency_mode =
        result.performance.latency_mode.as_ref().map(std::string::ToString::to_string);
    let bootstrap_replicates_ok =
        result.performance.bootstrap_replicates_ok.or(result.estimate.bootstrap_replicates_ok);
    let cancelled = result.performance.cancelled || result.estimate.bootstrap_cancelled;
    let stage_timings: Vec<(String, u64)> =
        result.performance.stage_timings_ns.iter().map(|(s, ns)| (s.to_string(), *ns)).collect();

    let identification = IdentificationSection {
        status: identification_status.clone(),
        method: method.clone(),
        adjustment_set: adjustment_set.clone(),
        assumption_count: result.estimate.assumptions.len(),
        derivation_step_count: result.identification.derivation.steps.len(),
    };
    let estimate = EstimateSection {
        ate: result.estimate.ate,
        se_analytic: result.estimate.se_analytic,
        se_bootstrap: result.estimate.se_bootstrap,
        estimator_id: estimator_id.clone(),
        method: method.clone(),
        overlap_ess,
        overlap_propensity_min,
    };
    let posterior = PosteriorSection {
        effect_mean: posterior_effect_mean,
        effect_sd: posterior_effect_sd,
        q025: posterior_q025,
        q975: posterior_q975,
        n_draws: posterior_n_draws,
        p_below_zero: posterior_p_below_zero,
        backend: posterior_backend.clone(),
        artifact: posterior_artifact.clone(),
        unidentified_mass: posterior_unidentified_mass,
    };
    let validation = ValidationSection::from_reports(refutations.clone());
    let performance = PerformanceSection {
        plan_id: plan_id.clone(),
        modality: modality.clone(),
        peak_memory_bytes: result.physical_plan.estimated_peak_memory_bytes,
        latency_mode: latency_mode.clone(),
        wall_time_ns: result.performance.wall_time_ns,
        bootstrap_replicates_requested: result.performance.bootstrap_replicates_requested,
        bootstrap_replicates_ok,
        n_draws: result.performance.n_draws,
        cancelled,
        early_stopped: result.performance.early_stopped,
        stage_timings: stage_timings.clone(),
        bytes_borrowed: result.performance.bytes_borrowed,
    };

    Ok(AteAnalysisResult {
        ate: result.estimate.ate,
        se_analytic: result.estimate.se_analytic,
        se_bootstrap: result.estimate.se_bootstrap,
        bootstrap_replicates_failed: result.estimate.bootstrap_replicates_failed,
        adjustment_set,
        identification_status,
        refutation_passed,
        refutation_ran,
        refutation_count: refutations.len(),
        refutations,
        assumption_count: result.estimate.assumptions.len(),
        derivation_step_count: result.identification.derivation.steps.len(),
        method,
        estimator_id,
        overlap_ess,
        overlap_propensity_min,
        posterior_effect_mean,
        posterior_effect_sd,
        posterior_q025,
        posterior_q975,
        posterior_n_draws,
        posterior_p_below_zero,
        posterior_backend,
        posterior_artifact,
        diagnostics: result
            .diagnostics
            .iter()
            .map(|d| format!("{}: {}", d.code, d.message))
            .collect(),
        provenance_node_count: result.provenance.len(),
        plan_id,
        modality,
        discovery_algorithm: result
            .logical_plan
            .discovery_algorithm
            .as_ref()
            .map(std::string::ToString::to_string),
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
        posterior_ppc_p_value: ppc.posterior_ppc_p_value,
        posterior_ppc_observed: ppc.posterior_ppc_observed,
        posterior_ppc_predictive_mean: ppc.posterior_ppc_predictive_mean,
        posterior_ppc_predictive_sd: ppc.posterior_ppc_predictive_sd,
        posterior_ppc_n_sims: ppc.posterior_ppc_n_sims,
        prior_sensitivity_scales,
        prior_sensitivity_alphas,
        prior_sensitivity_means,
        prior_sensitivity_sds,
        conflict_source_ids,
        conflict_alphas_requested,
        conflict_alphas_applied,
        posterior_unidentified_mass,
        latency_mode,
        wall_time_ns: result.performance.wall_time_ns,
        bootstrap_replicates_requested: result.performance.bootstrap_replicates_requested,
        bootstrap_replicates_ok,
        n_draws_effort: result.performance.n_draws,
        cancelled,
        early_stopped: result.performance.early_stopped,
        stage_timings,
        identification,
        estimate,
        posterior,
        validation,
        performance,
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

#[pyfunction]
#[pyo3(signature = (
    names, columns, edges, treatment, outcome, modifier, *,
    control_level=0.0, active_level=1.0,
    refute=None, validators=None, seed=1, bootstrap=50, threads=1
))]
fn analyze_conditional(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    modifier: String,
    control_level: f64,
    active_level: f64,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let w_id = data.schema().id_of(&modifier).map_err(py_err)?;
        let inner = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level)
            .with_effect_modifiers([w_id]);
        let cq = ConditionalEffectQuery::try_new(inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dag = dag_from_named_edges(data.schema(), &edges)?;
        let analysis = Study::tabular(data)
            .graph(dag)
            .query(CausalQuery::ConditionalEffect(cq))
            .refute(suite)
            .custom_validators(custom_validators)
            .bootstrap_replicates(bootstrap)
            .build()
            .map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        ate_result_from_analysis(&names, result, false)
    })
}

/// Static mediation (treatment → mediator(s) → outcome) via the facade.
#[pyfunction]
#[pyo3(signature = (
    names, columns, edges, treatment, outcome, mediators, *,
    contrast="mediated", control_level=0.0, active_level=1.0,
    refute=None, seed=1, bootstrap=0, threads=1
))]
fn analyze_mediation(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    mediators: Vec<String>,
    contrast: &str,
    control_level: f64,
    active_level: f64,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let suite = suite_from_refute(refute.as_ref())?;
    let contrast = contrast.to_string();
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let mut med_ids = Vec::with_capacity(mediators.len());
        for m in &mediators {
            med_ids.push(data.schema().id_of(m).map_err(py_err)?);
        }
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
        let mut q = MediationQuery::binary(t_id, y_id, med_ids, contrast);
        q.control = Intervention::set(t_id, Value::f64(control_level));
        q.active = Intervention::set(t_id, Value::f64(active_level));
        let dag = dag_from_named_edges(data.schema(), &edges)?;
        let analysis = Study::tabular(data)
            .graph(dag)
            .query(CausalQuery::Mediation(q))
            .refute(suite)
            .bootstrap_replicates(bootstrap)
            .build()
            .map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        ate_result_from_analysis(&names, result, false)
    })
}

/// Identify-only on a static ADMG (no estimation).
///
/// An ADMG carries bidirected edges, so it is the only static graph type that
/// can state "these two variables share an unmeasured common cause". Without
/// this entry point an unobserved confounder had to be flattened into a DAG
/// before identification, where it looks like an ordinary adjustable node and
/// the effect is reported as identified by adjusting on a variable no study
/// can measure.
#[pyfunction]
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
        let mut builder = Study::tabular(data)
            .graph(graph.admg)
            .query(AverageEffectQuery::binary_ate(t_id, y_id))
            .refute(RefuteSuite::None);
        if let Some(id) = identifier {
            builder = builder.identifier(
                id.parse::<antecedent::IdentifierId>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
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
#[pyfunction]
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
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
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

/// Temporal linear mediation (treatment → mediator → outcome).

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_ate, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_pag, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_cpdag, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_admg, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_arrow_c, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_many, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_discover, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_path_specific, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_conditional, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_mediation, m)?)?;
    m.add_function(wrap_pyfunction!(identify_ate, m)?)?;
    m.add_function(wrap_pyfunction!(identify_ate_admg, m)?)?;
    Ok(())
}
