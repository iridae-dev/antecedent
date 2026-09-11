//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

use crate::graphs;
use crate::*;
use antecedent::{AcceptedGraph, StudyBuilder};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ResponseQuery, VariableId,
};
use antecedent_graph::Dag;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

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

struct AteGil {
    custom_validators: Vec<std::sync::Arc<dyn antecedent_validate::CustomEffectValidator>>,
    suite: antecedent::RefuteSuite,
    threads: u32,
    prior_mapping: Option<antecedent_io::PriorMapping>,
    composed_prior: Option<crate::prior_bank::OwnedComposedPrior>,
    cancel_token: Option<antecedent_core::CancellationToken>,
    progress: Option<std::sync::Arc<dyn antecedent_core::ProgressSink>>,
    stage_sink: Option<std::sync::Arc<dyn antecedent::StageResultSink>>,
    latency_mode: Option<antecedent::LatencyMode>,
    parsed_estimator_config: crate::estimator_config::ParsedEstimatorConfig,
}

fn parse_ate_gil(
    estimator: Option<&str>,
    prior_mapping: Option<&Bound<'_, PyDict>>,
    composed_prior: Option<&Bound<'_, PyDict>>,
    refute: Option<&Bound<'_, PyAny>>,
    validators: Option<&Bound<'_, PyAny>>,
    estimator_config: Option<&Bound<'_, PyDict>>,
    bootstrap: u32,
    threads: u32,
    latency: Option<&str>,
    cancel: Option<PyCancellationToken>,
    on_progress: Option<&Bound<'_, PyAny>>,
    on_stage: Option<&Bound<'_, PyAny>>,
) -> PyResult<AteGil> {
    let custom_validators = callbacks::parse_validators(validators)?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
    let latency_mode = match latency {
        None => None,
        Some(s) => Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
            PyValueError::new_err(format!("unknown latency={s:?}; use interactive|standard|report"))
        })?),
    };
    Ok(AteGil {
        custom_validators,
        suite: suite_from_refute(refute)?,
        threads,
        prior_mapping: parse_prior_mapping(prior_mapping)?,
        composed_prior: match composed_prior {
            Some(d) => Some(crate::prior_bank::owned_composed_prior_from_dict(d)?),
            None => None,
        },
        cancel_token: cancel.map(|c| c.inner),
        progress: callbacks::progress_sink_from_py(on_progress)?,
        stage_sink: callbacks::stage_sink_from_py(on_stage)?,
        latency_mode,
        parsed_estimator_config: crate::estimator_config::parse_estimator_config(
            estimator_config,
            estimator,
            bootstrap,
        )?,
    })
}

fn finish_static_ate(
    names: &[String],
    data: antecedent_data::TabularData,
    bytes_borrowed: Option<u64>,
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
    gil: AteGil,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    seed: u64,
    bootstrap: u32,
    accepted: bool,
    include_posterior_artifact: bool,
    pop_spec: Option<antecedent_core::TargetPopulation>,
    registry: Option<antecedent_core::PopulationRegistry>,
    outcome_functional: Option<antecedent_core::OutcomeFunctional>,
) -> PyResult<AteAnalysisResult> {
    let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
    let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
    let dag = dag_from_named_edges(data.schema(), &edges)?;
    let mut query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
    if let Some(pop) = pop_spec {
        query = query.with_target_population(pop);
    }
    if let Some(functional) = outcome_functional {
        query = query.with_outcome_functional(functional);
    }
    let crate::estimator_config::ParsedEstimatorConfig {
        spec: configured_spec,
        rd_running_variable: configured_rv,
        rd_cutoff: configured_cutoff,
        rd_bandwidth: configured_bandwidth,
    } = gil.parsed_estimator_config;
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
    let mut builder = bind_dag(Study::tabular(data), dag, accepted)
        .query(query)
        .refute(gil.suite)
        .custom_validators(gil.custom_validators);
    if configured_spec.is_none() {
        builder = builder.bootstrap_replicates(bootstrap);
    }
    if let Some(mode) = gil.latency_mode {
        builder = builder.latency_mode(mode);
    }
    if let Some(sink) = gil.stage_sink {
        builder = builder.stage_sink(sink);
    }
    if let Some(reg) = registry {
        builder = builder.population_registry(reg);
    }
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
        names,
        builder,
        inference.as_deref(),
        n_draws,
        prior_scale,
        prior_artifact.as_deref(),
        gil.prior_mapping,
        gil.composed_prior,
        seed,
        gil.threads,
        gil.cancel_token,
        gil.progress,
        include_posterior_artifact,
        bytes_borrowed,
    )
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

// Python batch query: treatment, outcome, control, active, functional specification.
type PyBatchQuery<'py> = (String, String, f64, f64, Option<Bound<'py, PyDict>>);

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

pub(crate) fn parse_outcome_functional(
    spec: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<antecedent_core::OutcomeFunctional>> {
    let Some(d) = spec else {
        return Ok(None);
    };
    let kind: String = d
        .get_item("kind")?
        .ok_or_else(|| PyValueError::new_err("outcome_functional requires 'kind'"))?
        .extract()?;
    Ok(Some(match kind.to_ascii_lowercase().as_str() {
        "mean" => antecedent_core::OutcomeFunctional::Mean,
        "exceedance" => {
            let threshold: f64 = d
                .get_item("threshold")?
                .ok_or_else(|| PyValueError::new_err("exceedance requires 'threshold'"))?
                .extract()?;
            antecedent_core::OutcomeFunctional::exceedance(threshold)
        }
        "exceedance_grid" | "grid" => {
            let thresholds: Vec<f64> = d
                .get_item("thresholds")?
                .ok_or_else(|| PyValueError::new_err("exceedance_grid requires 'thresholds'"))?
                .extract()?;
            antecedent_core::OutcomeFunctional::exceedance_grid(thresholds)
        }
        "quantile" => {
            let tau: f64 = d
                .get_item("tau")?
                .ok_or_else(|| PyValueError::new_err("quantile requires 'tau'"))?
                .extract()?;
            antecedent_core::OutcomeFunctional::quantile(tau)
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown outcome_functional kind {other:?}"
            )));
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
            if let Ok(spec) = v.cast::<PyDict>() {
                let weights: Vec<f64> = spec
                    .get_item("weights")?
                    .ok_or_else(|| PyValueError::new_err("missing weights"))?
                    .extract()?;
                let depends: Vec<u32> = spec
                    .get_item("depends_on")?
                    .ok_or_else(|| PyValueError::new_err("missing depends_on"))?
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
    outcome_functional=None,
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
    outcome_functional: Option<Bound<'_, PyDict>>,
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
    let outcome_functional = parse_outcome_functional(outcome_functional.as_ref())?;
    let registry = parse_population_registry(
        population_predicates.as_ref(),
        population_distributions.as_ref(),
    )?;
    let gil = parse_ate_gil(
        estimator.as_deref(),
        prior_mapping,
        composed_prior,
        refute.as_ref(),
        validators.as_ref(),
        estimator_config.as_ref(),
        bootstrap,
        threads,
        latency.as_deref(),
        cancel,
        on_progress.as_ref(),
        on_stage.as_ref(),
    )?;
    let data = tabular_from_numpy(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        finish_static_ate(
            &names,
            data,
            None,
            edges,
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
            gil,
            running_variable,
            cutoff,
            bandwidth,
            seed,
            bootstrap,
            accepted,
            return_posterior_artifact,
            pop_spec,
            registry,
            outcome_functional,
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
    let gil = parse_ate_gil(
        estimator.as_deref(),
        prior_mapping,
        composed_prior,
        refute.as_ref(),
        validators.as_ref(),
        estimator_config.as_ref(),
        bootstrap,
        threads,
        latency.as_deref(),
        cancel,
        on_progress.as_ref(),
        on_stage.as_ref(),
    )?;
    detach_catch(py, move || {
        finish_static_ate(
            &names,
            data,
            Some(bytes_borrowed),
            edges,
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
            gil,
            running_variable,
            cutoff,
            bandwidth,
            seed,
            bootstrap,
            accepted,
            return_posterior_artifact,
            None,
            None,
            None,
        )
    })
}

fn parse_latency_mode(latency: Option<&str>) -> PyResult<Option<antecedent::LatencyMode>> {
    match latency {
        None => Ok(None),
        Some(s) => Ok(Some(antecedent::LatencyMode::parse(s).ok_or_else(|| {
            PyValueError::new_err(format!("unknown latency={s:?}; use interactive|standard|report"))
        })?)),
    }
}

type AteBatchQuerySpec = (String, String, f64, f64, Option<antecedent_core::OutcomeFunctional>);

fn parse_ate_batch_query_specs(queries: Vec<PyBatchQuery<'_>>) -> PyResult<Vec<AteBatchQuerySpec>> {
    let mut parsed = Vec::with_capacity(queries.len());
    for (treatment, outcome, control, active, functional) in queries {
        parsed.push((
            treatment,
            outcome,
            control,
            active,
            parse_outcome_functional(functional.as_ref())?,
        ));
    }
    Ok(parsed)
}

fn compile_batch_study(
    data: antecedent_data::TabularData,
    edges: Vec<(String, String)>,
    identifier: Option<String>,
    estimator: Option<String>,
    suite: antecedent::RefuteSuite,
    bootstrap: u32,
    latency_mode: Option<antecedent::LatencyMode>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u32>>,
    estimate_rows: Option<Vec<u32>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
) -> PyResult<antecedent::BatchStudy> {
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
    }
    .bootstrap_replicates(bootstrap)
    .refute(suite);
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
    suite: antecedent::RefuteSuite,
    bootstrap: u32,
    latency_mode: Option<antecedent::LatencyMode>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u32>>,
    estimate_rows: Option<Vec<u32>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
) -> PyResult<(antecedent::BatchStudy, Vec<AverageEffectQuery>)> {
    let mut ate_queries = Vec::with_capacity(parsed_queries.len());
    for (treatment, outcome, control, active, functional) in &parsed_queries {
        let t_id = data.schema().id_of(treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(outcome).map_err(py_err)?;
        let mut query = AverageEffectQuery::with_levels(t_id, y_id, *control, *active);
        if let Some(functional) = functional.clone() {
            query = query.with_outcome_functional(functional);
        }
        ate_queries.push(query);
    }
    let batch = compile_batch_study(
        data,
        edges,
        identifier,
        estimator,
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

type CellBatchQuerySpec =
    (String, Vec<String>, Vec<String>, Vec<Vec<f64>>, Option<antecedent_core::OutcomeFunctional>);
type PyCellBatchQuery<'py> =
    (String, Vec<String>, Vec<String>, Vec<Vec<f64>>, Option<Bound<'py, PyDict>>);

fn parse_cell_batch_query_specs(
    queries: Vec<PyCellBatchQuery<'_>>,
) -> PyResult<Vec<CellBatchQuerySpec>> {
    let mut parsed = Vec::with_capacity(queries.len());
    for (outcome, treatments, kinds, parameters, functional) in queries {
        parsed.push((
            outcome,
            treatments,
            kinds,
            parameters,
            parse_outcome_functional(functional.as_ref())?,
        ));
    }
    Ok(parsed)
}

fn compile_cell_batch(
    data: antecedent_data::TabularData,
    edges: Vec<(String, String)>,
    parsed_queries: Vec<CellBatchQuerySpec>,
    identifier: Option<String>,
    estimator: Option<String>,
    suite: antecedent::RefuteSuite,
    bootstrap: u32,
    latency_mode: Option<antecedent::LatencyMode>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u32>>,
    estimate_rows: Option<Vec<u32>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
    family_contrast: Option<String>,
) -> PyResult<(antecedent::BatchStudy, Vec<ResponseQuery>)> {
    let mut cell_queries = Vec::with_capacity(parsed_queries.len());
    for (outcome, treatments, kinds, parameters, functional) in &parsed_queries {
        let treatment_ids = crate::response_api::resolve_names(data.schema(), treatments)?;
        let outcome_ids = crate::response_api::resolve_names(data.schema(), &[outcome.clone()])?;
        let built = crate::response_api::build_functional(
            "intervention_response",
            &treatment_ids,
            &outcome_ids,
            None,
            None,
            None,
            Some(kinds.clone()),
            Some(parameters.clone()),
            1,
            antecedent_core::DerivativeScale::Identity,
            antecedent_core::DerivativeWeighting::Observed,
        )?;
        let mut query = ResponseQuery::new(built);
        if let Some(functional) = functional.clone() {
            query = query.with_outcome_functional(functional);
        }
        cell_queries.push(query);
    }
    let batch = compile_batch_study(
        data,
        edges,
        identifier,
        estimator,
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
    refute=None,
    seed=1,
    bootstrap=50,
    threads=1,
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
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    latency: Option<String>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u32>>,
    estimate_rows: Option<Vec<u32>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
) -> PyResult<Vec<AteAnalysisResult>> {
    let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
    let suite = suite_from_refute(refute.as_ref())?;
    let latency_mode = parse_latency_mode(latency.as_deref())?;
    let parsed_queries = parse_ate_batch_query_specs(queries)?;
    detach_catch(py, move || {
        let (batch, ate_queries) = compile_ate_batch(
            data,
            edges,
            parsed_queries,
            identifier,
            estimator,
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
        let ctx = py_execution_context(seed, threads);
        let results = batch.estimate_many(&ate_queries, &ctx).map_err(py_err)?;
        results.into_iter().map(|r| ate_result_from_analysis(&names, r, false)).collect()
    })
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
    data: antecedent_data::TabularData,
    bytes_borrowed: Option<u64>,
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
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
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
            data,
            bytes_borrowed,
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
    data: antecedent_data::TabularData,
    bytes_borrowed: Option<u64>,
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
            AcceptedGraph::cpdag(cpdag.clone()).map_err(py_err)?;
            builder.graph(cpdag)
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
        bytes_borrowed,
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
    require_named_graph_order(&graph.names, &names, "Pag")?;
    let data = tabular_from_numpy(&names, &columns)?;
    drop(columns);
    analyze_ate_typed_graph(
        py,
        names,
        data,
        None,
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
    require_named_graph_order(&graph.names, &names, "Cpdag")?;
    let data = tabular_from_numpy(&names, &columns)?;
    drop(columns);
    analyze_ate_typed_graph(
        py,
        names,
        data,
        None,
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
    require_named_graph_order(&graph.names, &names, "Admg")?;
    let data = tabular_from_numpy(&names, &columns)?;
    drop(columns);
    analyze_ate_typed_graph(
        py,
        names,
        data,
        None,
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

macro_rules! typed_ate_arrow_c {
    ($fn_name:ident, $graph_ty:ty, $variant:ident, $field:ident) => {
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
        fn $fn_name(
            py: Python<'_>,
            names: Vec<String>,
            columns: Vec<Bound<'_, PyAny>>,
            graph: $graph_ty,
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
            require_named_graph_order(&graph.names, &names, stringify!($variant))?;
            let (data, bytes_borrowed) = tabular_from_arrow_c_objs(py, names.clone(), columns)?;
            analyze_ate_typed_graph(
                py,
                names,
                data,
                Some(bytes_borrowed),
                StaticGraphInput::$variant(graph.$field),
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
    };
}

typed_ate_arrow_c!(analyze_ate_pag_arrow_c, graphs::Pag, Pag, pag);
typed_ate_arrow_c!(analyze_ate_cpdag_arrow_c, graphs::Cpdag, Cpdag, cpdag);
typed_ate_arrow_c!(analyze_ate_admg_arrow_c, graphs::Admg, Admg, admg);

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
    columns: Vec<Bound<'_, PyAny>>,
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
    let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let (ci_impl, _ci_name, is_ci_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    let parsed_estimator_config = crate::estimator_config::parse_estimator_config(
        estimator_config.as_ref(),
        estimator.as_deref(),
        bootstrap,
    )?;
    let threads = if is_ci_callback || !custom_validators.is_empty() { 1 } else { threads };
    let soft_weight = soft_weight.to_string();
    detach_catch(py, move || {
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

        // Graph-posterior discovery: no discrete graph to review — wire the
        // posterior directly. Frequentist inference is licensed for AverageEffect
        // on DAG atoms; other queries still refuse at `build()`.
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
    refute=None,
    seed=1,
    threads=1
))]
fn analyze_distribution(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    outcome: String,
    interventions: std::collections::HashMap<String, f64>,
    conditioning: Option<Vec<String>>,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
    let suite = suite_from_refute(refute.as_ref())?;
    detach_catch(py, move || {
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
            .refute(suite)
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
    threads=1,
    refute=None
))]
fn analyze_path_specific(
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
    seed: u64,
    bootstrap: u32,
    threads: u32,
    refute: Option<Bound<'_, PyAny>>,
) -> PyResult<AteAnalysisResult> {
    let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
    let suite = suite_from_refute(refute.as_ref())?;
    detach_catch(py, move || {
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
            .refute(suite)
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
    let certificate_json = crate::identification_details::analysis_to_json(&result, names)?;
    let adjustment_set: Vec<String> = crate::public_adjustment_set(
        result.identification.status,
        result
            .estimand
            .adjustment_set
            .iter()
            .map(|id| {
                names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("var{}", id.raw()))
            })
            .collect(),
    );

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
        functional_means: result
            .estimate
            .score_inference
            .as_ref()
            .map(|s| s.raw_means.clone())
            .or_else(|| {
                result
                    .estimate
                    .score_table
                    .as_ref()
                    .and_then(|t| t.summarize(None).ok().map(|s| s.means.to_vec()))
            }),
        joint_covariance: result
            .estimate
            .joint_covariance
            .as_ref()
            .map(|c| (0..c.dim).map(|i| (0..c.dim).map(|j| c.get(i, j)).collect()).collect()),
        score_inference: result.estimate.score_inference.as_ref().map(ScoreInferenceSection::from),
        scenario_effects: result.estimate.scenario_effects.as_ref().map(|v| v.to_vec()),
        scenario_intervals: result.estimate.scenario_intervals.as_ref().map(|v| v.to_vec()),
        exceedance_cdf: result.estimate.exceedance_cdf.as_ref().map(|v| v.to_vec()),
        monotone_rearranged: result.estimate.monotone_rearranged,
        interaction_structurally_zero: result
            .response
            .as_ref()
            .map(|r| r.interaction_structurally_zero)
            .or(Some(result.estimate.interaction_structurally_zero)),
        score_table: result.estimate.score_table.as_ref().map(|t| ScoreTableSection {
            n_rows: t.n_rows,
            n_folds: t.n_folds,
            provenance: t.nuisance_provenance.to_string(),
            columns: t.columns.iter().map(|c| (c.arm, c.threshold)).collect(),
            scores: t.scores.to_vec(),
            row_index: t.row_index.to_vec(),
            fold_ids: t.fold_ids.to_vec(),
            adjustment_set: t.adjustment_set.iter().map(|v| v.raw()).collect(),
            observed_arm: t.observed_arm.to_vec(),
            propensities: t.propensities.to_vec(),
            observed_outcome: t.observed_outcome.to_vec(),
            treatment: t.treatment.raw(),
            intervened: t.intervened.iter().map(|v| v.raw()).collect(),
        }),
        simultaneous_interval: result.estimate.simultaneous_interval,
        adjusted_p_values: result.estimate.adjusted_p_values,
        family_contrast: result.estimate.family_contrast,
        candidate_selection: result
            .estimate
            .candidate_selection
            .as_ref()
            .map(|s| CandidateSelectionSection {
                screen_id: s.screen_id.to_string(),
                procedure: s.procedure.to_string(),
                winner_index: s.winner_index,
                family_size: s.family_size,
                screen_rows: s.screen_rows.to_vec(),
                estimate_rows: s.estimate_rows.to_vec(),
                disjoint: s.disjoint,
            })
            .or_else(|| {
                result.candidate_selection.as_ref().map(|s| CandidateSelectionSection {
                    screen_id: s.screen_id.to_string(),
                    procedure: s.procedure.as_str().to_string(),
                    winner_index: s.winner_index,
                    family_size: s.family_size,
                    screen_rows: s.screen_rows.to_vec(),
                    estimate_rows: s.estimate_rows.to_vec(),
                    disjoint: s.disjoint,
                })
            }),
        evalue: result.estimate.evalue,
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
    let validation = ValidationSection::from_reports(refutations.clone(), &result.diagnostics);
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

    let (evidence_status, allowlist_reason, allowlist_parent) =
        crate::evidence_status_parts(result.support_status);

    Ok(AteAnalysisResult {
        certificate_json,
        ate: result.estimate.ate,
        se_analytic: result.estimate.se_analytic,
        se_bootstrap: result.estimate.se_bootstrap,
        bootstrap_replicates_failed: result.estimate.bootstrap_replicates_failed,
        adjustment_set,
        identification_status,
        refutation_passed: validation.passed,
        refutation_ran: validation.ran,
        refutation_count: validation.count,
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
        assumptions: result
            .estimate
            .assumptions
            .entries
            .iter()
            .map(|r| format!("{:?}", r.assumption))
            .collect(),
        support_diagnostics: result
            .diagnostics
            .iter()
            .filter(|d| d.code.contains("support") || d.code.contains("overlap"))
            .map(|d| d.message.to_string())
            .collect(),
        unit_effects: result.counterfactual.as_ref().map(|cf| cf.unit_effects.to_vec()),
        mediation_total: result.mediation.as_ref().and_then(|m| m.total),
        mediation_direct: result.mediation.as_ref().and_then(|m| m.direct),
        mediation_mediated: result.mediation.as_ref().and_then(|m| m.mediated),
        evidence_status,
        allowlist_reason,
        allowlist_parent,
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
    refute=None, validators=None, seed=1, bootstrap=50, threads=1, accepted=false,
    outcome_functional=None,
))]
fn analyze_conditional(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
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
    accepted: bool,
    outcome_functional: Option<Bound<'_, PyDict>>,
) -> PyResult<AteAnalysisResult> {
    let outcome_functional = parse_outcome_functional(outcome_functional.as_ref())?;
    let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
    detach_catch(py, move || {
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let w_id = data.schema().id_of(&modifier).map_err(py_err)?;
        let mut inner = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level)
            .with_effect_modifiers([w_id]);
        if let Some(functional) = outcome_functional {
            inner = inner.with_outcome_functional(functional);
        }
        let cq = ConditionalEffectQuery::try_new(inner)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dag = dag_from_named_edges(data.schema(), &edges)?;
        let analysis = bind_dag(Study::tabular(data), dag, accepted)
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
    columns: Vec<Bound<'_, PyAny>>,
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
    let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
    let suite = suite_from_refute(refute.as_ref())?;
    let contrast = contrast.to_string();
    detach_catch(py, move || {
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

fn structure_var_id(names: &[String], name: &str) -> PyResult<VariableId> {
    names
        .iter()
        .position(|n| n == name)
        .and_then(|i| u32::try_from(i).ok())
        .map(VariableId::from_raw)
        .ok_or_else(|| PyValueError::new_err(format!("unknown variable {name:?}")))
}

fn structure_query(
    kind: &str,
    names: &[String],
    treatment: &str,
    outcome: &str,
    modifier: Option<&str>,
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
            let modifier = modifier.ok_or_else(|| {
                PyValueError::new_err("ConditionalEffect identify requires modifier=")
            })?;
            let w_id = structure_var_id(names, modifier)?;
            let inner = AverageEffectQuery::binary_ate(t_id, y_id).with_effect_modifiers([w_id]);
            let cq = ConditionalEffectQuery::try_new(inner)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
                    return Err(PyValueError::new_err("window requires a sustained policy"));
                }
                query.policy = TemporalPolicy::sustained(from, until);
            }
            Ok(CausalQuery::TemporalEffect(query))
        }
        other => {
            Err(PyValueError::new_err(format!("identify_structure unknown query kind {other:?}")))
        }
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
#[pyfunction]
#[pyo3(signature = (
    graph,
    query_kind,
    treatment,
    outcome,
    *,
    identifier=None,
    modifier=None,
    policy=None,
    treatment_lag=1,
    horizon_steps=1,
    active_level=1.0,
    window=None,
    treatments=None,
    include_details=false,
))]
#[allow(clippy::too_many_arguments)]
fn identify_structure(
    py: Python<'_>,
    graph: Bound<'_, PyAny>,
    query_kind: String,
    treatment: String,
    outcome: String,
    identifier: Option<String>,
    modifier: Option<String>,
    policy: Option<String>,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    window: Option<(i32, i32)>,
    treatments: Option<Vec<String>>,
    include_details: bool,
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
            return Err(PyValueError::new_err(
                "identify_structure requires a Dag, Cpdag, Pag, TemporalDag, TemporalCpdag, or TemporalPag",
            ));
        };
        let query = structure_query(
            &query_kind,
            &names,
            &treatment,
            &outcome,
            modifier.as_deref(),
            policy.as_deref(),
            treatment_lag,
            horizon_steps,
            active_level,
            window,
            treatments,
        )?;
        let identification = if let Some(id) = identifier {
            let strategy = id
                .parse::<antecedent::IdentifierId>()
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            antecedent::identify_with(&structure, &query, strategy).map_err(py_err)?
        } else {
            antecedent::identify(&structure, &query).map_err(py_err)?
        };
        let details = if include_details {
            Some(crate::identification_details::to_json(
                &identification,
                &query,
                &names,
                structure.class().as_str(),
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

/// Average effect from a supplied graph posterior (known-truth / replay atoms).
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    posterior,
    treatment,
    outcome,
    *,
    control_level=0.0,
    active_level=1.0,
    inference="conjugate",
    n_draws=1000,
    prior_scale=10.0,
    refute=None,
    seed=1,
    bootstrap=0,
    threads=1,
    cancel=None,
    on_progress=None,
))]
fn analyze_ate_graph_posterior(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    posterior: Bound<'_, crate::bayesian::PyGraphPosterior>,
    treatment: String,
    outcome: String,
    control_level: f64,
    active_level: f64,
    inference: &str,
    n_draws: usize,
    prior_scale: f64,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    cancel: Option<PyCancellationToken>,
    on_progress: Option<Bound<'_, PyAny>>,
) -> PyResult<AteAnalysisResult> {
    let gp = {
        let posterior = posterior.borrow();
        posterior.require_bound_to(&names)?;
        posterior.to_rust()?
    };
    let suite = suite_from_refute(refute.as_ref())?;
    let cancel_token = cancel.map(|token| token.inner);
    let progress = callbacks::progress_sink_from_py(on_progress.as_ref())?;
    let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
    let inference = inference.to_string();
    detach_catch(py, move || {
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
        let builder = Study::tabular(data)
            .graph_posterior(gp)
            .query(query)
            .refute(suite)
            .bootstrap_replicates(bootstrap);
        run_static_ate_from_builder(
            &names,
            builder,
            Some(&inference),
            n_draws,
            prior_scale,
            None,
            None,
            None,
            seed,
            threads,
            cancel_token,
            progress,
            false,
            None,
        )
    })
}

#[pyfunction]
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
    refute=None,
    seed=1,
    bootstrap=0,
    threads=1,
    outcome_functional=None,
    latency=None,
    identifier=None,
    validators=None,
    cancel=None,
    on_progress=None,
))]
fn analyze_ate_tiered(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    tiers: Vec<Vec<String>>,
    within_tier: String,
    treatment: String,
    outcome: String,
    control_level: f64,
    active_level: f64,
    estimator: Option<String>,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    outcome_functional: Option<Bound<'_, pyo3::types::PyDict>>,
    latency: Option<String>,
    identifier: Option<String>,
    validators: Option<Bound<'_, PyAny>>,
    cancel: Option<PyCancellationToken>,
    on_progress: Option<Bound<'_, PyAny>>,
) -> PyResult<AteAnalysisResult> {
    if identifier.is_some() {
        return Err(PyValueError::new_err(
            "TieredBackground selects its own identifier; omit identifier",
        ));
    }
    if validators.is_some() {
        return Err(PyValueError::new_err(
            "analyze_ate_tiered does not take validators; omit validators",
        ));
    }
    if cancel.is_some() {
        return Err(PyValueError::new_err("analyze_ate_tiered does not take cancel; omit cancel"));
    }
    if on_progress.is_some() {
        return Err(PyValueError::new_err(
            "analyze_ate_tiered does not take on_progress; omit on_progress",
        ));
    }
    let latency_mode = parse_latency_mode(latency.as_deref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let outcome_functional = parse_outcome_functional(outcome_functional.as_ref())?;
    let data = tabular_from_numpy(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let within = crate::parse_within_tier(Some(within_tier.as_str()))?;
        let named: Vec<Vec<&str>> =
            tiers.iter().map(|tier| tier.iter().map(String::as_str).collect()).collect();
        let background =
            antecedent_graph::TieredBackground::from_named(data.schema(), &named, within)
                .map_err(py_err)?;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let mut query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
        if let Some(functional) = outcome_functional {
            query = query.with_outcome_functional(functional);
        }
        let mut builder = Study::tabular(data)
            .tiered_background(background)
            .map_err(py_err)?
            .query(query)
            .refute(suite)
            .bootstrap_replicates(bootstrap);
        if let Some(mode) = latency_mode {
            builder = builder.latency_mode(mode);
        }
        if let Some(est) = estimator {
            builder = builder.estimator(
                est.parse::<antecedent::EstimatorId>()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            );
        }
        let ctx = crate::py_execution_context_ext(
            seed,
            threads,
            None,
            None,
            Some(crate::PY_DEFAULT_CACHE_MAX_BYTES),
        );
        let result = builder.build().map_err(py_err)?.run(&ctx).map_err(py_err)?;
        ate_result_from_analysis(&names, result, false)
    })
}

fn parse_candidate_screen(
    screen_id: Option<String>,
    procedure: Option<String>,
    screen_rows: Option<Vec<u32>>,
    estimate_rows: Option<Vec<u32>>,
) -> PyResult<Option<antecedent::CandidateScreen>> {
    if screen_id.is_none()
        && procedure.is_none()
        && screen_rows.is_none()
        && estimate_rows.is_none()
    {
        return Ok(None);
    }
    let screen_id = screen_id.ok_or_else(|| {
        PyValueError::new_err(
            "candidate screen requires screen_id, procedure, screen_rows, and estimate_rows",
        )
    })?;
    let procedure = match procedure.as_deref().unwrap_or("unrecorded") {
        "max_t" => antecedent::CandidateProcedure::MaxT,
        "bh" => antecedent::CandidateProcedure::BenjaminiHochberg,
        "by" => antecedent::CandidateProcedure::BenjaminiYekutieli,
        "unrecorded" => antecedent::CandidateProcedure::Unrecorded,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown screen procedure {other:?}; use max_t|bh|by|unrecorded"
            )));
        }
    };
    Ok(Some(antecedent::CandidateScreen {
        screen_id: std::sync::Arc::from(screen_id),
        procedure,
        screen_rows: std::sync::Arc::from(screen_rows.unwrap_or_default()),
        estimate_rows: std::sync::Arc::from(estimate_rows.unwrap_or_default()),
    }))
}

/// Frozen multi-query batch: identify once, estimate on later tables.
#[pyclass(name = "PreparedBatch")]
pub struct PyPreparedBatch {
    inner: std::sync::Arc<antecedent::PreparedBatch>,
    names: Vec<String>,
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

    #[pyo3(signature = (names, columns, *, seed=1, threads=1))]
    fn estimate(
        &self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        seed: u64,
        threads: u32,
    ) -> PyResult<Vec<AteAnalysisResult>> {
        if names != self.names {
            return Err(PyValueError::new_err(
                "prepared batch estimate requires the same column names (order) as prepare",
            ));
        }
        let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
        let inner = std::sync::Arc::clone(&self.inner);
        detach_catch(py, move || {
            let ctx = py_execution_context(seed, threads);
            let results = inner.estimate(&data, &ctx).map_err(py_err)?;
            results.into_iter().map(|r| ate_result_from_analysis(&names, r, false)).collect()
        })
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
    refute=None,
    seed=1,
    bootstrap=50,
    threads=1,
    latency=None,
    screen_id=None,
    screen_procedure=None,
    screen_rows=None,
    estimate_rows=None,
    tiers=None,
    within_tier=None,
))]
fn prepare_ate_batch(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    queries: Vec<PyBatchQuery<'_>>,
    identifier: Option<String>,
    estimator: Option<String>,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    latency: Option<String>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u32>>,
    estimate_rows: Option<Vec<u32>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
) -> PyResult<PyPreparedBatch> {
    let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
    let suite = suite_from_refute(refute.as_ref())?;
    let latency_mode = parse_latency_mode(latency.as_deref())?;
    let parsed_queries = parse_ate_batch_query_specs(queries)?;
    detach_catch(py, move || {
        let (batch, ate_queries) = compile_ate_batch(
            data,
            edges,
            parsed_queries,
            identifier,
            estimator,
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
        let ctx = py_execution_context(seed, threads);
        let prepared = batch.prepare(&ate_queries, &ctx).map_err(py_err)?;
        Ok(PyPreparedBatch { inner: std::sync::Arc::new(prepared), names })
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
    refute=None,
    seed=1,
    bootstrap=50,
    threads=1,
    latency=None,
    screen_id=None,
    screen_procedure=None,
    screen_rows=None,
    estimate_rows=None,
    tiers=None,
    within_tier=None,
    family_contrast="cell_minus_control",
))]
fn prepare_cells_batch(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    queries: Vec<PyCellBatchQuery<'_>>,
    identifier: Option<String>,
    estimator: Option<String>,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    latency: Option<String>,
    screen_id: Option<String>,
    screen_procedure: Option<String>,
    screen_rows: Option<Vec<u32>>,
    estimate_rows: Option<Vec<u32>>,
    tiers: Option<Vec<Vec<String>>>,
    within_tier: Option<String>,
    family_contrast: Option<&str>,
) -> PyResult<PyPreparedBatch> {
    let (data, _) = tabular_from_py_columns(py, names.clone(), columns)?;
    let suite = suite_from_refute(refute.as_ref())?;
    let latency_mode = parse_latency_mode(latency.as_deref())?;
    let parsed_queries = parse_cell_batch_query_specs(queries)?;
    let family_contrast = family_contrast.map(str::to_owned);
    detach_catch(py, move || {
        let (batch, cell_queries) = compile_cell_batch(
            data,
            edges,
            parsed_queries,
            identifier,
            estimator,
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
        let ctx = py_execution_context(seed, threads);
        let prepared = batch.prepare_cells(&cell_queries, &ctx).map_err(py_err)?;
        Ok(PyPreparedBatch { inner: std::sync::Arc::new(prepared), names })
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_ate, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_tiered, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_pag, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_cpdag, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_admg, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_arrow_c, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_pag_arrow_c, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_cpdag_arrow_c, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_admg_arrow_c, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_many, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_ate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_cells_batch, m)?)?;
    m.add_class::<PyPreparedBatch>()?;
    m.add_function(wrap_pyfunction!(analyze_ate_discover, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_graph_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_path_specific, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_conditional, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_mediation, m)?)?;
    m.add_function(wrap_pyfunction!(identify_ate, m)?)?;
    m.add_function(wrap_pyfunction!(identify_ate_admg, m)?)?;
    m.add_function(wrap_pyfunction!(identify_structure, m)?)?;
    Ok(())
}
