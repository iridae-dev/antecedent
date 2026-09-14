//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

use crate::*;
use antecedent::StudyBuilder;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyfunction]
#[pyo3(signature = (names, columns, treatment, mediator, outcome, *, seed=1, threads=1))]
fn mediation_effects_summary(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    treatment: String,
    mediator: String,
    outcome: String,
    seed: u64,
    threads: u32,
) -> PyResult<MediationEffectsSummary> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (series, _) = series_from_batch(&batch)?;
        let t = schema_var_id(series.schema(), &treatment)?;
        let m = schema_var_id(series.schema(), &mediator)?;
        let y = schema_var_id(series.schema(), &outcome)?;
        let q = MediationQuery::binary(t, y, [m], MediationContrast::Total);
        let mut arena = CausalExprArena::new();
        let functional = arena.frontdoor_ate(t, y, &[m], Value::f64(1.0), Value::f64(0.0));
        let estimand =
            IdentifiedEstimand::frontdoor("temporal_mediation.total", Arc::from([m]), functional);
        let ctx = py_execution_context(seed, threads);
        let surface = TemporalMediationEstimator::new()
            .effect_surface(&series, &estimand, &q, &ctx)
            .map_err(py_err)?;
        Ok(MediationEffectsSummary {
            total: surface.total,
            direct: surface.direct,
            mediated: surface.mediated,
        })
    })
}

/// Intervene+predict summary (mean predicted outcome under do(parent=level)).
#[pyfunction]
#[pyo3(signature = (names, columns, target, parent, *, parent_lag=1, level=1.0))]
fn predict_intervened_summary(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    target: String,
    parent: String,
    parent_lag: u32,
    level: f64,
) -> PyResult<PredictSummary> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let (series, _) = series_from_batch(&batch)?;
        let y = schema_var_id(series.schema(), &target)?;
        let x = schema_var_id(series.schema(), &parent)?;
        let policy = KernelPolicy::default_policy();
        let pred = TemporalLinearPredictor::fit(
            &series,
            y,
            [antecedent_data::LaggedColumn { variable: x, lag: Lag::from_raw(parent_lag) }],
            &policy,
        )
        .map_err(py_err)?;
        let yhat = pred.predict_intervened(&series, x, level, &policy).map_err(py_err)?;
        let mean = yhat.iter().sum::<f64>() / yhat.len().max(1) as f64;
        Ok(PredictSummary { mean_prediction: mean, n: yhat.len() as u64 })
    })
}

/// RPCMCI summary (typed regimes, no single-graph collapse).
#[pyclass]
pub(crate) struct RpcmciDiscoverySummary {
    #[pyo3(get)]
    pub(crate) algorithm: String,
    #[pyo3(get)]
    pub(crate) n_regimes: u64,
    #[pyo3(get)]
    pub(crate) regime_ids: Vec<u32>,
    #[pyo3(get)]
    pub(crate) directed_edges: Vec<u64>,
    #[pyo3(get)]
    pub(crate) undirected_edges: Vec<u64>,
}

/// Mediation effects summary.
#[pyclass]
pub(crate) struct MediationEffectsSummary {
    #[pyo3(get)]
    pub(crate) total: f64,
    #[pyo3(get)]
    pub(crate) direct: f64,
    #[pyo3(get)]
    pub(crate) mediated: f64,
}

/// Prediction summary under intervention.
#[pyclass]
pub(crate) struct PredictSummary {
    #[pyo3(get)]
    pub(crate) mean_prediction: f64,
    #[pyo3(get)]
    pub(crate) n: u64,
}

/// Unified analysis result (static or temporal).
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct AnalysisResult {
    #[pyo3(get)]
    pub(crate) certificate_json: Option<String>,
    /// Scalar contrast when one exists. Function-valued results omit it.
    #[pyo3(get)]
    pub(crate) ate: Option<f64>,
    #[pyo3(get)]
    pub(crate) se_analytic: f64,
    #[pyo3(get)]
    pub(crate) se_bootstrap: Option<f64>,
    #[pyo3(get)]
    pub(crate) plan_id: String,
    #[pyo3(get)]
    pub(crate) modality: String,
    #[pyo3(get)]
    pub(crate) discovery_algorithm: Option<String>,
    #[pyo3(get)]
    pub(crate) structure_source: String,
    #[pyo3(get)]
    pub(crate) graph_review_required: bool,
    #[pyo3(get)]
    pub(crate) plan_identifier: Option<String>,
    #[pyo3(get)]
    pub(crate) plan_estimator: Option<String>,
    #[pyo3(get)]
    pub(crate) validation_suite: Option<String>,
    #[pyo3(get)]
    pub(crate) peak_memory_bytes: Option<u64>,
    #[pyo3(get)]
    pub(crate) identification_status: String,
    #[pyo3(get)]
    pub(crate) method: String,
    #[pyo3(get)]
    pub(crate) diagnostics: Vec<String>,
    #[pyo3(get)]
    pub(crate) provenance_node_count: usize,
    #[pyo3(get)]
    pub(crate) refutation_count: usize,
    /// Per-refuter records (name, comparison statistic, pass/fail), one per validator run.
    #[pyo3(get)]
    pub(crate) refutations: Vec<RefutationReportView>,
    #[pyo3(get)]
    pub(crate) worker_threads: u32,
    #[pyo3(get)]
    pub(crate) expected_python_crossings: u32,
    #[pyo3(get)]
    pub(crate) adjustment_set: Vec<String>,
    #[pyo3(get)]
    pub(crate) assumption_count: usize,
    #[pyo3(get)]
    pub(crate) derivation_step_count: usize,
    #[pyo3(get)]
    pub(crate) estimator_id: String,
    #[pyo3(get)]
    pub(crate) posterior_effect_mean: Option<f64>,
    #[pyo3(get)]
    pub(crate) posterior_effect_sd: Option<f64>,
    #[pyo3(get)]
    pub(crate) posterior_q025: Option<f64>,
    #[pyo3(get)]
    pub(crate) posterior_q975: Option<f64>,
    #[pyo3(get)]
    pub(crate) posterior_n_draws: Option<usize>,
    #[pyo3(get)]
    pub(crate) posterior_p_below_zero: Option<f64>,
    #[pyo3(get)]
    pub(crate) posterior_backend: Option<String>,
    #[pyo3(get)]
    pub(crate) posterior_artifact: Option<Vec<u8>>,
    #[pyo3(get)]
    pub(crate) posterior_unidentified_mass: Option<f64>,
    #[pyo3(get)]
    pub(crate) mediation_total: Option<f64>,
    #[pyo3(get)]
    pub(crate) mediation_direct: Option<f64>,
    #[pyo3(get)]
    pub(crate) mediation_mediated: Option<f64>,
    /// Horizon-indexed temporal mediation result. Empty for non-mediation queries.
    #[pyo3(get)]
    pub(crate) mediation_horizons: Vec<u32>,
    #[pyo3(get)]
    pub(crate) mediation_effects: Vec<f64>,
    #[pyo3(get)]
    pub(crate) mediation_totals: Vec<f64>,
    #[pyo3(get)]
    pub(crate) mediation_directs: Vec<f64>,
    #[pyo3(get)]
    pub(crate) mediation_mediated_effects: Vec<f64>,
    #[pyo3(get)]
    pub(crate) mediation_identification_statuses: Vec<String>,
    #[pyo3(get)]
    pub(crate) mediation_methods: Vec<String>,
    #[pyo3(get)]
    pub(crate) mediation_adjustments: Vec<Vec<(u32, i32)>>,
    #[pyo3(get)]
    pub(crate) mediation_uncertainty_kinds: Vec<String>,
    #[pyo3(get)]
    pub(crate) mediation_standard_deviations: Vec<Option<f64>>,
    #[pyo3(get)]
    pub(crate) mediation_q025: Vec<Option<f64>>,
    #[pyo3(get)]
    pub(crate) mediation_q975: Vec<Option<f64>>,
    #[pyo3(get)]
    pub(crate) mediation_identified_lower: Vec<Option<f64>>,
    #[pyo3(get)]
    pub(crate) mediation_identified_upper: Vec<Option<f64>>,
    #[pyo3(get)]
    pub(crate) mediation_joint_posterior: Option<bool>,
    /// Nested identification section — every field is populated on the temporal path
    /// (unlike `estimate`/`performance`, this DTO carries the full identification set).
    #[pyo3(get)]
    pub(crate) identification: IdentificationSection,
    /// Nested estimate section. `overlap_ess` / `overlap_propensity_min` read `None`
    /// in practice — every temporal path fixes `OverlapPolicy::ExplicitOverride`,
    /// under which the shared estimator never computes an overlap report — but this
    /// reads the real field rather than hardcoding the assumption.
    #[pyo3(get)]
    pub(crate) estimate: EstimateSection,
    /// Nested posterior section — every field is populated on the temporal path.
    #[pyo3(get)]
    pub(crate) posterior: PosteriorSection,
    /// Nested validation section, using the same pass/ran aggregate rule as the
    /// static DTO.
    #[pyo3(get)]
    pub(crate) validation: ValidationSection,
    /// Nested performance section. `bootstrap_replicates_ok`, `n_draws` (posterior
    /// draw effort), and `stage_timings` are genuinely never populated on any
    /// temporal execution path and read as `None` / empty; every other field is
    /// read from the same `StudyResult.performance` record the static DTO uses.
    #[pyo3(get)]
    pub(crate) performance: PerformanceSection,
    /// Support-matrix evidence contract (`licensed` or `allowed_unlicensed`).
    #[pyo3(get)]
    pub(crate) evidence_status: Option<String>,
    #[pyo3(get)]
    pub(crate) allowlist_reason: Option<String>,
    #[pyo3(get)]
    pub(crate) allowlist_parent: Option<String>,
    #[pyo3(get)]
    pub(crate) structural_weight_basis: Option<String>,
    #[pyo3(get)]
    pub(crate) structural_identified_mass: Option<f64>,
    #[pyo3(get)]
    pub(crate) structural_unidentified_mass: Option<f64>,
    #[pyo3(get)]
    pub(crate) structural_unevaluable_mass: Option<f64>,
    /// Scalar identified set `(lower, upper)` over identified completions.
    #[pyo3(get)]
    pub(crate) structural_identified_set: Option<(f64, f64)>,
    /// Interval for the identified set at `structural_identified_set_interval_level`
    /// (construction in `structural_identified_set_interval_method`).
    #[pyo3(get)]
    pub(crate) structural_identified_set_interval: Option<(f64, f64)>,
    #[pyo3(get)]
    pub(crate) structural_identified_set_interval_level: Option<f64>,
    /// `imbens_manski_shared_block` (Frequentist coverage of the true
    /// completion's effect) or `product_posterior_envelope_quantile` (Bayesian).
    #[pyo3(get)]
    pub(crate) structural_identified_set_interval_method: Option<String>,
    /// The set spans a capped completion enumeration (retained completions only).
    #[pyo3(get)]
    pub(crate) structural_identified_set_interval_truncated: Option<bool>,
}

/// Run temporal effect analysis with a supplied lagged edge list.
///
/// `edges` are `(source, source_lag, target, target_lag)` with lags ≥ 0.
/// `treatment_lag` is the pulse/sustained offset as a non-negative lag
/// (policy origin at `-treatment_lag`).
/// `policy` is `"pulse"` (default) or `"sustained"`.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    edges,
    treatment,
    outcome,
    *,
    treatment_lag=crate::temporal_license::DEFAULT_TREATMENT_LAG,
    horizon_steps=1,
    active_level=1.0,
    policy=crate::temporal_license::DEFAULT_POLICY,
    inference=None,
    n_draws=1000,
    prior_scale=10.0,
    prior_artifact=None,
    refute=None,
    validators=None,
    seed=1,
    bootstrap=0,
    threads=1
))]
fn analyze(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, u32, String, u32)>,
    treatment: String,
    outcome: String,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    policy: &str,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AnalysisResult> {
    let (tabular, _) = crate::tabular_from_py_columns(py, names.clone(), columns)?;
    let policy = policy.to_ascii_lowercase();
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
    detach_catch(py, move || {
        let series = series_from_tabular(tabular)?;

        let t_id = schema_var_id(series.schema(), &treatment)?;
        let y_id = schema_var_id(series.schema(), &outcome)?;

        let g = temporal_dag_from_schema_edges(series.schema(), &edges)?;

        let q = temporal_query_from_policy(
            &policy,
            t_id,
            y_id,
            treatment_lag,
            horizon_steps,
            active_level,
        )?;

        let mut builder = Study::series(series)
            .graph(g)
            .temporal_query(q)
            .refute(suite)
            .custom_validators(custom_validators)
            .bootstrap_replicates(bootstrap);
        builder = apply_temporal_inference(
            builder,
            inference.as_deref(),
            n_draws,
            prior_scale,
            prior_artifact.as_deref(),
        )?;
        let analysis = builder.build().map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        analysis_result_from_run(&names, result)
    })
}

pub(crate) enum TemporalClassGraph {
    Cpdag(antecedent_graph::TemporalCpdag),
    Pag(antecedent_graph::TemporalPag),
}

pub(crate) fn bind_temporal_class(
    builder: StudyBuilder,
    graph: TemporalClassGraph,
    accepted: bool,
) -> StudyBuilder {
    match (graph, accepted) {
        (TemporalClassGraph::Cpdag(g), true) => builder.graph(AcceptedGraph::from(g)),
        (TemporalClassGraph::Cpdag(g), false) => builder.graph(g),
        (TemporalClassGraph::Pag(g), true) => builder.graph(AcceptedGraph::from(g)),
        (TemporalClassGraph::Pag(g), false) => builder.graph(g),
    }
}

#[allow(clippy::too_many_arguments)]
fn analyze_temporal_class(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    graph: TemporalClassGraph,
    treatment: String,
    outcome: String,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    policy: &str,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    class_prior_ordered: Option<Vec<f64>>,
    class_prior_pairs: Option<Vec<(u64, f64)>>,
    max_completions: Option<usize>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    accepted: bool,
) -> PyResult<AnalysisResult> {
    let (tabular, _) = crate::tabular_from_py_columns(py, names.clone(), columns)?;
    let policy = policy.to_ascii_lowercase();
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
    detach_catch(py, move || {
        let series = series_from_tabular(tabular)?;
        let t_id = schema_var_id(series.schema(), &treatment)?;
        let y_id = schema_var_id(series.schema(), &outcome)?;
        let q = temporal_query_from_policy(
            &policy,
            t_id,
            y_id,
            treatment_lag,
            horizon_steps,
            active_level,
        )?;
        let mut builder = bind_temporal_class(Study::series(series), graph, accepted)
            .temporal_query(q)
            .refute(suite)
            .custom_validators(custom_validators)
            .bootstrap_replicates(bootstrap);
        builder = apply_temporal_inference(
            builder,
            inference.as_deref(),
            n_draws,
            prior_scale,
            prior_artifact.as_deref(),
        )?;
        builder = apply_class_prior(builder, class_prior_ordered, class_prior_pairs)?;
        builder = apply_max_completions(builder, max_completions);
        let analysis = builder.build().map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        analysis_result_from_run(&names, result)
    })
}

/// Temporal effect with a typed [`TemporalCpdag`]. Undirected marks stay on the
/// class; a fully oriented graph is the TemporalDag coordinate.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    graph,
    treatment,
    outcome,
    *,
    treatment_lag=crate::temporal_license::DEFAULT_TREATMENT_LAG,
    horizon_steps=1,
    active_level=1.0,
    policy=crate::temporal_license::DEFAULT_POLICY,
    inference=None,
    n_draws=1000,
    prior_scale=10.0,
    prior_artifact=None,
    class_prior_ordered=None,
    class_prior_pairs=None,
    max_completions=None,
    refute=None,
    validators=None,
    seed=1,
    bootstrap=0,
    threads=1,
    accepted=false,
))]
fn analyze_temporal_cpdag(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    graph: graphs::TemporalCpdag,
    treatment: String,
    outcome: String,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    policy: &str,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    class_prior_ordered: Option<Vec<f64>>,
    class_prior_pairs: Option<Vec<(u64, f64)>>,
    max_completions: Option<usize>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    accepted: bool,
) -> PyResult<AnalysisResult> {
    analyze_temporal_class(
        py,
        names,
        columns,
        TemporalClassGraph::Cpdag(graph.cpdag),
        treatment,
        outcome,
        treatment_lag,
        horizon_steps,
        active_level,
        policy,
        inference,
        n_draws,
        prior_scale,
        prior_artifact,
        class_prior_ordered,
        class_prior_pairs,
        max_completions,
        refute,
        validators,
        seed,
        bootstrap,
        threads,
        accepted,
    )
}

/// Temporal effect with a typed [`TemporalPag`]. Circle marks stay on the class.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    graph,
    treatment,
    outcome,
    *,
    treatment_lag=crate::temporal_license::DEFAULT_TREATMENT_LAG,
    horizon_steps=1,
    active_level=1.0,
    policy=crate::temporal_license::DEFAULT_POLICY,
    inference=None,
    n_draws=1000,
    prior_scale=10.0,
    prior_artifact=None,
    class_prior_ordered=None,
    class_prior_pairs=None,
    max_completions=None,
    refute=None,
    validators=None,
    seed=1,
    bootstrap=0,
    threads=1,
    accepted=false,
))]
fn analyze_temporal_pag(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    graph: graphs::TemporalPag,
    treatment: String,
    outcome: String,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    policy: &str,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    class_prior_ordered: Option<Vec<f64>>,
    class_prior_pairs: Option<Vec<(u64, f64)>>,
    max_completions: Option<usize>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    accepted: bool,
) -> PyResult<AnalysisResult> {
    analyze_temporal_class(
        py,
        names,
        columns,
        TemporalClassGraph::Pag(graph.pag),
        treatment,
        outcome,
        treatment_lag,
        horizon_steps,
        active_level,
        policy,
        inference,
        n_draws,
        prior_scale,
        prior_artifact,
        class_prior_ordered,
        class_prior_pairs,
        max_completions,
        refute,
        validators,
        seed,
        bootstrap,
        threads,
        accepted,
    )
}

/// Temporal effect on irregular event data (aligned via duration bins before estimation).
///
/// Optional `algorithm` enables PCMCI-family / DBN discovery after align-to-grid.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    event_times_ns,
    align_interval_ns,
    edges,
    treatment,
    outcome,
    *,
    treatment_lag=crate::temporal_license::DEFAULT_TREATMENT_LAG,
    horizon_steps=1,
    active_level=1.0,
    policy=crate::temporal_license::DEFAULT_POLICY,
    inference=None,
    n_draws=1000,
    prior_scale=10.0,
    prior_artifact=None,
    refute=None,
    validators=None,
    seed=1,
    bootstrap=0,
    threads=1,
    algorithm=None,
    max_lag=1,
    alpha=0.05,
    max_cond_size=2,
    fdr=true,
    accept_discovered=true,
    regimes=None,
    n_chains=2,
    n_warmup=100,
    mcmc_draws=200,
    force_mcmc=false,
    ci=None,
))]
fn analyze_events(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    event_times_ns: Vec<i64>,
    align_interval_ns: u64,
    edges: Vec<(String, u32, String, u32)>,
    treatment: String,
    outcome: String,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    policy: &str,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    algorithm: Option<String>,
    max_lag: u32,
    alpha: f64,
    max_cond_size: usize,
    fdr: bool,
    accept_discovered: bool,
    regimes: Option<Vec<u32>>,
    n_chains: u32,
    n_warmup: u32,
    mcmc_draws: u32,
    force_mcmc: bool,
    ci: Option<Bound<'_, PyAny>>,
) -> PyResult<AnalysisResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let (ci_impl, _ci_name, is_ci_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    let threads = if custom_validators.is_empty() && !is_ci_callback { threads } else { 1 };
    drop(columns);
    let policy = policy.to_string();
    let fdr_ctrl = if fdr { FdrControl::bh() } else { FdrControl::Off };
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let event = EventData::try_new(loaded.data.storage().clone(), Arc::from(event_times_ns))
            .map_err(py_err)?;
        let t_id = schema_var_id(event.schema(), &treatment)?;
        let y_id = schema_var_id(event.schema(), &outcome)?;
        let q = temporal_query_from_policy(
            &policy,
            t_id,
            y_id,
            treatment_lag,
            horizon_steps,
            active_level,
        )?;
        let ctx = py_execution_context(seed, threads);
        let mut builder = if let Some(algo) = algorithm.as_deref() {
            // Discovery needs the aligned series up front; `Study::events` re-runs the
            // same (pure, deterministic) alignment internally so the built `Study` still
            // reports `DataClassification::Event`.
            let aligned = event.align_to_grid(align_interval_ns).map_err(py_err)?;
            let vars: Vec<VariableId> = aligned.schema().variables().iter().map(|v| v.id).collect();
            let algo = algo.to_ascii_lowercase();
            let fdr = fdr_ctrl.adjustment();
            let builder = Study::events(&event, align_interval_ns).map_err(py_err)?;
            match algo.as_str() {
                "pcmci" => {
                    let params = DiscoverParams {
                        max_lag,
                        alpha,
                        fdr,
                        ci: ci_impl.clone(),
                        multi_dataset: MultiDatasetConstraints::default(),
                        max_cond_size,
                    };
                    let found =
                        facade_discover_pcmci(&aligned, &vars, &params, &ctx).map_err(py_err)?;
                    let accepted = accept_temporal_graph_review(found.review, accept_discovered)
                        .map_err(py_err)?;
                    builder.graph(accepted)
                }
                "pcmci_plus" => {
                    let params = DiscoverParams {
                        max_lag,
                        alpha,
                        fdr,
                        ci: ci_impl.clone(),
                        multi_dataset: MultiDatasetConstraints::default(),
                        max_cond_size,
                    };
                    let found = facade_discover_pcmci_plus(&aligned, &vars, &params, &ctx)
                        .map_err(py_err)?;
                    let accepted = accept_temporal_cpdag_review(found.review, accept_discovered)
                        .map_err(py_err)?;
                    builder.graph(accepted)
                }
                "lpcmci" => {
                    let params = DiscoverParams {
                        max_lag,
                        alpha,
                        fdr,
                        ci: ci_impl.clone(),
                        multi_dataset: MultiDatasetConstraints::default(),
                        max_cond_size,
                    };
                    let found =
                        facade_discover_lpcmci(&aligned, &vars, &params, &ctx).map_err(py_err)?;
                    let accepted = accept_temporal_pag_review(
                        found.evidence.graph.clone(),
                        found.review,
                        accept_discovered,
                    )
                    .map_err(py_err)?;
                    builder.graph(accepted)
                }
                "rpcmci" => {
                    let regimes = regimes.ok_or_else(|| {
                        PyValueError::new_err(
                            "analyze_events(algorithm='rpcmci') requires regimes=[…]",
                        )
                    })?;
                    let assign = RegimeAssignment::try_new(
                        regimes.into_iter().map(RegimeId::from_raw).collect::<Vec<_>>(),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    let params = DiscoverParams {
                        max_lag,
                        alpha,
                        fdr,
                        ci: ci_impl.clone(),
                        multi_dataset: MultiDatasetConstraints::default(),
                        max_cond_size,
                    };
                    let found =
                        facade_discover_rpcmci(&aligned, &vars, &assign, &params, None, &ctx)
                            .map_err(py_err)?;
                    let accepted =
                        accept_rpcmci_review(&found, accept_discovered).map_err(py_err)?;
                    builder.graph(accepted)
                }
                "dbn_posterior" => {
                    let params = antecedent::discovery::BayesianDiscoverParams::default();
                    let schedule = antecedent::discovery::GraphMcmcSchedule {
                        n_chains,
                        n_warmup,
                        n_draws: mcmc_draws,
                        thin: 1,
                    };
                    let gp = antecedent::discovery::discover_dbn_posterior(
                        &aligned, &vars, &params, max_lag, force_mcmc, &schedule, &ctx,
                    )
                    .map_err(py_err)?;
                    builder.graph_posterior(gp)
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "EventFrame discovery algorithm {other:?} unsupported"
                    )));
                }
            }
        } else {
            let g = temporal_dag_from_schema_edges(event.schema(), &edges)?;
            Study::events(&event, align_interval_ns).map_err(py_err)?.graph(g)
        };
        builder = builder
            .temporal_query(q)
            .refute(suite)
            .custom_validators(custom_validators)
            .bootstrap_replicates(bootstrap);
        builder = apply_temporal_inference(
            builder,
            inference.as_deref(),
            n_draws,
            prior_scale,
            prior_artifact.as_deref(),
        )?;
        let analysis = builder.build().map_err(py_err)?;
        let result = analysis.run(&ctx).map_err(py_err)?;
        analysis_result_from_run(&names, result)
    })
}

/// Temporal effect on multi-unit panel data (stacked estimate + PanelClusterHac SE).
#[pyfunction]
#[pyo3(signature = (
    names,
    unit_columns,
    unit_ids,
    edges,
    treatment,
    outcome,
    *,
    treatment_lag=crate::temporal_license::DEFAULT_TREATMENT_LAG,
    horizon_steps=1,
    active_level=1.0,
    policy=crate::temporal_license::DEFAULT_POLICY,
    inference=None,
    n_draws=1000,
    prior_scale=10.0,
    prior_artifact=None,
    refute=None,
    validators=None,
    seed=1,
    bootstrap=0,
    threads=1
))]
fn analyze_panel(
    py: Python<'_>,
    names: Vec<String>,
    unit_columns: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
    unit_ids: Vec<u32>,
    edges: Vec<(String, u32, String, u32)>,
    treatment: String,
    outcome: String,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    policy: &str,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AnalysisResult> {
    if unit_columns.is_empty() {
        return Err(PyValueError::new_err("panel needs ≥1 unit"));
    }
    if unit_columns.len() != unit_ids.len() {
        return Err(PyValueError::new_err("unit_columns and unit_ids length mismatch"));
    }
    let mut batches = Vec::with_capacity(unit_columns.len());
    for cols in &unit_columns {
        batches.push(columns_to_batch(&names, cols)?);
    }
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
    drop(unit_columns);
    let policy = policy.to_string();
    detach_catch(py, move || {
        let mut units = Vec::with_capacity(batches.len());
        for (batch, unit_id) in batches.iter().zip(unit_ids.into_iter()) {
            let (series, _) = series_from_batch(batch)?;
            units.push(PanelUnit { unit_id, series });
        }
        let panel = PanelData::try_new(Arc::from(units)).map_err(py_err)?;
        let t_id = schema_var_id(panel.schema(), &treatment)?;
        let y_id = schema_var_id(panel.schema(), &outcome)?;
        let g = temporal_dag_from_schema_edges(panel.schema(), &edges)?;
        let q = temporal_query_from_policy(
            &policy,
            t_id,
            y_id,
            treatment_lag,
            horizon_steps,
            active_level,
        )?;
        let mut builder = Study::panel(panel)
            .graph(g)
            .temporal_query(q)
            .refute(suite)
            .custom_validators(custom_validators)
            .bootstrap_replicates(bootstrap);
        builder = apply_temporal_inference(
            builder,
            inference.as_deref(),
            n_draws,
            prior_scale,
            prior_artifact.as_deref(),
        )?;
        let analysis = builder.build().map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        analysis_result_from_run(&names, result)
    })
}

/// Panel + J-PCMCI+ discover → stacked PanelClusterHac estimate.
#[pyfunction]
#[pyo3(signature = (
    names,
    unit_columns,
    unit_ids,
    treatment,
    outcome,
    *,
    algorithm="jpcmci_plus",
    max_lag=3,
    alpha=0.05,
    max_cond_size=2,
    fdr=true,
    accept_discovered=true,
    treatment_lag=crate::temporal_license::DEFAULT_TREATMENT_LAG,
    horizon_steps=1,
    active_level=1.0,
    policy=crate::temporal_license::DEFAULT_POLICY,
    inference=None,
    n_draws=1000,
    prior_scale=10.0,
    prior_artifact=None,
    refute=None,
    validators=None,
    seed=1,
    bootstrap=0,
    threads=1,
    context_names=None,
    include_space_dummy=true,
    include_time_dummy=false,
    space_dummy_ci=false,
    time_dummy_encoding="integer",
    time_dummy_ci=false,
    ci=None,
))]
fn analyze_panel_discover(
    py: Python<'_>,
    names: Vec<String>,
    unit_columns: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
    unit_ids: Vec<u32>,
    treatment: String,
    outcome: String,
    algorithm: &str,
    max_lag: u32,
    alpha: f64,
    max_cond_size: usize,
    fdr: bool,
    accept_discovered: bool,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    policy: &str,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    context_names: Option<Vec<String>>,
    include_space_dummy: bool,
    include_time_dummy: bool,
    space_dummy_ci: bool,
    time_dummy_encoding: &str,
    time_dummy_ci: bool,
    ci: Option<Bound<'_, PyAny>>,
) -> PyResult<AnalysisResult> {
    if unit_columns.is_empty() {
        return Err(PyValueError::new_err("panel needs ≥1 unit"));
    }
    if unit_columns.len() != unit_ids.len() {
        return Err(PyValueError::new_err("unit_columns and unit_ids length mismatch"));
    }
    let mut batches = Vec::with_capacity(unit_columns.len());
    for cols in &unit_columns {
        batches.push(columns_to_batch(&names, cols)?);
    }
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let (ci_impl, _ci_name, is_ci_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    let threads = if is_ci_callback || !custom_validators.is_empty() { 1 } else { threads };
    drop(unit_columns);
    let policy = policy.to_string();
    let time_dummy_encoding = time_dummy_encoding.to_string();
    let algorithm = algorithm.to_string();
    detach_catch(py, move || {
        let mut units = Vec::with_capacity(batches.len());
        for (batch, unit_id) in batches.iter().zip(unit_ids.into_iter()) {
            let (series, _) = series_from_batch(batch)?;
            units.push(PanelUnit { unit_id, series });
        }
        let panel = PanelData::try_new(Arc::from(units)).map_err(py_err)?;
        let t_id = schema_var_id(panel.schema(), &treatment)?;
        let y_id = schema_var_id(panel.schema(), &outcome)?;
        let q = temporal_query_from_policy(
            &policy,
            t_id,
            y_id,
            treatment_lag,
            horizon_steps,
            active_level,
        )?;
        let fdr_ctrl = if fdr { FdrControl::bh() } else { FdrControl::Off };
        let multi_dataset = panel_multi_dataset_constraints(
            &panel,
            context_names.unwrap_or_default(),
            include_space_dummy,
            include_time_dummy,
            space_dummy_ci,
            &time_dummy_encoding,
            time_dummy_ci,
        )?;
        let algo = algorithm.to_ascii_lowercase();
        let ctx = py_execution_context(seed, threads);
        let builder = panel_discovery_builder(
            panel,
            algo.as_str(),
            max_lag,
            alpha,
            max_cond_size,
            fdr_ctrl,
            accept_discovered,
            multi_dataset,
            ci_impl,
            &ctx,
        )?;
        let mut builder = builder
            .temporal_query(q)
            .refute(suite)
            .custom_validators(custom_validators)
            .bootstrap_replicates(bootstrap);
        builder = apply_temporal_inference(
            builder,
            inference.as_deref(),
            n_draws,
            prior_scale,
            prior_artifact.as_deref(),
        )?;
        let analysis = builder.build().map_err(py_err)?;
        let result = analysis.run(&ctx).map_err(py_err)?;
        analysis_result_from_run(&names, result)
    })
}

pub(crate) fn temporal_query_from_policy(
    policy: &str,
    t_id: VariableId,
    y_id: VariableId,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
) -> PyResult<TemporalEffectQuery> {
    let parsed = crate::temporal_license::policy_at_lag(policy, treatment_lag)?;
    let query = match &parsed {
        TemporalPolicy::Pulse { .. } => TemporalEffectQuery::pulse(t_id, y_id, active_level),
        TemporalPolicy::Sustained { .. } => {
            TemporalEffectQuery::sustained(t_id, y_id, 0, active_level)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown temporal policy {policy:?}; use pulse|sustained"
            )));
        }
    };
    Ok(query.with_policy(parsed).with_horizon_steps(horizon_steps))
}

/// Result of a GCM counterfactual ITE (single boundary crossing).
#[pyclass]
pub(crate) struct GcmIteResult {
    #[pyo3(get)]
    pub(crate) mean_ite: f64,
    #[pyo3(get)]
    pub(crate) n_units: usize,
    #[pyo3(get)]
    pub(crate) noise_inference: String,
    #[pyo3(get)]
    pub(crate) n_assignments: usize,
    /// Per-unit treatment effects (float64 NumPy array).
    #[pyo3(get)]
    pub(crate) unit_effects: Py<PyArray1<f64>>,
}

/// Interventional samples under hard `do` (means + full draws).
#[pyclass]
pub(crate) struct GcmSampleResult {
    #[pyo3(get)]
    pub(crate) column_means: Vec<f64>,
    #[pyo3(get)]
    pub(crate) n_draws: usize,
    #[pyo3(get)]
    pub(crate) n_nodes: usize,
    /// Column-major draws shaped `(n_nodes, n_draws)`.
    #[pyo3(get)]
    pub(crate) draws: Py<PyArray2<f64>>,
}

/// Fit a linear-Gaussian GCM and return mean ITE under hard interventions.
///
/// Crosses the Python boundary once: NumPy columns + edges in, arrays out.

fn temporal_multi_env_dummy_modes(
    space_dummy_ci: &str,
    time_dummy_encoding: &str,
    time_dummy_ci: &str,
) -> PyResult<(SpaceDummyCiMode, TimeDummyEncoding, TimeDummyCiMode)> {
    parse_dummy_ci_modes(space_dummy_ci, time_dummy_encoding, time_dummy_ci)
}

fn run_temporal_analysis(
    names: &[String],
    analysis: antecedent::Study,
    seed: u64,
    threads: u32,
) -> PyResult<AnalysisResult> {
    let ctx = py_execution_context(seed, threads);
    let result = analysis.run(&ctx).map_err(py_err)?;
    analysis_result_from_run(names, result)
}

fn temporal_discover_jpcmci_plus(
    names: &[String],
    batches: &[RecordBatch],
    treatment: &str,
    outcome: &str,
    context_names: &[String],
    include_space_dummy: bool,
    include_time_dummy: bool,
    space_dummy_ci: &str,
    time_dummy_encoding: &str,
    time_dummy_ci: &str,
    policy: &str,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    max_lag: u32,
    alpha: f64,
    max_cond_size: usize,
    fdr_ctrl: FdrControl,
    accept_discovered: bool,
    suite: RefuteSuite,
    custom_validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    bootstrap: u32,
    ci_impl: Arc<dyn antecedent_stats::ConditionalIndependence + Send + Sync>,
    seed: u64,
    threads: u32,
) -> PyResult<AnalysisResult> {
    let mut series_list = Vec::with_capacity(batches.len());
    for batch in batches {
        let (series, _) = series_from_batch(batch)?;
        series_list.push(series);
    }
    let multi = MultiEnvironmentData::try_new(Arc::from(series_list)).map_err(py_err)?;
    let t_id = multi.schema().id_of(treatment).map_err(py_err)?;
    let y_id = multi.schema().id_of(outcome).map_err(py_err)?;
    let mut context_ids = Vec::new();
    for cname in context_names {
        context_ids.push(multi.schema().id_of(cname).map_err(py_err)?);
    }
    let (space_mode, time_enc, time_mode) =
        temporal_multi_env_dummy_modes(space_dummy_ci, time_dummy_encoding, time_dummy_ci)?;
    let multi_dataset = MultiDatasetConstraints {
        context_variables: Arc::from(context_ids),
        include_space_dummy,
        include_time_dummy,
        space_dummy_ci: space_mode,
        time_dummy_encoding: time_enc,
        time_dummy_ci: time_mode,
        ..MultiDatasetConstraints::default()
    };
    let q =
        temporal_query_from_policy(policy, t_id, y_id, treatment_lag, horizon_steps, active_level)?;
    let vars: Vec<VariableId> = multi.schema().variables().iter().map(|v| v.id).collect();
    let ctx = py_execution_context(seed, threads);
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr: fdr_ctrl.adjustment(),
        ci: ci_impl,
        multi_dataset,
        max_cond_size,
    };
    let found = facade_discover_jpcmci_plus(&multi, &vars, &params, &ctx).map_err(py_err)?;
    let accepted = accept_temporal_cpdag_review(found.review, accept_discovered).map_err(py_err)?;
    let analysis = Study::series_multi(multi)
        .graph(accepted)
        .temporal_query(q)
        .refute(suite)
        .custom_validators(custom_validators)
        .bootstrap_replicates(bootstrap)
        .build()
        .map_err(py_err)?;
    run_temporal_analysis(names, analysis, seed, threads)
}

fn temporal_discover_rpcmci(
    names: &[String],
    batch: &RecordBatch,
    regimes: Vec<u32>,
    treatment: &str,
    outcome: &str,
    policy: &str,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    max_lag: u32,
    alpha: f64,
    max_cond_size: usize,
    fdr_ctrl: FdrControl,
    accept_discovered: bool,
    suite: RefuteSuite,
    custom_validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    bootstrap: u32,
    ci_impl: Arc<dyn antecedent_stats::ConditionalIndependence + Send + Sync>,
    seed: u64,
    threads: u32,
) -> PyResult<AnalysisResult> {
    let loaded = tabular_from_record_batch(batch).map_err(py_err)?;
    let tabular = loaded.data;
    let n = tabular.row_count();
    if regimes.len() != n {
        return Err(PyValueError::new_err(format!(
            "regimes length {} != series length {n}",
            regimes.len()
        )));
    }
    let series = series_from_tabular(tabular)?;
    let assign =
        RegimeAssignment::try_new(regimes.into_iter().map(RegimeId::from_raw).collect::<Vec<_>>())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let t_id = series.schema().id_of(treatment).map_err(py_err)?;
    let y_id = series.schema().id_of(outcome).map_err(py_err)?;
    let q =
        temporal_query_from_policy(policy, t_id, y_id, treatment_lag, horizon_steps, active_level)?;
    let vars: Vec<VariableId> = series.schema().variables().iter().map(|v| v.id).collect();
    let ctx = py_execution_context(seed, threads);
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr: fdr_ctrl.adjustment(),
        ci: ci_impl,
        multi_dataset: MultiDatasetConstraints::default(),
        max_cond_size,
    };
    let found =
        facade_discover_rpcmci(&series, &vars, &assign, &params, None, &ctx).map_err(py_err)?;
    let accepted = accept_rpcmci_review(&found, accept_discovered).map_err(py_err)?;
    let analysis = Study::series(series)
        .graph(accepted)
        .temporal_query(q)
        .refute(suite)
        .custom_validators(custom_validators)
        .bootstrap_replicates(bootstrap)
        .build()
        .map_err(py_err)?;
    run_temporal_analysis(names, analysis, seed, threads)
}

fn temporal_discover_pcmci_family(
    names: &[String],
    batch: &RecordBatch,
    algo: &str,
    treatment: &str,
    outcome: &str,
    policy: &str,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    max_lag: u32,
    alpha: f64,
    max_cond_size: usize,
    fdr_ctrl: FdrControl,
    accept_discovered: bool,
    suite: RefuteSuite,
    custom_validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    bootstrap: u32,
    ci_impl: Arc<dyn antecedent_stats::ConditionalIndependence + Send + Sync>,
    inference: Option<&str>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<&[u8]>,
    seed: u64,
    threads: u32,
) -> PyResult<AnalysisResult> {
    let loaded = tabular_from_record_batch(batch).map_err(py_err)?;
    let series = series_from_tabular(loaded.data)?;
    let t_id = series.schema().id_of(treatment).map_err(py_err)?;
    let y_id = series.schema().id_of(outcome).map_err(py_err)?;
    let q =
        temporal_query_from_policy(policy, t_id, y_id, treatment_lag, horizon_steps, active_level)?;
    let vars: Vec<VariableId> = series.schema().variables().iter().map(|v| v.id).collect();
    let ctx = py_execution_context(seed, threads);
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr: fdr_ctrl.adjustment(),
        ci: ci_impl,
        multi_dataset: MultiDatasetConstraints::default(),
        max_cond_size,
    };
    let accepted = match algo {
        "pcmci" => {
            let found = facade_discover_pcmci(&series, &vars, &params, &ctx).map_err(py_err)?;
            accept_temporal_graph_review(found.review, accept_discovered).map_err(py_err)?
        }
        "pcmci_plus" => {
            let found =
                facade_discover_pcmci_plus(&series, &vars, &params, &ctx).map_err(py_err)?;
            accept_temporal_cpdag_review(found.review, accept_discovered).map_err(py_err)?
        }
        "lpcmci" => {
            let found = facade_discover_lpcmci(&series, &vars, &params, &ctx).map_err(py_err)?;
            accept_temporal_pag_review(
                found.evidence.graph.clone(),
                found.review,
                accept_discovered,
            )
            .map_err(py_err)?
        }
        _ => unreachable!(),
    };
    let mut builder = Study::series(series)
        .graph(accepted)
        .temporal_query(q)
        .refute(suite)
        .custom_validators(custom_validators)
        .bootstrap_replicates(bootstrap);
    builder = apply_temporal_inference(builder, inference, n_draws, prior_scale, prior_artifact)?;
    let analysis = builder.build().map_err(py_err)?;
    run_temporal_analysis(names, analysis, seed, threads)
}

fn temporal_discover_dbn_posterior(
    names: &[String],
    batch: &RecordBatch,
    treatment: &str,
    outcome: &str,
    policy: &str,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    max_lag: u32,
    suite: RefuteSuite,
    custom_validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    bootstrap: u32,
    force_mcmc: bool,
    n_chains: u32,
    n_warmup: u32,
    mcmc_draws: u32,
    inference: Option<&str>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<&[u8]>,
    seed: u64,
    threads: u32,
) -> PyResult<AnalysisResult> {
    let loaded = tabular_from_record_batch(batch).map_err(py_err)?;
    let series = series_from_tabular(loaded.data)?;
    let t_id = series.schema().id_of(treatment).map_err(py_err)?;
    let y_id = series.schema().id_of(outcome).map_err(py_err)?;
    let q =
        temporal_query_from_policy(policy, t_id, y_id, treatment_lag, horizon_steps, active_level)?;
    let vars: Vec<VariableId> = series.schema().variables().iter().map(|v| v.id).collect();
    let ctx = py_execution_context(seed, threads);
    let params = antecedent::discovery::BayesianDiscoverParams::default();
    let schedule = antecedent::discovery::GraphMcmcSchedule {
        n_chains,
        n_warmup,
        n_draws: mcmc_draws,
        thin: 1,
    };
    let gp = antecedent::discovery::discover_dbn_posterior(
        &series, &vars, &params, max_lag, force_mcmc, &schedule, &ctx,
    )
    .map_err(py_err)?;
    let mut builder = Study::series(series)
        .graph_posterior(gp)
        .temporal_query(q)
        .refute(suite)
        .custom_validators(custom_validators)
        .bootstrap_replicates(bootstrap);
    builder = apply_temporal_inference(builder, inference, n_draws, prior_scale, prior_artifact)?;
    let analysis = builder.build().map_err(py_err)?;
    run_temporal_analysis(names, analysis, seed, threads)
}

// Discovery config carried across the temporal dispatch helpers. The bools are
// independent switches mirroring Python kwargs, not a state machine worth a type.
#[allow(clippy::struct_excessive_bools)]
struct TemporalDiscoverContext {
    names: Vec<String>,
    treatment: String,
    outcome: String,
    policy: String,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    max_lag: u32,
    alpha: f64,
    max_cond_size: usize,
    fdr_ctrl: FdrControl,
    accept_discovered: bool,
    suite: RefuteSuite,
    custom_validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    bootstrap: u32,
    ci_impl: Arc<dyn antecedent_stats::ConditionalIndependence + Send + Sync>,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    seed: u64,
    threads: u32,
    context_names: Vec<String>,
    include_space_dummy: bool,
    include_time_dummy: bool,
    space_dummy_ci: String,
    time_dummy_encoding: String,
    time_dummy_ci: String,
    force_mcmc: bool,
    n_chains: u32,
    n_warmup: u32,
    mcmc_draws: u32,
}

fn dispatch_temporal_discover(
    py: Python<'_>,
    algo: String,
    ctx: TemporalDiscoverContext,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    env_columns: Option<Vec<Vec<PyReadonlyArray1<'_, f64>>>>,
    regimes: Option<Vec<u32>>,
) -> PyResult<AnalysisResult> {
    match algo.as_str() {
        "jpcmci_plus" => dispatch_temporal_jpcmci_plus(py, ctx, env_columns, columns),
        "rpcmci" => dispatch_temporal_rpcmci(py, ctx, columns, regimes),
        "pcmci" | "pcmci_plus" | "lpcmci" => dispatch_temporal_pcmci_family(py, algo, ctx, columns),
        "dbn_posterior" => dispatch_temporal_dbn_posterior(py, ctx, columns),
        other => Err(PyValueError::new_err(format!(
            "unknown discovery algorithm {other:?}; use pcmci|pcmci_plus|lpcmci|jpcmci_plus|rpcmci|dbn_posterior"
        ))),
    }
}

fn dispatch_temporal_jpcmci_plus(
    py: Python<'_>,
    ctx: TemporalDiscoverContext,
    env_columns: Option<Vec<Vec<PyReadonlyArray1<'_, f64>>>>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
) -> PyResult<AnalysisResult> {
    let envs = env_columns.ok_or_else(|| {
        PyValueError::new_err(
            "analyze_temporal_discover(algorithm='jpcmci_plus') requires env_columns",
        )
    })?;
    if envs.is_empty() {
        return Err(PyValueError::new_err("jpcmci_plus needs ≥1 environment in env_columns"));
    }
    let mut batches = Vec::with_capacity(envs.len());
    for cols in &envs {
        batches.push(columns_to_batch(&ctx.names, cols)?);
    }
    drop(envs);
    drop(columns);
    detach_catch(py, move || {
        temporal_discover_jpcmci_plus(
            &ctx.names,
            &batches,
            &ctx.treatment,
            &ctx.outcome,
            &ctx.context_names,
            ctx.include_space_dummy,
            ctx.include_time_dummy,
            &ctx.space_dummy_ci,
            &ctx.time_dummy_encoding,
            &ctx.time_dummy_ci,
            &ctx.policy,
            ctx.treatment_lag,
            ctx.horizon_steps,
            ctx.active_level,
            ctx.max_lag,
            ctx.alpha,
            ctx.max_cond_size,
            ctx.fdr_ctrl,
            ctx.accept_discovered,
            ctx.suite,
            ctx.custom_validators,
            ctx.bootstrap,
            ctx.ci_impl,
            ctx.seed,
            ctx.threads,
        )
    })
}

fn dispatch_temporal_rpcmci(
    py: Python<'_>,
    ctx: TemporalDiscoverContext,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    regimes: Option<Vec<u32>>,
) -> PyResult<AnalysisResult> {
    let regimes = regimes.ok_or_else(|| {
        PyValueError::new_err("analyze_temporal_discover(algorithm='rpcmci') requires regimes=[…]")
    })?;
    let batch = columns_to_batch(&ctx.names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        temporal_discover_rpcmci(
            &ctx.names,
            &batch,
            regimes,
            &ctx.treatment,
            &ctx.outcome,
            &ctx.policy,
            ctx.treatment_lag,
            ctx.horizon_steps,
            ctx.active_level,
            ctx.max_lag,
            ctx.alpha,
            ctx.max_cond_size,
            ctx.fdr_ctrl,
            ctx.accept_discovered,
            ctx.suite,
            ctx.custom_validators,
            ctx.bootstrap,
            ctx.ci_impl,
            ctx.seed,
            ctx.threads,
        )
    })
}

fn dispatch_temporal_pcmci_family(
    py: Python<'_>,
    algo: String,
    ctx: TemporalDiscoverContext,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
) -> PyResult<AnalysisResult> {
    let batch = columns_to_batch(&ctx.names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        temporal_discover_pcmci_family(
            &ctx.names,
            &batch,
            algo.as_str(),
            &ctx.treatment,
            &ctx.outcome,
            &ctx.policy,
            ctx.treatment_lag,
            ctx.horizon_steps,
            ctx.active_level,
            ctx.max_lag,
            ctx.alpha,
            ctx.max_cond_size,
            ctx.fdr_ctrl,
            ctx.accept_discovered,
            ctx.suite,
            ctx.custom_validators,
            ctx.bootstrap,
            ctx.ci_impl,
            ctx.inference.as_deref(),
            ctx.n_draws,
            ctx.prior_scale,
            ctx.prior_artifact.as_deref(),
            ctx.seed,
            ctx.threads,
        )
    })
}

fn dispatch_temporal_dbn_posterior(
    py: Python<'_>,
    ctx: TemporalDiscoverContext,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
) -> PyResult<AnalysisResult> {
    let batch = columns_to_batch(&ctx.names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        temporal_discover_dbn_posterior(
            &ctx.names,
            &batch,
            &ctx.treatment,
            &ctx.outcome,
            &ctx.policy,
            ctx.treatment_lag,
            ctx.horizon_steps,
            ctx.active_level,
            ctx.max_lag,
            ctx.suite,
            ctx.custom_validators,
            ctx.bootstrap,
            ctx.force_mcmc,
            ctx.n_chains,
            ctx.n_warmup,
            ctx.mcmc_draws,
            ctx.inference.as_deref(),
            ctx.n_draws,
            ctx.prior_scale,
            ctx.prior_artifact.as_deref(),
            ctx.seed,
            ctx.threads,
        )
    })
}

/// Temporal effect analysis with PCMCI-family discovery (auto-accept when possible).
///
/// `algorithm` is one of `pcmci`, `pcmci_plus`, `lpcmci`. When discovery requires
/// human review and `accept_discovered` is false (or auto-accept is impossible),
/// raises [`CausalReviewError`].
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    treatment,
    outcome,
    *,
    algorithm="pcmci",
    max_lag=1,
    alpha=0.05,
    max_cond_size=2,
    fdr=true,
    accept_discovered=true,
    treatment_lag=crate::temporal_license::DEFAULT_TREATMENT_LAG,
    horizon_steps=1,
    active_level=1.0,
    policy=crate::temporal_license::DEFAULT_POLICY,
    inference=None,
    n_draws=1000,
    prior_scale=10.0,
    prior_artifact=None,
    refute=None,
    validators=None,
    seed=1,
    bootstrap=0,
    threads=1,
    env_columns=None,
    regimes=None,
    context_names=None,
    include_space_dummy=true,
    include_time_dummy=false,
    space_dummy_ci="scalar",
    time_dummy_encoding="integer",
    time_dummy_ci="scalar",
    ci=None,
    n_chains=2,
    n_warmup=100,
    mcmc_draws=200,
    force_mcmc=false,
))]
fn analyze_temporal_discover(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    treatment: String,
    outcome: String,
    algorithm: &str,
    max_lag: u32,
    alpha: f64,
    max_cond_size: usize,
    fdr: bool,
    accept_discovered: bool,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
    policy: &str,
    inference: Option<String>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<Vec<u8>>,
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    env_columns: Option<Vec<Vec<PyReadonlyArray1<'_, f64>>>>,
    regimes: Option<Vec<u32>>,
    context_names: Option<Vec<String>>,
    include_space_dummy: bool,
    include_time_dummy: bool,
    space_dummy_ci: &str,
    time_dummy_encoding: &str,
    time_dummy_ci: &str,
    ci: Option<Bound<'_, PyAny>>,
    n_chains: u32,
    n_warmup: u32,
    mcmc_draws: u32,
    force_mcmc: bool,
) -> PyResult<AnalysisResult> {
    let algo = algorithm.to_string();
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let policy = policy.to_ascii_lowercase();
    let fdr_ctrl = if fdr { FdrControl::bh() } else { FdrControl::Off };
    let context_names = context_names.unwrap_or_default();
    let space_dummy_ci = space_dummy_ci.to_string();
    let time_dummy_encoding = time_dummy_encoding.to_string();
    let time_dummy_ci = time_dummy_ci.to_string();
    let (ci_impl, _ci_name, is_ci_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    let threads = if is_ci_callback { 1 } else { threads };
    dispatch_temporal_discover(
        py,
        algo,
        TemporalDiscoverContext {
            names,
            treatment,
            outcome,
            policy,
            treatment_lag,
            horizon_steps,
            active_level,
            max_lag,
            alpha,
            max_cond_size,
            fdr_ctrl,
            accept_discovered,
            suite,
            custom_validators,
            bootstrap,
            ci_impl,
            inference,
            n_draws,
            prior_scale,
            prior_artifact,
            seed,
            threads,
            context_names,
            include_space_dummy,
            include_time_dummy,
            space_dummy_ci,
            time_dummy_encoding,
            time_dummy_ci,
            force_mcmc,
            n_chains,
            n_warmup,
            mcmc_draws,
        },
        columns,
        env_columns,
        regimes,
    )
}

fn analysis_result_from_run(
    names: &[String],
    result: antecedent::StudyResult,
) -> PyResult<AnalysisResult> {
    let certificate_json = crate::identification_details::analysis_to_json(&result, names)?;
    // The compiled plan names the executed estimator; Bayesian temporal fits
    // record `bayesian.temporal.gcomp` there, so no Python-side relabel is needed.
    let estimator_id = result
        .logical_plan
        .estimator
        .as_deref()
        .unwrap_or(antecedent::EstimatorId::TemporalLinearAdjustment.as_str())
        .to_string();
    let SharedStudySections {
        identification,
        estimate,
        posterior: posterior_section,
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
        ..
    } = crate::shared_study_sections(names, &result, estimator_id)?;
    let mut mediation_horizons = Vec::new();
    let mut mediation_effects = Vec::new();
    let mut mediation_totals = Vec::new();
    let mut mediation_directs = Vec::new();
    let mut mediation_mediated_effects = Vec::new();
    let mut mediation_identification_statuses = Vec::new();
    let mut mediation_methods = Vec::new();
    let mut mediation_adjustments = Vec::new();
    let mut mediation_uncertainty_kinds = Vec::new();
    let mut mediation_standard_deviations = Vec::new();
    let mut mediation_q025 = Vec::new();
    let mut mediation_q975 = Vec::new();
    let mut mediation_identified_lower = Vec::new();
    let mut mediation_identified_upper = Vec::new();
    if let Some(grid) = result.mediation_grid.as_ref() {
        for slice in grid.slices.iter() {
            mediation_horizons.push(slice.horizon);
            mediation_effects.push(slice.estimate.effect.ate);
            mediation_totals.push(slice.estimate.total.unwrap_or(f64::NAN));
            mediation_directs.push(slice.estimate.direct.unwrap_or(f64::NAN));
            mediation_mediated_effects.push(slice.estimate.mediated.unwrap_or(f64::NAN));
            mediation_identification_statuses.push(format!("{:?}", slice.identification_status));
            mediation_methods.push(slice.method.to_string());
            mediation_adjustments.push(
                slice.adjustment.iter().map(|key| (key.variable.raw(), key.offset)).collect(),
            );
            mediation_identified_lower.push(slice.identified_set.map(|set| set.lower));
            mediation_identified_upper.push(slice.identified_set.map(|set| set.upper));
            match &slice.uncertainty {
                antecedent_estimate::TemporalMediationUncertainty::FrequentistPointwise {
                    standard_error,
                }
                | antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
                    requested: standard_error,
                    ..
                } => {
                    mediation_uncertainty_kinds.push("frequentist_pointwise".to_string());
                    mediation_standard_deviations.push(*standard_error);
                    mediation_q025.push(None);
                    mediation_q975.push(None);
                }
                antecedent_estimate::TemporalMediationUncertainty::BayesianPointwise {
                    requested,
                    ..
                } => {
                    mediation_uncertainty_kinds.push("bayesian_pointwise".to_string());
                    mediation_standard_deviations.push(Some(requested.standard_deviation));
                    mediation_q025.push(Some(requested.q025));
                    mediation_q975.push(Some(requested.q975));
                }
                _ => {
                    mediation_uncertainty_kinds.push("unavailable".to_string());
                    mediation_standard_deviations.push(None);
                    mediation_q025.push(None);
                    mediation_q975.push(None);
                }
            }
        }
    }

    Ok(AnalysisResult {
        certificate_json,
        ate: result.estimate.ate.is_finite().then_some(result.estimate.ate),
        se_analytic: result.estimate.se_analytic,
        se_bootstrap: result.estimate.se_bootstrap,
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
        identification_status,
        method,
        diagnostics: result
            .diagnostics
            .iter()
            .map(|d| format!("{}: {}", d.code, d.message))
            .collect(),
        provenance_node_count: result.provenance.len(),
        refutation_count: validation.count,
        refutations,
        worker_threads: result.physical_plan.worker_threads,
        expected_python_crossings: result.physical_plan.expected_python_crossings,
        adjustment_set,
        assumption_count: result.estimate.assumptions.len(),
        derivation_step_count: result.identification.derivation.steps.len(),
        estimator_id,
        posterior_effect_mean: posterior_section.effect_mean,
        posterior_effect_sd: posterior_section.effect_sd,
        posterior_q025: posterior_section.q025,
        posterior_q975: posterior_section.q975,
        posterior_n_draws: posterior_section.n_draws,
        posterior_p_below_zero: posterior_section.p_below_zero,
        posterior_backend: posterior_section.backend.clone(),
        posterior_artifact: posterior_section.artifact.clone(),
        posterior_unidentified_mass: posterior_section.unidentified_mass,
        mediation_total: result.mediation.as_ref().and_then(|m| m.total),
        mediation_direct: result.mediation.as_ref().and_then(|m| m.direct),
        mediation_mediated: result.mediation.as_ref().and_then(|m| m.mediated),
        mediation_horizons,
        mediation_effects,
        mediation_totals,
        mediation_directs,
        mediation_mediated_effects,
        mediation_identification_statuses,
        mediation_methods,
        mediation_adjustments,
        mediation_uncertainty_kinds,
        mediation_standard_deviations,
        mediation_q025,
        mediation_q975,
        mediation_identified_lower,
        mediation_identified_upper,
        mediation_joint_posterior: result.mediation_grid.as_ref().map(|grid| grid.joint_posterior),
        identification,
        estimate,
        posterior: posterior_section,
        validation,
        performance,
        evidence_status,
        allowlist_reason,
        allowlist_parent,
        structural_weight_basis: result
            .structural_response
            .as_ref()
            .map(|mixture| mixture.weight_basis.as_str().to_string()),
        structural_identified_mass: result
            .structural_response
            .as_ref()
            .map(|mixture| mixture.identified_mass),
        structural_unidentified_mass: result
            .structural_response
            .as_ref()
            .map(|mixture| mixture.unidentified_mass),
        structural_unevaluable_mass: result
            .structural_response
            .as_ref()
            .map(|mixture| mixture.unevaluable_mass),
        structural_identified_set: identified_set.set,
        structural_identified_set_interval: identified_set.interval,
        structural_identified_set_interval_level: identified_set.level,
        structural_identified_set_interval_method: identified_set.method,
        structural_identified_set_interval_truncated: identified_set.truncated,
    })
}

pub(crate) fn apply_max_completions(
    builder: StudyBuilder,
    max_completions: Option<usize>,
) -> StudyBuilder {
    match max_completions {
        Some(n) => builder.max_completions(n),
        None => builder,
    }
}

pub(crate) fn apply_class_prior(
    builder: StudyBuilder,
    ordered: Option<Vec<f64>>,
    pairs: Option<Vec<(u64, f64)>>,
) -> PyResult<StudyBuilder> {
    match (ordered, pairs) {
        (None, None) => Ok(builder),
        (Some(masses), None) => {
            Ok(builder.class_prior(antecedent::ClassPrior::from_ordered(masses).map_err(py_err)?))
        }
        (None, Some(pairs)) => {
            Ok(builder.class_prior(antecedent::ClassPrior::from_pairs(pairs).map_err(py_err)?))
        }
        (Some(_), Some(_)) => Err(PyValueError::new_err(
            "class prior accepts ordered masses or key/mass pairs, not both",
        )),
    }
}

pub(crate) fn apply_temporal_inference(
    builder: antecedent::StudyBuilder,
    inference: Option<&str>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<&[u8]>,
) -> PyResult<antecedent::StudyBuilder> {
    apply_temporal_inference_transfer(
        builder,
        inference,
        n_draws,
        prior_scale,
        prior_artifact,
        None,
        None,
    )
}

pub(crate) fn apply_temporal_inference_transfer(
    builder: antecedent::StudyBuilder,
    inference: Option<&str>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<&[u8]>,
    prior_mapping: Option<antecedent_io::PriorMapping>,
    composed_prior: Option<crate::prior_bank::OwnedComposedPrior>,
) -> PyResult<antecedent::StudyBuilder> {
    let Some(mode) = inference else {
        return Ok(builder);
    };
    let mut cfg = match mode.to_ascii_lowercase().as_str() {
        "bayesian" | "bayesian.laplace" | "laplace" => {
            BayesianConfig::laplace().n_draws(n_draws).prior_scale(prior_scale)
        }
        "bayesian.conjugate" | "conjugate" => {
            BayesianConfig::conjugate().n_draws(n_draws).prior_scale(prior_scale)
        }
        "bayesian.hmc" | "hmc" => BayesianConfig::hmc().n_draws(n_draws).prior_scale(prior_scale),
        "frequentist" => {
            return Ok(builder.inference(InferenceMode::Frequentist));
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
    Ok(builder.inference(InferenceMode::Bayesian(cfg)))
}

/// Anomaly scores for listed outcomes.

#[pyfunction]
#[pyo3(signature = (
    names, columns, edges, treatment, mediator, outcome, *,
    contrast="mediated", control_level=0.0, active_level=1.0,
    horizons=None,
    seed=1, bootstrap=0, threads=1
))]
fn analyze_temporal_mediation(
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
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AnalysisResult> {
    let (tabular, _) = crate::tabular_from_py_columns(py, names.clone(), columns)?;
    let contrast = contrast.to_string();
    detach_catch(py, move || {
        let series = series_from_tabular(tabular)?;
        let t_id = schema_var_id(series.schema(), &treatment)?;
        let m_id = schema_var_id(series.schema(), &mediator)?;
        let y_id = schema_var_id(series.schema(), &outcome)?;
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
        let g = temporal_dag_from_schema_edges(series.schema(), &edges)?;
        let analysis = Study::series(series)
            .graph(g)
            .query(antecedent_core::CausalQuery::Mediation(q))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(bootstrap)
            .build()
            .map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        analysis_result_from_run(&names, result)
    })
}

/// Pulse/Sustained effect from a supplied DBN graph posterior.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    posterior,
    treatment,
    outcome,
    *,
    policy="pulse",
    window=None,
    treatment_lag=1,
    horizon_steps=1,
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
fn analyze_temporal_graph_posterior(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    posterior: Bound<'_, crate::bayesian::PyGraphPosterior>,
    treatment: String,
    outcome: String,
    policy: &str,
    window: Option<(i32, i32)>,
    treatment_lag: u32,
    horizon_steps: u32,
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
) -> PyResult<AnalysisResult> {
    let gp = {
        let posterior = posterior.borrow();
        posterior.require_bound_to(&names)?;
        posterior.to_rust()?
    };
    let suite = suite_from_refute(refute.as_ref())?;
    let cancel_token = cancel.map(|token| token.inner);
    let progress = callbacks::progress_sink_from_py(on_progress.as_ref())?;
    let (tabular, _) = crate::tabular_from_py_columns(py, names.clone(), columns)?;
    let policy = policy.to_ascii_lowercase();
    let inference = inference.to_string();
    detach_catch(py, move || {
        let series = series_from_tabular(tabular)?;
        let t_id = series.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = series.schema().id_of(&outcome).map_err(py_err)?;
        let mut q = temporal_query_from_policy(
            &policy,
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
        let mut builder = Study::series(series)
            .graph_posterior(gp)
            .temporal_query(q)
            .refute(suite)
            .bootstrap_replicates(bootstrap);
        builder = apply_temporal_inference(builder, Some(&inference), n_draws, prior_scale, None)?;
        let analysis = builder.build().map_err(py_err)?;
        let ctx = py_execution_context_ext(
            seed,
            threads,
            cancel_token,
            progress,
            Some(PY_DEFAULT_CACHE_MAX_BYTES),
        );
        let result = analysis.run(&ctx).map_err(py_err)?;
        analysis_result_from_run(&names, result)
    })
}

/// Temporal mediation mixture from a supplied DBN graph posterior.
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    posterior,
    treatment,
    mediator,
    outcome,
    *,
    contrast="mediated",
    control_level=0.0,
    active_level=1.0,
    horizons=None,
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
fn analyze_temporal_graph_posterior_mediation(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    posterior: Bound<'_, crate::bayesian::PyGraphPosterior>,
    treatment: String,
    mediator: String,
    outcome: String,
    contrast: &str,
    control_level: f64,
    active_level: f64,
    horizons: Option<Vec<u32>>,
    inference: &str,
    n_draws: usize,
    prior_scale: f64,
    refute: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    cancel: Option<PyCancellationToken>,
    on_progress: Option<Bound<'_, PyAny>>,
) -> PyResult<AnalysisResult> {
    let gp = {
        let posterior = posterior.borrow();
        posterior.require_bound_to(&names)?;
        posterior.to_rust()?
    };
    let suite = suite_from_refute(refute.as_ref())?;
    let cancel_token = cancel.map(|token| token.inner);
    let progress = callbacks::progress_sink_from_py(on_progress.as_ref())?;
    let (tabular, _) = crate::tabular_from_py_columns(py, names.clone(), columns)?;
    let contrast = contrast.to_string();
    let inference = inference.to_string();
    detach_catch(py, move || {
        let series = series_from_tabular(tabular)?;
        let t_id = series.schema().id_of(&treatment).map_err(py_err)?;
        let m_id = series.schema().id_of(&mediator).map_err(py_err)?;
        let y_id = series.schema().id_of(&outcome).map_err(py_err)?;
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
        let mut builder = Study::series(series)
            .graph_posterior(gp)
            .query(antecedent_core::CausalQuery::Mediation(q))
            .refute(suite)
            .bootstrap_replicates(bootstrap);
        builder = apply_temporal_inference(builder, Some(&inference), n_draws, prior_scale, None)?;
        let analysis = builder.build().map_err(py_err)?;
        let ctx = py_execution_context_ext(
            seed,
            threads,
            cancel_token,
            progress,
            Some(PY_DEFAULT_CACHE_MAX_BYTES),
        );
        let result = analysis.run(&ctx).map_err(py_err)?;
        analysis_result_from_run(&names, result)
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_temporal_cpdag, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_temporal_pag, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_events, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_panel, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_panel_discover, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_temporal_discover, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_temporal_graph_posterior, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_temporal_graph_posterior_mediation, m)?)?;
    m.add_function(wrap_pyfunction!(mediation_effects_summary, m)?)?;
    m.add_function(wrap_pyfunction!(predict_intervened_summary, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_temporal_mediation, m)?)?;
    Ok(())
}
