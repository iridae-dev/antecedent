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
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (names, columns, treatment, mediator, outcome, *, seed=1, threads=None))]
fn mediation_effects_summary(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    treatment: String,
    mediator: String,
    outcome: String,
    seed: u64,
    threads: Option<u32>,
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
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
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

/// Mean conditional prediction of `target` with the lagged `parent` held at `level`.
///
/// Fits `target_t ~ 1 + parent_{t-lag}` by OLS and averages the fitted line at
/// `parent = level`. No graph is involved and nothing is identified: this is the
/// association `E[target_t | parent_{t-lag} = level]` under a linear fit, which equals
/// an interventional mean only when the lagged parent is unconfounded with the target
/// and the relation is linear. For an identified interventional quantity with its
/// assumptions, support and uncertainty, run a temporal analysis with a graph instead.
#[pyfunction]
#[pyo3(signature = (names, columns, target, parent, *, parent_lag=1, level=1.0))]
fn predict_conditional_summary(
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
        let yhat = pred.predict_conditional(&series, x, level, &policy).map_err(py_err)?;
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

/// Mean of a conditional (associational) linear prediction over the lag-aligned rows.
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
    pub(crate) posterior_subsampled_out_mass: Option<f64>,
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

/// A supplied temporal structure that is not a `TemporalDag`: an equivalence class, or a
/// temporal ADMG whose pulse, sustained and conditional contrasts are identified on its unrolled
/// window.
pub(crate) enum TemporalClassGraph {
    Cpdag(antecedent_graph::TemporalCpdag),
    Pag(antecedent_graph::TemporalPag),
    Admg(antecedent_graph::TemporalAdmg),
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
            return Err(crate::value_err(format!(
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
    /// The compiled simulation route the effects came from.
    pub(crate) route: Option<crate::compiled_route_api::RouteRecord>,
}

#[pymethods]
impl GcmIteResult {
    /// The compiled unit-effect simulation route this result came from.
    #[getter]
    fn route(&self, py: Python<'_>) -> PyResult<crate::compiled_route_api::RouteHandle> {
        crate::compiled_route_api::route_object(py, self.route.as_ref())
    }

    /// The contract the unit-effect simulation executed under (`query_kind="Simulate"`).
    #[getter]
    fn contract(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.contract()
    }

    /// The claim sealed over the unit-effect simulation.
    #[getter]
    fn claim(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.claim()
    }

    /// Cheap inspection record of the compiled unit-effect simulation study.
    fn inspect(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.inspect()
    }
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
    /// The compiled simulation route the draws came from.
    pub(crate) route: Option<crate::compiled_route_api::RouteRecord>,
}

#[pymethods]
impl GcmSampleResult {
    /// The compiled interventional sample route this result came from.
    #[getter]
    fn route(&self, py: Python<'_>) -> PyResult<crate::compiled_route_api::RouteHandle> {
        crate::compiled_route_api::route_object(py, self.route.as_ref())
    }

    /// The contract the interventional sample executed under (`query_kind="Simulate"`).
    #[getter]
    fn contract(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.contract()
    }

    /// The claim sealed over the interventional sample.
    #[getter]
    fn claim(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.claim()
    }

    /// Cheap inspection record of the compiled interventional sample study.
    fn inspect(&self) -> PyResult<std::collections::HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.inspect()
    }
}

/// Fit a linear-Gaussian GCM and return mean ITE under hard interventions.
///
/// Crosses the Python boundary once: NumPy columns + edges in, arrays out.

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
        (Some(_), Some(_)) => {
            Err(crate::value_err("class prior accepts ordered masses or key/mass pairs, not both"))
        }
    }
}

pub(crate) fn apply_temporal_inference_transfer(
    builder: antecedent::StudyBuilder,
    inference: Option<&str>,
    n_draws: Option<usize>,
    prior_scale: f64,
    prior_artifact: Option<&[u8]>,
    prior_mapping: Option<antecedent_io::PriorMapping>,
    composed_prior: Option<crate::prior_bank::OwnedComposedPrior>,
    likelihood: antecedent_prob::BayesLikelihood,
) -> PyResult<antecedent::StudyBuilder> {
    let Some(mode) = inference else {
        return Ok(builder);
    };
    let apply_draws = |cfg: BayesianConfig| {
        if let Some(n) = n_draws { cfg.n_draws(n) } else { cfg }
    };
    let mut cfg = match mode.to_ascii_lowercase().as_str() {
        "bayesian" | "bayesian.laplace" | "laplace" => {
            apply_draws(BayesianConfig::laplace().prior_scale(prior_scale))
        }
        "bayesian.conjugate" | "conjugate" => {
            apply_draws(BayesianConfig::conjugate().prior_scale(prior_scale))
        }
        "bayesian.hmc" | "hmc" => apply_draws(BayesianConfig::hmc().prior_scale(prior_scale)),
        "frequentist" => {
            return Ok(builder.inference(InferenceMode::Frequentist));
        }
        other => {
            return Err(crate::value_err(format!(
                "unknown inference mode {other:?}; use frequentist|bayesian|conjugate|hmc"
            )));
        }
    };
    cfg = cfg.likelihood(likelihood);
    if let Some(comp) = composed_prior {
        cfg = crate::prior_bank::apply_owned_composed_prior(cfg, comp)?;
    } else if let Some(bytes) = prior_artifact {
        cfg = cfg.prior_from_artifact(bytes.to_vec(), prior_mapping);
    }
    let builder = builder.inference(InferenceMode::Bayesian(cfg));
    if let Some(n) = n_draws {
        Ok(builder.compute_budget(antecedent::ComputeBudget {
            n_draws: Some(n),
            ..antecedent::ComputeBudget::default()
        }))
    } else {
        Ok(builder)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mediation_effects_summary, m)?)?;
    m.add_function(wrap_pyfunction!(predict_conditional_summary, m)?)?;
    Ok(())
}
