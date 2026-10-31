//! Fitted GCM OO surface and typed attribution / design result pyclasses.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use antecedent::gcm::{
    ChangeAttributionResult as RustChangeAttributionResult, FittedGcm, RobustChangeOptions,
    fit_gcm, fit_gcm_with, rank_root_causes as facade_rank_root_causes,
};
use antecedent::{AcceptedGraph, AttributionOutcome, MechanismProgram, SimulationOutcome};
use antecedent_attribution::{
    AnomalyScores as RustAnomalyScores, CacheStats, ComponentContribution, ComputeBudget,
};
use antecedent_core::{
    AllocationMethod, AnomalyAttributionQuery, AttributeTarget, AttributionComponents,
    ChangeAttributionQuery, ChangeEstimator, ChangeMeasure, ComponentId, ExecutionContext,
    Intervention, MechanismChangeQuery, MechanismChangeTest, PopulationSelector, ShapleyConfig,
    SimulationTarget, UnitChangeQuery, Value, ValueExpr, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::Dag;
use antecedent_model::{
    MechanismAssignment, MechanismFamily, MechanismRegistry, NodeFamilyConstraint,
    SelectionOptions, ValidationSplits, ValueBatch,
};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::compiled_route_api::{ModelSource, RouteRecord, outcome_mismatch};
use crate::intervention_callback_api::{PyBatchCallbackHandle, program_intervention};
use crate::{
    GcmIteResult, GcmSampleResult, dag_from_named_edges, detach_catch, py_err,
    py_execution_context, py_msg,
};

/// `(mode, variable, callback, inputs)` of one callback program; see `FittedGcm.sample_do`.
type CallbackProgramArgs = (String, String, Py<PyBatchCallbackHandle>, Vec<Option<String>>);

/// The interventions of one call: hard `do` values, additive shifts and callback programs, all
/// named by variable. A variable both pinned and shifted is rejected.
fn interventions_of(
    data: &TabularData,
    interventions: &HashMap<String, f64>,
    shifts: Option<&HashMap<String, f64>>,
    programs: Option<&[CallbackProgramArgs]>,
) -> PyResult<Vec<Intervention>> {
    if let Some(shifts) = shifts {
        if let Some(name) = interventions.keys().find(|k| shifts.contains_key(*k)) {
            return Err(py_msg(format!(
                "variable {name:?} is present in both `interventions` and `shifts`; \
                 a variable cannot be simultaneously pinned (do(X := v)) and shifted \
                 (do(X := X + delta))"
            )));
        }
    }
    let mut ints = Vec::with_capacity(interventions.len() + shifts.map_or(0, HashMap::len));
    for (name, value) in interventions {
        let id = data.schema().id_of(name).map_err(py_err)?;
        ints.push(Intervention::set(id, Value::f64(*value)));
    }
    for (name, delta) in shifts.into_iter().flatten() {
        let id = data.schema().id_of(name).map_err(py_err)?;
        ints.push(Intervention::shift(id, Value::f64(*delta)));
    }
    for (mode, variable, callback, inputs) in programs.into_iter().flatten() {
        let id = data.schema().id_of(variable).map_err(py_err)?;
        let args = inputs
            .iter()
            .map(|input| match input {
                Some(name) => data.schema().id_of(name).map(ValueExpr::var).map_err(py_err),
                None => Ok(ValueExpr::generated()),
            })
            .collect::<PyResult<Vec<_>>>()?;
        ints.push(program_intervention(mode, id, callback.get(), args)?);
    }
    Ok(ints)
}

fn var_name(names: &[String], id: VariableId) -> String {
    names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("V{}", id.raw()))
}

fn component_name(names: &[String], id: ComponentId) -> String {
    names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("V{}", id.raw()))
}

fn dag_from_edges(data: &TabularData, edges: &[(String, String)]) -> PyResult<Dag> {
    dag_from_named_edges(data.schema(), edges)
}

fn path_breakdown_names(
    names: &[String],
    result: &RustChangeAttributionResult,
) -> Vec<(Vec<String>, f64)> {
    result
        .path_breakdown
        .iter()
        .map(|p| {
            let path: Vec<String> = p.path.iter().map(|id| var_name(names, *id)).collect();
            (path, p.contribution)
        })
        .collect()
}

fn contribution_pairs(
    names: &[String],
    result: &RustChangeAttributionResult,
) -> Vec<(String, f64)> {
    result
        .contributions
        .iter()
        .map(|c| (component_name(names, c.component), c.contribution))
        .collect()
}

pub(crate) fn anomaly_scores_from_rust(
    scores: impl IntoIterator<Item = RustAnomalyScores>,
    names: &[String],
) -> Vec<AnomalyScores> {
    scores
        .into_iter()
        .map(|s| {
            let mean = if s.scores.is_empty() {
                0.0
            } else {
                s.scores.iter().sum::<f64>() / s.scores.len() as f64
            };
            let top = s
                .scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_row = top.map(|(i, _)| s.rows[i]);
            AnomalyScores {
                outcome: var_name(names, s.target),
                mean_score: mean,
                n_units: s.rows.len(),
                scores: s.scores.to_vec(),
                rows: s.rows.to_vec(),
                top_row,
            }
        })
        .collect()
}

pub(crate) fn change_result_from_rust(
    result: RustChangeAttributionResult,
    names: &[String],
) -> ChangeAttributionResult {
    ChangeAttributionResult {
        total_change: result.total_change,
        contributions: contribution_pairs(names, &result),
        path_breakdown: path_breakdown_names(names, &result),
        rust: result,
        route: None,
        from_study: false,
    }
}

pub(crate) fn synthetic_change_result(
    outcome: VariableId,
    total_change: f64,
    pairs: Vec<(ComponentId, f64)>,
    names: &[String],
) -> ChangeAttributionResult {
    let contributions_rust: Vec<ComponentContribution> = pairs
        .iter()
        .map(|(c, v)| ComponentContribution {
            component: *c,
            contribution: *v,
            stderr: None,
            ci_low: None,
            ci_high: None,
        })
        .collect();
    let rust = RustChangeAttributionResult {
        outcome,
        total_change,
        observed_change: None,
        contributions: Arc::from(contributions_rust),
        interactions: Arc::from([]),
        path_breakdown: Arc::from([]),
        unidentified: Arc::from([]),
        graph_sensitivity: None,
        budget: ComputeBudget::default(),
        monte_carlo_stderr: None,
        component_mc_stderr: None,
        cache_stats: CacheStats::default(),
        fit_uncertainty: None,
    };
    change_result_from_rust(rust, names)
}

pub(crate) fn value_batch_to_sample_result(
    py: Python<'_>,
    samples: ValueBatch,
) -> PyResult<GcmSampleResult> {
    let n_rows = samples.n_rows;
    let n_nodes = samples.n_nodes;
    let mut means = Vec::with_capacity(n_nodes);
    for i in 0..n_nodes {
        let start = i * n_rows;
        let col = &samples.values[start..start + n_rows];
        let m = col.iter().sum::<f64>() / col.len().max(1) as f64;
        means.push(m);
    }
    let flat = samples.values.as_ref().to_vec();
    let draws = crate::attribution_api::reshape_draws(py, flat, n_nodes, n_rows)?;
    Ok(GcmSampleResult { column_means: means, n_draws: n_rows, n_nodes, draws, route: None })
}

/// Named contribution (node or path label).
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct Contribution {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub score: f64,
}

#[pymethods]
impl Contribution {
    fn __repr__(&self) -> String {
        format!("Contribution(name={:?}, score={})", self.name, self.score)
    }
}

/// Change-attribution summary (components and optional path breakdown).
#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct ChangeAttributionResult {
    #[pyo3(get)]
    pub total_change: f64,
    #[pyo3(get)]
    pub contributions: Vec<(String, f64)>,
    #[pyo3(get)]
    pub path_breakdown: Vec<(Vec<String>, f64)>,
    pub(crate) rust: RustChangeAttributionResult,
    /// The compiled attribution route this result came from.
    pub(crate) route: Option<RouteRecord>,
    /// Whether the attribution is the payload of a compiled study result, whose own
    /// contract states it.
    pub(crate) from_study: bool,
}

impl ChangeAttributionResult {
    pub(crate) fn with_route(mut self, route: RouteRecord) -> Self {
        self.route = Some(route);
        self
    }

    /// Mark the attribution as the payload of a compiled study result.
    pub(crate) fn with_compiled_study_origin(mut self) -> Self {
        self.from_study = true;
        self
    }
}

#[pymethods]
impl ChangeAttributionResult {
    /// The compiled attribution route this result came from.
    #[getter]
    fn route(&self, py: Python<'_>) -> PyResult<crate::compiled_route_api::RouteHandle> {
        crate::compiled_route_api::route_object(py, self.route.as_ref())
    }

    /// The contract the attribution executed under (`query_kind="Attribute"`, and the
    /// structure provenance when the model was fitted on an accepted discovery).
    #[getter]
    fn contract(&self) -> PyResult<HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.contract()
    }

    /// The claim sealed over the attribution.
    #[getter]
    fn claim(&self) -> PyResult<HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.claim()
    }

    /// Cheap inspection record of the compiled attribution study.
    fn inspect(&self) -> PyResult<HashMap<String, String>> {
        crate::compiled_route_api::require_route(self.route.as_ref())?.inspect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChangeAttributionResult(total_change={}, n_contributions={}, n_paths={})",
            self.total_change,
            self.contributions.len(),
            self.path_breakdown.len()
        )
    }
}

/// Anomaly scores for one outcome.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct AnomalyScores {
    #[pyo3(get)]
    pub outcome: String,
    #[pyo3(get)]
    pub mean_score: f64,
    #[pyo3(get)]
    pub n_units: usize,
    /// Per-unit IT scores, aligned with `rows`.
    #[pyo3(get)]
    pub scores: Vec<f64>,
    /// Row indices scored.
    #[pyo3(get)]
    pub rows: Vec<usize>,
    /// Row index of the highest IT score (`None` when no units were scored).
    #[pyo3(get)]
    pub top_row: Option<usize>,
}

#[pymethods]
impl AnomalyScores {
    fn __repr__(&self) -> String {
        format!(
            "AnomalyScores(outcome={:?}, mean_score={}, n_units={}, top_row={:?})",
            self.outcome, self.mean_score, self.n_units, self.top_row
        )
    }
}

/// Mechanism-change detection for one node.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct MechanismChangeDetection {
    #[pyo3(get)]
    pub node: String,
    #[pyo3(get)]
    pub statistic: f64,
    /// Raw per-target p-value, before any multiple-testing correction.
    #[pyo3(get)]
    pub p_value: f64,
    /// P-value after the correction applied across the target family, or `None` when no
    /// correction was applied. `changed` is decided from this when present.
    #[pyo3(get)]
    pub adjusted_p_value: Option<f64>,
    #[pyo3(get)]
    pub changed: bool,
    /// Permutations behind `p_value` when the null is a permutation distribution
    /// (kernel two-sample; small-segment likelihood ratio), else `None`.
    #[pyo3(get)]
    pub n_permutations: Option<usize>,
    /// Smallest attainable raw p-value `1 / (n_permutations + 1)`, or `None`.
    #[pyo3(get)]
    pub p_value_floor: Option<f64>,
}

#[pymethods]
impl MechanismChangeDetection {
    fn __repr__(&self) -> String {
        format!(
            "MechanismChangeDetection(node={:?}, statistic={}, p_value={}, \
             adjusted_p_value={:?}, changed={}, n_permutations={:?}, p_value_floor={:?})",
            self.node,
            self.statistic,
            self.p_value,
            self.adjusted_p_value,
            self.changed,
            self.n_permutations,
            self.p_value_floor
        )
    }
}

/// Feature relevance under interventions.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct FeatureRelevance {
    #[pyo3(get)]
    pub feature: String,
    #[pyo3(get)]
    pub score: f64,
}

#[pymethods]
impl FeatureRelevance {
    fn __repr__(&self) -> String {
        format!("FeatureRelevance(feature={:?}, score={})", self.feature, self.score)
    }
}

/// One ranked design candidate.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct RankedDesign {
    #[pyo3(get)]
    pub candidate_index: usize,
    #[pyo3(get)]
    pub kind: String,
    #[pyo3(get)]
    pub tag: u64,
    #[pyo3(get)]
    pub score: f64,
    #[pyo3(get)]
    pub stderr: f64,
    #[pyo3(get)]
    pub rank: usize,
    #[pyo3(get)]
    pub rank_uncertain: bool,
    #[pyo3(get)]
    pub implemented_functional: String,
    /// `exact` (closed form or finite enumeration) or `monte_carlo` (see `stderr`).
    #[pyo3(get)]
    pub evaluation: String,
}

#[pymethods]
impl RankedDesign {
    fn __repr__(&self) -> String {
        format!(
            "RankedDesign(index={}, kind={}, score={:.6}, rank={})",
            self.candidate_index, self.kind, self.score, self.rank
        )
    }
}

/// Constraint violation recorded during design ranking.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct DesignConstraintViolation {
    #[pyo3(get)]
    pub candidate_index: usize,
    #[pyo3(get)]
    pub constraint: String,
    #[pyo3(get)]
    pub detail: String,
}

#[pymethods]
impl DesignConstraintViolation {
    fn __repr__(&self) -> String {
        format!(
            "DesignConstraintViolation(index={}, constraint={})",
            self.candidate_index, self.constraint
        )
    }
}

/// Full design ranking result.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct DesignRanking {
    #[pyo3(get)]
    pub best_index: usize,
    #[pyo3(get)]
    pub scores: Vec<f64>,
    #[pyo3(get)]
    pub mc_samples: u64,
    #[pyo3(get)]
    pub early_stopped: bool,
    #[pyo3(get)]
    pub ranked: Vec<RankedDesign>,
    #[pyo3(get)]
    pub violations: Vec<DesignConstraintViolation>,
}

#[pymethods]
impl DesignRanking {
    fn __repr__(&self) -> String {
        format!(
            "DesignRanking(best_index={}, n_ranked={}, n_violations={}, mc_samples={})",
            self.best_index,
            self.ranked.len(),
            self.violations.len(),
            self.mc_samples
        )
    }
}

/// Decision evaluation under a utility callback.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct DecisionEvaluation {
    #[pyo3(get)]
    pub expected_utility: f64,
    /// Expected regret of the Bayes action against perfect information (EVPI).
    #[pyo3(get)]
    pub posterior_regret: f64,
    /// Index of the Bayes action.
    #[pyo3(get)]
    pub chosen_action: usize,
}

#[pymethods]
impl DecisionEvaluation {
    fn __repr__(&self) -> String {
        format!(
            "DecisionEvaluation(expected_utility={}, posterior_regret={}, chosen_action={})",
            self.expected_utility, self.posterior_regret, self.chosen_action
        )
    }
}

/// Fitted GCM retained across Python calls (fit once, sample / attribute many times).
///
/// Model, data, and names are [`Arc`]-shared so GIL-detached methods clone handles
/// instead of deep-copying the fitted session on every click.
///
/// Every attribution and simulation runs through the compiled fitted-model route on the
/// program the model was fitted under, reusing this fit; the result keeps that route, and a
/// method that returns a plain list or string records it as `last_route`.
#[pyclass(name = "FittedGcm")]
pub struct PyFittedGcm {
    pub(crate) inner: Arc<FittedGcm>,
    pub(crate) names: Arc<[String]>,
    pub(crate) data: Arc<TabularData>,
    /// The program the mechanisms were fitted under, carrying this fit.
    pub(crate) program: Arc<MechanismProgram>,
    /// The structure accepted from a compiled discovery, when the model was fitted on one.
    pub(crate) accepted: Option<Arc<AcceptedGraph>>,
    /// The route of the last method whose return value has no route of its own.
    pub(crate) last_route: Arc<Mutex<Option<RouteRecord>>>,
}

impl PyFittedGcm {
    fn ctx(seed: u64, threads: Option<u32>) -> ExecutionContext {
        py_execution_context(seed, crate::resolve_user_threads(threads))
    }

    pub(crate) fn new(
        fitted: FittedGcm,
        names: Vec<String>,
        data: TabularData,
        program: MechanismProgram,
        accepted: Option<AcceptedGraph>,
    ) -> Self {
        let inner = Arc::new(fitted);
        let program = program.with_prefit(Arc::clone(&inner), &data);
        Self {
            inner,
            names: Arc::from(names),
            data: Arc::new(data),
            program: Arc::new(program),
            accepted: accepted.map(Arc::new),
            last_route: Arc::new(Mutex::new(None)),
        }
    }

    /// What a compiled route on this fit runs on.
    pub(crate) fn source(&self) -> ModelSource {
        ModelSource {
            data: Arc::clone(&self.data),
            dag: (*self.inner.model.graph).clone(),
            accepted: self.accepted.as_deref().cloned(),
            program: (*self.program).clone(),
        }
    }

    /// Keep `route` as the route of the last call whose return value has none.
    pub(crate) fn record(&self, route: RouteRecord) {
        if let Ok(mut slot) = self.last_route.lock() {
            *slot = Some(route);
        }
    }
}

/// The change attribution of a compiled route, with the route attached.
fn change_of(
    outcome: AttributionOutcome,
    route: RouteRecord,
    names: &[String],
) -> PyResult<ChangeAttributionResult> {
    match outcome {
        AttributionOutcome::Change(result) => {
            Ok(change_result_from_rust(result, names).with_route(route))
        }
        _ => Err(outcome_mismatch()),
    }
}

/// The change population of a time-range comparison.
fn time_ranges(
    outcome: VariableId,
    baseline: (usize, usize),
    comparison: (usize, usize),
) -> ChangeAttributionQuery {
    ChangeAttributionQuery::new(
        outcome,
        PopulationSelector::TimeRange { start: baseline.0, end: baseline.1 },
        PopulationSelector::TimeRange { start: comparison.0, end: comparison.1 },
    )
}

#[pymethods]
impl PyFittedGcm {
    #[getter]
    fn names(&self) -> Vec<String> {
        self.names.to_vec()
    }

    #[getter]
    fn n_assignments(&self) -> usize {
        self.inner.assignments.len()
    }

    /// The compiled route of the last method whose return value carries no route of its
    /// own (feature relevance, anomaly scores, mechanism-change detections, intrinsic
    /// influence); `None` before any such call.
    #[getter]
    fn last_route(&self, py: Python<'_>) -> PyResult<crate::compiled_route_api::RouteHandle> {
        let route = self.last_route.lock().ok().and_then(|slot| slot.clone());
        crate::compiled_route_api::route_object(py, route.as_ref())
    }

    /// The provenance of the accepted discovery this model was fitted on, or `None`.
    fn structure_provenance(&self) -> Option<HashMap<String, String>> {
        self.accepted.as_deref().and_then(AcceptedGraph::provenance).map(|provenance| {
            let mut out = HashMap::new();
            out.insert("method".to_owned(), provenance.method.to_string());
            out.insert("inference".to_owned(), provenance.inference.to_string());
            out.insert("discovered_class".to_owned(), provenance.discovered_class.to_string());
            out.insert("review".to_owned(), provenance.review.to_string());
            out.insert("resolved_marks".to_owned(), provenance.resolved_marks.to_string());
            out
        })
    }

    /// Selected mechanism family id per variable name.
    fn mechanism_kinds(&self) -> Vec<(String, String)> {
        self.inner
            .assignments
            .iter()
            .map(|a| {
                let name = self
                    .names
                    .get(a.variable.as_usize())
                    .cloned()
                    .unwrap_or_else(|| format!("v{}", a.variable.raw()));
                (name, a.selected.id().to_string())
            })
            .collect()
    }

    /// Mechanism selection record per variable: every candidate's held-out score on the
    /// shared folds, the failed candidates with their reasons, the split scheme and seed,
    /// the candidate set and constraint that shaped it, and the selected family's noise
    /// semantics.
    fn mechanism_selection<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner.assignments.iter().map(|a| selection_record(py, &self.names, a)).collect()
    }

    /// Interventional ancestral sample under hard `do` values and/or additive shifts.
    ///
    /// `interventions` pins named variables to absolute values (`do(X := v)`).
    /// `shifts` (optional) instead adds a delta to a variable's structural
    /// assignment (`do(X := X + delta)`) without pinning it — useful for probing
    /// a mechanism's sensitivity rather than overriding it outright. A variable
    /// named in both maps is rejected: it cannot be simultaneously pinned and
    /// shifted.
    ///
    /// `programs` (optional) are batch-callback programs, each
    /// `(mode, variable, callback, inputs)` with `mode` `"assign"` or `"transform"` and
    /// `inputs` variable names (`None` passes the node's own generated value, transforms
    /// only). The callback runs once per batch inside the sampler; its inputs are
    /// dependency-checked like any read, and its output is validated against its declaration.
    #[pyo3(signature = (interventions, n, *, shifts=None, seed=0, threads=None, programs=None))]
    fn sample_do(
        &self,
        py: Python<'_>,
        interventions: HashMap<String, f64>,
        n: usize,
        shifts: Option<HashMap<String, f64>>,
        seed: u64,
        threads: Option<u32>,
        programs: Option<Vec<CallbackProgramArgs>>,
    ) -> PyResult<GcmSampleResult> {
        let source = self.source();
        let data = Arc::clone(&self.data);
        let (flat, n_rows, n_nodes, route) = detach_catch(py, move || {
            let ints =
                interventions_of(&data, &interventions, shifts.as_ref(), programs.as_deref())?;
            let ctx = Self::ctx(seed, threads);
            let target = SimulationTarget::InterventionalSample {
                interventions: Arc::from(ints),
                draws: n,
                seed,
            };
            let (outcome, route) = source.simulate(target, &ctx)?;
            let SimulationOutcome::Sample(samples) = outcome else {
                return Err(outcome_mismatch());
            };
            Ok::<_, PyErr>((
                samples.values.as_ref().to_vec(),
                samples.n_rows,
                samples.n_nodes,
                route,
            ))
        })?;
        let batch = ValueBatch { values: Arc::from(flat), n_rows, n_nodes };
        let mut result = value_batch_to_sample_result(py, batch)?;
        result.route = Some(route);
        Ok(result)
    }

    /// Interventional draws together with the exogenous noise that produced them.
    ///
    /// Returns `(draws, noise)`; `noise` is a `(n_variables, n)` array in the model's
    /// variable order. Replaying it with :meth:`replay_noise` under the same interventions
    /// reproduces the draws bit for bit, and replaying it under another policy gives that
    /// policy's counterfactual on the same units. Arguments are as in :meth:`sample_do`.
    #[pyo3(signature = (interventions, n, *, shifts=None, seed=0, threads=None, programs=None))]
    fn sample_noise<'py>(
        &self,
        py: Python<'py>,
        interventions: HashMap<String, f64>,
        n: usize,
        shifts: Option<HashMap<String, f64>>,
        seed: u64,
        threads: Option<u32>,
        programs: Option<Vec<CallbackProgramArgs>>,
    ) -> PyResult<(GcmSampleResult, Bound<'py, PyAny>)> {
        let source = self.source();
        let data = Arc::clone(&self.data);
        let (draws, noise, route) = detach_catch(py, move || {
            let ints =
                interventions_of(&data, &interventions, shifts.as_ref(), programs.as_deref())?;
            let ctx = Self::ctx(seed, threads);
            let target = SimulationTarget::NoiseSample { interventions: Arc::from(ints), draws: n };
            let (outcome, route) = source.simulate(target, &ctx)?;
            let SimulationOutcome::NoiseSample { draws, noise } = outcome else {
                return Err(outcome_mismatch());
            };
            Ok::<_, PyErr>((draws, noise.values().to_vec(), route))
        })?;
        let n_nodes = draws.n_nodes;
        let noise = PyArray1::from_vec(py, noise).reshape([n_nodes, n])?;
        let mut result = value_batch_to_sample_result(py, draws)?;
        result.route = Some(route);
        Ok((result, noise.into_any()))
    }

    /// Replay retained exogenous noise through the model under interventions.
    ///
    /// `noise` is a `(n_variables, n_rows)` float64 array in the model's variable order, as
    /// :meth:`sample_noise` returns it (or edited, or drawn elsewhere): it must be finite, and
    /// inside (0, 1) for a categorical variable. `interventions`, `shifts` and `programs` are
    /// as in :meth:`sample_do`. A stochastic policy draws from the stream keyed by
    /// `policy_salt`.
    #[pyo3(signature = (noise, interventions=None, *, shifts=None, policy_salt=0, seed=0, threads=None, programs=None))]
    #[allow(clippy::too_many_arguments, reason = "keyword options of one Python call")]
    fn replay_noise(
        &self,
        py: Python<'_>,
        noise: PyReadonlyArray2<'_, f64>,
        interventions: Option<HashMap<String, f64>>,
        shifts: Option<HashMap<String, f64>>,
        policy_salt: u64,
        seed: u64,
        threads: Option<u32>,
        programs: Option<Vec<CallbackProgramArgs>>,
    ) -> PyResult<GcmSampleResult> {
        let model_nodes = self.inner.model.n_nodes();
        let source = self.source();
        let data = Arc::clone(&self.data);
        let (n_nodes, n_rows) = noise.as_array().dim();
        if n_nodes != model_nodes {
            return Err(py_msg(format!(
                "noise has {n_nodes} rows but the model has {model_nodes} variables; pass a \
                 (n_variables, n_rows) array"
            )));
        }
        let values: Vec<f64> = noise.as_array().iter().copied().collect();
        let (batch, route) = detach_catch(py, move || {
            let ints = interventions_of(
                &data,
                &interventions.unwrap_or_default(),
                shifts.as_ref(),
                programs.as_deref(),
            )?;
            let ctx = Self::ctx(seed, threads);
            let target = SimulationTarget::NoiseReplay {
                interventions: Arc::from(ints),
                noise: Arc::from(values),
                n_rows,
                policy_salt,
            };
            let (outcome, route) = source.simulate(target, &ctx)?;
            let SimulationOutcome::Sample(batch) = outcome else {
                return Err(outcome_mismatch());
            };
            Ok::<_, PyErr>((batch, route))
        })?;
        let mut result = value_batch_to_sample_result(py, batch)?;
        result.route = Some(route);
        Ok(result)
    }

    /// Intrinsic causal influence of the noise terms upstream of `target` on its variance, from
    /// this fitted model (nothing is refitted, except the row-bootstrap refits `fit_bootstrap`
    /// asks for). Returns the result as JSON; see `antecedent.intrinsic_influence`.
    #[pyo3(signature = (target, *, independent_noise=false, bidirected_edges=None, allocation="auto", n_permutations=200, outer_draws=200, inner_draws=40, replicates=4, fit_bootstrap=0, posterior_draws=0, level=0.95, max_sources=64, seed=0, threads=None, cancel=None))]
    #[allow(clippy::too_many_arguments, reason = "keyword options of one Python call")]
    fn attribute_intrinsic_influence(
        &self,
        py: Python<'_>,
        target: String,
        independent_noise: bool,
        bidirected_edges: Option<Vec<(String, String)>>,
        allocation: &str,
        n_permutations: usize,
        outer_draws: usize,
        inner_draws: usize,
        replicates: usize,
        fit_bootstrap: u32,
        posterior_draws: usize,
        level: f64,
        max_sources: usize,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let allocation = crate::intrinsic_influence_api::allocation_of(allocation, n_permutations)?;
        let source = self.source();
        let names = Arc::clone(&self.names);
        let (json, route) = detach_catch(py, move || {
            let options = crate::intrinsic_influence_api::IntrinsicOptions {
                independent_noise,
                bidirected_edges,
                allocation,
                outer_draws,
                inner_draws,
                replicates,
                fit_bootstrap,
                posterior_draws,
                level,
                max_sources,
                seed,
                threads,
                cancel,
            };
            crate::intrinsic_influence_api::attribute(&source, &names, &target, options)
        })?;
        self.record(route);
        Ok(json)
    }

    /// Unit-level ITE under hard interventions on `treatment`.
    #[pyo3(signature = (treatment, outcome, active, control, *, seed=0, threads=None))]
    fn counterfactual_ite(
        &self,
        py: Python<'_>,
        treatment: String,
        outcome: String,
        active: f64,
        control: f64,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<GcmIteResult> {
        let source = self.source();
        let data = Arc::clone(&self.data);
        let n_assignments = self.inner.assignments.len();
        let (mean_ite, n_units, noise_inference, unit_vec, route) = detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let ctx = Self::ctx(seed, threads);
            let target =
                SimulationTarget::UnitEffects { treatment: t_id, outcome: y_id, active, control };
            let (outcome, route) = source.simulate(target, &ctx)?;
            let SimulationOutcome::UnitEffects(ite) = outcome else {
                return Err(outcome_mismatch());
            };
            Ok::<_, PyErr>((
                ite.mean_ite,
                ite.unit_effects.len(),
                format!("{:?}", ite.noise_inference),
                ite.unit_effects.as_ref().to_vec(),
                route,
            ))
        })?;
        Ok(GcmIteResult {
            mean_ite,
            n_units,
            noise_inference,
            n_assignments,
            unit_effects: PyArray1::from_vec(py, unit_vec).unbind(),
            route: Some(route),
        })
    }

    #[pyo3(signature = (treatment, outcome, *, path_nodes=None, max_paths=64, max_len=16, seed=0, threads=None))]
    fn attribute_path_specific(
        &self,
        py: Python<'_>,
        treatment: String,
        outcome: String,
        path_nodes: Option<Vec<String>>,
        max_paths: usize,
        max_len: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<ChangeAttributionResult> {
        let source = self.source();
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let mut intermediates = Vec::new();
            if let Some(nodes) = &path_nodes {
                for n in nodes {
                    intermediates.push(data.schema().id_of(n).map_err(py_err)?);
                }
            }
            let ctx = Self::ctx(seed, threads);
            let target = AttributeTarget::Paths {
                sources: Arc::from([t_id]),
                outcome: y_id,
                max_paths,
                max_len,
                through: Arc::from(intermediates),
            };
            let (outcome, route) = source.attribute(target, &ctx)?;
            change_of(outcome, route, &names)
        })
    }

    #[pyo3(signature = (sources, outcome, *, max_paths=64, max_len=16, seed=0, threads=None))]
    fn attribute_paths(
        &self,
        py: Python<'_>,
        sources: Vec<String>,
        outcome: String,
        max_paths: usize,
        max_len: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<ChangeAttributionResult> {
        let source = self.source();
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let ctx = Self::ctx(seed, threads);
            paths_on(&source, &names, &sources, &outcome, max_paths, max_len, &ctx)
        })
    }

    #[pyo3(signature = (outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=None))]
    fn attribute_distribution_change(
        &self,
        py: Python<'_>,
        outcome: String,
        baseline_start: usize,
        baseline_end: usize,
        comparison_start: usize,
        comparison_end: usize,
        n_samples: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<ChangeAttributionResult> {
        let source = self.source();
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let query = time_ranges(
                y_id,
                (baseline_start, baseline_end),
                (comparison_start, comparison_end),
            )
            .with_components(AttributionComponents::Mechanisms)
            .with_allocation(AllocationMethod::Shapley {
                approximation: ShapleyConfig::monte_carlo(n_samples).with_seed(seed),
            });
            let ctx = Self::ctx(seed, threads);
            let target = AttributeTarget::DistributionChange {
                change: query,
                estimator: ChangeEstimator::Sampled {
                    measure: ChangeMeasure::MeanDifference,
                    n_samples: n_samples.max(100),
                    seed,
                },
            };
            let (outcome, route) = source.attribute(target, &ctx)?;
            change_of(outcome, route, &names)
        })
    }

    #[pyo3(signature = (outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=None))]
    fn attribute_distribution_change_robust(
        &self,
        py: Python<'_>,
        outcome: String,
        baseline_start: usize,
        baseline_end: usize,
        comparison_start: usize,
        comparison_end: usize,
        n_samples: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<ChangeAttributionResult> {
        let _ = n_samples;
        let source = self.source();
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let query = ChangeAttributionQuery {
                outcome: y_id,
                baseline: PopulationSelector::TimeRange {
                    start: baseline_start,
                    end: baseline_end,
                },
                comparison: PopulationSelector::TimeRange {
                    start: comparison_start,
                    end: comparison_end,
                },
                components: AttributionComponents::Mechanisms,
                allocation: AllocationMethod::Shapley {
                    approximation: ShapleyConfig::monte_carlo(200),
                },
                max_components: 64,
            };
            let opts = RobustChangeOptions::default();
            let ctx = Self::ctx(seed, threads);
            let target = AttributeTarget::DistributionChange {
                change: query,
                estimator: ChangeEstimator::Robust {
                    max_rows: opts.max_rows,
                    n_samples: opts.n_samples,
                    seed: opts.seed,
                },
            };
            let (outcome, route) = source.attribute(target, &ctx)?;
            change_of(outcome, route, &names)
        })
    }

    #[pyo3(signature = (comparison_edges, outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=None))]
    fn attribute_structure_change(
        &self,
        py: Python<'_>,
        comparison_edges: Vec<(String, String)>,
        outcome: String,
        baseline_start: usize,
        baseline_end: usize,
        comparison_start: usize,
        comparison_end: usize,
        n_samples: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<ChangeAttributionResult> {
        let source = self.source();
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let comparison = comparison_pairs(&dag_from_edges(&data, &comparison_edges)?);
            let query = time_ranges(
                y_id,
                (baseline_start, baseline_end),
                (comparison_start, comparison_end),
            )
            .with_components(AttributionComponents::Structure)
            .with_allocation(AllocationMethod::Shapley {
                approximation: ShapleyConfig::monte_carlo(n_samples).with_seed(seed),
            });
            let ctx = Self::ctx(seed, threads);
            let target = AttributeTarget::StructureChange {
                change: query,
                comparison,
                measure: ChangeMeasure::MeanDifference,
                n_samples: n_samples.max(100),
                seed,
            };
            let (outcome, route) = source.attribute(target, &ctx)?;
            change_of(outcome, route, &names)
        })
    }

    #[pyo3(signature = (outcome, *, max_units=0, seed=0, threads=None))]
    fn attribute_unit_change(
        &self,
        py: Python<'_>,
        outcome: String,
        max_units: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<ChangeAttributionResult> {
        let source = self.source();
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let ctx = Self::ctx(seed, threads);
            let max_u = if max_units == 0 { data.row_count() } else { max_units };
            let target = AttributeTarget::UnitChange(UnitChangeQuery::new(y_id, max_u));
            let (outcome, route) = source.attribute(target, &ctx)?;
            let AttributionOutcome::UnitChange(result) = outcome else {
                return Err(outcome_mismatch());
            };
            let pairs: Vec<(ComponentId, f64)> = result
                .components
                .iter()
                .zip(result.mean_contributions.iter())
                .map(|(c, v)| (*c, *v))
                .collect();
            let total = result.mean_contributions.iter().map(|x| x.abs()).sum();
            Ok(synthetic_change_result(y_id, total, pairs, &names).with_route(route))
        })
    }

    #[pyo3(signature = (outcome, *, delta=1.0, n_samples=200, seed=0, threads=None))]
    fn attribute_feature_relevance(
        &self,
        py: Python<'_>,
        outcome: String,
        delta: f64,
        n_samples: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Vec<FeatureRelevance>> {
        let source = self.source();
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        let (scores, route) = detach_catch(py, move || {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let ctx = Self::ctx(seed, threads);
            let mut features = Vec::new();
            for i in 0..data.schema().len() {
                let id = VariableId::from_raw(crate::fit_u32(i)?);
                if id != y_id {
                    features.push(id);
                }
            }
            let max_features = features.len();
            let target = AttributeTarget::FeatureRelevance {
                outcome: y_id,
                features: Arc::from(features),
                delta,
                n_samples,
                max_features,
            };
            let (outcome, route) = source.attribute(target, &ctx)?;
            let AttributionOutcome::FeatureRelevance(scores) = outcome else {
                return Err(outcome_mismatch());
            };
            let scores: Vec<FeatureRelevance> = scores
                .into_iter()
                .map(|s| FeatureRelevance { feature: var_name(&names, s.feature), score: s.score })
                .collect();
            Ok((scores, route))
        })?;
        self.record(route);
        Ok(scores)
    }

    #[pyo3(signature = (outcomes, *, max_units=0))]
    fn anomaly_attribution(
        &self,
        py: Python<'_>,
        outcomes: Vec<String>,
        max_units: usize,
    ) -> PyResult<Vec<AnomalyScores>> {
        let source = self.source();
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        let (scores, route) = detach_catch(py, move || {
            let outcome_ids: Vec<VariableId> = outcomes
                .iter()
                .map(|n| data.schema().id_of(n).map_err(py_err))
                .collect::<PyResult<_>>()?;
            let max_u = if max_units == 0 { data.row_count() } else { max_units };
            let ctx = Self::ctx(0, None);
            let target = AttributeTarget::Anomaly(AnomalyAttributionQuery::new(outcome_ids, max_u));
            let (outcome, route) = source.attribute(target, &ctx)?;
            let AttributionOutcome::Anomaly(scores) = outcome else {
                return Err(outcome_mismatch());
            };
            Ok((anomaly_scores_from_rust(scores, &names), route))
        })?;
        self.record(route);
        Ok(scores)
    }

    #[pyo3(signature = (baseline_start, baseline_end, comparison_start, comparison_end, *, seed=0, threads=None))]
    fn mechanism_change_detection(
        &self,
        py: Python<'_>,
        baseline_start: usize,
        baseline_end: usize,
        comparison_start: usize,
        comparison_end: usize,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Vec<MechanismChangeDetection>> {
        let source = self.source();
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        let (detections, route) = detach_catch(py, move || {
            let ctx = Self::ctx(seed, threads);
            let mut targets = Vec::with_capacity(data.schema().len());
            for i in 0..data.schema().len() {
                targets.push(VariableId::from_raw(crate::fit_u32(i)?));
            }
            let query = MechanismChangeQuery::new(
                targets,
                PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
                PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
                0.05,
                data.schema().len(),
            );
            let target = AttributeTarget::MechanismChange {
                query,
                test: MechanismChangeTest::MeanDifference,
            };
            let (outcome, route) = source.attribute(target, &ctx)?;
            let AttributionOutcome::MechanismChange(detected) = outcome else {
                return Err(outcome_mismatch());
            };
            let detections: Vec<MechanismChangeDetection> = detected
                .into_iter()
                .map(|d| MechanismChangeDetection {
                    node: var_name(&names, d.variable),
                    statistic: d.statistic,
                    p_value: d.p_value,
                    adjusted_p_value: d.adjusted_p_value,
                    changed: d.changed,
                    n_permutations: d.n_permutations,
                    p_value_floor: d.p_value_floor,
                })
                .collect();
            Ok((detections, route))
        })?;
        self.record(route);
        Ok(detections)
    }

    #[pyo3(signature = (attribution, *, seed=0, threads=None))]
    fn rank_root_causes(
        &self,
        py: Python<'_>,
        attribution: &ChangeAttributionResult,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Vec<Contribution>> {
        let ranks = rank_root_causes(py, attribution, seed, threads)?;
        if let Some(route) = attribution.route.clone() {
            self.record(route);
        }
        Ok(ranks)
    }

    fn __repr__(&self) -> String {
        format!(
            "FittedGcm(n_vars={}, n_assignments={})",
            self.names.len(),
            self.inner.assignments.len()
        )
    }
}

/// Path decomposition of `sources` on `outcome` through the compiled route on `source`.
fn paths_on(
    source: &ModelSource,
    names: &[String],
    sources: &[String],
    outcome: &str,
    max_paths: usize,
    max_len: usize,
    ctx: &ExecutionContext,
) -> PyResult<ChangeAttributionResult> {
    let y_id = source.data.schema().id_of(outcome).map_err(py_err)?;
    let src_ids: Vec<VariableId> = sources
        .iter()
        .map(|n| source.data.schema().id_of(n).map_err(py_err))
        .collect::<PyResult<_>>()?;
    let target = AttributeTarget::Paths {
        sources: Arc::from(src_ids),
        outcome: y_id,
        max_paths,
        max_len,
        through: Arc::from([]),
    };
    let (outcome, route) = source.attribute(target, ctx)?;
    change_of(outcome, route, names)
}

/// The directed edges of a comparison structure, as variable pairs.
fn comparison_pairs(graph: &Dag) -> Arc<[(VariableId, VariableId)]> {
    graph
        .directed_edge_pairs_sorted()
        .into_iter()
        .map(|(from, to)| (VariableId::from_raw(from), VariableId::from_raw(to)))
        .collect()
}

fn family_ids(families: &[MechanismFamily]) -> Vec<&'static str> {
    families.iter().map(|f| f.id()).collect()
}

fn selection_record<'py>(
    py: Python<'py>,
    names: &[String],
    a: &MechanismAssignment,
) -> PyResult<Bound<'py, PyDict>> {
    let record = PyDict::new(py);
    record.set_item("variable", var_name(names, a.variable))?;
    record.set_item("provenance_id", "estimate.gcm.fit")?;
    record.set_item("selected", a.selected.id())?;
    record.set_item("typing", a.typing.id())?;
    record.set_item("noise_inference", a.selected.noise_inference_mode().id())?;
    record.set_item("attribution_compatible", a.selected.attribution_compatible())?;
    let provenance = &a.provenance;
    record.set_item("rule", provenance.rule.id())?;
    record.set_item("splits", provenance.splits.id())?;
    record.set_item(
        "purge",
        match provenance.splits {
            ValidationSplits::TimeOrdered { purge } => Some(purge),
            _ => None,
        },
    )?;
    record.set_item("folds", provenance.folds)?;
    record.set_item("n_rows", provenance.n_rows)?;
    record.set_item("seed", provenance.seed)?;
    record.set_item("candidate_set", family_ids(&provenance.candidate_set))?;
    let constraint = match &provenance.constraint {
        None => None,
        Some(c) => {
            let d = PyDict::new(py);
            d.set_item("allowed", c.allowed.as_deref().map(family_ids))?;
            d.set_item("excluded", family_ids(&c.excluded))?;
            d.set_item("required", c.required.map(MechanismFamily::id))?;
            Some(d)
        }
    };
    record.set_item("constraint", constraint)?;
    let candidates = a
        .candidates
        .iter()
        .map(|c| {
            let d = PyDict::new(py);
            d.set_item("family", c.family.id())?;
            d.set_item("score", c.score)?;
            d.set_item("held_out_log_likelihood", c.held_out_log_likelihood)?;
            d.set_item("complexity_penalty", c.complexity_penalty)?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    record.set_item("candidates", candidates)?;
    let failed = a
        .failed_families
        .iter()
        .map(|(family, reason)| {
            let d = PyDict::new(py);
            d.set_item("family", family.id())?;
            d.set_item("reason", reason)?;
            Ok(d)
        })
        .collect::<PyResult<Vec<_>>>()?;
    record.set_item("failed", failed)?;
    Ok(record)
}

/// Families a candidate token list names for one node constraint field.
fn constraint_families(tokens: &[String]) -> PyResult<Vec<MechanismFamily>> {
    let mut out: Vec<MechanismFamily> = Vec::new();
    for token in tokens {
        for family in antecedent::gcm::candidate_families(token).map_err(py_err)? {
            if !out.contains(&family) {
                out.push(family);
            }
        }
    }
    Ok(out)
}

/// Per-node constraints from `{node: {"allowed": [...], "excluded": [...], "required": id}}`,
/// in node-name order so the resolved options do not depend on dict iteration order.
fn parse_constraints(
    constraints: Option<&Bound<'_, PyDict>>,
    data: &TabularData,
) -> PyResult<Vec<(VariableId, NodeFamilyConstraint)>> {
    let mut entries: Vec<(String, Bound<'_, PyDict>)> = Vec::new();
    if let Some(map) = constraints {
        for (node, spec) in map.iter() {
            entries.push((node.extract()?, spec.cast::<PyDict>().map_err(py_msg)?.clone()));
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Vec::with_capacity(entries.len());
    for (node, spec) in entries {
        let variable = data.schema().id_of(&node).map_err(py_err)?;
        let mut constraint = NodeFamilyConstraint::default();
        for key in spec.keys() {
            let key: String = key.extract()?;
            let value = spec
                .get_item(&key)?
                .ok_or_else(|| py_msg(format!("constraint for `{node}`: missing `{key}`")))?;
            match key.as_str() {
                "allowed" => {
                    let tokens: Vec<String> = value.extract()?;
                    constraint.allowed = Some(Arc::from(constraint_families(&tokens)?));
                }
                "excluded" => {
                    let tokens: Vec<String> = value.extract()?;
                    constraint.excluded = Arc::from(constraint_families(&tokens)?);
                }
                "required" => {
                    let token: String = value.extract()?;
                    let families = antecedent::gcm::candidate_families(&token).map_err(py_err)?;
                    let [family] = families.as_slice() else {
                        return Err(py_msg(format!(
                            "constraint for `{node}`: `required` must name one family id, \
                             `{token}` names {}",
                            families.len()
                        )));
                    };
                    constraint.required = Some(*family);
                }
                other => {
                    return Err(py_msg(format!(
                        "constraint for `{node}`: unknown key `{other}` (expected allowed, \
                         excluded or required)"
                    )));
                }
            }
        }
        out.push((variable, constraint));
    }
    Ok(out)
}

fn parse_splits(splits: &str, purge: usize) -> PyResult<ValidationSplits> {
    match (splits, purge) {
        ("shuffled", 0) => Ok(ValidationSplits::Shuffled),
        ("clustered", 0) => Ok(ValidationSplits::Clustered),
        ("time_ordered", purge) => Ok(ValidationSplits::TimeOrdered { purge }),
        ("shuffled" | "clustered", _) => {
            Err(py_msg(format!("purge applies only to splits=\"time_ordered\", not `{splits}`")))
        }
        (other, _) => Err(py_msg(format!(
            "unknown splits `{other}` (expected shuffled, time_ordered or clustered)"
        ))),
    }
}

/// Fit a graphical causal model; return a reusable [`FittedGcm`].
///
/// With no keyword it fits the standard candidate set. `candidates` (family ids or
/// the aliases `linear`, `logistic`, `additive_spline`, `interactions`, `random_forest`,
/// `gradient_boosted_trees`, `tree_ensembles`, `tree_ensemble_classifier`) replaces the
/// candidate set; `constraints` narrows it per node; `splits` declares how rows are
/// cut for the held-out scores (`shuffled`, `time_ordered` with an optional `purge`,
/// or `clustered` by the `UnitId` column).
#[pyfunction]
#[pyo3(
    name = "fit_gcm",
    signature = (names, columns, edges, *, threads=None, candidates=None, constraints=None, splits="shuffled", purge=0)
)]
#[allow(clippy::too_many_arguments, reason = "keyword-only selection options of one Python call")]
fn fit_gcm_py(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    threads: Option<u32>,
    candidates: Option<Vec<String>>,
    constraints: Option<Bound<'_, PyDict>>,
    splits: &str,
    purge: usize,
) -> PyResult<PyFittedGcm> {
    let _ = threads;
    let (data, _) = crate::tabular_from_py_columns(names.clone(), columns)?;
    let selection = Selection::parse(candidates, constraints.as_ref(), &data, splits, purge)?;
    detach_catch(py, move || {
        let g = dag_from_edges(&data, &edges)?;
        selection.fit(g, data, names, None)
    })
}

/// The candidate set and selection rule of one fit, and the mechanism program that states them.
struct Selection {
    candidates: Option<Vec<String>>,
    registry: MechanismRegistry,
    options: SelectionOptions,
}

impl Selection {
    fn parse(
        candidates: Option<Vec<String>>,
        constraints: Option<&Bound<'_, PyDict>>,
        data: &TabularData,
        splits: &str,
        purge: usize,
    ) -> PyResult<Self> {
        let registry = match &candidates {
            None => MechanismRegistry::standard(),
            Some(tokens) => {
                let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
                antecedent::gcm::registry_from_candidates(&tokens).map_err(py_err)?
            }
        };
        let options = SelectionOptions {
            constraints: parse_constraints(constraints, data)?,
            splits: parse_splits(splits, purge)?,
        };
        Ok(Self { candidates, registry, options })
    }

    fn is_standard(&self) -> bool {
        self.candidates.is_none() && self.options == SelectionOptions::default()
    }

    fn program(&self) -> PyResult<MechanismProgram> {
        match &self.candidates {
            None if self.is_standard() => Ok(MechanismProgram::standard()),
            None => Ok(MechanismProgram::standard().with_options(self.options.clone())),
            Some(tokens) => {
                let tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
                MechanismProgram::from_candidates(&tokens, self.options.clone()).map_err(py_err)
            }
        }
    }

    /// Fit on `graph` and keep the program, so every later route reuses this fit.
    fn fit(
        &self,
        graph: Dag,
        data: TabularData,
        names: Vec<String>,
        accepted: Option<AcceptedGraph>,
    ) -> PyResult<PyFittedGcm> {
        let fit = if self.is_standard() {
            fit_gcm(graph, &data)
        } else {
            fit_gcm_with(graph, &data, &self.registry, &self.options)
        };
        let fitted = fit.map_err(py_err)?;
        Ok(PyFittedGcm::new(fitted, names, data, self.program()?, accepted))
    }
}

/// Fit a graphical causal model on the structure a compiled discovery returned.
///
/// `discovery` is a result of a `discover_*` entry point (PC, GES, LiNGAM, NOTEARS, ...).
/// Its route accepts the structure through the review gate as a fully oriented DAG (pending
/// directed edges are accepted; undirected or circle marks refuse), and the model is fitted
/// on that accepted DAG. Every attribution and simulation on the returned model states the
/// discovery's provenance (method, inference, query, data snapshot, program and claim) in
/// its contract. Returns `(fitted, edges)`, `edges` the accepted directed edges by name.
#[pyfunction]
#[pyo3(
    name = "fit_gcm_accepted",
    signature = (names, columns, discovery, *, threads=None, candidates=None, constraints=None, splits="shuffled", purge=0)
)]
#[allow(clippy::too_many_arguments, reason = "keyword-only selection options of one Python call")]
fn fit_gcm_accepted_py(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    discovery: PyRef<'_, crate::discovery_api::PcmciDiscoveryResult>,
    threads: Option<u32>,
    candidates: Option<Vec<String>>,
    constraints: Option<Bound<'_, PyDict>>,
    splits: &str,
    purge: usize,
) -> PyResult<(PyFittedGcm, Vec<(String, String)>)> {
    let _ = threads;
    let route = discovery
        .route
        .clone()
        .ok_or_else(|| crate::value_err("this discovery result has no compiled route to accept"))?;
    drop(discovery);
    let (data, _) = crate::tabular_from_py_columns(names.clone(), columns)?;
    let selection = Selection::parse(candidates, constraints.as_ref(), &data, splits, purge)?;
    detach_catch(py, move || {
        let accepted = route.execution.accept_dag(&route.ctx).map_err(py_err)?;
        let dag = accepted
            .as_dag()
            .cloned()
            .ok_or_else(|| crate::value_err("the accepted structure is not a DAG"))?;
        if dag.node_count() != data.schema().len() {
            return Err(crate::value_err(format!(
                "the discovery ran on {} variables but {} columns were supplied; pass the \
                 columns the discovery read, in the same order",
                dag.node_count(),
                data.schema().len()
            )));
        }
        let edges: Vec<(String, String)> = dag
            .directed_edge_pairs_sorted()
            .into_iter()
            .map(|(from, to)| {
                (
                    var_name(&names, VariableId::from_raw(from)),
                    var_name(&names, VariableId::from_raw(to)),
                )
            })
            .collect();
        let fitted = selection.fit(dag, data, names, Some(accepted))?;
        Ok((fitted, edges))
    })
}

/// Path decomposition (all paths from `sources` to `outcome`).
#[pyfunction]
#[pyo3(signature = (names, columns, edges, sources, outcome, *, max_paths=64, max_len=16, seed=0, threads=None))]
fn attribute_paths(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    edges: Vec<(String, String)>,
    sources: Vec<String>,
    outcome: String,
    max_paths: usize,
    max_len: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<ChangeAttributionResult> {
    let (data, _) = crate::tabular_from_py_columns(names.clone(), columns)?;
    detach_catch(py, move || {
        let source = ModelSource {
            dag: dag_from_edges(&data, &edges)?,
            data: Arc::new(data),
            accepted: None,
            program: MechanismProgram::standard(),
        };
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        paths_on(&source, &names, &sources, &outcome, max_paths, max_len, &ctx)
    })
}

/// Rank root causes from a [`ChangeAttributionResult`].
///
/// The ranking reads only the attribution it is given (no model, no data): it orders the
/// components of an attribution a compiled route returned, and that route states its
/// contract. An attribution with no compiled route is refused, so no ranking reaches a
/// caller without the contract of the answer it ranks.
#[pyfunction]
#[pyo3(signature = (attribution, *, seed=0, threads=None))]
fn rank_root_causes(
    py: Python<'_>,
    attribution: &ChangeAttributionResult,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<Vec<Contribution>> {
    if attribution.route.is_none() && !attribution.from_study {
        return Err(py_msg(
            "rank_root_causes ranks an attribution a compiled route returned; run the change \
             attribution through FittedGcm or antecedent.gcm first",
        ));
    }
    let rust = attribution.rust.clone();
    let name_map: HashMap<u32, String> = attribution
        .rust
        .contributions
        .iter()
        .zip(attribution.contributions.iter())
        .map(|(c, (name, _))| (c.component.raw(), name.clone()))
        .collect();
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let ranks = facade_rank_root_causes(&rust, &ctx).map_err(py_err)?;
        Ok(ranks
            .into_iter()
            .map(|r| Contribution {
                name: name_map
                    .get(&r.component.raw())
                    .cloned()
                    .unwrap_or_else(|| format!("V{}", r.component.raw())),
                score: r.score,
            })
            .collect())
    })
}

// A single successful fit is shared across callers. The physical mutex protects
// only state transitions; no computation or Python observer runs under it.
enum FitState<T> {
    Empty,
    Fitting { observer: bool },
    Ready(Arc<T>),
}

struct FitCache<T> {
    state: Mutex<FitState<T>>,
    published: AtomicBool,
}

impl<T> Default for FitCache<T> {
    fn default() -> Self {
        Self { state: Mutex::new(FitState::Empty), published: AtomicBool::new(false) }
    }
}

struct FitOwner<'a, T>(&'a FitCache<T>);
impl<T> Drop for FitOwner<'_, T> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if matches!(*state, FitState::Fitting { .. }) {
            *state = FitState::Empty;
        }
    }
}

impl<T> FitCache<T> {
    fn materialize(
        &self,
        ctx: &ExecutionContext,
        fit: impl FnOnce() -> PyResult<T>,
        has_observer: bool,
        mut observe: impl FnMut(f64),
        mut on_wait: impl FnMut(),
    ) -> PyResult<Arc<T>> {
        let cancelled = || py_err(antecedent::CausalError::Cancelled { stage: "policy_fit" });
        loop {
            if ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            match &*state {
                FitState::Ready(value) => return Ok(Arc::clone(value)),
                FitState::Empty => {
                    *state = FitState::Fitting { observer: false };
                    break;
                }
                FitState::Fitting { observer: true } => {
                    return Err(crate::CausalStateError::new_err(
                        "policy fit is busy invoking a progress observer; retry after it returns",
                    ));
                }
                FitState::Fitting { observer: false } => {}
            }
            drop(state);
            on_wait();
            std::thread::park_timeout(std::time::Duration::from_millis(5));
        }
        let _owner = FitOwner(self);
        let mut report = |fraction| -> PyResult<()> {
            if !has_observer {
                return if ctx.cancellation.is_cancelled() { Err(cancelled()) } else { Ok(()) };
            }
            {
                let mut state =
                    self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                *state = FitState::Fitting { observer: true };
            }
            observe(fraction);
            {
                let mut state =
                    self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                *state = FitState::Fitting { observer: false };
            }
            if ctx.cancellation.is_cancelled() {
                return Err(cancelled());
            }
            Ok(())
        };
        report(0.0)?;
        let value = Arc::new(fit()?);
        report(1.0)?;
        if ctx.cancellation.is_cancelled() {
            return Err(cancelled());
        }
        let mut state = self.state.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        *state = FitState::Ready(Arc::clone(&value));
        self.published.store(true, Ordering::Release);
        Ok(value)
    }
}

struct PolicyMaterialization {
    fitted: Arc<FittedGcm>,
    _retention: antecedent_core::execution::AdmissionLease,
}

/// Checked topology and program over owned observations, before statistical fitting.
/// Forks share successful fitted ownership; rebinding owns new rows and clears that fit.
#[pyclass(name = "_PreparedPolicyAnalysis", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyPreparedPolicyAnalysis {
    data: Arc<TabularData>,
    names: Arc<[String]>,
    graph: Dag,
    selection: Arc<Selection>,
    template: antecedent::gcm::PreparedPolicy,
    fitted: Arc<FitCache<PolicyMaterialization>>,
}

#[pymethods]
impl PyPreparedPolicyAnalysis {
    #[new]
    #[pyo3(signature = (names, columns, edges, policies, outcome, *, candidates=None, constraints=None, splits="shuffled", purge=0))]
    #[allow(clippy::too_many_arguments)]
    fn prepare(
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<Bound<'_, PyAny>>,
        edges: Vec<(String, String)>,
        policies: &str,
        outcome: &str,
        candidates: Option<Vec<String>>,
        constraints: Option<Bound<'_, PyDict>>,
        splits: &str,
        purge: usize,
    ) -> PyResult<Self> {
        let (data, _) = crate::tabular_from_py_columns(names.clone(), columns)?;
        data.schema().id_of(outcome).map_err(py_err)?;
        let selection = Selection::parse(candidates, constraints.as_ref(), &data, splits, purge)?;
        // Lower expressions and their dependency ordering now, without mechanisms or samples.
        let interventions = antecedent_io::policy_program_artifact::interventions_from_named_json(
            policies,
            &|name| data.schema().id_of(name).ok().map(VariableId::raw),
        )
        .map_err(crate::intervention_program_api::io_err)?;
        detach_catch(py, move || {
            let graph = dag_from_edges(&data, &edges)?;
            let topology =
                antecedent_model::CompiledCausalModel::compile(graph.clone()).map_err(py_err)?;
            let template = antecedent::gcm::PreparedPolicy::prepare(&topology, interventions)
                .map_err(py_err)?;
            Ok(Self {
                data: Arc::new(data),
                names: Arc::from(names),
                graph,
                selection: Arc::new(selection),
                template,
                fitted: Arc::new(FitCache::default()),
            })
        })
    }

    fn fork(&self) -> Self {
        self.clone()
    }

    fn bind(&self, names: Vec<String>, columns: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        if names.as_slice() != self.names.as_ref() {
            return Err(crate::value_err("policy observations must preserve variables and order"));
        }
        let (data, _) = crate::tabular_from_py_columns(names, columns)?;
        Ok(Self {
            data: Arc::new(data),
            names: Arc::clone(&self.names),
            graph: self.graph.clone(),
            selection: Arc::clone(&self.selection),
            template: self.template.clone(),
            fitted: Arc::new(FitCache::default()),
        })
    }

    #[getter]
    fn identity(&self) -> Option<String> {
        antecedent_io::policy_program_artifact::policy_program_identity(
            self.template.interventions(),
        )
        .ok()
        .map(|digest| antecedent_core::to_hex(&digest))
    }

    fn export_program<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let bytes = antecedent_io::policy_program_artifact::encode_policy_program(
            self.template.interventions(),
        )
        .map_err(crate::intervention_program_api::io_err)?;
        Ok(pyo3::types::PyBytes::new(py, &bytes))
    }

    /// Deterministic physical work dimensions, without fitting or drawing.
    fn quote(&self, draws: usize) -> PyResult<HashMap<String, u64>> {
        let width =
            u64::try_from(self.names.len()).map_err(|error| crate::value_err(error.to_string()))?;
        let values = |rows: usize| -> PyResult<u64> {
            u64::try_from(rows)
                .ok()
                .and_then(|rows| rows.checked_mul(width))
                .and_then(|cells| cells.checked_mul(8))
                .ok_or_else(|| {
                    crate::CausalResourceError::new_err("policy work dimensions overflow")
                })
        };
        Ok(HashMap::from([
            ("fit_workspace_bytes".into(), values(self.data.row_count())?),
            ("draw_matrix_bytes".into(), values(draws)?),
        ]))
    }

    #[getter]
    fn fit_materialized(&self) -> bool {
        self.fitted.published.load(Ordering::Acquire)
    }

    #[pyo3(signature = (seed=1, threads=None))]
    fn materialize(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<(PyFittedGcm, crate::intervention_program_api::PyPreparedPolicy)> {
        let fitted = detach_catch(py, || {
            let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
            // Admission also checks cancellation on a reused fit, before publishing a run.
            let admission_error = |error: antecedent_core::execution::AdmissionError| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    py_err(antecedent::CausalError::Cancelled { stage: "policy_fit" })
                }
                error => crate::CausalResourceError::new_err(error.to_string()),
            };
            let _check = ctx
                .acquire_admission_bounded(0, 0, std::time::Duration::ZERO)
                .map_err(admission_error)?;
            self.fitted.materialize(
                &ctx,
                || {
                    let bytes = (self.data.row_count() as u64)
                        .checked_mul(self.names.len() as u64)
                        .and_then(|cells| cells.checked_mul(8))
                        .ok_or_else(|| {
                            crate::CausalResourceError::new_err("policy fit workspace overflow")
                        })?;
                    let workspace = ctx
                        .acquire_admission_bounded(0, bytes, std::time::Duration::ZERO)
                        .map_err(admission_error)?;
                    let selection = if self.selection.is_standard() {
                        None
                    } else {
                        Some((&self.selection.registry, &self.selection.options))
                    };
                    let fitted = antecedent::gcm::fit_gcm_polled(
                        self.graph.clone(),
                        &self.data,
                        selection,
                        &ctx,
                    )
                    .map_err(py_err)?;
                    let _completed = ctx
                        .acquire_admission_bounded(0, 0, std::time::Duration::ZERO)
                        .map_err(admission_error)?;
                    Ok(PolicyMaterialization { fitted: Arc::new(fitted), _retention: workspace })
                },
                ctx.progress.is_some(),
                |fraction| {
                    if let Some(progress) = &ctx.progress {
                        progress.report(fraction, "policy_fit");
                    }
                },
                || {},
            )
        })?;
        let program = self.selection.program()?.with_prefit(Arc::clone(&fitted.fitted), &self.data);
        let model = PyFittedGcm {
            inner: Arc::clone(&fitted.fitted),
            names: Arc::clone(&self.names),
            data: Arc::clone(&self.data),
            program: Arc::new(program),
            accepted: None,
            last_route: Arc::new(Mutex::new(None)),
        };
        let core = self.template.rebind(&model.inner.model).map_err(py_err)?;
        let native_model = Py::new(py, model)?;
        let policy = crate::intervention_program_api::PyPreparedPolicy::from_template(
            native_model.clone_ref(py),
            core,
            self.identity(),
        );
        // Return a second header on the same native storage so result routes are independent.
        let model = native_model.bind(py).borrow();
        let header = PyFittedGcm {
            inner: Arc::clone(&model.inner),
            names: Arc::clone(&model.names),
            data: Arc::clone(&model.data),
            program: Arc::clone(&model.program),
            accepted: None,
            last_route: Arc::clone(&model.last_route),
        };
        Ok((header, policy))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Contribution>()?;
    m.add_class::<ChangeAttributionResult>()?;
    m.add_class::<AnomalyScores>()?;
    m.add_class::<MechanismChangeDetection>()?;
    m.add_class::<FeatureRelevance>()?;
    m.add_class::<RankedDesign>()?;
    m.add_class::<DesignConstraintViolation>()?;
    m.add_class::<DesignRanking>()?;
    m.add_class::<DecisionEvaluation>()?;
    m.add_class::<PyFittedGcm>()?;
    m.add_class::<PyPreparedPolicyAnalysis>()?;
    m.add_function(wrap_pyfunction!(fit_gcm_py, m)?)?;
    m.add_function(wrap_pyfunction!(fit_gcm_accepted_py, m)?)?;
    m.add_function(wrap_pyfunction!(attribute_paths, m)?)?;
    m.add_function(wrap_pyfunction!(rank_root_causes, m)?)?;
    Ok(())
}

#[cfg(test)]
mod fit_cache_tests {
    use super::*;
    use std::sync::mpsc;

    // Run blocking coordination in a child so even a broken lock protocol or a
    // panic in scoped threads cannot wedge the main test process indefinitely.
    fn watchdog(name: &str) -> bool {
        const CHILD: &str = "ANTECEDENT_S20_FIT_CACHE_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            return false;
        }
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg(format!("gcm_api::fit_cache_tests::{name}"))
            .arg("--nocapture")
            .env(CHILD, "1")
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success(), "isolated test {name} failed: {status}");
                return true;
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("isolated test {name} timed out");
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    #[test]
    fn queued_waiter_refuses_when_observer_begins() {
        if watchdog("queued_waiter_refuses_when_observer_begins") {
            return;
        }
        Python::initialize();
        let cache = Arc::new(FitCache::<usize>::default());
        let (fitting_tx, fitting_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel();
        let (observer_tx, observer_rx) = mpsc::channel();
        let (observer_release_tx, observer_release_rx) = mpsc::channel();
        let (waiting_tx, waiting_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let first_cache = Arc::clone(&cache);
            let first = scope.spawn(move || {
                let ctx = ExecutionContext::for_tests(4);
                first_cache.materialize(
                    &ctx,
                    || {
                        fitting_tx.send(()).unwrap();
                        finish_rx.recv().unwrap();
                        Ok(17)
                    },
                    true,
                    |fraction| {
                        if fraction == 1.0 {
                            observer_tx.send(()).unwrap();
                            observer_release_rx.recv().unwrap();
                        }
                    },
                    || {},
                )
            });
            fitting_rx.recv().unwrap();
            let waiter_cache = Arc::clone(&cache);
            let waiter = scope.spawn(move || {
                let ctx = ExecutionContext::for_tests(5);
                let mut notify = Some(waiting_tx);
                let result = waiter_cache.materialize(
                    &ctx,
                    || panic!("unexpected refit"),
                    false,
                    |_| {},
                    || {
                        if let Some(sender) = notify.take() {
                            sender.send(()).unwrap();
                        }
                    },
                );
                let typed = result.is_err_and(|error| {
                    Python::attach(|py| error.is_instance_of::<crate::CausalStateError>(py))
                });
                result_tx.send(typed).unwrap();
            });
            waiting_rx.recv().unwrap();
            finish_tx.send(()).unwrap();
            observer_rx.recv().unwrap();
            let refused_before_callback_returns =
                result_rx.recv_timeout(std::time::Duration::from_secs(2));
            observer_release_tx.send(()).unwrap();
            assert!(refused_before_callback_returns.unwrap());
            first.join().unwrap().unwrap();
            waiter.join().unwrap();
        });
    }

    #[test]
    fn waiter_receives_published_fit_after_observer_finishes() {
        if watchdog("waiter_receives_published_fit_after_observer_finishes") {
            return;
        }
        Python::initialize();
        let cache = Arc::new(FitCache::<usize>::default());
        let (fitting_tx, fitting_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel();
        let (waiting_tx, waiting_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let (observer_tx, observer_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let first_cache = Arc::clone(&cache);
            let first = scope.spawn(move || {
                let ctx = ExecutionContext::for_tests(6);
                first_cache.materialize(
                    &ctx,
                    || {
                        fitting_tx.send(()).unwrap();
                        finish_rx.recv().unwrap();
                        Ok(31)
                    },
                    true,
                    |fraction| {
                        if fraction == 1.0 {
                            observer_tx.send(()).unwrap();
                        }
                    },
                    || {},
                )
            });
            fitting_rx.recv().unwrap();
            let waiter_cache = Arc::clone(&cache);
            let waiter = scope.spawn(move || {
                let ctx = ExecutionContext::for_tests(7);
                let mut wait_gate = Some((waiting_tx, resume_rx));
                waiter_cache.materialize(
                    &ctx,
                    || panic!("unexpected refit"),
                    false,
                    |_| {},
                    || {
                        if let Some((sender, receiver)) = wait_gate.take() {
                            sender.send(()).unwrap();
                            receiver.recv().unwrap();
                        }
                    },
                )
            });
            waiting_rx.recv().unwrap();
            finish_tx.send(()).unwrap();
            observer_rx.recv().unwrap();
            let first = first.join().unwrap().unwrap();
            resume_tx.send(()).unwrap();
            let waiter = waiter.join().unwrap().unwrap();
            assert!(Arc::ptr_eq(&first, &waiter));
            assert_eq!(*waiter, 31);
        });
    }

    #[test]
    fn panicked_fit_releases_owner_for_retry() {
        if watchdog("panicked_fit_releases_owner_for_retry") {
            return;
        }
        Python::initialize();
        let cache = Arc::new(FitCache::<usize>::default());
        let ctx = ExecutionContext::for_tests(8);
        let first = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = cache.materialize(&ctx, || panic!("injected fit panic"), false, |_| {}, || {});
        }));
        assert!(first.is_err());
        assert!(!cache.published.load(Ordering::Acquire));
        let retry_cache = Arc::clone(&cache);
        let (sender, receiver) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            let ctx = ExecutionContext::for_tests(9);
            let result = retry_cache.materialize(&ctx, || Ok(44), false, |_| {}, || {});
            sender.send(result.map(|value| *value)).unwrap();
        });
        let retried = receiver.recv_timeout(std::time::Duration::from_secs(2));
        assert_eq!(retried.unwrap().unwrap(), 44);
        thread.join().unwrap();
        assert!(cache.published.load(Ordering::Acquire));
    }

    #[test]
    fn concurrent_callers_share_one_fit_and_queued_cancellation_is_prompt() {
        if watchdog("concurrent_callers_share_one_fit_and_queued_cancellation_is_prompt") {
            return;
        }
        Python::initialize();
        let cache = Arc::new(FitCache::<usize>::default());
        let ctx = ExecutionContext::for_tests(1);
        let waiting_ctx = ExecutionContext::for_tests(2);
        let cancel_ctx = waiting_ctx.clone();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (waiting_tx, waiting_rx) = mpsc::channel();
        let (cancel_waiting_tx, cancel_waiting_rx) = mpsc::channel();
        let (cancel_result_tx, cancel_result_rx) = mpsc::channel();
        std::thread::scope(|scope| {
            let first_cache = Arc::clone(&cache);
            let first = scope.spawn(move || {
                first_cache.materialize(
                    &ctx,
                    || {
                        entered_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                        Ok(42)
                    },
                    false,
                    |_| {},
                    || {},
                )
            });
            entered_rx.recv().unwrap();
            let second_cache = Arc::clone(&cache);
            let second = scope.spawn(move || {
                let ctx = ExecutionContext::for_tests(3);
                let mut notify = Some(waiting_tx);
                second_cache.materialize(
                    &ctx,
                    || panic!("second caller refit"),
                    false,
                    |_| {},
                    || {
                        if let Some(sender) = notify.take() {
                            sender.send(()).unwrap();
                        }
                    },
                )
            });
            waiting_rx.recv().unwrap();
            let third_cache = Arc::clone(&cache);
            let third = scope.spawn(move || {
                let mut notify = Some(cancel_waiting_tx);
                let result = third_cache.materialize(
                    &waiting_ctx,
                    || panic!("cancelled caller refit"),
                    false,
                    |_| {},
                    || {
                        if let Some(sender) = notify.take() {
                            sender.send(()).unwrap();
                        }
                    },
                );
                let typed = result.is_err_and(|error| {
                    Python::attach(|py| error.is_instance_of::<crate::CausalCancelledError>(py))
                });
                cancel_result_tx.send(typed).unwrap();
            });
            cancel_waiting_rx.recv().unwrap();
            cancel_ctx.cancellation.cancel();
            let cancelled_before_release =
                cancel_result_rx.recv_timeout(std::time::Duration::from_secs(2));
            release_tx.send(()).unwrap();
            assert!(cancelled_before_release.unwrap());
            let first = first.join().unwrap().unwrap();
            let second = second.join().unwrap().unwrap();
            third.join().unwrap();
            assert!(Arc::ptr_eq(&first, &second));
            assert_eq!(*first, 42);
        });
    }
}
