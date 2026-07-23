//! Fitted GCM OO surface and typed attribution / design result pyclasses.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent::gcm::{
    ChangeAttributionResult as RustChangeAttributionResult, DifferenceMeasure,
    DistributionChangeOptions, FittedGcm, RobustChangeOptions, StructureChangeOptions,
    anomaly_attribution as facade_anomaly_attribution,
    attribute_distribution_change as facade_attribute_distribution_change,
    attribute_distribution_change_robust as facade_attribute_distribution_change_robust,
    attribute_feature_relevance as facade_attribute_feature_relevance,
    attribute_path_specific as facade_attribute_path_specific,
    attribute_paths as facade_attribute_paths,
    attribute_structure_change as facade_attribute_structure_change,
    attribute_unit_change as facade_attribute_unit_change,
    counterfactual_ite as facade_counterfactual_ite, fit_gcm,
    mechanism_change_detection as facade_mechanism_change_detection,
    rank_root_causes as facade_rank_root_causes, sample_do as facade_sample_do,
};
use causal_attribution::{CacheStats, ComponentContribution, ComputeBudget};
use causal_core::{
    AllocationMethod, AttributionComponents, CausalRng, ChangeAttributionQuery, ComponentId,
    ExecutionContext, Intervention, MechanismChangeQuery, PathSpecificEffectQuery,
    PopulationSelector, ShapleyConfig, UnitChangeQuery, Value, VariableId,
};
use causal_data::{TableView, TabularData, tabular_from_record_batch};
use causal_graph::{Dag, DenseNodeId};
use causal_model::{CompiledCausalModel, ValueBatch};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::{
    GcmIteResult, GcmSampleResult, columns_to_batch, detach_catch, py_err, py_execution_context,
    py_msg,
};

fn var_name(names: &[String], id: VariableId) -> String {
    names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("V{}", id.raw()))
}

fn component_name(names: &[String], id: ComponentId) -> String {
    names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("V{}", id.raw()))
}

fn dag_from_edges(data: &TabularData, edges: &[(String, String)]) -> PyResult<Dag> {
    let n_vars = u32::try_from(data.schema().len())
        .map_err(|_| PyValueError::new_err("too many variables"))?;
    let mut g = Dag::with_variables(n_vars);
    for (from, to) in edges {
        let from_id = data.schema().id_of(from).map_err(py_err)?;
        let to_id = data.schema().id_of(to).map_err(py_err)?;
        g.insert_directed(DenseNodeId::from_raw(from_id.raw()), DenseNodeId::from_raw(to_id.raw()))
            .map_err(py_err)?;
    }
    Ok(g)
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

pub(crate) fn change_result_from_rust(
    result: RustChangeAttributionResult,
    names: &[String],
) -> ChangeAttributionResult {
    ChangeAttributionResult {
        total_change: result.total_change,
        contributions: contribution_pairs(names, &result),
        path_breakdown: path_breakdown_names(names, &result),
        rust: result,
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
        contributions: Arc::from(contributions_rust),
        interactions: Arc::from([]),
        path_breakdown: Arc::from([]),
        unidentified: Arc::from([]),
        graph_sensitivity: None,
        budget: ComputeBudget::default(),
        monte_carlo_stderr: None,
        component_mc_stderr: None,
        cache_stats: CacheStats::default(),
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
    let draws = PyArray1::from_vec(py, flat).reshape([n_nodes, n_rows])?.unbind();
    Ok(GcmSampleResult { column_means: means, n_draws: n_rows, n_nodes, draws })
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
}

#[pymethods]
impl ChangeAttributionResult {
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
}

#[pymethods]
impl AnomalyScores {
    fn __repr__(&self) -> String {
        format!(
            "AnomalyScores(outcome={:?}, mean_score={}, n_units={})",
            self.outcome, self.mean_score, self.n_units
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
    #[pyo3(get)]
    pub p_value: f64,
    #[pyo3(get)]
    pub changed: bool,
}

#[pymethods]
impl MechanismChangeDetection {
    fn __repr__(&self) -> String {
        format!(
            "MechanismChangeDetection(node={:?}, statistic={}, p_value={}, changed={})",
            self.node, self.statistic, self.p_value, self.changed
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
    #[pyo3(get)]
    pub posterior_regret: f64,
    #[pyo3(get)]
    pub chosen_action: Option<usize>,
}

#[pymethods]
impl DecisionEvaluation {
    fn __repr__(&self) -> String {
        format!(
            "DecisionEvaluation(expected_utility={}, posterior_regret={}, chosen_action={:?})",
            self.expected_utility, self.posterior_regret, self.chosen_action
        )
    }
}

/// Fitted GCM retained across Python calls (fit once, sample / attribute many times).
///
/// Model, data, and names are [`Arc`]-shared so GIL-detached methods clone handles
/// instead of deep-copying the fitted session on every click.
#[pyclass(name = "FittedGcm")]
pub struct PyFittedGcm {
    pub(crate) inner: Arc<FittedGcm>,
    pub(crate) names: Arc<[String]>,
    pub(crate) data: Arc<TabularData>,
}

impl PyFittedGcm {
    fn ctx(seed: u64, threads: u32) -> ExecutionContext {
        py_execution_context(seed, threads)
    }
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

    /// Interventional ancestral sample under hard `do` values.
    #[pyo3(signature = (interventions, n, *, seed=0, threads=1))]
    fn sample_do(
        &self,
        py: Python<'_>,
        interventions: HashMap<String, f64>,
        n: usize,
        seed: u64,
        threads: u32,
    ) -> PyResult<GcmSampleResult> {
        let inner = Arc::clone(&self.inner);
        let data = Arc::clone(&self.data);
        let (flat, n_rows, n_nodes) = detach_catch(py, move || {
            let mut ints = Vec::with_capacity(interventions.len());
            for (name, value) in interventions {
                let id = data.schema().id_of(&name).map_err(py_err)?;
                ints.push(Intervention::set(id, Value::f64(value)));
            }
            let ctx = Self::ctx(seed, threads);
            let mut rng = CausalRng::from_seed(seed);
            let samples =
                facade_sample_do(&inner.model, &ints, n, &mut rng, &ctx).map_err(py_err)?;
            Ok::<_, PyErr>((samples.values.as_ref().to_vec(), samples.n_rows, samples.n_nodes))
        })?;
        let batch = ValueBatch { values: Arc::from(flat), n_rows, n_nodes };
        value_batch_to_sample_result(py, batch)
    }

    /// Unit-level ITE under hard interventions on `treatment`.
    #[pyo3(signature = (treatment, outcome, active, control, *, seed=0, threads=1))]
    fn counterfactual_ite(
        &self,
        py: Python<'_>,
        treatment: String,
        outcome: String,
        active: f64,
        control: f64,
        seed: u64,
        threads: u32,
    ) -> PyResult<GcmIteResult> {
        let inner = Arc::clone(&self.inner);
        let data = Arc::clone(&self.data);
        let n_assignments = self.inner.assignments.len();
        let (mean_ite, n_units, noise_inference, unit_vec) = detach_catch(py, move || {
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let ctx = Self::ctx(seed, threads);
            let ite = facade_counterfactual_ite(
                inner.model.clone(),
                &data,
                t_id,
                y_id,
                active,
                control,
                &ctx,
            )
            .map_err(py_err)?;
            Ok::<_, PyErr>((
                ite.mean_ite,
                ite.unit_effects.len(),
                format!("{:?}", ite.noise_inference),
                ite.unit_effects.as_ref().to_vec(),
            ))
        })?;
        Ok(GcmIteResult {
            mean_ite,
            n_units,
            noise_inference,
            n_assignments,
            unit_effects: PyArray1::from_vec(py, unit_vec).unbind(),
        })
    }

    #[pyo3(signature = (treatment, outcome, *, path_nodes=None, max_paths=64, max_len=16, seed=0, threads=1))]
    fn attribute_path_specific(
        &self,
        py: Python<'_>,
        treatment: String,
        outcome: String,
        path_nodes: Option<Vec<String>>,
        max_paths: usize,
        max_len: usize,
        seed: u64,
        threads: u32,
    ) -> PyResult<ChangeAttributionResult> {
        let inner = Arc::clone(&self.inner);
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
            let mut query = PathSpecificEffectQuery::binary(t_id, y_id)
                .with_max_paths(max_paths)
                .with_max_len(max_len);
            if !intermediates.is_empty() {
                query = query.with_path_nodes(intermediates);
            }
            let ctx = Self::ctx(seed, threads);
            let result =
                facade_attribute_path_specific(&inner.model, &query, &ctx).map_err(py_err)?;
            Ok(change_result_from_rust(result, &names))
        })
    }

    #[pyo3(signature = (sources, outcome, *, max_paths=64, max_len=16, seed=0, threads=1))]
    fn attribute_paths(
        &self,
        py: Python<'_>,
        sources: Vec<String>,
        outcome: String,
        max_paths: usize,
        max_len: usize,
        seed: u64,
        threads: u32,
    ) -> PyResult<ChangeAttributionResult> {
        let inner = Arc::clone(&self.inner);
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let src_ids: Vec<VariableId> = sources
                .iter()
                .map(|n| data.schema().id_of(n).map_err(py_err))
                .collect::<PyResult<_>>()?;
            let ctx = Self::ctx(seed, threads);
            let result =
                facade_attribute_paths(&inner.model, &src_ids, y_id, max_paths, max_len, &ctx)
                    .map_err(py_err)?;
            Ok(change_result_from_rust(result, &names))
        })
    }

    #[pyo3(signature = (outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=1))]
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
        threads: u32,
    ) -> PyResult<ChangeAttributionResult> {
        let inner = Arc::clone(&self.inner);
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let query = ChangeAttributionQuery::new(
                y_id,
                PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
                PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
            )
            .with_components(AttributionComponents::Mechanisms)
            .with_allocation(AllocationMethod::Shapley {
                approximation: ShapleyConfig::monte_carlo(n_samples).with_seed(seed),
            });
            let ctx = Self::ctx(seed, threads);
            let opts = DistributionChangeOptions {
                measure: DifferenceMeasure::MeanDiff,
                n_samples: n_samples.max(100),
                seed,
            };
            let result =
                facade_attribute_distribution_change(&inner.model, &data, &query, &opts, &ctx)
                    .map_err(py_err)?;
            Ok(change_result_from_rust(result, &names))
        })
    }

    #[pyo3(signature = (outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=1))]
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
        threads: u32,
    ) -> PyResult<ChangeAttributionResult> {
        let _ = n_samples;
        let inner = Arc::clone(&self.inner);
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
            let result = facade_attribute_distribution_change_robust(
                &inner.model,
                &data,
                &query,
                &opts,
                &ctx,
            )
            .map_err(py_err)?;
            Ok(change_result_from_rust(result, &names))
        })
    }

    #[pyo3(signature = (comparison_edges, outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=1))]
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
        threads: u32,
    ) -> PyResult<ChangeAttributionResult> {
        let inner = Arc::clone(&self.inner);
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let baseline = &inner.model;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let g1 = dag_from_edges(&data, &comparison_edges)?;
            let comparison = CompiledCausalModel::compile(g1).map_err(py_msg)?;
            let query = ChangeAttributionQuery::new(
                y_id,
                PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
                PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
            )
            .with_components(AttributionComponents::Structure)
            .with_allocation(AllocationMethod::Shapley {
                approximation: ShapleyConfig::monte_carlo(n_samples).with_seed(seed),
            });
            let ctx = Self::ctx(seed, threads);
            let opts = StructureChangeOptions {
                measure: DifferenceMeasure::MeanDiff,
                n_samples: n_samples.max(100),
                seed,
            };
            let result = facade_attribute_structure_change(
                baseline,
                &comparison,
                &data,
                &query,
                &opts,
                &ctx,
            )
            .map_err(py_err)?;
            Ok(change_result_from_rust(result, &names))
        })
    }

    #[pyo3(signature = (outcome, *, max_units=0, seed=0, threads=1))]
    fn attribute_unit_change(
        &self,
        py: Python<'_>,
        outcome: String,
        max_units: usize,
        seed: u64,
        threads: u32,
    ) -> PyResult<ChangeAttributionResult> {
        let inner = Arc::clone(&self.inner);
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let ctx = Self::ctx(seed, threads);
            let max_u = if max_units == 0 { data.row_count() } else { max_units };
            let query = UnitChangeQuery::new(y_id, max_u);
            let result =
                facade_attribute_unit_change(&inner.model, &data, &query, &ctx).map_err(py_err)?;
            let pairs: Vec<(ComponentId, f64)> = result
                .components
                .iter()
                .zip(result.mean_contributions.iter())
                .map(|(c, v)| (*c, *v))
                .collect();
            let total = result.mean_contributions.iter().map(|x| x.abs()).sum();
            Ok(synthetic_change_result(y_id, total, pairs, &names))
        })
    }

    #[pyo3(signature = (outcome, *, delta=1.0, n_samples=200, seed=0, threads=1))]
    fn attribute_feature_relevance(
        &self,
        py: Python<'_>,
        outcome: String,
        delta: f64,
        n_samples: usize,
        seed: u64,
        threads: u32,
    ) -> PyResult<Vec<FeatureRelevance>> {
        let inner = Arc::clone(&self.inner);
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let ctx = Self::ctx(seed, threads);
            let features: Vec<VariableId> = (0..data.schema().len())
                .map(|i| VariableId::from_raw(u32::try_from(i).unwrap()))
                .filter(|id| *id != y_id)
                .collect();
            let scores = facade_attribute_feature_relevance(
                &inner.model,
                &data,
                y_id,
                &features,
                delta,
                n_samples,
                features.len(),
                &ctx,
            )
            .map_err(py_err)?;
            Ok(scores
                .into_iter()
                .map(|s| FeatureRelevance { feature: var_name(&names, s.feature), score: s.score })
                .collect())
        })
    }

    #[pyo3(signature = (outcomes, *, max_units=0))]
    fn anomaly_attribution(
        &self,
        py: Python<'_>,
        outcomes: Vec<String>,
        max_units: usize,
    ) -> PyResult<Vec<AnomalyScores>> {
        let inner = Arc::clone(&self.inner);
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let outcome_ids: Vec<VariableId> = outcomes
                .iter()
                .map(|n| data.schema().id_of(n).map_err(py_err))
                .collect::<PyResult<_>>()?;
            let max_u = if max_units == 0 { data.row_count() } else { max_units };
            let scores = facade_anomaly_attribution(&inner.model, &data, outcome_ids, max_u)
                .map_err(py_err)?;
            Ok(scores
                .into_iter()
                .map(|s| {
                    let mean = if s.scores.is_empty() {
                        0.0
                    } else {
                        s.scores.iter().sum::<f64>() / s.scores.len() as f64
                    };
                    AnomalyScores {
                        outcome: var_name(&names, s.target),
                        mean_score: mean,
                        n_units: s.rows.len(),
                    }
                })
                .collect())
        })
    }

    #[pyo3(signature = (baseline_start, baseline_end, comparison_start, comparison_end, *, seed=0, threads=1))]
    fn mechanism_change_detection(
        &self,
        py: Python<'_>,
        baseline_start: usize,
        baseline_end: usize,
        comparison_start: usize,
        comparison_end: usize,
        seed: u64,
        threads: u32,
    ) -> PyResult<Vec<MechanismChangeDetection>> {
        let inner = Arc::clone(&self.inner);
        let data = Arc::clone(&self.data);
        let names = Arc::clone(&self.names);
        detach_catch(py, move || {
            let ctx = Self::ctx(seed, threads);
            let targets: Vec<VariableId> = (0..data.schema().len())
                .map(|i| VariableId::from_raw(u32::try_from(i).unwrap()))
                .collect();
            let query = MechanismChangeQuery::new(
                targets,
                PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
                PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
                0.05,
                data.schema().len(),
            );
            let detected = facade_mechanism_change_detection(
                &inner.model,
                &data,
                &query,
                antecedent::gcm::MechanismChangeMethod::MeanDiff,
                &ctx,
            )
            .map_err(py_err)?;
            Ok(detected
                .into_iter()
                .map(|d| MechanismChangeDetection {
                    node: var_name(&names, d.variable),
                    statistic: d.statistic,
                    p_value: d.p_value,
                    changed: d.changed,
                })
                .collect())
        })
    }

    #[pyo3(signature = (attribution, *, seed=0, threads=1))]
    fn rank_root_causes(
        &self,
        py: Python<'_>,
        attribution: &ChangeAttributionResult,
        seed: u64,
        threads: u32,
    ) -> PyResult<Vec<Contribution>> {
        let _ = self;
        rank_root_causes(py, attribution, seed, threads)
    }

    fn __repr__(&self) -> String {
        format!(
            "FittedGcm(n_vars={}, n_assignments={})",
            self.names.len(),
            self.inner.assignments.len()
        )
    }
}

/// Fit a linear-Gaussian GCM; return a reusable [`FittedGcm`].
#[pyfunction]
#[pyo3(name = "fit_gcm", signature = (names, columns, edges, *, threads=1))]
fn fit_gcm_py(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    threads: u32,
) -> PyResult<PyFittedGcm> {
    let _ = threads;
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let g = dag_from_edges(&data, &edges)?;
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        Ok(PyFittedGcm { inner: Arc::new(fitted), names: Arc::from(names), data: Arc::new(data) })
    })
}

/// Path decomposition (all paths from `sources` to `outcome`).
#[pyfunction]
#[pyo3(signature = (names, columns, edges, sources, outcome, *, max_paths=64, max_len=16, seed=0, threads=1))]
fn attribute_paths(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    sources: Vec<String>,
    outcome: String,
    max_paths: usize,
    max_len: usize,
    seed: u64,
    threads: u32,
) -> PyResult<ChangeAttributionResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let g = dag_from_edges(&data, &edges)?;
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let src_ids: Vec<VariableId> = sources
            .iter()
            .map(|n| data.schema().id_of(n).map_err(py_err))
            .collect::<PyResult<_>>()?;
        let ctx = py_execution_context(seed, threads);
        let result =
            facade_attribute_paths(&fitted.model, &src_ids, y_id, max_paths, max_len, &ctx)
                .map_err(py_err)?;
        Ok(change_result_from_rust(result, &names))
    })
}

/// Rank root causes from a [`ChangeAttributionResult`].
#[pyfunction]
#[pyo3(signature = (attribution, *, seed=0, threads=1))]
fn rank_root_causes(
    py: Python<'_>,
    attribution: &ChangeAttributionResult,
    seed: u64,
    threads: u32,
) -> PyResult<Vec<Contribution>> {
    let rust = attribution.rust.clone();
    let name_map: HashMap<u32, String> = attribution
        .rust
        .contributions
        .iter()
        .zip(attribution.contributions.iter())
        .map(|(c, (name, _))| (c.component.raw(), name.clone()))
        .collect();
    detach_catch(py, move || {
        let ctx = py_execution_context(seed, threads);
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
    m.add_function(wrap_pyfunction!(fit_gcm_py, m)?)?;
    m.add_function(wrap_pyfunction!(attribute_paths, m)?)?;
    m.add_function(wrap_pyfunction!(rank_root_causes, m)?)?;
    Ok(())
}
