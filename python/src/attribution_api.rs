//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

use crate::*;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment, outcome, active, control, *, seed=0, threads=1))]
fn counterfactual_ite(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    active: f64,
    control: f64,
    seed: u64,
    threads: u32,
) -> PyResult<GcmIteResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    let (mean_ite, n_units, noise_inference, n_assignments, unit_vec) =
        detach_catch(py, move || {
            let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
            let data = loaded.data;
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let n_vars = u32::try_from(data.schema().len())
                .map_err(|_| PyValueError::new_err("too many variables"))?;
            let mut g = Dag::with_variables(n_vars);
            for (from, to) in &edges {
                let from_id = data.schema().id_of(from).map_err(py_err)?;
                let to_id = data.schema().id_of(to).map_err(py_err)?;
                g.insert_directed(
                    DenseNodeId::from_raw(from_id.raw()),
                    DenseNodeId::from_raw(to_id.raw()),
                )
                .map_err(py_err)?;
            }
            let fitted = fit_gcm(g, &data).map_err(py_err)?;
            let n_assignments = fitted.assignments.len();
            let ctx = py_execution_context(seed, threads);
            let ite =
                facade_counterfactual_ite(fitted.model, &data, t_id, y_id, active, control, &ctx)
                    .map_err(py_err)?;
            Ok::<_, PyErr>((
                ite.mean_ite,
                ite.unit_effects.len(),
                format!("{:?}", ite.noise_inference),
                n_assignments,
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

/// Fit GCM and return interventional column means + draws under hard `do(treatment=value)`.
///
/// `mechanism_wrappers` maps variable name → object with `sample_noise(n)` / `evaluate(parents, noise)`
/// ( slow path).
#[pyfunction]
#[pyo3(name = "sample_do", signature = (names, columns, edges, treatment, do_value, n_draws, *, seed=0, threads=1, mechanism_wrappers=None))]
fn sample_do_py(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    do_value: f64,
    n_draws: usize,
    seed: u64,
    threads: u32,
    mechanism_wrappers: Option<Bound<'_, PyDict>>,
) -> PyResult<GcmSampleResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    let wrappers = mechanism_wrappers.map(Bound::unbind);
    let threads = if wrappers.is_some() { 1 } else { threads };
    let (means, n_rows, n_nodes, flat) = detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let model = if let Some(w) = wrappers {
            Python::attach(|py| {
                let dict = w.bind(py);
                callbacks::apply_mechanism_wrappers(&fitted.model, &names, dict)
            })?
        } else {
            fitted.model
        };
        let ctx = py_execution_context(seed, threads);
        let mut rng = CausalRng::from_seed(seed);
        let samples = facade_sample_do(
            &model,
            &[Intervention::set(t_id, Value::f64(do_value))],
            n_draws,
            &mut rng,
            &ctx,
        )
        .map_err(py_err)?;
        let mut means = Vec::with_capacity(samples.n_nodes);
        for i in 0..samples.n_nodes {
            let start = i * samples.n_rows;
            let col = &samples.values[start..start + samples.n_rows];
            let m = col.iter().sum::<f64>() / col.len().max(1) as f64;
            means.push(m);
        }
        Ok::<_, PyErr>((means, samples.n_rows, samples.n_nodes, samples.values.as_ref().to_vec()))
    })?;
    let draws = PyArray1::from_vec(py, flat).reshape([n_nodes, n_rows])?.unbind();
    Ok(GcmSampleResult { column_means: means, n_draws: n_rows, n_nodes, draws })
}

/// Sample an interventional distribution via [`InterventionalDistributionQuery`].
///
/// Same return shape as [`gcm_sample_do`]; builds the typed query then samples.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment, do_value, n_draws, outcome=None, *, seed=0, threads=1))]
fn sample_interventional_distribution(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    do_value: f64,
    n_draws: usize,
    outcome: Option<String>,
    seed: u64,
    threads: u32,
) -> PyResult<GcmSampleResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    let (means, n_rows, n_nodes, flat) = detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_name = match &outcome {
            Some(o) => o.as_str(),
            None => names.last().map_or("y", String::as_str),
        };
        let y_id = data.schema().id_of(y_name).map_err(py_err)?;
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let query = InterventionalDistributionQuery::new(
            y_id,
            [Intervention::set(t_id, Value::f64(do_value))],
        );
        let ctx = py_execution_context(seed, threads);
        let mut rng = CausalRng::from_seed(seed);
        let samples = facade_sample_interventional_distribution(
            &fitted.model,
            &query,
            n_draws,
            &mut rng,
            &ctx,
        )
        .map_err(py_err)?;
        let mut means = Vec::with_capacity(samples.n_nodes);
        for i in 0..samples.n_nodes {
            let start = i * samples.n_rows;
            let col = &samples.values[start..start + samples.n_rows];
            let m = col.iter().sum::<f64>() / col.len().max(1) as f64;
            means.push(m);
        }
        Ok::<_, PyErr>((means, samples.n_rows, samples.n_nodes, samples.values.as_ref().to_vec()))
    })?;
    let draws = PyArray1::from_vec(py, flat).reshape([n_nodes, n_rows])?.unbind();
    Ok(GcmSampleResult { column_means: means, n_draws: n_rows, n_nodes, draws })
}

/// Path-specific contribution via [`PathSpecificEffectQuery`] / `path_decompose`.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment, outcome, *, path_nodes=None, max_paths=64, max_len=16, seed=0, threads=1))]
fn attribute_path_specific(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    path_nodes: Option<Vec<String>>,
    max_paths: usize,
    max_len: usize,
    seed: u64,
    threads: u32,
) -> PyResult<gcm_api::ChangeAttributionResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let mut intermediates = Vec::new();
        if let Some(nodes) = &path_nodes {
            for n in nodes {
                intermediates.push(data.schema().id_of(n).map_err(py_err)?);
            }
        }
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let mut query = PathSpecificEffectQuery::binary(t_id, y_id)
            .with_max_paths(max_paths)
            .with_max_len(max_len);
        if !intermediates.is_empty() {
            query = query.with_path_nodes(intermediates);
        }
        let ctx = py_execution_context(seed, threads);
        let result = facade_attribute_path_specific(&fitted.model, &query, &ctx).map_err(py_err)?;
        Ok(gcm_api::change_result_from_rust(result, &names))
    })
}

fn quantity_wire_name(q: &PosteriorQuantityWire) -> String {
    match q {
        PosteriorQuantityWire::Coefficient { index, name } => {
            name.clone().unwrap_or_else(|| format!("coef_{index}"))
        }
        PosteriorQuantityWire::ResidualVariance => "residual_variance".into(),
        PosteriorQuantityWire::Effect { name } | PosteriorQuantityWire::Scalar { name } => {
            name.clone()
        }
    }
}

/// Fit GCM and attribute distribution change between two row ranges via Shapley.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=1))]
fn attribute_distribution_change(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    outcome: String,
    baseline_start: usize,
    baseline_end: usize,
    comparison_start: usize,
    comparison_end: usize,
    n_samples: usize,
    seed: u64,
    threads: u32,
) -> PyResult<gcm_api::ChangeAttributionResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let query = ChangeAttributionQuery::new(
            y_id,
            PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
            PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
        )
        .with_components(AttributionComponents::Mechanisms)
        .with_allocation(AllocationMethod::Shapley {
            approximation: ShapleyConfig::monte_carlo(n_samples).with_seed(seed),
        });
        let ctx = py_execution_context(seed, threads);
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: n_samples.max(100),
            seed,
        };
        let result =
            facade_attribute_distribution_change(&fitted.model, &data, &query, &opts, &ctx)
                .map_err(py_err)?;
        Ok(gcm_api::change_result_from_rust(result, &names))
    })
}

/// Structure-change attribution between two edge lists (parent-set Shapley).
#[pyfunction]
#[pyo3(signature = (names, columns, baseline_edges, comparison_edges, outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=1))]
fn attribute_structure_change(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    baseline_edges: Vec<(String, String)>,
    comparison_edges: Vec<(String, String)>,
    outcome: String,
    baseline_start: usize,
    baseline_end: usize,
    comparison_start: usize,
    comparison_end: usize,
    n_samples: usize,
    seed: u64,
    threads: u32,
) -> PyResult<gcm_api::ChangeAttributionResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g0 = Dag::with_variables(n_vars);
        for (from, to) in &baseline_edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g0.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let mut g1 = Dag::with_variables(n_vars);
        for (from, to) in &comparison_edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g1.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let baseline = CompiledCausalModel::compile(g0).map_err(py_msg)?;
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
        let ctx = py_execution_context(seed, threads);
        let opts = StructureChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: n_samples.max(100),
            seed,
        };
        let result =
            facade_attribute_structure_change(&baseline, &comparison, &data, &query, &opts, &ctx)
                .map_err(py_err)?;
        Ok(gcm_api::change_result_from_rust(result, &names))
    })
}

/// Evaluate a decision problem under a Python utility callback .
///
/// `utility(actions, outcomes) -> flat float64 ndarray` of length `len(actions) * len(outcomes)`.
#[pyfunction]
#[pyo3(signature = (actions, outcomes, utility))]
fn evaluate_decision_py(
    py: Python<'_>,
    actions: Vec<f64>,
    outcomes: Vec<f64>,
    utility: Bound<'_, PyAny>,
) -> PyResult<gcm_api::DecisionEvaluation> {
    if !utility.is_callable() {
        return Err(PyValueError::new_err("utility must be callable"));
    }
    let util = Arc::new(callbacks::PyUtility::new(utility.unbind()));
    let problem = DecisionProblem::new(actions, util, Vec::new());
    // Keep GIL acquired: utility callback reacquires anyway; this is an explicit slow path.
    let eval = facade_evaluate_decision(&problem, &outcomes);
    let _ = py; // silence unused if optimized
    Ok(gcm_api::DecisionEvaluation {
        expected_utility: eval.expected_utility,
        posterior_regret: eval.posterior_regret,
        chosen_action: eval.chosen_action,
    })
}

/// Decode a serialized posterior artifact into summaries + column-major draws.
#[pyfunction]
fn decode_posterior_artifact(bytes: Vec<u8>) -> PyResult<PosteriorArtifact> {
    catch_ffi(|| {
        let (meta, draws) = decode_causal_posterior_bytes(&bytes).map_err(py_err)?;
        Ok(PosteriorArtifact {
            n_draws: meta.n_draws as usize,
            mean: meta.mean,
            sd: meta.sd,
            q025: meta.q025,
            q975: meta.q975,
            draws,
            backend_id: meta.backend_id,
            identification: meta.identification,
            unidentified_mass: meta.unidentified_mass,
            converged: meta.converged,
            hessian_condition: meta.hessian_condition,
            quantity_names: meta.quantities.iter().map(quantity_wire_name).collect(),
        })
    })
}

/// Re-encode a decoded [`PosteriorArtifact`] to container bytes (round-trip).
#[pyfunction]
fn encode_posterior_artifact(artifact: &PosteriorArtifact) -> PyResult<Vec<u8>> {
    catch_ffi(|| {
        let quantities: Vec<PosteriorQuantityWire> = artifact
            .quantity_names
            .iter()
            .map(|name| {
                if name == "residual_variance" {
                    PosteriorQuantityWire::ResidualVariance
                } else if name.starts_with("coef_") {
                    let index =
                        name.strip_prefix("coef_").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                    PosteriorQuantityWire::Coefficient { index, name: None }
                } else {
                    PosteriorQuantityWire::Effect { name: name.clone() }
                }
            })
            .collect();
        let meta = CausalPosteriorWire {
            quantities,
            n_draws: u32::try_from(artifact.n_draws)
                .map_err(|_| PyValueError::new_err("n_draws exceeds u32"))?,
            mean: artifact.mean.clone(),
            sd: artifact.sd.clone(),
            q025: artifact.q025.clone(),
            q975: artifact.q975.clone(),
            identification: artifact.identification.clone(),
            unidentified_mass: artifact.unidentified_mass,
            backend_id: artifact.backend_id.clone(),
            converged: artifact.converged,
            hessian_condition: artifact.hessian_condition,
            draws_encoding: "f64_le_colmajor".into(),
        };
        let art = encode_posterior_wire(&meta, &artifact.draws, "py-posterior", VERSION)
            .map_err(py_err)?;
        let mut buf = Vec::new();
        art.write_to(&mut buf).map_err(py_err)?;
        Ok(buf)
    })
}

/// Parse DOT digraph text; return `(node_count, edges)`.

#[pyfunction]
#[pyo3(signature = (names, columns, edges, outcomes, *, max_units=0))]
fn anomaly_attribution(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    outcomes: Vec<String>,
    max_units: usize,
) -> PyResult<Vec<gcm_api::AnomalyScores>> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let outcome_ids: Vec<VariableId> = outcomes
            .iter()
            .map(|n| data.schema().id_of(n).map_err(py_err))
            .collect::<PyResult<_>>()?;
        let max_u = if max_units == 0 { data.row_count() } else { max_units };
        let scores =
            facade_anomaly_attribution(&fitted.model, &data, outcome_ids, max_u).map_err(py_err)?;
        Ok(scores
            .into_iter()
            .map(|s| {
                let name = names
                    .get(s.target.as_usize())
                    .cloned()
                    .unwrap_or_else(|| format!("var{}", s.target.raw()));
                let mean = if s.scores.is_empty() {
                    0.0
                } else {
                    s.scores.iter().sum::<f64>() / s.scores.len() as f64
                };
                gcm_api::AnomalyScores { outcome: name, mean_score: mean, n_units: s.rows.len() }
            })
            .collect())
    })
}

/// Unit-level change attribution.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, outcome, *, max_units=0, seed=0, threads=1))]
fn attribute_unit_change(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    outcome: String,
    max_units: usize,
    seed: u64,
    threads: u32,
) -> PyResult<gcm_api::ChangeAttributionResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let max_u = if max_units == 0 { data.row_count() } else { max_units };
        let query = UnitChangeQuery::new(y_id, max_u);
        let result =
            facade_attribute_unit_change(&fitted.model, &data, &query, &ctx).map_err(py_err)?;
        let pairs: Vec<(antecedent_core::ComponentId, f64)> = result
            .components
            .iter()
            .zip(result.mean_contributions.iter())
            .map(|(c, v)| (*c, *v))
            .collect();
        let total = result.mean_contributions.iter().map(|x| x.abs()).sum();
        Ok(gcm_api::synthetic_change_result(y_id, total, pairs, &names))
    })
}

/// Feature relevance scores for parents of `outcome`.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, outcome, *, delta=1.0, n_samples=200, seed=0, threads=1))]
fn attribute_feature_relevance(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    outcome: String,
    delta: f64,
    n_samples: usize,
    seed: u64,
    threads: u32,
) -> PyResult<Vec<gcm_api::FeatureRelevance>> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let features: Vec<VariableId> = (0..data.schema().len())
            .map(|i| VariableId::from_raw(u32::try_from(i).unwrap()))
            .filter(|id| *id != y_id)
            .collect();
        let scores = facade_attribute_feature_relevance(
            &fitted.model,
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
            .map(|s| {
                let name = names
                    .get(s.feature.as_usize())
                    .cloned()
                    .unwrap_or_else(|| format!("var{}", s.feature.raw()));
                gcm_api::FeatureRelevance { feature: name, score: s.score }
            })
            .collect())
    })
}

/// Robust distribution-change attribution between two row ranges.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=1))]
fn attribute_distribution_change_robust(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    outcome: String,
    baseline_start: usize,
    baseline_end: usize,
    comparison_start: usize,
    comparison_end: usize,
    n_samples: usize,
    seed: u64,
    threads: u32,
) -> PyResult<gcm_api::ChangeAttributionResult> {
    let _ = n_samples;
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let query = ChangeAttributionQuery {
            outcome: y_id,
            baseline: PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
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
        let opts = antecedent::gcm::RobustChangeOptions::default();
        let ctx = py_execution_context(seed, threads);
        let result =
            facade_attribute_distribution_change_robust(&fitted.model, &data, &query, &opts, &ctx)
                .map_err(py_err)?;
        Ok(gcm_api::change_result_from_rust(result, &names))
    })
}

/// Detect mechanism changes across two row ranges.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, baseline_start, baseline_end, comparison_start, comparison_end, *, seed=0, threads=1))]
fn mechanism_change_detection(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    baseline_start: usize,
    baseline_end: usize,
    comparison_start: usize,
    comparison_end: usize,
    seed: u64,
    threads: u32,
) -> PyResult<Vec<gcm_api::MechanismChangeDetection>> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut g = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data.schema().id_of(from).map_err(py_err)?;
            let to_id = data.schema().id_of(to).map_err(py_err)?;
            g.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }
        let fitted = fit_gcm(g, &data).map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
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
            &fitted.model,
            &data,
            &query,
            antecedent::gcm::MechanismChangeMethod::MeanDiff,
            &ctx,
        )
        .map_err(py_err)?;
        Ok(detected
            .into_iter()
            .map(|d| {
                let name = names
                    .get(d.variable.as_usize())
                    .cloned()
                    .unwrap_or_else(|| format!("var{}", d.variable.raw()));
                gcm_api::MechanismChangeDetection {
                    node: name,
                    statistic: d.statistic,
                    p_value: d.p_value,
                    changed: d.changed,
                }
            })
            .collect())
    })
}

/// Parse NetworkX adjacency JSON; return `(node_count, edges)`.

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(counterfactual_ite, m)?)?;
    m.add_function(wrap_pyfunction!(sample_do_py, m)?)?;
    m.add_function(wrap_pyfunction!(sample_interventional_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(attribute_path_specific, m)?)?;
    m.add_function(wrap_pyfunction!(attribute_distribution_change, m)?)?;
    m.add_function(wrap_pyfunction!(attribute_distribution_change_robust, m)?)?;
    m.add_function(wrap_pyfunction!(attribute_structure_change, m)?)?;
    m.add_function(wrap_pyfunction!(anomaly_attribution, m)?)?;
    m.add_function(wrap_pyfunction!(attribute_unit_change, m)?)?;
    m.add_function(wrap_pyfunction!(attribute_feature_relevance, m)?)?;
    m.add_function(wrap_pyfunction!(mechanism_change_detection, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_decision_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_posterior_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(encode_posterior_artifact, m)?)?;
    Ok(())
}
