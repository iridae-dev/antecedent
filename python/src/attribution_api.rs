//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

use crate::compiled_route_api::{ModelSource, RouteRecord, outcome_mismatch};
use crate::*;
use antecedent::{AttributionOutcome, MechanismProgram, SimulationOutcome};
use antecedent_core::{AnomalyAttributionQuery, AttributeTarget, ChangeEstimator, ChangeMeasure};
use antecedent_core::{MechanismChangeTest, SimulationTarget};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// The standard mechanism program fitted on `edges` over `data`, as a compiled route runs it.
fn model_source(
    data: antecedent_data::TabularData,
    edges: &[(String, String)],
) -> PyResult<ModelSource> {
    Ok(ModelSource {
        dag: dag_from_named_edges(data.schema(), edges)?,
        data: Arc::new(data),
        accepted: None,
        program: MechanismProgram::standard(),
    })
}

/// The change attribution of a compiled route, with the route attached.
fn change_of(
    outcome: AttributionOutcome,
    route: RouteRecord,
    names: &[String],
) -> PyResult<gcm_api::ChangeAttributionResult> {
    match outcome {
        AttributionOutcome::Change(result) => {
            Ok(gcm_api::change_result_from_rust(result, names).with_route(route))
        }
        _ => Err(outcome_mismatch()),
    }
}

/// Column means of a draw batch, its shape and its flat values.
fn sample_parts(samples: &antecedent_model::ValueBatch) -> (Vec<f64>, usize, usize, Vec<f64>) {
    let mut means = Vec::with_capacity(samples.n_nodes);
    for i in 0..samples.n_nodes {
        let start = i * samples.n_rows;
        let col = &samples.values[start..start + samples.n_rows];
        let m = col.iter().sum::<f64>() / col.len().max(1) as f64;
        means.push(m);
    }
    (means, samples.n_rows, samples.n_nodes, samples.values.as_ref().to_vec())
}

/// Interventional draws of `iv` through the compiled simulation route on `source`.
fn sample_route(
    source: &ModelSource,
    iv: Intervention,
    n_draws: usize,
    seed: u64,
    ctx: &ExecutionContext,
) -> PyResult<((Vec<f64>, usize, usize, Vec<f64>), RouteRecord)> {
    let target = SimulationTarget::InterventionalSample {
        interventions: Arc::from([iv]),
        draws: n_draws,
        seed,
    };
    let (outcome, route) = source.simulate(target, ctx)?;
    let SimulationOutcome::Sample(samples) = outcome else {
        return Err(outcome_mismatch());
    };
    Ok((sample_parts(&samples), route))
}

#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment, outcome, active, control, *, seed=0, threads=None))]
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
    threads: Option<u32>,
) -> PyResult<GcmIteResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    let (mean_ite, n_units, noise_inference, n_assignments, unit_vec, route) =
        detach_catch(py, move || {
            let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
            let data = loaded.data;
            let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
            let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
            let source = model_source(data, &edges)?;
            let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
            let target =
                SimulationTarget::UnitEffects { treatment: t_id, outcome: y_id, active, control };
            let (outcome, route) = source.simulate(target, &ctx)?;
            let SimulationOutcome::UnitEffects(ite) = outcome else {
                return Err(outcome_mismatch());
            };
            let n_assignments =
                route.execution.result().simulation().map_or(0, |payload| payload.mechanisms.len());
            Ok::<_, PyErr>((
                ite.mean_ite,
                ite.unit_effects.len(),
                format!("{:?}", ite.noise_inference),
                n_assignments,
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

/// Fit GCM and return interventional column means + draws under `do(treatment=value)`
/// (hard set) or `do(treatment := treatment + value)` (additive shift, when `shift=True`).
///
/// `mechanism_wrappers` maps variable name → object with `sample_noise(n)` / `evaluate(parents, noise)`
/// ( slow path).
#[pyfunction]
#[pyo3(name = "sample_do", signature = (names, columns, edges, treatment, do_value, n_draws, *, seed=0, threads=None, mechanism_wrappers=None, shift=false))]
fn sample_do_py(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    treatment: String,
    do_value: f64,
    n_draws: usize,
    seed: u64,
    threads: Option<u32>,
    mechanism_wrappers: Option<Bound<'_, PyDict>>,
    shift: bool,
) -> PyResult<GcmSampleResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    let wrappers = mechanism_wrappers.map(Bound::unbind);
    let threads = if wrappers.is_some() { Some(1) } else { threads };
    let ((means, n_rows, n_nodes, flat), route) = detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let iv = if shift {
            Intervention::shift(t_id, Value::f64(do_value))
        } else {
            Intervention::set(t_id, Value::f64(do_value))
        };
        let Some(w) = wrappers else {
            let source = model_source(data, &edges)?;
            let (parts, route) = sample_route(&source, iv, n_draws, seed, &ctx)?;
            return Ok::<_, PyErr>((parts, Some(route)));
        };
        // Python mechanism wrappers replace fitted mechanisms with caller code. The wrapped
        // model runs through the compiled simulation route as a prefit the program names:
        // its label lists the wrapped variables, so the contract states that caller code, not
        // a fitted family, produced those mechanisms.
        let g = dag_from_named_edges(data.schema(), &edges)?;
        let fitted = fit_gcm(g.clone(), &data).map_err(py_err)?;
        let (model, wrapped) = Python::attach(|py| {
            let dict = w.bind(py);
            let wrapped = dict
                .keys()
                .iter()
                .map(|key| key.extract::<String>())
                .collect::<PyResult<Vec<String>>>()?;
            let model = callbacks::apply_mechanism_wrappers(&fitted.model, &names, dict)?;
            Ok::<_, PyErr>((model, wrapped))
        })?;
        let wrapped_names: Vec<&str> = wrapped.iter().map(String::as_str).collect();
        let wrapped_fit =
            Arc::new(antecedent::gcm::FittedGcm { model, assignments: fitted.assignments.clone() });
        let program =
            MechanismProgram::standard().with_wrapped_prefit(wrapped_fit, &data, &wrapped_names);
        let source = ModelSource { dag: g, data: Arc::new(data), accepted: None, program };
        let (parts, route) = sample_route(&source, iv, n_draws, seed, &ctx)?;
        Ok((parts, Some(route)))
    })?;
    let draws = PyArray1::from_vec(py, flat).reshape([n_nodes, n_rows])?.unbind();
    Ok(GcmSampleResult { column_means: means, n_draws: n_rows, n_nodes, draws, route })
}

/// Sample an interventional distribution via [`InterventionalDistributionQuery`].
///
/// Same return shape as [`gcm_sample_do`]; builds the typed query then samples.
/// `do_value` is a hard set (`do(treatment := do_value)`) unless `shift=True`, in
/// which case it is an additive delta (`do(treatment := treatment + do_value)`).
#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment, do_value, n_draws, outcome=None, *, seed=0, threads=None, shift=false))]
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
    threads: Option<u32>,
    shift: bool,
) -> PyResult<GcmSampleResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    let ((means, n_rows, n_nodes, flat), route) = detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_name = match &outcome {
            Some(o) => o.as_str(),
            None => names.last().map_or("y", String::as_str),
        };
        let y_id = data.schema().id_of(y_name).map_err(py_err)?;
        let iv = if shift {
            Intervention::shift(t_id, Value::f64(do_value))
        } else {
            Intervention::set(t_id, Value::f64(do_value))
        };
        // The interventional distribution of `y_id` is read off the interventional sample of
        // its interventions; the outcome only has to name a column.
        let _ = y_id;
        let source = model_source(data, &edges)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        sample_route(&source, iv, n_draws, seed, &ctx)
    })?;
    let draws = PyArray1::from_vec(py, flat).reshape([n_nodes, n_rows])?.unbind();
    let route = Some(route);
    Ok(GcmSampleResult { column_means: means, n_draws: n_rows, n_nodes, draws, route })
}

/// Path-specific contribution via [`PathSpecificEffectQuery`] / `path_decompose`.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment, outcome, *, path_nodes=None, max_paths=64, max_len=16, seed=0, threads=None))]
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
    threads: Option<u32>,
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
        let source = model_source(data, &edges)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
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
#[pyo3(signature = (names, columns, edges, outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=None))]
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
    threads: Option<u32>,
) -> PyResult<gcm_api::ChangeAttributionResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let source = model_source(data, &edges)?;
        let query = ChangeAttributionQuery::new(
            y_id,
            PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
            PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
        )
        .with_components(AttributionComponents::Mechanisms)
        .with_allocation(AllocationMethod::Shapley {
            approximation: ShapleyConfig::monte_carlo(n_samples).with_seed(seed),
        });
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
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

/// Structure-change attribution between two edge lists (parent-set Shapley).
#[pyfunction]
#[pyo3(signature = (names, columns, baseline_edges, comparison_edges, outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=None))]
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
    threads: Option<u32>,
) -> PyResult<gcm_api::ChangeAttributionResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let g1 = dag_from_named_edges(data.schema(), &comparison_edges)?;
        let comparison: Arc<[(VariableId, VariableId)]> = g1
            .directed_edge_pairs_sorted()
            .into_iter()
            .map(|(from, to)| (VariableId::from_raw(from), VariableId::from_raw(to)))
            .collect();
        let source = model_source(data, &baseline_edges)?;
        let query = ChangeAttributionQuery::new(
            y_id,
            PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
            PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
        )
        .with_components(AttributionComponents::Structure)
        .with_allocation(AllocationMethod::Shapley {
            approximation: ShapleyConfig::monte_carlo(n_samples).with_seed(seed),
        });
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
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
        return Err(crate::value_err("utility must be callable"));
    }
    let util = Arc::new(callbacks::PyUtility::new(utility.unbind()));
    let problem = DecisionProblem::new(actions, util, Vec::new());
    // Keep GIL acquired: utility callback reacquires anyway; this is an explicit slow path.
    let eval = facade_evaluate_decision(&problem, &outcomes).map_err(|err| match err {
        antecedent::design::DesignError::Callback { name, message } => {
            crate::py_err(antecedent::CausalError::Callback { name, message })
        }
        other => crate::py_err(antecedent::CausalError::from(other)),
    })?;
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
        let decoded = decode_causal_posterior_bytes(&bytes).or_else(|posterior_error| {
            let (_, _, composite) = antecedent_io::decode_analysis_result_artifact(&bytes)
                .map_err(|_| posterior_error)?;
            let nested = composite.posterior_artifact.ok_or_else(|| {
                antecedent_io::IoError::Convert(
                    "analysis_result artifact has no posterior axis".into(),
                )
            })?;
            decode_causal_posterior_bytes(&nested)
        });
        let (meta, draws) = decoded.map_err(py_err)?;
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
            subsampled_out_mass: meta.subsampled_out_mass,
            converged: meta.converged,
            hessian_condition: meta.hessian_condition,
            quantity_names: meta.quantities.iter().map(quantity_wire_name).collect(),
            treatment_contrast: meta.treatment_contrast,
            quantity_schema: Some(meta.quantities),
        })
    })
}

/// Re-encode a decoded [`PosteriorArtifact`] to container bytes (round-trip).
#[pyfunction]
fn encode_posterior_artifact(artifact: &PosteriorArtifact) -> PyResult<Vec<u8>> {
    catch_ffi(|| {
        let quantities: Vec<PosteriorQuantityWire> =
            artifact.quantity_schema.clone().unwrap_or_else(|| {
                artifact
                    .quantity_names
                    .iter()
                    .map(|name| {
                        if name == "residual_variance" {
                            PosteriorQuantityWire::ResidualVariance
                        } else if name.starts_with("coef_") {
                            let index = name
                                .strip_prefix("coef_")
                                .and_then(|s| s.parse::<u32>().ok())
                                .unwrap_or(0);
                            PosteriorQuantityWire::Coefficient { index, name: None }
                        } else {
                            PosteriorQuantityWire::Effect { name: name.clone() }
                        }
                    })
                    .collect()
            });
        // Artifacts built via `PosteriorArtifact.from_moments` (no draws) round-trip as
        // summary-only: an empty draws payload signals `draws_encoding = "none"` rather
        // than a (nonsensical) zero-length full-draws section.
        let draws_encoding = if artifact.draws.is_empty() { "none" } else { "f64_le_colmajor" };
        let meta = CausalPosteriorWire {
            quantities,
            n_draws: u32::try_from(artifact.n_draws)
                .map_err(|_| crate::value_err("n_draws exceeds u32"))?,
            mean: artifact.mean.clone(),
            sd: artifact.sd.clone(),
            q025: artifact.q025.clone(),
            q975: artifact.q975.clone(),
            identification: artifact.identification.clone(),
            unidentified_mass: artifact.unidentified_mass,
            subsampled_out_mass: artifact.subsampled_out_mass,
            backend_id: artifact.backend_id.clone(),
            converged: artifact.converged,
            hessian_condition: artifact.hessian_condition,
            draws_encoding: draws_encoding.into(),
            treatment_contrast: artifact.treatment_contrast,
            preparation: None,
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
        let outcome_ids: Vec<VariableId> = outcomes
            .iter()
            .map(|n| data.schema().id_of(n).map_err(py_err))
            .collect::<PyResult<_>>()?;
        let max_u = if max_units == 0 { data.row_count() } else { max_units };
        let source = model_source(data, &edges)?;
        let ctx = py_execution_context(0, crate::resolve_user_threads(None));
        let target = AttributeTarget::Anomaly(AnomalyAttributionQuery::new(outcome_ids, max_u));
        let (outcome, _route) = source.attribute(target, &ctx)?;
        let AttributionOutcome::Anomaly(scores) = outcome else {
            return Err(outcome_mismatch());
        };
        Ok(gcm_api::anomaly_scores_from_rust(scores, &names))
    })
}

/// Unit-level change attribution.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, outcome, *, max_units=0, seed=0, threads=None))]
fn attribute_unit_change(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    outcome: String,
    max_units: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<gcm_api::ChangeAttributionResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let max_u = if max_units == 0 { data.row_count() } else { max_units };
        let source = model_source(data, &edges)?;
        let target = AttributeTarget::UnitChange(UnitChangeQuery::new(y_id, max_u));
        let (outcome, route) = source.attribute(target, &ctx)?;
        let AttributionOutcome::UnitChange(result) = outcome else {
            return Err(outcome_mismatch());
        };
        let pairs: Vec<(antecedent_core::ComponentId, f64)> = result
            .components
            .iter()
            .zip(result.mean_contributions.iter())
            .map(|(c, v)| (*c, *v))
            .collect();
        let total = result.mean_contributions.iter().map(|x| x.abs()).sum();
        Ok(gcm_api::synthetic_change_result(y_id, total, pairs, &names).with_route(route))
    })
}

/// Feature relevance scores for parents of `outcome`.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, outcome, *, delta=1.0, n_samples=200, seed=0, threads=None))]
fn attribute_feature_relevance(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, String)>,
    outcome: String,
    delta: f64,
    n_samples: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<Vec<gcm_api::FeatureRelevance>> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let mut features = Vec::new();
        for i in 0..data.schema().len() {
            let id = VariableId::from_raw(crate::fit_u32(i)?);
            if id != y_id {
                features.push(id);
            }
        }
        let source = model_source(data, &edges)?;
        let max_features = features.len();
        let target = AttributeTarget::FeatureRelevance {
            outcome: y_id,
            features: Arc::from(features),
            delta,
            n_samples,
            max_features,
        };
        let (outcome, _route) = source.attribute(target, &ctx)?;
        let AttributionOutcome::FeatureRelevance(scores) = outcome else {
            return Err(outcome_mismatch());
        };
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
#[pyo3(signature = (names, columns, edges, outcome, baseline_start, baseline_end, comparison_start, comparison_end, *, n_samples=500, seed=0, threads=None))]
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
    threads: Option<u32>,
) -> PyResult<gcm_api::ChangeAttributionResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let source = model_source(data, &edges)?;
        let query = ChangeAttributionQuery {
            outcome: y_id,
            baseline: PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
            comparison: PopulationSelector::TimeRange {
                start: comparison_start,
                end: comparison_end,
            },
            components: AttributionComponents::Mechanisms,
            allocation: AllocationMethod::Shapley {
                approximation: ShapleyConfig::monte_carlo(n_samples).with_seed(seed),
            },
            max_components: 64,
        };
        // `RobustChangeOptions` only exposes `max_rows`; it has no `n_samples`/`seed`
        // fields to thread through (unlike `DistributionChangeOptions`), so the fix
        // above (Shapley `ShapleyConfig`) is the only place these two params can flow.
        let opts = antecedent::gcm::RobustChangeOptions::default();
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
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

/// Detect mechanism changes across two row ranges.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, baseline_start, baseline_end, comparison_start, comparison_end, *, seed=0, threads=None))]
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
    threads: Option<u32>,
) -> PyResult<Vec<gcm_api::MechanismChangeDetection>> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
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
        let source = model_source(data, &edges)?;
        let target =
            AttributeTarget::MechanismChange { query, test: MechanismChangeTest::MeanDifference };
        let (outcome, _route) = source.attribute(target, &ctx)?;
        let AttributionOutcome::MechanismChange(detected) = outcome else {
            return Err(outcome_mismatch());
        };
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
                    adjusted_p_value: d.adjusted_p_value,
                    changed: d.changed,
                    n_permutations: d.n_permutations,
                    p_value_floor: d.p_value_floor,
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
