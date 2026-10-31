//! Capability module extracted from `lib.rs` (SOLID/SRP cleanup).
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::wildcard_imports,
    clippy::empty_line_after_doc_comments
)]

use crate::NumericColumn;
use crate::compiled_route_api::{ModelSource, RouteRecord, outcome_mismatch};
use crate::*;
use antecedent::gcm::NoiseInferenceKind;
use antecedent::{AttributionOutcome, MechanismProgram, SimulationOutcome};
use antecedent_core::{AnomalyAttributionQuery, AttributeTarget, ChangeEstimator, ChangeMeasure};
use antecedent_core::{CausalQuery, MechanismChangeTest, SimulationTarget};
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

/// The table, graph and run settings every fitted-model entry point takes.
///
/// The NumPy columns are ingested once here and their borrows released before the GIL is, so
/// the entry point holds one value instead of repeating the `names/columns/edges/seed/threads`
/// plumbing: variable lookup, the compiled-route source and the execution context all come
/// from it.
struct GcmInput {
    data: antecedent_data::TabularData,
    edges: Vec<(String, String)>,
    seed: u64,
    threads: Option<u32>,
}

impl GcmInput {
    /// Ingest `columns` (consumed, so their NumPy borrows end here) under `names`.
    fn load(
        names: &[String],
        columns: Vec<NumericColumn<'_>>,
        edges: Vec<(String, String)>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<Self> {
        let loaded = tabular_from_numpy_receipt(names, &columns)?;
        drop(columns);
        Ok(Self { data: loaded.data, edges, seed, threads })
    }

    /// The variable named `name` in the table.
    fn id(&self, name: &str) -> PyResult<VariableId> {
        self.data.schema().id_of(name).map_err(py_err)
    }

    /// The execution context of this run.
    fn context(&self) -> ExecutionContext {
        py_execution_context(self.seed, crate::resolve_user_threads(self.threads))
    }

    /// The compiled-route source over the table and graph, with the run's context.
    fn into_source(self) -> PyResult<(ModelSource, ExecutionContext)> {
        let ctx = self.context();
        Ok((model_source(self.data, &self.edges)?, ctx))
    }
}

/// The `[n_nodes, n_rows]` draw matrix of a flat column-major draw buffer.
pub(crate) fn reshape_draws(
    py: Python<'_>,
    flat: Vec<f64>,
    n_nodes: usize,
    n_rows: usize,
) -> PyResult<Py<numpy::PyArray2<f64>>> {
    Ok(PyArray1::from_vec(py, flat).reshape([n_nodes, n_rows])?.unbind())
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

/// Column means of a draw batch, its row and column counts and its flat values.
type SampleParts = (Vec<f64>, usize, usize, Vec<f64>);

/// Column means of a draw batch, its shape and its flat values.
fn sample_parts(samples: &antecedent_model::ValueBatch) -> SampleParts {
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
) -> PyResult<(SampleParts, RouteRecord)> {
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
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    active: f64,
    control: f64,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<GcmIteResult> {
    let input = GcmInput::load(&names, columns, edges, seed, threads)?;
    let (mean_ite, n_units, noise_inference, n_assignments, unit_vec, route) =
        detach_catch(py, move || {
            let t_id = input.id(&treatment)?;
            let y_id = input.id(&outcome)?;
            let (source, ctx) = input.into_source()?;
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
                noise_inference_label(ite.noise_inference),
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
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    treatment: String,
    do_value: f64,
    n_draws: usize,
    seed: u64,
    threads: Option<u32>,
    mechanism_wrappers: Option<Bound<'_, PyDict>>,
    shift: bool,
) -> PyResult<GcmSampleResult> {
    let wrappers = mechanism_wrappers.map(Bound::unbind);
    let threads = if wrappers.is_some() { Some(1) } else { threads };
    let input = GcmInput::load(&names, columns, edges, seed, threads)?;
    let ((means, n_rows, n_nodes, flat), route) = detach_catch(py, move || {
        let t_id = input.id(&treatment)?;
        let ctx = input.context();
        let iv = if shift {
            Intervention::shift(t_id, Value::f64(do_value))
        } else {
            Intervention::set(t_id, Value::f64(do_value))
        };
        let Some(w) = wrappers else {
            let (source, _) = input.into_source()?;
            let (parts, route) = sample_route(&source, iv, n_draws, seed, &ctx)?;
            return Ok::<_, PyErr>((parts, Some(route)));
        };
        let GcmInput { data, edges, .. } = input;
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
    let draws = reshape_draws(py, flat, n_nodes, n_rows)?;
    Ok(GcmSampleResult { column_means: means, n_draws: n_rows, n_nodes, draws, route })
}

/// Sample an interventional distribution via [`InterventionalDistributionQuery`].
///
/// Same return shape as [`gcm_sample_do`]; builds the typed query then samples.
/// `do_value` is a hard set (`do(treatment := do_value)`) unless `shift=True`, in
/// which case it is an additive delta (`do(treatment := treatment + do_value)`).
/// Stable label of how the noise of an individual treatment effect was obtained. An explicit
/// mapping, so renaming the Rust variant never changes the string Python callers see.
fn noise_inference_label(kind: NoiseInferenceKind) -> String {
    match kind {
        NoiseInferenceKind::Invertible => "Invertible",
        NoiseInferenceKind::PosteriorNoise => "PosteriorNoise",
        NoiseInferenceKind::AssumedNoise => "AssumedNoise",
    }
    .to_owned()
}

#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment, do_value, n_draws, outcome=None, *, seed=0, threads=None, shift=false))]
fn sample_interventional_distribution(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    treatment: String,
    do_value: f64,
    n_draws: usize,
    outcome: Option<String>,
    seed: u64,
    threads: Option<u32>,
    shift: bool,
) -> PyResult<GcmSampleResult> {
    let input = GcmInput::load(&names, columns, edges, seed, threads)?;
    let ((means, n_rows, n_nodes, flat), route) = detach_catch(py, move || {
        let t_id = input.id(&treatment)?;
        let y_name = match &outcome {
            Some(o) => o.as_str(),
            None => names.last().map_or("y", String::as_str),
        };
        // The interventional distribution of the outcome is read off the interventional sample
        // of its interventions; the outcome only has to name a column of the table.
        input.id(y_name)?;
        let iv = if shift {
            Intervention::shift(t_id, Value::f64(do_value))
        } else {
            Intervention::set(t_id, Value::f64(do_value))
        };
        let (source, ctx) = input.into_source()?;
        sample_route(&source, iv, n_draws, seed, &ctx)
    })?;
    let draws = reshape_draws(py, flat, n_nodes, n_rows)?;
    let route = Some(route);
    Ok(GcmSampleResult { column_means: means, n_draws: n_rows, n_nodes, draws, route })
}

/// Path-specific contribution via [`PathSpecificEffectQuery`] / `path_decompose`.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, treatment, outcome, *, path_nodes=None, max_paths=64, max_len=16, seed=0, threads=None))]
fn attribute_path_specific(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    path_nodes: Option<Vec<String>>,
    max_paths: usize,
    max_len: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<gcm_api::ChangeAttributionResult> {
    let input = GcmInput::load(&names, columns, edges, seed, threads)?;
    detach_catch(py, move || {
        let t_id = input.id(&treatment)?;
        let y_id = input.id(&outcome)?;
        let mut intermediates = Vec::new();
        if let Some(nodes) = &path_nodes {
            for n in nodes {
                intermediates.push(input.id(n)?);
            }
        }
        let (source, ctx) = input.into_source()?;
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
    columns: Vec<NumericColumn<'_>>,
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
    gcm_api::require_samples(n_samples)?;
    let input = GcmInput::load(&names, columns, edges, seed, threads)?;
    detach_catch(py, move || {
        let y_id = input.id(&outcome)?;
        let (source, ctx) = input.into_source()?;
        let query = ChangeAttributionQuery::new(
            y_id,
            PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
            PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
        )
        .with_components(AttributionComponents::Mechanisms)
        .with_allocation(AllocationMethod::Shapley {
            approximation: ShapleyConfig::monte_carlo(n_samples).with_seed(seed),
        });
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
    columns: Vec<NumericColumn<'_>>,
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
    gcm_api::require_samples(n_samples)?;
    let input = GcmInput::load(&names, columns, baseline_edges, seed, threads)?;
    detach_catch(py, move || {
        let y_id = input.id(&outcome)?;
        let g1 = dag_from_named_edges(input.data.schema(), &comparison_edges)?;
        let comparison: Arc<[(VariableId, VariableId)]> = g1
            .directed_edge_pairs_sorted()
            .into_iter()
            .map(|(from, to)| (VariableId::from_raw(from), VariableId::from_raw(to)))
            .collect();
        let (source, ctx) = input.into_source()?;
        let query = ChangeAttributionQuery::new(
            y_id,
            PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
            PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
        )
        .with_components(AttributionComponents::Structure)
        .with_allocation(AllocationMethod::Shapley {
            approximation: ShapleyConfig::monte_carlo(n_samples).with_seed(seed),
        });
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

/// A quantity name such as `component[3]` or `path[0>3]` with its variable numbers replaced by
/// the variable names.
fn named_quantity(quantity: &str, names: &[String]) -> String {
    let Some((head, rest)) = quantity.split_once('[') else {
        return quantity.to_owned();
    };
    let inner = rest.trim_end_matches(']');
    let named: Vec<String> = inner
        .split('>')
        .map(|part| {
            part.parse::<usize>()
                .ok()
                .and_then(|index| names.get(index).cloned())
                .unwrap_or_else(|| part.to_owned())
        })
        .collect();
    format!("{head}[{}]", named.join(">"))
}

fn json_str(params: &serde_json::Value, key: &str) -> PyResult<String> {
    params[key]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| crate::value_err(format!("attribution parameter {key:?} is required")))
}

/// A count parameter; absent or null takes `default`, any other non-count is an error.
fn json_usize(params: &serde_json::Value, key: &str, default: usize) -> PyResult<usize> {
    match &params[key] {
        serde_json::Value::Null => Ok(default),
        value => value.as_u64().and_then(|v| usize::try_from(v).ok()).ok_or_else(|| {
            crate::value_err(format!(
                "attribution parameter {key:?} must be a non-negative integer"
            ))
        }),
    }
}

/// A list of names; absent or null is empty, any other shape is an error.
fn json_names(params: &serde_json::Value, key: &str) -> PyResult<Vec<String>> {
    match &params[key] {
        serde_json::Value::Null => Ok(Vec::new()),
        serde_json::Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str().map(str::to_owned).ok_or_else(|| {
                    crate::value_err(format!("attribution parameter {key:?} must list names"))
                })
            })
            .collect(),
        _ => Err(crate::value_err(format!("attribution parameter {key:?} must be a list"))),
    }
}

/// A finite number parameter; absent or null takes `default`, any other non-number is an error.
fn json_f64(params: &serde_json::Value, key: &str, default: f64) -> PyResult<f64> {
    Ok(json_opt_f64(params, key)?.unwrap_or(default))
}

fn json_opt_f64(params: &serde_json::Value, key: &str) -> PyResult<Option<f64>> {
    match &params[key] {
        serde_json::Value::Null => Ok(None),
        value => match value.as_f64() {
            Some(number) if number.is_finite() => Ok(Some(number)),
            _ => Err(crate::value_err(format!(
                "attribution parameter {key:?} must be a finite number"
            ))),
        },
    }
}

/// A step index; absent or null takes `default`, any other non-integer is an error.
fn json_i32(params: &serde_json::Value, key: &str, default: i32) -> PyResult<i32> {
    match &params[key] {
        serde_json::Value::Null => Ok(default),
        value => value.as_i64().and_then(|v| i32::try_from(v).ok()).ok_or_else(|| {
            crate::value_err(format!("attribution parameter {key:?} must be an integer step"))
        }),
    }
}

/// Edge pairs of names; a pair that is not two names is an error rather than dropped.
fn json_edge_pairs(params: &serde_json::Value, key: &str) -> PyResult<Vec<(String, String)>> {
    match &params[key] {
        serde_json::Value::Null => Ok(Vec::new()),
        serde_json::Value::Array(pairs) => pairs
            .iter()
            .map(|pair| match pair.as_array().map(Vec::as_slice) {
                Some([serde_json::Value::String(from), serde_json::Value::String(to)]) => {
                    Ok((from.clone(), to.clone()))
                }
                _ => Err(crate::value_err(format!(
                    "attribution parameter {key:?} holds an edge that is not a [from, to] pair"
                ))),
            })
            .collect(),
        _ => Err(crate::value_err(format!("attribution parameter {key:?} must be a list"))),
    }
}

/// The population change query of an attribution request.
fn change_query(
    params: &serde_json::Value,
    schema: &antecedent_core::CausalSchema,
    components: AttributionComponents,
    seed: u64,
) -> PyResult<(ChangeAttributionQuery, usize)> {
    let outcome = schema.id_of(&json_str(params, "outcome")?).map_err(py_err)?;
    let n_samples = json_usize(params, "n_samples", 500)?;
    let query = ChangeAttributionQuery::new(
        outcome,
        PopulationSelector::TimeRange {
            start: json_usize(params, "baseline_start", 0)?,
            end: json_usize(params, "baseline_end", 0)?,
        },
        PopulationSelector::TimeRange {
            start: json_usize(params, "comparison_start", 0)?,
            end: json_usize(params, "comparison_end", 0)?,
        },
    )
    .with_components(components)
    .with_allocation(AllocationMethod::Shapley {
        approximation: ShapleyConfig::monte_carlo(n_samples).with_seed(seed),
    });
    Ok((query, n_samples))
}

/// Run a fitted-model attribution through the compiled route under a posterior over the
/// mechanisms (`posterior_draws` Dirichlet-weighted refits) or the mechanism-refit row
/// bootstrap (`bootstrap_replicates`), and return its quantities as JSON.
///
/// `kind` is one of `distribution_change`, `robust_change`, `root_causes`, `structure_change`,
/// `paths`, `feature_relevance`, `anomaly` and `unit_change`; `params_json` carries the
/// parameters of that kind. Mechanism-change detection has no posterior over the mechanisms (a
/// two-sample test, not a smooth functional of the refit) and is refused.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, kind, params_json, *, posterior_draws=0, bootstrap_replicates=0, seed=0, threads=None))]
fn attribution_uncertainty(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    kind: String,
    params_json: String,
    posterior_draws: usize,
    bootstrap_replicates: u32,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    if (posterior_draws > 0) == (bootstrap_replicates > 0) {
        return Err(crate::value_err(
            "ask for exactly one of posterior_draws (Bayesian) and bootstrap_replicates \
             (Frequentist)",
        ));
    }
    let params: serde_json::Value = serde_json::from_str(&params_json)
        .map_err(|error| crate::value_err(format!("attribution parameters: {error}")))?;
    let input = GcmInput::load(&names, columns, edges, seed, threads)?;
    detach_catch(py, move || {
        let schema = input.data.schema().clone();
        let ids = |list: &[String]| -> PyResult<Vec<VariableId>> {
            list.iter().map(|name| schema.id_of(name).map_err(py_err)).collect()
        };
        let target = match kind.as_str() {
            "distribution_change" | "root_causes" | "robust_change" => {
                let (change, n_samples) =
                    change_query(&params, &schema, AttributionComponents::Mechanisms, seed)?;
                let estimator = if kind == "robust_change" {
                    ChangeEstimator::Robust { max_rows: 10_000, n_samples: 4_000, seed }
                } else {
                    ChangeEstimator::Sampled {
                        measure: ChangeMeasure::MeanDifference,
                        n_samples: n_samples.max(100),
                        seed,
                    }
                };
                if kind == "root_causes" {
                    AttributeTarget::RootCauses { change, estimator }
                } else {
                    AttributeTarget::DistributionChange { change, estimator }
                }
            }
            "structure_change" => {
                let (change, n_samples) =
                    change_query(&params, &schema, AttributionComponents::Structure, seed)?;
                let comparison_edges = json_edge_pairs(&params, "comparison_edges")?;
                let g1 = dag_from_named_edges(&schema, &comparison_edges)?;
                let comparison: Arc<[(VariableId, VariableId)]> = g1
                    .directed_edge_pairs_sorted()
                    .into_iter()
                    .map(|(from, to)| (VariableId::from_raw(from), VariableId::from_raw(to)))
                    .collect();
                AttributeTarget::StructureChange {
                    change,
                    comparison,
                    measure: ChangeMeasure::MeanDifference,
                    n_samples: n_samples.max(100),
                    seed,
                }
            }
            "paths" => AttributeTarget::Paths {
                sources: Arc::from(ids(&json_names(&params, "sources")?)?),
                outcome: schema.id_of(&json_str(&params, "outcome")?).map_err(py_err)?,
                max_paths: json_usize(&params, "max_paths", 64)?,
                max_len: json_usize(&params, "max_len", 16)?,
                through: Arc::from(ids(&json_names(&params, "through")?)?),
            },
            "feature_relevance" => {
                let outcome = schema.id_of(&json_str(&params, "outcome")?).map_err(py_err)?;
                let features: Vec<VariableId> = (0..schema.len())
                    .map(|i| crate::fit_u32(i).map(VariableId::from_raw))
                    .collect::<PyResult<Vec<_>>>()?
                    .into_iter()
                    .filter(|id| *id != outcome)
                    .collect();
                let max_features = features.len();
                AttributeTarget::FeatureRelevance {
                    outcome,
                    features: Arc::from(features),
                    delta: json_f64(&params, "delta", 1.0)?,
                    n_samples: json_usize(&params, "n_samples", 200)?,
                    max_features,
                }
            }
            "anomaly" => {
                let targets = ids(&json_names(&params, "outcomes")?)?;
                let max_units = match json_usize(&params, "max_units", 0)? {
                    0 => input.data.row_count(),
                    given => given,
                };
                AttributeTarget::Anomaly(AnomalyAttributionQuery::new(targets, max_units))
            }
            "unit_change" => {
                let outcome = schema.id_of(&json_str(&params, "outcome")?).map_err(py_err)?;
                let max_units = match json_usize(&params, "max_units", 0)? {
                    0 => input.data.row_count(),
                    given => given,
                };
                AttributeTarget::UnitChange(UnitChangeQuery::new(outcome, max_units))
            }
            other => {
                return Err(crate::value_err(format!(
                    "unknown attribution kind {other:?}; mechanism-change detection has no \
                     posterior or refit bootstrap (a test is not a smooth functional of the \
                     refitted mechanisms)"
                )));
            }
        };
        let (source, ctx) = input.into_source()?;
        let inference = if posterior_draws > 0 {
            antecedent::InferenceMode::Bayesian(
                antecedent::BayesianConfig::conjugate().n_draws(posterior_draws),
            )
        } else {
            antecedent::InferenceMode::Frequentist
        };
        let (outcome, route) =
            source.attribute_resampled(target, inference, bootstrap_replicates, &ctx)?;
        let point = antecedent::analysis::structure_route::attribution_quantities(&outcome)
            .unwrap_or_default();
        let result = route.execution.result();
        let posterior = result.posterior.as_ref().map(|posterior| {
            let summary = &posterior.summaries;
            let quantities: Vec<String> =
                point.iter().map(|(name, _)| named_quantity(name, &names)).collect();
            serde_json::json!({
                "draws": summary.n_draws,
                "level": 0.95,
                "names": quantities,
                "mean": summary.mean.to_vec(),
                "sd": summary.sd.to_vec(),
                "ci_low": summary.q025.to_vec(),
                "ci_high": summary.q975.to_vec(),
            })
        });
        let bootstrap =
            result.diagnostics.iter().find(|d| d.code.as_ref() == "gcm.frequentist_bootstrap").map(
                |d| {
                    let field = |key: &str| {
                        d.fields
                            .iter()
                            .find(|(k, _)| k.as_ref() == key)
                            .map(|(_, v)| v.to_string())
                            .unwrap_or_default()
                    };
                    let se: serde_json::Map<String, serde_json::Value> = d
                        .fields
                        .iter()
                        .filter_map(|(k, v)| {
                            let name = k.strip_prefix("bootstrap_se.")?;
                            Some((
                                named_quantity(name, &names),
                                v.parse::<f64>()
                                    .map_or(serde_json::Value::Null, serde_json::Value::from),
                            ))
                        })
                        .collect();
                    serde_json::json!({
                        "replicates_ok": field("replicates_ok").parse::<u32>().unwrap_or(0),
                        "replicates_failed": field("replicates_failed").parse::<u32>().unwrap_or(0),
                        "se": se,
                    })
                },
            );
        let point_json: Vec<serde_json::Value> = point
            .iter()
            .map(|(name, value)| serde_json::json!({ "name": named_quantity(name, &names), "value": value }))
            .collect();
        Ok(serde_json::json!({
            "kind": kind,
            "point": point_json,
            "posterior": posterior,
            "bootstrap": bootstrap,
        })
        .to_string())
    })
}

fn mechanism_test(name: &str) -> PyResult<MechanismChangeTest> {
    Ok(match name {
        "likelihood_ratio" => MechanismChangeTest::LikelihoodRatio,
        "mean_difference" => MechanismChangeTest::MeanDifference,
        "classifier_two_sample" => MechanismChangeTest::ClassifierTwoSample,
        "kernel_two_sample" => MechanismChangeTest::KernelTwoSample,
        "change_point" => MechanismChangeTest::ChangePoint,
        other => {
            return Err(crate::value_err(format!(
                "test must be likelihood_ratio, mean_difference, classifier_two_sample, \
                 kernel_two_sample or change_point, got {other:?}"
            )));
        }
    })
}

/// Run a standalone query on a fitted `TemporalDag` through the compiled route and return its
/// answer as JSON.
///
/// `kind = "mechanism_change"` tests whether the mechanism of each target changed between two
/// row ranges of one series: the null law is a circular-block permutation of the residuals (the
/// likelihood-ratio, mean-difference, kernel and classifier statistics), and with
/// `posterior_draws > 0` (Bayesian inference) the answer is the posterior probability of a change
/// under the `change_prior` the parameters declare (never assumed).
///
/// `kind = "rollout"` rolls the fitted dynamics forward from the last rows of the series under
/// the scheduled `interventions`, and returns the per-step means of every variable; with
/// `posterior_draws` or `bootstrap_replicates` also the posterior or bootstrap of those means over
/// block refits of the dynamics.
#[pyfunction]
#[pyo3(signature = (names, columns, lagged_edges, kind, params_json, *, posterior_draws=0, bootstrap_replicates=0, seed=0, threads=None))]
fn temporal_model_query(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    lagged_edges: Vec<(String, u32, String, u32)>,
    kind: String,
    params_json: String,
    posterior_draws: usize,
    bootstrap_replicates: u32,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<String> {
    if posterior_draws > 0 && bootstrap_replicates > 0 {
        return Err(crate::value_err("ask for posterior_draws or bootstrap_replicates, not both"));
    }
    let params: serde_json::Value = serde_json::from_str(&params_json)
        .map_err(|error| crate::value_err(format!("query parameters: {error}")))?;
    let batch = tabular_from_numpy_receipt(&names, &columns)?;
    drop(columns);
    detach_catch(py, move || {
        let loaded = batch;
        let schema = loaded.data.schema().clone();
        let series = series_from_tabular(loaded.data)?;
        let graph = temporal_dag_from_schema_edges(&schema, &lagged_edges)?;
        let ctx = py_execution_context(seed, crate::resolve_user_threads(threads));
        let inference = if posterior_draws > 0 {
            antecedent::InferenceMode::Bayesian(
                antecedent::BayesianConfig::conjugate().n_draws(posterior_draws),
            )
        } else {
            antecedent::InferenceMode::Frequentist
        };
        let variables = schema.len();
        let query = match kind.as_str() {
            "mechanism_change" => {
                let targets: Vec<VariableId> = json_names(&params, "targets")?
                    .iter()
                    .map(|name| schema.id_of(name).map_err(py_err))
                    .collect::<PyResult<_>>()?;
                let mut detection = MechanismChangeQuery::new(
                    targets,
                    PopulationSelector::TimeRange {
                        start: json_usize(&params, "baseline_start", 0)?,
                        end: json_usize(&params, "baseline_end", 0)?,
                    },
                    PopulationSelector::TimeRange {
                        start: json_usize(&params, "comparison_start", 0)?,
                        end: json_usize(&params, "comparison_end", 0)?,
                    },
                    json_f64(&params, "significance", 0.05)?,
                    json_usize(&params, "max_targets", 16)?,
                );
                if let Some(prior) = json_opt_f64(&params, "change_prior")? {
                    detection = detection.with_change_prior(prior);
                }
                let test = mechanism_test(match &params["test"] {
                    serde_json::Value::Null => "likelihood_ratio",
                    serde_json::Value::String(name) => name.as_str(),
                    _ => {
                        return Err(crate::value_err(
                            "attribution parameter \"test\" must be a name",
                        ));
                    }
                })?;
                CausalQuery::Attribute(antecedent_core::AttributeQuery::new(
                    AttributeTarget::MechanismChange { query: detection, test },
                ))
            }
            "rollout" => {
                let steps: Vec<antecedent_core::SequencedIntervention> = params["interventions"]
                    .as_array()
                    .map(|items| {
                        items
                            .iter()
                            .map(|item| {
                                let variable =
                                    schema.id_of(&json_str(item, "variable")?).map_err(py_err)?;
                                let from = json_i32(item, "from", 0)?;
                                let until = json_i32(item, "until", from)?;
                                let value = json_opt_f64(item, "value")?.ok_or_else(|| {
                                    crate::value_err("a rollout intervention needs its value")
                                })?;
                                Ok(antecedent_core::SequencedIntervention::new(
                                    Intervention::set(variable, Value::f64(value)),
                                    TemporalPolicy::Sustained { from, until },
                                ))
                            })
                            .collect::<PyResult<Vec<_>>>()
                    })
                    .transpose()?
                    .unwrap_or_default();
                CausalQuery::Simulate(antecedent_core::SimulateQuery::new(
                    antecedent_core::SimulationTarget::TemporalRollout {
                        schedule: antecedent_core::InterventionSequence::new(steps),
                        n_rows: json_usize(&params, "n_rows", 200)?,
                        n_steps: json_usize(&params, "n_steps", 8)?,
                        window: 0,
                        history: Arc::from([]),
                        seed: params["seed"].as_u64().unwrap_or(seed),
                    },
                ))
            }
            other => {
                return Err(crate::value_err(format!(
                    "unknown temporal query {other:?}; use mechanism_change or rollout"
                )));
            }
        };
        let study = Study::series(series.clone())
            .graph(graph)
            .query(query)
            .inference(inference)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(bootstrap_replicates)
            .build()
            .map_err(py_err)?;
        let result =
            study.prepare(&ctx).map_err(py_err)?.estimate_series(&series, &ctx).map_err(py_err)?;
        let body = if kind == "mechanism_change" {
            {
                let Some(AttributionOutcome::MechanismChange(found)) =
                    result.attribution().map(|payload| payload.outcome.clone())
                else {
                    return Err(outcome_mismatch());
                };
                let detections: Vec<serde_json::Value> = found
                    .iter()
                    .map(|d| {
                        serde_json::json!({
                            "variable": names.get(d.variable.as_usize()).cloned()
                                .unwrap_or_else(|| format!("V{}", d.variable.raw())),
                            "changed": d.changed,
                            "statistic": d.statistic,
                            "p_value": if d.p_value.is_nan() { serde_json::Value::Null } else { d.p_value.into() },
                            "adjusted_p_value": d.adjusted_p_value,
                            "method": d.method.as_ref(),
                            "n_permutations": d.n_permutations,
                            "posterior_probability": d.posterior_probability,
                        })
                    })
                    .collect();
                serde_json::json!({ "kind": kind, "detections": detections })
            }
        } else {
            {
                let Some(payload) = result.simulation() else {
                    return Err(outcome_mismatch());
                };
                let antecedent::SimulationOutcome::Temporal(path) = &payload.outcome else {
                    return Err(outcome_mismatch());
                };
                let steps: Vec<Vec<f64>> = path
                    .steps
                    .iter()
                    .map(|batch| {
                        (0..variables.min(path.n_nodes))
                            .map(|node| {
                                let column = batch.column(node).unwrap_or(&[]);
                                column.iter().sum::<f64>() / column.len().max(1) as f64
                            })
                            .collect()
                    })
                    .collect();
                let label = |step: usize, node: usize| {
                    format!("step[{step}].{}", names.get(node).map_or("?", String::as_str))
                };
                let n_steps = steps.len();
                let mut quantities: Vec<String> = (0..n_steps)
                    .flat_map(|step| (0..variables).map(move |node| (step, node)))
                    .map(|(step, node)| label(step, node))
                    .collect();
                // After the means come the simulated-path columns: draws of the predictive law.
                let path_labels: Vec<String> = (0..n_steps)
                    .flat_map(|step| (0..variables).map(move |node| (step, node)))
                    .map(|(step, node)| format!("path.{}", label(step, node)))
                    .collect();
                quantities.extend(path_labels);
                let posterior = result.posterior.as_ref().map(|posterior| {
                    let summary = &posterior.summaries;
                    serde_json::json!({
                        "draws": summary.n_draws,
                        "level": 0.95,
                        "names": quantities,
                        "mean": summary.mean.to_vec(),
                        "sd": summary.sd.to_vec(),
                        "ci_low": summary.q025.to_vec(),
                        "ci_high": summary.q975.to_vec(),
                    })
                });
                let bootstrap = result
                    .diagnostics
                    .iter()
                    .find(|d| d.code.as_ref() == "gcm.frequentist_bootstrap")
                    .map(|d| {
                        let se: serde_json::Map<String, serde_json::Value> = d
                            .fields
                            .iter()
                            .filter_map(|(k, v)| {
                                let rest = k.strip_prefix("bootstrap_se.step[")?;
                                let (step, node, is_path) =
                                    if let Some((step, node)) = rest.split_once("].node[") {
                                        (step, node, false)
                                    } else {
                                        let (step, node) = rest.split_once("].path[")?;
                                        (step, node, true)
                                    };
                                let node: usize = node.trim_end_matches(']').parse().ok()?;
                                let name = label(step.parse().ok()?, node);
                                Some((
                                    if is_path { format!("path.{name}") } else { name },
                                    v.parse::<f64>()
                                        .map_or(serde_json::Value::Null, serde_json::Value::from),
                                ))
                            })
                            .collect();
                        serde_json::json!({ "se": se })
                    });
                serde_json::json!({
                    "kind": kind,
                    "variables": names,
                    "steps": steps,
                    "posterior": posterior,
                    "bootstrap": bootstrap,
                })
            }
        };
        Ok(body.to_string())
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
    let eval = detach_catch(py, move || {
        facade_evaluate_decision(&problem, &outcomes).map_err(|err| match err {
            antecedent::design::DesignError::Callback { name, message } => {
                crate::py_err(antecedent::CausalError::Callback { name, message })
            }
            other => crate::py_err(antecedent::CausalError::from(other)),
        })
    })?;
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
        let quantities: Vec<PosteriorQuantityWire> = if let Some(schema) = &artifact.quantity_schema
        {
            schema.clone()
        } else {
            artifact.quantity_names.iter().map(|name| {
                if name == "residual_variance" {
                    Ok(PosteriorQuantityWire::ResidualVariance)
                } else if let Some(suffix) = name.strip_prefix("coef_") {
                    let index = suffix.parse::<u32>().map_err(|_| crate::value_err(format!(
                        "invalid indexed coefficient quantity `{name}`; expected coef_<u32>"
                    )))?;
                    if suffix != index.to_string() {
                        return Err(crate::value_err(format!(
                            "invalid indexed coefficient quantity `{name}`; expected canonical coef_<u32>"
                        )));
                    }
                    Ok(PosteriorQuantityWire::Coefficient { index, name: None })
                } else {
                    Ok(PosteriorQuantityWire::Effect { name: name.clone() })
                }
            }).collect::<PyResult<Vec<_>>>()?
        };
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
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    outcomes: Vec<String>,
    max_units: usize,
) -> PyResult<Vec<gcm_api::AnomalyScores>> {
    let input = GcmInput::load(&names, columns, edges, 0, None)?;
    detach_catch(py, move || {
        let outcome_ids: Vec<VariableId> =
            outcomes.iter().map(|n| input.id(n)).collect::<PyResult<_>>()?;
        let max_u = if max_units == 0 { input.data.row_count() } else { max_units };
        let (source, ctx) = input.into_source()?;
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
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    outcome: String,
    max_units: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<gcm_api::ChangeAttributionResult> {
    let input = GcmInput::load(&names, columns, edges, seed, threads)?;
    detach_catch(py, move || {
        let y_id = input.id(&outcome)?;
        let max_u = if max_units == 0 { input.data.row_count() } else { max_units };
        let (source, ctx) = input.into_source()?;
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
        let total = result.mean_contributions.iter().sum();
        Ok(gcm_api::synthetic_change_result(y_id, total, pairs, &names).with_route(route))
    })
}

/// Feature relevance scores for parents of `outcome`.
#[pyfunction]
#[pyo3(signature = (names, columns, edges, outcome, *, delta=1.0, n_samples=200, seed=0, threads=None))]
fn attribute_feature_relevance(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    outcome: String,
    delta: f64,
    n_samples: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<Vec<gcm_api::FeatureRelevance>> {
    let input = GcmInput::load(&names, columns, edges, seed, threads)?;
    detach_catch(py, move || {
        let y_id = input.id(&outcome)?;
        let mut features = Vec::new();
        for i in 0..input.data.schema().len() {
            let id = VariableId::from_raw(crate::fit_u32(i)?);
            if id != y_id {
                features.push(id);
            }
        }
        let (source, ctx) = input.into_source()?;
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
    columns: Vec<NumericColumn<'_>>,
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
    gcm_api::require_samples(n_samples)?;
    let input = GcmInput::load(&names, columns, edges, seed, threads)?;
    detach_catch(py, move || {
        let y_id = input.id(&outcome)?;
        let (source, ctx) = input.into_source()?;
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
        let opts = antecedent::gcm::RobustChangeOptions {
            n_samples: n_samples.max(100),
            seed,
            ..antecedent::gcm::RobustChangeOptions::default()
        };
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
    columns: Vec<NumericColumn<'_>>,
    edges: Vec<(String, String)>,
    baseline_start: usize,
    baseline_end: usize,
    comparison_start: usize,
    comparison_end: usize,
    seed: u64,
    threads: Option<u32>,
) -> PyResult<Vec<gcm_api::MechanismChangeDetection>> {
    let input = GcmInput::load(&names, columns, edges, seed, threads)?;
    detach_catch(py, move || {
        let n_variables = input.data.schema().len();
        let mut targets = Vec::with_capacity(n_variables);
        for i in 0..n_variables {
            targets.push(VariableId::from_raw(crate::fit_u32(i)?));
        }
        let query = MechanismChangeQuery::new(
            targets,
            PopulationSelector::TimeRange { start: baseline_start, end: baseline_end },
            PopulationSelector::TimeRange { start: comparison_start, end: comparison_end },
            0.05,
            n_variables,
        );
        let (source, ctx) = input.into_source()?;
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
    m.add_function(wrap_pyfunction!(attribution_uncertainty, m)?)?;
    m.add_function(wrap_pyfunction!(temporal_model_query, m)?)?;
    m.add_function(wrap_pyfunction!(attribute_feature_relevance, m)?)?;
    m.add_function(wrap_pyfunction!(mechanism_change_detection, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_decision_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_posterior_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(encode_posterior_artifact, m)?)?;
    Ok(())
}
