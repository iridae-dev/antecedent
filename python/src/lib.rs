//! `PyO3` bindings — : Arrow load, `analyze_ate` (incl. Bayesian),
//! `analyze`, `discover_pcmci`, `discover_pcmci_plus`, GCM fit/sample/CF.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs)]
#![allow(unsafe_code)] // required by PyO3
#![allow(
    clippy::doc_markdown,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

mod bayesian;
mod callbacks;
mod design_api;
mod gcm_api;
mod graphs;
mod stability;
mod state_api;

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use arrow_array::{Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use causal_data::TimeDummyEncoding;
use causal::{
    AnalysisError, BayesianConfig, CausalAnalysis, CompiledCausalModel, DecisionProblem,
    DifferenceMeasure, DiscoverParams, DiscoveryAccept, DiscoveryPerformanceRecord,
    DistributionChangeOptions, EstimatorId, FdrAdjustment, FdrControl, GraphInput, IdentifierId,
    InferenceMode, MultiDatasetConstraints, RefuteSuite, ScoredLink, SpaceDummyCiMode,
    StaticDiscoverParams, StructureChangeOptions, RegimeAssignment, TemporalLinearPredictor,
    TemporalMediationEstimator, TimeDummyCiMode,
    anomaly_attribution as facade_anomaly_attribution,
    attribute_distribution_change as facade_attribute_distribution_change,
    attribute_distribution_change_robust as facade_attribute_distribution_change_robust,
    attribute_feature_relevance as facade_attribute_feature_relevance,
    attribute_path_specific as facade_attribute_path_specific,
    attribute_structure_change as facade_attribute_structure_change,
    attribute_unit_change as facade_attribute_unit_change,
    counterfactual_ite as facade_counterfactual_ite, dag_from_dot as facade_dag_from_dot,
    dag_from_networkx_adjacency as facade_dag_from_networkx_adjacency,
    dag_to_dot as facade_dag_to_dot, dag_to_json as facade_dag_to_json,
    dag_to_networkx_adjacency as facade_dag_to_networkx_adjacency,
    decode_causal_posterior_bytes, discover_jpcmci_plus as facade_discover_jpcmci_plus,
    discover_lpcmci as facade_discover_lpcmci, discover_fci as facade_discover_fci,
    discover_ges as facade_discover_ges,
    discover_lingam as facade_discover_lingam,
    discover_notears as facade_discover_notears,
    discover_pc as facade_discover_pc, discover_pcmci as facade_discover_pcmci,
    discover_pcmci_plus as facade_discover_pcmci_plus,
    discover_rfci as facade_discover_rfci,
    discover_rpcmci as facade_discover_rpcmci, encode_causal_posterior_bytes,
    evaluate_decision as facade_evaluate_decision, fit_gcm,
    mechanism_change_detection as facade_mechanism_change_detection,
    pag_definite_directed_edge_count,
    sample_do as facade_sample_do,
    sample_interventional_distribution as facade_sample_interventional_distribution,
    two_regime_half_split,
};
use causal_stats::PartialCorrelation;
use causal_core::{
    AllocationMethod, AttributionComponents, AverageEffectQuery, CachePolicy,
    CausalQuery, CausalRng, ChangeAttributionQuery, ConditionalEffectQuery, DistributionRef,
    ExecutionContext, Intervention, InterventionalDistributionQuery, KernelPolicy, Lag,
    MechanismChangeQuery, MediationContrast, MediationQuery, PathSpecificEffectQuery,
    PopulationRegistry, PopulationSelector, PredicateExpr, RegimeId, SchemaError, ShapleyConfig,
    TargetPopulation, TemporalEffectQuery, TemporalPolicy, UnitChangeQuery, VERSION, Value,
    VariableId,
};
use causal_data::{
    ArrowCColumn, DataError, EventData, MultiEnvironmentData, PanelData, PanelUnit,
    SamplingRegularity, TableView, TimeIndex, TimeSeriesData,
    tabular_from_arrow_c_columns, tabular_from_record_batch,
};
use causal_expr::{CausalExprArena, IdentifiedEstimand};
use causal_graph::{
    Cpdag, Dag, DenseNodeId, Endpoint, GraphError, MarkedEdge, MiddleMark, NodeRef, Pag,
    TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged,
};
use causal_io::{
    CausalPosteriorWire, IoError, PosteriorQuantityWire,
    encode_posterior_artifact as encode_posterior_wire,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

create_exception!(causal._native, CausalError, PyException);
create_exception!(causal._native, CausalIdentifyError, CausalError);
create_exception!(causal._native, CausalEstimateError, CausalError);
create_exception!(causal._native, CausalValidateError, CausalError);
create_exception!(causal._native, CausalDiscoveryError, CausalError);
create_exception!(causal._native, CausalModelError, CausalError);
create_exception!(causal._native, CausalCounterfactualError, CausalError);
create_exception!(causal._native, CausalAttributionError, CausalError);
create_exception!(causal._native, CausalDataError, CausalError);
create_exception!(causal._native, CausalGraphError, CausalError);
create_exception!(causal._native, CausalDesignError, CausalError);
create_exception!(causal._native, CausalStateError, CausalError);
create_exception!(causal._native, CausalSerializationError, CausalError);
create_exception!(causal._native, CausalCompileError, CausalError);
create_exception!(causal._native, CausalResourceError, CausalError);
create_exception!(causal._native, CausalReviewError, CausalError);
create_exception!(causal._native, CausalUnsupportedError, CausalError);

/// Parse Python `refute=` — bool or suite name (`"full"` / `"placebo"` / `"none"`).
/// `None` (omitted kwarg) defaults to PlaceboAndRcc.
fn suite_from_refute(obj: Option<&Bound<'_, PyAny>>) -> PyResult<RefuteSuite> {
    let Some(obj) = obj else {
        return Ok(RefuteSuite::PlaceboAndRcc);
    };
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(if b { RefuteSuite::PlaceboAndRcc } else { RefuteSuite::None });
    }
    if let Ok(s) = obj.extract::<String>() {
        return match s.trim().to_ascii_lowercase().as_str() {
            "full" | "validation.full" => Ok(RefuteSuite::Full),
            "placebo" | "placebo_and_rcc" | "placebo+rcc" | "true" | "1" => {
                Ok(RefuteSuite::PlaceboAndRcc)
            }
            "none" | "off" | "false" | "0" => Ok(RefuteSuite::None),
            other => Err(PyValueError::new_err(format!(
                "unknown refute={other:?}; use True|False|\"full\"|\"placebo\"|\"none\""
            ))),
        };
    }
    Err(PyValueError::new_err(
        "refute= must be bool or str (True|False|\"full\"|\"placebo\"|\"none\")",
    ))
}

trait IntoCausalPyErr {
    fn into_causal_py_err(self) -> PyErr;
}

pub(crate) fn py_err<E: IntoCausalPyErr>(e: E) -> PyErr {
    e.into_causal_py_err()
}

/// Fallback for domain errors not re-exported at the binding crate boundary.
pub(crate) fn py_msg(e: impl ToString) -> PyErr {
    CausalError::new_err(e.to_string())
}

fn py_estimate(e: impl ToString) -> PyErr {
    CausalEstimateError::new_err(e.to_string())
}

/// Convert a Rust panic payload into a typed Python error so panics never cross FFI.
pub(crate) fn catch_ffi<F, T>(f: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T>,
{
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => Err(CausalError::new_err(format!(
            "internal Rust panic: {}",
            panic_payload_msg(payload.as_ref())
        ))),
    }
}

fn panic_payload_msg(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".into()
    }
}

/// Release the GIL for native work and convert any panic into [`CausalError`].
pub(crate) fn detach_catch<F, T>(py: Python<'_>, f: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T> + Send,
    T: Send,
{
    py.detach(|| catch_ffi(f))
}

impl IntoCausalPyErr for AnalysisError {
    fn into_causal_py_err(self) -> PyErr {
        match self {
            Self::Identify(e) => CausalIdentifyError::new_err(e.to_string()),
            Self::Estimate(e) => CausalEstimateError::new_err(e.to_string()),
            Self::Validate(e) => CausalValidateError::new_err(e.to_string()),
            Self::Discovery(e) => CausalDiscoveryError::new_err(e.to_string()),
            Self::Model(e) => CausalModelError::new_err(e.to_string()),
            Self::Counterfactual(e) => CausalCounterfactualError::new_err(e.to_string()),
            Self::Attribution(e) => CausalAttributionError::new_err(e.to_string()),
            Self::Serialization(e) => CausalSerializationError::new_err(e.to_string()),
            Self::Data(e) => CausalDataError::new_err(e.to_string()),
            Self::Graph(e) => CausalGraphError::new_err(e.to_string()),
            Self::Design(e) => CausalDesignError::new_err(e.to_string()),
            Self::State(e) => match &e {
                causal::StateError::CacheBudget { .. } => {
                    CausalResourceError::new_err(e.to_string())
                }
                _ => CausalStateError::new_err(e.to_string()),
            },
            Self::Schema(e) => CausalDataError::new_err(e.to_string()),
            Self::Compile { message } => CausalCompileError::new_err(message),
            Self::Resource { message } => CausalResourceError::new_err(message),
            Self::ReviewRequired {
                kind,
                algorithm,
                pending_edge_count,
                message,
                hint,
            } => {
                let err = CausalReviewError::new_err(message.clone());
                Python::attach(|py| {
                    let inst = err.value(py);
                    let _ = inst.setattr("kind", kind.as_str());
                    let _ = inst.setattr("algorithm", algorithm.as_deref());
                    let _ = inst.setattr("pending_edge_count", pending_edge_count);
                    let _ = inst.setattr("hint", hint.as_str());
                    let _ = inst.setattr("message", message.as_str());
                });
                err
            }
            Self::Unsupported { message } => CausalUnsupportedError::new_err(message),
            Self::Missing { field } => {
                CausalCompileError::new_err(format!("missing required field: {field}"))
            }
        }
    }
}

impl IntoCausalPyErr for DataError {
    fn into_causal_py_err(self) -> PyErr {
        AnalysisError::from(self).into_causal_py_err()
    }
}

impl IntoCausalPyErr for GraphError {
    fn into_causal_py_err(self) -> PyErr {
        AnalysisError::from(self).into_causal_py_err()
    }
}

impl IntoCausalPyErr for IoError {
    fn into_causal_py_err(self) -> PyErr {
        AnalysisError::from(self).into_causal_py_err()
    }
}

impl IntoCausalPyErr for SchemaError {
    fn into_causal_py_err(self) -> PyErr {
        AnalysisError::from(self).into_causal_py_err()
    }
}

impl IntoCausalPyErr for arrow_schema::ArrowError {
    fn into_causal_py_err(self) -> PyErr {
        CausalDataError::new_err(self.to_string())
    }
}

/// Result of the conversion probe (same Arrow→tabular path as analyze/discover).
#[pyclass]
struct ArrowLoadInfo {
    #[pyo3(get)]
    row_count: usize,
    #[pyo3(get)]
    column_count: usize,
    #[pyo3(get)]
    bytes_copied: u64,
    #[pyo3(get)]
    bytes_borrowed: u64,
    #[pyo3(get)]
    diagnostic_count: usize,
    /// Schema names after library-owned ingestion (proves the batch was parsed).
    #[pyo3(get)]
    column_names: Vec<String>,
}

/// Coarse-grained ATE analysis result (single boundary crossing).
#[pyclass]
struct AteAnalysisResult {
    #[pyo3(get)]
    ate: f64,
    #[pyo3(get)]
    se_analytic: f64,
    #[pyo3(get)]
    se_bootstrap: Option<f64>,
    /// Soft-failed bootstrap replicates (None if bootstrap was not requested).
    #[pyo3(get)]
    bootstrap_replicates_failed: Option<u32>,
    #[pyo3(get)]
    adjustment_set: Vec<String>,
    #[pyo3(get)]
    identification_status: String,
    #[pyo3(get)]
    refutation_passed: bool,
    /// Whether any refutation validators were actually run.
    #[pyo3(get)]
    refutation_ran: bool,
    #[pyo3(get)]
    refutation_count: usize,
    #[pyo3(get)]
    assumption_count: usize,
    #[pyo3(get)]
    derivation_step_count: usize,
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    estimator_id: String,
    #[pyo3(get)]
    overlap_ess: Option<f64>,
    #[pyo3(get)]
    overlap_propensity_min: Option<f64>,
    /// Posterior mean of the primary effect (Bayesian path).
    #[pyo3(get)]
    posterior_effect_mean: Option<f64>,
    /// Posterior SD of the primary effect.
    #[pyo3(get)]
    posterior_effect_sd: Option<f64>,
    /// 2.5% quantile of the primary effect.
    #[pyo3(get)]
    posterior_q025: Option<f64>,
    /// 97.5% quantile of the primary effect.
    #[pyo3(get)]
    posterior_q975: Option<f64>,
    /// Number of posterior draws.
    #[pyo3(get)]
    posterior_n_draws: Option<usize>,
    /// Empirical P(effect < 0).
    #[pyo3(get)]
    posterior_p_below_zero: Option<f64>,
    /// Inference backend id (e.g. laplace / conjugate_gaussian).
    #[pyo3(get)]
    posterior_backend: Option<String>,
    /// Serialized posterior artifact bytes (CBOR meta + f64 LE draws) when Bayesian.
    #[pyo3(get)]
    posterior_artifact: Option<Vec<u8>>,
    /// Human-readable diagnostic messages from the analysis.
    #[pyo3(get)]
    diagnostics: Vec<String>,
    /// Number of provenance nodes recorded for this run.
    #[pyo3(get)]
    provenance_node_count: usize,
    /// Logical plan id.
    #[pyo3(get)]
    plan_id: String,
    /// Data modality classification.
    #[pyo3(get)]
    modality: String,
    /// Estimated peak memory from the physical plan.
    #[pyo3(get)]
    peak_memory_bytes: Option<u64>,
    /// Worker threads from the physical plan (`0` = serial).
    #[pyo3(get)]
    worker_threads: u32,
    /// Expected Python boundary crossings recorded on the physical plan.
    /// Expected Python boundary crossings recorded on the physical plan.
    #[pyo3(get)]
    expected_python_crossings: u32,
    #[pyo3(get)]
    prior_ppc_p_value: Option<f64>,
    #[pyo3(get)]
    prior_ppc_observed: Option<f64>,
    #[pyo3(get)]
    prior_ppc_predictive_mean: Option<f64>,
    #[pyo3(get)]
    prior_ppc_predictive_sd: Option<f64>,
    #[pyo3(get)]
    prior_ppc_n_sims: Option<u32>,
    #[pyo3(get)]
    posterior_ppc_p_value: Option<f64>,
    #[pyo3(get)]
    posterior_ppc_observed: Option<f64>,
    #[pyo3(get)]
    posterior_ppc_predictive_mean: Option<f64>,
    #[pyo3(get)]
    posterior_ppc_predictive_sd: Option<f64>,
    #[pyo3(get)]
    posterior_ppc_n_sims: Option<u32>,
    #[pyo3(get)]
    prior_sensitivity_scales: Option<Vec<f64>>,
    #[pyo3(get)]
    prior_sensitivity_means: Option<Vec<f64>>,
    #[pyo3(get)]
    prior_sensitivity_sds: Option<Vec<f64>>,
    #[pyo3(get)]
    posterior_unidentified_mass: Option<f64>,
}


/// Decoded posterior artifact for Python consumers .
#[pyclass]
struct PosteriorArtifact {
    #[pyo3(get)]
    n_draws: usize,
    #[pyo3(get)]
    mean: Vec<f64>,
    #[pyo3(get)]
    sd: Vec<f64>,
    #[pyo3(get)]
    q025: Vec<f64>,
    #[pyo3(get)]
    q975: Vec<f64>,
    #[pyo3(get)]
    draws: Vec<f64>,
    #[pyo3(get)]
    backend_id: String,
    #[pyo3(get)]
    identification: String,
    #[pyo3(get)]
    unidentified_mass: f64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    hessian_condition: f64,
    #[pyo3(get)]
    quantity_names: Vec<String>,
}

pub(crate) fn columns_to_batch(
    names: &[String],
    columns: &[PyReadonlyArray1<'_, f64>],
) -> PyResult<RecordBatch> {
    if names.len() != columns.len() {
        return Err(PyValueError::new_err("names and columns must have the same length"));
    }
    if columns.is_empty() {
        return Err(PyValueError::new_err("at least one column required"));
    }
    let n = columns[0].as_array().len();
    for col in columns {
        if col.as_array().len() != n {
            return Err(PyValueError::new_err("column length mismatch"));
        }
    }
    let fields: Vec<Field> =
        names.iter().map(|nm| Field::new(nm, DataType::Float64, true)).collect();
    let schema = Schema::new(fields);
    // Contiguous copy from NumPy buffers (no Option-per-element intermediate).
    let arrays: Vec<Arc<dyn arrow_array::Array>> = columns
        .iter()
        .map(|c| {
            let slice = c.as_array();
            let values: Vec<f64> = slice.iter().copied().collect();
            Arc::new(Float64Array::from(values)) as Arc<dyn arrow_array::Array>
        })
        .collect();
    RecordBatch::try_new(Arc::new(schema), arrays).map_err(py_err)
}

/// Conversion probe: NumPy → Arrow → library-owned tabular storage.
///
/// Shares the same ingestion path as `analyze*` / `discover_*`. The loaded table is not
/// retained across the FFI boundary; call analysis APIs with the original NumPy columns.
#[pyfunction]
fn load_float64_columns(
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
) -> PyResult<ArrowLoadInfo> {
    catch_ffi(|| {
        let batch = columns_to_batch(&names, &columns)?;
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let column_names: Vec<String> =
            loaded.data.schema().variables().iter().map(|v| v.name.to_string()).collect();
        Ok(ArrowLoadInfo {
            row_count: loaded.data.row_count(),
            column_count: loaded.data.schema().len(),
            bytes_copied: loaded.bytes_copied,
            bytes_borrowed: loaded.bytes_borrowed,
            diagnostic_count: loaded.diagnostics.len(),
            column_names,
        })
    })
}

/// Load float64 columns from Arrow C Data Interface exporters (PyArrow / `__arrow_c_array__`).
///
/// Prefers zero-copy borrow of contiguous float64 value buffers.
#[pyfunction]
fn load_float64_arrow_c_columns(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<ArrowLoadInfo> {
    catch_ffi(|| {
        if names.len() != columns.len() {
            return Err(CausalDataError::new_err("names and columns length mismatch"));
        }
        let mut cdi_cols = Vec::with_capacity(columns.len());
        for (name, obj) in names.into_iter().zip(columns) {
            let (array, schema) = take_arrow_c_array(py, &obj)?;
            cdi_cols.push(ArrowCColumn { name, array, schema });
        }
        let loaded = tabular_from_arrow_c_columns(cdi_cols).map_err(py_err)?;
        let column_names: Vec<String> =
            loaded.data.schema().variables().iter().map(|v| v.name.to_string()).collect();
        Ok(ArrowLoadInfo {
            row_count: loaded.data.row_count(),
            column_count: loaded.data.schema().len(),
            bytes_copied: loaded.bytes_copied,
            bytes_borrowed: loaded.bytes_borrowed,
            diagnostic_count: loaded.diagnostics.len(),
            column_names,
        })
    })
}

/// Extract CDI structs from an object exporting `__arrow_c_array__`.
fn take_arrow_c_array(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<(causal_data::FfiArrowArray, causal_data::FfiArrowSchema)> {
    use std::ffi::CStr;

    use pyo3::types::PyCapsule;

    let export = obj.call_method0("__arrow_c_array__")?;
    let tuple = export.cast::<pyo3::types::PyTuple>()?;
    if tuple.len() != 2 {
        return Err(CausalDataError::new_err(
            "__arrow_c_array__ must return (schema_capsule, array_capsule)",
        ));
    }
    let schema_cap = tuple.get_item(0)?.cast_into::<PyCapsule>()?;
    let array_cap = tuple.get_item(1)?.cast_into::<PyCapsule>()?;

    let schema_name = CStr::from_bytes_with_nul(b"arrow_schema\0").unwrap();
    let array_name = CStr::from_bytes_with_nul(b"arrow_array\0").unwrap();

    let schema_ptr = schema_cap.pointer_checked(Some(schema_name))?.as_ptr()
        as *mut causal_data::FfiArrowSchema;
    let array_ptr =
        array_cap.pointer_checked(Some(array_name))?.as_ptr() as *mut causal_data::FfiArrowArray;
    if schema_ptr.is_null() || array_ptr.is_null() {
        return Err(CausalDataError::new_err("null Arrow C Data capsule pointer"));
    }

    // SAFETY: capsules export valid CDI structs; we move them out and leave released empties
    // so the capsule destructor is a no-op.
    let schema = unsafe { std::ptr::read(schema_ptr) };
    let array = unsafe { std::ptr::read(array_ptr) };
    unsafe {
        std::ptr::write(schema_ptr, causal_data::FfiArrowSchema::empty());
        std::ptr::write(array_ptr, causal_data::FfiArrowArray::empty());
    }
    let _ = py;
    Ok((array, schema))
}

fn tabular_from_arrow_c_objs(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<causal_data::TabularData> {
    if names.len() != columns.len() {
        return Err(CausalDataError::new_err("names and columns length mismatch"));
    }
    let mut cdi_cols = Vec::with_capacity(columns.len());
    for (name, obj) in names.into_iter().zip(columns) {
        let (array, schema) = take_arrow_c_array(py, &obj)?;
        cdi_cols.push(ArrowCColumn { name, array, schema });
    }
    let loaded = tabular_from_arrow_c_columns(cdi_cols).map_err(py_err)?;
    Ok(loaded.data)
}

pub(crate) fn py_execution_context(seed: u64, threads: u32) -> ExecutionContext {
    ExecutionContext::production(seed, threads)
}

/// Run static ATE: identify → estimate → optional refute .
///
/// Parse optional `target_population` dict from Python (`kind` + fields).
fn parse_target_population(
    spec: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<TargetPopulation>> {
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
            return Err(PyValueError::new_err(format!(
                "unknown target_population kind {other:?}"
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
            let weights: Vec<f64> = v.extract()?;
            reg.insert_distribution(DistributionRef::from_raw(id), weights);
        }
    }
    Ok(Some(reg))
}

/// `identifier`/`estimator` select the identification strategy and estimator; leaving both
/// `None` preserves the default (`backdoor.adjustment` + `linear.adjustment.ate`).
/// See [`causal::CausalAnalysisBuilder::identifier`] and
/// [`causal::CausalAnalysisBuilder::estimator`] for the supported ids.
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
    refute=None,
    validators=None,
    running_variable=None,
    cutoff=None,
    bandwidth=None,
    seed=1,
    bootstrap=50,
    threads=1,
    target_population=None,
    population_predicates=None,
    population_distributions=None,
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
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
    target_population: Option<Bound<'_, PyDict>>,
    population_predicates: Option<Bound<'_, PyDict>>,
    population_distributions: Option<Bound<'_, PyDict>>,
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
    // Drop NumPy borrows before releasing the GIL.
    drop(columns);

    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;

        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut dag = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data
                .schema()
                .id_of(from)
                .map_err(|e| CausalDataError::new_err(format!("edge from: {e}")))?;
            let to_id = data
                .schema()
                .id_of(to)
                .map_err(|e| CausalDataError::new_err(format!("edge to: {e}")))?;
            dag.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }

        let mut query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
        if let Some(pop) = pop_spec {
            query = query.with_target_population(pop);
        }

        let rd_ids = {
            let wants_rd = estimator.as_deref().map(|e| e.eq_ignore_ascii_case("rd.sharp")).unwrap_or(false)
                || running_variable.is_some()
                || cutoff.is_some()
                || bandwidth.is_some();
            if wants_rd {
                let (rv, cut, bw) = match (running_variable.as_deref(), cutoff, bandwidth) {
                    (Some(rv), Some(cut), Some(bw)) => (rv, cut, bw),
                    _ => {
                        return Err(PyValueError::new_err(
                            "rd.sharp (or any RD kwargs) requires running_variable, cutoff, and bandwidth",
                        ));
                    }
                };
                let rv_id = data.schema().id_of(rv).map_err(py_err)?;
                Some((rv_id, cut, bw))
            } else {
                None
            }
        };
        let mut builder = CausalAnalysis::builder()
            .data(data)
            .graph(dag)
            .query(query)
            .refute(suite)
            .custom_validators(custom_validators)
            .bootstrap_replicates(bootstrap);
        if let Some(reg) = registry {
            builder = builder.population_registry(reg);
        }
        if let Some(id) = identifier {
            builder = builder.identifier(id);
        }
        if let Some(est) = estimator {
            builder = builder.estimator(est);
        }
        if let Some((rv_id, cut, bw)) = rd_ids {
            builder = builder.rd_config(rv_id, cut, bw);
        }

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
                    let ctx = py_execution_context(seed, threads);
                    let result = analysis.run(&ctx).map_err(py_err)?;
                    return ate_result_from_analysis(&names, result);
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown inference mode {other:?}; use frequentist|bayesian|conjugate|hmc"
                    )));
                }
            };
            if let Some(bytes) = prior_artifact.as_deref() {
                let prior = causal::prior_set_from_posterior_bytes(bytes).map_err(py_err)?;
                cfg = cfg.prior(prior);
            }
            builder = builder.inference(InferenceMode::Bayesian(cfg));
            // Keep the caller's refute suite. Overwriting with None previously made
            // `refutation_passed=True` for checks that never ran.
        }
        let analysis = builder.build().map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        ate_result_from_analysis(&names, result)
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
    refute=None,
    validators=None,
    running_variable=None,
    cutoff=None,
    bandwidth=None,
    seed=1,
    bootstrap=50,
    threads=1
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
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    let data = tabular_from_arrow_c_objs(py, names.clone(), columns)?;
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };

    detach_catch(py, move || {
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;

        let n_vars = u32::try_from(data.schema().len())
            .map_err(|_| PyValueError::new_err("too many variables"))?;
        let mut dag = Dag::with_variables(n_vars);
        for (from, to) in &edges {
            let from_id = data
                .schema()
                .id_of(from)
                .map_err(|e| CausalDataError::new_err(format!("edge from: {e}")))?;
            let to_id = data
                .schema()
                .id_of(to)
                .map_err(|e| CausalDataError::new_err(format!("edge to: {e}")))?;
            dag.insert_directed(
                DenseNodeId::from_raw(from_id.raw()),
                DenseNodeId::from_raw(to_id.raw()),
            )
            .map_err(py_err)?;
        }

        let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
        
        let rd_ids = {
            let wants_rd = estimator.as_deref().map(|e| e.eq_ignore_ascii_case("rd.sharp")).unwrap_or(false)
                || running_variable.is_some()
                || cutoff.is_some()
                || bandwidth.is_some();
            if wants_rd {
                let (rv, cut, bw) = match (running_variable.as_deref(), cutoff, bandwidth) {
                    (Some(rv), Some(cut), Some(bw)) => (rv, cut, bw),
                    _ => {
                        return Err(PyValueError::new_err(
                            "rd.sharp (or any RD kwargs) requires running_variable, cutoff, and bandwidth",
                        ));
                    }
                };
                let rv_id = data.schema().id_of(rv).map_err(py_err)?;
                Some((rv_id, cut, bw))
            } else {
                None
            }
        };
        let mut builder = CausalAnalysis::builder()
            .data(data)
            .graph(dag)
            .query(query)
            .refute(suite)
            .custom_validators(custom_validators)
            .bootstrap_replicates(bootstrap);
        if let Some(id) = identifier {
            builder = builder.identifier(id);
        }
        if let Some(est) = estimator {
            builder = builder.estimator(est);
        }
        if let Some((rv_id, cut, bw)) = rd_ids {
            builder = builder.rd_config(rv_id, cut, bw);
        }

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
                    let ctx = py_execution_context(seed, threads);
                    let result = analysis.run(&ctx).map_err(py_err)?;
                    return ate_result_from_analysis(&names, result);
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown inference mode {other:?}; use frequentist|bayesian|conjugate|hmc"
                    )));
                }
            };
            if let Some(bytes) = prior_artifact.as_deref() {
                let prior = causal::prior_set_from_posterior_bytes(bytes).map_err(py_err)?;
                cfg = cfg.prior(prior);
            }
            builder = builder.inference(InferenceMode::Bayesian(cfg));
        }
        let analysis = builder.build().map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        ate_result_from_analysis(&names, result)
    })
}

fn run_ate_with_graph_input(
    names: &[String],
    batch: RecordBatch,
    graph: GraphInput,
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
    custom_validators: Vec<Arc<dyn causal_validate::CustomEffectValidator>>,
    running_variable: Option<String>,
    cutoff: Option<f64>,
    bandwidth: Option<f64>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
    let data = loaded.data;
    let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
    let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
    let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
    let rd_ids = {
        let wants_rd = estimator.as_deref().map(|e| e.eq_ignore_ascii_case("rd.sharp")).unwrap_or(false)
            || running_variable.is_some()
            || cutoff.is_some()
            || bandwidth.is_some();
        if wants_rd {
            let (rv, cut, bw) = match (running_variable.as_deref(), cutoff, bandwidth) {
                (Some(rv), Some(cut), Some(bw)) => (rv, cut, bw),
                _ => {
                    return Err(PyValueError::new_err(
                        "rd.sharp (or any RD kwargs) requires running_variable, cutoff, and bandwidth",
                    ));
                }
            };
            let rv_id = data.schema().id_of(rv).map_err(py_err)?;
            Some((rv_id, cut, bw))
        } else {
            None
        }
    };
    let mut builder = CausalAnalysis::builder()
        .data(data)
        .query(query)
        .refute(suite)
        .custom_validators(custom_validators)
        .bootstrap_replicates(bootstrap);
    builder = match graph {
        GraphInput::Static(dag) => builder.graph(dag),
        GraphInput::Pag(pag) => builder.pag(pag),
        GraphInput::Cpdag(cpdag) => builder.cpdag(cpdag),
        GraphInput::Admg(admg) => builder.admg(admg),
        _other => {
            return Err(PyValueError::new_err(
                "unsupported static graph input for analyze_ate_*",
            ));
        }
    };
    if let Some(id) = identifier {
        builder = builder.identifier(id);
    }
    if let Some(est) = estimator {
        builder = builder.estimator(est);
    }
    if let Some((rv_id, cut, bw)) = rd_ids {
        builder = builder.rd_config(rv_id, cut, bw);
    }
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
                let ctx = py_execution_context(seed, threads);
                let result = analysis.run(&ctx).map_err(py_err)?;
                return ate_result_from_analysis(names, result);
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown inference mode {other:?}; use frequentist|bayesian|conjugate|hmc"
                )));
            }
        };
        if let Some(bytes) = prior_artifact.as_deref() {
            let prior = causal::prior_set_from_posterior_bytes(bytes).map_err(py_err)?;
            cfg = cfg.prior(prior);
        }
        builder = builder.inference(InferenceMode::Bayesian(cfg));
    }
    let analysis = builder.build().map_err(py_err)?;
    let ctx = py_execution_context(seed, threads);
    let result = analysis.run(&ctx).map_err(py_err)?;
    ate_result_from_analysis(names, result)
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
    seed=1, bootstrap=50, threads=1
))]
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
        run_ate_with_graph_input(
            &names,
            batch,
            GraphInput::Pag(graph.pag),
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
            seed,
            bootstrap,
            threads,
        )
    })
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
    seed=1, bootstrap=50, threads=1
))]
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
        run_ate_with_graph_input(
            &names,
            batch,
            GraphInput::Cpdag(graph.cpdag),
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
            seed,
            bootstrap,
            threads,
        )
    })
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
    seed=1, bootstrap=50, threads=1
))]
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
        run_ate_with_graph_input(
            &names,
            batch,
            GraphInput::Admg(graph.admg),
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
            seed,
            bootstrap,
            threads,
        )
    })
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
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AteAnalysisResult> {
    let algo = algorithm.to_ascii_lowercase();
    let batch = columns_to_batch(&names, &columns)?;
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let (ci_impl, _ci_name, is_ci_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_ci_callback || !custom_validators.is_empty() {
        1
    } else {
        threads
    };
        let soft_weight = soft_weight.to_string();
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let data = loaded.data;
        let t_id = data.schema().id_of(&treatment).map_err(py_err)?;
        let y_id = data.schema().id_of(&outcome).map_err(py_err)?;
        let query = AverageEffectQuery::with_levels(t_id, y_id, control_level, active_level);
        let fdr_ctrl = if fdr { FdrControl::bh() } else { FdrControl::Off };
        let accept = if accept_discovered {
            DiscoveryAccept::AutoAccept
        } else {
            DiscoveryAccept::Review
        };
        
        let rd_ids = {
            let wants_rd = estimator.as_deref().map(|e| e.eq_ignore_ascii_case("rd.sharp")).unwrap_or(false)
                || running_variable.is_some()
                || cutoff.is_some()
                || bandwidth.is_some();
            if wants_rd {
                let (rv, cut, bw) = match (running_variable.as_deref(), cutoff, bandwidth) {
                    (Some(rv), Some(cut), Some(bw)) => (rv, cut, bw),
                    _ => {
                        return Err(PyValueError::new_err(
                            "rd.sharp (or any RD kwargs) requires running_variable, cutoff, and bandwidth",
                        ));
                    }
                };
                let rv_id = data.schema().id_of(rv).map_err(py_err)?;
                Some((rv_id, cut, bw))
            } else {
                None
            }
        };
        let mut builder = CausalAnalysis::builder().data(data);
        let soft = match soft_weight.as_str() {
            "none" | "" => causal::CiSoftWeight::None,
            "bayes_factor" | "bf" => causal::CiSoftWeight::BayesFactor,
            "posterior_dependence" | "pd" => causal::CiSoftWeight::PosteriorDependence,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown soft_weight {other:?}; use none|bayes_factor|posterior_dependence"
                )));
            }
        };
        builder = match algo.as_str() {
            "pc" => builder.discover_pc(alpha, max_cond_size, fdr_ctrl, accept),
            "ges" => builder.discover_ges(alpha, max_cond_size, fdr_ctrl, accept),
            "lingam" => builder.discover_lingam(max_cond_size, prune_threshold, accept),
            "notears" => {
                builder.discover_notears(max_cond_size, l1, threshold, standardize, accept)
            }
            "fci" => builder.discover_fci(alpha, max_cond_size, fdr_ctrl, accept),
            "rfci" => builder.discover_rfci(alpha, max_cond_size, fdr_ctrl, accept),
            "exact_dag_posterior" | "exact" => builder.discover_exact_dag_posterior(),
            "order_mcmc" => builder.discover_order_mcmc(
                n_chains, n_warmup, mcmc_draws, thin, require_diagnostics_gate,
            ),
            "structure_mcmc" => {
                builder.discover_structure_mcmc(n_chains, n_warmup, mcmc_draws, thin)
            }
            "ci_screened_posterior" | "ci_screened" => builder.discover_ci_screened_posterior(
                alpha, max_cond_size, fdr_ctrl, soft, n_chains, n_warmup, mcmc_draws, thin,
            ),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown static discovery algorithm {other:?};                      use pc|ges|lingam|notears|fci|rfci|exact_dag_posterior|order_mcmc|                     structure_mcmc|ci_screened_posterior"
                )));
            }
        };;
        let mut builder = builder
            .discovery_ci(ci_impl)
            .query(query)
            .refute(suite)
            .custom_validators(custom_validators)
            .bootstrap_replicates(bootstrap);
        if let Some(id) = identifier {
            builder = builder.identifier(id);
        }
        if let Some(est) = estimator {
            builder = builder.estimator(est);
        }
        if let Some((rv_id, cut, bw)) = rd_ids {
            builder = builder.rd_config(rv_id, cut, bw);
        }

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
                    let ctx = py_execution_context(seed, threads);
                    let result = analysis.run(&ctx).map_err(py_err)?;
                    return ate_result_from_analysis(&names, result);
                }
                other => {
                    return Err(PyValueError::new_err(format!(
                        "unknown inference mode {other:?}; use frequentist|bayesian|conjugate|hmc"
                    )));
                }
            };
            if let Some(bytes) = prior_artifact.as_deref() {
                let prior = causal::prior_set_from_posterior_bytes(bytes).map_err(py_err)?;
                cfg = cfg.prior(prior);
            }
            builder = builder.inference(InferenceMode::Bayesian(cfg));
        }
        let analysis = builder.build().map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        ate_result_from_analysis(&names, result)
    })
}

fn build_static_dag(
    data: &causal_data::TabularData,
    edges: &[(String, String)],
) -> PyResult<Dag> {
    let n_vars = u32::try_from(data.schema().len())
        .map_err(|_| PyValueError::new_err("too many variables"))?;
    let mut dag = Dag::with_variables(n_vars);
    for (from, to) in edges {
        let from_id = data
            .schema()
            .id_of(from)
            .map_err(|e| CausalDataError::new_err(format!("edge from: {e}")))?;
        let to_id = data
            .schema()
            .id_of(to)
            .map_err(|e| CausalDataError::new_err(format!("edge to: {e}")))?;
        dag.insert_directed(
            DenseNodeId::from_raw(from_id.raw()),
            DenseNodeId::from_raw(to_id.raw()),
        )
        .map_err(py_err)?;
    }
    Ok(dag)
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
        let dag = build_static_dag(&data, &edges)?;
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(dag)
            .causal_query(CausalQuery::Distribution(query))
            .identifier(IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalDistribution)
            .build()
            .map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        ate_result_from_analysis(&names, result)
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
        let dag = build_static_dag(&data, &edges)?;
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(dag)
            .causal_query(CausalQuery::PathSpecific(query))
            .identifier(IdentifierId::PathSpecificNatural)
            .estimator(EstimatorId::FunctionalEffect)
            .bootstrap_replicates(bootstrap)
            .build()
            .map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        ate_result_from_analysis(&names, result)
    })
}

fn ate_result_from_analysis(
    names: &[String],
    result: causal::CausalAnalysisResult,
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
    ) = if let Some(post) = result.posterior.as_ref() {
        let eq = post.effect_column().unwrap_or(0);
        let artifact = encode_causal_posterior_bytes(post, "ate-analysis").map_err(py_err)?;
        let p_below = post.probability_below(0.0).map_err(py_estimate)?;
        (
            Some(post.summaries.mean[eq]),
            Some(post.summaries.sd[eq]),
            Some(post.summaries.q025[eq]),
            Some(post.summaries.q975[eq]),
            Some(post.draws.n_draws),
            Some(p_below),
            Some(post.diagnostics.backend_id.to_string()),
            Some(artifact),
        )
    } else {
        (None, None, None, None, None, None, None, None)
    };


    use causal_validate::PredictiveCheckKind;
    let mut prior_ppc_p_value = None;
    let mut prior_ppc_observed = None;
    let mut prior_ppc_predictive_mean = None;
    let mut prior_ppc_predictive_sd = None;
    let mut prior_ppc_n_sims = None;
    let mut posterior_ppc_p_value = None;
    let mut posterior_ppc_observed = None;
    let mut posterior_ppc_predictive_mean = None;
    let mut posterior_ppc_predictive_sd = None;
    let mut posterior_ppc_n_sims = None;
    for pc in &result.predictive_checks {
        match pc.kind {
            PredictiveCheckKind::Prior => {
                prior_ppc_p_value = Some(pc.p_value);
                prior_ppc_observed = Some(pc.observed);
                prior_ppc_predictive_mean = Some(pc.predictive_mean);
                prior_ppc_predictive_sd = Some(pc.predictive_sd);
                prior_ppc_n_sims = Some(pc.n_sims);
            }
            PredictiveCheckKind::Posterior => {
                posterior_ppc_p_value = Some(pc.p_value);
                posterior_ppc_observed = Some(pc.observed);
                posterior_ppc_predictive_mean = Some(pc.predictive_mean);
                posterior_ppc_predictive_sd = Some(pc.predictive_sd);
                posterior_ppc_n_sims = Some(pc.n_sims);
            }
        }
    }
    let (prior_sensitivity_scales, prior_sensitivity_means, prior_sensitivity_sds) =
        if let Some(post) = result.posterior.as_ref() {
            if let Some(sens) = post.prior_sensitivity.as_ref() {
                (
                    Some(sens.prior_scales.iter().copied().collect()),
                    Some(sens.effect_means.iter().copied().collect()),
                    Some(sens.effect_sds.iter().copied().collect()),
                )
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        };
    let posterior_unidentified_mass = result.posterior.as_ref().map(|p| p.unidentified_mass);

    Ok(AteAnalysisResult {
        ate: result.estimate.ate,
        se_analytic: result.estimate.se_analytic,
        se_bootstrap: result.estimate.se_bootstrap,
        bootstrap_replicates_failed: result.estimate.bootstrap_replicates_failed,
        adjustment_set,
        identification_status: format!("{:?}", result.identification.status),
        refutation_passed,
        refutation_ran,
        refutation_count: result.refutations.len(),
        assumption_count: result.estimate.assumptions.len(),
        derivation_step_count: result.identification.derivation.steps.len(),
        method: result.estimand.method.to_string(),
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
        plan_id: result.logical_plan.plan_id.to_string(),
        modality: format!("{:?}", result.logical_plan.data_classification),
        peak_memory_bytes: result.physical_plan.estimated_peak_memory_bytes,
        worker_threads: result.physical_plan.worker_threads,
        expected_python_crossings: result.physical_plan.expected_python_crossings,
        prior_ppc_p_value,
        prior_ppc_observed,
        prior_ppc_predictive_mean,
        prior_ppc_predictive_sd,
        prior_ppc_n_sims,
        posterior_ppc_p_value,
        posterior_ppc_observed,
        posterior_ppc_predictive_mean,
        posterior_ppc_predictive_sd,
        posterior_ppc_n_sims,
        prior_sensitivity_scales,
        prior_sensitivity_means,
        prior_sensitivity_sds,
        posterior_unidentified_mass,
    })
}

/// One marked edge from an oriented temporal CPDAG/PAG.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct GraphEdge {
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    source_lag: u32,
    #[pyo3(get)]
    target: String,
    #[pyo3(get)]
    target_lag: u32,
    /// Endpoint mark at `source`: `tail` | `arrow` | `circle` | `conflict`.
    #[pyo3(get)]
    at_source: String,
    /// Endpoint mark at `target`: `tail` | `arrow` | `circle` | `conflict`.
    #[pyo3(get)]
    at_target: String,
}

fn endpoint_name(e: Endpoint) -> &'static str {
    match e {
        Endpoint::Tail => "tail",
        Endpoint::Arrow => "arrow",
        Endpoint::Circle => "circle",
        Endpoint::Conflict => "conflict",
    }
}

fn node_ref_parts(names: &[String], node: NodeRef) -> (String, u32) {
    match node {
        NodeRef::Lagged { variable, lag } => (
            names
                .get(variable.as_usize())
                .cloned()
                .unwrap_or_else(|| format!("var{}", variable.raw())),
            lag.raw(),
        ),
        NodeRef::Static(variable) | NodeRef::Context { variable, .. } => (
            names
                .get(variable.as_usize())
                .cloned()
                .unwrap_or_else(|| format!("var{}", variable.raw())),
            0,
        ),
    }
}

fn graph_edge_from_marked(names: &[String], nodes: &[NodeRef], edge: MarkedEdge) -> GraphEdge {
    let (source, source_lag) = node_ref_parts(names, nodes[edge.a.as_usize()]);
    let (target, target_lag) = node_ref_parts(names, nodes[edge.b.as_usize()]);
    GraphEdge {
        source,
        source_lag,
        target,
        target_lag,
        at_source: endpoint_name(edge.at_a).to_string(),
        at_target: endpoint_name(edge.at_b).to_string(),
    }
}

fn cpdag_graph_edges(names: &[String], cpdag: &TemporalCpdag) -> Vec<GraphEdge> {
    cpdag.edges().into_iter().map(|e| graph_edge_from_marked(names, cpdag.nodes(), e)).collect()
}

fn static_cpdag_graph_edges(names: &[String], cpdag: &Cpdag) -> Vec<GraphEdge> {
    cpdag.edges().into_iter().map(|e| graph_edge_from_marked(names, cpdag.nodes(), e)).collect()
}

fn static_dag_graph_edges(names: &[String], dag: &Dag) -> Vec<GraphEdge> {
    dag.edges().map(|e| graph_edge_from_marked(names, dag.nodes(), e)).collect()
}

fn pag_graph_edges(names: &[String], pag: &TemporalPag) -> Vec<GraphEdge> {
    let nodes = pag.nodes();
    let mut out = Vec::new();
    for i in 0..pag.node_count() {
        let a = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
        for (b, at_a, at_b) in pag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            out.push(graph_edge_from_marked(
                names,
                nodes,
                MarkedEdge { a, b, at_a, at_b, middle: MiddleMark::Empty },
            ));
        }
    }
    out
}

fn static_pag_graph_edges(names: &[String], pag: &Pag) -> Vec<GraphEdge> {
    let nodes = pag.nodes();
    let mut out = Vec::new();
    for i in 0..pag.node_count() {
        let a = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
        for (b, at_a, at_b) in pag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            out.push(graph_edge_from_marked(
                names,
                nodes,
                MarkedEdge { a, b, at_a, at_b, middle: MiddleMark::Empty },
            ));
        }
    }
    out
}

fn static_pag_definite_directed_count(pag: &Pag) -> u64 {
    let mut directed = 0u64;
    for i in 0..pag.node_count() {
        let a = DenseNodeId::from_raw(i as u32);
        for (b, at_a, at_b) in pag.neighbors(a) {
            if b.raw() < a.raw() {
                continue;
            }
            if matches!(
                (at_a, at_b),
                (Endpoint::Tail, Endpoint::Arrow) | (Endpoint::Arrow, Endpoint::Tail)
            ) {
                directed += 1;
            }
        }
    }
    directed
}

/// One discovered lagged link for Python.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct DiscoveredLink {
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    source_lag: u32,
    #[pyo3(get)]
    target: String,
    #[pyo3(get)]
    target_lag: u32,
    #[pyo3(get)]
    statistic: f64,
    #[pyo3(get)]
    p_value: f64,
    /// Benjamini–Hochberg adjusted p-value when FDR ran; otherwise `None`.
    #[pyo3(get)]
    adjusted_p_value: Option<f64>,
}

/// Coarse-grained PCMCI discovery result (single boundary crossing).
///
/// Field set is the stable Rust↔Python temporal discovery schema for .
#[pyclass]
struct PcmciDiscoveryResult {
    #[pyo3(get)]
    links: Vec<DiscoveredLink>,
    #[pyo3(get)]
    algorithm_id: String,
    #[pyo3(get)]
    algorithm_config: String,
    #[pyo3(get)]
    ci_tests: u64,
    #[pyo3(get)]
    links_retained: u64,
    #[pyo3(get)]
    pending_edge_count: u64,
    #[pyo3(get)]
    lagged_frame_bytes: u64,
    #[pyo3(get)]
    worker_threads: u32,
    #[pyo3(get)]
    ci_name: String,
    #[pyo3(get)]
    cpdag_nodes: u64,
    #[pyo3(get)]
    cpdag_directed_edges: u64,
    #[pyo3(get)]
    cpdag_undirected_edges: u64,
    /// Oriented graph body (CPDAG/PAG marks); empty for lagged-only PCMCI.
    #[pyo3(get)]
    graph_edges: Vec<GraphEdge>,
}

pub(crate) fn series_from_batch(batch: &RecordBatch) -> PyResult<(TimeSeriesData, Vec<VariableId>)> {
    let loaded = tabular_from_record_batch(batch).map_err(py_err)?;
    let tabular = loaded.data;
    let n = tabular.row_count();
    let series = TimeSeriesData::try_new(
        tabular.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .map_err(py_err)?;
    let variables: Vec<VariableId> = series.schema().variables().iter().map(|v| v.id).collect();
    Ok((series, variables))
}

pub(crate) fn tabular_from_batch(
    batch: &RecordBatch,
) -> PyResult<(causal_data::TabularData, Vec<VariableId>)> {
    let loaded = tabular_from_record_batch(batch).map_err(py_err)?;
    let tabular = loaded.data;
    let variables: Vec<VariableId> = tabular.schema().variables().iter().map(|v| v.id).collect();
    Ok((tabular, variables))
}

fn discovered_links(names: &[String], links: &[ScoredLink]) -> Vec<DiscoveredLink> {
    links
        .iter()
        .map(|s| DiscoveredLink {
            source: names
                .get(s.link.source.as_usize())
                .cloned()
                .unwrap_or_else(|| format!("var{}", s.link.source.raw())),
            source_lag: s.link.source_lag.raw(),
            target: names
                .get(s.link.target.as_usize())
                .cloned()
                .unwrap_or_else(|| format!("var{}", s.link.target.raw())),
            target_lag: s.link.target_lag.raw(),
            statistic: s.statistic,
            p_value: s.p_value,
            adjusted_p_value: s.adjusted_p_value,
        })
        .collect()
}

fn discovery_result_fields(
    names: &[String],
    links: &[ScoredLink],
    algorithm_id: &str,
    algorithm_config: &str,
    performance: &DiscoveryPerformanceRecord,
    pending_edge_count: u64,
    ci_name: String,
    cpdag_nodes: u64,
    cpdag_directed_edges: u64,
    cpdag_undirected_edges: u64,
    graph_edges: Vec<GraphEdge>,
) -> PcmciDiscoveryResult {
    PcmciDiscoveryResult {
        links: discovered_links(names, links),
        algorithm_id: algorithm_id.to_string(),
        algorithm_config: algorithm_config.to_string(),
        ci_tests: performance.ci_tests,
        links_retained: performance.links_retained,
        pending_edge_count,
        lagged_frame_bytes: performance.lagged_frame_bytes,
        worker_threads: performance.worker_threads,
        ci_name,
        cpdag_nodes,
        cpdag_directed_edges,
        cpdag_undirected_edges,
        graph_edges,
    }
}

/// Run lagged PCMCI discovery.
///
/// NumPy columns in, structured link list out once. Batch CI only (no per-query Python loop
/// unless `ci` is an explicit slow-path callable — ).
/// `ci` selects a named test (default `parcorr`) or a Python batch callable.
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=1))]
fn discover_pcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints::default(),
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_pcmci(&series, &variables, &params, &ctx).map_err(py_err)?;
        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            result.review.pending_edges.len() as u64,
            ci_name,
            0,
            0,
            0,
            Vec::new(),
        ))
    })
}

/// Run static PC discovery over tabular (non-temporal) columns.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=1))]
fn discover_pc(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc: false,
            max_subset: None,
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_pc(&data, &variables, &params, &ctx).map_err(py_err)?;

        let cpdag = &result.evidence.graph;
        let directed = cpdag.directed_edge_count() as u64;
        let undirected = cpdag.undirected_edge_count() as u64;
        let pending = result.review.pending_edges.len() as u64
            + result.review.pending_undirected.len() as u64;
        let graph_edges = static_cpdag_graph_edges(&names, cpdag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            cpdag.node_count() as u64,
            directed,
            undirected,
            graph_edges,
        ))
    })
}

/// Run GES discovery over tabular columns → CPDAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=1, screen_pc=false, max_subset=None))]
fn discover_ges(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: u32,
    screen_pc: bool,
    max_subset: Option<usize>,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc,
            max_subset,
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_ges(&data, &variables, &params, &ctx).map_err(py_err)?;

        let cpdag = &result.evidence.graph;
        let directed = cpdag.directed_edge_count() as u64;
        let undirected = cpdag.undirected_edge_count() as u64;
        let pending = result.review.pending_edges.len() as u64
            + result.review.pending_undirected.len() as u64;
        let graph_edges = static_cpdag_graph_edges(&names, cpdag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            cpdag.node_count() as u64,
            directed,
            undirected,
            graph_edges,
        ))
    })
}

/// Run DirectLiNGAM discovery over tabular columns → DAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, prune_threshold=0.05, seed=1, max_cond_size=8, threads=1))]
fn discover_lingam(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    prune_threshold: f64,
    seed: u64,
    max_cond_size: usize,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha: 0.05,
            max_cond_size,
            fdr: None,
            ci: Arc::new(PartialCorrelation::default()),
            screen_pc: false,
            max_subset: None,
        };
        let ctx = py_execution_context(seed, threads);
        let result =
            facade_discover_lingam(&data, &variables, &params, prune_threshold, &ctx).map_err(py_err)?;

        let dag = &result.evidence.graph;
        let directed = dag.edges().count() as u64;
        let pending = result.review.pending_edges.len() as u64;
        let graph_edges = static_dag_graph_edges(&names, dag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            "direct_lingam".into(),
            dag.node_count() as u64,
            directed,
            0,
            graph_edges,
        ))
    })
}

/// Run NOTEARS discovery over tabular columns → DAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, l1=0.1, threshold=0.3, standardize=true, seed=1, max_cond_size=8, threads=1))]
fn discover_notears(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    l1: f64,
    threshold: f64,
    standardize: bool,
    seed: u64,
    max_cond_size: usize,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    drop(columns);

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha: 0.05,
            max_cond_size,
            fdr: None,
            ci: Arc::new(PartialCorrelation::default()),
            screen_pc: false,
            max_subset: None,
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_notears(
            &data,
            &variables,
            &params,
            l1,
            threshold,
            standardize,
            &ctx,
        )
        .map_err(py_err)?;

        let dag = &result.discovery.evidence.graph;
        let directed = dag.edges().count() as u64;
        let pending = result.discovery.review.pending_edges.len() as u64;
        let graph_edges = static_dag_graph_edges(&names, dag);

        Ok(discovery_result_fields(
            &names,
            &result.discovery.evidence.links,
            result.discovery.algorithm.id.as_ref(),
            result.discovery.algorithm.config.as_ref(),
            &result.discovery.performance,
            pending,
            "notears".into(),
            dag.node_count() as u64,
            directed,
            0,
            graph_edges,
        ))
    })
}

/// Run classic static FCI discovery over tabular columns → PAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=1))]
fn discover_fci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc: false,
            max_subset: None,
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_fci(&data, &variables, &params, &ctx).map_err(py_err)?;

        let pag = &result.evidence.graph;
        let pending = result.review.pending_circles.len() as u64;
        let directed = static_pag_definite_directed_count(pag);
        let graph_edges = static_pag_graph_edges(&names, pag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            pag.node_count() as u64,
            directed,
            pending,
            graph_edges,
        ))
    })
}

/// Run classic static RFCI discovery over tabular columns → PAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, alpha=0.05, fdr=true, seed=1, ci=None, max_cond_size=2, threads=1))]
fn discover_rfci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    max_cond_size: usize,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (data, variables) = tabular_from_batch(&batch)?;
        let params = StaticDiscoverParams {
            alpha,
            max_cond_size,
            fdr: fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            ci: ci_impl,
            screen_pc: false,
            max_subset: None,
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_rfci(&data, &variables, &params, &ctx).map_err(py_err)?;

        let pag = &result.evidence.graph;
        let pending = result.review.pending_circles.len() as u64;
        let directed = static_pag_definite_directed_count(pag);
        let graph_edges = static_pag_graph_edges(&names, pag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            pag.node_count() as u64,
            directed,
            pending,
            graph_edges,
        ))
    })
}

/// Run PCMCI+ discovery returning links plus oriented temporal CPDAG summary.
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=1))]
fn discover_pcmci_plus(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints::default(),
        };
        let ctx = py_execution_context(seed, threads);
        let result =
            facade_discover_pcmci_plus(&series, &variables, &params, &ctx).map_err(py_err)?;

        let cpdag = &result.evidence.graph;
        let directed = cpdag.directed_edge_count() as u64;
        let undirected = cpdag.undirected_edge_count() as u64;
        let pending = result.review.pending_edges.len() as u64
            + result.review.pending_undirected.len() as u64;
        let graph_edges = cpdag_graph_edges(&names, cpdag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            cpdag.node_count() as u64,
            directed,
            undirected,
            graph_edges,
        ))
    })
}

/// Run LPCMCI discovery returning links plus temporal PAG summary (no per-edge GIL).
#[pyfunction]
#[pyo3(signature = (names, columns, *, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=1))]
fn discover_lpcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: u32,
) -> PyResult<PcmciDiscoveryResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };

    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints::default(),
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_lpcmci(&series, &variables, &params, &ctx).map_err(py_err)?;

        let pag = &result.evidence.graph;
        let pending = result.review.pending_circles.len() as u64;
        let directed = pag_definite_directed_edge_count(pag);
        let graph_edges = pag_graph_edges(&names, pag);

        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            pending,
            ci_name,
            pag.node_count() as u64,
            directed,
            pending, // undirected field reused as circle-pending count
            graph_edges,
        ))
    })
}

/// J-PCMCI+ over multiple environments (one GIL crossing).
///
/// `env_columns` is a list of column batches (each env: same `names` order).
/// Optional `context_names` lists observed context columns (must appear in `names`);
/// remaining names are treated as system variables.
#[pyfunction]
#[pyo3(signature = (
    names,
    env_columns,
    *,
    max_lag=1,
    alpha=0.05,
    fdr=true,
    seed=1,
    ci=None,
    weights=None,
    threads=1,
    context_names=None,
    include_space_dummy=true,
    include_time_dummy=false,
    space_dummy_ci="scalar",
    time_dummy_encoding="integer",
    time_dummy_ci="scalar",
))]
fn discover_jpcmci_plus(
    py: Python<'_>,
    names: Vec<String>,
    env_columns: Vec<Vec<PyReadonlyArray1<'_, f64>>>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: u32,
    context_names: Option<Vec<String>>,
    include_space_dummy: bool,
    include_time_dummy: bool,
    space_dummy_ci: &str,
    time_dummy_encoding: &str,
    time_dummy_ci: &str,
) -> PyResult<PcmciDiscoveryResult> {
    if env_columns.is_empty() {
        return Err(PyValueError::new_err("discover_jpcmci_plus needs ≥1 environment"));
    }
    let mut batches = Vec::with_capacity(env_columns.len());
    for cols in &env_columns {
        batches.push(columns_to_batch(&names, cols)?);
    }
    let (ci_impl, ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    let context_names = context_names.unwrap_or_default();
    let threads = if is_callback { 1 } else { threads };
    drop(env_columns);

    detach_catch(py, move || {
        let mut series_list = Vec::with_capacity(batches.len());
        let mut all_variables = Vec::new();
        for (i, batch) in batches.iter().enumerate() {
            let (series, vars) = series_from_batch(batch)?;
            if i == 0 {
                all_variables = vars;
            }
            series_list.push(series);
        }
        let multi = MultiEnvironmentData::try_new(Arc::from(series_list)).map_err(py_err)?;

        let mut context_ids = Vec::new();
        for cname in &context_names {
            let Some(idx) = names.iter().position(|n| n == cname) else {
                return Err(PyValueError::new_err(format!(
                    "context_names entry '{cname}' not found in names"
                )));
            };
            context_ids.push(all_variables[idx]);
        }
        let system: Vec<VariableId> = all_variables
            .iter()
            .copied()
            .filter(|v| !context_ids.contains(v))
            .collect();
        if system.is_empty() {
            return Err(PyValueError::new_err(
                "discover_jpcmci_plus needs ≥1 system variable after excluding context_names",
            ));
        }

        let space_dummy_ci = match space_dummy_ci {
            "scalar" | "scalar_one_hot" | "one_hot" => SpaceDummyCiMode::ScalarOneHot,
            "multivariate" | "multivariate_block" | "block" => SpaceDummyCiMode::MultivariateBlock,
            other => {
                return Err(PyValueError::new_err(format!(
                    "space_dummy_ci must be 'scalar' or 'multivariate', got '{other}'"
                )));
            }
        };
        let time_dummy_encoding = match time_dummy_encoding {
            "integer" | "integer_index" | "index" => TimeDummyEncoding::IntegerIndex,
            "one_hot" | "onehot" | "oh" => TimeDummyEncoding::OneHot,
            other => {
                return Err(PyValueError::new_err(format!(
                    "time_dummy_encoding must be 'integer' or 'one_hot', got '{other}'"
                )));
            }
        };
        let time_dummy_ci = match time_dummy_ci {
            "scalar" | "scalar_one_hot" | "one_hot" => TimeDummyCiMode::ScalarOneHot,
            "multivariate" | "multivariate_block" | "block" => TimeDummyCiMode::MultivariateBlock,
            other => {
                return Err(PyValueError::new_err(format!(
                    "time_dummy_ci must be 'scalar' or 'multivariate', got '{other}'"
                )));
            }
        };
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints {
                context_variables: Arc::from(context_ids),
                include_space_dummy,
                include_time_dummy,
                space_dummy_ci,
                time_dummy_encoding,
                time_dummy_ci,
                ..MultiDatasetConstraints::default()
            },
        };
        let ctx = py_execution_context(seed, threads);
        let result =
            facade_discover_jpcmci_plus(&multi, &system, &params, &ctx).map_err(py_err)?;
        let cpdag = &result.evidence.graph;
        let graph_edges = cpdag_graph_edges(&names, cpdag);
        Ok(discovery_result_fields(
            &names,
            &result.evidence.links,
            result.algorithm.id.as_ref(),
            result.algorithm.config.as_ref(),
            &result.performance,
            result.review.pending_undirected.len() as u64,
            ci_name,
            cpdag.node_count() as u64,
            cpdag.directed_edge_count() as u64,
            cpdag.undirected_edge_count() as u64,
            graph_edges,
        ))
    })
}

/// Two-regime half-split assignment (opt-in helper for RPCMCI; not applied by default).
#[pyfunction(name = "two_regime_half_split")]
fn two_regime_half_split_py(series_len: usize) -> Vec<u32> {
    two_regime_half_split(series_len)
        .regimes
        .iter()
        .map(|r| r.raw())
        .collect()
}

/// RPCMCI with caller-supplied regimes (required; no silent half-split).
#[pyfunction]
#[pyo3(signature = (names, columns, *, regimes, max_lag=1, alpha=0.05, fdr=true, seed=1, ci=None, weights=None, threads=1))]
fn discover_rpcmci(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    regimes: Vec<u32>,
    max_lag: u32,
    alpha: f64,
    fdr: bool,
    seed: u64,
    ci: Option<Bound<'_, PyAny>>,
    weights: Option<Vec<f64>>,
    threads: u32,
) -> PyResult<RpcmciDiscoverySummary> {
    let batch = columns_to_batch(&names, &columns)?;
    let (ci_impl, _ci_name, is_callback) = callbacks::resolve_ci_arg(ci.as_ref(), weights)?;
    drop(columns);
    let threads = if is_callback { 1 } else { threads };
    detach_catch(py, move || {
        let (series, variables) = series_from_batch(&batch)?;
        if regimes.len() != series.row_count() {
            return Err(PyValueError::new_err(format!(
                "regimes length {} != series length {}",
                regimes.len(),
                series.row_count()
            )));
        }
        let assign = RegimeAssignment::try_new(
            regimes.into_iter().map(RegimeId::from_raw).collect::<Vec<_>>(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let params = DiscoverParams {
            max_lag,
            alpha,
            fdr: fdr.then(FdrAdjustment::bh),
            ci: ci_impl,
            multi_dataset: MultiDatasetConstraints::default(),
        };
        let ctx = py_execution_context(seed, threads);
        let result = facade_discover_rpcmci(&series, &variables, &assign, &params, None, &ctx)
            .map_err(py_err)?;
        let mut regime_ids = Vec::new();
        let mut directed = Vec::new();
        let mut undirected = Vec::new();
        for (rid, g) in result.graphs.graphs.iter() {
            regime_ids.push(rid.raw());
            directed.push(g.directed_edge_count() as u64);
            undirected.push(g.undirected_edge_count() as u64);
        }
        Ok(RpcmciDiscoverySummary {
            algorithm: result.algorithm.id.to_string(),
            n_regimes: regime_ids.len() as u64,
            regime_ids,
            directed_edges: directed,
            undirected_edges: undirected,
        })
    })
}

/// Mediation effect surface summary (total / direct / mediated).
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
        let id = |nm: &str| {
            series
                .schema()
                .id_of(nm)
                .map_err(|e| CausalDataError::new_err(format!("unknown variable {nm}: {e}")))
        };
        let t = id(&treatment)?;
        let m = id(&mediator)?;
        let y = id(&outcome)?;
        let q = MediationQuery::binary(t, y, [m], MediationContrast::Total);
        let mut arena = CausalExprArena::new();
        let functional = arena.frontdoor_ate(t, y, &[m], Value::f64(1.0), Value::f64(0.0));
        let estimand =
            IdentifiedEstimand::frontdoor("temporal_mediation.total", Arc::from([m]), functional);
        let ctx = py_execution_context(seed, threads);
        let surface = TemporalMediationEstimator::new()
            .effect_surface(&series, &estimand, &q, &ctx)
            .map_err(py_estimate)?;
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
        let id = |nm: &str| {
            series
                .schema()
                .id_of(nm)
                .map_err(|e| CausalDataError::new_err(format!("unknown variable {nm}: {e}")))
        };
        let y = id(&target)?;
        let x = id(&parent)?;
        let policy = KernelPolicy::default_policy();
        let pred = TemporalLinearPredictor::fit(
            &series,
            y,
            [causal_data::LaggedColumn { variable: x, lag: Lag::from_raw(parent_lag) }],
            &policy,
        )
        .map_err(py_estimate)?;
        let yhat = pred.predict_intervened(&series, x, level, &policy).map_err(py_estimate)?;
        let mean = yhat.iter().sum::<f64>() / yhat.len().max(1) as f64;
        Ok(PredictSummary { mean_prediction: mean, n: yhat.len() as u64 })
    })
}

/// RPCMCI summary (typed regimes, no single-graph collapse).
#[pyclass]
struct RpcmciDiscoverySummary {
    #[pyo3(get)]
    algorithm: String,
    #[pyo3(get)]
    n_regimes: u64,
    #[pyo3(get)]
    regime_ids: Vec<u32>,
    #[pyo3(get)]
    directed_edges: Vec<u64>,
    #[pyo3(get)]
    undirected_edges: Vec<u64>,
}

/// Mediation effects summary.
#[pyclass]
struct MediationEffectsSummary {
    #[pyo3(get)]
    total: f64,
    #[pyo3(get)]
    direct: f64,
    #[pyo3(get)]
    mediated: f64,
}

/// Prediction summary under intervention.
#[pyclass]
struct PredictSummary {
    #[pyo3(get)]
    mean_prediction: f64,
    #[pyo3(get)]
    n: u64,
}

/// Unified analysis result (static or temporal).
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct AnalysisResult {
    #[pyo3(get)]
    ate: f64,
    #[pyo3(get)]
    se_analytic: f64,
    #[pyo3(get)]
    se_bootstrap: Option<f64>,
    #[pyo3(get)]
    plan_id: String,
    #[pyo3(get)]
    modality: String,
    #[pyo3(get)]
    peak_memory_bytes: Option<u64>,
    #[pyo3(get)]
    identification_status: String,
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    diagnostics: Vec<String>,
    #[pyo3(get)]
    provenance_node_count: usize,
    #[pyo3(get)]
    refutation_count: usize,
    #[pyo3(get)]
    worker_threads: u32,
    #[pyo3(get)]
    expected_python_crossings: u32,
    #[pyo3(get)]
    adjustment_set: Vec<String>,
    #[pyo3(get)]
    assumption_count: usize,
    #[pyo3(get)]
    derivation_step_count: usize,
    #[pyo3(get)]
    estimator_id: String,
    #[pyo3(get)]
    posterior_effect_mean: Option<f64>,
    #[pyo3(get)]
    posterior_effect_sd: Option<f64>,
    #[pyo3(get)]
    posterior_q025: Option<f64>,
    #[pyo3(get)]
    posterior_q975: Option<f64>,
    #[pyo3(get)]
    posterior_n_draws: Option<usize>,
    #[pyo3(get)]
    posterior_p_below_zero: Option<f64>,
    #[pyo3(get)]
    posterior_backend: Option<String>,
    #[pyo3(get)]
    posterior_artifact: Option<Vec<u8>>,
    #[pyo3(get)]
    posterior_unidentified_mass: Option<f64>,
    #[pyo3(get)]
    mediation_total: Option<f64>,
    #[pyo3(get)]
    mediation_direct: Option<f64>,
    #[pyo3(get)]
    mediation_mediated: Option<f64>,
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
    treatment_lag=1,
    horizon_steps=1,
    active_level=1.0,
    policy="pulse",
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
    columns: Vec<PyReadonlyArray1<'_, f64>>,
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
    let batch = columns_to_batch(&names, &columns)?;
    let policy = policy.to_ascii_lowercase();
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let tabular = loaded.data;
        let n = tabular.row_count();
        let series = TimeSeriesData::try_new(
            tabular.storage().clone(),
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .map_err(py_err)?;

        let name_to_id = |nm: &str| -> PyResult<VariableId> {
            series
                .schema()
                .id_of(nm)
                .map_err(|e| CausalDataError::new_err(format!("unknown variable {nm}: {e}")))
        };
        let t_id = name_to_id(&treatment)?;
        let y_id = name_to_id(&outcome)?;

        let mut g = TemporalDag::empty();
        for (src, slag, tgt, tlag) in &edges {
            let s =
                ensure_lagged(&mut g, name_to_id(src)?, Lag::from_raw(*slag)).map_err(py_err)?;
            let t =
                ensure_lagged(&mut g, name_to_id(tgt)?, Lag::from_raw(*tlag)).map_err(py_err)?;
            g.insert_directed(s, t).map_err(py_err)?;
        }

        let q = temporal_query_from_policy(
            &policy,
            t_id,
            y_id,
            treatment_lag,
            horizon_steps,
            active_level,
        )?;

        let mut builder = CausalAnalysis::builder()
            .series(series)
            .temporal_graph(g)
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

/// Temporal effect with a typed [`TemporalPag`] (review-required; same as LPCMCI without auto-accept).
#[pyfunction]
#[pyo3(signature = (
    names,
    columns,
    graph,
    treatment,
    outcome,
    *,
    treatment_lag=1,
    horizon_steps=1,
    active_level=1.0,
    policy="pulse",
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
fn analyze_temporal_pag(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
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
    refute: Option<Bound<'_, PyAny>>,
    validators: Option<Bound<'_, PyAny>>,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AnalysisResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let policy = policy.to_ascii_lowercase();
    let custom_validators = callbacks::parse_validators(validators.as_ref())?;
    let suite = suite_from_refute(refute.as_ref())?;
    let threads = if custom_validators.is_empty() { threads } else { 1 };
    drop(columns);
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let tabular = loaded.data;
        let n = tabular.row_count();
        let series = TimeSeriesData::try_new(
            tabular.storage().clone(),
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .map_err(py_err)?;

        let name_to_id = |nm: &str| -> PyResult<VariableId> {
            series
                .schema()
                .id_of(nm)
                .map_err(|e| CausalDataError::new_err(format!("unknown variable {nm}: {e}")))
        };
        let t_id = name_to_id(&treatment)?;
        let y_id = name_to_id(&outcome)?;
        let q = temporal_query_from_policy(
            &policy,
            t_id,
            y_id,
            treatment_lag,
            horizon_steps,
            active_level,
        )?;

        let mut builder = CausalAnalysis::builder()
            .series(series)
            .temporal_pag(graph.pag)
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
    treatment_lag=1,
    horizon_steps=1,
    active_level=1.0,
    policy="pulse",
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
    let accept = if accept_discovered {
        DiscoveryAccept::AutoAccept
    } else {
        DiscoveryAccept::Review
    };
    detach_catch(py, move || {
        let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
        let event = EventData::try_new(loaded.data.storage().clone(), Arc::from(event_times_ns))
            .map_err(py_err)?;
        let schema = event.schema().clone();
        let name_to_id = |nm: &str| {
            schema
                .id_of(nm)
                .map_err(|e| CausalDataError::new_err(format!("unknown variable {nm}: {e}")))
        };
        let t_id = name_to_id(&treatment)?;
        let y_id = name_to_id(&outcome)?;
        let q = temporal_query_from_policy(
            &policy,
            t_id,
            y_id,
            treatment_lag,
            horizon_steps,
            active_level,
        )?;
        let graph = if algorithm.is_none() {
            let mut g = TemporalDag::empty();
            for (src, slag, tgt, tlag) in &edges {
                let s =
                    ensure_lagged(&mut g, name_to_id(src)?, Lag::from_raw(*slag)).map_err(py_err)?;
                let t =
                    ensure_lagged(&mut g, name_to_id(tgt)?, Lag::from_raw(*tlag)).map_err(py_err)?;
                g.insert_directed(s, t).map_err(py_err)?;
            }
            Some(g)
        } else {
            None
        };
        let mut builder = CausalAnalysis::builder()
            .events(event, align_interval_ns)
            .temporal_query(q)
            .refute(suite)
            .custom_validators(custom_validators)
            .bootstrap_replicates(bootstrap)
            .discovery_ci(ci_impl);
        if let Some(g) = graph {
            builder = builder.temporal_graph(g);
        } else if let Some(algo) = algorithm.as_deref() {
            let algo = algo.to_ascii_lowercase();
            builder = match algo.as_str() {
                "pcmci" => builder.discover_pcmci(max_lag, alpha, fdr_ctrl, accept),
                "pcmci_plus" => builder.discover_pcmci_plus(max_lag, alpha, fdr_ctrl, accept),
                "lpcmci" => builder.discover_lpcmci(max_lag, alpha, fdr_ctrl, accept),
                "rpcmci" => {
                    let regimes = regimes.ok_or_else(|| {
                        PyValueError::new_err("analyze_events(algorithm='rpcmci') requires regimes=[…]")
                    })?;
                    let assign = RegimeAssignment::try_new(
                        regimes.into_iter().map(RegimeId::from_raw).collect::<Vec<_>>(),
                    )
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    builder.discover_rpcmci(max_lag, alpha, fdr_ctrl, accept, assign)
                }
                "dbn_posterior" => builder.discover_dbn_posterior(
                    max_lag, force_mcmc, n_chains, n_warmup, mcmc_draws,
                ),
                other => {
                    return Err(PyValueError::new_err(format!(
                        "EventFrame discovery algorithm {other:?} unsupported"
                    )));
                }
            };
        }
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
    treatment_lag=1,
    horizon_steps=1,
    active_level=1.0,
    policy="pulse",
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
        let name_to_id = |nm: &str| {
            panel
                .schema()
                .id_of(nm)
                .map_err(|e| CausalDataError::new_err(format!("unknown variable {nm}: {e}")))
        };
        let t_id = name_to_id(&treatment)?;
        let y_id = name_to_id(&outcome)?;
        let mut g = TemporalDag::empty();
        for (src, slag, tgt, tlag) in &edges {
            let s =
                ensure_lagged(&mut g, name_to_id(src)?, Lag::from_raw(*slag)).map_err(py_err)?;
            let t =
                ensure_lagged(&mut g, name_to_id(tgt)?, Lag::from_raw(*tlag)).map_err(py_err)?;
            g.insert_directed(s, t).map_err(py_err)?;
        }
        let q = temporal_query_from_policy(
            &policy,
            t_id,
            y_id,
            treatment_lag,
            horizon_steps,
            active_level,
        )?;
        let mut builder = CausalAnalysis::builder()
            .panel(panel)
            .temporal_graph(g)
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
        let analysis = builder.build()
            .map_err(py_err)?;
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
    fdr=true,
    accept_discovered=true,
    treatment_lag=1,
    horizon_steps=1,
    active_level=1.0,
    policy="pulse",
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
    time_dummy_ci=false
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
    let time_dummy_encoding = time_dummy_encoding.to_string();
        let algorithm = algorithm.to_string();
    detach_catch(py, move || {
        let mut units = Vec::with_capacity(batches.len());
        for (batch, unit_id) in batches.iter().zip(unit_ids.into_iter()) {
            let (series, _) = series_from_batch(batch)?;
            units.push(PanelUnit { unit_id, series });
        }
        let panel = PanelData::try_new(Arc::from(units)).map_err(py_err)?;
        let name_to_id = |nm: &str| {
            panel
                .schema()
                .id_of(nm)
                .map_err(|e| CausalDataError::new_err(format!("unknown variable {nm}: {e}")))
        };
        let t_id = name_to_id(&treatment)?;
        let y_id = name_to_id(&outcome)?;
        let q = temporal_query_from_policy(
            &policy,
            t_id,
            y_id,
            treatment_lag,
            horizon_steps,
            active_level,
        )?;
        let fdr_ctrl = if fdr { FdrControl::bh() } else { FdrControl::Off };
        let accept = if accept_discovered {
            DiscoveryAccept::AutoAccept
        } else {
            DiscoveryAccept::Review
        };
        let encoding = match time_dummy_encoding.as_str() {
            "integer" | "integer_index" | "index" => TimeDummyEncoding::IntegerIndex,
            "one_hot" | "onehot" | "oh" => TimeDummyEncoding::OneHot,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown time_dummy_encoding {other:?}"
                )));
            }
        };
        let space_mode = if space_dummy_ci {
            SpaceDummyCiMode::MultivariateBlock
        } else {
            SpaceDummyCiMode::ScalarOneHot
        };
        let time_mode = if time_dummy_ci {
            TimeDummyCiMode::MultivariateBlock
        } else {
            TimeDummyCiMode::ScalarOneHot
        };
        let mut context_ids = Vec::new();
        for cname in context_names.unwrap_or_default() {
            context_ids.push(panel.schema().id_of(&cname).map_err(py_err)?);
        }
        let multi_dataset = MultiDatasetConstraints {
            context_variables: Arc::from(context_ids),
            include_space_dummy,
            include_time_dummy,
            space_dummy_ci: space_mode,
            time_dummy_encoding: encoding,
            time_dummy_ci: time_mode,
            ..MultiDatasetConstraints::default()
        };
        let algo = algorithm.to_ascii_lowercase();
        let mut builder = CausalAnalysis::builder().panel(panel);
        builder = match algo.as_str() {
            "jpcmci_plus" | "jpcmci+" => {
                builder.discover_jpcmci_plus(max_lag, alpha, fdr_ctrl, accept, multi_dataset)
            }
            "pcmci" => builder.discover_pcmci(max_lag, alpha, fdr_ctrl, accept),
            "pcmci_plus" | "pcmci+" => {
                builder.discover_pcmci_plus(max_lag, alpha, fdr_ctrl, accept)
            }
            "lpcmci" => builder.discover_lpcmci(max_lag, alpha, fdr_ctrl, accept),
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown panel discovery algorithm {other:?};                      use jpcmci_plus|pcmci|pcmci_plus|lpcmci"
                )));
            }
        };
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
        let analysis = builder.build()
            .map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        analysis_result_from_run(&names, result)
    })
}

fn temporal_query_from_policy(
    policy: &str,
    t_id: VariableId,
    y_id: VariableId,
    treatment_lag: u32,
    horizon_steps: u32,
    active_level: f64,
) -> PyResult<TemporalEffectQuery> {
    let at = -i32::try_from(treatment_lag)
        .map_err(|_| PyValueError::new_err("treatment_lag too large"))?;
    match policy {
        "pulse" => Ok(TemporalEffectQuery::pulse(t_id, y_id, active_level)
            .with_policy(TemporalPolicy::pulse(at))
            .with_horizon_steps(horizon_steps)),
        "sustained" => {
            // Sustained from `-treatment_lag` through step 0; evaluate at `horizon_steps`.
            Ok(TemporalEffectQuery::sustained(t_id, y_id, 0, active_level)
                .with_policy(TemporalPolicy::sustained(at, 0))
                .with_horizon_steps(horizon_steps))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown temporal policy {other:?}; use pulse|sustained"
        ))),
    }
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
    let (mean_ite, n_units, noise_inference, n_assignments, unit_vec) = detach_catch(py, move || {
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
        let ite = facade_counterfactual_ite(fitted.model, &data, t_id, y_id, active, control, &ctx)
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
    let wrappers = mechanism_wrappers.map(|d| d.unbind());
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
    Ok(GcmSampleResult {
        column_means: means,
        n_draws: n_rows,
        n_nodes,
        draws,
    })
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
        let samples =
            facade_sample_interventional_distribution(&fitted.model, &query, n_draws, &mut rng, &ctx)
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
    Ok(GcmSampleResult {
        column_means: means,
        n_draws: n_rows,
        n_nodes,
        draws,
    })
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
        let mut ctx = py_execution_context(seed, threads);
        ctx.cache_policy = CachePolicy::enabled(Some(4_000_000));
        let opts = DistributionChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: n_samples.max(100),
            seed,
        };
        let result = facade_attribute_distribution_change(&fitted.model, &data, &query, &opts, &ctx)
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
        let mut ctx = py_execution_context(seed, threads);
        ctx.cache_policy = CachePolicy::enabled(Some(4_000_000));
        let opts = StructureChangeOptions {
            measure: DifferenceMeasure::MeanDiff,
            n_samples: n_samples.max(100),
            seed,
        };
        let result = facade_attribute_structure_change(
            &baseline,
            &comparison,
            &data,
            &query,
            &opts,
            &ctx,
        )
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
        let art =
            encode_posterior_wire(&meta, &artifact.draws, "py-posterior", VERSION).map_err(py_err)?;
        let mut buf = Vec::new();
        art.write_to(&mut buf).map_err(py_err)?;
        Ok(buf)
    })
}

/// Parse DOT digraph text; return `(node_count, edges)`.
#[pyfunction]
fn dag_from_dot(dot: &str) -> PyResult<(usize, Vec<(u32, u32)>)> {
    catch_ffi(|| {
        let dag = facade_dag_from_dot(dot).map_err(py_err)?;
        let wire = causal_io::dag_to_wire(&dag).map_err(py_err)?;
        Ok((wire.node_count as usize, wire.edges))
    })
}

/// Emit DOT for a numeric DAG given `node_count` and `edges`.
#[pyfunction]
fn dag_to_dot(node_count: u32, edges: Vec<(u32, u32)>) -> PyResult<String> {
    catch_ffi(|| {
        let wire = causal_io::DagWire { node_count, edges };
        let dag = causal_io::dag_from_wire(&wire).map_err(py_err)?;
        facade_dag_to_dot(&dag, None).map_err(py_err)
    })
}

/// Parsed JSON DAG: `(node_count, edges, variable_names)`.
type ParsedDagJson = (usize, Vec<(u32, u32)>, Option<Vec<String>>);

/// Parse JSON DAG document; return `(node_count, edges, variable_names|None)`.
#[pyfunction]
fn dag_from_json(json: &str) -> PyResult<ParsedDagJson> {
    catch_ffi(|| {
        let doc = causal_io::dag_json_from_str(json).map_err(py_err)?;
        let dag = causal_io::dag_from_wire(&doc.to_wire()).map_err(py_err)?;
        let _ = dag;
        Ok((doc.node_count as usize, doc.edges, doc.variable_names))
    })
}

/// Emit JSON for a numeric DAG.
#[pyfunction]
fn dag_to_json(
    node_count: u32,
    edges: Vec<(u32, u32)>,
    variable_names: Option<Vec<String>>,
) -> PyResult<String> {
    catch_ffi(|| {
        let wire = causal_io::DagWire { node_count, edges };
        let dag = causal_io::dag_from_wire(&wire).map_err(py_err)?;
        facade_dag_to_json(&dag, variable_names.as_deref()).map_err(py_err)
    })
}

/// Parse GML digraph text; return `(node_count, edges)`.
#[pyfunction]
fn dag_from_gml(gml: &str) -> PyResult<(usize, Vec<(u32, u32)>)> {
    catch_ffi(|| {
        let dag = causal::dag_from_gml(gml).map_err(py_err)?;
        let wire = causal_io::dag_to_wire(&dag).map_err(py_err)?;
        Ok((wire.node_count as usize, wire.edges))
    })
}

/// Emit GML for a numeric DAG.
#[pyfunction]
fn dag_to_gml(node_count: u32, edges: Vec<(u32, u32)>) -> PyResult<String> {
    catch_ffi(|| {
        let wire = causal_io::DagWire { node_count, edges };
        let dag = causal_io::dag_from_wire(&wire).map_err(py_err)?;
        causal::dag_to_gml(&dag, None).map_err(py_err)
    })
}

/// Parse NetworkX node-link JSON; return `(node_count, edges)`.
#[pyfunction]
fn dag_from_networkx_node_link(json: &str) -> PyResult<(usize, Vec<(u32, u32)>)> {
    catch_ffi(|| {
        let dag = causal::dag_from_networkx_node_link(json).map_err(py_err)?;
        let wire = causal_io::dag_to_wire(&dag).map_err(py_err)?;
        Ok((wire.node_count as usize, wire.edges))
    })
}

/// Emit NetworkX node-link JSON for a numeric DAG.
#[pyfunction]
fn dag_to_networkx_node_link(node_count: u32, edges: Vec<(u32, u32)>) -> PyResult<String> {
    catch_ffi(|| {
        let wire = causal_io::DagWire { node_count, edges };
        let dag = causal_io::dag_from_wire(&wire).map_err(py_err)?;
        causal::dag_to_networkx_node_link(&dag, None).map_err(py_err)
    })
}

/// Encode a minimal SCM model bundle (schema names + edges + mechanism slots).
///
/// `mechanisms` entries are `(kind, constant|intercept, coeffs|None, sigma|None)`
/// with `kind` in `{vacant, constant, linear_gaussian}`.
#[pyfunction]
fn encode_model_bundle(
    variable_names: Vec<String>,
    edges: Vec<(u32, u32)>,
    mechanisms: Vec<(String, Option<f64>, Option<Vec<f64>>, Option<f64>)>,
) -> PyResult<Vec<u8>> {
    catch_ffi(|| {
        use causal::{CompiledMechanismStore, MechanismSlot};
        use causal_core::{CausalSchemaBuilder, MeasurementSpec, SmallRoleSet, ValueType};
        use causal_io::{
            ModelBundleEncode, ModelBundleHeaderWire, ModelKindWire, encode_model_bundle as enc,
        };
        use std::sync::Arc;

        let mut b = CausalSchemaBuilder::new();
        for name in &variable_names {
            b.add_variable(
                name.as_str(),
                ValueType::Continuous,
                SmallRoleSet::empty(),
                None,
                None,
                MeasurementSpec::default(),
            )
            .map_err(|e| py_err(IoError::Convert(e.to_string())))?;
        }
        let schema = b.build().map_err(|e| py_err(IoError::Convert(e.to_string())))?;
        let wire = causal_io::DagWire {
            node_count: u32::try_from(variable_names.len()).unwrap_or(0),
            edges,
        };
        let dag = causal_io::dag_from_wire(&wire).map_err(py_err)?;
        let slots: Vec<MechanismSlot> = mechanisms
            .into_iter()
            .map(|(kind, constant, coeffs, sigma)| match kind.as_str() {
                "vacant" => MechanismSlot::Vacant,
                "constant" => MechanismSlot::Constant { value: constant.unwrap_or(0.0) },
                "linear_gaussian" => MechanismSlot::LinearGaussian {
                    intercept: constant.unwrap_or(0.0),
                    coeffs: Arc::from(coeffs.unwrap_or_default()),
                    sigma: sigma.unwrap_or(1.0),
                },
                _ => MechanismSlot::Vacant,
            })
            .collect();
        let store = CompiledMechanismStore { slots: slots.into() };
        let art = enc(ModelBundleEncode {
            header: ModelBundleHeaderWire { model_kind: ModelKindWire::Scm, label: None },
            schema: &schema,
            dag: &dag,
            mechanisms: &store,
            artifact_id: "py-model-bundle",
            contrast: None,
            query: None,
            analysis_trace: None,
            identification: None,
            estimate: None,
            refutations: None,
            logical_plan: None,
            physical_plan: None,
            performance: None,
            diagnostics: None,
            provenance: None,
            posterior: None,
            discovery: None,
        })
        .map_err(py_err)?;
        let mut buf = Vec::new();
        art.write_to(&mut buf).map_err(py_err)?;
        Ok(buf)
    })
}

/// Decode a model bundle; return `(variable_names, edges, n_mechanisms)`.
#[pyfunction]
fn decode_model_bundle(bytes: &[u8]) -> PyResult<(Vec<String>, Vec<(u32, u32)>, usize)> {
    catch_ffi(|| {
        let bundle = causal::decode_model_bundle_bytes(bytes).map_err(py_err)?;
        let names = bundle.schema.variables().iter().map(|v| v.name.to_string()).collect();
        let wire = causal_io::dag_to_wire(&bundle.dag).map_err(py_err)?;
        Ok((names, wire.edges, bundle.mechanisms.slots.len()))
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
    fdr=true,
    accept_discovered=true,
    treatment_lag=1,
    horizon_steps=1,
    active_level=1.0,
    policy="pulse",
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
    let accept = if accept_discovered {
        DiscoveryAccept::AutoAccept
    } else {
        DiscoveryAccept::Review
    };
    let context_names = context_names.unwrap_or_default();
    let space_dummy_ci = space_dummy_ci.to_string();
    let time_dummy_encoding = time_dummy_encoding.to_string();
    let time_dummy_ci = time_dummy_ci.to_string();
    let (ci_impl, _ci_name, is_ci_callback) = callbacks::resolve_ci_arg(ci.as_ref(), None)?;
    let threads = if is_ci_callback { 1 } else { threads };

    match algo.as_str() {
        "jpcmci_plus" => {
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
                batches.push(columns_to_batch(&names, cols)?);
            }
            drop(envs);
            drop(columns);
            detach_catch(py, move || {
                let mut series_list = Vec::with_capacity(batches.len());
                for batch in &batches {
                    let (series, _) = series_from_batch(batch)?;
                    series_list.push(series);
                }
                let multi = MultiEnvironmentData::try_new(Arc::from(series_list)).map_err(py_err)?;
                let t_id = multi.schema().id_of(&treatment).map_err(py_err)?;
                let y_id = multi.schema().id_of(&outcome).map_err(py_err)?;
                let mut context_ids = Vec::new();
                for cname in &context_names {
                    context_ids.push(multi.schema().id_of(cname).map_err(py_err)?);
                }
                let space_mode = match space_dummy_ci.as_str() {
                    "scalar" | "scalar_one_hot" | "one_hot" => SpaceDummyCiMode::ScalarOneHot,
                    "multivariate" | "multivariate_block" | "block" => {
                        SpaceDummyCiMode::MultivariateBlock
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "space_dummy_ci must be 'scalar' or 'multivariate', got '{other}'"
                        )));
                    }
                };
                let time_enc = match time_dummy_encoding.as_str() {
                    "integer" | "integer_index" | "index" => TimeDummyEncoding::IntegerIndex,
                    "one_hot" | "onehot" | "oh" => TimeDummyEncoding::OneHot,
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "time_dummy_encoding must be 'integer' or 'one_hot', got '{other}'"
                        )));
                    }
                };
                let time_mode = match time_dummy_ci.as_str() {
                    "scalar" | "scalar_one_hot" | "one_hot" => TimeDummyCiMode::ScalarOneHot,
                    "multivariate" | "multivariate_block" | "block" => {
                        TimeDummyCiMode::MultivariateBlock
                    }
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "time_dummy_ci must be 'scalar' or 'multivariate', got '{other}'"
                        )));
                    }
                };
                let multi_dataset = MultiDatasetConstraints {
                    context_variables: Arc::from(context_ids),
                    include_space_dummy,
                    include_time_dummy,
                    space_dummy_ci: space_mode,
                    time_dummy_encoding: time_enc,
                    time_dummy_ci: time_mode,
                    ..MultiDatasetConstraints::default()
                };
                let q = temporal_query_from_policy(
                    &policy,
                    t_id,
                    y_id,
                    treatment_lag,
                    horizon_steps,
                    active_level,
                )?;
                let analysis = CausalAnalysis::builder()
                    .series_multi(multi)
                    .temporal_query(q)
                    .refute(suite.clone())
                    .custom_validators(custom_validators.clone())
                    .refute(suite.clone())
                    .custom_validators(custom_validators.clone())
                    .bootstrap_replicates(bootstrap)
                    .discovery_ci(ci_impl)
                    .discover_jpcmci_plus(max_lag, alpha, fdr_ctrl, accept, multi_dataset)
                    .build()
                    .map_err(py_err)?;
                let ctx = py_execution_context(seed, threads);
                let result = analysis.run(&ctx).map_err(py_err)?;
                analysis_result_from_run(&names, result)
            })
        }
        "rpcmci" => {
            let regimes = regimes.ok_or_else(|| {
                PyValueError::new_err(
                    "analyze_temporal_discover(algorithm='rpcmci') requires regimes=[…]",
                )
            })?;
            let batch = columns_to_batch(&names, &columns)?;
            drop(columns);
            detach_catch(py, move || {
                let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
                let tabular = loaded.data;
                let n = tabular.row_count();
                if regimes.len() != n {
                    return Err(PyValueError::new_err(format!(
                        "regimes length {} != series length {n}",
                        regimes.len()
                    )));
                }
                let series = TimeSeriesData::try_new(
                    tabular.storage().clone(),
                    TimeIndex {
                        regularity: SamplingRegularity::Regular { interval_ns: 1 },
                        length: n,
                    },
                )
                .map_err(py_err)?;
                let assign = RegimeAssignment::try_new(
                    regimes
                        .into_iter()
                        .map(RegimeId::from_raw)
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let t_id = series.schema().id_of(&treatment).map_err(py_err)?;
                let y_id = series.schema().id_of(&outcome).map_err(py_err)?;
                let q = temporal_query_from_policy(
                    &policy,
                    t_id,
                    y_id,
                    treatment_lag,
                    horizon_steps,
                    active_level,
                )?;
                let analysis = CausalAnalysis::builder()
                    .series(series)
                    .temporal_query(q)
                    .refute(suite.clone())
                    .custom_validators(custom_validators.clone())
                    .bootstrap_replicates(bootstrap)
                    .discovery_ci(ci_impl)
                    .discover_rpcmci(max_lag, alpha, fdr_ctrl, accept, assign)
                    .build()
                    .map_err(py_err)?;
                let ctx = py_execution_context(seed, threads);
                let result = analysis.run(&ctx).map_err(py_err)?;
                analysis_result_from_run(&names, result)
            })
        }
        "pcmci" | "pcmci_plus" | "lpcmci" => {
            let batch = columns_to_batch(&names, &columns)?;
            drop(columns);
            detach_catch(py, move || {
                let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
                let tabular = loaded.data;
                let n = tabular.row_count();
                let series = TimeSeriesData::try_new(
                    tabular.storage().clone(),
                    TimeIndex {
                        regularity: SamplingRegularity::Regular { interval_ns: 1 },
                        length: n,
                    },
                )
                .map_err(py_err)?;

                let t_id = series.schema().id_of(&treatment).map_err(py_err)?;
                let y_id = series.schema().id_of(&outcome).map_err(py_err)?;
                let q = temporal_query_from_policy(
                    &policy,
                    t_id,
                    y_id,
                    treatment_lag,
                    horizon_steps,
                    active_level,
                )?;

                let mut builder = CausalAnalysis::builder()
                    .series(series)
                    .temporal_query(q)
                    .refute(suite.clone())
                    .custom_validators(custom_validators.clone())
                    .bootstrap_replicates(bootstrap)
                    .discovery_ci(ci_impl);
                builder = match algo.as_str() {
                    "pcmci" => builder.discover_pcmci(max_lag, alpha, fdr_ctrl, accept),
                    "pcmci_plus" => builder.discover_pcmci_plus(max_lag, alpha, fdr_ctrl, accept),
                    "lpcmci" => builder.discover_lpcmci(max_lag, alpha, fdr_ctrl, accept),
                    _ => unreachable!(),
                };
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
        "dbn_posterior" => {
            let batch = columns_to_batch(&names, &columns)?;
            drop(columns);
            detach_catch(py, move || {
                let loaded = tabular_from_record_batch(&batch).map_err(py_err)?;
                let tabular = loaded.data;
                let n = tabular.row_count();
                let series = TimeSeriesData::try_new(
                    tabular.storage().clone(),
                    TimeIndex {
                        regularity: SamplingRegularity::Regular { interval_ns: 1 },
                        length: n,
                    },
                )
                .map_err(py_err)?;
                let t_id = series.schema().id_of(&treatment).map_err(py_err)?;
                let y_id = series.schema().id_of(&outcome).map_err(py_err)?;
                let q = temporal_query_from_policy(
                    &policy,
                    t_id,
                    y_id,
                    treatment_lag,
                    horizon_steps,
                    active_level,
                )?;
                let mut builder = CausalAnalysis::builder()
                    .series(series)
                    .temporal_query(q)
                    .refute(suite)
                    .custom_validators(custom_validators)
                    .bootstrap_replicates(bootstrap)
                    .discover_dbn_posterior(max_lag, force_mcmc, n_chains, n_warmup, mcmc_draws);
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
        other => Err(PyValueError::new_err(format!(
            "unknown discovery algorithm {other:?}; use pcmci|pcmci_plus|lpcmci|jpcmci_plus|rpcmci|dbn_posterior"
        ))),
    }
}

fn analysis_result_from_run(
    names: &[String],
    result: causal::CausalAnalysisResult,
) -> PyResult<AnalysisResult> {
    let adjustment_set: Vec<String> = result
        .estimand
        .adjustment_set
        .iter()
        .map(|id| {
            names
                .get(id.as_usize())
                .cloned()
                .unwrap_or_else(|| format!("var{}", id.raw()))
        })
        .collect();
    let estimator_id = if result.posterior.is_some() {
        "bayesian.temporal.gcomp".to_string()
    } else {
        result
            .logical_plan
            .estimator
            .as_deref()
            .unwrap_or("temporal.linear.adjustment")
            .to_string()
    };
    let (
        posterior_effect_mean,
        posterior_effect_sd,
        posterior_q025,
        posterior_q975,
        posterior_n_draws,
        posterior_p_below_zero,
        posterior_backend,
        posterior_artifact,
        posterior_unidentified_mass,
    ) = if let Some(post) = result.posterior.as_ref() {
        let eq = post.effect_column().unwrap_or(0);
        let artifact = encode_causal_posterior_bytes(post, "temporal-analysis").map_err(py_err)?;
        let p_below = post.probability_below(0.0).map_err(py_estimate)?;
        (
            Some(post.summaries.mean[eq]),
            Some(post.summaries.sd[eq]),
            Some(post.summaries.q025[eq]),
            Some(post.summaries.q975[eq]),
            Some(post.draws.n_draws),
            Some(p_below),
            Some(post.diagnostics.backend_id.to_string()),
            Some(artifact),
            Some(post.unidentified_mass),
        )
    } else {
        (None, None, None, None, None, None, None, None, None)
    };
    Ok(AnalysisResult {
        ate: result.estimate.ate,
        se_analytic: result.estimate.se_analytic,
        se_bootstrap: result.estimate.se_bootstrap,
        plan_id: result.logical_plan.plan_id.to_string(),
        modality: format!("{:?}", result.logical_plan.data_classification),
        peak_memory_bytes: result.physical_plan.estimated_peak_memory_bytes,
        identification_status: format!("{:?}", result.identification.status),
        method: result.estimand.method.to_string(),
        diagnostics: result
            .diagnostics
            .iter()
            .map(|d| format!("{}: {}", d.code, d.message))
            .collect(),
        provenance_node_count: result.provenance.len(),
        refutation_count: result.refutations.len(),
        worker_threads: result.physical_plan.worker_threads,
        expected_python_crossings: result.physical_plan.expected_python_crossings,
        adjustment_set,
        assumption_count: result.estimate.assumptions.len(),
        derivation_step_count: result.identification.derivation.steps.len(),
        estimator_id,
        posterior_effect_mean,
        posterior_effect_sd,
        posterior_q025,
        posterior_q975,
        posterior_n_draws,
        posterior_p_below_zero,
        posterior_backend,
        posterior_artifact,
        posterior_unidentified_mass,
        mediation_total: result.mediation.as_ref().and_then(|m| m.total),
        mediation_direct: result.mediation.as_ref().and_then(|m| m.direct),
        mediation_mediated: result.mediation.as_ref().and_then(|m| m.mediated),
    })
}

fn apply_temporal_inference(
    mut builder: causal::CausalAnalysisBuilder,
    inference: Option<&str>,
    n_draws: usize,
    prior_scale: f64,
    prior_artifact: Option<&[u8]>,
) -> PyResult<causal::CausalAnalysisBuilder> {
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
    if let Some(bytes) = prior_artifact {
        let prior = causal::prior_set_from_posterior_bytes(bytes).map_err(py_err)?;
        cfg = cfg.prior(prior);
    }
    Ok(builder.inference(InferenceMode::Bayesian(cfg)))
}

/// Anomaly scores for listed outcomes.
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
        let scores = facade_anomaly_attribution(&fitted.model, &data, outcome_ids, max_u).map_err(py_err)?;
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
                gcm_api::AnomalyScores {
                    outcome: name,
                    mean_score: mean,
                    n_units: s.rows.len(),
                }
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
        let pairs: Vec<(causal_core::ComponentId, f64)> = result
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
        let opts = causal::RobustChangeOptions::default();
        let ctx = py_execution_context(seed, threads);
        let result = facade_attribute_distribution_change_robust(
            &fitted.model,
            &data,
            &query,
            &opts,
            &ctx,
        )
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
            PopulationSelector::TimeRange {
                start: baseline_start,
                end: baseline_end,
            },
            PopulationSelector::TimeRange {
                start: comparison_start,
                end: comparison_end,
            },
            0.05,
            data.schema().len(),
        );
        let detected = facade_mechanism_change_detection(
            &fitted.model,
            &data,
            &query,
            causal::MechanismChangeMethod::MeanDiff,
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
#[pyfunction]
fn dag_from_networkx_adjacency(json: &str) -> PyResult<(usize, Vec<(u32, u32)>)> {
    catch_ffi(|| {
        let dag = facade_dag_from_networkx_adjacency(json).map_err(py_err)?;
        let wire = causal_io::dag_to_wire(&dag).map_err(py_err)?;
        Ok((wire.node_count as usize, wire.edges))
    })
}

/// Emit NetworkX adjacency JSON for a numeric DAG.
#[pyfunction]
fn dag_to_networkx_adjacency(
    node_count: u32,
    edges: Vec<(u32, u32)>,
    variable_names: Option<Vec<String>>,
) -> PyResult<String> {
    catch_ffi(|| {
        let wire = causal_io::DagWire { node_count, edges };
        let dag = causal_io::dag_from_wire(&wire).map_err(py_err)?;
        facade_dag_to_networkx_adjacency(&dag, variable_names.as_deref()).map_err(py_err)
    })
}

/// Python module `causal._native`.

/// Conditional / context average effect (single effect modifier).
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
        let dag = build_static_dag(&data, &edges)?;
        let analysis = CausalAnalysis::builder()
            .data(data)
            .graph(dag)
            .causal_query(CausalQuery::ConditionalEffect(cq))
            .refute(suite)
            .custom_validators(custom_validators)
            .bootstrap_replicates(bootstrap)
            .build()
            .map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        ate_result_from_analysis(&names, result)
    })
}

/// Temporal linear mediation (treatment → mediator → outcome).
#[pyfunction]
#[pyo3(signature = (
    names, columns, edges, treatment, mediator, outcome, *,
    contrast="mediated", control_level=0.0, active_level=1.0,
    seed=1, bootstrap=0, threads=1
))]
fn analyze_temporal_mediation(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
    edges: Vec<(String, u32, String, u32)>,
    treatment: String,
    mediator: String,
    outcome: String,
    contrast: &str,
    control_level: f64,
    active_level: f64,
    seed: u64,
    bootstrap: u32,
    threads: u32,
) -> PyResult<AnalysisResult> {
    let batch = columns_to_batch(&names, &columns)?;
    let contrast = contrast.to_string();
    drop(columns);
    detach_catch(py, move || {
        let (series, _) = series_from_batch(&batch)?;
        let name_to_id = |nm: &str| {
            series
                .schema()
                .id_of(nm)
                .map_err(|e| CausalDataError::new_err(format!("unknown variable {nm}: {e}")))
        };
        let t_id = name_to_id(&treatment)?;
        let m_id = name_to_id(&mediator)?;
        let y_id = name_to_id(&outcome)?;
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
        let mut g = TemporalDag::empty();
        for (src, slag, tgt, tlag) in &edges {
            let s = ensure_lagged(&mut g, name_to_id(src)?, Lag::from_raw(*slag)).map_err(py_err)?;
            let t = ensure_lagged(&mut g, name_to_id(tgt)?, Lag::from_raw(*tlag)).map_err(py_err)?;
            g.insert_directed(s, t).map_err(py_err)?;
        }
        let analysis = CausalAnalysis::builder()
            .series(series)
            .temporal_graph(g)
            .causal_query(CausalQuery::Mediation(q))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(bootstrap)
            .build()
            .map_err(py_err)?;
        let ctx = py_execution_context(seed, threads);
        let result = analysis.run(&ctx).map_err(py_err)?;
        analysis_result_from_run(&names, result)
    })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CausalError", m.py().get_type::<CausalError>())?;
    m.add("CausalIdentifyError", m.py().get_type::<CausalIdentifyError>())?;
    m.add("CausalEstimateError", m.py().get_type::<CausalEstimateError>())?;
    m.add("CausalValidateError", m.py().get_type::<CausalValidateError>())?;
    m.add("CausalDiscoveryError", m.py().get_type::<CausalDiscoveryError>())?;
    m.add("CausalModelError", m.py().get_type::<CausalModelError>())?;
    m.add("CausalCounterfactualError", m.py().get_type::<CausalCounterfactualError>())?;
    m.add("CausalAttributionError", m.py().get_type::<CausalAttributionError>())?;
    m.add("CausalDataError", m.py().get_type::<CausalDataError>())?;
    m.add("CausalGraphError", m.py().get_type::<CausalGraphError>())?;
    m.add("CausalDesignError", m.py().get_type::<CausalDesignError>())?;
    m.add("CausalStateError", m.py().get_type::<CausalStateError>())?;
    m.add("CausalSerializationError", m.py().get_type::<CausalSerializationError>())?;
    m.add("CausalCompileError", m.py().get_type::<CausalCompileError>())?;
    m.add("CausalResourceError", m.py().get_type::<CausalResourceError>())?;
    m.add("CausalReviewError", m.py().get_type::<CausalReviewError>())?;
    m.add("CausalUnsupportedError", m.py().get_type::<CausalUnsupportedError>())?;
    m.add_function(wrap_pyfunction!(load_float64_columns, m)?)?;
    m.add_function(wrap_pyfunction!(load_float64_arrow_c_columns, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_pag, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_cpdag, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_admg, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_arrow_c, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_ate_discover, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_path_specific, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_conditional, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_temporal_mediation, m)?)?;
    m.add_function(wrap_pyfunction!(analyze, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_temporal_pag, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_events, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_panel, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_panel_discover, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_temporal_discover, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pcmci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pcmci_plus, m)?)?;
    m.add_function(wrap_pyfunction!(discover_pc, m)?)?;
    m.add_function(wrap_pyfunction!(discover_ges, m)?)?;
    m.add_function(wrap_pyfunction!(discover_lingam, m)?)?;
    m.add_function(wrap_pyfunction!(discover_notears, m)?)?;
    m.add_function(wrap_pyfunction!(discover_fci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_rfci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_lpcmci, m)?)?;
    m.add_function(wrap_pyfunction!(discover_jpcmci_plus, m)?)?;
    m.add_function(wrap_pyfunction!(discover_rpcmci, m)?)?;
    m.add_function(wrap_pyfunction!(two_regime_half_split_py, m)?)?;
    m.add_function(wrap_pyfunction!(mediation_effects_summary, m)?)?;
    m.add_function(wrap_pyfunction!(predict_intervened_summary, m)?)?;
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
    design_api::register(m)?;
    m.add_function(wrap_pyfunction!(evaluate_decision_py, m)?)?;
    m.add_function(wrap_pyfunction!(decode_posterior_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(encode_posterior_artifact, m)?)?;
    m.add_function(wrap_pyfunction!(dag_from_dot, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_dot, m)?)?;
    m.add_function(wrap_pyfunction!(dag_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(dag_from_gml, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_gml, m)?)?;
    m.add_function(wrap_pyfunction!(dag_from_networkx_node_link, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_networkx_node_link, m)?)?;
    m.add_function(wrap_pyfunction!(dag_from_networkx_adjacency, m)?)?;
    m.add_function(wrap_pyfunction!(dag_to_networkx_adjacency, m)?)?;
    m.add_function(wrap_pyfunction!(encode_model_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(decode_model_bundle, m)?)?;
    m.add_class::<ArrowLoadInfo>()?;
    m.add_class::<AteAnalysisResult>()?;
    m.add_class::<PosteriorArtifact>()?;
    m.add_class::<AnalysisResult>()?;
    m.add_class::<DiscoveredLink>()?;
    m.add_class::<GraphEdge>()?;
    m.add_class::<PcmciDiscoveryResult>()?;
    m.add_class::<RpcmciDiscoverySummary>()?;
    m.add_class::<MediationEffectsSummary>()?;
    m.add_class::<PredictSummary>()?;
    m.add_class::<GcmIteResult>()?;
    m.add_class::<GcmSampleResult>()?;
    m.add_class::<graphs::Dag>()?;
    m.add_class::<graphs::Cpdag>()?;
    m.add_class::<graphs::Pag>()?;
    m.add_class::<graphs::Admg>()?;
    m.add_class::<graphs::TemporalDag>()?;
    m.add_class::<graphs::TemporalCpdag>()?;
    m.add_class::<graphs::TemporalPag>()?;
    gcm_api::register(m)?;
    state_api::register(m)?;
    bayesian::register(m)?;
    stability::register(m)?;
    m.add("__version__", causal_core::VERSION)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::panic_payload_msg;
    use std::any::Any;

    #[test]
    fn panic_payload_formats_str_and_string() {
        let as_str: Box<dyn Any + Send> = Box::new("boom");
        assert_eq!(panic_payload_msg(as_str.as_ref()), "boom");
        let as_string: Box<dyn Any + Send> = Box::new(String::from("kaboom"));
        assert_eq!(panic_payload_msg(as_string.as_ref()), "kaboom");
        let other: Box<dyn Any + Send> = Box::new(42_u32);
        assert_eq!(panic_payload_msg(other.as_ref()), "unknown panic payload");
    }
}
