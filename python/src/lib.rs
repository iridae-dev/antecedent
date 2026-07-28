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
    clippy::too_many_lines,
    clippy::fn_params_excessive_bools,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

mod ate_api;
mod attribution_api;
mod bayesian;
mod callbacks;
mod design_api;
mod discovery_api;
mod gcm_api;
mod graph_build;
mod graph_io;
mod graphs;
mod prepared_api;
mod prior_bank;
mod stability;
mod state_api;
mod temporal_api;

pub(crate) use ate_api::{
    GraphEdge, ate_result_from_analysis, panel_discovery_builder, panel_multi_dataset_constraints,
};
pub(crate) use discovery_api::{
    DiscoveredLink, PcmciDiscoveryResult, series_from_batch, tabular_from_batch,
};
pub(crate) use graph_build::{
    dag_from_named_edges, parse_dummy_ci_modes, parse_time_dummy_encoding, schema_var_id,
    series_from_tabular, space_dummy_ci_from_bool, temporal_dag_from_lagged_edges,
    temporal_dag_from_schema_edges, time_dummy_ci_from_bool,
};
pub(crate) use temporal_api::{
    AnalysisResult, GcmIteResult, GcmSampleResult, MediationEffectsSummary, PredictSummary,
    RpcmciDiscoverySummary,
};

type MechanismWireEntry = (String, Option<f64>, Option<Vec<f64>>, Option<f64>);
type ModelBundleSummary = (Vec<String>, Vec<(u32, u32)>, usize);
type PriorSensitivityFields =
    (Option<Vec<f64>>, Option<Vec<f64>>, Option<Vec<f64>>, Option<Vec<f64>>);
type ConflictSummaryFields = (Option<Vec<String>>, Option<Vec<f64>>, Option<Vec<f64>>);

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use antecedent::design::{DecisionProblem, evaluate_decision as facade_evaluate_decision};
use antecedent::discovery::{
    DiscoverParams, DiscoveryPerformanceRecord, LaggedParent, MultiDatasetConstraints, PcSepsets,
    RegimeAssignment, ScoredLink, SpaceDummyCiMode, StaticDiscoverParams, TimeDummyCiMode,
    discover_fci as facade_discover_fci, discover_ges as facade_discover_ges,
    discover_jpcmci_plus as facade_discover_jpcmci_plus, discover_lingam as facade_discover_lingam,
    discover_lpcmci as facade_discover_lpcmci, discover_notears as facade_discover_notears,
    discover_pc as facade_discover_pc, discover_pcmci as facade_discover_pcmci,
    discover_pcmci_plus as facade_discover_pcmci_plus, discover_rfci as facade_discover_rfci,
    discover_rpcmci as facade_discover_rpcmci, pag_definite_directed_edge_count,
    two_regime_half_split,
};
use antecedent::estimate::{TemporalLinearPredictor, TemporalMediationEstimator};
use antecedent::gcm::{
    CompiledCausalModel, DifferenceMeasure, DistributionChangeOptions, StructureChangeOptions,
    anomaly_attribution as facade_anomaly_attribution,
    attribute_distribution_change as facade_attribute_distribution_change,
    attribute_distribution_change_robust as facade_attribute_distribution_change_robust,
    attribute_feature_relevance as facade_attribute_feature_relevance,
    attribute_path_specific as facade_attribute_path_specific,
    attribute_structure_change as facade_attribute_structure_change,
    attribute_unit_change as facade_attribute_unit_change,
    counterfactual_ite as facade_counterfactual_ite, fit_gcm,
    mechanism_change_detection as facade_mechanism_change_detection, sample_do as facade_sample_do,
    sample_interventional_distribution as facade_sample_interventional_distribution,
};
use antecedent::io::{
    dag_from_dot as facade_dag_from_dot,
    dag_from_networkx_adjacency as facade_dag_from_networkx_adjacency,
    dag_to_dot as facade_dag_to_dot, dag_to_json as facade_dag_to_json,
    dag_to_networkx_adjacency as facade_dag_to_networkx_adjacency, decode_causal_posterior_bytes,
    encode_causal_posterior_bytes,
};
use antecedent::{
    BayesianConfig, CausalError as RustCausalError, DiscoveryAccept, EstimatorId, FdrControl,
    GraphInput, IdentifierId, InferenceMode, RefuteSuite, Study,
};
use antecedent_core::{
    AllocationMethod, AttributionComponents, AverageEffectQuery, CachePolicy, CausalQuery,
    CausalRng, ChangeAttributionQuery, ConditionalEffectQuery, DistributionRef, ExecutionContext,
    Intervention, InterventionalDistributionQuery, KernelPolicy, Lag, MechanismChangeQuery,
    MediationContrast, MediationQuery, PathSpecificEffectQuery, PopulationRegistry,
    PopulationSelector, PredicateExpr, RegimeId, SchemaError, ShapleyConfig, TargetPopulation,
    TemporalEffectQuery, TemporalPolicy, UnitChangeQuery, VERSION, Value, VariableId,
};
use antecedent_data::TimeDummyEncoding;
use antecedent_data::{
    ArrowCColumn, DataError, EventData, MultiEnvironmentData, PanelData, PanelUnit, TableView,
    TimeSeriesData, tabular_from_arrow_c_columns, tabular_from_record_batch,
};
use antecedent_expr::{CausalExprArena, IdentifiedEstimand};
use antecedent_graph::{
    Cpdag, Dag, DenseNodeId, Endpoint, GraphError, MarkedEdge, MiddleMark, NodeRef, Pag,
    TemporalCpdag, TemporalPag,
};
use antecedent_io::{
    CausalPosteriorWire, IoError, PosteriorQuantityWire,
    encode_posterior_artifact as encode_posterior_wire,
};
use antecedent_stats::FdrAdjustment;
use antecedent_stats::PartialCorrelation;
use antecedent_validate::{PredictiveCheckKind, RefutationReport};
use arrow_array::{Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
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
create_exception!(causal._native, CausalCancelledError, CausalError);

/// Parse Python `refute=` — bool or suite name (`"full"` / `"placebo"` / `"none"`).
/// `None` (omitted kwarg) defaults to PlaceboAndRcc.
pub(crate) fn suite_from_refute(obj: Option<&Bound<'_, PyAny>>) -> PyResult<RefuteSuite> {
    let Some(obj) = obj else {
        return Ok(RefuteSuite::PlaceboAndRcc);
    };
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(if b { RefuteSuite::PlaceboAndRcc } else { RefuteSuite::None });
    }
    if let Ok(s) = obj.extract::<String>() {
        return match s.trim().to_ascii_lowercase().as_str() {
            "full" | "validation.full" => Ok(RefuteSuite::Full),
            "cheap" | "overlap" | "overlap+evalue" | "interactive" => Ok(RefuteSuite::Cheap),
            "placebo" | "placebo_and_rcc" | "placebo+rcc" | "true" | "1" => {
                Ok(RefuteSuite::PlaceboAndRcc)
            }
            "none" | "off" | "false" | "0" => Ok(RefuteSuite::None),
            other => Err(PyValueError::new_err(format!(
                "unknown refute={other:?}; use True|False|\"full\"|\"placebo\"|\"cheap\"|\"none\""
            ))),
        };
    }
    Err(PyValueError::new_err(
        "refute= must be bool or str (True|False|\"full\"|\"placebo\"|\"none\")",
    ))
}

trait IntoCausalPyErr {
    fn into_antecedent_py_err(self) -> PyErr;
}

pub(crate) fn py_err<E: IntoCausalPyErr>(e: E) -> PyErr {
    e.into_antecedent_py_err()
}

/// Fallback for domain errors not covered by [`IntoCausalPyErr`].
pub(crate) fn py_msg(e: impl ToString) -> PyErr {
    CausalError::new_err(e.to_string())
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

impl IntoCausalPyErr for RustCausalError {
    fn into_antecedent_py_err(self) -> PyErr {
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
                antecedent::state::StateError::CacheBudget { .. } => {
                    CausalResourceError::new_err(e.to_string())
                }
                _ => CausalStateError::new_err(e.to_string()),
            },
            Self::Schema(e) => CausalDataError::new_err(e.to_string()),
            Self::Compile { message } => CausalCompileError::new_err(message),
            Self::Resource { message } => CausalResourceError::new_err(message),
            Self::ReviewRequired { kind, algorithm, pending_edge_count, message, hint } => {
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
            Self::Cancelled { stage } => {
                CausalCancelledError::new_err(format!("cancelled during {stage}"))
            }
            // `CausalError` is `#[non_exhaustive]`: any variant added upstream maps to the
            // hierarchy root rather than failing the build. Give new variants an explicit
            // arm above when their Python-facing category is decided.
            ref other => CausalError::new_err(other.to_string()),
        }
    }
}

impl IntoCausalPyErr for DataError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for GraphError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for IoError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for SchemaError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for arrow_schema::ArrowError {
    fn into_antecedent_py_err(self) -> PyErr {
        CausalDataError::new_err(self.to_string())
    }
}

impl IntoCausalPyErr for antecedent_validate::ValidationError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for antecedent::state::StateError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for antecedent::estimate::EstimationError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for antecedent::gcm::ModelError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for antecedent::design::DesignError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
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
#[allow(clippy::struct_excessive_bools)] // FFI flat getters; effort flags are intentional
pub(crate) struct AteAnalysisResult {
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
    /// Per-refuter records (name, comparison statistic, pass/fail), one per validator run.
    #[pyo3(get)]
    refutations: Vec<RefutationReportView>,
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
    /// Discovery algorithm id from the logical plan, if any.
    #[pyo3(get)]
    discovery_algorithm: Option<String>,
    /// Whether graph review is required before estimation.
    #[pyo3(get)]
    graph_review_required: bool,
    /// Identifier algorithm id from the logical plan, if any.
    #[pyo3(get)]
    plan_identifier: Option<String>,
    /// Estimator id from the logical plan, if any.
    #[pyo3(get)]
    plan_estimator: Option<String>,
    /// Validation suite id from the logical plan, if any.
    #[pyo3(get)]
    validation_suite: Option<String>,
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
    prior_sensitivity_alphas: Option<Vec<f64>>,
    #[pyo3(get)]
    prior_sensitivity_means: Option<Vec<f64>>,
    #[pyo3(get)]
    prior_sensitivity_sds: Option<Vec<f64>>,
    #[pyo3(get)]
    conflict_source_ids: Option<Vec<String>>,
    #[pyo3(get)]
    conflict_alphas_requested: Option<Vec<f64>>,
    #[pyo3(get)]
    conflict_alphas_applied: Option<Vec<f64>>,
    #[pyo3(get)]
    posterior_unidentified_mass: Option<f64>,
    #[pyo3(get)]
    latency_mode: Option<String>,
    #[pyo3(get)]
    wall_time_ns: Option<u64>,
    #[pyo3(get)]
    bootstrap_replicates_requested: Option<u32>,
    #[pyo3(get)]
    bootstrap_replicates_ok: Option<u32>,
    #[pyo3(get)]
    n_draws_effort: Option<u32>,
    #[pyo3(get)]
    cancelled: bool,
    #[pyo3(get)]
    early_stopped: bool,
    #[pyo3(get)]
    stage_timings: Vec<(String, u64)>,
}

/// One refuter's record: which check ran, its comparison statistic, and pass/fail.
///
/// Exposes the fields of [`antecedent_validate::RefutationReport`] so callers can name the
/// refuter that failed instead of only seeing an aggregate pass/fail flag.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
struct RefutationReportView {
    /// Refuter id (e.g. `placebo.treatment`, `random.common_cause`, `bootstrap.ci_coverage`).
    #[pyo3(get)]
    refuter: String,
    /// Original ATE before refutation.
    #[pyo3(get)]
    original_ate: f64,
    /// Refuted / transformed ATE (mean across replicates when applicable).
    #[pyo3(get)]
    refuted_ate: f64,
    /// Scale-free comparison statistic (meaning is refuter-specific; see
    /// [`antecedent_validate::RefutationReport::comparison`]).
    #[pyo3(get)]
    comparison: f64,
    /// Whether the check is informative for the estimator used.
    #[pyo3(get)]
    informative: bool,
    /// Whether the check passed the configured threshold.
    #[pyo3(get)]
    passed: bool,
    /// Failure condition description when `passed` is `False`.
    #[pyo3(get)]
    failure_condition: Option<String>,
    /// Number of replicate estimates.
    #[pyo3(get)]
    replicates: u32,
}

impl From<&RefutationReport> for RefutationReportView {
    fn from(r: &RefutationReport) -> Self {
        Self {
            refuter: r.refuter.to_string(),
            original_ate: r.original_ate,
            refuted_ate: r.refuted_ate,
            comparison: r.comparison,
            informative: r.informative,
            passed: r.passed,
            failure_condition: r.failure_condition.as_ref().map(std::string::ToString::to_string),
            replicates: r.replicates,
        }
    }
}

#[pymethods]
impl RefutationReportView {
    fn __repr__(&self) -> String {
        format!(
            "RefutationReportView(refuter={:?}, passed={}, comparison={}, replicates={})",
            self.refuter, self.passed, self.comparison, self.replicates
        )
    }
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

#[pymethods]
impl PosteriorArtifact {
    #[new]
    #[pyo3(signature = (
        n_draws,
        mean,
        sd,
        q025,
        q975,
        draws,
        backend_id,
        identification,
        quantity_names,
        unidentified_mass=0.0,
        converged=true,
        hessian_condition=f64::NAN,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        n_draws: usize,
        mean: Vec<f64>,
        sd: Vec<f64>,
        q025: Vec<f64>,
        q975: Vec<f64>,
        draws: Vec<f64>,
        backend_id: String,
        identification: String,
        quantity_names: Vec<String>,
        unidentified_mass: f64,
        converged: bool,
        hessian_condition: f64,
    ) -> Self {
        Self {
            n_draws,
            mean,
            sd,
            q025,
            q975,
            draws,
            backend_id,
            identification,
            unidentified_mass,
            converged,
            hessian_condition,
            quantity_names,
        }
    }

    /// Build a summary-only artifact (mean / SD / quantiles, no draws).
    ///
    /// For callers who only hold posterior moments (e.g. from a conjugate update
    /// computed elsewhere) and would otherwise have to fabricate a fake draws array
    /// just to construct an artifact. `n_draws` records the draw count the moments
    /// were computed from; no samples are stored. `encode_posterior_artifact` emits
    /// `draws_encoding = "none"` for artifacts built this way, and
    /// `decode_posterior_artifact` round-trips them back with an empty `draws`.
    #[staticmethod]
    #[pyo3(signature = (
        n_draws,
        mean,
        sd,
        q025,
        q975,
        backend_id,
        identification,
        quantity_names,
        unidentified_mass=0.0,
        converged=true,
        hessian_condition=f64::NAN,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn from_moments(
        n_draws: usize,
        mean: Vec<f64>,
        sd: Vec<f64>,
        q025: Vec<f64>,
        q975: Vec<f64>,
        backend_id: String,
        identification: String,
        quantity_names: Vec<String>,
        unidentified_mass: f64,
        converged: bool,
        hessian_condition: f64,
    ) -> Self {
        Self {
            n_draws,
            mean,
            sd,
            q025,
            q975,
            draws: Vec::new(),
            backend_id,
            identification,
            unidentified_mass,
            converged,
            hessian_condition,
            quantity_names,
        }
    }
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
) -> PyResult<(antecedent_data::FfiArrowArray, antecedent_data::FfiArrowSchema)> {
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

    let schema_name = c"arrow_schema";
    let array_name = c"arrow_array";

    let schema_ptr = schema_cap
        .pointer_checked(Some(schema_name))?
        .as_ptr()
        .cast::<antecedent_data::FfiArrowSchema>();
    let array_ptr = array_cap
        .pointer_checked(Some(array_name))?
        .as_ptr()
        .cast::<antecedent_data::FfiArrowArray>();
    if schema_ptr.is_null() || array_ptr.is_null() {
        return Err(CausalDataError::new_err("null Arrow C Data capsule pointer"));
    }

    // SAFETY: capsules export valid CDI structs; we move them out and leave released empties
    // so the capsule destructor is a no-op.
    let schema = unsafe { std::ptr::read(schema_ptr) };
    let array = unsafe { std::ptr::read(array_ptr) };
    unsafe {
        std::ptr::write(schema_ptr, antecedent_data::FfiArrowSchema::empty());
        std::ptr::write(array_ptr, antecedent_data::FfiArrowArray::empty());
    }
    let _ = py;
    Ok((array, schema))
}

fn tabular_from_arrow_c_objs(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<antecedent_data::TabularData> {
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

/// Default coalition / semantic cache budget for Python production contexts
/// (matches attribution bench policy).
pub(crate) const PY_DEFAULT_CACHE_MAX_BYTES: u64 = 4_000_000;

pub(crate) fn py_execution_context(seed: u64, threads: u32) -> ExecutionContext {
    py_execution_context_ext(seed, threads, None, None, Some(PY_DEFAULT_CACHE_MAX_BYTES))
}

pub(crate) fn py_execution_context_ext(
    seed: u64,
    threads: u32,
    cancel: Option<antecedent_core::CancellationToken>,
    progress: Option<std::sync::Arc<dyn antecedent_core::ProgressSink>>,
    cache_max_bytes: Option<u64>,
) -> ExecutionContext {
    let mut ctx = ExecutionContext::production(seed, threads);
    ctx.cache_policy = CachePolicy::enabled(cache_max_bytes);
    if let Some(token) = cancel {
        ctx.cancellation = token;
    }
    ctx.progress = progress;
    ctx
}

/// Cooperative cancellation token shared with a running analysis.
#[pyclass(name = "CancellationToken", from_py_object)]
#[derive(Clone)]
pub struct PyCancellationToken {
    pub(crate) inner: antecedent_core::CancellationToken,
}

#[pymethods]
impl PyCancellationToken {
    #[new]
    fn new() -> Self {
        Self { inner: antecedent_core::CancellationToken::new() }
    }

    fn cancel(&self) {
        self.inner.cancel();
    }

    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_native_errors(m)?;
    register_native_functions(m)?;
    register_native_classes(m)?;
    gcm_api::register(m)?;
    state_api::register(m)?;
    prepared_api::register(m)?;
    bayesian::register(m)?;
    stability::register(m)?;
    prior_bank::register(m)?;
    m.add("__version__", antecedent_core::VERSION)?;
    Ok(())
}

fn register_native_errors(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    m.add("CausalCancelledError", m.py().get_type::<CausalCancelledError>())?;
    Ok(())
}

fn register_native_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_float64_columns, m)?)?;
    m.add_function(wrap_pyfunction!(load_float64_arrow_c_columns, m)?)?;
    ate_api::register(m)?;
    discovery_api::register(m)?;
    temporal_api::register(m)?;
    attribution_api::register(m)?;
    graph_io::register(m)?;
    design_api::register(m)?;
    Ok(())
}

fn register_native_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArrowLoadInfo>()?;
    m.add_class::<AteAnalysisResult>()?;
    m.add_class::<RefutationReportView>()?;
    m.add_class::<PyCancellationToken>()?;
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
