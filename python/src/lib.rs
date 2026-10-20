//! `PyO3` bindings for the Antecedent extension module (`antecedent._native`).
//!
//! Arrow and NumPy ingestion, graph handles, discovery, estimation, and the
//! prepared-analysis entry points live in the modules below. Unsafe FFI is
//! confined to [`ffi`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs)]
#![deny(clippy::disallowed_methods)]
#![allow(clippy::doc_markdown, reason = "docs quote Python names and signatures")]
#![allow(clippy::needless_pass_by_value, reason = "PyO3 extracts owned arguments from Python")]
#![allow(clippy::too_many_arguments, reason = "native signatures mirror the Python entry points")]
#![allow(
    clippy::too_many_lines,
    reason = "section builders and the module table are one function each"
)]
#![allow(
    clippy::fn_params_excessive_bools,
    reason = "effort and license flags are separate Python keywords"
)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

// The extension module does not link libpython. The test harness does.
#[cfg(test)]
include!(concat!(env!("OUT_DIR"), "/python_test_link.rs"));

mod adaptive_inference_api;
mod adaptive_participant_api;
mod adaptive_procedure_api;
mod array_fit_api;
mod artifact_api;
mod ate_api;
mod attribution_api;
mod batch_retarget_api;
pub(crate) mod bayesian;
mod bounds_api;
mod bridge;
mod callbacks;
mod candidate_screen_api;
mod categorical_treatment_api;
mod closed_pilot_api;
mod cohort_decision_api;
mod compact_export_api;
mod composition_api;
mod composition_bundle_api;
mod composition_verification_api;
mod contract_refusals_api;
mod counterfactual_id_api;
mod cpdag_scenario_api;
mod cross_world_api;
mod cross_world_execution_api;
mod cyclic_system_api;
mod decision_api;
mod decision_reliability_api;
#[allow(dead_code, reason = "signature literals are checked against this table in tests")]
mod defaults;
mod derived_treatment_api;
mod descriptive_api;
mod design_api;
mod design_ranking_api;
mod discovered_transport_api;
mod discovery_api;
mod discretized_composition_api;
mod distribution_api;
mod dose_grid_api;
mod effect_constancy_api;
mod effect_constancy_review_api;
mod estimator_config;
mod evidence_ledger_api;
mod execution_attempt_api;
mod experiment_api;
mod external_api;
mod factorial_api;
mod ffi;
mod forward_source_api;
mod functional_source_api;
mod gcm_api;
mod graph_build;
mod graph_falsification_api;
mod graph_io;
mod graphs;
mod identification_details;
mod inspect_receipt_api;
mod interference_saturation_api;
mod interrupt;
mod intrinsic_influence_api;
mod inverse_outcome_api;
mod inverse_query_api;
mod joint_law_propagation_api;
mod joint_transport_api;
mod latent_class_api;
mod learned_continuous_api;
mod learned_joint_api;
mod learned_trial_api;
mod matched_case_control_api;
mod measured_inference_api;
mod measurement_model_api;
mod mechanism_composition_api;
mod mechanism_discrepancy_api;
mod missing_report;
mod msm_sensitivity_api;
mod multi_level_treatment_api;
mod name_errors;
mod nested_bayesian_api;
mod nested_fisher_api;
mod nonlinear_mediation_api;
mod observation_api;
mod observation_query_api;
mod observational_interference_api;
mod observational_policy_api;
mod observational_survival_api;
mod policy_api;
mod predictive_assessment_api;
mod preflight_api;
mod prepared_api;
mod design_model_api;
mod prepared_options;
mod prior_bank;
mod prior_signal_api;
mod program_claims_api;
mod proposal_arrival_api;
mod provider_api;
mod quasi_api;
mod recalc_adjusted_api;
mod recalc_api;
mod recalc_bayesian_api;
mod recalc_bounds;
mod recalc_cell_api;
mod recalc_composite_api;
mod recalc_design_api;
mod recalc_dr_api;
mod recalc_external_api;
mod recalc_family_api;
mod recalc_static_api;
mod recalc_temporal_api;
mod recovery_api;
mod recovery_chain_api;
mod regimes_api;
mod unlicensed_design_api;
mod registry;
mod sequential_evaluation_api;
pub(crate) use registry::transport_defaults;
mod interface_contract_api;
mod intervention_callback_api;
mod intervention_program_api;
mod provider_callbacks_api;
mod repair_api;
mod response_api;
mod rollout_api;
mod scalar_decision_api;
mod scenario_invariance_api;
mod scientific_bridge_api;
mod selective_discovery_api;
mod sensitivity_decision_api;
mod sensitivity_source_api;
mod smoothed_dose_api;
mod source_evidence_api;
mod source_projection_api;
mod stability;
mod state_api;
mod study_planning_api;
mod posterior_boundary_api;
mod posterior_row_api;
mod survival_api;
mod period_refresh_api;
mod temporal_api;
mod temporal_counterfactual_api;
mod temporal_extension_api;
mod temporal_license;
mod temporal_transport_api;
mod transport_admg_conditional_api;
mod transport_common;
mod transport_counterfactual_api;
mod transport_counterfactual_sampled_api;
mod transport_exact_api;
mod transport_grid_api;
mod transport_interference_api;
mod transport_joint_sensitivity_api;
mod transport_mixed_api;
mod transport_mz_api;
mod transport_revalidation_api;
mod transport_scenario_api;
mod transport_statistical_api;
mod transport_z_api;
mod transported_constructors_api;
mod transported_counterfactual_api;
mod transported_response_api;
mod vector_treatment_api;

pub(crate) use ate_api::{GraphEdge, ate_result_from_analysis};
pub(crate) use discovery_api::{
    DiscoveredLink, PcmciDiscoveryResult, series_from_batch, tabular_from_batch,
};
pub(crate) use graph_build::{
    dag_from_named_edges, parse_dummy_ci_modes, require_named_graph_order, schema_var_id,
    series_from_tabular, temporal_dag_from_lagged_edges, temporal_dag_from_schema_edges,
};
pub(crate) use temporal_api::{
    AnalysisResult, GcmIteResult, GcmSampleResult, MediationEffectsSummary, PredictSummary,
    RpcmciDiscoverySummary,
};

pub(crate) fn parse_within_tier(
    within_tier: Option<&str>,
) -> PyResult<antecedent_graph::WithinTier> {
    let Some(raw) = within_tier else {
        return Err(crate::value_err("within_tier is required when tiers are supplied"));
    };
    match raw.to_ascii_lowercase().as_str() {
        "codetermined" => Ok(antecedent_graph::WithinTier::CoDetermined),
        "unknown" => Ok(antecedent_graph::WithinTier::Unknown),
        other => {
            Err(crate::value_err(format!("within_tier must be codetermined|unknown, got {other}")))
        }
    }
}

type MechanismWireEntry = (String, Option<f64>, Option<Vec<f64>>, Option<f64>);
type ModelBundleSummary = (Vec<String>, Vec<(u32, u32)>, usize);
type PriorSensitivityFields = (
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<Vec<f64>>,
    Option<String>,
    Option<Vec<f64>>,
);
type ConflictSummaryFields = (Option<Vec<String>>, Option<Vec<f64>>, Option<Vec<f64>>);

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use antecedent::design::{DecisionProblem, evaluate_decision as facade_evaluate_decision};
use antecedent::discovery::{
    DiscoverParams, DiscoveryPerformanceRecord, LaggedParent, MultiDatasetConstraints, PcSepsets,
    RegimeAssignment, ScoredLink, StaticDiscoverParams, discover_fci as facade_discover_fci,
    discover_ges as facade_discover_ges, discover_jpcmci_plus as facade_discover_jpcmci_plus,
    discover_lingam as facade_discover_lingam, discover_lpcmci as facade_discover_lpcmci,
    discover_lpcmci_panel as facade_discover_lpcmci_panel,
    discover_notears_with as facade_discover_notears, discover_pc as facade_discover_pc,
    discover_pcmci as facade_discover_pcmci, discover_pcmci_panel as facade_discover_pcmci_panel,
    discover_pcmci_plus as facade_discover_pcmci_plus,
    discover_pcmci_plus_panel as facade_discover_pcmci_plus_panel,
    discover_rfci as facade_discover_rfci, discover_rpcmci as facade_discover_rpcmci,
    pag_definite_directed_edge_count, two_regime_half_split,
};
use antecedent::error::PendingEdge as FacadePendingEdge;
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
use antecedent::io::{decode_causal_posterior_bytes, encode_causal_posterior_bytes};
use antecedent::review::{PendingCpdagReview, PendingGraphReview};
use antecedent::{
    AcceptedGraph, BayesianConfig, CausalError as RustCausalError, InferenceMode, RefuteSuite,
    Study,
};
use antecedent_core::{
    AllocationMethod, AttributionComponents, CachePolicy, CausalRng, ChangeAttributionQuery,
    DistributionRef, ExecutionContext, Intervention, InterventionalDistributionQuery, KernelPolicy,
    Lag, MechanismChangeQuery, MediationContrast, MediationQuery, PathSpecificEffectQuery,
    PopulationRegistry, PopulationSelector, PredicateExpr, RegimeId, SchemaError, ShapleyConfig,
    TargetPopulation, TemporalEffectQuery, TemporalPolicy, UnitChangeQuery, VERSION, Value,
    VariableId,
};
use antecedent_data::{
    DataError, MultiEnvironmentData, TableView, TimeSeriesData, tabular_from_arrow_c_columns,
    tabular_from_record_batch,
};
use antecedent_expr::{CausalExprArena, IdentifiedEstimand};
use antecedent_graph::{
    Cpdag, CpdagReview, Dag, DagReview, DenseNodeId, Endpoint, GraphError, MarkedEdge, MiddleMark,
    NodeRef, Pag, PagReview, TemporalCpdag, TemporalCpdagReview, TemporalGraphReview, TemporalPag,
    TemporalPagReview,
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
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

create_exception!(antecedent._native, CausalError, PyException);
create_exception!(antecedent._native, CausalIdentifyError, CausalError);
create_exception!(antecedent._native, CausalEstimateError, CausalError);
create_exception!(antecedent._native, CausalValidateError, CausalError);
create_exception!(antecedent._native, CausalDiscoveryError, CausalError);
create_exception!(antecedent._native, CausalModelError, CausalError);
create_exception!(antecedent._native, CausalCounterfactualError, CausalError);
create_exception!(antecedent._native, CausalAttributionError, CausalError);
create_exception!(antecedent._native, CausalDataError, CausalError);
create_exception!(antecedent._native, CausalGraphError, CausalError);
create_exception!(antecedent._native, CausalDesignError, CausalError);
create_exception!(antecedent._native, CausalStateError, CausalError);
create_exception!(antecedent._native, CausalSerializationError, CausalError);
create_exception!(antecedent._native, CausalCompileError, CausalError);
create_exception!(antecedent._native, CausalResourceError, CausalError);
create_exception!(antecedent._native, CausalReviewError, CausalError);
create_exception!(antecedent._native, CausalUnsupportedError, CausalError);
create_exception!(antecedent._native, CausalCancelledError, CausalError);

/// One entry in a raised `ReviewRequired`'s `pending_edges` sequence.
///
/// Read-only wire type mirroring `antecedent::error::PendingEdge`, exposed to Python
/// as plain attributes so `antecedent.errors.PendingEdge` (a frozen dataclass) can
/// normalize instances of this into its own type without depending on this class's
/// identity.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct CausalPendingEdge {
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    target: String,
    #[pyo3(get)]
    at_source: String,
    #[pyo3(get)]
    at_target: String,
}

impl From<&FacadePendingEdge> for CausalPendingEdge {
    fn from(edge: &FacadePendingEdge) -> Self {
        Self {
            source: edge.source.clone(),
            target: edge.target.clone(),
            at_source: edge.at_source.clone(),
            at_target: edge.at_target.clone(),
        }
    }
}

/// The Python-defined `ReviewRequired` class, registered by `antecedent.errors` at
/// import time via [`set_review_error_class`]. `None` when the native module is used
/// standalone (e.g. `import antecedent._native` without the `antecedent` package),
/// in which case [`review_required_py_err`] falls back to bare `CausalReviewError`.
static REVIEW_ERROR_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Register the Python `ReviewRequired` class the error mapper should instantiate for
/// `CausalError::ReviewRequired`, collapsing what used to be ad hoc `setattr` calls at
/// each Python-facing construction site into this one registration.
///
/// Called once by `antecedent.errors` at import time. A second call is a silent
/// no-op: [`PyOnceLock`] only accepts the first `set`, and the intended caller is a
/// single top-level import.
#[pyfunction]
fn set_review_error_class(py: Python<'_>, cls: Py<PyAny>) {
    let _ = REVIEW_ERROR_CLASS.get_or_init(py, || cls);
}

/// Build the Python exception for `CausalError::ReviewRequired`.
///
/// Instantiates the registered `ReviewRequired` class (see [`set_review_error_class`])
/// when available, so callers get the real class in `type(err).__mro__` plus a
/// structured `pending_edges` attribute; falls back to bare `CausalReviewError` when
/// nothing is registered. Either way this is the single place that builds a
/// review-required Python exception — `kind` / `algorithm` / `pending_edge_count` /
/// `pending_edges` / `hint` / `message` are attached identically in both branches, and
/// `str(err)` stays byte-identical to `message`.
fn review_required_py_err(
    kind: String,
    algorithm: Option<String>,
    pending_edge_count: usize,
    pending_edges: Arc<[FacadePendingEdge]>,
    message: String,
    hint: String,
) -> PyErr {
    Python::attach(|py| {
        let edges: Vec<Py<CausalPendingEdge>> = pending_edges
            .iter()
            .filter_map(|e| Py::new(py, CausalPendingEdge::from(e)).ok())
            .collect();

        let err: PyErr = REVIEW_ERROR_CLASS
            .get(py)
            .and_then(|cls| cls.bind(py).call1((message.as_str(),)).ok())
            .map_or_else(|| CausalReviewError::new_err(message.clone()), PyErr::from_value);

        let inst = err.value(py);
        let _ = inst.setattr("kind", kind.as_str());
        let _ = inst.setattr("algorithm", algorithm.as_deref());
        let _ = inst.setattr("pending_edge_count", pending_edge_count);
        let _ = inst.setattr("pending_edges", edges);
        let _ = inst.setattr("hint", hint.as_str());
        let _ = inst.setattr("message", message.as_str());
        err
    })
}

/// The Python `CausalUnsupportedError` subclass, registered by `antecedent.errors`.
static UNSUPPORTED_ERROR_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Register the Python `CausalUnsupportedError` the error mapper instantiates.
///
/// Keeps one refusal class across the boundary: a refusal raised in Rust is the
/// same class, with the same `reason_code`, as one raised in Python.
#[pyfunction]
fn set_unsupported_error_class(py: Python<'_>, cls: Py<PyAny>) {
    let _ = UNSUPPORTED_ERROR_CLASS.get_or_init(py, || cls);
}

/// The Python `CausalValueError` subclass, registered by `antecedent.errors`.
static VALUE_ERROR_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Register the Python `CausalValueError` invalid-input failures are raised as.
#[pyfunction]
fn set_value_error_class(py: Python<'_>, cls: Py<PyAny>) {
    let _ = VALUE_ERROR_CLASS.get_or_init(py, || cls);
}

/// Invalid input at a native entry point, as the registered `CausalValueError`
/// (a `CausalValidateError` that is also a `ValueError`).
///
/// The message is unchanged. A registered `reason=<code>:` prefix already in
/// the message is copied onto `reason_code`.
pub(crate) fn value_err(message: impl AsRef<str>) -> PyErr {
    let message = message.as_ref();
    attach_reason_from_message(value_error_of(message), message)
}

/// [`value_err`] with an explicit registered reason code.
///
/// The code is an attribute. It is not written into the message.
pub(crate) fn value_err_code(code: &str, message: impl AsRef<str>) -> PyErr {
    with_reason_code(value_error_of(message.as_ref()), code)
}

fn value_error_of(message: &str) -> PyErr {
    Python::attach(|py| {
        VALUE_ERROR_CLASS
            .get(py)
            .and_then(|cls| cls.bind(py).call1((message,)).ok())
            .map_or_else(|| CausalValidateError::new_err(message.to_string()), PyErr::from_value)
    })
}

/// A serialization failure, as `CausalSerializationError`.
///
/// A registered `reason=<code>:` prefix already in the message is copied onto
/// `reason_code`. The message text is the argument unchanged.
#[allow(
    clippy::disallowed_methods,
    reason = "this is the constructor the serialization ban redirects callers to"
)]
pub(crate) fn serialization_err(message: impl AsRef<str>) -> PyErr {
    let message = message.as_ref();
    attach_reason_from_message(CausalSerializationError::new_err(message.to_string()), message)
}

/// A refusal, as the registered Python class, carrying its `reason=<code>:` prefix.
///
/// Every refusal raised in this crate goes through here, so a caller catches one
/// class and reads one `reason_code` whether Rust or Python refused.
pub(crate) fn refusal(code: &str, message: impl AsRef<str>) -> PyErr {
    let prefix = antecedent_core::reason_code::PREFIX;
    unsupported_py_err(format!("{prefix}{code}: {}", message.as_ref()))
}

/// A refusal message with no registered reason code, as the Python class.
pub(crate) fn unsupported_message(message: impl AsRef<str>) -> PyErr {
    unsupported_py_err(message.as_ref().to_owned())
}

/// Build the Python exception for a refusal, carrying its `reason=<code>:` prefix.
fn unsupported_py_err(message: String) -> PyErr {
    // A support-matrix refusal renders its verdict first: `refused: reason=<code>: …`.
    let reason = reason_code_in_message(&message).map(str::to_string);
    Python::attach(|py| {
        let err: PyErr = UNSUPPORTED_ERROR_CLASS
            .get(py)
            .and_then(|cls| cls.bind(py).call1((message.as_str(),)).ok())
            .map_or_else(|| CausalUnsupportedError::new_err(message.clone()), PyErr::from_value);
        if let Some(code) = reason {
            let _ = err.value(py).setattr("reason_code", code.as_str());
        }
        err
    })
}

/// Attach a registered runtime-refusal reason code to a raised exception.
///
/// Callers pass a code checked by `antecedent_core::reason_code!`, or one read
/// back from a reason-coded Rust message, so an unregistered code is never
/// attached.
pub(crate) fn with_reason_code(err: PyErr, code: &str) -> PyErr {
    Python::attach(|py| {
        let _ = err.value(py).setattr("reason_code", code);
        err
    })
}

/// Attach a refusal's structured `remedy` (what the caller can change to proceed).
///
/// Additive: set only when the refusal names one, so every other exception reads the
/// `remedy = None` class default `antecedent.errors` installs on `CausalError`.
fn with_remedy(err: PyErr, remedy: Option<&str>) -> PyErr {
    if let Some(remedy) = remedy {
        Python::attach(|py| {
            let _ = err.value(py).setattr("remedy", remedy);
        });
    }
    err
}

/// Attach the route-native diagnostics a refusal carries (`route_diagnostics`).
///
/// Each record is a dict with `route`, `outcome`, `failed`, and `unevaluable_mass`. An
/// unreported mass is `None`, never `0.0`. Additive: set only when the refusal carries
/// records, so every other exception reads the empty-tuple class default.
fn with_route_diagnostics(
    err: PyErr,
    records: &[antecedent_core::diagnostic::RouteNativeDiagnostic],
) -> PyErr {
    if records.is_empty() {
        return err;
    }
    Python::attach(|py| {
        let rows: Vec<serde_json::Value> = records
            .iter()
            .map(|record| {
                serde_json::json!({
                    "route": record.route_name(),
                    "outcome": record.outcome_name(),
                    "failed": record.failed(),
                    "unevaluable_mass": record.unevaluable_mass().get(),
                })
            })
            .collect();
        if let Ok(list) = py
            .import("json")
            .and_then(|json| json.call_method1("loads", (serde_json::Value::from(rows).to_string(),)))
        {
            let _ = err.value(py).setattr("route_diagnostics", list);
        }
    });
    err
}

/// [`with_reason_code`] unless the exception already carries a finer code.
fn with_default_reason_code(err: PyErr, code: &str) -> PyErr {
    Python::attach(|py| {
        let value = err.value(py);
        let present = value.getattr("reason_code").ok().is_some_and(|existing| !existing.is_none());
        if !present {
            let _ = value.setattr("reason_code", code);
        }
        err
    })
}

/// The Python `EffectNotIdentified` class, registered by `antecedent.errors`.
static NOT_IDENTIFIED_ERROR_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Register the Python class a not-identified refusal is raised as.
#[pyfunction]
fn set_not_identified_error_class(py: Python<'_>, cls: Py<PyAny>) {
    let _ = NOT_IDENTIFIED_ERROR_CLASS.get_or_init(py, || cls);
}

/// A not-identified refusal carrying its identification outcome.
fn not_identified_py_err(status: &str, search_capped: bool, message: String) -> PyErr {
    Python::attach(|py| {
        let err: PyErr = NOT_IDENTIFIED_ERROR_CLASS
            .get(py)
            .and_then(|cls| cls.bind(py).call1((message.as_str(),)).ok())
            .map_or_else(|| CausalCompileError::new_err(message.clone()), PyErr::from_value);
        let value = err.value(py);
        let _ =
            value.setattr("reason_code", antecedent_core::reason_code!("effect_not_identified"));
        let _ = value.setattr("identification_status", status);
        let _ = value.setattr("search_capped", search_capped);
        let _ = value.setattr("search_complete", !search_capped);
        err
    })
}

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
            other => Err(crate::value_err(format!(
                "unknown refute={other:?}; use True|False|\"full\"|\"placebo\"|\"cheap\"|\"none\""
            ))),
        };
    }
    Err(crate::value_err("refute= must be bool or str (True|False|\"full\"|\"placebo\"|\"none\")"))
}

// --- Discovery review acceptance helpers -----------------------------------------
//
// The old `StudyBuilder` took a `DiscoveryAccept` flag (`AutoAccept` / `Review`) on each
// `.discover_*()` setter and deferred the actual review gate to `compile()` time. The new
// facade runs discovery standalone (see `antecedent::discovery`) and requires callers to
// explicitly turn the resulting review artifact into an `AcceptedGraph` — via
// `AcceptedGraph::accept(review)` (fallible; the review-artifact path) or the infallible
// `From<Dag|Admg|Pag|TemporalDag|TemporalPag>` conversions.
//
// These helpers reproduce the Python-visible `accept_discovered: bool` semantics on top of
// the new API, one per review-artifact shape:
// - DAG-shaped (`DagReview`, `TemporalGraphReview`): `true` clears pending directed edges
//   first (mirrors the old `DiscoveryAccept::AutoAccept`); `false` requires the graph to
//   already be fully oriented.
// - CPDAG-shaped (`CpdagReview`): `true` clears directed pending edges. Undirected
//   marks are the MEC and do not block acceptance (class-aware ATE consumes them).
//   `TemporalCpdagReview` is the same for undirected marks (class-aware pulse).
// - PAG-shaped (`PagReview`, `TemporalPagReview`): circle marks are informational, not
//   incompleteness (the class-aware identifiers handle them directly), so `true` accepts
//   the discovered graph as-is via the infallible `From` conversion — mirroring the old
//   unconditional `AutoAccept` success for FCI/RFCI/LPCMCI; `false` routes through the
//   review artifact, which fails with `ReviewRequired` while circles remain unreviewed.

/// DAG-shaped review (static `DagReview`; DirectLiNGAM, NOTEARS).
pub(crate) fn accept_dag_review(
    review: DagReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    let review = if accept_discovered { review.accept_all() } else { review };
    AcceptedGraph::accept(review)
}

/// Temporal DAG-shaped review (`TemporalGraphReview`; PCMCI).
pub(crate) fn accept_temporal_graph_review(
    review: TemporalGraphReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    let pending = PendingGraphReview::new(review);
    let pending = if accept_discovered { pending.accept_all() } else { pending };
    pending.finish()
}

/// Static CPDAG-shaped review (`CpdagReview`; PC, GES).
pub(crate) fn accept_cpdag_review(
    mut review: CpdagReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    if accept_discovered {
        review.pending_edges = std::sync::Arc::from([]);
    }
    AcceptedGraph::accept(review)
}

/// Temporal CPDAG-shaped review (`TemporalCpdagReview`; PCMCI+, J-PCMCI+, RPCMCI).
pub(crate) fn accept_temporal_cpdag_review(
    review: TemporalCpdagReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    let pending = PendingCpdagReview::new(review);
    let pending = if accept_discovered { pending.accept_all_directed() } else { pending };
    pending.finish()
}

/// Static PAG-shaped discovery output (FCI, RFCI).
pub(crate) fn accept_pag_review(
    graph: Pag,
    review: PagReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    // Static PAG circle marks are information the class-aware generalized-adjustment
    // identifier consumes, so auto-accept is safe here.
    if accept_discovered { Ok(AcceptedGraph::pag(graph)) } else { AcceptedGraph::accept(review) }
}

/// Temporal PAG-shaped discovery output (LPCMCI).
pub(crate) fn accept_temporal_pag_review(
    graph: TemporalPag,
    review: TemporalPagReview,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    // Circles are TemporalPag class information; auto-accept keeps the PAG.
    if accept_discovered {
        Ok(AcceptedGraph::temporal_pag(graph))
    } else {
        AcceptedGraph::accept(review)
    }
}

/// RPCMCI: reduce N per-regime CPDAG reviews to one accepted graph. A single accepted
/// graph is only possible when discovery found exactly one regime — multiple regimes
/// always require manual review (there is no single-graph collapse across regimes),
/// mirroring the old `compile()`'s `result.per_regime.len() == 1` gate.
pub(crate) fn accept_rpcmci_review(
    result: &antecedent::discovery::RpcmciDiscoveryResult,
    accept_discovered: bool,
) -> Result<AcceptedGraph, RustCausalError> {
    let Some(first) = result.per_regime.first() else {
        return Err(RustCausalError::review_required_msg("RPCMCI discovered no regime graphs"));
    };
    if result.per_regime.len() != 1 {
        return Err(RustCausalError::review_required_msg(format!(
            "RPCMCI discovered {} regimes; a single accepted graph requires exactly one \
             (review each regime's CPDAG separately)",
            result.per_regime.len()
        )));
    }
    accept_temporal_cpdag_review(first.review.clone(), accept_discovered)
}

trait IntoCausalPyErr {
    fn into_antecedent_py_err(self) -> PyErr;
}

pub(crate) fn py_err<E: IntoCausalPyErr>(e: E) -> PyErr {
    e.into_antecedent_py_err()
}

/// A `usize` index that has to cross the binding as a `u32` id.
///
/// Failure is an input error. It is not saturated and not zero.
pub(crate) fn fit_u32(index: usize) -> PyResult<u32> {
    u32::try_from(index).map_err(|_| value_err(format!("index {index} does not fit in u32")))
}

/// Fallback for domain errors not covered by [`IntoCausalPyErr`].
///
/// Stays a root `CausalError` (these are not invalid-argument failures). A
/// registered `reason=<code>:` prefix in the message is copied onto `reason_code`.
pub(crate) fn py_msg(e: impl ToString) -> PyErr {
    let message = e.to_string();
    attach_reason_from_message(domain_err(&message), &message)
}

#[allow(
    clippy::disallowed_methods,
    reason = "root CausalError is the domain fallback; invalid input uses value_err"
)]
fn domain_err(message: &str) -> PyErr {
    CausalError::new_err(message.to_string())
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

/// Release the GIL for native work, convert any panic into [`CausalError`], and let Ctrl-C in the
/// calling thread cancel the run (see [`interrupt`]).
pub(crate) fn detach_catch<F, T>(py: Python<'_>, f: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T> + Send,
    T: Send,
{
    interrupt::run(py, f)
}

/// The registered reason code a converted [`RustCausalError`] carries, if it has one.
///
/// The one owner of which native refusal raises which code: [`IntoCausalPyErr`] attaches
/// exactly this, so a caller reads `reason_code` off the exception and never parses the
/// message. A code the error names in its own `reason=<code>:` message wins over the
/// variant's default; an unregistered code is never attached.
fn reason_code_of(error: &RustCausalError) -> Option<String> {
    use antecedent_core::reason_code;
    let message = error.to_string();
    let named = reason_code_in_message(&message)
        .filter(|code| reason_code::is_registered(code))
        .map(str::to_string);
    let default = match error.peeled() {
        RustCausalError::Graph(e) => Some(if matches!(e, GraphError::UnknownVariableName { .. }) {
            reason_code!("unknown_variable")
        } else {
            reason_code!("graph_invalid")
        }),
        RustCausalError::Schema(
            SchemaError::UnknownVariableName { .. } | SchemaError::UnknownVariableId { .. },
        ) => Some(reason_code!("unknown_variable")),
        RustCausalError::SchemaMismatch { .. } => Some(reason_code!("schema_mismatch")),
        RustCausalError::NotIdentified { .. } => Some(reason_code!("effect_not_identified")),
        RustCausalError::Unsupported { .. } => Some(reason_code!("route_not_supported")),
        // A matrix refusal without a finer registered code carries the code of its cell verdict.
        RustCausalError::Support { id, .. } => Some(if id.as_str() == "not_applicable" {
            reason_code!("cell_not_applicable")
        } else {
            reason_code!("cell_not_licensed")
        }),
        RustCausalError::Missing { .. } => Some(reason_code!("required_option_missing")),
        RustCausalError::Conflict { .. } => Some(reason_code!("invalid_argument")),
        _ => None,
    };
    named.or_else(|| default.map(str::to_string))
}

/// The registered-shape `reason=<code>:` a message opens with, directly or after one
/// `<label>: ` (a matrix refusal renders its verdict first: `refused: reason=<code>: ...`).
fn reason_code_in_message(message: &str) -> Option<&str> {
    antecedent_core::reason_code::split_prefix(message)
        .or_else(|| {
            message
                .split_once(": ")
                .and_then(|(_, rest)| antecedent_core::reason_code::split_prefix(rest))
        })
        .map(|(code, _)| code)
}

/// Copy a registered reason code out of `message` onto `err`, leaving the message as it is.
fn attach_reason_from_message(err: PyErr, message: &str) -> PyErr {
    match reason_code_in_message(message)
        .filter(|code| antecedent_core::reason_code::is_registered(code))
    {
        Some(code) => with_reason_code(err, code),
        None => err,
    }
}

impl IntoCausalPyErr for RustCausalError {
    fn into_antecedent_py_err(self) -> PyErr {
        let code = reason_code_of(&self);
        let remedy = self.remedy();
        let fields = self.refusal_fields().map(std::borrow::Cow::into_owned);
        let routes = self.route_diagnostics();
        let err = with_remedy(uncoded_py_err(self), remedy);
        let err = preflight_api::with_refusal_fields(err, fields.as_ref());
        let err = with_route_diagnostics(err, &routes);
        match code {
            Some(code) => with_default_reason_code(err, &code),
            None => err,
        }
    }
}

/// The Python exception of an error's class, before its reason code is attached (see
/// [`reason_code_of`]).
fn uncoded_py_err(error: RustCausalError) -> PyErr {
    match error {
        // A refusal with structured fields is the refusal it wraps.
        RustCausalError::Diagnosed { error, .. } => uncoded_py_err(*error),
        RustCausalError::Identify(e) => CausalIdentifyError::new_err(e.to_string()),
        // A reason-coded estimator refusal is the one refusal class, as a Rust
        // `Unsupported` refusal is.
        RustCausalError::Estimate(
            e @ (antecedent_estimate::EstimationError::Refused { .. }
            | antecedent_estimate::EstimationError::RefusedWithFields { .. }
            | antecedent_estimate::EstimationError::TargetPopulation),
        ) => unsupported_py_err(e.to_string()),
        // An inference mode the route does not perform is the one refusal class, however the
        // route spelled it; a refusal with any other route record stays an estimate error.
        RustCausalError::Estimate(
            e @ antecedent_estimate::EstimationError::Prob(antecedent_prob::ProbError::Route {
                ..
            }),
        ) if {
            let records = RustCausalError::route_diagnostics(&RustCausalError::Estimate(e.clone()));
            !records.is_empty()
                && records.iter().all(|record| record.is_unsupported_inference())
        } =>
        {
            unsupported_py_err(e.to_string())
        }
        RustCausalError::Estimate(e) => CausalEstimateError::new_err(e.to_string()),
        RustCausalError::Validate(e) => CausalValidateError::new_err(e.to_string()),
        RustCausalError::Discovery(e) => CausalDiscoveryError::new_err(e.to_string()),
        // A model or attribution refusal that names its reason code is the one refusal
        // class too.
        RustCausalError::Model(e @ antecedent_model::ModelError::Unsupported { .. })
            if reason_code_in_message(&e.to_string()).is_some() =>
        {
            unsupported_py_err(e.to_string())
        }
        RustCausalError::Model(e) => CausalModelError::new_err(e.to_string()),
        RustCausalError::Counterfactual(e) => CausalCounterfactualError::new_err(e.to_string()),
        RustCausalError::Attribution(e @ antecedent_attribution::AttributionError::Message(_))
            if reason_code_in_message(&e.to_string()).is_some() =>
        {
            unsupported_py_err(e.to_string())
        }
        RustCausalError::Attribution(e) => CausalAttributionError::new_err(e.to_string()),
        RustCausalError::Serialization(e) => crate::serialization_err(e.to_string()),
        RustCausalError::Data(e) => CausalDataError::new_err(e.to_string()),
        RustCausalError::Graph(e) => CausalGraphError::new_err(e.to_string()),
        RustCausalError::Design(e) => CausalDesignError::new_err(e.to_string()),
        RustCausalError::Callback { name, message } => {
            CausalDesignError::new_err(format!("callback {name}: {message}"))
        }
        RustCausalError::State(e) => match &e {
            antecedent::state::StateError::CacheBudget { .. } => {
                CausalResourceError::new_err(e.to_string())
            }
            _ => CausalStateError::new_err(e.to_string()),
        },
        // A structure that does not describe the table is a data problem, and
        // callers should be able to catch it as one rather than the root class.
        RustCausalError::Schema(e) => CausalDataError::new_err(e.to_string()),
        RustCausalError::SchemaMismatch { detail } => CausalDataError::new_err(detail),
        RustCausalError::Compile { message } => CausalCompileError::new_err(message),
        RustCausalError::Rejected(failure) => {
            CausalCompileError::new_err(failure.message().to_owned())
        }
        RustCausalError::NotIdentified { status, search_capped, message } => {
            not_identified_py_err(status.as_str(), search_capped, message)
        }
        RustCausalError::Resource { message } => CausalResourceError::new_err(message),
        RustCausalError::ReviewRequired {
            kind,
            algorithm,
            pending_edge_count,
            pending_edges,
            message,
            hint,
        } => review_required_py_err(
            kind.as_str().to_owned(),
            algorithm,
            pending_edge_count,
            pending_edges,
            message,
            hint,
        ),
        RustCausalError::Unsupported { message } => unsupported_py_err(message.to_string()),
        RustCausalError::Support { id, message } => unsupported_py_err(format!("{id}: {message}")),
        RustCausalError::Missing { field } => {
            CausalCompileError::new_err(format!("missing required field: {field}"))
        }
        RustCausalError::Cancelled { stage } => {
            CausalCancelledError::new_err(format!("cancelled during {stage}"))
        }
        // `CausalError` is `#[non_exhaustive]`: any variant added upstream maps to the
        // hierarchy root rather than failing the build. Give new variants an explicit
        // arm above when their Python-facing category is decided.
        ref other => CausalError::new_err(other.to_string()),
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

pub(crate) fn public_adjustment_set(
    status: antecedent_core::IdentificationStatus,
    set: Vec<String>,
) -> Vec<String> {
    use antecedent_core::IdentificationStatus::{
        IdentifiedUnderParametricRestrictions, IdentifiedUnderPriorRestrictions,
        NonparametricallyIdentified,
    };
    match status {
        NonparametricallyIdentified
        | IdentifiedUnderParametricRestrictions
        | IdentifiedUnderPriorRestrictions => set,
        _ => Vec::new(),
    }
}

/// Scalar identified set `[lower, upper]` of a class-aware result and its
/// interval: endpoints, level, construction, and truncation flag (C-3).
#[derive(Default)]
pub(crate) struct IdentifiedSetFields {
    pub set: Option<(f64, f64)>,
    pub interval: Option<(f64, f64)>,
    pub level: Option<f64>,
    /// Wire tag of the construction (`imbens_manski_shared_block` or
    /// `product_posterior_envelope_quantile`).
    pub method: Option<String>,
    /// The set spans a capped (retained-only) completion enumeration.
    pub truncated: Option<bool>,
}

pub(crate) fn identified_set_fields(result: &antecedent::StudyResult) -> IdentifiedSetFields {
    let Some(mixture) = result.structural_response.as_ref() else {
        return IdentifiedSetFields::default();
    };
    let set = mixture
        .identified_set
        .as_ref()
        .filter(|set| set.lower.len() == 1 && set.upper.len() == 1)
        .map(|set| (set.lower[0], set.upper[0]));
    let interval = mixture.identified_set_interval.as_ref();
    IdentifiedSetFields {
        set,
        interval: interval.map(|i| (i.lower, i.upper)),
        level: interval.map(|i| i.level),
        method: interval.and_then(|i| identified_set_method_tag(i.method)),
        truncated: interval.map(|i| i.truncated),
    }
}

/// Wire tag of an identified-set interval construction (`None` for one the
/// artifact format has no tag for, which the artifact encoder refuses too).
fn identified_set_method_tag(
    method: antecedent_estimate::IdentifiedSetIntervalMethod,
) -> Option<String> {
    let wire = antecedent_io::IdentifiedSetIntervalMethodWire::try_from_method(method).ok()?;
    serde_json::to_value(wire).ok()?.as_str().map(str::to_owned)
}

pub(crate) fn evidence_status_parts(
    status: Option<antecedent::CellStatus>,
) -> (Option<String>, Option<String>, Option<String>) {
    match status {
        Some(antecedent::CellStatus::Licensed) => (Some("licensed".into()), None, None),
        Some(antecedent::CellStatus::Allowlisted { reason, parent }) => {
            (Some("allowed_unlicensed".into()), Some(reason.to_string()), Some(parent.to_string()))
        }
        _ => (None, None, None),
    }
}

/// Coarse-grained ATE analysis result (single boundary crossing).
#[pyclass]
#[allow(clippy::struct_excessive_bools)] // FFI flat getters; effort flags are intentional
pub(crate) struct AteAnalysisResult {
    pub(crate) effect_authority: Option<scalar_decision_api::NativeEffectAuthority>,
    /// Held-out doubly robust policy answer.
    #[pyo3(get)]
    pub(crate) policy_value: Option<policy_api::PolicyValueSection>,
    /// Conditional continuous-dose response grid.
    #[pyo3(get)]
    pub(crate) continuous_dose_response: Option<policy_api::ContinuousDoseResponseSection>,
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
    #[pyo3(get)]
    pub(crate) certificate_json: Option<String>,
    /// Missing-data policy, assumption and row accounting of this execution, as JSON.
    #[pyo3(get)]
    pub(crate) missing_data_json: Option<String>,
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
    /// Route-native diagnostics on the result, one `(route, outcome, failed, unevaluable_mass)`
    /// row per `route.*` diagnostic; the mass is `None` when the route did not report one.
    #[pyo3(get)]
    route_diagnostics: Vec<(String, String, bool, Option<f64>)>,
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
    /// Support-matrix structure axis (`explicit`, `accepted`, `graph_posterior`).
    #[pyo3(get)]
    structure_source: String,
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
    /// Perturbed prior family: `isotropic_scale`, `external_alpha`, or
    /// `resolved_prior_variance`.
    #[pyo3(get)]
    prior_sensitivity_family: Option<String>,
    /// Variance multipliers around the resolved prior (resolved-prior family only).
    #[pyo3(get)]
    prior_sensitivity_variance_multipliers: Option<Vec<f64>>,
    #[pyo3(get)]
    conflict_source_ids: Option<Vec<String>>,
    #[pyo3(get)]
    conflict_alphas_requested: Option<Vec<f64>>,
    #[pyo3(get)]
    conflict_alphas_applied: Option<Vec<f64>>,
    #[pyo3(get)]
    posterior_unidentified_mass: Option<f64>,
    #[pyo3(get)]
    posterior_subsampled_out_mass: Option<f64>,
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
    /// Nested identification section (view onto the fields above of the same name).
    #[pyo3(get)]
    identification: IdentificationSection,
    /// Nested estimate section.
    #[pyo3(get)]
    estimate: EstimateSection,
    /// Nested posterior section.
    #[pyo3(get)]
    posterior: PosteriorSection,
    /// Nested validation section.
    #[pyo3(get)]
    validation: ValidationSection,
    /// Nested performance section.
    #[pyo3(get)]
    performance: PerformanceSection,
    #[pyo3(get)]
    mediation_total: Option<f64>,
    #[pyo3(get)]
    unit_effects: Option<Vec<f64>>,
    /// Per-unit `(lower, upper)` intervals aligned with `unit_effects`.
    #[pyo3(get)]
    unit_effect_intervals: Option<Vec<(f64, f64)>>,
    /// Level of `unit_effect_intervals`.
    #[pyo3(get)]
    unit_effect_intervals_level: Option<f64>,
    /// Construction of `unit_effect_intervals`.
    #[pyo3(get)]
    unit_effect_intervals_method: Option<String>,
    /// Per-unit extrapolation flags aligned with `unit_effects`.
    #[pyo3(get)]
    unit_extrapolative: Option<Vec<bool>>,
    #[pyo3(get)]
    assumptions: Vec<String>,
    #[pyo3(get)]
    support_diagnostics: Vec<String>,
    #[pyo3(get)]
    mediation_direct: Option<f64>,
    #[pyo3(get)]
    mediation_mediated: Option<f64>,
    #[pyo3(get)]
    mediation_horizons: Vec<u32>,
    #[pyo3(get)]
    mediation_effects: Vec<f64>,
    #[pyo3(get)]
    mediation_totals: Vec<f64>,
    #[pyo3(get)]
    mediation_directs: Vec<f64>,
    #[pyo3(get)]
    mediation_mediated_effects: Vec<f64>,
    #[pyo3(get)]
    mediation_identification_statuses: Vec<String>,
    #[pyo3(get)]
    mediation_methods: Vec<String>,
    #[pyo3(get)]
    mediation_adjustments: Vec<Vec<(u32, i32)>>,
    #[pyo3(get)]
    mediation_uncertainty_kinds: Vec<String>,
    #[pyo3(get)]
    mediation_standard_deviations: Vec<Option<f64>>,
    #[pyo3(get)]
    mediation_q025: Vec<Option<f64>>,
    #[pyo3(get)]
    mediation_q975: Vec<Option<f64>>,
    #[pyo3(get)]
    mediation_identified_lower: Vec<Option<f64>>,
    #[pyo3(get)]
    mediation_identified_upper: Vec<Option<f64>>,
    #[pyo3(get)]
    mediation_joint_posterior: Option<bool>,
    /// Trial-to-target transport estimate and its overlap diagnostics (TransportQuery).
    #[pyo3(get)]
    transport: Option<transport_interference_api::TransportSection>,
    /// Randomized exposure contrast (InterferenceQuery).
    #[pyo3(get)]
    interference: Option<transport_interference_api::InterferenceSection>,
    /// Randomized Bernoulli ITT and design metadata.
    #[pyo3(get)]
    randomized_effect: Option<experiment_api::RandomizedEffectSection>,
    /// Balanced two-period panel DiD estimate and cluster standard error.
    #[pyo3(get)]
    panel_did: Option<quasi_api::PanelDidSection>,
    /// Synthetic-control point result and donor-support diagnostics.
    #[pyo3(get)]
    synthetic_control: Option<quasi_api::SyntheticControlSection>,
    /// Synthetic DiD point result and both fitted weight vectors.
    #[pyo3(get)]
    synthetic_did: Option<quasi_api::SyntheticDidSection>,
    /// Fixed-bandwidth fuzzy RD or regression-kink point result.
    #[pyo3(get)]
    local_polynomial_ratio: Option<quasi_api::LocalPolynomialRatioSection>,
    /// Randomized survival or competing-risk curves with no interval claim.
    #[pyo3(get)]
    survival: Option<survival_api::SurvivalSection>,
    /// Posterior of an unlicensed Bayesian randomized-design row (survival contrast, fixed
    /// group-dose policy value) with the covariance of its joint draws.
    #[pyo3(get)]
    posterior_row: Option<posterior_row_api::PosteriorRowSection>,
    /// Sequential randomized regime value and support diagnostics.
    #[pyo3(get)]
    longitudinal_regime: Option<regimes_api::LongitudinalRegimeSection>,
    /// Observational two-decision contrast of two static regimes (unlicensed).
    #[pyo3(get)]
    sequential_contrast: Option<unlicensed_design_api::SequentialContrastSection>,
    /// Within-period history-matched contrast (unlicensed).
    #[pyo3(get)]
    history_match: Option<unlicensed_design_api::HistoryMatchSection>,
    /// Per-target GCM anomaly scores (AnomalyAttribution).
    #[pyo3(get)]
    anomaly: Option<Vec<gcm_api::AnomalyScores>>,
    /// GCM distribution-change Shapley (ChangeAttribution).
    #[pyo3(get)]
    change_attribution: Option<gcm_api::ChangeAttributionResult>,
    /// Support-matrix evidence contract (`licensed` or `allowed_unlicensed`).
    #[pyo3(get)]
    evidence_status: Option<String>,
    /// Allowlist reason when `evidence_status` is `allowed_unlicensed`.
    #[pyo3(get)]
    allowlist_reason: Option<String>,
    /// Allowlist parent family when `evidence_status` is `allowed_unlicensed`.
    #[pyo3(get)]
    allowlist_parent: Option<String>,
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

// --- Nested result sections --------------------------------------------------------
//
// `AteAnalysisResult` (static) and `temporal_api::AnalysisResult` are both flat DTOs
// with a lot of field-name overlap. `shared_study_sections` is the one
// `StudyResult →` section builder; facades only attach modality extras (ATE
// posterior artifacts, temporal mediation grids). Nested sections mirror
// `antecedent.results._views`. Flat fields stay in place and are filled from
// the same builder so identified-set interval fields (`structural_identified_set_interval`,
// `distribution.mean_interval`) have a single owner.

/// Identification section (mirrors `antecedent.results.IdentificationView`).
///
/// Identical shape on both DTOs — the temporal facade always populates every field.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct IdentificationSection {
    /// Identification status string (e.g. `"NonparametricallyIdentified"`).
    #[pyo3(get)]
    status: String,
    /// Identification/estimand method id.
    #[pyo3(get)]
    method: String,
    /// Names of variables in the adjustment set.
    #[pyo3(get)]
    adjustment_set: Vec<String>,
    /// Number of assumptions recorded for the estimate.
    #[pyo3(get)]
    assumption_count: usize,
    /// Number of derivation steps in the identification proof.
    #[pyo3(get)]
    derivation_step_count: usize,
}

/// Estimate section (mirrors `antecedent.results.EstimateView`'s top-level scalar
/// fields; `mediation` is assembled separately by the Python wrapper).
///
/// `overlap_ess` / `overlap_propensity_min` are `None` on the temporal DTO — the
/// temporal facade has no overlap report to draw them from.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct EstimateSection {
    /// Scalar contrast when one exists. Function-valued results omit it.
    #[pyo3(get)]
    pub(crate) ate: Option<f64>,
    #[pyo3(get)]
    pub(crate) se_analytic: f64,
    #[pyo3(get)]
    pub(crate) se_bootstrap: Option<f64>,
    #[pyo3(get)]
    pub(crate) estimator_id: String,
    #[pyo3(get)]
    pub(crate) method: String,
    /// Effective sample size under overlap weighting. `None` on the temporal DTO.
    #[pyo3(get)]
    pub(crate) overlap_ess: Option<f64>,
    /// Minimum estimated propensity score. `None` on the temporal DTO.
    #[pyo3(get)]
    pub(crate) overlap_propensity_min: Option<f64>,
    /// Per-arm / per-threshold interventional means or exceedance probabilities.
    #[pyo3(get)]
    functional_means: Option<Vec<f64>>,
    /// Rearranged `F_a(c)` when an exceedance grid was estimated.
    #[pyo3(get)]
    exceedance_cdf: Option<Vec<f64>>,
    /// Whether `exceedance_cdf` was isotonically rearranged (cov/bands stay raw).
    #[pyo3(get)]
    monotone_rearranged: bool,
    /// Additive joint-response disclosure.
    #[pyo3(get)]
    interaction_structurally_zero: Option<bool>,
    /// Counterfactual disclosure: the selected mechanisms admit no effect
    /// modification, so every per-unit effect is the same number by construction.
    #[pyo3(get)]
    unit_effects_homogeneous: Option<bool>,
    /// Score-table metadata (n, folds, provenance). Full scores stay on the artifact.
    #[pyo3(get)]
    score_table: Option<ScoreTableSection>,
    #[pyo3(get)]
    joint_covariance: Option<Vec<Vec<f64>>>,
    #[pyo3(get)]
    score_inference: Option<ScoreInferenceSection>,
    /// What declared sampling weights did to the analysis, as
    /// `(rows_used, zero_weight_rows, effective_sample_size, weight_sum)`; `None` when no
    /// weights were declared.
    #[pyo3(get)]
    sampling_weights: Option<(u64, u64, f64, f64)>,
    #[pyo3(get)]
    scenario_effects: Option<Vec<f64>>,
    #[pyo3(get)]
    scenario_intervals: Option<Vec<(f64, f64)>>,
    #[pyo3(get)]
    simultaneous_interval: Option<(f64, f64, f64)>,
    #[pyo3(get)]
    adjusted_p_values: Option<(f64, f64)>,
    #[pyo3(get)]
    family_contrast: Option<(f64, f64)>,
    #[pyo3(get)]
    family_contrast_interval: Option<(f64, f64, f64)>,
    #[pyo3(get)]
    candidate_selection: Option<CandidateSelectionSection>,
    #[pyo3(get)]
    evalue: Option<f64>,
    /// Threshold the E-value refuter judged `evalue` against.
    #[pyo3(get)]
    evalue_threshold: Option<f64>,
    /// Interventional-distribution atoms with their probability intervals.
    /// `None` for non-distribution queries.
    #[pyo3(get)]
    distribution_atoms: Option<Vec<DistributionAtomSection>>,
    /// Bounded interval for the interventional mean of a binary `{0, 1}`
    /// outcome (the probability `P(Y = 1 | do(x))`). `None` otherwise.
    #[pyo3(get)]
    mean_interval: Option<ProbabilityIntervalSection>,
    /// Per-row CATE when a heterogeneous-effect estimator produced one.
    #[pyo3(get)]
    cate: Option<Vec<f64>>,
    /// Licensed pointwise CATE standard errors, when computed.
    #[pyo3(get)]
    cate_se: Option<Vec<f64>>,
    /// Forest leaf-dispersion diagnostic per row. Not a standard error.
    #[pyo3(get)]
    cate_leaf_dispersion: Option<Vec<f64>>,
    /// Actual fitted learner spec, implementation and version, in fit order.
    #[pyo3(get)]
    outcome_oof_r2: Option<f64>,
    #[pyo3(get)]
    treatment_oof_logloss: Option<f64>,
    #[pyo3(get)]
    crossfit_folds: Option<usize>,
    #[pyo3(get)]
    crossfit_seed: Option<u64>,
    #[pyo3(get)]
    learner_provenance: Vec<(String, String, String)>,
    /// A declared GLM-to-penalized fallback that ran: `(stage, fold, reason, message,
    /// destination)` of the failed GLM propensity fit and the canonical key of the fallback
    /// route that produced the result. `None` when no fallback ran.
    #[pyo3(get)]
    penalized_fallback: Option<(String, usize, String, String, String)>,
    /// Covariates a lasso propensity kept on each cross-fit fold: `(fold, names)`. Empty for
    /// ridge and for the unpenalized propensity.
    #[pyo3(get)]
    penalized_support: Vec<(usize, Vec<String>)>,
    /// The refit bootstrap of a penalized propensity: `(uncertainty_kind, se, requested, ok,
    /// failed, cancelled, penalties, influence_se)`, `penalties` holding `(replicate,
    /// per-fold selected penalties)` of each successful replicate. `None` without one.
    #[pyo3(get)]
    penalized_bootstrap: Option<PenalizedBootstrapRecord>,
}

/// `(uncertainty_kind, se, requested, ok, failed, cancelled, penalties, influence_se)`.
type PenalizedBootstrapRecord =
    (String, Option<f64>, u32, u32, u32, bool, Vec<(u32, Vec<f64>)>, f64);

/// Bounded interval for one interventional probability, or the reason none
/// could be formed.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct ProbabilityIntervalSection {
    /// Nominal two-sided level; `None` when unavailable.
    #[pyo3(get)]
    level: Option<f64>,
    /// Lower bound in `[0, 1]`; `None` when unavailable.
    #[pyo3(get)]
    lower: Option<f64>,
    /// Upper bound in `[0, 1]`; `None` when unavailable.
    #[pyo3(get)]
    upper: Option<f64>,
    /// Machine-readable reason when no interval could be formed.
    #[pyo3(get)]
    unavailable: Option<String>,
}

impl ProbabilityIntervalSection {
    pub(crate) fn new(interval: antecedent_estimate::ProbabilityInterval) -> Self {
        match interval {
            antecedent_estimate::ProbabilityInterval::Bounded { level, lower, upper } => Self {
                level: Some(level),
                lower: Some(lower),
                upper: Some(upper),
                unavailable: None,
            },
            antecedent_estimate::ProbabilityInterval::Unavailable(reason) => Self {
                level: None,
                lower: None,
                upper: None,
                unavailable: Some(reason.as_str().to_string()),
            },
        }
    }
}

/// One interventional-distribution atom.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct DistributionAtomSection {
    /// `(outcome name, level)` pairs in query order.
    #[pyo3(get)]
    outcomes: Vec<(String, Option<f64>)>,
    /// `(conditioning name, level)` pairs; empty when unconditional.
    #[pyo3(get)]
    conditioning: Vec<(String, Option<f64>)>,
    /// Estimated probability (posterior mean in Bayesian mode).
    #[pyo3(get)]
    probability: f64,
    /// Frequentist bootstrap SE of this probability.
    #[pyo3(get)]
    se_bootstrap: Option<f64>,
    /// Frequentist bounded interval; `None` when no bootstrap ran.
    #[pyo3(get)]
    interval: Option<ProbabilityIntervalSection>,
}

/// Distribution atoms and the binary-mean interval for the estimate section.
pub(crate) fn distribution_sections(
    names: &[String],
    distribution: Option<&antecedent_estimate::InterventionalDistributionEstimate>,
) -> (Option<Vec<DistributionAtomSection>>, Option<ProbabilityIntervalSection>) {
    let Some(dist) = distribution else {
        return (None, None);
    };
    let named = |pairs: &[(antecedent_core::VariableId, antecedent_core::Value)]| {
        pairs
            .iter()
            .map(|(id, value)| {
                let name =
                    names.get(id.as_usize()).cloned().unwrap_or_else(|| format!("var{}", id.raw()));
                (name, value.as_f64())
            })
            .collect::<Vec<_>>()
    };
    let atoms = dist
        .atoms
        .iter()
        .enumerate()
        .map(|(i, atom)| {
            let uncertainty = dist.atom_uncertainty.get(i);
            DistributionAtomSection {
                outcomes: named(&atom.outcomes),
                conditioning: named(&atom.conditioning),
                probability: atom.probability,
                se_bootstrap: uncertainty.and_then(|u| u.se_bootstrap),
                interval: uncertainty.map(|u| ProbabilityIntervalSection::new(u.interval)),
            }
        })
        .collect();
    let mean = dist.mean_interval.map(ProbabilityIntervalSection::new);
    (Some(atoms), mean)
}

/// Shared `StudyResult →` sections. Static and temporal facades attach extras only.
pub(crate) struct SharedStudySections {
    pub identification: IdentificationSection,
    pub estimate: EstimateSection,
    pub posterior: PosteriorSection,
    pub validation: ValidationSection,
    pub performance: PerformanceSection,
    pub identified_set: IdentifiedSetFields,
    pub identification_status: String,
    pub method: String,
    pub adjustment_set: Vec<String>,
    pub estimator_id: String,
    pub plan_id: String,
    pub modality: String,
    pub refutations: Vec<RefutationReportView>,
    pub evidence_status: Option<String>,
    pub allowlist_reason: Option<String>,
    pub allowlist_parent: Option<String>,
    pub structural_weight_basis: Option<String>,
    pub structural_identified_mass: Option<f64>,
    pub structural_unidentified_mass: Option<f64>,
    pub structural_unevaluable_mass: Option<f64>,
}

impl SharedStudySections {
    pub(crate) fn with_posterior_artifact(mut self, artifact: Option<Vec<u8>>) -> Self {
        self.posterior.artifact = artifact;
        self
    }
}

pub(crate) fn mediation_uncertainty_projection(
    slice: &antecedent_estimate::TemporalMediationSlice,
) -> (String, Option<f64>, Option<f64>, Option<f64>) {
    match &slice.uncertainty {
        antecedent_estimate::TemporalMediationUncertainty::FrequentistPointwise {
            standard_error,
        }
        | antecedent_estimate::TemporalMediationUncertainty::FrequentistBlockBootstrap {
            requested: standard_error,
            ..
        } => ("frequentist_pointwise".to_string(), *standard_error, None, None),
        antecedent_estimate::TemporalMediationUncertainty::BayesianPointwise {
            requested, ..
        } => (
            "bayesian_pointwise".to_string(),
            Some(requested.standard_deviation),
            Some(requested.q025),
            Some(requested.q975),
        ),
        _ => ("unavailable".to_string(), None, None, None),
    }
}

pub(crate) fn shared_study_sections(
    names: &[String],
    result: &antecedent::StudyResult,
    estimator_id: String,
) -> PyResult<SharedStudySections> {
    let absent_effect = antecedent::EffectEstimate::new(
        f64::NAN,
        f64::NAN,
        antecedent_core::AssumptionSet::default(),
        antecedent_estimate::OverlapPolicy::ExplicitOverride,
    );
    let estimate = result.estimate.as_effect().unwrap_or(&absent_effect);
    // A temporal effect's estimand ids are dense unfolded node ids; read them
    // through the certificate's unfolding coordinates, not as schema ids (a
    // dense id past the first window slice would otherwise name the wrong
    // variable, or none).
    let temporal_indexer = result.certificate.as_ref().and_then(|certificate| {
        match (&certificate.query, &certificate.identification) {
            (
                antecedent_core::CausalQuery::TemporalEffect(_),
                antecedent::Identification::Point { temporal_indexer, .. },
            ) => temporal_indexer.as_ref(),
            _ => None,
        }
    });
    let adjustment_set: Vec<String> = public_adjustment_set(
        result.identification.status,
        result
            .estimand
            .adjustment_set
            .iter()
            .map(|id| {
                let variable = temporal_indexer
                    .and_then(|indexer| indexer.key_of(id.raw()).ok())
                    .map_or(*id, |key| key.variable);
                names
                    .get(variable.as_usize())
                    .cloned()
                    .unwrap_or_else(|| format!("var{}", variable.raw()))
            })
            .collect(),
    );
    let identification_status = format!("{:?}", result.identification.status);
    let method = result.estimand.method.to_string();
    let refutations: Vec<RefutationReportView> =
        result.refutations.iter().map(RefutationReportView::from).collect();
    let plan_id = result.logical_plan.plan_id.to_string();
    let modality = format!("{:?}", result.logical_plan.data_classification);
    let latency_mode =
        result.performance.latency_mode.as_ref().map(std::string::ToString::to_string);
    let bootstrap_replicates_ok =
        result.performance.bootstrap_replicates_ok.or(estimate.uncertainty.bootstrap.replicates_ok);
    let cancelled = result.performance.cancelled || estimate.uncertainty.bootstrap.cancelled;
    let stage_timings: Vec<(String, u64)> =
        result.performance.stage_timings_ns.iter().map(|(s, ns)| (s.to_string(), *ns)).collect();
    let identification = IdentificationSection {
        status: identification_status.clone(),
        method: method.clone(),
        adjustment_set: adjustment_set.clone(),
        assumption_count: estimate.provenance.assumptions.len(),
        derivation_step_count: result.identification.derivation.steps.len(),
    };
    let (distribution_atoms, mean_interval) = distribution_sections(names, result.distribution());
    let overlap_ess = estimate.diagnostics.overlap_report.as_ref().and_then(|r| r.ess);
    let overlap_propensity_min =
        estimate.diagnostics.overlap_report.as_ref().map(|r| r.propensity_min);
    let estimate = EstimateSection {
        ate: estimate.ate.is_finite().then_some(estimate.ate),
        se_analytic: estimate.se_analytic(),
        se_bootstrap: estimate.se_bootstrap(),
        estimator_id: estimator_id.clone(),
        method: method.clone(),
        overlap_ess,
        overlap_propensity_min,
        functional_means: estimate.scores.inference.as_ref().map(|s| s.raw_means.clone()).or_else(
            || {
                estimate
                    .scores
                    .table
                    .as_ref()
                    .and_then(|t| t.summarize(None).ok().map(|s| s.means.to_vec()))
            },
        ),
        joint_covariance: estimate
            .joint
            .covariance
            .as_ref()
            .map(|c| (0..c.dim).map(|i| (0..c.dim).map(|j| c.get(i, j)).collect()).collect()),
        score_inference: estimate.scores.inference.as_ref().map(ScoreInferenceSection::from),
        sampling_weights: estimate.diagnostics.sampling_weights.map(|record| {
            (
                record.rows_used,
                record.zero_weight_rows,
                record.effective_sample_size,
                record.weight_sum,
            )
        }),
        scenario_effects: estimate.scenarios.effects.as_ref().map(|v| v.to_vec()),
        scenario_intervals: estimate.scenarios.intervals.as_ref().map(|v| v.to_vec()),
        exceedance_cdf: estimate.joint.exceedance_cdf.as_ref().map(|v| v.to_vec()),
        monotone_rearranged: estimate.joint.monotone_rearranged,
        interaction_structurally_zero: result
            .response()
            .as_ref()
            .map(|r| r.interaction_structurally_zero)
            .or(Some(estimate.joint.interaction_structurally_zero)),
        unit_effects_homogeneous: Some(estimate.heterogeneity.unit_effects_homogeneous),
        score_table: estimate.scores.table.as_ref().map(|t| ScoreTableSection {
            n_rows: t.n_rows,
            n_folds: t.n_folds,
            provenance: t.nuisance_provenance.to_string(),
            columns: t.columns.iter().map(|c| (c.arm, c.threshold)).collect(),
            scores: t.scores.to_vec(),
            row_index: t.row_index.iter().map(|id| id.get()).collect(),
            fold_ids: t.fold_ids.to_vec(),
            adjustment_set: t.adjustment_set.iter().map(|v| v.raw()).collect(),
            observed_arm: t.observed_arm.to_vec(),
            propensities: t.propensities.to_vec(),
            observed_outcome: t.observed_outcome.to_vec(),
            treatment: t.treatment.raw(),
            intervened: t.intervened.iter().map(|v| v.raw()).collect(),
        }),
        simultaneous_interval: estimate.family.simultaneous_interval,
        adjusted_p_values: estimate.family.adjusted_p_values,
        family_contrast: estimate.family.contrast,
        family_contrast_interval: estimate.family.contrast_interval,
        candidate_selection: estimate
            .family
            .candidate_selection
            .as_ref()
            .map(|s| CandidateSelectionSection {
                screen_id: s.screen_id.to_string(),
                procedure: s.procedure.to_string(),
                winner_index: s.winner_index,
                family_size: s.family_size,
                screen_rows: s.screen_rows.iter().map(|id| id.get()).collect(),
                estimate_rows: s.estimate_rows.iter().map(|id| id.get()).collect(),
                disjoint: s.disjoint,
            })
            .or_else(|| {
                result.candidate_selection.as_ref().map(|s| CandidateSelectionSection {
                    screen_id: s.screen_id.to_string(),
                    procedure: s.procedure.as_str().to_string(),
                    winner_index: s.winner_index,
                    family_size: s.family_size,
                    screen_rows: s.screen_rows.iter().map(|id| id.get()).collect(),
                    estimate_rows: s.estimate_rows.iter().map(|id| id.get()).collect(),
                    disjoint: s.disjoint,
                })
            }),
        evalue: estimate.diagnostics.evalue,
        evalue_threshold: estimate.diagnostics.evalue_threshold,
        distribution_atoms,
        mean_interval,
        cate: estimate.heterogeneity.cate.as_ref().map(|c| c.to_vec()),
        cate_se: estimate.heterogeneity.cate_se.as_ref().map(|c| c.to_vec()),
        cate_leaf_dispersion: estimate
            .heterogeneity
            .cate_leaf_dispersion
            .as_ref()
            .map(|c| c.to_vec()),
        outcome_oof_r2: estimate.crossfit.outcome_oof_r2,
        treatment_oof_logloss: estimate.crossfit.treatment_oof_logloss,
        crossfit_folds: estimate.crossfit.folds,
        crossfit_seed: estimate.crossfit.seed,
        learner_provenance: estimate
            .crossfit
            .learner_provenance
            .iter()
            .map(|p| (p.spec.clone(), p.implementation.clone(), p.version.clone()))
            .collect(),
        penalized_fallback: estimate.diagnostics.penalized.as_ref().and_then(|report| {
            report.fallback.as_ref().map(|f| {
                (
                    f.failed_fit.stage.to_string(),
                    f.failed_fit.fold,
                    f.failed_fit.reason.to_string(),
                    f.failed_fit.message.clone(),
                    f.destination.clone(),
                )
            })
        }),
        penalized_support: estimate
            .diagnostics
            .penalized
            .as_ref()
            .map_or_else(Vec::new, |report| {
                report.selected_support.iter().map(|s| (s.fold, s.names.clone())).collect()
            }),
        penalized_bootstrap: estimate
            .diagnostics
            .penalized
            .as_ref()
            .and_then(|report| report.variance.as_ref())
            .map(|v| {
                (
                    v.uncertainty_kind.to_string(),
                    v.refit_bootstrap_se,
                    v.replicates_requested,
                    v.replicates_ok,
                    v.replicates_failed,
                    v.cancelled,
                    v.replicate_penalties
                        .iter()
                        .map(|r| (r.replicate, r.lambdas.clone()))
                        .collect(),
                    v.influence_se,
                )
            }),
    };
    let (
        posterior_effect_mean,
        posterior_effect_sd,
        posterior_q025,
        posterior_q975,
        posterior_n_draws,
        posterior_p_below_zero,
        posterior_backend,
    ) = if let Some(post) = result.posterior.as_ref() {
        let eq = post.effect_column();
        let p_below = eq.map(|_| post.probability_below(0.0)).transpose().map_err(py_err)?;
        (
            eq.map(|i| post.summaries.mean[i]),
            eq.map(|i| post.summaries.sd[i]),
            eq.map(|i| post.summaries.q025[i]),
            eq.map(|i| post.summaries.q975[i]),
            Some(post.draws.n_draws),
            p_below,
            Some(post.diagnostics.backend_id.to_string()),
        )
    } else {
        (None, None, None, None, None, None, None)
    };
    let posterior = PosteriorSection {
        effect_mean: posterior_effect_mean,
        effect_sd: posterior_effect_sd,
        q025: posterior_q025,
        q975: posterior_q975,
        n_draws: posterior_n_draws,
        p_below_zero: posterior_p_below_zero,
        backend: posterior_backend,
        artifact: None,
        unidentified_mass: result.posterior.as_ref().map(|p| p.unidentified_mass),
        subsampled_out_mass: result.posterior.as_ref().map(|p| p.subsampled_out_mass),
    };
    let validation = ValidationSection::from_reports(refutations.clone(), &result.diagnostics);
    let performance = PerformanceSection {
        plan_id: plan_id.clone(),
        modality: modality.clone(),
        peak_memory_bytes: result.physical_plan.estimated_peak_memory_bytes,
        latency_mode,
        wall_time_ns: result.performance.wall_time_ns,
        bootstrap_replicates_requested: result.performance.bootstrap_replicates_requested,
        bootstrap_replicates_ok,
        n_draws: result.performance.n_draws,
        cancelled,
        early_stopped: result.performance.early_stopped,
        stage_timings,
        bytes_borrowed: result.performance.bytes_borrowed,
    };
    let (evidence_status, allowlist_reason, allowlist_parent) =
        evidence_status_parts(result.support_status);
    // A query with no matrix cell (mechanism / unit change) is off the axis by design; say so
    // rather than leaving a reader to mistake it for a licensed cell that was never stamped.
    let evidence_status = match (evidence_status, result.support_status) {
        (None, None) => Some("off_axis".to_owned()),
        (status, _) => status,
    };
    let structural = result.structural_response.as_ref();
    Ok(SharedStudySections {
        identification,
        estimate,
        posterior,
        validation,
        performance,
        identified_set: identified_set_fields(result),
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
        structural_weight_basis: structural.map(|mixture| mixture.weight_basis.as_str().into()),
        structural_identified_mass: structural.map(|m| m.identified_mass),
        structural_unidentified_mass: structural.map(|m| m.unidentified_mass),
        structural_unevaluable_mass: structural.map(|m| m.unevaluable_mass),
    })
}

/// Typed candidate-selection provenance on a batch result.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct CandidateSelectionSection {
    #[pyo3(get)]
    screen_id: String,
    #[pyo3(get)]
    procedure: String,
    #[pyo3(get)]
    winner_index: Option<usize>,
    #[pyo3(get)]
    family_size: usize,
    #[pyo3(get)]
    screen_rows: Vec<u64>,
    #[pyo3(get)]
    estimate_rows: Vec<u64>,
    #[pyo3(get)]
    disjoint: bool,
}

/// Typed score-table payload on an estimate section.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct ScoreTableSection {
    #[pyo3(get)]
    n_rows: usize,
    #[pyo3(get)]
    n_folds: u32,
    #[pyo3(get)]
    provenance: String,
    #[pyo3(get)]
    columns: Vec<(u32, Option<f64>)>,
    #[pyo3(get)]
    scores: Vec<f64>,
    #[pyo3(get)]
    row_index: Vec<u64>,
    #[pyo3(get)]
    fold_ids: Vec<u32>,
    #[pyo3(get)]
    adjustment_set: Vec<u32>,
    #[pyo3(get)]
    observed_arm: Vec<u32>,
    #[pyo3(get)]
    propensities: Vec<f64>,
    #[pyo3(get)]
    observed_outcome: Vec<f64>,
    #[pyo3(get)]
    treatment: u32,
    #[pyo3(get)]
    intervened: Vec<u32>,
}

/// Typed simultaneous inference and weighted support for score columns.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct ScoreInferenceSection {
    #[pyo3(get)]
    raw_means: Vec<f64>,
    #[pyo3(get)]
    lower: Vec<f64>,
    #[pyo3(get)]
    upper: Vec<f64>,
    #[pyo3(get)]
    level: f64,
    #[pyo3(get)]
    critical_value: f64,
    #[pyo3(get)]
    event_n_eff: Vec<f64>,
    #[pyo3(get)]
    threshold_supported: Vec<bool>,
    #[pyo3(get)]
    n_eff: f64,
    #[pyo3(get)]
    n_eff_by_arm: Vec<f64>,
    #[pyo3(get)]
    propensity_range: Option<(f64, f64)>,
    #[pyo3(get)]
    overlap_ok: bool,
}

impl From<&antecedent_estimate::scores::ScoreInference> for ScoreInferenceSection {
    fn from(s: &antecedent_estimate::scores::ScoreInference) -> Self {
        Self {
            raw_means: s.raw_means.clone(),
            lower: s.lower.clone(),
            upper: s.upper.clone(),
            level: s.level,
            critical_value: s.critical_value,
            event_n_eff: s.event_n_eff.clone(),
            threshold_supported: s.threshold_supported.clone(),
            n_eff: s.support.n_eff,
            n_eff_by_arm: s.support.n_eff_by_arm.clone(),
            propensity_range: s.support.propensity_range,
            overlap_ok: s.support.overlap_ok,
        }
    }
}

/// Posterior section (mirrors the top-level scalar fields of
/// `antecedent.results.PosteriorView`; `envelope` / `conflict` are assembled
/// separately by the Python wrapper from other raw fields).
///
/// Always present as a section; every field is `None` when no posterior was
/// computed (frequentist inference) — identical shape on both DTOs, since both
/// flat DTOs already carry these fields as `Option`.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PosteriorSection {
    #[pyo3(get)]
    pub(crate) effect_mean: Option<f64>,
    #[pyo3(get)]
    pub(crate) effect_sd: Option<f64>,
    #[pyo3(get)]
    pub(crate) q025: Option<f64>,
    #[pyo3(get)]
    pub(crate) q975: Option<f64>,
    #[pyo3(get)]
    pub(crate) n_draws: Option<usize>,
    #[pyo3(get)]
    pub(crate) p_below_zero: Option<f64>,
    #[pyo3(get)]
    pub(crate) backend: Option<String>,
    /// Serialized posterior artifact bytes, when requested and available.
    #[pyo3(get)]
    pub(crate) artifact: Option<Vec<u8>>,
    #[pyo3(get)]
    pub(crate) unidentified_mass: Option<f64>,
    /// Identified graph mass the Interactive tier left out of the envelope
    /// subsample (not evaluated; not unidentified).
    #[pyo3(get)]
    pub(crate) subsampled_out_mass: Option<f64>,
}

/// Validation section (mirrors the `passed` / `ran` / `count` / `reports` fields of
/// `antecedent.results.ValidationView`; `prior_predictive` / `posterior_predictive` /
/// `prior_sensitivity` are assembled separately by the Python wrapper from other raw
/// fields, which only the static DTO carries).
///
/// `passed` / `ran` follow the same aggregate rule on both DTOs: `ran` is whether any
/// refuter ran at all, and `passed` is `ran && reports.iter().all(|r| r.passed)` —
/// never `true` when nothing ran, matching the static `refutation_ran` /
/// `refutation_passed` fields exactly.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct ValidationSection {
    #[pyo3(get)]
    pub(crate) passed: bool,
    #[pyo3(get)]
    pub(crate) ran: bool,
    #[pyo3(get)]
    pub(crate) count: usize,
    /// Per-refuter records, one per validator run.
    #[pyo3(get)]
    reports: Vec<RefutationReportView>,
    #[pyo3(get)]
    computation_failures: Vec<ValidationFailureSection>,
}

/// A validator attempted computation but could not produce a scientific verdict.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
struct ValidationFailureSection {
    #[pyo3(get)]
    validator: String,
    #[pyo3(get)]
    reason: String,
}

impl ValidationSection {
    /// Build from a refutation-report slice, applying the shared pass/ran rule.
    ///
    /// The one place both DTOs compute `passed`/`ran` — see the struct doc for the
    /// rule. Kept as a plain function (not tied to `RefutationReport` internals) so
    /// both `ate_result_from_analysis` and `analysis_result_from_run` call the exact
    /// same logic instead of maintaining two copies of the aggregate rule.
    pub(crate) fn from_reports(
        reports: Vec<RefutationReportView>,
        diagnostics: &[antecedent_core::Diagnostic],
    ) -> Self {
        let computation_failures: Vec<_> = diagnostics
            .iter()
            .filter(|d| d.code.as_ref() == "refute.validator.failed")
            .map(|d| {
                let field = |name: &str| {
                    d.fields
                        .iter()
                        .find(|(k, _)| k.as_ref() == name)
                        .map(|(_, v)| v.to_string())
                        .unwrap_or_default()
                };
                ValidationFailureSection { validator: field("validator"), reason: field("reason") }
            })
            .collect();
        let count = reports.len() + computation_failures.len();
        let ran = count > 0;
        let passed = ran && computation_failures.is_empty() && reports.iter().all(|r| r.passed);
        Self { passed, ran, count, reports, computation_failures }
    }
}

/// Performance section (mirrors `antecedent.results.PerformanceView`).
///
/// Most fields are read straight from `StudyResult.performance`, which both the
/// static and temporal execution paths populate. `bootstrap_replicates_ok`,
/// `n_draws` (posterior draw effort), and `stage_timings` are the exception: no
/// temporal execution path currently records them, so they come through as
/// `None` / empty on the temporal DTO (see `analysis_result_from_run` in
/// `temporal_api.rs` for exactly which fields and why).
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PerformanceSection {
    #[pyo3(get)]
    pub(crate) plan_id: String,
    #[pyo3(get)]
    pub(crate) modality: String,
    #[pyo3(get)]
    pub(crate) peak_memory_bytes: Option<u64>,
    #[pyo3(get)]
    pub(crate) latency_mode: Option<String>,
    #[pyo3(get)]
    pub(crate) wall_time_ns: Option<u64>,
    #[pyo3(get)]
    pub(crate) bootstrap_replicates_requested: Option<u32>,
    #[pyo3(get)]
    pub(crate) bootstrap_replicates_ok: Option<u32>,
    #[pyo3(get)]
    pub(crate) n_draws: Option<u32>,
    #[pyo3(get)]
    pub(crate) cancelled: bool,
    #[pyo3(get)]
    pub(crate) early_stopped: bool,
    /// `(stage name, elapsed nanoseconds)` pairs. Empty when not recorded.
    #[pyo3(get)]
    pub(crate) stage_timings: Vec<(String, u64)>,
    /// Arrow CDI bytes borrowed at ingest (`None` when not an Arrow path).
    #[pyo3(get)]
    bytes_borrowed: Option<u64>,
}

/// Decoded posterior artifact for Python consumers.
#[pyclass(frozen)]
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
    draws: Vec<f64>,
    #[pyo3(get)]
    backend_id: String,
    #[pyo3(get)]
    identification: String,
    #[pyo3(get)]
    unidentified_mass: f64,
    /// Identified graph mass a latency tier left out of the envelope subsample.
    #[pyo3(get)]
    subsampled_out_mass: f64,
    #[pyo3(get)]
    converged: bool,
    #[pyo3(get)]
    hessian_condition: f64,
    #[pyo3(get)]
    quantity_names: Vec<String>,
    /// Source active minus control contrast for effect-to-slope hydration.
    #[pyo3(get)]
    treatment_contrast: Option<f64>,
    // Preserve roles, indices, and names across decode/encode. Python getters
    // return copies, so the decoded schema cannot drift from quantity_names.
    quantity_schema: Option<Vec<PosteriorQuantityWire>>,
}

#[pymethods]
impl PosteriorArtifact {
    /// Posterior draws as a float64 array (one memcpy, no boxed floats). `numpy.asarray(artifact)`
    /// views the same storage through the buffer protocol without copying at all.
    #[getter]
    fn draws<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.draws)
    }

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
        treatment_contrast=None,
        subsampled_out_mass=0.0,
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
        treatment_contrast: Option<f64>,
        subsampled_out_mass: f64,
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
            subsampled_out_mass,
            converged,
            hessian_condition,
            quantity_names,
            treatment_contrast,
            quantity_schema: None,
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
    /// Supply `treatment_contrast` (source active minus control) for an ATE
    /// summary that will be mapped onto a target regression slope.
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
        treatment_contrast=None,
        subsampled_out_mass=0.0,
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
        treatment_contrast: Option<f64>,
        subsampled_out_mass: f64,
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
            subsampled_out_mass,
            converged,
            hessian_condition,
            quantity_names,
            treatment_contrast,
            quantity_schema: None,
        }
    }

    /// Number of posterior draws (``n_draws``, independent of whether `draws` is populated).
    fn __len__(&self) -> usize {
        self.n_draws
    }

    /// NumPy array protocol — ``np.asarray(artifact)`` returns the flat posterior draws
    /// directly, without needing ``np.asarray(artifact.draws)``.
    ///
    /// `dtype` and `copy` match the NumPy 2 calling convention.
    /// Default / ``copy=False`` wraps the live `draws` buffer (no extra `Vec` clone).
    /// ``copy=True`` copies into a fresh NumPy-owned array.
    /// ``copy=None`` (default) copies only when a dtype cast requires it.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Tri-state: None = copy-if-needed, Some(false) = never, Some(true) = always.
        let force_copy = match &copy {
            Some(c) => Some(c.is_truthy()?),
            None => None,
        };
        let np = py.import("numpy")?;
        let float64 = np.getattr("float64")?;
        let arr = if force_copy == Some(true) {
            let draws = &slf.borrow().draws;
            PyArray1::from_slice(py, draws).into_any()
        } else {
            let mv = py.import("builtins")?.call_method1("memoryview", (slf,))?;
            np.getattr("asarray")?.call((mv, &float64), None)?
        };
        match dtype {
            None => Ok(arr),
            Some(dt) => match force_copy {
                // copy=None → NumPy default (copy only if the cast requires it).
                None => Ok(np.getattr("asarray")?.call((arr, dt), None)?),
                Some(force) => {
                    // NumPy 2: forward explicit copy= so False cannot silently allocate.
                    // NumPy 1.x lacks copy=; fall back to an explicit same-dtype check.
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("copy", force)?;
                    match np.getattr("asarray")?.call((&arr, &dt), Some(&kwargs)) {
                        Ok(out) => Ok(out),
                        Err(err) if !force => {
                            let msg = err.to_string();
                            if msg.contains("unexpected keyword argument") || msg.contains("copy") {
                                let cur = arr.getattr("dtype")?;
                                if cur.eq(dt)? {
                                    Ok(arr)
                                } else {
                                    Err(crate::value_err(
                                        "Unable to avoid a copy while casting PosteriorArtifact \
                                         dtype; pass copy=True to allow a conversion",
                                    ))
                                }
                            } else {
                                Err(err)
                            }
                        }
                        Err(err) => Err(err),
                    }
                }
            },
        }
    }
}

pub(crate) fn columns_to_batch(
    names: &[String],
    columns: &[PyReadonlyArray1<'_, f64>],
) -> PyResult<RecordBatch> {
    if names.len() != columns.len() {
        return Err(crate::value_err("names and columns must have the same length"));
    }
    if columns.is_empty() {
        return Err(crate::value_err("at least one column required"));
    }
    let n = columns[0].as_array().len();
    for (i, col) in columns.iter().enumerate() {
        let len = col.as_array().len();
        if len != n {
            return Err(crate::value_err(format!(
                "column length mismatch: '{}' has length {}, '{}' has length {}",
                names[0], n, names[i], len
            )));
        }
    }
    let fields: Vec<Field> =
        names.iter().map(|nm| Field::new(nm, DataType::Float64, true)).collect();
    let schema = Schema::new(fields);
    // One pass into the Arrow values buffer. NumPy has no validity bitmap (NaN is its
    // missing-value sentinel), so the Option-per-element route is taken only for a column that
    // actually holds a NaN.
    let arrays: Vec<Arc<dyn arrow_array::Array>> = columns
        .iter()
        .map(|c| {
            let view = c.as_array();
            let contiguous = view.as_slice();
            let has_nan = contiguous.map_or_else(
                || view.iter().any(|v| v.is_nan()),
                |values| values.iter().any(|v| v.is_nan()),
            );
            let array = if has_nan {
                Float64Array::from(
                    view.iter().map(|&v| (!v.is_nan()).then_some(v)).collect::<Vec<Option<f64>>>(),
                )
            } else if let Some(values) = contiguous {
                Float64Array::from(values.to_vec())
            } else {
                Float64Array::from_iter_values(view.iter().copied())
            };
            Arc::new(array) as Arc<dyn arrow_array::Array>
        })
        .collect();
    RecordBatch::try_new(Arc::new(schema), arrays).map_err(py_err)
}

pub(crate) fn tabular_from_numpy(
    names: &[String],
    columns: &[PyReadonlyArray1<'_, f64>],
) -> PyResult<antecedent_data::TabularData> {
    let batch = columns_to_batch(names, columns)?;
    tabular_from_record_batch(&batch).map(|loaded| loaded.data).map_err(py_err)
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
/// Prefers zero-copy borrow of contiguous float64 value buffers. Borrowed buffers must remain
/// unchanged, with no writes through writable aliases, during this call. This conversion probe
/// returns only metadata and retains no data; analysis and prepared-study calls can retain the
/// same kind of borrow for as long as their native data handles live.
#[pyfunction]
fn load_float64_arrow_c_columns(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<ArrowLoadInfo> {
    catch_ffi(|| {
        let cdi_cols = ffi::arrow_c_columns(names, columns)?;
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

pub(crate) fn tabular_from_arrow_c_objs(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<(antecedent_data::TabularData, u64)> {
    let cdi_cols = ffi::arrow_c_columns(names, columns)?;
    let loaded = tabular_from_arrow_c_columns(cdi_cols).map_err(py_err)?;
    Ok((loaded.data, loaded.bytes_borrowed))
}

/// NumPy float64 columns or Arrow CDI exporters → owned [`TabularData`].
pub(crate) fn tabular_from_py_columns(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<(antecedent_data::TabularData, Option<u64>)> {
    if names.len() != columns.len() {
        // Preserve the pre-unification ValueError for NumPy callers that catch it.
        return Err(crate::value_err("names and columns must have the same length"));
    }
    let use_arrow = !columns.is_empty()
        && columns.iter().all(|c| c.hasattr("__arrow_c_array__").is_ok_and(|ok| ok));
    if use_arrow {
        let (data, bytes_borrowed) = tabular_from_arrow_c_objs(names, columns)?;
        return Ok((data, Some(bytes_borrowed)));
    }
    let mut np = Vec::with_capacity(columns.len());
    for c in &columns {
        np.push(c.extract::<PyReadonlyArray1<'_, f64>>().map_err(PyErr::from)?);
    }
    Ok((tabular_from_numpy(&names, &np)?, None))
}

/// Default coalition / semantic cache budget for Python production contexts
/// (matches attribution bench policy).
pub(crate) const PY_DEFAULT_CACHE_MAX_BYTES: u64 = 4_000_000;

/// Resolve an omitted or zero `threads=` to [`antecedent_core::default_user_threads`].
/// An explicit `threads >= 1` is a pin and may exceed the default cap.
pub(crate) fn resolve_user_threads(threads: Option<u32>) -> u32 {
    match threads {
        Some(n) if n >= 1 => n,
        Some(_) => 1,
        None => antecedent_core::default_user_threads(),
    }
}

#[pyfunction]
fn default_user_threads() -> u32 {
    antecedent_core::default_user_threads()
}

pub(crate) fn py_execution_context(seed: u64, threads: u32) -> ExecutionContext {
    py_execution_context_ext(seed, threads, None, None, Some(PY_DEFAULT_CACHE_MAX_BYTES))
}

/// [`py_execution_context`] honouring an optional caller token.
pub(crate) fn py_execution_context_cancel(
    seed: u64,
    threads: u32,
    cancel: Option<PyCancellationToken>,
) -> ExecutionContext {
    let mut ctx = py_execution_context(seed, threads);
    apply_cancel(&mut ctx, cancel);
    ctx
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
    install_cancellation(&mut ctx, cancel);
    ctx.progress = progress;
    ctx
}

/// Install the caller's cancellation token on `ctx`; Ctrl-C also fires it (or, with no token, the
/// run's own).
pub(crate) fn apply_cancel(ctx: &mut ExecutionContext, cancel: Option<PyCancellationToken>) {
    install_cancellation(ctx, cancel.map(|token| token.inner));
}

fn install_cancellation(
    ctx: &mut ExecutionContext,
    cancel: Option<antecedent_core::CancellationToken>,
) {
    if let Some(token) = cancel {
        interrupt::link(&token);
        ctx.cancellation = token;
    } else if let Some(token) = interrupt::ambient_token() {
        ctx.cancellation = token;
    } else {
        // Built ahead of the detached call: the token joins the run that executes this context.
        interrupt::link(&ctx.cancellation);
    }
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
    contract_refusals_api::register(m)?;
    inspect_receipt_api::register(m)?;
    registry::register_module(m)
}

#[cfg(test)]
mod tests {
    use super::panic_payload_msg;
    use std::any::Any;

    #[test]
    fn computation_failure_cannot_be_a_validation_pass() {
        use antecedent_core::{Diagnostic, DiagnosticKind, DiagnosticSeverity};
        use std::sync::Arc;
        let mut d = Diagnostic::new(
            "refute.validator.failed",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "failed fit",
        );
        d.fields = Arc::from([
            (Arc::from("validator"), Arc::from("overlap")),
            (Arc::from("reason"), Arc::from("singular fit")),
        ]);
        let result = super::ValidationSection::from_reports(vec![], &[d]);
        assert!(result.ran);
        assert!(!result.passed);
        assert_eq!(result.count, 1);
        assert!(result.reports.is_empty());
        assert_eq!(result.computation_failures[0].validator, "overlap");
        assert_eq!(result.computation_failures[0].reason, "singular fit");
    }

    /// Every refusal-shaped `CausalError` variant raises one registered reason code, taken
    /// from its own `reason=<code>:` message when it names one and from its variant
    /// otherwise; a code the vocabulary does not list is never attached.
    #[test]
    fn every_native_refusal_variant_names_its_registered_reason_code() {
        use super::{RustCausalError as E, reason_code_of};
        use antecedent::support::SupportRefusal;
        use antecedent_core::{IdentificationStatus, SchemaError};
        use antecedent_graph::GraphError;

        let named =
            format!("{}data_modality_not_licensed: panel", antecedent_core::reason_code::PREFIX);
        let unregistered =
            format!("{}not_a_registered_code: x", antecedent_core::reason_code::PREFIX);
        let cases: Vec<(E, Option<&str>)> = vec![
            (
                E::Graph(GraphError::UnknownVariableName { name: "a".into() }),
                Some("unknown_variable"),
            ),
            (E::Graph(GraphError::Cycle { from: 0, to: 1 }), Some("graph_invalid")),
            (
                E::Schema(SchemaError::UnknownVariableName { name: "a".into() }),
                Some("unknown_variable"),
            ),
            (E::Schema(SchemaError::UnknownVariableId { id: 3 }), Some("unknown_variable")),
            (E::Schema(SchemaError::TooManyVariables), None),
            (E::SchemaMismatch { detail: "x".into() }, Some("schema_mismatch")),
            (
                E::NotIdentified {
                    status: IdentificationStatus::NotIdentified,
                    search_capped: false,
                    message: "no estimand".into(),
                },
                Some("effect_not_identified"),
            ),
            (E::Unsupported { message: "no such route" }, Some("route_not_supported")),
            (
                E::Support { id: SupportRefusal::NotApplicable, message: "typed-impossible" },
                Some("cell_not_applicable"),
            ),
            (
                E::Support { id: SupportRefusal::Refused, message: "not licensed" },
                Some("cell_not_licensed"),
            ),
            (E::Missing { field: "outcome" }, Some("required_option_missing")),
            (E::Conflict { what: "seed", detail: "set once" }, Some("invalid_argument")),
            // A message that names its own registered code wins over the variant default.
            (E::Compile { message: named.clone() }, Some("data_modality_not_licensed")),
            (
                E::Unsupported { message: "refused: reason=cell_not_licensed: x" },
                Some("cell_not_licensed"),
            ),
            // An unregistered code is not attached; the variant default stands.
            (E::Compile { message: unregistered }, None),
            (E::Compile { message: "plain compile failure".into() }, None),
            (E::Resource { message: "memory".into() }, None),
            (E::Cancelled { stage: "estimate" }, None),
        ];
        for (error, expected) in cases {
            let label = error.to_string();
            assert_eq!(reason_code_of(&error).as_deref(), expected, "{label}");
        }
    }

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
