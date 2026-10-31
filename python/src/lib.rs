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
mod adaptive_policy_api;
mod adaptive_procedure_api;
mod array_fit_api;
mod artifact_api;
mod association_api;
mod ate_api;
mod attribution_api;
mod batch_retarget_api;
pub(crate) mod bayesian;
mod bound_study_ranking_api;
mod bounds_api;
mod bridge;
mod callbacks;
mod candidate_screen_api;
mod categorical_treatment_api;
mod closed_pilot_api;
mod cohort_decision_api;
mod cohort_pushdown_api;
mod compact_export_api;
mod compiled_route_api;
mod composition_api;
mod composition_bundle_api;
mod composition_effects_api;
mod composition_verification_api;
mod contract_refusals_api;
mod counterfactual_id_api;
mod cpdag_scenario_api;
mod cross_world_api;
mod cross_world_execution_api;
mod cyclic_system_api;
mod data_diagnostics_api;
mod data_snapshot;
mod decision_api;
mod decision_compile_api;
mod decision_reliability_api;
#[cfg(test)]
mod defaults;
mod derived_treatment_api;
mod descriptive_api;
mod design_api;
mod design_model_api;
mod design_ranking_api;
mod discovered_transport_api;
mod discovered_transport_trial_api;
mod discovery_api;
mod discretized_composition_api;
mod distribution_api;
mod dose_grid_api;
mod effect_constancy_api;
mod effect_constancy_review_api;
mod errors;
mod estimator_config;
mod evidence_ledger_api;
mod evidence_plan_api;
mod execution_api;
mod execution_attempt_api;
mod experiment_api;
mod external_api;
mod external_mechanism_api;
mod factorial_api;
mod fallback_report_api;
mod fallback_stage_api;
mod ffi;
mod file_source_api;
mod finite_discrete_api;
mod forward_source_api;
mod functional_source_api;
mod gcm_api;
mod gcm_estimand_api;
mod graph_build;
mod graph_falsification_api;
mod graph_io;
mod graph_membership_api;
mod graph_score_api;
mod graph_selection_api;
mod graphs;
mod horizon_composition_api;
mod identification_details;
mod ingest;
mod inspect_receipt_api;
mod interference_saturation_api;
mod interrupt;
mod interval_binding_api;
mod intrinsic_influence_api;
mod inverse_outcome_api;
mod inverse_query_api;
mod joint_law_propagation_api;
mod joint_panel_api;
mod joint_response_api;
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
mod mixed_decision_api;
mod mnar_sensitivity_api;
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
mod pag_completion_api;
mod planning_api;
mod policy_api;
mod pooled_measurement_api;
mod posterior;
mod predictive_assessment_api;
mod preflight_api;
mod prepared_api;
mod prepared_options;
mod prior_bank;
mod prior_signal_api;
mod program_claims_api;
mod propagation_chain_api;
mod proposal_arrival_api;
mod provider_api;
mod published_decision_api;
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
#[cfg(feature = "calibration-internal")]
mod recalc_joint_bayesian_api;
mod recalc_standing_api;
mod recalc_static_api;
mod recalc_temporal_api;
mod recipe_logistic_api;
mod recovery_api;
mod recovery_chain_api;
mod regimes_api;
mod registry;
mod release_migration_api;
mod sequential_evaluation_api;
mod temporal_design_api;
mod tiered_identify_api;
mod unlicensed_design_api;
pub(crate) use registry::transport_defaults;
mod interface_contract_api;
mod intervention_callback_api;
mod intervention_program_api;
mod period_refresh_api;
mod posterior_boundary_api;
mod posterior_row_api;
mod provider_callbacks_api;
mod repair_api;
mod response_api;
mod retained_refresh_api;
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
mod staleness_api;
mod state_api;
mod structure_stability_api;
mod study_planning_api;
mod support_matrix_api;
mod survival_api;
mod target_validation_api;
mod temporal_api;
mod temporal_counterfactual_api;
mod temporal_extension_api;
mod temporal_license;
mod temporal_transport_api;
mod transport_admg_conditional_api;
mod transport_buffers;
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
mod typed_mechanism_transport_api;
mod vector_treatment_api;

pub(crate) use ate_api::{GraphEdge, ate_result_from_analysis};
pub(crate) use discovery_api::{
    DiscoveredLink, PcmciDiscoveryResult, series_from_batch, tabular_from_batch,
};
pub(crate) use errors::*;
pub(crate) use graph_build::{
    dag_from_named_edges, parse_dummy_ci_modes, require_named_graph_order, schema_var_id,
    series_from_tabular, temporal_dag_from_lagged_edges, temporal_dag_from_schema_edges,
};
pub(crate) use ingest::*;
pub(crate) use posterior::*;
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

/// Posterior mean, standard deviation, interval and draws of a family contrast.
type FamilyContrastPosterior = (f64, f64, (f64, f64), Arc<[f64]>);
/// One outcome's counterfactual contrast: outcome name, mean, standard error, unit effects,
/// unit intervals, unit extrapolation flags and whether the unit effects are homogeneous.
type CounterfactualContrastRow =
    (String, f64, Option<f64>, Vec<f64>, Option<Vec<(f64, f64)>>, Option<Vec<bool>>, bool);
type MechanismWireEntry = (String, Option<f64>, Option<Vec<f64>>, Option<f64>);
type ModelBundleSummary = (Vec<String>, Vec<(u32, u32)>, usize);

use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use antecedent::design::{DecisionProblem, evaluate_decision as facade_evaluate_decision};
use antecedent::discovery::{
    DiscoverParams, DiscoveryPerformanceRecord, LaggedParent, MultiDatasetConstraints, PcSepsets,
    RegimeAssignment, ScoredLink, StaticDiscoverParams, pag_definite_directed_edge_count,
    two_regime_half_split,
};
use antecedent::error::PendingEdge as FacadePendingEdge;
use antecedent::estimate::{TemporalLinearPredictor, TemporalMediationEstimator};
use antecedent::gcm::fit_gcm;
use antecedent::io::{decode_causal_posterior_bytes, encode_causal_posterior_bytes};
use antecedent::review::{PendingCpdagReview, PendingGraphReview};
use antecedent::{
    AcceptedGraph, BayesianConfig, CausalError as RustCausalError, InferenceMode, RefuteSuite,
    Study,
};
use antecedent_core::{
    AllocationMethod, AttributionComponents, CachePolicy, ChangeAttributionQuery, DistributionRef,
    ExecutionContext, Intervention, KernelPolicy, Lag, MechanismChangeQuery, MediationContrast,
    MediationQuery, PopulationRegistry, PopulationSelector, PredicateExpr, RegimeId, SchemaError,
    ShapleyConfig, TargetPopulation, TemporalEffectQuery, TemporalPolicy, UnitChangeQuery, VERSION,
    Value, VariableId,
};
use antecedent_data::{
    DataError, MultiEnvironmentData, TableView, TimeSeriesData, tabular_from_arrow_c_columns,
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
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;

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
    /// How identified graph atoms were combined (`same_estimand_weighted_mean`,
    /// `identified_set_envelope` or `graph_dependent_atoms`), when the route aggregates
    /// across graphs.
    #[pyo3(get)]
    pub(crate) structural_aggregation_policy: Option<String>,
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
    /// Every interval this execution reported (method, SE kind, level, dependence rule,
    /// replicate and draw counts), primary first, as JSON.
    #[pyo3(get)]
    pub(crate) interval_bindings_json: Option<String>,
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
    refutation_passed: Option<bool>,
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
    /// row per route record (passed or failed, whatever the diagnostic's code); the mass is `None` when the route did not report one.
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
    prior_ppc_dispersion_p_value: Option<f64>,
    #[pyo3(get)]
    prior_ppc_noise: Option<String>,
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
    posterior_ppc_dispersion_p_value: Option<f64>,
    #[pyo3(get)]
    posterior_ppc_noise: Option<String>,
    /// Posterior predictive replicates retained at fit time, when declared.
    #[pyo3(get)]
    predictive_replicates: Option<predictive_assessment_api::PredictiveReplicateLaws>,
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
    /// One contrast per outcome of a multi-outcome or joint-baseline counterfactual:
    /// `(outcome, mean, standard_error, unit_effects, intervals, extrapolative, homogeneous)`.
    #[pyo3(get)]
    counterfactual_contrasts: Option<Vec<CounterfactualContrastRow>>,
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
    /// Sequentially randomized unit-period design or regime survival (unlicensed).
    #[pyo3(get)]
    temporal_design: Option<temporal_design_api::TemporalDesignSection>,
    /// Sharp bounds on the average effect under a binary instrument (unlicensed): an identified
    /// set, never a point.
    #[pyo3(get)]
    iv_bounds: Option<bounds_api::IvBoundsSection>,
    /// Noncausal association of the observed data (unlicensed, calibration unmeasured): never
    /// an intervention effect, ranks no action.
    #[pyo3(get)]
    association: Option<association_api::AssociationSection>,
    /// Heterogeneity-robust or covariate-adjusted DiD, conformal or multi-unit synthetic
    /// panel, or optioned local ratio (unlicensed, calibration unmeasured).
    #[pyo3(get)]
    quasi_rollout: Option<quasi_api::QuasiRolloutSection>,
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
    /// Descriptive scope kind; absent for legacy reports.
    #[pyo3(get)]
    scope_kind: Option<String>,
    /// Scoped target identity in hexadecimal.
    #[pyo3(get)]
    scope_target_identity: Option<String>,
    /// Outer posterior atom key for a completion-scoped report.
    #[pyo3(get)]
    scope_outer_atom: Option<u64>,
    /// Full canonical completion digest in hexadecimal.
    #[pyo3(get)]
    scope_completion_digest: Option<String>,
    /// Temporal completion horizon index.
    #[pyo3(get)]
    scope_horizon_index: Option<usize>,
    /// Temporal completion horizon.
    #[pyo3(get)]
    scope_horizon: Option<u32>,
}

impl From<&RefutationReport> for RefutationReportView {
    fn from(r: &RefutationReport) -> Self {
        use antecedent_validate::RefutationScope;
        let (kind, target, atom, digest, horizon_index, horizon) = match r.scope.as_ref() {
            Some(RefutationScope::WholeTarget { target }) => {
                (Some("whole_target"), Some(target), None, None, None, None)
            }
            Some(RefutationScope::TemporalClassCompletion {
                outer_atom,
                horizon_index,
                horizon,
                completion_digest,
                target,
            }) => (
                Some("temporal_class_completion"),
                Some(target),
                Some(*outer_atom),
                Some(completion_digest),
                Some(*horizon_index),
                Some(*horizon),
            ),
            Some(RefutationScope::StaticClassCompletion {
                outer_atom,
                completion_digest,
                target,
            }) => (
                Some("static_class_completion"),
                Some(target),
                Some(*outer_atom),
                Some(completion_digest),
                None,
                None,
            ),
            None => (None, None, None, None, None, None),
        };
        Self {
            refuter: r.refuter.to_string(),
            original_ate: r.original_ate,
            refuted_ate: r.refuted_ate,
            comparison: r.comparison,
            informative: r.informative,
            passed: r.passed,
            failure_condition: r.failure_condition.as_ref().map(std::string::ToString::to_string),
            replicates: r.replicates,
            scope_kind: kind.map(str::to_owned),
            scope_target_identity: target.map(|target| target.identity_hex()),
            scope_outer_atom: atom,
            scope_completion_digest: digest.map(|digest| antecedent_core::identity::to_hex(digest)),
            scope_horizon_index: horizon_index,
            scope_horizon: horizon,
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
    /// Fraction of rows a declared overlap trim excluded (0 without a trim); the estimate
    /// targets the retained common-support rows. `None` on the temporal DTO.
    #[pyo3(get)]
    pub(crate) overlap_excluded_fraction: Option<f64>,
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
    /// `(posterior mean, standard deviation, (lower, upper), draws)` of the family contrast;
    /// draw `d` of every family member shares one row-weight vector. The draws are shared
    /// immutable storage; the getter builds the Python list from it once per access.
    family_contrast_posterior: Option<FamilyContrastPosterior>,
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

#[pymethods]
impl EstimateSection {
    /// `(posterior mean, standard deviation, (lower, upper), draws)` of the family contrast, or
    /// `None` without a family posterior.
    #[getter]
    #[allow(
        clippy::type_complexity,
        reason = "the tuple is the documented Python shape of the family contrast posterior"
    )]
    fn family_contrast_posterior<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(f64, f64, (f64, f64), Bound<'py, pyo3::types::PyList>)>> {
        self.family_contrast_posterior
            .as_ref()
            .map(|(mean, sd, interval, draws)| {
                Ok((*mean, *sd, *interval, pyo3::types::PyList::new(py, draws.iter().copied())?))
            })
            .transpose()
    }
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
    pub structural_aggregation_policy: Option<String>,
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
    let estimate = result.estimate().as_effect().unwrap_or(&absent_effect);
    let resampling = estimate.diagnostics.resampling.as_ref().map(ResamplingSection::from_report);
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
    let overlap_excluded_fraction =
        estimate.diagnostics.overlap_report.as_ref().map(|r| r.excluded_fraction);
    let estimate = EstimateSection {
        ate: estimate.ate.is_finite().then_some(estimate.ate),
        se_analytic: estimate.se_analytic(),
        se_bootstrap: estimate.se_bootstrap(),
        estimator_id: estimator_id.clone(),
        method: method.clone(),
        overlap_ess,
        overlap_propensity_min,
        overlap_excluded_fraction,
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
        family_contrast_posterior: estimate.family.contrast_posterior.as_ref().map(|posterior| {
            (
                posterior.posterior_mean,
                posterior.standard_deviation,
                posterior.interval,
                Arc::from(posterior.draws.as_slice()),
            )
        }),
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
                result.candidate_selection().as_ref().map(|s| CandidateSelectionSection {
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
        native_execution: result.performance.native_execution.clone(),
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
        resampling,
    };
    let (evidence_status, allowlist_reason, allowlist_parent) =
        evidence_status_parts(result.support_status());
    // A query with no matrix cell (mechanism / unit change) is off the axis by design; say so
    // rather than leaving a reader to mistake it for a licensed cell that was never stamped.
    let evidence_status = match (evidence_status, result.support_status()) {
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
        structural_aggregation_policy: structural
            .and_then(|m| m.aggregation_policy)
            .map(|policy| policy.as_str().to_owned()),
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
/// `ran` records whether any check ran. `passed` is absent when only descriptive
/// scoped checks ran, because they do not establish a whole-result verdict.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct ValidationSection {
    #[pyo3(get)]
    pub(crate) passed: Option<bool>,
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
        let unscoped: Vec<_> =
            reports.iter().filter(|report| report.scope_kind.is_none()).collect();
        let has_scoped = reports.iter().any(|report| report.scope_kind.is_some());
        let passed = if has_scoped {
            None
        } else {
            Some(ran && computation_failures.is_empty() && unscoped.iter().all(|r| r.passed))
        };
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
    native_execution: Option<antecedent_core::execution_facts::ExecutionFacts>,
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
    /// The resampling procedure behind a requested bootstrap interval (`None` when no
    /// interval was requested).
    #[pyo3(get)]
    resampling: Option<ResamplingSection>,
}

/// The complete resampling procedure behind a requested bootstrap interval.
#[pyclass(skip_from_py_object)]
#[derive(Clone, Debug)]
pub(crate) struct ResamplingSection {
    /// Nominal level of the percentile interval.
    #[pyo3(get)]
    level: f64,
    /// Replicates per wave.
    #[pyo3(get)]
    wave_size: u32,
    /// Successes required before a tolerance may stop the run.
    #[pyo3(get)]
    min_replicates: u32,
    /// Replicate cap (the estimator's declared replicate count).
    #[pyo3(get)]
    max_replicates: u32,
    /// Relative Monte Carlo error target of the standard error.
    #[pyo3(get)]
    se_rel_tol: Option<f64>,
    /// Relative Monte Carlo error target of the interval half-width.
    #[pyo3(get)]
    half_width_rel_tol: Option<f64>,
    /// Percentile interval of the successes, when licensed.
    #[pyo3(get)]
    interval: Option<(f64, f64)>,
    /// Why the run ended: `converged`, `max_replicates` or `cancelled`.
    #[pyo3(get)]
    stop: String,
    /// Replicates attempted in the completed waves, failures included.
    #[pyo3(get)]
    attempted: u32,
    /// Per wave: `(wave, first replicate, attempted, ok, estimator errors, non-finite, declined)`.
    #[pyo3(get)]
    waves: Vec<(u32, u32, u32, u32, u32, u32, u32)>,
    /// Relative Monte Carlo error of the standard error at the stop.
    #[pyo3(get)]
    se_rel_mc_error: Option<f64>,
    /// Relative Monte Carlo error of the interval half-width at the stop.
    #[pyo3(get)]
    half_width_rel_mc_error: Option<f64>,
}

impl ResamplingSection {
    fn from_report(report: &antecedent_estimate::ResamplingReport) -> Self {
        use antecedent_estimate::adaptive_resample_waves::StopReason;
        Self {
            level: report.rule.level,
            wave_size: report.rule.wave_size,
            min_replicates: report.rule.min_replicates,
            max_replicates: report.rule.max_replicates,
            se_rel_tol: report.rule.se_rel_tol,
            half_width_rel_tol: report.rule.half_width_rel_tol,
            interval: report.interval,
            stop: match report.stop {
                StopReason::Converged => "converged",
                StopReason::MaxReplicates => "max_replicates",
                StopReason::Cancelled => "cancelled",
            }
            .to_owned(),
            attempted: report.attempted,
            waves: report
                .waves
                .iter()
                .map(|w| {
                    (
                        w.wave,
                        w.first_replicate,
                        w.attempted,
                        w.ok,
                        w.estimator_errors,
                        w.non_finite,
                        w.declined,
                    )
                })
                .collect(),
            se_rel_mc_error: report.se_rel_mc_error,
            half_width_rel_mc_error: report.half_width_rel_mc_error,
        }
    }
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
    ctx.progress = interrupt::bind_progress(progress);
    execution_api::apply(&mut ctx);
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
    data_diagnostics_api::register(m)?;
    data_snapshot::register(m)?;
    execution_api::register(m)?;
    tiered_identify_api::register(m)?;
    contract_refusals_api::register(m)?;
    inspect_receipt_api::register(m)?;
    release_migration_api::register(m)?;
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
        assert_eq!(result.passed, Some(false));
        assert_eq!(result.count, 1);
        assert!(result.reports.is_empty());
        assert_eq!(result.computation_failures[0].validator, "overlap");
        assert_eq!(result.computation_failures[0].reason, "singular fit");
    }

    #[test]
    fn scoped_only_validation_has_no_whole_verdict() {
        let target = antecedent_validate::target_native::RefutationTarget::new(
            antecedent_validate::target_native::TargetFamily::TemporalEffect,
            antecedent_validate::target_native::TargetQuadrant::TEMPORAL_FREQUENTIST,
            ["completion-level"],
            [7; 32],
        )
        .unwrap();
        let mut report = antecedent_validate::RefutationReport::new(
            "target.subset_stability",
            12.0,
            12.1,
            0.5,
            true,
            true,
            None,
            30,
        );
        report.scope = Some(antecedent_validate::RefutationScope::TemporalClassCompletion {
            outer_atom: 2,
            horizon_index: 0,
            horizon: 1,
            completion_digest: [3; 32],
            target,
        });
        let view = super::RefutationReportView::from(&report);
        let validation = super::ValidationSection::from_reports(vec![view], &[]);
        assert!(validation.ran);
        assert_eq!(validation.passed, None);
        assert_eq!(validation.count, 1);
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
            // A structured callback failure names the code it carries; its text is not read.
            (
                E::Model(antecedent_model::ModelError::CallbackFailed(
                    antecedent_core::CallbackFailure::new(
                        Some(1),
                        "Python x failed",
                        Some("route_not_supported".into()),
                    ),
                )),
                Some("route_not_supported"),
            ),
            (
                E::Model(antecedent_model::ModelError::CallbackFailed(
                    antecedent_core::CallbackFailure::new(None, named.clone(), None),
                )),
                None,
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

#[pymethods]
impl PerformanceSection {
    /// Engine-produced execution observations, separate from plan estimates.
    #[getter]
    fn native_execution(&self) -> Option<inspect_receipt_api::PyExecutionFacts> {
        self.native_execution.clone().map(|inner| inspect_receipt_api::PyExecutionFacts { inner })
    }
}
