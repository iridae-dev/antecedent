//! Registration table for `antecedent._native`.
//!
//! The pymodule entry stays in `lib.rs` so `PyInit__native` is exported from
//! the crate root. The class and function lists live here.

use pyo3::prelude::*;

use super::array_fit_api;
use super::sequential_evaluation_api;
use super::staleness_api;
use super::temporal_design_api;
use super::unlicensed_design_api;
use super::{
    AnalysisResult, ArrowLoadInfo, AteAnalysisResult, BayesianConfig, CandidateSelectionSection,
    CausalAttributionError, CausalCancelledError, CausalCompileError, CausalCounterfactualError,
    CausalDataError, CausalDesignError, CausalDiscoveryError, CausalError, CausalEstimateError,
    CausalGraphError, CausalIdentifyError, CausalModelError, CausalPendingEdge,
    CausalResourceError, CausalReviewError, CausalSerializationError, CausalStateError,
    CausalUnsupportedError, CausalValidateError, DiscoveredLink, DistributionAtomSection,
    EstimateSection, GcmIteResult, GcmSampleResult, GraphEdge, IdentificationSection,
    MediationEffectsSummary, PcmciDiscoveryResult, PerformanceSection, PosteriorArtifact,
    PosteriorSection, PredictSummary, ProbabilityIntervalSection, PyCancellationToken, PyDict,
    RefutationReportView, RefuteSuite, RpcmciDiscoverySummary, ScoreInferenceSection,
    ScoreTableSection, ValidationFailureSection, ValidationSection, adaptive_inference_api,
    adaptive_participant_api, adaptive_procedure_api, artifact_api, association_api, ate_api,
    attribution_api, bayesian, bounds_api, candidate_screen_api, categorical_treatment_api,
    closed_pilot_api, cohort_decision_api, compact_export_api, composition_api,
    composition_bundle_api, composition_verification_api, counterfactual_id_api,
    cpdag_scenario_api, cross_world_api, cross_world_execution_api, decision_api,
    decision_reliability_api, default_user_threads, derived_treatment_api, descriptive_api,
    design_api, design_ranking_api, discovered_transport_api, discovery_api,
    discretized_composition_api, distribution_api, dose_grid_api, effect_constancy_api,
    effect_constancy_review_api, evidence_ledger_api, evidence_plan_api, execution_attempt_api,
    experiment_api, external_api, factorial_api, forward_source_api, gcm_api,
    graph_falsification_api, graph_io, graphs, horizon_composition_api, interface_contract_api,
    interference_saturation_api, inverse_outcome_api, inverse_query_api, joint_law_propagation_api,
    joint_transport_api, latent_class_api, learned_continuous_api, learned_joint_api,
    learned_trial_api, load_float64_arrow_c_columns, load_float64_columns,
    matched_case_control_api, measured_inference_api, measurement_model_api,
    mechanism_composition_api, mechanism_discrepancy_api, msm_sensitivity_api,
    multi_level_treatment_api, nested_bayesian_api, nested_fisher_api, nonlinear_mediation_api,
    observation_api, observation_query_api, observational_interference_api,
    observational_policy_api, observational_survival_api, policy_api, posterior_boundary_api,
    posterior_row_api, predictive_assessment_api, preflight_api, prepared_api, prior_bank,
    prior_signal_api, program_claims_api, propagation_chain_api, proposal_arrival_api,
    provider_api, provider_callbacks_api, quasi_api, recalc_adjusted_api, recalc_api,
    recalc_bayesian_api, recalc_cell_api, recalc_composite_api, recalc_design_api, recalc_dr_api,
    recalc_external_api, recalc_static_api, recalc_temporal_api, recovery_api, recovery_chain_api,
    regimes_api, repair_api, response_api, scenario_invariance_api, scientific_bridge_api,
    selective_discovery_api, sensitivity_decision_api, sensitivity_source_api,
    set_not_identified_error_class, set_review_error_class, set_unsupported_error_class,
    set_value_error_class, smoothed_dose_api, source_evidence_api, source_projection_api,
    stability, state_api, study_planning_api, survival_api, temporal_api,
    temporal_counterfactual_api, temporal_extension_api, temporal_license, temporal_transport_api,
    transport_admg_conditional_api, transport_counterfactual_api,
    transport_counterfactual_sampled_api, transport_exact_api, transport_grid_api,
    transport_interference_api, transport_joint_sensitivity_api, transport_mixed_api,
    transport_mz_api, transport_revalidation_api, transport_scenario_api,
    transport_statistical_api, transport_z_api, transported_constructors_api,
    transported_counterfactual_api, transported_response_api, value_err, vector_treatment_api,
};

pub(super) fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_native_errors(m)?;
    register_native_functions(m)?;
    register_native_classes(m)?;
    super::compiled_route_api::register(m)?;
    gcm_api::register(m)?;
    graph_falsification_api::register(m)?;
    super::intrinsic_influence_api::register(m)?;
    super::file_source_api::register(m)?;
    super::intervention_callback_api::register(m)?;
    super::intervention_program_api::register(m)?;
    state_api::register(m)?;
    prepared_api::register(m)?;
    bayesian::register(m)?;
    stability::register(m)?;
    prior_bank::register(m)?;
    prior_signal_api::register(m)?;
    response_api::register(m)?;
    temporal_license::register(m)?;
    transport_interference_api::register(m)?;
    transport_exact_api::register(m)?;
    transport_grid_api::register(m)?;
    transport_statistical_api::register(m)?;
    transport_z_api::register(m)?;
    transport_joint_sensitivity_api::register(m)?;
    transport_mixed_api::register(m)?;
    study_planning_api::register(m)?;
    repair_api::register(m)?;
    evidence_plan_api::register(m)?;
    recovery_api::register(m)?;
    closed_pilot_api::register(m)?;
    nested_fisher_api::register(m)?;
    nested_bayesian_api::register(m)?;
    learned_joint_api::register(m)?;
    measured_inference_api::register(m)?;
    joint_law_propagation_api::register(m)?;
    propagation_chain_api::register(m)?;
    cross_world_execution_api::register(m)?;
    joint_transport_api::register(m)?;
    msm_sensitivity_api::register(m)?;
    mechanism_discrepancy_api::register(m)?;
    inverse_outcome_api::register(m)?;
    inverse_query_api::register(m)?;
    forward_source_api::register(m)?;
    transported_response_api::register(m)?;
    transport_revalidation_api::register(m)?;
    discretized_composition_api::register(m)?;
    sensitivity_decision_api::register(m)?;
    sensitivity_source_api::register(m)?;
    transport_admg_conditional_api::register(m)?;
    transport_mz_api::register(m)?;
    transport_counterfactual_api::register(m)?;
    transport_counterfactual_sampled_api::register(m)?;
    transport_scenario_api::register(m)?;
    cpdag_scenario_api::register(m)?;
    temporal_transport_api::register(m)?;
    temporal_extension_api::register(m)?;
    cross_world_api::register(m)?;
    counterfactual_id_api::register(m)?;
    learned_trial_api::register(m)?;
    learned_continuous_api::register(m)?;
    smoothed_dose_api::register(m)?;
    derived_treatment_api::register(m)?;
    matched_case_control_api::register(m)?;
    descriptive_api::register(m)?;
    candidate_screen_api::register(m)?;
    preflight_api::register(m)?;
    observation_api::register(m)?;
    bounds_api::register(m)?;
    association_api::register(m)?;
    artifact_api::register(m)?;
    array_fit_api::register(m)?;
    decision_api::register(m)?;
    distribution_api::register(m)?;
    external_api::register(m)?;
    effect_constancy_api::register(m)?;
    effect_constancy_review_api::register(m)?;
    recalc_api::register(m)?;
    recalc_adjusted_api::register(m)?;
    recalc_dr_api::register(m)?;
    recalc_design_api::register(m)?;
    recalc_bayesian_api::register(m)?;
    recalc_temporal_api::register(m)?;
    recalc_external_api::register(m)?;
    recalc_composite_api::register(m)?;
    staleness_api::register(m)?;
    super::target_validation_api::register(m)?;
    execution_attempt_api::register(m)?;
    source_evidence_api::register(m)?;
    source_projection_api::register(m)?;
    proposal_arrival_api::register(m)?;
    recalc_static_api::register(m)?;
    recalc_cell_api::register(m)?;
    dose_grid_api::register(m)?;
    recovery_chain_api::register(m)?;
    vector_treatment_api::register(m)?;
    categorical_treatment_api::register(m)?;
    multi_level_treatment_api::register(m)?;
    observational_survival_api::register(m)?;
    observational_policy_api::register(m)?;
    compact_export_api::register(m)?;
    cohort_decision_api::register(m)?;
    nonlinear_mediation_api::register(m)?;
    crate::cyclic_system_api::register(m)?;
    crate::joint_panel_api::register(m)?;
    latent_class_api::register(m)?;
    temporal_counterfactual_api::register(m)?;
    transported_counterfactual_api::register(m)?;
    scenario_invariance_api::register(m)?;
    provider_api::register(m)?;
    composition_api::register(m)?;
    composition_bundle_api::register(m)?;
    measurement_model_api::register(m)?;
    mechanism_composition_api::register(m)?;
    observation_query_api::register(m)?;
    evidence_ledger_api::register(m)?;
    sequential_evaluation_api::register(m)?;
    scientific_bridge_api::register(m)?;
    composition_verification_api::register(m)?;
    interface_contract_api::register(m)?;
    provider_callbacks_api::register(m)?;
    selective_discovery_api::register(m)?;
    predictive_assessment_api::register(m)?;
    adaptive_inference_api::register(m)?;
    adaptive_participant_api::register(m)?;
    adaptive_procedure_api::register(m)?;
    discovered_transport_api::register(m)?;
    transported_constructors_api::register(m)?;
    decision_reliability_api::register(m)?;
    program_claims_api::register(m)?;
    m.add("__version__", antecedent_core::VERSION)?;
    // Surfaced so the Python package can refuse to stay silent when an
    // unoptimized extension sneaks in: a stale editable install rebuilt through
    // PEP 517/660 defaults to Cargo's debug profile, which keeps every estimate
    // bit-identical while making estimation ~50x slower.
    m.add("__build_optimized__", !cfg!(debug_assertions))?;
    Ok(())
}

fn register_native_errors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Every exception reads `remedy`: `None` unless the refusal names one (see
    // [`with_remedy`]), so a caller never needs `getattr(err, "remedy", None)`.
    m.py().get_type::<CausalError>().setattr("remedy", m.py().None())?;
    m.py().get_type::<CausalError>().setattr("refusal_fields", m.py().None())?;
    m.py()
        .get_type::<CausalError>()
        .setattr("route_diagnostics", pyo3::types::PyTuple::empty(m.py()))?;
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

/// Inference settings the transport query modalities apply when a caller omits them.
pub(crate) mod transport_defaults {
    /// Bootstrap replicates of a statistical transport query.
    pub(crate) const BOOTSTRAP: u32 = 199;
    /// Central coverage level of a statistical transport interval.
    pub(crate) const COVERAGE_LEVEL: f64 = 0.95;
}

/// The study builder's omitted-default table, read from the builder itself.
///
/// `bootstrap` applies on resampling routes (the builder omits it on static and
/// Bayesian response surfaces and counterfactuals); `refute` is downgraded to
/// `none` on a cell that does not license it; `latency` is never injected.
#[pyfunction]
fn omitted_defaults(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("bootstrap", antecedent::StudyBuilder::OMITTED_BOOTSTRAP)?;
    dict.set_item(
        "refute",
        match antecedent::StudyBuilder::OMITTED_REFUTE {
            RefuteSuite::None => "none",
            RefuteSuite::Cheap => "cheap",
            RefuteSuite::PlaceboAndRcc => "placebo",
            RefuteSuite::Full => "full",
        },
    )?;
    dict.set_item("latency", py.None())?;
    // Draw budgets belong to the Bayesian backend constructors.
    dict.set_item("n_draws", BayesianConfig::laplace().n_draws)?;
    dict.set_item("n_draws_hmc", BayesianConfig::hmc().n_draws)?;
    // Keyword defaults the native signatures repeat as literals; a test compares them here.
    dict.set_item("discovery_alpha", antecedent::discovery_defaults::DEFAULT_ALPHA)?;
    dict.set_item(
        "discovery_max_cond_size",
        antecedent::discovery_defaults::DEFAULT_MAX_COND_SIZE,
    )?;
    dict.set_item("prior_scale", BayesianConfig::laplace().prior_scale)?;
    // Propensity overlap default shared by every propensity-score estimator.
    let (clip, trim) = match antecedent_estimate::default_propensity_overlap() {
        antecedent_estimate::OverlapPolicy::RequireDiagnostics { clip, trim } => (clip, trim),
        antecedent_estimate::OverlapPolicy::ExplicitOverride => {
            return Err(value_err("the default propensity overlap policy requires diagnostics"));
        }
    };
    dict.set_item("overlap_clip", clip)?;
    dict.set_item("overlap_trim", trim)?;
    // Inference settings of the transport query modalities.
    dict.set_item("transport_bootstrap", transport_defaults::BOOTSTRAP)?;
    dict.set_item("transport_coverage_level", transport_defaults::COVERAGE_LEVEL)?;
    Ok(dict.into())
}

#[pyfunction]
fn identification_status_names() -> Vec<String> {
    use antecedent_core::IdentificationStatus as S;
    fn name(status: S) -> String {
        match status {
            S::NonparametricallyIdentified
            | S::IdentifiedUnderParametricRestrictions
            | S::IdentifiedUnderPriorRestrictions
            | S::PartiallyIdentified
            | S::GraphDependent
            | S::NotIdentified => format!("{status:?}"),
        }
    }
    [
        S::NonparametricallyIdentified,
        S::IdentifiedUnderParametricRestrictions,
        S::IdentifiedUnderPriorRestrictions,
        S::PartiallyIdentified,
        S::GraphDependent,
        S::NotIdentified,
    ]
    .into_iter()
    .map(name)
    .collect()
}

/// Registered runtime-refusal reason codes (`parity/reason_codes.toml`), the
/// list `CausalUnsupportedError(reason_code=...)` validates against.
#[pyfunction]
fn runtime_refusal_codes() -> Vec<&'static str> {
    antecedent_core::reason_code::RUNTIME_REFUSAL_CODES.to_vec()
}

/// The committed host-integration JSON Schema, exactly as generated.
#[pyfunction]
fn host_schema_text() -> &'static str {
    antecedent::host::HOST_SCHEMA_TEXT
}

/// Host status (`supported`, `partial`, `unsupported`) of a claim from its wire names;
/// `None` for a name outside the registered vocabularies.
#[pyfunction]
fn host_status_of_claim(
    kind: &str,
    identification: &str,
    support: &str,
    evaluated: &str,
) -> Option<&'static str> {
    antecedent::host::HostStatus::of_claim_names(kind, identification, support, evaluated)
        .map(antecedent::host::HostStatus::as_str)
}

/// Host status of an operation from its applicability and readiness wire names; `None` for a
/// name outside the registered vocabularies.
#[pyfunction]
#[pyo3(signature = (applicability, readiness=None))]
fn host_status_of_operation(applicability: &str, readiness: Option<&str>) -> Option<&'static str> {
    antecedent::host::HostStatus::of_operation_names(applicability, readiness)
        .map(antecedent::host::HostStatus::as_str)
}

fn register_native_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(host_schema_text, m)?)?;
    m.add_function(wrap_pyfunction!(host_status_of_claim, m)?)?;
    m.add_function(wrap_pyfunction!(host_status_of_operation, m)?)?;
    m.add_function(wrap_pyfunction!(load_float64_columns, m)?)?;
    m.add_function(wrap_pyfunction!(load_float64_arrow_c_columns, m)?)?;
    m.add_function(wrap_pyfunction!(set_review_error_class, m)?)?;
    m.add_function(wrap_pyfunction!(set_unsupported_error_class, m)?)?;
    m.add_function(wrap_pyfunction!(set_value_error_class, m)?)?;
    m.add_function(wrap_pyfunction!(set_not_identified_error_class, m)?)?;
    m.add_function(wrap_pyfunction!(omitted_defaults, m)?)?;
    m.add_function(wrap_pyfunction!(default_user_threads, m)?)?;
    m.add_function(wrap_pyfunction!(identification_status_names, m)?)?;
    m.add_function(wrap_pyfunction!(runtime_refusal_codes, m)?)?;
    ate_api::register(m)?;
    discovery_api::register(m)?;
    temporal_api::register(m)?;
    attribution_api::register(m)?;
    horizon_composition_api::register(m)?;
    graph_io::register(m)?;
    design_api::register(m)?;
    design_ranking_api::register(m)?;
    policy_api::register(m)?;
    experiment_api::register(m)?;
    interference_saturation_api::register(m)?;
    observational_interference_api::register(m)?;
    factorial_api::register(m)?;
    super::planning_api::register(m)?;
    regimes_api::register(m)?;
    unlicensed_design_api::register(m)?;
    temporal_design_api::register(m)?;
    super::pooled_measurement_api::register(m)?;
    quasi_api::register(m)?;
    survival_api::register(m)?;
    posterior_row_api::register(m)?;
    posterior_boundary_api::register(m)?;
    Ok(())
}

fn register_native_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CausalPendingEdge>()?;
    m.add_class::<ArrowLoadInfo>()?;
    m.add_class::<AteAnalysisResult>()?;
    m.add_class::<RefutationReportView>()?;
    m.add_class::<IdentificationSection>()?;
    m.add_class::<EstimateSection>()?;
    m.add_class::<CandidateSelectionSection>()?;
    m.add_class::<DistributionAtomSection>()?;
    m.add_class::<ProbabilityIntervalSection>()?;
    m.add_class::<ScoreTableSection>()?;
    m.add_class::<ScoreInferenceSection>()?;
    m.add_class::<PosteriorSection>()?;
    m.add_class::<ValidationSection>()?;
    m.add_class::<ValidationFailureSection>()?;
    m.add_class::<PerformanceSection>()?;
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
    m.add_class::<graphs::TemporalAdmg>()?;
    m.add_class::<graphs::TemporalCpdag>()?;
    m.add_class::<graphs::TemporalPag>()?;
    Ok(())
}
