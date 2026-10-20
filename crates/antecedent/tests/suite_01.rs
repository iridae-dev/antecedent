//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "adaptive_inference_facade.rs"]
mod adaptive_inference_facade;
#[path = "adaptive_participant_acceptance.rs"]
mod adaptive_participant_acceptance;
#[path = "adaptive_participant_facade.rs"]
mod adaptive_participant_facade;
#[path = "adaptive_procedure_facade.rs"]
mod adaptive_procedure_facade;
#[path = "adaptive_repeated_measures_consumer.rs"]
mod adaptive_repeated_measures_consumer;
#[path = "adaptive_repeated_measures_facade.rs"]
mod adaptive_repeated_measures_facade;
#[path = "admg_conditional_transport_lifecycle.rs"]
mod admg_conditional_transport_lifecycle;
#[path = "admg_functional_effect_route_evidence.rs"]
mod admg_functional_effect_route_evidence;
#[path = "admg_intervention_response_route_evidence.rs"]
mod admg_intervention_response_route_evidence;
#[path = "admg_posterior_ate.rs"]
mod admg_posterior_ate;
#[path = "admg_posterior_response.rs"]
mod admg_posterior_response;
#[path = "admg_response_curve_route_evidence.rs"]
mod admg_response_curve_route_evidence;
#[path = "arrow_null_ingest.rs"]
mod arrow_null_ingest;
#[path = "artifact_compat_earlier_release.rs"]
mod artifact_compat_earlier_release;
#[path = "artifact_migrate.rs"]
mod artifact_migrate;
#[path = "attribution.rs"]
mod attribution;
#[path = "augmented_panel_did_route.rs"]
mod augmented_panel_did_route;
#[path = "batch_analysis.rs"]
mod batch_analysis;
#[path = "batch_analyze_many.rs"]
mod batch_analyze_many;
#[path = "batch_retarget.rs"]
mod batch_retarget;
#[path = "batch_retarget_known_truth.rs"]
mod batch_retarget_known_truth;
#[path = "bayesian.rs"]
mod bayesian;
#[path = "bayesian_design_posterior_routes.rs"]
mod bayesian_design_posterior_routes;
#[path = "bayesian_functional_routes.rs"]
mod bayesian_functional_routes;
#[path = "bayesian_design_routes.rs"]
mod bayesian_design_routes;
#[path = "bayesian_iv_rd_staged.rs"]
mod bayesian_iv_rd_staged;
#[path = "bayesian_likelihood.rs"]
mod bayesian_likelihood;
#[path = "bayesian_prior_in_force.rs"]
mod bayesian_prior_in_force;
#[path = "bayesian_randomized_routes.rs"]
mod bayesian_randomized_routes;
#[path = "bayesian_robust_ate.rs"]
mod bayesian_robust_ate;
#[path = "bayesian_trial_transport.rs"]
mod bayesian_trial_transport;
#[path = "calibration_binding.rs"]
mod calibration_binding;
#[path = "calibration_grid_binding.rs"]
mod calibration_grid_binding;
#[path = "callback_nested_overlap_facade.rs"]
mod callback_nested_overlap_facade;
#[path = "cancellation_expansion_cells.rs"]
mod cancellation_expansion_cells;
#[path = "categorical_treatment_posterior_facade.rs"]
mod categorical_treatment_posterior_facade;
