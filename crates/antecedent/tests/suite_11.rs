//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "claims_03_dbn_posterior_truth.rs"]
mod claims_03_dbn_posterior_truth;
#[path = "class_and_design_validation_suites.rs"]
mod class_and_design_validation_suites;
#[path = "class_attribution_known_truth.rs"]
mod class_attribution_known_truth;
#[path = "class_derivative_known_truth.rs"]
mod class_derivative_known_truth;
#[path = "class_derivative_response.rs"]
mod class_derivative_response;
#[path = "class_explicit_known_truth.rs"]
mod class_explicit_known_truth;
#[path = "class_graph_posterior_derivative_known_truth.rs"]
mod class_graph_posterior_derivative_known_truth;
#[path = "class_posterior_known_truth.rs"]
mod class_posterior_known_truth;
#[path = "class_route_artifacts.rs"]
mod class_route_artifacts;
#[path = "codetermined_bayesian_response_truth.rs"]
mod codetermined_bayesian_response_truth;
#[path = "compile_from_decision.rs"]
mod compile_from_decision;
#[path = "compiled_robust_decisions.rs"]
mod compiled_robust_decisions;
#[path = "composition_chain.rs"]
mod composition_chain;
#[path = "composition_posterior_decision_bundle.rs"]
mod composition_posterior_decision_bundle;
#[path = "conditional_effect_admg_bayesian_interval.rs"]
mod conditional_effect_admg_bayesian_interval;
#[path = "conditional_effect_admg_validation_truth.rs"]
mod conditional_effect_admg_validation_truth;
#[path = "conditional_ipcw.rs"]
mod conditional_ipcw;
#[path = "continuous_measurement_facade.rs"]
mod continuous_measurement_facade;
#[path = "counterfactual_accepted_known_truth.rs"]
mod counterfactual_accepted_known_truth;
#[path = "counterfactual_contrasts.rs"]
mod counterfactual_contrasts;
#[path = "counterfactual_noise_posterior.rs"]
mod counterfactual_noise_posterior;
#[path = "cross_surface_cyclic_feedback.rs"]
mod cross_surface_cyclic_feedback;
#[path = "cross_surface_prepared_journey.rs"]
mod cross_surface_prepared_journey;
#[path = "cross_surface_previous_release.rs"]
mod cross_surface_previous_release;
#[path = "cross_surface_refresh_withdrawal.rs"]
mod cross_surface_refresh_withdrawal;
#[path = "cross_surface_retention_refresh.rs"]
mod cross_surface_retention_refresh;
#[path = "cross_world_class_truth.rs"]
mod cross_world_class_truth;
#[path = "cross_world_counterfactual_posterior_truth.rs"]
mod cross_world_counterfactual_posterior_truth;
#[path = "cross_world_dag_validation_truth.rs"]
mod cross_world_dag_validation_truth;
#[path = "cross_world_posterior_truth.rs"]
mod cross_world_posterior_truth;
