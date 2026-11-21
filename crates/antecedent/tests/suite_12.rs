//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "cross_world_temporal_truth.rs"]
mod cross_world_temporal_truth;
#[path = "cyclic_dag_regression.rs"]
mod cyclic_dag_regression;
#[path = "cyclic_independent_consumer.rs"]
mod cyclic_independent_consumer;
#[path = "dag_response_validation_known_truth.rs"]
mod dag_response_validation_known_truth;
#[path = "data_diagnostics.rs"]
mod data_diagnostics;
#[path = "dependence_aware_estimators.rs"]
mod dependence_aware_estimators;
#[path = "derivative_refuter_known_truth.rs"]
mod derivative_refuter_known_truth;
#[path = "derivative_validation_on_classes.rs"]
mod derivative_validation_on_classes;
#[path = "descriptive_fallback.rs"]
mod descriptive_fallback;
#[path = "dirichlet_update_exact.rs"]
mod dirichlet_update_exact;
#[path = "distribution_validation_route_evidence.rs"]
mod distribution_validation_route_evidence;
#[path = "estimate_response_kennedy_dr_declarations.rs"]
mod estimate_response_kennedy_dr_declarations;
#[path = "estimate_uncertainty_reference.rs"]
mod estimate_uncertainty_reference;
#[path = "estimator_validity_snapshot.rs"]
mod estimator_validity_snapshot;
#[path = "evidence_plan_study.rs"]
mod evidence_plan_study;
#[path = "experiment_planning.rs"]
mod experiment_planning;
#[path = "facade_route_replay.rs"]
mod facade_route_replay;
#[path = "fitted_model_support_routes.rs"]
mod fitted_model_support_routes;
#[path = "flagship_shared_parameter.rs"]
mod flagship_shared_parameter;
#[path = "functional_effect_average_dag_known_truth.rs"]
mod functional_effect_average_dag_known_truth;
#[path = "functional_graph_posterior_mixture_truth.rs"]
mod functional_graph_posterior_mixture_truth;
#[path = "gcm_bayesian_refit_reference.rs"]
mod gcm_bayesian_refit_reference;
#[path = "graph_imputation_known_truth.rs"]
mod graph_imputation_known_truth;
#[path = "graph_posterior_attribution_routes.rs"]
mod graph_posterior_attribution_routes;
#[path = "graph_posterior_cross_world_routes.rs"]
mod graph_posterior_cross_world_routes;
#[path = "graph_posterior_derivative_known_truth.rs"]
mod graph_posterior_derivative_known_truth;
#[path = "graph_posterior_exposure_routes.rs"]
mod graph_posterior_exposure_routes;
#[path = "graph_posterior_temporal_pag_mediation.rs"]
mod graph_posterior_temporal_pag_mediation;
#[path = "graph_posterior_weighted_target.rs"]
mod graph_posterior_weighted_target;
#[path = "graphless_did_known_truth.rs"]
mod graphless_did_known_truth;
