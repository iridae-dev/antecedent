//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "continuous_measurement_facade.rs"]
mod continuous_measurement_facade;
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
#[path = "cyclic_dag_regression.rs"]
mod cyclic_dag_regression;
#[path = "cyclic_independent_consumer.rs"]
mod cyclic_independent_consumer;
#[path = "data_diagnostics.rs"]
mod data_diagnostics;
#[path = "dependence_aware_estimators.rs"]
mod dependence_aware_estimators;
#[path = "derivative_validation_on_classes.rs"]
mod derivative_validation_on_classes;
#[path = "descriptive_fallback.rs"]
mod descriptive_fallback;
#[path = "estimate_response_kennedy_dr_declarations.rs"]
mod estimate_response_kennedy_dr_declarations;
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
#[path = "gcm_bayesian_refit_reference.rs"]
mod gcm_bayesian_refit_reference;
#[path = "graph_posterior_attribution_routes.rs"]
mod graph_posterior_attribution_routes;
#[path = "graph_posterior_cross_world_routes.rs"]
mod graph_posterior_cross_world_routes;
#[path = "graph_posterior_exposure_routes.rs"]
mod graph_posterior_exposure_routes;
#[path = "graph_posterior_temporal_pag_mediation.rs"]
mod graph_posterior_temporal_pag_mediation;
#[path = "horizon_one_static.rs"]
mod horizon_one_static;
#[path = "host_contract.rs"]
mod host_contract;
#[path = "joint_horizon_block_weight_construction.rs"]
mod joint_horizon_block_weight_construction;
