//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "horizon_composition.rs"]
mod horizon_composition;
#[path = "horizon_one_static.rs"]
mod horizon_one_static;
#[path = "host_contract.rs"]
mod host_contract;
#[path = "intervention_response_class_validation_truth.rs"]
mod intervention_response_class_validation_truth;
#[path = "interventional_distribution_class_route_evidence.rs"]
mod interventional_distribution_class_route_evidence;
#[path = "joint_horizon_block_weight_construction.rs"]
mod joint_horizon_block_weight_construction;
#[path = "joint_sensitivity_source_exact.rs"]
mod joint_sensitivity_source_exact;
#[path = "lag_zero_average_effect_truth.rs"]
mod lag_zero_average_effect_truth;
#[path = "mapped_posterior_transfer_reference.rs"]
mod mapped_posterior_transfer_reference;
#[path = "measured_exposure_known_truth.rs"]
mod measured_exposure_known_truth;
#[path = "mechanism_refit_bootstrap_construction.rs"]
mod mechanism_refit_bootstrap_construction;
#[path = "mediation_mixed_graph_route_evidence.rs"]
mod mediation_mixed_graph_route_evidence;
#[path = "missing_data_routes.rs"]
mod missing_data_routes;
#[path = "missing_imputation_aipw_weighting.rs"]
mod missing_imputation_aipw_weighting;
#[path = "missing_imputation_estimator_families.rs"]
mod missing_imputation_estimator_families;
#[path = "missing_imputation_routes.rs"]
mod missing_imputation_routes;
#[path = "missing_policy_support.rs"]
mod missing_policy_support;
#[path = "missing_policy_unmodelled_routes.rs"]
mod missing_policy_unmodelled_routes;
#[path = "native_response_claims_exact.rs"]
mod native_response_claims_exact;
#[path = "observational_survival_policy_routes.rs"]
mod observational_survival_policy_routes;
#[path = "panel_link_contrast_route.rs"]
mod panel_link_contrast_route;
#[path = "path_specific_mixed_graph_route_evidence.rs"]
mod path_specific_mixed_graph_route_evidence;
#[path = "period_refresh_reference.rs"]
mod period_refresh_reference;
#[path = "point_only_routes.rs"]
mod point_only_routes;
#[path = "pooled_structure_and_prior_selection.rs"]
mod pooled_structure_and_prior_selection;
#[path = "prepared_adjusted_retarget.rs"]
mod prepared_adjusted_retarget;
#[path = "prepared_retained_refresh.rs"]
mod prepared_retained_refresh;
#[path = "proposal_arrival_exact.rs"]
mod proposal_arrival_exact;
#[path = "quasi_rollout_routes.rs"]
mod quasi_rollout_routes;
#[path = "randomized_design_boundaries.rs"]
mod randomized_design_boundaries;
