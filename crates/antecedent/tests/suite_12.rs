//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "mapped_posterior_transfer_reference.rs"]
mod mapped_posterior_transfer_reference;
#[path = "mechanism_refit_bootstrap_construction.rs"]
mod mechanism_refit_bootstrap_construction;
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
#[path = "panel_link_contrast_route.rs"]
mod panel_link_contrast_route;
#[path = "point_only_routes.rs"]
mod point_only_routes;
#[path = "prepared_retained_refresh.rs"]
mod prepared_retained_refresh;
#[path = "randomized_design_boundaries.rs"]
mod randomized_design_boundaries;
#[path = "randomized_design_missing_policies.rs"]
mod randomized_design_missing_policies;
#[path = "randomized_quantile_artifact.rs"]
mod randomized_quantile_artifact;
#[path = "randomized_quantile_known_truth.rs"]
mod randomized_quantile_known_truth;
#[path = "sample_split_analysis_reference.rs"]
mod sample_split_analysis_reference;
#[path = "sampling_weight_routes.rs"]
mod sampling_weight_routes;
#[path = "scenario_class_route_boundaries.rs"]
mod scenario_class_route_boundaries;
#[path = "scenario_dependence_declarations.rs"]
mod scenario_dependence_declarations;
#[path = "scenario_equivalence_class_estimators.rs"]
mod scenario_equivalence_class_estimators;
#[path = "scenario_invalid_input_refusals.rs"]
mod scenario_invalid_input_refusals;
#[path = "scenario_iv_route_boundaries.rs"]
mod scenario_iv_route_boundaries;
#[path = "scenario_route_declarations.rs"]
mod scenario_route_declarations;
#[path = "scenario_treatment_levels.rs"]
mod scenario_treatment_levels;
#[path = "scientific_bridge_interval_reference.rs"]
mod scientific_bridge_interval_reference;
#[path = "static_route_input_boundaries.rs"]
mod static_route_input_boundaries;
#[path = "static_route_table_weights.rs"]
mod static_route_table_weights;
#[path = "structure_uncertainty_routes.rs"]
mod structure_uncertainty_routes;
#[path = "study_ranking_verified_unlocks.rs"]
mod study_ranking_verified_unlocks;
#[path = "target_constructed_refuters.rs"]
mod target_constructed_refuters;
