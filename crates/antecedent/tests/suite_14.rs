//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "randomized_design_missing_policies.rs"]
mod randomized_design_missing_policies;
#[path = "randomized_quantile_artifact.rs"]
mod randomized_quantile_artifact;
#[path = "randomized_quantile_known_truth.rs"]
mod randomized_quantile_known_truth;
#[path = "recalc_attempt_exact_effect.rs"]
mod recalc_attempt_exact_effect;
#[path = "recalc_cell_exact_effect.rs"]
mod recalc_cell_exact_effect;
#[path = "recalc_exact_effect.rs"]
mod recalc_exact_effect;
#[path = "response_curve_posterior_validation_truth.rs"]
mod response_curve_posterior_validation_truth;
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
#[path = "smoothed_dose_reference.rs"]
mod smoothed_dose_reference;
#[path = "source_evidence_exact.rs"]
mod source_evidence_exact;
#[path = "staged_derivatives.rs"]
mod staged_derivatives;
#[path = "staged_static_kinds.rs"]
mod staged_static_kinds;
#[path = "static_route_input_boundaries.rs"]
mod static_route_input_boundaries;
#[path = "static_route_table_weights.rs"]
mod static_route_table_weights;
#[path = "structure_prior_routes.rs"]
mod structure_prior_routes;
#[path = "structure_uncertainty_routes.rs"]
mod structure_uncertainty_routes;
#[path = "study_ranking_verified_unlocks.rs"]
mod study_ranking_verified_unlocks;
#[path = "target_constructed_refuters.rs"]
mod target_constructed_refuters;
#[path = "target_native_validation.rs"]
mod target_native_validation;
#[path = "target_refuter_extensions.rs"]
mod target_refuter_extensions;
#[path = "temporal_admg_derivative.rs"]
mod temporal_admg_derivative;
