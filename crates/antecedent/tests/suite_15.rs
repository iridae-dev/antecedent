//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "temporal_admg_derivative_known_truth.rs"]
mod temporal_admg_derivative_known_truth;
#[path = "temporal_admg_posterior.rs"]
mod temporal_admg_posterior;
#[path = "temporal_attribution_block_bootstrap.rs"]
mod temporal_attribution_block_bootstrap;
#[path = "temporal_class_derivative_known_truth.rs"]
mod temporal_class_derivative_known_truth;
#[path = "temporal_class_posterior_known_truth.rs"]
mod temporal_class_posterior_known_truth;
#[path = "temporal_conditional_validation_truth.rs"]
mod temporal_conditional_validation_truth;
#[path = "temporal_counterfactual_posterior.rs"]
mod temporal_counterfactual_posterior;
#[path = "temporal_derivative_refuter_known_truth.rs"]
mod temporal_derivative_refuter_known_truth;
#[path = "temporal_graph_posterior_derivative_known_truth.rs"]
mod temporal_graph_posterior_derivative_known_truth;
#[path = "temporal_lagged_instrument_route.rs"]
mod temporal_lagged_instrument_route;
#[path = "temporal_pag_mediation.rs"]
mod temporal_pag_mediation;
#[path = "temporal_point_uncertainty.rs"]
mod temporal_point_uncertainty;
#[path = "temporal_response_facade.rs"]
mod temporal_response_facade;
#[path = "temporal_scenario_boundaries.rs"]
mod temporal_scenario_boundaries;
#[path = "temporal_sequential_gcomp_multistep_known_truth.rs"]
mod temporal_sequential_gcomp_multistep_known_truth;
#[path = "temporal_treatment_score_product.rs"]
mod temporal_treatment_score_product;
#[path = "temporal_window_route_evidence.rs"]
mod temporal_window_route_evidence;
#[path = "tier_diagnostics_exact.rs"]
mod tier_diagnostics_exact;
#[path = "tiered_average_effect_closed_form.rs"]
mod tiered_average_effect_closed_form;
#[path = "tiered_bayesian.rs"]
mod tiered_bayesian;
#[path = "transport_interference_cell_evidence.rs"]
mod transport_interference_cell_evidence;
#[path = "transport_weighted_target.rs"]
mod transport_weighted_target;
#[path = "unavailable_b_measurement_route_declarations.rs"]
mod unavailable_b_measurement_route_declarations;
#[path = "unavailable_b_static_temporal_refusals.rs"]
mod unavailable_b_static_temporal_refusals;
#[path = "unsupported_refuter_states_and_mechanism_bootstrap.rs"]
mod unsupported_refuter_states_and_mechanism_bootstrap;
#[path = "weighted_route_boundary.rs"]
mod weighted_route_boundary;
#[path = "weighted_target_effect.rs"]
mod weighted_target_effect;
#[path = "white_innovation_posterior_construction.rs"]
mod white_innovation_posterior_construction;
#[path = "white_innovation_route.rs"]
mod white_innovation_route;
