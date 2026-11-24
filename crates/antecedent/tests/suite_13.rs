//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "compiled_twins_counterfactual_seal.rs"]
mod compiled_twins_counterfactual_seal;
#[path = "compiled_twins_tradeoff_decisions.rs"]
mod compiled_twins_tradeoff_decisions;
#[path = "target_native_validation.rs"]
mod target_native_validation;
#[path = "target_refuter_extensions.rs"]
mod target_refuter_extensions;
#[path = "temporal_admg_derivative.rs"]
mod temporal_admg_derivative;
#[path = "temporal_admg_posterior.rs"]
mod temporal_admg_posterior;
#[path = "temporal_attribution_block_bootstrap.rs"]
mod temporal_attribution_block_bootstrap;
#[path = "temporal_counterfactual_posterior.rs"]
mod temporal_counterfactual_posterior;
#[path = "temporal_lagged_instrument_route.rs"]
mod temporal_lagged_instrument_route;
#[path = "temporal_point_uncertainty.rs"]
mod temporal_point_uncertainty;
#[path = "temporal_scenario_boundaries.rs"]
mod temporal_scenario_boundaries;
#[path = "temporal_treatment_score_product.rs"]
mod temporal_treatment_score_product;
#[path = "tiered_bayesian.rs"]
mod tiered_bayesian;
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
#[path = "white_innovation_posterior_construction.rs"]
mod white_innovation_posterior_construction;
#[path = "white_innovation_route.rs"]
mod white_innovation_route;
