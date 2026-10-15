//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "staged_static_kinds.rs"]
mod staged_static_kinds;
#[path = "static_class_bounds.rs"]
mod static_class_bounds;
#[path = "statistical_transport_numeric.rs"]
mod statistical_transport_numeric;
#[path = "survival_route.rs"]
mod survival_route;
#[path = "switchback_interval_route.rs"]
mod switchback_interval_route;
#[path = "synthetic_control_route.rs"]
mod synthetic_control_route;
#[path = "synthetic_did_randomization_route.rs"]
mod synthetic_did_randomization_route;
#[path = "temporal_admg_route_evidence.rs"]
mod temporal_admg_route_evidence;
#[path = "temporal_attribution_route_evidence.rs"]
mod temporal_attribution_route_evidence;
#[path = "temporal_class_bayesian_envelope.rs"]
mod temporal_class_bayesian_envelope;
#[path = "temporal_class_envelope_numeric_pins.rs"]
mod temporal_class_envelope_numeric_pins;
#[path = "temporal_class_observation.rs"]
mod temporal_class_observation;
#[path = "temporal_class_point_queries.rs"]
mod temporal_class_point_queries;
#[path = "temporal_class_posterior.rs"]
mod temporal_class_posterior;
#[path = "temporal_class_sequence_policies.rs"]
mod temporal_class_sequence_policies;
#[path = "temporal_conditional_effect.rs"]
mod temporal_conditional_effect;
#[path = "temporal_cross_world_effect.rs"]
mod temporal_cross_world_effect;
#[path = "temporal_derivative_effect.rs"]
mod temporal_derivative_effect;
#[path = "temporal_incomplete_structure_parity.rs"]
mod temporal_incomplete_structure_parity;
#[path = "temporal_mag_numeric.rs"]
mod temporal_mag_numeric;
#[path = "temporal_mediation_horizons.rs"]
mod temporal_mediation_horizons;
#[path = "temporal_observation.rs"]
mod temporal_observation;
#[path = "temporal_observed_bayesian.rs"]
mod temporal_observed_bayesian;
#[path = "temporal_parent_adjustment.rs"]
mod temporal_parent_adjustment;
#[path = "temporal_prior_transfer.rs"]
mod temporal_prior_transfer;
#[path = "temporal_response_dose_identification.rs"]
mod temporal_response_dose_identification;
#[path = "temporal_response_facade.rs"]
mod temporal_response_facade;
#[path = "temporal_response_redundancy.rs"]
mod temporal_response_redundancy;
#[path = "temporal_soft_mean_mechanisms.rs"]
mod temporal_soft_mean_mechanisms;
#[path = "tier_diagnostics.rs"]
mod tier_diagnostics;
