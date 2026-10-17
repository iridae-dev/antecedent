//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "refute_validator_not_applicable.rs"]
mod refute_validator_not_applicable;
#[path = "rollout_artifact.rs"]
mod rollout_artifact;
#[path = "route_evidence_joint_law_propagation.rs"]
mod route_evidence_joint_law_propagation;
#[path = "route_evidence_observation_query.rs"]
mod route_evidence_observation_query;
#[path = "sample_split_analysis_facade.rs"]
mod sample_split_analysis_facade;
#[path = "sample_split_effect_facade.rs"]
mod sample_split_effect_facade;
#[path = "sample_split_parent_set_effect_facade.rs"]
mod sample_split_parent_set_effect_facade;
#[path = "sample_split_weighted_acceptance.rs"]
mod sample_split_weighted_acceptance;
#[path = "saturation_interference_route.rs"]
mod saturation_interference_route;
#[path = "schema_binding.rs"]
mod schema_binding;
#[path = "scientific_bridge_facade.rs"]
mod scientific_bridge_facade;
#[path = "scientific_bridge_interval_facade.rs"]
mod scientific_bridge_interval_facade;
#[path = "selective_discovery_facade.rs"]
mod selective_discovery_facade;
#[path = "sequential_evaluation_facade.rs"]
mod sequential_evaluation_facade;
#[path = "short_series_measurement.rs"]
mod short_series_measurement;
#[path = "smoothed_dose_lifecycle.rs"]
mod smoothed_dose_lifecycle;
#[path = "source_evidence.rs"]
mod source_evidence;
#[path = "source_projection.rs"]
mod source_projection;
#[path = "specification_conformance.rs"]
mod specification_conformance;
#[path = "staged_attribution_transport_interference.rs"]
mod staged_attribution_transport_interference;
#[path = "staged_bayesian_basis_gcomp.rs"]
mod staged_bayesian_basis_gcomp;
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
