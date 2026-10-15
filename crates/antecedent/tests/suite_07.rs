//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "recalc_dr.rs"]
mod recalc_dr;
#[path = "recalc_external.rs"]
mod recalc_external;
#[path = "recalc_joint_bayesian_standing.rs"]
mod recalc_joint_bayesian_standing;
#[path = "recalc_receipt.rs"]
mod recalc_receipt;
#[path = "recalc_static.rs"]
mod recalc_static;
#[path = "recalc_temporal.rs"]
mod recalc_temporal;
#[path = "refusal_fields_existing_sites.rs"]
mod refusal_fields_existing_sites;
#[path = "refusal_remedy.rs"]
mod refusal_remedy;
#[path = "refute_validator_not_applicable.rs"]
mod refute_validator_not_applicable;
#[path = "response_facade.rs"]
mod response_facade;
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
#[path = "smoothed_dose_calibration.rs"]
mod smoothed_dose_calibration;
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
#[path = "staged_derivatives.rs"]
mod staged_derivatives;
