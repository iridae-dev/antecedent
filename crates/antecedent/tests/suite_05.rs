//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "inverse_forward_sources_facade.rs"]
mod inverse_forward_sources_facade;
#[path = "joint_class_response.rs"]
mod joint_class_response;
#[path = "joint_law_propagation_facade.rs"]
mod joint_law_propagation_facade;
#[path = "joint_sensitivity_lifecycle.rs"]
mod joint_sensitivity_lifecycle;
#[path = "learned_continuous_lifecycle.rs"]
mod learned_continuous_lifecycle;
#[path = "local_polynomial_ratio_route.rs"]
mod local_polynomial_ratio_route;
#[path = "longitudinal_regime_route.rs"]
mod longitudinal_regime_route;
#[path = "mag_visible_numeric.rs"]
mod mag_visible_numeric;
#[path = "manufacturing_temporal.rs"]
mod manufacturing_temporal;
#[path = "mapped_posterior_transfer.rs"]
mod mapped_posterior_transfer;
#[path = "measurement_model_facade.rs"]
mod measurement_model_facade;
#[path = "mechanism_composition_facade.rs"]
mod mechanism_composition_facade;
#[path = "mnar_sensitivity_facade.rs"]
mod mnar_sensitivity_facade;
#[path = "msm_sensitivity_decision.rs"]
mod msm_sensitivity_decision;
#[path = "multi_source_acceptance.rs"]
mod multi_source_acceptance;
#[path = "mz_transport_lifecycle.rs"]
mod mz_transport_lifecycle;
#[path = "nan_cell_ingest.rs"]
mod nan_cell_ingest;
#[path = "native_claim_support_and_horizon_draws.rs"]
mod native_claim_support_and_horizon_draws;
#[path = "native_claims.rs"]
mod native_claims;
#[path = "native_coordinate_parity.rs"]
mod native_coordinate_parity;
#[path = "nested_counterfactual_route_evidence.rs"]
mod nested_counterfactual_route_evidence;
#[path = "numeric_pins.rs"]
mod numeric_pins;
#[path = "observation_query_facade.rs"]
mod observation_query_facade;
#[path = "observation_recovery_lifecycle.rs"]
mod observation_recovery_lifecycle;
#[path = "observational_interference_route.rs"]
mod observational_interference_route;
#[path = "pag.rs"]
mod pag;
#[path = "pag_admg_numeric_pins.rs"]
mod pag_admg_numeric_pins;
#[path = "pag_identified_envelope_numeric_pins.rs"]
mod pag_identified_envelope_numeric_pins;
#[path = "panel_did_interval_calibration.rs"]
mod panel_did_interval_calibration;
#[path = "panel_did_route.rs"]
mod panel_did_route;
#[path = "path_specific_edge_gformula_pins.rs"]
mod path_specific_edge_gformula_pins;
