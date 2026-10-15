//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "graph_typed_inputs.rs"]
mod graph_typed_inputs;
#[path = "interface_contract_facade.rs"]
mod interface_contract_facade;
#[path = "interference_interval_route.rs"]
mod interference_interval_route;
#[path = "inverse_cohort_source_facade.rs"]
mod inverse_cohort_source_facade;
#[path = "inverse_composition_refusals_facade.rs"]
mod inverse_composition_refusals_facade;
#[path = "inverse_forward_sources_facade.rs"]
mod inverse_forward_sources_facade;
#[path = "joint_class_response.rs"]
mod joint_class_response;
#[path = "joint_law_propagation_facade.rs"]
mod joint_law_propagation_facade;
#[path = "joint_sensitivity_calibration.rs"]
mod joint_sensitivity_calibration;
#[path = "joint_sensitivity_lifecycle.rs"]
mod joint_sensitivity_lifecycle;
#[path = "learned_continuous_calibration.rs"]
mod learned_continuous_calibration;
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
#[path = "msm_sensitivity_decision.rs"]
mod msm_sensitivity_decision;
#[path = "multi_source_acceptance.rs"]
mod multi_source_acceptance;
#[path = "mz_transport_calibration.rs"]
mod mz_transport_calibration;
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
