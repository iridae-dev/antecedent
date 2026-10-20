//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "period_refresh_contract.rs"]
mod period_refresh_contract;
#[path = "period_refresh_lifecycle.rs"]
mod period_refresh_lifecycle;
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
#[path = "temporal_response_redundancy.rs"]
mod temporal_response_redundancy;
#[path = "temporal_soft_mean_mechanisms.rs"]
mod temporal_soft_mean_mechanisms;
#[path = "tier_diagnostics.rs"]
mod tier_diagnostics;
#[path = "tiered_known_truth.rs"]
mod tiered_known_truth;
#[path = "transport_grid_missing_evidence.rs"]
mod transport_grid_missing_evidence;
#[path = "transport_revalidation_facade.rs"]
mod transport_revalidation_facade;
#[path = "transport_scenarios.rs"]
mod transport_scenarios;
#[path = "transport_scenarios_conditional.rs"]
mod transport_scenarios_conditional;
#[path = "transport_sensitivity_facade.rs"]
mod transport_sensitivity_facade;
#[path = "transport_standing_facade.rs"]
mod transport_standing_facade;
#[path = "transported_cohort_facade.rs"]
mod transported_cohort_facade;
#[path = "transported_counterfactual_facade.rs"]
mod transported_counterfactual_facade;
#[path = "transported_measurement_facade.rs"]
mod transported_measurement_facade;
#[path = "transported_mechanism_facade.rs"]
mod transported_mechanism_facade;
#[path = "whole_path_account_facade.rs"]
mod whole_path_account_facade;
#[path = "workspace_invariants.rs"]
mod workspace_invariants;
