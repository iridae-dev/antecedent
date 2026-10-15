//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

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
#[path = "transport_statistical_calibration.rs"]
mod transport_statistical_calibration;
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
#[path = "z_transport_repair.rs"]
mod z_transport_repair;
#[path = "z_transport_surrogate_calibration.rs"]
mod z_transport_surrogate_calibration;
