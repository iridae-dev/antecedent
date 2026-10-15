//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

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
#[path = "per_population_discovery_facade.rs"]
mod per_population_discovery_facade;
#[path = "policy_value_route.rs"]
mod policy_value_route;
#[path = "population_license.rs"]
mod population_license;
#[path = "portable_predictor_resume.rs"]
mod portable_predictor_resume;
#[path = "posterior_mean_decision.rs"]
mod posterior_mean_decision;
#[path = "practitioner_facade.rs"]
mod practitioner_facade;
#[path = "predictive_assessment_facade.rs"]
mod predictive_assessment_facade;
#[path = "preflight_reproducers.rs"]
mod preflight_reproducers;
#[path = "production_replicates.rs"]
mod production_replicates;
#[path = "projection_wide.rs"]
mod projection_wide;
#[path = "proposal_arrival.rs"]
mod proposal_arrival;
#[path = "provider_callbacks.rs"]
mod provider_callbacks;
#[path = "randomized_precision_intervals.rs"]
mod randomized_precision_intervals;
#[path = "randomized_trial_route.rs"]
mod randomized_trial_route;
#[path = "rank_drop_estimate.rs"]
mod rank_drop_estimate;
#[path = "recalc_adjusted.rs"]
mod recalc_adjusted;
#[path = "recalc_attempt.rs"]
mod recalc_attempt;
#[path = "recalc_bayesian.rs"]
mod recalc_bayesian;
#[path = "recalc_bayesian_standing.rs"]
mod recalc_bayesian_standing;
#[path = "recalc_cell.rs"]
mod recalc_cell;
#[path = "recalc_composite.rs"]
mod recalc_composite;
#[path = "recalc_design.rs"]
mod recalc_design;
