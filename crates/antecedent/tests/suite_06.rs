//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "per_population_discovery_facade.rs"]
mod per_population_discovery_facade;
#[path = "policy_python_parity.rs"]
mod policy_python_parity;
#[path = "policy_value_route.rs"]
mod policy_value_route;
#[path = "population_license.rs"]
mod population_license;
#[path = "portable_predictor_resume.rs"]
mod portable_predictor_resume;
#[path = "posterior_boundaries_facade.rs"]
mod posterior_boundaries_facade;
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
