//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "penalized_aipw.rs"]
mod penalized_aipw;
#[path = "posterior_mean_linear.rs"]
mod posterior_mean_linear;
#[path = "recovery_sampled.rs"]
mod recovery_sampled;
#[path = "refusal_fields_existing_sites.rs"]
mod refusal_fields_existing_sites;
#[path = "scenario_covariance.rs"]
mod scenario_covariance;
#[path = "scenario_invariance_report.rs"]
mod scenario_invariance_report;
#[path = "switchback_calibration.rs"]
mod switchback_calibration;
#[path = "temporal_dependent_interval.rs"]
mod temporal_dependent_interval;
#[path = "temporal_initial_state.rs"]
mod temporal_initial_state;
#[path = "temporal_refresh.rs"]
mod temporal_refresh;
#[path = "temporal_transport_execution.rs"]
mod temporal_transport_execution;
#[path = "vector_treatment.rs"]
mod vector_treatment;
#[path = "z_transport_c_factor_exchange.rs"]
mod z_transport_c_factor_exchange;
#[path = "z_transport_cited_evidence.rs"]
mod z_transport_cited_evidence;
#[path = "z_transport_sequential_exchange.rs"]
mod z_transport_sequential_exchange;
