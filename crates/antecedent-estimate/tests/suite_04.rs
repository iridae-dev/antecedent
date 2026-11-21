//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "temporal_mediation_reference.rs"]
mod temporal_mediation_reference;
#[path = "temporal_observed_bayes_reference.rs"]
mod temporal_observed_bayes_reference;
#[path = "temporal_propensity_reference.rs"]
mod temporal_propensity_reference;
#[path = "temporal_refresh.rs"]
mod temporal_refresh;
#[path = "temporal_response_gcomp_reference.rs"]
mod temporal_response_gcomp_reference;
#[path = "transport_design_reference.rs"]
mod transport_design_reference;
#[path = "transport_response_grid_reference.rs"]
mod transport_response_grid_reference;
#[path = "trial_bayesian_bootstrap_reference.rs"]
mod trial_bayesian_bootstrap_reference;
#[path = "vector_treatment_reference.rs"]
mod vector_treatment_reference;
#[path = "white_innovation_ljung_box_reference.rs"]
mod white_innovation_ljung_box_reference;
#[path = "z_transport_c_factor_exchange.rs"]
mod z_transport_c_factor_exchange;
#[path = "z_transport_sequential_exchange.rs"]
mod z_transport_sequential_exchange;
