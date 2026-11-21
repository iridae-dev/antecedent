//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "temporal_dependent_interval.rs"]
mod temporal_dependent_interval;
#[path = "temporal_design_dose_quantile_known_truth.rs"]
mod temporal_design_dose_quantile_known_truth;
#[path = "temporal_design_excursion_bayesian_bootstrap_reference.rs"]
mod temporal_design_excursion_bayesian_bootstrap_reference;
#[path = "temporal_design_excursion_reference.rs"]
mod temporal_design_excursion_reference;
#[path = "temporal_initial_state.rs"]
mod temporal_initial_state;
#[path = "temporal_initial_state_bayesian_exact.rs"]
mod temporal_initial_state_bayesian_exact;
#[path = "temporal_linear_adjustment_reference.rs"]
mod temporal_linear_adjustment_reference;
#[path = "temporal_linear_predictor_reference.rs"]
mod temporal_linear_predictor_reference;
#[path = "temporal_link_bayesian_reference.rs"]
mod temporal_link_bayesian_reference;
#[path = "temporal_link_reference.rs"]
mod temporal_link_reference;
#[path = "temporal_mediation_reference.rs"]
mod temporal_mediation_reference;
#[path = "temporal_observed_bayes_reference.rs"]
mod temporal_observed_bayes_reference;
#[path = "temporal_propensity_reference.rs"]
mod temporal_propensity_reference;
#[path = "temporal_refresh.rs"]
mod temporal_refresh;
#[path = "temporal_regime_survival_reference.rs"]
mod temporal_regime_survival_reference;
#[path = "temporal_response_gcomp_reference.rs"]
mod temporal_response_gcomp_reference;
#[path = "temporal_transport_execution.rs"]
mod temporal_transport_execution;
#[path = "transport_design_reference.rs"]
mod transport_design_reference;
#[path = "transport_response_grid_reference.rs"]
mod transport_response_grid_reference;
#[path = "trial_bayesian_bootstrap_reference.rs"]
mod trial_bayesian_bootstrap_reference;
#[path = "two_step_posterior_exact.rs"]
mod two_step_posterior_exact;
#[path = "vector_treatment_reference.rs"]
mod vector_treatment_reference;
#[path = "white_innovation_ljung_box_reference.rs"]
mod white_innovation_ljung_box_reference;
#[path = "z_transport_c_factor_exchange.rs"]
mod z_transport_c_factor_exchange;
#[path = "z_transport_cited_evidence.rs"]
mod z_transport_cited_evidence;
#[path = "z_transport_sequential_exchange.rs"]
mod z_transport_sequential_exchange;
