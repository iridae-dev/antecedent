//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "multiple_imputation_vector.rs"]
mod multiple_imputation_vector;
#[path = "mz_transport_interval_construction.rs"]
mod mz_transport_interval_construction;
#[path = "nested_markov_binary.rs"]
mod nested_markov_binary;
#[path = "nested_markov_fisher_reference.rs"]
mod nested_markov_fisher_reference;
#[path = "nonlinear_mediation_reference.rs"]
mod nonlinear_mediation_reference;
#[path = "overlap_reference.rs"]
mod overlap_reference;
#[path = "policy_value_posterior_reference.rs"]
mod policy_value_posterior_reference;
#[path = "population_aipw.rs"]
mod population_aipw;
#[path = "posterior_mean_linear.rs"]
mod posterior_mean_linear;
#[path = "posterior_mean_reference.rs"]
mod posterior_mean_reference;
#[path = "product_limit_reference.rs"]
mod product_limit_reference;
#[path = "propensity_posterior_reference.rs"]
mod propensity_posterior_reference;
#[path = "quadratic_dose_reproduction.rs"]
mod quadratic_dose_reproduction;
#[path = "quantile_inversion_reference.rs"]
mod quantile_inversion_reference;
#[path = "quantile_posterior_reference.rs"]
mod quantile_posterior_reference;
#[path = "randomized_design_enumeration.rs"]
mod randomized_design_enumeration;
#[path = "recovery_chain_observation_reference.rs"]
mod recovery_chain_observation_reference;
#[path = "refusal_fields_existing_sites.rs"]
mod refusal_fields_existing_sites;
#[path = "responder_weighted_reference.rs"]
mod responder_weighted_reference;
#[path = "sampled_count_validation.rs"]
mod sampled_count_validation;
#[path = "scenario_invariance_report.rs"]
mod scenario_invariance_report;
#[path = "sequential_aipw_noisy_reference.rs"]
mod sequential_aipw_noisy_reference;
#[path = "smoothed_dose_declared_reference.rs"]
mod smoothed_dose_declared_reference;
#[path = "staggered_did_reference.rs"]
mod staggered_did_reference;
#[path = "switchback_calibration.rs"]
mod switchback_calibration;
#[path = "temporal_conditional_effect_reference.rs"]
mod temporal_conditional_effect_reference;
#[path = "temporal_linear_adjustment_reference.rs"]
mod temporal_linear_adjustment_reference;
#[path = "temporal_linear_predictor_reference.rs"]
mod temporal_linear_predictor_reference;
#[path = "temporal_link_bayesian_reference.rs"]
mod temporal_link_bayesian_reference;
#[path = "temporal_link_reference.rs"]
mod temporal_link_reference;
