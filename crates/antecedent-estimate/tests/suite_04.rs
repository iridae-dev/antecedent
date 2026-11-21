//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "matching_ties.rs"]
mod matching_ties;
#[path = "measurement_exposure_reference.rs"]
mod measurement_exposure_reference;
#[path = "measurement_models.rs"]
mod measurement_models;
#[path = "mechanism_discrepancy_reference.rs"]
mod mechanism_discrepancy_reference;
#[path = "mechanism_ridge_exact.rs"]
mod mechanism_ridge_exact;
#[path = "mediation_posterior_exact.rs"]
mod mediation_posterior_exact;
#[path = "multi_source_posterior_exact.rs"]
mod multi_source_posterior_exact;
#[path = "multiple_imputation.rs"]
mod multiple_imputation;
#[path = "multiple_imputation_dynamics.rs"]
mod multiple_imputation_dynamics;
#[path = "multiple_imputation_vector.rs"]
mod multiple_imputation_vector;
#[path = "mz_transport_execution.rs"]
mod mz_transport_execution;
#[path = "mz_transport_interval_construction.rs"]
mod mz_transport_interval_construction;
#[path = "nested_markov_binary.rs"]
mod nested_markov_binary;
#[path = "nested_markov_binary_reference.rs"]
mod nested_markov_binary_reference;
#[path = "nested_markov_fisher_reference.rs"]
mod nested_markov_fisher_reference;
#[path = "nested_markov_uncertainty.rs"]
mod nested_markov_uncertainty;
#[path = "neyman_randomization_enumeration.rs"]
mod neyman_randomization_enumeration;
#[path = "nonlinear_mediation.rs"]
mod nonlinear_mediation;
#[path = "nonlinear_mediation_reference.rs"]
mod nonlinear_mediation_reference;
#[path = "observation_recovery_execution.rs"]
mod observation_recovery_execution;
#[path = "ordered_response_recovery_chain_truth.rs"]
mod ordered_response_recovery_chain_truth;
#[path = "overlap_reference.rs"]
mod overlap_reference;
#[path = "penalized_aipw.rs"]
mod penalized_aipw;
#[path = "policy_value_posterior_reference.rs"]
mod policy_value_posterior_reference;
#[path = "policy_value_reference.rs"]
mod policy_value_reference;
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
