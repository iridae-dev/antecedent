//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "quadratic_dose_reproduction.rs"]
mod quadratic_dose_reproduction;
#[path = "quantile_inversion_reference.rs"]
mod quantile_inversion_reference;
#[path = "quantile_posterior_reference.rs"]
mod quantile_posterior_reference;
#[path = "randomized_design_enumeration.rs"]
mod randomized_design_enumeration;
#[path = "randomized_posterior_reference.rs"]
mod randomized_posterior_reference;
#[path = "randomized_quantile_reference.rs"]
mod randomized_quantile_reference;
#[path = "randomized_scores_enumeration.rs"]
mod randomized_scores_enumeration;
#[path = "recovery_binary_posterior_exact.rs"]
mod recovery_binary_posterior_exact;
#[path = "recovery_chain_observation_reference.rs"]
mod recovery_chain_observation_reference;
#[path = "recovery_chain_posterior_exact.rs"]
mod recovery_chain_posterior_exact;
#[path = "recovery_sampled.rs"]
mod recovery_sampled;
#[path = "refusal_fields_existing_sites.rs"]
mod refusal_fields_existing_sites;
#[path = "responder_weighted_reference.rs"]
mod responder_weighted_reference;
#[path = "sampled_count_validation.rs"]
mod sampled_count_validation;
#[path = "sampling_weights.rs"]
mod sampling_weights;
#[path = "sampling_weights_reference.rs"]
mod sampling_weights_reference;
#[path = "scenario_covariance.rs"]
mod scenario_covariance;
#[path = "scenario_covariance_posterior_exact.rs"]
mod scenario_covariance_posterior_exact;
#[path = "scenario_invariance_report.rs"]
mod scenario_invariance_report;
#[path = "score_posterior_interval_known_truth.rs"]
mod score_posterior_interval_known_truth;
#[path = "sequential_aipw_noisy_reference.rs"]
mod sequential_aipw_noisy_reference;
#[path = "sequential_contrast_reference.rs"]
mod sequential_contrast_reference;
#[path = "shared_scenario_covariance_reference.rs"]
mod shared_scenario_covariance_reference;
#[path = "smoothed_dose_declared_reference.rs"]
mod smoothed_dose_declared_reference;
#[path = "staggered_did_reference.rs"]
mod staggered_did_reference;
#[path = "support_coordinate_ess_exact.rs"]
mod support_coordinate_ess_exact;
#[path = "survival_posterior_reference.rs"]
mod survival_posterior_reference;
#[path = "switchback_calibration.rs"]
mod switchback_calibration;
#[path = "switchback_posterior_reference.rs"]
mod switchback_posterior_reference;
#[path = "temporal_conditional_effect_reference.rs"]
mod temporal_conditional_effect_reference;
