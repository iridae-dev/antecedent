//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "gam_derivative_band_construction.rs"]
mod gam_derivative_band_construction;
#[path = "glm_adjustment_reference.rs"]
mod glm_adjustment_reference;
#[path = "graphless_interference_saturation_known_truth.rs"]
mod graphless_interference_saturation_known_truth;
#[path = "graphless_local_polynomial_ratio_known_truth.rs"]
mod graphless_local_polynomial_ratio_known_truth;
#[path = "graphless_published_intervals_known_truth.rs"]
mod graphless_published_intervals_known_truth;
#[path = "graphless_randomized_cuped_known_truth.rs"]
mod graphless_randomized_cuped_known_truth;
#[path = "graphless_survival_bootstrap_known_truth.rs"]
mod graphless_survival_bootstrap_known_truth;
#[path = "history_match_variance_reference.rs"]
mod history_match_variance_reference;
#[path = "history_outcome_model_reference.rs"]
mod history_outcome_model_reference;
#[path = "influence_se_reference.rs"]
mod influence_se_reference;
#[path = "interference_observational_calibration.rs"]
mod interference_observational_calibration;
#[path = "interference_saturation_calibration.rs"]
mod interference_saturation_calibration;
#[path = "intervention_support.rs"]
mod intervention_support;
#[path = "iv_bounds_interval_reference.rs"]
mod iv_bounds_interval_reference;
#[path = "joint_bayesian_transport.rs"]
mod joint_bayesian_transport;
#[path = "joint_bayesian_transport_reference.rs"]
mod joint_bayesian_transport_reference;
#[path = "joint_design_reference.rs"]
mod joint_design_reference;
#[path = "joint_if_reference.rs"]
mod joint_if_reference;
#[path = "lagged_instrument_ratio_reference.rs"]
mod lagged_instrument_ratio_reference;
#[path = "lagged_instrument_wald_reference.rs"]
mod lagged_instrument_wald_reference;
#[path = "latent_class_effects.rs"]
mod latent_class_effects;
#[path = "latent_class_posterior_exact.rs"]
mod latent_class_posterior_exact;
#[path = "latent_class_reference.rs"]
mod latent_class_reference;
#[path = "learned_joint_transport.rs"]
mod learned_joint_transport;
#[path = "learned_joint_transport_reference.rs"]
mod learned_joint_transport_reference;
#[path = "linear_adjustment_reference.rs"]
mod linear_adjustment_reference;
#[path = "local_polynomial_ratio_calibration.rs"]
mod local_polynomial_ratio_calibration;
#[path = "local_ratio_row_weights.rs"]
mod local_ratio_row_weights;
#[path = "longitudinal_g_formula_calibration.rs"]
mod longitudinal_g_formula_calibration;
#[path = "marginal_structural_model_reference.rs"]
mod marginal_structural_model_reference;
