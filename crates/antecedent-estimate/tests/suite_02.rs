//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "estimate_response_kennedy_dr_simultaneous_reference.rs"]
mod estimate_response_kennedy_dr_simultaneous_reference;
#[path = "estimate_response_observation_adjusted_reference.rs"]
mod estimate_response_observation_adjusted_reference;
#[path = "estimate_response_point_derivative_reference.rs"]
mod estimate_response_point_derivative_reference;
#[path = "estimate_response_riesz_ade_reference.rs"]
mod estimate_response_riesz_ade_reference;
#[path = "estimate_response_temporal_bayesian_reference.rs"]
mod estimate_response_temporal_bayesian_reference;
#[path = "estimator_degeneracy.rs"]
mod estimator_degeneracy;
#[path = "estimator_dependence_axes.rs"]
mod estimator_dependence_axes;
#[path = "gam_derivative_band_construction.rs"]
mod gam_derivative_band_construction;
#[path = "glm_adjustment_reference.rs"]
mod glm_adjustment_reference;
#[path = "history_match_variance_reference.rs"]
mod history_match_variance_reference;
#[path = "influence_se_reference.rs"]
mod influence_se_reference;
#[path = "interference_observational_calibration.rs"]
mod interference_observational_calibration;
#[path = "interference_saturation_calibration.rs"]
mod interference_saturation_calibration;
#[path = "intervention_support.rs"]
mod intervention_support;
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
#[path = "latent_class_reference.rs"]
mod latent_class_reference;
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
#[path = "matching_ties.rs"]
mod matching_ties;
#[path = "measurement_exposure_reference.rs"]
mod measurement_exposure_reference;
#[path = "mechanism_discrepancy_reference.rs"]
mod mechanism_discrepancy_reference;
#[path = "multiple_imputation.rs"]
mod multiple_imputation;
#[path = "multiple_imputation_dynamics.rs"]
mod multiple_imputation_dynamics;
