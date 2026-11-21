//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "cohort_linear_effect.rs"]
mod cohort_linear_effect;
#[path = "complier_posterior_reference.rs"]
mod complier_posterior_reference;
#[path = "conditional_reference.rs"]
mod conditional_reference;
#[path = "continuous_measurement_error.rs"]
mod continuous_measurement_error;
#[path = "count_fitted_posterior_reference.rs"]
mod count_fitted_posterior_reference;
#[path = "cpdag_scenarios.rs"]
mod cpdag_scenarios;
#[path = "cyclic_general_systems_reference.rs"]
mod cyclic_general_systems_reference;
#[path = "dependence_aipw_hac.rs"]
mod dependence_aipw_hac;
#[path = "dependence_aware_interval_reference.rs"]
mod dependence_aware_interval_reference;
#[path = "discrete_treatment_contrast.rs"]
mod discrete_treatment_contrast;
#[path = "dose_grid_declared_reference.rs"]
mod dose_grid_declared_reference;
#[path = "dose_grid_functional.rs"]
mod dose_grid_functional;
#[path = "dose_grid_functional_bayesian_reference.rs"]
mod dose_grid_functional_bayesian_reference;
#[path = "dose_grid_local_quadratic_reference.rs"]
mod dose_grid_local_quadratic_reference;
#[path = "dose_posterior_reference.rs"]
mod dose_posterior_reference;
#[path = "effect_constancy_calibration.rs"]
mod effect_constancy_calibration;
#[path = "effect_constancy_reference.rs"]
mod effect_constancy_reference;
#[path = "envelope_reference.rs"]
mod envelope_reference;
#[path = "estimate_policy_continuous_dose_reference.rs"]
mod estimate_policy_continuous_dose_reference;
#[path = "estimate_response_bayesian_reference.rs"]
mod estimate_response_bayesian_reference;
#[path = "estimate_response_gam_derivative_reference.rs"]
mod estimate_response_gam_derivative_reference;
#[path = "estimate_response_intervention_gcomp_reference.rs"]
mod estimate_response_intervention_gcomp_reference;
#[path = "estimate_response_kennedy_dr_reference.rs"]
mod estimate_response_kennedy_dr_reference;
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
