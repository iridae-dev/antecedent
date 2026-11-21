//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "adaptive_decision_stopping_posterior_reference.rs"]
mod adaptive_decision_stopping_posterior_reference;
#[path = "adaptive_repeated_measures_weighting_reference.rs"]
mod adaptive_repeated_measures_weighting_reference;
#[path = "adjusted_contrast_reference.rs"]
mod adjusted_contrast_reference;
#[path = "bayesian_conditional_reference.rs"]
mod bayesian_conditional_reference;
#[path = "bayesian_event_study_reference.rs"]
mod bayesian_event_study_reference;
#[path = "bayesian_gaussian_gcomp_reference.rs"]
mod bayesian_gaussian_gcomp_reference;
#[path = "bayesian_glm_gcomp_posterior.rs"]
mod bayesian_glm_gcomp_posterior;
#[path = "bayesian_interference_reference.rs"]
mod bayesian_interference_reference;
#[path = "bayesian_iv_reference.rs"]
mod bayesian_iv_reference;
#[path = "bayesian_mediation_reference.rs"]
mod bayesian_mediation_reference;
#[path = "bayesian_rd_reference.rs"]
mod bayesian_rd_reference;
#[path = "bayesian_robust_ate_calibration.rs"]
mod bayesian_robust_ate_calibration;
#[path = "categorical_declared_reference.rs"]
mod categorical_declared_reference;
#[path = "categorical_regression_reference.rs"]
mod categorical_regression_reference;
#[path = "cells_01_staggered_serial_reference.rs"]
mod cells_01_staggered_serial_reference;
#[path = "cluster_dml_known_truth.rs"]
mod cluster_dml_known_truth;
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
#[path = "discrete_treatment_contrast.rs"]
mod discrete_treatment_contrast;
#[path = "dose_grid_declared_reference.rs"]
mod dose_grid_declared_reference;
#[path = "dose_grid_local_quadratic_reference.rs"]
mod dose_grid_local_quadratic_reference;
#[path = "effect_constancy_reference.rs"]
mod effect_constancy_reference;
#[path = "envelope_reference.rs"]
mod envelope_reference;
#[path = "estimate_response_bayesian_reference.rs"]
mod estimate_response_bayesian_reference;
#[path = "estimate_response_intervention_gcomp_reference.rs"]
mod estimate_response_intervention_gcomp_reference;
#[path = "estimate_response_kennedy_dr_reference.rs"]
mod estimate_response_kennedy_dr_reference;
