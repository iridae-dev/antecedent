//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "adaptive_inference.rs"]
mod adaptive_inference;
#[path = "admg_conditional_transport_scm.rs"]
mod admg_conditional_transport_scm;
#[path = "bayesian_robust_ate_calibration.rs"]
mod bayesian_robust_ate_calibration;
#[path = "cancellation_expansion_cells.rs"]
mod cancellation_expansion_cells;
#[path = "categorical_treatment.rs"]
mod categorical_treatment;
#[path = "cluster_dml_aipw.rs"]
mod cluster_dml_aipw;
#[path = "cluster_dml_known_truth.rs"]
mod cluster_dml_known_truth;
#[path = "cohort_linear_effect.rs"]
mod cohort_linear_effect;
#[path = "cpdag_scenarios.rs"]
mod cpdag_scenarios;
#[path = "descriptive_comparison.rs"]
mod descriptive_comparison;
#[path = "dose_grid_functional.rs"]
mod dose_grid_functional;
#[path = "effect_constancy.rs"]
mod effect_constancy;
#[path = "effect_constancy_calibration.rs"]
mod effect_constancy_calibration;
#[path = "interference_observational_calibration.rs"]
mod interference_observational_calibration;
#[path = "interference_saturation_calibration.rs"]
mod interference_saturation_calibration;
#[path = "intervention_support.rs"]
mod intervention_support;
#[path = "joint_bayesian_transport.rs"]
mod joint_bayesian_transport;
#[path = "joint_cell_factorized.rs"]
mod joint_cell_factorized;
#[path = "latent_class_effects.rs"]
mod latent_class_effects;
#[path = "learned_joint_transport.rs"]
mod learned_joint_transport;
#[path = "local_polynomial_ratio_calibration.rs"]
mod local_polynomial_ratio_calibration;
#[path = "longitudinal_g_formula_calibration.rs"]
mod longitudinal_g_formula_calibration;
#[path = "matched_case_control.rs"]
mod matched_case_control;
#[path = "measurement_models.rs"]
mod measurement_models;
#[path = "mechanism_discrepancy.rs"]
mod mechanism_discrepancy;
#[path = "nested_markov_binary.rs"]
mod nested_markov_binary;
#[path = "nested_markov_uncertainty.rs"]
mod nested_markov_uncertainty;
#[path = "nonlinear_mediation.rs"]
mod nonlinear_mediation;
#[path = "observation_recovery_execution.rs"]
mod observation_recovery_execution;
#[path = "ordered_response_recovery_chain_truth.rs"]
mod ordered_response_recovery_chain_truth;
