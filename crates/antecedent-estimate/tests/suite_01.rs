//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "adaptive_decision_stopping_posterior_reference.rs"]
mod adaptive_decision_stopping_posterior_reference;
#[path = "adaptive_inference.rs"]
mod adaptive_inference;
#[path = "adaptive_interval_exact.rs"]
mod adaptive_interval_exact;
#[path = "adaptive_repeated_measures_weighting_reference.rs"]
mod adaptive_repeated_measures_weighting_reference;
#[path = "adjusted_contrast_reference.rs"]
mod adjusted_contrast_reference;
#[path = "ancova_reference.rs"]
mod ancova_reference;
#[path = "augmented_did_reference.rs"]
mod augmented_did_reference;
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
#[path = "bayesian_local_ratio_reference.rs"]
mod bayesian_local_ratio_reference;
#[path = "bayesian_mediation_reference.rs"]
mod bayesian_mediation_reference;
#[path = "bayesian_panel_did_reference.rs"]
mod bayesian_panel_did_reference;
#[path = "bayesian_rd_reference.rs"]
mod bayesian_rd_reference;
#[path = "bayesian_robust_ate_calibration.rs"]
mod bayesian_robust_ate_calibration;
#[path = "bayesian_route_exact.rs"]
mod bayesian_route_exact;
#[path = "bayesian_score_route_exact.rs"]
mod bayesian_score_route_exact;
#[path = "bayesian_synthetic_control_reference.rs"]
mod bayesian_synthetic_control_reference;
#[path = "bayesian_temporal_regime_survival_reference.rs"]
mod bayesian_temporal_regime_survival_reference;
#[path = "cancellation_expansion_cells.rs"]
mod cancellation_expansion_cells;
#[path = "categorical_declared_reference.rs"]
mod categorical_declared_reference;
#[path = "categorical_regression_reference.rs"]
mod categorical_regression_reference;
#[path = "cells_01_staggered_serial_reference.rs"]
mod cells_01_staggered_serial_reference;
#[path = "choice_interval_construction.rs"]
mod choice_interval_construction;
#[path = "cluster_dml_aipw.rs"]
mod cluster_dml_aipw;
#[path = "cluster_dml_known_truth.rs"]
mod cluster_dml_known_truth;
#[path = "clustered_score_posterior_exact.rs"]
mod clustered_score_posterior_exact;
