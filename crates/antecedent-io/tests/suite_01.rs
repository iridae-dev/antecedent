//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "adaptive_inference_result_artifact.rs"]
mod adaptive_inference_result_artifact;
#[path = "artifact_compat_adaptive_result.rs"]
mod artifact_compat_adaptive_result;
#[path = "artifact_compat_inherited.rs"]
mod artifact_compat_inherited;
#[path = "artifact_compat_predictive_adaptive_golden.rs"]
mod artifact_compat_predictive_adaptive_golden;
#[path = "artifact_compat_recalc_receipt.rs"]
mod artifact_compat_recalc_receipt;
#[path = "bayesian_posterior_row_artifact.rs"]
mod bayesian_posterior_row_artifact;
#[path = "categorical_treatment_artifact.rs"]
mod categorical_treatment_artifact;
#[path = "cohort_interval_artifact.rs"]
mod cohort_interval_artifact;
#[path = "compact_export.rs"]
mod compact_export;
#[path = "composition_cross_surface.rs"]
mod composition_cross_surface;
#[path = "cpdag_completion_artifact.rs"]
mod cpdag_completion_artifact;
#[path = "cross_surface_external_claim.rs"]
mod cross_surface_external_claim;
#[path = "cross_surface_joint_law.rs"]
mod cross_surface_joint_law;
#[path = "distribution_artifact.rs"]
mod distribution_artifact;
#[path = "distribution_meanings.rs"]
mod distribution_meanings;
#[path = "dose_grid_artifact.rs"]
mod dose_grid_artifact;
#[path = "effect_constancy_artifact.rs"]
mod effect_constancy_artifact;
#[path = "external_binding_wire.rs"]
mod external_binding_wire;
#[path = "external_claim_artifact.rs"]
mod external_claim_artifact;
#[path = "external_column_claim_artifact.rs"]
mod external_column_claim_artifact;
#[path = "flagship_decision_artifact.rs"]
mod flagship_decision_artifact;
#[path = "frozen_scores_artifact.rs"]
mod frozen_scores_artifact;
#[path = "joint_bayesian_source_dependence_ledger.rs"]
mod joint_bayesian_source_dependence_ledger;
#[path = "joint_bayesian_transport_artifact.rs"]
mod joint_bayesian_transport_artifact;
#[path = "joint_bayesian_transport_model_artifact.rs"]
mod joint_bayesian_transport_model_artifact;
#[path = "latent_class_artifact.rs"]
mod latent_class_artifact;
#[path = "learned_joint_transport_artifact.rs"]
mod learned_joint_transport_artifact;
#[path = "mechanism_discrepancy_artifact.rs"]
mod mechanism_discrepancy_artifact;
#[path = "msm_sensitivity_adapter.rs"]
mod msm_sensitivity_adapter;
#[path = "native_contract_identity.rs"]
mod native_contract_identity;
#[path = "nested_markov_artifact.rs"]
mod nested_markov_artifact;
#[path = "nested_markov_bayesian_artifact.rs"]
mod nested_markov_bayesian_artifact;
