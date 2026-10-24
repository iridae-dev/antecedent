//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "cyclic_family_typed.rs"]
mod cyclic_family_typed;
#[path = "nonlinear_mediation_artifact.rs"]
mod nonlinear_mediation_artifact;
#[path = "period_refresh_summary_artifact.rs"]
mod period_refresh_summary_artifact;
#[path = "predictive_assessment_artifact.rs"]
mod predictive_assessment_artifact;
#[path = "recalc_receipt_artifact.rs"]
mod recalc_receipt_artifact;
#[path = "recovery_chain_artifact.rs"]
mod recovery_chain_artifact;
#[path = "release_migration_consumers.rs"]
mod release_migration_consumers;
#[path = "sampled_recovery_artifact.rs"]
mod sampled_recovery_artifact;
#[path = "scenario_covariance_artifact.rs"]
mod scenario_covariance_artifact;
#[path = "sensitivity_artifact.rs"]
mod sensitivity_artifact;
#[path = "temporal_counterfactual_artifact.rs"]
mod temporal_counterfactual_artifact;
#[path = "temporal_initial_state_artifact.rs"]
mod temporal_initial_state_artifact;
#[path = "temporal_interval_artifact.rs"]
mod temporal_interval_artifact;
#[path = "temporal_refresh_artifact.rs"]
mod temporal_refresh_artifact;
#[path = "transported_counterfactual_artifact.rs"]
mod transported_counterfactual_artifact;
#[path = "vector_treatment_artifact.rs"]
mod vector_treatment_artifact;
