//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "adaptive_inference_oracle.rs"]
mod adaptive_inference_oracle;
#[path = "adaptive_inversion_oracle.rs"]
mod adaptive_inversion_oracle;
#[path = "adaptive_procedure_bounds.rs"]
mod adaptive_procedure_bounds;
#[path = "adaptive_procedure_uncertainty_oracle.rs"]
mod adaptive_procedure_uncertainty_oracle;
#[path = "adaptive_repeated_measures_oracle.rs"]
mod adaptive_repeated_measures_oracle;
#[path = "advanced_ci_oracle.rs"]
mod advanced_ci_oracle;
#[path = "bayesian_ci_oracle.rs"]
mod bayesian_ci_oracle;
#[path = "discovery_ci_stats.rs"]
mod discovery_ci_stats;
#[path = "foundations_oracle.rs"]
mod foundations_oracle;
#[path = "route_evidence_joint_law.rs"]
mod route_evidence_joint_law;
#[path = "synthesis_oracle.rs"]
mod synthesis_oracle;
#[path = "uncertainty_routing_contract.rs"]
mod uncertainty_routing_contract;
