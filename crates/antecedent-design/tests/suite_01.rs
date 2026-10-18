//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "artifact_compat_inherited.rs"]
mod artifact_compat_inherited;
#[path = "artifact_compat_sequential_reliability_golden.rs"]
mod artifact_compat_sequential_reliability_golden;
#[path = "cohort_decision_input.rs"]
mod cohort_decision_input;
#[path = "composition_boundary.rs"]
mod composition_boundary;
#[path = "composition_bundle.rs"]
mod composition_bundle;
#[path = "composition_verifiers.rs"]
mod composition_verifiers;
#[path = "coordinate_parity.rs"]
mod coordinate_parity;
#[path = "cross_surface_bundle.rs"]
mod cross_surface_bundle;
#[path = "cross_surface_decision.rs"]
mod cross_surface_decision;
#[path = "decision_acceptance.rs"]
mod decision_acceptance;
#[path = "decision_adapters.rs"]
mod decision_adapters;
#[path = "decision_artifact.rs"]
mod decision_artifact;
#[path = "decision_contract_eval.rs"]
mod decision_contract_eval;
#[path = "decision_mean_source.rs"]
mod decision_mean_source;
#[path = "decision_reliability.rs"]
mod decision_reliability;
#[path = "decision_reliability_candidate_counts.rs"]
mod decision_reliability_candidate_counts;
#[path = "decision_reliability_intervals.rs"]
mod decision_reliability_intervals;
#[path = "decision_reliability_producers.rs"]
mod decision_reliability_producers;
#[path = "decision_robust_artifact.rs"]
mod decision_robust_artifact;
#[path = "decision_robustness.rs"]
mod decision_robustness;
#[path = "decision_structural.rs"]
mod decision_structural;
#[path = "decision_structural_artifact.rs"]
mod decision_structural_artifact;
#[path = "decision_transported_source.rs"]
mod decision_transported_source;
#[path = "design_oracle.rs"]
mod design_oracle;
#[path = "design_ranking.rs"]
mod design_ranking;
#[path = "design_ranking_artifact.rs"]
mod design_ranking_artifact;
#[path = "evsi.rs"]
mod evsi;
#[path = "identity_properties_decision_contract.rs"]
mod identity_properties_decision_contract;
#[path = "inverse_forward_source.rs"]
mod inverse_forward_source;
#[path = "inverse_observational_scenarios.rs"]
mod inverse_observational_scenarios;
#[path = "inverse_query.rs"]
mod inverse_query;
