//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "evidence_plan.rs"]
mod evidence_plan;
#[path = "inverse_query_artifact.rs"]
mod inverse_query_artifact;
#[path = "paired_draws_alignment.rs"]
mod paired_draws_alignment;
#[path = "prior_signal.rs"]
mod prior_signal;
#[path = "proposal_receipt.rs"]
mod proposal_receipt;
#[path = "repair_artifact.rs"]
mod repair_artifact;
#[path = "repair_facade.rs"]
mod repair_facade;
#[path = "route_evidence_sequential_evaluation.rs"]
mod route_evidence_sequential_evaluation;
#[path = "sensitivity_decision.rs"]
mod sensitivity_decision;
#[path = "sequential_evaluation.rs"]
mod sequential_evaluation;
#[path = "signal_provider.rs"]
mod signal_provider;
#[path = "structured_refusals.rs"]
mod structured_refusals;
#[path = "study_candidate.rs"]
mod study_candidate;
#[path = "transport_sensitivity.rs"]
mod transport_sensitivity;
#[path = "transported_bridge.rs"]
mod transported_bridge;
#[path = "transported_evaluation.rs"]
mod transported_evaluation;
#[path = "transported_route_variants.rs"]
mod transported_route_variants;
#[path = "transported_synthesis.rs"]
mod transported_synthesis;
#[path = "transported_verify_replay.rs"]
mod transported_verify_replay;
#[path = "value_of_information.rs"]
mod value_of_information;
