//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "accepted_preferences.rs"]
mod accepted_preferences;
#[path = "adaptive_policy_search.rs"]
mod adaptive_policy_search;
#[path = "choice_record.rs"]
mod choice_record;
#[path = "choice_record_reference.rs"]
mod choice_record_reference;
#[path = "decision_reliability_resources.rs"]
mod decision_reliability_resources;
#[path = "evidence_plan.rs"]
mod evidence_plan;
#[path = "inverse_query_artifact.rs"]
mod inverse_query_artifact;
#[path = "paired_draws_alignment.rs"]
mod paired_draws_alignment;
#[path = "preference_artifact.rs"]
mod preference_artifact;
#[path = "preference_cancellation.rs"]
mod preference_cancellation;
#[path = "preference_certified.rs"]
mod preference_certified;
#[path = "preference_choice_model.rs"]
mod preference_choice_model;
#[path = "preference_contract.rs"]
mod preference_contract;
#[path = "preference_decision.rs"]
mod preference_decision;
#[path = "preference_decision_structural.rs"]
mod preference_decision_structural;
#[path = "preference_evaluation.rs"]
mod preference_evaluation;
#[path = "preference_model.rs"]
mod preference_model;
#[path = "preference_model_reference.rs"]
mod preference_model_reference;
#[path = "preference_set_learning.rs"]
mod preference_set_learning;
#[path = "preference_set_learning_reference.rs"]
mod preference_set_learning_reference;
#[path = "preference_shape_learning.rs"]
mod preference_shape_learning;
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
