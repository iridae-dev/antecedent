//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "choice_record.rs"]
mod choice_record;
#[path = "choice_record_reference.rs"]
mod choice_record_reference;
#[path = "preference_contract.rs"]
mod preference_contract;
#[path = "preference_model.rs"]
mod preference_model;
#[path = "preference_model_reference.rs"]
mod preference_model_reference;
#[path = "preference_set_learning.rs"]
mod preference_set_learning;
#[path = "preference_set_learning_reference.rs"]
mod preference_set_learning_reference;
#[path = "preference_decision.rs"]
mod preference_decision;
#[path = "preference_decision_structural.rs"]
mod preference_decision_structural;
