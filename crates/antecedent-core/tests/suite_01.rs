//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "identity_properties_standing.rs"]
mod identity_properties_standing;
#[path = "recalc_plan.rs"]
mod recalc_plan;
#[path = "unleased_parallelism.rs"]
mod unleased_parallelism;
