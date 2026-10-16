//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "arrow_cdi_zero_copy_gate.rs"]
mod arrow_cdi_zero_copy_gate;
#[path = "arrow_copy_gate.rs"]
mod arrow_copy_gate;
