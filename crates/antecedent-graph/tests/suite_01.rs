//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "dsep_exhaustive.rs"]
mod dsep_exhaustive;

#[path = "oracle_closure.rs"]
mod oracle_closure;
