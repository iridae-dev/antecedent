//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "particle_filter_oracle.rs"]
mod particle_filter_oracle;
#[path = "rolling_mechanism_oracle.rs"]
mod rolling_mechanism_oracle;
