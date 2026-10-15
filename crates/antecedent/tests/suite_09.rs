//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "z_transport_repair.rs"]
mod z_transport_repair;
#[path = "z_transport_surrogate_calibration.rs"]
mod z_transport_surrogate_calibration;
