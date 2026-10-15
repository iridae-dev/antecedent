//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "joint_mechanism_sensitivity.rs"]
mod joint_mechanism_sensitivity;
#[path = "msm_sensitivity.rs"]
mod msm_sensitivity;
#[path = "predictive_assessment.rs"]
mod predictive_assessment;
