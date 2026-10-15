//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "dense_v0_sequential.rs"]
mod dense_v0_sequential;
#[path = "mcmc_arviz_oracle.rs"]
mod mcmc_arviz_oracle;
#[path = "prior_support_oracle.rs"]
mod prior_support_oracle;
