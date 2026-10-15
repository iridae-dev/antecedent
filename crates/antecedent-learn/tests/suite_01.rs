//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "bayesian_basis.rs"]
mod bayesian_basis;
#[path = "nested_markov_bayesian_reference.rs"]
mod nested_markov_bayesian_reference;
#[path = "nested_markov_bayesian_refusals.rs"]
mod nested_markov_bayesian_refusals;
