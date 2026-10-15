//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "capability_matrix_registry.rs"]
mod capability_matrix_registry;
#[path = "cross_world_execution.rs"]
mod cross_world_execution;
#[path = "cross_world_execution_temporal.rs"]
mod cross_world_execution_temporal;
#[path = "temporal_cross_world.rs"]
mod temporal_cross_world;
#[path = "transported_path_specific.rs"]
mod transported_path_specific;
#[path = "whole_path_uncertainty.rs"]
mod whole_path_uncertainty;
