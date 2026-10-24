//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "class_attribution_route.rs"]
mod class_attribution_route;
#[path = "counterfactual_class_route.rs"]
mod counterfactual_class_route;
#[path = "mediation_mixed_graph.rs"]
mod mediation_mixed_graph;
#[path = "predictive_score_interval_facade.rs"]
mod predictive_score_interval_facade;
#[path = "released_surface.rs"]
mod released_surface;
#[path = "route_diagnostics.rs"]
mod route_diagnostics;
#[path = "sampling_design_declaration.rs"]
mod sampling_design_declaration;
#[path = "resampling_accuracy.rs"]
mod resampling_accuracy;
#[path = "z_transport_repair.rs"]
mod z_transport_repair;
#[path = "z_transport_surrogate_calibration.rs"]
mod z_transport_surrogate_calibration;
