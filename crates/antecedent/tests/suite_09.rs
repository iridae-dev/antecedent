//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "class_attribution_route.rs"]
mod class_attribution_route;
#[path = "compiled_twins_joint_panel.rs"]
mod compiled_twins_joint_panel;
#[path = "compiled_twins_longitudinal_regime.rs"]
mod compiled_twins_longitudinal_regime;
#[path = "compiled_twins_policy_value.rs"]
mod compiled_twins_policy_value;
#[path = "compiled_twins_randomized_policy.rs"]
mod compiled_twins_randomized_policy;
#[path = "compiled_twins_staggered_did.rs"]
mod compiled_twins_staggered_did;
#[path = "compiled_twins_survival_ipcw.rs"]
mod compiled_twins_survival_ipcw;
#[path = "counterfactual_class_route.rs"]
mod counterfactual_class_route;
#[path = "mediation_mixed_graph.rs"]
mod mediation_mixed_graph;
#[path = "predictive_score_interval_facade.rs"]
mod predictive_score_interval_facade;
#[path = "released_surface.rs"]
mod released_surface;
#[path = "resampling_accuracy.rs"]
mod resampling_accuracy;
#[path = "route_diagnostics.rs"]
mod route_diagnostics;
#[path = "sampling_design_declaration.rs"]
mod sampling_design_declaration;
#[path = "typed_transported_mechanism_facade.rs"]
mod typed_transported_mechanism_facade;
#[path = "z_transport_repair.rs"]
mod z_transport_repair;
#[path = "z_transport_surrogate_calibration.rs"]
mod z_transport_surrogate_calibration;
