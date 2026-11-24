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
#[path = "preference_decision_class_weighting.rs"]
mod preference_decision_class_weighting;
#[path = "preference_decision_compiled.rs"]
mod preference_decision_compiled;
#[path = "preference_decision_compiled_atoms.rs"]
mod preference_decision_compiled_atoms;
#[path = "preference_decision_compiled_chain_binding.rs"]
mod preference_decision_compiled_chain_binding;
#[path = "preference_decision_compiled_class_marks.rs"]
mod preference_decision_compiled_class_marks;
#[path = "preference_decision_compiled_design.rs"]
mod preference_decision_compiled_design;
#[path = "preference_decision_compiled_evidence_basis.rs"]
mod preference_decision_compiled_evidence_basis;
#[path = "preference_decision_compiled_joint_law.rs"]
mod preference_decision_compiled_joint_law;
#[path = "preference_decision_compiled_outcomes.rs"]
mod preference_decision_compiled_outcomes;
#[path = "preference_decision_compiled_plan_facts.rs"]
mod preference_decision_compiled_plan_facts;
#[path = "preference_decision_compiled_program_identity.rs"]
mod preference_decision_compiled_program_identity;
#[path = "preference_decision_compiled_structures.rs"]
mod preference_decision_compiled_structures;
#[path = "preference_decision_compiled_unit_keys.rs"]
mod preference_decision_compiled_unit_keys;
#[path = "preference_decision_end_to_end.rs"]
mod preference_decision_end_to_end;
#[path = "preference_decision_several_outcomes.rs"]
mod preference_decision_several_outcomes;
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
