//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "dbn_mediation_horizon_mass.rs"]
mod dbn_mediation_horizon_mass;
#[path = "dbn_posterior_atom_identity.rs"]
mod dbn_posterior_atom_identity;
#[path = "dbn_sustained_validation.rs"]
mod dbn_sustained_validation;
#[path = "decision_exit_gate.rs"]
mod decision_exit_gate;
#[path = "decision_reliability_derived_facade.rs"]
mod decision_reliability_derived_facade;
#[path = "decision_reliability_facade.rs"]
mod decision_reliability_facade;
#[path = "dense_prior_hydration.rs"]
mod dense_prior_hydration;
#[path = "derived_treatment.rs"]
mod derived_treatment;
#[path = "design_state.rs"]
mod design_state;
#[path = "discovered_transport_facade.rs"]
mod discovered_transport_facade;
#[path = "discretized_composition_facade.rs"]
mod discretized_composition_facade;
#[path = "distribution_route_evidence.rs"]
mod distribution_route_evidence;
#[path = "dml_scores_retarget.rs"]
mod dml_scores_retarget;
#[path = "effect_constancy_consumers.rs"]
mod effect_constancy_consumers;
#[path = "envelope_prior_anchor.rs"]
mod envelope_prior_anchor;
#[path = "envelope_refuter_targets.rs"]
mod envelope_refuter_targets;
#[path = "estimate_conformance.rs"]
mod estimate_conformance;
#[path = "estimate_linear_gaussian_ate.rs"]
mod estimate_linear_gaussian_ate;
#[path = "estimate_noisy_estimators.rs"]
mod estimate_noisy_estimators;
#[path = "evidence_ledger_facade.rs"]
mod evidence_ledger_facade;
#[path = "examples_run.rs"]
mod examples_run;
#[path = "facade_gaps.rs"]
mod facade_gaps;
#[path = "frontdoor_iv_route_evidence.rs"]
mod frontdoor_iv_route_evidence;
#[path = "graph_interchange.rs"]
mod graph_interchange;
#[path = "graph_posterior_repeated_atoms.rs"]
mod graph_posterior_repeated_atoms;
#[path = "graph_typed_inputs.rs"]
mod graph_typed_inputs;
#[path = "interface_contract_facade.rs"]
mod interface_contract_facade;
#[path = "interference_interval_route.rs"]
mod interference_interval_route;
#[path = "inverse_cohort_source_facade.rs"]
mod inverse_cohort_source_facade;
#[path = "inverse_composition_refusals_facade.rs"]
mod inverse_composition_refusals_facade;
