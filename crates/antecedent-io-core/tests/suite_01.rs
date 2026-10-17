//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "adaptive_monitoring_contract_artifact.rs"]
mod adaptive_monitoring_contract_artifact;
#[path = "artifact_compat_bridge_golden.rs"]
mod artifact_compat_bridge_golden;
#[path = "artifact_compat_callback_result_golden.rs"]
mod artifact_compat_callback_result_golden;
#[path = "artifact_compat_cross_world_execution_golden.rs"]
mod artifact_compat_cross_world_execution_golden;
#[path = "artifact_compat_golden.rs"]
mod artifact_compat_golden;
#[path = "artifact_compat_joint_law_propagation_golden.rs"]
mod artifact_compat_joint_law_propagation_golden;
#[path = "artifact_compat_mechanism_composition_golden.rs"]
mod artifact_compat_mechanism_composition_golden;
#[path = "artifact_compat_observation_query_golden.rs"]
mod artifact_compat_observation_query_golden;
#[path = "callback_result_artifact.rs"]
mod callback_result_artifact;
#[path = "callback_result_standing.rs"]
mod callback_result_standing;
#[path = "composition_verification_artifact.rs"]
mod composition_verification_artifact;
#[path = "cross_world_execution_artifact.rs"]
mod cross_world_execution_artifact;
#[path = "distribution_alignment_identity.rs"]
mod distribution_alignment_identity;
#[path = "evidence_ledger_artifact.rs"]
mod evidence_ledger_artifact;
#[path = "interface_contract_artifact.rs"]
mod interface_contract_artifact;
#[path = "joint_law_propagation_artifact.rs"]
mod joint_law_propagation_artifact;
#[path = "joint_law_propagation_bounds.rs"]
mod joint_law_propagation_bounds;
#[path = "mechanism_composition_artifact.rs"]
mod mechanism_composition_artifact;
#[path = "observation_query_artifact.rs"]
mod observation_query_artifact;
#[path = "policy_program_execution.rs"]
mod policy_program_execution;
#[path = "provider_envelope.rs"]
mod provider_envelope;
#[path = "scientific_bridge_artifact.rs"]
mod scientific_bridge_artifact;
#[path = "selective_discovery_artifact.rs"]
mod selective_discovery_artifact;
#[path = "transport_handoff_artifact.rs"]
mod transport_handoff_artifact;
#[path = "transport_handoff_provider_trust.rs"]
mod transport_handoff_provider_trust;
#[path = "transport_revalidation_artifact.rs"]
mod transport_revalidation_artifact;
#[path = "version_gate_before_shape.rs"]
mod version_gate_before_shape;
