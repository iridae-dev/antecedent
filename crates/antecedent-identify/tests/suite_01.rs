//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "admg_conditional_transport.rs"]
mod admg_conditional_transport;
#[path = "artifact_compat_recovery_identity.rs"]
mod artifact_compat_recovery_identity;
#[path = "binary_iv_lp_oracle.rs"]
mod binary_iv_lp_oracle;
#[path = "counterfactual_id_search.rs"]
mod counterfactual_id_search;
#[path = "counterfactual_id_y0_parity.rs"]
mod counterfactual_id_y0_parity;
#[path = "cpdag_completion.rs"]
mod cpdag_completion;
#[path = "discovered_transport.rs"]
mod discovered_transport;
#[path = "gac_candidate_cap.rs"]
mod gac_candidate_cap;
#[path = "id_completeness.rs"]
mod id_completeness;
#[path = "meta_transport_scm.rs"]
mod meta_transport_scm;
#[path = "mixed_source_external_parity.rs"]
mod mixed_source_external_parity;
#[path = "mixed_source_search.rs"]
mod mixed_source_search;
#[path = "mz_transport_paper_reference.rs"]
mod mz_transport_paper_reference;
#[path = "mz_transport_search.rs"]
mod mz_transport_search;
#[path = "numeric_identification_cases.rs"]
mod numeric_identification_cases;
#[path = "observation_recovery.rs"]
mod observation_recovery;
#[path = "ordered_response_recovery_chain.rs"]
mod ordered_response_recovery_chain;
#[path = "pag_path_specific_candidates.rs"]
mod pag_path_specific_candidates;
#[path = "path_specific_district_scm.rs"]
mod path_specific_district_scm;
#[path = "scenario_invariance.rs"]
mod scenario_invariance;
#[path = "temporal_pag_path_specific.rs"]
mod temporal_pag_path_specific;
#[path = "temporal_sequence_spec.rs"]
mod temporal_sequence_spec;
#[path = "transport_identification_status.rs"]
mod transport_identification_status;
#[path = "z_transport_negative_evidence.rs"]
mod z_transport_negative_evidence;
#[path = "z_transport_outcome_inspection.rs"]
mod z_transport_outcome_inspection;
#[path = "z_transport_two_source.rs"]
mod z_transport_two_source;
