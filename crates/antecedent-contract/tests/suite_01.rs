//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "adaptive_design.rs"]
mod adaptive_design;
#[path = "callback_protocol.rs"]
mod callback_protocol;
#[path = "composition_provenance.rs"]
mod composition_provenance;
#[path = "composition_verification.rs"]
mod composition_verification;
#[path = "evidence_ledger.rs"]
mod evidence_ledger;
#[path = "evidence_obligation.rs"]
mod evidence_obligation;
#[path = "interface_adapters.rs"]
mod interface_adapters;
#[path = "interface_contract.rs"]
mod interface_contract;
#[path = "native_response_coordinates.rs"]
mod native_response_coordinates;
#[path = "program_binding.rs"]
mod program_binding;
#[path = "provider_transport.rs"]
mod provider_transport;
#[path = "scientific_bridge.rs"]
mod scientific_bridge;
#[path = "transport_identity_binding.rs"]
mod transport_identity_binding;
#[path = "transport_revalidation.rs"]
mod transport_revalidation;
