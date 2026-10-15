//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "discretized_scm.rs"]
mod discretized_scm;
#[path = "discretized_scm_artifact.rs"]
mod discretized_scm_artifact;
#[path = "mechanism_composition.rs"]
mod mechanism_composition;
#[path = "mechanism_transport.rs"]
mod mechanism_transport;
#[path = "mechanism_transport_identity_fields.rs"]
mod mechanism_transport_identity_fields;
#[path = "observation_query.rs"]
mod observation_query;
#[path = "route_evidence_observation_query.rs"]
mod route_evidence_observation_query;
#[path = "scm_oracle.rs"]
mod scm_oracle;
#[path = "transport_obstruction_agreement.rs"]
mod transport_obstruction_agreement;
