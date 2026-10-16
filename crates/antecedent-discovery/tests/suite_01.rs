//! Consolidated integration tests: one binary for this group of files, each
//! included in place so its path, helpers and registry citations are unchanged.
#![allow(clippy::all, reason = "each included file carries its own lint scope")]
// consolidated-suite

#[path = "ci_search_threads.rs"]
mod ci_search_threads;
#[path = "dag_posterior_conformance.rs"]
mod dag_posterior_conformance;
#[path = "discovery_jpcmci_plus_two_env_edges.rs"]
mod discovery_jpcmci_plus_two_env_edges;
#[path = "discovery_masked_mci_lag1.rs"]
mod discovery_masked_mci_lag1;
#[path = "discovery_notears_chain.rs"]
mod discovery_notears_chain;
#[path = "discovery_pc_parcorr.rs"]
mod discovery_pc_parcorr;
#[path = "discovery_pcmci_lag1.rs"]
mod discovery_pcmci_lag1;
#[path = "discovery_pcmci_multivar.rs"]
mod discovery_pcmci_multivar;
#[path = "discovery_pcmci_plus_lag0.rs"]
mod discovery_pcmci_plus_lag0;
#[path = "discovery_vector_vars_pcmci.rs"]
mod discovery_vector_vars_pcmci;
#[path = "graph_mcmc_oracle.rs"]
mod graph_mcmc_oracle;
#[path = "jpcmci_plus_oracle_matrix.rs"]
mod jpcmci_plus_oracle_matrix;
#[path = "lpcmci_evidence_reconciliation.rs"]
mod lpcmci_evidence_reconciliation;
#[path = "lpcmci_oracle_matrix.rs"]
mod lpcmci_oracle_matrix;
#[path = "multiplicity_oracle.rs"]
mod multiplicity_oracle;
#[path = "rpcmci_fixed_regime_oracle.rs"]
mod rpcmci_fixed_regime_oracle;
#[path = "selective_discovery.rs"]
mod selective_discovery;
#[path = "selective_populations_and_split.rs"]
mod selective_populations_and_split;
#[path = "static_discovery_oracle.rs"]
mod static_discovery_oracle;
