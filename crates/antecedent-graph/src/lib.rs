//! Causal graph types, dense indexes, and traversal workspaces.
//!
//! Distinct concrete types ([`Dag`], [`Admg`], [`CyclicDigraph`],
//! [`CyclicMixedGraph`], [`Cpdag`], [`Pag`], [`TemporalDag`], …) preserve edge
//! semantics — they are not interchangeable aliases. [`Dag`] and [`Admg`] still
//! reject directed cycles.
//!
//! ```
//! use antecedent_core::CausalSchemaBuilder;
//! use antecedent_graph::Dag;
//!
//! let schema = CausalSchemaBuilder::new()
//!     .continuous("a")
//!     .finish()
//!     .continuous("b")
//!     .finish()
//!     .build()
//!     .unwrap();
//! let dag = Dag::from_named_edges(&schema, &[("a", "b")]).unwrap();
//! assert_eq!(dag.node_count(), 2);
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod admg;
pub mod algo;
pub mod ancestry;
pub mod completion;
pub mod cpdag;
pub mod cpdag_completion;
pub mod cyclic;
pub mod dag;
mod directed;
pub mod dsep;
pub mod error;
pub mod marked_storage;
mod mec_search;
pub mod msep;
pub mod named;
pub mod overlay;
pub mod pag;
pub mod projection;
mod scc;
pub mod selection;
pub mod sigma_separation;
pub mod temporal;
pub mod temporal_admg;
pub mod temporal_cpdag_completion;
pub mod temporal_pag;
pub mod temporal_pag_completion;
pub mod tiered;
pub mod types;
pub mod unfold;
pub mod unfold_pag;
pub mod workspace;

pub use admg::{Admg, CopyLabels};
pub use antecedent_core::NodeRef;
pub use completion::{
    CompletionSampler, CompletionValidationReport, DEFAULT_COMPLETION_OPERATIONS, PagCompletion,
    is_mag_completion,
};
pub use cpdag::{Cpdag, CpdagReview, TemporalCpdag};
pub use cpdag_completion::{CpdagCompletion, CpdagCompletionSampler, is_mec_member};
pub use cyclic::{CyclicDigraph, CyclicMixedGraph};
pub use dag::{Dag, DagReview};
pub use dsep::{DSeparationWorkspace, PathStep, SeparationCertificate, SeparationResult};
pub use error::{
    CyclicDagRefusal, CyclicNextAttempt, DagOperation, GraphError, directed_edges_contain_cycle,
    refuse_directed_cycle,
};
pub use msep::PagSeparation;
pub use overlay::{DagView, GraphOverlay};
pub use pag::{DefiniteStatusPath, DefiniteStatusPathSearch, Pag, PagReview};
pub use projection::{latent_project, projection_preserves_msep_sample};
pub use scc::Condensation;
pub use selection::SelectionDiagram;
pub use sigma_separation::{
    OrdinaryDSeparationOnCycles, SigmaSeparation, SigmaSeparationAssumptions,
    SigmaSeparationStatus, SigmaSeparationWorkspace, WALK_ORACLE_NODE_BOUND, sigma_separates,
    sigma_walk_oracle,
};
pub use temporal::TemporalDag;
pub use temporal_admg::TemporalAdmg;
pub use temporal_cpdag_completion::{TemporalCpdagCompletion, TemporalCpdagCompletionSampler};
pub use temporal_pag::{TemporalPag, TemporalPagReview};
pub use temporal_pag_completion::{TemporalPagCompletion, TemporalPagCompletionSampler};
pub use tiered::{TieredBackground, WithinTier};
pub use types::{DenseNodeId, Endpoint, MarkedEdge, MiddleMark};
pub use unfold::{
    LazyUnfoldedTemporalGraph, TemporalCpdagReview, TemporalGraphReview, UnfoldedTemporalGraph,
    ensure_lagged,
};
pub use unfold_pag::UnfoldedTemporalPag;
pub use workspace::{BitSet, GraphWorkspace};

/// Equation-variable causal ordering and determinism-aware B-separation.
pub mod bipartite;
