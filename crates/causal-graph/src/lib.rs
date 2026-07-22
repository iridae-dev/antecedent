//! Causal graph types, dense indexes, and traversal workspaces.
//!
//! Distinct concrete types ([`Dag`], [`Admg`], [`Cpdag`], [`Pag`], [`TemporalDag`], …)
//! preserve edge semantics — they are not interchangeable aliases.
//!
//! ```
//! use causal_graph::{Dag, DenseNodeId};
//!
//! let mut dag = Dag::empty();
//! let a = DenseNodeId::from_raw(0);
//! let b = DenseNodeId::from_raw(1);
//! // Prefer schema-aligned constructors in real code; empty DAGs are for scaffolding.
//! assert_eq!(dag.node_count(), 0);
//! let _ = (a, b, dag);
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
pub mod dag;
pub mod dsep;
pub mod error;
pub(crate) mod marked_storage;
pub mod msep;
pub mod overlay;
pub mod pag;
pub mod projection;
pub mod temporal;
pub mod temporal_pag;
pub mod types;
pub mod unfold;
pub mod workspace;

pub use admg::Admg;
pub use causal_core::NodeRef;
pub use completion::{CompletionSampler, PagCompletion, is_mag_completion};
pub use cpdag::{Cpdag, CpdagReview, TemporalCpdag};
pub use cpdag_completion::{CpdagCompletion, CpdagCompletionSampler, is_mec_member};
pub use dag::{Dag, DagReview};
pub use dsep::{DSeparationWorkspace, PathStep, SeparationCertificate, SeparationResult};
pub use error::GraphError;
pub use overlay::{DagView, GraphOverlay};
pub use pag::{DefiniteStatusPath, DefiniteStatusPathSearch, Pag, PagReview};
pub use projection::{latent_project, projection_preserves_msep_sample};
pub use temporal::TemporalDag;
pub use temporal_pag::{TemporalPag, TemporalPagReview};
pub use types::{DenseNodeId, Endpoint, MarkedEdge, MiddleMark};
pub use unfold::{
    LazyUnfoldedTemporalGraph, TemporalCpdagReview, TemporalGraphReview, UnfoldedTemporalGraph,
    ensure_lagged,
};
pub use workspace::{BitSet, GraphWorkspace};
