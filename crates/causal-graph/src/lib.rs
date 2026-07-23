//! Causal graph types, dense indexes, and traversal workspaces.
//!
//! Distinct concrete types ([`Dag`], [`Admg`], [`Cpdag`], [`Pag`], [`TemporalDag`], …)
//! preserve edge semantics — they are not interchangeable aliases.
//!
//! ```
//! use causal_core::CausalSchemaBuilder;
//! use causal_graph::Dag;
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
pub mod dag;
pub mod dsep;
pub mod error;
pub mod marked_storage;
pub mod msep;
pub mod named;
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
