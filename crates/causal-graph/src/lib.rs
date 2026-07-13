//! Causal graph types, dense indexes, and traversal workspaces.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod ancestry;
pub mod dag;
pub mod dsep;
pub mod error;
pub mod temporal;
pub mod types;
pub mod unfold;
pub mod workspace;

pub use dag::Dag;
pub use dsep::{DSeparationWorkspace, PathStep, SeparationCertificate, SeparationResult};
pub use error::GraphError;
pub use temporal::TemporalDag;
pub use types::{DenseNodeId, Endpoint, MarkedEdge, NodeRef};
pub use unfold::{LazyUnfoldedTemporalGraph, TemporalGraphReview, UnfoldedTemporalGraph, ensure_lagged};
pub use workspace::{BitSet, GraphWorkspace};
