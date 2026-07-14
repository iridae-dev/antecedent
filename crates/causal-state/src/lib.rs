//! Incremental causal state (DESIGN.md §20).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop
)]

pub mod error;
pub mod event;
pub mod invalidation;
pub mod retention;
pub mod state;
pub mod store;
pub mod suff_stats;

pub use error::StateError;
pub use event::StateEvent;
pub use invalidation::{InvalidationEntry, InvalidationLog, InvalidationTarget};
pub use retention::RetentionPolicy;
pub use state::CausalState;
pub use store::{
    CachedResult, ConstraintId, DataBatchRef, DataCatalog, DataVersion, GraphConstraintRecord,
    GraphEvidenceRecord, GraphEvidenceStore, InterventionRecord, ModelRecord, ModelStore,
    QueryRecord, QueryStore, ResultStore, SuffStatStore,
};
pub use suff_stats::{
    LagIndexCacheEntry, LagIndexCacheKey, LinearOlsSuffStats, StreamingCovariance,
};
