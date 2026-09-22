//! Incremental causal state.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_range_loop
)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

pub mod error;
pub mod event;
pub mod graph_score;
pub mod invalidation;
pub mod mechanism_diag;
pub mod particle_filter;
pub mod retention;
pub mod state;
pub mod store;
pub mod suff_stats;

pub use error::StateError;
pub use event::StateEvent;
pub use graph_score::{
    GraphScoreCacheKey, GraphScoreData, GraphScoreFamily, LocalScoreCache, ParentSetOp,
    full_graph_score,
};
pub use invalidation::{InvalidationEntry, InvalidationLog, InvalidationTarget};
pub use mechanism_diag::{
    RollingMechanismDiagnostics, evict_mechanism_diag, insert_mechanism_diag,
};
pub use particle_filter::{LgssmParams, ParticleFilterState};
pub use retention::RetentionPolicy;
pub use state::{CausalState, ResultPublication};
pub use store::{
    CachedResult, ConstraintId, DataBatchRef, DataCatalog, DataVersion, GraphConstraintRecord,
    GraphEvidenceRecord, GraphEvidenceStore, InterventionRecord, ModelRecord, ModelStore,
    PublishedLineage, QueryRecord, QueryStore, ResultStore, SuffStatStore,
};
pub use suff_stats::{
    LagIndexCacheEntry, LagIndexCacheKey, LinearOlsSuffStats, StreamingCovariance,
};
