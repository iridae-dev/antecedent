//! Incremental causal state.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod conjugate_block_refresh;
pub mod conjugate_refresh;
pub mod covariance_refresh;
pub mod disk_retention;
pub mod error;
pub mod event;
pub mod graph_score;
pub mod invalidation;
pub mod mechanism_diag;
pub mod ols_refresh;
pub mod particle_filter;
pub mod refresh_contract;
pub mod retention;
pub mod state;
pub mod store;
pub mod suff_stats;
pub mod weighted_refresh;

pub use conjugate_block_refresh::ConjugateBlockRefresh;
pub use conjugate_refresh::{ConjugateOlsPosterior, ConjugateOlsPrior, ConjugateOlsSummary};
pub use covariance_refresh::{CovarianceRefresh, CovarianceRow, CovarianceSnapshot};
pub use disk_retention::{SpillHandle, SpillRegistry, SpillStats, Spillable};
pub use error::StateError;
pub use event::StateEvent;
pub use graph_score::{
    GraphScoreCacheKey, GraphScoreData, GraphScoreFamily, LocalScoreCache, ParentSetOp,
    full_graph_score,
};
pub use invalidation::{
    ChangedInput, InvalidationEntry, InvalidationLog, InvalidationTarget, TemporalDisposition,
    TemporalFactor, TemporalProduct,
};
pub use mechanism_diag::{
    RollingMechanismDiagnostics, evict_mechanism_diag, insert_mechanism_diag,
};
pub use ols_refresh::{OlsLedgerState, OlsRefresh, OlsSnapshot, RefreshOutcome, RefreshRow};
pub use particle_filter::{LgssmParams, ParticleFilterState};
pub use refresh_contract::{RefreshChange, RefreshDisposition, RefreshRoute, route_contract};
pub use retention::RetentionPolicy;
pub use state::{CausalState, ResultPublication};
pub use store::{
    CachedResult, ConstraintId, DataBatchRef, DataCatalog, DataVersion, GraphConstraintRecord,
    GraphEvidenceRecord, GraphEvidenceStore, InterventionRecord, ModelRecord, ModelStore,
    PublishedLineage, QueryRecord, QueryStore, ResultStore, SuffStatStore,
};
pub use suff_stats::{
    LagIndexCacheEntry, LagIndexCacheKey, LinearOlsDelta, LinearOlsSuffStats, MERGE_ABS_TOL,
    MERGE_REL_TOL, StreamingCovariance, merge_within_tolerance,
};
pub use weighted_refresh::{WeightedLedger, WeightedOlsRefresh, WeightedRow, WeightedSnapshot};
