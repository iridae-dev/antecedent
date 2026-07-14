//! Anomaly attribution, change explanation, and root-cause ranking (DESIGN.md §17).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

mod anomaly;
mod coalition;
mod error;
mod population;
mod result;
mod shapley;

pub use anomaly::{
    AnomalyScores, ArrowStrength, arrow_strengths, intrinsic_influence, score_anomalies,
};
pub use coalition::{CoalitionCache, CoalitionKey};
pub use error::AttributionError;
pub use population::{
    multi_env_series, resolve_multi_env_rows, resolve_rows, subset_table,
};
pub use result::{
    CacheStats, ChangeAttributionResult, ComponentContribution, ComputeBudget, FeatureRelevance,
    InteractionTerm, MechanismChangeDetection, PathContribution, RootCauseRank, UnitChangeResult,
};
pub use shapley::{
    CoalitionPayoff, ShapleyEstimate, check_shapley_size, estimate_shapley, sequential_allocate,
};
