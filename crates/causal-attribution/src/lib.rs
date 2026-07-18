//! Anomaly attribution, change explanation, and root-cause ranking (DESIGN.md §17).
//!
//! # Modules
//!
//! Browse by concern: [`anomaly`], [`distribution_change`], [`unit_change`],
//! [`path`], [`feature_relevance`], [`mechanism_change`], [`robust`], [`root_cause`],
//! [`shapley`], [`builder`].
//!
//! # Example
//!
//! ```rust,ignore
//! use causal_attribution::score_anomalies;
//! use causal_core::{AnomalyAttributionQuery, VariableId};
//!
//! let query = AnomalyAttributionQuery::new([VariableId::from_raw(0)], 100);
//! let scores = score_anomalies(&model, &data, &query)?;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

pub mod anomaly;
pub mod builder;
pub mod coalition;
pub mod distribution_change;
pub mod error;
pub mod feature_relevance;
pub mod mechanism_change;
pub mod path;
pub mod population;
pub mod result;
pub mod robust;
pub mod root_cause;
pub mod shapley;
pub mod unit_change;

pub use anomaly::{
    AnomalyScores, ArrowStrength, arrow_strengths, population_do_contrast, score_anomalies,
};
pub use builder::ChangeAttribution;
pub use coalition::{CoalitionCache, CoalitionKey};
pub use distribution_change::{
    DifferenceMeasure, DistributionChangeOptions, distribution_change, distribution_change_shapley,
};
pub use error::AttributionError;
pub use feature_relevance::feature_relevance;
pub use mechanism_change::{MechanismChangeMethod, detect_mechanism_changes};
pub use path::path_decompose;
pub use population::{multi_env_series, resolve_multi_env_rows, resolve_rows, subset_table};
pub use result::{
    CacheStats, ChangeAttributionResult, ComponentContribution, ComputeBudget, FeatureRelevance,
    InteractionTerm, MechanismChangeDetection, PathContribution, RootCauseRank, UnitChangeResult,
};
pub use robust::{RobustChangeOptions, distribution_change_robust};
pub use root_cause::{
    aggregate_model_collection_ranks, contribution_posterior_from_rows,
    posterior_contribution_ranks, root_cause_rank,
};
pub use shapley::{
    CoalitionPayoff, ShapleyEstimate, check_shapley_size, estimate_shapley, sequential_allocate,
};
pub use unit_change::unit_change;
