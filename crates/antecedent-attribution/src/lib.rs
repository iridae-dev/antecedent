//! Anomaly attribution, change explanation, and root-cause ranking.
//!
//! # Modules
//!
//! Browse by concern: [`anomaly`], [`distribution_change`](mod@distribution_change), [`structure_change`](mod@structure_change),
//! [`unit_change`](mod@unit_change), [`path`], [`feature_relevance`](mod@feature_relevance), [`mechanism_change`], [`robust`],
//! [`root_cause`], [`shapley`], [`builder`].
//!
//! # Example
//!
//! ```rust,ignore
//! use antecedent_attribution::score_anomalies;
//! use antecedent_core::{AnomalyAttributionQuery, VariableId};
//!
//! let query = AnomalyAttributionQuery::new([VariableId::from_raw(0)], 100);
//! let scores = score_anomalies(&model, &data, &query)?;
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

pub mod anomaly;
pub mod builder;
pub mod change_common;
pub mod coalition;
pub mod distribution_change;
pub mod error;
pub mod feature_relevance;
pub mod mechanism_change;
pub mod path;
pub mod population;
pub(crate) mod prep;
pub mod result;
pub mod robust;
pub mod root_cause;
pub mod shapley;
pub mod structure_change;
pub mod unit_change;

pub use anomaly::{
    AnomalyScores, ArrowStrength, PopulationDoContrast, arrow_strengths, population_do_contrast,
    score_anomalies, score_anomalies_with,
};
pub use builder::ChangeAttribution;
pub use distribution_change::{
    DifferenceMeasure, DistributionChangeOptions, distribution_change,
    distribution_change_with_row_weights,
};
pub use error::AttributionError;
pub use feature_relevance::{FeatureRelevanceOptions, feature_relevance};
pub use mechanism_change::{MechanismChangeMethod, detect_mechanism_changes};
pub use path::path_decompose;
pub use population::resolve_rows;
pub use result::{
    CacheStats, ChangeAttributionResult, ComponentContribution, ComputeBudget, FeatureRelevance,
    FitUncertainty, InteractionTerm, MechanismChangeDetection, PathContribution, RootCauseRank,
    UnitChangeResult,
};
pub use robust::{RobustChangeOptions, distribution_change_robust};
pub use root_cause::root_cause_rank;
pub use shapley::{CoalitionPayoff, ShapleyEstimate, estimate_shapley};
pub use structure_change::{StructureChangeOptions, structure_change};
pub use unit_change::unit_change;
