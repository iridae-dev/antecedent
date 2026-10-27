//! Anomaly attribution, change explanation, and root-cause ranking.
//!
//! # Modules
//!
//! Browse by concern: [`anomaly`], [`distribution_change`](mod@distribution_change), [`structure_change`](mod@structure_change),
//! [`unit_change`](mod@unit_change), [`path`], [`feature_relevance`](mod@feature_relevance), [`mechanism_change`], [`robust`],
//! [`root_cause`], [`shapley`], [`builder`], [`intrinsic_influence`](mod@intrinsic_influence).
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
pub mod intrinsic_influence;
pub mod latent_blocks;
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
    score_anomalies, score_anomalies_blocks_with, score_anomalies_with,
};
pub use builder::ChangeAttribution;
pub use coalition::Coalition;
pub use distribution_change::{
    DifferenceMeasure, DistributionChangeOptions, PopulationRefit, distribution_change,
    distribution_change_with_frequency_weights, distribution_change_with_row_weights,
};
pub use error::AttributionError;
pub use feature_relevance::{FeatureRelevanceOptions, feature_relevance};
pub use intrinsic_influence::{
    ComponentStatus, ComponentUncertainty, FitResampling, INTRINSIC_MAX_EXACT_PLAYERS,
    IntrinsicAllocation, IntrinsicBudget, IntrinsicInfluenceQuery, IntrinsicInfluenceResult,
    IntrinsicMethod, NoiseIndependence, PartialStderr, ShapleyMethod, SourceInfluence,
    UncertaintyComponent, intrinsic_influence,
};
pub use latent_blocks::{
    HEDGE_WITNESS_ID, HEDGE_WITNESS_TEST, MechanismHedge, NoiseBlockIdentification, NoiseBlockPlan,
    distribution_change_blocks, plan_noise_blocks,
};
pub use mechanism_change::{
    MIN_BLOCK_PERMUTATIONS, MechanismChangeMethod, MechanismChangePosterior,
    detect_mechanism_changes, detect_mechanism_changes_block, mechanism_change_posterior,
};
pub use path::path_decompose;
pub use population::resolve_rows;
pub use result::{
    CacheStats, ChangeAttributionResult, ComponentContribution, ComputeBudget, FeatureRelevance,
    FitUncertainty, InteractionTerm, MechanismChangeDetection, PathContribution, RootCauseRank,
    UnitChangeResult,
};
pub use robust::{
    RobustChangeOptions, distribution_change_robust, distribution_change_robust_with_refit,
};
pub use root_cause::root_cause_rank;
pub use shapley::{CoalitionPayoff, ShapleyEstimate, estimate_shapley};
pub use structure_change::{StructureChangeOptions, structure_change, structure_change_with_refit};
pub use unit_change::unit_change;
