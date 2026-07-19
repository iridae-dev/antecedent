//! Abduction-action-prediction counterfactual evaluation (DESIGN.md §16).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod engine;
pub mod error;
pub mod trajectory;

pub use engine::{
    AbductionMissingPolicy, CompiledCounterfactualPlan, CounterfactualEngine, CounterfactualResult,
    CounterfactualWorld, ExogenousPosterior, NoiseInferenceKind, nested_hard_counterfactual,
    simultaneous_hard_counterfactual, streaming_matches_retained,
};
pub use error::CounterfactualError;
pub use trajectory::{
    CounterfactualTrajectoryRequest, TrajectoryArm, TrajectoryResult, evaluate_trajectories,
};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
