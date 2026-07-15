//! Abduction-action-prediction counterfactual evaluation (DESIGN.md §16).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod engine;
pub mod error;

pub use engine::{
    CompiledCounterfactualPlan, CounterfactualEngine, CounterfactualResult, CounterfactualWorld,
    ExogenousPosterior, MissingPolicy, NoiseInferenceKind, nested_hard_counterfactual,
    streaming_matches_retained,
};
pub use error::CounterfactualError;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
