//! Unified `CausalAnalysis` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::cast_precision_loss
)]

mod builder;
mod execute;
mod helpers;

pub use builder::{CausalAnalysisBuilder, RdConfig, RefuteSuite};
pub use execute::CausalAnalysis;
