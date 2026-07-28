//! Unified `Study` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::cast_precision_loss
)]

mod batch;
mod builder;
mod execute;
mod helpers;
mod latency;
mod prepared;
mod stage;

pub use batch::BatchStudy;
pub use builder::{RdConfig, RefuteSuite, StudyBuilder};
pub use execute::Study;
pub use latency::{
    ComputeBudget, INTERACTIVE_BOOTSTRAP, INTERACTIVE_MAX_ENVELOPE_GRAPHS, INTERACTIVE_N_DRAWS,
    LatencyMode, REPORT_BOOTSTRAP, REPORT_N_DRAWS, ResolvedLatencyBudget, STANDARD_BOOTSTRAP,
    STANDARD_N_DRAWS, refuse_discovery_under_interactive, refuse_non_report_hmc,
};
pub use prepared::PreparedStudy;
pub use stage::{StageEvent, StageResultSink};
