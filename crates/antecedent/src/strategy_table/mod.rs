//! Identifier / estimator strategy tables for plan compilation and static execution
//!. Incremental extraction from the analysis workflow — does not
//! replace [`crate::Study`] / plans / [`crate::StudyResult`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod dispatch;
mod ids;

pub use dispatch::*;
pub use ids::*;
