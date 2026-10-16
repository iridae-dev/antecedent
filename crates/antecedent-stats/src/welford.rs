//! Streaming mean / sum-of-squared-deviations accumulator.
//!
//! The update lives in `antecedent-kernels` so the HMC diagonal metric and this
//! crate share one recurrence. [`Welford`] is re-exported at the crate root.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_kernels::Welford;
