//! Retention policies for state components.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// What raw history a component requires.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RetentionPolicy {
    /// Full raw history must be retained.
    RawHistory,
    /// Only a bounded trailing window is required.
    BoundedWindow {
        /// Maximum rows / events retained.
        max_rows: u64,
    },
    /// Sufficient statistics alone reconstruct the needed view.
    SufficientStatisticsOnly,
}
