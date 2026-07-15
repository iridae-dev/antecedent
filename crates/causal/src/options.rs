//! Shared facade option enums (API hygiene).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Whether Benjamini–Hochberg FDR is applied after CI tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FdrControl {
    /// No FDR adjustment.
    Off,
    /// Benjamini–Hochberg FDR.
    Bh,
}

impl FdrControl {
    /// Whether FDR is enabled.
    #[must_use]
    pub const fn enabled(self) -> bool {
        matches!(self, Self::Bh)
    }
}

impl From<bool> for FdrControl {
    fn from(value: bool) -> Self {
        if value { Self::Bh } else { Self::Off }
    }
}

/// Whether a discovery result may be auto-accepted into an analysis plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DiscoveryAccept {
    /// Leave as review-required.
    Review,
    /// Auto-accept when the algorithm permits (no pending undirected marks, etc.).
    AutoAccept,
}

impl DiscoveryAccept {
    /// Whether auto-accept is requested.
    #[must_use]
    pub const fn auto(self) -> bool {
        matches!(self, Self::AutoAccept)
    }
}

impl From<bool> for DiscoveryAccept {
    fn from(value: bool) -> Self {
        if value { Self::AutoAccept } else { Self::Review }
    }
}
