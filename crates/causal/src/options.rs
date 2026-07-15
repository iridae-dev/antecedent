//! Shared facade option enums (API hygiene).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_stats::{FdrAdjustment, MultipleTestingMethod};

/// Whether / how FDR (or FWER) adjustment is applied after CI tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FdrControl {
    /// No multiple-testing adjustment.
    Off,
    /// Adjust with the given procedure / family options.
    On(FdrAdjustment),
}

impl FdrControl {
    /// Benjamini–Hochberg with tigramite's contemporaneous exclusion.
    #[must_use]
    pub const fn bh() -> Self {
        Self::On(FdrAdjustment::bh())
    }

    /// Benjamini–Yekutieli with contemporaneous exclusion.
    #[must_use]
    pub const fn by() -> Self {
        Self::On(FdrAdjustment::by())
    }

    /// Bonferroni FWER.
    #[must_use]
    pub const fn bonferroni() -> Self {
        Self::On(FdrAdjustment {
            method: MultipleTestingMethod::Bonferroni,
            exclude_contemporaneous: true,
        })
    }

    /// Holm–Bonferroni FWER.
    #[must_use]
    pub const fn holm() -> Self {
        Self::On(FdrAdjustment {
            method: MultipleTestingMethod::Holm,
            exclude_contemporaneous: true,
        })
    }

    /// Whether any adjustment is enabled.
    #[must_use]
    pub const fn enabled(self) -> bool {
        matches!(self, Self::On(_))
    }

    /// Adjustment config when enabled.
    #[must_use]
    pub const fn adjustment(self) -> Option<FdrAdjustment> {
        match self {
            Self::Off => None,
            Self::On(cfg) => Some(cfg),
        }
    }
}

impl From<bool> for FdrControl {
    fn from(value: bool) -> Self {
        if value { Self::bh() } else { Self::Off }
    }
}

impl From<FdrAdjustment> for FdrControl {
    fn from(value: FdrAdjustment) -> Self {
        Self::On(value)
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
