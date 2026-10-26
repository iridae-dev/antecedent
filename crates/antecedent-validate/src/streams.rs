//! RNG stream ids for refuters and stability checks.
//!
//! Each value is the historical stream. A rename does not change draws; a new
//! number would.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Replicates whose stream is the historical `base + r`. Bases of the refuters above are
/// `0x1000` apart at the closest, so those offsets cannot alias a neighbouring refuter.
const ADDITIVE_REPLICATES: u32 = 0x1000;

/// Stream of replicate `r` of a refuter whose streams start at `base`.
///
/// Replicates below `0x1000` keep the historical `base + r` stream. Larger indices would run
/// into the next refuter's base, so they use a derived, hash-separated stream instead; any
/// replicate count is therefore accepted without two refuters sharing a draw.
pub(crate) const fn replicate_stream(base: u64, r: u32) -> u64 {
    if r < ADDITIVE_REPLICATES {
        base.wrapping_add(r as u64)
    } else {
        antecedent_core::derive_stream(base, r as u64)
    }
}

/// Placebo refit replicates.
pub(crate) const PLACEBO_REFIT: u64 = 0xA7E0_0001_0000;
/// Placebo treatment shuffle within a replicate.
pub(crate) const PLACEBO_SHUFFLE: u64 = 0xA7E0_0001_1000;
/// Random-common-cause noise.
pub(crate) const RANDOM_COMMON_CAUSE: u64 = 0xA7E0_0002_0000;
/// Unobserved-common-cause noise.
pub(crate) const UNOBSERVED_COMMON_CAUSE: u64 = 0xA7E0_0006_0000;
/// Data-subset replicate.
pub(crate) const DATA_SUBSET: u64 = 0xA7E0_0007_0000;
/// Dummy-outcome noise.
pub(crate) const DUMMY_OUTCOME: u64 = 0xA7E0_0008_0000;
/// Bootstrap refutation resamples.
pub(crate) const BOOTSTRAP_REFUTE: u64 = 0xA7E0_0009_0000;
/// Linear sensitivity confounder.
pub(crate) const LINEAR_SENSITIVITY: u64 = 0xA7E0_000A_0000;
/// Partial-linear sensitivity confounder.
pub(crate) const PARTIAL_LINEAR_SENSITIVITY: u64 = 0xA7E0_000B_0000;
/// Nonparametric sensitivity confounder.
pub(crate) const NONPARAMETRIC_SENSITIVITY: u64 = 0xA7E0_000C_0000;
/// Neighborhood nonparametric sensitivity confounder.
pub(crate) const NEIGHBORHOOD_SENSITIVITY: u64 = 0xA7E0_000D_0000;

/// Functional-stability 80% subsets. Slot 0 of [`crate::functional`] keeps this base.
pub(crate) const FUNCTIONAL_SUBSET_80: u64 = 0xF012_0000;
/// Functional-stability 50% subsets.
pub(crate) const FUNCTIONAL_SUBSET_50: u64 = 0xF013_0000;
/// Window-contrast contiguous 80% stretches.
pub(crate) const WINDOW_STRETCH_80: u64 = 0xF016_0000;
/// Window-contrast contiguous 50% stretches.
pub(crate) const WINDOW_STRETCH_50: u64 = 0xF017_0000;
/// Window-contrast treatment rotations.
pub(crate) const WINDOW_PLACEBO_SHIFT: u64 = 0xF018_0000;

/// Discovery null calibration.
pub(crate) const STABILITY_NULL: u64 = 0x5011;
/// False-positive calibration.
pub(crate) const STABILITY_FALSE_POSITIVE: u64 = 0xF41E;
/// Orientation stability.
pub(crate) const STABILITY_ORIENTATION: u64 = 0x0E1E;
/// Regime stability.
pub(crate) const STABILITY_REGIME: u64 = 0x5E61;
/// PCMCI grid.
pub(crate) const STABILITY_PCMCI: u64 = 0x57AB;

/// Observation noise for predictive replicates.
pub(crate) const REPLICATE_NOISE: u64 = 0x0B5E_4A7E_0001_D00D;
/// Serial posterior-predictive residual noise.
pub(crate) const SERIAL_PPC: u64 = 0x5E71_A1D0_C0DE_0001;
