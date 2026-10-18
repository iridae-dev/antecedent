//! RNG stream ids for refuters and stability checks.
//!
//! Each value is the historical stream. A rename does not change draws; a new
//! number would.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
