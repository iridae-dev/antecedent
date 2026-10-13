//! `SplitMix64` (Steele, Lea & Flood 2014): the crate's one owner of the seed
//! scrambler and the small deterministic streams used for seeding and fixtures.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// The mixer, its increment and the stepper are owned by `antecedent_core`; this module
// re-exports them under the crate-local path and keeps the seed scrambler and fixtures.
pub(crate) use antecedent_core::{GOLDEN_GAMMA, mix64, splitmix64};

/// Scramble a caller seed so nearby seeds start unrelated streams: the first
/// `SplitMix64` output of a generator whose state is `seed`.
#[must_use]
pub(crate) const fn seed_mix(seed: u64) -> u64 {
    mix64(seed.wrapping_add(GOLDEN_GAMMA))
}

/// Second Weyl multiplier of the per-index stream key: replicate and row indices of test
/// fixtures are combined as `(replicate + 1)·GOLDEN_GAMMA ^ (index + 1)·INDEX_GAMMA`.
pub(crate) const INDEX_GAMMA: u64 = 0xD1B5_4A32_D192_ED03;

/// Multiplier of the third stream coordinate of the test-fixture key.
#[cfg(test)]
pub(crate) const STREAM_GAMMA: u64 = 0xA076_1D64_78BD_642F;

/// A uniform draw in `[0, 1)` keyed by `(replicate, index)`, so every fixture row of every
/// replicate is a pure function of its coordinates (test fixtures).
#[cfg(test)]
#[must_use]
pub(crate) fn keyed_unit(replicate: u64, index: u64) -> f64 {
    keyed_unit_stream(replicate, index, 0)
}

/// [`keyed_unit`] with a third stream coordinate (zero reproduces [`keyed_unit`]).
#[cfg(test)]
#[must_use]
pub(crate) fn keyed_unit_stream(replicate: u64, index: u64, stream_coordinate: u64) -> f64 {
    unit_f64(mix64(
        replicate.wrapping_add(1).wrapping_mul(GOLDEN_GAMMA)
            ^ index.wrapping_add(1).wrapping_mul(INDEX_GAMMA)
            ^ stream_coordinate.wrapping_mul(STREAM_GAMMA),
    ))
}

/// A sequential `SplitMix64` stream of test fixtures.
#[cfg(test)]
pub(crate) struct FixtureRng(u64);

#[cfg(test)]
impl FixtureRng {
    /// A stream whose first output is `splitmix64` of `seed`.
    pub(crate) const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 random bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        splitmix64(&mut self.0)
    }

    /// The next uniform draw in `[0, 1)`.
    pub(crate) fn uniform(&mut self) -> f64 {
        unit_f64(self.next_u64())
    }
}

/// The top 53 bits of `bits` as a uniform draw in `[0, 1)` (test fixtures).
#[cfg(test)]
#[must_use]
#[allow(clippy::cast_precision_loss, reason = "a 53-bit integer is exact in f64")]
pub(crate) fn unit_f64(bits: u64) -> f64 {
    (bits >> 11) as f64 / (1_u64 << 53) as f64
}

/// Advance a `SplitMix64` state and return a uniform draw in `[0, 1)` (test fixtures).
#[cfg(test)]
pub(crate) fn splitmix64_unit(state: &mut u64) -> f64 {
    unit_f64(splitmix64(state))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference outputs of `SplitMix64` seeded at 0 (Vigna's `splitmix64.c`).
    #[test]
    fn matches_the_reference_stream() {
        let mut state = 0;
        assert_eq!(splitmix64(&mut state), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(&mut state), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(seed_mix(0), 0xE220_A839_7B1D_CDAF);
    }
}
