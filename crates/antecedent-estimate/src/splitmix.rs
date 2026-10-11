//! `SplitMix64` (Steele, Lea & Flood 2014): the crate's one owner of the seed
//! scrambler and the small deterministic streams used for seeding and fixtures.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// The mixer, its increment and the stepper are owned by `antecedent_core`; this module
// re-exports them under the crate-local path and keeps the seed scrambler and fixtures.
pub(crate) use antecedent_core::{mix64, splitmix64, GOLDEN_GAMMA};

/// Scramble a caller seed so nearby seeds start unrelated streams: the first
/// `SplitMix64` output of a generator whose state is `seed`.
#[must_use]
pub(crate) const fn seed_mix(seed: u64) -> u64 {
    mix64(seed.wrapping_add(GOLDEN_GAMMA))
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
