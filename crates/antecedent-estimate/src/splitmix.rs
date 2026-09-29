//! `SplitMix64` (Steele, Lea & Flood 2014): the crate's one owner of the seed
//! scrambler and the small deterministic streams used for seeding and fixtures.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// The `SplitMix64` increment (the golden-ratio gamma).
pub(crate) const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The `SplitMix64` output finalizer: a bijective avalanche mix of one word.
#[must_use]
pub(crate) const fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Scramble a caller seed so nearby seeds start unrelated streams: the first
/// `SplitMix64` output of a generator whose state is `seed`.
#[must_use]
pub(crate) const fn seed_mix(seed: u64) -> u64 {
    mix64(seed.wrapping_add(GOLDEN_GAMMA))
}

/// Advance a `SplitMix64` state and return its next output.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(GOLDEN_GAMMA);
    mix64(*state)
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
