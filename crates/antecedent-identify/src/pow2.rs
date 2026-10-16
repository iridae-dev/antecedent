//! Power-of-two sizes that refuse a shift past the word width.
//!
//! `1 << exp` panics when `exp` is at least the width of the integer. Enumeration
//! sized from a graph or a query goes through these helpers so an oversized
//! exponent is `None` instead of a wrapped or panicking shift. An exponent that
//! fits produces the same value as the shift.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// `2^exp` as `usize`, or `None` when the shift does not fit in `usize`.
#[must_use]
pub(crate) fn checked_pow2(exp: usize) -> Option<usize> {
    u32::try_from(exp).ok().and_then(|exp| 1usize.checked_shl(exp))
}

/// `2^exp` as `u64`, or `None` when the shift does not fit in `u64`.
#[must_use]
pub(crate) fn checked_pow2_u64(exp: usize) -> Option<u64> {
    u32::try_from(exp).ok().and_then(|exp| 1u64.checked_shl(exp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_range_exponents_match_the_shift() {
        for exp in [0usize, 1, 12, 20, 31] {
            assert_eq!(checked_pow2(exp), Some(1usize << exp));
            assert_eq!(checked_pow2_u64(exp), Some(1u64 << exp));
        }
        assert_eq!(checked_pow2(usize::BITS as usize - 1), Some(1usize << (usize::BITS - 1)));
        assert_eq!(checked_pow2_u64(63), Some(1u64 << 63));
    }

    #[test]
    fn a_shift_past_the_word_is_none() {
        assert_eq!(checked_pow2(usize::BITS as usize), None);
        assert_eq!(checked_pow2(usize::MAX), None);
        assert_eq!(checked_pow2_u64(64), None);
        assert_eq!(checked_pow2_u64(128), None);
    }
}
