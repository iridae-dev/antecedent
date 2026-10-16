//! Lossless `usize` counts for artifact builders that do not return [`Result`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// `usize` as `u64` without saturating.
///
/// Supported targets have `usize` no wider than `u64`, so this is the word
/// `u64::try_from` would produce. A wider `usize` fails to compile instead of
/// aliasing `u64::MAX`.
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    reason = "usize::BITS <= 64 is asserted; the cast cannot truncate on a target this crate builds for"
)]
pub(crate) const fn lossless_u64(value: usize) -> u64 {
    const {
        assert!(usize::BITS <= 64, "refusing to narrow a usize that does not fit in u64");
    }
    value as u64
}

/// Little-endian IEEE-754 words. A trailing partial word is omitted.
///
/// Same copy as `antecedent_io_core`'s section decoder. It stays crate-private
/// so the public artifact API does not grow a second real codec.
#[must_use]
pub(crate) fn f64s_from_le_bytes(bytes: &[u8]) -> Vec<f64> {
    let mut values = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut word = [0_u8; 8];
        word.copy_from_slice(chunk);
        values.push(f64::from_le_bytes(word));
    }
    values
}

/// Little-endian IEEE-754 words, including the sign of zero and NaN payloads.
#[must_use]
pub(crate) fn f64s_to_le_bytes(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(8));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Next `len` values. A short slice panics, the same as indexing it.
pub(crate) fn split_reals<'a>(values: &mut &'a [f64], len: usize) -> &'a [f64] {
    let (head, tail) = values.split_at(len);
    *values = tail;
    head
}

/// `count` consecutive blocks of `width` values, in order.
///
/// A zero width yields `count` empty blocks and consumes nothing. Callers that
/// require `values` to hold exactly `count * width` words check that first.
pub(crate) fn take_blocks(values: &mut &[f64], width: usize, count: usize) -> Vec<Vec<f64>> {
    (0..count).map(|_| split_reals(values, width).to_vec()).collect()
}

/// Column-major little-endian words as `column_count` blocks of `rows`.
///
/// `None` when `rows * column_count * 8` is not the byte length, including
/// when that product does not fit in `usize`.
pub(crate) fn columns_from_le(
    numbers: &[u8],
    rows: usize,
    column_count: usize,
) -> Option<Vec<Vec<f64>>> {
    let expected = column_count.checked_mul(rows).and_then(|count| count.checked_mul(8))?;
    if expected != numbers.len() {
        return None;
    }
    let values = f64s_from_le_bytes(numbers);
    let mut rest = values.as_slice();
    Some(take_blocks(&mut rest, rows, column_count))
}

#[cfg(test)]
mod tests {
    use super::{columns_from_le, f64s_to_le_bytes};

    #[test]
    fn blocks_round_trip_keeps_signed_zero_and_nan() {
        let nan = f64::from_bits(0x7ff8_0000_0000_0001);
        let columns = [vec![-0.0, 1.0], vec![nan, -1.5], vec![0.0, f64::MIN_POSITIVE]];
        let flat: Vec<f64> = columns.iter().flatten().copied().collect();
        let bytes = f64s_to_le_bytes(&flat);
        let direct: Vec<u8> = flat.iter().flat_map(|value| value.to_le_bytes()).collect();
        assert_eq!(bytes, direct);
        let blocks = columns_from_le(&bytes, 2, 3).unwrap();
        let again: Vec<f64> = blocks.into_iter().flatten().collect();
        let bits = |sample: &[f64]| sample.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&again), bits(&flat));
        assert_eq!(f64s_to_le_bytes(&again), bytes);
        assert_ne!((-0.0_f64).to_bits(), 0.0_f64.to_bits());
        assert_eq!(again[0].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(again[2].to_bits(), nan.to_bits());
        assert_eq!(columns_from_le(&[], 0, 2), Some(vec![Vec::new(), Vec::new()]));
        assert!(columns_from_le(&bytes[..bytes.len() - 1], 2, 3).is_none());
    }
}
