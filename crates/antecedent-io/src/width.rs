//! Lossless `usize` counts and little-endian word blocks for artifact builders that do not
//! return [`Result`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// The word conversions are the ones `antecedent_io_core` owns; there is one copy.
pub(crate) use antecedent_io_core::artifact_codec::{
    f64s_from_le_bytes, f64s_to_le_bytes, split_reals,
};
pub(crate) use antecedent_io_core::error::lossless_u64;

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
