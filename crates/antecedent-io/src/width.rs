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
