//! Bit-packed row masks.
//!
//! One bit per row, least-significant bit first within each byte. The tabular data crate's
//! validity bitmap and the target-population keep-mask are both this type, so a selection
//! built by one is consumed by the other without a per-row conversion.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

/// Why a [`RowMask`] operation was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RowMaskError {
    /// The byte buffer holds fewer bits than the declared length.
    #[error("row mask buffer holds {actual} bytes but {needed} are needed")]
    BufferTooShort {
        /// Bytes required for the declared length.
        needed: usize,
        /// Bytes supplied.
        actual: usize,
    },
    /// A mapped row index is not a row of the mask.
    #[error("row {row} is outside a mask of {len} rows")]
    RowOutOfRange {
        /// Offending row.
        row: usize,
        /// Mask length.
        len: usize,
    },
    /// Two masks (or a mask and a column) disagree on the number of rows.
    #[error("row mask length {actual} does not match {expected}")]
    LengthMismatch {
        /// Required length.
        expected: usize,
        /// Length found.
        actual: usize,
    },
}

/// Packed set of rows (`1` = set), LSB-first.
///
/// Bits past [`Self::len`] in the last byte are padding and never contribute to a count or an
/// iteration. Equality compares the stored bytes, padding included.
#[derive(Clone, Debug, Default)]
pub struct RowMask {
    bytes: Arc<[u8]>,
    len: usize,
    /// Known all-set (established by [`Self::all_set`]); `false` means unknown.
    known_all_set: bool,
}

impl PartialEq for RowMask {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.bytes == other.bytes
    }
}

impl Eq for RowMask {}

impl RowMask {
    /// All-set mask of `len` bits.
    #[must_use]
    pub fn all_set(len: usize) -> Self {
        let n = len.div_ceil(8);
        Self { bytes: Arc::from(vec![0xFFu8; n].into_boxed_slice()), len, known_all_set: true }
    }

    /// Construct from raw bytes.
    ///
    /// # Errors
    ///
    /// [`RowMaskError::BufferTooShort`] when the buffer is shorter than `ceil(len / 8)`.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>, len: usize) -> Result<Self, RowMaskError> {
        let bytes = bytes.into();
        let needed = len.div_ceil(8);
        if bytes.len() < needed {
            return Err(RowMaskError::BufferTooShort { needed, actual: bytes.len() });
        }
        Ok(Self { bytes, len, known_all_set: false })
    }

    /// Pack one flag per row (`true` = set).
    #[must_use]
    pub fn from_bools(flags: &[bool]) -> Self {
        let mut bytes = vec![0u8; flags.len().div_ceil(8)];
        for (row, _) in flags.iter().enumerate().filter(|(_, flag)| **flag) {
            bytes[row / 8] |= 1 << (row % 8);
        }
        Self { bytes: Arc::from(bytes), len: flags.len(), known_all_set: false }
    }

    /// Mask of `len` rows whose bit is `keep(row)`.
    #[must_use]
    pub fn from_fn(len: usize, mut keep: impl FnMut(usize) -> bool) -> Self {
        let mut bytes = vec![0u8; len.div_ceil(8)];
        for row in 0..len {
            if keep(row) {
                bytes[row / 8] |= 1 << (row % 8);
            }
        }
        Self { bytes: Arc::from(bytes), len, known_all_set: false }
    }

    /// Mask of `len` rows with exactly the listed rows set (duplicates allowed).
    ///
    /// # Errors
    ///
    /// [`RowMaskError::RowOutOfRange`] for a row not below `len`.
    pub fn from_set_rows(
        len: usize,
        rows: impl IntoIterator<Item = usize>,
    ) -> Result<Self, RowMaskError> {
        let mut bytes = vec![0u8; len.div_ceil(8)];
        for row in rows {
            if row >= len {
                return Err(RowMaskError::RowOutOfRange { row, len });
            }
            bytes[row / 8] |= 1 << (row % 8);
        }
        Ok(Self { bytes: Arc::from(bytes), len, known_all_set: false })
    }

    /// Bit length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the mask has no rows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Stored bytes, least-significant bit first; bits past [`Self::len`] are padding.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Whether row `i` is set; `false` past the end.
    #[must_use]
    pub fn get(&self, i: usize) -> bool {
        i < self.len && (self.bytes[i / 8] >> (i % 8)) & 1 == 1
    }

    /// Whether every bit is set.
    #[must_use]
    pub fn is_all_set(&self) -> bool {
        if self.known_all_set {
            return true;
        }
        let full = self.len / 8;
        let tail = self.len % 8;
        let mut words = self.bytes[..full].chunks_exact(8);
        words.by_ref().all(|w| u64::from_le_bytes(w.try_into().unwrap_or([0; 8])) == u64::MAX)
            && words.remainder().iter().all(|&b| b == 0xFF)
            && (tail == 0 || {
                let want = (1u8 << tail) - 1;
                self.bytes[full] & want == want
            })
    }

    /// Gather bits through a row map (`out[i] = self[row_map[i]]`).
    ///
    /// # Errors
    ///
    /// [`RowMaskError::RowOutOfRange`] when a mapped row is out of range.
    pub fn gather(&self, row_map: &[u32]) -> Result<Self, RowMaskError> {
        self.gather_validated(row_map.len(), row_map.iter().map(|&r| r as usize))
    }

    /// Gather bits through a `usize` row map.
    ///
    /// # Errors
    ///
    /// [`RowMaskError::RowOutOfRange`] when a mapped row is out of range.
    pub fn gather_rows(&self, row_map: &[usize]) -> Result<Self, RowMaskError> {
        self.gather_validated(row_map.len(), row_map.iter().copied())
    }

    /// The one gather behind [`Self::gather`] and [`Self::gather_rows`].
    ///
    /// The map is range-checked in one pass before any bit is read, so the copy loop reads
    /// the stored bytes without a per-row bound test or error branch, and an all-set source
    /// writes its result without reading the map again.
    fn gather_validated(
        &self,
        n: usize,
        rows: impl Iterator<Item = usize> + Clone,
    ) -> Result<Self, RowMaskError> {
        if let Some(row) = rows.clone().find(|&r| r >= self.len) {
            return Err(RowMaskError::RowOutOfRange { row, len: self.len });
        }
        let mut bytes = vec![0u8; n.div_ceil(8)];
        if self.is_all_set() {
            let (full, tail) = (n / 8, n % 8);
            bytes[..full].fill(0xFF);
            if tail != 0 {
                bytes[full] = (1u8 << tail) - 1;
            }
        } else {
            for (i, r) in rows.enumerate() {
                if (self.bytes[r / 8] >> (r % 8)) & 1 == 1 {
                    bytes[i / 8] |= 1 << (i % 8);
                }
            }
        }
        Self::from_bytes(bytes, n)
    }

    /// Bits of stored byte `byte` that lie inside the mask (all eight for a full byte).
    fn used_bits_in_byte(&self, byte: usize) -> u8 {
        let start = byte * 8;
        if start + 8 <= self.len {
            0xFF
        } else {
            // Fewer than eight bits remain, so the shift is below eight.
            (1u8 << (self.len - start)) - 1
        }
    }

    /// Number of set rows.
    #[must_use]
    pub fn count_set(&self) -> usize {
        if self.known_all_set {
            return self.len;
        }
        (0..self.len.div_ceil(8))
            .map(|byte| (self.bytes[byte] & self.used_bits_in_byte(byte)).count_ones() as usize)
            .sum()
    }

    /// Rows whose bit equals `want`, ascending, found a byte at a time.
    fn rows_with_bit(&self, want: bool) -> impl Iterator<Item = usize> + '_ {
        (0..self.len.div_ceil(8)).flat_map(move |byte| {
            let stored = if want { self.bytes[byte] } else { !self.bytes[byte] };
            let mut bits = stored & self.used_bits_in_byte(byte);
            std::iter::from_fn(move || {
                if bits == 0 {
                    return None;
                }
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                Some(byte * 8 + bit)
            })
        })
    }

    /// Set rows in ascending order.
    pub fn iter_set(&self) -> impl Iterator<Item = usize> + '_ {
        self.rows_with_bit(true)
    }

    /// Unset rows in ascending order.
    pub fn iter_unset(&self) -> impl Iterator<Item = usize> + '_ {
        self.rows_with_bit(false)
    }

    /// One flag per row (`true` = set).
    #[must_use]
    pub fn to_bools(&self) -> Vec<bool> {
        let mut flags = vec![false; self.len];
        for row in self.iter_set() {
            flags[row] = true;
        }
        flags
    }

    /// Rows set in every one of `parts`, over `len` rows: a word-wise AND of the stored
    /// bytes, with the padding bits cleared. No parts is all-set.
    ///
    /// # Errors
    ///
    /// [`RowMaskError::LengthMismatch`] for a part whose length is not `len`.
    pub fn intersection(len: usize, parts: &[&Self]) -> Result<Self, RowMaskError> {
        if let Some(part) = parts.iter().find(|part| part.len != len) {
            return Err(RowMaskError::LengthMismatch { expected: len, actual: part.len });
        }
        let used = len.div_ceil(8);
        let mut bytes = vec![0xFFu8; used];
        for part in parts {
            let source = &part.bytes[..used];
            let mut dst = bytes.chunks_exact_mut(8);
            let mut src = source.chunks_exact(8);
            for (d, s) in dst.by_ref().zip(src.by_ref()) {
                let word = u64::from_le_bytes(d.try_into().unwrap_or([0; 8]))
                    & u64::from_le_bytes(s.try_into().unwrap_or([0; 8]));
                d.copy_from_slice(&word.to_le_bytes());
            }
            for (d, s) in dst.into_remainder().iter_mut().zip(src.remainder()) {
                *d &= *s;
            }
        }
        if len % 8 != 0 {
            bytes[used - 1] &= (1u8 << (len % 8)) - 1;
        }
        Self::from_bytes(bytes, len)
    }

    /// Compact to the rows `keep` marks set: the result has `keep.count_set()` rows and bit
    /// `j` is this mask's bit at the `j`-th set row of `keep`.
    ///
    /// # Errors
    ///
    /// [`RowMaskError::LengthMismatch`] when `keep` does not have this mask's length.
    pub fn compact(&self, keep: &Self) -> Result<Self, RowMaskError> {
        if keep.len != self.len {
            return Err(RowMaskError::LengthMismatch { expected: self.len, actual: keep.len });
        }
        let n_new = keep.count_set();
        let mut bytes = vec![0u8; n_new.div_ceil(8)];
        for (j, row) in keep.iter_set().enumerate() {
            if self.get(row) {
                bytes[j / 8] |= 1 << (j % 8);
            }
        }
        Self::from_bytes(bytes, n_new)
    }

    /// Concatenate masks end-to-end.
    ///
    /// A part that starts on a byte boundary is copied a byte at a time; others are placed
    /// bit by bit from their set rows.
    ///
    /// # Errors
    ///
    /// Propagates construction errors (none occur for well-formed parts).
    pub fn concat(parts: &[&Self]) -> Result<Self, RowMaskError> {
        let n: usize = parts.iter().map(|p| p.len).sum();
        let mut bytes = vec![0u8; n.div_ceil(8)];
        let mut offset = 0usize;
        for part in parts {
            if offset % 8 == 0 {
                let used = part.len.div_ceil(8);
                let start = offset / 8;
                for (byte, dst) in bytes[start..start + used].iter_mut().enumerate() {
                    *dst = part.bytes[byte] & part.used_bits_in_byte(byte);
                }
            } else {
                for i in part.iter_set() {
                    let j = offset + i;
                    bytes[j / 8] |= 1 << (j % 8);
                }
            }
            offset += part.len;
        }
        Self::from_bytes(bytes, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_and_unpacks_flags() {
        let flags = [true, false, true, true, false, false, false, false, true, true];
        let mask = RowMask::from_bools(&flags);
        assert_eq!(mask.len(), 10);
        assert_eq!(mask.count_set(), 5);
        assert_eq!(mask.to_bools(), flags);
        assert_eq!(mask.iter_set().collect::<Vec<_>>(), vec![0, 2, 3, 8, 9]);
        assert_eq!(mask.iter_unset().collect::<Vec<_>>(), vec![1, 4, 5, 6, 7]);
        assert!(mask.get(2) && !mask.get(1) && !mask.get(10));
    }

    #[test]
    fn builders_set_exactly_the_requested_rows() {
        let evens = RowMask::from_fn(11, |row| row % 2 == 0);
        assert_eq!(evens.iter_set().collect::<Vec<_>>(), vec![0, 2, 4, 6, 8, 10]);
        let listed = RowMask::from_set_rows(11, [3, 3, 10]).unwrap();
        assert_eq!(listed.iter_set().collect::<Vec<_>>(), vec![3, 10]);
        assert!(matches!(
            RowMask::from_set_rows(4, [1, 4]),
            Err(RowMaskError::RowOutOfRange { row: 4, len: 4 })
        ));
    }

    #[test]
    fn padding_bits_are_ignored() {
        let mask = RowMask::from_bytes(vec![0b1111_0111_u8], 3).unwrap();
        assert!(mask.is_all_set());
        assert_eq!(mask.count_set(), 3);
        assert_eq!(mask.iter_unset().count(), 0);
        assert!(RowMask::all_set(0).is_all_set());
        assert!(!RowMask::from_bytes(vec![0xFF, 0x7F], 16).unwrap().is_all_set());
        assert!(matches!(
            RowMask::from_bytes(vec![0u8], 9),
            Err(RowMaskError::BufferTooShort { needed: 2, actual: 1 })
        ));
    }

    #[test]
    fn gather_intersect_compact_and_concat_agree_with_flags() {
        let a = RowMask::from_bools(&[true, true, false, true, false, true, true, false, true]);
        let b = RowMask::from_bools(&[true, false, false, true, true, true, false, false, true]);
        let both = RowMask::intersection(9, &[&a, &b]).unwrap();
        assert_eq!(
            both.to_bools(),
            vec![true, false, false, true, false, true, false, false, true]
        );
        assert_eq!(
            a.gather_rows(&[8, 0, 2]).unwrap().to_bools(),
            vec![true, true, false]
        );
        assert!(matches!(
            a.gather(&[9]),
            Err(RowMaskError::RowOutOfRange { row: 9, len: 9 })
        ));
        assert_eq!(a.compact(&b).unwrap().to_bools(), vec![true, true, false, true, true]);
        let joined = RowMask::concat(&[&a, &b]).unwrap();
        assert_eq!(joined.len(), 18);
        let mut expected = a.to_bools();
        expected.extend(b.to_bools());
        assert_eq!(joined.to_bools(), expected);
        assert!(matches!(
            RowMask::intersection(8, &[&a]),
            Err(RowMaskError::LengthMismatch { expected: 8, actual: 9 })
        ));
    }
}
