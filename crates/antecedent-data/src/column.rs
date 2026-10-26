//! Columnar storage and typed column views.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_kernels::{BitMaskView, F64VectorView};

use crate::buffer::F64Buffer;
use crate::categorical::CategoricalColumn;
use crate::error::DataError;

/// Packed validity bitmap (`1` = valid), LSB-first.
///
/// The bitmap is authoritative for missingness. [`Float64Column`] keeps it in
/// lockstep with its values: a row is invalid exactly when it holds `NaN`, so
/// neither value-only nor bitmap-only readers can mistake a missing cell for an
/// observation.
#[derive(Clone, Debug, Default)]
pub struct ValidityBitmap {
    bytes: Arc<[u8]>,
    len: usize,
    /// Known all-valid (set by [`Self::all_valid`]); `false` means unknown.
    known_all_valid: bool,
}

impl PartialEq for ValidityBitmap {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len && self.bytes == other.bytes
    }
}

impl Eq for ValidityBitmap {}

impl ValidityBitmap {
    /// All-valid bitmap of `len` bits.
    #[must_use]
    pub fn all_valid(len: usize) -> Self {
        let n = len.div_ceil(8);
        Self { bytes: Arc::from(vec![0xFFu8; n].into_boxed_slice()), len, known_all_valid: true }
    }

    /// Construct from raw bytes.
    ///
    /// # Errors
    ///
    /// When the buffer is shorter than `ceil(len/8)`.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>, len: usize) -> Result<Self, DataError> {
        let bytes = bytes.into();
        if bytes.len() < len.div_ceil(8) {
            return Err(DataError::InvalidValidity { message: "validity buffer too short" });
        }
        Ok(Self { bytes, len, known_all_valid: false })
    }

    /// Bit length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Stored bytes, least-significant bit first; bits past [`Self::len`] are padding.
    pub(crate) fn raw_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Whether empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow as a kernel mask view.
    ///
    /// # Errors
    ///
    /// Propagates view construction errors.
    pub fn as_mask_view(&self) -> Result<BitMaskView<'_>, DataError> {
        BitMaskView::new(&self.bytes, self.len)
            .map_err(|_| DataError::InvalidValidity { message: "mask view rejected buffer" })
    }

    /// Whether row `i` is valid.
    #[must_use]
    pub fn is_valid(&self, i: usize) -> bool {
        self.as_mask_view().is_ok_and(|m| m.get(i))
    }

    /// Whether every bit is valid.
    #[must_use]
    pub fn is_all_valid(&self) -> bool {
        if self.known_all_valid {
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
    /// When a mapped row is out of range.
    pub fn gather(&self, row_map: &[u32]) -> Result<Self, DataError> {
        self.gather_validated(row_map.len(), row_map.iter().map(|&r| r as usize))
    }

    /// Gather bits through a `usize` row map.
    ///
    /// # Errors
    ///
    /// When a mapped row is out of range.
    pub fn gather_rows(&self, row_map: &[usize]) -> Result<Self, DataError> {
        self.gather_validated(row_map.len(), row_map.iter().copied())
    }

    /// The one gather behind [`Self::gather`] and [`Self::gather_rows`].
    ///
    /// The map is range-checked in one pass before any bit is read, so the copy loop reads
    /// the stored bytes without a per-row bound test or error branch, and an all-valid source
    /// writes its result without reading the map again.
    fn gather_validated(
        &self,
        n: usize,
        rows: impl Iterator<Item = usize> + Clone,
    ) -> Result<Self, DataError> {
        if rows.clone().any(|r| r >= self.len) {
            return Err(DataError::InvalidValidity { message: "row map exceeds bitmap" });
        }
        let mut bytes = vec![0u8; n.div_ceil(8)];
        if self.is_all_valid() {
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

    /// Bits of the last stored byte that lie inside the bitmap (all eight for a full byte).
    fn used_bits_in_byte(&self, byte: usize) -> u8 {
        let start = byte * 8;
        if start + 8 <= self.len {
            0xFF
        } else {
            // Fewer than eight bits remain, so the shift is below eight.
            (1u8 << (self.len - start)) - 1
        }
    }

    /// Number of valid rows.
    #[must_use]
    pub fn count_valid(&self) -> usize {
        if self.known_all_valid {
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

    /// Valid rows in ascending order.
    pub fn iter_valid(&self) -> impl Iterator<Item = usize> + '_ {
        self.rows_with_bit(true)
    }

    /// Invalid rows in ascending order.
    pub fn iter_invalid(&self) -> impl Iterator<Item = usize> + '_ {
        self.rows_with_bit(false)
    }

    /// Pack one flag per row (`true` = valid).
    #[must_use]
    pub fn from_bools(flags: &[bool]) -> Self {
        let mut bytes = vec![0u8; flags.len().div_ceil(8)];
        for (row, _) in flags.iter().enumerate().filter(|(_, flag)| **flag) {
            bytes[row / 8] |= 1 << (row % 8);
        }
        Self { bytes: Arc::from(bytes), len: flags.len(), known_all_valid: false }
    }

    /// One flag per row (`true` = valid).
    #[must_use]
    pub fn to_bools(&self) -> Vec<bool> {
        let mut flags = vec![false; self.len];
        for row in self.iter_valid() {
            flags[row] = true;
        }
        flags
    }

    /// Rows valid in every one of `parts`, over `len` rows: a word-wise AND of the stored
    /// bytes, with the padding bits cleared. No parts is all-valid.
    ///
    /// # Errors
    ///
    /// A part whose length is not `len`.
    pub fn intersection(len: usize, parts: &[&Self]) -> Result<Self, DataError> {
        if let Some(part) = parts.iter().find(|part| part.len != len) {
            return Err(DataError::LengthMismatch {
                expected: len,
                actual: part.len,
                context: "validity intersection",
            });
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

    /// Compact to the rows `keep` marks valid.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn compact(&self, keep: &Self) -> Result<Self, DataError> {
        if keep.len != self.len {
            return Err(DataError::LengthMismatch {
                expected: self.len,
                actual: keep.len,
                context: "validity compact keep",
            });
        }
        let n_new = keep.count_valid();
        let mut bytes = vec![0u8; n_new.div_ceil(8)];
        for (j, row) in keep.iter_valid().enumerate() {
            if self.is_valid(row) {
                bytes[j / 8] |= 1 << (j % 8);
            }
        }
        Self::from_bytes(bytes, n_new)
    }

    /// Concatenate bitmaps end-to-end.
    ///
    /// A part that starts on a byte boundary is copied a byte at a time; others are placed
    /// bit by bit from their valid rows.
    ///
    /// # Errors
    ///
    /// Propagates bitmap construction errors.
    pub fn concat(parts: &[&Self]) -> Result<Self, DataError> {
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
                for i in part.iter_valid() {
                    let j = offset + i;
                    bytes[j / 8] |= 1 << (j % 8);
                }
            }
            offset += part.len;
        }
        Self::from_bytes(bytes, n)
    }
}

/// Float64 column (owned or foreign-backed values).
///
/// Invariant: a row is invalid exactly when its value is `NaN`. `NaN` is the
/// missing-value sentinel for `f64` data (as it is for `NumPy` input), never an
/// observation. Borrowed readers ([`crate::TableView::float64_slice`]) hand the
/// buffer out without consulting validity, so a missing cell must never read
/// as a finite value; and bitmap readers (complete-case selection) must never
/// keep a `NaN` row as observed.
#[derive(Clone, Debug, PartialEq)]
pub struct Float64Column {
    /// Variable id.
    pub id: VariableId,
    /// Values; `NaN` exactly under the invalid rows.
    pub values: F64Buffer,
    /// Validity bitmap (authoritative for missingness).
    pub validity: ValidityBitmap,
}

impl Float64Column {
    /// Construct a column; lengths must match.
    ///
    /// The stored validity is `validity` with every `NaN` row cleared, and any
    /// invalid row whose value is not already `NaN` is rewritten to `NaN`
    /// (copying the buffer, including a foreign one, only when a rewrite is
    /// needed), so invalid ⇔ `NaN` holds for every construction path. When the
    /// values hold no `NaN` the bitmap is kept as given (no allocation).
    ///
    /// The invariant is established in one pass over the values. A buffer a
    /// previous scan found `NaN`-free ([`F64Buffer::is_nan_free`], carried by
    /// every clone of a constructed column's `values`) is not rescanned: with an
    /// all-valid bitmap the column is assembled without touching the values.
    ///
    /// # Errors
    ///
    /// [`DataError::LengthMismatch`] when validity length differs.
    pub fn new(
        id: VariableId,
        values: impl Into<F64Buffer>,
        validity: ValidityBitmap,
    ) -> Result<Self, DataError> {
        let mut values = values.into();
        if validity.len() != values.len() {
            return Err(DataError::LengthMismatch {
                expected: values.len(),
                actual: validity.len(),
                context: "float64 validity",
            });
        }
        let all_valid = validity.is_all_valid();
        if values.is_nan_free() {
            // No `NaN` anywhere: the bitmap has nothing to clear, and only the
            // invalid rows (if any) need `NaN` written under them.
            if !all_valid {
                values = write_nan_under_invalid(&values, &validity)?;
            }
            return Ok(Self { id, values, validity });
        }
        if all_valid {
            if contains_nan(values.as_slice()) {
                // Clearing the `NaN` rows leaves every invalid row holding `NaN`.
                let validity = clear_nan_rows(values.as_slice(), &validity)?;
                return Ok(Self { id, values, validity });
            }
            values.mark_nan_free();
            return Ok(Self { id, values, validity });
        }
        // Rows may be invalid: one fused pass answers both questions.
        let scan = scan_against_validity(values.as_slice(), &validity);
        let validity = if scan.nan_under_valid {
            clear_nan_rows(values.as_slice(), &validity)?
        } else {
            validity
        };
        if scan.observed_under_invalid {
            values = write_nan_under_invalid(&values, &validity)?;
        }
        Ok(Self { id, values, validity })
    }

    /// Row count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Borrowed contiguous view (no allocation).
    #[must_use]
    pub fn as_f64_view(&self) -> F64VectorView<'_> {
        F64VectorView::contiguous(self.values.as_slice())
    }
}

/// Whether every invalid row of `values` already holds `NaN`. Only the Arrow
/// adapter needs it, so it is compiled with the `arrow` feature.
#[cfg(feature = "arrow")]
pub(crate) fn invalid_rows_are_nan(values: &[f64], validity: &ValidityBitmap) -> bool {
    if validity.is_all_valid() {
        return true;
    }
    validity
        .as_mask_view()
        .is_ok_and(|mask| values.iter().enumerate().all(|(i, v)| mask.get(i) || v.is_nan()))
}

/// Whether any value is `NaN` (branch-free per chunk so the common all-finite
/// scan vectorizes; stops at the first chunk holding a `NaN`).
fn contains_nan(values: &[f64]) -> bool {
    values.chunks(64).any(|chunk| chunk.iter().fold(false, |acc, v| acc | v.is_nan()))
}

/// What one pass over `values` against a (not all-valid) bitmap found.
#[derive(Clone, Copy, Debug, Default)]
struct ValidityScan {
    /// Some valid row holds `NaN` (its bit must be cleared).
    nan_under_valid: bool,
    /// Some invalid row holds a non-`NaN` value (it must be rewritten to `NaN`).
    observed_under_invalid: bool,
}

/// One fused pass: walks the bitmap a byte (eight rows) at a time, accumulating
/// branch-free within a byte so the loop vectorizes, and stops early once both
/// answers are known.
fn scan_against_validity(values: &[f64], validity: &ValidityBitmap) -> ValidityScan {
    // Byte-wise flags (`1` = true) keep the inner loop free of short-circuits.
    fn flags(byte: u8, rows: &[f64]) -> (u8, u8) {
        let (mut nan_under_valid, mut observed_under_invalid) = (0u8, 0u8);
        for (bit, v) in rows.iter().enumerate() {
            let valid = (byte >> bit) & 1;
            let nan = u8::from(v.is_nan());
            nan_under_valid |= valid & nan;
            observed_under_invalid |= (1 - valid) & (1 - nan);
        }
        (nan_under_valid, observed_under_invalid)
    }
    let n = values.len();
    let mut scan = ValidityScan::default();
    let mut rows = values.chunks_exact(8);
    for (&byte, chunk) in validity.bytes.iter().zip(rows.by_ref()) {
        let (nan_under_valid, observed_under_invalid) = flags(byte, chunk);
        scan.nan_under_valid |= nan_under_valid == 1;
        scan.observed_under_invalid |= observed_under_invalid == 1;
        if scan.nan_under_valid && scan.observed_under_invalid {
            return scan;
        }
    }
    let tail = rows.remainder();
    if !tail.is_empty() {
        let (nan_under_valid, observed_under_invalid) = flags(validity.bytes[n / 8], tail);
        scan.nan_under_valid |= nan_under_valid == 1;
        scan.observed_under_invalid |= observed_under_invalid == 1;
    }
    scan
}

/// Clear the validity bit of every `NaN` row (copies the bitmap).
fn clear_nan_rows(values: &[f64], validity: &ValidityBitmap) -> Result<ValidityBitmap, DataError> {
    let mut bytes = validity.bytes[..values.len().div_ceil(8)].to_vec();
    for (i, v) in values.iter().enumerate() {
        if v.is_nan() {
            bytes[i / 8] &= !(1 << (i % 8));
        }
    }
    ValidityBitmap::from_bytes(bytes, values.len())
}

/// Copy `values` with `NaN` written under every invalid row.
fn write_nan_under_invalid(
    values: &F64Buffer,
    validity: &ValidityBitmap,
) -> Result<F64Buffer, DataError> {
    let mask = validity.as_mask_view()?;
    let normalized: Vec<f64> = values
        .as_slice()
        .iter()
        .enumerate()
        .map(|(i, &v)| if mask.get(i) { v } else { f64::NAN })
        .collect();
    Ok(F64Buffer::owned(Arc::<[f64]>::from(normalized)))
}

/// Owned int64 column.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Int64Column {
    /// Variable id.
    pub id: VariableId,
    /// Values.
    pub values: Arc<[i64]>,
    /// Validity.
    pub validity: ValidityBitmap,
}

impl Int64Column {
    /// Construct with matching lengths.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn new(
        id: VariableId,
        values: impl Into<Arc<[i64]>>,
        validity: ValidityBitmap,
    ) -> Result<Self, DataError> {
        let values = values.into();
        if validity.len() != values.len() {
            return Err(DataError::LengthMismatch {
                expected: values.len(),
                actual: validity.len(),
                context: "int64 validity",
            });
        }
        Ok(Self { id, values, validity })
    }

    /// Row count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Owned boolean column (bytes: 0/1 per row for simplicity).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BooleanColumn {
    /// Variable id.
    pub id: VariableId,
    /// Values as 0/1 bytes.
    pub values: Arc<[u8]>,
    /// Validity.
    pub validity: ValidityBitmap,
}

impl BooleanColumn {
    /// Construct with matching lengths.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn new(
        id: VariableId,
        values: impl Into<Arc<[u8]>>,
        validity: ValidityBitmap,
    ) -> Result<Self, DataError> {
        let values = values.into();
        if validity.len() != values.len() {
            return Err(DataError::LengthMismatch {
                expected: values.len(),
                actual: validity.len(),
                context: "bool validity",
            });
        }
        Ok(Self { id, values, validity })
    }

    /// Row count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Owned timestamp column (nanoseconds since epoch; timezone metadata lives in schema).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TimestampColumn {
    /// Variable id.
    pub id: VariableId,
    /// Values in nanoseconds.
    pub values_ns: Arc<[i64]>,
    /// Validity.
    pub validity: ValidityBitmap,
}

impl TimestampColumn {
    /// Construct with matching lengths.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn new(
        id: VariableId,
        values_ns: impl Into<Arc<[i64]>>,
        validity: ValidityBitmap,
    ) -> Result<Self, DataError> {
        let values_ns = values_ns.into();
        if validity.len() != values_ns.len() {
            return Err(DataError::LengthMismatch {
                expected: values_ns.len(),
                actual: validity.len(),
                context: "timestamp validity",
            });
        }
        Ok(Self { id, values_ns, validity })
    }

    /// Row count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values_ns.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values_ns.is_empty()
    }
}

/// Owned fixed-size vector column (row-major: `values[row * dim + component]`).
#[derive(Clone, Debug, PartialEq)]
pub struct FixedVectorColumn {
    /// Variable id.
    pub id: VariableId,
    /// Vector dimensionality.
    pub dim: usize,
    /// Flattened values.
    pub values: Arc<[f64]>,
    /// Per-row validity.
    pub validity: ValidityBitmap,
}

impl FixedVectorColumn {
    /// Construct a fixed-vector column.
    ///
    /// # Errors
    ///
    /// Length / shape mismatch.
    pub fn new(
        id: VariableId,
        dim: usize,
        values: impl Into<Arc<[f64]>>,
        validity: ValidityBitmap,
    ) -> Result<Self, DataError> {
        if dim == 0 {
            return Err(DataError::InvalidValidity { message: "fixed vector dim must be > 0" });
        }
        let values = values.into();
        let expected = validity
            .len()
            .checked_mul(dim)
            .ok_or(DataError::InvalidValidity { message: "fixed vector shape overflow" })?;
        if values.len() != expected {
            return Err(DataError::LengthMismatch {
                expected,
                actual: values.len(),
                context: "fixed vector values",
            });
        }
        Ok(Self { id, dim, values, validity })
    }

    /// Row count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.validity.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Borrowed typed column view (library-owned; not Arrow types).
#[derive(Clone, Copy, Debug)]
pub enum ColumnView<'a> {
    /// Float64 column.
    Float64(&'a Float64Column),
    /// Int64 column.
    Int64(&'a Int64Column),
    /// Boolean column.
    Boolean(&'a BooleanColumn),
    /// Dictionary categorical column.
    Categorical(&'a CategoricalColumn),
    /// Timestamp column.
    Timestamp(&'a TimestampColumn),
    /// Fixed-size vector column.
    FixedVector(&'a FixedVectorColumn),
}

impl<'a> ColumnView<'a> {
    /// Variable id.
    #[must_use]
    pub fn id(self) -> VariableId {
        match self {
            Self::Float64(c) => c.id,
            Self::Int64(c) => c.id,
            Self::Boolean(c) => c.id,
            Self::Categorical(c) => c.id,
            Self::Timestamp(c) => c.id,
            Self::FixedVector(c) => c.id,
        }
    }

    /// Row count.
    #[must_use]
    pub fn len(self) -> usize {
        match self {
            Self::Float64(c) => c.len(),
            Self::Int64(c) => c.len(),
            Self::Boolean(c) => c.len(),
            Self::Categorical(c) => c.len(),
            Self::Timestamp(c) => c.len(),
            Self::FixedVector(c) => c.len(),
        }
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Borrow the column validity bitmap.
    #[must_use]
    pub fn validity(self) -> &'a ValidityBitmap {
        match self {
            Self::Float64(c) => &c.validity,
            Self::Int64(c) => &c.validity,
            Self::Boolean(c) => &c.validity,
            Self::Categorical(c) => &c.validity,
            Self::Timestamp(c) => &c.validity,
            Self::FixedVector(c) => &c.validity,
        }
    }
}

/// Owned column enum stored in a table.
#[derive(Clone, Debug)]
pub enum OwnedColumn {
    /// Float64.
    Float64(Float64Column),
    /// Int64.
    Int64(Int64Column),
    /// Boolean.
    Boolean(BooleanColumn),
    /// Categorical.
    Categorical(CategoricalColumn),
    /// Timestamp.
    Timestamp(TimestampColumn),
    /// Fixed-size vector.
    FixedVector(FixedVectorColumn),
}

impl ColumnView<'_> {
    /// An owned column sharing this view's buffers (no value is copied).
    #[must_use]
    pub fn share(self) -> OwnedColumn {
        match self {
            Self::Float64(c) => OwnedColumn::Float64(c.clone()),
            Self::Int64(c) => OwnedColumn::Int64(c.clone()),
            Self::Boolean(c) => OwnedColumn::Boolean(c.clone()),
            Self::Categorical(c) => OwnedColumn::Categorical(c.clone()),
            Self::Timestamp(c) => OwnedColumn::Timestamp(c.clone()),
            Self::FixedVector(c) => OwnedColumn::FixedVector(c.clone()),
        }
    }

    /// A new column of the same type whose row `i` is this column's row `rows[i]`.
    ///
    /// Each replicate slot inherits the source row's validity, so a value under a null slot
    /// never resurfaces as an observation. A `Float64` gather from a `NaN`-free column is
    /// `NaN`-free and skips the rescan. A categorical gather keeps the source domain and its
    /// remapped-unknown provenance count.
    ///
    /// # Errors
    ///
    /// When a mapped row is out of range.
    pub fn gather_rows(self, rows: &[usize]) -> Result<OwnedColumn, DataError> {
        if rows.iter().any(|&r| r >= self.len()) {
            return Err(DataError::InvalidValidity { message: "row map exceeds column" });
        }
        let validity = self.validity().gather_rows(rows)?;
        Ok(match self {
            Self::Float64(c) => {
                let values: Vec<f64> = rows.iter().map(|&r| c.values[r]).collect();
                let mut values = F64Buffer::owned(Arc::<[f64]>::from(values));
                // Rows gathered from a NaN-free column are NaN-free: skip the rescan.
                if c.values.is_nan_free() {
                    values.mark_nan_free();
                }
                OwnedColumn::Float64(Float64Column::new(c.id, values, validity)?)
            }
            Self::Int64(c) => OwnedColumn::Int64(Int64Column::new(
                c.id,
                rows.iter().map(|&r| c.values[r]).collect::<Vec<i64>>(),
                validity,
            )?),
            Self::Boolean(c) => OwnedColumn::Boolean(BooleanColumn::new(
                c.id,
                rows.iter().map(|&r| c.values[r]).collect::<Vec<u8>>(),
                validity,
            )?),
            Self::Categorical(c) => OwnedColumn::Categorical(CategoricalColumn {
                id: c.id,
                codes: rows.iter().map(|&r| c.codes[r]).collect::<Vec<_>>().into(),
                validity,
                domain: Arc::clone(&c.domain),
                n_remapped_unknown: c.n_remapped_unknown,
            }),
            Self::Timestamp(c) => OwnedColumn::Timestamp(TimestampColumn::new(
                c.id,
                rows.iter().map(|&r| c.values_ns[r]).collect::<Vec<i64>>(),
                validity,
            )?),
            Self::FixedVector(c) => {
                let mut values = Vec::with_capacity(rows.len() * c.dim);
                for &r in rows {
                    values.extend_from_slice(&c.values[r * c.dim..(r + 1) * c.dim]);
                }
                OwnedColumn::FixedVector(FixedVectorColumn::new(c.id, c.dim, values, validity)?)
            }
        })
    }
}

impl OwnedColumn {
    /// Variable id.
    #[must_use]
    pub fn id(&self) -> VariableId {
        self.as_view().id()
    }

    /// Clone the column with a remapped dense id (value buffers stay shared).
    #[must_use]
    pub fn with_id(&self, id: VariableId) -> Self {
        match self {
            Self::Float64(c) => Self::Float64(Float64Column {
                id,
                values: c.values.clone(),
                validity: c.validity.clone(),
            }),
            Self::Int64(c) => Self::Int64(Int64Column {
                id,
                values: Arc::clone(&c.values),
                validity: c.validity.clone(),
            }),
            Self::Boolean(c) => Self::Boolean(BooleanColumn {
                id,
                values: Arc::clone(&c.values),
                validity: c.validity.clone(),
            }),
            Self::Categorical(c) => Self::Categorical(CategoricalColumn {
                id,
                codes: Arc::clone(&c.codes),
                validity: c.validity.clone(),
                domain: Arc::clone(&c.domain),
                n_remapped_unknown: c.n_remapped_unknown,
            }),
            Self::Timestamp(c) => Self::Timestamp(TimestampColumn {
                id,
                values_ns: Arc::clone(&c.values_ns),
                validity: c.validity.clone(),
            }),
            Self::FixedVector(c) => Self::FixedVector(FixedVectorColumn {
                id,
                values: Arc::clone(&c.values),
                dim: c.dim,
                validity: c.validity.clone(),
            }),
        }
    }

    /// Row count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_view().len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow as a [`ColumnView`].
    #[must_use]
    pub fn as_view(&self) -> ColumnView<'_> {
        match self {
            Self::Float64(c) => ColumnView::Float64(c),
            Self::Int64(c) => ColumnView::Int64(c),
            Self::Boolean(c) => ColumnView::Boolean(c),
            Self::Categorical(c) => ColumnView::Categorical(c),
            Self::Timestamp(c) => ColumnView::Timestamp(c),
            Self::FixedVector(c) => ColumnView::FixedVector(c),
        }
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss, reason = "values are copied bit-for-bit")]
mod tests {
    use super::*;

    #[test]
    fn gather_rows_keeps_each_column_type_and_its_validity() {
        let validity = ValidityBitmap::from_bytes(vec![0b101], 3).unwrap();
        let col = Int64Column::new(VariableId::from_raw(0), vec![10, 20, 30], validity).unwrap();
        let OwnedColumn::Int64(out) = ColumnView::Int64(&col).gather_rows(&[2, 2, 1, 0]).unwrap()
        else {
            panic!("an int64 column gathers to an int64 column");
        };
        assert_eq!(&*out.values, &[30, 30, 20, 10]);
        let valid: Vec<bool> = (0..4).map(|i| out.validity.is_valid(i)).collect();
        assert_eq!(valid, [true, true, false, true]);
        assert!(ColumnView::Int64(&col).gather_rows(&[3]).is_err());

        let vectors = FixedVectorColumn::new(
            VariableId::from_raw(1),
            2,
            vec![1.0, 2.0, 3.0, 4.0],
            ValidityBitmap::all_valid(2),
        )
        .unwrap();
        let OwnedColumn::FixedVector(out) =
            ColumnView::FixedVector(&vectors).gather_rows(&[1, 0, 1]).unwrap()
        else {
            panic!("a vector column gathers to a vector column");
        };
        assert_eq!(&*out.values, &[3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(out.validity.is_all_valid());
    }

    #[test]
    fn is_all_valid_ignores_padding_bits() {
        assert!(ValidityBitmap::from_bytes(vec![0b0000_0111], 3).unwrap().is_all_valid());
        assert!(!ValidityBitmap::from_bytes(vec![0b1111_1011], 3).unwrap().is_all_valid());
        assert!(ValidityBitmap::from_bytes(vec![0xFF, 0x7F], 15).unwrap().is_all_valid());
        assert!(!ValidityBitmap::from_bytes(vec![0xFF, 0x7F], 16).unwrap().is_all_valid());
        assert!(!ValidityBitmap::from_bytes(vec![0xFE, 0xFF], 16).unwrap().is_all_valid());
        assert!(ValidityBitmap::all_valid(0).is_all_valid());
        assert!(ValidityBitmap::all_valid(17).is_all_valid());
    }

    #[test]
    fn float64_column_writes_nan_under_invalid_rows() {
        let validity = ValidityBitmap::from_bytes(vec![0b101], 3).unwrap();
        let col =
            Float64Column::new(VariableId::from_raw(0), vec![1.0, 0.0, 3.0], validity).unwrap();
        assert_eq!(col.values[0], 1.0);
        assert!(col.values[1].is_nan());
        assert_eq!(col.values[2], 3.0);
    }

    #[test]
    fn float64_column_keeps_buffer_when_invariant_holds() {
        let values: Arc<[f64]> = Arc::from(vec![1.0, f64::NAN, 3.0]);
        let ptr = values.as_ptr();
        let validity = ValidityBitmap::from_bytes(vec![0b101], 3).unwrap();
        let col =
            Float64Column::new(VariableId::from_raw(0), Arc::clone(&values), validity).unwrap();
        assert_eq!(col.values.as_slice().as_ptr(), ptr, "no copy when nulls already hold NaN");

        // Without NaN values the given bitmap is kept as is.
        let validity = ValidityBitmap::all_valid(3);
        let bits = validity.bytes.as_ptr();
        let col =
            Float64Column::new(VariableId::from_raw(0), vec![1.0, 2.0, 3.0], validity).unwrap();
        assert_eq!(col.validity.bytes.as_ptr(), bits, "no bitmap copy without NaN values");
    }

    #[test]
    fn float64_column_marks_nan_rows_invalid() {
        // A NaN under a valid bit is a missing cell, not an observation.
        let n = 70_usize;
        let mut values: Vec<f64> = (0..70_u32).map(f64::from).collect();
        values[3] = f64::NAN;
        values[66] = f64::NAN;
        let mut bytes = vec![0xFFu8; 9];
        bytes[1] &= !(1 << 2); // row 10 invalid through the bitmap
        let validity = ValidityBitmap::from_bytes(bytes, n).unwrap();
        let col = Float64Column::new(VariableId::from_raw(0), values, validity).unwrap();
        for i in 0..n {
            let missing = matches!(i, 3 | 10 | 66);
            assert_eq!(col.validity.is_valid(i), !missing, "row {i}");
            assert_eq!(col.values[i].is_nan(), missing, "row {i}");
        }
    }

    #[test]
    fn clean_scan_marks_the_buffer_nan_free_and_rewraps_without_a_copy() {
        let n = 70_usize;
        let values: Vec<f64> = (0..70_u32).map(f64::from).collect();
        let col = Float64Column::new(VariableId::from_raw(0), values, ValidityBitmap::all_valid(n))
            .unwrap();
        assert!(col.values.is_nan_free());
        let ptr = col.values.as_slice().as_ptr();
        // Re-wrapping the constructed buffer keeps it (no scan, no copy).
        let again = Float64Column::new(
            VariableId::from_raw(1),
            col.values.clone(),
            ValidityBitmap::all_valid(n),
        )
        .unwrap();
        assert!(again.values.is_nan_free());
        assert_eq!(again.values.as_slice().as_ptr(), ptr);
        // A NaN-free buffer under a bitmap with invalid rows still gets NaN written.
        let mut bytes = vec![0xFFu8; 9];
        bytes[8] &= !(1 << 1); // row 65 invalid
        let masked = Float64Column::new(
            VariableId::from_raw(2),
            col.values.clone(),
            ValidityBitmap::from_bytes(bytes, n).unwrap(),
        )
        .unwrap();
        assert!(!masked.values.is_nan_free());
        assert!(masked.values[65].is_nan());
        assert_eq!(masked.values[64], 64.0);
        assert!(!masked.validity.is_valid(65));
        // Raw buffers are unknown until scanned; a NaN leaves them unmarked.
        let mut with_nan = (0..70_u32).map(f64::from).collect::<Vec<_>>();
        with_nan[9] = f64::NAN;
        let col =
            Float64Column::new(VariableId::from_raw(0), with_nan, ValidityBitmap::all_valid(n))
                .unwrap();
        assert!(!col.values.is_nan_free());
        assert!(!col.validity.is_valid(9));
    }

    /// The fused pass matches a row-by-row reference on every mix of NaN under
    /// valid bits and observations under invalid bits, across the 8-row tail.
    #[test]
    fn fused_scan_establishes_the_invariant_on_every_row_mix() {
        let mut rng = antecedent_core::CausalRng::from_seed(0x9E37_79B9_7F4A_7C15);
        let mut next = move || rng.next_u64();
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 40, 130] {
            for _case in 0..40 {
                let mut values: Vec<f64> = (0..n).map(|i| i as f64 + 0.5).collect();
                let mut bytes = vec![0xFFu8; n.div_ceil(8)];
                for i in 0..n {
                    match next() % 5 {
                        0 => values[i] = f64::NAN,            // NaN under a valid bit
                        1 => bytes[i / 8] &= !(1 << (i % 8)), // observation under invalid
                        2 => {
                            values[i] = f64::NAN;
                            bytes[i / 8] &= !(1 << (i % 8));
                        }
                        _ => {}
                    }
                }
                let validity = ValidityBitmap::from_bytes(bytes.clone(), n).unwrap();
                let scan = scan_against_validity(&values, &validity);
                let expect_nan_under_valid =
                    (0..n).any(|i| values[i].is_nan() && (bytes[i / 8] >> (i % 8)) & 1 == 1);
                let expect_observed_under_invalid =
                    (0..n).any(|i| !values[i].is_nan() && (bytes[i / 8] >> (i % 8)) & 1 == 0);
                assert_eq!(scan.nan_under_valid, expect_nan_under_valid, "n={n}");
                assert_eq!(scan.observed_under_invalid, expect_observed_under_invalid, "n={n}");
                let col =
                    Float64Column::new(VariableId::from_raw(0), values.clone(), validity).unwrap();
                for i in 0..n {
                    let missing = values[i].is_nan() || (bytes[i / 8] >> (i % 8)) & 1 == 0;
                    assert_eq!(col.validity.is_valid(i), !missing, "n={n} row {i}");
                    assert_eq!(col.values[i].is_nan(), missing, "n={n} row {i}");
                    if !missing {
                        assert_eq!(col.values[i], values[i]);
                    }
                }
                let no_missing = (0..n).all(|i| col.validity.is_valid(i));
                assert_eq!(col.values.is_nan_free(), no_missing, "n={n}");
            }
        }
    }
}
