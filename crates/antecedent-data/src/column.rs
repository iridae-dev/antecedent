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
        let mask = self.as_mask_view()?;
        let n = row_map.len();
        let mut bytes = vec![0u8; n.div_ceil(8)];
        for (i, &r) in row_map.iter().enumerate() {
            let r = r as usize;
            if r >= self.len {
                return Err(DataError::InvalidValidity { message: "row map exceeds bitmap" });
            }
            if mask.get(r) {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        Self::from_bytes(bytes, n)
    }

    /// Gather bits through a `usize` row map.
    ///
    /// # Errors
    ///
    /// When a mapped row is out of range.
    pub fn gather_rows(&self, row_map: &[usize]) -> Result<Self, DataError> {
        let mask = self.as_mask_view()?;
        let n = row_map.len();
        let mut bytes = vec![0u8; n.div_ceil(8)];
        for (i, &r) in row_map.iter().enumerate() {
            if r >= self.len {
                return Err(DataError::InvalidValidity { message: "row map exceeds bitmap" });
            }
            if mask.get(r) {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        Self::from_bytes(bytes, n)
    }

    /// Compact to rows where `keep[i]` is true.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn compact(&self, keep: &[bool]) -> Result<Self, DataError> {
        if keep.len() != self.len {
            return Err(DataError::LengthMismatch {
                expected: self.len,
                actual: keep.len(),
                context: "validity compact keep",
            });
        }
        let n_new = keep.iter().filter(|&&k| k).count();
        let mut bytes = vec![0u8; n_new.div_ceil(8)];
        let mut j = 0usize;
        for (i, &k) in keep.iter().enumerate() {
            if k {
                if self.is_valid(i) {
                    bytes[j / 8] |= 1 << (j % 8);
                }
                j += 1;
            }
        }
        Self::from_bytes(bytes, n_new)
    }

    /// Concatenate bitmaps end-to-end.
    ///
    /// # Errors
    ///
    /// Propagates bitmap construction errors.
    pub fn concat(parts: &[&Self]) -> Result<Self, DataError> {
        let n: usize = parts.iter().map(|p| p.len).sum();
        let mut bytes = vec![0u8; n.div_ceil(8)];
        let mut offset = 0usize;
        for part in parts {
            for i in 0..part.len {
                if part.is_valid(i) {
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

/// Whether every invalid row of `values` already holds `NaN`.
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

impl OwnedColumn {
    /// Variable id.
    #[must_use]
    pub fn id(&self) -> VariableId {
        match self {
            Self::Float64(c) => c.id,
            Self::Int64(c) => c.id,
            Self::Boolean(c) => c.id,
            Self::Categorical(c) => c.id,
            Self::Timestamp(c) => c.id,
            Self::FixedVector(c) => c.id,
        }
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
#[allow(clippy::float_cmp, clippy::cast_precision_loss)] // values are copied bit-for-bit
mod tests {
    use super::*;

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
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
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
