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
/// The bitmap is authoritative for missingness. [`Float64Column`] additionally
/// stores `NaN` under every invalid row so value-only readers cannot mistake a
/// missing cell for an observation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ValidityBitmap {
    bytes: Arc<[u8]>,
    len: usize,
}

impl ValidityBitmap {
    /// All-valid bitmap of `len` bits.
    #[must_use]
    pub fn all_valid(len: usize) -> Self {
        let n = len.div_ceil(8);
        Self { bytes: Arc::from(vec![0xFFu8; n].into_boxed_slice()), len }
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
        Ok(Self { bytes, len })
    }

    /// Bit length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
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
        let full = self.len / 8;
        let tail = self.len % 8;
        self.bytes[..full].iter().all(|&b| b == 0xFF)
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
/// Invariant: every invalid row holds `NaN` in `values`. Borrowed readers
/// ([`crate::TableView::float64_slice`]) hand the buffer out without
/// consulting validity, so a missing cell must never read as a finite value.
#[derive(Clone, Debug, PartialEq)]
pub struct Float64Column {
    /// Variable id.
    pub id: VariableId,
    /// Values; `NaN` under every invalid row.
    pub values: F64Buffer,
    /// Validity bitmap (authoritative for missingness).
    pub validity: ValidityBitmap,
}

impl Float64Column {
    /// Construct a column; lengths must match.
    ///
    /// Any invalid row whose value is not already `NaN` is rewritten to `NaN`
    /// (copying the buffer, including a foreign one, only when a rewrite is
    /// needed), so the missing-means-`NaN` invariant holds for every
    /// construction path.
    ///
    /// # Errors
    ///
    /// [`DataError::LengthMismatch`] when validity length differs.
    pub fn new(
        id: VariableId,
        values: impl Into<F64Buffer>,
        validity: ValidityBitmap,
    ) -> Result<Self, DataError> {
        let values = values.into();
        if validity.len() != values.len() {
            return Err(DataError::LengthMismatch {
                expected: values.len(),
                actual: validity.len(),
                context: "float64 validity",
            });
        }
        let values = nan_under_invalid(values, &validity)?;
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

/// Rewrite invalid rows to `NaN`, returning `values` untouched when they already are.
fn nan_under_invalid(values: F64Buffer, validity: &ValidityBitmap) -> Result<F64Buffer, DataError> {
    if invalid_rows_are_nan(values.as_slice(), validity) {
        return Ok(values);
    }
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
#[allow(clippy::float_cmp)] // values are copied bit-for-bit
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

        // A NaN under a valid bit is data, not missingness; it is left alone.
        let col = Float64Column::new(
            VariableId::from_raw(0),
            vec![f64::NAN, 2.0],
            ValidityBitmap::all_valid(2),
        )
        .unwrap();
        assert!(col.validity.is_valid(0));
        assert!(col.values[0].is_nan());
    }
}
