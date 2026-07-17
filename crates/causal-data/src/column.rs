//! Columnar storage and typed column views (DESIGN.md §5.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::VariableId;
use causal_kernels::{BitMaskView, F64VectorView};

use crate::buffer::F64Buffer;
use crate::categorical::CategoricalColumn;
use crate::error::DataError;

/// Packed validity bitmap (`1` = valid). Missingness is never a sentinel value.
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
        self.as_mask_view().is_ok_and(|m| (0..self.len).all(|i| m.get(i)))
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
#[derive(Clone, Debug, PartialEq)]
pub struct Float64Column {
    /// Variable id.
    pub id: VariableId,
    /// Values (sentinel-free; use validity for missing).
    pub values: F64Buffer,
    /// Validity bitmap.
    pub validity: ValidityBitmap,
}

impl Float64Column {
    /// Construct a column from owned values; lengths must match.
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
