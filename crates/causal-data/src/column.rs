//! Columnar storage and typed column views (DESIGN.md §5.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::VariableId;
use causal_kernels::{BitMaskView, F64VectorView};

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
}

/// Owned float64 column.
#[derive(Clone, Debug, PartialEq)]
pub struct Float64Column {
    /// Variable id.
    pub id: VariableId,
    /// Values (sentinel-free; use validity for missing).
    pub values: Arc<[f64]>,
    /// Validity bitmap.
    pub validity: ValidityBitmap,
}

impl Float64Column {
    /// Construct a column; lengths must match.
    ///
    /// # Errors
    ///
    /// [`DataError::LengthMismatch`] when validity length differs.
    pub fn new(
        id: VariableId,
        values: impl Into<Arc<[f64]>>,
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
        F64VectorView::contiguous(&self.values)
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

/// Owned boolean column (bytes: 0/1 per row for Phase 0 simplicity).
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

impl ColumnView<'_> {
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
