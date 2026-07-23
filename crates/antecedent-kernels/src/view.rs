//! Library-owned borrowed vector and matrix views.
//!
//! Public APIs expose these views; SIMD types are never part of the public API.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

/// Errors when constructing or indexing views.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ViewError {
    /// Shape or stride inconsistency.
    InvalidShape {
        /// Explanation.
        message: &'static str,
    },
    /// Index out of bounds.
    OutOfBounds {
        /// Requested index.
        index: usize,
        /// Valid length.
        len: usize,
    },
}

impl fmt::Display for ViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape { message } => write!(f, "invalid view shape: {message}"),
            Self::OutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for length {len}")
            }
        }
    }
}

impl std::error::Error for ViewError {}

/// Borrowed strided `f64` vector view.
#[derive(Clone, Copy, Debug)]
pub struct F64VectorView<'a> {
    data: &'a [f64],
    len: usize,
    stride: usize,
}

impl<'a> F64VectorView<'a> {
    /// Contiguous vector view over `data`.
    #[must_use]
    pub const fn contiguous(data: &'a [f64]) -> Self {
        Self { data, len: data.len(), stride: 1 }
    }

    /// Strided view. The underlying slice must contain at least
    /// `(len.saturating_sub(1)) * stride + 1` elements when `len > 0`.
    ///
    /// # Errors
    ///
    /// Returns [`ViewError::InvalidShape`] when the slice is too short.
    pub fn strided(data: &'a [f64], len: usize, stride: usize) -> Result<Self, ViewError> {
        if len == 0 {
            return Ok(Self { data, len: 0, stride: stride.max(1) });
        }
        if stride == 0 {
            return Err(ViewError::InvalidShape { message: "stride must be non-zero" });
        }
        let need = (len - 1)
            .checked_mul(stride)
            .and_then(|v| v.checked_add(1))
            .ok_or(ViewError::InvalidShape { message: "stride*len overflow" })?;
        if data.len() < need {
            return Err(ViewError::InvalidShape {
                message: "underlying slice shorter than strided extent",
            });
        }
        Ok(Self { data, len, stride })
    }

    /// Number of logical elements.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Whether empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Stride between logical elements.
    #[must_use]
    pub const fn stride(self) -> usize {
        self.stride
    }

    /// Whether the view is unit-stride contiguous over `len` elements.
    #[must_use]
    pub const fn is_contiguous(self) -> bool {
        self.stride == 1
    }

    /// Element at logical index `i`.
    ///
    /// # Errors
    ///
    /// Returns [`ViewError::OutOfBounds`] when `i >= len`.
    pub fn get(self, i: usize) -> Result<f64, ViewError> {
        if i >= self.len {
            return Err(ViewError::OutOfBounds { index: i, len: self.len });
        }
        Ok(self.data[i * self.stride])
    }

    /// Unchecked element access for hot loops after bounds are established.
    ///
    /// # Safety
    ///
    /// Caller must ensure `i < self.len`.
    #[inline]
    #[must_use]
    pub unsafe fn get_unchecked(self, i: usize) -> f64 {
        // SAFETY: caller guarantees i < len; stride construction validated extent.
        unsafe { *self.data.get_unchecked(i * self.stride) }
    }

    /// Contiguous slice when unit-stride; otherwise `None`.
    #[must_use]
    pub fn as_slice(self) -> Option<&'a [f64]> {
        if self.is_contiguous() { Some(&self.data[..self.len]) } else { None }
    }
}

/// Borrowed column-major or row-major `f64` matrix view.
#[derive(Clone, Copy, Debug)]
pub struct F64MatrixView<'a> {
    data: &'a [f64],
    nrows: usize,
    ncols: usize,
    row_stride: usize,
    col_stride: usize,
}

impl<'a> F64MatrixView<'a> {
    /// Column-major contiguous matrix (`faer`-friendly default).
    ///
    /// # Errors
    ///
    /// Returns [`ViewError::InvalidShape`] when `data.len() < nrows * ncols`.
    pub fn column_major(data: &'a [f64], nrows: usize, ncols: usize) -> Result<Self, ViewError> {
        let need = nrows
            .checked_mul(ncols)
            .ok_or(ViewError::InvalidShape { message: "nrows*ncols overflow" })?;
        if data.len() < need {
            return Err(ViewError::InvalidShape { message: "buffer shorter than matrix" });
        }
        Ok(Self { data, nrows, ncols, row_stride: 1, col_stride: nrows })
    }

    /// Number of rows.
    #[must_use]
    pub const fn nrows(self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[must_use]
    pub const fn ncols(self) -> usize {
        self.ncols
    }

    /// Element at `(row, col)`.
    ///
    /// # Errors
    ///
    /// Out-of-bounds indices.
    pub fn get(self, row: usize, col: usize) -> Result<f64, ViewError> {
        if row >= self.nrows {
            return Err(ViewError::OutOfBounds { index: row, len: self.nrows });
        }
        if col >= self.ncols {
            return Err(ViewError::OutOfBounds { index: col, len: self.ncols });
        }
        Ok(self.data[row * self.row_stride + col * self.col_stride])
    }

    /// Column `j` as a vector view.
    ///
    /// # Errors
    ///
    /// Out-of-bounds column.
    pub fn column(self, j: usize) -> Result<F64VectorView<'a>, ViewError> {
        if j >= self.ncols {
            return Err(ViewError::OutOfBounds { index: j, len: self.ncols });
        }
        let offset = j * self.col_stride;
        F64VectorView::strided(&self.data[offset..], self.nrows, self.row_stride)
    }
}

/// Optional validity / analysis mask as a packed bitmap (`1` = valid/included).
#[derive(Clone, Copy, Debug)]
pub struct BitMaskView<'a> {
    bytes: &'a [u8],
    len: usize,
}

impl<'a> BitMaskView<'a> {
    /// Create a mask covering `len` bits.
    ///
    /// # Errors
    ///
    /// When `bytes` is shorter than `ceil(len / 8)`.
    pub fn new(bytes: &'a [u8], len: usize) -> Result<Self, ViewError> {
        let need = len.div_ceil(8);
        if bytes.len() < need {
            return Err(ViewError::InvalidShape { message: "mask buffer too short" });
        }
        Ok(Self { bytes, len })
    }

    /// Number of bits.
    #[must_use]
    pub const fn len(self) -> usize {
        self.len
    }

    /// Whether empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Whether bit `i` is set.
    #[must_use]
    pub fn get(self, i: usize) -> bool {
        if i >= self.len {
            return false;
        }
        let byte = self.bytes[i / 8];
        (byte >> (i % 8)) & 1 == 1
    }
}
