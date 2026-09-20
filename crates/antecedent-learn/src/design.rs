//! Antecedent-owned design interchange. Adapters translate; foreign matrix
//! types never appear here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::F64MatrixView;
use antecedent_stats::CompiledDesign;

use crate::error::LearnError;

/// Memory layout of a dense design buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layout {
    /// `values[col * rows + row]`.
    ColumnMajor,
    /// `values[row * cols + col]`.
    RowMajor,
}

/// Borrowed dense design matrix.
#[derive(Clone, Copy, Debug)]
pub struct DenseDesign<'a> {
    values: &'a [f64],
    rows: usize,
    cols: usize,
    layout: Layout,
}

impl<'a> DenseDesign<'a> {
    /// Borrow a packed dense buffer.
    ///
    /// # Errors
    ///
    /// Buffer shorter than `rows * cols`, or overflow.
    pub fn new(
        values: &'a [f64],
        rows: usize,
        cols: usize,
        layout: Layout,
    ) -> Result<Self, LearnError> {
        let need =
            rows.checked_mul(cols).ok_or(LearnError::Shape { message: "rows*cols overflow" })?;
        if values.len() < need {
            return Err(LearnError::Shape { message: "dense buffer shorter than matrix" });
        }
        Ok(Self { values, rows, cols, layout })
    }

    /// Underlying values (at least `rows * cols` long).
    #[must_use]
    pub const fn values(self) -> &'a [f64] {
        self.values
    }

    /// Physical row count.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Column count.
    #[must_use]
    pub const fn cols(self) -> usize {
        self.cols
    }

    /// Storage layout.
    #[must_use]
    pub const fn layout(self) -> Layout {
        self.layout
    }

    /// Element at a physical `(row, col)`.
    ///
    /// # Errors
    ///
    /// Out of bounds.
    pub fn get(self, row: usize, col: usize) -> Result<f64, LearnError> {
        if row >= self.rows {
            return Err(LearnError::Shape { message: "dense row out of bounds" });
        }
        if col >= self.cols {
            return Err(LearnError::Shape { message: "dense col out of bounds" });
        }
        let idx = match self.layout {
            Layout::ColumnMajor => col * self.rows + row,
            Layout::RowMajor => row * self.cols + col,
        };
        Ok(self.values[idx])
    }
}

/// Borrowed CSR sparse design. Typed so a planner can choose it; solvers in
/// A–C do not consume this variant.
#[derive(Clone, Copy, Debug)]
pub struct SparseDesignView<'a> {
    values: &'a [f64],
    col_indices: &'a [u32],
    row_ptrs: &'a [u32],
    rows: usize,
    cols: usize,
}

impl<'a> SparseDesignView<'a> {
    /// Construct a CSR view.
    ///
    /// # Errors
    ///
    /// `row_ptrs` must have length `rows + 1`; `values` and `col_indices` must
    /// share a length matching `row_ptrs[rows]`.
    pub fn new(
        values: &'a [f64],
        col_indices: &'a [u32],
        row_ptrs: &'a [u32],
        rows: usize,
        cols: usize,
    ) -> Result<Self, LearnError> {
        if row_ptrs.len() != rows.saturating_add(1) {
            return Err(LearnError::Shape { message: "CSR row_ptrs length must be rows+1" });
        }
        let nnz = usize::try_from(*row_ptrs.last().unwrap_or(&0))
            .map_err(|_| LearnError::Shape { message: "CSR nnz overflow" })?;
        if row_ptrs.first() != Some(&0) || row_ptrs.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(LearnError::Shape {
                message: "CSR row pointers must start at zero and be monotone",
            });
        }
        if values.len() != nnz || col_indices.len() != nnz {
            return Err(LearnError::Shape { message: "CSR values/indices shorter than nnz" });
        }
        if col_indices.iter().any(|&col| col as usize >= cols) {
            return Err(LearnError::Shape { message: "CSR column index out of bounds" });
        }
        Ok(Self { values, col_indices, row_ptrs, rows, cols })
    }

    /// Non-zero values.
    #[must_use]
    pub const fn values(self) -> &'a [f64] {
        self.values
    }

    /// Column index per non-zero.
    #[must_use]
    pub const fn col_indices(self) -> &'a [u32] {
        self.col_indices
    }

    /// Row pointers (`rows + 1`).
    #[must_use]
    pub const fn row_ptrs(self) -> &'a [u32] {
        self.row_ptrs
    }

    /// Physical row count.
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Column count.
    #[must_use]
    pub const fn cols(self) -> usize {
        self.cols
    }
}

/// Physical storage behind a [`DesignView`].
#[derive(Clone, Copy, Debug)]
pub enum DesignStorage<'a> {
    /// Packed dense matrix.
    Dense(DenseDesign<'a>),
    /// CSR placeholder.
    SparseCsr(SparseDesignView<'a>),
}

impl DesignStorage<'_> {
    /// Physical rows in storage (ignores fold selection).
    #[must_use]
    pub const fn physical_rows(self) -> usize {
        match self {
            Self::Dense(d) => d.rows(),
            Self::SparseCsr(s) => s.rows(),
        }
    }

    /// Column count.
    #[must_use]
    pub const fn cols(self) -> usize {
        match self {
            Self::Dense(d) => d.cols(),
            Self::SparseCsr(s) => s.cols(),
        }
    }
}

/// Fold / train row subset over an immutable design buffer.
#[derive(Clone, Copy, Debug)]
pub struct RowSelection<'a> {
    indices: &'a [u32],
}

impl<'a> RowSelection<'a> {
    /// Borrow row indices into a physical design.
    #[must_use]
    pub const fn new(indices: &'a [u32]) -> Self {
        Self { indices }
    }

    /// Index vector.
    #[must_use]
    pub const fn indices(self) -> &'a [u32] {
        self.indices
    }

    /// Number of selected rows.
    #[must_use]
    pub const fn len(self) -> usize {
        self.indices.len()
    }

    /// Whether empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.indices.is_empty()
    }
}

/// Borrowed design: storage plus optional fold row indices.
#[derive(Clone, Copy, Debug)]
pub struct DesignView<'a> {
    storage: DesignStorage<'a>,
    rows: Option<RowSelection<'a>>,
}

impl<'a> DesignView<'a> {
    /// View a column-major dense buffer.
    ///
    /// # Errors
    ///
    /// Shape errors from [`DenseDesign::new`].
    pub fn from_column_major(
        values: &'a [f64],
        rows: usize,
        cols: usize,
    ) -> Result<Self, LearnError> {
        Ok(Self {
            storage: DesignStorage::Dense(DenseDesign::new(
                values,
                rows,
                cols,
                Layout::ColumnMajor,
            )?),
            rows: None,
        })
    }

    /// View a row-major dense buffer.
    ///
    /// # Errors
    ///
    /// Shape errors from [`DenseDesign::new`].
    pub fn from_row_major(values: &'a [f64], rows: usize, cols: usize) -> Result<Self, LearnError> {
        Ok(Self {
            storage: DesignStorage::Dense(DenseDesign::new(values, rows, cols, Layout::RowMajor)?),
            rows: None,
        })
    }

    /// View a kernel matrix if it is packed column-major.
    ///
    /// # Errors
    ///
    /// Non-contiguous or non-column-major kernel views.
    pub fn from_f64_matrix_view(view: F64MatrixView<'a>) -> Result<Self, LearnError> {
        let values = view.as_column_major_slice().ok_or(LearnError::Unsupported {
            message: "F64MatrixView is not packed column-major",
        })?;
        Self::from_column_major(values, view.nrows(), view.ncols())
    }

    /// View a compiled causal design's matrix without copying. Ignores causal
    /// metadata and complete-case [`CompiledDesign::row_selection`].
    ///
    /// # Errors
    ///
    /// Shape errors if the compiled buffer is inconsistent.
    pub fn from_compiled(design: &'a CompiledDesign) -> Result<Self, LearnError> {
        Self::from_column_major(design.matrix.as_ref(), design.nrows, design.ncols)
    }

    /// Restrict to a fold / train subset. Indices address physical rows.
    ///
    /// # Errors
    ///
    /// An index at or beyond the physical row count.
    pub fn with_rows(self, rows: RowSelection<'a>) -> Result<Self, LearnError> {
        let n = self.storage.physical_rows();
        for &i in rows.indices() {
            if (i as usize) >= n {
                return Err(LearnError::Shape { message: "row selection index out of bounds" });
            }
        }
        Ok(Self { storage: self.storage, rows: Some(rows) })
    }

    /// Storage (dense or sparse).
    #[must_use]
    pub const fn storage(self) -> DesignStorage<'a> {
        self.storage
    }

    /// Fold selection, if any.
    #[must_use]
    pub const fn row_selection(self) -> Option<RowSelection<'a>> {
        self.rows
    }

    /// Logical row count (selected rows, or all physical rows).
    #[must_use]
    pub fn nrows(self) -> usize {
        self.rows.map_or_else(|| self.storage.physical_rows(), RowSelection::len)
    }

    /// Physical rows in the backing buffer.
    #[must_use]
    pub const fn physical_nrows(self) -> usize {
        self.storage.physical_rows()
    }

    /// Column count.
    #[must_use]
    pub const fn ncols(self) -> usize {
        self.storage.cols()
    }

    /// Physical row for logical row `i`.
    ///
    /// # Errors
    ///
    /// Logical row out of bounds.
    pub fn physical_index(self, logical_row: usize) -> Result<usize, LearnError> {
        if logical_row >= self.nrows() {
            return Err(LearnError::Shape { message: "logical row out of bounds" });
        }
        match self.rows {
            Some(sel) => Ok(sel.indices()[logical_row] as usize),
            None => Ok(logical_row),
        }
    }

    /// Element at a logical `(row, col)`.
    ///
    /// # Errors
    ///
    /// Out of bounds, or sparse storage (no dense getter).
    pub fn get(self, row: usize, col: usize) -> Result<f64, LearnError> {
        let physical = self.physical_index(row)?;
        match self.storage {
            DesignStorage::Dense(d) => d.get(physical, col),
            DesignStorage::SparseCsr(_) => {
                Err(LearnError::Unsupported { message: "dense get on sparse design" })
            }
        }
    }
}

/// Borrowed target aligned with the design's **physical** rows.
#[derive(Clone, Copy, Debug)]
pub struct TargetView<'a> {
    values: &'a [f64],
}

impl<'a> TargetView<'a> {
    /// Borrow a target vector.
    #[must_use]
    pub const fn new(values: &'a [f64]) -> Self {
        Self { values }
    }

    /// Physical target values.
    #[must_use]
    pub const fn values(self) -> &'a [f64] {
        self.values
    }

    /// Length.
    #[must_use]
    pub const fn len(self) -> usize {
        self.values.len()
    }

    /// Whether empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn index_only_folds_read_one_buffer() {
        // Column-major 4×2: col0 = [1,1,1,1], col1 = [0,1,2,3]
        let x = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let view = DesignView::from_column_major(&x, 4, 2).unwrap();
        let fold1 = [0_u32, 2];
        let fold2 = [1_u32, 3];
        let a = view.with_rows(RowSelection::new(&fold1)).unwrap();
        let b = view.with_rows(RowSelection::new(&fold2)).unwrap();
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.get(0, 1).unwrap(), 0.0);
        assert_eq!(a.get(1, 1).unwrap(), 2.0);
        assert_eq!(b.get(0, 1).unwrap(), 1.0);
        assert_eq!(b.get(1, 1).unwrap(), 3.0);
        assert!(std::ptr::eq(
            match a.storage() {
                DesignStorage::Dense(d) => d.values().as_ptr(),
                DesignStorage::SparseCsr(_) => unreachable!(),
            },
            x.as_ptr()
        ));
    }

    #[test]
    fn compiled_design_is_a_zero_copy_view() {
        let t = [0.0, 1.0, 0.0];
        let y = [4.0, 5.0, 6.0];
        let compiled = CompiledDesign::linear_adjustment(&t, &[], &y, &[]).unwrap();
        let view = DesignView::from_compiled(&compiled).unwrap();
        assert_eq!(view.nrows(), 3);
        assert_eq!(view.ncols(), 2);
        assert!(std::ptr::eq(
            match view.storage() {
                DesignStorage::Dense(d) => d.values().as_ptr(),
                DesignStorage::SparseCsr(_) => unreachable!(),
            },
            compiled.matrix.as_ptr()
        ));
    }
}

#[cfg(test)]
mod csr_audit_tests {
    use super::*;
    #[test]
    fn malformed_csr_is_rejected_at_construction() {
        assert!(SparseDesignView::new(&[1.0], &[0], &[1, 1], 1, 1).is_err());
        assert!(SparseDesignView::new(&[1.0], &[0], &[0, 2, 1], 2, 1).is_err());
        assert!(SparseDesignView::new(&[1.0], &[1], &[0, 1], 1, 1).is_err());
        assert!(SparseDesignView::new(&[1.0], &[0], &[0, 1], 1, 1).is_ok());
    }
}
