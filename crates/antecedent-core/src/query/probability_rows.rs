//! Row-major table of per-row action probabilities stored in one allocation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::Index;
use std::sync::Arc;

/// Known action probabilities, one row per unit and one column per declared action.
///
/// The rows live in a single flat `Arc<[f64]>` addressed with a fixed stride (the number of
/// actions), so a query over `n` units holds one allocation, not `n`, and a row access is an
/// offset computation. Rows built from input of unequal length cannot be laid out with a
/// stride; such a table keeps its row count but reports a stride of zero, every row reads as
/// empty, and [`Self::is_rectangular`] is `false`. A query's validation refuses it because an
/// empty row never matches the declared action count.
#[derive(Clone, Debug, PartialEq)]
pub struct ProbabilityRows {
    values: Arc<[f64]>,
    rows: usize,
    stride: usize,
}

impl ProbabilityRows {
    /// A table of `values.len() / stride` rows of `stride` entries each.
    ///
    /// A zero `stride`, or a length that is not a multiple of it, is malformed: the table keeps
    /// the values, reports no rectangular layout, and every row reads as empty.
    #[must_use]
    pub fn new(values: impl Into<Arc<[f64]>>, stride: usize) -> Self {
        let values = values.into();
        if stride > 0 && values.len() % stride == 0 {
            let rows = values.len() / stride;
            Self { values, rows, stride }
        } else {
            Self { values, rows: 0, stride: 0 }
        }
    }

    /// Build from explicit rows. Rows of unequal length produce a malformed table (see the
    /// type documentation) that keeps the row count.
    #[must_use]
    pub fn from_rows<R: AsRef<[f64]>>(rows: &[R]) -> Self {
        let stride = rows.first().map_or(0, |row| row.as_ref().len());
        let rectangular = rows.iter().all(|row| row.as_ref().len() == stride);
        let values: Arc<[f64]> = rows.iter().flat_map(|row| row.as_ref().iter().copied()).collect();
        if rectangular {
            Self { values, rows: rows.len(), stride }
        } else {
            Self { values, rows: rows.len(), stride: 0 }
        }
    }

    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Entries per row; zero for a malformed table.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Whether every row has the same length, so rows can be addressed by stride.
    #[must_use]
    pub fn is_rectangular(&self) -> bool {
        self.stride > 0 && self.stride * self.rows == self.values.len()
    }

    /// All entries, row after row.
    #[must_use]
    pub fn as_flat(&self) -> &[f64] {
        &self.values
    }

    /// Row `i`, or `None` past the end or for a malformed table.
    #[must_use]
    pub fn row(&self, i: usize) -> Option<&[f64]> {
        if !self.is_rectangular() || i >= self.rows {
            return None;
        }
        self.values.get(i * self.stride..(i + 1) * self.stride)
    }

    /// Rows in order; a malformed table yields `len()` empty rows.
    #[must_use]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[f64]> + '_ {
        (0..self.rows).map(|i| self.row(i).unwrap_or(&[]))
    }

    /// Rows copied into owned vectors.
    #[must_use]
    pub fn to_vec(&self) -> Vec<Vec<f64>> {
        self.iter().map(<[f64]>::to_vec).collect()
    }
}

/// Read access to a table of per-row action probabilities, so an estimator accepts the flat
/// [`ProbabilityRows`] a query stores as well as a plain `Vec` of rows.
pub trait ProbabilityTable {
    /// Number of rows.
    fn row_count(&self) -> usize;

    /// Row `i`.
    ///
    /// # Panics
    ///
    /// Panics when `i >= row_count()`.
    fn row_at(&self, i: usize) -> &[f64];

    /// The first row, or `None` for an empty table.
    fn first_row(&self) -> Option<&[f64]> {
        (self.row_count() > 0).then(|| self.row_at(0))
    }
}

impl ProbabilityTable for ProbabilityRows {
    fn row_count(&self) -> usize {
        self.len()
    }

    fn row_at(&self, i: usize) -> &[f64] {
        &self[i]
    }
}

impl ProbabilityTable for [Vec<f64>] {
    fn row_count(&self) -> usize {
        self.len()
    }

    fn row_at(&self, i: usize) -> &[f64] {
        &self[i]
    }
}

impl ProbabilityTable for Vec<Vec<f64>> {
    fn row_count(&self) -> usize {
        self.len()
    }

    fn row_at(&self, i: usize) -> &[f64] {
        &self[i]
    }
}

impl Index<usize> for ProbabilityRows {
    type Output = [f64];

    /// # Panics
    ///
    /// Panics when `i >= len()`. A malformed table yields an empty row.
    fn index(&self, i: usize) -> &[f64] {
        assert!(i < self.rows, "probability row {i} out of range for {} rows", self.rows);
        self.row(i).unwrap_or(&[])
    }
}

impl From<Vec<Vec<f64>>> for ProbabilityRows {
    fn from(rows: Vec<Vec<f64>>) -> Self {
        Self::from_rows(&rows)
    }
}

impl From<&[Vec<f64>]> for ProbabilityRows {
    fn from(rows: &[Vec<f64>]) -> Self {
        Self::from_rows(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_addressed_by_stride() {
        let rows = ProbabilityRows::from(vec![vec![0.2, 0.3, 0.5], vec![0.1, 0.1, 0.8]]);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows.stride(), 3);
        assert!(rows.is_rectangular());
        assert_eq!(&rows[1], &[0.1, 0.1, 0.8]);
        assert_eq!(rows.as_flat().len(), 6);
        assert_eq!(rows.to_vec(), vec![vec![0.2, 0.3, 0.5], vec![0.1, 0.1, 0.8]]);
        assert_eq!(rows.iter().len(), 2);
        assert_eq!(rows, ProbabilityRows::new(vec![0.2, 0.3, 0.5, 0.1, 0.1, 0.8], 3));
    }

    #[test]
    fn ragged_input_keeps_its_row_count_and_reads_empty() {
        let ragged = ProbabilityRows::from(vec![vec![0.5, 0.5], vec![1.0]]);
        assert_eq!(ragged.len(), 2);
        assert!(!ragged.is_rectangular());
        assert!(ragged.row(0).is_none());
        assert!(ragged[0].is_empty());
        let misaligned = ProbabilityRows::new(vec![0.1; 5], 2);
        assert!(!misaligned.is_rectangular());
        assert_eq!(ProbabilityRows::from(Vec::<Vec<f64>>::new()).len(), 0);
    }
}
