//! Columnar value / noise batches and mechanism workspaces.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::error::ModelError;

/// Borrow column `node` of a column-major `n_rows x n_nodes` buffer, validating the
/// declared shape against the buffer so a mismatched batch is a [`ModelError::Shape`]
/// rather than a slice panic.
fn checked_column<'a>(
    values: &'a [f64],
    n_rows: usize,
    n_nodes: usize,
    node: usize,
    label: &str,
) -> Result<&'a [f64], ModelError> {
    if node >= n_nodes {
        return Err(ModelError::Shape { message: format!("{label} column out of range") });
    }
    let range = column_range(values.len(), n_rows, node, label)?;
    Ok(&values[range])
}

/// Mutable counterpart of [`checked_column`].
fn checked_column_mut<'a>(
    values: &'a mut [f64],
    n_rows: usize,
    n_nodes: usize,
    node: usize,
    label: &str,
) -> Result<&'a mut [f64], ModelError> {
    if node >= n_nodes {
        return Err(ModelError::Shape { message: format!("{label} column out of range") });
    }
    let range = column_range(values.len(), n_rows, node, label)?;
    Ok(&mut values[range])
}

fn column_range(
    len: usize,
    n_rows: usize,
    node: usize,
    label: &str,
) -> Result<std::ops::Range<usize>, ModelError> {
    let start = node.checked_mul(n_rows);
    let end = start.and_then(|s| s.checked_add(n_rows));
    match (start, end) {
        (Some(start), Some(end)) if end <= len => Ok(start..end),
        _ => Err(ModelError::Shape {
            message: format!("{label} batch buffer is shorter than its declared shape"),
        }),
    }
}

/// Value at `(row, node)` of a checked column-major buffer.
fn checked_get(
    values: &[f64],
    n_rows: usize,
    n_nodes: usize,
    row: usize,
    node: usize,
) -> Result<f64, ModelError> {
    if row >= n_rows || node >= n_nodes {
        return Err(ModelError::Shape { message: "value index out of range".into() });
    }
    Ok(checked_column(values, n_rows, n_nodes, node, "value")?[row])
}

/// Column-major batch of continuous values: `values[node * n_rows + row]`.
#[derive(Clone, Debug, Default)]
pub struct ValueBatch {
    /// Number of rows (samples / units).
    pub n_rows: usize,
    /// Number of nodes (columns).
    pub n_nodes: usize,
    /// Flat column-major storage.
    pub values: Arc<[f64]>,
}

impl ValueBatch {
    /// Allocate zeros.
    #[must_use]
    pub fn zeros(n_rows: usize, n_nodes: usize) -> Self {
        Self { n_rows, n_nodes, values: Arc::from(vec![0.0; n_rows.saturating_mul(n_nodes)]) }
    }

    /// Borrow a column.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn column(&self, node: usize) -> Result<&[f64], ModelError> {
        checked_column(&self.values, self.n_rows, self.n_nodes, node, "value")
    }

    /// Value at `(row, node)`.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn get(&self, row: usize, node: usize) -> Result<f64, ModelError> {
        checked_get(&self.values, self.n_rows, self.n_nodes, row, node)
    }
}

/// Mutable view into a value batch (owned buffer).
#[derive(Debug)]
pub struct ValueBatchMut<'a> {
    /// Rows.
    pub n_rows: usize,
    /// Nodes.
    pub n_nodes: usize,
    /// Flat column-major storage.
    pub values: &'a mut [f64],
}

impl<'a> ValueBatchMut<'a> {
    /// Wrap a buffer.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn new(n_rows: usize, n_nodes: usize, values: &'a mut [f64]) -> Result<Self, ModelError> {
        if values.len() < n_rows.saturating_mul(n_nodes) {
            return Err(ModelError::Shape { message: "value buffer too short".into() });
        }
        Ok(Self { n_rows, n_nodes, values })
    }

    /// Mutable column slice.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn column_mut(&mut self, node: usize) -> Result<&mut [f64], ModelError> {
        checked_column_mut(self.values, self.n_rows, self.n_nodes, node, "value")
    }

    /// Set `(row, node)`.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn set(&mut self, row: usize, node: usize, v: f64) -> Result<(), ModelError> {
        if row >= self.n_rows || node >= self.n_nodes {
            return Err(ModelError::Shape { message: "value index out of range".into() });
        }
        checked_column_mut(self.values, self.n_rows, self.n_nodes, node, "value")?[row] = v;
        Ok(())
    }

    /// Freeze into an owned [`ValueBatch`].
    #[must_use]
    pub fn into_batch(self) -> ValueBatch {
        ValueBatch {
            n_rows: self.n_rows,
            n_nodes: self.n_nodes,
            values: Arc::from(self.values.to_vec()),
        }
    }
}

/// Columnar exogenous noise batch (same layout as [`ValueBatch`]).
#[derive(Clone, Debug, Default)]
pub struct NoiseBatch {
    /// Rows.
    pub n_rows: usize,
    /// Nodes.
    pub n_nodes: usize,
    /// Flat storage.
    pub values: Arc<[f64]>,
}

impl NoiseBatch {
    /// Zeros.
    #[must_use]
    pub fn zeros(n_rows: usize, n_nodes: usize) -> Self {
        Self { n_rows, n_nodes, values: Arc::from(vec![0.0; n_rows.saturating_mul(n_nodes)]) }
    }

    /// Column.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn column(&self, node: usize) -> Result<&[f64], ModelError> {
        checked_column(&self.values, self.n_rows, self.n_nodes, node, "noise")
    }
}

/// Mutable noise batch.
#[derive(Debug)]
pub struct NoiseBatchMut<'a> {
    /// Rows.
    pub n_rows: usize,
    /// Nodes.
    pub n_nodes: usize,
    /// Storage.
    pub values: &'a mut [f64],
}

impl<'a> NoiseBatchMut<'a> {
    /// Wrap.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn new(n_rows: usize, n_nodes: usize, values: &'a mut [f64]) -> Result<Self, ModelError> {
        if values.len() < n_rows.saturating_mul(n_nodes) {
            return Err(ModelError::Shape { message: "noise buffer too short".into() });
        }
        Ok(Self { n_rows, n_nodes, values })
    }

    /// Immutable column.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn column(&self, node: usize) -> Result<&[f64], ModelError> {
        checked_column(&*self.values, self.n_rows, self.n_nodes, node, "noise")
    }

    /// Mutable column.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn column_mut(&mut self, node: usize) -> Result<&mut [f64], ModelError> {
        checked_column_mut(self.values, self.n_rows, self.n_nodes, node, "noise")
    }

    /// Freeze.
    #[must_use]
    pub fn into_batch(self) -> NoiseBatch {
        NoiseBatch {
            n_rows: self.n_rows,
            n_nodes: self.n_nodes,
            values: Arc::from(self.values.to_vec()),
        }
    }
}

/// Borrowed parent columns for one node (aligned row-major view into gathered parents).
#[derive(Clone, Copy, Debug)]
pub struct ParentBatch<'a> {
    /// Number of rows.
    pub n_rows: usize,
    /// Number of parents.
    pub n_parents: usize,
    /// Flat `parent * n_rows + row`.
    pub values: &'a [f64],
}

impl<'a> ParentBatch<'a> {
    /// Empty parents.
    #[must_use]
    pub const fn empty(n_rows: usize) -> Self {
        Self { n_rows, n_parents: 0, values: &[] }
    }

    /// Parent column `p`.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn column(&self, p: usize) -> Result<&'a [f64], ModelError> {
        if p >= self.n_parents {
            return Err(ModelError::Shape { message: "parent column out of range".into() });
        }
        let range = column_range(self.values.len(), self.n_rows, p, "parent")?;
        Ok(&self.values[range])
    }

    /// Every parent column, validated once. Row loops index the returned slices instead of
    /// re-validating and re-slicing through [`Self::column`] per element.
    ///
    /// # Errors
    ///
    /// The flat buffer is shorter than `n_parents * n_rows`.
    pub fn columns(&self) -> Result<Vec<&'a [f64]>, ModelError> {
        (0..self.n_parents).map(|p| self.column(p)).collect()
    }
}

/// Reusable scratch for mechanism evaluation.
#[derive(Clone, Debug, Default)]
pub struct MechanismWorkspace {
    /// Gathered parent matrix (column-major over parents).
    pub parents: Vec<f64>,
    /// Scratch residuals / linear predictors.
    pub scratch: Vec<f64>,
    /// Work columns of the intervention-program executor (grown, never shrunk).
    pub program_scratch: Vec<f64>,
    /// Grow counter (tests / reuse gates).
    pub grow_count: u32,
}

impl MechanismWorkspace {
    /// Ensure capacity for `n_rows` × `n_parents` gather + scratch of `n_rows`.
    pub fn prepare(&mut self, n_rows: usize, n_parents: usize) {
        let need = n_rows.saturating_mul(n_parents.max(1));
        if self.parents.capacity() < need {
            self.parents.reserve(need - self.parents.len());
            self.grow_count = self.grow_count.saturating_add(1);
        }
        self.parents.resize(need, 0.0);
        if self.scratch.capacity() < n_rows {
            self.scratch.reserve(n_rows.saturating_sub(self.scratch.len()));
            self.grow_count = self.grow_count.saturating_add(1);
        }
        self.scratch.resize(n_rows, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_reserves_from_length_after_a_smaller_reuse() {
        let mut ws = MechanismWorkspace::default();
        ws.prepare(64, 1);
        ws.prepare(1, 1);
        let target = ws.parents.capacity().max(ws.scratch.capacity()) + 1;
        ws.prepare(target, 1);
        assert!(ws.parents.capacity() >= target);
        assert!(ws.scratch.capacity() >= target);
        assert_eq!(ws.grow_count, 4);
        ws.prepare(target, 1);
        assert_eq!(ws.grow_count, 4);
    }

    #[test]
    fn a_batch_whose_buffer_is_shorter_than_its_shape_reports_a_shape_error() {
        let batch = ValueBatch { n_rows: 3, n_nodes: 2, values: Arc::from(vec![0.0; 4]) };
        assert!(batch.column(0).is_ok());
        assert!(matches!(batch.column(1), Err(ModelError::Shape { .. })));
        assert!(matches!(batch.get(2, 1), Err(ModelError::Shape { .. })));
        let mut buffer = vec![0.0; 4];
        let mut mutable = ValueBatchMut { n_rows: 3, n_nodes: 2, values: &mut buffer };
        assert!(matches!(mutable.column_mut(1), Err(ModelError::Shape { .. })));
        assert!(matches!(mutable.set(2, 1, 1.0), Err(ModelError::Shape { .. })));
    }
}
