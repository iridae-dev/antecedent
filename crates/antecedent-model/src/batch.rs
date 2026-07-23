//! Columnar value / noise batches and mechanism workspaces.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use crate::error::ModelError;

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
        if node >= self.n_nodes {
            return Err(ModelError::Shape { message: "value column out of range".into() });
        }
        let start = node * self.n_rows;
        Ok(&self.values[start..start + self.n_rows])
    }

    /// Value at `(row, node)`.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn get(&self, row: usize, node: usize) -> Result<f64, ModelError> {
        if row >= self.n_rows || node >= self.n_nodes {
            return Err(ModelError::Shape { message: "value index out of range".into() });
        }
        Ok(self.values[node * self.n_rows + row])
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
        if node >= self.n_nodes {
            return Err(ModelError::Shape { message: "value column out of range".into() });
        }
        let start = node * self.n_rows;
        Ok(&mut self.values[start..start + self.n_rows])
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
        self.values[node * self.n_rows + row] = v;
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
        if node >= self.n_nodes {
            return Err(ModelError::Shape { message: "noise column out of range".into() });
        }
        let start = node * self.n_rows;
        Ok(&self.values[start..start + self.n_rows])
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
        if node >= self.n_nodes {
            return Err(ModelError::Shape { message: "noise column out of range".into() });
        }
        let start = node * self.n_rows;
        Ok(&self.values[start..start + self.n_rows])
    }

    /// Mutable column.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn column_mut(&mut self, node: usize) -> Result<&mut [f64], ModelError> {
        if node >= self.n_nodes {
            return Err(ModelError::Shape { message: "noise column out of range".into() });
        }
        let start = node * self.n_rows;
        Ok(&mut self.values[start..start + self.n_rows])
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
        let start = p * self.n_rows;
        Ok(&self.values[start..start + self.n_rows])
    }
}

/// Reusable scratch for mechanism evaluation.
#[derive(Clone, Debug, Default)]
pub struct MechanismWorkspace {
    /// Gathered parent matrix (column-major over parents).
    pub parents: Vec<f64>,
    /// Scratch residuals / linear predictors.
    pub scratch: Vec<f64>,
    /// Grow counter (tests / reuse gates).
    pub grow_count: u32,
}

impl MechanismWorkspace {
    /// Ensure capacity for `n_rows` × `n_parents` gather + scratch of `n_rows`.
    pub fn prepare(&mut self, n_rows: usize, n_parents: usize) {
        let need = n_rows.saturating_mul(n_parents.max(1));
        if self.parents.capacity() < need {
            self.parents.reserve(need - self.parents.capacity());
            self.grow_count = self.grow_count.saturating_add(1);
        }
        self.parents.resize(need, 0.0);
        if self.scratch.capacity() < n_rows {
            self.scratch.reserve(n_rows.saturating_sub(self.scratch.capacity()));
            self.grow_count = self.grow_count.saturating_add(1);
        }
        self.scratch.resize(n_rows, 0.0);
    }
}
