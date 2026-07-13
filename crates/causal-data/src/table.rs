//! [`TableView`] trait — public causal table API (ADR 0004).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::{CausalSchema, VariableId};

use crate::column::ColumnView;
use crate::error::DataError;

/// Borrowed table access used by algorithms.
pub trait TableView {
    /// Immutable causal schema.
    fn schema(&self) -> &CausalSchema;

    /// Number of rows.
    fn row_count(&self) -> usize;

    /// Column view for `id`.
    ///
    /// # Errors
    ///
    /// Unknown variable or type issues.
    fn column(&self, id: VariableId) -> Result<ColumnView<'_>, DataError>;

    /// Copy a float64 column into an owned buffer.
    ///
    /// # Errors
    ///
    /// Unknown variable or non-float64 column.
    fn float64_values(&self, id: VariableId) -> Result<Vec<f64>, DataError> {
        match self.column(id)? {
            ColumnView::Float64(c) => Ok(c.values.to_vec()),
            _ => Err(DataError::TypeMismatch { id, expected: "float64" }),
        }
    }
}
