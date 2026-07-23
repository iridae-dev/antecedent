//! Owned tabular storage implementing [`TableView`](crate::table::TableView).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalSchema, VariableId};

use crate::column::{ColumnView, OwnedColumn};
use crate::error::DataError;
use crate::table::TableView;

/// Owned columnar table with optional analysis mask and weights.
#[derive(Clone, Debug)]
pub struct OwnedColumnarStorage {
    schema: CausalSchema,
    columns: Arc<[OwnedColumn]>,
    row_count: usize,
    /// Optional analysis inclusion mask (1 = include), independent of validity.
    analysis_mask: Option<crate::column::ValidityBitmap>,
    /// Optional observation weights.
    weights: Option<Arc<[f64]>>,
}

impl OwnedColumnarStorage {
    /// Build a table from schema-aligned columns.
    ///
    /// # Errors
    ///
    /// Length mismatches, missing schema variables, or duplicate columns.
    pub fn try_new(
        schema: CausalSchema,
        columns: Vec<OwnedColumn>,
        analysis_mask: Option<crate::column::ValidityBitmap>,
        weights: Option<Arc<[f64]>>,
    ) -> Result<Self, DataError> {
        if columns.len() != schema.len() {
            return Err(DataError::LengthMismatch {
                expected: schema.len(),
                actual: columns.len(),
                context: "column count vs schema",
            });
        }
        let row_count = columns.first().map_or(0, OwnedColumn::len);
        for (i, col) in columns.iter().enumerate() {
            let expected_id = VariableId::from_raw(u32::try_from(i).map_err(|_| {
                DataError::InvalidArgument { message: "schema exceeds VariableId range".into() }
            })?);
            if col.id() != expected_id {
                return Err(DataError::UnknownVariable { id: col.id() });
            }
            if col.len() != row_count {
                return Err(DataError::LengthMismatch {
                    expected: row_count,
                    actual: col.len(),
                    context: "column row count",
                });
            }
        }
        if let Some(mask) = &analysis_mask {
            if mask.len() != row_count {
                return Err(DataError::LengthMismatch {
                    expected: row_count,
                    actual: mask.len(),
                    context: "analysis mask",
                });
            }
        }
        if let Some(w) = &weights {
            if w.len() != row_count {
                return Err(DataError::LengthMismatch {
                    expected: row_count,
                    actual: w.len(),
                    context: "weights",
                });
            }
        }
        Ok(Self { schema, columns: Arc::from(columns), row_count, analysis_mask, weights })
    }

    /// Optional analysis mask.
    #[must_use]
    pub fn analysis_mask(&self) -> Option<&crate::column::ValidityBitmap> {
        self.analysis_mask.as_ref()
    }

    /// Optional weights.
    #[must_use]
    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    /// Borrow owned columns in dense id order.
    #[must_use]
    pub fn columns(&self) -> &[OwnedColumn] {
        &self.columns
    }

    /// Shared column Arc (identity for copy-avoidance checks).
    #[must_use]
    pub fn columns_arc(&self) -> &Arc<[OwnedColumn]> {
        &self.columns
    }
}

impl TableView for OwnedColumnarStorage {
    fn schema(&self) -> &CausalSchema {
        &self.schema
    }

    fn row_count(&self) -> usize {
        self.row_count
    }

    fn column(&self, id: VariableId) -> Result<ColumnView<'_>, DataError> {
        self.columns
            .get(id.as_usize())
            .map(OwnedColumn::as_view)
            .ok_or(DataError::UnknownVariable { id })
    }
}
