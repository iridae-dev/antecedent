//! Arrow-backed adapters. Arrow types stay inside this module (ADR 0004).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use arrow_array::{Array, Float64Array, RecordBatch};
use causal_core::{
    CausalSchema, CausalSchemaBuilder, DiagnosticSet, MeasurementSpec, RoleHint, SmallRoleSet,
    ValueType, VariableId,
};

use crate::column::{Float64Column, OwnedColumn, ValidityBitmap};
use crate::dataset::TabularData;
use crate::error::DataError;
use crate::materialize::{MaterializationReason, materialization_diagnostic};
use crate::storage::OwnedColumnarStorage;

/// Result of loading an Arrow [`RecordBatch`] into library-owned storage.
#[derive(Clone, Debug)]
pub struct ArrowLoadResult {
    /// Loaded tabular data.
    pub data: TabularData,
    /// Copy / materialization diagnostics.
    pub diagnostics: DiagnosticSet,
    /// Total bytes copied into owned buffers.
    pub bytes_copied: u64,
}

/// Load float64 columns from an Arrow record batch into [`TabularData`].
///
/// Phase 0 supports contiguous float64 columns. Nullability is translated into
/// [`ValidityBitmap`]. The Arrow arrays are **copied** into library-owned
/// `Arc<[f64]>` buffers so the public API never exposes Arrow types; the copy
/// is recorded in diagnostics (Phase 0 exit criterion).
///
/// # Errors
///
/// Unsupported column types, empty batches, or schema construction failures.
pub fn tabular_from_record_batch(batch: &RecordBatch) -> Result<ArrowLoadResult, DataError> {
    if batch.num_columns() == 0 {
        return Err(DataError::InvalidArgument {
            message: "record batch has no columns".into(),
        });
    }
    let mut builder = CausalSchemaBuilder::new();
    let mut columns = Vec::with_capacity(batch.num_columns());
    let mut diagnostics = DiagnosticSet::new();
    let mut bytes_copied = 0u64;
    let n_rows = batch.num_rows();

    for (i, field) in batch.schema().fields().iter().enumerate() {
        let name = field.name().clone();
        builder
            .add_variable(
                Arc::<str>::from(name),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .map_err(|e| DataError::Schema(e.to_string()))?;

        let array = batch.column(i);
        let floats =
            array.as_any().downcast_ref::<Float64Array>().ok_or(DataError::TypeMismatch {
                id: VariableId::from_raw(u32::try_from(i).expect("col")),
                expected: "float64",
            })?;

        let mut values = Vec::with_capacity(n_rows);
        let mut validity_bytes = vec![0u8; n_rows.div_ceil(8)];
        for row in 0..n_rows {
            if floats.is_null(row) {
                values.push(0.0);
            } else {
                values.push(floats.value(row));
                validity_bytes[row / 8] |= 1 << (row % 8);
            }
        }
        let copied = (values.len() * core::mem::size_of::<f64>() + validity_bytes.len()) as u64;
        bytes_copied += copied;
        diagnostics.push(materialization_diagnostic(
            MaterializationReason::ForeignBufferIncompatible,
            copied,
        ));

        let col = Float64Column::new(
            VariableId::from_raw(u32::try_from(i).expect("col")),
            Arc::<[f64]>::from(values),
            ValidityBitmap::from_bytes(validity_bytes, n_rows)?,
        )?;
        columns.push(OwnedColumn::Float64(col));
    }

    let schema: CausalSchema = builder.build().map_err(|e| DataError::Schema(e.to_string()))?;
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None)?;
    Ok(ArrowLoadResult { data: TabularData::new(storage), diagnostics, bytes_copied })
}

#[cfg(test)]
mod tests {
    use arrow_array::Float64Array;
    use arrow_schema::{DataType, Field, Schema};
    use causal_core::VariableId;

    use super::*;
    use crate::table::TableView;

    #[test]
    fn arrow_load_copies_and_exposes_table_view() {
        let schema = Schema::new(vec![
            Field::new("x", DataType::Float64, true),
            Field::new("y", DataType::Float64, true),
        ]);
        let x = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let y = Float64Array::from(vec![Some(4.0), Some(5.0), Some(6.0)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(x), Arc::new(y)]).unwrap();

        let loaded = tabular_from_record_batch(&batch).unwrap();
        assert!(loaded.bytes_copied > 0);
        assert!(!loaded.diagnostics.is_empty());
        assert_eq!(loaded.data.row_count(), 3);
        let col = loaded.data.column(VariableId::from_raw(0)).unwrap();
        match col {
            crate::column::ColumnView::Float64(c) => {
                assert!(c.validity.is_valid(0));
                assert!(!c.validity.is_valid(1));
                assert!((c.values[2] - 3.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected float"),
        }
    }
}
