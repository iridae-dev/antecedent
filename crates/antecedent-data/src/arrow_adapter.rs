//! Arrow-backed adapters. Arrow types stay inside this module (ADR 0004).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::num::NonZeroU32;
use std::sync::Arc;

use antecedent_core::{
    CausalSchema, CausalSchemaBuilder, DiagnosticSet, MeasurementSpec, RoleHint, ScalarType,
    SmallRoleSet, ValueType, VariableId,
};
use arrow_array::{Array, FixedSizeListArray, Float64Array, RecordBatch};

use crate::arrow_ffi::{ArrowCColumn, float64_column_from_array};
use crate::buffer::F64Buffer;
use crate::column::{FixedVectorColumn, Float64Column, OwnedColumn, ValidityBitmap};
use crate::dataset::TabularData;
use crate::error::DataError;
use crate::materialize::{MaterializationReason, materialization_diagnostic};
use crate::storage::OwnedColumnarStorage;

/// Result of loading Arrow input into library-owned storage.
#[derive(Clone, Debug)]
pub struct ArrowLoadResult {
    /// Loaded tabular data.
    pub data: TabularData,
    /// Copy / materialization diagnostics.
    pub diagnostics: DiagnosticSet,
    /// Total bytes copied into owned buffers.
    pub bytes_copied: u64,
    /// Total bytes borrowed zero-copy from foreign buffers.
    pub bytes_borrowed: u64,
}

/// Load float64 and fixed-size float64-list columns from an Arrow record batch.
///
/// Always copies into library-owned buffers (in-process `RecordBatch` path).
/// Prefer [`tabular_from_arrow_c_columns`] for Arrow C Data Interface zero-copy.
/// Null cells are invalid and hold `NaN`; a fixed-size-list row with a null
/// component is an invalid row.
///
/// # Errors
///
/// Unsupported column types, empty batches, schema construction failures, or more
/// than `u32::MAX` columns.
pub fn tabular_from_record_batch(batch: &RecordBatch) -> Result<ArrowLoadResult, DataError> {
    if batch.num_columns() == 0 {
        return Err(DataError::InvalidArgument { message: "record batch has no columns".into() });
    }
    let mut builder = CausalSchemaBuilder::new();
    let mut columns = Vec::with_capacity(batch.num_columns());
    let mut diagnostics = DiagnosticSet::new();
    let mut bytes_copied = 0u64;
    let n_rows = batch.num_rows();

    for (i, field) in batch.schema().fields().iter().enumerate() {
        let name = field.name().clone();
        let id = VariableId::from_raw(u32::try_from(i).map_err(|_| {
            DataError::InvalidArgument { message: "too many Arrow columns for VariableId".into() }
        })?);
        let array = batch.column(i);

        if let Some(floats) = array.as_any().downcast_ref::<Float64Array>() {
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
            let (col, copied) = float64_owned_from_array(id, floats, n_rows)?;
            bytes_copied += copied;
            diagnostics.push(materialization_diagnostic(
                MaterializationReason::ForeignBufferIncompatible,
                copied,
            ));
            columns.push(OwnedColumn::Float64(col));
            continue;
        }

        if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
            let dim = usize::try_from(list.value_length()).map_err(|_| {
                DataError::InvalidArgument { message: "FixedSizeList width must fit usize".into() }
            })?;
            if dim == 0 {
                return Err(DataError::InvalidArgument {
                    message: "FixedSizeList width must be > 0".into(),
                });
            }
            let width = NonZeroU32::new(u32::try_from(dim).map_err(|_| {
                DataError::InvalidArgument { message: "FixedSizeList width must fit u32".into() }
            })?)
            .ok_or(DataError::InvalidArgument {
                message: "FixedSizeList width must be > 0".into(),
            })?;
            builder
                .add_variable(
                    Arc::<str>::from(name),
                    ValueType::Vector { width, element: ScalarType::Float64 },
                    SmallRoleSet::from_hint(RoleHint::Context),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .map_err(|e| DataError::Schema(e.to_string()))?;
            let (col, copied) = fixed_vector_from_list(id, list, n_rows, dim)?;
            bytes_copied += copied;
            diagnostics.push(materialization_diagnostic(
                MaterializationReason::ForeignBufferIncompatible,
                copied,
            ));
            columns.push(OwnedColumn::FixedVector(col));
            continue;
        }

        return Err(DataError::TypeMismatch { id, expected: "float64 or FixedSizeList<float64>" });
    }

    let schema: CausalSchema = builder.build().map_err(|e| DataError::Schema(e.to_string()))?;
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None)?;
    Ok(ArrowLoadResult {
        data: TabularData::new(storage),
        diagnostics,
        bytes_copied,
        bytes_borrowed: 0,
    })
}

fn float64_owned_from_array(
    id: VariableId,
    floats: &Float64Array,
    n_rows: usize,
) -> Result<(Float64Column, u64), DataError> {
    let mut values = Vec::with_capacity(n_rows);
    let mut validity_bytes = vec![0u8; n_rows.div_ceil(8)];
    for row in 0..n_rows {
        if floats.is_null(row) {
            // Missing cells hold NaN so value-only readers never see an observation.
            values.push(f64::NAN);
        } else {
            values.push(floats.value(row));
            validity_bytes[row / 8] |= 1 << (row % 8);
        }
    }
    let copied = (values.len() * core::mem::size_of::<f64>() + validity_bytes.len()) as u64;
    let col = Float64Column::new(
        id,
        F64Buffer::owned(Arc::from(values)),
        ValidityBitmap::from_bytes(validity_bytes, n_rows)?,
    )?;
    Ok((col, copied))
}

fn fixed_vector_from_list(
    id: VariableId,
    list: &FixedSizeListArray,
    n_rows: usize,
    dim: usize,
) -> Result<(FixedVectorColumn, u64), DataError> {
    let values = list.values();
    let floats = values
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or(DataError::TypeMismatch { id, expected: "FixedSizeList<float64>" })?;
    // Validity is per row, so a vector with any null component is a missing row;
    // every component of a missing row holds NaN.
    let mut flat = Vec::with_capacity(n_rows.saturating_mul(dim));
    let mut validity_bytes = vec![0u8; n_rows.div_ceil(8)];
    for row in 0..n_rows {
        let start = row.saturating_mul(dim);
        if list.is_null(row) || (start..start + dim).any(|idx| floats.is_null(idx)) {
            flat.extend(std::iter::repeat_n(f64::NAN, dim));
            continue;
        }
        validity_bytes[row / 8] |= 1 << (row % 8);
        flat.extend((start..start + dim).map(|idx| floats.value(idx)));
    }
    let copied = (flat.len() * core::mem::size_of::<f64>() + validity_bytes.len()) as u64;
    let col = FixedVectorColumn::new(
        id,
        dim,
        Arc::from(flat),
        ValidityBitmap::from_bytes(validity_bytes, n_rows)?,
    )?;
    Ok((col, copied))
}

/// Load float64 columns from Arrow C Data Interface exports, preferring zero-copy.
///
/// Consumes each [`ArrowCColumn`]'s FFI structs. Contiguous float64 value buffers
/// are borrowed; validity bitmaps are copied into library storage. A column with
/// nulls is borrowed only when every null slot already holds `NaN`; otherwise its
/// values are copied with `NaN` written under each null.
///
/// # Errors
///
/// Empty input, non-float64 columns, CDI import failure, schema errors, or more
/// than `u32::MAX` columns.
pub fn tabular_from_arrow_c_columns(
    columns: Vec<ArrowCColumn>,
) -> Result<ArrowLoadResult, DataError> {
    if columns.is_empty() {
        return Err(DataError::InvalidArgument {
            message: "Arrow CDI import needs ≥1 column".into(),
        });
    }
    let mut builder = CausalSchemaBuilder::new();
    let mut owned_cols = Vec::with_capacity(columns.len());
    let mut diagnostics = DiagnosticSet::new();
    let mut bytes_copied = 0u64;
    let mut bytes_borrowed = 0u64;
    let mut n_rows = None;

    for (i, col) in columns.into_iter().enumerate() {
        let name = col.name.clone();
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

        let array = col.into_array()?;
        if let Some(n) = n_rows {
            if array.len() != n {
                return Err(DataError::LengthMismatch {
                    expected: n,
                    actual: array.len(),
                    context: "Arrow CDI column lengths",
                });
            }
        } else {
            n_rows = Some(array.len());
        }

        let id = VariableId::from_raw(u32::try_from(i).map_err(|_| {
            DataError::InvalidArgument { message: "too many Arrow columns for VariableId".into() }
        })?);
        let (owned, borrowed, copied, diag) = float64_column_from_array(id, array)?;
        bytes_borrowed += borrowed;
        bytes_copied += copied;
        if let Some(diag) = diag {
            diagnostics.push(diag);
        }
        owned_cols.push(owned);
    }

    let schema: CausalSchema = builder.build().map_err(|e| DataError::Schema(e.to_string()))?;
    let storage = OwnedColumnarStorage::try_new(schema, owned_cols, None, None)?;
    Ok(ArrowLoadResult {
        data: TabularData::new(storage),
        diagnostics,
        bytes_copied,
        bytes_borrowed,
    })
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // values are copied bit-for-bit
mod tests {
    use antecedent_core::VariableId;
    use arrow_array::Float64Array;
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::arrow_ffi::ArrowCColumn;
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
        assert_eq!(loaded.bytes_borrowed, 0);
        assert!(!loaded.diagnostics.is_empty());
        assert_eq!(loaded.data.row_count(), 3);
        let col = loaded.data.column(VariableId::from_raw(0)).unwrap();
        match col {
            crate::column::ColumnView::Float64(c) => {
                assert!(c.validity.is_valid(0));
                assert!(!c.validity.is_valid(1));
                assert!((c.values[2] - 3.0).abs() < f64::EPSILON);
                assert!(!c.values.is_foreign());
            }
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn arrow_load_fixed_size_list_float64() {
        use arrow_array::FixedSizeListArray;
        use arrow_buffer::NullBuffer;

        let values = Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let list = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float64, true)),
            2,
            Arc::new(values),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let schema = Schema::new(vec![Field::new(
            "v",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, true)), 2),
            true,
        )]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list)]).unwrap();
        let loaded = tabular_from_record_batch(&batch).unwrap();
        assert_eq!(loaded.data.row_count(), 3);
        match loaded.data.column(VariableId::from_raw(0)).unwrap() {
            crate::column::ColumnView::FixedVector(c) => {
                assert_eq!(c.dim, 2);
                assert!(c.validity.is_valid(0));
                assert!(!c.validity.is_valid(1));
                assert!(c.validity.is_valid(2));
                assert!((c.values[0] - 1.0).abs() < f64::EPSILON);
                assert!((c.values[1] - 2.0).abs() < f64::EPSILON);
                assert!((c.values[4] - 5.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected FixedVector"),
        }
    }

    fn float_col(data: &TabularData, id: u32) -> &Float64Column {
        match data.column(VariableId::from_raw(id)).unwrap() {
            crate::column::ColumnView::Float64(c) => c,
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn arrow_null_float64_cell_reads_nan_on_every_path() {
        let schema = Schema::new(vec![Field::new("x", DataType::Float64, true)]);
        let x = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(x)]).unwrap();
        let loaded = tabular_from_record_batch(&batch).unwrap();
        let id = VariableId::from_raw(0);
        assert!(!float_col(&loaded.data, 0).validity.is_valid(1));
        let slice = loaded.data.float64_slice(id).unwrap();
        assert!(slice[1].is_nan(), "borrowed slice exposed {} for a null cell", slice[1]);
        assert!(loaded.data.float64_cow(id).unwrap()[1].is_nan());
        let values = loaded.data.float64_values(id).unwrap();
        assert!(values[1].is_nan());
        assert_eq!((values[0], values[2]), (1.0, 3.0));
    }

    #[test]
    fn arrow_cdi_nulls_copy_with_nan_when_slots_hold_placeholders() {
        // arrow-rs writes 0.0 under null slots, like most exporters.
        let x = Float64Array::from(vec![Some(1.0), None, Some(3.0)]);
        assert_eq!(x.values()[1], 0.0);
        let loaded =
            tabular_from_arrow_c_columns(vec![ArrowCColumn::from_arrow_array("x", &x).unwrap()])
                .unwrap();
        assert_eq!(loaded.bytes_borrowed, 0);
        let c = float_col(&loaded.data, 0);
        assert!(!c.values.is_foreign());
        assert!(!c.validity.is_valid(1));
        assert!(c.values[1].is_nan());
        assert!(loaded.data.float64_slice(VariableId::from_raw(0)).unwrap()[1].is_nan());
        assert!((c.values[2] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn arrow_cdi_nulls_stay_zero_copy_when_slots_hold_nan() {
        use arrow_buffer::{NullBuffer, ScalarBuffer};
        let x = Float64Array::new(
            ScalarBuffer::from(vec![1.0, f64::NAN, 3.0]),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let loaded =
            tabular_from_arrow_c_columns(vec![ArrowCColumn::from_arrow_array("x", &x).unwrap()])
                .unwrap();
        assert!(loaded.bytes_borrowed > 0);
        let c = float_col(&loaded.data, 0);
        assert!(c.values.is_foreign());
        assert!(!c.validity.is_valid(1));
        assert!(c.values[1].is_nan());
    }

    #[test]
    fn arrow_fixed_size_list_null_component_invalidates_row() {
        use arrow_array::FixedSizeListArray;
        use arrow_buffer::NullBuffer;

        let values =
            Float64Array::from(vec![Some(1.0), Some(2.0), Some(3.0), None, Some(5.0), Some(6.0)]);
        let item = Arc::new(Field::new("item", DataType::Float64, true));
        let list = FixedSizeListArray::new(
            Arc::clone(&item),
            2,
            Arc::new(values),
            Some(NullBuffer::from(vec![true, true, false])),
        );
        let schema = Schema::new(vec![Field::new("v", DataType::FixedSizeList(item, 2), true)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list)]).unwrap();
        let loaded = tabular_from_record_batch(&batch).unwrap();
        match loaded.data.column(VariableId::from_raw(0)).unwrap() {
            crate::column::ColumnView::FixedVector(c) => {
                assert!(c.validity.is_valid(0));
                assert_eq!(&c.values[0..2], &[1.0, 2.0]);
                // Row 1 has a null component; row 2 is a null list slot.
                assert!(!c.validity.is_valid(1));
                assert!(!c.validity.is_valid(2));
                assert!(c.values[2..].iter().all(|v| v.is_nan()));
            }
            _ => panic!("expected FixedVector"),
        }
    }

    #[test]
    fn arrow_cdi_zero_copy_borrows_values() {
        let x = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let y = Float64Array::from(vec![4.0, 5.0, 6.0]);
        let loaded = tabular_from_arrow_c_columns(vec![
            ArrowCColumn::from_arrow_array("x", &x).unwrap(),
            ArrowCColumn::from_arrow_array("y", &y).unwrap(),
        ])
        .unwrap();
        assert!(loaded.bytes_borrowed > 0);
        assert_eq!(loaded.data.row_count(), 3);
        let col = loaded.data.column(VariableId::from_raw(0)).unwrap();
        match col {
            crate::column::ColumnView::Float64(c) => {
                assert!(c.values.is_foreign());
                assert!((c.values[0] - 1.0).abs() < f64::EPSILON);
                assert!((c.values[2] - 3.0).abs() < f64::EPSILON);
            }
            _ => panic!("expected float"),
        }
    }
}
