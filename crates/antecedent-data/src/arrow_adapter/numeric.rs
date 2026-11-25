//! Numeric Arrow columns widened directly into their final table buffers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{DiagnosticSet, VariableId};
use arrow_array::types::{ArrowPrimitiveType, Float16Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float16Array, Float32Array, Float64Array, Int8Array, Int16Array,
    Int32Array, Int64Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::DataType;

use super::ArrowLoadResult;
use super::chunks::{checked_row_total, variable_id};
use super::finish_load;
use crate::arrow_ffi::{ArrowCColumn, float64_column_from_array};
use crate::column::{Float64Column, OwnedColumn, ValidityBitmap};
use crate::error::DataError;
use crate::materialize::{MaterializationReason, materialization_diagnostic};

const EXACT_INT_LIMIT: i64 = 1 << 53;
type Float16 = <Float16Type as ArrowPrimitiveType>::Native;

/// Load continuous numeric columns, optionally borrowing a contiguous double buffer.
///
/// The columns keep their individual chunk layouts until the final buffer is filled.
/// Integer values must fit the exact range of float64.
///
/// # Errors
///
/// Malformed Arrow exports, unsupported or conflicting chunk types, unequal column lengths,
/// or integers outside the exact range.
pub fn load_numeric_columns(
    columns: Vec<ArrowCColumn>,
    retain: bool,
) -> Result<ArrowLoadResult, DataError> {
    if columns.is_empty() {
        return Err(DataError::empty("Arrow CDI import columns"));
    }
    let mut names = Vec::with_capacity(columns.len());
    let mut owned = Vec::with_capacity(columns.len());
    let mut diagnostics = DiagnosticSet::new();
    let mut bytes_copied = 0_u64;
    let mut bytes_borrowed = 0_u64;
    for (position, column) in columns.into_iter().enumerate() {
        names.push(column.name.clone());
        let id = variable_id(position)?;
        let mut arrays = column.into_chunk_arrays()?;
        if !retain && arrays.len() == 1 && arrays[0].data_type() == &DataType::Float64 {
            let array = arrays.pop().ok_or_else(|| DataError::empty("Arrow column chunks"))?;
            let (column, borrowed, copied, diag) = float64_column_from_array(id, array)?;
            bytes_copied += copied;
            bytes_borrowed += borrowed;
            if let Some(diag) = diag {
                diagnostics.push(diag);
            }
            owned.push(column);
        } else {
            let name = &names[position];
            let (column, copied) = own_numeric(id, name, &arrays)?;
            bytes_copied += copied;
            diagnostics.push(materialization_diagnostic(
                MaterializationReason::ForeignBufferIncompatible,
                copied,
            ));
            owned.push(OwnedColumn::Float64(column));
        }
    }
    finish_load(&names, owned, diagnostics, bytes_copied, bytes_borrowed)
}

#[allow(
    clippy::cast_precision_loss,
    reason = "64-bit integers are validated against the exact float64 interval before casting"
)]
fn own_numeric(
    id: VariableId,
    name: &str,
    arrays: &[ArrayRef],
) -> Result<(Float64Column, u64), DataError> {
    let first = arrays.first().ok_or_else(|| DataError::empty("Arrow column chunks"))?;
    if arrays.len() == 1
        && let Some(floats) = first.as_any().downcast_ref::<Float64Array>()
        && floats.null_count() == 0
    {
        let values = Arc::<[f64]>::from(&floats.values()[..]);
        let bytes = std::mem::size_of_val(values.as_ref()) as u64;
        return Ok((
            Float64Column::new(id, values, ValidityBitmap::all_valid(floats.len()))?,
            bytes,
        ));
    }
    let rows = checked_row_total(arrays.iter().map(Array::len))?;
    macro_rules! widen {
        ($array:ty, $convert:expr $(, $check:expr)?) => {{
            let chunks = arrays
                .iter()
                .map(|array| {
                    array.as_any().downcast_ref::<$array>().ok_or_else(|| {
                        DataError::mismatch("one numeric type across all Arrow column chunks")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            $(
                if chunks.iter().any(|array| array.iter().flatten().any($check)) {
                    return Err(DataError::external(
                        "Arrow integer widening",
                        format!("column {name:?} holds a value beyond 2**53"),
                    ));
                }
            )?
            let convert = $convert;
            own_values(
                id,
                rows,
                chunks.len() > 1 || chunks.iter().any(|array| array.null_count() != 0),
                chunks.iter().flat_map(|array| array.iter().map(|v| v.map(convert))),
            )
        }};
    }
    match first.data_type() {
        DataType::Boolean => widen!(BooleanArray, |v| f64::from(u8::from(v))),
        DataType::Float16 => widen!(Float16Array, |v: Float16| v.to_f64()),
        DataType::Float32 => widen!(Float32Array, f64::from),
        DataType::Float64 => widen!(Float64Array, |v| v),
        DataType::Int8 => widen!(Int8Array, f64::from),
        DataType::Int16 => widen!(Int16Array, f64::from),
        DataType::Int32 => widen!(Int32Array, f64::from),
        DataType::Int64 => {
            widen!(Int64Array, |v| v as f64, |v| !(-EXACT_INT_LIMIT..=EXACT_INT_LIMIT).contains(&v))
        }
        DataType::UInt8 => widen!(UInt8Array, f64::from),
        DataType::UInt16 => widen!(UInt16Array, f64::from),
        DataType::UInt32 => widen!(UInt32Array, f64::from),
        DataType::UInt64 => widen!(UInt64Array, |v| v as f64, |v| v > EXACT_INT_LIMIT as u64),
        _ => Err(DataError::TypeMismatch { id, expected: "numeric Arrow values" }),
    }
}

fn own_values(
    id: VariableId,
    rows: usize,
    copy_mask: bool,
    mut values: impl Iterator<Item = Option<f64>>,
) -> Result<(Float64Column, u64), DataError> {
    let mut validity = if copy_mask { vec![0u8; rows.div_ceil(8)] } else { Vec::new() };
    // Each chunk's full length is included in `rows`. The range supplies that exact length
    // to Arc's iterator collector, so all values go straight into the retained slice.
    let values: Arc<[f64]> = (0..rows)
        .map(|row| match values.next().flatten() {
            Some(value) => {
                if copy_mask {
                    validity[row / 8] |= 1 << (row % 8);
                }
                value
            }
            None => f64::NAN,
        })
        .collect();
    let copied = (std::mem::size_of_val(values.as_ref()) + validity.len()) as u64;
    let validity = if copy_mask {
        ValidityBitmap::from_bytes(validity, rows)?
    } else {
        ValidityBitmap::all_valid(rows)
    };
    Ok((Float64Column::new(id, values, validity)?, copied))
}
