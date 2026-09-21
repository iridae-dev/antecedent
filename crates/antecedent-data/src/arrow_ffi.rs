//! Isolated Arrow C Data Interface FFI.
//!
//! All `unsafe` for Arrow CDI lives in this module. Safe wrappers validate
//! types and lengths before exposing library-owned views.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unsafe_code)]

use std::sync::Arc;

use antecedent_core::VariableId;
use arrow_array::ffi::{FFI_ArrowArray, from_ffi, to_ffi};
use arrow_array::{Array, ArrayRef, Float64Array};
use arrow_schema::ffi::FFI_ArrowSchema;

use crate::buffer::{F64Buffer, ForeignF64Buffer};
use crate::column::{Float64Column, OwnedColumn, ValidityBitmap, invalid_rows_are_nan};
use crate::error::DataError;
use crate::materialize::{MaterializationReason, materialization_diagnostic};

/// Keeps an Arrow array alive for foreign buffer borrows.
#[derive(Debug)]
struct ArrayOwner(#[allow(dead_code)] ArrayRef);

/// One column imported from the Arrow C Data Interface.
///
/// The FFI pair is private so a mismatched array/schema cannot be assembled
/// with only safe Rust. Construct via [`Self::from_ffi`] at a compliant
/// exporter boundary, or [`Self::from_arrow_array`] from an Arrow array.
pub struct ArrowCColumn {
    /// Column name.
    pub name: String,
    array: FFI_ArrowArray,
    schema: FFI_ArrowSchema,
}

impl ArrowCColumn {
    /// Wrap a C Data Interface pair produced by a compliant exporter.
    ///
    /// # Safety
    ///
    /// `array` and `schema` must form a valid Arrow C Data Interface pair.
    /// The wrapper assumes those layouts agree; later type checks cannot
    /// establish the original allocation's validity.
    #[must_use]
    pub unsafe fn from_ffi(
        name: impl Into<String>,
        array: FFI_ArrowArray,
        schema: FFI_ArrowSchema,
    ) -> Self {
        Self { name: name.into(), array, schema }
    }

    /// Wrap an already-validated owned Arrow array as a CDI column.
    ///
    /// # Errors
    ///
    /// When Arrow cannot export the array through the C Data Interface.
    pub fn from_arrow_array(name: impl Into<String>, array: &dyn Array) -> Result<Self, DataError> {
        let (ffi_array, ffi_schema) = to_ffi(&array.to_data()).map_err(|e| {
            DataError::InvalidArgument { message: format!("Arrow CDI export failed: {e}") }
        })?;
        Ok(Self { name: name.into(), array: ffi_array, schema: ffi_schema })
    }

    /// Import this CDI column into an Arrow array (takes ownership of FFI structs).
    ///
    /// # Errors
    ///
    /// When the CDI pair is malformed or unsupported.
    pub fn into_array(self) -> Result<ArrayRef, DataError> {
        // SAFETY: only [`Self::from_ffi`] (unsafe) or [`Self::from_arrow_array`]
        // (exports a consistent pair) can construct this type.
        let Self { array, schema, .. } = self;
        unsafe { import_array(array, &schema) }
    }
}

/// Import a single Arrow array via CDI.
///
/// # Safety
///
/// `array` and `schema` must form a valid Arrow C Data Interface pair produced
/// by a compliant exporter. This function takes ownership and will release them.
///
/// # Errors
///
/// Returns [`DataError::InvalidArgument`] when the FFI pair cannot be decoded.
pub unsafe fn import_array(
    array: FFI_ArrowArray,
    schema: &FFI_ArrowSchema,
) -> Result<ArrayRef, DataError> {
    // SAFETY: caller guarantees a valid CDI pair; from_ffi releases on drop/error.
    let data = unsafe { from_ffi(array, schema) }.map_err(|e| DataError::InvalidArgument {
        message: format!("Arrow CDI import failed: {e}"),
    })?;
    Ok(arrow_array::make_array(data))
}

/// Convert a float64 Arrow array into a library column, preferring zero-copy.
pub(crate) fn float64_column_from_array(
    id: VariableId,
    array: ArrayRef,
) -> Result<(OwnedColumn, u64, u64, Option<antecedent_core::Diagnostic>), DataError> {
    let floats = array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or(DataError::TypeMismatch { id, expected: "float64" })?;
    let n = floats.len();
    let (validity, validity_copied) = validity_from_arrow(floats)?;

    let values_slice = floats.values().as_ref();
    let aligned = values_slice.as_ptr() as usize % core::mem::align_of::<f64>() == 0;
    // Arrow leaves the value under a null slot unspecified (exporters commonly
    // write 0.0). Borrow only when every null slot already holds NaN; otherwise
    // copy and write NaN there, so a borrowed slice never shows a missing cell
    // as an observation.
    let can_borrow =
        aligned && values_slice.len() == n && invalid_rows_are_nan(values_slice, &validity);

    if can_borrow {
        let ptr = values_slice.as_ptr();
        // SAFETY: `array` owns the buffer for the lifetime of ForeignF64Buffer.
        let foreign = unsafe { ForeignF64Buffer::new(Arc::new(ArrayOwner(array)), ptr, n) };
        // Ensure owner holds the array (read through for dead_code).
        debug_assert!(!foreign.as_slice().is_empty() || n == 0);
        let _ = foreign.len();
        let values = F64Buffer::foreign(foreign);
        let borrowed = values.nbytes();
        let col = Float64Column::new(id, values, validity)?;
        // The values are borrowed; only a bitmap built from Arrow null slots is a copy.
        let diag = (validity_copied > 0).then(|| {
            materialization_diagnostic(MaterializationReason::ValidityBitmap, validity_copied)
        });
        Ok((OwnedColumn::Float64(col), borrowed, validity_copied, diag))
    } else {
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            values.push(if floats.is_null(i) { f64::NAN } else { floats.value(i) });
        }
        let value_bytes = (values.len() * core::mem::size_of::<f64>()) as u64;
        let copied = value_bytes + validity_copied;
        let col = Float64Column::new(id, F64Buffer::owned(Arc::from(values)), validity)?;
        let diag =
            materialization_diagnostic(MaterializationReason::ForeignBufferIncompatible, copied);
        Ok((OwnedColumn::Float64(col), 0, copied, Some(diag)))
    }
}

fn validity_from_arrow(floats: &Float64Array) -> Result<(ValidityBitmap, u64), DataError> {
    let n = floats.len();
    if floats.null_count() == 0 {
        // Synthesised locally: nothing is copied from the Arrow buffers.
        return Ok((ValidityBitmap::all_valid(n), 0));
    }
    let mut validity_bytes = vec![0u8; n.div_ceil(8)];
    for row in 0..n {
        if !floats.is_null(row) {
            validity_bytes[row / 8] |= 1 << (row % 8);
        }
    }
    let copied = validity_bytes.len() as u64;
    Ok((ValidityBitmap::from_bytes(validity_bytes, n)?, copied))
}

/// Re-export FFI types for the Python / foreign boundary only.
pub use arrow_array::ffi::FFI_ArrowArray as FfiArrowArray;
pub use arrow_schema::ffi::FFI_ArrowSchema as FfiArrowSchema;
