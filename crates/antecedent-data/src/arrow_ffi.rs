//! Isolated Arrow C Data Interface FFI.
//!
//! All `unsafe` for Arrow CDI lives in this module. Safe wrappers validate
//! types and lengths before exposing library-owned views.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    unsafe_code,
    reason = "the isolated Arrow C Data Interface boundary; safe wrappers validate types and lengths first"
)]

use std::sync::Arc;

use antecedent_core::VariableId;
use arrow_array::ffi::{FFI_ArrowArray, from_ffi, to_ffi};
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch, RecordBatchReader};
use arrow_schema::SchemaRef;
use arrow_schema::ffi::FFI_ArrowSchema;

use crate::buffer::{F64Buffer, ForeignBufferOwner, ForeignF64Buffer};
use crate::column::{Float64Column, OwnedColumn, ValidityBitmap, invalid_rows_are_nan};
use crate::error::DataError;
use crate::materialize::{MaterializationReason, materialization_diagnostic};

/// Keeps an Arrow array alive for foreign buffer borrows.
#[derive(Debug)]
struct ArrayOwner {
    /// Held only so the imported Arrow array outlives every borrow of its buffer.
    _array: ArrayRef,
}

impl ForeignBufferOwner for ArrayOwner {}

struct FfiPair {
    array: FFI_ArrowArray,
    schema: FFI_ArrowSchema,
}

/// One column imported from the Arrow C Data Interface.
///
/// The FFI pair is private so a mismatched array/schema cannot be assembled
/// with only safe Rust. Construct via [`Self::from_ffi`] at a compliant
/// exporter boundary, or [`Self::from_arrow_array`] from an Arrow array.
/// A column may hold several chunks. They stay separate until a consumer
/// asks for one array.
pub struct ArrowCColumn {
    /// Column name.
    pub name: String,
    parts: Vec<FfiPair>,
}

impl ArrowCColumn {
    /// Wrap a C Data Interface pair produced by a compliant exporter.
    ///
    /// # Safety
    ///
    /// `array` and `schema` must form a valid Arrow C Data Interface pair.
    /// The wrapper assumes those layouts agree; later type checks cannot
    /// establish the original allocation's validity.
    /// Exported buffers must remain immutable until the imported storage and
    /// all its clones release them. No mutable alias may write to retained
    /// buffers. Cached content identities and fitted preparations rely on this
    /// lifetime premise.
    #[must_use]
    pub unsafe fn from_ffi(
        name: impl Into<String>,
        array: FFI_ArrowArray,
        schema: FFI_ArrowSchema,
    ) -> Self {
        Self { name: name.into(), parts: vec![FfiPair { array, schema }] }
    }

    /// Wrap chunk FFI pairs produced by a compliant exporter.
    ///
    /// # Safety
    ///
    /// Each pair must be a valid Arrow C Data Interface pair for one chunk of
    /// the same column. The same immutability rule as [`Self::from_ffi`] applies
    /// to every chunk.
    ///
    /// # Errors
    ///
    /// When `chunks` is empty.
    pub unsafe fn from_ffi_chunks(
        name: impl Into<String>,
        chunks: Vec<(FFI_ArrowArray, FFI_ArrowSchema)>,
    ) -> Result<Self, DataError> {
        if chunks.is_empty() {
            return Err(DataError::InvalidArgument {
                message: "Arrow CDI column needs ≥1 chunk".into(),
            });
        }
        let parts = chunks.into_iter().map(|(array, schema)| FfiPair { array, schema }).collect();
        Ok(Self { name: name.into(), parts })
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
        Ok(Self {
            name: name.into(),
            parts: vec![FfiPair { array: ffi_array, schema: ffi_schema }],
        })
    }

    /// How many chunks this column still holds.
    #[must_use]
    pub fn n_chunks(&self) -> usize {
        self.parts.len()
    }

    /// Import this CDI column into an Arrow array (takes ownership of FFI structs).
    ///
    /// A column with more than one chunk is not concatenated here.
    ///
    /// # Errors
    ///
    /// When the CDI pair is malformed or the column still has several chunks.
    pub fn into_array(self) -> Result<ArrayRef, DataError> {
        let mut chunks = self.into_chunk_arrays()?;
        if chunks.len() == 1 {
            return chunks.pop().ok_or_else(|| DataError::InvalidArgument {
                message: "Arrow chunk list was empty".into(),
            });
        }
        Err(DataError::InvalidArgument {
            message: "Arrow column has several chunks; import the frame view instead of one array"
                .into(),
        })
    }

    /// Import every chunk. A failure releases the chunks already imported and
    /// the ones not yet imported.
    ///
    /// # Errors
    ///
    /// When a CDI pair is malformed or unsupported.
    pub fn into_chunk_arrays(self) -> Result<Vec<ArrayRef>, DataError> {
        let mut out = Vec::with_capacity(self.parts.len());
        for part in self.parts {
            // SAFETY: only [`Self::from_ffi`], [`Self::from_ffi_chunks`], or
            // [`Self::from_arrow_array`] can construct a pair.
            out.push(unsafe { import_array(part.array, &part.schema) }?);
        }
        Ok(out)
    }
}

/// An Arrow C Stream Interface of record batches.
///
/// Dropping the stream calls its release callback, including when import fails
/// part-way through the batches. The stream is in-process exchange, not a
/// stored file and not a cross-process handle.
pub struct ArrowCStream {
    inner: FFI_ArrowArrayStream,
}

impl ArrowCStream {
    /// Take ownership of a C stream produced by a compliant exporter.
    ///
    /// # Safety
    ///
    /// `stream` must be a valid, unreleased Arrow C Stream Interface.
    /// Its batches must remain immutable until this stream and every array
    /// imported from it are dropped. No mutable alias may write to those buffers.
    #[must_use]
    pub unsafe fn from_ffi(stream: FFI_ArrowArrayStream) -> Self {
        Self { inner: stream }
    }

    /// Export a record-batch reader as a C stream.
    #[must_use]
    pub fn from_reader(reader: Box<dyn RecordBatchReader + Send>) -> Self {
        Self { inner: FFI_ArrowArrayStream::new(reader) }
    }

    /// Import the stream schema and return a reader that yields one batch at a time.
    ///
    /// The stream is released when the reader is dropped.
    pub(crate) fn into_reader(self) -> Result<ArrowArrayStreamReader, DataError> {
        ArrowArrayStreamReader::try_new(self.inner).map_err(|e| DataError::InvalidArgument {
            message: format!("Arrow stream schema import failed: {e}"),
        })
    }

    /// Read every batch, up to 65_536. Chunks are not concatenated.
    ///
    /// # Errors
    ///
    /// When the stream schema or a later batch cannot be imported, or the stream
    /// yields more than 65_536 batches. The stream is released on both success
    /// and failure.
    pub fn into_batches(self) -> Result<(SchemaRef, Vec<RecordBatch>), DataError> {
        let reader = self.into_reader()?;
        let schema = reader.schema();
        let mut batches = Vec::new();
        for batch in reader {
            if batches.len() == MAX_STREAM_BATCHES {
                return Err(DataError::InvalidArgument {
                    message: format!("Arrow stream exceeds {MAX_STREAM_BATCHES} batches"),
                });
            }
            let batch = batch.map_err(|e| DataError::InvalidArgument {
                message: format!("Arrow stream batch import failed: {e}"),
            })?;
            batches.push(batch);
        }
        Ok((schema, batches))
    }
}

/// Most record batches [`ArrowCStream::into_batches`] will read.
const MAX_STREAM_BATCHES: usize = 65_536;

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
pub(crate) unsafe fn import_array(
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
        // SAFETY: `array` owns the buffer for the lifetime of ForeignF64Buffer;
        // the exporter contract keeps those buffers immutable until release.
        let owner = Arc::new(ArrayOwner { _array: array });
        let foreign = unsafe { ForeignF64Buffer::new(owner, ptr, n) };
        debug_assert!(!foreign.as_slice().is_empty() || n == 0);
        let values = F64Buffer::foreign(foreign);
        let borrowed = values.nbytes();
        let col = Float64Column::new(id, values, validity)?;
        // The values are borrowed; only a bitmap built from Arrow null slots is a copy.
        let diag = (validity_copied > 0).then(|| {
            materialization_diagnostic(MaterializationReason::ValidityBitmap, validity_copied)
        });
        Ok((OwnedColumn::Float64(col), borrowed, validity_copied, diag))
    } else {
        let raw = floats.values();
        // `values()` is the physical buffer. Use it only when it is exactly the
        // logical column; a non-zero offset or a longer buffer goes through `value`.
        let values = if floats.offset() == 0 && raw.len() == n {
            let mut values = raw.to_vec();
            for (i, value) in values.iter_mut().enumerate() {
                if floats.is_null(i) {
                    *value = f64::NAN;
                }
            }
            values
        } else {
            let mut values = Vec::with_capacity(n);
            for i in 0..n {
                values.push(if floats.is_null(i) { f64::NAN } else { floats.value(i) });
            }
            values
        };
        let value_bytes = (values.len() * core::mem::size_of::<f64>()) as u64;
        let copied = value_bytes + validity_copied;
        let col = Float64Column::new(id, F64Buffer::owned(Arc::from(values)), validity)?;
        let diag =
            materialization_diagnostic(MaterializationReason::ForeignBufferIncompatible, copied);
        Ok((OwnedColumn::Float64(col), 0, copied, Some(diag)))
    }
}

fn validity_from_arrow(floats: &Float64Array) -> Result<(ValidityBitmap, u64), DataError> {
    validity_from_array(floats)
}

/// Copy an Arrow null bitmap into a library validity bitmap.
///
/// A column with no nulls synthesises an all-valid bitmap and copies nothing.
pub(crate) fn validity_from_array(array: &dyn Array) -> Result<(ValidityBitmap, u64), DataError> {
    let n = array.len();
    if array.null_count() == 0 {
        return Ok((ValidityBitmap::all_valid(n), 0));
    }
    let mut validity_bytes = vec![0u8; n.div_ceil(8)];
    for row in 0..n {
        if !array.is_null(row) {
            validity_bytes[row / 8] |= 1 << (row % 8);
        }
    }
    let copied = u64::try_from(validity_bytes.len()).map_err(|_| DataError::InvalidArgument {
        message: "validity bitmap length does not fit u64".into(),
    })?;
    Ok((ValidityBitmap::from_bytes(validity_bytes, n)?, copied))
}

/// Re-export FFI types for the Python / foreign boundary only.
pub use arrow_array::ffi::FFI_ArrowArray as FfiArrowArray;
pub use arrow_array::ffi_stream::FFI_ArrowArrayStream as FfiArrowArrayStream;
pub use arrow_schema::ffi::FFI_ArrowSchema as FfiArrowSchema;

#[cfg(test)]
mod tests {
    use std::ptr::NonNull;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use arrow_array::Float64Array;
    use arrow_buffer::{Buffer, ScalarBuffer};

    use super::*;
    use crate::arrow_adapter::{ArrowFrameView, tabular_from_arrow_c_columns};
    use crate::column::ColumnView;
    use crate::table::TableView;

    struct Owner {
        flag: Arc<AtomicBool>,
        _data: Box<[f64]>,
    }

    impl Drop for Owner {
        fn drop(&mut self) {
            self.flag.store(true, Ordering::SeqCst);
        }
    }

    fn foreign_f64(flag: &Arc<AtomicBool>, mut data: Box<[f64]>) -> Float64Array {
        let len = data.len();
        let ptr = NonNull::new(data.as_mut_ptr().cast::<u8>()).expect("allocation pointer");
        let nbytes = std::mem::size_of_val(data.as_ref());
        let owner = Arc::new(Owner { flag: Arc::clone(flag), _data: data });
        // SAFETY: `ptr` addresses `owner`'s heap buffer for `nbytes` bytes. `owner`
        // keeps that buffer alive and is the allocation the buffer releases.
        let buffer = unsafe { Buffer::from_custom_allocation(ptr, nbytes, owner) };
        Float64Array::new(ScalarBuffer::new(buffer, 0, len), None)
    }

    #[test]
    fn owner_outlives_the_source_array() {
        let flag = Arc::new(AtomicBool::new(false));
        let array = foreign_f64(&flag, vec![1.0, 2.0, 3.0].into_boxed_slice());
        let column = ArrowCColumn::from_arrow_array("x", &array).unwrap();
        drop(array);
        assert!(!flag.load(Ordering::SeqCst));
        let loaded = tabular_from_arrow_c_columns(vec![column]).unwrap();
        assert!(!flag.load(Ordering::SeqCst));
        let values = loaded.data.float64_slice(antecedent_core::VariableId::from_raw(0)).unwrap();
        assert_eq!(values, &[1.0, 2.0, 3.0]);
        drop(loaded);
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn sliced_float_window_does_not_read_the_prefix() {
        let array = Float64Array::from(vec![9.0, 1.0, 2.0, 3.0]);
        let window = array.slice(1, 3);
        let loaded = tabular_from_arrow_c_columns(vec![
            ArrowCColumn::from_arrow_array("x", &window).unwrap(),
        ])
        .unwrap();
        let values = loaded.data.float64_slice(antecedent_core::VariableId::from_raw(0)).unwrap();
        assert_eq!(values, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn frozen_study_keeps_bytes_after_the_alias_is_mutated() {
        let mut values = vec![1.0_f64, 2.0, 3.0];
        let ptr = NonNull::new(values.as_mut_ptr().cast::<u8>()).expect("value pointer");
        let nbytes = values.len() * core::mem::size_of::<f64>();
        // SAFETY: `ptr` addresses `values` for `nbytes` bytes. The allocation does not
        // free that vector; the test keeps `values` and does not write it while the
        // imported frame is alive.
        let buffer = unsafe { Buffer::from_custom_allocation(ptr, nbytes, Arc::new(())) };
        let array = Float64Array::new(ScalarBuffer::new(buffer, 0, values.len()), None);
        let frame = ArrowFrameView::from_c_columns(vec![
            ArrowCColumn::from_arrow_array("x", &array).unwrap(),
        ])
        .unwrap();
        match &frame.column_chunks(0).unwrap()[0] {
            crate::column::OwnedColumn::Float64(column) => assert!(column.values.is_foreign()),
            _ => panic!("expected float"),
        }
        let frozen = frame.freeze().unwrap();
        drop(frame);
        drop(array);
        values[0] = 9.0;
        match frozen.data().column(antecedent_core::VariableId::from_raw(0)).unwrap() {
            ColumnView::Float64(column) => {
                assert!(!column.values.is_foreign());
                assert_eq!(column.values.as_slice()[0].to_bits(), 1.0_f64.to_bits());
            }
            _ => panic!("expected float"),
        }
        let alias = ArrowCColumn::from_arrow_array("x", &Float64Array::from(vec![9.0])).unwrap();
        let refused = frozen.bind_foreign_alias(alias).unwrap_err();
        assert!(refused.to_string().contains("refuses a mutable foreign alias"));
        match frozen.data().column(antecedent_core::VariableId::from_raw(0)).unwrap() {
            ColumnView::Float64(column) => {
                assert_eq!(column.values.as_slice()[0].to_bits(), 1.0_f64.to_bits());
            }
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn failed_import_releases_the_first_column_owner() {
        use arrow_array::StringArray;

        let flag = Arc::new(AtomicBool::new(false));
        let array = foreign_f64(&flag, vec![1.0, 2.0].into_boxed_slice());
        let good = ArrowCColumn::from_arrow_array("x", &array).unwrap();
        drop(array);
        assert!(!flag.load(Ordering::SeqCst));
        let bad = ArrowCColumn::from_arrow_array("s", &StringArray::from(vec!["no"])).unwrap();
        let loaded = tabular_from_arrow_c_columns(vec![good, bad]);
        assert!(loaded.is_err());
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn stream_batch_count_is_bounded() {
        use arrow_array::{RecordBatch, RecordBatchReader};
        use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};

        use crate::error::DataError;

        struct Endless {
            schema: SchemaRef,
        }
        impl Iterator for Endless {
            type Item = Result<RecordBatch, ArrowError>;
            fn next(&mut self) -> Option<Self::Item> {
                Some(Ok(RecordBatch::new_empty(Arc::clone(&self.schema))))
            }
        }
        impl RecordBatchReader for Endless {
            fn schema(&self) -> SchemaRef {
                Arc::clone(&self.schema)
            }
        }
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
        let err = ArrowCStream::from_reader(Box::new(Endless { schema }))
            .into_batches()
            .expect_err("batch cap");
        match err {
            DataError::InvalidArgument { message } => {
                assert!(message.contains(&MAX_STREAM_BATCHES.to_string()));
            }
            other => panic!("expected the batch cap, got {other}"),
        }
    }
}
