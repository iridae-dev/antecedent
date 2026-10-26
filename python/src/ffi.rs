//! The only `unsafe` in the Python extension.
//!
//! Arrow C Data Interface capsules and read-only numeric buffer protocols. Every
//! other module in this crate stays safe.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unsafe_code)]

use antecedent_data::arrow_adapter::ArrowFrameView;
use antecedent_data::arrow_ffi::{ArrowCStream, FfiArrowArrayStream};
use antecedent_data::{ArrowCColumn, FfiArrowArray, FfiArrowSchema};
use pyo3::exceptions::PyBufferError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::{CausalDataError, PosteriorArtifact};

/// Read-only buffers retain their exporting Python owner until CPython releases them.
/// The two callers below own immutable storage whose allocation cannot move.
unsafe fn readonly_numeric_buffer(
    owner: Bound<'_, PyAny>,
    data: *const std::ffi::c_void,
    len: usize,
    format_code: &std::ffi::CStr,
    view: *mut pyo3::ffi::Py_buffer,
    flags: std::os::raw::c_int,
) -> PyResult<()> {
    use pyo3::ffi;
    if view.is_null() {
        return Err(PyBufferError::new_err("View is null"));
    }
    if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
        return Err(PyBufferError::new_err("native numeric views are read-only"));
    }
    let n = isize::try_from(len).map_err(|_| PyBufferError::new_err("buffer too large"))?;
    let itemsize = isize::try_from(std::mem::size_of::<f64>())
        .map_err(|_| PyBufferError::new_err("f64 size does not fit isize"))?;
    let nbytes =
        n.checked_mul(itemsize).ok_or_else(|| PyBufferError::new_err("buffer size overflow"))?;
    let format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
        format_code.to_owned().into_raw()
    } else {
        std::ptr::null_mut()
    };
    // SAFETY: both callers derive data from owner's immutable retained allocation.
    // The owned reference installed in obj pins that allocation until release.
    // No writable buffer is exported, and shape/strides have exactly one dimension.
    unsafe {
        (*view).obj = owner.as_ptr();
        ffi::Py_INCREF((*view).obj);
        (*view).buf = data.cast_mut();
        (*view).len = nbytes;
        (*view).readonly = 1;
        (*view).itemsize = itemsize;
        (*view).format = format;
        (*view).ndim = 1;
        (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
            Box::into_raw(Box::new([n])).cast()
        } else {
            std::ptr::null_mut()
        };
        (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
            Box::into_raw(Box::new([itemsize])).cast()
        } else {
            std::ptr::null_mut()
        };
        (*view).suboffsets = std::ptr::null_mut();
        (*view).internal = std::ptr::null_mut();
    }
    Ok(())
}

unsafe fn release_readonly_buffer(view: *mut pyo3::ffi::Py_buffer) {
    if view.is_null() {
        return;
    }
    // SAFETY: these optional allocations were created only by readonly_numeric_buffer.
    // CPython calls release exactly once for each successfully acquired view.
    unsafe {
        if !(*view).format.is_null() {
            drop(std::ffi::CString::from_raw((*view).format));
            (*view).format = std::ptr::null_mut();
        }
        if !(*view).shape.is_null() {
            drop(Box::from_raw((*view).shape.cast::<[isize; 1]>()));
            (*view).shape = std::ptr::null_mut();
        }
        if !(*view).strides.is_null() {
            drop(Box::from_raw((*view).strides.cast::<[isize; 1]>()));
            (*view).strides = std::ptr::null_mut();
        }
    }
}

/// Zero-copy float64 view of posterior draws.
#[pymethods]
impl PosteriorArtifact {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        let borrowed = slf.borrow();
        // SAFETY: PosteriorArtifact owns immutable draws and is retained by the view.
        unsafe {
            readonly_numeric_buffer(
                slf.clone().into_any(),
                borrowed.draws.as_ptr().cast(),
                borrowed.draws.len(),
                c"d",
                view,
                flags,
            )
        }
    }

    #[allow(
        clippy::unused_self,
        reason = "CPython buffer-release protocol requires an instance receiver"
    )]
    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        // SAFETY: CPython passes the buffer initialized by __getbuffer__.
        unsafe {
            release_readonly_buffer(view);
        }
    }
}

/// Read-only int64 views pin the admitted nanosecond index allocation.
#[pymethods]
impl crate::data_snapshot::IndexSnapshot {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        let borrowed = slf.borrow();
        // SAFETY: the owner retains immutable, aligned i64 values; dtype q and f64 have the same item width.
        unsafe {
            readonly_numeric_buffer(
                slf.clone().into_any(),
                borrowed.values.as_ptr().cast(),
                borrowed.values.len(),
                c"q",
                view,
                flags,
            )
        }
    }
    #[allow(
        clippy::unused_self,
        reason = "CPython buffer-release protocol requires an instance receiver"
    )]
    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        // SAFETY: CPython returns the view initialized by __getbuffer__ exactly once.
        unsafe {
            release_readonly_buffer(view);
        }
    }
}

/// Snapshot columns share immutable native table allocations, including across branches.
#[pymethods]
impl crate::data_snapshot::SnapshotColumn {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        let borrowed = slf.borrow();
        // SAFETY: SnapshotColumn pins the table and its immutable F64Buffer.
        let values = borrowed.values()?;
        unsafe {
            readonly_numeric_buffer(
                slf.clone().into_any(),
                values.as_ptr().cast(),
                values.len(),
                c"d",
                view,
                flags,
            )
        }
    }

    #[allow(
        clippy::unused_self,
        reason = "CPython buffer-release protocol requires an instance receiver"
    )]
    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        // SAFETY: CPython passes the buffer initialized by __getbuffer__.
        unsafe {
            release_readonly_buffer(view);
        }
    }
}

/// Columns from Arrow C Data Interface exporters, in `names` order.
///
/// A length mismatch is `CausalDataError` with the message both Arrow entry
/// points already used. The NumPy path keeps its own message.
pub(crate) fn arrow_c_columns(
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
) -> PyResult<Vec<ArrowCColumn>> {
    if names.len() != columns.len() {
        return Err(CausalDataError::new_err("names and columns length mismatch"));
    }
    let mut cdi_cols = Vec::with_capacity(columns.len());
    for (name, obj) in names.into_iter().zip(columns) {
        if let Some(chunks) = take_chunk_pairs(&obj)? {
            // SAFETY: each pair was moved out of a compliant exporter by `take_arrow_c_array`.
            let column = unsafe { ArrowCColumn::from_ffi_chunks(name, chunks) }
                .map_err(|error| CausalDataError::new_err(error.to_string()))?;
            cdi_cols.push(column);
            continue;
        }
        let (array, schema) = take_arrow_c_array(&obj)?;
        // SAFETY: `take_arrow_c_array` moved the exporter's CDI structs out and
        // left released empty structs in the capsules.
        cdi_cols.push(unsafe { ArrowCColumn::from_ffi(name, array, schema) });
    }
    Ok(cdi_cols)
}

/// The chunks of a column that keeps several, each moved out as a CDI pair.
///
/// A column object exposing `chunks` (several Arrow arrays, each an `__arrow_c_array__`
/// exporter) is imported chunk by chunk: the loader joins them into the table's one owned
/// buffer, a single copy, so the caller never concatenates them first. A column with fewer
/// than two chunks, or any other object, takes the single-array route.
fn take_chunk_pairs(
    obj: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<(FfiArrowArray, FfiArrowSchema)>>> {
    if obj.cast::<pyo3::types::PyTuple>().is_ok() {
        return Ok(None);
    }
    let Ok(chunks) = obj.getattr("chunks") else { return Ok(None) };
    let Ok(iter) = chunks.try_iter() else { return Ok(None) };
    let items = iter.collect::<PyResult<Vec<Bound<'_, PyAny>>>>()?;
    if items.len() < 2 || !items.iter().all(|c| c.hasattr("__arrow_c_array__").unwrap_or(false)) {
        return Ok(None);
    }
    items.iter().map(take_arrow_c_array).collect::<PyResult<Vec<_>>>().map(Some)
}

/// Move CDI structs out of an `__arrow_c_array__` exporter or a `(schema, array)` capsule pair.
fn take_arrow_c_array(obj: &Bound<'_, PyAny>) -> PyResult<(FfiArrowArray, FfiArrowSchema)> {
    if let Ok(tuple) = obj.cast::<pyo3::types::PyTuple>() {
        if tuple.len() == 2 && tuple.get_item(0)?.is_instance_of::<PyCapsule>() {
            return move_array_capsules(tuple);
        }
    }
    if obj.hasattr("__arrow_c_array__")? {
        let export = obj.call_method0("__arrow_c_array__")?;
        let tuple = export.cast::<pyo3::types::PyTuple>()?;
        return move_array_capsules(tuple);
    }
    if obj.hasattr("__arrow_c_stream__")? || obj.is_instance_of::<PyCapsule>() {
        let frame = arrow_frame_from_py(obj)?;
        let chunks = frame.n_chunks();
        return Err(CausalDataError::new_err(format!(
            "Arrow stream import kept {chunks} chunk(s) and was not flattened into one array; \
             use ingest_arrow_frame"
        )));
    }
    Err(CausalDataError::new_err(
        "expected an Arrow C Data array exporter, a (schema, array) capsule pair, or an Arrow \
         stream",
    ))
}

fn move_array_capsules(
    tuple: &Bound<'_, pyo3::types::PyTuple>,
) -> PyResult<(FfiArrowArray, FfiArrowSchema)> {
    if tuple.len() != 2 {
        return Err(CausalDataError::new_err(
            "__arrow_c_array__ must return (schema_capsule, array_capsule)",
        ));
    }
    let schema_cap = tuple.get_item(0)?.cast_into::<PyCapsule>()?;
    let array_cap = tuple.get_item(1)?.cast_into::<PyCapsule>()?;

    let schema_name = c"arrow_schema";
    let array_name = c"arrow_array";

    let schema_ptr =
        schema_cap.pointer_checked(Some(schema_name))?.as_ptr().cast::<FfiArrowSchema>();
    let array_ptr = array_cap.pointer_checked(Some(array_name))?.as_ptr().cast::<FfiArrowArray>();
    if schema_ptr.is_null() || array_ptr.is_null() {
        return Err(CausalDataError::new_err("null Arrow C Data capsule pointer"));
    }

    // SAFETY: capsules export valid CDI structs; we move them out and leave released empties
    // so the capsule destructor is a no-op.
    let schema = unsafe { std::ptr::read(schema_ptr) };
    let array = unsafe { std::ptr::read(array_ptr) };
    unsafe {
        std::ptr::write(schema_ptr, FfiArrowSchema::empty());
        std::ptr::write(array_ptr, FfiArrowArray::empty());
    }
    Ok((array, schema))
}

/// Import an Arrow C Stream capsule, or an object with `__arrow_c_stream__`,
/// without concatenating record batches.
pub(crate) fn arrow_frame_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ArrowFrameView> {
    let stream = take_arrow_c_stream(obj)?;
    ArrowFrameView::from_stream(stream).map_err(|err| CausalDataError::new_err(err.to_string()))
}

fn take_arrow_c_stream(obj: &Bound<'_, PyAny>) -> PyResult<ArrowCStream> {
    let capsule = if obj.is_instance_of::<PyCapsule>() {
        obj.clone().cast_into::<PyCapsule>()?
    } else {
        obj.call_method0("__arrow_c_stream__")?.cast_into::<PyCapsule>()?
    };
    let name = c"arrow_array_stream";
    let ptr = capsule.pointer_checked(Some(name))?.as_ptr().cast::<FfiArrowArrayStream>();
    if ptr.is_null() {
        return Err(CausalDataError::new_err("null Arrow C Stream capsule pointer"));
    }
    // SAFETY: the capsule exports a valid stream. We move it out and leave a released
    // empty stream so the capsule destructor is a no-op.
    let stream = unsafe { std::ptr::read(ptr) };
    unsafe {
        std::ptr::write(ptr, FfiArrowArrayStream::empty());
    }
    Ok(unsafe { ArrowCStream::from_ffi(stream) })
}

/// Owns an immutable allocation and its admission until the final NumPy view is dropped.
/// Python may mutate elements, but no Rust method can resize or access the allocation.
#[pyclass(frozen)]
struct AdmittedNumericBuffer {
    array: numpy::ndarray::Array1<f64>,
    retention: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
}

#[pymethods]
impl AdmittedNumericBuffer {
    #[getter]
    fn admitted_bytes(&self) -> u64 {
        self.retention.remainder().bytes()
    }
}

/// Expose the buffer owner class so its `admitted_bytes` is documented and typed.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AdmittedNumericBuffer>()
}

/// Convert an owned vector without copying, retaining admission in the array's base owner.
pub(crate) fn admitted_numeric_array(
    py: Python<'_>,
    values: Vec<f64>,
    retention: std::sync::Arc<antecedent_core::execution::AdmissionLease>,
) -> PyResult<Py<numpy::PyArray1<f64>>> {
    let owner = Bound::new(
        py,
        AdmittedNumericBuffer { array: numpy::ndarray::Array1::from_vec(values), retention },
    )?;
    let buffer = owner.borrow();
    // SAFETY: NumPy retains `owner` as its base. The allocation is private and no
    // exposed Rust method can access, resize, or replace it. Python element writes
    // do not reallocate it. Slices retain the base transitively, along with its lease.
    let array =
        unsafe { numpy::PyArray1::borrow_from_array(&buffer.array, owner.clone().into_any()) };
    Ok(array.unbind())
}
