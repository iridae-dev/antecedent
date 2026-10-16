//! The only `unsafe` in the Python extension.
//!
//! Arrow C Data Interface capsules and the posterior buffer protocol. Every
//! other module in this crate stays safe.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unsafe_code)]

use antecedent_data::{ArrowCColumn, FfiArrowArray, FfiArrowSchema};
use pyo3::exceptions::PyBufferError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::{CausalDataError, PosteriorArtifact};

/// Zero-copy float64 view of `draws` for `np.asarray` / `memoryview`.
#[pymethods]
impl PosteriorArtifact {
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        use pyo3::ffi;
        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("PosteriorArtifact draws are read-only"));
        }
        let borrowed = slf.borrow();
        let data = borrowed.draws.as_slice();
        let n =
            isize::try_from(data.len()).map_err(|_| PyBufferError::new_err("draws too large"))?;
        let itemsize = isize::try_from(std::mem::size_of::<f64>())
            .map_err(|_| PyBufferError::new_err("f64 size does not fit isize"))?;
        let nbytes = n.saturating_mul(itemsize);
        unsafe {
            (*view).obj = slf.as_ptr();
            ffi::Py_INCREF((*view).obj);
            (*view).buf = data.as_ptr().cast::<std::ffi::c_void>().cast_mut();
            (*view).len = nbytes;
            (*view).readonly = 1;
            (*view).itemsize = itemsize;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                std::ffi::CString::new("d")
                    .map_err(|_| PyBufferError::new_err("buffer format is not a C string"))?
                    .into_raw()
            } else {
                std::ptr::null_mut()
            };
            (*view).ndim = 1;
            (*view).shape = if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                let shape = Box::new([n]);
                Box::into_raw(shape).cast()
            } else {
                std::ptr::null_mut()
            };
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                let strides = Box::new([itemsize]);
                Box::into_raw(strides).cast()
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
        }
        Ok(())
    }

    #[allow(clippy::unused_self)]
    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        if view.is_null() {
            return;
        }
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
        let (array, schema) = take_arrow_c_array(&obj)?;
        // SAFETY: `take_arrow_c_array` moved the exporter's CDI structs out and
        // left released empty structs in the capsules.
        cdi_cols.push(unsafe { ArrowCColumn::from_ffi(name, array, schema) });
    }
    Ok(cdi_cols)
}

/// Move CDI structs out of an object exporting `__arrow_c_array__`.
fn take_arrow_c_array(obj: &Bound<'_, PyAny>) -> PyResult<(FfiArrowArray, FfiArrowSchema)> {
    let export = obj.call_method0("__arrow_c_array__")?;
    let tuple = export.cast::<pyo3::types::PyTuple>()?;
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
