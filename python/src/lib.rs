//! PyO3 bindings — Phase 0 skeleton: Arrow float64 load with copy diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs)]
#![allow(unsafe_code)] // required by PyO3

use std::sync::Arc;

use arrow_array::{Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use causal_data::{TableView, tabular_from_record_batch};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Result of loading columns into the Rust data layer.
#[pyclass]
struct ArrowLoadInfo {
    /// Number of rows.
    #[pyo3(get)]
    row_count: usize,
    /// Number of columns.
    #[pyo3(get)]
    column_count: usize,
    /// Bytes copied into owned Rust buffers.
    #[pyo3(get)]
    bytes_copied: u64,
    /// Number of materialization diagnostics recorded.
    #[pyo3(get)]
    diagnostic_count: usize,
}

/// Load float64 NumPy columns (copied into Arrow, then into library-owned storage).
///
/// Phase 0 measures copy behavior explicitly: inputs cross the Python boundary
/// as owned buffers; `bytes_copied` reports Rust-side materialization size.
#[pyfunction]
fn load_float64_columns(
    names: Vec<String>,
    columns: Vec<PyReadonlyArray1<'_, f64>>,
) -> PyResult<ArrowLoadInfo> {
    if names.len() != columns.len() {
        return Err(PyValueError::new_err("names and columns must have the same length"));
    }
    if columns.is_empty() {
        return Err(PyValueError::new_err("at least one column required"));
    }
    let n = columns[0].as_array().len();
    for col in &columns {
        if col.as_array().len() != n {
            return Err(PyValueError::new_err("column length mismatch"));
        }
    }

    let fields: Vec<Field> = names.iter().map(|n| Field::new(n, DataType::Float64, true)).collect();
    let schema = Schema::new(fields);
    let arrays: Vec<Arc<dyn arrow_array::Array>> = columns
        .iter()
        .map(|c| {
            let slice = c.as_array();
            let values: Vec<Option<f64>> = slice.iter().copied().map(Some).collect();
            Arc::new(Float64Array::from(values)) as Arc<dyn arrow_array::Array>
        })
        .collect();
    let batch = RecordBatch::try_new(Arc::new(schema), arrays)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let loaded =
        tabular_from_record_batch(&batch).map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(ArrowLoadInfo {
        row_count: loaded.data.row_count(),
        column_count: loaded.data.schema().len(),
        bytes_copied: loaded.bytes_copied,
        diagnostic_count: loaded.diagnostics.len(),
    })
}

/// Python module `causal._native`.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_float64_columns, m)?)?;
    m.add_class::<ArrowLoadInfo>()?;
    m.add("__version__", causal_core::VERSION)?;
    Ok(())
}
