//! Immutable Python table snapshots. Numeric buffers cross the ownership boundary once.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_data::{
    F64Buffer, Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData,
};
use pyo3::prelude::*;

/// Owned nanosecond indices, shared with event preparation without another input copy.
#[pyclass(name = "_IndexSnapshot", frozen)]
pub(crate) struct IndexSnapshot {
    pub(crate) values: Arc<[i64]>,
    pub(crate) retention: Arc<antecedent_core::execution::AdmissionLease>,
}

#[pymethods]
impl IndexSnapshot {
    #[new]
    fn new(values: &Bound<'_, PyAny>) -> PyResult<Self> {
        let len = values.len()?;
        let bytes = u64::try_from(len)
            .ok()
            .and_then(|len| len.checked_mul(8))
            .ok_or_else(|| crate::value_err("event index size overflow"))?;
        let mut ctx = antecedent_core::ExecutionContext::production_default(1);
        crate::execution_api::apply(&mut ctx);
        let retention =
            Arc::new(ctx.acquire_admission_bounded(0, bytes, std::time::Duration::ZERO).map_err(
                |error| match error {
                    antecedent_core::execution::AdmissionError::Cancelled => {
                        crate::py_err(antecedent::CausalError::Cancelled { stage: "ingest" })
                    }
                    error => crate::CausalResourceError::new_err(error.to_string()),
                },
            )?);
        let values = if let Ok(array) = values.extract::<numpy::PyReadonlyArray1<'_, i64>>() {
            match array.as_slice() {
                Ok(slice) => Arc::from(slice),
                Err(_) => array.as_array().iter().copied().collect(),
            }
        } else {
            Arc::from(values.extract::<Vec<i64>>()?)
        };
        Ok(Self { values, retention })
    }

    fn __len__(&self) -> usize {
        self.values.len()
    }
}

/// A column view pins its original table. The buffer protocol is read-only in ffi.rs.
#[pyclass(name = "_SnapshotColumn", frozen)]
pub(crate) struct SnapshotColumn {
    pub(crate) table: Arc<TabularData>,
    pub(crate) id: VariableId,
}

impl SnapshotColumn {
    pub(crate) fn buffer(&self) -> PyResult<F64Buffer> {
        match self.table.storage().columns().get(self.id.as_usize()) {
            Some(OwnedColumn::Float64(column)) => Ok(column.values.clone()),
            _ => Err(crate::value_err("snapshot column is not numeric")),
        }
    }

    pub(crate) fn values(&self) -> PyResult<&[f64]> {
        match self.table.storage().columns().get(self.id.as_usize()) {
            Some(OwnedColumn::Float64(column)) => Ok(column.values.as_slice()),
            _ => Err(crate::value_err("snapshot column is not a numeric column")),
        }
    }
}

#[pymethods]
impl SnapshotColumn {
    fn __len__(&self) -> usize {
        self.table.row_count()
    }

    fn __getitem__(&self, index: isize) -> PyResult<f64> {
        let len = isize::try_from(self.table.row_count())
            .map_err(|_| crate::value_err("snapshot row count exceeds Python index range"))?;
        let index = if index < 0 { index.checked_add(len) } else { Some(index) };
        let values = self.values()?;
        let value = index.and_then(|i| usize::try_from(i).ok()).and_then(|i| values.get(i));
        value
            .copied()
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err("snapshot row out of range"))
    }

    #[getter]
    fn name(&self) -> PyResult<String> {
        self.table.schema().get(self.id).map(|v| v.name.to_string()).map_err(crate::py_err)
    }
}

#[pyclass(name = "_TableSnapshot", frozen)]
pub(crate) struct TableSnapshot {
    data: Arc<TabularData>,
    copied_bytes: u64,
    borrowed_bytes: u64,
}

#[pymethods]
impl TableSnapshot {
    #[new]
    #[pyo3(signature = (names, columns, *, borrow=false))]
    fn new(names: Vec<String>, columns: Vec<Bound<'_, PyAny>>, borrow: bool) -> PyResult<Self> {
        if let Some(data) = from_columns(&names, &columns)? {
            return Ok(Self { data: Arc::new(data), copied_bytes: 0, borrowed_bytes: 0 });
        }
        if names.len() != columns.len() || columns.is_empty() {
            return Err(crate::value_err("snapshot needs one nonempty column per name"));
        }
        let rows = columns[0].len()?;
        let mut arrow = true;
        for column in &columns {
            if column.len()? != rows {
                return Err(crate::value_err("snapshot columns must have equal lengths"));
            }
            arrow &= column.hasattr("__arrow_c_array__")?;
        }
        let numeric = if arrow {
            None
        } else {
            Some(columns.iter().map(crate::NumericColumn::from_object).collect::<PyResult<Vec<_>>>()?)
        };
        let shared = numeric
            .as_ref()
            .map(|columns| {
                columns
                    .iter()
                    .map(|column| match column {
                        crate::NumericColumn::F64(array) => {
                            numpy_buffer(array).map(|buffer| buffer.is_some())
                        }
                        _ => Ok(false),
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_else(|| vec![false; columns.len()]);
        let allocated = shared.iter().filter(|shared| !**shared).count();
        let bytes_to_retain = rows
            .checked_mul(allocated)
            .and_then(|count| count.checked_mul(8))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| crate::value_err("snapshot size overflow"))?;
        let mut ctx = antecedent_core::ExecutionContext::production_default(1);
        crate::execution_api::apply(&mut ctx);
        let retention = Arc::new(
            ctx.acquire_admission_bounded(0, bytes_to_retain, std::time::Duration::ZERO).map_err(
                |error| match error {
                    antecedent_core::execution::AdmissionError::Cancelled => {
                        crate::py_err(antecedent::CausalError::Cancelled { stage: "ingest" })
                    }
                    error => crate::CausalResourceError::new_err(error.to_string()),
                },
            )?,
        );
        let (mut data, borrowed, mut copied_bytes) = if let Some(numeric) = numeric {
            let loaded = crate::numeric_columns_to_batch(&names, &numeric)?.loaded()?;
            (loaded.data, None, loaded.receipt.bytes_copied)
        } else {
            let (data, borrowed) = crate::tabular_from_py_columns(names, columns)?;
            (data, borrowed, bytes_to_retain.saturating_sub(borrowed.unwrap_or(0)))
        };
        let bytes = data
            .row_count()
            .checked_mul(data.schema().len())
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()))
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| crate::value_err("snapshot size overflow"))?;
        if !borrow && borrowed.is_some_and(|n| n > 0) {
            copied_bytes = bytes;
            // Arrow can alias writable NumPy arrays. Default snapshots therefore take
            // ownership too; only the explicit borrow constructor accepts that contract.
            let storage = data.storage();
            let columns = storage
                .columns()
                .iter()
                .map(|column| match column {
                    OwnedColumn::Float64(column) => Float64Column::new(
                        column.id,
                        F64Buffer::owned(Arc::<[f64]>::from(column.values.as_slice())),
                        column.validity.clone(),
                    )
                    .map(OwnedColumn::Float64),
                    _ => Err(antecedent_data::DataError::InvalidArgument {
                        message: "table snapshots require numeric columns".into(),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(crate::py_err)?;
            data = TabularData::new(
                OwnedColumnarStorage::try_new(
                    storage.schema().clone(),
                    columns,
                    storage.analysis_mask().cloned(),
                    storage.weights().map(Arc::from),
                )
                .map_err(crate::py_err)?,
            );
        }
        let storage = data.storage();
        let columns = storage
            .columns()
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, column)| match column {
                OwnedColumn::Float64(mut column) => {
                    if !shared[index] {
                        column.values = column.values.with_admission(Arc::clone(&retention));
                    }
                    OwnedColumn::Float64(column)
                }
                column => column,
            })
            .collect();
        data = TabularData::new(
            OwnedColumnarStorage::try_new(
                storage.schema().clone(),
                columns,
                storage.analysis_mask().cloned(),
                storage.weights().map(Arc::from),
            )
            .map_err(crate::py_err)?,
        );
        let borrowed_bytes = if borrow { borrowed.unwrap_or(0) } else { 0 };
        Ok(Self { data: Arc::new(data), copied_bytes, borrowed_bytes })
    }

    #[getter]
    fn column_names(&self) -> Vec<String> {
        self.data.schema().variables().iter().map(|v| v.name.to_string()).collect()
    }

    #[getter]
    fn row_count(&self) -> usize {
        self.data.row_count()
    }

    #[getter]
    fn bytes_copied(&self) -> u64 {
        self.copied_bytes
    }

    #[getter]
    fn bytes_borrowed(&self) -> u64 {
        self.borrowed_bytes
    }

    fn column(&self, name: &str) -> PyResult<SnapshotColumn> {
        let id = self
            .data
            .schema()
            .variables()
            .iter()
            .position(|v| &*v.name == name)
            .and_then(|i| u32::try_from(i).ok())
            .map(VariableId::from_raw)
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_owned()))?;
        Ok(SnapshotColumn { table: Arc::clone(&self.data), id })
    }
}

/// Recognize native snapshot columns before any NumPy extraction. Projection shares
/// immutable buffers and never gathers numeric values through Python.
pub(crate) fn from_columns(
    names: &[String],
    columns: &[Bound<'_, PyAny>],
) -> PyResult<Option<TabularData>> {
    let Some(first) = columns.first() else { return Ok(None) };
    let Ok(first) = first.extract::<PyRef<'_, SnapshotColumn>>() else { return Ok(None) };
    if names.len() != columns.len() {
        return Err(crate::value_err("names and columns must have the same length"));
    }
    let mut ids = Vec::with_capacity(columns.len());
    for (name, column) in names.iter().zip(columns) {
        let column = column.extract::<PyRef<'_, SnapshotColumn>>()?;
        if !Arc::ptr_eq(&first.table, &column.table) || column.name()? != *name {
            return Err(crate::value_err(
                "snapshot columns must retain one table and their original names",
            ));
        }
        ids.push(column.id);
    }
    let complete = ids.len() == first.table.schema().len()
        && ids.iter().enumerate().all(|(i, id)| id.as_usize() == i);
    Ok(Some(if complete {
        (*first.table).clone()
    } else {
        first.table.project(&ids).map_err(crate::py_err)?.0
    }))
}

pub(crate) fn register(module: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    module.add_class::<IndexSnapshot>()?;
    module.add_class::<TableSnapshot>()?;
    module.add_class::<SnapshotColumn>()
}

/// A NumPy view over a complete snapshot column shares its proven native buffer.
/// Slices and arbitrary foreign arrays continue through the explicit ownership copy.
pub(crate) fn numpy_buffer(
    column: &numpy::PyReadonlyArray1<'_, f64>,
) -> PyResult<Option<F64Buffer>> {
    let view = column.as_array();
    if !view.is_standard_layout() {
        return Ok(None);
    }
    let mut owner = column.as_any().clone();
    for _ in 0..8 {
        if let Ok(snapshot) = owner.extract::<PyRef<'_, SnapshotColumn>>() {
            let buffer = snapshot.buffer()?;
            return Ok(
                (buffer.len() == view.len() && buffer.as_ptr() == view.as_ptr()).then_some(buffer)
            );
        }
        let next = if owner.is_instance_of::<pyo3::types::PyMemoryView>() {
            owner.getattr("obj")?
        } else if owner.is_instance_of::<numpy::PyArray1<f64>>() {
            owner.getattr("base")?
        } else {
            return Ok(None);
        };
        if next.is_none() {
            return Ok(None);
        }
        owner = next;
    }
    Ok(None)
}
