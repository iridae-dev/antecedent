//! Shared bounded extraction for selective-recalculation bridges.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::recalc_api::invalid;

pub(crate) const MAX_ROWS: usize = 100_000;
pub(crate) const MAX_COLUMNS: usize = 256;
pub(crate) const MAX_VALUES: usize = 1_000_000;

pub(crate) fn check_columns(
    names: &[String],
    columns: &[PyReadonlyArray1<'_, f64>],
) -> PyResult<()> {
    if names.len() > MAX_COLUMNS
        || columns.len() > MAX_COLUMNS
        || columns.iter().any(|c| c.len().unwrap_or(usize::MAX) > MAX_ROWS)
        || columns
            .iter()
            .try_fold(0usize, |n, c| c.len().ok().and_then(|len| n.checked_add(len)))
            .is_none_or(|n| n > MAX_VALUES)
    {
        return Err(invalid("recalc.limits_exceeded", "data exceed row/column/value limits"));
    }
    Ok(())
}

pub(crate) fn sequence_len(sequence: &Bound<'_, PyAny>, detail: &str) -> PyResult<usize> {
    if let Ok(list) = sequence.cast::<PyList>() {
        Ok(list.len())
    } else if let Ok(tuple) = sequence.cast::<PyTuple>() {
        Ok(tuple.len())
    } else {
        Err(invalid(detail, "prediction rows must be lists or tuples"))
    }
}

pub(crate) fn sequence_item<'py>(
    sequence: &Bound<'py, PyAny>,
    index: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(list) = sequence.cast::<PyList>() {
        list.get_item(index)
    } else {
        sequence.cast::<PyTuple>()?.get_item(index)
    }
}

// Inspect Python's actual list/tuple dimensions before any numeric extraction.
// Direct indexing bypasses arbitrary subclass iterators and their allocation behavior.
pub(crate) fn bounded_prediction_rows(
    raw_rows: &Bound<'_, PyAny>,
    detail: &str,
) -> PyResult<Vec<Vec<f64>>> {
    let nrows = sequence_len(raw_rows, detail)?;
    if nrows > MAX_ROWS {
        return Err(invalid("recalc.limits_exceeded", "predictions exceed row limit"));
    }
    let mut total = 0usize;
    for i in 0..nrows {
        let row = sequence_item(raw_rows, i)?;
        let width = sequence_len(&row, detail)?;
        total = total
            .checked_add(width)
            .ok_or_else(|| invalid("recalc.limits_exceeded", "prediction size overflow"))?;
        if width > MAX_COLUMNS || total > MAX_VALUES {
            return Err(invalid(
                "recalc.limits_exceeded",
                "predictions exceed feature/value limits",
            ));
        }
    }
    let mut rows = Vec::with_capacity(nrows);
    total = 0;
    for i in 0..nrows {
        let row = sequence_item(raw_rows, i)?;
        let width = sequence_len(&row, detail)?;
        total = total
            .checked_add(width)
            .ok_or_else(|| invalid("recalc.limits_exceeded", "prediction size overflow"))?;
        if width > MAX_COLUMNS || total > MAX_VALUES {
            return Err(invalid(
                "recalc.limits_exceeded",
                "predictions exceed feature/value limits",
            ));
        }
        let mut values = Vec::with_capacity(width);
        for j in 0..width {
            values.push(sequence_item(&row, j)?.extract::<f64>()?);
        }
        rows.push(values);
    }
    Ok(rows)
}
