//! Shared resource-checked extraction for selective-recalculation bridges.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::recalc_api::invalid;

/// Bytes the bridge allocates per `f64` value it takes from the caller.
const BYTES_PER_VALUE: u64 = 8;

/// Admit `values` numbers (plus `extra_bytes` of container overhead) against the context's
/// memory budget, using checked arithmetic and before the bridge allocates them.
///
/// Cancellation is not decided here: the context carries the caller's token into the run,
/// which refuses with the structured `recalc.cancelled` result of its own family.
pub(crate) fn check_values(
    values: Option<usize>,
    extra_bytes: u64,
    ctx: &ExecutionContext,
) -> PyResult<()> {
    let bytes = values
        .and_then(|n| u64::try_from(n).ok())
        .and_then(|n| n.checked_mul(BYTES_PER_VALUE))
        .and_then(|n| n.checked_add(extra_bytes))
        .ok_or_else(|| invalid("recalc.limits_exceeded", "input size overflows"))?;
    let limit = ctx.memory_limit_bytes().into_iter().min();
    if let Some(limit) = limit.filter(|limit| bytes > *limit) {
        return Err(invalid(
            "recalc.memory_budget_exceeded",
            format!("planned input copy of {bytes} bytes exceeds memory limit of {limit} bytes"),
        ));
    }
    Ok(())
}

/// Admit the caller's columns: the copy the bridge will make is checked against the memory
/// budget of `ctx`.
pub(crate) fn check_columns(
    columns: &[PyReadonlyArray1<'_, f64>],
    ctx: &ExecutionContext,
) -> PyResult<()> {
    let values =
        columns.iter().try_fold(0usize, |n, c| c.len().ok().and_then(|len| n.checked_add(len)));
    check_values(values, 0, ctx)
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
    ctx: &ExecutionContext,
) -> PyResult<Vec<Vec<f64>>> {
    let nrows = sequence_len(raw_rows, detail)?;
    let mut total = Some(0usize);
    for i in 0..nrows {
        let width = sequence_len(&sequence_item(raw_rows, i)?, detail)?;
        total = total.and_then(|n| n.checked_add(width));
    }
    let row_overhead = u64::try_from(nrows)
        .ok()
        .and_then(|n| n.checked_mul(std::mem::size_of::<Vec<f64>>() as u64))
        .ok_or_else(|| invalid("recalc.limits_exceeded", "prediction size overflow"))?;
    check_values(total, row_overhead, ctx)?;
    let mut rows = Vec::with_capacity(nrows);
    for i in 0..nrows {
        let row = sequence_item(raw_rows, i)?;
        let width = sequence_len(&row, detail)?;
        let mut values = Vec::with_capacity(width);
        for j in 0..width {
            values.push(sequence_item(&row, j)?.extract::<f64>()?);
        }
        rows.push(values);
    }
    Ok(rows)
}
