//! Shared dense materialization for faer-backed learners.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]

use crate::design::{DesignStorage, DesignView, Layout};
use crate::error::LearnError;

/// Gather a dense column-major design for the view's logical rows.
pub(crate) fn materialize_dense_colmajor(
    x: DesignView<'_>,
) -> Result<(std::borrow::Cow<'_, [f64]>, usize, usize), LearnError> {
    let nrows = x.nrows();
    let ncols = x.ncols();
    match x.storage() {
        DesignStorage::SparseCsr(_) => {
            return Err(LearnError::Unsupported { message: "learner requires dense design" });
        }
        DesignStorage::Dense(d) => {
            if x.row_selection().is_none() && d.layout() == Layout::ColumnMajor {
                let need = nrows.saturating_mul(ncols);
                return Ok((std::borrow::Cow::Borrowed(&d.values()[..need]), nrows, ncols));
            }
        }
    }
    let mut out = vec![0.0; nrows.saturating_mul(ncols)];
    for c in 0..ncols {
        for r in 0..nrows {
            out[c * nrows + r] = x.get(r, c)?;
        }
    }
    Ok((std::borrow::Cow::Owned(out), nrows, ncols))
}

/// Gather physical-aligned values onto logical rows of `x`.
pub(crate) fn gather_physical(
    values: &[f64],
    x: DesignView<'_>,
    logical_rows: usize,
) -> Result<Vec<f64>, LearnError> {
    let mut out = vec![0.0; logical_rows];
    for r in 0..logical_rows {
        out[r] = values[x.physical_index(r)?];
    }
    Ok(out)
}

/// Linear predictor into `out` (logical rows).
pub(crate) fn predict_linear(
    coefficients: &[f64],
    x: DesignView<'_>,
    out: &mut [f64],
) -> Result<(), LearnError> {
    if out.len() != x.nrows() {
        return Err(LearnError::Shape { message: "predict out length != logical rows" });
    }
    if coefficients.len() != x.ncols() {
        return Err(LearnError::Shape { message: "coefficient length != ncols" });
    }
    for (r, slot) in out.iter_mut().enumerate() {
        let mut pred = 0.0;
        for (c, beta) in coefficients.iter().enumerate() {
            pred += x.get(r, c)? * beta;
        }
        *slot = pred;
    }
    Ok(())
}
