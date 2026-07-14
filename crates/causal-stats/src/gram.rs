//! Form Gram matrices and related dense helpers shared by OLS paths.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]

/// Fill symmetric `ncols×ncols` `XtX` (row-major) from column-major `X`.
pub fn form_xtx(x_colmajor: &[f64], nrows: usize, ncols: usize, xtx: &mut [f64]) {
    debug_assert!(xtx.len() >= ncols * ncols);
    xtx[..ncols * ncols].fill(0.0);
    accumulate_xtx(x_colmajor, nrows, ncols, xtx);
}

/// Accumulate `XᵀX` into an existing symmetric Gram (row-major) from column-major `X`.
///
/// Used by incremental OLS sufficient statistics (DESIGN.md §20).
pub fn accumulate_xtx(x_colmajor: &[f64], nrows: usize, ncols: usize, xtx: &mut [f64]) {
    debug_assert!(xtx.len() >= ncols * ncols);
    for c1 in 0..ncols {
        for c2 in c1..ncols {
            let mut acc = 0.0;
            let col1 = &x_colmajor[c1 * nrows..(c1 + 1) * nrows];
            let col2 = &x_colmajor[c2 * nrows..(c2 + 1) * nrows];
            for r in 0..nrows {
                acc += col1[r] * col2[r];
            }
            xtx[c1 * ncols + c2] += acc;
            if c1 != c2 {
                xtx[c2 * ncols + c1] += acc;
            }
        }
    }
}

/// Accumulate one design row into `XtX` and `Xty` (row-major Gram).
#[allow(clippy::similar_names)] // xtx / xty are standard OLS Gram notation
pub fn accumulate_xtx_xty_row(row: &[f64], y: f64, xtx: &mut [f64], xty: &mut [f64]) {
    let ncols = row.len();
    debug_assert!(xtx.len() >= ncols * ncols);
    debug_assert!(xty.len() >= ncols);
    for c1 in 0..ncols {
        xty[c1] += row[c1] * y;
        for c2 in c1..ncols {
            let v = row[c1] * row[c2];
            xtx[c1 * ncols + c2] += v;
            if c1 != c2 {
                xtx[c2 * ncols + c1] += v;
            }
        }
    }
}

/// Invert a small dense matrix via Gauss–Jordan; returns `None` on singular pivot.
#[must_use]
pub fn invert_square(a_in: &[f64], ncols: usize) -> Option<Vec<f64>> {
    let mut a = a_in.to_vec();
    let mut inv = vec![0.0; ncols * ncols];
    for i in 0..ncols {
        inv[i * ncols + i] = 1.0;
    }
    for col in 0..ncols {
        // Partial pivoting: pick the largest |pivot| in the remaining rows.
        let mut best = col;
        for row in (col + 1)..ncols {
            if a[row * ncols + col].abs() > a[best * ncols + col].abs() {
                best = row;
            }
        }
        if a[best * ncols + col].abs() < 1e-14 {
            return None;
        }
        if best != col {
            for j in 0..ncols {
                a.swap(col * ncols + j, best * ncols + j);
                inv.swap(col * ncols + j, best * ncols + j);
            }
        }
        let pivot = a[col * ncols + col];
        for j in 0..ncols {
            a[col * ncols + j] /= pivot;
            inv[col * ncols + j] /= pivot;
        }
        for row in 0..ncols {
            if row == col {
                continue;
            }
            let factor = a[row * ncols + col];
            for j in 0..ncols {
                a[row * ncols + j] -= factor * a[col * ncols + j];
                inv[row * ncols + j] -= factor * inv[col * ncols + j];
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulate_row_matches_form_xtx() {
        let nrows = 4;
        let ncols = 2;
        // Column-major: col0 = [1,2,3,4], col1 = [0.5,1.5,2.5,3.5]
        let x = [1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5];
        let mut full = vec![0.0; 4];
        form_xtx(&x, nrows, ncols, &mut full);
        let mut row_acc = vec![0.0; 4];
        let mut xty = vec![0.0; 2];
        for r in 0..nrows {
            let row = [x[r], x[nrows + r]];
            accumulate_xtx_xty_row(&row, 0.0, &mut row_acc, &mut xty);
        }
        for i in 0..4 {
            assert!((full[i] - row_acc[i]).abs() < 1e-12, "{i}: {} vs {}", full[i], row_acc[i]);
        }
    }
}
