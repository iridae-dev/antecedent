//! Form Gram matrices and related dense helpers shared by OLS paths.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]

/// Fill symmetric `ncols×ncols` `XtX` (row-major) from column-major `X`.
pub fn form_xtx(x_colmajor: &[f64], nrows: usize, ncols: usize, xtx: &mut [f64]) {
    debug_assert!(xtx.len() >= ncols * ncols);
    xtx[..ncols * ncols].fill(0.0);
    for c1 in 0..ncols {
        for c2 in c1..ncols {
            let mut acc = 0.0;
            let col1 = &x_colmajor[c1 * nrows..(c1 + 1) * nrows];
            let col2 = &x_colmajor[c2 * nrows..(c2 + 1) * nrows];
            for r in 0..nrows {
                acc += col1[r] * col2[r];
            }
            xtx[c1 * ncols + c2] = acc;
            xtx[c2 * ncols + c1] = acc;
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
