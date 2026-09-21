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

/// Fill `Xᵀy` (length `ncols`) from column-major `X` and `y`.
pub fn form_xty(x_colmajor: &[f64], nrows: usize, ncols: usize, y: &[f64], xty: &mut [f64]) {
    debug_assert!(x_colmajor.len() >= nrows * ncols);
    debug_assert!(y.len() >= nrows);
    debug_assert!(xty.len() >= ncols);
    for c in 0..ncols {
        let col = &x_colmajor[c * nrows..(c + 1) * nrows];
        let mut acc = 0.0;
        for r in 0..nrows {
            acc += col[r] * y[r];
        }
        xty[c] = acc;
    }
}

/// Accumulate `XᵀX` into an existing symmetric Gram (row-major) from column-major `X`.
///
/// Used by incremental OLS sufficient statistics.
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

/// Relative pivot tolerance shared by [`cholesky_spd`] and [`invert_square`]: a pivot that
/// has lost all but `64 ε` of its original magnitude to cancellation carries no information.
const PIVOT_REL_TOL: f64 = 64.0 * f64::EPSILON;

/// Whether column `col` of column-major `X` is constant, judged against the column's own
/// magnitude (`|v − v₀| ≤ 64 ε · max|v|`), so the verdict does not depend on the unit the
/// column is measured in. An all-zero column is constant; an empty column is constant.
#[must_use]
pub fn column_is_constant(x_colmajor: &[f64], nrows: usize, col: usize) -> bool {
    if nrows == 0 {
        return true;
    }
    let base = col * nrows;
    let column = &x_colmajor[base..base + nrows];
    let v0 = column[0];
    let max_abs = column.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    column.iter().all(|&v| (v - v0).abs() <= PIVOT_REL_TOL * max_abs)
}

/// Lower-triangular Cholesky of an SPD matrix (row-major `n×n`).
///
/// Returns `None` on a NaN, non-positive, or numerically singular pivot. The
/// factorization is [`antecedent_kernels::cholesky_spd_into`], shared with the
/// Bayesian backends.
#[must_use]
pub fn cholesky_spd(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n.checked_mul(n)?];
    antecedent_kernels::cholesky_spd_into(a, n, &mut l).ok()?;
    Some(l)
}

/// `log|A| = 2 Σ log Lᵢᵢ` from a Cholesky factor of SPD `A`.
#[must_use]
pub fn chol_log_det(chol: &[f64], n: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        s += chol[i * n + i].ln();
    }
    2.0 * s
}

/// Solve `A x = b` given Cholesky factor `L` of SPD `A = L L'`.
#[must_use]
pub fn chol_solve(chol: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    if chol.len() < n * n || b.len() < n {
        return None;
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut acc = b[i];
        for j in 0..i {
            acc -= chol[i * n + j] * y[j];
        }
        let diag = chol[i * n + i];
        if diag.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return None;
        }
        y[i] = acc / diag;
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut acc = y[i];
        for j in (i + 1)..n {
            acc -= chol[j * n + i] * x[j];
        }
        x[i] = acc / chol[i * n + i];
    }
    Some(x)
}

/// Invert a small dense matrix via Gauss–Jordan; returns `None` on singular pivot.
///
/// The matrix is first equilibrated symmetrically, `B = D⁻¹ A D⁻¹` with `Dᵢ = √|aᵢᵢ|` (the
/// row/column magnitude when the diagonal entry is zero), and singularity is judged on the
/// pivots of `B`, whose unit-scale rows make the tolerance independent of each column's
/// unit. A Gram of `[1, x]` with `x` of order `1e6` is therefore inverted (its
/// equilibrated form is the identity), while a genuinely collinear pair is still refused.
/// The inverse is un-scaled, `A⁻¹ = D⁻¹ B⁻¹ D⁻¹`. Any non-finite entry returns `None`.
#[must_use]
pub fn invert_square(a_in: &[f64], ncols: usize) -> Option<Vec<f64>> {
    if ncols == 0
        || a_in.len() < ncols * ncols
        || a_in[..ncols * ncols].iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let mut d = vec![0.0_f64; ncols];
    for i in 0..ncols {
        let diag = a_in[i * ncols + i].abs();
        let magnitude = if diag > 0.0 {
            diag.sqrt()
        } else {
            (0..ncols).fold(0.0_f64, |m, j| {
                m.max(a_in[i * ncols + j].abs().max(a_in[j * ncols + i].abs()))
            })
        };
        if !(magnitude.is_finite() && magnitude > 0.0) {
            return None;
        }
        d[i] = magnitude;
    }
    let tol = 1e-12;

    let mut a = vec![0.0; ncols * ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            a[i * ncols + j] = a_in[i * ncols + j] / (d[i] * d[j]);
        }
    }
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
        if a[best * ncols + col].abs() < tol {
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
    for i in 0..ncols {
        for j in 0..ncols {
            inv[i * ncols + j] /= d[i] * d[j];
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

    #[allow(clippy::float_cmp)] // exact constants: the values compared are representable results, not measurements
    #[test]
    fn form_xty_matches_hand_dot_products_and_the_row_accumulator() {
        // Columns c0 = [1, 1, 1], c1 = [2, 0, 1]; y = [1, 2, 3]: c0·y = 6, c1·y = 2 + 0 + 3 = 5.
        let x = [1.0, 1.0, 1.0, 2.0, 0.0, 1.0];
        let y = [1.0, 2.0, 3.0];
        let mut xty = [0.0; 2];
        form_xty(&x, 3, 2, &y, &mut xty);
        assert_eq!(xty, [6.0, 5.0]);
        let mut acc_xtx = [0.0; 4];
        let mut acc_xty = [0.0; 2];
        for r in 0..3 {
            accumulate_xtx_xty_row(&[x[r], x[3 + r]], y[r], &mut acc_xtx, &mut acc_xty);
        }
        assert_eq!(acc_xty, xty);
    }

    #[test]
    fn chol_log_det_matches_direct_2x2() {
        // A = [[4, 1], [1, 3]]; det = 11.
        let a = [4.0, 1.0, 1.0, 3.0];
        let chol = cholesky_spd(&a, 2).expect("spd");
        let log_det = chol_log_det(&chol, 2);
        assert!((log_det - 11.0_f64.ln()).abs() < 1e-12, "log_det={log_det}");
        let b = [5.0, 4.0];
        let x = chol_solve(&chol, 2, &b).expect("solve");
        // A x = b ⇒ [4,1;1,3] x = [5,4] ⇒ x = [1,1]
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn invert_square_rejects_badly_scaled_near_singular_matrix() {
        // Rows are nearly parallel at large magnitude: after one elimination step the
        // remaining pivot is ~1e-6 in absolute terms — comfortably above a fixed 1e-14
        // absolute threshold (which would wrongly accept this and hand back a garbage
        // inverse), but far below 1e-12 * scale (~1e-2) once the tolerance is scaled to
        // the matrix's own magnitude (~1e10).
        let a = [1e10, 1e10, 1e10, 1e10 + 1e-6];
        assert!(invert_square(&a, 2).is_none());
    }

    #[test]
    fn invert_square_accepts_heterogeneous_column_units() {
        // Gram of [1, x] with x of sd 3e6 and n = 100: diag(100, 9e14) is perfectly
        // conditioned once each column is measured in its own unit. A tolerance tied to the
        // largest diagonal (9e14 · 1e-12 = 900 > 100) wrongly declared it singular.
        let a = [100.0, 0.0, 0.0, 9e14];
        let inv = invert_square(&a, 2).expect("well-conditioned after column scaling");
        assert!((inv[0] - 0.01).abs() < 1e-15);
        assert!(inv[1].abs() < 1e-30 && inv[2].abs() < 1e-30);
        assert!((inv[3] / (1.0 / 9e14) - 1.0).abs() < 1e-12);

        // A = D B D with D = diag(10, 3e7), B = [[1, .5], [.5, 1]]:
        // A⁻¹ = D⁻¹ B⁻¹ D⁻¹, B⁻¹ = (4/3) [[1, -.5], [-.5, 1]].
        let a = [100.0, 1.5e8, 1.5e8, 9e14];
        let inv = invert_square(&a, 2).expect("coupled, well-conditioned after scaling");
        let expect = [4.0 / 3.0 / 100.0, -2.0 / 3.0 / 3e8, -2.0 / 3.0 / 3e8, 4.0 / 3.0 / 9e14];
        for (g, e) in inv.iter().zip(expect) {
            assert!((g / e - 1.0).abs() < 1e-10, "got {inv:?} expected {expect:?}");
        }
    }

    #[test]
    fn invert_square_rejects_collinear_columns_at_any_unit() {
        // Second column is exactly 3e6 × the first: singular regardless of units.
        let a = [1.0, 3e6, 3e6, 9e12];
        assert!(invert_square(&a, 2).is_none());
        assert!(invert_square(&[1.0, f64::NAN, f64::NAN, 1.0], 2).is_none());
    }

    #[test]
    fn cholesky_spd_tolerance_is_relative_and_rejects_nan() {
        // Unit-free: tiny and huge well-conditioned diagonals both factor.
        assert!(cholesky_spd(&[1e-20, 0.0, 0.0, 1.0], 2).is_some());
        assert!(cholesky_spd(&[100.0, 0.0, 0.0, 9e14], 2).is_some());
        // Numerically singular (pivot cancels to ~1e-15 of the diagonal).
        assert!(cholesky_spd(&[1.0, 1.0, 1.0, 1.0 + 1e-15], 2).is_none());
        assert!(cholesky_spd(&[f64::NAN, 0.0, 0.0, 1.0], 2).is_none());
    }

    #[test]
    fn column_is_constant_is_relative_to_column_magnitude() {
        // Small-unit columns that genuinely vary are not constant, at any magnitude.
        let varying = [1e-8, 2e-8, 3e-8, 4e-8];
        assert!(!column_is_constant(&varying, 4, 0));
        let varying_tiny = [1e-13, 2e-13, 3e-13, 4e-13];
        assert!(!column_is_constant(&varying_tiny, 4, 0));
        assert!(column_is_constant(&[5.0, 5.0, 5.0, 5.0], 4, 0));
        assert!(column_is_constant(&[0.0; 4], 4, 0));
        // Two columns: only the second varies.
        let two = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0];
        assert!(column_is_constant(&two, 4, 0));
        assert!(!column_is_constant(&two, 4, 1));
    }

    #[test]
    fn invert_square_still_inverts_well_scaled_matrix() {
        // Sanity check that the new relative tolerance doesn't reject ordinary,
        // well-conditioned matrices.
        let a = [4.0, 1.0, 1.0, 3.0];
        let inv = invert_square(&a, 2).expect("well-conditioned");
        // A^-1 = 1/11 * [[3, -1], [-1, 4]]
        assert!((inv[0] - 3.0 / 11.0).abs() < 1e-12);
        assert!((inv[1] - (-1.0 / 11.0)).abs() < 1e-12);
        assert!((inv[2] - (-1.0 / 11.0)).abs() < 1e-12);
        assert!((inv[3] - 4.0 / 11.0).abs() < 1e-12);
    }
}
