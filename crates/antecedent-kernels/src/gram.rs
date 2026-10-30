//! Symmetric equilibration and Gauss-Jordan elimination.
//!
//! One pivot policy for partial-correlation solves, dense inversion, and the
//! conjugate posterior precision.
//!
//! # Pivot policy
//!
//! 1. **Scales.** For a finite positive diagonal `a_ii`, `d_i` is the power of
//!    two nearest `sqrt(a_ii)` (`2^{round(1/2 log2 a_ii)}`, exponent clamped to
//!    `[-1000, 1000]`). A power of two divides exactly in `f64`. Any other
//!    diagonal uses `d_i = 1`.
//! 2. **Equilibrated matrix.** `B = D^{-1} A D^{-1}`.
//! 3. **Elimination.** Gauss-Jordan with partial pivoting. The scale `s` is
//!    `max_i |b_ii|` when that is positive, and `max_ij |b_ij|` when every
//!    diagonal is zero, so a zero diagonal with a real off-diagonal pivot is
//!    not declared singular before pivoting. The matrix is singular when `s`
//!    is non-finite or not positive, or when the best remaining pivot is
//!    non-finite or `<= EQUILIBRATED_PIVOT_FLOOR * s` (`1e-12`). Equality with
//!    the floor is singular. On a partial-correlation Gram the diagonal is
//!    already 1, so `s` is that diagonal and the elimination matches the
//!    previous in-place solver.
//! 4. **Undo.** `x = D^{-1} x_tilde`. An inverse is `A^{-1} = D^{-1} B^{-1} D^{-1}`.
//!
//! Cholesky keeps its own `64 eps` floor. That is a different algorithm.
//! [`invert_square`] and [`gauss_jordan_solve`] on the same equilibrated
//! entries share this accept/reject predicate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Pivot floor on a matrix whose diagonal is order 1.
///
/// Multiplied by the scale inside [`gauss_jordan_solve`].
pub const EQUILIBRATED_PIVOT_FLOOR: f64 = 1e-12;

/// Diagonal scales `D` for `B = D^{-1} A D^{-1}`.
///
/// `a` is row-major `n x n`. A short buffer yields an empty vector.
#[must_use]
pub fn symmetric_scales(a: &[f64], n: usize) -> Vec<f64> {
    let nn = n.saturating_mul(n);
    if n == 0 || a.len() < nn {
        return Vec::new();
    }
    (0..n)
        .map(|j| {
            let diag = a[j * n + j];
            if diag.is_finite() && diag > 0.0 {
                #[allow(clippy::cast_possible_truncation, reason = "exponent of a finite f64")]
                let exponent = (0.5 * diag.log2()).round().clamp(-1000.0, 1000.0) as i32;
                2.0_f64.powi(exponent)
            } else {
                1.0
            }
        })
        .collect()
}

/// Write `B = D^{-1} A D^{-1}` into `out`.
pub fn equilibrate_into(a: &[f64], scales: &[f64], n: usize, out: &mut [f64]) {
    let nn = n.saturating_mul(n);
    if scales.len() < n || a.len() < nn || out.len() < nn {
        return;
    }
    for i in 0..n {
        for j in 0..n {
            // Use one division when the scale product is representable. This preserves
            // mixed-scale entries that successive division could underflow prematurely.
            // Fall back to dividing by the larger scale first only when the product
            // itself overflows or underflows.
            out[i * n + j] = divide_by_scale_pair(a[i * n + j], scales[i], scales[j]);
        }
    }
}

/// Divide by two positive scales without losing a representable result to the scale product.
fn divide_by_scale_pair(value: f64, left: f64, right: f64) -> f64 {
    let product = left * right;
    if product.is_finite() && product > 0.0 {
        value / product
    } else if left >= right {
        value / left / right
    } else {
        value / right / left
    }
}

/// In-place Gauss-Jordan with partial pivoting on `nrhs` right-hand sides.
///
/// Right-hand side `k` occupies `rhs[k * n .. (k + 1) * n]`. Returns `false`
/// when the pivot policy declares the matrix singular.
#[must_use]
pub fn gauss_jordan_solve(a: &mut [f64], rhs: &mut [f64], n: usize, nrhs: usize) -> bool {
    gauss_jordan_solve_with_pivot(a, rhs, n, nrhs).is_some()
}

/// [`gauss_jordan_solve`] that also reports the smallest accepted pivot as a fraction of the
/// matrix scale (the largest diagonal magnitude). `None` exactly when the solve refuses.
///
/// For a unit-diagonal Gram this is an inexpensive conditioning indicator: a small value
/// means the matrix is close to the refusal floor and the solution has lost digits.
#[must_use]
pub fn gauss_jordan_solve_with_pivot(
    a: &mut [f64],
    rhs: &mut [f64],
    n: usize,
    nrhs: usize,
) -> Option<f64> {
    let nn = n.saturating_mul(n);
    if n == 0 || a.len() < nn || rhs.len() < nrhs.saturating_mul(n) {
        return None;
    }
    let rhs_cells = n.saturating_mul(nrhs);
    if a[..nn].iter().chain(&rhs[..rhs_cells]).any(|value| !value.is_finite()) {
        return None;
    }
    let mut diag_scale = 0.0_f64;
    let mut entry_scale = 0.0_f64;
    for i in 0..n {
        diag_scale = diag_scale.max(a[i * n + i].abs());
        for j in 0..n {
            entry_scale = entry_scale.max(a[i * n + j].abs());
        }
    }
    let scale = if diag_scale > 0.0 { diag_scale } else { entry_scale };
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    let tol = EQUILIBRATED_PIVOT_FLOOR * scale;
    let mut smallest = f64::INFINITY;
    for col in 0..n {
        let mut best_row = col;
        let mut best = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > best {
                best = v;
                best_row = row;
            }
        }
        if !best.is_finite() || best <= tol {
            return None;
        }
        smallest = smallest.min(best);
        if best_row != col {
            for j in 0..n {
                a.swap(col * n + j, best_row * n + j);
            }
            for k in 0..nrhs {
                rhs.swap(k * n + col, k * n + best_row);
            }
        }
        let pivot = a[col * n + col];
        for j in 0..n {
            a[col * n + j] /= pivot;
        }
        for k in 0..nrhs {
            rhs[k * n + col] /= pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row * n + col];
            for j in 0..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            for k in 0..nrhs {
                rhs[k * n + row] -= factor * rhs[k * n + col];
            }
        }
    }
    if a[..nn].iter().chain(&rhs[..rhs_cells]).any(|value| !value.is_finite()) {
        return None;
    }
    Some(smallest / scale)
}

/// Invert a small dense row-major matrix.
///
/// Equilibrates with [`symmetric_scales`], inverts with [`gauss_jordan_solve`],
/// and unscales. `None` on a short or non-finite buffer, a zero dimension, or a
/// singular pivot under the module policy, or an inverse that is not representable as
/// finite `f64` values.
#[must_use]
pub fn invert_square(a_in: &[f64], n: usize) -> Option<Vec<f64>> {
    let nn = n.checked_mul(n)?;
    if n == 0 || a_in.len() < nn || a_in[..nn].iter().any(|v| !v.is_finite()) {
        return None;
    }
    let scales = symmetric_scales(a_in, n);
    if scales.len() != n || scales.iter().any(|d| !(d.is_finite() && *d > 0.0)) {
        return None;
    }
    let mut a = vec![0.0; nn];
    equilibrate_into(a_in, &scales, n, &mut a);
    let mut rhs = vec![0.0; nn];
    for j in 0..n {
        rhs[j * n + j] = 1.0;
    }
    if !gauss_jordan_solve(&mut a, &mut rhs, n, n) {
        return None;
    }
    let mut inv = vec![0.0; nn];
    for i in 0..n {
        for j in 0..n {
            // As above, do not form a scale product that can overflow.
            inv[i * n + j] = divide_by_scale_pair(rhs[j * n + i], scales[i], scales[j]);
        }
    }
    if inv.iter().any(|value| !value.is_finite()) {
        return None;
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_refuses_nonfinite_entries_and_unrepresentable_solutions() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for slot in 0..4 {
                let mut matrix = [1.0, 0.0, 0.0, 1.0];
                matrix[slot] = value;
                assert!(
                    gauss_jordan_solve_with_pivot(&mut matrix, &mut [1.0, 2.0], 2, 1).is_none()
                );
            }
            assert!(
                gauss_jordan_solve_with_pivot(&mut [1.0, 0.0, 0.0, 1.0], &mut [value, 1.0], 2, 1)
                    .is_none()
            );
        }
        assert!(gauss_jordan_solve_with_pivot(&mut [1e-300], &mut [1e300], 1, 1).is_none());
        let mut rhs = [3.0, 4.0];
        assert!(gauss_jordan_solve_with_pivot(&mut [1.0, 0.0, 0.0, 1.0], &mut rhs, 2, 1).is_some());
        assert_eq!(rhs, [3.0, 4.0]);
    }

    #[test]
    fn power_of_two_scales_match_the_conjugate_formula() {
        let a = [100.0, 0.0, 0.0, 9e14];
        assert_eq!(symmetric_scales(&a, 2), [8.0, 2.0_f64.powi(25)]);
        assert_eq!(symmetric_scales(&[0.0, 1.0, 1.0, 4.0], 2), [1.0, 2.0]);
    }

    #[test]
    fn solve_with_pivot_reports_the_smallest_relative_pivot() {
        let mut a = [1.0, 0.0, 0.0, 1e-6];
        let mut rhs = [1.0, 1.0];
        let pivot = gauss_jordan_solve_with_pivot(&mut a, &mut rhs, 2, 1).expect("regular");
        assert_eq!(pivot, 1e-6);
        assert_eq!(rhs, [1.0, 1e6]);
        let mut singular = [1.0, 0.0, 0.0, 1e-13];
        assert!(gauss_jordan_solve_with_pivot(&mut singular, &mut [1.0, 1.0], 2, 1).is_none());
        assert!(!gauss_jordan_solve(&mut [1.0, 0.0, 0.0, 1e-13], &mut [1.0, 1.0], 2, 1));
    }

    #[test]
    fn invert_and_solve_agree_on_singularity() {
        let cases: &[&[f64]] = &[
            &[4.0, 1.0, 1.0, 3.0],
            &[100.0, 0.0, 0.0, 9e14],
            &[100.0, 1.5e8, 1.5e8, 9e14],
            &[1.0, 3e6, 3e6, 9e12],
            &[1e10, 1e10, 1e10, 1e10 + 1e-6],
            &[1.0, f64::NAN, f64::NAN, 1.0],
            &[0.0, 1.0, 1.0, 0.0],
        ];
        for a_in in cases {
            let inverted = invert_square(a_in, 2);
            let scales = symmetric_scales(a_in, 2);
            let mut b = vec![0.0; 4];
            equilibrate_into(a_in, &scales, 2, &mut b);
            let mut rhs = [1.0, 0.0];
            let solved = if a_in.iter().any(|v| !v.is_finite()) {
                false
            } else {
                gauss_jordan_solve(&mut b, &mut rhs, 2, 1)
            };
            assert_eq!(inverted.is_some(), solved, "verdict split on {a_in:?}");
        }
    }

    #[test]
    fn heterogeneous_diagonal_inverse_stays_exact() {
        let inv = invert_square(&[100.0, 0.0, 0.0, 9e14], 2).expect("diagonal");
        assert!((inv[0] - 0.01).abs() < 1e-15, "{inv:?}");
        assert!(inv[1].abs() < 1e-30 && inv[2].abs() < 1e-30, "{inv:?}");
        assert!((inv[3] * 9e14 - 1.0).abs() < 1e-12, "{inv:?}");
    }

    #[test]
    fn inversion_handles_finite_diagonals_whose_scale_product_overflows() {
        let value = 1e308;
        let inv = invert_square(&[value], 1).expect("finite nonzero diagonal");
        assert!(inv[0].is_finite() && inv[0] > 0.0, "{inv:?}");
        assert!((inv[0] * value - 1.0).abs() < 1e-12, "{inv:?}");

        let inv =
            invert_square(&[value, 0.0, 0.0, value], 2).expect("finite two-dimensional diagonal");
        assert!(inv.iter().all(|entry| entry.is_finite()), "{inv:?}");
        assert!((inv[0] * value - 1.0).abs() < 1e-12, "{inv:?}");
        assert!((inv[3] * value - 1.0).abs() < 1e-12, "{inv:?}");
    }

    #[test]
    fn mixed_scales_preserve_representable_subnormal_entries() {
        let scales = [2.0_f64.powi(512), 2.0_f64.powi(-511)];
        let off_diagonal = 1e-320;
        let mut equilibrated = [0.0; 4];
        equilibrate_into(&[0.0, off_diagonal, off_diagonal, 0.0], &scales, 2, &mut equilibrated);

        let expected = off_diagonal / 2.0;
        assert!(
            equilibrated[1] > 0.0,
            "mixed-scale entry was prematurely underflowed: {equilibrated:?}"
        );
        assert_eq!(equilibrated[1].to_bits(), expected.to_bits());
        assert_eq!(equilibrated[2].to_bits(), expected.to_bits());
    }

    #[test]
    fn tiny_diagonal_inverse_is_finite_when_representable() {
        let value = 1e-308;
        let inv = invert_square(&[value], 1).expect("representable tiny-diagonal inverse");
        assert!(inv[0].is_finite(), "{inv:?}");
        assert!((inv[0] / 1e308 - 1.0).abs() < 1e-12, "{inv:?}");
    }

    #[test]
    fn tiny_diagonal_inverse_refuses_unrepresentable_result() {
        assert!(invert_square(&[1e-320], 1).is_none());
    }

    #[test]
    fn zero_diagonal_permutation_still_inverts() {
        let inv = invert_square(&[0.0, 1.0, 1.0, 0.0], 2).expect("permutation");
        assert!(inv[0].abs() < 1e-15 && inv[3].abs() < 1e-15, "{inv:?}");
        assert!((inv[1] - 1.0).abs() < 1e-15 && (inv[2] - 1.0).abs() < 1e-15, "{inv:?}");
    }
}
