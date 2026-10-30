// SPDX-License-Identifier: MIT OR Apache-2.0
//! Verified invertibility of represented affine residual systems, dense or sparse.
//! A numerical inverse is only a candidate. Outward interval arithmetic checks
//! ||I - R A||_infinity < 1 for the exact represented A = I-W; the Neumann
//! argument then proves invertibility independently of any numerical root.
use crate::equation_system::Interval;
use crate::nonlinear_equilibrium::MAX_DENSE_MATRIX_BYTES;
use crate::sparse_equilibrium::{CsrMatrix, SparseLu};
/// Prove invertibility of I-W using an outward-rounded inverse residual.
/// Returns false when the bounded certificate fails, including ill conditioning.
/// This is no assertion about dynamic stability or contraction of W.
#[must_use]
pub fn certifies_affine_bijection(weights: &[f64], order: usize) -> bool {
    if order == 0 {
        return weights.is_empty();
    }
    if order.checked_mul(order) != Some(weights.len()) || weights.iter().any(|x| !x.is_finite()) {
        return false;
    }
    let rows: Vec<Vec<(usize, f64)>> = (0..order)
        .map(|i| {
            (0..order)
                .filter_map(|j| {
                    let w = weights[i * order + j];
                    (w != 0.0).then_some((j, w))
                })
                .collect()
        })
        .collect();
    certifies_sparse_affine_bijection(&rows, order)
}
/// Sparse form of [`certifies_affine_bijection`]. `rows[i]` holds the nonzero local
/// `(column, w_ij)` entries of `W`. A sparse LU of `A = I - W` supplies a numerical inverse
/// `R` (one solve per column); outward interval arithmetic then bounds
/// `||I - R A||_inf` using only the nonzeros of `A`. A bound below one proves `R A`, hence `A`,
/// invertible by a Neumann series, independently of how `R` was found. The explicit `R` is a
/// dense `order x order` array, so the certificate is not attempted (returns false) when that
/// array would exceed [`MAX_DENSE_MATRIX_BYTES`]; contraction certificates still apply there.
#[must_use]
pub fn certifies_sparse_affine_bijection(rows: &[Vec<(usize, f64)>], order: usize) -> bool {
    if order == 0 {
        return rows.is_empty();
    }
    if rows.len() != order
        || order
            .checked_mul(order)
            .and_then(|c| c.checked_mul(8))
            .is_none_or(|b| b > MAX_DENSE_MATRIX_BYTES)
        || rows.iter().flatten().any(|&(c, w)| c >= order || !w.is_finite())
    {
        return false;
    }
    let one = Interval { lower: 1.0, upper: 1.0 };
    let mut a_rows: Vec<Vec<(usize, Interval)>> = Vec::with_capacity(order);
    let mut numeric: Vec<Vec<(usize, f64)>> = Vec::with_capacity(order);
    for (i, row) in rows.iter().enumerate() {
        // Represent the identity exactly once. Keeping every -W entry separate
        // lets the interval residual enclose even duplicate sums that round in
        // the numerical CSR matrix used to obtain the candidate inverse.
        let mut interval_row = Vec::with_capacity(row.len() + 1);
        let mut numeric_row = Vec::with_capacity(row.len() + 1);
        interval_row.push((i, one));
        numeric_row.push((i, 1.0));
        for &(c, w) in row {
            interval_row.push((c, Interval { lower: -w, upper: -w }));
            numeric_row.push((c, -w));
        }
        a_rows.push(interval_row);
        numeric.push(numeric_row);
    }
    let Ok(matrix) = CsrMatrix::from_rows(order, numeric) else {
        return false;
    };
    let Ok(lu) = SparseLu::factor(&matrix, usize::MAX, &mut || Ok(())) else {
        return false;
    };
    let mut inverse = vec![0.0; order * order];
    for k in 0..order {
        let mut unit = vec![0.0; order];
        unit[k] = 1.0;
        for (i, value) in lu.solve(&unit).into_iter().enumerate() {
            inverse[i * order + k] = value;
        }
    }
    if inverse.iter().any(|x| !x.is_finite()) {
        return false;
    }
    for i in 0..order {
        let mut residual: Vec<Interval> = (0..order)
            .map(|j| {
                let d = if i == j { 1.0 } else { 0.0 };
                Interval { lower: d, upper: d }
            })
            .collect();
        for k in 0..order {
            let r = inverse[i * order + k];
            if r == 0.0 {
                continue;
            }
            for &(j, a) in &a_rows[k] {
                residual[j] = residual[j].add_outward(a.scale_outward(-r));
            }
        }
        let mut norm = Interval { lower: 0.0, upper: 0.0 };
        for entry in residual {
            let magnitude = entry.lower.abs().max(entry.upper.abs());
            norm = norm.add_outward(Interval { lower: 0.0, upper: magnitude });
        }
        if !norm.upper.is_finite() || norm.upper >= 1.0 {
            return false;
        }
    }
    true
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_bijection_is_separate_from_contraction_stability_and_numeric_success() {
        assert!(certifies_affine_bijection(&[0.0, 2.0, -2.0, 0.0], 2));
        assert!(!certifies_affine_bijection(&[0.0, 1.0, 1.0, 0.0], 2));
        assert!(!certifies_affine_bijection(&[f64::NAN], 1));
        assert!(certifies_affine_bijection(&[2.0], 1));
        assert!(!certifies_affine_bijection(&[1.0], 1));
    }
    #[test]
    fn duplicate_diagonals_represent_one_identity_and_sum_all_weights() {
        assert!(!certifies_sparse_affine_bijection(&[vec![(0, 0.5), (0, 0.5)]], 1));
        assert!(certifies_sparse_affine_bijection(&[vec![(0, 0.25), (0, 0.25)]], 1));
        assert!(certifies_sparse_affine_bijection(&[vec![(0, 1.0), (0, 1.0)]], 1));
        assert!(
            !certifies_sparse_affine_bijection(&[vec![(1, 0.5), (1, 0.5)], vec![(0, 1.0)]], 2,)
        );
    }
    #[test]
    fn sparse_certificate_agrees_with_the_dense_one_and_scales_past_the_dense_order_limit() {
        let rows = vec![vec![(1, 2.0)], vec![(0, -2.0)]];
        assert!(certifies_sparse_affine_bijection(&rows, 2));
        assert!(!certifies_sparse_affine_bijection(&[vec![(1, 1.0)], vec![(0, 1.0)]], 2));
        assert!(!certifies_sparse_affine_bijection(&[vec![(0, 1.0)]], 1));
        // Directed ring with weight 1.5: noncontractive but I - 1.5 P is invertible.
        let n = 300;
        let ring: Vec<Vec<(usize, f64)>> = (0..n).map(|i| vec![((i + n - 1) % n, 1.5)]).collect();
        assert!(certifies_sparse_affine_bijection(&ring, n));
        // Weight exactly one on the ring makes I - W singular.
        let unit: Vec<Vec<(usize, f64)>> = (0..n).map(|i| vec![((i + n - 1) % n, 1.0)]).collect();
        assert!(!certifies_sparse_affine_bijection(&unit, n));
    }
}
