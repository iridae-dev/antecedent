//! The one symmetric positive-definite factorization of the estimators.
//!
//! Every estimator that solves, inverts or takes the log-determinant of a Gram, information or
//! posterior-precision matrix goes through [`SpdFactor`], so "is this matrix positive definite?"
//! has one answer: the kernel acceptance rule, a pivot must be finite and exceed `64·ε` times the
//! diagonal entry it was reduced from. The only caller-chosen policy is the explicit [`Jitter`]:
//! a retry with a relative diagonal shift, never an implicit absolute tolerance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::{chol_inverse, chol_solve_into, cholesky_spd_into};

/// What to do when the strict factorization of a matrix is refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Jitter {
    /// Refuse: the matrix must be positive definite as given.
    None,
    /// Retry once with `relative · max(maxᵢ |aᵢᵢ|, MIN_POSITIVE)` added to the diagonal.
    Relative(f64),
}

/// Lower Cholesky factor (row-major `n × n`) of a symmetric positive-definite matrix.
#[derive(Clone, Debug)]
pub(crate) struct SpdFactor {
    lower: Vec<f64>,
    n: usize,
}

impl SpdFactor {
    /// Factor `a` (row-major `n × n`) under the strict kernel rule; `None` when it is not
    /// positive definite to working precision, contains a non-finite entry, or has the wrong size.
    #[must_use]
    pub(crate) fn new(a: &[f64], n: usize) -> Option<Self> {
        Self::with_jitter(a, n, Jitter::None)
    }

    /// [`Self::new`], retrying once under `jitter` when the strict factorization is refused.
    #[must_use]
    pub(crate) fn with_jitter(a: &[f64], n: usize, jitter: Jitter) -> Option<Self> {
        let nn = n.checked_mul(n)?;
        if a.len() != nn {
            return None;
        }
        let mut lower = vec![0.0; nn];
        if cholesky_spd_into(a, n, &mut lower).is_ok() {
            return Some(Self { lower, n });
        }
        let Jitter::Relative(relative) = jitter else {
            return None;
        };
        if !(relative.is_finite() && relative > 0.0) {
            return None;
        }
        let scale =
            (0..n).map(|i| a[i * n + i].abs()).fold(0.0_f64, f64::max).max(f64::MIN_POSITIVE);
        let mut shifted = a.to_vec();
        for i in 0..n {
            shifted[i * n + i] += relative * scale;
        }
        cholesky_spd_into(&shifted, n, &mut lower).ok()?;
        Some(Self { lower, n })
    }

    /// Consume the factor, returning the lower triangle buffer.
    #[must_use]
    pub(crate) fn into_lower(self) -> Vec<f64> {
        self.lower
    }

    /// Solve `A x = b`; `None` when `b` has the wrong length or the solution is not finite.
    #[must_use]
    pub(crate) fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        if b.len() != self.n {
            return None;
        }
        let mut y = vec![0.0; self.n];
        let mut x = vec![0.0; self.n];
        chol_solve_into(&self.lower, self.n, b, &mut y, &mut x);
        x.iter().all(|v| v.is_finite()).then_some(x)
    }

    /// `A⁻¹`, row-major and exactly symmetric (the two triangles are averaged).
    #[must_use]
    pub(crate) fn inverse(&self) -> Vec<f64> {
        inverse_from_lower(&self.lower, self.n)
    }

    /// `ln |A| = 2 Σ ln Lᵢᵢ`.
    #[must_use]
    pub(crate) fn log_det(&self) -> f64 {
        2.0 * (0..self.n).map(|i| self.lower[i * self.n + i].ln()).sum::<f64>()
    }
}

/// `A⁻¹` (row-major, exactly symmetric) from the lower Cholesky factor of `A`.
#[must_use]
pub(crate) fn inverse_from_lower(lower: &[f64], n: usize) -> Vec<f64> {
    let mut inverse = chol_inverse(lower, n);
    symmetrize_average(&mut inverse, n);
    inverse
}

/// Replace both triangles of the row-major `n × n` matrix `m` by their average.
pub(crate) fn symmetrize_average(m: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..i {
            let value = 0.5 * (m[i * n + j] + m[j * n + i]);
            m[i * n + j] = value;
            m[j * n + i] = value;
        }
    }
}

/// Solve `A x = b` for symmetric positive-definite row-major `A` (`n × n`) under the strict
/// kernel rule.
#[must_use]
pub(crate) fn solve_spd(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    SpdFactor::new(a, n)?.solve(b)
}

/// Solve `A x = b` retrying once under `jitter` when `A` is refused.
#[must_use]
pub(crate) fn solve_spd_with(a: &[f64], b: &[f64], n: usize, jitter: Jitter) -> Option<Vec<f64>> {
    SpdFactor::with_jitter(a, n, jitter)?.solve(b)
}

/// `(XᵀX)⁻¹` (row-major `ncols × ncols`) of a column-major `nrows × ncols` design, from the
/// triangular factor of a Householder QR of the design itself: `(XᵀX)⁻¹ = R⁻¹ R⁻ᵀ`. The Gram
/// matrix is never formed, so the conditioning is that of `X` rather than its square, and the
/// rank rule is the one the least-squares point estimate applies (a diagonal of `R` no larger
/// than `√nrows · ε` times the largest, after scaling every column to unit length).
///
/// `None` when the design is empty, has fewer rows than columns, has a non-finite entry, a zero
/// column, or is rank deficient.
#[must_use]
pub(crate) fn xtx_inverse_qr(x_colmajor: &[f64], nrows: usize, ncols: usize) -> Option<Vec<f64>> {
    let cells = nrows.checked_mul(ncols)?;
    if ncols == 0 || nrows < ncols || x_colmajor.len() < cells {
        return None;
    }
    let mut a = x_colmajor[..cells].to_vec();
    if a.iter().any(|v| !v.is_finite()) {
        return None;
    }
    // Equilibrate: unit-length columns, so the rank test does not depend on the units.
    let mut scale = vec![0.0; ncols];
    for (c, s) in scale.iter_mut().enumerate() {
        let column = &mut a[c * nrows..(c + 1) * nrows];
        let norm = column.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return None;
        }
        column.iter_mut().for_each(|v| *v /= norm);
        *s = norm;
    }
    // Householder triangularisation; `r[k]` is the diagonal entry of the k-th reflection.
    let mut r_diag = vec![0.0; ncols];
    for k in 0..ncols {
        let norm = a[k * nrows + k..(k + 1) * nrows].iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 {
            return None;
        }
        let alpha = if a[k * nrows + k] > 0.0 { -norm } else { norm };
        a[k * nrows + k] -= alpha;
        let v_norm_sq: f64 = a[k * nrows + k..(k + 1) * nrows].iter().map(|v| v * v).sum();
        r_diag[k] = alpha;
        if v_norm_sq > 0.0 {
            for c in (k + 1)..ncols {
                let (head, tail) = a.split_at_mut(c * nrows);
                let v = &head[k * nrows + k..(k + 1) * nrows];
                let column = &mut tail[k..nrows];
                let projection =
                    2.0 * v.iter().zip(column.iter()).map(|(x, y)| x * y).sum::<f64>() / v_norm_sq;
                for (y, x) in column.iter_mut().zip(v) {
                    *y -= projection * x;
                }
            }
        }
    }
    let largest = r_diag.iter().fold(0.0_f64, |m, d| m.max(d.abs()));
    let tolerance = (nrows as f64).sqrt() * f64::EPSILON * largest;
    if r_diag.iter().any(|d| !(d.abs() > tolerance)) {
        return None;
    }
    // Upper-triangular R (row-major): the diagonal from `r_diag`, the strict upper triangle from
    // the reduced columns (rows above the diagonal are untouched by later reflections).
    let mut r = vec![0.0; ncols * ncols];
    for c in 0..ncols {
        for row in 0..c {
            r[row * ncols + c] = a[c * nrows + row];
        }
        r[c * ncols + c] = r_diag[c];
    }
    // R⁻¹ by back substitution, one column at a time.
    let mut r_inv = vec![0.0; ncols * ncols];
    for col in 0..ncols {
        r_inv[col * ncols + col] = 1.0 / r[col * ncols + col];
        for row in (0..col).rev() {
            let mut sum = 0.0;
            for k in (row + 1)..=col {
                sum += r[row * ncols + k] * r_inv[k * ncols + col];
            }
            r_inv[row * ncols + col] = -sum / r[row * ncols + row];
        }
    }
    // (R̃ᵀR̃)⁻¹ = R̃⁻¹ R̃⁻ᵀ, unscaled by the column norms.
    let mut inverse = vec![0.0; ncols * ncols];
    for i in 0..ncols {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in i.max(j)..ncols {
                sum += r_inv[i * ncols + k] * r_inv[j * ncols + k];
            }
            let value = sum / (scale[i] * scale[j]);
            inverse[i * ncols + j] = value;
            inverse[j * ncols + i] = value;
        }
    }
    inverse.iter().all(|v| v.is_finite()).then_some(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_inverts_and_takes_log_determinant() {
        // [[4, 2], [2, 5]]: det 16, inverse [[5, -2], [-2, 4]] / 16.
        let a = [4.0, 2.0, 2.0, 5.0];
        let factor = SpdFactor::new(&a, 2).expect("positive definite");
        assert!((factor.log_det() - 16.0_f64.ln()).abs() < 1e-12);
        let inverse = factor.inverse();
        assert!((inverse[0] - 5.0 / 16.0).abs() < 1e-12);
        assert!((inverse[1] + 2.0 / 16.0).abs() < 1e-12);
        assert_eq!(inverse[1].to_bits(), inverse[2].to_bits());
        let x = factor.solve(&[1.0, 2.0]).expect("finite solution");
        assert!((4.0 * x[0] + 2.0 * x[1] - 1.0).abs() < 1e-12);
        assert!((2.0 * x[0] + 5.0 * x[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn acceptance_is_relative_to_the_diagonal() {
        // A well-conditioned matrix in tiny units is positive definite, not "singular".
        let tiny = [4.0e-15, 2.0e-15, 2.0e-15, 5.0e-15];
        assert!(SpdFactor::new(&tiny, 2).is_some());
        // A perfectly collinear pair and an indefinite matrix are refused at any scale.
        assert!(SpdFactor::new(&[1.0, 1.0, 1.0, 1.0], 2).is_none());
        assert!(SpdFactor::new(&[1.0e-15, 1.0e-15, 1.0e-15, 1.0e-15], 2).is_none());
        assert!(SpdFactor::new(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
        // Wrong size and non-finite entries are refused.
        assert!(SpdFactor::new(&[1.0, 0.0, 0.0], 2).is_none());
        assert!(SpdFactor::new(&[f64::NAN, 0.0, 0.0, 1.0], 2).is_none());
    }

    #[test]
    fn jitter_is_an_explicit_retry() {
        let singular = [1.0, 1.0, 1.0, 1.0];
        assert!(solve_spd(&singular, &[1.0, 1.0], 2).is_none());
        assert!(solve_spd_with(&singular, &[1.0, 1.0], 2, Jitter::None).is_none());
        assert!(solve_spd_with(&singular, &[1.0, 1.0], 2, Jitter::Relative(1e-8)).is_some());
        assert!(SpdFactor::with_jitter(&singular, 2, Jitter::Relative(f64::NAN)).is_none());
    }

    #[test]
    fn qr_inverse_matches_the_gram_inverse_and_survives_bad_scaling() {
        // y ~ 1 + x on x = 0..5; (XtX) = [[6, 15], [15, 55]], det 105.
        let x: Vec<f64> = [vec![1.0; 6], (0..6).map(f64::from).collect::<Vec<_>>()].concat();
        let inverse = xtx_inverse_qr(&x, 6, 2).expect("full rank");
        assert!((inverse[0] - 55.0 / 105.0).abs() < 1e-12);
        assert!((inverse[1] + 15.0 / 105.0).abs() < 1e-12);
        assert!((inverse[3] - 6.0 / 105.0).abs() < 1e-12);
        assert_eq!(inverse[1].to_bits(), inverse[2].to_bits());
        // Rescaling a column rescales the inverse accordingly; it is not a rank failure.
        let scaled: Vec<f64> =
            x.iter().enumerate().map(|(i, v)| if i >= 6 { v * 1e-9 } else { *v }).collect();
        let scaled_inverse = xtx_inverse_qr(&scaled, 6, 2).expect("scale-free");
        assert!((scaled_inverse[3] * 1e-18 - 6.0 / 105.0).abs() < 1e-9);
    }

    #[test]
    fn qr_inverse_refuses_rank_deficiency_and_bad_shapes() {
        let collinear: Vec<f64> = [vec![1.0; 4], vec![2.0; 4]].concat();
        assert!(xtx_inverse_qr(&collinear, 4, 2).is_none());
        assert!(xtx_inverse_qr(&[1.0, 2.0], 1, 2).is_none());
        assert!(xtx_inverse_qr(&[], 0, 0).is_none());
        assert!(xtx_inverse_qr(&[1.0, f64::NAN, 1.0, 2.0], 2, 2).is_none());
        assert!(xtx_inverse_qr(&[0.0, 0.0, 1.0, 2.0], 2, 2).is_none());
    }
}
