//! Dense Cholesky factorization of symmetric positive-definite matrices.
//!
//! The one implementation shared by the statistics and Bayesian crates. A pivot
//! must be finite and positive *and* not lost to cancellation relative to the
//! diagonal it was reduced from, so NaN input and numerically singular matrices
//! are refused instead of yielding a factor that looks valid.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

/// Why a Cholesky factorization was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CholeskyError {
    /// The input or output buffer is shorter than `n²`.
    BufferTooShort,
    /// The pivot at `index` is NaN/infinite, non-positive, or no larger than
    /// rounding noise on the diagonal entry it was reduced from.
    NotPositiveDefinite {
        /// Zero-based diagonal position of the failing pivot.
        index: usize,
    },
}

/// Lower-triangular Cholesky factor of an SPD matrix (row-major `n×n`), written
/// into `l` (upper triangle zeroed).
///
/// A pivot `s = aᵢᵢ − Σₖ lᵢₖ²` is accepted only when it is finite and exceeds
/// `n·ε·aᵢᵢ`: below that the subtraction has cancelled to rounding noise, the
/// matrix is singular to working precision, and any inverse or solve from it is
/// noise. A NaN entry reaches some pivot and is refused there.
///
/// # Errors
///
/// [`CholeskyError::BufferTooShort`] or [`CholeskyError::NotPositiveDefinite`].
pub fn cholesky_spd_into(a: &[f64], n: usize, l: &mut [f64]) -> Result<(), CholeskyError> {
    let nn = n.saturating_mul(n);
    if a.len() < nn || l.len() < nn {
        return Err(CholeskyError::BufferTooShort);
    }
    l[..nn].fill(0.0);
    let rel_tol = (n.max(1) as f64) * f64::EPSILON;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let floor = rel_tol * a[i * n + i].max(0.0);
                if !(sum.is_finite() && sum > 0.0 && sum > floor) {
                    return Err(CholeskyError::NotPositiveDefinite { index: i });
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Ok(())
}

/// Lower bound on the 2-norm condition number from a Cholesky factor:
/// `(max Lᵢᵢ / min Lᵢᵢ)²`.
///
/// The true κ₂ can be larger (`[[1, .999], [.999, 1]]` gives 500 against 1999),
/// so this is a floor: a large value proves ill-conditioning, a small one does
/// not prove good conditioning. Any non-finite or non-positive diagonal gives
/// `+∞`.
#[must_use]
pub fn cholesky_condition_lower_bound(chol: &[f64], n: usize) -> f64 {
    let mut min_d = f64::INFINITY;
    let mut max_d = 0.0_f64;
    for i in 0..n {
        let d = chol[i * n + i];
        if !(d.is_finite() && d > 0.0) {
            return f64::INFINITY;
        }
        min_d = min_d.min(d);
        max_d = max_d.max(d);
    }
    let ratio = max_d / min_d;
    ratio * ratio
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factors_hand_matrix() {
        // [[4, 2], [2, 5]] = L L' with L = [[2, 0], [1, 2]].
        let mut l = [9.0; 4];
        cholesky_spd_into(&[4.0, 2.0, 2.0, 5.0], 2, &mut l).unwrap();
        assert_eq!(l, [2.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn nan_pivot_is_refused() {
        let mut l = [0.0; 4];
        let nan_diag = cholesky_spd_into(&[f64::NAN, 0.0, 0.0, 1.0], 2, &mut l);
        assert_eq!(nan_diag, Err(CholeskyError::NotPositiveDefinite { index: 0 }));
        // NaN off-diagonal poisons the next pivot.
        let nan_off = cholesky_spd_into(&[1.0, f64::NAN, f64::NAN, 1.0], 2, &mut l);
        assert_eq!(nan_off, Err(CholeskyError::NotPositiveDefinite { index: 1 }));
        let inf = cholesky_spd_into(&[f64::INFINITY, 0.0, 0.0, 1.0], 2, &mut l);
        assert_eq!(inf, Err(CholeskyError::NotPositiveDefinite { index: 0 }));
    }

    #[test]
    fn numerically_singular_pivot_is_refused() {
        // Second pivot is 1 - 1² = 0 exactly; a 1e-300 pivot is noise against a
        // unit diagonal only when it is reduced from one, so use a rank-one
        // matrix plus a rounding-level remainder.
        let mut l = [0.0; 4];
        assert_eq!(
            cholesky_spd_into(&[1.0, 1.0, 1.0, 1.0], 2, &mut l),
            Err(CholeskyError::NotPositiveDefinite { index: 1 })
        );
        assert_eq!(
            cholesky_spd_into(&[1.0, 1.0, 1.0, 1.0 + 1e-17], 2, &mut l),
            Err(CholeskyError::NotPositiveDefinite { index: 1 })
        );
        // A small but resolvable pivot (1e-8 relative) is still accepted.
        assert!(cholesky_spd_into(&[1.0, 1.0, 1.0, 1.0 + 1e-8], 2, &mut l).is_ok());
    }

    #[test]
    fn short_buffer_is_refused() {
        let mut l = [0.0; 3];
        assert_eq!(
            cholesky_spd_into(&[1.0, 0.0, 0.0, 1.0], 2, &mut l),
            Err(CholeskyError::BufferTooShort)
        );
    }

    #[test]
    fn condition_bound_matches_hand_and_rejects_nan() {
        // Diagonals 4 and 1 -> (4/1)² = 16.
        assert_eq!(cholesky_condition_lower_bound(&[4.0, 0.0, 3.0, 1.0], 2), 16.0);
        assert!(cholesky_condition_lower_bound(&[f64::NAN, 0.0, 0.0, 1.0], 2).is_infinite());
        assert!(cholesky_condition_lower_bound(&[1.0, 0.0, 0.0, 0.0], 2).is_infinite());
    }
}
