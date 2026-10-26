//! Dense Cholesky factorization of symmetric positive-definite matrices.
//!
//! The one implementation shared by the statistics and Bayesian crates. A pivot
//! must be finite and positive *and* not lost to cancellation relative to the
//! diagonal it was reduced from, so NaN input and numerically singular matrices
//! are refused instead of yielding a factor that looks valid.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
/// `64·ε·aᵢᵢ`: below that the subtraction has cancelled to rounding noise, the
/// matrix is singular to working precision, and any inverse or solve from it is
/// noise. A NaN entry reaches some pivot and is refused there.
///
/// # Errors
///
/// [`CholeskyError::BufferTooShort`] or [`CholeskyError::NotPositiveDefinite`].
pub fn cholesky_spd_into(a: &[f64], n: usize, l: &mut [f64]) -> Result<(), CholeskyError> {
    cholesky_spd_pivots_into(a, n, l).map(|_| ()).map_err(|refusal| refusal.error)
}

/// A refused factorization together with the relative size of the pivot that failed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CholeskyRefusal {
    /// Why the factorization was refused.
    pub error: CholeskyError,
    /// The failing pivot `sᵢ / aᵢᵢ`, reduced from a diagonal entry floored at the smallest
    /// positive `f64`; `NaN` when the pivot is NaN or the buffer was too short.
    pub ratio: f64,
}

/// [`cholesky_spd_into`] with the pivot diagnostics a caller reports.
///
/// On success returns the smallest relative pivot `minᵢ sᵢ / aᵢᵢ` (a floor on how far the
/// matrix is from singular, 1 for the identity); on refusal returns the failing position
/// and its relative pivot. The acceptance rule is exactly that of [`cholesky_spd_into`].
///
/// # Errors
///
/// A [`CholeskyRefusal`] carrying [`CholeskyError::BufferTooShort`] or
/// [`CholeskyError::NotPositiveDefinite`].
pub fn cholesky_spd_pivots_into(
    a: &[f64],
    n: usize,
    l: &mut [f64],
) -> Result<f64, CholeskyRefusal> {
    let nn = n.saturating_mul(n);
    if a.len() < nn || l.len() < nn {
        return Err(CholeskyRefusal { error: CholeskyError::BufferTooShort, ratio: f64::NAN });
    }
    l[..nn].fill(0.0);
    let rel_tol = 64.0 * f64::EPSILON;
    let mut smallest = f64::INFINITY;
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diagonal = a[i * n + i];
                let floor = rel_tol * diagonal.max(0.0);
                if !(sum.is_finite() && sum > 0.0 && sum > floor) {
                    return Err(CholeskyRefusal {
                        error: CholeskyError::NotPositiveDefinite { index: i },
                        ratio: sum / diagonal.abs().max(f64::MIN_POSITIVE),
                    });
                }
                smallest = smallest.min(sum / diagonal.abs().max(f64::MIN_POSITIVE));
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Ok(smallest)
}

/// Lower-triangular factor of a symmetric positive *semi*-definite matrix (row-major
/// `n×n`), written into `l`; a rank-deficient direction becomes a zero column.
///
/// The one PSD-tolerant variant, for correlation and covariance matrices that are
/// singular by construction (perfectly collinear scores). With `scale = maxᵢ aᵢᵢ` and
/// `slack = tolerance · scale`, a pivot below `−slack` refuses the matrix as indefinite; a
/// pivot at or below `slack` is a rank-deficient direction whose column is zero, and every
/// entry below it must then lie within `slack` of zero (otherwise the matrix is
/// indefinite). Use [`cholesky_spd_into`] when singularity must be refused.
///
/// # Errors
///
/// [`CholeskyError::BufferTooShort`], or [`CholeskyError::NotPositiveDefinite`] (at the
/// offending position) for a non-finite entry or an indefinite matrix.
pub fn cholesky_psd_into(
    a: &[f64],
    n: usize,
    tolerance: f64,
    l: &mut [f64],
) -> Result<(), CholeskyError> {
    let nn = n.saturating_mul(n);
    if a.len() < nn || l.len() < nn {
        return Err(CholeskyError::BufferTooShort);
    }
    l[..nn].fill(0.0);
    let scale = (0..n).map(|i| a[i * n + i]).fold(0.0_f64, f64::max);
    let slack = tolerance * scale;
    if !(scale.is_finite() && slack.is_finite()) {
        return Err(CholeskyError::NotPositiveDefinite { index: 0 });
    }
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if !sum.is_finite() {
                return Err(CholeskyError::NotPositiveDefinite { index: i });
            }
            if i == j {
                if sum < -slack {
                    return Err(CholeskyError::NotPositiveDefinite { index: i });
                }
                l[i * n + j] = if sum <= slack { 0.0 } else { sum.sqrt() };
            } else if l[j * n + j] > 0.0 {
                l[i * n + j] = sum / l[j * n + j];
            } else if sum.abs() > slack {
                return Err(CholeskyError::NotPositiveDefinite { index: i });
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

/// Forward substitution `L y = b` for a lower-triangular row-major `L` (`n×n`).
///
/// Divides by the diagonal as stored; callers that need a positive-diagonal
/// guard check it first. `b` and `y` must hold at least `n` entries.
pub fn forward_solve_into(chol: &[f64], n: usize, b: &[f64], y: &mut [f64]) {
    for i in 0..n {
        let mut acc = b[i];
        for j in 0..i {
            acc -= chol[i * n + j] * y[j];
        }
        y[i] = acc / chol[i * n + i];
    }
}

/// Back substitution `Lᵀ x = y` for a lower-triangular row-major `L` (`n×n`).
///
/// `y` and `x` must hold at least `n` entries; `x` is fully overwritten.
pub fn back_solve_into(chol: &[f64], n: usize, y: &[f64], x: &mut [f64]) {
    for i in (0..n).rev() {
        let mut acc = y[i];
        for j in (i + 1)..n {
            acc -= chol[j * n + i] * x[j];
        }
        x[i] = acc / chol[i * n + i];
    }
}

/// Solve `L Lᵀ x = b` from a lower-triangular Cholesky factor, writing into `x`.
///
/// `y` is forward-substitution scratch of length `n`. No buffer or pivot
/// validation: the factor is assumed to come from [`cholesky_spd_into`].
pub fn chol_solve_into(chol: &[f64], n: usize, b: &[f64], y: &mut [f64], x: &mut [f64]) {
    forward_solve_into(chol, n, b, y);
    back_solve_into(chol, n, y, x);
}

/// Inverse `A⁻¹` (row-major `n×n`) from the Cholesky factor of SPD `A`, one
/// identity column at a time, so no refactorization is needed.
#[must_use]
pub fn chol_inverse(chol: &[f64], n: usize) -> Vec<f64> {
    let mut inv = vec![0.0; n * n];
    let mut eye_col = vec![0.0; n];
    let mut y = vec![0.0; n];
    for col in 0..n {
        eye_col.fill(0.0);
        eye_col[col] = 1.0;
        forward_solve_into(chol, n, &eye_col, &mut y);
        for i in (0..n).rev() {
            let mut acc = y[i];
            for j in (i + 1)..n {
                acc -= chol[j * n + i] * inv[j * n + col];
            }
            inv[i * n + col] = acc / chol[i * n + i];
        }
    }
    inv
}

/// Lower Cholesky factor `L` (row-major `p x p`) of the Gram matrix `XᵀX` of a column-major
/// `n x p` design, or `None` when the design is rank deficient.
///
/// The factorization runs on the column-equilibrated Gram (unit diagonal, so each pivot reads
/// as the share of its column the preceding ones leave unexplained): a column's scale cannot
/// hide or fake a rank loss. A column norm that is zero or non-finite, a pivot at or below
/// `max(n, p) · ε`, or any non-finite entry refuses the design. The factor is rescaled back
/// to the original columns before it is returned. `None` also when `design` holds fewer than
/// `n * p` entries.
#[must_use]
pub fn design_gram_cholesky(design: &[f64], n: usize, p: usize) -> Option<Vec<f64>> {
    if design.len() < n.checked_mul(p)? {
        return None;
    }
    let column = |c: usize| &design[c * n..(c + 1) * n];
    let norms: Vec<f64> =
        (0..p).map(|c| column(c).iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    if norms.iter().any(|norm| !(norm.is_finite() && *norm > 0.0)) {
        return None;
    }
    let mut l = vec![0.0; p * p];
    let floor = n.max(p) as f64 * f64::EPSILON;
    for i in 0..p {
        for j in 0..=i {
            let dot: f64 = column(i)
                .iter()
                .zip(column(j))
                .map(|(a, b)| (a / norms[i]) * (b / norms[j]))
                .sum();
            let mut acc = dot;
            for q in 0..j {
                acc -= l[i * p + q] * l[j * p + q];
            }
            if i == j {
                if !(acc.is_finite() && acc > floor) {
                    return None;
                }
                l[i * p + i] = acc.sqrt();
            } else {
                l[i * p + j] = acc / l[j * p + j];
            }
        }
    }
    for i in 0..p {
        for j in 0..=i {
            l[i * p + j] *= norms[i];
        }
    }
    l.iter().all(|v| v.is_finite()).then_some(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "small-integer Cholesky factors of an integer matrix are exact in f64"
    )]
    fn factors_hand_matrix() {
        // [[4, 2], [2, 5]] = L L' with L = [[2, 0], [1, 2]].
        let mut l = [9.0; 4];
        cholesky_spd_into(&[4.0, 2.0, 2.0, 5.0], 2, &mut l).unwrap();
        assert_eq!(l, [2.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn design_gram_factor_reproduces_the_gram_and_refuses_rank_loss() {
        // Columns of very different scale: the factor must still satisfy L L' = X'X.
        let n = 5;
        let design = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0e3, 2.0e3, 3.0e3, 4.0e3];
        let l = design_gram_cholesky(&design, n, 2).expect("full rank");
        let gram = |i: usize, j: usize| -> f64 {
            (0..n).map(|r| design[i * n + r] * design[j * n + r]).sum()
        };
        for i in 0..2 {
            for j in 0..=i {
                let rebuilt: f64 = (0..=j).map(|q| l[i * 2 + q] * l[j * 2 + q]).sum();
                assert!((rebuilt - gram(i, j)).abs() <= 1e-9 * gram(i, i).max(gram(j, j)));
            }
        }
        let collinear = [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        assert!(design_gram_cholesky(&collinear, n, 2).is_none());
        assert!(design_gram_cholesky(&[1.0, f64::NAN, 2.0, 3.0], 2, 2).is_none());
        assert!(design_gram_cholesky(&[1.0, 2.0], 2, 2).is_none());
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
    fn pivot_diagnostics_report_the_smallest_and_the_failing_relative_pivot() {
        let mut l = [0.0; 4];
        // [[4, 2], [2, 5]]: pivots 4 and 5 - 1 = 4, relative 1 and 4/5.
        let smallest = cholesky_spd_pivots_into(&[4.0, 2.0, 2.0, 5.0], 2, &mut l).unwrap();
        assert!((smallest - 0.8).abs() < 1e-15, "{smallest}");
        // [[1, 1], [1, 1 + 1e-8]]: the second pivot is 1e-8 against a diagonal of 1 + 1e-8.
        let ok = cholesky_spd_pivots_into(&[1.0, 1.0, 1.0, 1.0 + 1e-8], 2, &mut l).unwrap();
        assert!((ok - 1e-8 / (1.0 + 1e-8)).abs() < 1e-12, "{ok}");
        // Singular: refused at index 1 with a relative pivot at rounding level.
        let refused = cholesky_spd_pivots_into(&[1.0, 1.0, 1.0, 1.0], 2, &mut l).unwrap_err();
        assert_eq!(refused.error, CholeskyError::NotPositiveDefinite { index: 1 });
        assert!(refused.ratio.abs() < 1e-15, "{}", refused.ratio);
        // Indefinite: [[1, 2], [2, 1]] has pivot 1 - 4 = -3.
        let indefinite = cholesky_spd_pivots_into(&[1.0, 2.0, 2.0, 1.0], 2, &mut l).unwrap_err();
        assert!((indefinite.ratio + 3.0).abs() < 1e-15, "{}", indefinite.ratio);
        let short = cholesky_spd_pivots_into(&[1.0, 0.0, 0.0, 1.0], 2, &mut [0.0; 3]);
        assert_eq!(short.unwrap_err().error, CholeskyError::BufferTooShort);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison against exactly representable or bit-identical reference values"
    )]
    fn psd_variant_zeroes_rank_deficient_directions_and_refuses_indefinite_matrices() {
        let mut l = [9.0; 4];
        // Perfectly correlated pair: [[1, 1], [1, 1]] = L L' with L = [[1, 0], [1, 0]].
        cholesky_psd_into(&[1.0, 1.0, 1.0, 1.0], 2, 1e-10, &mut l).unwrap();
        assert_eq!(l, [1.0, 0.0, 1.0, 0.0]);
        // A full-rank matrix factors as the SPD factorization does.
        let mut spd = [0.0; 4];
        cholesky_spd_into(&[4.0, 2.0, 2.0, 5.0], 2, &mut spd).unwrap();
        cholesky_psd_into(&[4.0, 2.0, 2.0, 5.0], 2, 1e-10, &mut l).unwrap();
        assert_eq!(l, spd);
        // Indefinite, and a zero pivot with a non-zero entry below it, are refused.
        assert_eq!(
            cholesky_psd_into(&[1.0, 2.0, 2.0, 1.0], 2, 1e-10, &mut l),
            Err(CholeskyError::NotPositiveDefinite { index: 1 })
        );
        assert_eq!(
            cholesky_psd_into(&[0.0, 0.5, 0.5, 1.0], 2, 1e-10, &mut l),
            Err(CholeskyError::NotPositiveDefinite { index: 1 })
        );
        assert!(cholesky_psd_into(&[f64::NAN, 0.0, 0.0, 1.0], 2, 1e-10, &mut l).is_err());
        // The slack is relative to the largest diagonal entry.
        assert!(cholesky_psd_into(&[1e8, 1e8, 1e8, 1e8 - 1e-4], 2, 1e-10, &mut l).is_ok());
        assert!(cholesky_psd_into(&[1.0, 1.0, 1.0, 1.0 - 1e-4], 2, 1e-10, &mut l).is_err());
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
    #[allow(clippy::float_cmp, reason = "the hand-computed bound (4/1)^2 = 16 is exact in f64")]
    fn condition_bound_matches_hand_and_rejects_nan() {
        // Diagonals 4 and 1 -> (4/1)² = 16.
        assert_eq!(cholesky_condition_lower_bound(&[4.0, 0.0, 3.0, 1.0], 2), 16.0);
        assert!(cholesky_condition_lower_bound(&[f64::NAN, 0.0, 0.0, 1.0], 2).is_infinite());
        assert!(cholesky_condition_lower_bound(&[1.0, 0.0, 0.0, 0.0], 2).is_infinite());
    }

    #[test]
    fn solve_and_inverse_match_hand_values() {
        // A = [[4, 2], [2, 5]]; A⁻¹ = [[5, -2], [-2, 4]] / 16; b = [1, 2] → x = [1/16, 6/16].
        let a = [4.0, 2.0, 2.0, 5.0];
        let mut l = [0.0; 4];
        cholesky_spd_into(&a, 2, &mut l).unwrap();
        let (mut y, mut x) = ([0.0; 2], [0.0; 2]);
        chol_solve_into(&l, 2, &[1.0, 2.0], &mut y, &mut x);
        assert!((x[0] - 1.0 / 16.0).abs() < 1e-15 && (x[1] - 6.0 / 16.0).abs() < 1e-15);
        let inv = chol_inverse(&l, 2);
        let expected = [5.0 / 16.0, -2.0 / 16.0, -2.0 / 16.0, 4.0 / 16.0];
        for (g, e) in inv.iter().zip(expected) {
            assert!((g - e).abs() < 1e-15);
        }
    }
}
