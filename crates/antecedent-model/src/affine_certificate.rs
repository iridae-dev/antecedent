// SPDX-License-Identifier: MIT OR Apache-2.0
//! Verified invertibility of represented affine residual systems, dense or sparse.
//! A numerical inverse is only a candidate. Outward interval arithmetic checks
//! ||I - R A||_infinity < 1 for the exact represented A = I-W; the Neumann
//! argument then proves invertibility independently of any numerical root.
use crate::equation_system::Interval;
use crate::nonlinear_equilibrium::{EquilibriumError, MAX_DENSE_MATRIX_BYTES};
use crate::sparse_equilibrium::{CsrMatrix, FactorError, SolverBudgetCode, SparseLu};
use antecedent_core::ExecutionContext;
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
    certifies_sparse_affine_checked(rows, order, usize::MAX, &mut || Ok(())).unwrap_or(false)
}

/// [`certifies_affine_bijection`] under the caller's execution controls: the worst-case dense
/// inverse, interval rows and fill are acquired from the context's admission pool before any is
/// allocated, and cancellation is polled between rows. The context-free form is unadmitted.
///
/// # Errors
///
/// [`EquilibriumError::Resource`] when the context cancels or its pool cannot admit the
/// worst case, or when the dense matrix limit is exceeded.
pub fn certifies_affine_bijection_in(
    weights: &[f64],
    order: usize,
    ctx: &ExecutionContext,
) -> Result<bool, EquilibriumError> {
    if order == 0 {
        return Ok(weights.is_empty());
    }
    if order.checked_mul(order) != Some(weights.len()) || weights.iter().any(|x| !x.is_finite()) {
        return Ok(false);
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
    certifies_sparse_affine_bijection_in(&rows, order, ctx)
}

/// [`certifies_sparse_affine_bijection`] under the caller's execution controls; see
/// [`certifies_affine_bijection_in`].
///
/// # Errors
///
/// As [`certifies_affine_bijection_in`].
pub fn certifies_sparse_affine_bijection_in(
    rows: &[Vec<(usize, f64)>],
    order: usize,
    ctx: &ExecutionContext,
) -> Result<bool, EquilibriumError> {
    let square = order
        .checked_mul(order)
        .ok_or_else(|| EquilibriumError::Resource("affine certificate size overflow".into()))?;
    // Dense inverse, interval and numeric rows, CSR, ordering and worst-case fill: the same
    // quote the model's own certificate holds, with fill at its dense worst case.
    let bytes = square
        .checked_mul(192)
        .and_then(|v| v.checked_add(square.checked_mul(48)?))
        .and_then(|v| v.checked_add(order.checked_mul(1024)?))
        .and_then(|v| v.checked_add(4096))
        .and_then(|v| u64::try_from(v).ok())
        .ok_or_else(|| EquilibriumError::Resource("affine certificate size overflow".into()))?;
    let _lease =
        ctx.acquire_admission(0, bytes).map_err(|e| EquilibriumError::Resource(e.to_string()))?;
    certifies_sparse_affine_checked(rows, order, square, &mut || {
        crate::nonlinear_equilibrium::check_cancel(ctx)
    })
}

// The caller admits worst-case dense/fill storage before constructing rows.
pub(crate) fn certifies_sparse_affine_checked(
    rows: &[Vec<(usize, f64)>],
    order: usize,
    max_fill: usize,
    tick: &mut dyn FnMut() -> Result<(), EquilibriumError>,
) -> Result<bool, EquilibriumError> {
    if order == 0 {
        return Ok(rows.is_empty());
    }
    if rows.len() != order || rows.iter().flatten().any(|&(c, w)| c >= order || !w.is_finite()) {
        return Ok(false);
    }
    if order
        .checked_mul(order)
        .and_then(|c| c.checked_mul(8))
        .is_none_or(|b| b > MAX_DENSE_MATRIX_BYTES)
    {
        return Err(EquilibriumError::Resource("affine certificate dense matrix limit".into()));
    }
    let one = Interval { lower: 1.0, upper: 1.0 };
    let mut a_rows: Vec<Vec<(usize, Interval)>> = Vec::with_capacity(order);
    let mut numeric: Vec<Vec<(usize, f64)>> = Vec::with_capacity(order);
    for (i, row) in rows.iter().enumerate() {
        tick()?;
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
        return Ok(false);
    };
    let lu = match SparseLu::factor(&matrix, max_fill, tick) {
        Ok(lu) => lu,
        Err(FactorError::Singular | FactorError::NonFinite) => return Ok(false),
        Err(FactorError::FillBudget(required)) => {
            return Err(SolverBudgetCode::FactorFill.refusal(required as u64, max_fill as u64));
        }
        Err(FactorError::Interrupted(error)) => return Err(error),
    };
    let mut inverse = vec![0.0; order * order];
    for k in 0..order {
        tick()?;
        let mut unit = vec![0.0; order];
        unit[k] = 1.0;
        for (i, value) in lu.solve(&unit).into_iter().enumerate() {
            inverse[i * order + k] = value;
        }
    }
    if inverse.iter().any(|x| !x.is_finite()) {
        return Ok(false);
    }
    for i in 0..order {
        tick()?;
        let mut residual: Vec<Interval> = (0..order)
            .map(|j| {
                let d = if i == j { 1.0 } else { 0.0 };
                Interval { lower: d, upper: d }
            })
            .collect();
        for k in 0..order {
            if k % 32 == 0 {
                tick()?;
            }
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
            return Ok(false);
        }
    }
    Ok(true)
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
    fn the_admitted_certificate_agrees_and_honours_memory_and_cancellation() {
        let ctx = ExecutionContext::for_tests(0);
        assert!(matches!(certifies_affine_bijection_in(&[0.0, 2.0, -2.0, 0.0], 2, &ctx), Ok(true)));
        assert!(matches!(certifies_affine_bijection_in(&[0.0, 1.0, 1.0, 0.0], 2, &ctx), Ok(false)));
        // A 64 x 64 order needs more than 1 KiB: a tiny pool refuses before allocating.
        let tiny = ExecutionContext::for_tests(0).with_memory_limit(Some(1024));
        let weights = vec![0.0; 64 * 64];
        assert!(matches!(
            certifies_affine_bijection_in(&weights, 64, &tiny),
            Err(EquilibriumError::Resource(_))
        ));
        let cancelled = ExecutionContext::for_tests(0);
        cancelled.cancellation.cancel();
        assert!(matches!(
            certifies_affine_bijection_in(&[0.0, 2.0, -2.0, 0.0], 2, &cancelled),
            Err(EquilibriumError::Resource(_))
        ));
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
