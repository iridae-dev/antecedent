//! Exact pairs-bootstrap solves in a QR-orthonormalized basis.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use faer::linalg::householder::{
    apply_block_householder_sequence_on_the_left_in_place_scratch,
    apply_block_householder_sequence_on_the_left_in_place_with_conj,
};

use super::{
    Conj, FactorScratch, Mat, MemStack, StatsError, factor_equilibrated, serial_scope,
    solve_equilibrated, write_column_scales, write_residuals,
};

/// One immutable QR basis for repeated row-resampled least squares.
///
/// Replicate multiplicities come from the original row-index stream. Only
/// demonstrably well-conditioned systems use the small solve; `None` requests
/// the caller's original pivoted QR, including its numerical-rank policy.
pub struct ResampledLeastSquares {
    basis: Mat<f64>,
    residuals: Vec<f64>,
    anchor: Vec<f64>,
    r: Mat<f64>,
    permutation: Vec<usize>,
    scales: Vec<f64>,
}

/// Worker-owned count and small-system scratch for a pairs bootstrap.
#[derive(Default)]
pub struct ResampledLeastSquaresWorkspace {
    counts: Vec<usize>,
    gram: Vec<f64>,
    rhs: Vec<f64>,
}

impl ResampledLeastSquares {
    /// Prepare a bounded QR basis for an immutable source design.
    ///
    /// Returns `None` for small, large, budget-constrained, offset-dominated or
    /// poorly conditioned designs. No approximate resampling law is introduced.
    #[must_use]
    pub fn prepare(
        x: &[f64],
        n: usize,
        p: usize,
        y: &[f64],
        ctx: &ExecutionContext,
    ) -> Option<Self> {
        let cells = n.checked_mul(p)?;
        let retained = cells.checked_add(n)?.checked_add(4 * p + p * p)?.checked_mul(8)?;
        if n < 256
            || !(2..=16).contains(&p)
            || cells > 1_000_000
            || y.len() != n
            || x.len() < cells
            || ctx.kernel_policy.force_scalar
            || !ctx.cache_policy.enabled
            || ctx.cancellation.is_cancelled()
            || ctx.memory.hard_limit_bytes.is_some()
            || ctx.memory.soft_limit_bytes.is_some()
            || ctx.admission.byte_budget() != u64::MAX
            || ctx.cache_policy.max_bytes.is_some_and(|cap| retained as u64 > cap)
            || x[..cells].iter().chain(y).any(|v| !v.is_finite())
        {
            return None;
        }
        let mut scales = Vec::new();
        write_column_scales(x, n, p, &mut scales);
        let scope = serial_scope();
        let mut scratch = FactorScratch::new();
        let factor = factor_equilibrated(x, n, p, &scales, scope, &mut scratch).ok()?;
        let (rank, rcond) = factor.rank_and_rcond(n);
        if rank < p || rcond? < 1e-5 || ctx.cancellation.is_cancelled() {
            return None;
        }
        let anchor = solve_equilibrated(&factor, y, &scales, scope, &mut scratch).ok()?;
        let mut residuals = vec![0.0; n];
        let rss = write_residuals(x, n, p, y, &anchor, &mut residuals);
        let rms = (rss / n as f64).sqrt();
        let max_y = y.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        if !rms.is_finite() || rms == 0.0 || max_y / rms > 1e6 {
            return None;
        }
        let required = apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
            n,
            factor.q_coeff.nrows(),
            p,
        );
        scratch.ensure(scope, required).ok()?;
        let mut basis = Mat::from_fn(n, p, |i, j| if i == j { 1.0 } else { 0.0 });
        apply_block_householder_sequence_on_the_left_in_place_with_conj(
            factor.q_basis.as_ref(),
            factor.q_coeff.as_ref(),
            Conj::No,
            basis.as_mut(),
            faer::Par::Seq,
            MemStack::new(scratch.buffer_mut().ok()?),
        );
        if ctx.cancellation.is_cancelled() {
            return None;
        }
        let actual_bytes = usize::try_from(basis.col_stride())
            .ok()?
            .checked_mul(p)?
            .checked_add(usize::try_from(factor.thin_r.col_stride()).ok()?.checked_mul(p)?)?
            .checked_add(residuals.capacity())?
            .checked_add(anchor.capacity())?
            .checked_add(scales.capacity())?
            .checked_add(factor.perm_forward.capacity())?
            .checked_mul(size_of::<f64>())?
            .checked_add(size_of::<Self>())?;
        if ctx.cache_policy.max_bytes.is_some_and(|cap| actual_bytes as u64 > cap) {
            return None;
        }
        Some(Self {
            basis,
            residuals,
            anchor,
            r: factor.thin_r,
            permutation: factor.perm_forward,
            scales,
        })
    }

    /// Solve a row map with its exact integer multiplicities.
    ///
    /// `Ok(None)` requires pivoted-QR fallback. Rejected small systems are not
    /// counted as completed fits. No rows, uncertainty draws or rank failures
    /// are silently discarded.
    ///
    /// # Errors
    /// Execution-attempt failure, as for the ordinary least-squares backend.
    pub fn try_coefficients(
        &self,
        rows: &[usize],
        ws: &mut ResampledLeastSquaresWorkspace,
    ) -> Result<Option<Vec<f64>>, StatsError> {
        let n = self.basis.nrows();
        let p = self.basis.ncols();
        if rows.len() != n || rows.iter().any(|&r| r >= n) {
            return Ok(None);
        }
        ws.counts.resize(n, 0);
        ws.counts.fill(0);
        for &row in rows {
            ws.counts[row] += 1;
        }
        ws.gram.resize(p * p, 0.0);
        ws.gram.fill(0.0);
        ws.rhs.resize(p, 0.0);
        ws.rhs.fill(0.0);
        for (row, &count) in ws.counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let weight = count as f64;
            for i in 0..p {
                let qi = self.basis[(row, i)];
                ws.rhs[i] += weight * qi * self.residuals[row];
                for j in i..p {
                    ws.gram[i * p + j] += weight * qi * self.basis[(row, j)];
                }
            }
        }
        for i in 0..p {
            for j in i + 1..p {
                ws.gram[j * p + i] = ws.gram[i * p + j];
            }
        }
        // Gershgorin bounds in an orthonormal basis. This explicitly avoids
        // normal equations in the original, potentially badly scaled design.
        let mut lower = f64::INFINITY;
        let mut upper = 0.0_f64;
        for i in 0..p {
            let radius: f64 = (0..p).filter(|&j| j != i).map(|j| ws.gram[i * p + j].abs()).sum();
            lower = lower.min(ws.gram[i * p + i] - radius);
            upper = upper.max(ws.gram[i * p + i] + radius);
        }
        if !lower.is_finite() || lower <= 0.05 || upper / lower > 100.0 {
            return Ok(None);
        }
        let Some(chol) = crate::gram::cholesky_spd(&ws.gram, p) else { return Ok(None) };
        let Some(mut delta) = crate::gram::chol_solve(&chol, p, &ws.rhs) else { return Ok(None) };
        let mut defect = 0.0_f64;
        let mut scale = 0.0_f64;
        for i in 0..p {
            let product: f64 = (0..p).map(|j| ws.gram[i * p + j] * delta[j]).sum();
            defect = defect.max((product - ws.rhs[i]).abs());
            scale = scale.max(
                ws.rhs[i].abs()
                    + (0..p).map(|j| (ws.gram[i * p + j] * delta[j]).abs()).sum::<f64>(),
            );
        }
        if !defect.is_finite() || defect > 64.0 * p as f64 * f64::EPSILON * scale {
            return Ok(None);
        }
        for i in (0..p).rev() {
            let tail: f64 = (i + 1..p).map(|j| self.r[(i, j)] * delta[j]).sum();
            delta[i] = (delta[i] - tail) / self.r[(i, i)];
        }
        let mut coefficients = self.anchor.clone();
        for (i, &change) in delta.iter().enumerate() {
            let original = self.permutation[i];
            coefficients[original] += change / self.scales[original];
        }
        if coefficients.iter().any(|v| !v.is_finite()) {
            return Ok(None);
        }
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::LeastSquaresSolve,
            || {
                crate::fit_counts::completed_solve();
                Ok(Some(coefficients))
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FaerBackend;

    #[test]
    fn same_pairs_resamples_match_pivoted_qr_with_permutation_and_scaling() {
        let n = 512;
        let p = 3;
        let mut x = vec![0.0; n * p];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1e-3 * (i as f64 * 0.31).sin();
            x[n + i] = 100.0;
            x[2 * n + i] = (i as f64 * 0.17).cos();
            y[i] = 0.2 + 17.0 * x[i] + 1.5 * x[2 * n + i] + (i as f64 * 0.71).sin();
        }
        let ctx = ExecutionContext::production(19, 1);
        let prepared = ResampledLeastSquares::prepare(&x, n, p, &y, &ctx).unwrap();
        let mut ws = ResampledLeastSquaresWorkspace::default();
        let mut state = 7u64;
        let mut accepted = 0;
        for _ in 0..64 {
            let rows: Vec<_> = (0..n)
                .map(|_| {
                    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    usize::try_from((z ^ (z >> 31)) % n as u64).unwrap()
                })
                .collect();
            let source = &x;
            let bx: Vec<_> =
                (0..p).flat_map(|c| rows.iter().map(move |&i| source[c * n + i])).collect();
            let by: Vec<_> = rows.iter().map(|&i| y[i]).collect();
            let expected = FaerBackend.least_squares_coefficients(&bx, n, p, &by).unwrap();
            if let Some(actual) = prepared.try_coefficients(&rows, &mut ws).unwrap() {
                accepted += 1;
                for (a, b) in actual.iter().zip(expected) {
                    assert!((a - b).abs() <= 1e-9 * (1.0 + b.abs()), "{actual:?}, reference {b}");
                }
            }
        }
        assert_eq!(accepted, 64);
        assert!(prepared.try_coefficients(&vec![0; n], &mut ws).unwrap().is_none());
        let offset: Vec<_> = y.iter().map(|v| v + 1e12).collect();
        assert!(ResampledLeastSquares::prepare(&x, n, p, &offset, &ctx).is_none());
        let mut bounded = ctx;
        bounded.cache_policy = antecedent_core::CachePolicy::enabled(Some(0));
        assert!(ResampledLeastSquares::prepare(&x, n, p, &y, &bounded).is_none());
    }
}
