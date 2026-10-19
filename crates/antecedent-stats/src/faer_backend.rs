//! `faer` dense backend — column-pivoted QR least squares.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use faer::diag::Diag;
use faer::dyn_stack::MemStack;
use faer::linalg::qr::col_pivoting::factor::{
    qr_in_place, qr_in_place_scratch, recommended_block_size,
};
use faer::linalg::qr::col_pivoting::solve::{
    solve_lstsq_in_place_scratch, solve_lstsq_in_place_with_conj,
};
use faer::linalg::svd::{ComputeSvdVectors, svd, svd_scratch};
use faer::{Conj, Mat, MatRef, Spec};

use crate::error::StatsError;
use crate::faer_scope::FaerScope;
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresFit, LeastSquaresWorkspace};

/// Default `faer` backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct FaerBackend;

impl DenseLinearAlgebra for FaerBackend {
    fn least_squares(
        &self,
        x_colmajor: &[f64],
        nrows: usize,
        ncols: usize,
        y: &[f64],
        workspace: &mut LeastSquaresWorkspace,
    ) -> Result<LeastSquaresFit, StatsError> {
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::LeastSquaresSolve,
            || {
                if y.len() != nrows {
                    return Err(StatsError::Shape { message: "y length != nrows" });
                }
                if x_colmajor.len() < nrows.saturating_mul(ncols) {
                    return Err(StatsError::Shape { message: "X buffer too short" });
                }
                if nrows < ncols {
                    return Err(StatsError::Shape { message: "nrows < ncols" });
                }
                workspace.prepare(nrows, ncols);

                // Equilibrate columns before QR so predictor units do not determine
                // numerical rank. Solve in these coordinates, then restore coefficients.
                let scales: Vec<_> = (0..ncols)
                    .map(|c| {
                        let scale = x_colmajor[c * nrows..(c + 1) * nrows]
                            .iter()
                            .copied()
                            .map(f64::abs)
                            .fold(0.0_f64, f64::max);
                        if scale == 0.0 { 1.0 } else { scale }
                    })
                    .collect();
                let a =
                    Mat::<f64>::from_fn(nrows, ncols, |r, c| x_colmajor[c * nrows + r] / scales[c]);
                let scope = serial_scope();
                let factored = column_pivoted_qr(a, scope)?;

                // Rank from |R_ii| relative to the largest pivot.
                let r_factor = factored.thin_r.as_ref();
                let size = r_factor.nrows().min(r_factor.ncols());
                let mut max_diag = 0.0_f64;
                let mut min_diag = f64::INFINITY;
                for i in 0..size {
                    let d = r_factor[(i, i)].abs();
                    max_diag = max_diag.max(d);
                    if d > 0.0 {
                        min_diag = min_diag.min(d);
                    }
                }
                let tol = (nrows as f64).sqrt() * f64::EPSILON * max_diag;
                let mut rank = 0usize;
                for i in 0..size {
                    if r_factor[(i, i)].abs() > tol {
                        rank += 1;
                    }
                }
                if rank < ncols {
                    return Err(StatsError::RankDeficient { rank, ncols });
                }
                let rcond = if max_diag > 0.0 && min_diag.is_finite() {
                    Some(min_diag / max_diag)
                } else {
                    None
                };

                // solve_lstsq writes β into the leading ncols entries of the RHS.
                let mut rhs = Mat::<f64>::from_fn(nrows, 1, |r, _| y[r]);
                solve_with_scope(&factored, rhs.as_mut(), scope)?;

                let mut coefficients = vec![0.0; ncols];
                for i in 0..ncols {
                    coefficients[i] = rhs[(i, 0)] / scales[i];
                }

                let residuals = &mut workspace.residuals[..nrows];
                for r in 0..nrows {
                    let mut pred = 0.0;
                    for c in 0..ncols {
                        pred += x_colmajor[c * nrows + r] * coefficients[c];
                    }
                    residuals[r] = y[r] - pred;
                }
                let rss: f64 = residuals.iter().map(|e| e * e).sum();

                crate::fit_counts::completed_solve();
                Ok(LeastSquaresFit {
                    coefficients,
                    residuals: residuals.to_vec(),
                    rank,
                    rss,
                    diagnostics: FitDiagnostics::new(rank, rcond, "faer", workspace.grow_count),
                })
            },
        )
    }
}

impl FaerBackend {
    /// Minimum-norm least squares `β = X⁺ y` through a thin SVD, for designs that may be rank
    /// deficient.
    ///
    /// Singular values at or below `max(nrows, ncols) · ε · σ_max` are treated as zero. When
    /// the columns are collinear the coefficients are not identified individually; this picks
    /// the solution of smallest Euclidean norm, whose fitted values and residuals are the
    /// orthogonal projection of `y` on the column space, the same as any least-squares solution.
    /// Prefer [`DenseLinearAlgebra::least_squares`] when the design is full rank.
    ///
    /// # Errors
    ///
    /// Shape mismatch, or an SVD that fails to converge.
    pub fn min_norm_least_squares(
        &self,
        x_colmajor: &[f64],
        nrows: usize,
        ncols: usize,
        y: &[f64],
    ) -> Result<LeastSquaresFit, StatsError> {
        if y.len() != nrows {
            return Err(StatsError::Shape { message: "y length != nrows" });
        }
        if x_colmajor.len() < nrows.saturating_mul(ncols) {
            return Err(StatsError::Shape { message: "X buffer too short" });
        }
        let a = Mat::<f64>::from_fn(nrows, ncols, |r, c| x_colmajor[c * nrows + r]);
        let svd = thin_svd_in_scope(a.as_ref(), serial_scope()).map_err(|_| {
            StatsError::Backend("minimum-norm least squares: SVD did not converge".into())
        })?;
        let (u, s, v) = (svd.u.as_ref(), svd.s.as_ref().column_vector(), svd.v.as_ref());
        let size = nrows.min(ncols);
        let s_max = (0..size).map(|i| s[i].abs()).fold(0.0_f64, f64::max);
        let tol = (nrows.max(ncols) as f64) * f64::EPSILON * s_max;
        let mut coefficients = vec![0.0; ncols];
        let mut rank = 0usize;
        let mut s_min = f64::INFINITY;
        for i in 0..size {
            let si = s[i].abs();
            if si <= tol {
                continue;
            }
            rank += 1;
            s_min = s_min.min(si);
            let uty: f64 = (0..nrows).map(|r| u[(r, i)] * y[r]).sum();
            let w = uty / s[i];
            for (c, coef) in coefficients.iter_mut().enumerate() {
                *coef += v[(c, i)] * w;
            }
        }
        let residuals: Vec<f64> = (0..nrows)
            .map(|r| {
                y[r] - (0..ncols).map(|c| x_colmajor[c * nrows + r] * coefficients[c]).sum::<f64>()
            })
            .collect();
        let rss = residuals.iter().map(|e| e * e).sum();
        let rcond = (rank > 0).then(|| s_min / s_max);
        Ok(LeastSquaresFit {
            coefficients,
            residuals,
            rank,
            rss,
            diagnostics: FitDiagnostics::new(rank, rcond, "faer-svd", 0),
        })
    }
}

struct ScopedQr {
    q_basis: Mat<f64>,
    q_coeff: Mat<f64>,
    r_full: Mat<f64>,
    thin_r: Mat<f64>,
    perm_forward: Vec<usize>,
    perm_inverse: Vec<usize>,
}

pub(crate) fn serial_scope() -> FaerScope {
    FaerScope::try_new(1, u64::MAX).unwrap_or_else(|_| unreachable!("one worker is a scope"))
}

fn column_pivoted_qr(mut qr: Mat<f64>, scope: FaerScope) -> Result<ScopedQr, StatsError> {
    let (nrows, ncols) = qr.shape();
    let par = scope.parallelism();
    let block_size = recommended_block_size::<f64>(nrows, ncols);
    let scratch = faer::dyn_stack::StackReq::and(
        qr_in_place_scratch::<usize, f64>(nrows, ncols, block_size, par, Spec::default()),
        solve_lstsq_in_place_scratch::<usize, f64>(nrows, ncols, block_size, 1, par),
    );
    scope.admit_scratch(scratch).map_err(|error| StatsError::Backend(error.to_string()))?;
    let mut buffer =
        scope.scratch_buffer(scratch).map_err(|error| StatsError::Backend(error.to_string()))?;
    let size = nrows.min(ncols);
    let mut q_coeff = Mat::<f64>::zeros(block_size, size);
    let mut perm_forward = vec![0usize; ncols];
    let mut perm_inverse = vec![0usize; ncols];
    let _ = qr_in_place(
        qr.as_mut(),
        q_coeff.as_mut(),
        &mut perm_forward,
        &mut perm_inverse,
        par,
        MemStack::new(&mut buffer),
        Spec::default(),
    );
    let (q_basis, r_full) = split_triangular(qr);
    let thin_r = r_full.get(..size, ..size).to_owned();
    Ok(ScopedQr { q_basis, q_coeff, r_full, thin_r, perm_forward, perm_inverse })
}

fn solve_with_scope(
    factored: &ScopedQr,
    rhs: faer::MatMut<'_, f64>,
    scope: FaerScope,
) -> Result<(), StatsError> {
    let par = scope.parallelism();
    let (nrows, ncols) = factored.q_basis.shape();
    let block_size = factored.q_coeff.nrows();
    let scratch =
        solve_lstsq_in_place_scratch::<usize, f64>(nrows, ncols, block_size, rhs.ncols(), par);
    scope.admit_scratch(scratch).map_err(|error| StatsError::Backend(error.to_string()))?;
    let mut buffer =
        scope.scratch_buffer(scratch).map_err(|error| StatsError::Backend(error.to_string()))?;
    let perm =
        faer::perm::PermRef::new_checked(&factored.perm_forward, &factored.perm_inverse, ncols);
    solve_lstsq_in_place_with_conj(
        factored.q_basis.as_ref(),
        factored.q_coeff.as_ref(),
        factored.r_full.as_ref(),
        perm,
        Conj::No,
        rhs,
        par,
        MemStack::new(&mut buffer),
    );
    Ok(())
}

fn split_triangular(lu: Mat<f64>) -> (Mat<f64>, Mat<f64>) {
    let (nrows, ncols) = lu.shape();
    let size = nrows.min(ncols);
    if nrows >= ncols {
        let mut lower = lu;
        let mut upper = Mat::<f64>::zeros(size, size);
        upper.copy_from_triangular_upper(lower.get(..size, ..size));
        for c in 0..size {
            for r in 0..c {
                lower[(r, c)] = 0.0;
            }
            lower[(c, c)] = 1.0;
        }
        (lower, upper)
    } else {
        let mut upper = lu;
        let mut lower = Mat::<f64>::zeros(size, size);
        lower.copy_from_strict_triangular_lower(upper.get(..size, ..size));
        for c in 0..ncols {
            for r in (c + 1)..size {
                upper[(r, c)] = 0.0;
            }
        }
        for i in 0..size {
            lower[(i, i)] = 1.0;
        }
        (lower, upper)
    }
}

pub(crate) fn symmetric_eigen_in_scope(
    matrix: MatRef<'_, f64>,
    scope: FaerScope,
) -> Result<(Diag<f64>, Mat<f64>), StatsError> {
    let n = matrix.nrows();
    let par = scope.parallelism();
    let scratch = faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
        n,
        faer::linalg::evd::ComputeEigenvectors::Yes,
        par,
        Spec::default(),
    );
    scope.admit_scratch(scratch).map_err(|error| StatsError::Backend(error.to_string()))?;
    let mut buffer =
        scope.scratch_buffer(scratch).map_err(|error| StatsError::Backend(error.to_string()))?;
    let mut values = Diag::<f64>::zeros(n);
    let mut vectors = Mat::<f64>::zeros(n, n);
    faer::linalg::evd::self_adjoint_evd(
        matrix,
        values.as_mut(),
        Some(vectors.as_mut()),
        par,
        MemStack::new(&mut buffer),
        Spec::default(),
    )
    .map_err(|_| StatsError::Backend("symmetric eigendecomposition failed".into()))?;
    Ok((values, vectors))
}

struct ThinSvd {
    u: Mat<f64>,
    s: Diag<f64>,
    v: Mat<f64>,
}

fn thin_svd_in_scope(
    a: MatRef<'_, f64>,
    scope: FaerScope,
) -> Result<ThinSvd, faer::linalg::svd::SvdError> {
    let (nrows, ncols) = a.shape();
    let size = nrows.min(ncols);
    let par = scope.parallelism();
    let compute = ComputeSvdVectors::Thin;
    let scratch = svd_scratch::<f64>(nrows, ncols, compute, compute, par, Spec::default());
    scope.admit_scratch(scratch).map_err(|_| faer::linalg::svd::SvdError::NoConvergence)?;
    let mut buffer =
        scope.scratch_buffer(scratch).map_err(|_| faer::linalg::svd::SvdError::NoConvergence)?;
    let mut u = Mat::<f64>::zeros(nrows, size);
    let mut v = Mat::<f64>::zeros(ncols, size);
    let mut s = Diag::<f64>::zeros(size);
    svd(
        a,
        s.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        par,
        MemStack::new(&mut buffer),
        Spec::default(),
    )?;
    Ok(ThinSvd { u, s, v })
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "tests assert exactly representable values (rank counts, copied inputs) and cast small fixture indices"
)]
mod tests {
    #[test]
    fn review_qr_rank_is_invariant_to_design_units() {
        for scale in [1e-30, 1.0, 1e30] {
            let x = [scale, scale, scale, scale, 0.0, scale, 2.0 * scale, 3.0 * scale];
            let y = [3.0, 7.0, 11.0, 15.0];
            let fit = FaerBackend
                .least_squares(&x, 4, 2, &y, &mut LeastSquaresWorkspace::default())
                .unwrap();
            assert!((fit.coefficients[0] * scale - 3.0).abs() < 1e-12);
            assert!((fit.coefficients[1] * scale - 4.0).abs() < 1e-12);
            let independent_units = [1.0, 1.0, 1.0, 1.0, 0.0, scale, 2.0 * scale, 3.0 * scale];
            let fit = FaerBackend
                .least_squares(&independent_units, 4, 2, &y, &mut LeastSquaresWorkspace::default())
                .unwrap();
            assert!((fit.coefficients[0] - 3.0).abs() < 1e-12);
            assert!((fit.coefficients[1] * scale - 4.0).abs() < 1e-12);
        }
    }

    use super::*;

    /// `x1 = 2·x0`: the minimum-norm solution splits the effect `β0 + 2β1 = 3` as
    /// `(3/5, 6/5)` and reproduces the least-squares fit exactly.
    #[test]
    fn min_norm_least_squares_resolves_collinear_columns() {
        let n = 20usize;
        let x0: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
        let mut x = x0.clone();
        x.extend(x0.iter().map(|v| 2.0 * v));
        let y: Vec<f64> = x0.iter().map(|v| 3.0 * v).collect();
        assert!(
            FaerBackend.least_squares(&x, n, 2, &y, &mut LeastSquaresWorkspace::default()).is_err()
        );
        let fit = FaerBackend.min_norm_least_squares(&x, n, 2, &y).unwrap();
        assert_eq!(fit.rank, 1);
        assert!((fit.coefficients[0] - 0.6).abs() < 1e-12, "{:?}", fit.coefficients);
        assert!((fit.coefficients[1] - 1.2).abs() < 1e-12, "{:?}", fit.coefficients);
        assert!(fit.rss < 1e-24);

        // Full rank: agrees with the QR solve.
        let mut full = x0.clone();
        full.extend((0..n).map(|i| i as f64 / n as f64));
        let yf: Vec<f64> = (0..n).map(|i| 1.5 * full[i] - 0.5 * full[n + i]).collect();
        let qr = FaerBackend
            .least_squares(&full, n, 2, &yf, &mut LeastSquaresWorkspace::default())
            .unwrap();
        let mn = FaerBackend.min_norm_least_squares(&full, n, 2, &yf).unwrap();
        for (a, b) in qr.coefficients.iter().zip(&mn.coefficients) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    #[test]
    fn qr_recovers_known_line() {
        let n = 50usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            y[i] = 3.0 + 4.0 * (i as f64);
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = FaerBackend.least_squares(&x, n, 2, &y, &mut ws).unwrap();
        assert!((fit.coefficients[0] - 3.0).abs() < 1e-10);
        assert!((fit.coefficients[1] - 4.0).abs() < 1e-10);
        assert_eq!(fit.rank, 2);
        assert!(fit.rss < 1e-20);
    }

    #[test]
    fn rank_deficient_rejected() {
        let n = 10usize;
        let mut x = vec![0.0; n * 2];
        let y = vec![1.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = 2.0; // duplicate of intercept up to scale
        }
        let mut ws = LeastSquaresWorkspace::default();
        let err = FaerBackend.least_squares(&x, n, 2, &y, &mut ws).unwrap_err();
        assert!(matches!(err, StatsError::RankDeficient { .. }));
    }
}
