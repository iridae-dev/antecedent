//! `faer` dense backend — column-pivoted QR least squares (DESIGN.md §11.6).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names, clippy::cast_precision_loss)]

use faer::linalg::solvers::{ColPivQr, SolveLstsqCore};
use faer::{Conj, Mat};

use crate::error::StatsError;
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

        // Column-pivoted QR on X (not normal equations — DESIGN.md §11.6).
        let a = Mat::<f64>::from_fn(nrows, ncols, |r, c| x_colmajor[c * nrows + r]);
        let qr = ColPivQr::new(a.as_ref());

        // Rank from |R_ii| relative to the largest pivot.
        let r_factor = qr.thin_R();
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
        let tol = (nrows as f64).sqrt() * f64::EPSILON * max_diag.max(1.0);
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
        qr.solve_lstsq_in_place_with_conj(Conj::No, rhs.as_mut());

        let mut coefficients = vec![0.0; ncols];
        for i in 0..ncols {
            coefficients[i] = rhs[(i, 0)];
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

        Ok(LeastSquaresFit {
            coefficients,
            residuals: residuals.to_vec(),
            rank,
            rss,
            diagnostics: FitDiagnostics::new(rank, rcond, "faer", workspace.grow_count),
        })
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use super::*;

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
