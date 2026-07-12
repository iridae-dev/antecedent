//! `faer` dense backend.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names)]

use faer::Mat;
use faer::linalg::solvers::{FullPivLu, Solve};

use crate::error::StatsError;
use crate::linalg::{DenseLinearAlgebra, LeastSquaresFit, LeastSquaresWorkspace};

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

        // Form XᵀX and Xᵀy via explicit loops (stable, no per-call Mat alloc beyond temps).
        let xtx = &mut workspace.scratch[..ncols * ncols];
        xtx.fill(0.0);
        for c1 in 0..ncols {
            for c2 in c1..ncols {
                let mut acc = 0.0;
                let col1 = &x_colmajor[c1 * nrows..(c1 + 1) * nrows];
                let col2 = &x_colmajor[c2 * nrows..(c2 + 1) * nrows];
                for r in 0..nrows {
                    acc += col1[r] * col2[r];
                }
                xtx[c1 * ncols + c2] = acc;
                xtx[c2 * ncols + c1] = acc;
            }
        }
        let xty = &mut workspace.rhs[..ncols];
        xty.fill(0.0);
        for c in 0..ncols {
            let col = &x_colmajor[c * nrows..(c + 1) * nrows];
            let mut acc = 0.0;
            for r in 0..nrows {
                acc += col[r] * y[r];
            }
            xty[c] = acc;
        }

        // Solve XᵀX β = Xᵀy with faer FullPivLu.
        let a = Mat::<f64>::from_fn(ncols, ncols, |i, j| xtx[j * ncols + i]);
        let mut b = Mat::<f64>::from_fn(ncols, 1, |i, _| xty[i]);
        let lu = FullPivLu::new(a.as_ref());
        lu.solve_in_place(b.as_mut());
        let mut coefficients = vec![0.0; ncols];
        for i in 0..ncols {
            coefficients[i] = b[(i, 0)];
        }

        // Residuals and RSS.
        let residuals = &mut workspace.residuals[..nrows];
        for r in 0..nrows {
            let mut pred = 0.0;
            for c in 0..ncols {
                pred += x_colmajor[c * nrows + r] * coefficients[c];
            }
            residuals[r] = y[r] - pred;
        }
        let rss: f64 = residuals.iter().map(|e| e * e).sum();
        // Rank: count pivots with sufficient magnitude (simple heuristic).
        let mut rank = 0usize;
        for i in 0..ncols {
            if xtx[i * ncols + i].abs() > 1e-12 {
                rank += 1;
            }
        }
        if rank < ncols {
            return Err(StatsError::RankDeficient { rank, ncols });
        }

        Ok(LeastSquaresFit { coefficients, residuals: residuals.to_vec(), rank, rss })
    }
}
