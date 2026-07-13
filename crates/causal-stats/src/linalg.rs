//! Dense linear algebra backend abstraction (ADR 0001).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::StatsError;

/// Result of a least-squares solve.
#[derive(Clone, Debug)]
pub struct LeastSquaresFit {
    /// Coefficient vector (length = ncols).
    pub coefficients: Vec<f64>,
    /// Residuals (length = nrows).
    pub residuals: Vec<f64>,
    /// Numerical rank estimate.
    pub rank: usize,
    /// Residual sum of squares.
    pub rss: f64,
}

/// Operation-level dense LA interface. Public causal APIs do not expose backend types.
pub trait DenseLinearAlgebra: Send + Sync {
    /// Solve `min ||X β − y||_2` for column-major `x` with `nrows` × `ncols`.
    ///
    /// # Errors
    ///
    /// Shape or numerical failure.
    fn least_squares(
        &self,
        x_colmajor: &[f64],
        nrows: usize,
        ncols: usize,
        y: &[f64],
        workspace: &mut LeastSquaresWorkspace,
    ) -> Result<LeastSquaresFit, StatsError>;
}

/// Reusable scratch for repeated least-squares fits.
#[derive(Clone, Debug, Default)]
pub struct LeastSquaresWorkspace {
    /// Scratch for `XtX` / factorizations (backend-specific packing).
    pub scratch: Vec<f64>,
    /// Scratch for `Xty` / coefficients.
    pub rhs: Vec<f64>,
    /// Residual buffer.
    pub residuals: Vec<f64>,
    /// Times [`Self::prepare`] grew any buffer (reuse diagnostics).
    pub grow_count: u32,
}

impl LeastSquaresWorkspace {
    /// Ensure capacity for a design of the given shape (grows, does not shrink).
    pub fn prepare(&mut self, nrows: usize, ncols: usize) {
        let need_scratch = ncols.saturating_mul(ncols).max(nrows.saturating_mul(ncols));
        let mut grew = false;
        if self.scratch.len() < need_scratch {
            self.scratch.resize(need_scratch, 0.0);
            grew = true;
        }
        if self.rhs.len() < ncols {
            self.rhs.resize(ncols, 0.0);
            grew = true;
        }
        if self.residuals.len() < nrows {
            self.residuals.resize(nrows, 0.0);
            grew = true;
        }
        if grew {
            self.grow_count = self.grow_count.saturating_add(1);
        }
    }
}
