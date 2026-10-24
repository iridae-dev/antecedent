//! Dense linear algebra backend abstraction (ADR 0001).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::StatsError;

/// Fit diagnostics required by.
#[derive(Clone, Debug, PartialEq)]
pub struct FitDiagnostics {
    /// Numerical rank estimate.
    pub rank: usize,
    /// Reciprocal condition estimate when available (`min|R_ii| / max|R_ii|`).
    pub rcond: Option<f64>,
    /// Backend identifier (`"faer"`, `"ridge"`, `"lasso"`, `"huber"`, …).
    pub backend: &'static str,
    /// Workspace grow count observed at fit time.
    pub grow_count: u32,
}

impl FitDiagnostics {
    /// Build diagnostics with optional condition number.
    #[must_use]
    pub const fn new(
        rank: usize,
        rcond: Option<f64>,
        backend: &'static str,
        grow_count: u32,
    ) -> Self {
        Self { rank, rcond, backend, grow_count }
    }
}

impl Default for FitDiagnostics {
    fn default() -> Self {
        Self { rank: 0, rcond: None, backend: "unknown", grow_count: 0 }
    }
}

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
    /// Rank / condition / backend / allocation diagnostics.
    pub diagnostics: FitDiagnostics,
}

/// Operation-level dense LA interface. Public causal APIs do not expose backend types.
pub trait DenseLinearAlgebra: Send + Sync {
    /// Solve `min ||X β − y||_2` for column-major `x` with `nrows` × `ncols`.
    ///
    /// When `nrows < ncols` and the numerical rank equals `nrows`, the solution
    /// is the minimum-norm coefficient vector. Rank deficiency is a numerical
    /// rank below `min(nrows, ncols)`, not the rectangular shape by itself.
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

    /// Solve `min ||diag(row_scale) X β − y||_2`, where `y` is already scaled by `row_scale`.
    ///
    /// This is the weighted-least-squares and IRLS step: the caller's `x` is read once, and a
    /// backend that can fold the row scale into its own factorization copy does so, so the
    /// scaled design never exists as a second buffer. The default builds that buffer and
    /// calls [`Self::least_squares`]; both give the same coefficients bit for bit.
    ///
    /// # Errors
    ///
    /// Shape mismatch (including a `row_scale` that is not `nrows` long), or the failures of
    /// [`Self::least_squares`].
    fn row_scaled_least_squares(
        &self,
        x_colmajor: &[f64],
        nrows: usize,
        ncols: usize,
        row_scale: &[f64],
        y: &[f64],
        workspace: &mut LeastSquaresWorkspace,
    ) -> Result<LeastSquaresFit, StatsError> {
        let Some(cells) = nrows.checked_mul(ncols) else {
            return Err(StatsError::Shape { message: "X buffer too short" });
        };
        if row_scale.len() != nrows {
            return Err(StatsError::Shape { message: "row scale length != nrows" });
        }
        if x_colmajor.len() < cells {
            return Err(StatsError::Shape { message: "X buffer too short" });
        }
        let mut scaled = vec![0.0; cells];
        if nrows > 0 {
            for (source, target) in
                x_colmajor[..cells].chunks_exact(nrows).zip(scaled.chunks_exact_mut(nrows))
            {
                for ((slot, &value), &scale) in target.iter_mut().zip(source).zip(row_scale) {
                    *slot = value * scale;
                }
            }
        }
        self.least_squares(&scaled, nrows, ncols, y, workspace)
    }
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
