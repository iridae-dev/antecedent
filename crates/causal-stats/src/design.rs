//! Compiled design matrices for repeated estimator fits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]

use std::sync::Arc;

use causal_core::VariableId;

use crate::error::StatsError;
use crate::linalg::{DenseLinearAlgebra, LeastSquaresFit, LeastSquaresWorkspace};

/// Column role in a compiled design.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DesignColumnRole {
    /// Intercept.
    Intercept,
    /// Treatment variable.
    Treatment,
    /// Covariate / adjustment variable.
    Covariate(VariableId),
}

/// Compiled design matrix (column-major) reusable across fits.
#[derive(Clone, Debug)]
pub struct CompiledDesign {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Column-major values.
    pub matrix: Arc<[f64]>,
    /// Column roles.
    pub columns: Arc<[DesignColumnRole]>,
    /// Outcome vector aligned with rows.
    pub outcome: Arc<[f64]>,
    /// Original row indices retained after validity / analysis-mask filtering.
    pub row_selection: Arc<[usize]>,
}

impl CompiledDesign {
    /// Build `[1 | T | Z…]` design from contiguous float columns (same length).
    ///
    /// `row_selection` records provenance of retained rows (empty → `0..nrows`).
    ///
    /// # Errors
    ///
    /// Length mismatches or empty data.
    pub fn linear_adjustment(
        treatment: &[f64],
        covariates: &[(VariableId, &[f64])],
        outcome: &[f64],
        row_selection: &[usize],
    ) -> Result<Self, StatsError> {
        let nrows = outcome.len();
        if nrows == 0 {
            return Err(StatsError::Shape { message: "empty design" });
        }
        if treatment.len() != nrows {
            return Err(StatsError::Shape { message: "treatment length mismatch" });
        }
        for (_, col) in covariates {
            if col.len() != nrows {
                return Err(StatsError::Shape { message: "covariate length mismatch" });
            }
        }
        let selection: Arc<[usize]> = if row_selection.is_empty() {
            Arc::from((0..nrows).collect::<Vec<_>>())
        } else if row_selection.len() == nrows {
            Arc::from(row_selection.to_vec())
        } else {
            return Err(StatsError::Shape { message: "row_selection length mismatch" });
        };
        let ncols = 2 + covariates.len();
        let mut matrix = vec![0.0; nrows * ncols];
        // col 0: intercept
        for r in 0..nrows {
            matrix[r] = 1.0;
        }
        // col 1: treatment
        for r in 0..nrows {
            matrix[nrows + r] = treatment[r];
        }
        let mut roles = vec![DesignColumnRole::Intercept, DesignColumnRole::Treatment];
        for (i, (id, col)) in covariates.iter().enumerate() {
            let base = (2 + i) * nrows;
            for r in 0..nrows {
                matrix[base + r] = col[r];
            }
            roles.push(DesignColumnRole::Covariate(*id));
        }
        Ok(Self {
            nrows,
            ncols,
            matrix: Arc::from(matrix),
            columns: Arc::from(roles),
            outcome: Arc::from(outcome.to_vec()),
            row_selection: selection,
        })
    }

    /// Fit OLS using `backend` and reusable `workspace`.
    ///
    /// # Errors
    ///
    /// Propagates backend errors.
    pub fn fit_ols(
        &self,
        backend: &impl DenseLinearAlgebra,
        workspace: &mut LeastSquaresWorkspace,
    ) -> Result<LeastSquaresFit, StatsError> {
        backend.least_squares(&self.matrix, self.nrows, self.ncols, &self.outcome, workspace)
    }

    /// Index of the treatment column (always 1 for [`linear_adjustment`]).
    #[must_use]
    pub fn treatment_column(&self) -> Option<usize> {
        self.columns.iter().position(|c| matches!(c, DesignColumnRole::Treatment))
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use causal_core::VariableId;

    use super::*;
    use crate::faer_backend::FaerBackend;

    #[test]
    fn ols_recovers_known_coefficients() {
        // y = 1 + 2 t + 3 z + noise
        let n = 200usize;
        let t: Vec<f64> = (0..n).map(|i| (i as f64) / n as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| ((i * 3) % 7) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + 3.0 * z[i]).collect();
        let design = CompiledDesign::linear_adjustment(
            &t,
            &[(VariableId::from_raw(2), z.as_slice())],
            &y,
            &[],
        )
        .unwrap();
        let backend = FaerBackend;
        let mut ws = LeastSquaresWorkspace::default();
        let fit = design.fit_ols(&backend, &mut ws).unwrap();
        assert!((fit.coefficients[0] - 1.0).abs() < 1e-8);
        assert!((fit.coefficients[1] - 2.0).abs() < 1e-8);
        assert!((fit.coefficients[2] - 3.0).abs() < 1e-8);
        assert!(fit.rss < 1e-16);
    }

    #[test]
    fn repeated_fits_reuse_workspace_capacity() {
        let t = vec![0.0, 1.0, 0.0, 1.0];
        let y = vec![1.0, 3.0, 1.5, 2.5];
        let design = CompiledDesign::linear_adjustment(&t, &[], &y, &[]).unwrap();
        let backend = FaerBackend;
        let mut ws = LeastSquaresWorkspace::default();
        let _ = design.fit_ols(&backend, &mut ws).unwrap();
        let cap_scratch = ws.scratch.capacity();
        let ptr = ws.scratch.as_ptr();
        for _ in 0..20 {
            let _ = design.fit_ols(&backend, &mut ws).unwrap();
            assert_eq!(ws.scratch.as_ptr(), ptr);
            assert_eq!(ws.scratch.capacity(), cap_scratch);
        }
    }
}
