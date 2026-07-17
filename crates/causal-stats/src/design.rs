//! Compiled design matrices for repeated estimator fits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop, clippy::manual_memcpy, clippy::cast_precision_loss)]

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

/// Contrast coding kind recorded on a compiled design (DESIGN.md §11.2 provenance).
///
/// Mirrors `causal_data::Contrast` without depending on that crate.
#[derive(Clone, Debug, PartialEq)]
pub enum ContrastCodingKind {
    /// Treatment coding relative to a reference level code.
    Treatment {
        /// Reference category code.
        reference: u32,
    },
    /// Sum-to-zero.
    SumToZero,
    /// Helmert.
    Helmert,
    /// Polynomial.
    Polynomial,
    /// Full-rank indicator (no drop).
    FullRankIndicator,
    /// Caller-supplied contrast matrix.
    Custom,
}

/// Provenance for one categorical variable expanded into design columns.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordedContrast {
    /// Source variable.
    pub variable: VariableId,
    /// Coding applied at compilation.
    pub coding: ContrastCodingKind,
    /// Ordered level labels at compilation time.
    pub level_labels: Arc<[Arc<str>]>,
    /// Half-open column range `[start, end)` into the design matrix.
    pub column_range: (usize, usize),
    /// Contrast matrix snapshot (`n_levels × n_columns`, column-major).
    pub matrix: Arc<[f64]>,
}

/// One column's standardization parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StandardizedColumn {
    /// Design-matrix column index.
    pub col_idx: usize,
    /// Mean subtracted before scaling.
    pub mean: f64,
    /// Scale divisor (`max(sd, eps)`); 1.0 if not scaled.
    pub scale: f64,
}

/// Record of column standardization applied at design compilation (DESIGN.md §11.2).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StandardizationRecord {
    /// Per-column mean/scale entries (empty = no standardization).
    pub entries: Vec<StandardizedColumn>,
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
    /// Categorical contrast provenance (empty for float-only designs).
    pub contrasts: Vec<RecordedContrast>,
    /// Standardization provenance (empty when columns are raw).
    pub standardization: StandardizationRecord,
}

impl CompiledDesign {
    /// Build `[1 | T | Z…]` design from contiguous float columns (same length).
    ///
    /// `row_selection` records provenance of retained rows (empty → `0..nrows`).
    /// Contrast and standardization records start empty.
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
            contrasts: Vec::new(),
            standardization: StandardizationRecord::default(),
        })
    }

    /// Attach contrast / standardization provenance without rebuilding the matrix.
    #[must_use]
    pub fn with_provenance(
        mut self,
        contrasts: Vec<RecordedContrast>,
        standardization: StandardizationRecord,
    ) -> Self {
        self.contrasts = contrasts;
        self.standardization = standardization;
        self
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

/// Standardize selected columns of a column-major design matrix in place.
///
/// For each column index in `col_idxs`, subtract the column mean and divide by
/// `max(sample_sd, eps)`. Returns the standardization record for bootstrap reuse.
///
/// # Errors
///
/// Out-of-range column index or empty matrix.
pub fn standardize_columns(
    matrix: &mut [f64],
    nrows: usize,
    ncols: usize,
    col_idxs: &[usize],
    eps: f64,
) -> Result<StandardizationRecord, StatsError> {
    if nrows == 0 || ncols == 0 {
        return Err(StatsError::Shape { message: "empty matrix for standardization" });
    }
    if matrix.len() != nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "matrix length mismatch" });
    }
    let eps = if eps.is_finite() && eps > 0.0 { eps } else { 1e-12 };
    let mut entries = Vec::with_capacity(col_idxs.len());
    for &col in col_idxs {
        if col >= ncols {
            return Err(StatsError::Shape { message: "standardize column out of range" });
        }
        let base = col * nrows;
        let slice = &mut matrix[base..base + nrows];
        let mean = slice.iter().sum::<f64>() / nrows as f64;
        let mut var = 0.0;
        for &v in slice.iter() {
            let d = v - mean;
            var += d * d;
        }
        let scale = if nrows > 1 {
            (var / (nrows - 1) as f64).sqrt().max(eps)
        } else {
            1.0_f64.max(eps)
        };
        for v in slice.iter_mut() {
            *v = (*v - mean) / scale;
        }
        entries.push(StandardizedColumn { col_idx: col, mean, scale });
    }
    Ok(StandardizationRecord { entries })
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
        assert!(design.contrasts.is_empty());
        assert!(design.standardization.entries.is_empty());
        let backend = FaerBackend;
        let mut ws = LeastSquaresWorkspace::default();
        let fit = design.fit_ols(&backend, &mut ws).unwrap();
        assert!((fit.coefficients[0] - 1.0).abs() < 1e-8);
        assert!((fit.coefficients[1] - 2.0).abs() < 1e-8);
        assert!((fit.coefficients[2] - 3.0).abs() < 1e-8);
        assert!(fit.rss < 1e-16);
        assert_eq!(fit.diagnostics.backend, "faer");
        assert_eq!(fit.diagnostics.rank, 3);
        assert!(fit.diagnostics.rcond.is_some());
        assert!(fit.diagnostics.grow_count >= 1);
    }

    #[test]
    fn standardize_columns_records_mean_scale() {
        let mut matrix = vec![
            1.0, 1.0, 1.0, 1.0, // intercept
            0.0, 2.0, 4.0, 6.0, // to standardize
        ];
        let rec = standardize_columns(&mut matrix, 4, 2, &[1], 1e-12).unwrap();
        assert_eq!(rec.entries.len(), 1);
        assert!((rec.entries[0].mean - 3.0).abs() < 1e-12);
        let col: Vec<f64> = matrix[4..8].to_vec();
        let mean: f64 = col.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-12);
    }

    #[test]
    fn with_provenance_attaches_without_rebuild() {
        let t = vec![0.0, 1.0];
        let y = vec![1.0, 2.0];
        let design = CompiledDesign::linear_adjustment(&t, &[], &y, &[]).unwrap();
        let ptr = design.matrix.as_ptr();
        let contrast = RecordedContrast {
            variable: VariableId::from_raw(9),
            coding: ContrastCodingKind::SumToZero,
            level_labels: Arc::from([Arc::<str>::from("a"), Arc::<str>::from("b")]),
            column_range: (2, 3),
            matrix: Arc::from([1.0_f64, -1.0]),
        };
        let design = design.with_provenance(
            vec![contrast],
            StandardizationRecord {
                entries: vec![StandardizedColumn { col_idx: 1, mean: 0.5, scale: 1.0 }],
            },
        );
        assert_eq!(design.matrix.as_ptr(), ptr);
        assert_eq!(design.contrasts.len(), 1);
        assert_eq!(design.standardization.entries.len(), 1);
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
        let res_ptr = ws.residuals.as_ptr();
        let res_cap = ws.residuals.capacity();
        for _ in 0..20 {
            let _ = design.fit_ols(&backend, &mut ws).unwrap();
            assert_eq!(ws.scratch.as_ptr(), ptr);
            assert_eq!(ws.scratch.capacity(), cap_scratch);
            assert_eq!(ws.residuals.as_ptr(), res_ptr);
            assert_eq!(ws.residuals.capacity(), res_cap);
        }
    }

    #[test]
    fn hot_path_allocation_gate_no_scratch_growth() {
        // Prepared OLS hot path performs no repeated scratch allocation.
        let n = 400usize;
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + z[i]).collect();
        let design = CompiledDesign::linear_adjustment(
            &t,
            &[(VariableId::from_raw(2), z.as_slice())],
            &y,
            &[],
        )
        .unwrap();
        let backend = FaerBackend;
        let mut ws = LeastSquaresWorkspace::default();
        let _ = design.fit_ols(&backend, &mut ws).unwrap();
        let scratch_ptr = ws.scratch.as_ptr();
        let scratch_cap = ws.scratch.capacity();
        let rhs_ptr = ws.rhs.as_ptr();
        let rhs_cap = ws.rhs.capacity();
        for _ in 0..250 {
            let _ = design.fit_ols(&backend, &mut ws).unwrap();
            assert_eq!(ws.scratch.as_ptr(), scratch_ptr);
            assert_eq!(ws.scratch.capacity(), scratch_cap);
            assert_eq!(ws.rhs.as_ptr(), rhs_ptr);
            assert_eq!(ws.rhs.capacity(), rhs_cap);
        }
    }
}
