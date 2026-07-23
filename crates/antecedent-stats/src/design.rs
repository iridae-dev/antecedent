//! Compiled design matrices for repeated estimator fits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop, clippy::manual_memcpy, clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent_core::{KernelPolicy, VariableId};
use antecedent_kernels::standardize_inplace;

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

/// One design-matrix column with role and provenance links.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct DesignColumn {
    /// Role of this column.
    pub role: DesignColumnRole,
    /// Index into [`CompiledDesign::contrasts`] when expanded from a categorical.
    pub contrast_idx: Option<usize>,
    /// Index into [`StandardizationRecord::entries`] when this column was standardized.
    pub standardization_idx: Option<usize>,
    /// Index into [`CompiledDesign::smooths`] when expanded from a smooth term.
    pub smooth_idx: Option<usize>,
}

impl DesignColumn {
    /// Column with role only (no contrast / standardization / smooth link).
    #[must_use]
    pub const fn from_role(role: DesignColumnRole) -> Self {
        Self { role, contrast_idx: None, standardization_idx: None, smooth_idx: None }
    }
}

/// Richer column-metadata map for [`CompiledDesign`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DesignColumnMap {
    columns: Arc<[DesignColumn]>,
}

impl DesignColumnMap {
    /// Build from a slice of columns.
    #[must_use]
    pub fn from_columns(columns: impl Into<Arc<[DesignColumn]>>) -> Self {
        Self { columns: columns.into() }
    }

    /// Build from roles only (no provenance links).
    #[must_use]
    pub fn from_roles(roles: impl IntoIterator<Item = DesignColumnRole>) -> Self {
        let cols: Vec<DesignColumn> = roles.into_iter().map(DesignColumn::from_role).collect();
        Self { columns: Arc::from(cols) }
    }

    /// Number of columns.
    #[must_use]
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Whether the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Role at column index, if in range.
    #[must_use]
    pub fn role(&self, index: usize) -> Option<DesignColumnRole> {
        self.columns.get(index).map(|c| c.role)
    }

    /// Full column metadata at index, if in range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&DesignColumn> {
        self.columns.get(index)
    }

    /// Iterate column metadata in matrix order.
    pub fn iter(&self) -> impl Iterator<Item = &DesignColumn> {
        self.columns.iter()
    }

    /// Underlying slice.
    #[must_use]
    pub fn as_slice(&self) -> &[DesignColumn] {
        &self.columns
    }

    /// Index of the treatment column, if present.
    #[must_use]
    pub fn treatment_column(&self) -> Option<usize> {
        self.columns.iter().position(|c| matches!(c.role, DesignColumnRole::Treatment))
    }

    /// Attach standardization entry indexes from a [`StandardizationRecord`].
    ///
    /// Clears previous `standardization_idx` values, then sets each entry's
    /// `col_idx` → entry index in `record.entries`.
    #[must_use]
    pub fn with_standardization_links(mut self, record: &StandardizationRecord) -> Self {
        let mut cols = self.columns.as_ref().to_vec();
        for c in &mut cols {
            c.standardization_idx = None;
        }
        for (ei, entry) in record.entries.iter().enumerate() {
            if let Some(col) = cols.get_mut(entry.col_idx) {
                col.standardization_idx = Some(ei);
            }
        }
        self.columns = Arc::from(cols);
        self
    }

    /// Attach contrast indexes from recorded contrasts (`column_range` → contrast index).
    #[must_use]
    pub fn with_contrast_links(mut self, contrasts: &[RecordedContrast]) -> Self {
        let mut cols = self.columns.as_ref().to_vec();
        for c in &mut cols {
            c.contrast_idx = None;
        }
        for (ci, contrast) in contrasts.iter().enumerate() {
            let (start, end) = contrast.column_range;
            for col in cols.iter_mut().take(end).skip(start) {
                col.contrast_idx = Some(ci);
            }
        }
        self.columns = Arc::from(cols);
        self
    }

    /// Attach smooth indexes from recorded smooths (`column_range` → smooth index).
    #[must_use]
    pub fn with_smooth_links(mut self, smooths: &[RecordedSmooth]) -> Self {
        let mut cols = self.columns.as_ref().to_vec();
        for c in &mut cols {
            c.smooth_idx = None;
        }
        for (si, smooth) in smooths.iter().enumerate() {
            let (start, end) = smooth.column_range;
            for col in cols.iter_mut().take(end).skip(start) {
                col.smooth_idx = Some(si);
            }
        }
        self.columns = Arc::from(cols);
        self
    }
}

impl From<Vec<DesignColumn>> for DesignColumnMap {
    fn from(value: Vec<DesignColumn>) -> Self {
        Self { columns: Arc::from(value) }
    }
}

impl From<Arc<[DesignColumn]>> for DesignColumnMap {
    fn from(value: Arc<[DesignColumn]>) -> Self {
        Self { columns: value }
    }
}

/// Contrast coding kind recorded on a compiled design.
///
/// Mirrors `antecedent_data::Contrast` without depending on that crate.
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

/// Spline / smooth basis kind recorded on a compiled additive design.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BasisKind {
    /// Cubic B-spline basis.
    CubicBSpline,
}

/// Provenance for one smooth term expanded into design columns.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordedSmooth {
    /// Source covariate variable (when known).
    pub variable: Option<VariableId>,
    /// Basis family used for expansion.
    pub basis: BasisKind,
    /// Knot sequence used at expansion (including boundary knots).
    pub knots: Arc<[f64]>,
    /// Ridge penalty λ applied to basis coefficients within this smooth.
    pub lambda: f64,
    /// Half-open column range `[start, end)` into the design matrix.
    pub column_range: (usize, usize),
    /// Number of basis columns (`end - start`).
    pub n_basis: usize,
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

/// Record of column standardization applied at design compilation.
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
    /// Column metadata (roles + provenance links); dense storage stays on `matrix`.
    pub columns: DesignColumnMap,
    /// Outcome vector aligned with rows.
    pub outcome: Arc<[f64]>,
    /// Original row indices retained after validity / analysis-mask filtering.
    pub row_selection: Arc<[usize]>,
    /// Categorical contrast provenance (empty for float-only designs).
    pub contrasts: Vec<RecordedContrast>,
    /// Smooth / spline provenance (empty when no additive terms).
    pub smooths: Vec<RecordedSmooth>,
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
            columns: DesignColumnMap::from_roles(roles),
            outcome: Arc::from(outcome.to_vec()),
            row_selection: selection,
            contrasts: Vec::new(),
            smooths: Vec::new(),
            standardization: StandardizationRecord::default(),
        })
    }

    /// Attach contrast / standardization provenance without rebuilding the matrix.
    ///
    /// Also refreshes per-column `contrast_idx` / `standardization_idx` links.
    /// Existing [`Self::smooths`] links are preserved via [`DesignColumnMap::with_smooth_links`].
    #[must_use]
    pub fn with_provenance(
        mut self,
        contrasts: Vec<RecordedContrast>,
        standardization: StandardizationRecord,
    ) -> Self {
        self.columns = self
            .columns
            .with_contrast_links(&contrasts)
            .with_standardization_links(&standardization)
            .with_smooth_links(&self.smooths);
        self.contrasts = contrasts;
        self.standardization = standardization;
        self
    }

    /// Attach smooth provenance without rebuilding the matrix.
    ///
    /// Refreshes per-column `smooth_idx` links; contrast / standardization links are preserved.
    #[must_use]
    pub fn with_smooth_provenance(mut self, smooths: Vec<RecordedSmooth>) -> Self {
        self.columns = self
            .columns
            .with_smooth_links(&smooths)
            .with_contrast_links(&self.contrasts)
            .with_standardization_links(&self.standardization);
        self.smooths = smooths;
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
        self.columns.treatment_column()
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
    policy: &KernelPolicy,
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
        let (mean, scale) = standardize_inplace(policy, slice, eps);
        entries.push(StandardizedColumn { col_idx: col, mean, scale });
    }
    Ok(StandardizationRecord { entries })
}

#[cfg(test)]
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
mod tests {
    use antecedent_core::VariableId;

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
        let rec =
            standardize_columns(&mut matrix, 4, 2, &[1], 1e-12, &KernelPolicy::default_policy())
                .unwrap();
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
        assert_eq!(design.columns.get(2).and_then(|c| c.contrast_idx), None); // range (2,3) but ncols=2
        assert_eq!(design.columns.get(1).and_then(|c| c.standardization_idx), Some(0));
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
