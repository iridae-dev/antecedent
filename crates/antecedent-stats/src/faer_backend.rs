//! `faer` dense backend — column-pivoted QR least squares.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use faer::diag::Diag;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::qr::col_pivoting::factor::{
    qr_in_place, qr_in_place_scratch, recommended_block_size,
};
use faer::linalg::qr::col_pivoting::solve::{
    solve_lstsq_in_place_scratch, solve_lstsq_in_place_with_conj,
};
use faer::linalg::svd::{ComputeSvdVectors, SvdError, svd, svd_scratch};
use faer::{Conj, Mat, MatRef, Spec};

use crate::error::StatsError;
use antecedent_core::execution::AdmissionLease;

use crate::faer_scope::{FaerScope, FaerScopeError, detached_serial_lease};
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresFit, LeastSquaresWorkspace};

mod resampling;
pub use resampling::{ResampledLeastSquares, ResampledLeastSquaresWorkspace};

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
                let (coefficients, rank, rcond) = solve_full_rank(x_colmajor, nrows, ncols, y)?;
                workspace.prepare(nrows, ncols);
                let rss = write_residuals(
                    x_colmajor,
                    nrows,
                    ncols,
                    y,
                    &coefficients,
                    &mut workspace.residuals,
                );

                crate::fit_counts::completed_solve();
                Ok(LeastSquaresFit {
                    coefficients,
                    residuals: workspace.residuals[..nrows].to_vec(),
                    rank,
                    rss,
                    diagnostics: FitDiagnostics::new(rank, rcond, "faer", workspace.grow_count),
                })
            },
        )
    }
}

impl FaerBackend {
    /// Solve full-rank least squares when only the coefficients are consumed.
    ///
    /// Uses the same equilibrated, column-pivoted QR and numerical-rank check as
    /// the full fit, without allocating or evaluating unused residual vectors.
    ///
    /// # Errors
    ///
    /// Shape mismatch or a rank-deficient design.
    pub fn least_squares_coefficients(
        &self,
        x_colmajor: &[f64],
        nrows: usize,
        ncols: usize,
        y: &[f64],
    ) -> Result<Vec<f64>, StatsError> {
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::LeastSquaresSolve,
            || {
                let (coefficients, _, _) = solve_full_rank(x_colmajor, nrows, ncols, y)?;
                crate::fit_counts::completed_solve();
                Ok(coefficients)
            },
        )
    }

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
        let Some(cells) = nrows.checked_mul(ncols) else {
            return Err(StatsError::Shape { message: "X buffer too short" });
        };
        if x_colmajor.len() < cells {
            return Err(StatsError::Shape { message: "X buffer too short" });
        }
        let mut scratch = FactorScratch::new();
        min_norm_on_column_major(x_colmajor, nrows, ncols, y, serial_scope(), &mut scratch, 0)
    }
}

/// Shared factorization, rank decision and solve for both output contracts.
fn solve_full_rank(
    x: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
) -> Result<(Vec<f64>, usize, Option<f64>), StatsError> {
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    if x.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    if nrows < ncols {
        return Err(StatsError::Shape { message: "nrows < ncols" });
    }
    let mut scales = Vec::new();
    write_column_scales(x, nrows, ncols, &mut scales);
    let scope = serial_scope();
    let mut scratch = FactorScratch::new();
    let factored = factor_equilibrated(x, nrows, ncols, &scales, scope, &mut scratch)?;
    let (rank, rcond) = factored.rank_and_rcond(nrows);
    if rank < ncols {
        return Err(StatsError::RankDeficient { rank, ncols });
    }
    let coefficients = solve_equilibrated(&factored, y, &scales, scope, &mut scratch)?;
    Ok((coefficients, rank, rcond))
}

/// Column-pivoted QR kept with the design that produced it.
pub(crate) struct ScopedQr {
    q_basis: Mat<f64>,
    q_coeff: Mat<f64>,
    r_full: Mat<f64>,
    thin_r: Mat<f64>,
    perm_forward: Vec<usize>,
    perm_inverse: Vec<usize>,
}

impl ScopedQr {
    /// Numerical rank and `min|R_ii| / max|R_ii|` from the leading triangle.
    ///
    /// The tolerance is `sqrt(nrows) · ε · max|R_ii|`, the same rule as
    /// [`DenseLinearAlgebra::least_squares`].
    #[must_use]
    pub(crate) fn rank_and_rcond(&self, nrows: usize) -> (usize, Option<f64>) {
        let r_factor = self.thin_r.as_ref();
        let size = r_factor.nrows().min(r_factor.ncols());
        let mut max_diag = 0.0_f64;
        let mut min_diag = f64::INFINITY;
        for index in 0..size {
            let diagonal = r_factor[(index, index)].abs();
            max_diag = max_diag.max(diagonal);
            if diagonal > 0.0 {
                min_diag = min_diag.min(diagonal);
            }
        }
        let tolerance = (nrows as f64).sqrt() * f64::EPSILON * max_diag;
        let mut rank = 0usize;
        for index in 0..size {
            if r_factor[(index, index)].abs() > tolerance {
                rank += 1;
            }
        }
        let rcond =
            if max_diag > 0.0 && min_diag.is_finite() { Some(min_diag / max_diag) } else { None };
        (rank, rcond)
    }
}

/// One scratch buffer reused across QR, the triangular solve, and a fallback SVD.
pub(crate) struct FactorScratch {
    buffer: Option<MemBuffer>,
    root: Option<AdmissionLease>,
    child: Option<AdmissionLease>,
}

impl FactorScratch {
    /// Empty scratch. Nothing is allocated. The first [`Self::ensure`] draws a
    /// detached serial lease because [`FaerBackend::least_squares`] has no study
    /// [`ExecutionContext`](antecedent_core::ExecutionContext) to lease from.
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self { buffer: None, root: None, child: None }
    }

    /// Scratch paid by `root` instead of a detached pool.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn under(root: AdmissionLease) -> Self {
        Self { buffer: None, root: Some(root), child: None }
    }

    /// Bytes of the retained buffer. Zero when nothing has been allocated.
    #[must_use]
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> u64 {
        match self.buffer.as_ref() {
            Some(buffer) => match u64::try_from(buffer.len()) {
                Ok(bytes) => bytes,
                Err(_) => u64::MAX,
            },
            None => 0,
        }
    }

    /// Whether this scratch holds one buffer.
    #[must_use]
    #[cfg(test)]
    pub(crate) const fn is_allocated(&self) -> bool {
        self.buffer.is_some()
    }

    fn ensure(&mut self, scope: FaerScope, required: StackReq) -> Result<(), StatsError> {
        let bytes = match u64::try_from(required.size_bytes()) {
            Ok(bytes) if required.align_bytes() != 0 => bytes,
            _ => {
                return Err(backend_error(FaerScopeError::ScratchOverflow));
            }
        };
        if buffer_fits(self.buffer.as_ref(), required, bytes) {
            return Ok(());
        }
        self.child = None;
        if self.root.is_none() {
            self.root = Some(detached_serial_lease());
        }
        let child = {
            let root = self.root.as_ref().expect("root lease was just installed");
            scope.admit_scratch(root, required).map_err(backend_error)?
        };
        self.buffer = Some(MemBuffer::new(required));
        self.child = Some(child);
        Ok(())
    }

    fn buffer_mut(&mut self) -> Result<&mut MemBuffer, StatsError> {
        self.buffer.as_mut().ok_or(StatsError::Backend("faer scratch was not allocated".into()))
    }
}

/// One worker. Scratch for this scope is drawn by [`FactorScratch`] from a
/// detached serial lease: this function has no study [`ExecutionContext`].
#[must_use]
pub(crate) fn serial_scope() -> FaerScope {
    FaerScope::try_new(1).unwrap_or_else(|_| unreachable!("one worker is a scope"))
}

/// Column scales: the max absolute entry, with a zero column stored as `1`.
pub(crate) fn write_column_scales(
    design: &[f64],
    nrows: usize,
    ncols: usize,
    scales: &mut Vec<f64>,
) {
    scales.resize(ncols, 0.0);
    for column in 0..ncols {
        let mut scale = 0.0_f64;
        for value in &design[column * nrows..(column + 1) * nrows] {
            scale = scale.max(value.abs());
        }
        scales[column] = if scale == 0.0 { 1.0 } else { scale };
    }
}

/// Equilibrated column-pivoted QR into `scratch`.
///
/// The input design is not modified. Column `c` of the factor is the design
/// column divided by `scales[c]`.
pub(crate) fn factor_equilibrated(
    design: &[f64],
    nrows: usize,
    ncols: usize,
    scales: &[f64],
    scope: FaerScope,
    scratch: &mut FactorScratch,
) -> Result<ScopedQr, StatsError> {
    let matrix = Mat::<f64>::from_fn(nrows, ncols, |row, column| {
        design[column * nrows + row] / scales[column]
    });
    column_pivoted_qr(matrix, scope, scratch)
}

/// Solve the equilibrated system and divide the coefficients by `scales`.
pub(crate) fn solve_equilibrated(
    factored: &ScopedQr,
    y: &[f64],
    scales: &[f64],
    scope: FaerScope,
    scratch: &mut FactorScratch,
) -> Result<Vec<f64>, StatsError> {
    let nrows = y.len();
    let ncols = scales.len();
    let mut rhs = Mat::<f64>::from_fn(nrows, 1, |row, _| y[row]);
    solve_scoped_qr(factored, rhs.as_mut(), scope, scratch)?;
    let mut coefficients = vec![0.0; ncols];
    for column in 0..ncols {
        coefficients[column] = rhs[(column, 0)] / scales[column];
    }
    Ok(coefficients)
}

/// Write `y - Xβ` into `residuals` and return the residual sum of squares.
pub(crate) fn write_residuals(
    design: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    coefficients: &[f64],
    residuals: &mut [f64],
) -> f64 {
    let residuals = &mut residuals[..nrows];
    for row in 0..nrows {
        let mut predicted = 0.0;
        for column in 0..ncols {
            predicted += design[column * nrows + row] * coefficients[column];
        }
        residuals[row] = y[row] - predicted;
    }
    residuals.iter().map(|residual| residual * residual).sum()
}

/// Minimum-norm coefficients of a column-major design, viewed in place.
///
pub(crate) fn min_norm_on_column_major(
    design: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    scope: FaerScope,
    scratch: &mut FactorScratch,
    grow_count: u32,
) -> Result<LeastSquaresFit, StatsError> {
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    let Some(cells) = nrows.checked_mul(ncols) else {
        return Err(StatsError::Shape { message: "X buffer too short" });
    };
    if design.len() < cells {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    let view = MatRef::from_column_major_slice(&design[..cells], nrows, ncols);
    let svd = thin_svd_in_scope(view, scope, scratch).map_err(|_| {
        StatsError::Backend("minimum-norm least squares: SVD did not converge".into())
    })?;
    let (u, s, v) = (svd.u.as_ref(), svd.s.as_ref().column_vector(), svd.v.as_ref());
    let size = nrows.min(ncols);
    let s_max = (0..size).map(|index| s[index].abs()).fold(0.0_f64, f64::max);
    let tolerance = (nrows.max(ncols) as f64) * f64::EPSILON * s_max;
    let mut coefficients = vec![0.0; ncols];
    let mut rank = 0usize;
    let mut s_min = f64::INFINITY;
    for index in 0..size {
        let singular = s[index].abs();
        if singular <= tolerance {
            continue;
        }
        rank += 1;
        s_min = s_min.min(singular);
        let projection: f64 = (0..nrows).map(|row| u[(row, index)] * y[row]).sum();
        let weight = projection / s[index];
        for (column, coefficient) in coefficients.iter_mut().enumerate() {
            *coefficient += v[(column, index)] * weight;
        }
    }
    let mut residuals = vec![0.0; nrows];
    let rss = write_residuals(design, nrows, ncols, y, &coefficients, &mut residuals);
    let rcond = (rank > 0).then(|| s_min / s_max);
    Ok(LeastSquaresFit {
        coefficients,
        residuals,
        rank,
        rss,
        diagnostics: FitDiagnostics::new(rank, rcond, "faer-svd", grow_count),
    })
}

fn column_pivoted_qr(
    mut qr: Mat<f64>,
    scope: FaerScope,
    scratch: &mut FactorScratch,
) -> Result<ScopedQr, StatsError> {
    let (nrows, ncols) = qr.shape();
    let par = scope.parallelism();
    let block_size = recommended_block_size::<f64>(nrows, ncols);
    let required = StackReq::and(
        qr_in_place_scratch::<usize, f64>(nrows, ncols, block_size, par, Spec::default()),
        solve_lstsq_in_place_scratch::<usize, f64>(nrows, ncols, block_size, 1, par),
    );
    scratch.ensure(scope, required)?;
    let buffer = scratch.buffer_mut()?;
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
        MemStack::new(buffer),
        Spec::default(),
    );
    let (q_basis, r_full) = split_triangular(qr);
    let thin_r = r_full.get(..size, ..size).to_owned();
    Ok(ScopedQr { q_basis, q_coeff, r_full, thin_r, perm_forward, perm_inverse })
}

fn solve_scoped_qr(
    factored: &ScopedQr,
    rhs: faer::MatMut<'_, f64>,
    scope: FaerScope,
    scratch: &mut FactorScratch,
) -> Result<(), StatsError> {
    let par = scope.parallelism();
    let (nrows, ncols) = factored.q_basis.shape();
    let block_size = factored.q_coeff.nrows();
    let required =
        solve_lstsq_in_place_scratch::<usize, f64>(nrows, ncols, block_size, rhs.ncols(), par);
    scratch.ensure(scope, required)?;
    let buffer = scratch.buffer_mut()?;
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
        MemStack::new(buffer),
    );
    Ok(())
}

fn buffer_fits(buffer: Option<&MemBuffer>, required: StackReq, bytes: u64) -> bool {
    let Some(buffer) = buffer else {
        return false;
    };
    let size_ok = match u64::try_from(buffer.len()) {
        Ok(len) => len >= bytes,
        Err(_) => true,
    };
    let align = required.align_bytes();
    let align_ok = align <= 1 || buffer.as_ptr().cast::<u8>().align_offset(align) == 0;
    size_ok && align_ok
}

fn backend_error(error: FaerScopeError) -> StatsError {
    StatsError::Backend(error.to_string())
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

/// Symmetric eigendecomposition under an explicit scope.
///
/// This calls `self_adjoint_evd` with the scope's degree. It does not call
/// `self_adjoint_eigenvalues` or `self_adjoint_eigen`.
pub(crate) fn symmetric_eigen_in_scope(
    matrix: MatRef<'_, f64>,
    scope: FaerScope,
) -> Result<(Diag<f64>, Mat<f64>), StatsError> {
    let n = matrix.nrows();
    let par = scope.parallelism();
    let required = faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
        n,
        faer::linalg::evd::ComputeEigenvectors::Yes,
        par,
        Spec::default(),
    );
    let mut scratch = FactorScratch::new();
    scratch.ensure(scope, required)?;
    let buffer = scratch.buffer_mut()?;
    let mut values = Diag::<f64>::zeros(n);
    let mut vectors = Mat::<f64>::zeros(n, n);
    faer::linalg::evd::self_adjoint_evd(
        matrix,
        values.as_mut(),
        Some(vectors.as_mut()),
        par,
        MemStack::new(buffer),
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
    scratch: &mut FactorScratch,
) -> Result<ThinSvd, SvdError> {
    let (nrows, ncols) = a.shape();
    let size = nrows.min(ncols);
    let par = scope.parallelism();
    let compute = ComputeSvdVectors::Thin;
    let required = svd_scratch::<f64>(nrows, ncols, compute, compute, par, Spec::default());
    scratch.ensure(scope, required).map_err(|_| SvdError::NoConvergence)?;
    let buffer = scratch.buffer_mut().map_err(|_| SvdError::NoConvergence)?;
    let mut u = Mat::<f64>::zeros(nrows, size);
    let mut v = Mat::<f64>::zeros(ncols, size);
    let mut s = Diag::<f64>::zeros(size);
    svd(
        a,
        s.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        par,
        MemStack::new(buffer),
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
    use super::*;

    #[test]
    fn coefficients_only_matches_full_qr_including_rank_refusal() {
        for n in [8, 31, 128] {
            for scale in [1e-9, 1.0, 1e9] {
                let mut x = vec![1.0; n * 3];
                let y: Vec<_> = (0..n)
                    .map(|i| {
                        x[n + i] = scale * i as f64;
                        x[2 * n + i] = ((i * 17) % 23) as f64;
                        2.0 + 0.7 * i as f64 + 0.3 * x[2 * n + i]
                    })
                    .collect();
                let full = FaerBackend
                    .least_squares(&x, n, 3, &y, &mut LeastSquaresWorkspace::default())
                    .unwrap();
                let coefs = FaerBackend.least_squares_coefficients(&x, n, 3, &y).unwrap();
                assert_eq!(coefs, full.coefficients);
            }
        }
        let x = [1.0; 8];
        let y = [1.0; 4];
        assert!(matches!(
            FaerBackend.least_squares_coefficients(&x, 4, 2, &y),
            Err(StatsError::RankDeficient { .. })
        ));
        assert!(FaerBackend.least_squares_coefficients(&x, 4, 2, &y[..3]).is_err());
        assert!(FaerBackend.least_squares_coefficients(&x[..7], 4, 2, &y).is_err());
    }

    #[test]
    fn scratch_above_the_budget_is_refused_before_allocation() {
        let scope = FaerScope::try_new(2).unwrap();
        assert_eq!(scope.parallelism().degree(), 2);
        let root = {
            let pool = antecedent_core::execution::AdmissionPool::fixed(1, 8);
            antecedent_core::ExecutionContext::for_tests(1)
                .with_admission(pool)
                .acquire_admission(1, 8)
                .expect("pool funds the scratch lease")
        };
        let mut scratch = FactorScratch::under(root);
        let error = scratch.ensure(scope, StackReq::new::<u8>(9)).unwrap_err();
        assert!(matches!(error, StatsError::Backend(_)));
        assert!(!scratch.is_allocated());
        assert_eq!(scratch.bytes(), 0);
    }

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
