//! `faer` dense backend — column-pivoted QR least squares.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::Cell;

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

use crate::faer_scope::{FaerScope, FaerScopeError};
use crate::layout_rules::{LayoutOp, LayoutRefusal, LayoutRequest, StorageLayout, select_layout};
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresFit, LeastSquaresWorkspace};

mod resampling;
pub use resampling::{ResampledLeastSquares, ResampledLeastSquaresWorkspace};

/// Design copies and scratch the dense solver paths performed on this thread.
///
/// Counted where the work happens: a gather is the one pass that writes the equilibrated
/// `nrows × ncols` factor from the caller's column-major design, a scratch allocation is a
/// `faer` buffer that was actually created (a buffer that already fits is reused and not
/// counted), and an in-place view is a factorization that read the caller's design without
/// copying it. Read it through [`observe_solver_traffic`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SolverTraffic {
    /// Passes that copied a whole design.
    pub design_gathers: u64,
    /// Design cells those passes wrote.
    pub gathered_cells: u64,
    /// `faer` scratch buffers created.
    pub scratch_allocations: u64,
    /// Bytes of every scratch buffer created.
    pub scratch_bytes_allocated: u64,
    /// Largest single scratch buffer created while observing.
    pub largest_scratch_bytes: u64,
    /// Factorizations that read the caller's design in place.
    pub in_place_views: u64,
}

thread_local! {
    static TRAFFIC: Cell<SolverTraffic> = const {
        Cell::new(SolverTraffic {
            design_gathers: 0,
            gathered_cells: 0,
            scratch_allocations: 0,
            scratch_bytes_allocated: 0,
            largest_scratch_bytes: 0,
            in_place_views: 0,
        })
    };
}

fn record_traffic(update: impl FnOnce(&mut SolverTraffic)) {
    TRAFFIC.with(|cell| {
        let mut traffic = cell.get();
        update(&mut traffic);
        cell.set(traffic);
    });
}

/// Run `work` and report the design gathers, scratch allocations and in-place views the solver
/// paths performed on this thread while it ran.
///
/// This counts what the backend executed, not what a caller declares. Nested observations
/// each see the work inside them.
pub fn observe_solver_traffic<R>(work: impl FnOnce() -> R) -> (R, SolverTraffic) {
    let before = TRAFFIC.with(Cell::get);
    TRAFFIC.with(|cell| cell.set(SolverTraffic { largest_scratch_bytes: 0, ..before }));
    let result = work();
    let after = TRAFFIC.with(Cell::get);
    TRAFFIC.with(|cell| {
        cell.set(SolverTraffic {
            largest_scratch_bytes: before.largest_scratch_bytes.max(after.largest_scratch_bytes),
            ..after
        });
    });
    let observed = SolverTraffic {
        design_gathers: after.design_gathers.saturating_sub(before.design_gathers),
        gathered_cells: after.gathered_cells.saturating_sub(before.gathered_cells),
        scratch_allocations: after.scratch_allocations.saturating_sub(before.scratch_allocations),
        scratch_bytes_allocated: after
            .scratch_bytes_allocated
            .saturating_sub(before.scratch_bytes_allocated),
        largest_scratch_bytes: after.largest_scratch_bytes,
        in_place_views: after.in_place_views.saturating_sub(before.in_place_views),
    };
    (result, observed)
}

/// Size the residual and right-hand-side buffers a fit writes.
///
/// The `faer` paths never read `workspace.scratch`, so it is left alone rather than grown to
/// a second design-sized buffer. A growth counts once, as [`LeastSquaresWorkspace::prepare`]
/// counts it.
fn prepare_workspace(workspace: &mut LeastSquaresWorkspace, nrows: usize, ncols: usize) {
    let mut grew = false;
    if workspace.rhs.len() < ncols {
        workspace.rhs.resize(ncols, 0.0);
        grew = true;
    }
    if workspace.residuals.len() < nrows {
        workspace.residuals.resize(nrows, 0.0);
        grew = true;
    }
    if grew {
        workspace.grow_count = workspace.grow_count.saturating_add(1);
    }
}

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
                prepare_workspace(workspace, nrows, ncols);
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

    fn row_scaled_least_squares(
        &self,
        x_colmajor: &[f64],
        nrows: usize,
        ncols: usize,
        row_scale: &[f64],
        y: &[f64],
        workspace: &mut LeastSquaresWorkspace,
    ) -> Result<LeastSquaresFit, StatsError> {
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::LeastSquaresSolve,
            || {
                checked_design_cells(x_colmajor, nrows, ncols, y)?;
                if row_scale.len() != nrows {
                    return Err(StatsError::Shape { message: "row scale length != nrows" });
                }
                if row_scale.iter().any(|scale| !scale.is_finite()) {
                    return Err(StatsError::Unsupported {
                        message: "least squares requires finite row scales",
                    });
                }
                if nrows < ncols {
                    return Err(StatsError::RankDeficient { rank: nrows, ncols });
                }
                // Column scales of the row-scaled design, without building it.
                let mut scales = vec![0.0; ncols];
                for (column, scale) in scales.iter_mut().enumerate() {
                    let mut largest = 0.0_f64;
                    for (value, row_scale) in
                        x_colmajor[column * nrows..(column + 1) * nrows].iter().zip(row_scale)
                    {
                        let scaled = value * row_scale;
                        if !scaled.is_finite() {
                            return Err(StatsError::Unsupported {
                                message: "least squares requires a finite scaled design",
                            });
                        }
                        largest = largest.max(scaled.abs());
                    }
                    *scale = if largest == 0.0 { 1.0 } else { largest };
                }
                let scope = serial_scope();
                let mut scratch = FactorScratch::new();
                let factored = factor_equilibrated_rows(
                    x_colmajor,
                    nrows,
                    ncols,
                    Some(row_scale),
                    &scales,
                    scope,
                    &mut scratch,
                )?;
                let (rank, rcond) = factored.rank_and_rcond(nrows);
                if rank < ncols {
                    return Err(StatsError::RankDeficient { rank, ncols });
                }
                let coefficients = solve_equilibrated(&factored, y, &scales, scope, &mut scratch)?;
                drop(factored);
                prepare_workspace(workspace, nrows, ncols);
                let residuals = &mut workspace.residuals[..nrows];
                for row in 0..nrows {
                    let mut predicted = 0.0;
                    for column in 0..ncols {
                        predicted += (x_colmajor[column * nrows + row] * row_scale[row])
                            * coefficients[column];
                    }
                    residuals[row] = y[row] - predicted;
                }
                let rss = residuals.iter().map(|residual| residual * residual).sum();
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
    /// Shape mismatch, or a rank below the column count. A design with fewer rows than
    /// columns never identifies its coefficients; it is rank deficient, and
    /// [`Self::min_norm_least_squares`] is the explicit way to pick one solution.
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

    /// Least squares by column-pivoted QR, falling back to the minimum-norm solution when the
    /// design is rank deficient or has fewer rows than columns.
    ///
    /// One scratch buffer serves the QR, the triangular solve and the fallback SVD, and the
    /// fallback reads the caller's design in place: the only design copy is the equilibrated
    /// factor the QR needs, and it is dropped before the SVD runs. A design with fewer rows
    /// than columns is never gathered at all. Calling [`DenseLinearAlgebra::least_squares`]
    /// and, on its rank refusal, [`Self::min_norm_least_squares`] pays a second scratch buffer.
    /// The fit's diagnostics name the path that produced it (`"faer"` or `"faer-svd"`).
    ///
    /// # Errors
    ///
    /// Shape mismatch, or an SVD that fails to converge.
    pub fn least_squares_or_min_norm(
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
                checked_design_cells(x_colmajor, nrows, ncols, y)?;
                let scope = serial_scope();
                let mut scratch = FactorScratch::new();
                if nrows >= ncols {
                    let mut scales = Vec::new();
                    write_column_scales(x_colmajor, nrows, ncols, &mut scales);
                    let factored = factor_equilibrated(
                        x_colmajor,
                        nrows,
                        ncols,
                        &scales,
                        scope,
                        &mut scratch,
                    )?;
                    let (rank, rcond) = factored.rank_and_rcond(nrows);
                    if rank == ncols {
                        let coefficients =
                            solve_equilibrated(&factored, y, &scales, scope, &mut scratch)?;
                        drop(factored);
                        prepare_workspace(workspace, nrows, ncols);
                        let rss = write_residuals(
                            x_colmajor,
                            nrows,
                            ncols,
                            y,
                            &coefficients,
                            &mut workspace.residuals,
                        );
                        crate::fit_counts::completed_solve();
                        return Ok(LeastSquaresFit {
                            coefficients,
                            residuals: workspace.residuals[..nrows].to_vec(),
                            rank,
                            rss,
                            diagnostics: FitDiagnostics::new(
                                rank,
                                rcond,
                                "faer",
                                workspace.grow_count,
                            ),
                        });
                    }
                }
                let fit = min_norm_on_column_major(
                    x_colmajor,
                    nrows,
                    ncols,
                    y,
                    scope,
                    &mut scratch,
                    workspace.grow_count,
                )?;
                crate::fit_counts::completed_solve();
                Ok(fit)
            },
        )
    }

    /// Full-rank least-squares coefficients with the worker count and scratch both taken from
    /// the caller's budget.
    ///
    /// The worker count is the measured choice for this shape ([`select_layout`]) capped at
    /// `available_workers`. Scratch is sized from `faer`'s own requirement for that choice and
    /// drawn from `scratch`, the caller's lease, before any buffer is allocated: a requirement
    /// the lease cannot hold is refused with nothing gathered and nothing allocated.
    ///
    /// # Errors
    ///
    /// Shape mismatch, a rank below the column count, no workers, or
    /// [`StatsError::ResourceLimit`] when the scratch does not fit the lease.
    pub fn least_squares_coefficients_leased(
        &self,
        x_colmajor: &[f64],
        nrows: usize,
        ncols: usize,
        y: &[f64],
        available_workers: u32,
        scratch: &AdmissionLease,
    ) -> Result<Vec<f64>, StatsError> {
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::LeastSquaresSolve,
            || {
                checked_design_cells(x_colmajor, nrows, ncols, y)?;
                if nrows < ncols {
                    return Err(StatsError::RankDeficient { rank: nrows, ncols });
                }
                let budget = scratch.remainder().bytes();
                let choice = select_layout(&LayoutRequest {
                    op: LayoutOp::ColumnPivotedQr,
                    storage: StorageLayout::Dense,
                    rows: nrows,
                    cols: ncols,
                    available_workers,
                    scratch_budget_bytes: budget,
                })
                .map_err(|refusal| match refusal {
                    LayoutRefusal::ScratchOverBudget { required_bytes, budget_bytes } => {
                        StatsError::ResourceLimit {
                            message: "column-pivoted QR scratch",
                            required_bytes,
                            limit_bytes: budget_bytes,
                        }
                    }
                    LayoutRefusal::NoWorkers => {
                        StatsError::Shape { message: "least squares needs at least one worker" }
                    }
                    _ => StatsError::Shape { message: "least squares needs a non-empty design" },
                })?;
                let scope = FaerScope::try_new(choice.workers).map_err(backend_error)?;
                let held = scratch
                    .acquire_child(0, budget)
                    .map_err(|error| StatsError::Backend(error.to_string()))?;
                let mut factor_scratch = FactorScratch::under(held);
                let (coefficients, _, _) =
                    solve_full_rank_in(x_colmajor, nrows, ncols, y, scope, &mut factor_scratch)?;
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
///
/// A short, wide design cannot identify its coefficients, so it is refused as rank
/// deficient with the rank its rows allow; [`StatsError::Shape`] is reserved for a buffer
/// that does not match `nrows` and `ncols`. Callers that want the minimum-norm selection
/// ask for it through [`FaerBackend::min_norm_least_squares`].
fn solve_full_rank(
    x: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
) -> Result<(Vec<f64>, usize, Option<f64>), StatsError> {
    let mut scratch = FactorScratch::new();
    solve_full_rank_in(x, nrows, ncols, y, serial_scope(), &mut scratch)
}

/// [`solve_full_rank`] under the caller's scope and scratch.
fn solve_full_rank_in(
    x: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    scope: FaerScope,
    scratch: &mut FactorScratch,
) -> Result<(Vec<f64>, usize, Option<f64>), StatsError> {
    checked_design_cells(x, nrows, ncols, y)?;
    if nrows < ncols {
        return Err(StatsError::RankDeficient { rank: nrows, ncols });
    }
    let mut scales = Vec::new();
    write_column_scales(x, nrows, ncols, &mut scales);
    let factored = factor_equilibrated(x, nrows, ncols, &scales, scope, scratch)?;
    let (rank, rcond) = factored.rank_and_rcond(nrows);
    if rank < ncols {
        return Err(StatsError::RankDeficient { rank, ncols });
    }
    let coefficients = solve_equilibrated(&factored, y, &scales, scope, scratch)?;
    Ok((coefficients, rank, rcond))
}

/// Cells of a `nrows × ncols` design after the shape checks every solve makes.
fn checked_design_cells(
    x: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
) -> Result<usize, StatsError> {
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    let Some(cells) = nrows.checked_mul(ncols) else {
        return Err(StatsError::Shape { message: "X buffer too short" });
    };
    if x.len() < cells {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    if x[..cells].iter().chain(y).any(|value| !value.is_finite()) {
        return Err(StatsError::Unsupported {
            message: "least squares requires finite design and outcome values",
        });
    }
    Ok(cells)
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
        self.buffer = None;
        self.child = None;
        let child = if let Some(root) = self.root.as_ref() {
            scope.admit_scratch(root, required).map_err(backend_error)?
        } else if let Some(admitted) = antecedent_core::numerical_admission::acquire(bytes) {
            admitted
                .map_err(|error| StatsError::Backend(format!("faer scratch admission: {error}")))?
        } else {
            let context = antecedent_core::ExecutionContext::for_tests(0)
                .with_admission(antecedent_core::execution::AdmissionPool::fixed(0, bytes));
            context
                .acquire_admission(0, bytes)
                .map_err(|error| StatsError::Backend(error.to_string()))?
        };
        self.buffer = Some(MemBuffer::new(required));
        self.child = Some(child);
        record_traffic(|traffic| {
            traffic.scratch_allocations = traffic.scratch_allocations.saturating_add(1);
            traffic.scratch_bytes_allocated = traffic.scratch_bytes_allocated.saturating_add(bytes);
            traffic.largest_scratch_bytes = traffic.largest_scratch_bytes.max(bytes);
        });
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
    factor_equilibrated_rows(design, nrows, ncols, None, scales, scope, scratch)
}

/// [`factor_equilibrated`] with an optional per-row scale folded into the one gather.
///
/// Entry `(row, column)` of the factor is `design · row_scale[row] / scales[column]`, with the
/// product rounded before the division exactly as a materialized scaled design would be, so
/// the scaled design never exists as a second buffer.
pub(crate) fn factor_equilibrated_rows(
    design: &[f64],
    nrows: usize,
    ncols: usize,
    row_scale: Option<&[f64]>,
    scales: &[f64],
    scope: FaerScope,
    scratch: &mut FactorScratch,
) -> Result<ScopedQr, StatsError> {
    let matrix = match row_scale {
        None => Mat::<f64>::from_fn(nrows, ncols, |row, column| {
            design[column * nrows + row] / scales[column]
        }),
        Some(rows) => Mat::<f64>::from_fn(nrows, ncols, |row, column| {
            (design[column * nrows + row] * rows[row]) / scales[column]
        }),
    };
    record_traffic(|traffic| {
        traffic.design_gathers = traffic.design_gathers.saturating_add(1);
        traffic.gathered_cells =
            traffic.gathered_cells.saturating_add(u64::try_from(nrows * ncols).unwrap_or(u64::MAX));
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
    let cells = checked_design_cells(design, nrows, ncols, y)?;
    let view = MatRef::from_column_major_slice(&design[..cells], nrows, ncols);
    record_traffic(|traffic| traffic.in_place_views = traffic.in_place_views.saturating_add(1));
    // A tall or square design is reduced by an unpivoted QR first, exactly as `faer`'s own
    // tall SVD does, so the SVD sees the `ncols x ncols` triangular factor and the right-hand
    // side is reduced to `Q^T y`. Memory: the QR factor lives in the one scratch buffer (a
    // design-sized phase), and that phase ends, leaving only the small triangular factor and
    // `Q^T y`, before the SVD of the triangular factor takes the same buffer. The peak is
    // therefore the caller's vectors plus one design-sized scratch, never the design-sized left
    // factor `U` beside the SVD's own design-sized workspace. The singular values of the
    // triangular factor are those of the design, so the rank test and the solution are the same.
    let reduced;
    let (svd_input, rhs): (MatRef<'_, f64>, &[f64]);
    let triangular;
    if nrows >= ncols {
        reduced = qr_reduce(view, y, scope, scratch)?;
        triangular = reduced.0;
        svd_input = triangular.as_ref();
        rhs = &reduced.1;
    } else {
        svd_input = view;
        rhs = y;
    }
    let svd = thin_svd_in_scope(svd_input, scope, scratch).map_err(|_| {
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
        let projection: f64 = (0..u.nrows()).map(|row| u[(row, index)] * rhs[row]).sum();
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

/// Unpivoted QR of a tall or square column-major design held in `scratch`: the
/// `ncols x ncols` upper-triangular factor and the first `ncols` entries of `Q^T y`.
///
/// The design-sized factor, its Householder coefficients and the transformed right-hand side
/// are all carved from the one scratch buffer, so the phase's memory is that buffer and the two
/// small owned results.
fn qr_reduce(
    design: MatRef<'_, f64>,
    y: &[f64],
    scope: FaerScope,
    scratch: &mut FactorScratch,
) -> Result<(Mat<f64>, Vec<f64>), StatsError> {
    use faer::linalg::householder::{
        apply_block_householder_sequence_transpose_on_the_left_in_place_scratch as apply_scratch,
        apply_block_householder_sequence_transpose_on_the_left_in_place_with_conj as apply_transpose,
    };
    use faer::linalg::qr::no_pivoting::factor as plain;
    use faer::linalg::{temp_mat_scratch, temp_mat_zeroed};
    use faer::mat::AsMatMut;
    use faer::reborrow::{Reborrow, ReborrowMut};

    let (nrows, ncols) = design.shape();
    let par = scope.parallelism();
    let block_size = plain::recommended_block_size::<f64>(nrows, ncols);
    let required = temp_mat_scratch::<f64>(nrows, ncols)
        .and(temp_mat_scratch::<f64>(block_size, ncols))
        .and(temp_mat_scratch::<f64>(nrows, 1))
        .and(
            plain::qr_in_place_scratch::<f64>(nrows, ncols, block_size, par, Spec::default())
                .or(apply_scratch::<f64>(nrows, block_size, 1)),
        );
    scratch.ensure(scope, required)?;
    let stack = MemStack::new(scratch.buffer_mut()?);
    let (mut qr, stack) = temp_mat_zeroed::<f64, _, _>(nrows, ncols, stack);
    let mut qr = qr.as_mat_mut();
    let (mut coeff, stack) = temp_mat_zeroed::<f64, _, _>(block_size, ncols, stack);
    let mut coeff = coeff.as_mat_mut();
    let (mut rhs, stack) = temp_mat_zeroed::<f64, _, _>(nrows, 1, stack);
    let mut rhs = rhs.as_mat_mut();
    qr.rb_mut().copy_from(design);
    let _ = plain::qr_in_place(qr.rb_mut(), coeff.rb_mut(), par, stack, Spec::default());
    for (row, value) in y.iter().enumerate() {
        rhs[(row, 0)] = *value;
    }
    apply_transpose(qr.rb(), coeff.rb(), Conj::No, rhs.rb_mut(), par, stack);
    let triangular =
        Mat::<f64>::from_fn(
            ncols,
            ncols,
            |row, column| {
                if row <= column { qr[(row, column)] } else { 0.0 }
            },
        );
    let reduced = (0..ncols).map(|row| rhs[(row, 0)]).collect();
    Ok((triangular, reduced))
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
    fn least_squares_refuses_nonfinite_design_and_outcome_on_every_solve_path() {
        let design = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let outcome = [1.0, 3.0, 5.0, 7.0];
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for in_design in [false, true] {
                let mut x = design;
                let mut y = outcome;
                if in_design {
                    x[6] = value;
                } else {
                    y[2] = value;
                }
                let mut workspace = LeastSquaresWorkspace::default();
                assert!(matches!(
                    FaerBackend.least_squares(&x, 4, 2, &y, &mut workspace),
                    Err(StatsError::Unsupported { .. })
                ));
                assert!(matches!(
                    FaerBackend.least_squares_coefficients(&x, 4, 2, &y),
                    Err(StatsError::Unsupported { .. })
                ));
                assert!(matches!(
                    FaerBackend.min_norm_least_squares(&x, 4, 2, &y),
                    Err(StatsError::Unsupported { .. })
                ));
                assert!(matches!(
                    FaerBackend.least_squares_or_min_norm(&x, 4, 2, &y, &mut workspace),
                    Err(StatsError::Unsupported { .. })
                ));
                assert!(matches!(
                    FaerBackend.row_scaled_least_squares(&x, 4, 2, &[1.0; 4], &y, &mut workspace),
                    Err(StatsError::Unsupported { .. })
                ));
            }
        }
        // An unused tail is not part of the declared design.
        let mut with_tail = design.to_vec();
        with_tail.push(f64::NAN);
        assert!(FaerBackend.least_squares_coefficients(&with_tail, 4, 2, &outcome).is_ok());
        for scale in [[1.0, f64::NAN, 1.0, 1.0], [f64::INFINITY; 4]] {
            assert!(matches!(
                FaerBackend.row_scaled_least_squares(
                    &design,
                    4,
                    2,
                    &scale,
                    &outcome,
                    &mut LeastSquaresWorkspace::default(),
                ),
                Err(StatsError::Unsupported { .. })
            ));
        }
        assert!(matches!(
            FaerBackend.row_scaled_least_squares(
                &[1e308; 4],
                4,
                1,
                &[2.0; 4],
                &outcome,
                &mut LeastSquaresWorkspace::default(),
            ),
            Err(StatsError::Unsupported { .. })
        ));
    }

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

    fn smooth_design(n: usize, p: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0; n * p];
        let mut y = vec![0.0; n];
        for row in 0..n {
            x[row] = 1.0;
            y[row] = 0.5;
            for column in 1..p {
                let value = ((row as f64 + 1.0) * (column as f64 + 0.7) * 0.31).sin();
                x[column * n + row] = value;
                y[row] += value * column as f64 * 0.1;
            }
        }
        (x, y)
    }

    #[test]
    fn full_rank_fit_gathers_the_design_once_and_reuses_one_scratch_buffer() {
        let (n, p) = (200usize, 6usize);
        let (x, y) = smooth_design(n, p);
        let (fit, traffic) = observe_solver_traffic(|| {
            FaerBackend
                .least_squares_or_min_norm(&x, n, p, &y, &mut LeastSquaresWorkspace::default())
                .unwrap()
        });
        assert_eq!(fit.diagnostics.backend, "faer");
        assert_eq!((traffic.design_gathers, traffic.gathered_cells), (1, (n * p) as u64));
        // QR and the triangular solve share the buffer the QR sized for both.
        assert_eq!(traffic.scratch_allocations, 1);
        assert_eq!(traffic.in_place_views, 0);
        let plain =
            FaerBackend.least_squares(&x, n, p, &y, &mut LeastSquaresWorkspace::default()).unwrap();
        assert_eq!(fit.coefficients, plain.coefficients);
        // The fit does not size a design-sized workspace buffer it never reads.
        let mut workspace = LeastSquaresWorkspace::default();
        FaerBackend.least_squares(&x, n, p, &y, &mut workspace).unwrap();
        assert!(workspace.scratch.is_empty());
        assert!(workspace.residuals.len() >= n && workspace.rhs.len() >= p);
    }

    #[test]
    fn rank_deficient_fallback_reads_the_design_in_place_after_one_gather() {
        let (n, p) = (200usize, 6usize);
        let (mut x, y) = smooth_design(n, p);
        for row in 0..n {
            x[5 * n + row] = 2.0 * x[2 * n + row];
        }
        let ((fallback, shared), shared_traffic) = observe_solver_traffic(|| {
            let fit = FaerBackend
                .least_squares_or_min_norm(&x, n, p, &y, &mut LeastSquaresWorkspace::default())
                .unwrap();
            (fit.clone(), fit)
        });
        assert_eq!(shared.diagnostics.backend, "faer-svd");
        assert_eq!(shared.rank, p - 1);
        assert_eq!(shared_traffic.design_gathers, 1);
        assert_eq!(shared_traffic.in_place_views, 1);
        assert!(shared_traffic.scratch_allocations <= 2);
        let ((), separate) = observe_solver_traffic(|| {
            assert!(matches!(
                FaerBackend.least_squares(&x, n, p, &y, &mut LeastSquaresWorkspace::default()),
                Err(StatsError::RankDeficient { .. })
            ));
            FaerBackend.min_norm_least_squares(&x, n, p, &y).unwrap();
        });
        assert_eq!(separate.design_gathers, 1);
        assert!(shared_traffic.scratch_allocations <= separate.scratch_allocations);
        assert!(shared_traffic.scratch_bytes_allocated <= separate.scratch_bytes_allocated);
        let direct = FaerBackend.min_norm_least_squares(&x, n, p, &y).unwrap();
        assert_eq!(fallback.coefficients, direct.coefficients);
    }

    #[test]
    fn wide_design_goes_to_the_svd_without_any_design_gather() {
        let x = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = [1.0, 1.0];
        let (fit, traffic) = observe_solver_traffic(|| {
            FaerBackend
                .least_squares_or_min_norm(&x, 2, 3, &y, &mut LeastSquaresWorkspace::default())
                .unwrap()
        });
        assert_eq!(fit.diagnostics.backend, "faer-svd");
        assert_eq!((traffic.design_gathers, traffic.gathered_cells), (0, 0));
        assert_eq!(traffic.in_place_views, 1);
        assert_eq!(traffic.scratch_allocations, 1);
    }

    #[test]
    fn observation_nests_and_leaves_outer_totals_intact() {
        let (n, p) = (64usize, 3usize);
        let (x, y) = smooth_design(n, p);
        let ((), outer) = observe_solver_traffic(|| {
            FaerBackend.least_squares_coefficients(&x, n, p, &y).unwrap();
            let ((), inner) = observe_solver_traffic(|| {
                FaerBackend.least_squares_coefficients(&x, n, p, &y).unwrap();
            });
            assert_eq!(inner.design_gathers, 1);
            FaerBackend.least_squares_coefficients(&x, n, p, &y).unwrap();
        });
        assert_eq!(outer.design_gathers, 3);
        assert!(outer.largest_scratch_bytes > 0);
        assert!(outer.largest_scratch_bytes <= outer.scratch_bytes_allocated);
    }

    #[test]
    fn row_scaled_solve_equals_the_materialized_scaled_design_bit_for_bit() {
        let (n, p) = (150usize, 5usize);
        let (x, y) = smooth_design(n, p);
        let root: Vec<f64> = (0..n).map(|row| 0.5 + (row % 7) as f64 * 0.25).collect();
        let y_w: Vec<f64> = y.iter().zip(&root).map(|(v, s)| v * s).collect();
        let mut x_w = x.clone();
        for column in 0..p {
            for row in 0..n {
                x_w[column * n + row] = x[column * n + row] * root[row];
            }
        }
        let direct = FaerBackend
            .least_squares(&x_w, n, p, &y_w, &mut LeastSquaresWorkspace::default())
            .unwrap();
        let (folded, traffic) = observe_solver_traffic(|| {
            FaerBackend
                .row_scaled_least_squares(
                    &x,
                    n,
                    p,
                    &root,
                    &y_w,
                    &mut LeastSquaresWorkspace::default(),
                )
                .unwrap()
        });
        assert_eq!(traffic.design_gathers, 1);
        assert_eq!(folded.coefficients, direct.coefficients);
        assert_eq!(folded.residuals, direct.residuals);
        assert_eq!(folded.rss.to_bits(), direct.rss.to_bits());
        assert_eq!(folded.rank, direct.rank);
        let mut ws = LeastSquaresWorkspace::default();
        assert!(matches!(
            FaerBackend.row_scaled_least_squares(&x, n, p, &root[..n - 1], &y_w, &mut ws),
            Err(StatsError::Shape { .. })
        ));
        let mut duplicate = x.clone();
        for row in 0..n {
            duplicate[4 * n + row] = 3.0 * duplicate[n + row];
        }
        assert!(matches!(
            FaerBackend.row_scaled_least_squares(&duplicate, n, p, &root, &y_w, &mut ws),
            Err(StatsError::RankDeficient { .. })
        ));
    }

    fn pool_lease(bytes: u64) -> AdmissionLease {
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, bytes);
        antecedent_core::ExecutionContext::for_tests(1)
            .with_admission(pool)
            .acquire_admission(0, bytes)
            .expect("the pool funds the lease")
    }

    #[test]
    fn leased_solve_draws_scratch_from_the_callers_lease_and_refuses_before_gathering() {
        let (n, p) = (96usize, 5usize);
        let (x, y) = smooth_design(n, p);
        let expected = FaerBackend.least_squares_coefficients(&x, n, p, &y).unwrap();
        let lease = pool_lease(1 << 24);
        let (coefficients, traffic) = observe_solver_traffic(|| {
            FaerBackend.least_squares_coefficients_leased(&x, n, p, &y, 4, &lease).unwrap()
        });
        for (a, b) in coefficients.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
        assert_eq!(traffic.design_gathers, 1);
        assert!(traffic.largest_scratch_bytes > 0);
        assert!(traffic.largest_scratch_bytes <= 1 << 24);
        // The lease is whole again once the solve returns.
        assert_eq!(lease.remainder().bytes(), 1 << 24);

        let tight = pool_lease(8);
        let (refused, traffic) = observe_solver_traffic(|| {
            FaerBackend.least_squares_coefficients_leased(&x, n, p, &y, 4, &tight)
        });
        assert!(matches!(
            refused,
            Err(StatsError::ResourceLimit {
                message: "column-pivoted QR scratch",
                limit_bytes: 8,
                ..
            })
        ));
        assert_eq!(traffic, SolverTraffic::default());
        assert_eq!(tight.remainder().bytes(), 8);
        assert!(matches!(
            FaerBackend.least_squares_coefficients_leased(&x, n, p, &y, 0, &lease),
            Err(StatsError::Shape { .. })
        ));
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

    /// Two independent rows and three columns identify no coefficient vector: the default
    /// solve refuses, and the explicit minimum-norm solve of `X β = [1, 1]` is
    /// `(1/3, 1/3, 2/3)`.
    #[test]
    fn short_design_is_refused_and_minimum_norm_is_explicit() {
        let x = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = [1.0, 1.0];
        let mut ws = LeastSquaresWorkspace::default();
        assert!(matches!(
            FaerBackend.least_squares(&x, 2, 3, &y, &mut ws),
            Err(StatsError::RankDeficient { rank: 2, ncols: 3 })
        ));
        assert!(matches!(
            FaerBackend.least_squares_coefficients(&x, 2, 3, &y),
            Err(StatsError::RankDeficient { .. })
        ));
        let fit = FaerBackend.min_norm_least_squares(&x, 2, 3, &y).expect("minimum norm");
        assert_eq!(fit.rank, 2);
        let expect = [1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0];
        for (got, want) in fit.coefficients.iter().zip(expect) {
            assert!((got - want).abs() < 1e-10, "{got} vs {want}");
        }
        assert!(fit.rss < 1e-18, "rss {}", fit.rss);
    }

    /// A short design is rank deficient, not a shape error, whatever its rows are.
    #[test]
    fn short_rank_deficiency_is_not_a_shape_error() {
        let x = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let y = [1.0, 1.0];
        let mut ws = LeastSquaresWorkspace::default();
        let err = FaerBackend.least_squares(&x, 2, 3, &y, &mut ws).unwrap_err();
        assert!(matches!(err, StatsError::RankDeficient { ncols: 3, .. }), "{err:?}");
        assert!(matches!(
            FaerBackend.least_squares(&x, 2, 3, &y[..1], &mut ws),
            Err(StatsError::Shape { .. })
        ));
    }
}
