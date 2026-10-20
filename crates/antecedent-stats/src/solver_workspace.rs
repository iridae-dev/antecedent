//! Selected design, equilibration, factorization, and solver scratch.
//!
//! A gather copies the selected rows once into a column-major design. Column
//! scales are a vector of length `ncols`. The QR factor and one scratch buffer
//! are reused for a later solve of that same gather. A rank-deficient solve
//! drops the factor and falls back to the minimum-norm SVD of that same
//! design, without gathering again.
//!
//! [`qr_serial_and_parallel_durations`] runs that QR under [`serial_scope`] and
//! under a caller-supplied wider [`FaerScope`]. It returns both durations. It
//! does not choose a crossover.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;
use std::time::Duration;

use crate::error::StatsError;
use crate::faer_backend::{
    FactorScratch, ScopedQr, column_pivoted_qr_duration, counted_bytes, factor_equilibrated,
    min_norm_on_column_major, serial_scope, solve_equilibrated, write_column_scales,
    write_residuals,
};
use crate::faer_scope::FaerScope;
use crate::linalg::{FitDiagnostics, LeastSquaresFit};

/// Durations of one column-pivoted QR under two explicit scopes.
///
/// `serial` is [`serial_scope`]: one worker and a scratch budget of [`u64::MAX`].
/// `parallel` is the wider scope passed by the caller. Neither duration is a
/// crossover threshold.
#[derive(Clone, Copy, Debug)]
pub struct QrScopeDurations {
    /// QR under [`serial_scope`].
    pub serial: Duration,
    /// QR under the explicit wider scope.
    pub parallel: Duration,
}

/// Factor `design` twice and return both durations.
///
/// The serial run uses [`serial_scope`]. The parallel run uses `wide`, which
/// must declare at least two workers. The caller records where the serial
/// scope wins. This function does not.
///
/// # Errors
///
/// * `wide` has fewer than two workers.
/// * `design` is empty or shorter than `nrows * ncols`.
/// * Either scope refuses its scratch.
#[must_use = "both durations are the measurement; neither one is a crossover threshold"]
pub fn qr_serial_and_parallel_durations(
    design: &[f64],
    nrows: usize,
    ncols: usize,
    wide: FaerScope,
) -> Result<QrScopeDurations, StatsError> {
    if wide.workers() < 2 {
        return Err(StatsError::Shape {
            message: "parallel factorization scope needs at least two workers",
        });
    }
    let serial = column_pivoted_qr_duration(design, nrows, ncols, serial_scope())?;
    let parallel = column_pivoted_qr_duration(design, nrows, ncols, wide)?;
    Ok(QrScopeDurations { serial, parallel })
}

/// Owned selected rows, column-major design, column scales, QR, and scratch.
///
/// There is no [`Default`]. A scope has to be passed in; a default would be a
/// place for the process pool to hide. [`FaerScope`] has no [`Default`] either.
pub struct SolverWorkspace {
    scope: FaerScope,
    selected: Vec<usize>,
    outcome: Vec<f64>,
    design: Vec<f64>,
    nrows: usize,
    ncols: usize,
    scales: Vec<f64>,
    factor: Option<ScopedQr>,
    factor_ready: bool,
    scratch: FactorScratch,
    /// Owned column-major layouts. Zero before the first gather, then one.
    layout_copies: u64,
    /// Successful gathers. A repeated gather rewrites the same design buffer.
    gather_writes: u64,
    /// Gathers that grew a retained buffer.
    buffer_grows: u64,
    /// Times the scratch allocation was replaced.
    scratch_allocations: u64,
    /// High-water mark of [`Self::retained_bytes`].
    peak_bytes: u64,
    /// SVD factor bytes counted for the fallback, then released.
    transient_bytes: u64,
    used_min_norm_fallback: bool,
}

impl SolverWorkspace {
    /// Empty workspace for `scope`. Nothing is allocated.
    #[must_use]
    pub const fn new(scope: FaerScope) -> Self {
        Self {
            scope,
            selected: Vec::new(),
            outcome: Vec::new(),
            design: Vec::new(),
            nrows: 0,
            ncols: 0,
            scales: Vec::new(),
            factor: None,
            factor_ready: false,
            scratch: FactorScratch::new(),
            layout_copies: 0,
            gather_writes: 0,
            buffer_grows: 0,
            scratch_allocations: 0,
            peak_bytes: 0,
            transient_bytes: 0,
            used_min_norm_fallback: false,
        }
    }

    /// Copy `selected` rows of a column-major source into one design buffer.
    ///
    /// A second gather of the same shape rewrites that buffer. Column scales
    /// are stored beside it. Any previous factor is dropped.
    ///
    /// # Errors
    ///
    /// The selection is empty, a row is out of range, or the source shape does
    /// not match `source_rows` and `ncols`.
    #[must_use = "a refused gather leaves the previous design in place"]
    pub fn gather_selected(
        &mut self,
        source_colmajor: &[f64],
        source_rows: usize,
        ncols: usize,
        outcome: &[f64],
        selected: &[usize],
    ) -> Result<(), StatsError> {
        if selected.is_empty() || ncols == 0 {
            return Err(StatsError::Shape { message: "empty selected design" });
        }
        if outcome.len() != source_rows {
            return Err(StatsError::Shape { message: "y length != source rows" });
        }
        let Some(source_cells) = source_rows.checked_mul(ncols) else {
            return Err(StatsError::Shape { message: "X buffer too short" });
        };
        if source_colmajor.len() < source_cells {
            return Err(StatsError::Shape { message: "X buffer too short" });
        }
        if selected.iter().any(|&row| row >= source_rows) {
            return Err(StatsError::Shape { message: "selected row out of range" });
        }
        let nrows = selected.len();
        let Some(cells) = nrows.checked_mul(ncols) else {
            return Err(StatsError::Shape { message: "selected design overflows" });
        };

        let grew_design = resize_f64(&mut self.design, cells);
        for column in 0..ncols {
            for (slot, &row) in selected.iter().enumerate() {
                self.design[column * nrows + slot] = source_colmajor[column * source_rows + row];
            }
        }
        let grew_outcome = resize_f64(&mut self.outcome, nrows);
        for (slot, &row) in selected.iter().enumerate() {
            self.outcome[slot] = outcome[row];
        }
        let grew_selected = replace_indices(&mut self.selected, selected);
        self.nrows = nrows;
        self.ncols = ncols;
        self.layout_copies = 1;
        self.gather_writes = self.gather_writes.saturating_add(1);
        self.factor = None;
        self.factor_ready = false;
        self.used_min_norm_fallback = false;
        let grew_scales = self.assign_scales();
        if grew_design || grew_outcome || grew_selected || grew_scales {
            self.buffer_grows = self.buffer_grows.saturating_add(1);
        }
        self.note_peak();
        Ok(())
    }

    /// Full-rank least squares of the gathered design.
    ///
    /// A second call reuses the factor and the scratch buffer.
    ///
    /// # Errors
    ///
    /// Nothing has been gathered, `nrows < ncols`, or the QR rank is short.
    /// The short-rank error is [`StatsError::RankDeficient`].
    #[must_use = "a refused full-rank solve is not a coefficient vector"]
    pub fn solve_full_rank(&mut self) -> Result<LeastSquaresFit, StatsError> {
        self.used_min_norm_fallback = false;
        self.ensure_gathered()?;
        if self.nrows < self.ncols {
            return Err(StatsError::Shape { message: "nrows < ncols" });
        }
        self.factorize()?;
        let (rank, rcond) = self
            .factor
            .as_ref()
            .ok_or(StatsError::Backend("qr factor was not retained".into()))?
            .rank_and_rcond(self.nrows);
        if rank < self.ncols {
            self.note_peak();
            return Err(StatsError::RankDeficient { rank, ncols: self.ncols });
        }
        let coefficients = self.coefficients_from_factor()?;
        let grow_count = u32::try_from(self.buffer_grows).unwrap_or(u32::MAX);
        let mut residuals = vec![0.0; self.nrows];
        let rss = write_residuals(
            &self.design,
            self.nrows,
            self.ncols,
            &self.outcome,
            &coefficients,
            &mut residuals,
        );
        Ok(LeastSquaresFit {
            coefficients,
            residuals,
            rank,
            rss,
            diagnostics: FitDiagnostics::new(rank, rcond, "faer", grow_count),
        })
    }

    /// Full-rank least squares, or the minimum-norm SVD when the QR rank is short.
    ///
    /// The fallback reads the gathered design in place. It does not gather again.
    /// A later call on that same gather repeats the SVD and does not rebuild the
    /// QR factor that the short rank already discarded.
    ///
    /// # Errors
    ///
    /// Nothing has been gathered, the shape is not a least-squares shape and is
    /// not rank-deficient, or the SVD does not converge.
    #[must_use = "a refused solve is not a coefficient vector"]
    pub fn solve(&mut self) -> Result<LeastSquaresFit, StatsError> {
        if self.used_min_norm_fallback {
            return self.solve_min_norm();
        }
        match self.solve_full_rank() {
            Err(StatsError::RankDeficient { .. }) => self.solve_min_norm(),
            other => other,
        }
    }

    /// Owned column-major designs. One after a gather, including a repeated gather.
    #[must_use]
    pub const fn layout_copies(&self) -> u64 {
        self.layout_copies
    }

    /// How many times selected rows were written into the design buffer.
    #[must_use]
    pub const fn gather_writes(&self) -> u64 {
        self.gather_writes
    }

    /// Scratch buffers held. Zero or one.
    #[must_use]
    pub const fn scratch_copies(&self) -> u64 {
        if self.scratch.is_allocated() { 1 } else { 0 }
    }

    /// Times the scratch buffer was allocated or replaced.
    #[must_use]
    pub const fn scratch_allocations(&self) -> u64 {
        self.scratch_allocations
    }

    /// Bytes of the column-major design.
    #[must_use]
    pub fn design_bytes(&self) -> u64 {
        counted_bytes(self.design.len(), size_of::<f64>())
    }

    /// Bytes of the column scales. This is not a second design.
    #[must_use]
    pub fn equilibration_bytes(&self) -> u64 {
        counted_bytes(self.scales.len(), size_of::<f64>())
    }

    /// High-water mark of counted workspace bytes.
    ///
    /// The returned fit is not included. A reused factor and scratch buffer do
    /// not raise it.
    #[must_use]
    pub const fn peak_bytes(&self) -> u64 {
        self.peak_bytes
    }

    /// Bytes currently retained: selection, outcome, design, scales, factor, and scratch.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        counted_bytes(self.selected.len(), size_of::<usize>())
            .saturating_add(counted_bytes(self.outcome.len(), size_of::<f64>()))
            .saturating_add(self.design_bytes())
            .saturating_add(self.equilibration_bytes())
            .saturating_add(self.factor.as_ref().map_or(0, ScopedQr::counted_bytes))
            .saturating_add(self.scratch.bytes())
            .saturating_add(self.transient_bytes)
    }

    /// Whether the latest successful [`Self::solve`] used the minimum-norm fallback.
    #[must_use]
    pub const fn used_min_norm_fallback(&self) -> bool {
        self.used_min_norm_fallback
    }

    fn assign_scales(&mut self) -> bool {
        let before = self.scales.capacity();
        write_column_scales(&self.design, self.nrows, self.ncols, &mut self.scales);
        self.scales.capacity() > before
    }

    fn ensure_gathered(&self) -> Result<(), StatsError> {
        if self.layout_copies == 0 {
            Err(StatsError::Shape { message: "solver workspace has no selected design" })
        } else {
            Ok(())
        }
    }

    fn factorize(&mut self) -> Result<(), StatsError> {
        if self.factor_ready {
            return Ok(());
        }
        let scope = self.scope;
        let before = self.scratch.address();
        let factored = factor_equilibrated(
            &self.design,
            self.nrows,
            self.ncols,
            &self.scales,
            scope,
            &mut self.scratch,
        )?;
        self.factor = Some(factored);
        self.factor_ready = true;
        self.account_scratch(before);
        Ok(())
    }

    fn coefficients_from_factor(&mut self) -> Result<Vec<f64>, StatsError> {
        let scope = self.scope;
        let before = self.scratch.address();
        let solved = {
            let Self { factor, outcome, scales, scratch, .. } = self;
            let Some(factored) = factor.as_ref() else {
                return Err(StatsError::Backend("qr factor was not retained".into()));
            };
            solve_equilibrated(factored, outcome, scales, scope, scratch)
        };
        self.account_scratch(before);
        solved
    }

    fn solve_min_norm(&mut self) -> Result<LeastSquaresFit, StatsError> {
        self.factor = None;
        self.factor_ready = false;
        self.note_peak();
        let grow_count = u32::try_from(self.buffer_grows).unwrap_or(u32::MAX);
        let scope = self.scope;
        let nrows = self.nrows;
        let ncols = self.ncols;
        let before = self.scratch.address();
        let solved = min_norm_on_column_major(
            &self.design,
            nrows,
            ncols,
            &self.outcome,
            scope,
            &mut self.scratch,
            grow_count,
        );
        self.account_scratch(before);
        let (fit, factor_bytes) = solved?;
        self.transient_bytes = factor_bytes;
        self.note_peak();
        self.transient_bytes = 0;
        self.used_min_norm_fallback = true;
        Ok(fit)
    }

    fn account_scratch(&mut self, before: Option<usize>) {
        if self.scratch.address() != before {
            self.scratch_allocations = self.scratch_allocations.saturating_add(1);
        }
        self.note_peak();
    }

    fn note_peak(&mut self) {
        let retained = self.retained_bytes();
        if retained > self.peak_bytes {
            self.peak_bytes = retained;
        }
    }
}

fn resize_f64(slot: &mut Vec<f64>, len: usize) -> bool {
    let before = slot.capacity();
    slot.resize(len, 0.0);
    slot.capacity() > before
}

fn replace_indices(slot: &mut Vec<usize>, rows: &[usize]) -> bool {
    let before = slot.capacity();
    slot.clear();
    slot.extend_from_slice(rows);
    slot.capacity() > before
}

#[cfg(test)]
#[allow(
    clippy::cast_precision_loss,
    reason = "fixtures use small integer coordinates that are exactly representable"
)]
mod tests {
    use std::mem::size_of;

    use super::{SolverWorkspace, qr_serial_and_parallel_durations};
    use crate::error::StatsError;
    use crate::faer_backend::{FaerBackend, serial_scope};
    use crate::faer_scope::FaerScope;

    fn scoped(workers: u32) -> SolverWorkspace {
        SolverWorkspace::new(FaerScope::try_new(workers, u64::MAX).unwrap())
    }

    fn line_source() -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let source_rows = 80usize;
        let mut source = vec![0.0; source_rows * 2];
        let mut outcome = vec![0.0; source_rows];
        for row in 0..source_rows {
            source[row] = 1.0;
            source[source_rows + row] = row as f64;
            outcome[row] = 3.0 + 4.0 * (row as f64);
        }
        (source, outcome, (10..60).collect())
    }

    #[test]
    fn solver_workspace_does_not_default_its_scope() {
        assert_no_default::<SolverWorkspace>();
        assert_no_default::<FaerScope>();
    }

    #[test]
    fn gather_once_reuses_scratch_and_matches_known_line() {
        let (source, outcome, selected) = line_source();
        let mut workspace = scoped(1);
        assert_eq!(workspace.layout_copies(), 0);
        assert_eq!(workspace.peak_bytes(), 0);
        workspace.gather_selected(&source, 80, 2, &outcome, &selected).unwrap();
        let design_bytes = u64::try_from(50 * 2 * size_of::<f64>()).unwrap();
        assert_eq!(workspace.layout_copies(), 1);
        assert_eq!(workspace.gather_writes(), 1);
        assert_eq!(workspace.design_bytes(), design_bytes);
        assert!(workspace.design_bytes() < u64::try_from(source.len() * size_of::<f64>()).unwrap());
        assert_eq!(workspace.equilibration_bytes(), u64::try_from(2 * size_of::<f64>()).unwrap());
        assert!(workspace.equilibration_bytes() < workspace.design_bytes());
        assert_eq!(workspace.scratch_copies(), 0);
        let gathered = workspace.peak_bytes();
        assert_eq!(gathered, workspace.retained_bytes());

        let fit = workspace.solve_full_rank().unwrap();
        assert!((fit.coefficients[0] - 3.0).abs() < 1e-10);
        assert!((fit.coefficients[1] - 4.0).abs() < 1e-10);
        assert_eq!(fit.rank, 2);
        assert!(fit.rss < 1e-20);
        assert_eq!(fit.diagnostics.backend, "faer");
        assert_eq!(fit.diagnostics.grow_count, 1);
        assert!(!workspace.used_min_norm_fallback());
        assert_eq!(workspace.scratch_copies(), 1);
        assert_eq!(workspace.scratch_allocations(), 1);
        assert!(workspace.peak_bytes() > gathered);
        assert_eq!(workspace.peak_bytes(), workspace.retained_bytes());

        let peak = workspace.peak_bytes();
        let again = workspace.solve_full_rank().unwrap();
        assert!((again.coefficients[0] - 3.0).abs() < 1e-10);
        assert!((again.coefficients[1] - 4.0).abs() < 1e-10);
        assert_eq!(again.diagnostics.grow_count, 1);
        assert_eq!(workspace.scratch_allocations(), 1);
        assert_eq!(workspace.scratch_copies(), 1);
        assert_eq!(workspace.peak_bytes(), peak);
        assert_eq!(workspace.layout_copies(), 1);
        assert_eq!(workspace.gather_writes(), 1);

        workspace.gather_selected(&source, 80, 2, &outcome, &selected).unwrap();
        assert_eq!(workspace.layout_copies(), 1);
        assert_eq!(workspace.gather_writes(), 2);
        assert_eq!(workspace.design_bytes(), design_bytes);
        assert_eq!(workspace.peak_bytes(), peak);
    }

    #[test]
    fn wider_scope_matches_the_known_line() {
        let (source, outcome, selected) = line_source();
        let mut workspace = scoped(2);
        workspace.gather_selected(&source, 80, 2, &outcome, &selected).unwrap();
        let fit = workspace.solve_full_rank().unwrap();
        assert!((fit.coefficients[0] - 3.0).abs() < 1e-10);
        assert!((fit.coefficients[1] - 4.0).abs() < 1e-10);
        assert_eq!(fit.rank, 2);
        assert_eq!(workspace.layout_copies(), 1);
        assert_eq!(workspace.scratch_copies(), 1);
        assert_eq!(workspace.scratch_allocations(), 1);
        assert!(!workspace.used_min_norm_fallback());
    }

    #[test]
    fn ill_conditioned_fallback_keeps_one_design_copy() {
        let n = 20usize;
        let column: Vec<f64> = (0..n).map(|index| (index as f64 * 0.7).sin()).collect();
        let mut source = column.clone();
        source.extend(column.iter().map(|value| 2.0 * value));
        let outcome: Vec<f64> = column.iter().map(|value| 3.0 * value).collect();
        let selected: Vec<usize> = (0..n).collect();
        let mut workspace = scoped(1);
        workspace.gather_selected(&source, n, 2, &outcome, &selected).unwrap();
        let design_bytes = u64::try_from(n * 2 * size_of::<f64>()).unwrap();
        assert_eq!(workspace.design_bytes(), design_bytes);
        assert_eq!(workspace.equilibration_bytes(), u64::try_from(2 * size_of::<f64>()).unwrap());

        let strict = workspace.solve_full_rank();
        assert!(matches!(strict, Err(StatsError::RankDeficient { rank: 1, ncols: 2 })));
        assert!(!workspace.used_min_norm_fallback());
        assert_eq!(workspace.layout_copies(), 1);
        assert_eq!(workspace.gather_writes(), 1);

        let fit = workspace.solve().unwrap();
        let backend = FaerBackend.min_norm_least_squares(&source, n, 2, &outcome).unwrap();
        assert!(workspace.used_min_norm_fallback());
        assert_eq!(fit.rank, 1);
        assert!((fit.coefficients[0] - 0.6).abs() < 1e-12, "{:?}", fit.coefficients);
        assert!((fit.coefficients[1] - 1.2).abs() < 1e-12, "{:?}", fit.coefficients);
        assert!(fit.rss < 1e-24);
        assert_eq!(fit.diagnostics.backend, "faer-svd");
        for (left, right) in fit.coefficients.iter().zip(&backend.coefficients) {
            assert!((left - right).abs() < 1e-12, "{left} vs {right}");
        }
        assert_eq!(workspace.layout_copies(), 1);
        assert_eq!(workspace.gather_writes(), 1);
        assert_eq!(workspace.design_bytes(), design_bytes);
        assert_eq!(workspace.scratch_copies(), 1);
        assert!(workspace.peak_bytes() > workspace.retained_bytes());

        let allocations = workspace.scratch_allocations();
        let peak = workspace.peak_bytes();
        let again = workspace.solve().unwrap();
        assert!((again.coefficients[0] - 0.6).abs() < 1e-12);
        assert!((again.coefficients[1] - 1.2).abs() < 1e-12);
        assert_eq!(workspace.scratch_allocations(), allocations);
        assert_eq!(workspace.scratch_copies(), 1);
        assert_eq!(workspace.peak_bytes(), peak);
        assert_eq!(workspace.design_bytes(), design_bytes);
        assert_eq!(workspace.layout_copies(), 1);
        assert_eq!(workspace.gather_writes(), 1);
    }

    #[test]
    fn qr_serial_and_parallel_durations_return_both_clocks() {
        let (design, nrows, ncols) = dense_line();
        let serial = serial_scope();
        assert_eq!(serial.workers(), 1);
        assert_eq!(serial.scratch_budget(), u64::MAX);
        let wide = FaerScope::try_new(2, u64::MAX).unwrap();
        assert!(wide.workers() > serial.workers());
        let measured = qr_serial_and_parallel_durations(&design, nrows, ncols, wide).unwrap();
        // Both clocks are returned. Which one is smaller is the caller's record.
        std::hint::black_box((measured.serial, measured.parallel));
    }

    #[test]
    fn one_worker_scope_is_refused_for_the_parallel_leg() {
        let (design, nrows, ncols) = dense_line();
        let serial = FaerScope::try_new(1, u64::MAX).unwrap();
        let error = qr_serial_and_parallel_durations(&design, nrows, ncols, serial).unwrap_err();
        assert_eq!(
            error,
            StatsError::Shape {
                message: "parallel factorization scope needs at least two workers",
            }
        );
    }

    fn dense_line() -> (Vec<f64>, usize, usize) {
        let nrows = 50usize;
        let mut design = vec![0.0; nrows * 2];
        for row in 0..nrows {
            design[row] = 1.0;
            design[nrows + row] = row as f64;
        }
        (design, nrows, 2)
    }

    fn assert_no_default<T: ?Sized>() {
        struct Marker;
        trait Probe<A> {
            fn probe() {}
        }
        impl<T: ?Sized> Probe<()> for T {}
        #[allow(dead_code)]
        impl<T: Default> Probe<Marker> for T {}
        <T as Probe<_>>::probe();
    }
}
