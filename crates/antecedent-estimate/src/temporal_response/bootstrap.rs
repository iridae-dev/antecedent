//! Joint circular-block bootstrap of the response surface.

use crate::temporal_block::{
    AlignedRows, GappedAlignedRows, RowBlockDraws, aligned_block_bootstrap, common_time_window,
    gapped_aligned_block_bootstrap,
};
use crate::util::BOOTSTRAP_MAX_FAILURE_FRAC;
use antecedent_core::ExecutionContext;
use antecedent_data::fill_circular_block_indexes;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use super::{band::HORIZON_BOOTSTRAP_STREAM, surface::FittedHorizon};

/// Fill `out` with a circular-block resample; reuses `out`'s allocation.
pub fn circular_block_positions_into(
    m: usize,
    block: usize,
    rng: &mut antecedent_core::CausalRng,
    out: &mut Vec<usize>,
) {
    out.clear();
    if m == 0 {
        return;
    }
    let block = block.clamp(1, m);
    out.resize(m, 0);
    if fill_circular_block_indexes(m, block, rng, out).is_err() {
        out.clear();
    }
}

/// Joint bootstrap draws of every horizon's fit.
///
/// The surface evaluates `cbar(a)'β` at many cells, so each replicate keeps the whole
/// coefficient vector and the resampled design column means for every horizon, rather
/// than collapsing to a single SE the way [`crate::util::bootstrap_se_with_scratch`] does. Failure
/// accounting matches that helper: too few survivors, or more than
/// [`BOOTSTRAP_MAX_FAILURE_FRAC`] soft failures, means no bootstrap SE is reported.
pub(super) struct SurfaceBootstrap {
    /// `draws[r]` is flattened `[coefs_h, means_h]` per horizon in horizon order.
    pub(super) draws: Vec<Vec<f64>>,
    /// `ncols[h]` is horizon `h`'s design column count (each horizon occupies `2 * p`).
    ncols: Vec<usize>,
    pub(super) cancelled: bool,
}

impl SurfaceBootstrap {
    pub(super) fn horizon_fit<'a>(&self, flat: &'a [f64], h: usize) -> (&'a [f64], &'a [f64]) {
        let mut off = 0;
        for (i, &p) in self.ncols.iter().enumerate() {
            if i == h {
                return (&flat[off..off + p], &flat[off + p..off + 2 * p]);
            }
            off += 2 * p;
        }
        (&[], &[])
    }
}

/// Joint circular-block bootstrap of the whole dose × horizon surface.
///
/// Serially dependent lag-aligned rows are resampled in blocks (never row by row):
/// an iid pairs bootstrap misses the autocovariance of the level's mean component and
/// cannot support a simultaneous band. Each replicate resamples circular blocks of
/// consecutive series times over the window every horizon can evaluate
/// ([`aligned_block_bootstrap`]), refits every horizon on its own lag-aligned rows at
/// those same times, and keeps the replicate only if every horizon fit succeeds, so
/// each kept replicate is one draw of the whole surface on one calendar resample.
/// Blocks resample lag-aligned tuples, so no replicate row straddles a block junction.
///
/// Returns `None` when no bootstrap was requested, when fewer than two replicates
/// survived, or when singular resamples pushed the failure fraction past the crate-wide
/// threshold. Callers then publish no band and must say so.
pub(super) fn bootstrap_surface(
    horizons: &[FittedHorizon],
    rows: &[AlignedRows],
    block_length: usize,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Option<SurfaceBootstrap> {
    if replicates == 0 || horizons.is_empty() || rows.iter().any(|r| r.rows == 0) {
        return None;
    }
    let mut refit = SurfaceRefit::new(horizons);
    let boot = aligned_block_bootstrap(
        rows,
        block_length,
        replicates,
        HORIZON_BOOTSTRAP_STREAM,
        ctx,
        |maps| refit.run(horizons, maps),
    )?;
    refit.finish(boot)
}

/// Resample stream base of the Bayesian block-weight draws of the joint surface.
const HORIZON_BAYES_STREAM: u64 = 0x7E51_u64;

/// Joint Bayesian block bootstrap of the whole dose × horizon surface: the posterior over the
/// surface given the empirical law of the time-aligned lag windows.
///
/// Each replicate draws one Exp(1) weight per block of `block_length` consecutive series times
/// of the window every horizon can evaluate, shared by every horizon, and refits every horizon
/// by weighted least squares on its own lag-aligned rows at those times (the row weights and
/// the weighted design column means). A kept replicate is therefore one draw of the whole
/// surface under one weight vector, coupled across horizons, which independent per-horizon
/// fits are not. Returns `None` on the conditions [`bootstrap_surface`] does.
pub(super) fn bayesian_bootstrap_surface(
    horizons: &[FittedHorizon],
    rows: &[AlignedRows],
    block_length: usize,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Option<SurfaceBootstrap> {
    if replicates == 0 || horizons.is_empty() || rows.iter().any(|r| r.rows == 0) {
        return None;
    }
    let (start, len) = common_time_window(rows)?;
    let block = block_length.clamp(1, len);
    let mut refit = SurfaceRefit::new(horizons);
    let mut draws = Vec::with_capacity(replicates as usize);
    let (mut attempted, mut cancelled) = (0u32, false);
    let mut weights = Vec::with_capacity(len);
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        attempted += 1;
        let mut rng = ctx.rng.stream_for(
            antecedent_core::StreamDomain::TemporalBlock,
            antecedent_core::derive_stream(HORIZON_BAYES_STREAM, u64::from(replicate)),
        );
        weights.clear();
        while weights.len() < len {
            let weight = antecedent_kernels::bayesian_bootstrap_weights(&mut rng, 1, false)[0];
            weights.extend(std::iter::repeat_n(weight, block.min(len - weights.len())));
        }
        if let Some(flat) = refit.run_weighted(horizons, rows, start, &weights) {
            if flat.iter().all(|x| x.is_finite()) {
                draws.push(flat);
            }
        }
    }
    if draws.len() < 2 {
        return None;
    }
    let failed = f64::from(attempted) - draws.len() as f64;
    if failed / f64::from(attempted) > BOOTSTRAP_MAX_FAILURE_FRAC {
        return None;
    }
    Some(SurfaceBootstrap { draws, ncols: refit.ncols, cancelled })
}

/// [`bootstrap_surface`] for horizons fit on a series with missing time points: one block plan
/// on the original window axis over the windows every horizon reads, keeping the times intact
/// in every horizon, with every horizon refit on those same times
/// ([`gapped_aligned_block_bootstrap`]).
pub(super) fn bootstrap_surface_gapped(
    horizons: &[FittedHorizon],
    rows: &[GappedAlignedRows],
    block_length: usize,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Option<SurfaceBootstrap> {
    if replicates == 0 || horizons.is_empty() {
        return None;
    }
    let mut refit = SurfaceRefit::new(horizons);
    let boot = gapped_aligned_block_bootstrap(
        rows,
        block_length,
        replicates,
        HORIZON_BOOTSTRAP_STREAM,
        ctx,
        |maps| refit.run(horizons, maps),
    )?;
    refit.finish(boot)
}

/// Buffers and shape of the per-replicate refit of every horizon.
struct SurfaceRefit {
    ols_ws: LeastSquaresWorkspace,
    x_boot: Vec<f64>,
    y_boot: Vec<f64>,
    ncols: Vec<usize>,
}

impl SurfaceRefit {
    fn new(horizons: &[FittedHorizon]) -> Self {
        Self {
            ols_ws: LeastSquaresWorkspace::default(),
            x_boot: Vec::new(),
            y_boot: Vec::new(),
            ncols: horizons.iter().map(|fitted| fitted.prepared.design.ncols).collect(),
        }
    }

    /// Every horizon's coefficients and design column means on its resampled rows.
    fn run(&mut self, horizons: &[FittedHorizon], maps: &[Vec<usize>]) -> Option<Vec<f64>> {
        let mut flat = Vec::new();
        for (fitted, map) in horizons.iter().zip(maps) {
            let design = &fitted.prepared.design;
            let (n, p, m) = (design.nrows, design.ncols, map.len());
            if m <= p {
                return None;
            }
            self.x_boot.resize(m * p, 0.0);
            self.y_boot.resize(m, 0.0);
            for (i, &src) in map.iter().enumerate() {
                self.y_boot[i] = design.outcome[src];
            }
            for c in 0..p {
                let column = &design.matrix[c * n..(c + 1) * n];
                for (i, &src) in map.iter().enumerate() {
                    self.x_boot[c * m + i] = column[src];
                }
            }
            let fit = FaerBackend
                .least_squares(&self.x_boot, m, p, &self.y_boot, &mut self.ols_ws)
                .ok()?;
            flat.extend(fit.coefficients);
            flat.extend(
                (0..p).map(|c| self.x_boot[c * m..(c + 1) * m].iter().sum::<f64>() / m as f64),
            );
        }
        Some(flat)
    }

    /// Every horizon's weighted-least-squares coefficients and weighted design column means,
    /// the lag-aligned rows at series times `start..start + weights.len()` weighted by
    /// `weights`.
    fn run_weighted(
        &mut self,
        horizons: &[FittedHorizon],
        rows: &[AlignedRows],
        start: usize,
        weights: &[f64],
    ) -> Option<Vec<f64>> {
        let m = weights.len();
        let total: f64 = weights.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return None;
        }
        let root: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
        let mut flat = Vec::new();
        for (fitted, aligned) in horizons.iter().zip(rows) {
            let design = &fitted.prepared.design;
            let (n, p) = (design.nrows, design.ncols);
            if m <= p {
                return None;
            }
            let offset = start.checked_sub(aligned.first_time)?;
            if offset + m > n {
                return None;
            }
            self.x_boot.resize(m * p, 0.0);
            self.y_boot.resize(m, 0.0);
            for (i, &r) in root.iter().enumerate() {
                self.y_boot[i] = design.outcome[offset + i] * r;
            }
            let mut means = Vec::with_capacity(p);
            for c in 0..p {
                let column = &design.matrix[c * n..(c + 1) * n];
                let mut weighted_sum = 0.0;
                for (i, &r) in root.iter().enumerate() {
                    let value = column[offset + i];
                    self.x_boot[c * m + i] = value * r;
                    weighted_sum += weights[i] * value;
                }
                means.push(weighted_sum / total);
            }
            let fit = FaerBackend
                .least_squares(&self.x_boot, m, p, &self.y_boot, &mut self.ols_ws)
                .ok()?;
            flat.extend(fit.coefficients);
            flat.extend(means);
        }
        Some(flat)
    }

    fn finish(self, boot: RowBlockDraws) -> Option<SurfaceBootstrap> {
        if boot.draws.len() < 2 {
            return None;
        }
        // Unattempted replicates after cancellation are not failures (mirrors
        // `finalize_bootstrap_se_ex`); singular resamples among those attempted are.
        let failed = f64::from(boot.attempted) - boot.draws.len() as f64;
        if failed / f64::from(boot.attempted) > BOOTSTRAP_MAX_FAILURE_FRAC {
            return None;
        }
        Some(SurfaceBootstrap { draws: boot.draws, ncols: self.ncols, cancelled: boot.cancelled })
    }
}

/// Unbiased sample SD of column `cell` across `draws`.
pub(super) fn sample_std_cell(draws: &[Vec<f64>], cell: usize) -> f64 {
    let column: Vec<f64> = draws.iter().map(|draw| draw[cell]).collect();
    antecedent_stats::sample_std(&column)
}
