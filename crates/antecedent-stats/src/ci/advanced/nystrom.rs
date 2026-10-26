//! Scalable Nystrom GPDC conditional independence test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};
use antecedent_kernels::unbiased_index;
use crate::ci::block_shuffle::query_stream_salt;
use crate::ci::types::{CiBatchRequest, CiBatchResult, CiWorkspace, ConditionalIndependenceTest, PreparedCiTest, nonparametric_permutation_count, requested_block_size};
use crate::error::StatsError;
use super::gpdc::{GPDC_FACTOR_CACHE, ResidualNull, ensure_residual_information, residual_dcor_permutation_result, standardized_z};

/// Largest Nyström rank [`GpdcNystrom`] accepts: the `m × m` normal equations are accumulated
/// per row block, so memory grows as `m²` times the block count.
pub const GPDC_NYSTROM_MAX_INDUCING_POINTS: usize = 1_024;

/// Row blocks the Nyström accumulations are split into at most. The split depends only on
/// the row count, so the sums (and every result) are identical at any thread count.
const NYSTROM_MAX_ROW_BLOCKS: usize = 32;

/// Rows per Nyström row block, at least this many.
const NYSTROM_MIN_BLOCK_ROWS: usize = 2_048;

/// Relative diagonal jitter of the Nyström normal equations: the RBF kernel matrix of
/// inducing points is numerically rank deficient, and without it the factorization of
/// `K_mn K_nm + λ K_mm` would fail on exactly the smooth conditioning sets GPDC is for.
pub(super) const NYSTROM_JITTER: f64 = 1e-10;

/// Scalable GPDC: Nyström (subset-of-regressors) RBF-GP residualization of X and Y on Z
/// followed by the same distance correlation and permutation null as [`Gpdc`](super::Gpdc).
///
/// This is a separate construction from [`Gpdc`](super::Gpdc), not an approximation switch: it matches
/// `Gpdc` in kernel, standardisation and ridge, but replaces the full-rank predictor
/// `K (K + λI)⁻¹ y` by the rank-`m` one `K_nm (K_mn K_nm + λ K_mm + εI)⁻¹ K_mn y`, where the
/// `m = inducing_points` inducing rows are a seeded draw without replacement from the
/// sample (`inducing_seed`, independent of the execution seed, so the residual series is a
/// function of the data and these two fields alone) and `ε` is [`NYSTROM_JITTER`] times the
/// mean diagonal. Normal equations are accumulated in row blocks and X and Y are two
/// right-hand sides of one factorization, so time is `O(n m²)` for the fit and `O(n m)` per
/// series, and memory is `O(n + m²)` — never `n × n` — which makes 100 000 rows with
/// `m = 256` a matter of seconds, where the exact [`Gpdc`](super::Gpdc) kernel's `O(n²)` memory and
/// `O(n³)` work are quoted and refused when the caller's budget does not hold them. With
/// `inducing_points >= n` every row is an inducing point.
///
/// The rank limits what Z structure is removed, not which null is used: a Z effect that the
/// `m`-point kernel cannot represent leaves Z-dependence in both residuals and inflates the
/// rejection rate, which is why the construction has its own Type I / power calibration.
/// Factory id: `gpdc_nystrom`.
#[derive(Clone, Debug)]
pub struct GpdcNystrom {
    /// RBF length scale (on standardised Z).
    pub(super) length_scale: f64,
    /// Ridge.
    pub(super) ridge: f64,
    /// Nyström rank `m`: inducing rows drawn from the sample (at most
    /// [`GPDC_NYSTROM_MAX_INDUCING_POINTS`]).
    pub(super) inducing_points: usize,
    /// Seed of the inducing-row draw.
    pub(super) inducing_seed: u64,
}

impl Default for GpdcNystrom {
    fn default() -> Self {
        Self::new()
    }
}

impl GpdcNystrom {
    /// Construct with defaults: `Gpdc`'s length scale and ridge, 256 inducing points.
    #[must_use]
    pub fn new() -> Self {
        Self {
            length_scale: 1.0,
            ridge: 1e-3,
            inducing_points: 256,
            inducing_seed: 0x4E59_5354_0001,
        }
    }
}

impl GpdcNystrom {
    /// RBF length scale on the standardized conditioning columns.
    #[must_use]
    pub const fn length_scale(&self) -> f64 {
        self.length_scale
    }

    /// Ridge `λ` of the kernel.
    #[must_use]
    pub const fn ridge(&self) -> f64 {
        self.ridge
    }

    /// Nyström rank `m`.
    #[must_use]
    pub const fn inducing_points(&self) -> usize {
        self.inducing_points
    }

    /// Seed of the inducing-row draw.
    #[must_use]
    pub const fn inducing_seed(&self) -> u64 {
        self.inducing_seed
    }

    /// These settings with the given RBF length scale.
    ///
    /// # Errors
    ///
    /// [`StatsError::Shape`] unless `length_scale` is finite and positive.
    pub fn with_length_scale(mut self, length_scale: f64) -> Result<Self, StatsError> {
        if !(length_scale.is_finite() && length_scale > 0.0) {
            return Err(StatsError::Shape {
                message: "GPDC length scale must be finite and positive",
            });
        }
        self.length_scale = length_scale;
        Ok(self)
    }

    /// These settings with the given kernel ridge.
    ///
    /// # Errors
    ///
    /// [`StatsError::Shape`] unless `ridge` is finite and positive.
    pub fn with_ridge(mut self, ridge: f64) -> Result<Self, StatsError> {
        if !(ridge.is_finite() && ridge > 0.0) {
            return Err(StatsError::Shape { message: "GPDC ridge must be finite and positive" });
        }
        self.ridge = ridge;
        Ok(self)
    }

    /// These settings with Nyström rank `m` inducing rows.
    ///
    /// # Errors
    ///
    /// [`StatsError::Shape`] unless `1 <= m <= GPDC_NYSTROM_MAX_INDUCING_POINTS`.
    pub fn with_inducing_points(mut self, m: usize) -> Result<Self, StatsError> {
        if m == 0 || m > GPDC_NYSTROM_MAX_INDUCING_POINTS {
            return Err(StatsError::Shape {
                message: "GPDC Nystrom inducing_points must be between 1 and 1024",
            });
        }
        self.inducing_points = m;
        Ok(self)
    }

    /// These settings with the given seed for the inducing-row draw.
    #[must_use]
    pub const fn with_inducing_seed(mut self, seed: u64) -> Self {
        self.inducing_seed = seed;
        self
    }
}

impl ConditionalIndependenceTest for GpdcNystrom {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(StatsError::Shape {
                message: "GPDC length scale must be finite and positive",
            });
        }
        if !self.ridge.is_finite() || self.ridge <= 0.0 {
            return Err(StatsError::Shape { message: "GPDC ridge must be finite and positive" });
        }
        if self.inducing_points == 0 || self.inducing_points > GPDC_NYSTROM_MAX_INDUCING_POINTS {
            return Err(StatsError::Shape {
                message: "GPDC Nystrom inducing_points must be between 1 and 1024",
            });
        }
        let request = &prepared.bind_request(request);
        let n = request.columns.first().map_or(0, |c| c.len());
        if n == 0 {
            return Err(StatsError::Shape { message: "no columns" });
        }
        // As for `Gpdc`, both series are residualized on Z before permuting, so a
        // contiguous-block permutation of the Y residual is a valid null.
        let block_size = requested_block_size(request.significance);
        let n_perm = nonparametric_permutation_count(request.significance);
        let mut fits: Vec<(Vec<usize>, NystromFit)> = Vec::new();
        let mut results = Vec::with_capacity(request.queries.len());
        for q in request.queries {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let fit: Option<&NystromFit> = if z.is_empty() {
                None
            } else {
                let at = if let Some(at) = fits.iter().position(|(key, _)| key.as_slice() == z) {
                    at
                } else {
                    let fit = nystrom_fit(request.columns, z, n, self, ctx)?;
                    if fits.len() >= GPDC_FACTOR_CACHE {
                        fits.remove(0);
                    }
                    fits.push((z.to_vec(), fit));
                    fits.len() - 1
                };
                Some(&fits[at].1)
            };
            let [rx, ry] =
                nystrom_residuals(fit, [request.columns[q.x], request.columns[q.y]], ctx)?;
            ensure_residual_information(request.columns[q.x], &rx)?;
            ensure_residual_information(request.columns[q.y], &ry)?;
            results.push(residual_dcor_permutation_result(
                &rx,
                &ry,
                ResidualNull {
                    salt: 0x69DD_u64 ^ query_stream_salt(&[q.x], &[q.y], z),
                    n_perm,
                    block_size,
                },
                ctx,
            ));
        }
        Ok(CiBatchResult { results })
    }

    fn attainable_p(&self) -> crate::ci::types::AttainableP {
        crate::ci::types::AttainableP::NonparametricPermutation
    }
}

/// The Nyström fit for one conditioning set: standardised Z, inducing rows and the Cholesky
/// factor of `K_mn K_nm + λ K_mm + εI`.
#[derive(Debug)]
pub(super) struct NystromFit {
    z_std: Vec<f64>,
    zdim: usize,
    inducing: Vec<f64>,
    rank: usize,
    ls2: f64,
    chol: Vec<f64>,
}

impl NystromFit {
    /// Kernel row of one standardised Z row against the inducing rows.
    fn kernel_row(&self, row: usize, out: &mut [f64]) {
        rbf_row(
            &self.z_std[row * self.zdim..(row + 1) * self.zdim],
            &self.inducing,
            self.zdim,
            self.ls2,
            out,
        );
    }
}

/// `exp(-d² / (2 ℓ²))` between `point` and each row of row-major `centres`.
fn rbf_row(point: &[f64], centres: &[f64], zdim: usize, ls2: f64, out: &mut [f64]) {
    for (slot, centre) in out.iter_mut().zip(centres.chunks_exact(zdim)) {
        let mut d2 = 0.0;
        for (a, b) in point.iter().zip(centre) {
            let diff = a - b;
            d2 += diff * diff;
        }
        *slot = (-0.5 * d2 / ls2).exp();
    }
}

/// `(rows per block, block count)` for the Nyström accumulations over `n` rows.
fn nystrom_blocks(n: usize) -> (usize, usize) {
    let block_rows = n.div_ceil(NYSTROM_MAX_ROW_BLOCKS).max(NYSTROM_MIN_BLOCK_ROWS);
    (block_rows, n.div_ceil(block_rows))
}

/// Bytes a Nyström fit over `n` rows, `zdim` conditioning columns and rank `m` plans to hold.
fn nystrom_bytes(n: usize, zdim: usize, m: usize) -> Option<u64> {
    let (_, blocks) = nystrom_blocks(n);
    let per_row = zdim.checked_add(9)?;
    let words = n
        .checked_mul(per_row)?
        .checked_add(blocks.checked_add(3)?.checked_mul(m.checked_mul(m)?)?)?
        .checked_add(m.checked_mul(zdim)?)?;
    u64::try_from(words.checked_mul(core::mem::size_of::<f64>())?).ok()
}

/// The `m` inducing rows (ascending): a seeded draw without replacement from `n` rows,
/// keyed by the conditioning set so different sets draw independent rows.
pub(super) fn nystrom_inducing_rows(n: usize, m: usize, seed: u64, z: &[usize]) -> Vec<usize> {
    let mut rng = antecedent_core::RngFactory::from_seed(seed)
        .stream_for(StreamDomain::StatsCi, query_stream_salt(&[], &[], z));
    let mut rows: Vec<usize> = (0..n).collect();
    for i in 0..m {
        let j = i + unbiased_index(&mut rng, n - i);
        rows.swap(i, j);
    }
    rows.truncate(m);
    rows.sort_unstable();
    rows
}

/// Fit the Nyström predictor for conditioning set `z`.
pub(super) fn nystrom_fit(
    columns: &[&[f64]],
    z: &[usize],
    n: usize,
    gp: &GpdcNystrom,
    ctx: &ExecutionContext,
) -> Result<NystromFit, StatsError> {
    let m = gp.inducing_points.min(n);
    let zdim = z.len();
    let _admitted =
        crate::matching::check_budget(ctx, "GPDC Nystrom fit", nystrom_bytes(n, zdim, m))?;
    let z_std = standardized_z(columns, z, n)?;
    let rows = nystrom_inducing_rows(n, m, gp.inducing_seed, z);
    let mut inducing = Vec::with_capacity(m * zdim);
    for &r in &rows {
        inducing.extend_from_slice(&z_std[r * zdim..(r + 1) * zdim]);
    }
    let ls2 = gp.length_scale * gp.length_scale;
    let mut fit = NystromFit { z_std, zdim, inducing, rank: m, ls2, chol: Vec::new() };

    // K_mn K_nm, lower triangle, summed over row blocks in block order.
    let (block_rows, blocks) = nystrom_blocks(n);
    let partials = ctx.map_indexed(blocks, |block, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(StatsError::Cancelled);
        }
        let mut acc = vec![0.0; m * m];
        let mut k = vec![0.0; m];
        for row in block * block_rows..((block + 1) * block_rows).min(n) {
            fit.kernel_row(row, &mut k);
            for (j, &kj) in k.iter().enumerate() {
                for (slot, kl) in acc[j * m..=j * m + j].iter_mut().zip(&k) {
                    *slot += kj * kl;
                }
            }
        }
        Ok(acc)
    })?;
    let mut normal = vec![0.0; m * m];
    for partial in &partials {
        for (slot, v) in normal.iter_mut().zip(partial) {
            *slot += v;
        }
    }
    drop(partials);
    // + λ K_mm, mirrored to the full symmetric matrix, + jitter on the diagonal.
    let mut k_row = vec![0.0; m];
    for j in 0..m {
        rbf_row(&fit.inducing[j * zdim..(j + 1) * zdim], &fit.inducing, zdim, ls2, &mut k_row);
        for l in 0..=j {
            normal[j * m + l] += gp.ridge * k_row[l];
        }
    }
    let mean_diagonal = (0..m).map(|j| normal[j * m + j]).sum::<f64>() / m as f64;
    for j in 0..m {
        normal[j * m + j] += NYSTROM_JITTER * mean_diagonal;
        for l in 0..j {
            normal[l * m + j] = normal[j * m + l];
        }
    }
    fit.chol = crate::gram::cholesky_spd(&normal, m).ok_or_else(|| {
        StatsError::Backend("GPDC Nystrom normal equations not positive definite".into())
    })?;
    Ok(fit)
}

/// Nyström residuals of two series given the fit (`None`: empty Z, only centred).
pub(super) fn nystrom_residuals(
    fit: Option<&NystromFit>,
    targets: [&[f64]; 2],
    ctx: &ExecutionContext,
) -> Result<[Vec<f64>; 2], StatsError> {
    let n = targets[0].len();
    let mut centered: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for (slot, y) in centered.iter_mut().zip(targets) {
        if y.len() != n || y.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::Shape {
                message: "non-finite response in GPDC residualization",
            });
        }
        let mean = y.iter().sum::<f64>() / n as f64;
        *slot = y.iter().map(|value| value - mean).collect();
    }
    let Some(fit) = fit else {
        return Ok(centered);
    };
    let m = fit.rank;
    let (block_rows, blocks) = nystrom_blocks(n);
    // K_mn y for both series.
    let moments = ctx.map_indexed(blocks, |block, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(StatsError::Cancelled);
        }
        let mut sums = [vec![0.0; m], vec![0.0; m]];
        let mut k = vec![0.0; m];
        for row in block * block_rows..((block + 1) * block_rows).min(n) {
            fit.kernel_row(row, &mut k);
            for (sum, y) in sums.iter_mut().zip(&centered) {
                for (slot, kl) in sum.iter_mut().zip(&k) {
                    *slot += kl * y[row];
                }
            }
        }
        Ok(sums)
    })?;
    let mut coefficients: [Vec<f64>; 2] = [vec![0.0; m], vec![0.0; m]];
    for sums in &moments {
        for (total, sum) in coefficients.iter_mut().zip(sums) {
            for (slot, v) in total.iter_mut().zip(sum) {
                *slot += v;
            }
        }
    }
    for coefficient in &mut coefficients {
        *coefficient = crate::gram::chol_solve(&fit.chol, m, coefficient)
            .ok_or_else(|| StatsError::Backend("GPDC Nystrom solve failed".into()))?;
    }
    // Residual = centred y - K_nm c.
    let parts = ctx.map_indexed(blocks, |block, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(StatsError::Cancelled);
        }
        let range = block * block_rows..((block + 1) * block_rows).min(n);
        let mut out = [Vec::with_capacity(range.len()), Vec::with_capacity(range.len())];
        let mut k = vec![0.0; m];
        for row in range {
            fit.kernel_row(row, &mut k);
            for ((part, y), c) in out.iter_mut().zip(&centered).zip(&coefficients) {
                let fitted: f64 = k.iter().zip(c).map(|(kl, cl)| kl * cl).sum();
                part.push(y[row] - fitted);
            }
        }
        Ok(out)
    })?;
    let mut residuals: [Vec<f64>; 2] = [Vec::with_capacity(n), Vec::with_capacity(n)];
    for part in parts {
        for (total, piece) in residuals.iter_mut().zip(part) {
            total.extend(piece);
        }
    }
    Ok(residuals)
}
