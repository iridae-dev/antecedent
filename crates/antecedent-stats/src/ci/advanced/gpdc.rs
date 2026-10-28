//! Exact Gaussian-process distance correlation (GPDC) conditional independence test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ci::block_shuffle::{add_one_pvalue, permutation_null_exceedances, query_stream_salt};
use crate::ci::residualize::residual_is_uninformative;
use crate::ci::types::{
    CiBatchRequest, CiBatchResult, CiQuery, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, nonparametric_permutation_count, requested_block_size,
};
use crate::dcor::distance_correlation;
use crate::error::StatsError;
use antecedent_core::search_budget::GpdcKernel;
use antecedent_core::{ExecutionContext, StreamDomain};

/// Conditioning sets whose kernel factorization one batch keeps for reuse, when the caller's
/// memory budget holds that many beside the working kernel; one when it holds only one.
pub(super) const GPDC_FACTOR_CACHE: usize = 2;

/// Bytes of one binary64 kernel or factor entry.
const GPDC_ENTRY_BYTES: u64 = 8;

/// Work and memory an exact conditional GPDC batch needs, quoted before any kernel exists.
///
/// The figures come from the declared exact-mode charge of
/// [`GpdcKernel`](antecedent_core::search_budget::GpdcKernel): a factorization holds the dense
/// kernel and its Cholesky factor (`2 · rows² · 8` bytes) plus the standardized conditioning
/// block, and costs `rows³ + rows² · conditioning columns` operations. The batch keeps up to
/// [`Self::cache_slots`] factors, so its peak adds the cached factors that stay live while the
/// next one is built. The cache is as wide as the caller's free memory allows; the batch is
/// refused, not shrunk, when even one factorization does not fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpdcExactQuote {
    /// Rows of every kernel.
    pub rows: u64,
    /// Widest conditioning set in the batch.
    pub conditioning_columns: u64,
    /// Kernel factorizations the batch will perform, counting rebuilds after eviction.
    pub factorizations: u64,
    /// Operations of all those factorizations.
    pub work: u64,
    /// Bytes leased from the caller's admission pool before the first kernel is allocated.
    pub peak_bytes: u64,
    /// Factors kept for reuse between queries.
    pub cache_slots: usize,
}

#[cfg(test)]
thread_local! {
    /// Kernel factorizations this thread has performed (test observation of the quote).
    pub(super) static GPDC_FACTORIZATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Factorizations a batch performs when it keeps `cache_slots` factors, oldest evicted first.
fn count_factorizations(queries: &[CiQuery], z_flat: &[usize], cache_slots: usize) -> u64 {
    let mut cached: Vec<&[usize]> = Vec::with_capacity(cache_slots);
    let mut builds = 0u64;
    for query in queries {
        if query.z_len == 0 {
            continue;
        }
        let z = &z_flat[query.z_start..query.z_start + query.z_len];
        if cached.contains(&z) {
            continue;
        }
        builds += 1;
        while cached.len() >= cache_slots {
            cached.remove(0);
        }
        cached.push(z);
    }
    builds
}

/// Native GPDC: RBF-GP residualization (ridge) + distance-correlation on residuals.
///
/// Residualization centers the response and factors `K+λI` once per distinct Z set with
/// Cholesky (X and Y are two right-hand sides of the same factorization, and queries of one
/// batch sharing a Z set reuse it). The GP mean prediction is `Kα` with `(K+λI)α = y_c`, so the
/// residual is exactly `y_c − Kα = λα` (MM-008). The earlier Jacobi-on-raw-`y` path rejected
/// conditional nulls and missed two-conditioner alternatives against the pinned advanced-CI
/// oracle.
///
/// The conditional test has no row ceiling. Its work and memory are quoted from the declared
/// exact-mode charge ([`GpdcExactQuote`]) before anything is allocated: the kernel, factor and
/// cached-factor bytes are leased from the caller's admission pool, and a batch that does not
/// fit is refused with the numbers, never run on a smaller kernel. [`Self::with_work_limit`]
/// adds the caller's operation ceiling. [`GpdcNystrom`](super::GpdcNystrom) is the separately identified
/// low-rank construction; it is chosen by name, never as a fallback from this one.
#[derive(Clone, Debug)]
pub struct Gpdc {
    /// RBF length scale (validated finite and positive when set through the builder).
    pub(super) length_scale: f64,
    /// Ridge (validated finite and positive when set through the builder).
    pub(super) ridge: f64,
    /// Ceiling on the operations of one batch's factorizations, when the caller declares one.
    work_limit: Option<u64>,
}

impl Default for Gpdc {
    fn default() -> Self {
        Self::new()
    }
}

impl Gpdc {
    /// Construct with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self { length_scale: 1.0, ridge: 1e-3, work_limit: None }
    }

    /// RBF length scale on the standardized conditioning columns.
    #[must_use]
    pub const fn length_scale(&self) -> f64 {
        self.length_scale
    }

    /// Ridge `λ` of the kernel `K + λI`.
    #[must_use]
    pub const fn ridge(&self) -> f64 {
        self.ridge
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

    /// Declare the most operations a batch's kernel factorizations may take. A batch whose
    /// quote exceeds it is refused before any kernel is built.
    #[must_use]
    pub const fn with_work_limit(mut self, operations: u64) -> Self {
        self.work_limit = Some(operations);
        self
    }

    /// Quote the exact conditional work of `request` against the caller's budget, without
    /// allocating a kernel. `None` when no query conditions on anything.
    ///
    /// The memory figure is the one [`ConditionalIndependenceTest::test_batch`] leases.
    ///
    /// # Errors
    ///
    /// [`StatsError::ResourceLimit`] when the quote overflows `u64`,
    /// [`StatsError::WorkLimit`] when it exceeds [`Self::with_work_limit`], and
    /// [`StatsError::Shape`] for an empty request.
    pub fn quote(
        &self,
        request: &CiBatchRequest<'_>,
        ctx: &ExecutionContext,
    ) -> Result<Option<GpdcExactQuote>, StatsError> {
        let n = request.columns.first().map_or(0, |c| c.len());
        if n == 0 {
            return Err(StatsError::Shape { message: "no columns" });
        }
        self.quote_rows(n, request.queries, request.z_flat, ctx)
    }

    fn quote_rows(
        &self,
        n: usize,
        queries: &[CiQuery],
        z_flat: &[usize],
        ctx: &ExecutionContext,
    ) -> Result<Option<GpdcExactQuote>, StatsError> {
        let widest = queries.iter().map(|q| q.z_len).max().unwrap_or(0);
        if widest == 0 {
            return Ok(None);
        }
        let limit_bytes =
            ctx.memory_limit_bytes().unwrap_or(u64::MAX).min(ctx.admission().available().bytes());
        let overflow = || StatsError::ResourceLimit {
            message: "GPDC exact kernel quote overflows u64",
            required_bytes: u64::MAX,
            limit_bytes,
        };
        let (rows, columns) = (
            u64::try_from(n).map_err(|_| overflow())?,
            u64::try_from(widest).map_err(|_| overflow())?,
        );
        let per_factor = GpdcKernel::new(rows, columns).estimate().map_err(|_| overflow())?;
        let factor_bytes = rows
            .checked_mul(rows)
            .and_then(|c| c.checked_mul(GPDC_ENTRY_BYTES))
            .ok_or_else(overflow)?;
        let kept_beside =
            (GPDC_FACTOR_CACHE as u64 - 1).checked_mul(factor_bytes).ok_or_else(overflow)?;
        let wide = per_factor.memory_bytes().checked_add(kept_beside);
        let (cache_slots, peak_bytes) = match wide {
            Some(bytes) if bytes <= limit_bytes => (GPDC_FACTOR_CACHE, bytes),
            _ => (1, per_factor.memory_bytes()),
        };
        let factorizations = count_factorizations(queries, z_flat, cache_slots);
        let work = per_factor.work().checked_mul(factorizations).ok_or_else(overflow)?;
        if let Some(limit_work) = self.work_limit {
            if work > limit_work {
                return Err(StatsError::WorkLimit {
                    message: "GPDC exact kernel factorization",
                    required_work: work,
                    limit_work,
                });
            }
        }
        Ok(Some(GpdcExactQuote {
            rows,
            conditioning_columns: columns,
            factorizations,
            work,
            peak_bytes,
            cache_slots,
        }))
    }
}

impl ConditionalIndependenceTest for Gpdc {
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
        let request = &prepared.bind_request(request);
        let n = request.columns.first().map_or(0, |c| c.len());
        if n == 0 {
            return Err(StatsError::Shape { message: "no columns" });
        }
        // Unlike KnnDependence / SymbolicCmi, GPDC residualizes X and Y on Z through the GP
        // regression *before* permuting (see `gp_residual`), so by the time the null is built
        // there is nothing left to stratify — Z's influence is already removed from both
        // residual series. That makes a contiguous-block permutation of the Y residual a
        // direct, valid substitution for the exchangeable shuffle: the same residual-null
        // architecture the ParCorr family uses. So `block_size` is honoured, not rejected.
        let block_size = requested_block_size(request.significance);
        let n_perm = nonparametric_permutation_count(request.significance);
        // The kernel, its factor and the cached factors are quoted from the caller's budget and
        // leased before the first one is allocated. A batch that does not fit is refused here.
        let quote = self.quote_rows(n, request.queries, request.z_flat, ctx)?;
        let _kernel_lease = match quote {
            Some(quote) => Some(crate::matching::check_budget(
                ctx,
                "GPDC exact kernel, factor and cache (next attempt: a larger memory limit, or \
                 request gpdc_nystrom explicitly)",
                Some(quote.peak_bytes),
            )?),
            None => None,
        };
        let cache_slots = quote.map_or(GPDC_FACTOR_CACHE, |quote| quote.cache_slots);
        let mut factors: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();
        let mut results = Vec::with_capacity(request.queries.len());
        for q in request.queries {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let factor: Option<&[f64]> = if z.is_empty() {
                None
            } else {
                let at = if let Some(at) = factors.iter().position(|(key, _)| key.as_slice() == z) {
                    at
                } else {
                    if ctx.cancellation.is_cancelled() {
                        return Err(StatsError::Cancelled);
                    }
                    // Evict before building so the cached factors and the working kernel
                    // never exceed the leased peak.
                    while factors.len() >= cache_slots {
                        factors.remove(0);
                    }
                    let chol = gp_factor(request.columns, z, n, self)?;
                    factors.push((z.to_vec(), chol));
                    factors.len() - 1
                };
                Some(factors[at].1.as_slice())
            };
            let rx = gp_residual(request.columns[q.x], factor, self.ridge)?;
            let ry = gp_residual(request.columns[q.y], factor, self.ridge)?;
            // A constant series, or one the GP explains completely from Z, has no distance
            // variance left: dCor is undefined, and reporting 0 (p = 1) would be an
            // independence verdict manufactured by the absence of information.
            ensure_residual_information(request.columns[q.x], &rx)?;
            ensure_residual_information(request.columns[q.y], &ry)?;
            results.push(residual_dcor_permutation_result(
                &rx,
                &ry,
                ResidualNull {
                    salt: 0x69DC_u64 ^ query_stream_salt(&[q.x], &[q.y], z),
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

/// How a GPDC-family test permutes its residuals.
#[derive(Clone, Copy)]
pub(super) struct ResidualNull {
    /// Stream salt of the query (distinguishes the test family and the query).
    pub(super) salt: u64,
    /// Permutation replicates.
    pub(super) n_perm: usize,
    /// Contiguous block length (1: exchangeable shuffle).
    pub(super) block_size: usize,
}

/// Distance correlation of two Z-free residual series with its permutation p-value.
///
/// The distance correlation is `O(n log n)` time and `O(n)` memory (sorting / Fenwick
/// identities): neither the statistic nor any permutation replicate builds an `n x n`
/// distance matrix. The null permutes the Y residuals (Z influence already removed) and
/// recomputes dCor; the add-one p-value keeps it in `(0, 1]`. `block_size > 1` permutes
/// contiguous blocks so the residual's serial dependence survives into the null; otherwise
/// it is an ordinary exchangeable shuffle.
pub(super) fn residual_dcor_permutation_result(
    rx: &[f64],
    ry: &[f64],
    null: ResidualNull,
    ctx: &ExecutionContext,
) -> CiResult {
    let dcor = distance_correlation(rx, ry);
    let mut ry_perm = ry.to_vec();
    let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, null.salt);
    // One stratum holding every row: the exchangeable shuffle.
    let everything = vec![(0..ry.len()).collect::<Vec<usize>>()];
    let null_ge = permutation_null_exceedances(
        null.n_perm,
        &mut ry_perm,
        &everything,
        null.block_size,
        &mut rng,
        |permuted| {
            Ok::<_, std::convert::Infallible>(dcor_reaches_observed(
                distance_correlation(rx, permuted),
                dcor,
            ))
        },
    )
    .unwrap_or_else(|never| match never {});
    let p = add_one_pvalue(null_ge, null.n_perm);
    CiResult { statistic: dcor, p_value: p, df: 0.0, ci: None }
}

/// Whether a permuted distance correlation reaches the observed one.
///
/// A permutation that is a symmetry of the observed series pair gives the same distance
/// correlation as a real number, but `dCov²` is a difference of three nearly equal sums, so the
/// two floating-point values can differ by about `1e-16` in `dCov²` (about `1e-8` in `dCor` near
/// zero). Counted strictly, those tied permutations fall below the observed value at random and the
/// p-value is too small. The comparison is made on the squared correlation with an absolute slack
/// of `1e-12`, far above that rounding and far below the squared-correlation gap between
/// genuinely different arrangements of a realistic sample.
fn dcor_reaches_observed(null: f64, observed: f64) -> bool {
    null * null >= observed * observed - 1e-12
}

/// Refuse a residual carrying no information about its column.
pub(super) fn ensure_residual_information(raw: &[f64], resid: &[f64]) -> Result<(), StatsError> {
    if residual_is_uninformative(raw, resid) {
        return Err(StatsError::Unsupported {
            message: "GPDC: a series is constant or fully explained by the conditioning set, so \
                      its distance correlation is undefined (not zero)",
        });
    }
    Ok(())
}

/// Z columns standardised to zero mean and unit SD (row-major `n × |Z|`; a constant column
/// becomes zeros), so the fixed RBF length scale means the same thing at any scale of Z.
pub(super) fn standardized_z(
    columns: &[&[f64]],
    z: &[usize],
    n: usize,
) -> Result<Vec<f64>, StatsError> {
    // Standardise each Z column before the fixed RBF length scale so conditioning
    // does not silently fail when Z is far from unit scale (stats-ci-4).
    let zdim = z.len();
    let mut z_std = vec![0.0; n * zdim];
    for (j, &zc) in z.iter().enumerate() {
        let col = columns
            .get(zc)
            .ok_or(StatsError::Shape { message: "Z column index out of range for GPDC" })?;
        if col.len() < n || col.iter().take(n).any(|v| !v.is_finite()) {
            return Err(StatsError::Shape { message: "non-finite Z in GPDC residualization" });
        }
        let mean_z = col.iter().take(n).sum::<f64>() / n as f64;
        let var_z = col
            .iter()
            .take(n)
            .map(|v| {
                let d = v - mean_z;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let sd = var_z.sqrt();
        if sd.is_finite() && sd > 0.0 {
            for r in 0..n {
                z_std[r * zdim + j] = (col[r] - mean_z) / sd;
            }
        }
    }
    Ok(z_std)
}

/// Cholesky factor of `K + λI` on the standardised Z columns (row-major `n × n`).
pub(super) fn gp_factor(
    columns: &[&[f64]],
    z: &[usize],
    n: usize,
    gp: &Gpdc,
) -> Result<Vec<f64>, StatsError> {
    #[cfg(test)]
    GPDC_FACTORIZATIONS.with(|count| count.set(count.get() + 1));
    let zdim = z.len();
    let z_std = standardized_z(columns, z, n)?;
    // Gram on standardised Z (sum of RBF over Z dims) plus the ridge.
    let mut k = vec![0.0; n * n];
    let ls2 = gp.length_scale * gp.length_scale;
    for i in 0..n {
        for j in 0..=i {
            let mut d2 = 0.0;
            for d in 0..zdim {
                let diff = z_std[i * zdim + d] - z_std[j * zdim + d];
                d2 += diff * diff;
            }
            let kij = (-0.5 * d2 / ls2).exp();
            k[i * n + j] = kij;
            k[j * n + i] = kij;
        }
        k[i * n + i] += gp.ridge;
    }
    crate::gram::cholesky_spd(&k, n)
        .ok_or_else(|| StatsError::Backend("GPDC kernel factorization failed".into()))
}

/// GP residual of `y` given the Cholesky `factor` of `K + λI` (`None`: empty Z).
///
/// With `(K+λI)α = y_c` the mean prediction is `Kα = y_c − λα` (MM-008), so the residual is
/// `y_c − Kα = λα` — one triangular solve, no `O(n²)` matrix-vector product.
pub(super) fn gp_residual(
    y: &[f64],
    factor: Option<&[f64]>,
    ridge: f64,
) -> Result<Vec<f64>, StatsError> {
    let n = y.len();
    if y.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::Shape { message: "non-finite response in GPDC residualization" });
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = y.iter().map(|value| value - mean).collect();
    let Some(chol) = factor else {
        return Ok(centered);
    };
    let alpha = crate::gram::chol_solve(chol, n, &centered)
        .ok_or_else(|| StatsError::Backend("GPDC kernel solve failed".into()))?;
    Ok(alpha.into_iter().map(|a| ridge * a).collect())
}
