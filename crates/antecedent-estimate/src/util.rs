//! Shared estimation helpers (SOLID/DRY).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::{AdaptiveBootstrapBudget, ExecutionContext, RouteBudget};
use antecedent_data::{resample::fill_resample_index_replicate, ResamplingPlan};
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, StatsError, chol_solve, cholesky_spd,
    form_xtx, invert_square,
};

use crate::bootstrap_wave::{cluster_worker_slots, fill_worker_replicate, retained_index_slots};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;

/// Map a stats-layer error into [`EstimationError::Stats`]. A refusal record the stats layer
/// carries (a refused GLM fit, a rank-deficient design, too few clusters) stays in the error and
/// is read back as structured fields by [`EstimationError::refusal_fields`]; the text and the
/// variant are unchanged.
#[allow(clippy::needless_pass_by_value, reason = "statsError is small / owned at call sites")]
pub(crate) fn stats_err(e: StatsError) -> EstimationError {
    EstimationError::from(e)
}

/// Solve `A x = b` for symmetric positive-definite row-major `A` (`p × p`).
#[must_use]
pub(crate) fn solve_spd(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    let chol = cholesky_spd(a, p)?;
    chol_solve(&chol, p, b)
}

/// The 64-bit identity word of a finished canonical stream: the first eight bytes of its
/// BLAKE3 digest, little-endian. For identities that are carried as a `u64` field; anything
/// stored as a whole digest uses [`CanonicalWriter::finish`] directly.
#[must_use]
pub(crate) fn canonical_word(writer: &antecedent_core::CanonicalWriter) -> u64 {
    let digest = writer.finish();
    u64::from_le_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ])
}

/// `n` draws from `N(mean, variance)` on the seeded direct-draw stream of the shared
/// multivariate sampler; `None` when `variance` is not positive and finite (the sampler's
/// Cholesky refuses it) or the draw fails.
#[must_use]
pub(crate) fn normal_draws(mean: f64, variance: f64, n: usize, seed: u64) -> Option<Vec<f64>> {
    if !(variance.is_finite() && variance > 0.0) {
        return None;
    }
    antecedent_prob::sample_gaussian_mvn(
        &[mean],
        &[variance],
        n,
        seed,
        &mut antecedent_prob::LaplaceWorkspace::default(),
    )
    .ok()
    .map(|draws| draws.to_vec())
}

/// `X'X` (row-major `p × p`) of a column-major `n × p` design.
#[must_use]
pub(crate) fn gram(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut xtx = vec![0.0; p * p];
    form_xtx(x, n, p, &mut xtx);
    xtx
}

/// Require [`OverlapPolicy::ExplicitOverride`] (linear / IV / RD / front-door / GLM paths).
pub(crate) fn require_explicit_override(
    overlap: OverlapPolicy,
    message: &'static str,
) -> Result<(), EstimationError> {
    if overlap != OverlapPolicy::ExplicitOverride {
        return Err(EstimationError::Overlap { message });
    }
    Ok(())
}

/// Refuse [`OverlapPolicy::ExplicitOverride`] (propensity / AIPW paths — positivity mandatory).
pub(crate) fn refuse_explicit_override(
    overlap: OverlapPolicy,
    message: &'static str,
) -> Result<(), EstimationError> {
    if matches!(overlap, OverlapPolicy::ExplicitOverride) {
        return Err(EstimationError::Overlap { message });
    }
    Ok(())
}

/// A structured refusal whose message is `"{detail}: {message}"`: `detail` is the namespaced
/// reason literal a caller keys on, `message` the human text.
pub(crate) fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

/// Slack of every correlation-matrix factorization, relative to its unit diagonal: a pivot
/// within it of zero is a rank-deficient direction (perfectly collinear scores, an identical
/// estimand), one below it is an indefinite matrix. Also the tolerance of the unit-diagonal
/// and symmetry checks that precede the factorization.
pub(crate) const CORRELATION_PSD_TOLERANCE: f64 = 1e-9;

/// Two-sided normal p-value `2·(1 − Φ(|z|))` of a standardized statistic; `NaN` propagates.
pub(crate) fn normal_two_sided_p(z: f64) -> f64 {
    2.0 * antecedent_kernels::norm_sf(z.abs())
}

/// A row or parameter count as `f64`.
#[allow(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
pub(crate) const fn count(n: usize) -> f64 {
    n as f64
}

pub(crate) use antecedent_kernels::{dot, mean};

/// Unbiased sample standard deviation; `NaN` if fewer than 2 observations.
pub(crate) use antecedent_kernels::sample_std;

/// Monte Carlo critical value of a max-statistic band: the `⌈level·(B+1)⌉`-th
/// smallest of `B` ascending-sorted replicate maxima.
///
/// The `B + 1` rank treats the observed statistic as one more exchangeable draw
/// (Davison & Hinkley 1997, §4.2), so the band is conservative wherever the rank
/// exists; `⌈level·B⌉` sits one rank lower whenever `level·B` is not an integer.
/// When `⌈level·(B+1)⌉ > B` (`B < level/(1−level)`: 19 at 0.95, 99 at 0.99) the
/// exact critical value is `+∞` — the sample maximum only covers `B/(B+1) < level`
/// — so `+∞` is returned rather than a narrower, anti-conservative maximum. Every
/// simultaneous response band (multiplier, Gaussian max-t and replicate sup-t)
/// reads its critical value here. `NaN` for an empty slice.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the rank is at least 1 and the code below rejects any rank above b"
)]
#[allow(clippy::cast_sign_loss, reason = "the rank is clamped to at least 1 before the cast")]
pub(crate) fn monte_carlo_critical(sorted_maxima: &[f64], level: f64) -> f64 {
    let b = sorted_maxima.len();
    if b == 0 {
        return f64::NAN;
    }
    // The tolerance keeps a product that lands one ulp above an exact integer (e.g.
    // `0.95 · 20`) from bumping the rank past `B`.
    let rank = ((level * (b as f64 + 1.0) - 1e-9).ceil().max(1.0)) as usize;
    if rank > b {
        return f64::INFINITY;
    }
    sorted_maxima[rank - 1]
}

/// How many seeded posterior or resampling draws an estimator takes, and from which seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawSpec {
    /// Number of draws.
    pub draws: usize,
    /// Master seed of the draw stream.
    pub seed: u64,
}

impl DrawSpec {
    /// `draws` draws from `seed`.
    #[must_use]
    pub const fn new(draws: usize, seed: u64) -> Self {
        Self { draws, seed }
    }
}

/// Outcome of an IID bootstrap SE computation with failure accounting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapSeResult {
    /// Sample SD of successful replicate ATEs; `None` if too few survivors or too many failures.
    pub se: Option<f64>,
    /// Replicates that produced a finite estimate.
    pub replicates_ok: u32,
    /// Replicates that soft-failed (skipped) or did not yield an estimate.
    pub replicates_failed: u32,
    /// Cooperative cancellation stopped the loop early.
    pub cancelled: bool,
    /// Adaptive SE early-stop (orthogonal to [`Self::cancelled`]).
    pub early_stopped: bool,
}

impl BootstrapSeResult {
    /// Empty / skipped bootstrap (zero replicates requested).
    #[must_use]
    pub const fn skipped() -> Self {
        Self {
            se: None,
            replicates_ok: 0,
            replicates_failed: 0,
            cancelled: false,
            early_stopped: false,
        }
    }
}

/// Maximum allowed soft-failure fraction before refusing to report a bootstrap
/// summary: more than half the attempted replicates failing means the survivors
/// are not a resample of the estimator.
pub(crate) const BOOTSTRAP_MAX_FAILURE_FRAC: f64 = 0.5;

/// When a set of bootstrap replicates or posterior draws may be summarized.
///
/// One rule owns the decision for every transport interval: the estimator's
/// own failure-fraction refusal, the learned-trial refusal of any failed
/// replicate, and the facade's licensing floor are the same struct with
/// different thresholds, so no path re-derives the arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplicatePolicy {
    /// Fewest requested replicates that can earn the summary's label.
    pub min_requested: u32,
    /// Fewest successful replicates a summary may be computed from.
    pub min_ok: u32,
    /// Largest failed share of the attempted replicates.
    pub max_failure_frac: f64,
    /// Stable reason published when the summary is withheld.
    pub reason: &'static str,
}

impl ReplicatePolicy {
    /// The estimator's own rule: at least two survivors and no more than
    /// [`BOOTSTRAP_MAX_FAILURE_FRAC`] failures.
    pub const BOOTSTRAP: Self = Self {
        min_requested: 0,
        min_ok: 2,
        max_failure_frac: BOOTSTRAP_MAX_FAILURE_FRAC,
        reason: "bootstrap_failure_fraction",
    };
    /// The learned-trial rule: any failed replicate withholds the interval.
    pub const STRICT: Self = Self {
        min_requested: 2,
        min_ok: 2,
        max_failure_frac: 0.0,
        reason: "bootstrap_replicate_failure",
    };

    /// A nominal-coverage licensing floor on requested and successful replicates.
    #[must_use]
    pub const fn percentile_floor(min_replicates: u32) -> Self {
        Self {
            min_requested: min_replicates,
            min_ok: min_replicates,
            max_failure_frac: 1.0,
            reason: "insufficient_bootstrap_replicates",
        }
    }

    /// `Ok` when `ok` successful replicates out of `ok + failed` attempted, from
    /// `requested`, may be summarized; otherwise the withholding reason.
    ///
    /// # Errors
    /// The policy's `reason` when the summary must be withheld.
    pub fn decide(self, requested: u32, ok: u32, failed: u32) -> Result<(), &'static str> {
        let attempted = ok.saturating_add(failed);
        let fail_frac = if attempted == 0 { 1.0 } else { f64::from(failed) / f64::from(attempted) };
        if requested < self.min_requested || ok < self.min_ok || fail_frac > self.max_failure_frac {
            Err(self.reason)
        } else {
            Ok(())
        }
    }
}

/// Finalize a bootstrap SE from successful replicate ATEs.
#[cfg(test)]
#[must_use]
pub(crate) fn finalize_bootstrap_se(ates: &[f64], replicates: u32) -> BootstrapSeResult {
    finalize_bootstrap_se_ex(ates, replicates, false, false)
}

/// Finalize bootstrap SE with explicit cancellation / early-stop flags.
///
/// `attempted` counts only evaluated replicates, so unattempted draws are never
/// failures, while genuine failures remain visible even after cancellation.
#[must_use]
pub(crate) fn finalize_bootstrap_se_ex(
    ates: &[f64],
    attempted: u32,
    cancelled: bool,
    early_stopped: bool,
) -> BootstrapSeResult {
    let ok = u32::try_from(ates.len()).unwrap_or(u32::MAX);
    let failed = attempted.saturating_sub(ok);
    let too_few = ates.len() < 2;
    let fail_frac = if attempted == 0 { 0.0 } else { f64::from(failed) / f64::from(attempted) };
    let se = if too_few || fail_frac > BOOTSTRAP_MAX_FAILURE_FRAC {
        None
    } else {
        let s = sample_std(ates);
        if s.is_finite() { Some(s) } else { None }
    };
    BootstrapSeResult { se, replicates_ok: ok, replicates_failed: failed, cancelled, early_stopped }
}

/// Whether an enabled bootstrap budget may stop after `successful` replicates.
///
/// The stop is a replicate floor derived from the Monte Carlo error of a
/// bootstrap SE ([`AdaptiveBootstrapBudget::required_replicates`]); it does not
/// look at the SE trajectory, so the count is pinned by the budget alone.
#[must_use]
pub(crate) fn adaptive_bootstrap_should_stop(
    budget: AdaptiveBootstrapBudget,
    successful: u32,
) -> bool {
    budget.enabled && successful >= budget.required_replicates()
}

/// IID bootstrap standard error with failure accounting and optional early-stop.
///
/// Each replicate draws `n` row indexes from the resample stream keyed by
/// `stream_base ^ replicate_id`, so the draw is fixed by the seed and that id even when
/// evaluation stops early. Workers write those indexes into a reused buffer of length
/// `n`; an `n` by replicates index matrix is not retained.
///
/// `estimate` should return `Ok(Some(ate))` on success, `Ok(None)` for a soft-failed replicate
/// (counted as failed, bootstrap continues), or `Err` to abort the whole bootstrap.
///
/// Cooperative cancellation: when `ctx.cancellation` trips mid-loop, returns a partial result
/// with `cancelled = true` rather than inventing a full run.
///
/// Early-stop: when [`ExecutionContext::adaptive_bootstrap`] is enabled (no constructor
/// enables it) and the successful count reaches the budget's Monte Carlo error floor
/// before the request is exhausted, returns with `early_stopped = true` and the actual
/// count. Otherwise every requested replicate is evaluated.
pub(crate) fn bootstrap_se(
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    estimate: impl Fn(&[usize]) -> Result<Option<f64>, EstimationError> + Sync,
) -> Result<BootstrapSeResult, EstimationError> {
    bootstrap_se_with_scratch(replicates, ctx, stream_base, n, || (), |(), idx| estimate(idx))
}

/// [`bootstrap_se`] with per-worker scratch so replicate evaluation can run
/// under [`ExecutionContext::parallelism`] without a shared `&mut` workspace.
///
/// Retained index memory is `n` times the worker count (one buffer on the serial and
/// adaptive paths). Adaptive early-stop stays serial (the floor is a running success
/// count). When an outer map already saturates the thread budget, `ctx` is serial and
/// this path does not spawn.
pub(crate) fn bootstrap_se_with_scratch<S, F>(
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    make_scratch: impl Fn() -> S + Sync,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    S: Send,
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError> + Sync,
{
    if replicates == 0 || n == 0 {
        return Ok(BootstrapSeResult::skipped());
    }
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let n_rep = replicates as usize;
    let threads = (ctx.parallelism.max_threads.get() as usize).clamp(1, n_rep);
    let parallel = threads > 1 && !ctx.adaptive_bootstrap.enabled;
    // One length-`n` buffer per worker. Adaptive early-stop does not spawn.
    let workers = if parallel { threads } else { 1 };
    retained_index_slots(n, workers)?;
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let rows = ReplicateRows::Rows { n, ctx, stream_base };
    if parallel {
        return evaluate_bootstrap_parallel(ctx, &rows, n_rep, make_scratch, estimate);
    }
    evaluate_bootstrap_serial(ctx, &rows, n_rep, make_scratch, estimate)
}

/// The rows of each cluster, in row order, for whole-cluster resampling.
#[derive(Clone, Debug)]
pub(crate) struct ClusterRows {
    members: Vec<Vec<antecedent_data::GlobalId>>,
    rows: usize,
}

impl ClusterRows {
    /// Group row indices by cluster label (clusters in order of first appearance).
    ///
    /// # Errors
    ///
    /// Fewer than two clusters (a one-cluster resample is the data), or a row that does
    /// not fit in a [`antecedent_data::GlobalId`].
    pub(crate) fn new(labels: &[u32]) -> Result<Self, EstimationError> {
        let rows = labels.len();
        let mut slot_of: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut members: Vec<Vec<antecedent_data::GlobalId>> = Vec::new();
        for (row, label) in labels.iter().enumerate() {
            let row = antecedent_data::GlobalId::new(u64::try_from(row).map_err(|_| {
                EstimationError::data_msg(
                    "row position does not fit in u64; rows are addressed by GlobalId and ChunkIndex only spans one bounded chunk",
                )
            })?);
            let next = members.len();
            let slot = *slot_of.entry(*label).or_insert(next);
            if slot == members.len() {
                members.push(Vec::new());
            }
            members[slot].push(row);
        }
        if members.len() < 2 {
            return Err(crate::se::few_clusters(members.len()));
        }
        Ok(Self { members, rows })
    }

    /// Rows in the sample these clusters partition.
    pub(crate) fn row_count(&self) -> usize {
        self.rows
    }

    /// Number of clusters.
    pub(crate) fn len(&self) -> usize {
        self.members.len()
    }

    /// The rows of each cluster, in row order.
    pub(crate) fn members(&self) -> &[Vec<antecedent_data::GlobalId>] {
        &self.members
    }
}

/// Whole-cluster bootstrap standard error with failure accounting.
///
/// Each replicate draws as many clusters as the data has, with replacement, and presents the
/// union of their rows (clusters in draw order, rows in data order) to `estimate`. A
/// replicate therefore has a *variable* number of rows: a cluster drawn twice contributes
/// twice, one not drawn not at all. This is the resampling a cluster-robust variance
/// assumes, with no row ever split from its cluster; a fixed-length row index would break
/// the clusters or the sample size.
///
/// Otherwise behaves as [`bootstrap_se_with_scratch`]: soft failures, cancellation, and
/// parallel evaluation are the same, and the draws are common-random-number stable under a
/// fixed seed.
///
/// Each worker writes one replicate. Retained index memory is `n` times the worker count.
/// A groups-by-replicates matrix is admitted against the route budget before any index
/// buffer is allocated, and that matrix is not retained.
///
/// # Errors
///
/// The route budget refuses the worker buffer, the matrix size does not fit in `u64`,
/// or an `Err` from `estimate`.
pub(crate) fn bootstrap_se_clusters_with_scratch<S, F>(
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    clusters: &ClusterRows,
    make_scratch: impl Fn() -> S + Sync,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    S: Send,
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError> + Sync,
{
    let groups = clusters.len();
    if replicates == 0 || groups == 0 {
        return Ok(BootstrapSeResult::skipped());
    }
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let n_rep = replicates as usize;
    let threads = (ctx.parallelism.max_threads.get() as usize).clamp(1, n_rep);
    let parallel = threads > 1 && !ctx.adaptive_bootstrap.enabled;
    let workers = if parallel { threads } else { 1 };
    let limit = ctx.memory.hard_limit_bytes.unwrap_or(u64::MAX);
    // Judge the groups-by-replicates matrix, then keep one replicate per worker.
    cluster_worker_slots(clusters.row_count(), workers, groups, n_rep, RouteBudget::new(limit))?;
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let rows = ReplicateRows::Clusters { members: clusters.members(), ctx, stream_base };
    if parallel {
        return evaluate_bootstrap_parallel(ctx, &rows, n_rep, make_scratch, estimate);
    }
    evaluate_bootstrap_serial(ctx, &rows, n_rep, make_scratch, estimate)
}

/// How one bootstrap replicate's row indices are supplied to the estimator.
enum ReplicateRows<'a> {
    /// `n` iid row draws for replicate `r`, generated from `stream_base ^ r`.
    ///
    /// Each worker fills its own length-`n` buffer. The `n` by replicates matrix is not
    /// retained.
    Rows { n: usize, ctx: &'a ExecutionContext, stream_base: u64 },
    /// One whole-cluster replicate, drawn into the caller buffer. The groups-by-replicates
    /// matrix is not retained. Length varies with the clusters drawn.
    Clusters {
        members: &'a [Vec<antecedent_data::GlobalId>],
        ctx: &'a ExecutionContext,
        stream_base: u64,
    },
}

impl ReplicateRows<'_> {
    fn fill(&self, r: usize, out: &mut Vec<usize>) -> Result<(), EstimationError> {
        out.clear();
        let row = |id: antecedent_data::GlobalId| -> Result<usize, EstimationError> {
            usize::try_from(id.get()).map_err(|_| {
                EstimationError::data_msg(
                    "row position does not fit in u64; rows are addressed by GlobalId and ChunkIndex only spans one bounded chunk",
                )
            })
        };
        match self {
            Self::Rows { n, ctx, stream_base } => {
                let replicate = u64::try_from(r)
                    .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds u64"))?;
                fill_worker_replicate(*n, replicate, ctx, *stream_base, out)?;
            }
            Self::Clusters { members, ctx, stream_base } => {
                let groups = members.len();
                let replicate = u64::try_from(r).map_err(|_| {
                    EstimationError::data_msg("bootstrap replicate id exceeds u64")
                })?;
                let mut drawn = Vec::new();
                fill_resample_index_replicate(
                    ResamplingPlan::IidBootstrap,
                    groups,
                    None,
                    ctx,
                    *stream_base,
                    replicate,
                    &mut drawn,
                )?;
                for cluster in &drawn {
                    let slot = row(*cluster)?;
                    let Some(sources) = members.get(slot) else {
                        return Err(EstimationError::data_msg(
                            "cluster draw is outside the observed clusters",
                        ));
                    };
                    for src in sources {
                        out.push(row(*src)?);
                    }
                }
            }
        }
        Ok(())
    }
}

fn evaluate_bootstrap_serial<S, F>(
    ctx: &ExecutionContext,
    rows: &ReplicateRows<'_>,
    n_rep: usize,
    make_scratch: impl Fn() -> S,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError>,
{
    let mut scratch = make_scratch();
    let mut ates = Vec::with_capacity(n_rep);
    let mut idx = Vec::new();
    let mut cancelled = false;
    let mut early_stopped = false;
    let mut attempted = 0u32;
    let budget = ctx.adaptive_bootstrap;
    for r in 0..n_rep {
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        rows.fill(r, &mut idx)?;
        attempted += 1;
        let estimate = estimate(&mut scratch, &idx)?.filter(|ate| ate.is_finite());
        if let Some(ate) = estimate {
            ates.push(ate);
        }
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        if estimate.is_some() && r + 1 < n_rep {
            let ok_count = u32::try_from(ates.len()).unwrap_or(u32::MAX);
            if adaptive_bootstrap_should_stop(budget, ok_count) {
                early_stopped = true;
                if let Some(p) = &ctx.progress {
                    p.report((r + 1) as f64 / n_rep as f64, "bootstrap");
                }
                break;
            }
        }
        if let Some(p) = &ctx.progress {
            p.report((r + 1) as f64 / n_rep as f64, "bootstrap");
        }
    }
    Ok(finalize_bootstrap_se_ex(&ates, attempted, cancelled, early_stopped))
}

fn evaluate_bootstrap_parallel<S, F>(
    ctx: &ExecutionContext,
    rows: &ReplicateRows<'_>,
    n_rep: usize,
    make_scratch: impl Fn() -> S + Sync,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    S: Send,
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError> + Sync,
{
    let threads = (ctx.parallelism.max_threads.get() as usize).clamp(1, n_rep);
    let mut slots: Vec<Option<Result<Option<f64>, EstimationError>>> =
        (0..n_rep).map(|_| None).collect();
    std::thread::scope(|scope| {
        let estimate = &estimate;
        let make_scratch = &make_scratch;
        let mut rest = slots.as_mut_slice();
        let mut start = 0usize;
        for t in 0..threads {
            let take = rest.len().div_ceil(threads - t);
            let (mine, next) = rest.split_at_mut(take);
            let begin = start;
            scope.spawn(move || {
                let mut scratch = make_scratch();
                let mut idx = Vec::new();
                for (k, slot) in mine.iter_mut().enumerate() {
                    let r = begin + k;
                    if ctx.cancellation.is_cancelled() {
                        break;
                    }
                    match rows.fill(r, &mut idx) {
                        Ok(()) => {
                            *slot = Some(
                                estimate(&mut scratch, &idx)
                                    .map(|v| v.filter(|ate| ate.is_finite())),
                            );
                        }
                        Err(error) => *slot = Some(Err(error)),
                    }
                }
            });
            rest = next;
            start += take;
            if rest.is_empty() {
                break;
            }
        }
    });
    let mut ates = Vec::with_capacity(n_rep);
    let mut attempted = 0u32;
    let mut cancelled = ctx.cancellation.is_cancelled();
    for slot in slots {
        match slot {
            None => {
                cancelled = true;
                break;
            }
            Some(Err(error)) => return Err(error),
            Some(Ok(value)) => {
                attempted += 1;
                if let Some(ate) = value {
                    ates.push(ate);
                }
            }
        }
    }
    if let Some(p) = &ctx.progress {
        p.report(1.0, "bootstrap");
    }
    Ok(finalize_bootstrap_se_ex(&ates, attempted, cancelled, false))
}

/// Coefficients and residuals of one `FaerBackend` least-squares solve.
pub(crate) struct LsFit {
    /// Coefficient vector, length `ncols`.
    pub coefficients: Vec<f64>,
    /// Residuals, length `nrows`.
    pub residuals: Vec<f64>,
    /// Numerical rank.
    pub rank: usize,
}

/// Solve `min ||X β − y||` on a column-major design.
///
/// The coefficients, residuals, and rank are the backend fit's fields.
///
/// # Errors
///
/// A stats-layer shape or numerical failure.
pub(crate) fn ls_fit(
    backend: FaerBackend,
    design_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    workspace: &mut LeastSquaresWorkspace,
) -> Result<LsFit, EstimationError> {
    let fit =
        backend.least_squares(design_colmajor, nrows, ncols, y, workspace).map_err(stats_err)?;
    Ok(LsFit { coefficients: fit.coefficients, residuals: fit.residuals, rank: fit.rank })
}

/// OLS residual variance `σ² = RSS / (n − p)` for a fitted coefficient vector.
pub(crate) fn ols_sigma2(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    beta: &[f64],
) -> f64 {
    let mut rss = 0.0;
    for r in 0..nrows {
        let mut pred = 0.0;
        for c in 0..ncols {
            pred += x_colmajor[c * nrows + r] * beta[c];
        }
        let e = y[r] - pred;
        rss += e * e;
    }
    rss / (nrows.saturating_sub(ncols)).max(1) as f64
}

/// Variance of a single OLS coefficient: `σ² · [(XᵀX)⁻¹]_{jj}`.
///
/// Returns `NaN` if `XᵀX` is singular.
pub(crate) fn coefficient_variance(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    col: usize,
    sigma2: f64,
) -> f64 {
    let mut xtx = vec![0.0; ncols * ncols];
    form_xtx(x_colmajor, nrows, ncols, &mut xtx);
    let Some(inv) = invert_square(&xtx, ncols) else {
        return f64::NAN;
    };
    sigma2 * inv[col * ncols + col].max(0.0)
}

/// Compute `(XᵀX)⁻¹` for a column-major `nrows × ncols` design, or `None` if singular.
///
/// Shared `form_xtx` → `invert_square` idiom used by every analytic-SE helper that needs the
/// unscaled inverse-information matrix.
pub(crate) fn xtx_inverse(x_colmajor: &[f64], nrows: usize, ncols: usize) -> Option<Vec<f64>> {
    let mut xtx = vec![0.0; ncols * ncols];
    form_xtx(x_colmajor, nrows, ncols, &mut xtx);
    invert_square(&xtx, ncols)
}

/// Invert a small fixed-size matrix with [`invert_square`]'s scale-free rule, or `None` when it
/// is singular or non-finite.
///
/// The one inverse for the Bayesian IV / RD and local-polynomial families, which carry their
/// moment matrices as `[[f64; N]; N]`.
#[must_use]
pub(crate) fn invert_fixed<const N: usize>(a: &[[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let flat: Vec<f64> = a.iter().flatten().copied().collect();
    let inv = invert_square(&flat, N)?;
    Some(std::array::from_fn(|i| std::array::from_fn(|j| inv[i * N + j])))
}

/// [`invert_fixed`] for a runtime-sized square matrix held as rows; `None` when the rows are not
/// square, or the matrix is singular or non-finite.
#[must_use]
pub(crate) fn invert_rows(rows: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = rows.len();
    if rows.iter().any(|r| r.len() != n) {
        return None;
    }
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    let inv = invert_square(&flat, n)?;
    Some(inv.chunks(n).map(<[f64]>::to_vec).collect())
}

/// Gather `idx`-selected entries of `src` into `out` (a reused scratch buffer) for one IID
/// bootstrap resample: `out[r] = src[idx[r]]`.
pub(crate) fn gather_bootstrap_vector(out: &mut [f64], src: &[f64], idx: &[usize]) {
    for (r, &i) in idx.iter().enumerate() {
        out[r] = src[i];
    }
}

/// Gather `idx`-selected rows of a column-major `n × ncols` matrix `src` into `out` (a reused
/// scratch buffer) for one IID bootstrap resample: `out[c*n+r] = src[c*n+idx[r]]`.
pub(crate) fn gather_bootstrap_design(
    out: &mut [f64],
    src: &[f64],
    n: usize,
    ncols: usize,
    idx: &[usize],
) {
    for (r, &i) in idx.iter().enumerate() {
        for c in 0..ncols {
            out[c * n + r] = src[c * n + i];
        }
    }
}

/// Gather `idx`-selected rows of a column-major matrix `src` of `src_rows` rows into `out`,
/// whose row count is `idx.len()`: `out[c·m + r] = src[c·src_rows + idx[r]]` with
/// `m = idx.len()`. For a resample as long as the data this is [`gather_bootstrap_design`];
/// a whole-cluster resample has its own length.
pub(crate) fn gather_resampled_design(
    out: &mut [f64],
    src: &[f64],
    src_rows: usize,
    ncols: usize,
    idx: &[usize],
) {
    let m = idx.len();
    for (r, &i) in idx.iter().enumerate() {
        for c in 0..ncols {
            out[c * m + r] = src[c * src_rows + i];
        }
    }
}

/// Delta-method SE for a linear contrast `gᵀ β`: `sqrt(σ² · gᵀ (XᵀX)⁻¹ g)`.
///
/// Returns `NaN` if the Gram matrix is singular or the quadratic form is non-finite.
pub(crate) fn delta_method_se(inv_xtx: &[f64], ncols: usize, g: &[f64], sigma2: f64) -> f64 {
    if g.len() != ncols || inv_xtx.len() != ncols * ncols {
        return f64::NAN;
    }
    // v = inv · g
    let mut v = vec![0.0; ncols];
    for i in 0..ncols {
        let mut s = 0.0;
        for j in 0..ncols {
            s += inv_xtx[i * ncols + j] * g[j];
        }
        v[i] = s;
    }
    let mut q = 0.0;
    for i in 0..ncols {
        q += g[i] * v[i];
    }
    let var = sigma2 * q;
    if !var.is_finite() {
        return f64::NAN;
    }
    var.max(0.0).sqrt()
}

/// Single-pass `(min, max)` fold.
///
/// Shared by the static and temporal response estimators so neither carries its own
/// min/max pair (two full-slice folds where one suffices).
pub(crate) fn range(values: &[f64]) -> (f64, f64) {
    values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    use antecedent_core::{AdaptiveBootstrapBudget, CancellationToken, ExecutionContext};

    #[test]
    fn fixed_inverse_judges_singularity_scale_free() {
        // A design in units of 1e-8 has pivots far below the old absolute 1e-12 cutoff but is
        // perfectly conditioned once equilibrated.
        let tiny = [[1e-16, 0.0, 0.0], [0.0, 4e-16, 0.0], [0.0, 0.0, 9e-16]];
        let inv = invert_fixed(&tiny).expect("a scaled diagonal is not singular");
        assert!((inv[1][1] * 4e-16 - 1.0).abs() < 1e-12);
        // A genuinely collinear pair is still refused, at any scale.
        let collinear = [[1.0, 2.0], [2.0, 4.0]];
        assert!(invert_fixed(&collinear).is_none());
        let scaled = [[1e12, 2e12], [2e12, 4e12]];
        assert!(invert_fixed(&scaled).is_none());
        // Non-finite entries are refused rather than propagated.
        assert!(invert_fixed(&[[f64::NAN, 0.0], [0.0, 1.0]]).is_none());
    }

    #[test]
    fn row_inverse_matches_the_fixed_inverse_and_refuses_ragged_input() {
        let rows = vec![vec![1e-16, 0.0], vec![0.0, 4e-16]];
        let inv = invert_rows(&rows).expect("a scaled diagonal is not singular");
        assert!((inv[0][0] * 1e-16 - 1.0).abs() < 1e-12 && inv[0][1] == 0.0);
        assert!(invert_rows(&[vec![1.0, 2.0], vec![2.0, 4.0]]).is_none());
        assert!(invert_rows(&[vec![1.0, 0.0], vec![0.0]]).is_none());
    }

    #[test]
    fn cancelled_bootstrap_retains_actual_failures() {
        let ctx = ExecutionContext::for_tests(7);
        let calls = AtomicU32::new(0);
        let result = bootstrap_se(20, &ctx, 1, 10, |_| {
            let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
            if n == 4 {
                ctx.cancellation.cancel();
            }
            Ok(if n <= 2 { Some(f64::from(n)) } else { None })
        })
        .unwrap();
        assert!(result.cancelled);
        assert!(!result.early_stopped);
        assert_eq!(result.replicates_ok, 2);
        assert_eq!(result.replicates_failed, 2);
        assert!(result.se.is_some());
    }

    #[test]
    fn early_stopped_bootstrap_retains_actual_failures() {
        let mut ctx = ExecutionContext::for_tests(7);
        // ε = 0.5 → floor ⌈1 + 1/(2·0.25)⌉ = 3 successes.
        ctx.adaptive_bootstrap =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 2, se_rel_epsilon: 0.5 };
        let calls = AtomicU32::new(0);
        let result = bootstrap_se(20, &ctx, 1, 10, |_| {
            let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
            Ok(if n <= 4 { None } else { Some(1.0) })
        })
        .unwrap();
        assert!(result.early_stopped);
        assert_eq!(result.replicates_ok, 3);
        assert_eq!(result.replicates_failed, 4);
        assert!(result.se.is_none(), "majority failure must not be hidden by early stopping");
    }

    #[test]
    fn bootstrap_nonfinite_estimates_are_failures() {
        let ctx = ExecutionContext::for_tests(7);
        let values = [Some(1.0), Some(f64::NAN), Some(f64::INFINITY), Some(2.0)];
        let i = AtomicU32::new(0);
        let result = bootstrap_se(4, &ctx, 1, 5, |_| {
            let value = values[i.fetch_add(1, Ordering::Relaxed) as usize];
            Ok(value)
        })
        .unwrap();
        assert_eq!(result.replicates_ok, 2);
        assert_eq!(result.replicates_failed, 2);
        assert!(result.se.is_some());
    }

    #[test]
    fn bootstrap_cancellation_does_not_hide_unrelated_estimation_error() {
        let ctx = ExecutionContext::for_tests(7);
        let error = EstimationError::data_msg("unrelated malformed data");
        let result = bootstrap_se(4, &ctx, 1, 5, |_| {
            ctx.cancellation.cancel();
            Err(error.clone())
        });
        assert_eq!(result.unwrap_err(), error);
    }

    #[test]
    fn bootstrap_checks_capacity_before_allocation() {
        let ctx = ExecutionContext::for_tests(7);
        let result = bootstrap_se(2, &ctx, 1, usize::MAX, |_| panic!("must not estimate"));
        assert!(result.unwrap_err().to_string().contains("capacity"));
        ctx.cancellation.cancel();
        let cancelled =
            bootstrap_se(2, &ctx, 1, usize::MAX, |_| panic!("must not estimate")).unwrap();
        assert!(cancelled.cancelled);
        assert_eq!(cancelled.replicates_failed, 0);
    }

    #[test]
    fn finalize_refuses_se_when_too_few_or_too_many_failures() {
        assert!(finalize_bootstrap_se(&[], 10).se.is_none());
        assert!(finalize_bootstrap_se(&[1.0], 10).se.is_none());
        let many_fail = finalize_bootstrap_se(&[1.0, 1.1], 10);
        assert_eq!(many_fail.replicates_ok, 2);
        assert_eq!(many_fail.replicates_failed, 8);
        assert!(many_fail.se.is_none(), "80% failure must refuse SE");
        let ok = finalize_bootstrap_se(&[1.0, 1.2, 0.9, 1.1], 4);
        assert!(ok.se.is_some());
        assert_eq!(ok.replicates_failed, 0);
        assert!(!ok.early_stopped);
    }

    #[test]
    fn finalize_early_stop_does_not_count_unattempted_as_failures() {
        let r = finalize_bootstrap_se_ex(&[1.0, 1.1, 0.9, 1.05], 4, false, true);
        assert_eq!(r.replicates_ok, 4);
        assert_eq!(r.replicates_failed, 0);
        assert!(r.early_stopped);
        assert!(r.se.is_some());
    }

    #[test]
    fn adaptive_stop_is_the_monte_carlo_error_floor() {
        // ε = 5% needs 201 successes; the explicit floor of 4 is below that.
        let budget =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 4, se_rel_epsilon: 0.05 };
        assert!(!adaptive_bootstrap_should_stop(budget, 4));
        assert!(!adaptive_bootstrap_should_stop(budget, 200));
        assert!(adaptive_bootstrap_should_stop(budget, 201));
        // A loose ε defers to the explicit floor.
        let floored =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 10, se_rel_epsilon: 0.5 };
        assert!(!adaptive_bootstrap_should_stop(floored, 9));
        assert!(adaptive_bootstrap_should_stop(floored, 10));
        assert!(!adaptive_bootstrap_should_stop(AdaptiveBootstrapBudget::disabled(), u32::MAX));
    }

    #[test]
    fn bootstrap_early_stop_count_is_pinned_by_the_budget() {
        let mut ctx = ExecutionContext::for_tests(42);
        // ε = 0.2 → ⌈1 + 1/(2·0.04)⌉ = 14 successes.
        ctx.adaptive_bootstrap =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 5, se_rel_epsilon: 0.2 };
        // A constant ATE would have tripped the old relative-change rule at
        // once; the floor ignores the trajectory.
        let r1 = bootstrap_se(80, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        let r2 = bootstrap_se(80, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        assert!(r1.early_stopped);
        assert_eq!(r1.replicates_ok, 14);
        assert_eq!(r1.replicates_ok, r2.replicates_ok);
        assert!(!r1.cancelled);
        // A request at or below the floor is evaluated in full, not "stopped".
        let exact = bootstrap_se(14, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        assert!(!exact.early_stopped);
        assert_eq!(exact.replicates_ok, 14);
        let below = bootstrap_se(10, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        assert!(!below.early_stopped);
        assert_eq!(below.replicates_ok, 10);
    }

    #[test]
    fn enabled_default_budget_never_stops_below_its_floor() {
        let mut ctx = ExecutionContext::for_tests(42);
        ctx.adaptive_bootstrap = AdaptiveBootstrapBudget::enabled_default();
        let r = bootstrap_se(199, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        assert!(!r.early_stopped);
        assert_eq!(r.replicates_ok, 199);
        let r = bootstrap_se(400, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        assert!(r.early_stopped);
        assert_eq!(r.replicates_ok, 201);
    }

    #[test]
    fn bootstrap_cancel_and_early_stop_are_independent() {
        let mut ctx = ExecutionContext::for_tests(7);
        ctx.adaptive_bootstrap = AdaptiveBootstrapBudget::disabled();
        let token = CancellationToken::new();
        token.cancel();
        ctx.cancellation = token;
        let r = bootstrap_se(40, &ctx, 0x1111, 10, |_| Ok(Some(1.0))).unwrap();
        assert!(r.cancelled);
        assert!(!r.early_stopped);
        assert_eq!(r.replicates_ok, 0);
    }

    #[test]
    fn monte_carlo_critical_is_infinite_when_the_exact_rank_does_not_exist() {
        let maxima: Vec<f64> = (1..=10).map(f64::from).collect();
        // ⌈0.95·11⌉ = 11 > 10: the sample maximum only covers 10/11 < 0.95.
        assert!(monte_carlo_critical(&maxima, 0.95).is_infinite());
        // B = 19: ⌈0.95·20⌉ = 19 ≤ 19, so the maximum is the exact critical value.
        let nineteen: Vec<f64> = (1..=19).map(f64::from).collect();
        assert!((monte_carlo_critical(&nineteen, 0.95) - 19.0).abs() < 1e-12);
        // B = 99 at 0.99: ⌈0.99·100⌉ = 99.
        let ninety_nine: Vec<f64> = (1..=99).map(f64::from).collect();
        assert!((monte_carlo_critical(&ninety_nine, 0.99) - 99.0).abs() < 1e-12);
        let ninety_eight: Vec<f64> = (1..=98).map(f64::from).collect();
        assert!(monte_carlo_critical(&ninety_eight, 0.99).is_infinite());
        // Interior rank: B = 100, ⌈0.95·101⌉ = 96.
        let hundred: Vec<f64> = (1..=100).map(f64::from).collect();
        assert!((monte_carlo_critical(&hundred, 0.95) - 96.0).abs() < 1e-12);
        assert!(monte_carlo_critical(&[], 0.95).is_nan());
    }
    #[test]
    fn cluster_bootstrap_replicates_are_unions_of_whole_clusters_with_variable_length() {
        // Clusters of 1, 2, 3 and 5 rows, interleaved in the data.
        let labels = [0_u32, 1, 2, 3, 1, 2, 3, 2, 3, 3, 3];
        let clusters = ClusterRows::new(&labels).unwrap();
        assert_eq!(clusters.len(), 4);
        let sizes = [1_usize, 2, 3, 5];
        let lengths = std::sync::Mutex::new(std::collections::BTreeSet::new());
        let ctx = ExecutionContext::for_tests(11);
        let run = || {
            bootstrap_se_clusters_with_scratch(
                200,
                &ctx,
                9,
                &clusters,
                || (),
                |(), rows| {
                    // Every drawn cluster is present in full, as many times as it was drawn.
                    let mut seen = [0_usize; 4];
                    for &row in rows {
                        seen[labels[row] as usize] += 1;
                    }
                    let mut draws = 0;
                    for (label, count) in seen.iter().enumerate() {
                        assert_eq!(count % sizes[label], 0, "cluster {label} was split");
                        draws += count / sizes[label];
                    }
                    assert_eq!(draws, 4, "a replicate draws as many clusters as the data has");
                    lengths.lock().unwrap().insert(rows.len());
                    Ok(Some(rows.len() as f64))
                },
            )
            .unwrap()
        };
        let first = run();
        assert_eq!(first.replicates_ok, 200);
        assert!(lengths.lock().unwrap().len() > 3, "replicate lengths must vary");
        // Same seed, same draws.
        assert_eq!(run().se, first.se);
        // One cluster is not resamplable.
        assert!(ClusterRows::new(&[7, 7, 7]).is_err());
    }

    #[test]
    fn cluster_bootstrap_writes_one_replicate_per_worker() {
        let labels = [0_u32, 1, 2, 3, 1, 2, 3, 2, 3, 3, 3];
        let clusters = ClusterRows::new(&labels).unwrap();
        let n = clusters.row_count();
        let groups = clusters.len();
        let replicates = 80u32;
        let workers = 4usize;
        let matrix_bytes = groups * usize::try_from(replicates).unwrap() * size_of::<u64>();
        let retained_bytes = n * workers * size_of::<u64>();
        assert!(retained_bytes < matrix_bytes);
        let mut ctx = ExecutionContext::production(11, u32::try_from(workers).unwrap());
        ctx.memory.hard_limit_bytes = Some(u64::try_from(retained_bytes).unwrap());
        let calls = AtomicU32::new(0);
        let result = bootstrap_se_clusters_with_scratch(
            replicates,
            &ctx,
            9,
            &clusters,
            || (),
            |(), rows| {
                calls.fetch_add(1, Ordering::Relaxed);
                assert!(!rows.is_empty());
                Ok(Some(rows.len() as f64))
            },
        )
        .unwrap();
        assert_eq!(result.replicates_ok, replicates);
        assert_eq!(calls.load(Ordering::Relaxed), replicates);
        let mut refused = ctx.clone();
        refused.memory.hard_limit_bytes = Some(u64::try_from(retained_bytes - 1).unwrap());
        let refused_calls = AtomicU32::new(0);
        let error = bootstrap_se_clusters_with_scratch(
            replicates,
            &refused,
            9,
            &clusters,
            || (),
            |(), _rows| {
                refused_calls.fetch_add(1, Ordering::Relaxed);
                Ok(Some(1.0_f64))
            },
        )
        .unwrap_err();
        assert_eq!(refused_calls.load(Ordering::Relaxed), 0);
        assert!(error.to_string().contains("route budget"), "{error}");
    }
}
