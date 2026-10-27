//! The replicate rows of a bootstrap and the loops that evaluate them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{DeclaredSamplingDesign, ExecutionContext, ResamplingRule};
use antecedent_data::{ResamplingPlan, resample::fill_resample_index_replicate};

use super::{
    BootstrapSeResult, adaptive_bootstrap_should_stop, design_err, finalize_bootstrap_se_ex,
};
use crate::adaptive_resample_waves::{DriveError, StopReason, WaveStoppingRule, run_waves_serial};
use crate::bootstrap_wave::{fill_worker_replicate, retained_index_slots};
use crate::error::EstimationError;
use crate::util::scratch::admit_bootstrap_scratch;

/// The rows of each cluster, in ascending label order, for whole-cluster resampling.
#[derive(Clone, Debug)]
pub(crate) struct ClusterRows {
    members: Vec<Vec<antecedent_data::GlobalId>>,
    rows: usize,
    largest: usize,
}

impl ClusterRows {
    /// Group row indices by cluster label, clusters in ascending label order.
    ///
    /// That is the order [`antecedent_data`] uses for [`antecedent_data::ResamplingPlan::ClusterBootstrap`].
    /// First appearance is a different order when a later label is smaller, so the same stream
    /// would expand different clusters.
    ///
    /// # Errors
    ///
    /// Fewer than two clusters (a one-cluster resample is the data), or a row that does
    /// not fit in a [`antecedent_data::GlobalId`].
    pub(crate) fn new(labels: &[u32]) -> Result<Self, EstimationError> {
        let rows = labels.len();
        let mut grouped: std::collections::BTreeMap<u32, Vec<antecedent_data::GlobalId>> =
            std::collections::BTreeMap::new();
        for (row, label) in labels.iter().enumerate() {
            let row = antecedent_data::GlobalId::new(u64::try_from(row).map_err(|_| {
                EstimationError::data_msg(
                    "row position does not fit in u64; rows are addressed by GlobalId and ChunkIndex only spans one bounded chunk",
                )
            })?);
            grouped.entry(*label).or_default().push(row);
        }
        if grouped.len() < 2 {
            return Err(crate::se::few_clusters(grouped.len()));
        }
        let members: Vec<Vec<antecedent_data::GlobalId>> = grouped.into_values().collect();
        let largest = members.iter().map(Vec::len).max().unwrap_or(0);
        Ok(Self { members, rows, largest })
    }

    /// Rows in the sample these clusters partition.
    #[cfg(test)]
    pub(crate) fn row_count(&self) -> usize {
        self.rows
    }

    /// Rows one replicate may hold, which admission charges and the draw enforces.
    ///
    /// A replicate is a sum of `groups` independent cluster sizes in `[0, largest]` with mean
    /// `rows`, so by Hoeffding's inequality it exceeds `rows + t` with probability at most
    /// `exp(-2 t² / (groups · largest²))`. With `t = 4 · largest · √groups` that is below
    /// `1e-13`, and the sure ceiling `groups · largest` caps it when there are few clusters.
    pub(crate) fn replicate_row_ceiling(&self) -> usize {
        let groups = self.members.len();
        let sure = groups.saturating_mul(self.largest);
        let slack = groups.isqrt().saturating_add(1).saturating_mul(4).saturating_mul(self.largest);
        sure.min(self.rows.saturating_add(slack))
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
/// Each worker writes one replicate. Retained index memory is the replicate row ceiling
/// ([`ClusterRows::replicate_row_ceiling`]) times the worker count, admitted before any draw
/// and enforced as each cluster is appended. The groups-by-replicates matrix is not allocated.
///
/// # Errors
///
/// The admission pool cannot fund that worker buffer, the buffer is not addressable,
/// or an `Err` from `estimate`.
#[cfg(test)]
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
    bootstrap_se_clusters_under(
        DeclaredSamplingDesign::ClusteredPanel,
        replicates,
        ctx,
        stream_base,
        clusters,
        make_scratch,
        estimate,
    )
}

/// Whole-cluster bootstrap under the `declared` sampling design. The design refuses the
/// draw when it does not bind whole-cluster resampling.
///
/// # Errors
///
/// The design refuses the draw, or the failures of [`bootstrap_se_clusters_with_scratch`].
pub(crate) fn bootstrap_se_clusters_under<S, F>(
    declared: DeclaredSamplingDesign,
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
    // Under independent subject trajectories the labels are subject ids, and drawing whole
    // subjects is that design's own rule; every other design draws whole clusters.
    let rule = match declared {
        DeclaredSamplingDesign::IndependentSubjectTrajectories => ResamplingRule::RowSubjectIid,
        DeclaredSamplingDesign::ClusteredPanel | DeclaredSamplingDesign::DependentTimeSeries => {
            ResamplingRule::WholeCluster
        }
    };
    declared.apply_resampling(rule).map_err(design_err)?;
    let groups = clusters.len();
    if replicates == 0 || groups == 0 {
        return Ok(BootstrapSeResult::skipped());
    }
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let n_rep = replicates as usize;
    // One replicate buffer per worker; no groups-by-replicates matrix exists to admit.
    let ceiling = clusters.replicate_row_ceiling();
    let Some(scratch) = admit_bootstrap_scratch(ctx, ceiling, n_rep)? else {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    };
    let workers = scratch.workers();
    retained_index_slots(ceiling, workers)?;
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let rows = ReplicateRows::Clusters { members: clusters.members(), ceiling, ctx, stream_base };
    if workers > 1 {
        return evaluate_bootstrap_parallel(ctx, &rows, n_rep, make_scratch, estimate);
    }
    evaluate_bootstrap_serial(ctx, &rows, n_rep, make_scratch, estimate)
}

/// How one bootstrap replicate's row indices are supplied to the estimator.
pub(super) enum ReplicateRows<'a> {
    /// `n` iid row draws for replicate `r`, generated from `derive_stream(stream_base, r)`.
    ///
    /// Each worker fills its own length-`n` buffer. The `n` by replicates matrix is not
    /// retained.
    Rows { n: usize, ctx: &'a ExecutionContext, stream_base: u64 },
    /// One whole-cluster replicate, drawn into the caller buffer. The groups-by-replicates
    /// matrix is not retained. Length varies with the clusters drawn.
    Clusters {
        members: &'a [Vec<antecedent_data::GlobalId>],
        ceiling: usize,
        ctx: &'a ExecutionContext,
        stream_base: u64,
    },
    /// Contiguous blocks of `block_length` over `n` series rows.
    Blocks { n: usize, block_length: usize, ctx: &'a ExecutionContext, stream_base: u64 },
}

impl ReplicateRows<'_> {
    /// Fill `out` with the row indices of replicate `r`. `drawn` is the per-worker buffer the
    /// cluster and block draws land in before they are widened to rows, so no replicate allocates.
    pub(super) fn fill(
        &self,
        r: usize,
        out: &mut Vec<usize>,
        drawn: &mut Vec<antecedent_data::GlobalId>,
    ) -> Result<(), EstimationError> {
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
            Self::Clusters { members, ceiling, ctx, stream_base } => {
                let groups = members.len();
                let replicate = u64::try_from(r)
                    .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds u64"))?;
                fill_resample_index_replicate(
                    ResamplingPlan::IidBootstrap,
                    groups,
                    None,
                    ctx,
                    *stream_base,
                    replicate,
                    drawn,
                )?;
                for cluster in drawn.iter() {
                    let slot = row(*cluster)?;
                    let Some(sources) = members.get(slot) else {
                        return Err(EstimationError::data_msg(
                            "cluster draw is outside the observed clusters",
                        ));
                    };
                    if out.len().saturating_add(sources.len()) > *ceiling {
                        return Err(EstimationError::data_msg(
                            "cluster replicate exceeds the admitted index buffer",
                        ));
                    }
                    for src in sources {
                        out.push(row(*src)?);
                    }
                }
            }
            Self::Blocks { n, block_length, ctx, stream_base } => {
                let replicate = u64::try_from(r)
                    .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds u64"))?;
                fill_resample_index_replicate(
                    ResamplingPlan::MovingBlock { length: *block_length },
                    *n,
                    None,
                    ctx,
                    *stream_base,
                    replicate,
                    drawn,
                )?;
                for id in drawn.iter() {
                    out.push(row(*id)?);
                }
            }
        }
        Ok(())
    }
}

/// The enabled adaptive budget, run as a wave procedure: one replicate per wave on the caller,
/// stopping once the budget's success floor is reached. The replicates drawn and the standard
/// error are those of the fixed-count loop on the same seed; a replicate whose estimator errs
/// aborts the bootstrap.
fn evaluate_bootstrap_waves<S, F>(
    ctx: &ExecutionContext,
    rows: &ReplicateRows<'_>,
    n_rep: usize,
    make_scratch: impl Fn() -> S,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError>,
{
    let max_replicates = u32::try_from(n_rep)
        .map_err(|_| EstimationError::data_msg("bootstrap replicate count exceeds u32"))?;
    let rule = WaveStoppingRule::success_floor_rule(
        max_replicates,
        ctx.adaptive_bootstrap.required_replicates(),
    );
    let mut state = (make_scratch(), Vec::<usize>::new(), Vec::new());
    let run = run_waves_serial(&rule, ctx, 0, &mut state, |(scratch, idx, drawn), id, _rng| {
        let replicate = usize::try_from(id)
            .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds usize"))?;
        rows.fill(replicate, idx, drawn)?;
        estimate(scratch, idx)
    });
    match run {
        Ok(run) => Ok(finalize_bootstrap_se_ex(
            &run.successes,
            run.attempted,
            run.stop == StopReason::Cancelled,
            run.stop == StopReason::Converged,
        )),
        Err(DriveError::Estimator(error)) => Err(error),
        Err(DriveError::Wave(error)) => Err(EstimationError::Wave(error)),
    }
}

pub(super) fn evaluate_bootstrap_serial<S, F>(
    ctx: &ExecutionContext,
    rows: &ReplicateRows<'_>,
    n_rep: usize,
    make_scratch: impl Fn() -> S,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError>,
{
    if ctx.adaptive_bootstrap.enabled {
        return evaluate_bootstrap_waves(ctx, rows, n_rep, make_scratch, estimate);
    }
    let mut scratch = make_scratch();
    let mut ates = Vec::with_capacity(n_rep);
    let mut idx = Vec::new();
    let mut drawn = Vec::new();
    let mut cancelled = false;
    let mut early_stopped = false;
    let mut attempted = 0u32;
    let budget = ctx.adaptive_bootstrap;
    for r in 0..n_rep {
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        rows.fill(r, &mut idx, &mut drawn)?;
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

pub(super) fn evaluate_bootstrap_parallel<S, F>(
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
    let finished = std::sync::atomic::AtomicUsize::new(0);
    let first_cancel = std::sync::atomic::AtomicUsize::new(usize::MAX);
    // The workers are leased without waiting: a bootstrap that finds the pool held by
    // another study stays on its caller instead of queueing behind it.
    let values = ctx.map_indexed_with_scratch_if_free(
        n_rep,
        || (make_scratch(), Vec::<usize>::new(), Vec::new()),
        |r, inner, (scratch, idx, drawn): &mut (S, Vec<usize>, Vec<antecedent_data::GlobalId>)| {
            if inner.cancellation.is_cancelled() {
                first_cancel.fetch_min(r, std::sync::atomic::Ordering::Relaxed);
                return Ok::<Option<f64>, EstimationError>(None);
            }
            rows.fill(r, idx, drawn)?;
            let value = estimate(scratch, idx)?.filter(|ate| ate.is_finite());
            let done = finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if let Some(progress) = &inner.progress {
                progress.report(done as f64 / n_rep as f64, "bootstrap");
            }
            Ok(value)
        },
    )?;
    let stop = first_cancel.load(std::sync::atomic::Ordering::Relaxed);
    let mut ates = Vec::with_capacity(n_rep);
    let mut attempted = 0u32;
    for (r, value) in values.into_iter().enumerate() {
        if r >= stop {
            break;
        }
        attempted += 1;
        if let Some(ate) = value {
            ates.push(ate);
        }
    }
    let cancelled = stop != usize::MAX || ctx.cancellation.is_cancelled();
    Ok(finalize_bootstrap_se_ex(&ates, attempted, cancelled, false))
}
