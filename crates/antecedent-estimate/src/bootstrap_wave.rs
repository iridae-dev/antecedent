//! Bounded resampling waves.
//!
//! A wave retains one reused draw buffer, or `width` fixed-length replicates, rather than an
//! `n` by replicates matrix. Replicate `i` is drawn from [`StreamDomain::Resample`] at
//! `stream_base ^ i`: the same seed and replicate id always produce the same rows. IID, block,
//! permutation, Bayesian, and equal-cluster plans keep those streams. An unequal cluster
//! replicate stays variable length; this module does not build the groups-by-replicates index.
//!
//! The adaptive procedure stops on a Monte Carlo replicate floor. Stopping early ends replicate
//! evaluation only. It does not reduce the caller's full-sample asymptotic fit. Cancellation is
//! a recorded stop, not a success. Failed replicates stay in the attempt count.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AdmittedRouteMemory, EncodedWidth, ExecutionContext, RouteBudget, RouteMemoryDeclaration,
    RouteMemoryError, StreamDomain,
};
use antecedent_kernels::unbiased_index;

use crate::error::EstimationError;

/// Slots in an addressable `n × workers` index buffer.
///
/// The product is the retained index memory. It does not grow with the replicate count.
///
/// # Errors
///
/// `n * workers` indexes of `u64` do not fit in an addressable allocation.
pub(crate) fn retained_index_slots(n: usize, workers: usize) -> Result<usize, EstimationError> {
    n.checked_mul(workers)
        .filter(|&slots| {
            slots.checked_mul(size_of::<u64>()).is_some_and(|bytes| isize::try_from(bytes).is_ok())
        })
        .ok_or_else(|| {
            EstimationError::data_msg("bootstrap index allocation exceeds addressable capacity")
        })
}

/// Slots of one cluster-bootstrap replicate buffer per worker.
///
/// The retained buffer is `n` indexes times `workers`. A groups-by-replicates
/// matrix is admitted against `budget` first. That admission allocates nothing.
/// When the budget refuses the matrix, this still returns the worker-buffer
/// length, and the caller must not allocate the matrix. When the worker buffer
/// itself exceeds `budget`, this refuses before either buffer exists.
///
/// # Errors
///
/// `n * workers` is not an addressable index buffer, or that buffer exceeds
/// `budget`. A matrix the budget refuses, or that does not fit in `u64`, is not
/// allocated.
pub(crate) fn cluster_worker_slots(
    n: usize,
    workers: usize,
    groups: usize,
    replicates: usize,
    budget: RouteBudget,
) -> Result<usize, EstimationError> {
    let matrix = admit_group_replicate_matrix(groups, replicates, budget);
    let slots = retained_index_slots(n, workers)?;
    let retained_bytes = index_bytes(slots)?;
    if retained_bytes > budget.bytes() {
        return Err(EstimationError::data_msg(
            "bootstrap index allocation exceeds the route budget",
        ));
    }
    match matrix {
        Ok(_)
        | Err(RouteMemoryError::ExceedsBudget { .. } | RouteMemoryError::Unrepresentable) => {
            Ok(slots)
        }
        Err(error) => Err(EstimationError::data_msg(error.to_string())),
    }
}

/// Admit a groups-by-replicates index matrix under `budget`.
///
/// The estimate's total is the matrix size in bytes. This allocates nothing.
/// A refusal is the decision not to allocate that matrix.
///
/// # Errors
///
/// [`RouteMemoryError::Unrepresentable`] when the matrix does not fit in `u64`,
/// and [`RouteMemoryError::ExceedsBudget`] when it is larger than `budget`.
pub(crate) fn admit_group_replicate_matrix(
    groups: usize,
    replicates: usize,
    budget: RouteBudget,
) -> Result<AdmittedRouteMemory, RouteMemoryError> {
    let matrix_bytes = groups
        .checked_mul(replicates)
        .and_then(|slots| u64::try_from(slots).ok())
        .and_then(|slots| slots.checked_mul(index_byte_width()))
        .ok_or(RouteMemoryError::Unrepresentable)?;
    let width = EncodedWidth::int32().bytes();
    let Some(retained) = matrix_bytes.checked_sub(width) else {
        return Err(RouteMemoryError::Unrepresentable);
    };
    RouteMemoryDeclaration::new()
        .rows(1)
        .encoded_column(EncodedWidth::int32())
        .masks(0)
        .design_expansion(0)
        .learner_fold_copies(0, 0)
        .solver_scratch(0)
        .retained_scores_draws(retained)
        .output(0)
        .workers(1)
        .admit(budget)
}

fn index_byte_width() -> u64 {
    u64::try_from(size_of::<u64>()).unwrap_or(8)
}

fn index_bytes(slots: usize) -> Result<u64, EstimationError> {
    u64::try_from(slots)
        .ok()
        .and_then(|slots| slots.checked_mul(index_byte_width()))
        .ok_or_else(|| {
            EstimationError::data_msg("bootstrap index allocation exceeds addressable capacity")
        })
}

/// Retained IID bootstrap indexes for at most `width` replicates.
///
/// The buffer length is `n * width`.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct BootstrapIndexWave {
    indexes: Vec<u64>,
    n: usize,
    width: usize,
    filled: usize,
}

#[cfg(test)]
impl BootstrapIndexWave {
    /// Allocate a buffer for `width` replicates of length `n`.
    ///
    /// # Errors
    ///
    /// `n` or `width` is zero, or [`retained_index_slots`] refuses the product.
    pub(crate) fn allocate(n: usize, width: usize) -> Result<Self, EstimationError> {
        if n == 0 || width == 0 {
            return Err(EstimationError::data_msg(
                "bootstrap index wave needs a positive row count and width",
            ));
        }
        let slots = retained_index_slots(n, width)?;
        let mut indexes = Vec::new();
        indexes.try_reserve_exact(slots).map_err(|e| {
            EstimationError::data_msg(format!("bootstrap index allocation failed: {e}"))
        })?;
        indexes.resize(slots, 0);
        Ok(Self { indexes, n, width, filled: 0 })
    }

    /// Number of row indexes the wave retains (`n * width`).
    #[must_use]
    pub(crate) fn index_len(&self) -> usize {
        self.indexes.len()
    }

    /// Draw replicates `origin .. origin + count` into this buffer.
    ///
    /// `count` must be at most the allocated width. Replicate `origin + local` uses
    /// [`StreamDomain::Resample`] keyed by `stream_base ^ replicate`.
    ///
    /// # Errors
    ///
    /// `count` exceeds the width, a replicate id does not fit in `u64`, or a draw does
    /// not fit in a row index.
    pub(crate) fn fill(
        &mut self,
        ctx: &ExecutionContext,
        stream_base: u64,
        origin: u64,
        count: usize,
    ) -> Result<(), EstimationError> {
        if count > self.width {
            return Err(EstimationError::data_msg(
                "bootstrap index wave count exceeds the allocated width",
            ));
        }
        if self.n == 0 {
            return Err(EstimationError::data_msg("bootstrap replicate needs n > 0"));
        }
        let n = self.n;
        self.filled = 0;
        for (local, row) in self.indexes.chunks_mut(n).take(count).enumerate() {
            let local_id = u64::try_from(local).map_err(|_| replicate_id_error())?;
            let replicate = origin.checked_add(local_id).ok_or_else(replicate_id_error)?;
            let mut at = 0;
            with_replicate_draws(n, replicate, ctx, stream_base, |index| {
                row[at] = u64::try_from(index).map_err(|_| row_index_error())?;
                at += 1;
                Ok(())
            })?;
        }
        self.filled = count;
        Ok(())
    }

    /// Row indexes of local slot `local` from the last successful [`Self::fill`].
    #[must_use]
    pub(crate) fn replicate(&self, local: usize) -> Option<&[u64]> {
        if local >= self.filled {
            return None;
        }
        let start = local.checked_mul(self.n)?;
        let end = start.checked_add(self.n)?;
        self.indexes.get(start..end)
    }
}

/// Write one replicate's row indexes into `out`, replacing its previous contents.
///
/// The draw is [`StreamDomain::Resample`] at `stream_base ^ replicate`. `out` ends at length `n`.
///
/// # Errors
///
/// `n` is zero, the buffer cannot be reserved, or a replicate draw fails.
pub(crate) fn fill_worker_replicate(
    n: usize,
    replicate: u64,
    ctx: &ExecutionContext,
    stream_base: u64,
    out: &mut Vec<usize>,
) -> Result<(), EstimationError> {
    out.clear();
    out.try_reserve_exact(n).map_err(|e| {
        EstimationError::data_msg(format!("bootstrap index allocation failed: {e}"))
    })?;
    with_replicate_draws(n, replicate, ctx, stream_base, |index| {
        out.push(index);
        Ok(())
    })
}

#[inline]
fn with_replicate_draws(
    n: usize,
    replicate: u64,
    ctx: &ExecutionContext,
    stream_base: u64,
    mut store: impl FnMut(usize) -> Result<(), EstimationError>,
) -> Result<(), EstimationError> {
    if n == 0 {
        return Err(EstimationError::data_msg("bootstrap replicate needs n > 0"));
    }
    let mut rng = ctx.rng.stream_for(StreamDomain::Resample, stream_base ^ replicate);
    for _ in 0..n {
        store(unbiased_index(&mut rng, n))?;
    }
    Ok(())
}

#[cfg(test)]
fn replicate_id_error() -> EstimationError {
    EstimationError::data_msg("bootstrap replicate id exceeds u64")
}

#[cfg(test)]
fn row_index_error() -> EstimationError {
    EstimationError::data_msg(
        "row position does not fit in u64; rows are addressed by GlobalId and \
         ChunkIndex only spans one bounded chunk",
    )
}

#[cfg_attr(
    not(test),
    allow(
        dead_code,
        reason = "production still uses the serial evaluator; tests cover this procedure"
    )
)]
mod adaptive_resample {
    use antecedent_core::AdaptiveBootstrapBudget;
    use antecedent_data::{DataError, GlobalId, ResamplingPlan};

    use super::retained_index_slots;
    use crate::error::EstimationError;
    use crate::statistical_transport::percentile_interval;
    use crate::util::{adaptive_bootstrap_should_stop, finalize_bootstrap_se_ex};

    /// Row-index or Dirichlet-weight design. Streams stay on the data-layer plans.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum AdaptiveDesign<'a> {
        /// Index plan. Cluster labels are required when the plan needs them.
        Indexes {
            /// Resampling plan. Bayesian weights use [`AdaptiveDesign::Bayesian`].
            plan: ResamplingPlan,
            /// Rows in the original sample.
            n: usize,
            /// Cluster labels, required for clustered plans.
            cluster_ids: Option<&'a [u32]>,
        },
        /// Dirichlet(1, …, 1) weights on `n` rows, in original row order.
        Bayesian {
            /// Rows in the original sample.
            n: usize,
        },
    }

    impl AdaptiveDesign<'_> {
        fn n(self) -> usize {
            match self {
                Self::Indexes { n, .. } | Self::Bayesian { n } => n,
            }
        }

        fn variable_cluster(self) -> bool {
            match self {
                Self::Indexes {
                    plan: ResamplingPlan::ClusterBootstrap { .. },
                    cluster_ids: Some(ids),
                    ..
                } => !cluster_sizes_equal(ids),
                _ => false,
            }
        }
    }

    /// One finished draw. The slice is the reused wave buffer.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum ResampleDraw<'a> {
        /// Row ids. Length is `n` except for an unequal cluster replicate.
        Rows(&'a [GlobalId]),
        /// Dirichlet weights summing to `n`.
        Weights(&'a [f64]),
    }

    /// Arguments of one adaptive wave.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct AdaptiveWaveRequest<'a> {
        /// Design and its data-layer stream.
        pub design: AdaptiveDesign<'a>,
        /// Stream key mixed with the replicate id.
        pub stream_base: u64,
        /// Cap on replicates. Unattempted ids are not failures.
        pub requested: u32,
        /// Fixed-length replicates kept at once.
        ///
        /// Unequal cluster plans keep one variable-length buffer instead.
        pub width: usize,
        /// Monte Carlo floor. Disabled budgets evaluate the whole request.
        pub budget: AdaptiveBootstrapBudget,
        /// Nominal level of the type-7 percentile interval. Not the Monte Carlo error target.
        pub level: f64,
    }

    /// Finished or stopped wave.
    ///
    /// `interval` is the type-7 percentile of the finite successes when the bootstrap failure
    /// rule allows a standard error. `normal_sd_monte_carlo_error` is `1/√(2(B − 1))` at that
    /// success count. It is the error the floor bounds. It is not a coverage error of `interval`.
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) struct AdaptiveWaveResult {
        /// Finite successful estimates, in attempt order.
        pub successes: Vec<f64>,
        /// Replicates the caller asked for.
        pub requested: u32,
        /// Replicates that finished with a finite value or a soft failure.
        pub attempted: u32,
        /// Finite successes.
        pub replicates_ok: u32,
        /// Soft failures inside [`Self::attempted`].
        ///
        /// Unattempted replicates and cancelled slots are not failures.
        pub replicates_failed: u32,
        /// The token stopped the loop. Not a successful replicate.
        pub cancelled: bool,
        /// The success floor was met before the request was exhausted.
        pub early_stopped: bool,
        /// Sample standard deviation of [`Self::successes`], withheld by the failure rule.
        pub se: Option<f64>,
        /// Type-7 percentile interval of the successes when [`Self::se`] is present.
        pub interval: Option<(f64, f64)>,
        /// Relative Monte Carlo error of a normal sample standard deviation at `replicates_ok`.
        pub normal_sd_monte_carlo_error: Option<f64>,
        /// Entries held in the wave buffer. Not `n` times the request, and not a cluster matrix.
        pub resident_entries: usize,
    }

    /// Relative Monte Carlo error `1/√(2(B − 1))` of a normal sample standard deviation.
    ///
    /// `None` when fewer than two successes leave the formula undefined. The value is not a
    /// coverage error and not a count of full-sample fits.
    pub(crate) fn normal_sd_relative_monte_carlo_error(successes: u32) -> Option<f64> {
        if successes < 2 {
            return None;
        }
        Some(1.0 / (2.0 * f64::from(successes - 1)).sqrt())
    }

    /// Draw and evaluate `request` until the Monte Carlo floor, the cap, or cancellation.
    ///
    /// Each replicate is drawn into the reused wave buffer from
    /// `StreamDomain::Resample` at `stream_base ^ replicate`. The same seed and id match
    /// `fill_resample_index_replicate` or `fill_resample_weight_replicate`.
    ///
    /// `estimate` returns `Ok(Some(value))` for a finite success, `Ok(None)` or a non-finite
    /// value for a soft failure, or `Err` to abort. [`DataError::Cancelled`] from the draw or
    /// from `estimate` records [`AdaptiveWaveResult::cancelled`] and does not count that
    /// replicate as a success or a failure.
    ///
    /// The stop is [`AdaptiveBootstrapBudget::required_replicates`] successes, and only when
    /// the budget is enabled and replicates remain. Failures do not advance the floor.
    /// Reaching the floor sets `early_stopped` and does not change how many times the caller
    /// fits the original sample: this function never fits it.
    ///
    /// # Errors
    ///
    /// `width` is zero, the wave allocation does not fit, a draw fails for a reason other than
    /// cancellation, a fixed-length plan changes length, or `estimate` returns any error other
    /// than cancellation.
    #[allow(
        clippy::too_many_lines,
        reason = "one loop draws, scores, and applies the stopping rule"
    )]
    pub(crate) fn run_adaptive_resample_wave(
        request: &AdaptiveWaveRequest<'_>,
        ctx: &antecedent_core::ExecutionContext,
        mut estimate: impl FnMut(ResampleDraw<'_>) -> Result<Option<f64>, EstimationError>,
    ) -> Result<AdaptiveWaveResult, EstimationError> {
        let n = request.design.n();
        if request.requested == 0 || n == 0 {
            return Ok(AdaptiveWaveResult::empty(request.requested, false, 0));
        }
        if request.width == 0 {
            return Err(EstimationError::data_msg("resample wave needs a positive width"));
        }
        if ctx.cancellation.is_cancelled() {
            return Ok(AdaptiveWaveResult::empty(request.requested, true, 0));
        }
        let variable = request.design.variable_cluster();
        let slots = if variable {
            1
        } else {
            request.width.min(usize::try_from(request.requested).unwrap_or(usize::MAX))
        };
        let mut index_slots = Vec::new();
        let mut weight_slots = Vec::new();
        let mut scratch_indexes = Vec::new();
        let mut scratch_weights = Vec::new();
        match request.design {
            AdaptiveDesign::Bayesian { .. } => {
                let count = retained_index_slots(n, slots)?;
                weight_slots.try_reserve_exact(count).map_err(|error| {
                    EstimationError::data_msg(format!("resample wave allocation failed: {error}"))
                })?;
                weight_slots.resize(count, 0.0);
            }
            AdaptiveDesign::Indexes { .. } if !variable => {
                let count = retained_index_slots(n, slots)?;
                index_slots.try_reserve_exact(count).map_err(|error| {
                    EstimationError::data_msg(format!("resample wave allocation failed: {error}"))
                })?;
                index_slots.resize(count, GlobalId::new(0));
            }
            AdaptiveDesign::Indexes { .. } => {}
        }
        let allocated = index_slots.len().max(weight_slots.len());
        let mut high_water = allocated;
        let mut successes = Vec::new();
        let mut attempted = 0u32;
        let mut cancelled = false;
        let mut early_stopped = false;
        for replicate in 0..request.requested {
            if ctx.cancellation.is_cancelled() {
                cancelled = true;
                break;
            }
            let index = usize::try_from(replicate).map_err(|_| {
                EstimationError::data_msg("bootstrap replicate id exceeds the wave index")
            })?;
            let local = index % slots;
            let fill = fill_replicate(
                request,
                ctx,
                u64::from(replicate),
                local,
                n,
                variable,
                &mut index_slots,
                &mut weight_slots,
                &mut scratch_indexes,
                &mut scratch_weights,
            )?;
            let ready = match fill {
                Fill::Cancelled => {
                    cancelled = true;
                    break;
                }
                Fill::FixedIndex => Ready::FixedIndex,
                Fill::VariableIndex => Ready::VariableIndex,
                Fill::FixedWeight => Ready::FixedWeight,
            };
            if variable {
                high_water = high_water.max(scratch_indexes.len());
            }
            let outcome = match ready {
                Ready::FixedIndex => {
                    let start = local * n;
                    estimate(ResampleDraw::Rows(&index_slots[start..start + n]))
                }
                Ready::VariableIndex => estimate(ResampleDraw::Rows(&scratch_indexes)),
                Ready::FixedWeight => {
                    let start = local * n;
                    estimate(ResampleDraw::Weights(&weight_slots[start..start + n]))
                }
            };
            match outcome {
                Err(EstimationError::Data(DataError::Cancelled)) => {
                    cancelled = true;
                    break;
                }
                Err(error) => return Err(error),
                Ok(value) => {
                    attempted += 1;
                    let finite = value.filter(|item| item.is_finite());
                    if let Some(item) = finite {
                        successes.push(item);
                    }
                    if ctx.cancellation.is_cancelled() {
                        cancelled = true;
                        report(ctx, replicate, request.requested);
                        break;
                    }
                    let more =
                        replicate.checked_add(1).is_some_and(|next| next < request.requested);
                    if finite.is_some() && more {
                        let ok = u32::try_from(successes.len()).unwrap_or(u32::MAX);
                        if adaptive_bootstrap_should_stop(request.budget, ok) {
                            early_stopped = true;
                            report(ctx, replicate, request.requested);
                            break;
                        }
                    }
                    report(ctx, replicate, request.requested);
                }
            }
        }
        Ok(AdaptiveWaveResult::from_successes(
            successes,
            request.requested,
            attempted,
            cancelled,
            early_stopped,
            request.level,
            high_water,
        ))
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Fill {
        Cancelled,
        FixedIndex,
        VariableIndex,
        FixedWeight,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Ready {
        FixedIndex,
        VariableIndex,
        FixedWeight,
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one draw writes the reused index buffer and the reused weight buffer"
    )]
    fn fill_replicate(
        request: &AdaptiveWaveRequest<'_>,
        ctx: &antecedent_core::ExecutionContext,
        replicate: u64,
        local: usize,
        n: usize,
        variable: bool,
        index_slots: &mut [GlobalId],
        weight_slots: &mut [f64],
        scratch_indexes: &mut Vec<GlobalId>,
        scratch_weights: &mut Vec<f64>,
    ) -> Result<Fill, EstimationError> {
        match request.design {
            AdaptiveDesign::Bayesian { .. } => {
                if let Err(error) = antecedent_data::resample::fill_resample_weight_replicate(
                    ResamplingPlan::BayesianBootstrap,
                    n,
                    ctx,
                    request.stream_base,
                    replicate,
                    scratch_weights,
                ) {
                    return cancelled_or(error);
                }
                if scratch_weights.len() != n {
                    return Err(EstimationError::data_msg(
                        "Bayesian resample wave expected one weight per row",
                    ));
                }
                let start = local * n;
                weight_slots[start..start + n].copy_from_slice(scratch_weights);
                Ok(Fill::FixedWeight)
            }
            AdaptiveDesign::Indexes { plan, cluster_ids, .. } => {
                if let Err(error) = antecedent_data::resample::fill_resample_index_replicate(
                    plan,
                    n,
                    cluster_ids,
                    ctx,
                    request.stream_base,
                    replicate,
                    scratch_indexes,
                ) {
                    return cancelled_or(error);
                }
                if variable {
                    return Ok(Fill::VariableIndex);
                }
                if scratch_indexes.len() != n {
                    return Err(EstimationError::data_msg(
                        "fixed-length resample returned a variable row count",
                    ));
                }
                let start = local * n;
                index_slots[start..start + n].copy_from_slice(scratch_indexes);
                Ok(Fill::FixedIndex)
            }
        }
    }

    fn cancelled_or(error: DataError) -> Result<Fill, EstimationError> {
        match error {
            DataError::Cancelled => Ok(Fill::Cancelled),
            other => Err(EstimationError::Data(other)),
        }
    }

    fn report(ctx: &antecedent_core::ExecutionContext, replicate: u32, requested: u32) {
        if let Some(progress) = &ctx.progress {
            let done = f64::from(replicate.saturating_add(1));
            progress.report(done / f64::from(requested), "adaptive resample");
        }
    }

    fn cluster_sizes_equal(ids: &[u32]) -> bool {
        let mut counts = std::collections::BTreeMap::<u32, usize>::new();
        for id in ids {
            *counts.entry(*id).or_insert(0) += 1;
        }
        let mut sizes = counts.values();
        let Some(first) = sizes.next() else {
            return true;
        };
        sizes.all(|size| size == first)
    }

    impl AdaptiveWaveResult {
        fn empty(requested: u32, cancelled: bool, resident_entries: usize) -> Self {
            Self::from_successes(Vec::new(), requested, 0, cancelled, false, 0.95, resident_entries)
        }

        fn from_successes(
            successes: Vec<f64>,
            requested: u32,
            attempted: u32,
            cancelled: bool,
            early_stopped: bool,
            level: f64,
            resident_entries: usize,
        ) -> Self {
            let summary = finalize_bootstrap_se_ex(&successes, attempted, cancelled, early_stopped);
            let interval = summary.se.and_then(|_| percentile_interval(&successes, level));
            Self {
                normal_sd_monte_carlo_error: normal_sd_relative_monte_carlo_error(
                    summary.replicates_ok,
                ),
                successes,
                requested,
                attempted,
                replicates_ok: summary.replicates_ok,
                replicates_failed: summary.replicates_failed,
                cancelled: summary.cancelled,
                early_stopped: summary.early_stopped,
                se: summary.se,
                interval,
                resident_entries,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use antecedent_core::{AdaptiveBootstrapBudget, ExecutionContext, VariableId};
        use antecedent_data::{DataError, GlobalId, PermutationScheme, ResamplingPlan};
        use antecedent_kernels::sample_std;

        use super::{
            AdaptiveDesign, AdaptiveWaveRequest, AdaptiveWaveResult, ResampleDraw,
            normal_sd_relative_monte_carlo_error, run_adaptive_resample_wave,
        };
        use crate::error::EstimationError;
        use crate::statistical_transport::percentile_interval;

        const DATA: [f64; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        fn floor_budget() -> AdaptiveBootstrapBudget {
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 2, se_rel_epsilon: 0.2 }
        }

        fn iid() -> AdaptiveDesign<'static> {
            AdaptiveDesign::Indexes {
                plan: ResamplingPlan::IidBootstrap,
                n: DATA.len(),
                cluster_ids: None,
            }
        }

        fn request(
            design: AdaptiveDesign<'_>,
            requested: u32,
            width: usize,
            budget: AdaptiveBootstrapBudget,
        ) -> AdaptiveWaveRequest<'_> {
            AdaptiveWaveRequest { design, stream_base: 0x51, requested, width, budget, level: 0.95 }
        }

        fn mean_rows(data: &[f64], draw: ResampleDraw<'_>) -> Result<Option<f64>, EstimationError> {
            match draw {
                ResampleDraw::Rows(rows) => {
                    let sum: f64 =
                        rows.iter().map(|id| data[usize::try_from(id.get()).unwrap()]).sum();
                    Ok(Some(sum / rows.len() as f64))
                }
                ResampleDraw::Weights(_) => {
                    Err(EstimationError::data_msg("index design returned weights"))
                }
            }
        }

        fn near(left: f64, right: f64) {
            assert!((left - right).abs() <= 1e-12);
        }

        fn bits_eq(left: &[f64], right: &[f64]) {
            assert_eq!(left.len(), right.len());
            for (left, right) in left.iter().zip(right) {
                assert_eq!(left.to_bits(), right.to_bits());
            }
        }

        fn accounted(result: &AdaptiveWaveResult) {
            assert_eq!(
                result.replicates_ok.saturating_add(result.replicates_failed),
                result.attempted
            );
            assert_eq!(usize::try_from(result.replicates_ok).unwrap(), result.successes.len());
            assert!(result.attempted <= result.requested);
            assert!(result.successes.iter().all(|value| value.is_finite()));
        }

        #[test]
        fn stopped_interval_matches_the_successful_prefix() {
            let budget = floor_budget();
            let ctx = ExecutionContext::for_tests(99);
            let mut full_sample_fits = 0u32;
            let point = {
                full_sample_fits += 1;
                DATA.iter().sum::<f64>() / DATA.len() as f64
            };
            let request = request(iid(), 40, 1, budget);
            let stopped =
                run_adaptive_resample_wave(&request, &ctx, |draw| mean_rows(&DATA, draw)).unwrap();
            accounted(&stopped);
            assert_eq!(full_sample_fits, 1);
            near(point, 3.5);
            assert!(stopped.early_stopped);
            assert!(!stopped.cancelled);
            assert_eq!(stopped.replicates_failed, 0);
            assert_eq!(stopped.replicates_ok, budget.required_replicates());
            assert!(stopped.attempted < stopped.requested);
            assert_ne!(stopped.replicates_failed, stopped.requested - stopped.replicates_ok);
            let at = stopped.normal_sd_monte_carlo_error.expect("monte carlo error");
            let before = normal_sd_relative_monte_carlo_error(stopped.replicates_ok - 1).unwrap();
            assert!(at <= budget.se_rel_epsilon);
            assert!(before > budget.se_rel_epsilon);
            assert!((0.95 - at).abs() > 0.5);
            let (lo, hi) = stopped.interval.expect("interval");
            let published = percentile_interval(&stopped.successes, 0.95).unwrap();
            near(lo, published.0);
            near(hi, published.1);
            near(stopped.se.unwrap(), sample_std(&stopped.successes));
            let mut exact = request;
            exact.requested = budget.required_replicates();
            let exact_run =
                run_adaptive_resample_wave(&exact, &ctx, |draw| mean_rows(&DATA, draw)).unwrap();
            assert!(!exact_run.early_stopped);
            bits_eq(&exact_run.successes, &stopped.successes);
            let mut disabled = request;
            disabled.budget = AdaptiveBootstrapBudget::disabled();
            let full =
                run_adaptive_resample_wave(&disabled, &ctx, |draw| mean_rows(&DATA, draw)).unwrap();
            assert!(!full.early_stopped);
            assert_eq!(full.attempted, full.requested);
            bits_eq(&stopped.successes, &full.successes[..stopped.successes.len()]);
            let mut wide = request;
            wide.width = 5;
            let wide_run =
                run_adaptive_resample_wave(&wide, &ctx, |draw| mean_rows(&DATA, draw)).unwrap();
            bits_eq(&wide_run.successes, &stopped.successes);
            assert_eq!(stopped.resident_entries, DATA.len());
            assert_eq!(wide_run.resident_entries, DATA.len() * 5);
            let wide_cap = DATA.len() * usize::try_from(wide.requested).unwrap();
            assert!(wide_run.resident_entries < wide_cap);
            let threaded = ExecutionContext::production(99, 4);
            let parallel =
                run_adaptive_resample_wave(&request, &threaded, |draw| mean_rows(&DATA, draw))
                    .unwrap();
            bits_eq(&parallel.successes, &stopped.successes);
            let shifted = DATA.map(|value| value + 3.0);
            let shifted_run =
                run_adaptive_resample_wave(&request, &ctx, |draw| mean_rows(&shifted, draw))
                    .unwrap();
            let (slo, shi) = shifted_run.interval.expect("shifted interval");
            near(slo, lo + 3.0);
            near(shi, hi + 3.0);
            near(shifted_run.se.unwrap(), stopped.se.unwrap());
            assert_eq!(full_sample_fits, 1);
        }

        #[test]
        fn failed_replicates_stay_in_the_count() {
            let budget = floor_budget();
            let ctx = ExecutionContext::for_tests(3);
            let request = request(iid(), 40, 3, budget);
            let mut id = 0u32;
            let stopped = run_adaptive_resample_wave(&request, &ctx, |draw| {
                let current = id;
                id += 1;
                if current % 3 == 0 { Ok(None) } else { mean_rows(&DATA, draw) }
            })
            .unwrap();
            let mut expect_ok = 0u32;
            let mut expect_attempted = 0u32;
            let mut replicate = 0u32;
            while expect_ok < budget.required_replicates() {
                expect_attempted += 1;
                if replicate % 3 != 0 {
                    expect_ok += 1;
                }
                replicate += 1;
            }
            accounted(&stopped);
            assert!(stopped.early_stopped);
            assert_eq!(stopped.replicates_ok, expect_ok);
            assert_eq!(stopped.attempted, expect_attempted);
            assert_eq!(stopped.replicates_failed, expect_attempted - expect_ok);
            assert!(stopped.attempted < stopped.requested);
            assert_ne!(stopped.replicates_failed, stopped.requested - stopped.replicates_ok);
            let at = stopped.normal_sd_monte_carlo_error.unwrap();
            assert!(at <= budget.se_rel_epsilon);
            assert!(stopped.interval.is_some());
            let mut prefix = request;
            prefix.budget = AdaptiveBootstrapBudget::disabled();
            prefix.requested = expect_attempted;
            let mut prefix_id = 0u32;
            let prefix_run = run_adaptive_resample_wave(&prefix, &ctx, |draw| {
                let current = prefix_id;
                prefix_id += 1;
                if current % 3 == 0 { Ok(None) } else { mean_rows(&DATA, draw) }
            })
            .unwrap();
            assert!(!prefix_run.early_stopped);
            assert_eq!(prefix_run.replicates_failed, stopped.replicates_failed);
            bits_eq(&prefix_run.successes, &stopped.successes);
        }

        #[test]
        fn soft_failures_withhold_the_interval_without_dropping_the_count() {
            let ctx = ExecutionContext::for_tests(3);
            let half = request(iid(), 4, 3, AdaptiveBootstrapBudget::disabled());
            let mut half_id = 0u32;
            let half_run = run_adaptive_resample_wave(&half, &ctx, |draw| {
                let current = half_id;
                half_id += 1;
                if current % 2 == 1 { Ok(None) } else { mean_rows(&DATA, draw) }
            })
            .unwrap();
            accounted(&half_run);
            assert_eq!(half_run.replicates_ok, 2);
            assert_eq!(half_run.replicates_failed, 2);
            assert!(half_run.se.is_some());
            assert!(half_run.interval.is_some());

            let mut majority = half;
            majority.requested = 6;
            let mut majority_id = 0u32;
            let majority_run = run_adaptive_resample_wave(&majority, &ctx, |draw| {
                let current = majority_id;
                majority_id += 1;
                if current % 3 == 0 { mean_rows(&DATA, draw) } else { Ok(None) }
            })
            .unwrap();
            accounted(&majority_run);
            assert_eq!(majority_run.replicates_ok, 2);
            assert_eq!(majority_run.replicates_failed, 4);
            assert!(majority_run.se.is_none());
            assert!(majority_run.interval.is_none());
            assert_eq!(majority_run.successes.len(), 2);

            let non_finite = half;
            let mut finite_id = 0u32;
            let finite_run = run_adaptive_resample_wave(&non_finite, &ctx, |draw| {
                let current = finite_id;
                finite_id += 1;
                if current == 1 { Ok(Some(f64::NAN)) } else { mean_rows(&DATA, draw) }
            })
            .unwrap();
            accounted(&finite_run);
            assert_eq!(finite_run.replicates_failed, 1);
            assert_eq!(finite_run.replicates_ok, 3);

            let empty = request(iid(), 0, 3, AdaptiveBootstrapBudget::disabled());
            let empty_run =
                run_adaptive_resample_wave(&empty, &ctx, |draw| mean_rows(&DATA, draw)).unwrap();
            assert_eq!(empty_run.attempted, 0);
            assert_eq!(empty_run.replicates_failed, 0);
            assert!(!empty_run.cancelled);
            let narrow = request(iid(), 4, 0, AdaptiveBootstrapBudget::disabled());
            let refused = run_adaptive_resample_wave(&narrow, &ctx, |draw| mean_rows(&DATA, draw));
            assert!(refused.is_err());
            let hard = request(iid(), 4, 3, AdaptiveBootstrapBudget::disabled());
            let error = run_adaptive_resample_wave(&hard, &ctx, |_| {
                Err(EstimationError::data_msg("replicate refused"))
            })
            .unwrap_err();
            assert!(error.to_string().contains("replicate refused"));
        }

        #[test]
        fn cancellation_is_a_recorded_stop() {
            let request = request(iid(), 20, 4, floor_budget());
            let before_ctx = ExecutionContext::for_tests(5);
            before_ctx.cancellation.cancel();
            let mut before_calls = 0u32;
            let before = run_adaptive_resample_wave(&request, &before_ctx, |draw| {
                before_calls += 1;
                mean_rows(&DATA, draw)
            })
            .unwrap();
            assert_eq!(before_calls, 0);
            assert!(before.cancelled);
            assert!(!before.early_stopped);
            assert_eq!(before.attempted, 0);
            assert_eq!(before.replicates_ok, 0);
            assert_eq!(before.replicates_failed, 0);
            assert!(before.interval.is_none());
            assert_eq!(before.resident_entries, 0);

            let after_ctx = ExecutionContext::for_tests(5);
            let mut after_calls = 0u32;
            let after = run_adaptive_resample_wave(&request, &after_ctx, |draw| {
                after_calls += 1;
                if after_calls == 3 {
                    after_ctx.cancellation.cancel();
                }
                mean_rows(&DATA, draw)
            })
            .unwrap();
            accounted(&after);
            assert_eq!(after_calls, 3);
            assert_eq!(after.replicates_ok, 3);
            assert_eq!(after.replicates_failed, 0);
            assert!(after.cancelled);
            assert!(!after.early_stopped);
            assert!(after.interval.is_some());

            let estimate_ctx = ExecutionContext::for_tests(5);
            let mut estimate_calls = 0u32;
            let from_estimate = run_adaptive_resample_wave(&request, &estimate_ctx, |_| {
                estimate_calls += 1;
                if estimate_calls == 3 {
                    Err(EstimationError::Data(DataError::Cancelled))
                } else {
                    Ok(Some(1.0))
                }
            })
            .unwrap();
            assert_eq!(estimate_calls, 3);
            assert_eq!(from_estimate.attempted, 2);
            assert_eq!(from_estimate.replicates_ok, 2);
            assert_eq!(from_estimate.replicates_failed, 0);
            assert!(from_estimate.cancelled);
            assert!(!from_estimate.early_stopped);

            let tiny =
                AdaptiveBootstrapBudget { enabled: true, min_replicates: 2, se_rel_epsilon: 1.0 };
            let need = tiny.required_replicates();
            let mut floor_request = request;
            floor_request.budget = tiny;
            let preempt_ctx = ExecutionContext::for_tests(5);
            let mut produced = 0u32;
            let preempted = run_adaptive_resample_wave(&floor_request, &preempt_ctx, |_| {
                produced += 1;
                if produced == need {
                    preempt_ctx.cancellation.cancel();
                }
                Ok(Some(1.0))
            })
            .unwrap();
            assert_eq!(preempted.replicates_ok, need);
            assert!(preempted.cancelled);
            assert!(!preempted.early_stopped);
            assert_eq!(produced, need);
            let complete_ctx = ExecutionContext::for_tests(5);
            let completed =
                run_adaptive_resample_wave(&floor_request, &complete_ctx, |_| Ok(Some(1.0)))
                    .unwrap();
            assert!(completed.early_stopped);
            assert!(!completed.cancelled);
            assert_eq!(completed.replicates_ok, need);
            assert!(completed.attempted < completed.requested);
        }

        #[test]
        fn design_streams_stay_on_their_plans() {
            let plans = [
                ResamplingPlan::IidBootstrap,
                ResamplingPlan::MovingBlock { length: 3 },
                ResamplingPlan::CircularBlock { length: 3 },
                ResamplingPlan::StationaryBlock { expected_length: 2.0 },
                ResamplingPlan::Permutation(PermutationScheme::Full),
            ];
            for plan in plans {
                assert_index_stream(plan, 12, None, 2);
            }
            assert_index_stream(
                ResamplingPlan::Permutation(PermutationScheme::WithinCluster),
                6,
                Some(&[0, 0, 0, 1, 1, 2]),
                2,
            );
            assert_bayesian_stream();
            assert_equal_clusters_match_the_batch();
            assert_unequal_clusters_stay_variable_length();
        }

        fn assert_index_stream(plan: ResamplingPlan, n: usize, ids: Option<&[u32]>, width: usize) {
            let requested = 5u32;
            let ctx = ExecutionContext::for_tests(11);
            let wave = request(
                AdaptiveDesign::Indexes { plan, n, cluster_ids: ids },
                requested,
                width,
                AdaptiveBootstrapBudget::disabled(),
            );
            let mut recorded = Vec::new();
            let result = run_adaptive_resample_wave(&wave, &ctx, |draw| {
                let ResampleDraw::Rows(rows) = draw else {
                    return Err(EstimationError::data_msg("index design returned weights"));
                };
                recorded.push(rows.to_vec());
                Ok(Some(1.0))
            })
            .unwrap();
            assert_eq!(recorded.len(), usize::try_from(requested).unwrap());
            assert!(!result.early_stopped);
            assert!(!result.cancelled);
            let slots = width.min(usize::try_from(requested).unwrap());
            assert_eq!(result.resident_entries, n * slots);
            let mut scratch = Vec::new();
            for (replicate, rows) in recorded.iter().enumerate() {
                antecedent_data::resample::fill_resample_index_replicate(
                    plan,
                    n,
                    ids,
                    &ctx,
                    wave.stream_base,
                    u64::try_from(replicate).unwrap(),
                    &mut scratch,
                )
                .unwrap();
                assert_eq!(rows, &scratch);
            }
        }

        fn assert_bayesian_stream() {
            let n = 9usize;
            let requested = 4u32;
            let width = 2usize;
            let ctx = ExecutionContext::for_tests(11);
            let wave = request(
                AdaptiveDesign::Bayesian { n },
                requested,
                width,
                AdaptiveBootstrapBudget::disabled(),
            );
            let mut recorded = Vec::new();
            let result = run_adaptive_resample_wave(&wave, &ctx, |draw| {
                let ResampleDraw::Weights(weights) = draw else {
                    return Err(EstimationError::data_msg("weight design returned rows"));
                };
                recorded.push(weights.to_vec());
                Ok(Some(1.0))
            })
            .unwrap();
            assert_eq!(result.resident_entries, n * width);
            let mut scratch = Vec::new();
            for (replicate, weights) in recorded.iter().enumerate() {
                antecedent_data::resample::fill_resample_weight_replicate(
                    ResamplingPlan::BayesianBootstrap,
                    n,
                    &ctx,
                    wave.stream_base,
                    u64::try_from(replicate).unwrap(),
                    &mut scratch,
                )
                .unwrap();
                assert_eq!(weights.len(), scratch.len());
                for (left, right) in weights.iter().zip(&scratch) {
                    assert_eq!(left.to_bits(), right.to_bits());
                }
                let sum: f64 = weights.iter().sum();
                near(sum, n as f64);
            }
        }

        fn assert_equal_clusters_match_the_batch() {
            let ids = [0u32, 0, 1, 1, 2, 2];
            let n = ids.len();
            let requested = 8usize;
            let width = 2usize;
            let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(3) };
            let ctx = ExecutionContext::for_tests(11);
            let wave = request(
                AdaptiveDesign::Indexes { plan, n, cluster_ids: Some(&ids) },
                u32::try_from(requested).unwrap(),
                width,
                AdaptiveBootstrapBudget::disabled(),
            );
            let mut recorded = Vec::new();
            let result = run_adaptive_resample_wave(&wave, &ctx, |draw| {
                let ResampleDraw::Rows(rows) = draw else {
                    return Err(EstimationError::data_msg("index design returned weights"));
                };
                recorded.push(rows.to_vec());
                Ok(Some(1.0))
            })
            .unwrap();
            assert_eq!(result.resident_entries, n * width);
            assert!(result.resident_entries < 3 * requested);
            let mut batch = vec![GlobalId::new(0); n * requested];
            antecedent_data::fill_resample_index_batch(
                plan,
                n,
                requested,
                Some(&ids),
                &ctx,
                wave.stream_base,
                &mut batch,
            )
            .unwrap();
            for (replicate, rows) in recorded.iter().enumerate() {
                assert_eq!(rows.as_slice(), &batch[replicate * n..(replicate + 1) * n]);
            }
        }

        fn assert_unequal_clusters_stay_variable_length() {
            let ids = [0u32, 0, 0, 1];
            let n = ids.len();
            let requested = 24u32;
            let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(4) };
            let ctx = ExecutionContext::for_tests(11);
            let wave = request(
                AdaptiveDesign::Indexes { plan, n, cluster_ids: Some(&ids) },
                requested,
                8,
                AdaptiveBootstrapBudget::disabled(),
            );
            let mut recorded = Vec::new();
            let result = run_adaptive_resample_wave(&wave, &ctx, |draw| {
                let ResampleDraw::Rows(rows) = draw else {
                    return Err(EstimationError::data_msg("index design returned weights"));
                };
                recorded.push(rows.to_vec());
                Ok(Some(1.0))
            })
            .unwrap();
            let mut scratch = Vec::new();
            for (replicate, rows) in recorded.iter().enumerate() {
                antecedent_data::resample::fill_resample_index_replicate(
                    plan,
                    n,
                    Some(&ids),
                    &ctx,
                    wave.stream_base,
                    u64::try_from(replicate).unwrap(),
                    &mut scratch,
                )
                .unwrap();
                assert_eq!(rows, &scratch);
            }
            let longest = recorded.iter().map(Vec::len).max().unwrap();
            assert_eq!(result.resident_entries, longest);
            assert!(longest <= 6);
            assert!(longest < 8);
            assert!(longest < 2 * usize::try_from(requested).unwrap());
            assert!(recorded.iter().any(|rows| rows.len() != n));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use antecedent_core::ExecutionContext;
    use antecedent_data::ResamplingPlan;

    use super::*;
    use crate::util::{bootstrap_se, finalize_bootstrap_se};

    fn reference_ids(n: usize, replicates: usize, seed: u64, stream_base: u64) -> Vec<u64> {
        let ctx = ExecutionContext::for_tests(seed);
        let mut row = Vec::new();
        let mut full = Vec::with_capacity(n * replicates);
        for replicate in 0..replicates {
            antecedent_data::resample::fill_resample_index_replicate(
                ResamplingPlan::IidBootstrap,
                n,
                None,
                &ctx,
                stream_base,
                u64::try_from(replicate).unwrap(),
                &mut row,
            )
            .unwrap();
            full.extend(row.iter().map(|id| id.get()));
        }
        full
    }

    fn as_rows(flat: &[u64], n: usize) -> Vec<Vec<usize>> {
        flat.chunks(n)
            .map(|row| row.iter().map(|&id| usize::try_from(id).unwrap()).collect())
            .collect()
    }

    fn row_sum(idx: &[usize]) -> f64 {
        let sum = idx.iter().fold(0u32, |acc, &row| acc + u32::try_from(row).unwrap());
        f64::from(sum)
    }

    fn sums_of(rows: &[Vec<usize>]) -> Vec<f64> {
        rows.iter().map(|idx| row_sum(idx)).collect()
    }

    fn assert_bootstrap_matches(
        n: usize,
        replicates: u32,
        stream_base: u64,
        seed: u64,
        threads: u32,
        expected: &[Vec<usize>],
    ) {
        let ctx = if threads == 1 {
            ExecutionContext::for_tests(seed)
        } else {
            ExecutionContext::production(seed, threads)
        };
        let recorded = Mutex::new(Vec::new());
        let result = bootstrap_se(replicates, &ctx, stream_base, n, |idx| {
            recorded.lock().unwrap().push(idx.to_vec());
            Ok(Some(row_sum(idx)))
        })
        .unwrap();
        let mut got = recorded.into_inner().unwrap();
        let mut expect = expected.to_vec();
        if threads > 1 {
            got.sort();
            expect.sort();
        }
        assert_eq!(got, expect);
        let reference = finalize_bootstrap_se(&sums_of(expected), replicates);
        let got_se = result.se.expect("bootstrap se");
        let expect_se = reference.se.expect("reference se");
        assert!((got_se - expect_se).abs() < 1e-12);
        assert_eq!(result.replicates_ok, replicates);
        assert!(!result.cancelled);
    }

    #[test]
    fn wave_matches_full_index_reference() {
        let n = 5usize;
        let replicates = 11u32;
        let n_rep = replicates as usize;
        let stream_base = 0x51u64;
        let seed = 99u64;
        let reference = reference_ids(n, n_rep, seed, stream_base);
        let expected = as_rows(&reference, n);
        let ctx = ExecutionContext::for_tests(seed);
        let width = 3usize;
        let mut wave = BootstrapIndexWave::allocate(n, width).unwrap();
        let mut origin = 0usize;
        while origin < n_rep {
            let count = width.min(n_rep - origin);
            wave.fill(&ctx, stream_base, u64::try_from(origin).unwrap(), count).unwrap();
            for local in 0..count {
                let got = wave.replicate(local).expect("filled slot");
                let start = (origin + local) * n;
                assert_eq!(got, &reference[start..start + n]);
            }
            origin += count;
        }
        let mut alone = BootstrapIndexWave::allocate(n, 1).unwrap();
        alone.fill(&ctx, stream_base, 7, 1).unwrap();
        assert_eq!(alone.replicate(0).expect("replicate 7"), &reference[7 * n..8 * n]);
        assert_bootstrap_matches(n, replicates, stream_base, seed, 1, &expected);
        assert_bootstrap_matches(n, replicates, stream_base, seed, 4, &expected);
    }

    #[test]
    fn index_buffer_len_is_bounded_by_workers() {
        let n = 17usize;
        let workers = 4usize;
        let replicates = 200usize;
        let mut wave = BootstrapIndexWave::allocate(n, workers).unwrap();
        assert_eq!(wave.index_len(), n * workers);
        assert!(wave.index_len() < n * replicates);
        let ctx = ExecutionContext::for_tests(3);
        let mut origin = 0u64;
        let mut seen = 0usize;
        while seen < replicates {
            let count = workers.min(replicates - seen);
            wave.fill(&ctx, 11, origin, count).unwrap();
            assert_eq!(wave.index_len(), n * workers);
            origin += u64::try_from(count).unwrap();
            seen += count;
        }
    }

    #[test]
    fn cluster_index_buffer_is_rows_times_workers() {
        let n = 11usize;
        let workers = 4usize;
        let groups = 4usize;
        let replicates = 200usize;
        let matrix_bytes = groups * replicates * size_of::<u64>();
        let retained_bytes = n * workers * size_of::<u64>();
        assert!(retained_bytes < matrix_bytes);
        let budget = RouteBudget::new(u64::try_from(retained_bytes).unwrap());
        let slots = cluster_worker_slots(n, workers, groups, replicates, budget).unwrap();
        assert_eq!(slots, n * workers);
        assert!(slots < groups * replicates);
        let tight = RouteBudget::new(u64::try_from(retained_bytes - 1).unwrap());
        let refused = cluster_worker_slots(n, workers, groups, replicates, tight).unwrap_err();
        assert!(refused.to_string().contains("route budget"), "{refused}");
    }

    #[test]
    fn route_budget_refuses_groups_by_replicates_before_allocation() {
        let groups = 4usize;
        let replicates = 200usize;
        let matrix_bytes = u64::try_from(groups * replicates * size_of::<u64>()).unwrap();
        let budget = RouteBudget::new(matrix_bytes - 1);
        let refused = admit_group_replicate_matrix(groups, replicates, budget).unwrap_err();
        assert!(matches!(refused, RouteMemoryError::ExceedsBudget { .. }));
        assert!(refused.to_string().contains("route budget"));
        let admitted =
            admit_group_replicate_matrix(groups, replicates, RouteBudget::new(matrix_bytes))
                .expect("a budget of the matrix size admits the estimate and allocates nothing");
        assert_eq!(admitted.total(), matrix_bytes);
    }
}
