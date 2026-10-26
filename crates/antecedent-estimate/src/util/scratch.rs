//! Admission of the index and gather buffers a resampling loop retains, and of draw buffers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, ScratchLease, ScratchRefusal};

use crate::bootstrap_wave::retained_index_slots;
use crate::error::{EstimationError, ResampleBufferFault};

/// Reserve the index bytes for one bootstrap.
///
/// The lease holds `n` index slots per worker and no CPU slots. The parallel map
/// leases the workers, so this reservation does not block that acquire. Adaptive
/// early-stop runs on the caller with one buffer. `None` means the study was cancelled.
///
/// # Errors
///
/// An index buffer that overflows or no longer fits in the pool. Nothing is allocated.
pub(super) fn admit_bootstrap_scratch(
    ctx: &ExecutionContext,
    n: usize,
    n_rep: usize,
) -> Result<Option<ScratchLease>, EstimationError> {
    let per_worker = index_buffer_bytes(retained_index_slots(n, 1)?)?;
    let work_items = if ctx.adaptive_bootstrap.enabled { 1 } else { n_rep };
    match ctx.admit_index_bytes(work_items, per_worker) {
        Ok(lease) => {
            index_buffer_bytes(retained_index_slots(n, lease.workers())?)?;
            Ok(Some(lease))
        }
        Err(ScratchRefusal::Cancelled) => Ok(None),
        Err(ScratchRefusal::PoolShort { .. }) => {
            Err(EstimationError::ResampleBuffer(ResampleBufferFault::IndexScratchPool))
        }
        Err(_) => Err(EstimationError::ResampleBuffer(ResampleBufferFault::IndexCapacity)),
    }
}

fn index_buffer_bytes(slots: usize) -> Result<u64, EstimationError> {
    u64::try_from(slots)
        .ok()
        .and_then(|slots| slots.checked_mul(u64::try_from(size_of::<u64>()).unwrap_or(8)))
        .ok_or(EstimationError::ResampleBuffer(ResampleBufferFault::IndexCapacity))
}

/// Lease `count * width` values of `T` that a draw loop retains, before the buffer is built.
///
/// The lease holds bytes on the pool and no CPU slots. Hold it until the buffer is dropped or
/// handed to an owner that holds its own reservation.
///
/// # Errors
///
/// The study is cancelled, the byte product overflows, or the admission pool cannot fund it.
/// Nothing is allocated.
pub(crate) fn admit_array<T>(
    ctx: &ExecutionContext,
    count: usize,
    width: usize,
) -> Result<antecedent_core::execution::AdmissionLease, EstimationError> {
    let bytes = count
        .checked_mul(width)
        .and_then(|cells| cells.checked_mul(size_of::<T>().max(1)))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(EstimationError::ResampleBuffer(ResampleBufferFault::DrawBufferOverflow))?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => EstimationError::unsupported(
            "execution cancelled before the draw buffers were admitted",
        ),
        other => EstimationError::ResampleBuffer(ResampleBufferFault::DrawBufferAdmission(other)),
    })
}

/// [`admit_array`] for a function with no [`ExecutionContext`]: the workflow account bound by
/// [`antecedent_core::numerical_admission`], or `None` when the call is made outside any
/// workflow and nothing is charged.
///
/// # Errors
///
/// As [`admit_array`].
pub(crate) fn admit_bound_array<T>(
    count: usize,
    width: usize,
) -> Result<Option<antecedent_core::execution::AdmissionLease>, EstimationError> {
    let bytes = count
        .checked_mul(width)
        .and_then(|cells| cells.checked_mul(size_of::<T>().max(1)))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(EstimationError::ResampleBuffer(ResampleBufferFault::DrawBufferOverflow))?;
    antecedent_core::numerical_admission::acquire(bytes).transpose().map_err(|error| {
        EstimationError::ResampleBuffer(ResampleBufferFault::DrawBufferAdmission(error))
    })
}

/// Lease the per-worker gather buffers of one bootstrap, before any replicate runs.
///
/// Each worker holds `rows * cells_per_row` `f64` cells (the resampled design, outcome and
/// working vectors it gathers into). The lease holds those bytes for all workers and no CPU
/// slots, because the parallel map leases the workers itself. Adaptive early-stop runs one
/// worker. A whole-cluster resample may exceed `rows` by the cluster-size slack, which
/// [`bootstrap_by_design`] admits separately as its index lease. Hold the returned lease
/// across the bootstrap call. `None` means the study was cancelled; the bootstrap that
/// follows reports that itself, so a caller may ignore it.
///
/// # Errors
///
/// The byte product overflows or the admission pool cannot fund it. Nothing is allocated.
pub fn admit_gather_scratch(
    ctx: &ExecutionContext,
    rows: usize,
    cells_per_row: usize,
    replicates: u32,
) -> Result<Option<ScratchLease>, EstimationError> {
    if replicates == 0 || rows == 0 {
        return Ok(None);
    }
    let per_worker = rows
        .checked_mul(cells_per_row)
        .and_then(|cells| cells.checked_mul(size_of::<f64>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(EstimationError::ResampleBuffer(ResampleBufferFault::GatherCapacity))?;
    let work_items = if ctx.adaptive_bootstrap.enabled { 1 } else { replicates as usize };
    match ctx.admit_index_bytes(work_items, per_worker) {
        Ok(lease) => Ok(Some(lease)),
        Err(ScratchRefusal::Cancelled) => Ok(None),
        Err(ScratchRefusal::PoolShort { .. }) => {
            Err(EstimationError::ResampleBuffer(ResampleBufferFault::GatherPool))
        }
        Err(_) => Err(EstimationError::ResampleBuffer(ResampleBufferFault::GatherCapacity)),
    }
}
