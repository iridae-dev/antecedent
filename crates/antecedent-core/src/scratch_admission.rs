//! Worker caps and scratch leases for parallel resampling.
//!
//! [`ExecutionContext::admit_scratch`] takes one atomic lease on the admission pool for a
//! batch's workers and their scratch bytes, so concurrent callers share one remainder and
//! cannot together exceed it. The lease is released when the returned [`ScratchLease`]
//! drops.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

use crate::execution::{AdmissionError, AdmissionLease, AdmissionRemainder, ExecutionContext};

/// Why scratch was not admitted. Nothing is allocated and no lease is held.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Error)]
#[non_exhaustive]
pub enum ScratchRefusal {
    /// The byte product does not fit. It was not wrapped.
    #[error("scratch size overflows")]
    Overflow,
    /// The admission pool cannot hold the scratch beside the leases already open.
    #[error("admission pool cannot fund {bytes} bytes of scratch")]
    PoolShort {
        /// Bytes requested.
        bytes: u64,
    },
    /// The study was cancelled before the lease was granted.
    #[error("scratch admission stopped because the study was cancelled")]
    Cancelled,
}

/// A lease on the workers and scratch bytes of one batch.
///
/// Dropping it returns both to the pool.
#[must_use = "dropping the lease releases its workers and bytes"]
#[derive(Debug)]
pub struct ScratchLease {
    held: AdmissionLease,
    workers: usize,
    bytes: u64,
}

impl ScratchLease {
    /// Workers the batch may run. One means the batch stays on the caller.
    #[must_use]
    pub const fn workers(&self) -> usize {
        self.workers
    }

    /// Scratch bytes held for all workers.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// CPU and RAM this lease still holds.
    #[must_use]
    pub fn remainder(&self) -> AdmissionRemainder {
        self.held.remainder()
    }
}

impl ExecutionContext {
    /// Workers one batch of `work_items` asks for: the context's thread count, the item
    /// count, and the admission pool's thread capacity, whichever is smallest. One stays
    /// on the caller.
    ///
    /// This is a request, not a grant. It reads only the pool's fixed capacity, never its
    /// free slots, so there is no window between a read and a lease. The lease itself
    /// decides what is granted.
    fn requested_workers(&self, work_items: usize) -> usize {
        if work_items < 2 {
            return 1;
        }
        let requested = usize::try_from(self.parallelism.max_threads.get()).unwrap_or(usize::MAX);
        let requested = requested.clamp(1, work_items);
        let budget = usize::try_from(self.admission.thread_budget()).unwrap_or(usize::MAX);
        if budget <= 1 { 1 } else { requested.min(budget) }
    }

    /// Lease the workers and scratch for a batch of `work_items`, where each worker holds
    /// `per_worker_bytes`.
    ///
    /// The lease is taken in one atomic step on the admission pool. When the pool reports
    /// fewer free slots than requested, the request shrinks to what that refusal reported
    /// and is made again, so concurrent callers share one remainder and cannot together
    /// exceed it. Parallel workers hold their CPU slots on the lease; a serial batch holds
    /// bytes only.
    ///
    /// # Errors
    ///
    /// [`ScratchRefusal::Overflow`] when `per_worker_bytes * workers` does not fit,
    /// [`ScratchRefusal::PoolShort`] when the pool cannot fund it even for one worker's
    /// share of slots, and [`ScratchRefusal::Cancelled`] when the study is cancelled.
    pub fn admit_scratch(
        &self,
        work_items: usize,
        per_worker_bytes: u64,
    ) -> Result<ScratchLease, ScratchRefusal> {
        let mut workers = self.requested_workers(work_items);
        loop {
            let count = u64::try_from(workers).map_err(|_| ScratchRefusal::Overflow)?;
            let bytes = per_worker_bytes.checked_mul(count).ok_or(ScratchRefusal::Overflow)?;
            let threads = if workers > 1 {
                u32::try_from(workers).map_err(|_| ScratchRefusal::Overflow)?
            } else {
                0
            };
            match self.acquire_admission(threads, bytes) {
                Ok(held) => return Ok(ScratchLease { held, workers, bytes }),
                Err(AdmissionError::Cancelled) => return Err(ScratchRefusal::Cancelled),
                Err(AdmissionError::Overflow) => return Err(ScratchRefusal::Overflow),
                Err(AdmissionError::ExceedsRemainder { remaining_threads, .. }) => {
                    // The refusal carries the free slots the pool saw under its own lock.
                    let free = usize::try_from(remaining_threads).unwrap_or(usize::MAX);
                    let next = if free <= 1 { 1 } else { workers.min(free) };
                    if next >= workers {
                        return Err(ScratchRefusal::PoolShort { bytes });
                    }
                    workers = next;
                }
                Err(_) => return Err(ScratchRefusal::PoolShort { bytes }),
            }
        }
    }

    /// Reserve index bytes for a batch whose workers are leased later by
    /// [`Self::map_indexed_with_scratch`](crate::execution::ExecutionContext::map_indexed_with_scratch).
    ///
    /// The returned lease records the requested worker count and holds
    /// `workers * per_worker_bytes`. It holds no CPU slots, so the map can take them without
    /// waiting on this lease; the map shrinks to the slots it is granted.
    ///
    /// # Errors
    ///
    /// [`ScratchRefusal::Overflow`] when the byte product does not fit, and
    /// [`ScratchRefusal::PoolShort`] when the pool cannot fund those bytes.
    /// [`ScratchRefusal::Cancelled`] when the study is already cancelled.
    pub fn admit_index_bytes(
        &self,
        work_items: usize,
        per_worker_bytes: u64,
    ) -> Result<ScratchLease, ScratchRefusal> {
        let workers = self.requested_workers(work_items);
        let count = u64::try_from(workers).map_err(|_| ScratchRefusal::Overflow)?;
        let bytes = per_worker_bytes.checked_mul(count).ok_or(ScratchRefusal::Overflow)?;
        match self.acquire_admission(0, bytes) {
            Ok(held) => Ok(ScratchLease { held, workers, bytes }),
            Err(AdmissionError::Cancelled) => Err(ScratchRefusal::Cancelled),
            Err(AdmissionError::Overflow) => Err(ScratchRefusal::Overflow),
            Err(_) => Err(ScratchRefusal::PoolShort { bytes }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::execution::AdmissionPool;

    fn pooled(threads: u32, bytes: u64) -> (ExecutionContext, AdmissionPool) {
        let pool = AdmissionPool::fixed(threads, bytes);
        (ExecutionContext::production(1, 8).with_admission(pool.clone()), pool)
    }

    #[test]
    fn a_third_worker_is_refused_when_the_pool_is_two() {
        let (ctx, pool) = pooled(2, 1 << 20);
        let first = ctx.admit_scratch(8, 8).unwrap();
        assert_eq!(first.workers(), 2);
        assert_eq!(first.remainder().threads(), 2);
        assert_eq!(pool.available().threads(), 0);
        // The next batch finds no slot to share and stays on its caller.
        let second = ctx.admit_scratch(8, 8).unwrap();
        assert_eq!(second.workers(), 1);
        assert_eq!(pool.available().threads(), 0);
        // A third worker slot cannot be taken from the exhausted pool.
        let third = ctx.acquire_admission(1, 0).unwrap_err();
        assert!(matches!(third, AdmissionError::ExceedsRemainder { .. }), "{third}");
        drop(first);
        assert_eq!(pool.available().threads(), 2);
        drop(second);
        assert_eq!(pool.available().bytes(), 1 << 20);
    }

    #[test]
    fn concurrent_callers_cannot_hold_more_workers_than_the_pool() {
        let callers = 8usize;
        let (ctx, pool) = pooled(2, 1 << 20);
        let barrier = Barrier::new(callers);
        let parallel_held = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..callers {
                scope.spawn(|| {
                    let lease = ctx.admit_scratch(8, 8).unwrap();
                    let extra = if lease.workers() > 1 { lease.workers() } else { 0 };
                    let now = parallel_held.fetch_add(extra, Ordering::SeqCst) + extra;
                    peak.fetch_max(now, Ordering::SeqCst);
                    // Every caller holds its lease before any releases.
                    barrier.wait();
                    parallel_held.fetch_sub(extra, Ordering::SeqCst);
                });
            }
        });
        assert!(peak.load(Ordering::SeqCst) <= 2, "{}", peak.load(Ordering::SeqCst));
        assert_eq!(parallel_held.load(Ordering::SeqCst), 0);
        assert_eq!(pool.available().threads(), 2);
        assert_eq!(pool.available().bytes(), 1 << 20);
    }

    #[test]
    fn a_byte_product_that_overflows_is_refused() {
        let ctx = ExecutionContext::production(1, 4);
        let refused = ctx.admit_scratch(4, (u64::MAX / 2) + 1).unwrap_err();
        assert_eq!(refused, ScratchRefusal::Overflow);
    }

    #[test]
    fn scratch_is_funded_by_the_pool_capacity_that_is_also_the_planner_limit() {
        let (ctx, pool) = pooled(2, 1 << 20);
        assert_eq!(ctx.memory_limit_bytes(), Some(1 << 20));
        let lease = ctx.admit_scratch(8, 8).unwrap();
        assert_eq!(lease.workers(), 2);
        assert_eq!(lease.bytes(), 16);
        drop(lease);
        assert_eq!(pool.available().threads(), 2);
        assert_eq!(pool.available().bytes(), 1 << 20);
    }

    #[test]
    fn a_pool_short_of_bytes_is_refused_without_a_lease() {
        let (ctx, pool) = pooled(2, 15);
        let refused = ctx.admit_scratch(8, 8).unwrap_err();
        assert_eq!(refused, ScratchRefusal::PoolShort { bytes: 16 });
        assert_eq!(pool.available().threads(), 2);
        assert_eq!(pool.available().bytes(), 15);
    }
}
