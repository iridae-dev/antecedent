//! Execution context: indexed maps and bounded pipelines.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::admission::{AdmissionError, AdmissionLease, AdmissionRequest, AdmissionWait};
use super::budget::{NonZeroThreadCount, Parallelism};
use super::context::ExecutionContext;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;
use thiserror::Error;

/// What every spawned worker of one parallel call carries so its work is accounted
/// under the caller's lease: the lease itself (it stays charged until the workers
/// join), the caller's execution-observation counters, and the numerical-admission
/// binding with the lease installed as the worker reservation.
struct WorkerScope {
    lease: Arc<AdmissionLease>,
    observation: crate::execution_attempt::ObservationToken,
    numerical: crate::numerical_admission::NumericalAdmissionToken,
}

impl WorkerScope {
    fn capture(lease: AdmissionLease) -> Self {
        let observation = crate::execution_attempt::ObservationToken::capture();
        let lease = Arc::new(lease);
        let numerical = crate::numerical_admission::with_worker(
            Arc::clone(&lease),
            crate::numerical_admission::NumericalAdmissionToken::capture,
        );
        Self { lease, observation, numerical }
    }
}

impl ExecutionContext {
    /// `f(0), …, f(n-1)` under this context. Results return in index order.
    ///
    /// When `n > 1` and `max_threads > 1`, work is chunked on `thread::scope`
    /// and `f` receives [`Self::serial_inner`] so nested pools stay serial.
    /// When `n <= 1` or the context is already serial, `f` receives a context
    /// bounded by this pool's worker budget; an already compliant context is reused.
    ///
    /// The parallel path acquires one admission lease for its workers before it
    /// spawns. The lease is dropped after the workers join, including when `f`
    /// fails. Cancellation keeps the lease charged until the workers join and cleanup
    /// drops it. A call that cannot take the lease,
    /// including one whose context is already cancelled, evaluates `f` on the
    /// caller with a serial inner context and spawns nothing. A second parallel call waits until the pool
    /// can hold that lease. The serial path does not take a lease, so a nested
    /// map on [`Self::serial_inner`] does not wait on the outer hold.
    ///
    /// # Errors
    ///
    /// Returns the lowest-index error produced by `f`. Work above a known failure
    /// is skipped, so a failing run does not evaluate every remaining index.
    ///
    /// # Panics
    ///
    /// Panics if a worker fails to write a slot at or below the first error (a
    /// programming error in the pool, not in `f`).
    pub fn map_indexed<T, E, F>(&self, n: usize, f: F) -> Result<Vec<T>, E>
    where
        T: Send,
        E: Send,
        F: Fn(usize, &Self) -> Result<T, E> + Sync,
    {
        self.map_indexed_with_scratch(n, || (), |index, ctx, _scratch| f(index, ctx))
    }

    /// Execute an indexed map with shared admission even for a serial caller.
    /// Nested work reuses a live enclosing worker reservation and remains serial.
    ///
    /// # Errors
    /// Cancellation/admission refusal, or the original lowest-index task error.
    pub fn try_map_indexed<T, E, F>(&self, n: usize, f: F) -> Result<Vec<T>, MapExecutionError<E>>
    where
        T: Send,
        E: Send,
        F: Fn(usize, &Self) -> Result<T, E> + Sync,
    {
        self.try_map_indexed_with_scratch(n, || (), |index, ctx, ()| f(index, ctx))
    }

    /// Admitted map with one reusable scratch value per worker, including serial work.
    /// Scratch construction occurs only after admission and a cancellation check.
    ///
    /// # Errors
    /// Cancellation/admission refusal, or the original lowest-index task error.
    pub fn try_map_indexed_with_scratch<T, S, E, F>(
        &self,
        n: usize,
        make_scratch: impl Fn() -> S + Sync,
        f: F,
    ) -> Result<Vec<T>, MapExecutionError<E>>
    where
        T: Send,
        S: Send,
        E: Send,
        F: Fn(usize, &Self, &mut S) -> Result<T, E> + Sync,
    {
        if n == 0 {
            return Ok(Vec::new());
        }
        if self.cancellation.is_cancelled() {
            return Err(MapExecutionError::Admission(AdmissionError::Cancelled));
        }
        let call = |index, ctx: &Self, scratch: &mut S| {
            if ctx.cancellation.is_cancelled() {
                return Err(MapExecutionError::Admission(AdmissionError::Cancelled));
            }
            if !crate::numerical_admission::worker()
                .is_some_and(|lease| lease.belongs_to(&ctx.admission))
            {
                return Err(MapExecutionError::Admission(AdmissionError::Released));
            }
            f(index, ctx, scratch).map_err(MapExecutionError::Work)
        };
        if let Some(worker) =
            crate::numerical_admission::worker().filter(|lease| lease.belongs_to(&self.admission))
        {
            let inner = self.serial_inner();
            return crate::numerical_admission::with_worker(worker, || {
                inner.map_indexed_with_scratch(n, make_scratch, call)
            });
        }
        if self.map_worker_count(n) <= 1 {
            let lease = Arc::new(self.acquire_map_lease(1)?);
            let inner = self.serial_inner();
            return crate::numerical_admission::with_worker(lease, || {
                inner.map_indexed_with_scratch(n, make_scratch, call)
            });
        }
        self.map_indexed_with_scratch(n, make_scratch, call)
    }

    /// [`Self::map_indexed`] with one scratch value per worker.
    ///
    /// The serial path builds one scratch and reuses it. The parallel path builds
    /// one scratch inside each worker after the admission lease is held, and
    /// stops scheduling indexes above the first error.
    ///
    /// # Errors
    ///
    /// Returns the lowest-index error produced by `f`.
    ///
    /// # Panics
    ///
    /// Panics if a worker fails to write a slot at or below the first error.
    pub fn map_indexed_with_scratch<T, S, E, F>(
        &self,
        n: usize,
        make_scratch: impl Fn() -> S + Sync,
        f: F,
    ) -> Result<Vec<T>, E>
    where
        T: Send,
        S: Send,
        E: Send,
        F: Fn(usize, &Self, &mut S) -> Result<T, E> + Sync,
    {
        self.map_indexed_with_scratch_waiting(n, make_scratch, f, MapWait::UntilAvailable)
    }

    /// [`Self::map_indexed_with_scratch`] that never waits for worker slots.
    ///
    /// The worker lease is taken in one immediate step. When the pool has fewer free slots
    /// than requested, the map runs on as many workers as that refusal reported free, and
    /// on the caller with a serial inner context when fewer than two are free. A caller
    /// that already holds other leases on this pool cannot be blocked by a competing study.
    ///
    /// # Errors
    ///
    /// Returns the lowest-index error produced by `f`.
    ///
    /// # Panics
    ///
    /// Panics if a worker fails to write a slot at or below the first error.
    pub fn map_indexed_with_scratch_if_free<T, S, E, F>(
        &self,
        n: usize,
        make_scratch: impl Fn() -> S + Sync,
        f: F,
    ) -> Result<Vec<T>, E>
    where
        T: Send,
        S: Send,
        E: Send,
        F: Fn(usize, &Self, &mut S) -> Result<T, E> + Sync,
    {
        self.map_indexed_with_scratch_waiting(n, make_scratch, f, MapWait::IfFree)
    }

    pub(super) fn map_indexed_with_scratch_waiting<T, S, E, F>(
        &self,
        n: usize,
        make_scratch: impl Fn() -> S + Sync,
        f: F,
        wait: MapWait,
    ) -> Result<Vec<T>, E>
    where
        T: Send,
        S: Send,
        E: Send,
        F: Fn(usize, &Self, &mut S) -> Result<T, E> + Sync,
    {
        if n == 0 {
            return Ok(Vec::new());
        }
        let threads = self.map_worker_count(n);
        if threads <= 1 {
            let mut scratch = make_scratch();
            // A caller can request more workers than its shared account funds.
            // Keep nested numerical kernels within that account on the serial path too.
            let bounded;
            let pool_threads = self.admission.thread_budget().max(1);
            let ctx = if self.parallelism.max_threads.get() > pool_threads {
                bounded = Self {
                    parallelism: Parallelism::bounded(
                        NonZeroThreadCount::new(pool_threads).unwrap_or(NonZeroThreadCount::MIN),
                    ),
                    ..self.clone()
                };
                &bounded
            } else {
                self
            };
            return (0..n).map(|index| f(index, ctx, &mut scratch)).collect();
        }
        let Ok(worker_threads) = u32::try_from(threads) else {
            let mut scratch = make_scratch();
            return (0..n).map(|index| f(index, self, &mut scratch)).collect();
        };
        let acquired = match wait {
            MapWait::UntilAvailable => {
                self.acquire_map_lease(worker_threads).map(|lease| (worker_threads, lease))
            }
            MapWait::IfFree => self.acquire_map_lease_if_free(worker_threads),
        };
        let Ok((granted, lease)) = acquired else {
            let mut scratch = make_scratch();
            let inner = self.serial_inner();
            return (0..n).map(|index| f(index, &inner, &mut scratch)).collect();
        };
        let threads = usize::try_from(granted).unwrap_or(threads).min(threads);
        let inner = self.serial_inner();
        let mut slots: Vec<Option<Result<T, E>>> = (0..n).map(|_| None).collect();
        let first_failure = AtomicUsize::new(usize::MAX);
        let WorkerScope { lease, observation, numerical } = WorkerScope::capture(lease);
        std::thread::scope(|scope| {
            let f = &f;
            let inner = &inner;
            let first_failure = &first_failure;
            let make_scratch = &make_scratch;
            let mut rest = slots.as_mut_slice();
            let mut start = 0usize;
            for t in 0..threads {
                let take = rest.len().div_ceil(threads - t);
                let (mine, next) = rest.split_at_mut(take);
                let begin = start;
                let observation = &observation;
                let numerical = &numerical;
                scope.spawn(move || {
                    observation.execute(|| {
                        let mut scratch = make_scratch();
                        for (k, slot) in mine.iter_mut().enumerate() {
                            let index = begin + k;
                            if first_failure.load(Ordering::Relaxed) < index {
                                break;
                            }
                            let outcome =
                                numerical.execute(inner, || f(index, inner, &mut scratch));
                            if outcome.is_err() {
                                first_failure.fetch_min(index, Ordering::Relaxed);
                            }
                            *slot = Some(outcome);
                        }
                    })
                });
                rest = next;
                start += take;
                if rest.is_empty() {
                    break;
                }
            }
        });
        drop(lease);
        slots
            .into_iter()
            .map(|slot| slot.expect("every index below the first error was filled"))
            .collect()
    }

    /// Worker count for [`Self::map_indexed`], never above the admission pool.
    ///
    /// `n` is at least one. A count of one stays on the caller and takes no lease.
    pub(super) fn map_worker_count(&self, n: usize) -> usize {
        let requested = usize::try_from(self.parallelism.max_threads.get()).unwrap_or(usize::MAX);
        let requested = requested.clamp(1, n);
        if requested == 1 {
            return 1;
        }
        let pool_slots = usize::try_from(self.admission().thread_budget()).unwrap_or(usize::MAX);
        requested.min(pool_slots)
    }

    /// Immediate worker lease that shrinks to the free slots the pool reports when it
    /// refuses, down to two. The count granted is returned with the lease.
    pub(super) fn acquire_map_lease_if_free(
        &self,
        threads: u32,
    ) -> Result<(u32, AdmissionLease), AdmissionError> {
        let mut want = threads;
        loop {
            match self.acquire_request(AdmissionRequest::new(want, 0), None) {
                Ok(lease) => return Ok((want, lease)),
                Err(AdmissionError::ExceedsRemainder { remaining_threads, .. })
                    if remaining_threads >= 2 && remaining_threads < want =>
                {
                    want = remaining_threads;
                }
                Err(error) => return Err(error),
            }
        }
    }

    pub(super) fn acquire_map_lease(&self, threads: u32) -> Result<AdmissionLease, AdmissionError> {
        self.admit_request(AdmissionRequest::new(threads, 0), AdmissionWait::UntilAvailable)
    }
}

/// What a pipeline consumer wants after taking an item.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PipelineStep {
    /// Take the next item.
    Continue,
    /// Stop after this item. Production above it is dropped.
    Stop,
}

/// What a bounded pipeline did.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PipelineReceipt {
    /// Items the consumer took, in index order.
    pub items_consumed: usize,
    /// Most items produced-or-being-produced and not yet consumed at one time. It
    /// never exceeds the window, however long the consumer stalls.
    pub peak_live: usize,
    /// Workers that produced items; one means the caller produced inline.
    pub workers: usize,
}

pub(super) struct PipelineState<T, E> {
    pub(super) next_claim: usize,
    pub(super) next_consume: usize,
    pub(super) ready: std::collections::BTreeMap<usize, Result<T, E>>,
    pub(super) live: usize,
    pub(super) peak_live: usize,
    pub(super) stop: bool,
    pub(super) workers_alive: usize,
}

/// Marks a pipeline worker gone, including when it unwinds, so the consumer
/// never waits for an item nobody will produce.
pub(super) struct PipelineWorkerExit<'a, T, E> {
    pub(super) shared: &'a Mutex<PipelineState<T, E>>,
    pub(super) wake: &'a Condvar,
}

impl<T, E> Drop for PipelineWorkerExit<'_, T, E> {
    fn drop(&mut self) {
        let mut state = self.shared.lock().unwrap_or_else(PoisonError::into_inner);
        state.workers_alive = state.workers_alive.saturating_sub(1);
        // A panicking producer can leave a claimed index permanently missing. Stop
        // peers and wake the ordered consumer so the scope can join and propagate it.
        if std::thread::panicking() {
            state.stop = true;
        }
        drop(state);
        self.wake.notify_all();
    }
}

/// Tells pipeline workers to stop and frees items nobody will consume.
pub(super) struct PipelineStop<'a, T, E> {
    pub(super) shared: &'a Mutex<PipelineState<T, E>>,
    pub(super) wake: &'a Condvar,
}

impl<T, E> Drop for PipelineStop<'_, T, E> {
    fn drop(&mut self) {
        let mut state = self.shared.lock().unwrap_or_else(PoisonError::into_inner);
        state.stop = true;
        state.ready.clear();
        drop(state);
        self.wake.notify_all();
    }
}

impl ExecutionContext {
    /// Produce `0..n` on leased workers and consume them in index order on the caller,
    /// with at most `window` items produced and not yet consumed.
    ///
    /// A worker may claim index `i` only while `i < consumed + window`, so a consumer
    /// that stalls stops the producers after `window` items and the live buffers stay at
    /// `window` instead of growing to `n`. The consumer runs on the calling thread, one
    /// item at a time, in index order, so folds and sinks need no synchronization. An
    /// item stays live until its consumer call returns.
    ///
    /// Workers are leased like [`Self::map_indexed`], without waiting: when fewer than two
    /// slots are free the caller produces and consumes inline, one item live at a time.
    /// `produce` receives [`Self::serial_inner`].
    ///
    /// Cancellation is observed before each item is consumed; items produced beyond the
    /// last consumed one are dropped, and the lease returns when the workers join. A
    /// consumer that returns [`PipelineStep::Stop`] ends the run with a receipt.
    ///
    /// # Errors
    ///
    /// Cancellation, or the lowest-index error of `produce`, or the first error of
    /// `consume`; production above a known failure is not consumed.
    ///
    /// # Panics
    ///
    /// A panic in `produce` or `consume` propagates after the workers stop.
    #[allow(
        clippy::too_many_lines,
        reason = "one function owns the claim window, the ordered hand-off and the shutdown"
    )]
    pub fn pipeline_indexed_with_scratch<T, S, E, P, C>(
        &self,
        n: usize,
        window: usize,
        make_scratch: impl Fn() -> S + Sync,
        produce: P,
        mut consume: C,
    ) -> Result<PipelineReceipt, MapExecutionError<E>>
    where
        T: Send,
        S: Send,
        E: Send,
        P: Fn(usize, &Self, &mut S) -> Result<T, E> + Sync,
        C: FnMut(usize, T) -> Result<PipelineStep, E>,
    {
        let window = window.max(1);
        if n == 0 {
            return Ok(PipelineReceipt { items_consumed: 0, peak_live: 0, workers: 0 });
        }
        if self.cancellation.is_cancelled() {
            return Err(MapExecutionError::Admission(AdmissionError::Cancelled));
        }
        let inner = self.serial_inner();
        let threads = self.map_worker_count(n).min(window);
        let granted = if threads > 1 {
            u32::try_from(threads).ok().and_then(|t| self.acquire_map_lease_if_free(t).ok())
        } else {
            None
        };
        let Some((granted, lease)) = granted else {
            let mut scratch = make_scratch();
            for index in 0..n {
                if self.cancellation.is_cancelled() {
                    return Err(MapExecutionError::Admission(AdmissionError::Cancelled));
                }
                let item = produce(index, &inner, &mut scratch).map_err(MapExecutionError::Work)?;
                let step = consume(index, item).map_err(MapExecutionError::Work)?;
                if step == PipelineStep::Stop {
                    return Ok(PipelineReceipt {
                        items_consumed: index + 1,
                        peak_live: 1,
                        workers: 1,
                    });
                }
            }
            return Ok(PipelineReceipt { items_consumed: n, peak_live: 1, workers: 1 });
        };
        let workers = usize::try_from(granted).unwrap_or(threads).clamp(2, threads);
        let WorkerScope { lease, observation, numerical } = WorkerScope::capture(lease);
        let shared = Mutex::new(PipelineState::<T, E> {
            next_claim: 0,
            next_consume: 0,
            ready: std::collections::BTreeMap::new(),
            live: 0,
            peak_live: 0,
            stop: false,
            workers_alive: workers,
        });
        let wake = Condvar::new();
        let lock = || shared.lock().unwrap_or_else(PoisonError::into_inner);
        let mut consumed = 0usize;
        let mut outcome: Result<(), MapExecutionError<E>> = Ok(());
        std::thread::scope(|scope| {
            let produce = &produce;
            let make_scratch = &make_scratch;
            let inner = &inner;
            let observation = &observation;
            let numerical = &numerical;
            let shared_ref = &shared;
            let wake_ref = &wake;
            for _ in 0..workers {
                scope.spawn(move || {
                    observation.execute(|| {
                        let _exit = PipelineWorkerExit { shared: shared_ref, wake: wake_ref };
                        let mut scratch = make_scratch();
                        loop {
                            let claimed = {
                                let mut state =
                                    shared_ref.lock().unwrap_or_else(PoisonError::into_inner);
                                loop {
                                    if state.stop || state.next_claim >= n {
                                        break None;
                                    }
                                    if state.next_claim < state.next_consume + window {
                                        let index = state.next_claim;
                                        state.next_claim += 1;
                                        state.live += 1;
                                        state.peak_live = state.peak_live.max(state.live);
                                        break Some(index);
                                    }
                                    state = wake_ref
                                        .wait(state)
                                        .unwrap_or_else(PoisonError::into_inner);
                                }
                            };
                            let Some(index) = claimed else { break };
                            let result =
                                numerical.execute(inner, || produce(index, inner, &mut scratch));
                            shared_ref
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .ready
                                .insert(index, result);
                            wake_ref.notify_all();
                        }
                    });
                });
            }
            // Stops the workers on every exit from this closure, including a panic in
            // `consume`, so the scope can join them.
            let _stop = PipelineStop { shared: shared_ref, wake: wake_ref };
            'items: for index in 0..n {
                let item = {
                    let mut state = lock();
                    loop {
                        if state.stop {
                            break Err(AdmissionError::Released);
                        }
                        if let Some(item) = state.ready.remove(&index) {
                            break Ok(item);
                        }
                        if state.workers_alive == 0 {
                            break Err(AdmissionError::Released);
                        }
                        if self.cancellation.is_cancelled() {
                            break Err(AdmissionError::Cancelled);
                        }
                        state = wake
                            .wait_timeout(state, Duration::from_millis(5))
                            .unwrap_or_else(PoisonError::into_inner)
                            .0;
                    }
                };
                let step = match item {
                    Err(refusal) => Err(MapExecutionError::Admission(refusal)),
                    Ok(Err(error)) => Err(MapExecutionError::Work(error)),
                    Ok(Ok(value)) => {
                        if self.cancellation.is_cancelled() {
                            Err(MapExecutionError::Admission(AdmissionError::Cancelled))
                        } else {
                            consume(index, value).map_err(MapExecutionError::Work)
                        }
                    }
                };
                match step {
                    Err(error) => {
                        outcome = Err(error);
                        break 'items;
                    }
                    Ok(PipelineStep::Stop) => {
                        consumed += 1;
                        break 'items;
                    }
                    Ok(PipelineStep::Continue) => {}
                }
                consumed += 1;
                let mut state = lock();
                state.live = state.live.saturating_sub(1);
                state.next_consume = index + 1;
                drop(state);
                wake.notify_all();
            }
        });
        drop(lease);
        let peak_live = lock().peak_live;
        outcome.map(|()| PipelineReceipt { items_consumed: consumed, peak_live, workers })
    }
}

impl ExecutionContext {
    /// [`Self::pipeline_indexed_with_scratch`] without per-worker scratch.
    ///
    /// # Errors
    ///
    /// As [`Self::pipeline_indexed_with_scratch`].
    pub fn pipeline_indexed<T, E, P, C>(
        &self,
        n: usize,
        window: usize,
        produce: P,
        consume: C,
    ) -> Result<PipelineReceipt, MapExecutionError<E>>
    where
        T: Send,
        E: Send,
        P: Fn(usize, &Self) -> Result<T, E> + Sync,
        C: FnMut(usize, T) -> Result<PipelineStep, E>,
    {
        self.pipeline_indexed_with_scratch(
            n,
            window,
            || (),
            |index, ctx, ()| produce(index, ctx),
            consume,
        )
    }
}

/// Whether a parallel map waits for its worker slots.
#[derive(Clone, Copy)]
pub(super) enum MapWait {
    UntilAvailable,
    IfFree,
}

/// A shared-scheduler refusal or the original lowest-index task error.
#[derive(Debug, Error)]
pub enum MapExecutionError<E> {
    /// No work started without its required shared CPU reservation.
    #[error(transparent)]
    Admission(#[from] AdmissionError),
    /// The original task error, unchanged.
    #[error("worker execution failed: {0}")]
    Work(E),
}
