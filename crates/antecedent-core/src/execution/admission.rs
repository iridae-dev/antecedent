//! Execution context: shared CPU and RAM admission pool and leases.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::cancel::CancelFlag;
#[cfg(doc)]
use super::context::ExecutionContext;
use core::sync::atomic::{AtomicBool, Ordering};
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};
use thiserror::Error;

/// CPU slots and RAM bytes requested from an admission budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AdmissionRequest {
    pub(super) threads: u32,
    pub(super) bytes: u64,
}

impl AdmissionRequest {
    /// Request `threads` CPU slots and `bytes` of RAM.
    #[must_use]
    pub const fn new(threads: u32, bytes: u64) -> Self {
        Self { threads, bytes }
    }

    /// CPU slots.
    #[must_use]
    pub const fn threads(self) -> u32 {
        self.threads
    }

    /// RAM bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// Sum `parts` into one byte request.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::Overflow`] when the sum does not fit in `u64`. The sum is not wrapped.
    pub const fn try_sum(threads: u32, parts: &[u64]) -> Result<Self, AdmissionError> {
        let mut bytes = 0u64;
        let mut index = 0;
        while index < parts.len() {
            let Some(sum) = checked_byte_sum(bytes, parts[index]) else {
                return Err(AdmissionError::Overflow);
            };
            bytes = sum;
            index += 1;
        }
        Ok(Self::new(threads, bytes))
    }

    /// Request `count * width` bytes.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::Overflow`] when the product does not fit in `u64`.
    pub const fn try_product(threads: u32, count: u64, width: u64) -> Result<Self, AdmissionError> {
        let Some(bytes) = checked_byte_product(count, width) else {
            return Err(AdmissionError::Overflow);
        };
        Ok(Self::new(threads, bytes))
    }
}

/// CPU slots and RAM bytes still held by one lease, or still free in a pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct AdmissionRemainder {
    pub(super) threads: u32,
    pub(super) bytes: u64,
}

impl AdmissionRemainder {
    /// Remainder of `threads` slots and `bytes` of RAM.
    #[must_use]
    pub const fn new(threads: u32, bytes: u64) -> Self {
        Self { threads, bytes }
    }

    /// Nothing held.
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0, 0)
    }

    /// CPU slots.
    #[must_use]
    pub const fn threads(self) -> u32 {
        self.threads
    }

    /// RAM bytes.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.bytes
    }
}

/// Why an admission lease was not granted, or a size was not accepted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Error)]
#[non_exhaustive]
pub enum AdmissionError {
    /// The request is larger than the pool or parent remainder.
    #[error(
        "admission request of {requested_threads} threads and {requested_bytes} bytes exceeds \
         the remainder of {remaining_threads} threads and {remaining_bytes} bytes"
    )]
    ExceedsRemainder {
        /// CPU slots asked for.
        requested_threads: u32,
        /// RAM bytes asked for.
        requested_bytes: u64,
        /// CPU slots still available to this account.
        remaining_threads: u32,
        /// RAM bytes still available to this account.
        remaining_bytes: u64,
    },
    /// A sum or product did not fit. It was not wrapped.
    #[error("admission arithmetic overflowed and was refused")]
    Overflow,
    /// The wait ended and the budget still could not hold the request.
    #[error("admission wait of {bound:?} ended before the budget had room")]
    WaitBoundExceeded {
        /// Bound that elapsed.
        bound: Duration,
    },
    /// The study was cancelled before this admission completed.
    #[error("admission stopped because the study was cancelled")]
    Cancelled,
    /// The parent lease no longer holds a remainder.
    #[error("admission lease is already released")]
    Released,
}

/// `count * width`, or `None` when the product does not fit in `u64`.
#[must_use]
pub const fn checked_byte_product(count: u64, width: u64) -> Option<u64> {
    count.checked_mul(width)
}

/// `left + right`, or `None` when the sum does not fit in `u64`.
#[must_use]
pub const fn checked_byte_sum(left: u64, right: u64) -> Option<u64> {
    left.checked_add(right)
}

/// Shared CPU and RAM budget.
///
/// [`Self::available`] is what no open lease holds. Granting a lease decrements it. Two contexts
/// that install the same pool compete for that one remainder. The byte capacity is also the planner
/// limit ([`ExecutionContext::memory_limit_bytes`]). [`ExecutionContext::map_indexed`] consults
/// this pool for its one worker lease.
#[derive(Clone)]
pub struct AdmissionPool {
    pub(super) shared: Arc<SharedAdmission>,
}

pub(super) struct SharedAdmission {
    pub(super) state: Mutex<AdmissionState>,
    pub(super) released: Arc<Condvar>,
}

/// Free counters plus every open lease remainder equal the budget. That equality is the
/// enforcement: a descriptor that never changes these counters admits nothing.
pub(super) struct AdmissionState {
    pub(super) thread_budget: u32,
    pub(super) byte_budget: u64,
    pub(super) threads_free: u32,
    pub(super) bytes_free: u64,
    pub(super) leases: Vec<LeaseSlot>,
    pub(super) next_id: u64,
    pub(super) blocked_waiters: usize,
}

pub(super) struct LeaseSlot {
    pub(super) id: u64,
    pub(super) parent: Option<u64>,
    pub(super) threads: u32,
    pub(super) bytes: u64,
    pub(super) observation: Option<crate::execution_attempt::ObservedAdmission>,
}

pub(super) enum GrantError {
    Short(AdmissionError),
    Overflow,
}

impl AdmissionPool {
    /// Budget of `threads` CPU slots and `bytes` of RAM, all of it free.
    #[must_use]
    pub fn fixed(threads: u32, bytes: u64) -> Self {
        Self {
            shared: Arc::new(SharedAdmission {
                state: Mutex::new(AdmissionState {
                    thread_budget: threads,
                    byte_budget: bytes,
                    threads_free: threads,
                    bytes_free: bytes,
                    leases: Vec::new(),
                    next_id: 1,
                    blocked_waiters: 0,
                }),
                released: Arc::new(Condvar::new()),
            }),
        }
    }

    /// CPU and RAM no open lease holds.
    ///
    /// Cancellation does not free reservations still held by workers or retained buffers.
    /// Their owners release them after cleanup, so another job cannot overcommit live memory.
    #[must_use]
    pub fn available(&self) -> AdmissionRemainder {
        let guard = lock_mutex(&self.shared.state);
        AdmissionRemainder::new(guard.threads_free, guard.bytes_free)
    }

    /// CPU slots this pool was created with.
    #[must_use]
    pub fn thread_budget(&self) -> u32 {
        lock_mutex(&self.shared.state).thread_budget
    }

    /// RAM bytes this pool was created with.
    #[must_use]
    pub fn byte_budget(&self) -> u64 {
        lock_mutex(&self.shared.state).byte_budget
    }

    /// Studies blocked inside a bounded acquire.
    #[must_use]
    pub fn blocked_waiters(&self) -> usize {
        lock_mutex(&self.shared.state).blocked_waiters
    }
}

impl fmt::Debug for AdmissionPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let guard = lock_mutex(&self.shared.state);
        formatter
            .debug_struct("AdmissionPool")
            .field("thread_budget", &guard.thread_budget)
            .field("byte_budget", &guard.byte_budget)
            .field("threads_free", &guard.threads_free)
            .field("bytes_free", &guard.bytes_free)
            .field("open_leases", &guard.leases.len())
            .field("blocked_waiters", &guard.blocked_waiters)
            .finish()
    }
}

pub(super) fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(super) fn lease_index(leases: &[LeaseSlot], id: u64) -> Option<usize> {
    leases.iter().position(|slot| slot.id == id)
}

pub(super) fn exceeds(
    request: AdmissionRequest,
    remaining_threads: u32,
    remaining_bytes: u64,
) -> AdmissionError {
    AdmissionError::ExceedsRemainder {
        requested_threads: request.threads,
        requested_bytes: request.bytes,
        remaining_threads,
        remaining_bytes,
    }
}

pub(super) fn reserve(
    available_threads: u32,
    available_bytes: u64,
    request: AdmissionRequest,
) -> Result<(u32, u64), AdmissionError> {
    let Some(threads) = available_threads.checked_sub(request.threads) else {
        return Err(exceeds(request, available_threads, available_bytes));
    };
    let Some(bytes) = available_bytes.checked_sub(request.bytes) else {
        return Err(exceeds(request, available_threads, available_bytes));
    };
    Ok((threads, bytes))
}

pub(super) fn grant_root(
    state: &mut AdmissionState,
    request: AdmissionRequest,
) -> Result<u64, GrantError> {
    let Some(next_id) = state.next_id.checked_add(1) else {
        return Err(GrantError::Overflow);
    };
    let (threads_free, bytes_free) = match reserve(state.threads_free, state.bytes_free, request) {
        Ok(reserved) => reserved,
        Err(error) => return Err(GrantError::Short(error)),
    };
    state.threads_free = threads_free;
    state.bytes_free = bytes_free;
    Ok(push_lease(state, None, request, next_id))
}

pub(super) fn grant_child(
    state: &mut AdmissionState,
    parent_id: u64,
    request: AdmissionRequest,
) -> Result<u64, AdmissionError> {
    let Some(next_id) = state.next_id.checked_add(1) else {
        return Err(AdmissionError::Overflow);
    };
    let Some(index) = lease_index(&state.leases, parent_id) else {
        return Err(AdmissionError::Released);
    };
    let remaining_threads = state.leases[index].threads;
    let remaining_bytes = state.leases[index].bytes;
    let (threads, bytes) = reserve(remaining_threads, remaining_bytes, request)?;
    state.leases[index].threads = threads;
    state.leases[index].bytes = bytes;
    Ok(push_lease(state, Some(parent_id), request, next_id))
}

pub(super) fn push_lease(
    state: &mut AdmissionState,
    parent: Option<u64>,
    request: AdmissionRequest,
    next_id: u64,
) -> u64 {
    let id = state.next_id;
    state.next_id = next_id;
    let observation = match parent.and_then(|id| lease_index(&state.leases, id)) {
        Some(index) => state.leases[index]
            .observation
            .as_mut()
            .map(|held| held.split(request.threads, request.bytes)),
        None => {
            Some(crate::execution_attempt::ObservedAdmission::new(request.threads, request.bytes))
        }
    };
    state.leases.push(LeaseSlot {
        id,
        parent,
        threads: request.threads,
        bytes: request.bytes,
        observation,
    });
    id
}

pub(super) fn credit(
    state: &mut AdmissionState,
    parent: Option<u64>,
    threads: u32,
    bytes: u64,
) -> Result<(), AdmissionError> {
    let parent_index = parent.and_then(|id| lease_index(&state.leases, id));
    if let Some(index) = parent_index {
        let Some(held_threads) = state.leases[index].threads.checked_add(threads) else {
            return Err(AdmissionError::Overflow);
        };
        let Some(held_bytes) = state.leases[index].bytes.checked_add(bytes) else {
            return Err(AdmissionError::Overflow);
        };
        state.leases[index].threads = held_threads;
        state.leases[index].bytes = held_bytes;
        return Ok(());
    }
    let Some(threads_free) = state.threads_free.checked_add(threads) else {
        return Err(AdmissionError::Overflow);
    };
    if threads_free > state.thread_budget {
        return Err(AdmissionError::Overflow);
    }
    let Some(bytes_free) = state.bytes_free.checked_add(bytes) else {
        return Err(AdmissionError::Overflow);
    };
    if bytes_free > state.byte_budget {
        return Err(AdmissionError::Overflow);
    }
    state.threads_free = threads_free;
    state.bytes_free = bytes_free;
    Ok(())
}

/// Return slot `index` to its parent or the pool.
///
/// When the credit does not fit, the slot is left in the table with its holdings, nothing is
/// returned, and the overflow is reported. Nothing is dropped silently.
pub(super) fn close_slot(
    state: &mut AdmissionState,
    index: usize,
) -> Result<AdmissionRemainder, AdmissionError> {
    let id = state.leases[index].id;
    let parent = state.leases[index].parent;
    let threads = state.leases[index].threads;
    let bytes = state.leases[index].bytes;
    credit(state, parent, threads, bytes)?;
    for slot in &mut state.leases {
        if slot.parent == Some(id) {
            slot.parent = parent;
        }
    }
    let mut removed = state.leases.remove(index);
    if let Some(parent_index) = parent.and_then(|id| lease_index(&state.leases, id)) {
        if let (Some(held), Some(returned)) =
            (state.leases[parent_index].observation.as_mut(), removed.observation.as_mut())
        {
            held.merge(returned);
        }
    }
    Ok(AdmissionRemainder::new(threads, bytes))
}

pub(super) fn close_and_notify(
    shared: &SharedAdmission,
    id: u64,
) -> Result<AdmissionRemainder, AdmissionError> {
    let mut guard = lock_mutex(&shared.state);
    let Some(index) = lease_index(&guard.leases, id) else {
        return Ok(AdmissionRemainder::zero());
    };
    let remainder = close_slot(&mut guard, index)?;
    shared.released.notify_all();
    Ok(remainder)
}

/// A hold on CPU slots and RAM bytes.
///
/// Dropping or explicitly releasing the lease returns its remainder. Cancellation
/// refuses new child work and wakes waiters, but keeps live CPU and memory charged
/// until the owner finishes cleanup. Retained buffers keep their leases until drop.
#[must_use = "dropping the lease releases its CPU and RAM"]
pub struct AdmissionLease {
    pub(super) shared: Arc<SharedAdmission>,
    pub(super) id: u64,
    pub(super) flag: Arc<AtomicBool>,
    pub(super) active: AtomicBool,
}

impl AdmissionLease {
    /// Whether this reservation is charged to the specified shared admission account.
    #[must_use]
    pub fn belongs_to(&self, pool: &AdmissionPool) -> bool {
        Arc::ptr_eq(&self.shared, &pool.shared)
    }

    pub(super) fn new(shared: Arc<SharedAdmission>, id: u64, flag: Arc<AtomicBool>) -> Self {
        Self { shared, id, flag, active: AtomicBool::new(true) }
    }

    /// Include an existing reservation in the current native execution observation.
    /// This neither grants resources nor changes cancellation or release behavior.
    #[doc(hidden)]
    pub fn observe_current(&self) {
        let mut state = lock_mutex(&self.shared.state);
        if let Some(index) = lease_index(&state.leases, self.id) {
            if let Some(observation) = state.leases[index].observation.as_mut() {
                observation.attach_current();
            }
        }
    }

    /// Take a child lease from this lease's remainder.
    ///
    /// The child cannot use RAM or CPU the parent does not still hold, even when the pool has
    /// more free. The pool's free count stays put: the child is a portion of the parent.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::ExceedsRemainder`] when the request is larger than this lease's
    /// remainder. [`AdmissionError::Cancelled`] when the study was cancelled.
    /// [`AdmissionError::Released`] when this lease is already gone.
    /// [`AdmissionError::Overflow`] when naming another lease does not fit in `u64`.
    pub fn acquire_child(&self, threads: u32, bytes: u64) -> Result<Self, AdmissionError> {
        self.acquire_child_request(AdmissionRequest::new(threads, bytes))
    }

    /// Child lease of `count * width` bytes, refused when the product overflows.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::Overflow`] when `count * width` does not fit in `u64`. Otherwise the
    /// errors of [`Self::acquire_child`].
    pub fn acquire_child_product(
        &self,
        threads: u32,
        count: u64,
        width: u64,
    ) -> Result<Self, AdmissionError> {
        self.acquire_child_request(AdmissionRequest::try_product(threads, count, width)?)
    }

    /// Child lease for a request already checked for overflow.
    ///
    /// # Errors
    ///
    /// See [`Self::acquire_child`].
    pub fn acquire_child_request(&self, request: AdmissionRequest) -> Result<Self, AdmissionError> {
        let mut guard = lock_mutex(&self.shared.state);
        if self.flag.load(Ordering::SeqCst) {
            return Err(AdmissionError::Cancelled);
        }
        if lease_index(&guard.leases, self.id).is_none() {
            self.active.store(false, Ordering::SeqCst);
            return Err(AdmissionError::Released);
        }
        let id = grant_child(&mut guard, self.id, request)?;
        Ok(Self::new(Arc::clone(&self.shared), id, Arc::clone(&self.flag)))
    }

    /// Root lease on the pool that issued `self`.
    ///
    /// The new lease is not a portion of this remainder. This remainder stays
    /// put. The pool's free counters pay for it, under the same cancellation flag.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::ExceedsRemainder`] when the pool's free counters are short.
    /// [`AdmissionError::Cancelled`] when the study was cancelled.
    /// [`AdmissionError::Released`] when this lease is already gone.
    /// [`AdmissionError::Overflow`] when naming another lease does not fit in `u64`.
    pub(crate) fn acquire_from_pool(
        &self,
        threads: u32,
        bytes: u64,
    ) -> Result<Self, AdmissionError> {
        let mut guard = lock_mutex(&self.shared.state);
        if self.flag.load(Ordering::SeqCst) {
            return Err(AdmissionError::Cancelled);
        }
        if lease_index(&guard.leases, self.id).is_none() {
            self.active.store(false, Ordering::SeqCst);
            return Err(AdmissionError::Released);
        }
        let request = AdmissionRequest::new(threads, bytes);
        let id = match grant_root(&mut guard, request) {
            Ok(id) => id,
            Err(GrantError::Overflow) => return Err(AdmissionError::Overflow),
            Err(GrantError::Short(error)) => return Err(error),
        };
        Ok(Self::new(Arc::clone(&self.shared), id, Arc::clone(&self.flag)))
    }

    /// CPU and RAM this lease still holds, including during cancellation cleanup.
    #[must_use]
    pub fn remainder(&self) -> AdmissionRemainder {
        let guard = lock_mutex(&self.shared.state);
        if let Some(index) = lease_index(&guard.leases, self.id) {
            let slot = &guard.leases[index];
            AdmissionRemainder::new(slot.threads, slot.bytes)
        } else {
            self.active.store(false, Ordering::SeqCst);
            AdmissionRemainder::zero()
        }
    }

    /// Return this lease's remainder to its parent, or to the pool when it is a root.
    ///
    /// Outstanding children keep their portion. That portion is not part of the returned
    /// remainder; each child returns it when that child is released.
    ///
    /// # Panics
    ///
    /// Panics when the credit does not fit ([`Self::try_release`] reports it instead). The
    /// lease then stays in the pool's table with its holdings charged.
    #[must_use = "the remainder is the CPU and RAM returned to the parent"]
    pub fn release(self) -> AdmissionRemainder {
        match self.try_release() {
            Ok(remainder) => remainder,
            Err(error) => panic!("admission release failed: {error}"),
        }
    }

    /// [`Self::release`], reporting a credit that does not fit instead of panicking.
    ///
    /// # Errors
    ///
    /// [`AdmissionError::Overflow`] when returning the holdings would overflow the parent or
    /// the pool counters. The lease stays in the table with its holdings still charged.
    pub fn try_release(self) -> Result<AdmissionRemainder, AdmissionError> {
        if !self.active.swap(false, Ordering::SeqCst) {
            return Ok(AdmissionRemainder::zero());
        }
        close_and_notify(&self.shared, self.id)
    }
}

impl Drop for AdmissionLease {
    fn drop(&mut self) {
        if self.active.swap(false, Ordering::SeqCst) {
            if let Err(error) = close_and_notify(&self.shared, self.id) {
                // The lease stays charged in the table. An accounting break is loud, but
                // never a second panic while one is already unwinding.
                assert!(std::thread::panicking(), "admission release failed: {error}");
            }
        }
    }
}

impl fmt::Debug for AdmissionLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AdmissionLease")
            .field("id", &self.id)
            .field("active", &self.active.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
pub(super) enum AdmissionWait {
    Immediate,
    Bounded(Duration),
    UntilAvailable,
}

pub(super) fn admit(
    shared: &Arc<SharedAdmission>,
    request: AdmissionRequest,
    flag: &Arc<AtomicBool>,
    waiters: &CancelFlag,
    wait: AdmissionWait,
) -> Result<AdmissionLease, AdmissionError> {
    waiters.register_pool(shared);
    let mut guard = lock_mutex(&shared.state);
    let limit = match wait {
        AdmissionWait::Bounded(bound) => {
            let deadline = Instant::now().checked_add(bound).ok_or(AdmissionError::Overflow)?;
            Some((deadline, bound))
        }
        AdmissionWait::Immediate | AdmissionWait::UntilAvailable => None,
    };
    loop {
        if flag.load(Ordering::SeqCst) {
            return Err(AdmissionError::Cancelled);
        }
        match grant_root(&mut guard, request) {
            Ok(id) => {
                return Ok(AdmissionLease::new(Arc::clone(shared), id, Arc::clone(flag)));
            }
            Err(GrantError::Overflow) => return Err(AdmissionError::Overflow),
            Err(GrantError::Short(refusal)) => {
                let never_fits =
                    request.threads > guard.thread_budget || request.bytes > guard.byte_budget;
                if never_fits && matches!(wait, AdmissionWait::UntilAvailable) {
                    return Err(refusal);
                }
                guard = match wait {
                    AdmissionWait::Immediate => return Err(refusal),
                    AdmissionWait::Bounded(_) => wait_or_stop(shared, guard, refusal, limit)?,
                    AdmissionWait::UntilAvailable => block_on_release(shared, guard, None)?,
                };
            }
        }
    }
}

pub(super) fn wait_or_stop<'a>(
    shared: &'a SharedAdmission,
    guard: MutexGuard<'a, AdmissionState>,
    refusal: AdmissionError,
    limit: Option<(Instant, Duration)>,
) -> Result<MutexGuard<'a, AdmissionState>, AdmissionError> {
    let Some((deadline, bound)) = limit else {
        return Err(refusal);
    };
    if Instant::now() >= deadline {
        return Err(AdmissionError::WaitBoundExceeded { bound });
    }
    wait_for_room(shared, guard, deadline)
}

pub(super) fn wait_for_room<'a>(
    shared: &'a SharedAdmission,
    guard: MutexGuard<'a, AdmissionState>,
    deadline: Instant,
) -> Result<MutexGuard<'a, AdmissionState>, AdmissionError> {
    let timeout = deadline.saturating_duration_since(Instant::now());
    block_on_release(shared, guard, Some(timeout))
}

pub(super) fn block_on_release<'a>(
    shared: &'a SharedAdmission,
    mut guard: MutexGuard<'a, AdmissionState>,
    timeout: Option<Duration>,
) -> Result<MutexGuard<'a, AdmissionState>, AdmissionError> {
    let Some(waiting) = guard.blocked_waiters.checked_add(1) else {
        return Err(AdmissionError::Overflow);
    };
    guard.blocked_waiters = waiting;
    let mut locked = match timeout {
        Some(timeout) => {
            let (locked, _timed_out) = shared
                .released
                .wait_timeout(guard, timeout)
                .unwrap_or_else(PoisonError::into_inner);
            locked
        }
        None => shared.released.wait(guard).unwrap_or_else(PoisonError::into_inner),
    };
    locked.blocked_waiters = locked.blocked_waiters.saturating_sub(1);
    Ok(locked)
}
