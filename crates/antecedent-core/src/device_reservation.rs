//! Device-memory and spill reservation.
//!
//! This is a byte account. It does not allocate device memory, spill storage,
//! or wait, and it links no device library. [`DeviceReservationPool::fixed`]
//! names the device-byte and spill-byte budgets. A lease decrements both.
//! [`DeviceReservationLease::release`] returns that remainder after successful
//! work, [`DeviceReservationLease::cancel`] and
//! [`DeviceReservationLease::release_on_cancel`] return it on cancellation, and
//! [`Drop`] returns it on unwind, including a worker that fails while holding
//! the lease. A request larger than the remainder is refused and leaves the
//! account unchanged. Sums and products that do not fit in `u64` are refused
//! rather than wrapped.
//!
//! [`crate::execution::AdmissionPool`] is a separate account and is not decremented here.
//!
//! No accelerator is promoted. [`select_compute`] refuses one and does not grant
//! device bytes. A small serial job on [`ComputeSelection::Cpu`] leaves this
//! account unchanged. Its CPU and RAM lease, cancellation, and scratch cleanup
//! stay on the admission pool.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use thiserror::Error;

use crate::execution::CancellationToken;

mod cpu_admission;
pub use cpu_admission::{
    ArithmeticPath, ComputeSelection, CpuAdmission, CpuAdmissionError, FoldAdmission,
    LearnerAdmission, PredictionOrderError, admit_fold, nested_thread_request, neural_baseline,
    place_physical_predictions, retain_scientific_f64, scientific_arithmetic, scientific_tolerance,
    select_compute, within_declared_tolerance,
};

/// Device bytes and spill bytes requested from a reservation pool.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct DeviceReservationRequest {
    device_bytes: u64,
    spill_bytes: u64,
}

impl DeviceReservationRequest {
    /// Request `device_bytes` of device memory and `spill_bytes` of spill.
    #[must_use]
    pub const fn new(device_bytes: u64, spill_bytes: u64) -> Self {
        Self { device_bytes, spill_bytes }
    }

    /// Device bytes.
    #[must_use]
    pub const fn device_bytes(self) -> u64 {
        self.device_bytes
    }

    /// Spill bytes.
    #[must_use]
    pub const fn spill_bytes(self) -> u64 {
        self.spill_bytes
    }

    /// Sum device parts and spill parts into one request.
    ///
    /// # Errors
    ///
    /// [`DeviceReservationError::Overflow`] when either sum does not fit in
    /// `u64`. The sum is not wrapped.
    pub const fn try_sum(
        device_parts: &[u64],
        spill_parts: &[u64],
    ) -> Result<Self, DeviceReservationError> {
        let Some(device_bytes) = sum_parts(device_parts) else {
            return Err(DeviceReservationError::Overflow);
        };
        let Some(spill_bytes) = sum_parts(spill_parts) else {
            return Err(DeviceReservationError::Overflow);
        };
        Ok(Self::new(device_bytes, spill_bytes))
    }

    /// Request `device_count * device_width` device bytes and
    /// `spill_count * spill_width` spill bytes.
    ///
    /// # Errors
    ///
    /// [`DeviceReservationError::Overflow`] when either product does not fit in
    /// `u64`. The product is not wrapped.
    pub const fn try_product(
        device_count: u64,
        device_width: u64,
        spill_count: u64,
        spill_width: u64,
    ) -> Result<Self, DeviceReservationError> {
        let Some(device_bytes) = device_count.checked_mul(device_width) else {
            return Err(DeviceReservationError::Overflow);
        };
        let Some(spill_bytes) = spill_count.checked_mul(spill_width) else {
            return Err(DeviceReservationError::Overflow);
        };
        Ok(Self::new(device_bytes, spill_bytes))
    }
}

impl fmt::Debug for DeviceReservationRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceReservationRequest")
            .field("device_bytes", &self.device_bytes)
            .field("spill_bytes", &self.spill_bytes)
            .finish_non_exhaustive()
    }
}

/// Device bytes and spill bytes still held by one lease, or still free in a pool.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct DeviceReservationRemainder {
    device_bytes: u64,
    spill_bytes: u64,
}

impl DeviceReservationRemainder {
    /// Remainder of `device_bytes` and `spill_bytes`.
    #[must_use]
    pub const fn new(device_bytes: u64, spill_bytes: u64) -> Self {
        Self { device_bytes, spill_bytes }
    }

    /// Nothing held.
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0, 0)
    }

    /// Device bytes.
    #[must_use]
    pub const fn device_bytes(self) -> u64 {
        self.device_bytes
    }

    /// Spill bytes.
    #[must_use]
    pub const fn spill_bytes(self) -> u64 {
        self.spill_bytes
    }
}

impl fmt::Debug for DeviceReservationRemainder {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceReservationRemainder")
            .field("device_bytes", &self.device_bytes)
            .field("spill_bytes", &self.spill_bytes)
            .finish_non_exhaustive()
    }
}

/// Why a device reservation was not granted, or a size was not accepted.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Error)]
#[non_exhaustive]
pub enum DeviceReservationError {
    /// The request is larger than the pool or parent remainder.
    #[error(
        "device reservation of {requested_device_bytes} device bytes and \
         {requested_spill_bytes} spill bytes exceeds the remainder of \
         {remaining_device_bytes} device bytes and {remaining_spill_bytes} spill bytes"
    )]
    ExceedsRemainder {
        /// Device bytes asked for.
        requested_device_bytes: u64,
        /// Spill bytes asked for.
        requested_spill_bytes: u64,
        /// Device bytes still available to this account.
        remaining_device_bytes: u64,
        /// Spill bytes still available to this account.
        remaining_spill_bytes: u64,
    },
    /// A sum or product did not fit. It was not wrapped.
    #[error("device reservation arithmetic overflowed and was refused")]
    Overflow,
    /// Cancellation was requested before the reservation was granted.
    #[error("device reservation stopped because the work was cancelled")]
    Cancelled,
    /// The parent lease no longer holds a remainder.
    #[error("device reservation lease is already released")]
    Released,
    /// An accelerator was requested. None is promoted, so no device bytes are reserved.
    #[error("accelerator selection is refused because no device is promoted")]
    Unpromoted,
}

/// Shared device-byte and spill-byte budget.
///
/// Clones name the same account. [`Self::available`] is what no open lease holds.
/// Granting a lease decrements both counters. Two leases that together exceed the
/// budget refuse the second. [`crate::execution::AdmissionPool`] is not consulted
/// and is not decremented.
#[derive(Clone)]
pub struct DeviceReservationPool {
    shared: Arc<SharedReservation>,
}

/// Free counters plus every open lease equal the budget.
struct SharedReservation {
    state: Mutex<PoolState>,
}

struct PoolState {
    device_budget: u64,
    spill_budget: u64,
    device_free: u64,
    spill_free: u64,
    leases: Vec<LeaseSlot>,
    next_id: u64,
}

struct LeaseSlot {
    id: u64,
    parent: Option<u64>,
    device_bytes: u64,
    spill_bytes: u64,
}

impl DeviceReservationPool {
    /// Budget of `device_bytes` and `spill_bytes`, all of it free.
    #[must_use]
    pub fn fixed(device_bytes: u64, spill_bytes: u64) -> Self {
        Self {
            shared: Arc::new(SharedReservation {
                state: Mutex::new(PoolState {
                    device_budget: device_bytes,
                    spill_budget: spill_bytes,
                    device_free: device_bytes,
                    spill_free: spill_bytes,
                    leases: Vec::new(),
                    next_id: 1,
                }),
            }),
        }
    }

    /// Device bytes and spill bytes no open lease holds.
    #[must_use]
    pub fn available(&self) -> DeviceReservationRemainder {
        let (device_bytes, spill_bytes) = {
            let guard = lock_state(&self.shared.state);
            (guard.device_free, guard.spill_free)
        };
        DeviceReservationRemainder::new(device_bytes, spill_bytes)
    }

    /// Device bytes this pool was created with.
    #[must_use]
    pub fn device_budget(&self) -> u64 {
        lock_state(&self.shared.state).device_budget
    }

    /// Spill bytes this pool was created with.
    #[must_use]
    pub fn spill_budget(&self) -> u64 {
        lock_state(&self.shared.state).spill_budget
    }

    /// Take a root lease, decrementing both remainders.
    ///
    /// `cancelled` is checked before any counter changes. A refusal leaves the
    /// pool as it was.
    ///
    /// # Errors
    ///
    /// [`DeviceReservationError::Cancelled`] when `cancelled` is set.
    /// [`DeviceReservationError::ExceedsRemainder`] when the request is larger
    /// than the free remainder. [`DeviceReservationError::Overflow`] when naming
    /// another lease does not fit in `u64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use antecedent_core::device_reservation::DeviceReservationPool;
    ///
    /// let pool = DeviceReservationPool::fixed(100, 40);
    /// let lease = pool.try_grant(30, 10, false).unwrap();
    /// assert_eq!(pool.available().device_bytes(), 70);
    /// assert_eq!(lease.release().spill_bytes(), 10);
    /// assert_eq!(pool.available().spill_bytes(), 40);
    /// ```
    #[must_use = "a refused reservation is not permission to allocate"]
    pub fn try_grant(
        &self,
        device_bytes: u64,
        spill_bytes: u64,
        cancelled: bool,
    ) -> Result<DeviceReservationLease, DeviceReservationError> {
        self.try_grant_request(DeviceReservationRequest::new(device_bytes, spill_bytes), cancelled)
    }

    /// Root lease for a request already checked for overflow.
    ///
    /// # Errors
    ///
    /// See [`Self::try_grant`].
    #[must_use = "a refused reservation is not permission to allocate"]
    pub fn try_grant_request(
        &self,
        request: DeviceReservationRequest,
        cancelled: bool,
    ) -> Result<DeviceReservationLease, DeviceReservationError> {
        if cancelled {
            return Err(DeviceReservationError::Cancelled);
        }
        let mut guard = lock_state(&self.shared.state);
        let id = grant_root(&mut guard, request)?;
        drop(guard);
        Ok(DeviceReservationLease::new(Arc::clone(&self.shared), id))
    }

    /// [`Self::try_grant`] after observing `token` once.
    ///
    /// A cancelled token grants nothing and leaves both counters unchanged.
    ///
    /// # Errors
    ///
    /// See [`Self::try_grant`]. [`DeviceReservationError::Cancelled`] when
    /// [`CancellationToken::is_cancelled`] is set.
    #[must_use = "a refused reservation is not permission to allocate"]
    pub fn try_grant_with_token(
        &self,
        device_bytes: u64,
        spill_bytes: u64,
        token: &CancellationToken,
    ) -> Result<DeviceReservationLease, DeviceReservationError> {
        self.try_grant(device_bytes, spill_bytes, token.is_cancelled())
    }

    /// Root lease of a device product and a spill product.
    ///
    /// # Errors
    ///
    /// [`DeviceReservationError::Overflow`] when either product does not fit in
    /// `u64`. The pool is not decremented. Otherwise the errors of [`Self::try_grant`].
    #[must_use = "a refused reservation is not permission to allocate"]
    pub fn try_grant_product(
        &self,
        device_count: u64,
        device_width: u64,
        spill_count: u64,
        spill_width: u64,
        cancelled: bool,
    ) -> Result<DeviceReservationLease, DeviceReservationError> {
        let request = DeviceReservationRequest::try_product(
            device_count,
            device_width,
            spill_count,
            spill_width,
        )?;
        self.try_grant_request(request, cancelled)
    }
}

impl fmt::Debug for DeviceReservationPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let guard = lock_state(&self.shared.state);
        let device_budget = guard.device_budget;
        let spill_budget = guard.spill_budget;
        let device_free = guard.device_free;
        let spill_free = guard.spill_free;
        let open_leases = guard.leases.len();
        drop(guard);
        formatter
            .debug_struct("DeviceReservationPool")
            .field("device_budget", &device_budget)
            .field("spill_budget", &spill_budget)
            .field("device_free", &device_free)
            .field("spill_free", &spill_free)
            .field("open_leases", &open_leases)
            .finish_non_exhaustive()
    }
}

/// A hold on device bytes and spill bytes.
///
/// [`Self::release`] returns the remainder after successful work. [`Self::cancel`]
/// and [`Self::release_on_cancel`] return it on cancellation. [`Drop`] returns it
/// when the holder unwinds, including a worker that panics or otherwise fails.
/// Outstanding children keep their portion; each child returns that portion when
/// that child is released.
#[must_use = "dropping the lease returns its device and spill bytes"]
pub struct DeviceReservationLease {
    shared: Arc<SharedReservation>,
    id: u64,
    active: AtomicBool,
}

impl DeviceReservationLease {
    fn new(shared: Arc<SharedReservation>, id: u64) -> Self {
        Self { shared, id, active: AtomicBool::new(true) }
    }

    /// Take a child lease from this lease's remainder.
    ///
    /// The child cannot use device or spill bytes the parent does not still hold,
    /// even when the pool has more free. The pool's free count stays put: the child
    /// is a portion of the parent. `cancelled` is checked before either remainder
    /// changes.
    ///
    /// # Errors
    ///
    /// [`DeviceReservationError::Cancelled`] when `cancelled` is set.
    /// [`DeviceReservationError::ExceedsRemainder`] when the request is larger than
    /// this lease's remainder. [`DeviceReservationError::Released`] when this lease
    /// is already gone. [`DeviceReservationError::Overflow`] when naming another
    /// lease does not fit in `u64`.
    #[must_use = "a refused reservation is not permission to allocate"]
    pub fn acquire_child(
        &self,
        device_bytes: u64,
        spill_bytes: u64,
        cancelled: bool,
    ) -> Result<Self, DeviceReservationError> {
        self.acquire_child_request(
            DeviceReservationRequest::new(device_bytes, spill_bytes),
            cancelled,
        )
    }

    /// Child lease for a request already checked for overflow.
    ///
    /// # Errors
    ///
    /// See [`Self::acquire_child`].
    #[must_use = "a refused reservation is not permission to allocate"]
    pub fn acquire_child_request(
        &self,
        request: DeviceReservationRequest,
        cancelled: bool,
    ) -> Result<Self, DeviceReservationError> {
        if cancelled {
            return Err(DeviceReservationError::Cancelled);
        }
        let mut guard = lock_state(&self.shared.state);
        if !self.active.load(Ordering::SeqCst) || lease_index(&guard.leases, self.id).is_none() {
            drop(guard);
            self.active.store(false, Ordering::SeqCst);
            return Err(DeviceReservationError::Released);
        }
        let id = grant_child(&mut guard, self.id, request)?;
        drop(guard);
        Ok(Self::new(Arc::clone(&self.shared), id))
    }

    /// Child lease of a device product and a spill product.
    ///
    /// # Errors
    ///
    /// [`DeviceReservationError::Overflow`] when either product does not fit in
    /// `u64`. This lease is not decremented. Otherwise the errors of
    /// [`Self::acquire_child`].
    #[must_use = "a refused reservation is not permission to allocate"]
    pub fn acquire_child_product(
        &self,
        device_count: u64,
        device_width: u64,
        spill_count: u64,
        spill_width: u64,
        cancelled: bool,
    ) -> Result<Self, DeviceReservationError> {
        let request = DeviceReservationRequest::try_product(
            device_count,
            device_width,
            spill_count,
            spill_width,
        )?;
        self.acquire_child_request(request, cancelled)
    }

    /// Device and spill bytes this lease still holds.
    #[must_use]
    pub fn remainder(&self) -> DeviceReservationRemainder {
        let guard = lock_state(&self.shared.state);
        if let Some(index) = lease_index(&guard.leases, self.id) {
            let device_bytes = guard.leases[index].device_bytes;
            let spill_bytes = guard.leases[index].spill_bytes;
            drop(guard);
            DeviceReservationRemainder::new(device_bytes, spill_bytes)
        } else {
            drop(guard);
            self.active.store(false, Ordering::SeqCst);
            DeviceReservationRemainder::zero()
        }
    }

    /// Return this lease's remainder to its parent, or to the pool when it is a root.
    ///
    /// Outstanding children keep their portion. That portion is not part of the
    /// returned remainder; each child returns it when that child is released.
    #[must_use = "the remainder is the device and spill bytes returned to the pool"]
    pub fn release(self) -> DeviceReservationRemainder {
        relinquish(&self.shared, self.id, &self.active)
    }

    /// Return this lease's remainder because the work was cancelled.
    #[must_use = "the remainder is the device and spill bytes returned to the pool"]
    pub fn cancel(self) -> DeviceReservationRemainder {
        self.release()
    }

    /// Return this lease's remainder when `cancelled` is set.
    ///
    /// `false` leaves the lease and the pool unchanged and returns
    /// [`DeviceReservationRemainder::zero`]. A later call after the bytes were
    /// returned also yields zero.
    #[must_use = "the remainder is the device and spill bytes returned to the pool"]
    pub fn release_on_cancel(&self, cancelled: bool) -> DeviceReservationRemainder {
        if !cancelled {
            return DeviceReservationRemainder::zero();
        }
        relinquish(&self.shared, self.id, &self.active)
    }

    /// [`Self::release_on_cancel`] after observing `token` once.
    #[must_use = "the remainder is the device and spill bytes returned to the pool"]
    pub fn release_on_token(&self, token: &CancellationToken) -> DeviceReservationRemainder {
        self.release_on_cancel(token.is_cancelled())
    }
}

impl Drop for DeviceReservationLease {
    fn drop(&mut self) {
        let _ = relinquish(&self.shared, self.id, &self.active);
    }
}

impl fmt::Debug for DeviceReservationLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceReservationLease")
            .field("id", &self.id)
            .field("active", &self.active.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

const fn sum_parts(parts: &[u64]) -> Option<u64> {
    let mut total = 0u64;
    let mut index = 0;
    while index < parts.len() {
        let Some(sum) = total.checked_add(parts[index]) else {
            return None;
        };
        total = sum;
        index += 1;
    }
    Some(total)
}

fn lock_state(mutex: &Mutex<PoolState>) -> MutexGuard<'_, PoolState> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

const fn lease_index(leases: &[LeaseSlot], id: u64) -> Option<usize> {
    let mut index = 0;
    while index < leases.len() {
        if leases[index].id == id {
            return Some(index);
        }
        index += 1;
    }
    None
}

const fn exceeds(
    request: DeviceReservationRequest,
    remaining_device_bytes: u64,
    remaining_spill_bytes: u64,
) -> DeviceReservationError {
    DeviceReservationError::ExceedsRemainder {
        requested_device_bytes: request.device_bytes,
        requested_spill_bytes: request.spill_bytes,
        remaining_device_bytes,
        remaining_spill_bytes,
    }
}

const fn reserve(
    available_device: u64,
    available_spill: u64,
    request: DeviceReservationRequest,
) -> Result<(u64, u64), DeviceReservationError> {
    let Some(device_bytes) = available_device.checked_sub(request.device_bytes) else {
        return Err(exceeds(request, available_device, available_spill));
    };
    let Some(spill_bytes) = available_spill.checked_sub(request.spill_bytes) else {
        return Err(exceeds(request, available_device, available_spill));
    };
    Ok((device_bytes, spill_bytes))
}

fn grant_root(
    state: &mut PoolState,
    request: DeviceReservationRequest,
) -> Result<u64, DeviceReservationError> {
    let (device_free, spill_free) = reserve(state.device_free, state.spill_free, request)?;
    let Some(next_id) = state.next_id.checked_add(1) else {
        return Err(DeviceReservationError::Overflow);
    };
    state.device_free = device_free;
    state.spill_free = spill_free;
    let id = push_lease(state, None, request, next_id);
    debug_accounted(state);
    Ok(id)
}

fn grant_child(
    state: &mut PoolState,
    parent_id: u64,
    request: DeviceReservationRequest,
) -> Result<u64, DeviceReservationError> {
    let Some(index) = lease_index(&state.leases, parent_id) else {
        return Err(DeviceReservationError::Released);
    };
    let (device_bytes, spill_bytes) =
        reserve(state.leases[index].device_bytes, state.leases[index].spill_bytes, request)?;
    let Some(next_id) = state.next_id.checked_add(1) else {
        return Err(DeviceReservationError::Overflow);
    };
    state.leases[index].device_bytes = device_bytes;
    state.leases[index].spill_bytes = spill_bytes;
    let id = push_lease(state, Some(parent_id), request, next_id);
    debug_accounted(state);
    Ok(id)
}

fn push_lease(
    state: &mut PoolState,
    parent: Option<u64>,
    request: DeviceReservationRequest,
    next_id: u64,
) -> u64 {
    let id = state.next_id;
    state.next_id = next_id;
    state.leases.push(LeaseSlot {
        id,
        parent,
        device_bytes: request.device_bytes,
        spill_bytes: request.spill_bytes,
    });
    id
}

fn credit(
    state: &mut PoolState,
    parent: Option<u64>,
    device_bytes: u64,
    spill_bytes: u64,
) -> Result<(), DeviceReservationError> {
    if let Some(index) = parent.and_then(|id| lease_index(&state.leases, id)) {
        let Some(held_device) = state.leases[index].device_bytes.checked_add(device_bytes) else {
            return Err(DeviceReservationError::Overflow);
        };
        let Some(held_spill) = state.leases[index].spill_bytes.checked_add(spill_bytes) else {
            return Err(DeviceReservationError::Overflow);
        };
        state.leases[index].device_bytes = held_device;
        state.leases[index].spill_bytes = held_spill;
        return Ok(());
    }
    let Some(device_free) = state.device_free.checked_add(device_bytes) else {
        return Err(DeviceReservationError::Overflow);
    };
    if device_free > state.device_budget {
        return Err(DeviceReservationError::Overflow);
    }
    let Some(spill_free) = state.spill_free.checked_add(spill_bytes) else {
        return Err(DeviceReservationError::Overflow);
    };
    if spill_free > state.spill_budget {
        return Err(DeviceReservationError::Overflow);
    }
    state.device_free = device_free;
    state.spill_free = spill_free;
    Ok(())
}

fn close_slot(state: &mut PoolState, index: usize) -> DeviceReservationRemainder {
    let id = state.leases[index].id;
    let parent = state.leases[index].parent;
    let device_bytes = state.leases[index].device_bytes;
    let spill_bytes = state.leases[index].spill_bytes;
    if let Err(error) = credit(state, parent, device_bytes, spill_bytes) {
        panic!("device reservation credit overflow ({error}); lease {id} stays open");
    }
    for slot in &mut state.leases {
        if slot.parent == Some(id) {
            slot.parent = parent;
        }
    }
    state.leases.remove(index);
    debug_accounted(state);
    DeviceReservationRemainder::new(device_bytes, spill_bytes)
}

fn relinquish(
    shared: &SharedReservation,
    id: u64,
    active: &AtomicBool,
) -> DeviceReservationRemainder {
    if !active.swap(false, Ordering::SeqCst) {
        return DeviceReservationRemainder::zero();
    }
    let mut guard = lock_state(&shared.state);
    let Some(index) = lease_index(&guard.leases, id) else {
        return DeviceReservationRemainder::zero();
    };
    close_slot(&mut guard, index)
}

#[cfg_attr(
    not(debug_assertions),
    allow(unused_variables, reason = "the budget check is debug-only")
)]
fn debug_accounted(state: &PoolState) {
    #[cfg(debug_assertions)]
    debug_assert!(
        bytes_cover_budget(state),
        "device and spill reservations must sum to the pool budget"
    );
}

#[cfg(debug_assertions)]
fn bytes_cover_budget(state: &PoolState) -> bool {
    let mut device = state.device_free;
    let mut spill = state.spill_free;
    for slot in &state.leases {
        let Some(next_device) = device.checked_add(slot.device_bytes) else {
            return false;
        };
        let Some(next_spill) = spill.checked_add(slot.spill_bytes) else {
            return false;
        };
        device = next_device;
        spill = next_spill;
    }
    device == state.device_budget && spill == state.spill_budget
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::thread;

    use super::*;
    use crate::execution::{AdmissionPool, CancellationToken};

    #[test]
    fn credit_overflow_panics_and_leaves_the_lease() {
        let mut state = PoolState {
            device_budget: 0,
            spill_budget: 0,
            device_free: 0,
            spill_free: 0,
            leases: vec![LeaseSlot { id: 7, parent: None, device_bytes: 1, spill_bytes: 0 }],
            next_id: 8,
        };
        let panicked = catch_unwind(AssertUnwindSafe(|| close_slot(&mut state, 0)));
        assert!(panicked.is_err());
        assert_eq!(state.leases.len(), 1);
        assert_eq!(state.leases[0].id, 7);
    }

    #[test]
    fn grant_and_release() {
        let pool = DeviceReservationPool::fixed(100, 40);
        let first = pool.try_grant(60, 25, false).unwrap();
        let second = pool.try_grant(40, 15, false).unwrap();
        assert_eq!(pool.available(), DeviceReservationRemainder::zero());
        assert_eq!(first.remainder(), DeviceReservationRemainder::new(60, 25));
        assert_eq!(first.release(), DeviceReservationRemainder::new(60, 25));
        assert_eq!(pool.available(), DeviceReservationRemainder::new(60, 25));
        assert_eq!(second.release(), DeviceReservationRemainder::new(40, 15));
        assert_eq!(pool.available(), DeviceReservationRemainder::new(100, 40));
        assert_eq!(pool.device_budget(), 100);
        assert_eq!(pool.spill_budget(), 40);
    }

    #[test]
    fn refusal_does_not_decrement() {
        let pool = DeviceReservationPool::fixed(100, 40);
        let first = pool.try_grant(80, 30, false).unwrap();
        assert_eq!(
            pool.try_grant(30, 5, false).unwrap_err(),
            DeviceReservationError::ExceedsRemainder {
                requested_device_bytes: 30,
                requested_spill_bytes: 5,
                remaining_device_bytes: 20,
                remaining_spill_bytes: 10,
            }
        );
        assert_eq!(
            pool.try_grant(4, 11, false).unwrap_err(),
            DeviceReservationError::ExceedsRemainder {
                requested_device_bytes: 4,
                requested_spill_bytes: 11,
                remaining_device_bytes: 20,
                remaining_spill_bytes: 10,
            }
        );
        assert_eq!(pool.available(), DeviceReservationRemainder::new(20, 10));
        assert_eq!(first.remainder(), DeviceReservationRemainder::new(80, 30));
        drop(first);
        assert_eq!(pool.available(), DeviceReservationRemainder::new(100, 40));
        assert_eq!(
            pool.try_grant(30, 5, false).unwrap().release(),
            DeviceReservationRemainder::new(30, 5)
        );
    }

    #[test]
    fn drop_returns_bytes() {
        let pool = DeviceReservationPool::fixed(50, 25);
        let lease = pool.try_grant(50, 25, false).unwrap();
        assert_eq!(pool.available(), DeviceReservationRemainder::zero());
        drop(lease);
        assert_eq!(pool.available(), DeviceReservationRemainder::new(50, 25));
        let in_panic = pool.clone();
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            let _lease = in_panic.try_grant(4, 3, false).expect("grant");
            panic!("worker unwind");
        }));
        assert!(panicked.is_err());
        assert_eq!(pool.available(), DeviceReservationRemainder::new(50, 25));
    }

    #[test]
    fn cancel_releases() {
        let pool = DeviceReservationPool::fixed(80, 16);
        let token = CancellationToken::new();
        token.cancel();
        assert_eq!(
            pool.try_grant_with_token(10, 4, &token).unwrap_err(),
            DeviceReservationError::Cancelled
        );
        assert_eq!(pool.try_grant(10, 4, true).unwrap_err(), DeviceReservationError::Cancelled);
        assert_eq!(pool.available(), DeviceReservationRemainder::new(80, 16));

        let token = CancellationToken::new();
        let lease = pool.try_grant_with_token(20, 8, &token).unwrap();
        assert_eq!(lease.release_on_cancel(false), DeviceReservationRemainder::zero());
        assert_eq!(pool.available(), DeviceReservationRemainder::new(60, 8));
        token.cancel();
        assert_eq!(lease.release_on_token(&token), DeviceReservationRemainder::new(20, 8));
        assert_eq!(pool.available(), DeviceReservationRemainder::new(80, 16));
        assert_eq!(lease.cancel(), DeviceReservationRemainder::zero());

        let lease = pool.try_grant(5, 1, false).unwrap();
        assert_eq!(lease.cancel(), DeviceReservationRemainder::new(5, 1));
        assert_eq!(pool.available(), DeviceReservationRemainder::new(80, 16));
    }

    #[test]
    fn overflow_refuses() {
        let pool = DeviceReservationPool::fixed(100, 100);
        assert_eq!(
            pool.try_grant_product(u64::MAX, 2, 1, 1, false).unwrap_err(),
            DeviceReservationError::Overflow
        );
        assert_eq!(
            pool.try_grant_product(1, 1, u64::MAX, 2, false).unwrap_err(),
            DeviceReservationError::Overflow
        );
        assert_eq!(
            DeviceReservationRequest::try_sum(&[u64::MAX, 1], &[4]).unwrap_err(),
            DeviceReservationError::Overflow
        );
        assert_eq!(
            DeviceReservationRequest::try_sum(&[4], &[u64::MAX, 1]).unwrap_err(),
            DeviceReservationError::Overflow
        );
        assert_eq!(pool.available(), DeviceReservationRemainder::new(100, 100));
        let parent = pool.try_grant(10, 10, false).unwrap();
        assert_eq!(
            parent.acquire_child_product(u64::MAX, 2, 1, 1, false).unwrap_err(),
            DeviceReservationError::Overflow
        );
        assert_eq!(parent.remainder(), DeviceReservationRemainder::new(10, 10));
        assert_eq!(pool.available(), DeviceReservationRemainder::new(90, 90));
        assert_eq!(parent.release(), DeviceReservationRemainder::new(10, 10));
    }

    #[test]
    fn cpu_pool_types_need_no_device_library() {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DeviceReservationPool>();
        assert_send_sync::<DeviceReservationLease>();
        let cpu = AdmissionPool::fixed(2, 128);
        let token = CancellationToken::new();
        let device = DeviceReservationPool::fixed(64, 32);
        let lease = device.try_grant(8, 4, token.is_cancelled()).unwrap();
        assert_eq!(cpu.available().threads(), 2);
        assert_eq!(cpu.available().bytes(), 128);
        assert_eq!(cpu.thread_budget(), 2);
        assert!(!token.is_cancelled());
        assert_eq!(lease.release(), DeviceReservationRemainder::new(8, 4));
        assert_eq!(device.available(), DeviceReservationRemainder::new(64, 32));
    }

    #[test]
    fn child_lease_draws_from_parent() {
        let pool = DeviceReservationPool::fixed(100, 40);
        let parent = pool.try_grant(100, 40, false).unwrap();
        assert_eq!(
            parent.acquire_child(101, 0, false).unwrap_err(),
            DeviceReservationError::ExceedsRemainder {
                requested_device_bytes: 101,
                requested_spill_bytes: 0,
                remaining_device_bytes: 100,
                remaining_spill_bytes: 40,
            }
        );
        assert_eq!(
            parent.acquire_child(1, 1, true).unwrap_err(),
            DeviceReservationError::Cancelled
        );
        assert_eq!(pool.available(), DeviceReservationRemainder::zero());
        let child = parent.acquire_child(25, 10, false).unwrap();
        assert_eq!(parent.remainder(), DeviceReservationRemainder::new(75, 30));
        assert_eq!(child.release(), DeviceReservationRemainder::new(25, 10));
        assert_eq!(parent.remainder(), DeviceReservationRemainder::new(100, 40));
        let child = parent.acquire_child(25, 10, false).unwrap();
        assert_eq!(parent.release(), DeviceReservationRemainder::new(75, 30));
        assert_eq!(pool.available(), DeviceReservationRemainder::new(75, 30));
        assert_eq!(child.release(), DeviceReservationRemainder::new(25, 10));
        assert_eq!(pool.available(), DeviceReservationRemainder::new(100, 40));

        let parent = pool.try_grant(100, 40, false).unwrap();
        let child = parent.acquire_child(25, 10, false).unwrap();
        drop(parent);
        assert_eq!(pool.available(), DeviceReservationRemainder::new(75, 30));
        drop(child);
        assert_eq!(pool.available(), DeviceReservationRemainder::new(100, 40));

        let parent = pool.try_grant(10, 4, false).unwrap();
        assert_eq!(parent.release_on_cancel(true), DeviceReservationRemainder::new(10, 4));
        assert_eq!(
            parent.acquire_child(1, 1, false).unwrap_err(),
            DeviceReservationError::Released
        );
        assert_eq!(pool.available(), DeviceReservationRemainder::new(100, 40));
    }

    #[test]
    fn worker_failure_returns_bytes() {
        let pool = DeviceReservationPool::fixed(40, 20);
        let worker_pool = pool.clone();
        let worker = thread::spawn(move || {
            let _lease = worker_pool.try_grant(15, 7, false).expect("grant");
            panic!("worker failed");
        });
        assert!(worker.join().is_err());
        assert_eq!(pool.available(), DeviceReservationRemainder::new(40, 20));
    }
}
