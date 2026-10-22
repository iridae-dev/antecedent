//! CPU admission for a fold and its learner.
//!
//! No accelerator is promoted. Selecting one refuses and does not touch the
//! device account. A granted fold draws CPU slots, RAM, and its scratch bytes
//! from the shared [`ExecutionContext`] admission pool. A learner draws its CPU
//! and RAM from that fold and its scratch from the same pool. A request larger
//! than the remainder is refused and leaves both accounts unchanged. Cancellation
//! returns the scratch
//! and the lease on the CPU path.
//!
//! Predictions are written by physical row. Non-neural values stay `f64`.
//! The neural baseline stays `f32` and is not selected here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use thiserror::Error;

use crate::execution::{AdmissionError, AdmissionLease, ExecutionContext};
use crate::physical_operator::DeclaredTolerance;

use super::{DeviceReservationError, DeviceReservationPool, DeviceReservationRemainder};

/// Where work asked to run.
///
/// [`ComputeSelection::Accelerator`] is not a device that can be reserved.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ComputeSelection {
    /// Host CPU. Device bytes stay where they were.
    Cpu,
    /// An accelerator. Selection is refused.
    Accelerator,
}

/// CPU path after [`select_compute`] accepts the host.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[must_use = "cpu admission is not a device reservation"]
pub struct CpuAdmission;

impl CpuAdmission {
    /// Device remainder, unread as a grant.
    ///
    /// The pool is not decremented. A small serial job uses this and then the
    /// CPU admission pool.
    #[must_use = "the unchanged remainder shows that no device was reserved"]
    pub fn reserve_nothing(self, pool: &DeviceReservationPool) -> DeviceReservationRemainder {
        pool.available()
    }
}

/// Refuse an accelerator. Accept the host without reserving device bytes.
///
/// # Errors
///
/// [`DeviceReservationError::Unpromoted`] when `selection` is
/// [`ComputeSelection::Accelerator`].
#[must_use = "a refused accelerator is not a device reservation"]
pub const fn select_compute(
    selection: ComputeSelection,
) -> Result<CpuAdmission, DeviceReservationError> {
    match selection {
        ComputeSelection::Cpu => Ok(CpuAdmission),
        ComputeSelection::Accelerator => Err(DeviceReservationError::Unpromoted),
    }
}

/// Arithmetic width of a path.
///
/// [`Self::ScientificF64`] is the default. [`Self::NeuralF32`] is the neural
/// learner's existing baseline and is not applied by this module.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ArithmeticPath {
    /// Host scientific arithmetic.
    ScientificF64,
    /// Neural `f32` baseline.
    NeuralF32,
}

impl fmt::Display for ArithmeticPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScientificF64 => formatter.write_str("f64"),
            Self::NeuralF32 => formatter.write_str("f32"),
        }
    }
}

/// Default arithmetic. This is not the neural baseline.
#[must_use = "non-neural arithmetic stays f64"]
pub const fn scientific_arithmetic() -> ArithmeticPath {
    ArithmeticPath::ScientificF64
}

/// Neural learner baseline. Other paths do not use it.
#[must_use = "the neural baseline is f32 and is not the default"]
pub const fn neural_baseline() -> ArithmeticPath {
    ArithmeticPath::NeuralF32
}

/// Keep a non-neural value in `f64`.
#[must_use = "the f64 value is the non-neural result"]
pub const fn retain_scientific_f64(value: f64) -> f64 {
    value
}

/// Absolute tolerance for a non-neural comparison.
///
/// The bound is tighter than an `f32` ulp near order-1 values, so a value that
/// has been narrowed to `f32` misses it.
#[must_use = "a missing tolerance is not permission to compare"]
pub const fn scientific_tolerance() -> Option<DeclaredTolerance> {
    DeclaredTolerance::new(1e-12, 0.0)
}

/// Whether `left` and `right` meet `tolerance`.
///
/// A non-finite gap misses. The absolute bound is tried first. A zero relative
/// bound does not accept a larger gap.
#[allow(
    clippy::missing_const_for_fn,
    reason = "the tolerance comparison is the runtime numerical check"
)]
#[must_use = "a tolerance miss is not an accepted comparison"]
pub fn within_declared_tolerance(left: f64, right: f64, tolerance: DeclaredTolerance) -> bool {
    let gap = (left - right).abs();
    if !gap.is_finite() {
        return false;
    }
    if gap <= tolerance.absolute() {
        return true;
    }
    let scale = left.abs().max(right.abs());
    tolerance.relative() > 0.0 && gap <= tolerance.relative() * scale
}

/// One CPU slot when the pool has a free slot, otherwise none.
///
/// A parent that already holds every slot, including a parallel map, is asked
/// for no further slot. The request is not shrunk when the caller asked for
/// more than the remainder: that request is refused by [`admit_fold`].
#[must_use = "zero threads means the caller already holds the cpu lease"]
pub const fn nested_thread_request(free_threads: u32) -> u32 {
    if free_threads > 0 { 1 } else { 0 }
}

/// Why a fold or learner was not admitted on the CPU path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum CpuAdmissionError {
    /// The fold request is larger than the shared remainder.
    #[error(
        "fold request of {requested_threads} threads and {requested_bytes} bytes exceeds the \
         lease remainder of {remaining_threads} threads and {remaining_bytes} bytes"
    )]
    FoldExceedsLease {
        /// CPU slots asked for.
        requested_threads: u32,
        /// RAM bytes asked for.
        requested_bytes: u64,
        /// CPU slots still free.
        remaining_threads: u32,
        /// RAM bytes still free.
        remaining_bytes: u64,
    },
    /// The learner request is larger than the fold remainder.
    #[error(
        "learner request of {requested_threads} threads and {requested_bytes} bytes exceeds the \
         fold lease of {remaining_threads} threads and {remaining_bytes} bytes"
    )]
    LearnerExceedsLease {
        /// CPU slots asked for.
        requested_threads: u32,
        /// RAM bytes asked for.
        requested_bytes: u64,
        /// CPU slots the fold still holds.
        remaining_threads: u32,
        /// RAM bytes the fold still holds.
        remaining_bytes: u64,
    },
    /// Cancellation won before the hold was granted, or the hold was cleared.
    #[error("admission stopped because the work was cancelled")]
    Cancelled,
    /// The fold lease is already gone.
    #[error("admission lease is already released")]
    Released,
    /// A size did not fit in `u64`.
    #[error("admission arithmetic overflowed and was refused")]
    Overflow,
    /// The admission pool could not fund the scratch buffer. The CPU lease was returned.
    #[error(
        "the admission pool cannot fund {bytes} bytes of scratch; next attempt: release a lease \
         or raise the pool"
    )]
    Scratch {
        /// Scratch bytes asked for.
        bytes: u64,
    },
}

/// A fold's CPU, RAM, and scratch hold.
#[derive(Debug)]
#[must_use = "dropping the fold admission releases its cpu, ram, and scratch"]
pub struct FoldAdmission {
    hold: Hold,
}

impl FoldAdmission {
    /// CPU slots this fold still holds.
    #[must_use = "the held cpu count is the learner thread budget"]
    pub fn held_threads(&self) -> u32 {
        self.hold.held_threads()
    }

    /// Draw a learner from this fold's remainder, plus `scratch_bytes` of scratch from
    /// the shared pool.
    ///
    /// The learner cannot take CPU or RAM this fold does not still hold. A
    /// refusal takes no scratch.
    ///
    /// # Errors
    ///
    /// [`CpuAdmissionError::LearnerExceedsLease`] when the request is larger
    /// than this fold's remainder. [`CpuAdmissionError::Cancelled`] when the
    /// study is cancelled. [`CpuAdmissionError::Released`] when this fold is
    /// already gone. [`CpuAdmissionError::Scratch`] when the pool cannot fund the
    /// scratch; the child lease is returned. [`CpuAdmissionError::Overflow`]
    /// when the lease id does not fit.
    #[must_use = "a refused learner is not permission to fit"]
    pub fn admit_learner(
        &self,
        threads: u32,
        bytes: u64,
        scratch_bytes: u64,
    ) -> Result<LearnerAdmission, CpuAdmissionError> {
        let Some(parent) = self.hold.lease.as_ref() else {
            return Err(CpuAdmissionError::Released);
        };
        let child = parent.acquire_child(threads, bytes).map_err(learner_error)?;
        let scratch = draw_scratch(&child, scratch_bytes)?;
        Ok(LearnerAdmission { hold: Hold::from_parts(child, scratch) })
    }

    /// Return this fold's scratch and lease when `token` is cancelled.
    ///
    /// `false` leaves the hold in place.
    #[must_use = "a cancelled fold has no result"]
    pub fn observe_cancel(&mut self, token: &crate::execution::CancellationToken) -> bool {
        self.hold.observe_cancel(token)
    }
}

/// A learner's portion of a fold lease, plus one scratch buffer.
#[derive(Debug)]
#[must_use = "dropping the learner admission releases its cpu, ram, and scratch"]
pub struct LearnerAdmission {
    hold: Hold,
}

impl LearnerAdmission {
    /// Return this learner's scratch and lease when `token` is cancelled.
    ///
    /// `false` leaves the hold in place.
    #[must_use = "a cancelled learner has no result"]
    pub fn observe_cancel(&mut self, token: &crate::execution::CancellationToken) -> bool {
        self.hold.observe_cancel(token)
    }
}

/// Take a fold lease and `scratch_bytes` of scratch.
///
/// Cancellation and a short remainder grant nothing. Scratch bytes are a second
/// root lease on the same pool. If that draw fails, the CPU lease is returned.
///
/// # Errors
///
/// [`CpuAdmissionError::Cancelled`] when `ctx` is cancelled.
/// [`CpuAdmissionError::FoldExceedsLease`] when `threads` or `bytes` exceed the
/// shared remainder. [`CpuAdmissionError::Scratch`] when the pool cannot fund
/// `scratch_bytes`. [`CpuAdmissionError::Overflow`] when the lease id
/// does not fit.
#[must_use = "a refused fold is not permission to run it"]
pub fn admit_fold(
    ctx: &ExecutionContext,
    threads: u32,
    bytes: u64,
    scratch_bytes: u64,
) -> Result<FoldAdmission, CpuAdmissionError> {
    if ctx.cancellation.is_requested() {
        return Err(CpuAdmissionError::Cancelled);
    }
    let lease = ctx.acquire_admission(threads, bytes).map_err(fold_error)?;
    if ctx.cancellation.is_requested() {
        drop(lease);
        return Err(CpuAdmissionError::Cancelled);
    }
    let scratch = draw_scratch(&lease, scratch_bytes)?;
    Ok(FoldAdmission { hold: Hold::from_parts(lease, scratch) })
}

/// Draw `bytes` of scratch as a root lease on `parent`'s pool. A refusal returns nothing
/// held; the caller's `parent` is dropped on the error path by the caller's scope.
fn draw_scratch(parent: &AdmissionLease, bytes: u64) -> Result<AdmissionLease, CpuAdmissionError> {
    match parent.acquire_from_pool(0, bytes) {
        Ok(scratch) => Ok(scratch),
        Err(AdmissionError::Cancelled) => Err(CpuAdmissionError::Cancelled),
        Err(AdmissionError::Released) => Err(CpuAdmissionError::Released),
        Err(AdmissionError::Overflow) => Err(CpuAdmissionError::Overflow),
        Err(_) => Err(CpuAdmissionError::Scratch { bytes }),
    }
}

/// Write each fold's values onto physical rows.
///
/// The order of `folds` does not change the result. Every physical row is
/// written once.
///
/// # Errors
///
/// [`PredictionOrderError`] when a fold's rows and values differ in length, a
/// row is outside `n_rows`, a row is repeated, or a row is missing.
#[must_use = "misplaced predictions are not a physical-order result"]
pub fn place_physical_predictions(
    n_rows: usize,
    folds: &[(&[u64], &[f64])],
) -> Result<Vec<f64>, PredictionOrderError> {
    let mut predictions = vec![0.0; n_rows];
    let mut filled = vec![false; n_rows];
    for pair in folds {
        let (rows, values) = *pair;
        if rows.len() != values.len() {
            return Err(PredictionOrderError::LengthMismatch);
        }
        for (row, value) in rows.iter().zip(values) {
            let Some(index) = usize::try_from(*row).ok().filter(|index| *index < n_rows) else {
                return Err(PredictionOrderError::Outside { row: *row });
            };
            if filled[index] {
                return Err(PredictionOrderError::Duplicate { row: *row });
            }
            filled[index] = true;
            predictions[index] = retain_scientific_f64(*value);
        }
    }
    if let Some(index) = filled.iter().position(|seen| !seen) {
        let Ok(row) = u64::try_from(index) else {
            return Err(PredictionOrderError::Outside { row: u64::MAX });
        };
        return Err(PredictionOrderError::Missing { row });
    }
    Ok(predictions)
}

/// Why physical placement was refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum PredictionOrderError {
    /// A fold's row ids and values differ in length.
    #[error("fold predictions and row ids differ in length")]
    LengthMismatch,
    /// A row id is outside the physical row count.
    #[error("prediction row {row} is outside the physical row count")]
    Outside {
        /// Row id that did not fit.
        row: u64,
    },
    /// Two folds wrote the same physical row.
    #[error("physical row {row} was predicted more than once")]
    Duplicate {
        /// Row id written twice.
        row: u64,
    },
    /// A physical row received no prediction.
    #[error("physical row {row} has no prediction")]
    Missing {
        /// Row id that stayed empty.
        row: u64,
    },
}

#[derive(Debug)]
struct Hold {
    lease: Option<AdmissionLease>,
    scratch: Option<AdmissionLease>,
}

impl Hold {
    fn from_parts(lease: AdmissionLease, scratch: AdmissionLease) -> Self {
        Self { lease: Some(lease), scratch: Some(scratch) }
    }

    fn held_threads(&self) -> u32 {
        self.lease.as_ref().map_or(0, |lease| lease.remainder().threads())
    }

    fn observe_cancel(&mut self, token: &crate::execution::CancellationToken) -> bool {
        if !token.is_requested() {
            return false;
        }
        self.scratch.take();
        if let Some(lease) = self.lease.take() {
            let _ = lease.release();
        }
        true
    }
}

/// Map an admission refusal; `exceeds` builds the variant for a request over the remainder, which
/// is the only place a fold and a learner differ.
fn admission_error(
    error: AdmissionError,
    exceeds: impl FnOnce(u32, u64, u32, u64) -> CpuAdmissionError,
) -> CpuAdmissionError {
    match error {
        AdmissionError::ExceedsRemainder {
            requested_threads,
            requested_bytes,
            remaining_threads,
            remaining_bytes,
        } => exceeds(requested_threads, requested_bytes, remaining_threads, remaining_bytes),
        // Immediate admission does not wait, so a wait bound granted nothing.
        AdmissionError::Overflow | AdmissionError::WaitBoundExceeded { .. } => {
            CpuAdmissionError::Overflow
        }
        AdmissionError::Cancelled => CpuAdmissionError::Cancelled,
        AdmissionError::Released => CpuAdmissionError::Released,
    }
}

fn fold_error(error: AdmissionError) -> CpuAdmissionError {
    admission_error(
        error,
        |requested_threads, requested_bytes, remaining_threads, remaining_bytes| {
            CpuAdmissionError::FoldExceedsLease {
                requested_threads,
                requested_bytes,
                remaining_threads,
                remaining_bytes,
            }
        },
    )
}

fn learner_error(error: AdmissionError) -> CpuAdmissionError {
    admission_error(
        error,
        |requested_threads, requested_bytes, remaining_threads, remaining_bytes| {
            CpuAdmissionError::LearnerExceedsLease {
                requested_threads,
                requested_bytes,
                remaining_threads,
                remaining_bytes,
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{AdmissionPool, ExecutionContext};

    #[test]
    fn unpromoted_accelerator_is_refused() {
        let pool = DeviceReservationPool::fixed(32, 8);
        let error = select_compute(ComputeSelection::Accelerator).unwrap_err();
        assert_eq!(error, DeviceReservationError::Unpromoted);
        assert_eq!(pool.available(), DeviceReservationRemainder::new(32, 8));
    }

    #[test]
    fn small_serial_job_reserves_no_device() {
        let pool = DeviceReservationPool::fixed(32, 8);
        let before = pool.available();
        let cpu = select_compute(ComputeSelection::Cpu).unwrap();
        assert_eq!(cpu.reserve_nothing(&pool), before);
        assert_eq!(pool.available(), before);
        assert_eq!(scientific_arithmetic(), ArithmeticPath::ScientificF64);
        assert_eq!(neural_baseline(), ArithmeticPath::NeuralF32);
    }

    #[test]
    fn fold_that_exceeds_the_lease_is_refused() {
        let pool = AdmissionPool::fixed(2, 50);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let held = admit_fold(&ctx, 1, 40, 8).unwrap();
        let error = admit_fold(&ctx, 1, 20, 8).unwrap_err();
        assert_eq!(
            error,
            CpuAdmissionError::FoldExceedsLease {
                requested_threads: 1,
                requested_bytes: 20,
                remaining_threads: 1,
                remaining_bytes: 2,
            }
        );
        assert_eq!(pool.available().bytes(), 2);
        drop(held);
        assert_eq!(pool.available().bytes(), 50);
        assert_eq!(pool.available().threads(), 2);
    }

    #[test]
    fn scratch_refusal_returns_the_cpu_lease() {
        let pool = AdmissionPool::fixed(2, 10);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let error = admit_fold(&ctx, 1, 10, 8).unwrap_err();
        assert!(matches!(error, CpuAdmissionError::Scratch { bytes: 8 }));
        assert_eq!(pool.available().bytes(), 10);
        assert_eq!(pool.available().threads(), 2);
    }

    #[test]
    fn learner_that_exceeds_the_fold_lease_is_refused() {
        let pool = AdmissionPool::fixed(2, 80);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let fold = admit_fold(&ctx, 1, 40, 8).unwrap();
        let learner = fold.admit_learner(1, 30, 8).unwrap();
        let error = fold.admit_learner(1, 20, 8).unwrap_err();
        assert_eq!(
            error,
            CpuAdmissionError::LearnerExceedsLease {
                requested_threads: 1,
                requested_bytes: 20,
                remaining_threads: 0,
                remaining_bytes: 10,
            }
        );
        assert_eq!(pool.available().bytes(), 24);
        drop(learner);
        assert_eq!(pool.available().bytes(), 32);
        drop(fold);
        assert_eq!(pool.available().bytes(), 80);
        assert_eq!(pool.available().threads(), 2);
    }

    #[test]
    fn cancellation_releases_scratch() {
        let device = DeviceReservationPool::fixed(32, 8);
        let cpu = select_compute(ComputeSelection::Cpu).unwrap();
        assert_eq!(cpu.reserve_nothing(&device).device_bytes(), 32);
        let pool = AdmissionPool::fixed(2, 100);
        let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
        let mut hold = admit_fold(&ctx, 1, 40, 16).unwrap();
        assert!(!hold.observe_cancel(&ctx.cancellation));
        assert_eq!(pool.available().bytes(), 44);
        ctx.cancellation.cancel();
        assert!(hold.observe_cancel(&ctx.cancellation));
        assert_eq!(pool.available().bytes(), 100);
        assert_eq!(pool.available().threads(), 2);
        assert_eq!(device.available(), DeviceReservationRemainder::new(32, 8));
        let refused = admit_fold(&ctx, 1, 4, 16).unwrap_err();
        assert_eq!(refused, CpuAdmissionError::Cancelled);
        assert_eq!(pool.available().bytes(), 100);
    }

    #[test]
    fn prediction_order_follows_physical_rows() {
        let first_rows = [0_u64, 2];
        let third = 1.0 / 3.0;
        let first_values = [third, 0.5];
        let second_rows = [1_u64];
        let second_values = [2.0 / 3.0];
        let forward = place_physical_predictions(
            3,
            &[(&first_rows, &first_values), (&second_rows, &second_values)],
        )
        .unwrap();
        let reverse = place_physical_predictions(
            3,
            &[(&second_rows, &second_values), (&first_rows, &first_values)],
        )
        .unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward[0].to_bits(), third.to_bits());
        assert_eq!(forward[1].to_bits(), (2.0_f64 / 3.0).to_bits());
        assert_eq!(forward[2].to_bits(), 0.5_f64.to_bits());
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the f32 round trip is the narrowing this placement must not do"
        )]
        let narrowed = f64::from(third as f32);
        assert_ne!(forward[0].to_bits(), narrowed.to_bits());
        let repeated = place_physical_predictions(1, &[(&[0], &[third]), (&[0], &[third])]);
        assert_eq!(repeated.unwrap_err(), PredictionOrderError::Duplicate { row: 0 });
        let gap = place_physical_predictions(2, &[(&[0], &[third])]);
        assert_eq!(gap.unwrap_err(), PredictionOrderError::Missing { row: 1 });
    }

    #[test]
    fn non_neural_arithmetic_does_not_drop_to_f32() {
        let value = 1.0 / 3.0;
        let kept = retain_scientific_f64(value);
        assert_eq!(kept.to_bits(), value.to_bits());
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the f32 round trip is the drop a non-neural path must not take"
        )]
        let narrowed = f64::from(value as f32);
        assert_ne!(kept.to_bits(), narrowed.to_bits());
        let tolerance = scientific_tolerance().unwrap();
        assert!(within_declared_tolerance(kept, value, tolerance));
        assert!(!within_declared_tolerance(narrowed, value, tolerance));
        assert_eq!(scientific_arithmetic(), ArithmeticPath::ScientificF64);
        assert_ne!(neural_baseline(), scientific_arithmetic());
    }
}
