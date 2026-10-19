//! One dominant parallel axis for a physical operator.
//!
//! A plan chooses study, fold, chain, replicate, graph test, or numerical
//! kernel. That choice is the only axis the plan owns. Inner work, while the
//! plan holds the workers, receives a [`SerialInner`] context: one worker,
//! labeled with the outer axis. The inner context does not reserve workers
//! and does not start a pool.
//!
//! [`WorkerCap`] is the ceiling for concurrent plans. [`WorkerCap`] clones
//! share one counter. Admission reserves before any worker would start, and a
//! refusal leaves that counter and the limit unchanged. A request of zero
//! workers, or of more workers than the limit, is refused before the counter
//! is touched. A request that fits the limit but would make the shared total
//! exceed it is refused the same way. A total equal to the limit is admitted.
//!
//! This module does not spawn threads.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use thiserror::Error;

/// Workers on a serial inner context.
///
/// The outer plan owns the pool. Inner work stays on one worker.
const SERIAL_INNER_WORKERS: u32 = 1;

/// Dominant parallel axis of one physical operator.
///
/// Study is concurrent queries. Fold is cross-fitting. Chain is a set of
/// Markov chains. Replicate is bootstrap or resampling. Graph test is
/// conditional-independence or graph-atom work. Numerical kernel is one large
/// matrix operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ParallelAxis {
    /// Concurrent queries in one study.
    Study,
    /// Cross-fitting folds.
    Fold,
    /// Markov chains.
    Chain,
    /// Bootstrap or resampling replicates.
    Replicate,
    /// Conditional-independence tests or graph atoms.
    GraphTest,
    /// One large matrix kernel.
    NumericalKernel,
}

impl ParallelAxis {
    /// Study, fold, chain, replicate, graph test, and numerical kernel.
    #[must_use]
    pub const fn all() -> [Self; 6] {
        [
            Self::Study,
            Self::Fold,
            Self::Chain,
            Self::Replicate,
            Self::GraphTest,
            Self::NumericalKernel,
        ]
    }

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Study => "study",
            Self::Fold => "fold",
            Self::Chain => "chain",
            Self::Replicate => "replicate",
            Self::GraphTest => "graph_test",
            Self::NumericalKernel => "numerical_kernel",
        }
    }
}

impl fmt::Display for ParallelAxis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Shared ceiling on workers held by concurrent plans.
///
/// [`Clone`] copies the handle. The limit and the reserved count stay shared.
/// A limit of zero is a declared ceiling that admits no plan.
#[derive(Clone, Debug)]
pub struct WorkerCap {
    limit: u32,
    reserved: Arc<AtomicU32>,
}

impl WorkerCap {
    /// Cap of `limit` workers, with nothing reserved.
    #[must_use]
    pub fn new(limit: u32) -> Self {
        Self { limit, reserved: Arc::new(AtomicU32::new(0)) }
    }

    /// Declared ceiling. This does not change after construction.
    #[must_use]
    pub const fn limit(&self) -> u32 {
        self.limit
    }

    /// Workers currently reserved by plans that share this cap.
    #[must_use]
    pub fn reserved(&self) -> u32 {
        self.reserved.load(Ordering::Acquire)
    }

    /// Reserve `workers`, or leave the counter untouched.
    ///
    /// `workers` is already non-zero and within [`Self::limit`]. The update
    /// stores a new total only when that total still fits. A miss does not
    /// write.
    fn try_reserve(&self, workers: u32) -> Result<Reservation, ParallelPlanError> {
        let limit = self.limit;
        let updated = self.reserved.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            let next = current.checked_add(workers)?;
            (next <= limit).then_some(next)
        });
        match updated {
            Ok(_) => Ok(Reservation { workers, reserved: Arc::clone(&self.reserved) }),
            Err(reserved) => {
                Err(ParallelPlanError::ConcurrentCap { workers, reserved, cap: limit })
            }
        }
    }
}

/// Workers this plan reserved. Drop returns them to the cap.
#[derive(Debug)]
struct Reservation {
    workers: u32,
    reserved: Arc<AtomicU32>,
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.reserved.fetch_sub(self.workers, Ordering::Release);
    }
}

/// One dominant axis and the workers reserved for it.
///
/// Dropping the plan returns its workers to the [`WorkerCap`] it reserved
/// from. The plan does not spawn threads.
#[derive(Debug)]
#[must_use = "dropping the plan releases the workers it reserved"]
pub struct ParallelPlan {
    axis: ParallelAxis,
    reservation: Reservation,
}

impl ParallelPlan {
    /// Admit `axis` and reserve `workers` from `cap`.
    ///
    /// The plan has that one axis. `workers` must be at least one and no
    /// greater than the cap's limit. Zero, a count above the limit, and a
    /// count that would push the shared total past the limit are refused
    /// before any worker starts. A refusal does not change the cap.
    ///
    /// # Errors
    ///
    /// [`ParallelPlanError::ZeroWorkers`] when `workers` is zero.
    /// [`ParallelPlanError::WorkersAboveCap`] when `workers` is greater than
    /// the limit. [`ParallelPlanError::ConcurrentCap`] when `workers` fits
    /// the limit alone but not together with workers already reserved.
    #[must_use = "a refused plan does not reserve workers and does not start work"]
    pub fn new(
        axis: ParallelAxis,
        workers: u32,
        cap: &WorkerCap,
    ) -> Result<Self, ParallelPlanError> {
        if workers == 0 {
            return Err(ParallelPlanError::ZeroWorkers);
        }
        let limit = cap.limit();
        if workers > limit {
            return Err(ParallelPlanError::WorkersAboveCap { workers, cap: limit });
        }
        let reservation = cap.try_reserve(workers)?;
        Ok(Self { axis, reservation })
    }

    /// Dominant axis chosen for this plan.
    #[must_use]
    pub const fn axis(&self) -> ParallelAxis {
        self.axis
    }

    /// Workers this plan reserved. At least one.
    #[must_use]
    pub const fn workers(&self) -> u32 {
        self.reservation.workers
    }

    /// Keep `axis` when it is already this plan's dominant axis.
    ///
    /// A different axis is a second axis. It is refused. The plan and the
    /// cap stay as they were: one plan has one dominant axis.
    ///
    /// # Errors
    ///
    /// [`ParallelPlanError::SecondAxis`] when `axis` is not [`Self::axis`].
    #[must_use = "a second axis is refused and the plan keeps its first axis"]
    pub fn choose(&self, axis: ParallelAxis) -> Result<ParallelAxis, ParallelPlanError> {
        if axis == self.axis {
            Ok(self.axis)
        } else {
            Err(ParallelPlanError::SecondAxis { held: self.axis, requested: axis })
        }
    }

    /// Inner context for work nested under this plan.
    ///
    /// The outer plan owns the pool. The inner context has one worker and
    /// records [`Self::axis`]. It does not reserve from the cap.
    #[must_use]
    pub const fn serial_inner(&self) -> SerialInner {
        SerialInner::from_outer(self.axis)
    }
}

/// Inner context used when an outer plan already owns the workers.
///
/// [`Self::workers`] is one. [`Self::outer_axis`] is the axis of the plan
/// that owns the pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SerialInner {
    outer: ParallelAxis,
    workers: u32,
}

impl SerialInner {
    const fn from_outer(outer: ParallelAxis) -> Self {
        Self { outer, workers: SERIAL_INNER_WORKERS }
    }

    /// Axis of the plan that owns the pool.
    #[must_use]
    pub const fn outer_axis(self) -> ParallelAxis {
        self.outer
    }

    /// Workers for the inner context. Always one.
    #[must_use]
    pub const fn workers(self) -> u32 {
        self.workers
    }
}

/// Refusal to admit a parallel plan.
///
/// Admission does not start workers. A refusal reserves nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum ParallelPlanError {
    /// The plan asked for no workers.
    #[error(
        "parallel plan requires at least one worker; next attempt: request a positive worker \
         count within the cap"
    )]
    ZeroWorkers,
    /// The plan asked for more workers than the cap's limit.
    ///
    /// The shared counter was not changed.
    #[error(
        "parallel plan requested {workers} workers, above the cap of {cap}; the cap is \
         unchanged; next attempt: request no more workers than the cap"
    )]
    WorkersAboveCap {
        /// Workers the plan requested.
        workers: u32,
        /// Cap limit the request exceeded.
        cap: u32,
    },
    /// This plan would make the shared total larger than the cap.
    ///
    /// `reserved` is the count observed at the refusal. The counter was not
    /// increased.
    #[error(
        "parallel plan requested {workers} workers in addition to {reserved} already reserved, \
         exceeding the cap of {cap}; the cap is unchanged; next attempt: wait for a plan to \
         finish or request fewer workers"
    )]
    ConcurrentCap {
        /// Workers the plan requested.
        workers: u32,
        /// Workers already reserved when the plan was refused.
        reserved: u32,
        /// Cap limit the combined total exceeded.
        cap: u32,
    },
    /// The plan already has a dominant axis.
    #[error(
        "parallel plan already chose {held} and refuses {requested}; one plan has one dominant \
         axis; next attempt: keep {held} or build a separate plan"
    )]
    SecondAxis {
        /// Axis this plan already holds.
        held: ParallelAxis,
        /// Axis that was refused.
        requested: ParallelAxis,
    },
}

#[cfg(test)]
mod tests {
    use super::{ParallelAxis, ParallelPlan, ParallelPlanError, WorkerCap};

    #[test]
    fn each_axis_can_be_chosen_alone() {
        for axis in ParallelAxis::all() {
            let cap = WorkerCap::new(2);
            let plan = ParallelPlan::new(axis, 2, &cap).unwrap();
            assert_eq!(plan.choose(axis).unwrap(), axis);
            assert_eq!(plan.axis(), axis);
            assert_eq!(plan.workers(), 2);
            assert_eq!(cap.reserved(), 2);
            assert_eq!(cap.limit(), 2);
        }
    }

    #[test]
    fn two_axes_on_one_plan_are_refused() {
        let cap = WorkerCap::new(1);
        let plan = ParallelPlan::new(ParallelAxis::Study, 1, &cap).unwrap();
        for axis in ParallelAxis::all() {
            if axis == ParallelAxis::Study {
                assert_eq!(plan.choose(axis).unwrap(), ParallelAxis::Study);
                continue;
            }
            let err = plan.choose(axis).unwrap_err();
            assert_eq!(
                err,
                ParallelPlanError::SecondAxis { held: ParallelAxis::Study, requested: axis }
            );
        }
        assert_eq!(plan.axis(), ParallelAxis::Study);
        assert_eq!(plan.workers(), 1);
        assert_eq!(cap.reserved(), 1);
        assert_eq!(cap.limit(), 1);
    }

    #[test]
    fn serial_inner_has_one_worker() {
        for axis in ParallelAxis::all() {
            let cap = WorkerCap::new(3);
            let plan = ParallelPlan::new(axis, 3, &cap).unwrap();
            let inner = plan.serial_inner();
            assert_eq!(inner.workers(), 1);
            assert_eq!(inner.outer_axis(), axis);
            assert_eq!(plan.workers(), 3);
            assert_eq!(cap.reserved(), 3);
        }
    }

    #[test]
    fn two_plans_past_the_cap_refuse_the_second() {
        let cap = WorkerCap::new(4);
        let first = ParallelPlan::new(ParallelAxis::Study, 3, &cap).unwrap();
        let reserved = cap.reserved();
        let err = ParallelPlan::new(ParallelAxis::GraphTest, 2, &cap).unwrap_err();
        assert_eq!(err, ParallelPlanError::ConcurrentCap { workers: 2, reserved: 3, cap: 4 });
        assert_eq!(cap.reserved(), reserved);
        assert_eq!(cap.limit(), 4);
        assert_eq!(first.axis(), ParallelAxis::Study);
        assert_eq!(first.workers(), 3);
    }

    #[test]
    fn zero_workers_are_refused_before_work() {
        let cap = WorkerCap::new(4);
        let held = ParallelPlan::new(ParallelAxis::Fold, 2, &cap).unwrap();
        let err = ParallelPlan::new(ParallelAxis::Chain, 0, &cap).unwrap_err();
        assert_eq!(err, ParallelPlanError::ZeroWorkers);
        assert_eq!(cap.reserved(), 2);
        assert_eq!(cap.limit(), 4);
        assert_eq!(held.workers(), 2);
    }

    #[test]
    fn workers_above_the_cap_are_refused_before_work() {
        let cap = WorkerCap::new(4);
        let held = ParallelPlan::new(ParallelAxis::Chain, 1, &cap).unwrap();
        let err = ParallelPlan::new(ParallelAxis::Replicate, 5, &cap).unwrap_err();
        assert_eq!(err, ParallelPlanError::WorkersAboveCap { workers: 5, cap: 4 });
        assert_eq!(cap.reserved(), 1);
        assert_eq!(cap.limit(), 4);
        assert_eq!(held.axis(), ParallelAxis::Chain);
    }

    #[test]
    fn plans_that_sum_to_the_cap_are_admitted() {
        let cap = WorkerCap::new(4);
        let first = ParallelPlan::new(ParallelAxis::Fold, 3, &cap).unwrap();
        let second = ParallelPlan::new(ParallelAxis::Chain, 1, &cap).unwrap();
        assert_eq!(cap.reserved(), 4);
        let err = ParallelPlan::new(ParallelAxis::Study, 1, &cap).unwrap_err();
        assert_eq!(err, ParallelPlanError::ConcurrentCap { workers: 1, reserved: 4, cap: 4 });
        assert_eq!(cap.reserved(), 4);
        drop(first);
        assert_eq!(cap.reserved(), 1);
        let third = ParallelPlan::new(ParallelAxis::NumericalKernel, 3, &cap).unwrap();
        assert_eq!(cap.reserved(), 4);
        drop(second);
        drop(third);
        assert_eq!(cap.reserved(), 0);
        assert_eq!(cap.limit(), 4);
    }
}
