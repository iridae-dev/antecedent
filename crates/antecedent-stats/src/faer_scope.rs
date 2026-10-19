//! Per-operation worker limit and scratch budget for `faer`.
//!
//! A [`FaerScope`] carries the caller's worker limit and a scratch budget in
//! bytes. [`FaerScope::parallelism`] is the degree a kernel passes into `faer`:
//! [`faer::Par::Seq`] for one worker, and [`faer::Par::Rayon`] with that exact
//! count when the limit is larger. The count is never zero. This module does
//! not call [`faer::Par::rayon`], [`faer::get_global_parallelism`], or
//! [`faer::set_global_parallelism`]. `Par::rayon(0)` reads the process-global
//! rayon pool's current width. That operation is not provided, and neither are
//! the convenience factorizations that read the same pool (`ColPivQr::new`,
//! `thin_svd`, `self_adjoint_eigenvalues`, `self_adjoint_eigen`).
//!
//! [`FaerScope::admit_scratch`] checks a [`faer::dyn_stack::StackReq`] against
//! the budget before [`FaerScope::scratch_buffer`] allocates. A requirement
//! equal to the budget is admitted. A larger requirement is refused and
//! allocates nothing.
//!
//! [`FaerScope`] has no [`Default`]. A call that omits the scope is not filled
//! in by a default that hides the process pool. Zero workers are refused. The
//! scope does not spawn threads. `Par::Rayon` here is a degree for the caller
//! to pass down; this module does not run a `faer` kernel.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::num::{NonZeroU32, NonZeroUsize};

use faer::Par;
use faer::dyn_stack::{MemBuffer, StackReq};
use thiserror::Error;

/// Caller worker limit and scratch budget for one `faer` kernel.
///
/// The limit is a degree, not a pool. Dropping the scope releases nothing and
/// starts nothing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FaerScope {
    workers: NonZeroU32,
    scratch_budget: u64,
}

impl FaerScope {
    /// Scope with `workers` and a scratch budget of `scratch_budget` bytes.
    ///
    /// `workers` must be at least one. A scratch budget of zero admits only an
    /// empty requirement. This does not spawn threads and does not read the
    /// process pool.
    ///
    /// # Errors
    ///
    /// [`FaerScopeError::ZeroWorkers`] when `workers` is zero.
    #[must_use = "a refused scope does not carry a worker limit or a scratch budget"]
    pub const fn try_new(workers: u32, scratch_budget: u64) -> Result<Self, FaerScopeError> {
        match NonZeroU32::new(workers) {
            Some(workers) => Ok(Self { workers, scratch_budget }),
            None => Err(FaerScopeError::ZeroWorkers),
        }
    }

    /// Worker limit declared for this scope. At least one.
    #[must_use]
    pub const fn workers(self) -> u32 {
        self.workers.get()
    }

    /// Scratch budget in bytes. This does not allocate.
    #[must_use]
    pub const fn scratch_budget(self) -> u64 {
        self.scratch_budget
    }

    /// Explicit `faer` degree for a kernel that receives this scope.
    ///
    /// One worker is [`Par::Seq`], which stays on the caller thread. A larger
    /// limit is [`Par::Rayon`] of that same count. This is not `Par::rayon(0)`
    /// and does not call [`faer::get_global_parallelism`].
    #[must_use]
    pub fn parallelism(self) -> Par {
        let workers = self.workers.get();
        if workers == 1 {
            Par::Seq
        } else {
            match NonZeroUsize::new(usize::try_from(workers).unwrap_or(1)) {
                Some(workers) => Par::Rayon(workers),
                None => Par::Seq,
            }
        }
    }

    /// Bytes of `required` when they fit in the scratch budget.
    ///
    /// A layout `faer` cannot satisfy is [`FaerScopeError::ScratchOverflow`].
    /// A requirement above the budget is [`FaerScopeError::ScratchOverBudget`].
    /// Neither path allocates.
    ///
    /// # Errors
    ///
    /// [`FaerScopeError::ScratchOverflow`] when `required` has no layout.
    /// [`FaerScopeError::ScratchOverBudget`] when the layout's byte size is
    /// greater than [`Self::scratch_budget`].
    #[must_use = "a scratch refusal means the kernel must not allocate that buffer"]
    pub fn admit_scratch(self, required: StackReq) -> Result<u64, FaerScopeError> {
        let required_bytes = scratch_bytes(required)?;
        if required_bytes > self.scratch_budget {
            Err(FaerScopeError::ScratchOverBudget {
                required_bytes,
                budget_bytes: self.scratch_budget,
            })
        } else {
            Ok(required_bytes)
        }
    }

    /// Allocate `required` after [`Self::admit_scratch`] accepts it.
    ///
    /// The buffer is the kernel's scratch. This does not spawn threads. An
    /// allocation that the process cannot satisfy still aborts through the
    /// global allocator, after the budget check has passed.
    ///
    /// # Errors
    ///
    /// The same refusals as [`Self::admit_scratch`]. A refusal allocates nothing.
    #[must_use = "a scratch refusal means the kernel must not allocate that buffer"]
    pub fn scratch_buffer(self, required: StackReq) -> Result<MemBuffer, FaerScopeError> {
        self.admit_scratch(required)?;
        Ok(MemBuffer::new(required))
    }
}

/// Byte size of `required`, or a layout this budget cannot admit.
fn scratch_bytes(required: StackReq) -> Result<u64, FaerScopeError> {
    if required.align_bytes() == 0 {
        return Err(FaerScopeError::ScratchOverflow);
    }
    match u64::try_from(required.size_bytes()) {
        Ok(bytes) => Ok(bytes),
        Err(_) => Err(FaerScopeError::ScratchOverflow),
    }
}

/// Refusal to build a [`FaerScope`] or to admit its scratch.
///
/// A refusal does not spawn threads and does not allocate scratch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum FaerScopeError {
    /// The caller asked for no workers.
    ///
    /// Zero is not a stand-in for the process pool.
    #[error(
        "faer scope requires at least one worker; next attempt: declare a positive worker limit"
    )]
    ZeroWorkers,
    /// `faer` reported a scratch layout that cannot be allocated.
    #[error(
        "faer scratch requirement has no layout; next attempt: shrink the factorization that \
         produced it"
    )]
    ScratchOverflow,
    /// The scratch layout is larger than this scope's budget.
    ///
    /// Nothing was allocated.
    #[error(
        "faer scratch needs {required_bytes} bytes, above the budget of {budget_bytes}; the \
         budget is unchanged; next attempt: raise the scratch budget or shrink the kernel"
    )]
    ScratchOverBudget {
        /// Bytes the kernel asked to admit.
        required_bytes: u64,
        /// Budget the request exceeded.
        budget_bytes: u64,
    },
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use faer::Par;
    use faer::dyn_stack::StackReq;

    use super::{FaerScope, FaerScopeError};

    #[test]
    fn two_scopes_declare_different_worker_limits() {
        let narrow = FaerScope::try_new(1, 16).unwrap();
        let wide = FaerScope::try_new(4, 64).unwrap();
        assert_eq!(narrow.workers(), 1);
        assert_eq!(wide.workers(), 4);
        assert_ne!(narrow.workers(), wide.workers());
        assert_eq!(narrow.scratch_budget(), 16);
        assert_eq!(wide.scratch_budget(), 64);
        assert_eq!(narrow.parallelism(), Par::Seq);
        assert_eq!(narrow.parallelism().degree(), 1);
        let four = NonZeroUsize::new(4).unwrap();
        assert_eq!(wide.parallelism(), Par::Rayon(four));
        assert_eq!(wide.parallelism().degree(), 4);
    }

    #[test]
    fn call_without_a_scope_is_not_constructed_by_default() {
        // `FaerScope: Default` would make `<T as Probe<_>>::probe` ambiguous.
        // A call with no scope is therefore not `FaerScope::default()`, which
        // would be the place a process-global pool could hide.
        assert_no_default::<FaerScope>();
    }

    #[test]
    fn zero_workers_are_refused() {
        assert_eq!(FaerScope::try_new(0, 32), Err(FaerScopeError::ZeroWorkers));
        let held = FaerScope::try_new(2, 32).unwrap();
        assert_eq!(held.workers(), 2);
        assert_eq!(FaerScope::try_new(0, 0), Err(FaerScopeError::ZeroWorkers));
        assert_eq!(held.scratch_budget(), 32);
    }

    #[test]
    fn scratch_above_the_budget_is_refused() {
        let scope = FaerScope::try_new(1, 8).unwrap();
        let eight = StackReq::new::<u8>(8);
        assert_eq!(scope.admit_scratch(eight), Ok(8));
        assert_eq!(
            scope.admit_scratch(StackReq::new::<u8>(9)),
            Err(FaerScopeError::ScratchOverBudget { required_bytes: 9, budget_bytes: 8 })
        );
        assert_eq!(scope.admit_scratch(StackReq::OVERFLOW), Err(FaerScopeError::ScratchOverflow));
        let buffer = scope.scratch_buffer(eight).unwrap();
        assert!(buffer.len() >= 8);

        let empty_budget = FaerScope::try_new(2, 0).unwrap();
        assert_eq!(empty_budget.admit_scratch(StackReq::empty()), Ok(0));
        assert_eq!(
            empty_budget.admit_scratch(StackReq::new::<u8>(1)),
            Err(FaerScopeError::ScratchOverBudget { required_bytes: 1, budget_bytes: 0 })
        );
        assert_eq!(empty_budget.workers(), 2);
        assert_ne!(empty_budget.workers(), scope.workers());
    }

    fn assert_no_default<T: ?Sized>() {
        struct Marker;
        trait Probe<A> {
            fn probe() {}
        }
        impl<T: ?Sized> Probe<()> for T {}
        #[allow(dead_code)]
        impl<T: Default> Probe<Marker> for T {}
        <T as Probe<_>>::probe();
    }
}
